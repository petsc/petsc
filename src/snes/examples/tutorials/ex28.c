/*$Id: ex28.c,v 1.30 2001/08/07 21:31:17 bsmith Exp $*/

/*  mpiexec ./ex28 -da_grid_x 48 -da_grid_y 48 -localfunction 0 -cfl_ini 1 -max_st 2000 */

#define EQ

static char help[] = "Nonlinear MHD with multigrid and pusedo timestepping 2d.\n\
  \n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\
The flow can be driven with the lid or with bouyancy or both:\n\
  -viscosity <nu>\n\
  -skin_depth <d_e>\n\
  -larmor_radius <rho_s>\n\
  -contours : draw contour plots of solution\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

/* ------------------------------------------------------------------------

    We thank David E. Keyes for contributing the driven cavity discretization
    within this example code.

    This problem is modeled by the partial differential equation system
  
	               - Lap(ux)     + Grad_y(U)    = 0
		       - Lap(uy)     - Grad_x(U)    = 0
	dU/dt          - nu* Lap(U)  + u * Grad(U)  = 0

    in the unit square, which is uniformly discretized in each of x and
    y in this simple encoding.

    XXX
    No-slip, rigid-wall Dirichlet conditions are used for [U,V].
    Dirichlet conditions are used for Omega, based on the definition of
    vorticity: Omega = - Grad_y(U) + Grad_x(V), where along each
    constant coordinate boundary, the tangential derivative is zero.
    Dirichlet conditions are used for T on the left and right walls,
    and insulation homogeneous Neumann conditions are used for T on
    the top and bottom walls. 

    A finite difference approximation with the usual 5-point stencil 
    is used to discretize the boundary value problem to obtain a 
    nonlinear system of equations.  Upwinding is used for the divergence
    (convective) terms and central for the gradient (source) terms.
    
    The Jacobian can be either
      * formed via finite differencing using coloring (the default), or
      * applied matrix-free via the option -snes_mf 
        (for larger grid problems this variant may not converge 
        without a preconditioner due to ill-conditioning).

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscsles.h   - linear solvers 
*/
#include <stdlib.h>

#include "petscsnes.h"
#include "petscda.h"

#define HERE do { ierr = PetscPrintf(PETSC_COMM_WORLD,"LINE %d (%s)\n", __LINE__, __FUNCTION__);CHKERRQ(ierr); } while (0)
/* 
   User-defined routines and data structures
*/

#define sqr(a) ((a)*(a))

typedef struct {
  PassiveScalar  fnorm_ini,dt_ini;
  PassiveScalar  fnorm,dt;
  PassiveScalar  ptime;
  PassiveScalar  max_time;
  PassiveScalar  fnorm_ratio;
  int          ires,itstep;
  int          max_steps,print_freq;
  int          LocalTimeStepping;                         
  PassiveScalar  t;
} TstepCtx;

typedef struct {
  PassiveScalar ux,uy,U,Bx,By,F;
} PassiveField;

typedef struct {
  PetscScalar ux,uy,U,Bx,By,F;
} Field;


typedef struct {
  int          mglevels;
  int          cycles;         /* numbers of time steps for integration */ 
  PassiveReal  nu,d_e,rho_s;  /* physical parameters */
  PetscTruth   draw_contours;                /* flag - 1 indicates drawing contours */
  PetscTruth   PreLoading;
} Parameter;

typedef struct {
  Vec          Xold,func;
  TstepCtx     *tsCtx;
  Parameter    *param;
} AppCtx;

extern int FormInitialGuess(SNES,Vec,void*);
extern int FormFunction(SNES,Vec,Vec,void*);
extern int FormFunctionLocal(DALocalInfo*,Field**,Field**,void*);
extern int FormFunctionLocali(DALocalInfo*,MatStencil*,Field**,PetscScalar*,void*);
extern int Update(DMMG *);
extern int Initialize(DMMG *);
extern int ComputeTimeStep(SNES,void*);
extern int AddTSTerm(SNES,Vec,Vec,void*);
extern int AddTSTermLocal(DALocalInfo*,Field**,Field**,void*);
extern int Gnuplot(DA da, Vec X, double time);


PetscReal lx = 2.*M_PI;
PetscReal ly = 4.*M_PI;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG       *dmmg;               /* multilevel grid structure */
  AppCtx     *user;                /* user-defined work context */
  TstepCtx   tsCtx;
  Parameter  param;
  int        mx,my;
  int        i,ierr;
  MPI_Comm   comm;
  DA         da;
  PetscTruth localfunction = PETSC_TRUE;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;


  PreLoadBegin(PETSC_TRUE,"SetUp");

    param.PreLoading = PreLoading;
    ierr = DMMGCreate(comm,1,&user,&dmmg);CHKERRQ(ierr);
    param.mglevels = DMMGGetLevels(dmmg);


    /*
      Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
    */
    ierr = DACreate2d(comm,DA_XYPERIODIC,DA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,6,1,0,0,&da);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);

    ierr = DAGetInfo(DMMGGetDA(dmmg),0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
    /* 
     Problem parameters
    */
    param.nu          = 0.0;
    param.rho_s       = 0.0;
    param.d_e         = 0.2;
    ierr = PetscOptionsGetReal(PETSC_NULL,"-viscosity",&param.nu,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-skin_depth",&param.d_e,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-larmor_radius",&param.rho_s,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL,"-contours",&param.draw_contours);CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg),0,"ux");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),1,"uy");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),2,"U");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),3,"Bx");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),4,"By");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),5,"F");CHKERRQ(ierr);

    /*======================================================================*/
    /* Initilize stuff related to time stepping */
    /*======================================================================*/
    tsCtx.fnorm_ini = 0.0;  
    tsCtx.max_steps = 50;   tsCtx.max_time    = 1.0e+12;
    tsCtx.dt        = 0.01; tsCtx.fnorm_ratio = 1.0e+10; tsCtx.t       = 0.;
    tsCtx.LocalTimeStepping = 0;
    ierr = PetscOptionsGetInt(PETSC_NULL,"-max_st",&tsCtx.max_steps,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-ts_rtol",&tsCtx.fnorm_ratio,PETSC_NULL);CHKERRQ(ierr);
    tsCtx.print_freq = tsCtx.max_steps; 
    ierr = PetscOptionsGetInt(PETSC_NULL,"-print_freq",&tsCtx.print_freq,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-deltat",&tsCtx.dt,PETSC_NULL);CHKERRQ(ierr);
    
    ierr = PetscMalloc(param.mglevels*sizeof(AppCtx),&user); CHKERRQ(ierr);
    for (i=0; i<param.mglevels; i++) {
      ierr = VecDuplicate(dmmg[i]->x, &(user[i].Xold)); CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x, &(user[i].func)); CHKERRQ(ierr);
      user[i].tsCtx = &tsCtx;
      user[i].param = &param;
      dmmg[i]->user = &user[i];
    }
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create user context, set problem data, create vector data structures.
       Also, compute the initial guess.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver context
       
       Process adiC: FormFunctionLocal FormFunctionLocali AddTSTermLocal
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscOptionsGetLogical(PETSC_NULL,"-localfunction",&localfunction,PETSC_IGNORE);CHKERRQ(ierr);
    if (localfunction) {
      ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,ad_FormFunctionLocal,admf_FormFunctionLocal);CHKERRQ(ierr);
      ierr = DMMGSetSNESLocali(dmmg,FormFunctionLocali,ad_FormFunctionLocali,admf_FormFunctionLocali);CHKERRQ(ierr);
    } else {
      ierr = DMMGSetSNES(dmmg,FormFunction,0);CHKERRQ(ierr);
    }
    
    ierr = PetscPrintf(comm,"# viscosity = %g, skin_depth # = %g, larmor_radius # = %g\n",
		       param.nu,param.d_e,param.rho_s);CHKERRQ(ierr);
    
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);
    
    PreLoadStage("Solve");
    ierr = Initialize(dmmg); CHKERRQ(ierr);

#if 0
    {
      da = (DA)(dmmg[param.mglevels-1]->dm);
      ierr = Gnuplot(da, ((AppCtx*)dmmg[param.mglevels-1]->user)->Xold); CHKERRQ(ierr);
    }
#endif

    if (param.draw_contours) {
      ierr = VecView(((AppCtx*)dmmg[param.mglevels-1]->user)->Xold,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    }
    ierr = Update(dmmg); CHKERRQ(ierr);

#if 0
    {
      da = (DA)(dmmg[param.mglevels-1]->dm);
      ierr = Gnuplot(da, dmmg[param.mglevels-1]->x); CHKERRQ(ierr);
    }
#endif
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    for (i=0; i<param.mglevels; i++) {
      ierr = VecDestroy(user[i].Xold); CHKERRQ(ierr);
      ierr = VecDestroy(user[i].func); CHKERRQ(ierr);
    }
    ierr = PetscFree(user); CHKERRQ(ierr);
    ierr = DMMGDestroy(dmmg); CHKERRQ(ierr);
    PreLoadEnd();
    
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

#if 1
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Gnuplot"
/* ------------------------------------------------------------------- */
int Gnuplot(DA da, Vec X, double time)
{
  int          i,j,xs,ys,xm,ym;
  int          xints,xinte,yints,yinte;
  int          ierr;
  Field        **x;
  FILE         *f;
  char         fname[100];
  int          cpu;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&cpu);CHKERRQ(ierr);
  sprintf(fname, "out-%g-%d.dat", time, cpu);
  f = fopen(fname, "w");
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  ierr = DAVecGetArray(da,X,(void**)&x);CHKERRQ(ierr);

  xints = xs; xinte = xs+xm; yints = ys; yinte = ys+ym;

  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      fprintf(f, "%d %d %g %g %g %g %g %g\n", i, j, x[j][i].ux, x[j][i].uy, x[j][i].U, x[j][i].Bx, x[j][i].By, x[j][i].F);
    }
    fprintf(f, "\n");
  }
  ierr = DAVecRestoreArray(da,X,(void**)&x);CHKERRQ(ierr);
  fclose(f);
  return 0;
}
#endif

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Initialize"
/* ------------------------------------------------------------------- */
int Initialize(DMMG *dmmg)
{
  AppCtx    *user = (AppCtx*)dmmg[0]->user;
  DA        da;
  /*  TstepCtx  *tsCtx = user->tsCtx; */
  Parameter *param = user->param;
  int       i,j,mx,my,ierr,xs,ys,xm,ym;
  int       mglevel;
  PetscReal d_e,rho_s,de2,dhx,hx,dhy,hy,xx,yy;
  Field     **x;

  da = (DA)(dmmg[param->mglevels-1]->dm);
  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  dhx = mx/lx;     dhy = my/ly;
  hx = 1.0/dhx;    hy = 1.0/dhy;

  d_e = user->param->d_e;
  rho_s = user->param->rho_s;
  de2     = sqr(user->param->d_e);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DAVecGetArray(da,((AppCtx*)dmmg[param->mglevels-1]->user)->Xold,(void**)&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
#if 0
  for (j=ys; j<ys+ym; j++) {
    yy = j * hy;
    for (i=xs; i<xs+xm; i++) {
      xx = i * hx;
      x[j][i].U     = -2.0*cos(xx)*cos(yy);
      x[j][i].ux    = 0;//-cos(xx)*sin(yy);
      x[j][i].uy    = 0;// sin(xx)*cos(yy);
      x[j][i].F     = 0;
      x[j][i].Bx    = 0;
      x[j][i].By    = 0;
    }
  }
#else
  {
    PetscReal eps = lx/ly;
    PetscReal pert = 1e-5;
    PetscReal k = 1.*eps;
/*     PetscReal kappa = sqrt(1-sqr(k)); */
    PetscReal gam; 

    if (d_e < rho_s) d_e = rho_s;
    gam = k * d_e;

/*     cout << "Delta = " << 2 * kappa * tan(kappa*M_PI/2) << endl; */
/*     cout << "Delta d_e = " << 2 * kappa * tan(kappa*M_PI/2) * de << endl; */

    for (j=ys; j<ys+ym; j++) {
      yy = j * hy;
      for (i=xs; i<xs+xm; i++) {
	xx = i * hx;

	if (xx < -M_PI/2) {
	  x[j][i].U = pert * gam / k * erf((xx + M_PI) / (sqrt(2) * d_e)) * (-sin(k*yy));
	} else if (xx < M_PI/2) {
	  x[j][i].U = - pert * gam / k * erf(xx / (sqrt(2) * d_e)) * (-sin(k*yy));
	} else if (xx < 3*M_PI/2){
	  x[j][i].U = pert * gam / k * erf((xx - M_PI) / (sqrt(2) * d_e)) * (-sin(k*yy));
	} else {
	  x[j][i].U = - pert * gam / k * erf((xx - 2.*M_PI) / (sqrt(2) * d_e)) * (-sin(k*yy));
	}
#ifdef EQ
	x[j][i].F = 0.;
#else
	x[j][i].F = (1. + de2) * cos(xx);
#endif
      }
    }
  }
#endif

  /*
     Restore vector
  */
  ierr = DAVecRestoreArray(da,((AppCtx*)dmmg[param->mglevels-1]->user)->Xold,(void**)&x);CHKERRQ(ierr);

  /* Restrict Xold to coarser levels */
  for (mglevel=param->mglevels-1; mglevel>0; mglevel--) {
    ierr = MatRestrict(dmmg[mglevel]->R, ((AppCtx*)dmmg[mglevel]->user)->Xold, ((AppCtx*)dmmg[mglevel-1]->user)->Xold);CHKERRQ(ierr);
    ierr = VecPointwiseMult(dmmg[mglevel]->Rscale,((AppCtx*)dmmg[mglevel-1]->user)->Xold,((AppCtx*)dmmg[mglevel-1]->user)->Xold);CHKERRQ(ierr);
  }
  
  /* Store X in the qold for time stepping */
  /*ierr = VecDuplicate(X,&tsCtx->qold);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&tsCtx->func);CHKERRQ(ierr);
  ierr = VecCopy(X,tsCtx->Xold);CHKERRQ(ierr);
  tsCtx->ires = 0;
  ierr = SNESComputeFunction(snes,tsCtx->Xold,tsCtx->func);
  ierr = VecNorm(tsCtx->func,NORM_2,&tsCtx->fnorm_ini);CHKERRQ(ierr);
  tsCtx->ires = 1;
  PetscPrintf(PETSC_COMM_WORLD,"Initial function norm is %g\n",tsCtx->fnorm_ini);*/
  return 0;
} 
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
int FormInitialGuess(SNES snes,Vec X,void *ptr)
{
  DMMG      dmmg = (DMMG)ptr;
  AppCtx    *user = (AppCtx*)dmmg->user;
  TstepCtx  *tsCtx = user->tsCtx;
  int       ierr;
  ierr = VecCopy(user->Xold, X); CHKERRQ(ierr);

  /* calculate the residual on fine mesh */
  if (user->tsCtx->fnorm_ini == 0.0) {
    tsCtx->ires = 0;
    ierr = SNESComputeFunction(snes,user->Xold,user->func);
    ierr = VecNorm(user->func,NORM_2,&tsCtx->fnorm_ini);CHKERRQ(ierr);
    tsCtx->ires = 1;
  }

  return 0;
} 

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
/* 
   FormFunction - Evaluates the nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector

   Notes:
   We process the boundary nodes before handling the interior
   nodes, so that no conditional statements are needed within the
   PetscReal loop over the local grid indices. 
 */
int FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG         dmmg = (DMMG)ptr;
  AppCtx       *user = (AppCtx*)dmmg->user;
  TstepCtx     *tsCtx = user->tsCtx;
  int          ierr,i,j,mx,my,xs,ys,xm,ym;
  int          xints,xinte,yints,yinte;
  PetscReal    two = 2.0,one = 1.0,p5 = 0.5,hx,hy,dhx,dhy,hxdhy,hydhx,hxhy;
  PetscReal    rho_s,nu,de2;
  PetscScalar  u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;
  PetscScalar  Bx,By,aBx,aBy,Bxp,Bxm,Byp,Bym;
  Field        **x,**f;
  Vec          localX;
  DA           da = (DA)dmmg->dm;

  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  de2     = sqr(user->param->d_e);
  rho_s   = user->param->rho_s;
  nu      = user->param->nu;

  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
  */
  dhx = mx/lx;              dhy = my/ly;
  hx = one/dhx;             hy = one/dhy;
  hxdhy = hx*dhy;           hydhx = hy*dhx;
  hxhy = hx*hy;

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DAVecGetArray((DA)dmmg->dm,localX,(void**)&x);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,F,(void**)&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
     (physical corner points are set twice to avoid more conditionals).
  */
  xints = xs; xinte = xs+xm; yints = ys; yinte = ys+ym;

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
#ifdef EQ
      PetscReal xx = i * hx;
      PetscScalar F_eq_x = - (1. + de2) * sin(xx);
#else
      PetscScalar F_eq_x = 0.;
#endif
	/*
	  convective coefficients for upwinding
        */
	vx = x[j][i].ux; avx = PetscAbsScalar(vx);
        vxp = p5*(vx+avx); vxm = p5*(vx-avx);
	vy = x[j][i].uy; avy = PetscAbsScalar(vy);
        vyp = p5*(vy+avy); vym = p5*(vy-avy);
	vxp = vxm = p5*vx;
	vyp = vym = p5*vy;

	Bx = x[j][i].Bx; aBx = PetscAbsScalar(Bx);
        Bxp = p5*(Bx+aBx); Bxm = p5*(Bx-aBx);
#ifdef EQ
	By = x[j][i].By + sin(xx); aBy = PetscAbsScalar(By);
#else
	By = x[j][i].By; aBy = PetscAbsScalar(By);
#endif
        Byp = p5*(By+aBy); Bym = p5*(By-aBy);
	Bxp = Bxm = p5*Bx;
	Byp = Bym = p5*By;

	/* ux velocity */
        u          = x[j][i].ux;
        uxx        = -(two*u - x[j][i-1].ux - x[j][i+1].ux)*hydhx;
        uyy        = -(two*u - x[j-1][i].ux - x[j+1][i].ux)*hxdhy;
        f[j][i].ux = -(uxx + uyy) + p5*(x[j+1][i].U-x[j-1][i].U)*hx;

	/* uy velocity */
        u          = x[j][i].uy;
        uxx        = -(two*u - x[j][i-1].uy - x[j][i+1].uy)*hydhx;
        uyy        = -(two*u - x[j-1][i].uy - x[j+1][i].uy)*hxdhy;
        f[j][i].uy = -(uxx + uyy) - p5*(x[j][i+1].U-x[j][i-1].U)*hy;

	/* U */
        u          = x[j][i].U;
        uxx        = -(two*u - x[j][i-1].U - x[j][i+1].U)*hydhx;
        uyy        = -(two*u - x[j-1][i].U - x[j+1][i].U)*hxdhy;
	f[j][i].U  = - nu * (uxx + uyy) +
	  ((vxp*(u - x[j][i-1].U) +
	    vxm*(x[j][i+1].U - u)) * hy +
	   (vyp*(u - x[j-1][i].U) +
	    vym*(x[j+1][i].U - u)) * hx) -
	  ((Bxp*(u - x[j][i-1].F + F_eq_x * hx) +
	    Bxm*(x[j][i+1].F - u + F_eq_x * hx)) * hy +
	   (Byp*(u - x[j-1][i].F) +
	    Bym*(x[j+1][i].F - u)) * hx)/de2;
	

	/* Bx */
        u          = x[j][i].Bx;
        uxx        = -(two*u - x[j][i-1].Bx - x[j][i+1].Bx)*hydhx;
        uyy        = -(two*u - x[j-1][i].Bx - x[j+1][i].Bx)*hxdhy;
        f[j][i].Bx = -u*hxhy + de2 * (uxx + uyy) + p5*(x[j+1][i].F-x[j-1][i].F)*hx;

	/* By */
        u          = x[j][i].By;
        uxx        = -(two*u - x[j][i-1].By - x[j][i+1].By)*hydhx;
        uyy        = -(two*u - x[j-1][i].By - x[j+1][i].By)*hxdhy;
        f[j][i].By = -u*hxhy + de2 * (uxx + uyy) + p5*(x[j][i+1].F-x[j][i-1].F)*hy;

	/* F */
        u          = x[j][i].F;
        uxx        = -(two*u - x[j][i-1].F - x[j][i+1].F)*hydhx;
        uyy        = -(two*u - x[j-1][i].F - x[j+1][i].F)*hxdhy;
	f[j][i].F  = - nu * (uxx + uyy)/de2 +  // not quite right
#if 1
			(vxp*(u - x[j][i-1].F + F_eq_x * hx) +
			 vxm*(x[j][i+1].F - u + F_eq_x * hx)) * hy +
			(vyp*(u - x[j-1][i].F) +
			 vym*(x[j+1][i].F - u)) * hx;
#endif
/* 	f[j][i].ux = x[j][i].ux; */
/* 	f[j][i].uy = x[j][i].uy; */
/* 	f[j][i].Bx = x[j][i].Bx; */
/* 	f[j][i].By = x[j][i].By; */
/*   	f[j][i].F = x[j][i].F - (1+de2)*cos(xx); */
/* 	f[j][i].U = x[j][i].U; */
    }
  }

  /*
     Restore vectors
  */
  ierr = DAVecRestoreArray((DA)dmmg->dm,localX,(void**)&x);CHKERRQ(ierr);
  ierr = DAVecRestoreArray((DA)dmmg->dm,F,(void**)&f);CHKERRQ(ierr);

  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);

  /* Add time step contribution */
  if (tsCtx->ires) {
    ierr = AddTSTerm(snes,X,F,ptr); CHKERRQ(ierr);
  } else {
    PetscReal norm[6] = { };

    ierr = DAVecGetArray((DA)dmmg->dm,user->Xold,(void**)&x);CHKERRQ(ierr);
    for (j=yints; j<yinte; j++) {
      for (i=xints; i<xinte; i++) {
#define max(a,b) ((a)>(b)?(a):(b))

	norm[0] = max(norm[0],x[j][i].ux);
	norm[1] = max(norm[1],x[j][i].uy);
	norm[2] = max(norm[2],x[j][i].U);
	norm[3] = max(norm[3],x[j][i].Bx);
	norm[4] = max(norm[4],x[j][i].By);
	norm[5] = max(norm[5],x[j][i].F);
      }
    }
    ierr = DAVecRestoreArray((DA)dmmg->dm,user->Xold,(void**)&x);CHKERRQ(ierr);
    fprintf(stderr, "%g\t%g\t%g\t%g\t%g\t%g\t%g\n", tsCtx->t, norm[0],
	    norm[1], norm[2], norm[3], norm[4], norm[5]);
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84*ym*xm);CHKERRQ(ierr);


  return 0; 
} 

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
int FormFunctionLocal(DALocalInfo *info,Field **x,Field **f,void *ptr)
 {
  AppCtx       *user = (AppCtx*)ptr;
  TstepCtx     *tsCtx = user->tsCtx;
  int          ierr,i,j;
  int          xints,xinte,yints,yinte;
  PetscReal    hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal    d_e,rho_s,nu;
  PetscScalar  u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  exit(1);

  PetscFunctionBegin;
  d_e     = user->param->d_e;
  rho_s   = user->param->rho_s;
  nu      = user->param->nu;

  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
  */
  dhx = (PetscReal)(info->mx);  dhy = (PetscReal)(info->my);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
        f[j][i].ux    = x[j][i].ux;
        f[j][i].uy    = x[j][i].uy;
        f[j][i].U     = x[j][i].U + (x[j+1][i].ux - x[j][i].ux)*dhy; 
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
        f[j][i].ux    = x[j][i].ux;
        f[j][i].uy    = x[j][i].uy;
        f[j][i].U     = x[j][i].U + (x[j][i].ux - x[j-1][i].ux)*dhy; 
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].ux    = x[j][i].ux;
      f[j][i].uy    = x[j][i].uy;
      f[j][i].U     = x[j][i].U - (x[j][i+1].uy - x[j][i].uy)*dhx; 
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */ 
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].ux    = x[j][i].ux;
      f[j][i].uy    = x[j][i].uy;
      f[j][i].U     = x[j][i].U - (x[j][i].uy - x[j][i-1].uy)*dhx; 
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {

	/*
	  convective coefficients for upwinding
        */
	vx = x[j][i].ux; avx = PetscAbsScalar(vx);
        vxp = .5*(vx+avx); vxm = .5*(vx-avx);
	vy = x[j][i].uy; avy = PetscAbsScalar(vy);
        vyp = .5*(vy+avy); vym = .5*(vy-avy);

	/* U velocity */
        u          = x[j][i].ux;
        uxx        = (2.0*u - x[j][i-1].ux - x[j][i+1].ux)*hydhx;
        uyy        = (2.0*u - x[j-1][i].ux - x[j+1][i].ux)*hxdhy;
        f[j][i].ux  = uxx + uyy - .5*(x[j+1][i].U-x[j-1][i].U)*hx;

	/* V velocity */
        u          = x[j][i].uy;
        uxx        = (2.0*u - x[j][i-1].uy - x[j][i+1].uy)*hydhx;
        uyy        = (2.0*u - x[j-1][i].uy - x[j+1][i].uy)*hxdhy;
        f[j][i].uy  = uxx + uyy + .5*(x[j][i+1].U-x[j][i-1].U)*hy;

	/* Omega */
        u          = x[j][i].U;
        uxx        = (2.0*u - x[j][i-1].U - x[j][i+1].U)*hydhx;
        uyy        = (2.0*u - x[j-1][i].U - x[j+1][i].U)*hxdhy;
	f[j][i].U = uxx + uyy + 
			(vxp*(u - x[j][i-1].U) +
			 vxm*(x[j][i+1].U - u)) * hy +
			(vyp*(u - x[j-1][i].U) +
			 vym*(x[j+1][i].U - u)) * hx;

    }
  }

  /* Add time step contribution */
  if (tsCtx->ires) {
    ierr = AddTSTermLocal(info,x,f,ptr); CHKERRQ(ierr);
  }
  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocali"
int FormFunctionLocali(DALocalInfo *info,MatStencil *st,Field **x,PetscScalar *f,void *ptr)
 {
  AppCtx      *user = (AppCtx*)ptr;
  int         i,j,c;
  PassiveReal hx,hy,dhx,dhy,hxdhy,hydhx;
  PassiveReal d_e,rho_s,nu;
  PetscScalar u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  exit(1);
  PetscFunctionBegin;
  d_e     = user->param->d_e;  
  rho_s   = user->param->rho_s;
  nu      = user->param->nu;

  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
  */
  dhx = (PetscReal)(info->mx);     dhy = (PetscReal)(info->my);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  i = st->i; j = st->j; c = st->c;

  /* Test whether we are on the right edge of the global array */
  if (i == info->mx-1) {
    if (c == 0) *f     = x[j][i].ux;
    else if (c == 1) *f     = x[j][i].uy;
    else if (c == 2) *f = x[j][i].U - (x[j][i].uy - x[j][i-1].uy)*dhx; 

  /* Test whether we are on the left edge of the global array */
  } else if (i == 0) {
    if (c == 0) *f     = x[j][i].ux;
    else if (c == 1) *f     = x[j][i].uy;
    else if (c == 2) *f = x[j][i].U - (x[j][i+1].uy - x[j][i].uy)*dhx; 

  /* Test whether we are on the top edge of the global array */
  } else if (j == info->my-1) {
    if (c == 0) *f     = x[j][i].ux;
    else if (c == 1) *f     = x[j][i].uy;
    else if (c == 2) *f = x[j][i].U + (x[j][i].ux - x[j-1][i].ux)*dhy; 

  /* Test whether we are on the bottom edge of the global array */
  } else if (j == 0) {
    if (c == 0) *f     = x[j][i].ux;
    else if (c == 1) *f     = x[j][i].uy;
    else if (c == 2) *f = x[j][i].U + (x[j+1][i].ux - x[j][i].ux)*dhy; 

  /* Compute over the interior points */
  } else {
    /*
      convective coefficients for upwinding
    */
    vx = x[j][i].ux; avx = PetscAbsScalar(vx);
    vxp = .5*(vx+avx); vxm = .5*(vx-avx);
    vy = x[j][i].uy; avy = PetscAbsScalar(vy);
    vyp = .5*(vy+avy); vym = .5*(vy-avy);

    /* U velocity */
    if (c == 0) {
      u          = x[j][i].ux;
      uxx        = -(2.0*u - x[j][i-1].ux - x[j][i+1].ux)*hydhx;
      uyy        = -(2.0*u - x[j-1][i].ux - x[j+1][i].ux)*hxdhy;
      *f         = -(uxx + uyy) + .5*(x[j+1][i].U-x[j-1][i].U)*hx;

    /* V velocity */
    } else if (c == 1) {
      u          = x[j][i].uy;
      uxx        = -(2.0*u - x[j][i-1].uy - x[j][i+1].uy)*hydhx;
      uyy        = -(2.0*u - x[j-1][i].uy - x[j+1][i].uy)*hxdhy;
      *f         = -(uxx + uyy) - .5*(x[j][i+1].U-x[j][i-1].U)*hy;
    
    /* Omega */
    } else if (c == 2) {
      u          = x[j][i].U;
      uxx        = (2.0*u - x[j][i-1].U - x[j][i+1].U)*hydhx;
      uyy        = (2.0*u - x[j-1][i].U - x[j+1][i].U)*hxdhy;
      *f         = uxx + uyy + 
	(vxp*(u - x[j][i-1].U) +
	 vxm*(x[j][i+1].U - u)) * hy +
	(vyp*(u - x[j-1][i].U) +
	 vym*(x[j+1][i].U - u)) * hx;
    }
  }

  PetscFunctionReturn(0);
} 


/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "Update"
int Update(DMMG *dmmg)
/*---------------------------------------------------------------------*/
{
 
 AppCtx         *user = (AppCtx *) ((dmmg[0])->user);
 TstepCtx 	*tsCtx = user->tsCtx;
 Parameter      *param = user->param;
 SNES           snes;
 int 		ierr,its;
 PetscScalar 	fratio;
 PetscScalar 	time1,time2,cpuloc = 0.;
 int 		max_steps;
 PetscTruth     print_flag = PETSC_FALSE;
 int		nfailsCum = 0,nfails = 0;

  PetscFunctionBegin;

  ierr = PetscOptionsHasName(PETSC_NULL,"-print",&print_flag);CHKERRQ(ierr);
  if (user->param->PreLoading) 
   max_steps = 1;
  else
   max_steps = tsCtx->max_steps;
  fratio = 1.0;
  
  ierr = PetscGetTime(&time1);CHKERRQ(ierr);
  for (tsCtx->itstep = 0; tsCtx->itstep < max_steps; tsCtx->itstep++) {
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n", its);CHKERRQ(ierr);
    ierr = SNESGetNumberUnsuccessfulSteps(snes,&nfails);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of unsuccessfull = %d\n", nfails);CHKERRQ(ierr);
    nfailsCum += nfails; nfails = 0;
    if (nfailsCum >= 2) SETERRQ(1,"Unable to find a Newton Step");
    /*tsCtx->qcur = DMMGGetx(dmmg);
      ierr = VecCopy(tsCtx->qcur,tsCtx->qold);CHKERRQ(ierr);*/

    ierr = SNESComputeFunction(snes,dmmg[param->mglevels-1]->x,user->func);
    ierr = VecNorm(user->func,NORM_2,&tsCtx->fnorm);CHKERRQ(ierr);
    //    PetscPrintf(PETSC_COMM_WORLD, "%g\t%g\n", tsCtx->t, tsCtx->fnorm);
    
    ierr = VecCopy(dmmg[param->mglevels-1]->x, ((AppCtx*)dmmg[param->mglevels-1]->user)->Xold); CHKERRQ(ierr);
    for (its=param->mglevels-1; its>0 ;its--) {
      ierr = MatRestrict(dmmg[its]->R, ((AppCtx*)dmmg[its]->user)->Xold, ((AppCtx*)dmmg[its-1]->user)->Xold);CHKERRQ(ierr);
      ierr = VecPointwiseMult(dmmg[its]->Rscale,((AppCtx*)dmmg[its-1]->user)->Xold,((AppCtx*)dmmg[its-1]->user)->Xold);CHKERRQ(ierr);
    }
    
    tsCtx->t += tsCtx->dt;
    ierr = ComputeTimeStep(snes,((AppCtx*)dmmg[param->mglevels-1]->user));CHKERRQ(ierr);
    fratio = tsCtx->fnorm_ini/tsCtx->fnorm;
    ierr = PetscGetTime(&time2);CHKERRQ(ierr);
    cpuloc = time2-time1;            
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    if (print_flag) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"After Time Step %d and fnorm = %g\n",
			 tsCtx->itstep,tsCtx->fnorm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Wall clock time needed %g seconds for %d time steps\n",
			 cpuloc,tsCtx->itstep+1);CHKERRQ(ierr);    
    }
    if (param->draw_contours && !param->PreLoading) {
      ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    }
    ierr = Gnuplot((DA) dmmg[param->mglevels-1]->dm,
		   dmmg[param->mglevels-1]->x, tsCtx->t);
  } /* End of time step loop */
  
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Total wall clock time needed %g seconds for %d time steps\n",
		     cpuloc,tsCtx->itstep);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"fnorm = %g\n",tsCtx->fnorm);CHKERRQ(ierr);
  if (user->param->PreLoading) {
    tsCtx->fnorm_ini = 0.0;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Preloading done ...\n");CHKERRQ(ierr);
  }
  /*
  {
    Vec xx,yy;
    PetscScalar fnorm,fnorm1;
    ierr = SNESGetFunctionNorm(snes,&fnorm); CHKERRQ(ierr);
    xx = DMMGGetx(dmmg);
    ierr = VecDuplicate(xx,&yy);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes,xx,yy);
    ierr = VecNorm(yy,NORM_2,&fnorm1);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"fnorm = %g, fnorm1 = %g\n",fnorm,fnorm1);
    
  }
  */

  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "ComputeTimeStep"
int ComputeTimeStep(SNES snes,void *ptr)
/*---------------------------------------------------------------------*/
{
  AppCtx       *user = (AppCtx*)ptr;
  TstepCtx     *tsCtx = user->tsCtx;
  Vec	       func = user->func;
  int          ierr;
 
  PetscFunctionBegin; 
  tsCtx->ires = 0;
  ierr = SNESComputeFunction(snes,user->Xold,user->func);
  tsCtx->ires = 1;
  ierr = VecNorm(func,NORM_2,&tsCtx->fnorm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AddTSTerm"
int AddTSTerm(SNES snes,Vec X,Vec F,void *ptr)
/*---------------------------------------------------------------------*/
{
  DMMG         dmmg = (DMMG)ptr;
  AppCtx       *user = (AppCtx*)dmmg->user;
  TstepCtx     *tsCtx = user->tsCtx;
  DA           da = (DA)dmmg->dm;
  int          ierr,i,j,mx,my,xs,ys,xm,ym;
  int          xints,xinte,yints,yinte;
  PetscReal    one = 1.0,hx,hy,dhx,dhy,hxhy;
  PetscScalar  dtinv;
  Field        **x,**f;
  PassiveField **xold;

  PetscFunctionBegin;
  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  xints = xs; xinte = xs+xm; yints = ys; yinte = ys+ym;
  dhx = mx/lx;              dhy = my/ly;
  hx = one/dhx;             hy = one/dhy;
  hxhy = hx*hy;
  ierr = DAVecGetArray(da,X,(void**)&x);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,F,(void**)&f);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,user->Xold,(void**)&xold);CHKERRQ(ierr);
  dtinv = hxhy/(tsCtx->dt);
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      f[j][i].U += dtinv*(x[j][i].U-xold[j][i].U);
      f[j][i].F += dtinv*(x[j][i].F-xold[j][i].F);
    }
  }
  ierr = DAVecRestoreArray(da,X,(void**)&x);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,F,(void**)&f);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,user->Xold,(void**)&xold);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AddTSTermLocal"
int AddTSTermLocal(DALocalInfo* info,Field **x,Field **f,void *ptr)
/*---------------------------------------------------------------------*/
{
  AppCtx       *user = (AppCtx*)ptr;
  TstepCtx     *tsCtx = user->tsCtx;
  DA           da = info->da;
  int          ierr,i,j;
  int          xints,xinte,yints,yinte;
  PetscReal    hx,hy,dhx,dhy,hxhy;
  PassiveScalar  dtinv;
  PassiveField **xold;
  PetscFunctionBegin; 
  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;
  dhx = (PetscReal)(info->mx);  dhy = (PetscReal)(info->my);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxhy = hx*hy;
  ierr = DAVecGetArray(da,user->Xold,(void**)&xold);CHKERRQ(ierr);
  dtinv = hxhy/(tsCtx->dt);
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      f[j][i].U += dtinv*(x[j][i].U-xold[j][i].U);
    }
  }
  ierr = DAVecRestoreArray(da,user->Xold,(void**)&xold);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




