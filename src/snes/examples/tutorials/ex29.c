/*$Id: $*/

/*#define HAVE_DA_HDF*/

/* solve the equations for the perturbed magnetic field only */
#define EQ 

/* turning this on causes instability?!? */
/* #define UPWINDING  */

static char help[] = "Hall MHD with in two dimensions with time stepping and multigrid.\n\n\
-da_grid_x 5 -da_grid_y 5 -dmmg_nlevels 3\n\
-mg_coarse_pc_lu_damping\n\
-mg_levels_pc_type sor -mg_levels_pc_sor_symmetric\n\
-mg_levels_ksp_type richardson\n\
-mg_coarse_pc_type sor -mg_coarse_pc_sor_symmetric\n\
-snes_atol 1.e-10\n\
-ksp_atol 1.e-10\n\
-max_st 5\n\
-resistivity <eta>\n\
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

    We thank Kai Germaschewski for contributing the FormFunctionLocal()

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  
   Include "petscmg.h" to control the multigrid solvers. 
   Note that these automatically include:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscsles.h   - linear solvers 
*/
#include "petscsnes.h"
#include "petscda.h"
#include "petscmg.h"

#ifdef HAVE_DA_HDF
int DAVecHDFOutput(DA da, Vec X, char *fname);
#endif

#define D_x(x,m,i,j)  (p5 * (x[(j)][(i)+1].m - x[(j)][(i)-1].m) * dhx)
#define D_xm(x,m,i,j) ((x[(j)][(i)].m - x[(j)][(i)-1].m) * dhx)
#define D_xp(x,m,i,j) ((x[(j)][(i+1)].m - x[(j)][(i)].m) * dhx)
#define D_y(x,m,i,j)  (p5 * (x[(j)+1][(i)].m - x[(j)-1][(i)].m) * dhy)
#define D_ym(x,m,i,j) ((x[(j)][(i)].m - x[(j)-1][(i)].m) * dhy)
#define D_yp(x,m,i,j) ((x[(j)+1][(i)].m - x[(j)][(i)].m) * dhy)
#define D_xx(x,m,i,j) ((x[(j)][(i)+1].m - two*x[(j)][(i)].m + x[(j)][(i)-1].m) * hydhx * dhxdhy)
#define D_yy(x,m,i,j) ((x[(j)+1][(i)].m - two*x[(j)][(i)].m + x[(j)-1][(i)].m) * hxdhy * dhxdhy)
#define Lapl(x,m,i,j) (D_xx(x,m,i,j) + D_yy(x,m,i,j))
#define lx            (2.*M_PI)
#define ly            (4.*M_PI)
#define sqr(a)        ((a)*(a))

/* 
   User-defined routines and data structures
*/

typedef struct {
  PassiveScalar  fnorm_ini,dt_ini;
  PassiveScalar  fnorm,dt,dt_out;
  PassiveScalar  ptime;
  PassiveScalar  max_time;
  PassiveScalar  fnorm_ratio;
  int            ires,itstep;
  int            max_steps,print_freq;
  PassiveScalar  t;

  PetscTruth     ts_monitor;           /* print information about each time step */
  PetscReal      dump_time;            /* time to dump solution to a file */
  PetscViewer    socketviewer;         /* socket to dump solution at each timestep for visualization */
} TstepCtx;

typedef struct {
  PetscScalar phi,psi,U,F;
} Field;

typedef struct {
  PassiveScalar phi,psi,U,F;
} PassiveField;

typedef struct {
  int          mglevels;
  int          cycles;           /* number of time steps for integration */ 
  PassiveReal  nu,eta,d_e,rho_s; /* physical parameters */
  PetscTruth   draw_contours;    /* flag - 1 indicates drawing contours */
  PetscTruth   second_order;
  PetscTruth   PreLoading;
} Parameter;

typedef struct {
  Vec          Xoldold,Xold,func;
  TstepCtx     *tsCtx;
  Parameter    *param;
} AppCtx;

extern int DAGetMatrix_Specialized(DA,MatType,Mat*);
extern int FormFunctionLocal(DALocalInfo*,Field**,Field**,void*);
extern int Update(DMMG *);
extern int Initialize(DMMG *);
extern int AddTSTermLocal(DALocalInfo* info,Field **x,Field **f,AppCtx *user);
extern int AddTSTermLocal2(DALocalInfo* info,Field **x,Field **f,AppCtx *user);
extern int Gnuplot(DA da, Vec X, double time);
extern int AttachNullSpace(PC,Vec);
extern int FormFunctionLocali(DALocalInfo*,MatStencil*,Field**,PetscScalar*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG       *dmmg;       /* multilevel grid structure */
  AppCtx     *user;       /* user-defined work context (one for each level) */
  TstepCtx   tsCtx;       /* time-step parameters (one total) */
  Parameter  param;       /* physical parameters (one total) */
  int        i,ierr,m,n,mx,my;
  MPI_Comm   comm;
  DA         da;
  PetscTruth defaultnonzerostructure = PETSC_FALSE,flg;
  PetscReal  dt_ratio;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;


  PreLoadBegin(PETSC_TRUE,"SetUp");

  param.PreLoading = PreLoading;
    ierr = DMMGCreate(comm,1,&user,&dmmg);CHKERRQ(ierr);
    param.mglevels = DMMGGetLevels(dmmg);

    /*
      Create distributed array multigrid object (DMMG) to manage
      parallel grid and vectors for principal unknowns (x) and
      governing residuals (f)
    */
    ierr = DACreate2d(comm, DA_XYPERIODIC, DA_STENCIL_STAR, -5, -5,
                      PETSC_DECIDE, PETSC_DECIDE, 4, 1, 0, 0, &da);
    CHKERRQ(ierr);

    /* overwrite the matrix allocation routine with one specific for
       this codes nonzero structure */
    ierr = PetscOptionsHasName(PETSC_NULL, "-default_nonzero_structure",
                               &defaultnonzerostructure);
    CHKERRQ(ierr);
    if (!defaultnonzerostructure) {
      ierr = DASetGetMatrix(da,DAGetMatrix_Specialized);CHKERRQ(ierr);
    }

    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);

    /* default physical parameters */
    param.nu    = 0;
    param.eta   = 1e-3;
    param.d_e   = 0.2;
    param.rho_s = 0;

    ierr = PetscOptionsGetReal(PETSC_NULL, "-viscosity", &param.nu,
                               PETSC_NULL);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetReal(PETSC_NULL, "-resistivity", &param.eta,
                               PETSC_NULL);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetReal(PETSC_NULL, "-skin_depth", &param.d_e,
                               PETSC_NULL);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetReal(PETSC_NULL, "-larmor_radius", &param.rho_s,
                               PETSC_NULL);
    CHKERRQ(ierr);

    ierr = PetscOptionsHasName(PETSC_NULL, "-contours", &param.draw_contours);
    CHKERRQ(ierr);

    ierr = PetscOptionsHasName(PETSC_NULL, "-second_order", &param.second_order);
    CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg), 0, "phi");
    CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg), 1, "psi");
    CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg), 2, "U");
    CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg), 3, "F");
    CHKERRQ(ierr);

    /*======================================================================*/
    /* Initialize stuff related to time stepping */
    /*======================================================================*/

    ierr = DAGetInfo(DMMGGetDA(dmmg),0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

    tsCtx.fnorm_ini   = 0;  
    tsCtx.max_steps   = 1000000;   
    tsCtx.max_time    = 1.0e+12;
    /* use for dt = min(dx,dy); multiplied by dt_ratio below */
    tsCtx.dt          = PetscMin(lx/mx,ly/my);  
    tsCtx.fnorm_ratio = 1.0e+10;
    tsCtx.t           = 0;
    tsCtx.dt_out      = 10;
    tsCtx.print_freq  = tsCtx.max_steps; 
    tsCtx.ts_monitor  = PETSC_FALSE;
    tsCtx.dump_time   = -1.0;

    ierr = PetscOptionsGetInt(PETSC_NULL, "-max_st", &tsCtx.max_steps,
                              PETSC_NULL);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetReal(PETSC_NULL, "-ts_rtol", &tsCtx.fnorm_ratio,
                               PETSC_NULL);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetInt(PETSC_NULL, "-print_freq", &tsCtx.print_freq,
                              PETSC_NULL);
    CHKERRQ(ierr);

    dt_ratio = 1.0;
    ierr = PetscOptionsGetReal(PETSC_NULL, "-dt_ratio", &dt_ratio,
                                 PETSC_NULL);
    CHKERRQ(ierr);
    tsCtx.dt *= dt_ratio;


    ierr = PetscOptionsHasName(PETSC_NULL, "-ts_monitor", &tsCtx.ts_monitor);
    CHKERRQ(ierr);

    ierr = PetscOptionsGetReal(PETSC_NULL, "-dump", &tsCtx.dump_time,
                               PETSC_NULL);
    CHKERRQ(ierr);

    tsCtx.socketviewer = 0;
    ierr = PetscOptionsHasName(PETSC_NULL, "-socket_viewer", &flg);
    CHKERRQ(ierr);
    if (flg && !PreLoading) {
      ierr = PetscViewerSocketOpen(PETSC_COMM_WORLD,0,PETSC_DECIDE,&tsCtx.socketviewer);CHKERRQ(ierr);
    }
 

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create user context, set problem data, create vector data structures.
       Also, compute the initial guess.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* create application context for each level */    
    ierr = PetscMalloc(param.mglevels*sizeof(AppCtx),&user); CHKERRQ(ierr);
    for (i=0; i<param.mglevels; i++) {
      /* create work vectors to hold previous time-step solution and
         function value */
      ierr = VecDuplicate(dmmg[i]->x, &user[i].Xoldold); CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x, &user[i].Xold); CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x, &user[i].func); CHKERRQ(ierr);
      user[i].tsCtx = &tsCtx;
      user[i].param = &param;
      dmmg[i]->user = &user[i];
    }
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver context
       
       Process adiC(20):  AddTSTermLocal FormFunctionLocal
       Process adiC(20):  AddTSTermLocal AddTSTermLocal2 FormFunctionLocal
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMMGSetSNESLocal(dmmg, FormFunctionLocal, 0,
                            ad_FormFunctionLocal, admf_FormFunctionLocal);
    CHKERRQ(ierr);
    /*    ierr = DMMGSetSNESLocali(dmmg,FormFunctionLocali,ad_FormFunctionLocali,admf_FormFunctionLocali);CHKERRQ(ierr);*/

    /* attach nullspace to each level of the preconditioner */
    {
      SLES       subsles,sles;
      PC         pc,subpc;
      PetscTruth mg;

      ierr = SNESGetSLES(DMMGGetSNES(dmmg),&sles);CHKERRQ(ierr);
      ierr = SLESGetPC(sles,&pc);
      ierr = AttachNullSpace(pc,DMMGGetx(dmmg));CHKERRQ(ierr);
      ierr = PetscTypeCompare((PetscObject)pc,PCMG,&mg);CHKERRQ(ierr);
      if (mg) {
        for (i=0; i<param.mglevels; i++) {
	  ierr = MGGetSmoother(pc,i,&subsles);CHKERRQ(ierr);
	  ierr = SLESGetPC(subsles,&subpc);CHKERRQ(ierr);
	  ierr = AttachNullSpace(subpc,dmmg[i]->x);CHKERRQ(ierr);
        }
      }
    }
    
    if (PreLoading) {
      ierr = PetscPrintf(comm, "# viscosity = %g, resistivity = %g, "
			 "skin_depth # = %g, larmor_radius # = %g\n",
			 param.nu, param.eta, param.d_e, param.rho_s);
      CHKERRQ(ierr);
      ierr = DAGetInfo(DMMGGetDA(dmmg),0,&m,&n,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"Problem size %d by %d\n",m,n);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"dx %g dy %g dt %g ratio dt/min(dx,dy) %g\n",lx/mx,ly/my,tsCtx.dt,dt_ratio);CHKERRQ(ierr);
    }

    
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    PreLoadStage("Solve");

    if (param.draw_contours) {
      ierr = VecView(((AppCtx*)dmmg[param.mglevels-1]->user)->Xold,
                     PETSC_VIEWER_DRAW_WORLD);
      CHKERRQ(ierr);
    }

    ierr = Update(dmmg);
    CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    for (i=0; i<param.mglevels; i++) {
      ierr = VecDestroy(user[i].Xoldold); CHKERRQ(ierr);
      ierr = VecDestroy(user[i].Xold); CHKERRQ(ierr);
      ierr = VecDestroy(user[i].func); CHKERRQ(ierr);
    }
    ierr = PetscFree(user); CHKERRQ(ierr);
    ierr = DMMGDestroy(dmmg); CHKERRQ(ierr);

    PreLoadEnd();
    
  ierr = PetscFinalize();
  CHKERRQ(ierr);

  return 0;
}

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

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&cpu);
  CHKERRQ(ierr);

  sprintf(fname, "out-%g-%d.dat", time, cpu);

  f = fopen(fname, "w");

  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);
  CHKERRQ(ierr);

  ierr = DAVecGetArray(da,X,(void**)&x);
  CHKERRQ(ierr);

  xints = xs; xinte = xs+xm; yints = ys; yinte = ys+ym;

  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      ierr = PetscFPrintf(PETSC_COMM_WORLD, f,
                          "%d %d %g %g %g %g %g %g\n",
                          i, j, 0.0, 0.0,
                          x[j][i].U, x[j][i].F, x[j][i].phi, x[j][i].psi);
      CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,f, "\n");
    CHKERRQ(ierr);
  }

  ierr = DAVecRestoreArray(da,X,(void**)&x);
  CHKERRQ(ierr);

  fclose(f);

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Initialize"
/* ------------------------------------------------------------------- */
int Initialize(DMMG *dmmg)
{
  AppCtx     *appCtx = (AppCtx*)dmmg[0]->user;
  Parameter  *param = appCtx->param;
  DA         da;
  int        i,j,mx,my,ierr,xs,ys,xm,ym;
  PetscReal  two = 2.0,one = 1.0;
  PetscReal  hx,hy,dhx,dhy,hxdhy,hydhx,hxhy,dhxdhy;
  PetscReal  d_e,rho_s,de2,xx,yy;
  Field      **x, **localx;
  Vec        localX;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(0,"-restart",&flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"binaryoutput",PETSC_BINARY_RDONLY,&viewer);CHKERRQ(ierr);
    ierr = VecLoadIntoVector(viewer,dmmg[param->mglevels-1]->x);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  d_e   = param->d_e;
  rho_s = param->rho_s;
  de2   = sqr(param->d_e);

  da   = (DA)(dmmg[param->mglevels-1]->dm);
  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  dhx   = mx/lx;              dhy = my/ly;
  hx    = one/dhx;             hy = one/dhy;
  hxdhy = hx*dhy;           hydhx = hy*dhx;
  hxhy  = hx*hy;           dhxdhy = dhx*dhy;

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  ierr = DAGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DAVecGetArray(da,dmmg[param->mglevels-1]->x,(void**)&x);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,localX,(void**)&localx);CHKERRQ(ierr);

  /*
     Compute initial solution over the locally owned part of the grid
  */
  {
    PetscReal eps = lx/ly;
    PetscReal pert = 1e-4;
    PetscReal k = 1.*eps;
    PetscReal gam; 

    if (d_e < rho_s) d_e = rho_s;
    gam = k * d_e;

    for (j=ys-1; j<ys+ym+1; j++) {
      yy = j * hy;
      for (i=xs-1; i<xs+xm+1; i++) {
	xx = i * hx;

	if (xx < -M_PI/2) {
	  localx[j][i].phi = pert * gam / k * erf((xx + M_PI) / (sqrt(2) * d_e)) * (-sin(k*yy));
	} else if (xx < M_PI/2) {
	  localx[j][i].phi = - pert * gam / k * erf(xx / (sqrt(2) * d_e)) * (-sin(k*yy));
	} else if (xx < 3*M_PI/2){
	  localx[j][i].phi = pert * gam / k * erf((xx - M_PI) / (sqrt(2) * d_e)) * (-sin(k*yy));
	} else {
	  localx[j][i].phi = - pert * gam / k * erf((xx - 2.*M_PI) / (sqrt(2) * d_e)) * (-sin(k*yy));
	}
#ifdef EQ
	localx[j][i].psi = 0.;
#else
	localx[j][i].psi = cos(xx);
#endif
      }
    }
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
	x[j][i].U   = Lapl(localx,phi,i,j);
	x[j][i].F   = localx[j][i].psi - de2 * Lapl(localx,psi,i,j);
	x[j][i].phi = localx[j][i].phi;
	x[j][i].psi = localx[j][i].psi;
      }
    }
  }
  /*
     Restore vector
  */
  ierr = DAVecRestoreArray(da,dmmg[param->mglevels-1]->x,(void**)&x);
  CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,localX,(void**)&localx);
  CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "ComputeMaxima"
int ComputeMaxima(DA da, Vec X, PetscReal t)
{
  int      ierr,i,j,mx,my,xs,ys,xm,ym;
  int      xints,xinte,yints,yinte;
  Field    **x;
  double   norm[4] = {0,0,0,0};
  double   gnorm[4];
  MPI_Comm comm;

  PetscFunctionBegin;

  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);
  CHKERRQ(ierr);

  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);
  CHKERRQ(ierr);

  xints = xs; xinte = xs+xm; yints = ys; yinte = ys+ym;

  ierr = DAVecGetArray(da, X, (void**)&x);
  CHKERRQ(ierr);

  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      norm[0] = PetscMax(norm[0],x[j][i].U);
      norm[1] = PetscMax(norm[1],x[j][i].F);
      norm[2] = PetscMax(norm[2],x[j][i].phi);
      norm[3] = PetscMax(norm[3],x[j][i].psi);
    }
  }

  ierr = DAVecRestoreArray(da,X,(void**)&x);
  CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)da, &comm);
  CHKERRQ(ierr);
  ierr = MPI_Allreduce(norm, gnorm, 4, MPI_DOUBLE, MPI_MAX, comm);
  CHKERRQ(ierr);

  ierr = PetscFPrintf(PETSC_COMM_WORLD, stderr,
                      "%g\t%g\t%g\t%g\t%g\n",
                      t, gnorm[0], gnorm[1], gnorm[2], gnorm[3]);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
int FormFunctionLocal(DALocalInfo *info,Field **x,Field **f,void *ptr)
{
  AppCtx        *user = (AppCtx*)ptr;
  TstepCtx      *tsCtx = user->tsCtx;
  int           ierr,i,j;
  int           xints,xinte,yints,yinte;
  PassiveReal   hx,hy,dhx,dhy,hxdhy,hydhx,hxhy,dhxdhy;
  PassiveReal   de2,rhos2,nu,eta,dde2;
  PassiveReal   two = 2.0,one = 1.0,p5 = 0.5;
  PassiveReal   F_eq_x,By_eq;

  PetscScalar   xx;
  PetscScalar   vx,vy,avx,avy,vxp,vxm,vyp,vym;
  PetscScalar   Bx,By,aBx,aBy,Bxp,Bxm,Byp,Bym;

  PetscFunctionBegin;
  de2     = sqr(user->param->d_e);
  rhos2   = sqr(user->param->rho_s);
  nu      = user->param->nu;
  eta     = user->param->eta;
  dde2    = one/de2;

  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
  */
  dhx   = info->mx/lx;        dhy   = info->my/ly;
  hx    = one/dhx;             hy   = one/dhy;
  hxdhy = hx*dhy;           hydhx   = hy*dhx;
  hxhy  = hx*hy;             dhxdhy = dhx*dhy;

  xints = info->xs; xinte = info->xs+info->xm;
  yints = info->ys; yinte = info->ys+info->ym;

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
#ifdef EQ
      xx = i * hx;
      F_eq_x = - (1. + de2) * sin(xx);
      By_eq = sin(xx);
#else
      F_eq_x = 0.;
      By_eq = 0.;
#endif

      /*
       * convective coefficients for upwinding
       */

      vx  = - D_y(x,phi,i,j);
      vy  =   D_x(x,phi,i,j);
      avx = PetscAbsScalar(vx); vxp = p5*(vx+avx); vxm = p5*(vx-avx);
      avy = PetscAbsScalar(vy); vyp = p5*(vy+avy); vym = p5*(vy-avy);
#ifndef UPWINDING
      vxp = vxm = p5*vx;
      vyp = vym = p5*vy;
#endif

      Bx  =   D_y(x,psi,i,j);
      By  = - D_x(x,psi,i,j) + By_eq;
      aBx = PetscAbsScalar(Bx); Bxp = p5*(Bx+aBx); Bxm = p5*(Bx-aBx);
      aBy = PetscAbsScalar(By); Byp = p5*(By+aBy); Bym = p5*(By-aBy);
#ifndef UPWINDING
      Bxp = Bxm = p5*Bx;
      Byp = Bym = p5*By;
#endif

      /* Lap(phi) - U */
      f[j][i].phi = (Lapl(x,phi,i,j) - x[j][i].U) * hxhy;

      /* psi - d_e^2 * Lap(psi) - F */
      f[j][i].psi = (x[j][i].psi - de2 * Lapl(x,psi,i,j) - x[j][i].F) * hxhy;

      /* vx * U_x + vy * U_y - (B_x * F_x + B_y * F_y) / d_e^2
         - nu Lap(U) */
      f[j][i].U  =
        ((vxp * (D_xm(x,U,i,j)) + vxm * (D_xp(x,U,i,j)) +
          vyp * (D_ym(x,U,i,j)) + vym * (D_yp(x,U,i,j))) -
         (Bxp * (D_xm(x,F,i,j) + F_eq_x) + Bxm * (D_xp(x,F,i,j) + F_eq_x) +
          Byp * (D_ym(x,F,i,j)) + Bym * (D_yp(x,F,i,j))) * dde2 -
         nu * Lapl(x,U,i,j)) * hxhy;
      
      /* vx * F_x + vy * F_y - rho_s^2 * (B_x * U_x + B_y * U_y)
         - eta * Lap(psi) */
      f[j][i].F  =
        ((vxp * (D_xm(x,F,i,j) + F_eq_x) + vxm * (D_xp(x,F,i,j) + F_eq_x) +
          vyp * (D_ym(x,F,i,j)) + vym * (D_yp(x,F,i,j))) -
         (Bxp * (D_xm(x,U,i,j)) + Bxm * (D_xp(x,U,i,j)) +
          Byp * (D_ym(x,U,i,j)) + Bym * (D_yp(x,U,i,j))) * rhos2 -
         eta * Lapl(x,psi,i,j)) * hxhy;
    }
  }

  /* Add time step contribution */
  if (tsCtx->ires) {
    ierr = AddTSTermLocal(info,x,f,user); CHKERRQ(ierr);
  }
  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  /*  ierr = PetscLogFlops(84*info->ym*info->xm);CHKERRQ(ierr); FIXME */
  PetscFunctionReturn(0);
} 

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "Update"
int Update(DMMG *dmmg)
/*---------------------------------------------------------------------*/
{
  AppCtx         *user = (AppCtx *) ((dmmg[0])->user);
  TstepCtx       *tsCtx = user->tsCtx;
  Parameter      *param = user->param;
  SNES           snes;
  int            ierr,its,lits,i;
  int            max_steps;
  int            nfailsCum = 0,nfails = 0;
  static int     ic_out;
  PetscTruth     ts_monitor = tsCtx->ts_monitor && !user->param->PreLoading;

  PetscFunctionBegin;

  if (user->param->PreLoading) 
    max_steps = 1;
  else
    max_steps = tsCtx->max_steps;
  
  ierr = Initialize(dmmg);
  CHKERRQ(ierr);

  for (tsCtx->itstep = 0; tsCtx->itstep < max_steps; tsCtx->itstep++) {

    if ((param->second_order) && (tsCtx->itstep == 0))
    {
      for (i=param->mglevels-1; i>=0 ;i--)
      {
        ierr = VecCopy(((AppCtx*)dmmg[i]->user)->Xold,
                       ((AppCtx*)dmmg[i]->user)->Xoldold);
        CHKERRQ(ierr);
      }
    }

    for (i=param->mglevels-1; i>0 ;i--) {
      ierr = MatRestrict(dmmg[i]->R, dmmg[i]->x, dmmg[i-1]->x);
      CHKERRQ(ierr);
      ierr = VecPointwiseMult(dmmg[i]->Rscale,dmmg[i-1]->x,dmmg[i-1]->x);
      CHKERRQ(ierr);
      ierr = VecCopy(dmmg[i]->x, ((AppCtx*)dmmg[i]->user)->Xold);
      CHKERRQ(ierr);
    }
    ierr = VecCopy(dmmg[0]->x, ((AppCtx*)dmmg[0]->user)->Xold);
    CHKERRQ(ierr);

    ierr = DMMGSolve(dmmg);
    CHKERRQ(ierr); 

    snes = DMMGGetSNES(dmmg);

#ifdef SAVE_JACOBIAN 

    if (tsCtx->itstep == 665000)
    {
      SLES sles;
      PC pc;
      Mat mat, pmat;
      MatStructure flag;
      PetscViewer viewer;
      char file[128];

      strcpy(file, "matrix");

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file,
                                   PETSC_BINARY_CREATE, &viewer);
      CHKERRQ(ierr);

      ierr = SNESGetSLES(snes, &sles);
      CHKERRQ(ierr);

      ierr = SLESGetPC(sles, &pc);
      CHKERRQ(ierr);

      ierr = PCGetOperators(pc, &mat, &pmat, &flag);
      CHKERRQ(ierr);

      ierr = MatView(mat, viewer);
      CHKERRQ(ierr);

      ierr = PetscViewerDestroy(viewer);
      CHKERRQ(ierr);
      SETERRQ(1,"Done saving Jacobian");
    }

#endif

    tsCtx->t += tsCtx->dt;

    /* save restart solution if requested at a particular time, then exit */
    if (tsCtx->dump_time > 0.0 && tsCtx->t >= tsCtx->dump_time) {
      Vec v;
      ierr = SNESGetSolution(snes,&v);CHKERRQ(ierr);
      ierr = VecView(v,PETSC_VIEWER_BINARY_WORLD);CHKERRQ(ierr);
      SETERRQ1(1,"Saved solution at time %d",tsCtx->t);
    }

    if (ts_monitor) {
      ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
      ierr = SNESGetNumberLinearIterations(snes,&lits);CHKERRQ(ierr);
      ierr = SNESGetNumberUnsuccessfulSteps(snes,&nfails);CHKERRQ(ierr);
      ierr = SNESGetFunctionNorm(snes,&tsCtx->fnorm);CHKERRQ(ierr);
      nfailsCum += nfails;
      if (nfailsCum >= 2) SETERRQ(1,"Unable to find a Newton Step");
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Time Step %d time = %g Newton steps %d linear steps %d fnorm %g\n",
			 tsCtx->itstep, tsCtx->t, its, lits, tsCtx->fnorm);
      CHKERRQ(ierr);

      /* send solution over to Matlab, to be visualized (using ex29.m) */
      if (!user->param->PreLoading && tsCtx->socketviewer){
        Vec v;
        ierr = SNESGetSolution(snes,&v);CHKERRQ(ierr);
        ierr = VecView(v,tsCtx->socketviewer);CHKERRQ(ierr);
      }
    }

    if (!param->PreLoading) {
      if (param->draw_contours) {
	ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);
        CHKERRQ(ierr);
      }

      if (0 && ts_monitor) {
	/* compute maxima */
	ComputeMaxima((DA) dmmg[param->mglevels-1]->dm,
                      dmmg[param->mglevels-1]->x,
                      tsCtx->t);
	/* output */
	if (ic_out++ == (int)(tsCtx->dt_out / tsCtx->dt + .5)) {
#ifdef HAVE_DA_HDF
          char fname[128];

          sprintf(fname, "out-%g.hdf", tsCtx->t);
          ierr = DAVecHDFOutput(DMMGGetDA(dmmg), DMMGGetx(dmmg), fname);
          CHKERRQ(ierr);
#else
/*
          ierr = Gnuplot(DMMGGetDA(dmmg), DMMGGetx(dmmg), tsCtx->t);
          CHKERRQ(ierr);
*/
#endif
	  ic_out = 1;
        }
      }
    }
  } /* End of time step loop */
 
  if (!user->param->PreLoading){ 
    ierr = SNESGetFunctionNorm(snes,&tsCtx->fnorm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "timesteps %d fnorm = %g\n",
		       tsCtx->itstep, tsCtx->fnorm);
    CHKERRQ(ierr);
  }

  if (user->param->PreLoading) {
    tsCtx->fnorm_ini = 0.0;
  }
 
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AddTSTermLocal"
int AddTSTermLocal(DALocalInfo* info,Field **x,Field **f,AppCtx *user)
/*---------------------------------------------------------------------*/
{
  TstepCtx       *tsCtx = user->tsCtx;
  DA             da = info->da;
  int            ierr,i,j;
  int            xints,xinte,yints,yinte;
  PassiveReal    hx,hy,dhx,dhy,hxhy;
  PassiveReal    one = 1.0;
  PassiveScalar  dtinv;
  PassiveField   **xold;

  PetscFunctionBegin; 

  xints = info->xs; xinte = info->xs+info->xm;
  yints = info->ys; yinte = info->ys+info->ym;

  dhx  = info->mx/lx;            dhy = info->my/ly;
  hx   = one/dhx;                 hy = one/dhy;
  hxhy = hx*hy;

  dtinv = hxhy/(tsCtx->dt);

  ierr  = DAVecGetArray(da,user->Xold,(void**)&xold);CHKERRQ(ierr);
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      f[j][i].U += dtinv*(x[j][i].U-xold[j][i].U);
      f[j][i].F += dtinv*(x[j][i].F-xold[j][i].F);
    }
  }
  ierr = DAVecRestoreArray(da,user->Xold,(void**)&xold);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AddTSTermLocal2"
int AddTSTermLocal2(DALocalInfo* info,Field **x,Field **f,AppCtx *user)
/*---------------------------------------------------------------------*/
{
  TstepCtx       *tsCtx = user->tsCtx;
  DA             da = info->da;
  int            ierr,i,j;
  int            xints,xinte,yints,yinte;
  PassiveReal    hx,hy,dhx,dhy,hxhy;
  PassiveReal    one = 1.0, onep5 = 1.5, two = 2.0, p5 = 0.5;
  PassiveScalar  dtinv;
  PassiveField   **xoldold,**xold;

  PetscFunctionBegin; 

  xints = info->xs; xinte = info->xs+info->xm;
  yints = info->ys; yinte = info->ys+info->ym;

  dhx  = info->mx/lx;            dhy = info->my/ly;
  hx   = one/dhx;                 hy = one/dhy;
  hxhy = hx*hy;

  dtinv = hxhy/(tsCtx->dt);

  ierr  = DAVecGetArray(da,user->Xoldold,(void**)&xoldold);CHKERRQ(ierr);
  ierr  = DAVecGetArray(da,user->Xold,(void**)&xold);CHKERRQ(ierr);
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      f[j][i].U += dtinv * (onep5 * x[j][i].U - two * xold[j][i].U +
                            p5 * xoldold[j][i].U);
      f[j][i].F += dtinv * (onep5 * x[j][i].F - two * xold[j][i].F +
                            p5 * xoldold[j][i].F);
    }
  }
  ierr = DAVecRestoreArray(da,user->Xoldold,(void**)&xoldold);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,user->Xold,(void**)&xold);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AttachNullSpace"
int AttachNullSpace(PC pc,Vec model)
{
  int          i,ierr,rstart,rend,N;
  MatNullSpace sp;
  Vec          v,vs[1];
  PetscScalar  *vx,scale;

  PetscFunctionBegin;
  ierr  = VecDuplicate(model,&v);CHKERRQ(ierr);
  ierr  = VecGetSize(model,&N);CHKERRQ(ierr);
  scale = 2.0/sqrt(N); 
  ierr  = VecGetArray(v,&vx);CHKERRQ(ierr);
  ierr  = VecGetOwnershipRange(v,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    if (!(i % 4)) vx[i-rstart] = scale;
    else          vx[i-rstart] = 0.0;
  }
  ierr  = VecRestoreArray(v,&vx);CHKERRQ(ierr);
  vs[0] = v;
  ierr  = MatNullSpaceCreate(PETSC_COMM_WORLD,0,1,vs,&sp);CHKERRQ(ierr);
  ierr  = VecDestroy(v);CHKERRQ(ierr);
  ierr  = PCNullSpaceAttach(pc,sp);CHKERRQ(ierr);
  ierr  = MatNullSpaceDestroy(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Computes the nonzero structure of the Jacobian matrix for exactly this problem with 
  this discretization. Note the general (default) matrix generation will assume complete coupling 
  within and between the 4 by 4 blocks at each grid point
*/
#undef __FUNCT__  
#define __FUNCT__ "DAGetMatrix_Specialized" 
int DAGetMatrix_Specialized(DA da,MatType ignored,Mat *J)
{
  int                    ierr,xs,ys,nx,ny,i,j,slot,gxs,gys,gnx,gny;           
  int                    m,n,dim,s,*cols,nc,*rows,col,cnt,l,p;
  int                    lstart,lend,pstart,pend,*dnz,*onz,size;
  int                    dims[2],starts[2];
  MPI_Comm               comm;
  PetscScalar            *values;
  ISLocalToGlobalMapping ltog;
  DAStencilType          st;

  PetscFunctionBegin;
  /*     
         nc - number of components per grid point 
         col - number of colors needed in one direction for single component problem
  
  */
  ierr = DAGetInfo(da,&dim,&m,&n,0,0,0,0,&nc,&s,0,&st);CHKERRQ(ierr);
  col = 2*s + 1;

  ierr = DAGetCorners(da,&xs,&ys,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,0,&gnx,&gny,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* create empty Jacobian matrix */
  ierr = MatCreate(comm,nc*nx*ny,nc*nx*ny,PETSC_DECIDE,PETSC_DECIDE,J);CHKERRQ(ierr);  

  ierr = PetscMalloc(col*col*nc*nc*sizeof(PetscScalar),&values);CHKERRQ(ierr);
  ierr = PetscMemzero(values,col*col*nc*nc*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(nc*sizeof(int),&rows);CHKERRQ(ierr);
  ierr = PetscMalloc(col*col*nc*nc*sizeof(int),&cols);CHKERRQ(ierr);
  ierr = DAGetISLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  
  /* determine the matrix preallocation information */
  ierr = MatPreallocateInitialize(comm,nc*nx*ny,nc*nx*ny,dnz,onz);CHKERRQ(ierr);
  for (i=xs; i<xs+nx; i++) {

    pstart = -s;
    pend   = s;

    for (j=ys; j<ys+ny; j++) {
      slot = i - gxs + gnx*(j - gys);

      lstart = -s;
      lend   = s;

      /* phi row */
      cnt     = 0;
      for (l=lstart; l<lend+1; l++) {
	for (p=pstart; p<pend+1; p++) {
	  if ((!l || !p)) {
	    cols[cnt++]  = 0 + nc*(slot + gnx*l + p); /* coupling to phi */ 
            if (!l && !p) {
  	      cols[cnt++]  = 2 + nc*(slot + gnx*l + p); /* coupling to U */ 
            }
	  }
	}
      }
      rows[0] = 0 + nc*slot;
      ierr = MatPreallocateSetLocal(ltog,1,rows,cnt,cols,dnz,onz);CHKERRQ(ierr);

      /* psi row */
      cnt     = 0;
      for (l=lstart; l<lend+1; l++) {
	for (p=pstart; p<pend+1; p++) {
	  if ((!l || !p)) {
	    cols[cnt++]  = 1 + nc*(slot + gnx*l + p); /* coupling to psi */ 
            if (!l && !p) {
  	      cols[cnt++]  = 3 + nc*(slot + gnx*l + p); /* coupling to F */ 
            }
	  }
	}
      }
      rows[0] = 1 + nc*slot;
      ierr = MatPreallocateSetLocal(ltog,1,rows,cnt,cols,dnz,onz);CHKERRQ(ierr);

      /* U row */
      cnt     = 0;
      for (l=lstart; l<lend+1; l++) {
	for (p=pstart; p<pend+1; p++) {
	  if ((!l || !p)) {
            if (l || p) {
	      cols[cnt++]  = 0 + nc*(slot + gnx*l + p); /* coupling to phi */ 
	      cols[cnt++]  = 1 + nc*(slot + gnx*l + p); /* coupling to psi */
	    }
	    cols[cnt++]  = 2 + nc*(slot + gnx*l + p); /* coupling to U */ 
	    cols[cnt++]  = 3 + nc*(slot + gnx*l + p); /* coupling to F */
	  }
	}
      }
      rows[0] = 2 + nc*slot;
      ierr = MatPreallocateSetLocal(ltog,1,rows,cnt,cols,dnz,onz);CHKERRQ(ierr);

      /* F row */
      cnt     = 0;
      for (l=lstart; l<lend+1; l++) {
	for (p=pstart; p<pend+1; p++) {
	  if ((!l || !p)) {
            if (l || p) {
	      cols[cnt++]  = 0 + nc*(slot + gnx*l + p); /* coupling to phi */ 
	    }
            cols[cnt++]  = 1 + nc*(slot + gnx*l + p); /* coupling to psi */
	    cols[cnt++]  = 2 + nc*(slot + gnx*l + p); /* coupling to U */ 
	    cols[cnt++]  = 3 + nc*(slot + gnx*l + p); /* coupling to F */
	  }
	}
      }
      rows[0] = 3 + nc*slot;
      ierr = MatPreallocateSetLocal(ltog,1,rows,cnt,cols,dnz,onz);CHKERRQ(ierr);
    }
  }
  /* set matrix type and preallocation information */
  if (size > 1) {
    ierr = MatSetType(*J,MATMPIAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*J,MATSEQAIJ);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(*J,0,dnz);CHKERRQ(ierr);  
  ierr = MatMPIAIJSetPreallocation(*J,0,dnz,0,onz);CHKERRQ(ierr);  
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*J,ltog);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&starts[0],&starts[1],PETSC_IGNORE,&dims[0],&dims[1],PETSC_IGNORE);CHKERRQ(ierr);
  ierr = MatSetStencil(*J,2,dims,starts,nc);CHKERRQ(ierr);

  /*
    For each node in the grid: we get the neighbors in the local (on processor ordering
    that includes the ghost points) then MatSetValuesLocal() maps those indices to the global
    PETSc ordering.
  */
  /* loop over x grid points */
  for (i=xs; i<xs+nx; i++) {
    
    pstart = -s;
    pend   = s;
      
    /* loop over y grid points */
    for (j=ys; j<ys+ny; j++) {
      slot = i - gxs + gnx*(j - gys);
      
      lstart = -s;
      lend   = s;

      /* phi row */
      cnt     = 0;
      /* loop over neighbor points (and myself) that MAY be in stencil */
      for (l=lstart; l<lend+1; l++) {
	for (p=pstart; p<pend+1; p++) {
	  if ((!l || !p)) { /* skips the 4 corners (i.e. this gives you the 5 point stencil */
	    cols[cnt++]  = 0 + nc*(slot + gnx*l + p); /* coupling to phi */ 
            if (!l && !p) { /* skips all the points except the center (i.e. coupling to same point) */
  	      cols[cnt++]  = 2 + nc*(slot + gnx*l + p); /* coupling to U */ 
            }
	  }
	}
      }
      rows[0] = 0 + nc*slot;
      ierr = MatSetValuesLocal(*J,1,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);

      /* psi row */
      cnt     = 0;
      for (l=lstart; l<lend+1; l++) {
	for (p=pstart; p<pend+1; p++) {
	  if ((!l || !p)) {
	    cols[cnt++]  = 1 + nc*(slot + gnx*l + p); /* coupling to psi */ 
            if (!l && !p) {
  	      cols[cnt++]  = 3 + nc*(slot + gnx*l + p); /* coupling to F */ 
            }
	  }
	}
      }
      rows[0] = 1 + nc*slot;
      ierr = MatSetValuesLocal(*J,1,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);

      /* U row */
      cnt     = 0;
      for (l=lstart; l<lend+1; l++) {
	for (p=pstart; p<pend+1; p++) {
	  if ((!l || !p)) {
            if (l || p) { /* U is not coupled to phi or psi at the center */
	      cols[cnt++]  = 0 + nc*(slot + gnx*l + p); /* coupling to phi */ 
	      cols[cnt++]  = 1 + nc*(slot + gnx*l + p); /* coupling to psi */
	    }
	    cols[cnt++]  = 2 + nc*(slot + gnx*l + p); /* coupling to U */ 
	    cols[cnt++]  = 3 + nc*(slot + gnx*l + p); /* coupling to F */
	  }
	}
      }
      rows[0] = 2 + nc*slot;
      ierr = MatSetValuesLocal(*J,1,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);

      /* F row */
      cnt     = 0;
      for (l=lstart; l<lend+1; l++) {
	for (p=pstart; p<pend+1; p++) {
	  if ((!l || !p)) {
	    if (l || p) { /* U is not coupled to phi at the center */
	      cols[cnt++]  = 0 + nc*(slot + gnx*l + p); /* coupling to phi */ 
	    }
	    cols[cnt++]  = 1 + nc*(slot + gnx*l + p); /* coupling to psi */
	    cols[cnt++]  = 2 + nc*(slot + gnx*l + p); /* coupling to U */ 
	    cols[cnt++]  = 3 + nc*(slot + gnx*l + p); /* coupling to F */
	  }
	}
      }
      rows[0] = 3 + nc*slot;
      ierr = MatSetValuesLocal(*J,1,rows,cnt,cols,values,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

/*
    This is an experimental function and can be safely ignored.
*/
int FormFunctionLocali(DALocalInfo *info,MatStencil *st,Field **x,PetscScalar *f,void *ptr)
 {
  AppCtx        *user = (AppCtx*)ptr;
  TstepCtx      *tsCtx = user->tsCtx;
  int           ierr,i,j,c;
  int           xints,xinte,yints,yinte;
  PassiveReal   hx,hy,dhx,dhy,hxdhy,hydhx,hxhy,dhxdhy;
  PassiveReal   de2,rhos2,nu,eta,dde2;
  PassiveReal   two = 2.0,one = 1.0,p5 = 0.5;
  PassiveReal   F_eq_x,By_eq,dtinv;

  PetscScalar   xx;
  PetscScalar   vx,vy,avx,avy,vxp,vxm,vyp,vym;
  PetscScalar   Bx,By,aBx,aBy,Bxp,Bxm,Byp,Bym;
  PassiveField  **xold;

  PetscFunctionBegin;
  de2     = sqr(user->param->d_e);
  rhos2   = sqr(user->param->rho_s);
  nu      = user->param->nu;
  eta     = user->param->eta;
  dde2    = one/de2;

  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
  */
  dhx   = info->mx/lx;        dhy   = info->my/ly;
  hx    = one/dhx;             hy   = one/dhy;
  hxdhy = hx*dhy;           hydhx   = hy*dhx;
  hxhy  = hx*hy;             dhxdhy = dhx*dhy;

  xints = info->xs; xinte = info->xs+info->xm;
  yints = info->ys; yinte = info->ys+info->ym;


  i = st->i; j = st->j; c = st->c;

#ifdef EQ
      xx = i * hx;
      F_eq_x = - (1. + de2) * sin(xx);
      By_eq = sin(xx);
#else
      F_eq_x = 0.;
      By_eq = 0.;
#endif

      /*
       * convective coefficients for upwinding
       */

      vx  = - D_y(x,phi,i,j);
      vy  =   D_x(x,phi,i,j);
      avx = PetscAbsScalar(vx); vxp = p5*(vx+avx); vxm = p5*(vx-avx);
      avy = PetscAbsScalar(vy); vyp = p5*(vy+avy); vym = p5*(vy-avy);
#ifndef UPWINDING
      vxp = vxm = p5*vx;
      vyp = vym = p5*vy;
#endif

      Bx  =   D_y(x,psi,i,j);
      By  = - D_x(x,psi,i,j) + By_eq;
      aBx = PetscAbsScalar(Bx); Bxp = p5*(Bx+aBx); Bxm = p5*(Bx-aBx);
      aBy = PetscAbsScalar(By); Byp = p5*(By+aBy); Bym = p5*(By-aBy);
#ifndef UPWINDING
      Bxp = Bxm = p5*Bx;
      Byp = Bym = p5*By;
#endif

      ierr  = DAVecGetArray(info->da,user->Xold,(void**)&xold);CHKERRQ(ierr);
      dtinv = hxhy/(tsCtx->dt);
      switch(c) {

        case 0:
          /* Lap(phi) - U */
          *f = (Lapl(x,phi,i,j) - x[j][i].U) * hxhy;
          break;

        case 1:
	  /* psi - d_e^2 * Lap(psi) - F */
	  *f = (x[j][i].psi - de2 * Lapl(x,psi,i,j) - x[j][i].F) * hxhy;
          break;

        case 2:
          /* vx * U_x + vy * U_y - (B_x * F_x + B_y * F_y) / d_e^2
            - nu Lap(U) */
	  *f  =
	    ((vxp * (D_xm(x,U,i,j)) + vxm * (D_xp(x,U,i,j)) +
	      vyp * (D_ym(x,U,i,j)) + vym * (D_yp(x,U,i,j))) -
	     (Bxp * (D_xm(x,F,i,j) + F_eq_x) + Bxm * (D_xp(x,F,i,j) + F_eq_x) +
	      Byp * (D_ym(x,F,i,j)) + Bym * (D_yp(x,F,i,j))) * dde2 -
	     nu * Lapl(x,U,i,j)) * hxhy;
          *f += dtinv*(x[j][i].U-xold[j][i].U);
	  break;

        case 3:
          /* vx * F_x + vy * F_y - rho_s^2 * (B_x * U_x + B_y * U_y)
           - eta * Lap(psi) */
          *f  =
            ((vxp * (D_xm(x,F,i,j) + F_eq_x) + vxm * (D_xp(x,F,i,j) + F_eq_x) +
             vyp * (D_ym(x,F,i,j)) + vym * (D_yp(x,F,i,j))) -
            (Bxp * (D_xm(x,U,i,j)) + Bxm * (D_xp(x,U,i,j)) +
             Byp * (D_ym(x,U,i,j)) + Bym * (D_yp(x,U,i,j))) * rhos2 -
            eta * Lapl(x,psi,i,j)) * hxhy;
          *f += dtinv*(x[j][i].F-xold[j][i].F);
      }
      ierr = DAVecRestoreArray(info->da,user->Xold,(void**)&xold);CHKERRQ(ierr);


  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  /*  ierr = PetscLogFlops(84*info->ym*info->xm);CHKERRQ(ierr); FIXME */
  PetscFunctionReturn(0);
} 
