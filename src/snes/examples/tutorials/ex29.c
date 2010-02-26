
/*#define HAVE_DA_HDF*/

/* solve the equations for the perturbed magnetic field only */
#define EQ 

/* turning this on causes instability?!? */
/* #define UPWINDING  */

static char help[] = "Hall MHD with in two dimensions with time stepping and multigrid.\n\n\
-options_file ex29.options\n\
other PETSc options\n\
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
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers 
*/
#include "petscsnes.h"
#include "petscda.h"
#include "petscmg.h"
#include "petscdmmg.h"

#ifdef HAVE_DA_HDF
PetscInt DAVecHDFOutput(DA,Vec,char*);
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
#define lx            (2.*3.1415926)
#define ly            (4.*3.1415926)
#define sqr(a)        ((a)*(a))

/* 
   User-defined routines and data structures
*/

typedef struct {
  PetscReal      fnorm_ini,dt_ini;
  PetscReal      dt,dt_out;
  PetscReal      ptime;
  PetscReal      max_time;
  PetscReal      fnorm_ratio;
  PetscInt       ires,itstep;
  PetscInt       max_steps,print_freq;
  PetscReal      t;
  PetscReal      fnorm;

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
  PetscInt     mglevels;
  PetscInt     cycles;           /* number of time steps for integration */ 
  PassiveReal  nu,eta,d_e,rho_s; /* physical parameters */
  PetscTruth   draw_contours;    /* flag - 1 indicates drawing contours */
  PetscTruth   second_order;
  PetscTruth   PreLoading;
} Parameters;

typedef struct {
  Vec          Xoldold,Xold;
  TstepCtx     *tsCtx;
  Parameters    *param;
} AppCtx;

extern PetscErrorCode FormFunctionLocal(DALocalInfo*,Field**,Field**,void*);
extern PetscErrorCode Update(DMMG *);
extern PetscErrorCode Initialize(DMMG *);
extern PetscErrorCode AddTSTermLocal(DALocalInfo*,Field **,Field **,AppCtx *);
extern PetscErrorCode AddTSTermLocal2(DALocalInfo*,Field **,Field **,AppCtx *);
extern PetscErrorCode Gnuplot(DA, Vec, double);
extern PetscErrorCode FormFunctionLocali(DALocalInfo*,MatStencil*,Field**,PetscScalar*,void*);
extern PetscErrorCode FormFunctionLocali4(DALocalInfo*,MatStencil*,Field**,PetscScalar*,void*);
extern PetscErrorCode CreateNullSpace(DMMG,Vec*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DMMG           *dmmg;       /* multilevel grid structure */
  AppCtx         *user;       /* user-defined work context (one for each level) */
  TstepCtx       tsCtx;       /* time-step parameters (one total) */
  Parameters      param;       /* physical parameters (one total) */
  PetscInt       i,m,n,mx,my;
  DA             da;
  PetscReal      dt_ratio;
  PetscInt       dfill[16] = {1,0,1,0,
                              0,1,0,1,
                              0,0,1,1,
                              0,1,1,1};
  PetscInt       ofill[16] = {1,0,0,0,
                              0,1,0,0,
                              1,1,1,1,
                              1,1,1,1};

  PetscInitialize(&argc,&argv,(char *)0,help);


  PreLoadBegin(PETSC_TRUE,"SetUp");

  param.PreLoading = PreLoading;
    ierr = DMMGCreate(PETSC_COMM_WORLD,1,&user,&dmmg);CHKERRQ(ierr);
    param.mglevels = DMMGGetLevels(dmmg);

    /*
      Create distributed array multigrid object (DMMG) to manage
      parallel grid and vectors for principal unknowns (x) and
      governing residuals (f)
    */
    ierr = DACreate2d(PETSC_COMM_WORLD, DA_XYPERIODIC, DA_STENCIL_STAR, -5, -5,
                      PETSC_DECIDE, PETSC_DECIDE, 4, 1, 0, 0, &da);CHKERRQ(ierr);

    /* overwrite the default sparsity pattern toone specific for
       this code's nonzero structure */
    ierr = DASetBlockFills(da,dfill,ofill);CHKERRQ(ierr);

    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);

    /* default physical parameters */
    param.nu    = 0;
    param.eta   = 1e-3;
    param.d_e   = 0.2;
    param.rho_s = 0;
    param.second_order = PETSC_FALSE;

    ierr = PetscOptionsGetReal(PETSC_NULL, "-viscosity", &param.nu,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsGetReal(PETSC_NULL, "-resistivity", &param.eta,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsGetReal(PETSC_NULL, "-skin_depth", &param.d_e,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsGetReal(PETSC_NULL, "-larmor_radius", &param.rho_s,PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsHasName(PETSC_NULL, "-contours", &param.draw_contours);CHKERRQ(ierr);

    ierr = PetscOptionsHasName(PETSC_NULL, "-second_order", &param.second_order);CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg), 0, "phi");CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg), 1, "psi");CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg), 2, "U");CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg), 3, "F");CHKERRQ(ierr);

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

    ierr = PetscOptionsGetInt(PETSC_NULL, "-max_st", &tsCtx.max_steps,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL, "-ts_rtol", &tsCtx.fnorm_ratio,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL, "-print_freq", &tsCtx.print_freq,PETSC_NULL);CHKERRQ(ierr);

    dt_ratio = 1.0;
    ierr = PetscOptionsGetReal(PETSC_NULL, "-dt_ratio", &dt_ratio,PETSC_NULL);CHKERRQ(ierr);
    tsCtx.dt *= dt_ratio;

    ierr = PetscOptionsHasName(PETSC_NULL, "-ts_monitor", &tsCtx.ts_monitor);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL, "-dump", &tsCtx.dump_time,PETSC_NULL);CHKERRQ(ierr);


    tsCtx.socketviewer = 0;
#if defined(PETSC_USE_SOCKET_VIEWER)
    {
      PetscTruth flg;
      ierr = PetscOptionsHasName(PETSC_NULL, "-socket_viewer", &flg);CHKERRQ(ierr);
      if (flg && !PreLoading) {
        tsCtx.ts_monitor = PETSC_TRUE;
        ierr = PetscViewerSocketOpen(PETSC_COMM_WORLD,0,PETSC_DECIDE,&tsCtx.socketviewer);CHKERRQ(ierr);
      }
    }
#endif

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create user context, set problem data, create vector data structures.
       Also, compute the initial guess.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* create application context for each level */    
    ierr = PetscMalloc(param.mglevels*sizeof(AppCtx),&user);CHKERRQ(ierr);
    for (i=0; i<param.mglevels; i++) {
      /* create work vectors to hold previous time-step solution and
         function value */
      ierr = VecDuplicate(dmmg[i]->x, &user[i].Xoldold);CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x, &user[i].Xold);CHKERRQ(ierr);
      user[i].tsCtx = &tsCtx;
      user[i].param = &param;
      ierr = DMMGSetUser(dmmg,i,&user[i]);CHKERRQ(ierr);
    }
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver context
       
       Process adiC(20):  AddTSTermLocal AddTSTermLocal2 FormFunctionLocal FormFunctionLocali
       Process blockadiC(4):  FormFunctionLocali4
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMMGSetSNESLocal(dmmg, FormFunctionLocal, 0,ad_FormFunctionLocal, admf_FormFunctionLocal);CHKERRQ(ierr);
    ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);
    ierr = DMMGSetSNESLocali(dmmg,FormFunctionLocali,0,admf_FormFunctionLocali);CHKERRQ(ierr);
    ierr = DMMGSetSNESLocalib(dmmg,FormFunctionLocali4,0,admfb_FormFunctionLocali4);CHKERRQ(ierr);
  
   /* attach nullspace to each level of the preconditioner */
    ierr = DMMGSetNullSpace(dmmg,PETSC_FALSE,1,CreateNullSpace);CHKERRQ(ierr);
    
    ierr = PetscPrintf(PETSC_COMM_WORLD, "finish setupNull!");

    if (PreLoading) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "# viscosity = %G, resistivity = %G, "
			 "skin_depth # = %G, larmor_radius # = %G\n",
			 param.nu, param.eta, param.d_e, param.rho_s);CHKERRQ(ierr);
      ierr = DAGetInfo(DMMGGetDA(dmmg),0,&m,&n,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Problem size %D by %D\n",m,n);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"dx %G dy %G dt %G ratio dt/min(dx,dy) %G\n",lx/mx,ly/my,tsCtx.dt,dt_ratio);CHKERRQ(ierr);
    }

    
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    PreLoadStage("Solve");

    if (param.draw_contours) {
      ierr = VecView(((AppCtx*)DMMGGetUser(dmmg,param.mglevels-1))->Xold,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    }

    ierr = Update(dmmg);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    for (i=0; i<param.mglevels; i++) {
      ierr = VecDestroy(user[i].Xoldold);CHKERRQ(ierr);
      ierr = VecDestroy(user[i].Xold);CHKERRQ(ierr);
    }
    ierr = PetscFree(user);CHKERRQ(ierr);
    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);

    PreLoadEnd();
    
  PetscFinalize();
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Gnuplot"
/* ------------------------------------------------------------------- */
PetscErrorCode Gnuplot(DA da, Vec X, double mtime)
{
  PetscInt       i,j,xs,ys,xm,ym;
  PetscInt       xints,xinte,yints,yinte;
  PetscErrorCode ierr;
  Field          **x;
  FILE           *f;
  char           fname[PETSC_MAX_PATH_LEN];
  PetscMPIInt    rank;

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  sprintf(fname, "out-%g-%d.dat", mtime, rank);

  f = fopen(fname, "w");

  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  ierr = DAVecGetArray(da,X,&x);CHKERRQ(ierr);

  xints = xs; xinte = xs+xm; yints = ys; yinte = ys+ym;

  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      ierr = PetscFPrintf(PETSC_COMM_WORLD, f,
                          "%D %D %G %G %G %G %G %G\n",
                          i, j, 0.0, 0.0,
                          PetscAbsScalar(x[j][i].U), PetscAbsScalar(x[j][i].F), 
                          PetscAbsScalar(x[j][i].phi), PetscAbsScalar(x[j][i].psi));CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_WORLD,f, "\n");CHKERRQ(ierr);
  }
  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  fclose(f);
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Initialize"
/* ------------------------------------------------------------------- */
PetscErrorCode Initialize(DMMG *dmmg)
{
  AppCtx         *appCtx = (AppCtx*)DMMGGetUser(dmmg,0);
  Parameters      *param = appCtx->param;
  DA             da;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xs,ys,xm,ym;
  PetscReal      two = 2.0,one = 1.0;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx,hxhy,dhxdhy;
  PetscReal      d_e,rho_s,de2,xx,yy;
  Field          **x, **localx;
  Vec            localX;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(0,"-restart",&flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"binaryoutput",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
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
  ierr = DAVecGetArray(da,dmmg[param->mglevels-1]->x,&x);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,localX,&localx);CHKERRQ(ierr);

  /*
     Compute initial solution over the locally owned part of the grid
  */
#if defined (PETSC_HAVE_ERF)
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
	if (xx < -3.1416926/2) {
	  localx[j][i].phi = pert * gam / k * erf((xx + 3.1415926) / (sqrt(2.0) * d_e)) * (-sin(k*yy));
	} else if (xx < 3.1415926/2) {
	  localx[j][i].phi = - pert * gam / k * erf(xx / (sqrt(2.0) * d_e)) * (-sin(k*yy));
	} else if (xx < 3*3.1415926/2){
	  localx[j][i].phi = pert * gam / k * erf((xx - 3.1415926) / (sqrt(2.0) * d_e)) * (-sin(k*yy));
	} else {
	  localx[j][i].phi = - pert * gam / k * erf((xx - 2.*3.1415926) / (sqrt(2.0) * d_e)) * (-sin(k*yy));
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
#else
  SETERRQ(1,"erf() not available on this machine");
#endif

  /*
     Restore vector
  */
  ierr = DAVecRestoreArray(da,dmmg[param->mglevels-1]->x,&x);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,localX,&localx);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&localX);CHKERRQ(ierr);

  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "ComputeMaxima"
PetscErrorCode ComputeMaxima(DA da, Vec X, PetscReal t)
{
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xs,ys,xm,ym;
  PetscInt       xints,xinte,yints,yinte;
  Field          **x;
  double         norm[4] = {0,0,0,0};
  double         gnorm[4];
  MPI_Comm       comm;

  PetscFunctionBegin;

  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  xints = xs; xinte = xs+xm; yints = ys; yinte = ys+ym;

  ierr = DAVecGetArray(da, X, &x);CHKERRQ(ierr);

  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      norm[0] = PetscMax(norm[0],PetscAbsScalar(x[j][i].U));
      norm[1] = PetscMax(norm[1],PetscAbsScalar(x[j][i].F));
      norm[2] = PetscMax(norm[2],PetscAbsScalar(x[j][i].phi));
      norm[3] = PetscMax(norm[3],PetscAbsScalar(x[j][i].psi));
    }
  }

  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)da, &comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(norm, gnorm, 4, MPI_DOUBLE, MPI_MAX, comm);CHKERRQ(ierr);

  ierr = PetscFPrintf(PETSC_COMM_WORLD, stderr,"%G\t%G\t%G\t%G\t%G\n",t, gnorm[0], gnorm[1], gnorm[2], gnorm[3]);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
PetscErrorCode FormFunctionLocal(DALocalInfo *info,Field **x,Field **f,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  TstepCtx       *tsCtx = user->tsCtx;
  Parameters      *param = user->param;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PassiveReal    hx,hy,dhx,dhy,hxdhy,hydhx,hxhy,dhxdhy;
  PassiveReal    de2,rhos2,nu,eta,dde2;
  PassiveReal    two = 2.0,one = 1.0,p5 = 0.5;
  PassiveReal    F_eq_x,By_eq;
  PetscScalar    xx;
  PetscScalar    vx,vy,avx,avy,vxp,vxm,vyp,vym;
  PetscScalar    Bx,By,aBx,aBy,Bxp,Bxm,Byp,Bym;

  PetscFunctionBegin;
  de2     = sqr(param->d_e);
  rhos2   = sqr(param->rho_s);
  nu      = param->nu;
  eta     = param->eta;
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
      F_eq_x = - (1. + de2) * sin(PetscAbsScalar(xx));
      By_eq = sin(PetscAbsScalar(xx));
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
    if ((param->second_order) && (tsCtx->itstep > 0)){
      ierr = AddTSTermLocal2(info,x,f,user);CHKERRQ(ierr);
    } else {
      ierr = AddTSTermLocal(info,x,f,user);CHKERRQ(ierr);
    }
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  /*  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr); FIXME */
  PetscFunctionReturn(0);
} 

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "Update"
PetscErrorCode Update(DMMG *dmmg)
/*---------------------------------------------------------------------*/
{
  AppCtx          *user = (AppCtx *) DMMGGetUser(dmmg,0);
  TstepCtx        *tsCtx = user->tsCtx;
  Parameters       *param = user->param;
  SNES            snes;
  PetscErrorCode  ierr;
  PetscInt        its,lits,i;
  PetscInt        max_steps;
  PetscInt        nfailsCum = 0,nfails = 0;
  static PetscInt ic_out;
  PetscTruth      ts_monitor = (tsCtx->ts_monitor && !param->PreLoading) ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;

  if (param->PreLoading) 
    max_steps = 1;
  else
    max_steps = tsCtx->max_steps;
  
  ierr = Initialize(dmmg);CHKERRQ(ierr);

  snes = DMMGGetSNES(dmmg);

  for (tsCtx->itstep = 0; tsCtx->itstep < max_steps; tsCtx->itstep++) {
    
    ierr = PetscPrintf(PETSC_COMM_WORLD, "time step=%d!\n",tsCtx->itstep);
    if ((param->second_order) && (tsCtx->itstep > 0))
    {
      for (i=param->mglevels-1; i>=0 ;i--)
      {
        ierr = VecCopy(((AppCtx*)DMMGGetUser(dmmg,i))->Xold,((AppCtx*)DMMGGetUser(dmmg,i))->Xoldold);CHKERRQ(ierr);
      }
    }

    for (i=param->mglevels-1; i>0 ;i--) {
      ierr = MatRestrict(dmmg[i]->R, dmmg[i]->x, dmmg[i-1]->x);CHKERRQ(ierr);
      ierr = VecPointwiseMult(dmmg[i-1]->x,dmmg[i-1]->x,dmmg[i]->Rscale);CHKERRQ(ierr);
      ierr = VecCopy(dmmg[i]->x, ((AppCtx*)DMMGGetUser(dmmg,i))->Xold);CHKERRQ(ierr);
    }
    ierr = VecCopy(dmmg[0]->x, ((AppCtx*)DMMGGetUser(dmmg,0))->Xold);CHKERRQ(ierr);

    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 



    if (tsCtx->itstep == 665000)
    {
      KSP          ksp;
      PC           pc;
      Mat          mat, pmat;
      MatStructure flag;
      PetscViewer  viewer;
      char         file[PETSC_MAX_PATH_LEN];

      ierr = PetscStrcpy(file, "matrix");CHKERRQ(ierr);

      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);

      ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);

      ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);

      ierr = PCGetOperators(pc, &mat, &pmat, &flag);CHKERRQ(ierr);

      ierr = MatView(mat, viewer);CHKERRQ(ierr);

      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
      SETERRQ(1,"Done saving Jacobian");
    }


    tsCtx->t += tsCtx->dt;

    /* save restart solution if requested at a particular time, then exit */
    if (tsCtx->dump_time > 0.0 && tsCtx->t >= tsCtx->dump_time) {
      Vec v = DMMGGetx(dmmg);
      ierr = VecView(v,PETSC_VIEWER_BINARY_WORLD);CHKERRQ(ierr);
      SETERRQ1(1,"Saved solution at time %G",tsCtx->t);
    }

    if (ts_monitor)
    {
      ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
      ierr = SNESGetLinearSolveIterations(snes, &lits);CHKERRQ(ierr);
      ierr = SNESGetNonlinearStepFailures(snes, &nfails);CHKERRQ(ierr);
      ierr = SNESGetFunctionNorm(snes, &tsCtx->fnorm);CHKERRQ(ierr);

      nfailsCum += nfails;
      if (nfailsCum >= 2)
        SETERRQ(1, "unable to find a newton step");

      ierr = PetscPrintf(PETSC_COMM_WORLD,
                         "time step = %D, time = %G, number of nonlinear steps = %D, "
                         "number of linear steps = %D, norm of the function = %G\n",
			 tsCtx->itstep + 1, tsCtx->t, its, lits, PetscAbsScalar(tsCtx->fnorm));CHKERRQ(ierr);

      /* send solution over to Matlab, to be visualized (using ex29.m) */
      if (!param->PreLoading && tsCtx->socketviewer)
      {
        Vec v;
        ierr = SNESGetSolution(snes, &v);CHKERRQ(ierr);
#if defined(PETSC_USE_SOCKET_VIEWER)
        ierr = VecView(v, tsCtx->socketviewer);CHKERRQ(ierr);
#endif
      }
    }

    if (!param->PreLoading) {
      if (param->draw_contours) {
	ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
      }

      if (1 && ts_monitor) {
	/* compute maxima */
	ComputeMaxima((DA) dmmg[param->mglevels-1]->dm,dmmg[param->mglevels-1]->x,tsCtx->t);
	/* output */
	if (ic_out++ == (int)(tsCtx->dt_out / tsCtx->dt + .5)) {
#ifdef HAVE_DA_HDF
          char fname[PETSC_MAX_PATH_LEN];

          sprintf(fname, "out-%G.hdf", tsCtx->t);
          ierr = DAVecHDFOutput(DMMGGetDA(dmmg), DMMGGetx(dmmg), fname);CHKERRQ(ierr);
#else
/*
          ierr = Gnuplot(DMMGGetDA(dmmg), DMMGGetx(dmmg), tsCtx->t);CHKERRQ(ierr);
*/
#endif
	  ic_out = 1;
        }
      }
    }
  } /* End of time step loop */
 
  if (!param->PreLoading){ 
    ierr = SNESGetFunctionNorm(snes,&tsCtx->fnorm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "timesteps %D fnorm = %G\n",
		       tsCtx->itstep, PetscAbsScalar(tsCtx->fnorm));CHKERRQ(ierr);
  }

  if (param->PreLoading) {
    tsCtx->fnorm_ini = 0.0;
  }
 
  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AddTSTermLocal"
PetscErrorCode AddTSTermLocal(DALocalInfo* info,Field **x,Field **f,AppCtx *user)
/*---------------------------------------------------------------------*/
{
  TstepCtx       *tsCtx = user->tsCtx;
  DA             da = info->da;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscInt       xints,xinte,yints,yinte;
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

  ierr  = DAVecGetArray(da,user->Xold,&xold);CHKERRQ(ierr);
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      f[j][i].U += dtinv*(x[j][i].U-xold[j][i].U);
      f[j][i].F += dtinv*(x[j][i].F-xold[j][i].F);
    }
  }
  ierr = DAVecRestoreArray(da,user->Xold,&xold);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AddTSTermLocal2"
PetscErrorCode AddTSTermLocal2(DALocalInfo* info,Field **x,Field **f,AppCtx *user)
/*---------------------------------------------------------------------*/
{
  TstepCtx       *tsCtx = user->tsCtx;
  DA             da = info->da;
  PetscErrorCode ierr;
  PetscInt       i,j,xints,xinte,yints,yinte;
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

  ierr  = DAVecGetArray(da,user->Xoldold,&xoldold);CHKERRQ(ierr);
  ierr  = DAVecGetArray(da,user->Xold,&xold);CHKERRQ(ierr);
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      f[j][i].U += dtinv * (onep5 * x[j][i].U - two * xold[j][i].U +
                            p5 * xoldold[j][i].U);
      f[j][i].F += dtinv * (onep5 * x[j][i].F - two * xold[j][i].F +
                            p5 * xoldold[j][i].F);
    }
  }
  ierr = DAVecRestoreArray(da,user->Xoldold,&xoldold);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,user->Xold,&xold);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Creates the null space of the Jacobian for a particular level */
#undef __FUNCT__
#define __FUNCT__ "CreateNullSpace"
PetscErrorCode CreateNullSpace(DMMG dmmg,Vec *nulls)
{
  PetscErrorCode ierr;
  PetscInt       i,N,rstart,rend;
  PetscScalar    scale,*vx;

  PetscFunctionBegin;
  ierr  = VecGetSize(nulls[0],&N);CHKERRQ(ierr);
  scale = 2.0/sqrt((PetscReal)N); 
  ierr  = VecGetArray(nulls[0],&vx);CHKERRQ(ierr);
  ierr  = VecGetOwnershipRange(nulls[0],&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    if (!(i % 4)) vx[i-rstart] = scale;
    else          vx[i-rstart] = 0.0;
  }
  ierr  = VecRestoreArray(nulls[0],&vx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    This is an experimental function and can be safely ignored.
*/
PetscErrorCode FormFunctionLocali(DALocalInfo *info,MatStencil *st,Field **x,PetscScalar *f,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  TstepCtx       *tsCtx = user->tsCtx;
  Parameters      *param = user->param;
  PetscErrorCode ierr;
  PetscInt       i,j,c;
  PetscInt       xints,xinte,yints,yinte;
  PassiveReal    hx,hy,dhx,dhy,hxdhy,hydhx,hxhy,dhxdhy;
  PassiveReal    de2,rhos2,nu,eta,dde2;
  PassiveReal    two = 2.0,one = 1.0,p5 = 0.5;
  PassiveReal    F_eq_x,By_eq,dtinv;
  PetscScalar    xx;
  PetscScalar    vx,vy,avx,avy,vxp,vxm,vyp,vym;
  PetscScalar    Bx,By,aBx,aBy,Bxp,Bxm,Byp,Bym;
  PassiveField   **xold;

  PetscFunctionBegin;
  de2     = sqr(param->d_e);
  rhos2   = sqr(param->rho_s);
  nu      = param->nu;
  eta     = param->eta;
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
      F_eq_x = - (1. + de2) * sin(PetscAbsScalar(xx));
      By_eq = sin(PetscAbsScalar(xx));
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

      ierr  = DAVecGetArray(info->da,user->Xold,&xold);CHKERRQ(ierr);
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
          break;
      }
      ierr = DAVecRestoreArray(info->da,user->Xold,&xold);CHKERRQ(ierr);


  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  /*  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr); FIXME */
  PetscFunctionReturn(0);
} 
/*
    This is an experimental function and can be safely ignored.
*/
PetscErrorCode FormFunctionLocali4(DALocalInfo *info,MatStencil *st,Field **x,PetscScalar *ff,void *ptr)
{
  Field          *f  = (Field *)ff;
  AppCtx         *user = (AppCtx*)ptr;
  TstepCtx       *tsCtx = user->tsCtx;
  Parameters     *param = user->param;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscInt       xints,xinte,yints,yinte;
  PassiveReal    hx,hy,dhx,dhy,hxdhy,hydhx,hxhy,dhxdhy;
  PassiveReal    de2,rhos2,nu,eta,dde2;
  PassiveReal    two = 2.0,one = 1.0,p5 = 0.5;
  PassiveReal    F_eq_x,By_eq,dtinv;
  PetscScalar    xx;
  PetscScalar    vx,vy,avx,avy,vxp,vxm,vyp,vym;
  PetscScalar    Bx,By,aBx,aBy,Bxp,Bxm,Byp,Bym;
  PassiveField   **xold;

  PetscFunctionBegin;
  de2     = sqr(param->d_e);
  rhos2   = sqr(param->rho_s);
  nu      = param->nu;
  eta     = param->eta;
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


  i = st->i; j = st->j; 

#ifdef EQ
      xx = i * hx;
      F_eq_x = - (1. + de2) * sin(PetscAbsScalar(xx));
      By_eq = sin(PetscAbsScalar(xx));
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

      ierr  = DAVecGetArray(info->da,user->Xold,&xold);CHKERRQ(ierr);
      dtinv = hxhy/(tsCtx->dt);
      

        
          /* Lap(phi) - U */
          f->phi = (Lapl(x,phi,i,j) - x[j][i].U) * hxhy;
          

          
	  /* psi - d_e^2 * Lap(psi) - F */
	  f->psi = (x[j][i].psi - de2 * Lapl(x,psi,i,j) - x[j][i].F) * hxhy;
          

        
          /* vx * U_x + vy * U_y - (B_x * F_x + B_y * F_y) / d_e^2
            - nu Lap(U) */
	  f->U  =
	    ((vxp * (D_xm(x,U,i,j)) + vxm * (D_xp(x,U,i,j)) +
	      vyp * (D_ym(x,U,i,j)) + vym * (D_yp(x,U,i,j))) -
	     (Bxp * (D_xm(x,F,i,j) + F_eq_x) + Bxm * (D_xp(x,F,i,j) + F_eq_x) +
	      Byp * (D_ym(x,F,i,j)) + Bym * (D_yp(x,F,i,j))) * dde2 -
	     nu * Lapl(x,U,i,j)) * hxhy;
          f->U += dtinv*(x[j][i].U-xold[j][i].U);
	  

        
          /* vx * F_x + vy * F_y - rho_s^2 * (B_x * U_x + B_y * U_y)
           - eta * Lap(psi) */
          f->F  =
            ((vxp * (D_xm(x,F,i,j) + F_eq_x) + vxm * (D_xp(x,F,i,j) + F_eq_x) +
             vyp * (D_ym(x,F,i,j)) + vym * (D_yp(x,F,i,j))) -
            (Bxp * (D_xm(x,U,i,j)) + Bxm * (D_xp(x,U,i,j)) +
             Byp * (D_ym(x,U,i,j)) + Bym * (D_yp(x,U,i,j))) * rhos2 -
            eta * Lapl(x,psi,i,j)) * hxhy;
          f->F += dtinv*(x[j][i].F-xold[j][i].F);
         
      
      ierr = DAVecRestoreArray(info->da,user->Xold,&xold);CHKERRQ(ierr);


  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  /*  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr); FIXME */
  PetscFunctionReturn(0);
} 
