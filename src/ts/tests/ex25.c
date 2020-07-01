static const char help[] = "Call PetscInitialize multiple times.\n";
/*
   This example is based on the Brusselator tutorial of the same name, but tests multiple calls to PetscInitialize().
   This is a bad "convergence study" because it only compares min and max values of the solution rather than comparing
   norms of the errors.  For convergence studies, we recommend invoking PetscInitialize() only once and comparing norms
   of errors (perhaps estimated using an accurate reference solution).

   Time-dependent Brusselator reaction-diffusion PDE in 1d. Demonstrates IMEX methods and multiple solves.

   u_t - alpha u_xx = A + u^2 v - (B+1) u
   v_t - alpha v_xx = B u - u^2 v
   0 < x < 1;
   A = 1, B = 3, alpha = 1/10

   Initial conditions:
   u(x,0) = 1 + sin(2 pi x)
   v(x,0) = 3

   Boundary conditions:
   u(0,t) = u(1,t) = 1
   v(0,t) = v(1,t) = 3
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct _User *User;
struct _User {
  PetscReal A,B;                /* Reaction coefficients */
  PetscReal alpha;              /* Diffusion coefficient */
  PetscReal uleft,uright;       /* Dirichlet boundary conditions */
  PetscReal vleft,vright;       /* Dirichlet boundary conditions */
};

static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);
static int Brusselator(int,char**,PetscInt);

int main(int argc,char **argv)
{
  PetscInt       cycle;
  PetscErrorCode ierr;

  ierr = MPI_Init(&argc,&argv);if (ierr) return ierr;
  for (cycle=0; cycle<4; cycle++) {
    ierr = Brusselator(argc,argv,cycle);
    if (ierr) return 1;
  }
  ierr = MPI_Finalize();
  return ierr;
}

PetscErrorCode Brusselator(int argc,char **argv,PetscInt cycle)
{
  TS                ts;         /* nonlinear solver */
  Vec               X;          /* solution, residual vectors */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps,mx;
  PetscErrorCode    ierr;
  DM                da;
  PetscReal         ftime,hx,dt,xmax,xmin;
  struct _User      user;       /* user-defined work context */
  TSConvergedReason reason;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,11,2,2,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);

  /* Initialize user application context */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Advection-reaction options","");
  {
    user.A      = 1;
    user.B      = 3;
    user.alpha  = 0.1;
    user.uleft  = 1;
    user.uright = 1;
    user.vleft  = 3;
    user.vright = 3;
    ierr        = PetscOptionsReal("-A","Reaction rate","",user.A,&user.A,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-B","Reaction rate","",user.B,&user.B,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-alpha","Diffusion coefficient","",user.alpha,&user.alpha,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-uleft","Dirichlet boundary condition","",user.uleft,&user.uleft,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-uright","Dirichlet boundary condition","",user.uright,&user.uright,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-vleft","Dirichlet boundary condition","",user.vleft,&user.vleft,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-vright","Dirichlet boundary condition","",user.vright,&user.vright,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,FormIFunction,&user);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,&user);CHKERRQ(ierr);

  ftime = 1.0;
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = VecGetSize(X,&mx);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(mx/2-1);
  dt = 0.4 * PetscSqr(hx) / user.alpha; /* Diffusive stability limit */
  dt *= PetscPowRealInt(0.2,cycle);     /* Shrink the time step in convergence study. */
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetTolerances(ts,1e-3*PetscPowRealInt(0.5,cycle),NULL,1e-3*PetscPowRealInt(0.5,cycle),NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = VecMin(X,NULL,&xmin);CHKERRQ(ierr);
  ierr = VecMax(X,NULL,&xmax);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after % 3D steps. Range [%6.4f,%6.4f]\n",TSConvergedReasons[reason],(double)ftime,steps,(double)xmin,(double)xmax);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
  User           user = (User)ptr;
  DM             da;
  DMDALocalInfo  info;
  PetscInt       i;
  Field          *x,*xdot,*f;
  PetscReal      hx;
  Vec            Xloc;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx-1);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    if (i == 0) {
      f[i].u = hx * (x[i].u - user->uleft);
      f[i].v = hx * (x[i].v - user->vleft);
    } else if (i == info.mx-1) {
      f[i].u = hx * (x[i].u - user->uright);
      f[i].v = hx * (x[i].v - user->vright);
    } else {
      f[i].u = hx * xdot[i].u - user->alpha * (x[i-1].u - 2.*x[i].u + x[i+1].u) / hx;
      f[i].v = hx * xdot[i].v - user->alpha * (x[i-1].v - 2.*x[i].v + x[i+1].v) / hx;
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArrayRead(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User           user = (User)ptr;
  DM             da;
  DMDALocalInfo  info;
  PetscInt       i;
  PetscReal      hx;
  Field          *x,*f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscScalar u = x[i].u,v = x[i].v;
    f[i].u = hx*(user->A + u*u*v - (user->B+1)*u);
    f[i].v = hx*(user->B*u - u*u*v);
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArrayRead(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ptr)
{
  User           user = (User)ptr;
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscInt       i;
  PetscReal      hx;
  DM             da;
  Field          *x,*xdot;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,Xdot,&xdot);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    if (i == 0 || i == info.mx-1) {
      const PetscInt    row        = i,col = i;
      const PetscScalar vals[2][2] = {{hx,0},{0,hx}};
      ierr = MatSetValuesBlocked(Jpre,1,&row,1,&col,&vals[0][0],INSERT_VALUES);CHKERRQ(ierr);
    } else {
      const PetscInt    row           = i,col[] = {i-1,i,i+1};
      const PetscScalar dxxL          = -user->alpha/hx,dxx0 = 2.*user->alpha/hx,dxxR = -user->alpha/hx;
      const PetscScalar vals[2][3][2] = {{{dxxL,0},{a *hx+dxx0,0},{dxxR,0}},
                                         {{0,dxxL},{0,a*hx+dxx0},{0,dxxR}}};
      ierr = MatSetValuesBlocked(Jpre,1,&row,3,col,&vals[0][0][0],INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArrayRead(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Xdot,&xdot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  User           user = (User)ctx;
  DM             da;
  PetscInt       i;
  DMDALocalInfo  info;
  Field          *x;
  PetscReal      hx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal xi = i*hx;
    x[i].u = user->uleft*(1.-xi) + user->uright*xi + PetscSinReal(2.*PETSC_PI*xi);
    x[i].v = user->vleft*(1.-xi) + user->vright*xi;
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -ts_exact_final_time INTERPOLATE -snes_rtol 1.e-3

    test:
      suffix: 2
      args:   -ts_exact_final_time INTERPOLATE -snes_rtol 1.e-3

TEST*/

