static const char help[] = "Time-dependent Brusselator reaction-diffusion PDE in 1d formulated as a PDAE. Demonstrates solving PDEs with algebraic constraints (PDAE).\n";
/*
   u_t - alpha u_xx = A + u^2 v - (B+1) u
   v_t - alpha v_xx = B u - u^2 v
   0 < x < 1;
   A = 1, B = 3, alpha = 1/50

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

int main(int argc,char **argv)
{
  TS                ts;         /* nonlinear solver */
  Vec               X;          /* solution, residual vectors */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps,mx;
  PetscErrorCode    ierr;
  DM                da;
  PetscReal         ftime,hx,dt;
  struct _User      user;       /* user-defined work context */
  TSConvergedReason reason;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,11,2,2,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMCreateGlobalVector(da,&X));

  /* Initialize user application context */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Advection-reaction options","");CHKERRQ(ierr);
  {
    user.A      = 1;
    user.B      = 3;
    user.alpha  = 0.02;
    user.uleft  = 1;
    user.uright = 1;
    user.vleft  = 3;
    user.vright = 3;
    CHKERRQ(PetscOptionsReal("-A","Reaction rate","",user.A,&user.A,NULL));
    CHKERRQ(PetscOptionsReal("-B","Reaction rate","",user.B,&user.B,NULL));
    CHKERRQ(PetscOptionsReal("-alpha","Diffusion coefficient","",user.alpha,&user.alpha,NULL));
    CHKERRQ(PetscOptionsReal("-uleft","Dirichlet boundary condition","",user.uleft,&user.uleft,NULL));
    CHKERRQ(PetscOptionsReal("-uright","Dirichlet boundary condition","",user.uright,&user.uright,NULL));
    CHKERRQ(PetscOptionsReal("-vleft","Dirichlet boundary condition","",user.vleft,&user.vleft,NULL));
    CHKERRQ(PetscOptionsReal("-vright","Dirichlet boundary condition","",user.vright,&user.vright,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetDM(ts,da));
  CHKERRQ(TSSetType(ts,TSARKIMEX));
  CHKERRQ(TSSetEquationType(ts,TS_EQ_DAE_IMPLICIT_INDEX1));
  CHKERRQ(TSSetRHSFunction(ts,NULL,FormRHSFunction,&user));
  CHKERRQ(TSSetIFunction(ts,NULL,FormIFunction,&user));
  CHKERRQ(DMSetMatType(da,MATAIJ));
  CHKERRQ(DMCreateMatrix(da,&J));
  CHKERRQ(TSSetIJacobian(ts,J,J,FormIJacobian,&user));

  ftime = 10.0;
  CHKERRQ(TSSetMaxTime(ts,ftime));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(FormInitialSolution(ts,X,&user));
  CHKERRQ(TSSetSolution(ts,X));
  CHKERRQ(VecGetSize(X,&mx));
  hx = 1.0/(PetscReal)(mx/2-1);
  dt = 0.4 * PetscSqr(hx) / user.alpha; /* Diffusive stability limit */
  CHKERRQ(TSSetTimeStep(ts,dt));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,X));
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(TSGetConvergedReason(ts,&reason));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&da));
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

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  hx   = 1.0/(PetscReal)(info.mx-1);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  CHKERRQ(DMGetLocalVector(da,&Xloc));
  CHKERRQ(DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc));
  CHKERRQ(DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc));

  /* Get pointers to vector data */
  CHKERRQ(DMDAVecGetArrayRead(da,Xloc,&x));
  CHKERRQ(DMDAVecGetArrayRead(da,Xdot,&xdot));
  CHKERRQ(DMDAVecGetArray(da,F,&f));

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
  CHKERRQ(DMDAVecRestoreArrayRead(da,Xloc,&x));
  CHKERRQ(DMDAVecRestoreArrayRead(da,Xdot,&xdot));
  CHKERRQ(DMDAVecRestoreArray(da,F,&f));
  CHKERRQ(DMRestoreLocalVector(da,&Xloc));
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

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  hx   = 1.0/(PetscReal)(info.mx-1);

  /* Get pointers to vector data */
  CHKERRQ(DMDAVecGetArrayRead(da,X,&x));
  CHKERRQ(DMDAVecGetArray(da,F,&f));

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscScalar u = x[i].u,v = x[i].v;
    f[i].u = hx*(user->A + u*u*v - (user->B+1)*u);
    f[i].v = hx*(user->B*u - u*u*v);
  }

  /* Restore vectors */
  CHKERRQ(DMDAVecRestoreArrayRead(da,X,&x));
  CHKERRQ(DMDAVecRestoreArray(da,F,&f));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat J,Mat Jpre,void *ptr)
{
  User           user = (User)ptr;
  DMDALocalInfo  info;
  PetscInt       i;
  PetscReal      hx;
  DM             da;
  Field          *x,*xdot;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  hx   = 1.0/(PetscReal)(info.mx-1);

  /* Get pointers to vector data */
  CHKERRQ(DMDAVecGetArrayRead(da,X,&x));
  CHKERRQ(DMDAVecGetArrayRead(da,Xdot,&xdot));

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    if (i == 0 || i == info.mx-1) {
      const PetscInt    row        = i,col = i;
      const PetscScalar vals[2][2] = {{hx,0},{0,hx}};
      CHKERRQ(MatSetValuesBlocked(Jpre,1,&row,1,&col,&vals[0][0],INSERT_VALUES));
    } else {
      const PetscInt    row           = i,col[] = {i-1,i,i+1};
      const PetscScalar dxxL          = -user->alpha/hx,dxx0 = 2.*user->alpha/hx,dxxR = -user->alpha/hx;
      const PetscScalar vals[2][3][2] = {{{dxxL,0},{a *hx+dxx0,0},{dxxR,0}},
                                         {{0,dxxL},{0,a*hx+dxx0},{0,dxxR}}};
      CHKERRQ(MatSetValuesBlocked(Jpre,1,&row,3,col,&vals[0][0][0],INSERT_VALUES));
    }
  }

  /* Restore vectors */
  CHKERRQ(DMDAVecRestoreArrayRead(da,X,&x));
  CHKERRQ(DMDAVecRestoreArrayRead(da,Xdot,&xdot));

  CHKERRQ(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
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

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  hx   = 1.0/(PetscReal)(info.mx-1);

  /* Get pointers to vector data */
  CHKERRQ(DMDAVecGetArray(da,X,&x));

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal xi = i*hx;
    x[i].u = user->uleft*(1.-xi) + user->uright*xi + PetscSinReal(2.*PETSC_PI*xi);
    x[i].v = user->vleft*(1.-xi) + user->vright*xi;
  }
  CHKERRQ(DMDAVecRestoreArray(da,X,&x));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -nox -da_grid_x 20 -ts_monitor_draw_solution -ts_type rosw -ts_rosw_type 2p -ts_dt 5e-2 -ts_adapt_type none

TEST*/
