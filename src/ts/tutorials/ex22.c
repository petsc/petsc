static const char help[] = "Time-dependent advection-reaction PDE in 1d, demonstrates IMEX methods.\n";
/*
   u_t + a1*u_x = -k1*u + k2*v + s1
   v_t + a2*v_x = k1*u - k2*v + s2
   0 < x < 1;
   a1 = 1, k1 = 10^6, s1 = 0,
   a2 = 0, k2 = 2*k1, s2 = 1

   Initial conditions:
   u(x,0) = 1 + s2*x
   v(x,0) = k0/k1*u(x,0) + s1/k1

   Upstream boundary conditions:
   u(0,t) = 1-sin(12*t)^4

   Useful command-line parameters:
   -ts_arkimex_fully_implicit - treats advection implicitly, only relevant with -ts_type arkimex (default)
   -ts_type rosw - use Rosenbrock-W method (linearly implicit IMEX, amortizes assembly and preconditioner setup)
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

typedef PetscScalar Field[2];

typedef struct _User *User;
struct _User {
  PetscReal a[2];              /* Advection speeds */
  PetscReal k[2];              /* Reaction coefficients */
  PetscReal s[2];              /* Source terms */
};

static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

int main(int argc,char **argv)
{
  TS                ts;         /* time integrator */
  SNES              snes;       /* nonlinear solver */
  SNESLineSearch    linesearch; /* line search */
  Vec               X;          /* solution, residual vectors */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps,mx;
  PetscErrorCode    ierr;
  DM                da;
  PetscReal         ftime,dt;
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
    user.a[0] = 1;           ierr = PetscOptionsReal("-a0","Advection rate 0","",user.a[0],&user.a[0],NULL);CHKERRQ(ierr);
    user.a[1] = 0;           ierr = PetscOptionsReal("-a1","Advection rate 1","",user.a[1],&user.a[1],NULL);CHKERRQ(ierr);
    user.k[0] = 1e6;         ierr = PetscOptionsReal("-k0","Reaction rate 0","",user.k[0],&user.k[0],NULL);CHKERRQ(ierr);
    user.k[1] = 2*user.k[0]; ierr = PetscOptionsReal("-k1","Reaction rate 1","",user.k[1],&user.k[1],NULL);CHKERRQ(ierr);
    user.s[0] = 0;           ierr = PetscOptionsReal("-s0","Source 0","",user.s[0],&user.s[0],NULL);CHKERRQ(ierr);
    user.s[1] = 1;           ierr = PetscOptionsReal("-s1","Source 1","",user.s[1],&user.s[1],NULL);CHKERRQ(ierr);
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

  /* A line search in the nonlinear solve can fail due to ill-conditioning unless an absolute tolerance is set. Since
   * this problem is linear, we deactivate the line search. For a linear problem, it is usually recommended to also use
   * SNESSetType(snes,SNESKSPONLY). */
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESGetLineSearch(snes,&linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);CHKERRQ(ierr);

  ftime = .1;
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = VecGetSize(X,&mx);CHKERRQ(ierr);
  dt   = .1 * PetscMax(user.a[0],user.a[1]) / mx; /* Advective CFL, I don't know why it needs so much safety factor. */
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);

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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,steps);CHKERRQ(ierr);

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
  Field          *f;
  const Field    *x,*xdot;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,X,(void*)&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,Xdot,(void*)&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    f[i][0] = xdot[i][0] + user->k[0]*x[i][0] - user->k[1]*x[i][1] - user->s[0];
    f[i][1] = xdot[i][1] - user->k[0]*x[i][0] + user->k[1]*x[i][1] - user->s[1];
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArrayRead(da,X,(void*)&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Xdot,(void*)&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User           user = (User)ptr;
  DM             da;
  Vec            Xloc;
  DMDALocalInfo  info;
  PetscInt       i,j;
  PetscReal      hx;
  Field          *f;
  const Field    *x;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)info.mx;

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
  ierr = DMDAVecGetArrayRead(da,Xloc,(void*)&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    const PetscReal *a  = user->a;
    PetscReal       u0t[2];
    u0t[0] = 1.0 - PetscPowRealInt(PetscSinReal(12*t),4);
    u0t[1] = 0.0;
    for (j=0; j<2; j++) {
      if (i == 0)              f[i][j] = a[j]/hx*(1./3*u0t[j] + 0.5*x[i][j] - x[i+1][j] + 1./6*x[i+2][j]);
      else if (i == 1)         f[i][j] = a[j]/hx*(-1./12*u0t[j] + 2./3*x[i-1][j] - 2./3*x[i+1][j] + 1./12*x[i+2][j]);
      else if (i == info.mx-2) f[i][j] = a[j]/hx*(-1./6*x[i-2][j] + x[i-1][j] - 0.5*x[i][j] - 1./3*x[i+1][j]);
      else if (i == info.mx-1) f[i][j] = a[j]/hx*(-x[i][j] + x[i-1][j]);
      else                     f[i][j] = a[j]/hx*(-1./12*x[i-2][j] + 2./3*x[i-1][j] - 2./3*x[i+1][j] + 1./12*x[i+2][j]);
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArrayRead(da,Xloc,(void*)&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
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
  DM             da;
  const Field    *x,*xdot;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,X,(void*)&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,Xdot,(void*)&xdot);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    const PetscReal *k = user->k;
    PetscScalar     v[2][2];

    v[0][0] = a + k[0]; v[0][1] =  -k[1];
    v[1][0] =    -k[0]; v[1][1] = a+k[1];
    ierr    = MatSetValuesBlocked(Jpre,1,&i,1,&i,&v[0][0],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArrayRead(da,X,(void*)&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Xdot,(void*)&xdot);CHKERRQ(ierr);

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
  hx   = 1.0/(PetscReal)info.mx;

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal r = (i+1)*hx,ik = user->k[1] != 0.0 ? 1.0/user->k[1] : 1.0;
    x[i][0] = 1 + user->s[1]*r;
    x[i][1] = user->k[0]*ik*x[i][0] + user->s[1]*ik;
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -nox -da_grid_x 200 -ts_monitor_draw_solution -ts_arkimex_type 4 -ts_adapt_type none -ts_dt .005 -ts_max_time .1
      requires: !single

    test:
      suffix: 2
      args: -nox -da_grid_x 200 -ts_monitor_draw_solution -ts_type rosw -ts_dt 1e-3 -ts_adapt_type none -ts_dt .005 -ts_max_time .1
      nsize: 2

    test:
      suffix: 3
      args: -nox -da_grid_x 200 -ts_monitor_draw_solution -ts_type rosw -ts_rosw_type ra34pw2 -ts_dt 5e-3 -ts_max_time .1  -ts_adapt_type none
      nsize: 2

    test:
      suffix: 4
      args: -ts_type eimex -da_grid_x 200 -ts_eimex_order_adapt true -ts_dt 0.001 -ts_monitor -ts_max_steps 100
      filter: sed "s/ITS/TIME/g"
      nsize: 2

TEST*/
