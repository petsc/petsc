static const char help[] = "Time-dependent PDE in 1d. Simplified from ex15.c for illustrating how to solve DAEs. \n";
/*
   u_t = uxx
   0 < x < 1;
   At t=0: u(x) = exp(c*r*r*r), if r=PetscSqrtReal((x-.5)*(x-.5)) < .125
           u(x) = 0.0           if r >= .125

   Boundary conditions:
   Dirichlet BC:
   At x=0, x=1, u = 0.0

   Neumann BC:
   At x=0, x=1: du(x,t)/dx = 0

   mpiexec -n 2 ./ex17 -da_grid_x 40 -ts_max_steps 2 -snes_monitor -ksp_monitor
         ./ex17 -da_grid_x 40 -monitor_solution
         ./ex17 -da_grid_x 100  -ts_type theta -ts_theta_theta 0.5 # Midpoint is not L-stable
         ./ex17 -jac_type fd_coloring  -da_grid_x 500 -boundary 1
         ./ex17 -da_grid_x 100  -ts_type gl -ts_adapt_type none -ts_max_steps 2
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

typedef enum {JACOBIAN_ANALYTIC,JACOBIAN_FD_COLORING,JACOBIAN_FD_FULL} JacobianType;
static const char *const JacobianTypes[] = {"analytic","fd_coloring","fd_full","JacobianType","fd_",0};

/*
   User-defined data structures and routines
*/
typedef struct {
  PetscReal c;
  PetscInt  boundary;            /* Type of boundary condition */
  PetscBool viewJacobian;
} AppCtx;

static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

int main(int argc,char **argv)
{
  TS             ts;                   /* nonlinear solver */
  Vec            u;                    /* solution, residual vectors */
  Mat            J;                    /* Jacobian matrix */
  PetscInt       nsteps;
  PetscReal      vmin,vmax,norm;
  PetscErrorCode ierr;
  DM             da;
  PetscReal      ftime,dt;
  AppCtx         user;              /* user-defined work context */
  JacobianType   jacType;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,11,1,1,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMCreateGlobalVector(da,&u));

  /* Initialize user application context */
  user.c            = -30.0;
  user.boundary     = 0;  /* 0: Dirichlet BC; 1: Neumann BC */
  user.viewJacobian = PETSC_FALSE;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-boundary",&user.boundary,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-viewJacobian",&user.viewJacobian));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSTHETA));
  CHKERRQ(TSThetaSetTheta(ts,1.0)); /* Make the Theta method behave like backward Euler */
  CHKERRQ(TSSetIFunction(ts,NULL,FormIFunction,&user));

  CHKERRQ(DMSetMatType(da,MATAIJ));
  CHKERRQ(DMCreateMatrix(da,&J));
  jacType = JACOBIAN_ANALYTIC; /* use user-provide Jacobian */

  CHKERRQ(TSSetDM(ts,da)); /* Use TSGetDM() to access. Setting here allows easy use of geometric multigrid. */

  ftime = 1.0;
  CHKERRQ(TSSetMaxTime(ts,ftime));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(FormInitialSolution(ts,u,&user));
  CHKERRQ(TSSetSolution(ts,u));
  dt   = .01;
  CHKERRQ(TSSetTimeStep(ts,dt));

  /* Use slow fd Jacobian or fast fd Jacobian with colorings.
     Note: this requirs snes which is not created until TSSetUp()/TSSetFromOptions() is called */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options for Jacobian evaluation",NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsEnum("-jac_type","Type of Jacobian","",JacobianTypes,(PetscEnum)jacType,(PetscEnum*)&jacType,0));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (jacType == JACOBIAN_ANALYTIC) {
    CHKERRQ(TSSetIJacobian(ts,J,J,FormIJacobian,&user));
  } else if (jacType == JACOBIAN_FD_COLORING) {
    SNES snes;
    CHKERRQ(TSGetSNES(ts,&snes));
    CHKERRQ(SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,0));
  } else if (jacType == JACOBIAN_FD_FULL) {
    SNES snes;
    CHKERRQ(TSGetSNES(ts,&snes));
    CHKERRQ(SNESSetJacobian(snes,J,J,SNESComputeJacobianDefault,&user));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Integrate ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Compute diagnostics of the solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecNorm(u,NORM_1,&norm));
  CHKERRQ(VecMax(u,NULL,&vmax));
  CHKERRQ(VecMin(u,NULL,&vmin));
  CHKERRQ(TSGetStepNumber(ts,&nsteps));
  CHKERRQ(TSGetTime(ts,&ftime));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"timestep %D: time %g, solution norm %g, max %g, min %g\n",nsteps,(double)ftime,(double)norm,(double)vmax,(double)vmin));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------------- */
static PetscErrorCode FormIFunction(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  AppCtx         *user=(AppCtx*)ptr;
  DM             da;
  PetscInt       i,Mx,xs,xm;
  PetscReal      hx,sx;
  PetscScalar    *u,*udot,*f;
  Vec            localU;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMGetLocalVector(da,&localU));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  CHKERRQ(DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU));
  CHKERRQ(DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU));

  /* Get pointers to vector data */
  CHKERRQ(DMDAVecGetArrayRead(da,localU,&u));
  CHKERRQ(DMDAVecGetArrayRead(da,Udot,&udot));
  CHKERRQ(DMDAVecGetArray(da,F,&f));

  /* Get local grid boundaries */
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

  /* Compute function over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {
    if (user->boundary == 0) { /* Dirichlet BC */
      if (i == 0 || i == Mx-1) f[i] = u[i]; /* F = U */
      else                     f[i] = udot[i] + (2.*u[i] - u[i-1] - u[i+1])*sx;
    } else { /* Neumann BC */
      if (i == 0)         f[i] = u[0] - u[1];
      else if (i == Mx-1) f[i] = u[i] - u[i-1];
      else                f[i] = udot[i] + (2.*u[i] - u[i-1] - u[i+1])*sx;
    }
  }

  /* Restore vectors */
  CHKERRQ(DMDAVecRestoreArrayRead(da,localU,&u));
  CHKERRQ(DMDAVecRestoreArrayRead(da,Udot,&udot));
  CHKERRQ(DMDAVecRestoreArray(da,F,&f));
  CHKERRQ(DMRestoreLocalVector(da,&localU));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat J,Mat Jpre,void *ctx)
{
  PetscInt       i,rstart,rend,Mx;
  PetscReal      hx,sx;
  AppCtx         *user = (AppCtx*)ctx;
  DM             da;
  MatStencil     col[3],row;
  PetscInt       nc;
  PetscScalar    vals[3];

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(MatGetOwnershipRange(Jpre,&rstart,&rend));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));
  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
  for (i=rstart; i<rend; i++) {
    nc    = 0;
    row.i = i;
    if (user->boundary == 0 && (i == 0 || i == Mx-1)) {
      col[nc].i = i; vals[nc++] = 1.0;
    } else if (user->boundary > 0 && i == 0) { /* Left Neumann */
      col[nc].i = i;   vals[nc++] = 1.0;
      col[nc].i = i+1; vals[nc++] = -1.0;
    } else if (user->boundary > 0 && i == Mx-1) { /* Right Neumann */
      col[nc].i = i-1; vals[nc++] = -1.0;
      col[nc].i = i;   vals[nc++] = 1.0;
    } else {                    /* Interior */
      col[nc].i = i-1; vals[nc++] = -1.0*sx;
      col[nc].i = i;   vals[nc++] = 2.0*sx + a;
      col[nc].i = i+1; vals[nc++] = -1.0*sx;
    }
    CHKERRQ(MatSetValuesStencil(Jpre,1,&row,nc,col,vals,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  if (user->viewJacobian) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Jpre:\n"));
    CHKERRQ(MatView(Jpre,PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(TS ts,Vec U,void *ptr)
{
  AppCtx         *user=(AppCtx*)ptr;
  PetscReal      c    =user->c;
  DM             da;
  PetscInt       i,xs,xm,Mx;
  PetscScalar    *u;
  PetscReal      hx,x,r;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetDM(ts,&da));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1);

  /* Get pointers to vector data */
  CHKERRQ(DMDAVecGetArray(da,U,&u));

  /* Get local grid boundaries */
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

  /* Compute function over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    r = PetscSqrtReal((x-.5)*(x-.5));
    if (r < .125) u[i] = PetscExpReal(c*r*r*r);
    else          u[i] = 0.0;
  }

  /* Restore vectors */
  CHKERRQ(DMDAVecRestoreArray(da,U,&u));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      requires: !single
      args: -da_grid_x 40 -ts_max_steps 2 -snes_monitor_short -ksp_monitor_short -ts_monitor

    test:
      suffix: 2
      requires: !single
      args: -da_grid_x 100 -ts_type theta -ts_theta_theta 0.5

TEST*/
