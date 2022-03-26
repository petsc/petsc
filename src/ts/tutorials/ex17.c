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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,11,1,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da,&u));

  /* Initialize user application context */
  user.c            = -30.0;
  user.boundary     = 0;  /* 0: Dirichlet BC; 1: Neumann BC */
  user.viewJacobian = PETSC_FALSE;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-boundary",&user.boundary,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-viewJacobian",&user.viewJacobian));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetType(ts,TSTHETA));
  PetscCall(TSThetaSetTheta(ts,1.0)); /* Make the Theta method behave like backward Euler */
  PetscCall(TSSetIFunction(ts,NULL,FormIFunction,&user));

  PetscCall(DMSetMatType(da,MATAIJ));
  PetscCall(DMCreateMatrix(da,&J));
  jacType = JACOBIAN_ANALYTIC; /* use user-provide Jacobian */

  PetscCall(TSSetDM(ts,da)); /* Use TSGetDM() to access. Setting here allows easy use of geometric multigrid. */

  ftime = 1.0;
  PetscCall(TSSetMaxTime(ts,ftime));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(ts,u,&user));
  PetscCall(TSSetSolution(ts,u));
  dt   = .01;
  PetscCall(TSSetTimeStep(ts,dt));

  /* Use slow fd Jacobian or fast fd Jacobian with colorings.
     Note: this requirs snes which is not created until TSSetUp()/TSSetFromOptions() is called */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Options for Jacobian evaluation",NULL);PetscCall(ierr);
  PetscCall(PetscOptionsEnum("-jac_type","Type of Jacobian","",JacobianTypes,(PetscEnum)jacType,(PetscEnum*)&jacType,0));
  ierr = PetscOptionsEnd();PetscCall(ierr);
  if (jacType == JACOBIAN_ANALYTIC) {
    PetscCall(TSSetIJacobian(ts,J,J,FormIJacobian,&user));
  } else if (jacType == JACOBIAN_FD_COLORING) {
    SNES snes;
    PetscCall(TSGetSNES(ts,&snes));
    PetscCall(SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,0));
  } else if (jacType == JACOBIAN_FD_FULL) {
    SNES snes;
    PetscCall(TSGetSNES(ts,&snes));
    PetscCall(SNESSetJacobian(snes,J,J,SNESComputeJacobianDefault,&user));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Integrate ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Compute diagnostics of the solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecNorm(u,NORM_1,&norm));
  PetscCall(VecMax(u,NULL,&vmax));
  PetscCall(VecMin(u,NULL,&vmin));
  PetscCall(TSGetStepNumber(ts,&nsteps));
  PetscCall(TSGetTime(ts,&ftime));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"timestep %D: time %g, solution norm %g, max %g, min %g\n",nsteps,(double)ftime,(double)norm,(double)vmax,(double)vmin));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&u));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
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
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMGetLocalVector(da,&localU));
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU));
  PetscCall(DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU));

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArrayRead(da,localU,&u));
  PetscCall(DMDAVecGetArrayRead(da,Udot,&udot));
  PetscCall(DMDAVecGetArray(da,F,&f));

  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

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
  PetscCall(DMDAVecRestoreArrayRead(da,localU,&u));
  PetscCall(DMDAVecRestoreArrayRead(da,Udot,&udot));
  PetscCall(DMDAVecRestoreArray(da,F,&f));
  PetscCall(DMRestoreLocalVector(da,&localU));
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
  PetscCall(TSGetDM(ts,&da));
  PetscCall(MatGetOwnershipRange(Jpre,&rstart,&rend));
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));
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
    PetscCall(MatSetValuesStencil(Jpre,1,&row,nc,col,vals,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  if (user->viewJacobian) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Jpre:\n"));
    PetscCall(MatView(Jpre,PETSC_VIEWER_STDOUT_WORLD));
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
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  hx = 1.0/(PetscReal)(Mx-1);

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArray(da,U,&u));

  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

  /* Compute function over the locally owned part of the grid */
  for (i=xs; i<xs+xm; i++) {
    x = i*hx;
    r = PetscSqrtReal((x-.5)*(x-.5));
    if (r < .125) u[i] = PetscExpReal(c*r*r*r);
    else          u[i] = 0.0;
  }

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArray(da,U,&u));
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
