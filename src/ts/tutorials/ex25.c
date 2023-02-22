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
  PetscScalar u, v;
} Field;

typedef struct _User *User;
struct _User {
  PetscReal A, B;          /* Reaction coefficients */
  PetscReal alpha;         /* Diffusion coefficient */
  PetscReal uleft, uright; /* Dirichlet boundary conditions */
  PetscReal vleft, vright; /* Dirichlet boundary conditions */
};

static PetscErrorCode FormRHSFunction(TS, PetscReal, Vec, Vec, void *);
static PetscErrorCode FormIFunction(TS, PetscReal, Vec, Vec, Vec, void *);
static PetscErrorCode FormIJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
static PetscErrorCode FormInitialSolution(TS, Vec, void *);

int main(int argc, char **argv)
{
  TS                ts; /* nonlinear solver */
  Vec               X;  /* solution, residual vectors */
  Mat               J;  /* Jacobian matrix */
  PetscInt          steps, mx;
  DM                da;
  PetscReal         ftime, hx, dt;
  struct _User      user; /* user-defined work context */
  TSConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 11, 2, 2, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da, &X));

  /* Initialize user application context */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Advection-reaction options", "");
  {
    user.A      = 1;
    user.B      = 3;
    user.alpha  = 0.02;
    user.uleft  = 1;
    user.uright = 1;
    user.vleft  = 3;
    user.vright = 3;
    PetscCall(PetscOptionsReal("-A", "Reaction rate", "", user.A, &user.A, NULL));
    PetscCall(PetscOptionsReal("-B", "Reaction rate", "", user.B, &user.B, NULL));
    PetscCall(PetscOptionsReal("-alpha", "Diffusion coefficient", "", user.alpha, &user.alpha, NULL));
    PetscCall(PetscOptionsReal("-uleft", "Dirichlet boundary condition", "", user.uleft, &user.uleft, NULL));
    PetscCall(PetscOptionsReal("-uright", "Dirichlet boundary condition", "", user.uright, &user.uright, NULL));
    PetscCall(PetscOptionsReal("-vleft", "Dirichlet boundary condition", "", user.vleft, &user.vleft, NULL));
    PetscCall(PetscOptionsReal("-vright", "Dirichlet boundary condition", "", user.vright, &user.vright, NULL));
  }
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, da));
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSSetEquationType(ts, TS_EQ_DAE_IMPLICIT_INDEX1));
  PetscCall(TSSetRHSFunction(ts, NULL, FormRHSFunction, &user));
  PetscCall(TSSetIFunction(ts, NULL, FormIFunction, &user));
  PetscCall(DMSetMatType(da, MATAIJ));
  PetscCall(DMCreateMatrix(da, &J));
  PetscCall(TSSetIJacobian(ts, J, J, FormIJacobian, &user));

  ftime = 10.0;
  PetscCall(TSSetMaxTime(ts, ftime));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(ts, X, &user));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(VecGetSize(X, &mx));
  hx = 1.0 / (PetscReal)(mx / 2 - 1);
  dt = 0.4 * PetscSqr(hx) / user.alpha; /* Diffusive stability limit */
  PetscCall(TSSetTimeStep(ts, dt));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, X));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %" PetscInt_FMT " steps\n", TSConvergedReasons[reason], (double)ftime, steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&X));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode FormIFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ptr)
{
  User          user = (User)ptr;
  DM            da;
  DMDALocalInfo info;
  PetscInt      i;
  Field        *x, *xdot, *f;
  PetscReal     hx;
  Vec           Xloc;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  hx = 1.0 / (PetscReal)(info.mx - 1);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGetLocalVector(da, &Xloc));
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, Xloc));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, Xloc));

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArrayRead(da, Xloc, &x));
  PetscCall(DMDAVecGetArrayRead(da, Xdot, &xdot));
  PetscCall(DMDAVecGetArray(da, F, &f));

  /* Compute function over the locally owned part of the grid */
  for (i = info.xs; i < info.xs + info.xm; i++) {
    if (i == 0) {
      f[i].u = hx * (x[i].u - user->uleft);
      f[i].v = hx * (x[i].v - user->vleft);
    } else if (i == info.mx - 1) {
      f[i].u = hx * (x[i].u - user->uright);
      f[i].v = hx * (x[i].v - user->vright);
    } else {
      f[i].u = hx * xdot[i].u - user->alpha * (x[i - 1].u - 2. * x[i].u + x[i + 1].u) / hx;
      f[i].v = hx * xdot[i].v - user->alpha * (x[i - 1].v - 2. * x[i].v + x[i + 1].v) / hx;
    }
  }

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArrayRead(da, Xloc, &x));
  PetscCall(DMDAVecRestoreArrayRead(da, Xdot, &xdot));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(DMRestoreLocalVector(da, &Xloc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ptr)
{
  User          user = (User)ptr;
  DM            da;
  DMDALocalInfo info;
  PetscInt      i;
  PetscReal     hx;
  Field        *x, *f;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  hx = 1.0 / (PetscReal)(info.mx - 1);

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArrayRead(da, X, &x));
  PetscCall(DMDAVecGetArray(da, F, &f));

  /* Compute function over the locally owned part of the grid */
  for (i = info.xs; i < info.xs + info.xm; i++) {
    PetscScalar u = x[i].u, v = x[i].v;
    f[i].u = hx * (user->A + u * u * v - (user->B + 1) * u);
    f[i].v = hx * (user->B * u - u * u * v);
  }

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArrayRead(da, X, &x));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --------------------------------------------------------------------- */
/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
PetscErrorCode FormIJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat J, Mat Jpre, void *ptr)
{
  User          user = (User)ptr;
  DMDALocalInfo info;
  PetscInt      i;
  PetscReal     hx;
  DM            da;
  Field        *x, *xdot;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  hx = 1.0 / (PetscReal)(info.mx - 1);

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArrayRead(da, X, &x));
  PetscCall(DMDAVecGetArrayRead(da, Xdot, &xdot));

  /* Compute function over the locally owned part of the grid */
  for (i = info.xs; i < info.xs + info.xm; i++) {
    if (i == 0 || i == info.mx - 1) {
      const PetscInt    row = i, col = i;
      const PetscScalar vals[2][2] = {
        {hx, 0 },
        {0,  hx}
      };
      PetscCall(MatSetValuesBlocked(Jpre, 1, &row, 1, &col, &vals[0][0], INSERT_VALUES));
    } else {
      const PetscInt    row = i, col[] = {i - 1, i, i + 1};
      const PetscScalar dxxL = -user->alpha / hx, dxx0 = 2. * user->alpha / hx, dxxR = -user->alpha / hx;
      const PetscScalar vals[2][3][2] = {
        {{dxxL, 0}, {a * hx + dxx0, 0}, {dxxR, 0}},
        {{0, dxxL}, {0, a * hx + dxx0}, {0, dxxR}}
      };
      PetscCall(MatSetValuesBlocked(Jpre, 1, &row, 3, col, &vals[0][0][0], INSERT_VALUES));
    }
  }

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArrayRead(da, X, &x));
  PetscCall(DMDAVecRestoreArrayRead(da, Xdot, &xdot));

  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormInitialSolution(TS ts, Vec X, void *ctx)
{
  User          user = (User)ctx;
  DM            da;
  PetscInt      i;
  DMDALocalInfo info;
  Field        *x;
  PetscReal     hx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  hx = 1.0 / (PetscReal)(info.mx - 1);

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArray(da, X, &x));

  /* Compute function over the locally owned part of the grid */
  for (i = info.xs; i < info.xs + info.xm; i++) {
    PetscReal xi = i * hx;
    x[i].u       = user->uleft * (1. - xi) + user->uright * xi + PetscSinReal(2. * PETSC_PI * xi);
    x[i].v       = user->vleft * (1. - xi) + user->vright * xi;
  }
  PetscCall(DMDAVecRestoreArray(da, X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    test:
      args: -nox -da_grid_x 20 -ts_monitor_draw_solution -ts_type rosw -ts_rosw_type 2p -ts_dt 5e-2 -ts_adapt_type none

TEST*/
