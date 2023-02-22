
static char help[] = "Solves a time-dependent linear PDE with discontinuous right hand side.\n";

/* ------------------------------------------------------------------------

   This program solves the one-dimensional quench front problem modeling a cooled
   liquid rising on a hot metal rod
       u_t = u_xx + g(u),
   with
       g(u) = -Au if u <= u_c,
            =   0 if u >  u_c
   on the domain 0 <= x <= 1, with the boundary conditions
       u(t,0) = 0, u_x(t,1) = 0,
   and the initial condition
       u(0,x) = 0              if 0 <= x <= 0.1,
              = (x - 0.1)/0.15 if 0.1 < x < 0.25
              = 1              if 0.25 <= x <= 1
   We discretize the right-hand side using finite differences with
   uniform grid spacing h:
       u_xx = (u_{i+1} - 2u_{i} + u_{i-1})/(h^2)

Reference: L. Shampine and S. Thompson, "Event Location for Ordinary Differential Equations",
           http://www.radford.edu/~thompson/webddes/eventsweb.pdf
  ------------------------------------------------------------------------- */

#include <petscdmda.h>
#include <petscts.h>
/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  PetscReal A;
  PetscReal uc;
  PetscInt *sw;
} AppCtx;

PetscErrorCode InitialConditions(Vec U, DM da, AppCtx *app)
{
  Vec          xcoord;
  PetscScalar *x, *u;
  PetscInt     lsize, M, xs, xm, i;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinates(da, &xcoord));
  PetscCall(DMDAVecGetArrayRead(da, xcoord, &x));

  PetscCall(VecGetLocalSize(U, &lsize));
  PetscCall(PetscMalloc1(lsize, &app->sw));

  PetscCall(DMDAVecGetArray(da, U, &u));

  PetscCall(DMDAGetInfo(da, 0, &M, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0));

  for (i = xs; i < xs + xm; i++) {
    if (x[i] <= 0.1) u[i] = 0.;
    else if (x[i] > 0.1 && x[i] < 0.25) u[i] = (x[i] - 0.1) / 0.15;
    else u[i] = 1.0;

    app->sw[i - xs] = 1;
  }
  PetscCall(DMDAVecRestoreArray(da, U, &u));
  PetscCall(DMDAVecRestoreArrayRead(da, xcoord, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EventFunction(TS ts, PetscReal t, Vec U, PetscScalar *fvalue, void *ctx)
{
  AppCtx            *app = (AppCtx *)ctx;
  const PetscScalar *u;
  PetscInt           i, lsize;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(U, &lsize));
  PetscCall(VecGetArrayRead(U, &u));
  for (i = 0; i < lsize; i++) fvalue[i] = u[i] - app->uc;
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PostEventFunction(TS ts, PetscInt nevents_zero, PetscInt events_zero[], PetscReal t, Vec U, PetscBool forwardsolve, void *ctx)
{
  AppCtx  *app = (AppCtx *)ctx;
  PetscInt i, idx;

  PetscFunctionBeginUser;
  for (i = 0; i < nevents_zero; i++) {
    idx          = events_zero[i];
    app->sw[idx] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
  AppCtx            *app = (AppCtx *)ctx;
  PetscScalar       *f;
  const PetscScalar *u, *udot;
  DM                 da;
  PetscInt           M, xs, xm, i;
  PetscReal          h, h2;
  Vec                Ulocal;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));

  PetscCall(DMDAGetInfo(da, 0, &M, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0));

  PetscCall(DMGetLocalVector(da, &Ulocal));
  PetscCall(DMGlobalToLocalBegin(da, U, INSERT_VALUES, Ulocal));
  PetscCall(DMGlobalToLocalEnd(da, U, INSERT_VALUES, Ulocal));

  h  = 1.0 / (M - 1);
  h2 = h * h;
  PetscCall(DMDAVecGetArrayRead(da, Udot, &udot));
  PetscCall(DMDAVecGetArrayRead(da, Ulocal, &u));
  PetscCall(DMDAVecGetArray(da, F, &f));

  for (i = xs; i < xs + xm; i++) {
    if (i == 0) {
      f[i] = u[i];
    } else if (i == M - 1) {
      f[i] = (u[i] - u[i - 1]) / h;
    } else {
      f[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / h2 + app->sw[i - xs] * (-app->A * u[i]) - udot[i];
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(da, Udot, &udot));
  PetscCall(DMDAVecRestoreArrayRead(da, Ulocal, &u));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(DMRestoreLocalVector(da, &Ulocal));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal a, Mat A, Mat B, void *ctx)
{
  AppCtx     *app = (AppCtx *)ctx;
  DM          da;
  MatStencil  row, col[3];
  PetscScalar v[3];
  PetscInt    M, xs, xm, i;
  PetscReal   h, h2;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));

  PetscCall(DMDAGetInfo(da, 0, &M, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0));

  h  = 1.0 / (M - 1);
  h2 = h * h;
  for (i = xs; i < xs + xm; i++) {
    row.i = i;
    if (i == 0) {
      v[0] = 1.0;
      PetscCall(MatSetValuesStencil(A, 1, &row, 1, &row, v, INSERT_VALUES));
    } else if (i == M - 1) {
      col[0].i = i;
      v[0]     = 1 / h;
      col[1].i = i - 1;
      v[1]     = -1 / h;
      PetscCall(MatSetValuesStencil(A, 1, &row, 2, col, v, INSERT_VALUES));
    } else {
      col[0].i = i + 1;
      v[0]     = 1 / h2;
      col[1].i = i;
      v[1]     = -2 / h2 + app->sw[i - xs] * (-app->A) - a;
      col[2].i = i - 1;
      v[2]     = 1 / h2;
      PetscCall(MatSetValuesStencil(A, 1, &row, 3, col, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS       ts; /* ODE integrator */
  Vec      U;  /* solution will be stored here */
  Mat      J;  /* Jacobian matrix */
  PetscInt n = 16;
  AppCtx   app;
  DM       da;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "ex22 options", "");
  {
    app.A = 200000;
    PetscCall(PetscOptionsReal("-A", "", "", app.A, &app.A, NULL));
    app.uc = 0.5;
    PetscCall(PetscOptionsReal("-uc", "", "", app.uc, &app.uc, NULL));
  }
  PetscOptionsEnd();

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, -n, 1, 1, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0, 0, 0, 0));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateMatrix(da, &J));
  PetscCall(DMCreateGlobalVector(da, &U));

  PetscCall(InitialConditions(U, da, &app));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSROSW));
  PetscCall(TSSetIFunction(ts, NULL, (TSIFunction)IFunction, (void *)&app));
  PetscCall(TSSetIJacobian(ts, J, J, (TSIJacobian)IJacobian, (void *)&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSolution(ts, U));

  PetscCall(TSSetDM(ts, da));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetTimeStep(ts, 0.1));
  PetscCall(TSSetMaxTime(ts, 30.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  PetscInt lsize;
  PetscCall(VecGetLocalSize(U, &lsize));
  PetscInt  *direction;
  PetscBool *terminate;
  PetscInt   i;
  PetscCall(PetscMalloc1(lsize, &direction));
  PetscCall(PetscMalloc1(lsize, &terminate));
  for (i = 0; i < lsize; i++) {
    direction[i] = -1;
    terminate[i] = PETSC_FALSE;
  }
  PetscCall(TSSetEventHandler(ts, lsize, direction, terminate, EventFunction, PostEventFunction, (void *)&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&U));
  PetscCall(DMDestroy(&da));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFree(direction));
  PetscCall(PetscFree(terminate));

  PetscCall(PetscFree(app.sw));
  PetscCall(PetscFinalize());
  return 0;
}
