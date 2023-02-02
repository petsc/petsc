
static char help[] = "Reaction Equation from Chemistry\n";

/*

     Page 6, An example from Atomospheric Chemistry

                 u_1_t =
                 u_2_t =
                 u_3_t =
                 u_4_t =

  -ts_monitor_lg_error -ts_monitor_lg_solution  -ts_view -ts_max_time 2.e4

*/

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscts.h>

typedef struct {
  PetscScalar k1, k2, k3;
  PetscScalar sigma2;
  Vec         initialsolution;
} AppCtx;

PetscScalar k1(AppCtx *ctx, PetscReal t)
{
  PetscReal th    = t / 3600.0;
  PetscReal barth = th - 24.0 * PetscFloorReal(th / 24.0);
  if (((((PetscInt)th) % 24) < 4) || ((((PetscInt)th) % 24) >= 20)) return (1.0e-40);
  else return (ctx->k1 * PetscExpReal(7.0 * PetscPowReal(PetscSinReal(.0625 * PETSC_PI * (barth - 4.0)), .2)));
}

static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, AppCtx *ctx)
{
  PetscScalar       *f;
  const PetscScalar *u, *udot;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArrayWrite(F, &f));
  f[0] = udot[0] - k1(ctx, t) * u[2] + ctx->k2 * u[0];
  f[1] = udot[1] - k1(ctx, t) * u[2] + ctx->k3 * u[1] * u[3] - ctx->sigma2;
  f[2] = udot[2] - ctx->k3 * u[1] * u[3] + k1(ctx, t) * u[2];
  f[3] = udot[3] - ctx->k2 * u[0] + ctx->k3 * u[1] * u[3];
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArrayWrite(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal a, Mat A, Mat B, AppCtx *ctx)
{
  PetscInt           rowcol[] = {0, 1, 2, 3};
  PetscScalar        J[4][4];
  const PetscScalar *u, *udot;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(Udot, &udot));
  J[0][0] = a + ctx->k2;
  J[0][1] = 0.0;
  J[0][2] = -k1(ctx, t);
  J[0][3] = 0.0;
  J[1][0] = 0.0;
  J[1][1] = a + ctx->k3 * u[3];
  J[1][2] = -k1(ctx, t);
  J[1][3] = ctx->k3 * u[1];
  J[2][0] = 0.0;
  J[2][1] = -ctx->k3 * u[3];
  J[2][2] = a + k1(ctx, t);
  J[2][3] = -ctx->k3 * u[1];
  J[3][0] = -ctx->k2;
  J[3][1] = ctx->k3 * u[3];
  J[3][2] = 0.0;
  J[3][3] = a + ctx->k3 * u[1];
  PetscCall(MatSetValues(B, 4, rowcol, 4, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Solution(TS ts, PetscReal t, Vec U, AppCtx *ctx)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(ctx->initialsolution, U));
  PetscCheck(t <= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Solution not given");
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS           ts; /* ODE integrator */
  Vec          U;  /* solution */
  Mat          A;  /* Jacobian matrix */
  PetscMPIInt  size;
  PetscInt     n = 4;
  AppCtx       ctx;
  PetscScalar *u;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A, &U, NULL));

  ctx.k1     = 1.0e-5;
  ctx.k2     = 1.0e5;
  ctx.k3     = 1.0e-16;
  ctx.sigma2 = 1.0e6;

  PetscCall(VecDuplicate(U, &ctx.initialsolution));
  PetscCall(VecGetArrayWrite(ctx.initialsolution, &u));
  u[0] = 0.0;
  u[1] = 1.3e8;
  u[2] = 5.0e11;
  u[3] = 8.0e11;
  PetscCall(VecRestoreArrayWrite(ctx.initialsolution, &u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSROSW));
  PetscCall(TSSetIFunction(ts, NULL, (TSIFunction)IFunction, &ctx));
  PetscCall(TSSetIJacobian(ts, A, A, (TSIJacobian)IJacobian, &ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(Solution(ts, 0, U, &ctx));
  PetscCall(TSSetTime(ts, 4.0 * 3600));
  PetscCall(TSSetTimeStep(ts, 1.0));
  PetscCall(TSSetSolution(ts, U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts, 518400.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetMaxStepRejections(ts, 100));
  PetscCall(TSSetMaxSNESFailures(ts, -1)); /* unlimited */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&ctx.initialsolution));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -ts_view -ts_max_time 2.e4
     timeoutfactor: 15
     requires: !single

TEST*/
