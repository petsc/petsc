static char help[] = "Basic problem for multi-rate method.\n";

/*F

\begin{eqnarray}
                 ys' = -2.0*\frac{-1.0+ys^2.0-\cos(t)}{2.0*ys}+0.05*\frac{-2.0+yf^2-\cos(5.0*t)}{2.0*yf}-\frac{\sin(t)}{2.0*ys}\\
                 yf' = 0.05*\frac{-1.0+ys^2-\cos(t)}{2.0*ys}-\frac{-2.0+yf^2-\cos(5.0*t)}{2.0*yf}-5.0*\frac{\sin(5.0*t)}{2.0*yf}\\
\end{eqnarray}

F*/

#include <petscts.h>

typedef struct {
  PetscReal Tf, dt;
} AppCtx;

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, AppCtx *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));
  f[0] = -2.0 * (-1.0 + u[0] * u[0] - PetscCosScalar(t)) / (2.0 * u[0]) + 0.05 * (-2.0 + u[1] * u[1] - PetscCosScalar(5.0 * t)) / (2.0 * u[1]) - PetscSinScalar(t) / (2.0 * u[0]);
  f[1] = 0.05 * (-1.0 + u[0] * u[0] - PetscCosScalar(t)) / (2.0 * u[0]) - (-2.0 + u[1] * u[1] - PetscCosScalar(5.0 * t)) / (2.0 * u[1]) - 5.0 * PetscSinScalar(5.0 * t) / (2.0 * u[1]);
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionslow(TS ts, PetscReal t, Vec U, Vec F, AppCtx *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));
  f[0] = -2.0 * (-1.0 + u[0] * u[0] - PetscCosScalar(t)) / (2.0 * u[0]) + 0.05 * (-2.0 + u[1] * u[1] - PetscCosScalar(5.0 * t)) / (2.0 * u[1]) - PetscSinScalar(t) / (2.0 * u[0]);
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSFunctionfast(TS ts, PetscReal t, Vec U, Vec F, AppCtx *ctx)
{
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(F, &f));
  f[0] = 0.05 * (-1.0 + u[0] * u[0] - PetscCosScalar(t)) / (2.0 * u[0]) - (-2.0 + u[1] * u[1] - PetscCosScalar(5.0 * t)) / (2.0 * u[1]) - 5.0 * PetscSinScalar(5.0 * t) / (2.0 * u[1]);
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

static PetscErrorCode sol_true(PetscReal t, Vec U)
{
  PetscScalar *u;

  PetscFunctionBegin;
  PetscCall(VecGetArray(U, &u));
  u[0] = PetscSqrtScalar(1.0 + PetscCosScalar(t));
  u[1] = PetscSqrtScalar(2.0 + PetscCosScalar(5.0 * t));
  PetscCall(VecRestoreArray(U, &u));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS           ts; /* ODE integrator */
  Vec          U;  /* solution will be stored here */
  Vec          Utrue;
  PetscMPIInt  size;
  AppCtx       ctx;
  PetscScalar *u;
  IS           iss;
  IS           isf;
  PetscInt    *indicess;
  PetscInt    *indicesf;
  PetscInt     n = 2;
  PetscReal    error, tt;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create index for slow part and fast part
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscMalloc1(1, &indicess));
  indicess[0] = 0;
  PetscCall(PetscMalloc1(1, &indicesf));
  indicesf[0] = 1;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 1, indicess, PETSC_COPY_VALUES, &iss));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 1, indicesf, PETSC_COPY_VALUES, &isf));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary vector
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &U));
  PetscCall(VecSetSizes(U, n, PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(U));
  PetscCall(VecDuplicate(U, &Utrue));
  PetscCall(VecCopy(U, Utrue));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set initial condition
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(U, &u));
  u[0] = PetscSqrtScalar(2.0);
  u[1] = PetscSqrtScalar(3.0);
  PetscCall(VecRestoreArray(U, &u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSMPRK));

  PetscCall(TSSetRHSFunction(ts, NULL, (TSRHSFunction)RHSFunction, &ctx));
  PetscCall(TSRHSSplitSetIS(ts, "slow", iss));
  PetscCall(TSRHSSplitSetIS(ts, "fast", isf));
  PetscCall(TSRHSSplitSetRHSFunction(ts, "slow", NULL, (TSRHSFunction)RHSFunctionslow, &ctx));
  PetscCall(TSRHSSplitSetRHSFunction(ts, "fast", NULL, (TSRHSFunction)RHSFunctionfast, &ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSolution(ts, U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "ODE options", "");
  {
    ctx.Tf = 0.3;
    ctx.dt = 0.01;
    PetscCall(PetscOptionsScalar("-Tf", "", "", ctx.Tf, &ctx.Tf, NULL));
    PetscCall(PetscOptionsScalar("-dt", "", "", ctx.dt, &ctx.dt, NULL));
  }
  PetscOptionsEnd();
  PetscCall(TSSetMaxTime(ts, ctx.Tf));
  PetscCall(TSSetTimeStep(ts, ctx.dt));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, U));
  PetscCall(VecView(U, PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check the error of the Petsc solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSGetTime(ts, &tt));
  PetscCall(sol_true(tt, Utrue));
  PetscCall(VecAXPY(Utrue, -1.0, U));
  PetscCall(VecNorm(Utrue, NORM_2, &error));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Print norm2 error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "l2 error norm: %g\n", (double)error));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&Utrue));
  PetscCall(ISDestroy(&iss));
  PetscCall(ISDestroy(&isf));
  PetscCall(PetscFree(indicess));
  PetscCall(PetscFree(indicesf));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
    build:
      requires: !complex

    test:

TEST*/
