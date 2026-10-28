static char help[] = "Order check for TSARKIMEX started at a nonzero time with an implicit-only problem.\n\
Solves u' = cos(t), u(t0) = sin(t0), as an IFunction only, from t0 to t0 + 1 with two time step sizes.\n";

#include <petscts.h>

static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, PetscCtx ctx)
{
  const PetscScalar *udot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArrayWrite(F, &f));
  f[0] = udot[0] - PetscCosReal(t);
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArrayWrite(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(MatSetValue(B, 0, 0, shift, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveError(PetscReal t0, PetscReal dt, PetscReal *err)
{
  TS          ts;
  TSAdapt     adapt;
  Vec         U;
  Mat         J;
  PetscScalar u;
  PetscInt    row = 0;

  PetscFunctionBeginUser;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &U));
  PetscCall(VecSet(U, PetscSinReal(t0)));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &J));
  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, NULL));
  PetscCall(TSSetIJacobian(ts, J, J, IJacobian, NULL));
  PetscCall(TSSetTime(ts, t0));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetMaxTime(ts, t0 + 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTolerances(ts, 1e-12, NULL, 1e-12, NULL));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptSetType(adapt, TSADAPTNONE));
  PetscCall(TSSolve(ts, U));
  PetscCall(VecGetValues(U, 1, &row, &u));
  *err = PetscAbsScalar(u - PetscSinReal(t0 + 1.0));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscReal t0 = 1.0, dt = 0.01, e1, e2, order, minorder = 2.5;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-t0", &t0, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-min_order", &minorder, NULL));
  PetscCall(SolveError(t0, dt, &e1));
  PetscCall(SolveError(t0, dt / 2, &e2));
  order = PetscLog2Real(e1 / e2);
  PetscCall(PetscInfo(NULL, "errors %g %g observed order %g\n", (double)e1, (double)e2, (double)order));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Observed order >= %g: %s\n", (double)minorder, order >= minorder ? "yes" : "no"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    testset:
      requires: !single
      output_file: output/ex82.out
      test:
        suffix: 3
        args: -ts_arkimex_type 3
      test:
        suffix: 4
        args: -ts_arkimex_type 4
      test:
        suffix: 3_t0
        args: -ts_arkimex_type 3 -t0 0
      test:
        suffix: 4_fully_implicit
        args: -ts_arkimex_type 4 -ts_arkimex_fully_implicit

TEST*/
