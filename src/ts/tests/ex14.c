static char help[] = "Tests all TSRK types \n\n";

#include <petscts.h>

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  PetscInt           i, n;
  const PetscScalar *xx;
  /* */ PetscScalar *ff;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetArrayRead(X, &xx));
  PetscCall(VecGetArray(F, &ff));

  if (n >= 1) ff[0] = 1;
  for (i = 1; i < n; i++) ff[i] = (i + 1) * (xx[i - 1] + PetscPowReal(t, i)) / 2;

  PetscCall(VecRestoreArrayRead(X, &xx));
  PetscCall(VecRestoreArray(F, &ff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TestCheckStage(TSAdapt adapt, TS ts, PetscReal t, Vec X, PetscBool *accept)
{
  PetscInt step;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts, &step));
  *accept = (step >= 2) ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestExplicitTS(TS ts, PetscInt order, const char subtype[])
{
  PetscInt           i;
  PetscReal          t;
  Vec                U, X, Y;
  TSType             type;
  PetscBool          done;
  const PetscScalar *xx;
  const PetscScalar *yy;
  const PetscReal    Tf  = 1;
  const PetscReal    dt  = Tf / 8;
  const PetscReal    eps = 100 * PETSC_MACHINE_EPSILON;
  TSAdapt            adapt;
  PetscInt           step;
  TSConvergedReason  reason;

  PetscFunctionBeginUser;
  PetscCall(TSGetType(ts, &type));
  PetscCall(TSGetSolution(ts, &U));
  PetscCall(VecZeroEntries(U));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetTime(ts, 0));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetMaxTime(ts, Tf));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSolve(ts, NULL));
  PetscCall(TSRollBack(ts));
  PetscCall(TSSolve(ts, NULL));
  PetscCall(TSGetTime(ts, &t));

  PetscCall(TSGetSolution(ts, &U));
  PetscCall(VecDuplicate(U, &X));
  PetscCall(TSEvaluateStep(ts, order, X, NULL));
  PetscCall(VecGetArrayRead(X, &xx));
  for (i = 0; i < order; i++) {
    PetscReal error = PetscAbsReal(PetscRealPart(xx[i]) - PetscPowReal(t, i + 1));
    PetscCheck(error <= eps, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Bad solution, error %g, %s '%s'", (double)error, type, subtype);
  }
  PetscCall(VecRestoreArrayRead(X, &xx));
  PetscCall(VecDestroy(&X));

  PetscCall(TSGetSolution(ts, &U));
  PetscCall(VecDuplicate(U, &Y));
  PetscCall(TSEvaluateStep(ts, order - 1, Y, &done));
  PetscCall(VecGetArrayRead(Y, &yy));
  for (i = 0; done && i < order - 1; i++) {
    PetscReal error = PetscAbsReal(PetscRealPart(yy[i]) - PetscPowReal(t, i + 1));
    PetscCheck(error <= eps, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Bad estimator, error %g, %s '%s'", (double)error, type, subtype);
  }
  PetscCall(VecRestoreArrayRead(Y, &yy));
  PetscCall(VecDestroy(&Y));

  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptSetCheckStage(adapt, TestCheckStage));
  PetscCall(TSSetErrorIfStepFails(ts, PETSC_FALSE));
  PetscCall(TSSetStepNumber(ts, 0));
  PetscCall(TSSetTime(ts, 0));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSolve(ts, NULL));
  PetscCall(TSAdaptSetCheckStage(adapt, NULL));
  PetscCall(TSSetErrorIfStepFails(ts, PETSC_TRUE));
  PetscCall(TSGetStepNumber(ts, &step));
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCheck(step == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Bad step number %" PetscInt_FMT ", %s '%s'", step, type, subtype);
  PetscCheck(reason == TS_DIVERGED_STEP_REJECTED, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Bad reason %s, %s '%s'", TSConvergedReasons[reason], type, subtype);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestTSRK(TS ts, TSRKType type)
{
  PetscInt    order;
  TSAdapt     adapt;
  PetscBool   rk1, rk3, rk4;
  TSAdaptType adapttype;
  char        savetype[32];

  PetscFunctionBeginUser;
  PetscCall(TSRKSetType(ts, type));
  PetscCall(TSRKGetType(ts, &type));
  PetscCall(TSRKGetOrder(ts, &order));

  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptGetType(adapt, &adapttype));
  PetscCall(PetscStrncpy(savetype, adapttype, sizeof(savetype)));
  PetscCall(PetscStrcmp(type, TSRK1FE, &rk1));
  PetscCall(PetscStrcmp(type, TSRK3, &rk3));
  PetscCall(PetscStrcmp(type, TSRK4, &rk4));
  if (rk1 || rk3 || rk4) PetscCall(TSAdaptSetType(adapt, TSADAPTNONE));

  PetscCall(TestExplicitTS(ts, order, type));

  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptSetType(adapt, savetype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  TS       ts;
  Vec      X;
  PetscInt N = 9;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, NULL));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, N, &X));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(VecDestroy(&X));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TestTSRK(ts, TSRK1FE));
  PetscCall(TestTSRK(ts, TSRK2A));
  PetscCall(TestTSRK(ts, TSRK3));
  PetscCall(TestTSRK(ts, TSRK3BS));
  PetscCall(TestTSRK(ts, TSRK4));
  PetscCall(TestTSRK(ts, TSRK5F));
  PetscCall(TestTSRK(ts, TSRK5DP));
  PetscCall(TestTSRK(ts, TSRK5BS));
  PetscCall(TestTSRK(ts, TSRK6VR));
  PetscCall(TestTSRK(ts, TSRK7VR));
  PetscCall(TestTSRK(ts, TSRK8VR));

  PetscCall(TSRollBack(ts));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

testset:
  output_file: output/ex14.out
  test:
    suffix: 0
  test:
    suffix: 1
    args: -ts_adapt_type none
  test:
    suffix: 2
    args: -ts_adapt_type basic
  test:
    suffix: 3
    args: -ts_adapt_type dsp

TEST*/
