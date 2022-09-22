static char help[] = "Tests TSTrajectoryGetVecs. \n\n";
/*
  This example tests TSTrajectory and the ability of TSTrajectoryGetVecs
  to reconstructs states and derivatives via interpolation (if necessary).
  It also tests TSTrajectory{Get|Restore}UpdatedHistoryVecs
*/
#include <petscts.h>

PetscScalar func(PetscInt p, PetscReal t)
{
  return p ? t * func(p - 1, t) : 1.0;
}
PetscScalar dfunc(PetscInt p, PetscReal t)
{
  return p > 0 ? (PetscReal)p * func(p - 1, t) : 0.0;
}

int main(int argc, char **argv)
{
  TS           ts;
  Vec          W, W2, Wdot;
  TSTrajectory tj;
  PetscReal    times[10], tol = PETSC_SMALL;
  PetscReal    TT[10] = {2, 9, 1, 3, 6, 7, 5, 10, 4, 8};
  PetscInt     i, p = 1, Nt = 10;
  PetscInt     II[10] = {1, 4, 9, 2, 3, 6, 5, 8, 0, 7};
  PetscBool    sort, use1, use2, check = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &W));
  PetscCall(VecSetSizes(W, 1, PETSC_DECIDE));
  PetscCall(VecSetUp(W));
  PetscCall(VecDuplicate(W, &Wdot));
  PetscCall(VecDuplicate(W, &W2));
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetSolution(ts, W2));
  PetscCall(TSSetMaxSteps(ts, 10));
  PetscCall(TSSetSaveTrajectory(ts));
  PetscCall(TSGetTrajectory(ts, &tj));
  PetscCall(TSTrajectorySetType(tj, ts, TSTRAJECTORYBASIC));
  PetscCall(TSTrajectorySetFromOptions(tj, ts));
  PetscCall(TSTrajectorySetSolutionOnly(tj, PETSC_TRUE));
  PetscCall(TSTrajectorySetUp(tj, ts));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check", &check, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-p", &p, NULL));
  PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-interptimes", times, &Nt, NULL));
  sort = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-sorttimes", &sort, NULL));
  if (sort) PetscCall(PetscSortReal(10, TT));
  sort = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-sortkeys", &sort, NULL));
  if (sort) PetscCall(PetscSortInt(10, II));
  p = PetscMax(p, -p);

  /* populate trajectory */
  for (i = 0; i < 10; i++) {
    PetscCall(VecSet(W, func(p, TT[i])));
    PetscCall(TSSetStepNumber(ts, II[i]));
    PetscCall(TSTrajectorySet(tj, ts, II[i], TT[i], W));
  }
  for (i = 0; i < Nt; i++) {
    PetscReal          testtime = times[i], serr, derr;
    const PetscScalar *aW, *aWdot;

    PetscCall(TSTrajectoryGetVecs(tj, ts, PETSC_DECIDE, &testtime, W, Wdot));
    PetscCall(VecGetArrayRead(W, &aW));
    PetscCall(VecGetArrayRead(Wdot, &aWdot));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " f(%g) = %g (reconstructed %g)\n", (double)testtime, (double)PetscRealPart(func(p, testtime)), (double)PetscRealPart(aW[0])));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "df(%g) = %g (reconstructed %g)\n", (double)testtime, (double)PetscRealPart(dfunc(p, testtime)), (double)PetscRealPart(aWdot[0])));
    serr = PetscAbsScalar(func(p, testtime) - aW[0]);
    derr = PetscAbsScalar(dfunc(p, testtime) - aWdot[0]);
    PetscCall(VecRestoreArrayRead(W, &aW));
    PetscCall(VecRestoreArrayRead(Wdot, &aWdot));
    PetscCheck(!check || serr <= tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in state %g > %g", (double)serr, (double)tol);
    PetscCheck(!check || derr <= tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in state derivative %g > %g", (double)derr, (double)tol);
  }
  for (i = Nt - 1; i >= 0; i--) {
    PetscReal          testtime = times[i], serr;
    const PetscScalar *aW;

    PetscCall(TSTrajectoryGetVecs(tj, ts, PETSC_DECIDE, &testtime, W, NULL));
    PetscCall(VecGetArrayRead(W, &aW));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " f(%g) = %g (reconstructed %g)\n", (double)testtime, (double)PetscRealPart(func(p, testtime)), (double)PetscRealPart(aW[0])));
    serr = PetscAbsScalar(func(p, testtime) - aW[0]);
    PetscCall(VecRestoreArrayRead(W, &aW));
    PetscCheck(!check || serr <= tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in state %g > %g", (double)serr, (double)tol);
  }
  for (i = Nt - 1; i >= 0; i--) {
    PetscReal          testtime = times[i], derr;
    const PetscScalar *aWdot;

    PetscCall(TSTrajectoryGetVecs(tj, ts, PETSC_DECIDE, &testtime, NULL, Wdot));
    PetscCall(VecGetArrayRead(Wdot, &aWdot));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "df(%g) = %g (reconstructed %g)\n", (double)testtime, (double)PetscRealPart(dfunc(p, testtime)), (double)PetscRealPart(aWdot[0])));
    derr = PetscAbsScalar(dfunc(p, testtime) - aWdot[0]);
    PetscCall(VecRestoreArrayRead(Wdot, &aWdot));
    PetscCheck(!check || derr <= tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in state derivative %g > %g", (double)derr, (double)tol);
  }
  for (i = 0; i < Nt; i++) {
    PetscReal          testtime = times[i], serr, derr;
    const PetscScalar *aW, *aWdot;
    Vec                hW, hWdot;

    PetscCall(TSTrajectoryGetUpdatedHistoryVecs(tj, ts, testtime, &hW, &hWdot));
    PetscCall(VecGetArrayRead(hW, &aW));
    PetscCall(VecGetArrayRead(hWdot, &aWdot));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " f(%g) = %g (reconstructed %g)\n", (double)testtime, (double)PetscRealPart(func(p, testtime)), (double)PetscRealPart(aW[0])));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "df(%g) = %g (reconstructed %g)\n", (double)testtime, (double)PetscRealPart(dfunc(p, testtime)), (double)PetscRealPart(aWdot[0])));
    serr = PetscAbsScalar(func(p, testtime) - aW[0]);
    derr = PetscAbsScalar(dfunc(p, testtime) - aWdot[0]);
    PetscCall(VecRestoreArrayRead(hW, &aW));
    PetscCall(VecRestoreArrayRead(hWdot, &aWdot));
    PetscCall(TSTrajectoryRestoreUpdatedHistoryVecs(tj, &hW, &hWdot));
    PetscCheck(!check || serr <= tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in state %g > %g", (double)serr, (double)tol);
    PetscCheck(!check || derr <= tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in state derivative %g > %g", (double)derr, (double)tol);
  }

  /* Test on-the-fly reconstruction */
  PetscCall(TSDestroy(&ts));
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetSolution(ts, W2));
  PetscCall(TSSetMaxSteps(ts, 10));
  PetscCall(TSSetSaveTrajectory(ts));
  PetscCall(TSGetTrajectory(ts, &tj));
  PetscCall(TSTrajectorySetType(tj, ts, TSTRAJECTORYBASIC));
  PetscCall(TSTrajectorySetFromOptions(tj, ts));
  PetscCall(TSTrajectorySetSolutionOnly(tj, PETSC_TRUE));
  PetscCall(TSTrajectorySetUp(tj, ts));
  use1 = PETSC_FALSE;
  use2 = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_state", &use1, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_der", &use2, NULL));
  PetscCall(PetscSortReal(10, TT));
  for (i = 0; i < 10; i++) {
    PetscCall(TSSetStepNumber(ts, i));
    PetscCall(VecSet(W, func(p, TT[i])));
    PetscCall(TSTrajectorySet(tj, ts, i, TT[i], W));
    if (i) {
      const PetscScalar *aW, *aWdot;
      Vec                hW, hWdot;
      PetscReal          testtime = TT[i], serr, derr;

      PetscCall(TSTrajectoryGetUpdatedHistoryVecs(tj, ts, testtime, use1 ? &hW : NULL, use2 ? &hWdot : NULL));
      if (use1) {
        PetscCall(VecGetArrayRead(hW, &aW));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, " f(%g) = %g (reconstructed %g)\n", (double)testtime, (double)PetscRealPart(func(p, testtime)), (double)PetscRealPart(aW[0])));
        serr = PetscAbsScalar(func(p, testtime) - aW[0]);
        PetscCall(VecRestoreArrayRead(hW, &aW));
        PetscCheck(!check || serr <= tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in state %g > %g", (double)serr, (double)tol);
      }
      if (use2) {
        PetscCall(VecGetArrayRead(hWdot, &aWdot));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "df(%g) = %g (reconstructed %g)\n", (double)testtime, (double)PetscRealPart(dfunc(p, testtime)), (double)PetscRealPart(aWdot[0])));
        derr = PetscAbsScalar(dfunc(p, testtime) - aWdot[0]);
        PetscCall(VecRestoreArrayRead(hWdot, &aWdot));
        PetscCheck(!check || i < p || derr <= tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in state derivative %g > %g", (double)derr, (double)tol);
      }
      PetscCall(TSTrajectoryRestoreUpdatedHistoryVecs(tj, use1 ? &hW : NULL, use2 ? &hWdot : NULL));
    }
  }
  PetscCall(TSRemoveTrajectory(ts));
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&W));
  PetscCall(VecDestroy(&W2));
  PetscCall(VecDestroy(&Wdot));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

test:
  suffix: 1
  requires: !single
  args: -ts_trajectory_monitor -p 3 -ts_trajectory_reconstruction_order 3 -interptimes 1,9.9,3,1.1,1.1,5.6 -check

test:
  suffix: 2
  requires: !single
  args: -sortkeys -ts_trajectory_monitor -ts_trajectory_type memory -p 3 -ts_trajectory_reconstruction_order 3 -ts_trajectory_adjointmode 0 -interptimes 1,9.9,3,1.1,1.1,5.6 -check

test:
  suffix: 3
  requires: !single
  args: -ts_trajectory_monitor -p 3 -ts_trajectory_reconstruction_order 5 -interptimes 1,9.9,3,1.1,1.1,5.6 -check

TEST*/
