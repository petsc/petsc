static char help[] = "Tests TSTrajectoryGetVecs. \n\n";
/*
  This example tests TSTrajectory and the ability of TSTrajectoryGetVecs
  to reconstructs states and derivatives via interpolation (if necessary).
  It also tests TSTrajectory{Get|Restore}UpdatedHistoryVecs
*/
#include <petscts.h>

PetscScalar func(PetscInt p, PetscReal t)  { return p ? t*func(p-1,t) : 1.0; }
PetscScalar dfunc(PetscInt p, PetscReal t)  { return p > 0 ? (PetscReal)p*func(p-1,t) : 0.0; }

int main(int argc,char **argv)
{
  TS             ts;
  Vec            W,W2,Wdot;
  TSTrajectory   tj;
  PetscReal      times[10], tol = PETSC_SMALL;
  PetscReal      TT[10] = { 2, 9, 1, 3, 6, 7, 5, 10, 4, 8 };
  PetscInt       i, p = 1, Nt = 10;
  PetscInt       II[10] = { 1, 4, 9, 2, 3, 6, 5, 8, 0, 7 };
  PetscBool      sort,use1,use2,check = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&W));
  CHKERRQ(VecSetSizes(W,1,PETSC_DECIDE));
  CHKERRQ(VecSetUp(W));
  CHKERRQ(VecDuplicate(W,&Wdot));
  CHKERRQ(VecDuplicate(W,&W2));
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetSolution(ts,W2));
  CHKERRQ(TSSetMaxSteps(ts,10));
  CHKERRQ(TSSetSaveTrajectory(ts));
  CHKERRQ(TSGetTrajectory(ts,&tj));
  CHKERRQ(TSTrajectorySetType(tj,ts,TSTRAJECTORYBASIC));
  CHKERRQ(TSTrajectorySetFromOptions(tj,ts));
  CHKERRQ(TSTrajectorySetSolutionOnly(tj,PETSC_TRUE));
  CHKERRQ(TSTrajectorySetUp(tj,ts));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-check",&check,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  CHKERRQ(PetscOptionsGetRealArray(NULL,NULL,"-interptimes",times,&Nt,NULL));
  sort = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-sorttimes",&sort,NULL));
  if (sort) {
    CHKERRQ(PetscSortReal(10,TT));
  }
  sort = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-sortkeys",&sort,NULL));
  if (sort) {
    CHKERRQ(PetscSortInt(10,II));
  }
  p = PetscMax(p,-p);

  /* populate trajectory */
  for (i = 0; i < 10; i++) {
    CHKERRQ(VecSet(W,func(p,TT[i])));
    CHKERRQ(TSSetStepNumber(ts,II[i]));
    CHKERRQ(TSTrajectorySet(tj,ts,II[i],TT[i],W));
  }
  for (i = 0; i < Nt; i++) {
    PetscReal testtime = times[i], serr, derr;
    const PetscScalar *aW,*aWdot;

    CHKERRQ(TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&testtime,W,Wdot));
    CHKERRQ(VecGetArrayRead(W,&aW));
    CHKERRQ(VecGetArrayRead(Wdot,&aWdot));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0])));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0])));
    serr = PetscAbsScalar(func(p,testtime)-aW[0]);
    derr = PetscAbsScalar(dfunc(p,testtime)-aWdot[0]);
    CHKERRQ(VecRestoreArrayRead(W,&aW));
    CHKERRQ(VecRestoreArrayRead(Wdot,&aWdot));
    PetscCheck(!check || serr <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state %g > %g",(double)serr,(double)tol);
    PetscCheck(!check || derr <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state derivative %g > %g",(double)derr,(double)tol);
  }
  for (i = Nt-1; i >= 0; i--) {
    PetscReal         testtime = times[i], serr;
    const PetscScalar *aW;

    CHKERRQ(TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&testtime,W,NULL));
    CHKERRQ(VecGetArrayRead(W,&aW));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0])));
    serr = PetscAbsScalar(func(p,testtime)-aW[0]);
    CHKERRQ(VecRestoreArrayRead(W,&aW));
    PetscCheck(!check || serr <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state %g > %g",(double)serr,(double)tol);
  }
  for (i = Nt-1; i >= 0; i--) {
    PetscReal         testtime = times[i], derr;
    const PetscScalar *aWdot;

    CHKERRQ(TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&testtime,NULL,Wdot));
    CHKERRQ(VecGetArrayRead(Wdot,&aWdot));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0])));
    derr = PetscAbsScalar(dfunc(p,testtime)-aWdot[0]);
    CHKERRQ(VecRestoreArrayRead(Wdot,&aWdot));
    PetscCheck(!check || derr <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state derivative %g > %g",(double)derr,(double)tol);
  }
  for (i = 0; i < Nt; i++) {
    PetscReal         testtime = times[i], serr, derr;
    const PetscScalar *aW,*aWdot;
    Vec               hW,hWdot;

    CHKERRQ(TSTrajectoryGetUpdatedHistoryVecs(tj,ts,testtime,&hW,&hWdot));
    CHKERRQ(VecGetArrayRead(hW,&aW));
    CHKERRQ(VecGetArrayRead(hWdot,&aWdot));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0])));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0])));
    serr = PetscAbsScalar(func(p,testtime)-aW[0]);
    derr = PetscAbsScalar(dfunc(p,testtime)-aWdot[0]);
    CHKERRQ(VecRestoreArrayRead(hW,&aW));
    CHKERRQ(VecRestoreArrayRead(hWdot,&aWdot));
    CHKERRQ(TSTrajectoryRestoreUpdatedHistoryVecs(tj,&hW,&hWdot));
    PetscCheck(!check || serr <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state %g > %g",(double)serr,(double)tol);
    PetscCheck(!check || derr <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state derivative %g > %g",(double)derr,(double)tol);
  }

  /* Test on-the-fly reconstruction */
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetSolution(ts,W2));
  CHKERRQ(TSSetMaxSteps(ts,10));
  CHKERRQ(TSSetSaveTrajectory(ts));
  CHKERRQ(TSGetTrajectory(ts,&tj));
  CHKERRQ(TSTrajectorySetType(tj,ts,TSTRAJECTORYBASIC));
  CHKERRQ(TSTrajectorySetFromOptions(tj,ts));
  CHKERRQ(TSTrajectorySetSolutionOnly(tj,PETSC_TRUE));
  CHKERRQ(TSTrajectorySetUp(tj,ts));
  use1 = PETSC_FALSE;
  use2 = PETSC_TRUE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-use_state",&use1,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-use_der",&use2,NULL));
  CHKERRQ(PetscSortReal(10,TT));
  for (i = 0; i < 10; i++) {

    CHKERRQ(TSSetStepNumber(ts,i));
    CHKERRQ(VecSet(W,func(p,TT[i])));
    CHKERRQ(TSTrajectorySet(tj,ts,i,TT[i],W));
    if (i) {
      const PetscScalar *aW,*aWdot;
      Vec               hW,hWdot;
      PetscReal         testtime = TT[i], serr, derr;

      CHKERRQ(TSTrajectoryGetUpdatedHistoryVecs(tj,ts,testtime,use1 ? &hW : NULL,use2 ? &hWdot : NULL));
      if (use1) {
        CHKERRQ(VecGetArrayRead(hW,&aW));
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0])));
        serr = PetscAbsScalar(func(p,testtime)-aW[0]);
        CHKERRQ(VecRestoreArrayRead(hW,&aW));
        PetscCheck(!check || serr <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state %g > %g",(double)serr,(double)tol);
      }
      if (use2) {
        CHKERRQ(VecGetArrayRead(hWdot,&aWdot));
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0])));
        derr = PetscAbsScalar(dfunc(p,testtime)-aWdot[0]);
        CHKERRQ(VecRestoreArrayRead(hWdot,&aWdot));
        PetscCheck(!check || i < p || derr <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state derivative %g > %g",(double)derr,(double)tol);
      }
      CHKERRQ(TSTrajectoryRestoreUpdatedHistoryVecs(tj,use1 ? &hW : NULL,use2 ? &hWdot : NULL));
    }
  }
  CHKERRQ(TSRemoveTrajectory(ts));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(VecDestroy(&W));
  CHKERRQ(VecDestroy(&W2));
  CHKERRQ(VecDestroy(&Wdot));
  CHKERRQ(PetscFinalize());
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
