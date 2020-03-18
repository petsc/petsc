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
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = VecCreate(PETSC_COMM_WORLD,&W);CHKERRQ(ierr);
  ierr = VecSetSizes(W,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetUp(W);CHKERRQ(ierr);
  ierr = VecDuplicate(W,&Wdot);CHKERRQ(ierr);
  ierr = VecDuplicate(W,&W2);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,W2);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,10);CHKERRQ(ierr);
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
  ierr = TSGetTrajectory(ts,&tj);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(tj,ts,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(tj,ts);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(tj,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetUp(tj,ts);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-check",&check,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-interptimes",times,&Nt,NULL);CHKERRQ(ierr);
  sort = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-sorttimes",&sort,NULL);CHKERRQ(ierr);
  if (sort) {
    ierr = PetscSortReal(10,TT);CHKERRQ(ierr);
  }
  sort = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-sortkeys",&sort,NULL);CHKERRQ(ierr);
  if (sort) {
    ierr = PetscSortInt(10,II);CHKERRQ(ierr);
  }
  p = PetscMax(p,-p);

  /* populate trajectory */
  for (i = 0; i < 10; i++) {
    ierr = VecSet(W,func(p,TT[i]));CHKERRQ(ierr);
    ierr = TSSetStepNumber(ts,II[i]);CHKERRQ(ierr);
    ierr = TSTrajectorySet(tj,ts,II[i],TT[i],W);CHKERRQ(ierr);
  }
  for (i = 0; i < Nt; i++) {
    PetscReal testtime = times[i], serr, derr;
    const PetscScalar *aW,*aWdot;

    ierr = TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&testtime,W,Wdot);CHKERRQ(ierr);
    ierr = VecGetArrayRead(W,&aW);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Wdot,&aWdot);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0]));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0]));CHKERRQ(ierr);
    serr = PetscAbsScalar(func(p,testtime)-aW[0]);
    derr = PetscAbsScalar(dfunc(p,testtime)-aWdot[0]);
    ierr = VecRestoreArrayRead(W,&aW);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Wdot,&aWdot);CHKERRQ(ierr);
    if (check && serr > tol) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state %g > %g",(double)serr,(double)tol);
    if (check && derr > tol) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state derivative %g > %g",(double)derr,(double)tol);
  }
  for (i = Nt-1; i >= 0; i--) {
    PetscReal         testtime = times[i], serr;
    const PetscScalar *aW;

    ierr = TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&testtime,W,NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(W,&aW);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0]));CHKERRQ(ierr);
    serr = PetscAbsScalar(func(p,testtime)-aW[0]);
    ierr = VecRestoreArrayRead(W,&aW);CHKERRQ(ierr);
    if (check && serr > tol) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state %g > %g",(double)serr,(double)tol);
  }
  for (i = Nt-1; i >= 0; i--) {
    PetscReal         testtime = times[i], derr;
    const PetscScalar *aWdot;

    ierr = TSTrajectoryGetVecs(tj,ts,PETSC_DECIDE,&testtime,NULL,Wdot);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Wdot,&aWdot);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0]));CHKERRQ(ierr);
    derr = PetscAbsScalar(dfunc(p,testtime)-aWdot[0]);
    ierr = VecRestoreArrayRead(Wdot,&aWdot);CHKERRQ(ierr);
    if (check && derr > tol) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state derivative %g > %g",(double)derr,(double)tol);
  }
  for (i = 0; i < Nt; i++) {
    PetscReal         testtime = times[i], serr, derr;
    const PetscScalar *aW,*aWdot;
    Vec               hW,hWdot;

    ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,ts,testtime,&hW,&hWdot);CHKERRQ(ierr);
    ierr = VecGetArrayRead(hW,&aW);CHKERRQ(ierr);
    ierr = VecGetArrayRead(hWdot,&aWdot);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0]));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0]));CHKERRQ(ierr);
    serr = PetscAbsScalar(func(p,testtime)-aW[0]);
    derr = PetscAbsScalar(dfunc(p,testtime)-aWdot[0]);
    ierr = VecRestoreArrayRead(hW,&aW);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(hWdot,&aWdot);CHKERRQ(ierr);
    ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,&hW,&hWdot);CHKERRQ(ierr);
    if (check && serr > tol) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state %g > %g",(double)serr,(double)tol);
    if (check && derr > tol) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state derivative %g > %g",(double)derr,(double)tol);
  }

  /* Test on-the-fly reconstruction */
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,W2);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,10);CHKERRQ(ierr);
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
  ierr = TSGetTrajectory(ts,&tj);CHKERRQ(ierr);
  ierr = TSTrajectorySetType(tj,ts,TSTRAJECTORYBASIC);CHKERRQ(ierr);
  ierr = TSTrajectorySetFromOptions(tj,ts);CHKERRQ(ierr);
  ierr = TSTrajectorySetSolutionOnly(tj,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSTrajectorySetUp(tj,ts);CHKERRQ(ierr);
  use1 = PETSC_FALSE;
  use2 = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_state",&use1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_der",&use2,NULL);CHKERRQ(ierr);
  ierr = PetscSortReal(10,TT);CHKERRQ(ierr);
  for (i = 0; i < 10; i++) {

    ierr = TSSetStepNumber(ts,i);CHKERRQ(ierr);
    ierr = VecSet(W,func(p,TT[i]));CHKERRQ(ierr);
    ierr = TSTrajectorySet(tj,ts,i,TT[i],W);CHKERRQ(ierr);
    if (i) {
      const PetscScalar *aW,*aWdot;
      Vec               hW,hWdot;
      PetscReal         testtime = TT[i], serr, derr;

      ierr = TSTrajectoryGetUpdatedHistoryVecs(tj,ts,testtime,use1 ? &hW : NULL,use2 ? &hWdot : NULL);CHKERRQ(ierr);
      if (use1) {
        ierr = VecGetArrayRead(hW,&aW);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," f(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(func(p,testtime)),(double)PetscRealPart(aW[0]));CHKERRQ(ierr);
        serr = PetscAbsScalar(func(p,testtime)-aW[0]);
        ierr = VecRestoreArrayRead(hW,&aW);CHKERRQ(ierr);
        if (check && serr > tol) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state %g > %g",(double)serr,(double)tol);
      }
      if (use2) {
        ierr = VecGetArrayRead(hWdot,&aWdot);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"df(%g) = %g (reconstructed %g)\n",(double)testtime,(double)PetscRealPart(dfunc(p,testtime)),(double)PetscRealPart(aWdot[0]));CHKERRQ(ierr);
        derr = PetscAbsScalar(dfunc(p,testtime)-aWdot[0]);
        ierr = VecRestoreArrayRead(hWdot,&aWdot);CHKERRQ(ierr);
        if (check && i >= p && derr > tol) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in state derivative %g > %g",(double)derr,(double)tol);
      }
      ierr = TSTrajectoryRestoreUpdatedHistoryVecs(tj,use1 ? &hW : NULL,use2 ? &hWdot : NULL);CHKERRQ(ierr);
    }
  }
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&W);CHKERRQ(ierr);
  ierr = VecDestroy(&W2);CHKERRQ(ierr);
  ierr = VecDestroy(&Wdot);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
