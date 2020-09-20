static char help[] ="Tests all TSRK types \n\n";

#include <petscts.h>

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscInt          i,n;
  const PetscScalar *xx;
  /* */ PetscScalar *ff;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(F,&ff);CHKERRQ(ierr);

  if (n >= 1)
    ff[0] = 1;
  for (i = 1; i < n; i++)
    ff[i] = (i+1)*(xx[i-1]+PetscPowReal(t,i))/2;

  ierr = VecRestoreArrayRead(X,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&ff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TestCheckStage(TSAdapt adapt,TS ts,PetscReal t,Vec X,PetscBool *accept)
{
  PetscInt       step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  *accept = (step >= 2) ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TestExplicitTS(TS ts,PetscInt order,const char subtype[])
{
  PetscInt          i;
  PetscReal         t;
  Vec               U,X,Y;
  TSType            type;
  PetscBool         done;
  const PetscScalar *xx;
  const PetscScalar *yy;
  const PetscReal   Tf  = 1;
  const PetscReal   dt  = Tf/8;
  const PetscReal   eps = 100*PETSC_MACHINE_EPSILON;
  TSAdapt           adapt;
  PetscInt          step;
  TSConvergedReason reason;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = TSGetType(ts,&type);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecZeroEntries(U);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,Tf);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);
  ierr = TSRollBack(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);

  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&X);CHKERRQ(ierr);
  ierr = TSEvaluateStep(ts,order,X,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&xx);CHKERRQ(ierr);
  for (i = 0; i < order; i++) {
    PetscReal error = PetscAbsReal(PetscRealPart(xx[i]) - PetscPowReal(t,i+1));
    if (error > eps) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad solution, error %g, %s '%s'",(double)error,type,subtype);
  }
  ierr = VecRestoreArrayRead(X,&xx);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);

  ierr = TSGetSolution(ts,&U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&Y);CHKERRQ(ierr);
  ierr = TSEvaluateStep(ts,order-1,Y,&done);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Y,&yy);CHKERRQ(ierr);
  for (i = 0; done && i < order-1; i++) {
    PetscReal error = PetscAbsReal(PetscRealPart(yy[i]) - PetscPowReal(t,i+1));
    if (error > eps) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad estimator, error %g, %s '%s'",(double)error,type,subtype);
  }
  ierr = VecRestoreArrayRead(Y,&yy);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetCheckStage(adapt,TestCheckStage);CHKERRQ(ierr);
  ierr = TSSetErrorIfStepFails(ts,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);
  ierr = TSAdaptSetCheckStage(adapt,NULL);CHKERRQ(ierr);
  ierr = TSSetErrorIfStepFails(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  if (step != 2) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad step number %D, %s '%s'",step,type,subtype);
  if (reason != TS_DIVERGED_STEP_REJECTED) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad reason %s, %s '%s'",TSConvergedReasons[reason],type,subtype);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestTSRK(TS ts,TSRKType type)
{
  PetscInt       order;
  TSAdapt        adapt;
  PetscBool      rk1,rk3,rk4;
  TSAdaptType    adapttype;
  char           savetype[32];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSRKSetType(ts,type);CHKERRQ(ierr);
  ierr = TSRKGetType(ts,&type);CHKERRQ(ierr);
  ierr = TSRKGetOrder(ts,&order);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptGetType(adapt,&adapttype);CHKERRQ(ierr);
  ierr = PetscStrncpy(savetype,adapttype,sizeof(savetype));CHKERRQ(ierr);
  ierr = PetscStrcmp(type,TSRK1FE,&rk1);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,TSRK3,&rk3);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,TSRK4,&rk4);CHKERRQ(ierr);
  if (rk1 || rk3 || rk4) {ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);}

  ierr = TestExplicitTS(ts,order,type);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,savetype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  TS             ts;
  Vec            X;
  PetscInt       N = 9;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,NULL);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&X);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TestTSRK(ts,TSRK1FE);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK2A);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK3);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK3BS);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK4);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK5F);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK5DP);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK5BS);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK6VR);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK7VR);CHKERRQ(ierr);
  ierr = TestTSRK(ts,TSRK8VR);CHKERRQ(ierr);

  ierr = TSRollBack(ts);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
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
