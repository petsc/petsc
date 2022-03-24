static char help[] ="Tests all TSRK types \n\n";

#include <petscts.h>

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscInt          i,n;
  const PetscScalar *xx;
  /* */ PetscScalar *ff;

  PetscFunctionBegin;
  CHKERRQ(VecGetLocalSize(X,&n));
  CHKERRQ(VecGetArrayRead(X,&xx));
  CHKERRQ(VecGetArray(F,&ff));

  if (n >= 1)
    ff[0] = 1;
  for (i = 1; i < n; i++)
    ff[i] = (i+1)*(xx[i-1]+PetscPowReal(t,i))/2;

  CHKERRQ(VecRestoreArrayRead(X,&xx));
  CHKERRQ(VecRestoreArray(F,&ff));
  PetscFunctionReturn(0);
}

PetscErrorCode TestCheckStage(TSAdapt adapt,TS ts,PetscReal t,Vec X,PetscBool *accept)
{
  PetscInt       step;

  PetscFunctionBegin;
  CHKERRQ(TSGetStepNumber(ts,&step));
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

  PetscFunctionBegin;
  CHKERRQ(TSGetType(ts,&type));
  CHKERRQ(TSGetSolution(ts,&U));
  CHKERRQ(VecZeroEntries(U));
  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetTime(ts,0));
  CHKERRQ(TSSetTimeStep(ts,dt));
  CHKERRQ(TSSetMaxTime(ts,Tf));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSolve(ts,NULL));
  CHKERRQ(TSRollBack(ts));
  CHKERRQ(TSSolve(ts,NULL));
  CHKERRQ(TSGetTime(ts,&t));

  CHKERRQ(TSGetSolution(ts,&U));
  CHKERRQ(VecDuplicate(U,&X));
  CHKERRQ(TSEvaluateStep(ts,order,X,NULL));
  CHKERRQ(VecGetArrayRead(X,&xx));
  for (i = 0; i < order; i++) {
    PetscReal error = PetscAbsReal(PetscRealPart(xx[i]) - PetscPowReal(t,i+1));
    PetscCheck(error <= eps,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad solution, error %g, %s '%s'",(double)error,type,subtype);
  }
  CHKERRQ(VecRestoreArrayRead(X,&xx));
  CHKERRQ(VecDestroy(&X));

  CHKERRQ(TSGetSolution(ts,&U));
  CHKERRQ(VecDuplicate(U,&Y));
  CHKERRQ(TSEvaluateStep(ts,order-1,Y,&done));
  CHKERRQ(VecGetArrayRead(Y,&yy));
  for (i = 0; done && i < order-1; i++) {
    PetscReal error = PetscAbsReal(PetscRealPart(yy[i]) - PetscPowReal(t,i+1));
    PetscCheck(error <= eps,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad estimator, error %g, %s '%s'",(double)error,type,subtype);
  }
  CHKERRQ(VecRestoreArrayRead(Y,&yy));
  CHKERRQ(VecDestroy(&Y));

  CHKERRQ(TSGetAdapt(ts,&adapt));
  CHKERRQ(TSAdaptSetCheckStage(adapt,TestCheckStage));
  CHKERRQ(TSSetErrorIfStepFails(ts,PETSC_FALSE));
  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetTime(ts,0));
  CHKERRQ(TSSetTimeStep(ts,dt));
  CHKERRQ(TSSolve(ts,NULL));
  CHKERRQ(TSAdaptSetCheckStage(adapt,NULL));
  CHKERRQ(TSSetErrorIfStepFails(ts,PETSC_TRUE));
  CHKERRQ(TSGetStepNumber(ts,&step));
  CHKERRQ(TSGetConvergedReason(ts,&reason));
  PetscCheck(step == 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad step number %D, %s '%s'",step,type,subtype);
  PetscCheck(reason == TS_DIVERGED_STEP_REJECTED,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad reason %s, %s '%s'",TSConvergedReasons[reason],type,subtype);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestTSRK(TS ts,TSRKType type)
{
  PetscInt       order;
  TSAdapt        adapt;
  PetscBool      rk1,rk3,rk4;
  TSAdaptType    adapttype;
  char           savetype[32];

  PetscFunctionBegin;
  CHKERRQ(TSRKSetType(ts,type));
  CHKERRQ(TSRKGetType(ts,&type));
  CHKERRQ(TSRKGetOrder(ts,&order));

  CHKERRQ(TSGetAdapt(ts,&adapt));
  CHKERRQ(TSAdaptGetType(adapt,&adapttype));
  CHKERRQ(PetscStrncpy(savetype,adapttype,sizeof(savetype)));
  CHKERRQ(PetscStrcmp(type,TSRK1FE,&rk1));
  CHKERRQ(PetscStrcmp(type,TSRK3,&rk3));
  CHKERRQ(PetscStrcmp(type,TSRK4,&rk4));
  if (rk1 || rk3 || rk4) CHKERRQ(TSAdaptSetType(adapt,TSADAPTNONE));

  CHKERRQ(TestExplicitTS(ts,order,type));

  CHKERRQ(TSGetAdapt(ts,&adapt));
  CHKERRQ(TSAdaptSetType(adapt,savetype));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  TS             ts;
  Vec            X;
  PetscInt       N = 9;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));

  CHKERRQ(TSCreate(PETSC_COMM_SELF,&ts));
  CHKERRQ(TSSetType(ts,TSRK));
  CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunction,NULL));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N,&X));
  CHKERRQ(TSSetSolution(ts,X));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(TestTSRK(ts,TSRK1FE));
  CHKERRQ(TestTSRK(ts,TSRK2A));
  CHKERRQ(TestTSRK(ts,TSRK3));
  CHKERRQ(TestTSRK(ts,TSRK3BS));
  CHKERRQ(TestTSRK(ts,TSRK4));
  CHKERRQ(TestTSRK(ts,TSRK5F));
  CHKERRQ(TestTSRK(ts,TSRK5DP));
  CHKERRQ(TestTSRK(ts,TSRK5BS));
  CHKERRQ(TestTSRK(ts,TSRK6VR));
  CHKERRQ(TestTSRK(ts,TSRK7VR));
  CHKERRQ(TestTSRK(ts,TSRK8VR));

  CHKERRQ(TSRollBack(ts));
  CHKERRQ(TSDestroy(&ts));

  CHKERRQ(PetscFinalize());
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
