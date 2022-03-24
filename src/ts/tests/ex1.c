
static char help[] = "Solves the trival ODE du/dt = 1, u(0) = 0. \n\n";

#include <petscts.h>
#include <petscpc.h>

static PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);

static PetscErrorCode PreStep(TS);
static PetscErrorCode PostStep(TS);
static PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);
static PetscErrorCode Event(TS,PetscReal,Vec,PetscScalar*,void*);
static PetscErrorCode PostEvent(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*);

int main(int argc,char **argv)
{
  TS              ts;
  PetscInt        n;
  const PetscInt  n_end = 11;
  PetscReal       t;
  const PetscReal t_end = 11;
  Vec             x;
  Vec             f;
  Mat             A;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&f));
  CHKERRQ(VecSetSizes(f,1,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(f));
  CHKERRQ(VecSetUp(f));
  CHKERRQ(TSSetRHSFunction(ts,f,RHSFunction,NULL));
  CHKERRQ(VecDestroy(&f));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,1,1,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  /* ensure that the Jacobian matrix has diagonal entries since that is required by TS */
  CHKERRQ(MatShift(A,(PetscReal)1));
  CHKERRQ(MatShift(A,(PetscReal)-1));
  CHKERRQ(TSSetRHSJacobian(ts,A,A,RHSJacobian,NULL));
  CHKERRQ(MatDestroy(&A));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,1,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSetUp(x));
  CHKERRQ(TSSetSolution(ts,x));
  CHKERRQ(VecDestroy(&x));

  CHKERRQ(TSMonitorSet(ts,Monitor,NULL,NULL));
  CHKERRQ(TSSetPreStep(ts,PreStep));
  CHKERRQ(TSSetPostStep(ts,PostStep));

  {
    TSAdapt adapt;
    CHKERRQ(TSGetAdapt(ts,&adapt));
    CHKERRQ(TSAdaptSetType(adapt,TSADAPTNONE));
  }
  {
    PetscInt  direction[3];
    PetscBool terminate[3];
    direction[0] = +1; terminate[0] = PETSC_FALSE;
    direction[1] = -1; terminate[1] = PETSC_FALSE;
    direction[2] =  0; terminate[2] = PETSC_FALSE;
    CHKERRQ(TSSetEventHandler(ts,3,direction,terminate,Event,PostEvent,NULL));
  }
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetFromOptions(ts));

  /* --- First Solve --- */

  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetTimeStep(ts,1));
  CHKERRQ(TSSetTime(ts,0));
  CHKERRQ(TSSetMaxTime(ts,PETSC_MAX_REAL));
  CHKERRQ(TSSetMaxSteps(ts,3));

  CHKERRQ(TSGetTime(ts,&t));
  CHKERRQ(TSGetSolution(ts,&x));
  CHKERRQ(VecSet(x,t));
  while (t < t_end) {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n"));
    CHKERRQ(TSSolve(ts,NULL));
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n"));
    CHKERRQ(TSGetTime(ts,&t));
    CHKERRQ(TSGetStepNumber(ts,&n));
    CHKERRQ(TSSetMaxSteps(ts,PetscMin(n+3,n_end)));
  }
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n"));
  CHKERRQ(TSSolve(ts,NULL));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n"));

  /* --- Second Solve --- */

  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetTimeStep(ts,1));
  CHKERRQ(TSSetTime(ts,0));
  CHKERRQ(TSSetMaxTime(ts,3));
  CHKERRQ(TSSetMaxSteps(ts,PETSC_MAX_INT));

  CHKERRQ(TSGetTime(ts,&t));
  CHKERRQ(TSGetSolution(ts,&x));
  CHKERRQ(VecSet(x,t));
  while (t < t_end) {
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n"));
    CHKERRQ(TSSolve(ts,NULL));
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n"));
    CHKERRQ(TSGetTime(ts,&t));
    CHKERRQ(TSSetMaxTime(ts,PetscMin(t+3,t_end)));
  }
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n"));
  CHKERRQ(TSSolve(ts,NULL));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n"));

  /* --- */

  CHKERRQ(TSDestroy(&ts));

  CHKERRQ(PetscFinalize());
  return 0;
}

/* -------------------------------------------------------------------*/

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec x,Vec f,void *ctx)
{
  PetscFunctionBegin;
  CHKERRQ(VecSet(f,(PetscReal)1));
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat A,Mat B,void *ctx)
{
  PetscFunctionBegin;
  CHKERRQ(MatZeroEntries(B));
  if (B != A) CHKERRQ(MatZeroEntries(A));
  PetscFunctionReturn(0);
}

PetscErrorCode PreStep(TS ts)
{
  PetscInt          n;
  PetscReal         t;
  Vec               x;
  const PetscScalar *a;

  PetscFunctionBegin;
  CHKERRQ(TSGetStepNumber(ts,&n));
  CHKERRQ(TSGetTime(ts,&t));
  CHKERRQ(TSGetSolution(ts,&x));
  CHKERRQ(VecGetArrayRead(x,&a));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)PetscRealPart(a[0])));
  CHKERRQ(VecRestoreArrayRead(x,&a));
  PetscFunctionReturn(0);
}

PetscErrorCode PostStep(TS ts)
{
  PetscInt          n;
  PetscReal         t;
  Vec               x;
  const PetscScalar *a;

  PetscFunctionBegin;
  CHKERRQ(TSGetStepNumber(ts,&n));
  CHKERRQ(TSGetTime(ts,&t));
  CHKERRQ(TSGetSolution(ts,&x));
  CHKERRQ(VecGetArrayRead(x,&a));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)PetscRealPart(a[0])));
  CHKERRQ(VecRestoreArrayRead(x,&a));
  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(TS ts,PetscInt n,PetscReal t,Vec x,void *ctx)
{
  const PetscScalar *a;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(x,&a));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)PetscRealPart(a[0])));
  CHKERRQ(VecRestoreArrayRead(x,&a));
  PetscFunctionReturn(0);
}

PetscErrorCode Event(TS ts,PetscReal t,Vec x,PetscScalar *fvalue,void *ctx)
{
  PetscFunctionBegin;
  fvalue[0] = t - 5;
  fvalue[1] = 7 - t;
  fvalue[2] = t - 9;
  PetscFunctionReturn(0);
}

PetscErrorCode PostEvent(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec x,PetscBool forwardsolve,void* ctx)
{
  PetscInt          i;
  const PetscScalar *a;

  PetscFunctionBegin;
  CHKERRQ(TSGetStepNumber(ts,&i));
  CHKERRQ(VecGetArrayRead(x,&a));
  CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,i,(double)t,(double)PetscRealPart(a[0])));
  CHKERRQ(VecRestoreArrayRead(x,&a));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      suffix: euler
      args: -ts_type euler
      output_file: output/ex1.out

    test:
      suffix: ssp
      args:   -ts_type ssp
      output_file: output/ex1.out

    test:
      suffix: rk
      args: -ts_type rk
      output_file: output/ex1.out

    test:
      suffix: beuler
      args: -ts_type beuler
      output_file: output/ex1.out

    test:
      suffix: cn
      args: -ts_type cn
      output_file: output/ex1.out

    test:
      suffix: theta
      args: -ts_type theta
      output_file: output/ex1.out

    test:
      suffix: bdf
      args: -ts_type bdf
      output_file: output/ex1.out

    test:
      suffix: alpha
      args: -ts_type alpha
      output_file: output/ex1.out

    test:
      suffix: rosw
      args: -ts_type rosw
      output_file: output/ex1.out

    test:
      suffix: arkimex
      args: -ts_type arkimex
      output_file: output/ex1.out

TEST*/
