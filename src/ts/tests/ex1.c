
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&f));
  PetscCall(VecSetSizes(f,1,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(f));
  PetscCall(VecSetUp(f));
  PetscCall(TSSetRHSFunction(ts,f,RHSFunction,NULL));
  PetscCall(VecDestroy(&f));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,1,1,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  /* ensure that the Jacobian matrix has diagonal entries since that is required by TS */
  PetscCall(MatShift(A,(PetscReal)1));
  PetscCall(MatShift(A,(PetscReal)-1));
  PetscCall(TSSetRHSJacobian(ts,A,A,RHSJacobian,NULL));
  PetscCall(MatDestroy(&A));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,1,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetUp(x));
  PetscCall(TSSetSolution(ts,x));
  PetscCall(VecDestroy(&x));

  PetscCall(TSMonitorSet(ts,Monitor,NULL,NULL));
  PetscCall(TSSetPreStep(ts,PreStep));
  PetscCall(TSSetPostStep(ts,PostStep));

  {
    TSAdapt adapt;
    PetscCall(TSGetAdapt(ts,&adapt));
    PetscCall(TSAdaptSetType(adapt,TSADAPTNONE));
  }
  {
    PetscInt  direction[3];
    PetscBool terminate[3];
    direction[0] = +1; terminate[0] = PETSC_FALSE;
    direction[1] = -1; terminate[1] = PETSC_FALSE;
    direction[2] =  0; terminate[2] = PETSC_FALSE;
    PetscCall(TSSetEventHandler(ts,3,direction,terminate,Event,PostEvent,NULL));
  }
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  /* --- First Solve --- */

  PetscCall(TSSetStepNumber(ts,0));
  PetscCall(TSSetTimeStep(ts,1));
  PetscCall(TSSetTime(ts,0));
  PetscCall(TSSetMaxTime(ts,PETSC_MAX_REAL));
  PetscCall(TSSetMaxSteps(ts,3));

  PetscCall(TSGetTime(ts,&t));
  PetscCall(TSGetSolution(ts,&x));
  PetscCall(VecSet(x,t));
  while (t < t_end) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n"));
    PetscCall(TSSolve(ts,NULL));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n"));
    PetscCall(TSGetTime(ts,&t));
    PetscCall(TSGetStepNumber(ts,&n));
    PetscCall(TSSetMaxSteps(ts,PetscMin(n+3,n_end)));
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n"));
  PetscCall(TSSolve(ts,NULL));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n"));

  /* --- Second Solve --- */

  PetscCall(TSSetStepNumber(ts,0));
  PetscCall(TSSetTimeStep(ts,1));
  PetscCall(TSSetTime(ts,0));
  PetscCall(TSSetMaxTime(ts,3));
  PetscCall(TSSetMaxSteps(ts,PETSC_MAX_INT));

  PetscCall(TSGetTime(ts,&t));
  PetscCall(TSGetSolution(ts,&x));
  PetscCall(VecSet(x,t));
  while (t < t_end) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n"));
    PetscCall(TSSolve(ts,NULL));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n"));
    PetscCall(TSGetTime(ts,&t));
    PetscCall(TSSetMaxTime(ts,PetscMin(t+3,t_end)));
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n"));
  PetscCall(TSSolve(ts,NULL));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n"));

  /* --- */

  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
  return 0;
}

/* -------------------------------------------------------------------*/

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec x,Vec f,void *ctx)
{
  PetscFunctionBegin;
  PetscCall(VecSet(f,(PetscReal)1));
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat A,Mat B,void *ctx)
{
  PetscFunctionBegin;
  PetscCall(MatZeroEntries(B));
  if (B != A) PetscCall(MatZeroEntries(A));
  PetscFunctionReturn(0);
}

PetscErrorCode PreStep(TS ts)
{
  PetscInt          n;
  PetscReal         t;
  Vec               x;
  const PetscScalar *a;

  PetscFunctionBegin;
  PetscCall(TSGetStepNumber(ts,&n));
  PetscCall(TSGetTime(ts,&t));
  PetscCall(TSGetSolution(ts,&x));
  PetscCall(VecGetArrayRead(x,&a));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)PetscRealPart(a[0])));
  PetscCall(VecRestoreArrayRead(x,&a));
  PetscFunctionReturn(0);
}

PetscErrorCode PostStep(TS ts)
{
  PetscInt          n;
  PetscReal         t;
  Vec               x;
  const PetscScalar *a;

  PetscFunctionBegin;
  PetscCall(TSGetStepNumber(ts,&n));
  PetscCall(TSGetTime(ts,&t));
  PetscCall(TSGetSolution(ts,&x));
  PetscCall(VecGetArrayRead(x,&a));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)PetscRealPart(a[0])));
  PetscCall(VecRestoreArrayRead(x,&a));
  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(TS ts,PetscInt n,PetscReal t,Vec x,void *ctx)
{
  const PetscScalar *a;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&a));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)PetscRealPart(a[0])));
  PetscCall(VecRestoreArrayRead(x,&a));
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
  PetscCall(TSGetStepNumber(ts,&i));
  PetscCall(VecGetArrayRead(x,&a));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,i,(double)t,(double)PetscRealPart(a[0])));
  PetscCall(VecRestoreArrayRead(x,&a));
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
