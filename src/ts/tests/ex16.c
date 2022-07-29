
static char help[] = "Solves the trivial ODE du/dt = 1, u(0) = 0. \n\n";
/*
  This example tests TSEvent's capability to handle complicated cases.
  Test 1: an event close to endpoint of a time step should not be detected twice.
  Test 2: two events in the same time step should be detected in the correct order.
  Test 3: three events in the same time step should be detected in the correct order.
*/

#include <petscts.h>

static PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);

static PetscErrorCode Event(TS,PetscReal,Vec,PetscScalar*,void*);
static PetscErrorCode PostEvent(TS,PetscInt,PetscInt[],PetscReal,Vec,PetscBool,void*);

int main(int argc,char **argv)
{
  TS              ts;
  const PetscReal t_end = 2.0;
  Vec             x;
  Vec             f;
  Mat             A;

  PetscFunctionBeginUser;
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

  {
    PetscInt  direction[3];
    PetscBool terminate[3];
    direction[0] = +1; terminate[0] = PETSC_FALSE;
    direction[1] = -1; terminate[1] = PETSC_FALSE;
    direction[2] =  0; terminate[2] = PETSC_FALSE;
    PetscCall(TSSetEventHandler(ts,3,direction,terminate,Event,PostEvent,NULL));
  }
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetMaxTime(ts,t_end));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts,NULL));

  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec x,Vec f,void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(f,(PetscReal)1));
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat A,Mat B,void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(B));
  if (B != A) PetscCall(MatZeroEntries(A));
  PetscFunctionReturn(0);
}

PetscErrorCode Event(TS ts,PetscReal t,Vec x,PetscScalar *fvalue,void *ctx)
{
  PetscFunctionBeginUser;
  fvalue[0] = t - 1.1;
  fvalue[1] = 1.2 - t;
  fvalue[2] = t - 1.3;
  PetscFunctionReturn(0);
}

PetscErrorCode PostEvent(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec x,PetscBool forwardsolve,void* ctx)
{
  PetscInt          i;
  const PetscScalar *a;

  PetscFunctionBeginUser;
  PetscCall(TSGetStepNumber(ts,&i));
  PetscCall(VecGetArrayRead(x,&a));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %" PetscInt_FMT " time %g value %g\n",PETSC_FUNCTION_NAME,i,(double)t,(double)PetscRealPart(a[0])));
  PetscCall(VecRestoreArrayRead(x,&a));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -ts_type beuler -ts_dt 0.1 -ts_event_monitor

    test:
      suffix: 2
      args: -ts_type beuler -ts_dt 0.2 -ts_event_monitor

    test:
      suffix: 3
      args: -ts_type beuler -ts_dt 0.5 -ts_event_monitor
TEST*/
