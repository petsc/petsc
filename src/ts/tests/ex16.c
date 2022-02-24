
static char help[] = "Solves the trival ODE du/dt = 1, u(0) = 0. \n\n";
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
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

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

  {
    PetscInt  direction[3];
    PetscBool terminate[3];
    direction[0] = +1; terminate[0] = PETSC_FALSE;
    direction[1] = -1; terminate[1] = PETSC_FALSE;
    direction[2] =  0; terminate[2] = PETSC_FALSE;
    CHKERRQ(TSSetEventHandler(ts,3,direction,terminate,Event,PostEvent,NULL));
  }
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetMaxTime(ts,t_end));
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(TSSolve(ts,NULL));

  CHKERRQ(TSDestroy(&ts));
  ierr = PetscFinalize();
  return ierr;
}

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

PetscErrorCode Event(TS ts,PetscReal t,Vec x,PetscScalar *fvalue,void *ctx)
{
  PetscFunctionBegin;
  fvalue[0] = t - 1.1;
  fvalue[1] = 1.2 - t;
  fvalue[2] = t - 1.3;
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
      args: -ts_type beuler -ts_dt 0.1 -ts_event_monitor

    test:
      suffix: 2
      args: -ts_type beuler -ts_dt 0.2 -ts_event_monitor

    test:
      suffix: 3
      args: -ts_type beuler -ts_dt 0.5 -ts_event_monitor
TEST*/
