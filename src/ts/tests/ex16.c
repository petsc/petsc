
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

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&f);CHKERRQ(ierr);
  ierr = VecSetSizes(f,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(f);CHKERRQ(ierr);
  ierr = VecSetUp(f);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,f,RHSFunction,NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,1,1,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* ensure that the Jacobian matrix has diagonal entries since that is required by TS */
  ierr = MatShift(A,(PetscReal)1);CHKERRQ(ierr);
  ierr = MatShift(A,(PetscReal)-1);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,RHSJacobian,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetUp(x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);

  {
    PetscInt  direction[3];
    PetscBool terminate[3];
    direction[0] = +1; terminate[0] = PETSC_FALSE;
    direction[1] = -1; terminate[1] = PETSC_FALSE;
    direction[2] =  0; terminate[2] = PETSC_FALSE;
    ierr = TSSetEventHandler(ts,3,direction,terminate,Event,PostEvent,NULL);CHKERRQ(ierr);
  }
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,t_end);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(f,(PetscReal)1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec x,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  if (B != A) {ierr = MatZeroEntries(A);CHKERRQ(ierr);}
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = TSGetStepNumber(ts,&i);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&a);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,i,(double)t,(double)PetscRealPart(a[0]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&a);CHKERRQ(ierr);
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
