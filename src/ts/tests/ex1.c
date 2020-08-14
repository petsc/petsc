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

  ierr = TSMonitorSet(ts,Monitor,NULL,NULL);CHKERRQ(ierr);
  ierr = TSSetPreStep(ts,PreStep);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,PostStep);CHKERRQ(ierr);

  {
    TSAdapt adapt;
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);
  }
  {
    PetscInt  direction[3];
    PetscBool terminate[3];
    direction[0] = +1; terminate[0] = PETSC_FALSE;
    direction[1] = -1; terminate[1] = PETSC_FALSE;
    direction[2] =  0; terminate[2] = PETSC_FALSE;
    ierr = TSSetEventHandler(ts,3,direction,terminate,Event,PostEvent,NULL);CHKERRQ(ierr);
  }
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* --- First Solve --- */

  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,PETSC_MAX_REAL);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,3);CHKERRQ(ierr);

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&x);CHKERRQ(ierr);
  ierr = VecSet(x,t);CHKERRQ(ierr);
  while (t < t_end) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n");CHKERRQ(ierr);
    ierr = TSSolve(ts,NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n");CHKERRQ(ierr);
    ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
    ierr = TSGetStepNumber(ts,&n);CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ts,PetscMin(n+3,n_end));CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n");CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n");CHKERRQ(ierr);

  /* --- Second Solve --- */

  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,3);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,PETSC_MAX_INT);CHKERRQ(ierr);

  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&x);CHKERRQ(ierr);
  ierr = VecSet(x,t);CHKERRQ(ierr);
  while (t < t_end) {
    ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n");CHKERRQ(ierr);
    ierr = TSSolve(ts,NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n");CHKERRQ(ierr);
    ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,PetscMin(t+3,t_end));CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: Begin\n");CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"TSSolve: End\n\n");CHKERRQ(ierr);

  /* --- */

  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/* -------------------------------------------------------------------*/

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

PetscErrorCode PreStep(TS ts)
{
  PetscInt          n;
  PetscReal         t;
  Vec               x;
  const PetscScalar *a;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = TSGetStepNumber(ts,&n);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&a);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)PetscRealPart(a[0]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PostStep(TS ts)
{
  PetscInt          n;
  PetscReal         t;
  Vec               x;
  const PetscScalar *a;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = TSGetStepNumber(ts,&n);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&a);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)PetscRealPart(a[0]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(TS ts,PetscInt n,PetscReal t,Vec x,void *ctx)
{
  const PetscScalar *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&a);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"%-10s-> step %D time %g value %g\n",PETSC_FUNCTION_NAME,n,(double)t,(double)PetscRealPart(a[0]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&a);CHKERRQ(ierr);
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
