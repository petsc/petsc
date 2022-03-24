static char help[] = "Solves the trival ODE 2 du/dt = 1, u(0) = 0. \n\n";

#include <petscts.h>
#include <petscpc.h>

PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);

int main(int argc,char **argv)
{
  TS              ts;
  Vec             x;
  Vec             f;
  Mat             A;
  PetscErrorCode  ierr;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetEquationType(ts,TS_EQ_ODE_IMPLICIT));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&f));
  CHKERRQ(VecSetSizes(f,1,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(f));
  CHKERRQ(VecSetUp(f));
  CHKERRQ(TSSetIFunction(ts,f,IFunction,NULL));
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
  CHKERRQ(TSSetIJacobian(ts,A,A,IJacobian,NULL));
  CHKERRQ(MatDestroy(&A));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,1,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSetUp(x));
  CHKERRQ(TSSetSolution(ts,x));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetTimeStep(ts,1));
  CHKERRQ(TSSetTime(ts,0));
  CHKERRQ(TSSetMaxTime(ts,PETSC_MAX_REAL));
  CHKERRQ(TSSetMaxSteps(ts,3));

  /*
      When an ARKIMEX scheme with an explicit stage is used this will error with a message informing the user it is not possible to use
      a non-trivial mass matrix with ARKIMEX schemes with explicit stages.
  */
  ierr = TSSolve(ts,NULL);
  if (ierr != PETSC_ERR_ARG_INCOMP) CHKERRQ(ierr);

  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode IFunction(TS ts,PetscReal t,Vec x,Vec xdot,Vec f,void *ctx)
{
  PetscFunctionBegin;
  CHKERRQ(VecCopy(xdot,f));
  CHKERRQ(VecScale(f,2.0));
  CHKERRQ(VecShift(f,-1.0));
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobian(TS ts,PetscReal t,Vec x,Vec xdot,PetscReal shift,Mat A,Mat B,void *ctx)
{
  PetscScalar    j;

  PetscFunctionBegin;
  j = shift*2.0;
  CHKERRQ(MatSetValue(B,0,0,j,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      suffix: arkimex_explicit_stage
      requires: defined(PETSC_USE_DEBUG)
      args: -ts_type arkimex -error_output_stdout
      filter:  egrep -v "(Petsc|on a| at |Configure)"

    test:
      suffix: arkimex_implicit_stage
      args: -ts_type arkimex -ts_arkimex_type l2 -ts_monitor_solution -ts_monitor

TEST*/
