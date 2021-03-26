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

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_ODE_IMPLICIT);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&f);CHKERRQ(ierr);
  ierr = VecSetSizes(f,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(f);CHKERRQ(ierr);
  ierr = VecSetUp(f);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,f,IFunction,NULL);CHKERRQ(ierr);
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
  ierr = TSSetIJacobian(ts,A,A,IJacobian,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,1,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetUp(x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1);CHKERRQ(ierr);
  ierr = TSSetTime(ts,0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,PETSC_MAX_REAL);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,3);CHKERRQ(ierr);

  /*
      When an ARKIMEX scheme with an explicit stage is used this will error with a message informing the user it is not possible to use
      a non-trivial mass matrix with ARKIMEX schemes with explicit stages.
  */
  ierr = TSSolve(ts,NULL);
  if (ierr != PETSC_ERR_ARG_INCOMP) CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode IFunction(TS ts,PetscReal t,Vec x,Vec xdot,Vec f,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(xdot,f);CHKERRQ(ierr);
  ierr = VecScale(f,2.0);CHKERRQ(ierr);
  ierr = VecShift(f,-1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobian(TS ts,PetscReal t,Vec x,Vec xdot,PetscReal shift,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    j;

  PetscFunctionBegin;
  j = shift*2.0;
  ierr = MatSetValue(B,0,0,j,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*TEST

    test:
      suffix: arkimex_explicit_stage
      requires: define(PETSC_USE_DEBUG)
      args: -ts_type arkimex -error_output_stdout
      filter:  egrep -v "(Petsc|on a| at |Configure)"

    test:
      suffix: arkimex_implicit_stage
      args: -ts_type arkimex -ts_arkimex_type l2 -ts_monitor_solution -ts_monitor

TEST*/
