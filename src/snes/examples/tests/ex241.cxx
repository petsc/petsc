static char help[] = "Artificial test to check that snes->domainerror is being reset appropriately";

/* ------------------------------------------------------------------------

    Artificial test to check that snes->domainerror is being reset appropriately

  ------------------------------------------------------------------------- */

#define PETSC_SKIP_COMPLEX
#include <petscsnes.h>

typedef struct {
  PetscReal value;              /* parameter in nonlinear function */
} AppCtx;

PetscErrorCode UserFunction(SNES,Vec,Vec,void*);
PetscErrorCode UserJacobian(SNES,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  SNES           snes;
  Vec            x,r;
  Mat            J;
  PetscErrorCode ierr;
  PetscInt       its;
  AppCtx         user;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* Allocate vectors / matrix */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,1,1,1,NULL,&J);CHKERRQ(ierr);

  /* Create / set-up SNES */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,r,UserFunction,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,UserJacobian,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* Set initial guess (=1) and target value */
  user.value = 1e-4;

  ierr = VecSet(x,1.0);CHKERRQ(ierr);

  /* Set initial guess / solve */
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  
  /* Done */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*
    UserFunction - for nonlinear function x^2 - value = 0
*/
PetscErrorCode UserFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx            *user = (AppCtx*)ptr;
  PetscInt          N,i;
  PetscScalar       *f;
  PetscReal         half;
  const PetscScalar *x;
  PetscErrorCode    ierr;

  half = 0.5;

  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /* Calculate residual */
  for(i=0; i<N; ++i) {
    /* 
       Test for domain error.
       Artifical test is applied.  With starting value 1.0, first iterate will be 0.5 + user->value/2.
       Declare (0.5-value,0.5+value) to be infeasible.
       In later iterations, snes->domainerror should be cleared, allowing iterations in the feasible region to be accepted.
    */
    if( (half-user->value) < PetscRealPart(x[i]) && PetscRealPart(x[i]) < (half+user->value) ) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"DOMAIN ERROR: x=%g\n",(double)PetscRealPart(x[i]));CHKERRQ(ierr);
      ierr = SNESSetFunctionDomainError(snes);CHKERRQ(ierr);
    }
    f[i] = x[i]*x[i] - user->value;
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  return 0;
}

/*
    UserJacobian - for nonlinear function x^2 - value = 0
*/
PetscErrorCode UserJacobian(SNES snes,Vec X,Mat J,Mat jac,void *ptr)
{
  PetscInt          N,i,row,col;
  const PetscScalar *x;
  PetscScalar       v;
  PetscErrorCode    ierr;

  ierr = VecGetSize(X,&N);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  
  /* Calculate Jacobian */
  for (i=0; i<N; ++i) {
    row = i;
    col = i;
    v = 2*x[i];
    ierr = MatSetValues(jac,1,&row,1,&col,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (jac != J) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  return 0;
}

/*TEST

   build:
      requires: !single !define(PETSC_HAVE_SUN_CXX) !complex

   test:
      args:  -snes_monitor_solution -snes_linesearch_monitor

TEST*/

