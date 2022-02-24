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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Allocate vectors / matrix */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,1));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecDuplicate(x,&r));

  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD,1,1,1,NULL,&J));

  /* Create / set-up SNES */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
  CHKERRQ(SNESSetFunction(snes,r,UserFunction,&user));
  CHKERRQ(SNESSetJacobian(snes,J,J,UserJacobian,&user));
  CHKERRQ(SNESSetFromOptions(snes));

  /* Set initial guess (=1) and target value */
  user.value = 1e-4;

  CHKERRQ(VecSet(x,1.0));

  /* Set initial guess / solve */
  CHKERRQ(SNESSolve(snes,NULL,x));
  CHKERRQ(SNESGetIterationNumber(snes,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Done */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(SNESDestroy(&snes));
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

  half = 0.5;

  CHKERRQ(VecGetSize(X,&N));
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(F,&f));

  /* Calculate residual */
  for (i=0; i<N; ++i) {
    /*
       Test for domain error.
       Artifical test is applied.  With starting value 1.0, first iterate will be 0.5 + user->value/2.
       Declare (0.5-value,0.5+value) to be infeasible.
       In later iterations, snes->domainerror should be cleared, allowing iterations in the feasible region to be accepted.
    */
    if ((half-user->value) < PetscRealPart(x[i]) && PetscRealPart(x[i]) < (half+user->value)) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"DOMAIN ERROR: x=%g\n",(double)PetscRealPart(x[i])));
      CHKERRQ(SNESSetFunctionDomainError(snes));
    }
    f[i] = x[i]*x[i] - user->value;
  }
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
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

  CHKERRQ(VecGetSize(X,&N));
  CHKERRQ(VecGetArrayRead(X,&x));

  /* Calculate Jacobian */
  for (i=0; i<N; ++i) {
    row = i;
    col = i;
    v = 2*x[i];
    CHKERRQ(MatSetValues(jac,1,&row,1,&col,&v,INSERT_VALUES));
  }
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));

  if (jac != J) {
    CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  return 0;
}

/*TEST

   build:
      requires: !single !defined(PETSC_HAVE_SUN_CXX) !complex

   test:
      args:  -snes_monitor_solution -snes_linesearch_monitor

TEST*/
