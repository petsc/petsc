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
  PetscInt       its;
  AppCtx         user;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Allocate vectors / matrix */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,1));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&r));

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD,1,1,1,NULL,&J));

  /* Create / set-up SNES */
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetFunction(snes,r,UserFunction,&user));
  PetscCall(SNESSetJacobian(snes,J,J,UserJacobian,&user));
  PetscCall(SNESSetFromOptions(snes));

  /* Set initial guess (=1) and target value */
  user.value = 1e-4;

  PetscCall(VecSet(x,1.0));

  /* Set initial guess / solve */
  PetscCall(SNESSolve(snes,NULL,x));
  PetscCall(SNESGetIterationNumber(snes,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Done */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
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

  PetscCall(VecGetSize(X,&N));
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArray(F,&f));

  /* Calculate residual */
  for (i=0; i<N; ++i) {
    /*
       Test for domain error.
       Artifical test is applied.  With starting value 1.0, first iterate will be 0.5 + user->value/2.
       Declare (0.5-value,0.5+value) to be infeasible.
       In later iterations, snes->domainerror should be cleared, allowing iterations in the feasible region to be accepted.
    */
    if ((half-user->value) < PetscRealPart(x[i]) && PetscRealPart(x[i]) < (half+user->value)) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"DOMAIN ERROR: x=%g\n",(double)PetscRealPart(x[i])));
      PetscCall(SNESSetFunctionDomainError(snes));
    }
    f[i] = x[i]*x[i] - user->value;
  }
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArray(F,&f));
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

  PetscCall(VecGetSize(X,&N));
  PetscCall(VecGetArrayRead(X,&x));

  /* Calculate Jacobian */
  for (i=0; i<N; ++i) {
    row = i;
    col = i;
    v = 2*x[i];
    PetscCall(MatSetValues(jac,1,&row,1,&col,&v,INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));

  if (jac != J) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  return 0;
}

/*TEST

   build:
      requires: !single !defined(PETSC_HAVE_SUN_CXX) !complex

   test:
      args:  -snes_monitor_solution -snes_linesearch_monitor

TEST*/
