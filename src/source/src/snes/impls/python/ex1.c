#include <Python.h>
#include <petscsnes.h>

static char help[] = "Python-implemented SNES.\n\n";

extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  PetscInt N;
  Vec      x,F;
  Mat      A;
  SNES     snes;

  PetscErrorCode ierr;
  
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);
  Py_InitializeEx(0); PyRun_SimpleString("from petsc4py import PETSc\n");

  ierr = PetscOptionsSetValue("-snes_python", "mysolver,MyNewton");CHKERRQ(ierr);

  N = 10;
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  
  ierr = MatGetVecs(A,&x,&F);;CHKERRQ(ierr);
  
  ierr = SNESCreate(PETSC_COMM_SELF,&snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, F,    FormFunction, PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, A, A, FormJacobian, PETSC_NULL);CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = VecSetRandom(x, PETSC_NULL);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);

  ierr = VecDestroy(F);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = SNESDestroy(snes);CHKERRQ(ierr);

  ierr = PetscOptionsClearValue("-snes_python");CHKERRQ(ierr);

  Py_Finalize();
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;

}

PetscErrorCode FormFunction(SNES snes,Vec x,Vec F,void* dummy)
{
  VecPointwiseMult(F, x, x);
  return 0;
}

PetscErrorCode FormJacobian(SNES snes,Vec x,Mat* J,Mat* P,MatStructure* flag,void* dummy)
{
  MatDiagonalSet(*P,x,INSERT_VALUES);
  MatScale(*P, 2);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}
