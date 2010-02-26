/*
      Demonstrates the use of the "extra", polymorphic versions of many functions
*/
#include "petscmat.h"

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Vec            x;
  Mat            mat,matb,matsb;

#if defined(__cplusplus)
  PetscInitialize(&argc,&args);
#else
  PetscInitialize(&argc,&args,0,0);
#endif

#if defined(__cplusplus)
  ierr = VecCreate(&x);CHKERRQ(ierr);
#else
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
#endif
  ierr = VecSetSizes(x,6,0);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

#if defined(__cplusplus)
  mat   = MatCreateSeqAIJ(6,6);
  matb  = MatCreateSeqBAIJ(2,6,6,5);
  matsb = MatCreateSeqSBAIJ(2,6,6,5);
#else
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,6,6,0,0,&mat);CHKERRQ(ierr);
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,2,6,6,5,0,&mat);CHKERRQ(ierr);
  ierr = MatCreateSeqSBAIJ(PETSC_COMM_SELF,2,6,6,5,0,&mat);CHKERRQ(ierr);
#endif

  ierr = MatDestroy(mat);CHKERRQ(ierr);
  ierr = MatDestroy(matb);CHKERRQ(ierr);
  ierr = MatDestroy(matsb);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
