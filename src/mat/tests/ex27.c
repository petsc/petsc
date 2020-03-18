static char help[]= "Test MatSetRandom on MATMPIAIJ matrices\n\n";

/*
   Adapted from an example Contributed-by: Jakub Kruzik <jakub.kruzik@vsb.cz>
*/
#include <petscmat.h>
int main(int argc,char **args)
{
  Mat            A[2];
  PetscErrorCode ierr;
  PetscReal      nrm,tol=10*PETSC_SMALL;
  PetscRandom    rctx;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);

  /* Call MatSetRandom on unassembled matrices */
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,20,20,3,NULL,3,NULL,&A[0]);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,20,20,3,NULL,3,NULL,&A[1]);CHKERRQ(ierr);
  ierr = MatSetRandom(A[0],rctx);CHKERRQ(ierr);
  ierr = MatSetRandom(A[1],rctx);CHKERRQ(ierr);

  ierr = MatAXPY(A[0],1.0,A[1],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(A[0],-1.0,A[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(A[0],NORM_1,&nrm);CHKERRQ(ierr);
  if (nrm > tol) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm(), norm1=: %g\n",(double)nrm);CHKERRQ(ierr);}

  /* Call MatSetRandom on assembled matrices */
  ierr = MatSetRandom(A[0],rctx);CHKERRQ(ierr);
  ierr = MatSetRandom(A[1],rctx);CHKERRQ(ierr);

  ierr = MatAXPY(A[0],1.0,A[1],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(A[0],-1.0,A[0],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(A[0],NORM_1,&nrm);CHKERRQ(ierr);
  if (nrm > tol) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm(), norm1=: %g\n",(double)nrm);CHKERRQ(ierr);}

  ierr = MatDestroy(&A[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&A[1]);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   test:
      nsize: 3
TEST*/
