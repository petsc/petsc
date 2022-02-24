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
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));

  /* Call MatSetRandom on unassembled matrices */
  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,20,20,3,NULL,3,NULL,&A[0]));
  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,20,20,3,NULL,3,NULL,&A[1]));
  CHKERRQ(MatSetRandom(A[0],rctx));
  CHKERRQ(MatSetRandom(A[1],rctx));

  CHKERRQ(MatAXPY(A[0],1.0,A[1],DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(MatAXPY(A[0],-1.0,A[0],SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(A[0],NORM_1,&nrm));
  if (nrm > tol) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm(), norm1=: %g\n",(double)nrm));

  /* Call MatSetRandom on assembled matrices */
  CHKERRQ(MatSetRandom(A[0],rctx));
  CHKERRQ(MatSetRandom(A[1],rctx));

  CHKERRQ(MatAXPY(A[0],1.0,A[1],DIFFERENT_NONZERO_PATTERN));
  CHKERRQ(MatAXPY(A[0],-1.0,A[0],SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(A[0],NORM_1,&nrm));
  if (nrm > tol) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm(), norm1=: %g\n",(double)nrm));

  CHKERRQ(MatDestroy(&A[0]));
  CHKERRQ(MatDestroy(&A[1]));
  CHKERRQ(PetscRandomDestroy(&rctx));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   test:
      nsize: 3
TEST*/
