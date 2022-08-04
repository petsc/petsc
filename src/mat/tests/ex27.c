static char help[]= "Test MatSetRandom on MATMPIAIJ matrices\n\n";

/*
   Adapted from an example Contributed-by: Jakub Kruzik <jakub.kruzik@vsb.cz>
*/
#include <petscmat.h>
int main(int argc,char **args)
{
  Mat            A[2];
  PetscReal      nrm,tol=10*PETSC_SMALL;
  PetscRandom    rctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));

  /* Call MatSetRandom on unassembled matrices */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,20,20,3,NULL,3,NULL,&A[0]));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,20,20,3,NULL,3,NULL,&A[1]));
  PetscCall(MatSetRandom(A[0],rctx));
  PetscCall(MatSetRandom(A[1],rctx));

  PetscCall(MatAXPY(A[0],1.0,A[1],DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatAXPY(A[0],-1.0,A[0],SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(A[0],NORM_1,&nrm));
  if (nrm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm(), norm1=: %g\n",(double)nrm));

  /* Call MatSetRandom on assembled matrices */
  PetscCall(MatSetRandom(A[0],rctx));
  PetscCall(MatSetRandom(A[1],rctx));

  PetscCall(MatAXPY(A[0],1.0,A[1],DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatAXPY(A[0],-1.0,A[0],SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(A[0],NORM_1,&nrm));
  if (nrm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error: MatNorm(), norm1=: %g\n",(double)nrm));

  PetscCall(MatDestroy(&A[0]));
  PetscCall(MatDestroy(&A[1]));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   test:
      nsize: 3
TEST*/
