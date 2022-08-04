static char help[] = "Test the Fischer-3 initial guess routine.\n\n";

#include <petscksp.h>

#define SIZE 3

int main(int argc,char **args)
{
  PetscInt i;
  {
    Mat         A;
    PetscInt    indices[SIZE] = {0,1,2};
    PetscScalar values[SIZE] = {1.0,1.0,1.0};
    Vec         sol,rhs,newsol,newrhs;

    PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));

    /* common data structures */
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,SIZE,SIZE,NULL,&A));
    for (i = 0; i < SIZE; ++i) {
      PetscCall(MatSetValue(A,i,i,1.0,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF,SIZE,&sol));
    PetscCall(VecDuplicate(sol,&rhs));
    PetscCall(VecDuplicate(sol,&newrhs));
    PetscCall(VecDuplicate(sol,&newsol));

    PetscCall(VecSetValues(sol,SIZE,indices,values,INSERT_VALUES));
    PetscCall(VecSetValues(rhs,SIZE - 1,indices,values,INSERT_VALUES));
    PetscCall(VecSetValues(newrhs,SIZE - 2,indices,values,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(sol));
    PetscCall(VecAssemblyBegin(rhs));
    PetscCall(VecAssemblyBegin(newrhs));
    PetscCall(VecAssemblyEnd(sol));
    PetscCall(VecAssemblyEnd(rhs));
    PetscCall(VecAssemblyEnd(newrhs));

    /* Test one vector */
    {
      KSP      ksp;
      KSPGuess guess;

      PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
      PetscCall(KSPSetOperators(ksp,A,A));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(KSPGetGuess(ksp,&guess));
      /* we aren't calling through the KSP so we call this ourselves */
      PetscCall(KSPGuessSetUp(guess));

      PetscCall(KSPGuessUpdate(guess,rhs,sol));
      PetscCall(KSPGuessFormGuess(guess,newrhs,newsol));
      PetscCall(VecView(newsol,PETSC_VIEWER_STDOUT_SELF));

      PetscCall(KSPDestroy(&ksp));
    }

    /* Test a singular projection matrix */
    {
      KSP      ksp;
      KSPGuess guess;

      PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
      PetscCall(KSPSetOperators(ksp,A,A));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(KSPGetGuess(ksp,&guess));
      PetscCall(KSPGuessSetUp(guess));

      for (i = 0; i < 15; ++i) {
        PetscCall(KSPGuessUpdate(guess,rhs,sol));
      }
      PetscCall(KSPGuessFormGuess(guess,newrhs,newsol));
      PetscCall(VecView(newsol,PETSC_VIEWER_STDOUT_SELF));

      PetscCall(KSPDestroy(&ksp));
    }
    PetscCall(VecDestroy(&newsol));
    PetscCall(VecDestroy(&newrhs));
    PetscCall(VecDestroy(&rhs));
    PetscCall(VecDestroy(&sol));

    PetscCall(MatDestroy(&A));
  }

  /* Test something triangular */
  {
    PetscInt triangle_size = 10;
    Mat      A;

    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,triangle_size,triangle_size,NULL,&A));
    for (i = 0; i < triangle_size; ++i) {
      PetscCall(MatSetValue(A,i,i,1.0,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    {
      KSP         ksp;
      KSPGuess    guess;
      Vec         sol,rhs;
      PetscInt    j,indices[] = {0,1,2,3,4};
      PetscScalar values[] = {1.0,2.0,3.0,4.0,5.0};

      PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));
      PetscCall(KSPSetOperators(ksp,A,A));
      PetscCall(KSPSetFromOptions(ksp));
      PetscCall(KSPGetGuess(ksp,&guess));
      PetscCall(KSPGuessSetUp(guess));

      for (i = 0; i < 5; ++i) {
        PetscCall(VecCreateSeq(PETSC_COMM_SELF,triangle_size,&sol));
        PetscCall(VecCreateSeq(PETSC_COMM_SELF,triangle_size,&rhs));
        for (j = 0; j < i; ++j) {
          PetscCall(VecSetValue(sol,j,(PetscScalar)j,INSERT_VALUES));
          PetscCall(VecSetValue(rhs,j,(PetscScalar)j,INSERT_VALUES));
        }
        PetscCall(VecAssemblyBegin(sol));
        PetscCall(VecAssemblyBegin(rhs));
        PetscCall(VecAssemblyEnd(sol));
        PetscCall(VecAssemblyEnd(rhs));

        PetscCall(KSPGuessUpdate(guess,rhs,sol));

        PetscCall(VecDestroy(&rhs));
        PetscCall(VecDestroy(&sol));
      }

      PetscCall(VecCreateSeq(PETSC_COMM_SELF,triangle_size,&sol));
      PetscCall(VecCreateSeq(PETSC_COMM_SELF,triangle_size,&rhs));
      PetscCall(VecSetValues(rhs,5,indices,values,INSERT_VALUES));
      PetscCall(VecAssemblyBegin(sol));
      PetscCall(VecAssemblyEnd(sol));

      PetscCall(KSPGuessFormGuess(guess,rhs,sol));
      PetscCall(VecView(sol,PETSC_VIEWER_STDOUT_SELF));

      PetscCall(VecDestroy(&rhs));
      PetscCall(VecDestroy(&sol));
      PetscCall(KSPDestroy(&ksp));
    }
    PetscCall(MatDestroy(&A));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/* The relative tolerance here is strict enough to get rid of all the noise in both single and double precision: values as low as 5e-7 also work */

/*TEST

   test:
      args: -ksp_guess_type fischer -ksp_guess_fischer_model 3,10 -ksp_guess_fischer_monitor -ksp_guess_fischer_tol 1e-6

TEST*/
