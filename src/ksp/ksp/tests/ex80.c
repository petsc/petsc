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

    CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));

    /* common data structures */
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,SIZE,SIZE,NULL,&A));
    for (i = 0; i < SIZE; ++i) {
      CHKERRQ(MatSetValue(A,i,i,1.0,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,SIZE,&sol));
    CHKERRQ(VecDuplicate(sol,&rhs));
    CHKERRQ(VecDuplicate(sol,&newrhs));
    CHKERRQ(VecDuplicate(sol,&newsol));

    CHKERRQ(VecSetValues(sol,SIZE,indices,values,INSERT_VALUES));
    CHKERRQ(VecSetValues(rhs,SIZE - 1,indices,values,INSERT_VALUES));
    CHKERRQ(VecSetValues(newrhs,SIZE - 2,indices,values,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(sol));
    CHKERRQ(VecAssemblyBegin(rhs));
    CHKERRQ(VecAssemblyBegin(newrhs));
    CHKERRQ(VecAssemblyEnd(sol));
    CHKERRQ(VecAssemblyEnd(rhs));
    CHKERRQ(VecAssemblyEnd(newrhs));

    /* Test one vector */
    {
      KSP      ksp;
      KSPGuess guess;

      CHKERRQ(KSPCreate(PETSC_COMM_SELF,&ksp));
      CHKERRQ(KSPSetOperators(ksp,A,A));
      CHKERRQ(KSPSetFromOptions(ksp));
      CHKERRQ(KSPGetGuess(ksp,&guess));
      /* we aren't calling through the KSP so we call this ourselves */
      CHKERRQ(KSPGuessSetUp(guess));

      CHKERRQ(KSPGuessUpdate(guess,rhs,sol));
      CHKERRQ(KSPGuessFormGuess(guess,newrhs,newsol));
      CHKERRQ(VecView(newsol,PETSC_VIEWER_STDOUT_SELF));

      CHKERRQ(KSPDestroy(&ksp));
    }

    /* Test a singular projection matrix */
    {
      KSP      ksp;
      KSPGuess guess;

      CHKERRQ(KSPCreate(PETSC_COMM_SELF,&ksp));
      CHKERRQ(KSPSetOperators(ksp,A,A));
      CHKERRQ(KSPSetFromOptions(ksp));
      CHKERRQ(KSPGetGuess(ksp,&guess));
      CHKERRQ(KSPGuessSetUp(guess));

      for (i = 0; i < 15; ++i) {
        CHKERRQ(KSPGuessUpdate(guess,rhs,sol));
      }
      CHKERRQ(KSPGuessFormGuess(guess,newrhs,newsol));
      CHKERRQ(VecView(newsol,PETSC_VIEWER_STDOUT_SELF));

      CHKERRQ(KSPDestroy(&ksp));
    }
    CHKERRQ(VecDestroy(&newsol));
    CHKERRQ(VecDestroy(&newrhs));
    CHKERRQ(VecDestroy(&rhs));
    CHKERRQ(VecDestroy(&sol));

    CHKERRQ(MatDestroy(&A));
  }

  /* Test something triangular */
  {
    PetscInt triangle_size = 10;
    Mat      A;

    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,triangle_size,triangle_size,NULL,&A));
    for (i = 0; i < triangle_size; ++i) {
      CHKERRQ(MatSetValue(A,i,i,1.0,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

    {
      KSP         ksp;
      KSPGuess    guess;
      Vec         sol,rhs;
      PetscInt    j,indices[] = {0,1,2,3,4};
      PetscScalar values[] = {1.0,2.0,3.0,4.0,5.0};

      CHKERRQ(KSPCreate(PETSC_COMM_SELF,&ksp));
      CHKERRQ(KSPSetOperators(ksp,A,A));
      CHKERRQ(KSPSetFromOptions(ksp));
      CHKERRQ(KSPGetGuess(ksp,&guess));
      CHKERRQ(KSPGuessSetUp(guess));

      for (i = 0; i < 5; ++i) {
        CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,triangle_size,&sol));
        CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,triangle_size,&rhs));
        for (j = 0; j < i; ++j) {
          CHKERRQ(VecSetValue(sol,j,(PetscScalar)j,INSERT_VALUES));
          CHKERRQ(VecSetValue(rhs,j,(PetscScalar)j,INSERT_VALUES));
        }
        CHKERRQ(VecAssemblyBegin(sol));
        CHKERRQ(VecAssemblyBegin(rhs));
        CHKERRQ(VecAssemblyEnd(sol));
        CHKERRQ(VecAssemblyEnd(rhs));

        CHKERRQ(KSPGuessUpdate(guess,rhs,sol));

        CHKERRQ(VecDestroy(&rhs));
        CHKERRQ(VecDestroy(&sol));
      }

      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,triangle_size,&sol));
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,triangle_size,&rhs));
      CHKERRQ(VecSetValues(rhs,5,indices,values,INSERT_VALUES));
      CHKERRQ(VecAssemblyBegin(sol));
      CHKERRQ(VecAssemblyEnd(sol));

      CHKERRQ(KSPGuessFormGuess(guess,rhs,sol));
      CHKERRQ(VecView(sol,PETSC_VIEWER_STDOUT_SELF));

      CHKERRQ(VecDestroy(&rhs));
      CHKERRQ(VecDestroy(&sol));
      CHKERRQ(KSPDestroy(&ksp));
    }
    CHKERRQ(MatDestroy(&A));
  }
  CHKERRQ(PetscFinalize());
  return 0;
}

/* The relative tolerance here is strict enough to get rid of all the noise in both single and double precision: values as low as 5e-7 also work */

/*TEST

   test:
      args: -ksp_guess_type fischer -ksp_guess_fischer_model 3,10 -ksp_guess_fischer_monitor -ksp_guess_fischer_tol 1e-6

TEST*/
