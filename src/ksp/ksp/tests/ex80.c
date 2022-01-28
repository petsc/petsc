static char help[] = "Test the Fischer-3 initial guess routine.\n\n";

#include <petscksp.h>

#define SIZE 3

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt i;
  {
    Mat         A;
    PetscInt    indices[SIZE] = {0,1,2};
    PetscScalar values[SIZE] = {1.0,1.0,1.0};
    Vec         sol,rhs,newsol,newrhs;

    ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

    /* common data structures */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,SIZE,SIZE,NULL,&A);CHKERRQ(ierr);
    for (i = 0; i < SIZE; ++i) {
      ierr = MatSetValue(A,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = VecCreateSeq(PETSC_COMM_SELF,SIZE,&sol);CHKERRQ(ierr);
    ierr = VecDuplicate(sol,&rhs);CHKERRQ(ierr);
    ierr = VecDuplicate(sol,&newrhs);CHKERRQ(ierr);
    ierr = VecDuplicate(sol,&newsol);CHKERRQ(ierr);

    ierr = VecSetValues(sol,SIZE,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(rhs,SIZE - 1,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(newrhs,SIZE - 2,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(sol);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(newrhs);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(sol);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(newrhs);CHKERRQ(ierr);

    /* Test one vector */
    {
      KSP      ksp;
      KSPGuess guess;

      ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = KSPGetGuess(ksp,&guess);CHKERRQ(ierr);
      /* we aren't calling through the KSP so we call this ourselves */
      ierr = KSPGuessSetUp(guess);CHKERRQ(ierr);

      ierr = KSPGuessUpdate(guess,rhs,sol);CHKERRQ(ierr);
      ierr = KSPGuessFormGuess(guess,newrhs,newsol);CHKERRQ(ierr);
      ierr = VecView(newsol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

      ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    }

    /* Test a singular projection matrix */
    {
      KSP      ksp;
      KSPGuess guess;

      ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = KSPGetGuess(ksp,&guess);CHKERRQ(ierr);
      ierr = KSPGuessSetUp(guess);CHKERRQ(ierr);

      for (i = 0; i < 15; ++i) {
        ierr = KSPGuessUpdate(guess,rhs,sol);CHKERRQ(ierr);
      }
      ierr = KSPGuessFormGuess(guess,newrhs,newsol);CHKERRQ(ierr);
      ierr = VecView(newsol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

      ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&newsol);CHKERRQ(ierr);
    ierr = VecDestroy(&newrhs);CHKERRQ(ierr);
    ierr = VecDestroy(&rhs);CHKERRQ(ierr);
    ierr = VecDestroy(&sol);CHKERRQ(ierr);

    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }

  /* Test something triangular */
  {
    PetscInt triangle_size = 10;
    Mat      A;

    ierr = MatCreateSeqDense(PETSC_COMM_SELF,triangle_size,triangle_size,NULL,&A);CHKERRQ(ierr);
    for (i = 0; i < triangle_size; ++i) {
      ierr = MatSetValue(A,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    {
      KSP         ksp;
      KSPGuess    guess;
      Vec         sol,rhs;
      PetscInt    j,indices[] = {0,1,2,3,4};
      PetscScalar values[] = {1.0,2.0,3.0,4.0,5.0};

      ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = KSPGetGuess(ksp,&guess);CHKERRQ(ierr);
      ierr = KSPGuessSetUp(guess);CHKERRQ(ierr);

      for (i = 0; i < 5; ++i) {
        ierr = VecCreateSeq(PETSC_COMM_SELF,triangle_size,&sol);CHKERRQ(ierr);
        ierr = VecCreateSeq(PETSC_COMM_SELF,triangle_size,&rhs);CHKERRQ(ierr);
        for (j = 0; j < i; ++j) {
          ierr = VecSetValue(sol,j,(PetscScalar)j,INSERT_VALUES);CHKERRQ(ierr);
          ierr = VecSetValue(rhs,j,(PetscScalar)j,INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(sol);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(rhs);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(sol);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(rhs);CHKERRQ(ierr);

        ierr = KSPGuessUpdate(guess,rhs,sol);CHKERRQ(ierr);

        ierr = VecDestroy(&rhs);CHKERRQ(ierr);
        ierr = VecDestroy(&sol);CHKERRQ(ierr);
      }

      ierr = VecCreateSeq(PETSC_COMM_SELF,triangle_size,&sol);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,triangle_size,&rhs);CHKERRQ(ierr);
      ierr = VecSetValues(rhs,5,indices,values,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(sol);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(sol);CHKERRQ(ierr);

      ierr = KSPGuessFormGuess(guess,rhs,sol);CHKERRQ(ierr);
      ierr = VecView(sol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

      ierr = VecDestroy(&rhs);CHKERRQ(ierr);
      ierr = VecDestroy(&sol);CHKERRQ(ierr);
      ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/* The relative tolerance here is strict enough to get rid of all the noise in both single and double precision: values as low as 5e-7 also work */

/*TEST

   test:
      args: -ksp_guess_type fischer -ksp_guess_fischer_model 3,10 -ksp_guess_fischer_monitor -ksp_guess_fischer_tol 1e-6

TEST*/
