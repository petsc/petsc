static char help[] = "Test the Fischer-1 initial guess routine with VECNEST.\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
  /* Test that exceeding the number of stored vectors works correctly - this used to not work with VecNest */
  PetscInt    triangle_size = 10;
  Mat         A, A_nest;
  KSP         ksp;
  KSPGuess    guess;
  Vec         sol, rhs, sol_nest, rhs_nest;
  PetscInt    i, j, indices[] = {0, 1, 2, 3, 4};
  PetscScalar values[] = {1.0, 2.0, 3.0, 4.0, 5.0};

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, triangle_size, triangle_size, NULL, &A));
  for (i = 0; i < triangle_size; ++i) PetscCall(MatSetValue(A, i, i, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateNest(PETSC_COMM_SELF, 1, NULL, 1, NULL, &A, &A_nest));
  PetscCall(MatNestSetVecType(A_nest, VECNEST));

  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCall(KSPSetOperators(ksp, A_nest, A_nest));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetGuess(ksp, &guess));
  PetscCall(KSPGuessSetUp(guess));

  for (i = 0; i < 5; ++i) {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, triangle_size, &sol));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, triangle_size, &rhs));
    for (j = 0; j < i; ++j) {
      PetscCall(VecSetValue(sol, j, (PetscScalar)j, INSERT_VALUES));
      PetscCall(VecSetValue(rhs, j, (PetscScalar)j, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(sol));
    PetscCall(VecAssemblyBegin(rhs));
    PetscCall(VecAssemblyEnd(sol));
    PetscCall(VecAssemblyEnd(rhs));

    PetscCall(VecCreateNest(PETSC_COMM_SELF, 1, NULL, &sol, &sol_nest));
    PetscCall(VecCreateNest(PETSC_COMM_SELF, 1, NULL, &rhs, &rhs_nest));

    PetscCall(KSPGuessUpdate(guess, rhs_nest, sol_nest));

    PetscCall(VecDestroy(&rhs_nest));
    PetscCall(VecDestroy(&sol_nest));
    PetscCall(VecDestroy(&rhs));
    PetscCall(VecDestroy(&sol));
  }

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, triangle_size, &sol));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, triangle_size, &rhs));
  PetscCall(VecSetValues(rhs, 5, indices, values, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(rhs));
  PetscCall(VecAssemblyEnd(rhs));

  PetscCall(VecCreateNest(PETSC_COMM_SELF, 1, NULL, &sol, &sol_nest));
  PetscCall(VecCreateNest(PETSC_COMM_SELF, 1, NULL, &rhs, &rhs_nest));

  PetscCall(KSPGuessFormGuess(guess, rhs_nest, sol_nest));
  PetscCall(VecView(sol_nest, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(VecDestroy(&rhs_nest));
  PetscCall(VecDestroy(&sol_nest));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&sol));

  PetscCall(KSPDestroy(&ksp));

  PetscCall(MatDestroy(&A_nest));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_guess_type fischer -ksp_guess_fischer_model 1,3 -ksp_guess_fischer_monitor

TEST*/
