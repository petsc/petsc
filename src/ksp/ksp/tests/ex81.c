static char help[] = "Test different KSP and Mat prefixes.\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
  KSP ksp;
  PC  pc;
  Mat A, B, C;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, 2, 2, PETSC_DECIDE, PETSC_DECIDE, 1.0, &A));
  PetscCall(MatConvert(A, MATAIJ, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &C));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(MatSetOptionsPrefix(A, "alpha_"));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetOptionsPrefix(ksp, "beta_"));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetUp(pc));
  PetscCall(PCSetUpOnBlocks(pc));
  PetscCall(PCView(pc, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatSetOptionsPrefix(C, "gamma_"));
  PetscCall(KSPSetOperators(ksp, C, C));
  PetscCall(PCSetUp(pc));
  PetscCall(PCSetUpOnBlocks(pc));
  PetscCall(PCView(pc, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      requires: mumps
      args: -beta_pc_type lu -beta_pc_factor_mat_solver_type mumps -beta_mat_mumps_icntl_14 30

   test:
      nsize: 2
      suffix: 2
      requires: mumps
      args: -beta_pc_type asm -beta_sub_pc_factor_mat_solver_type mumps -beta_sub_mat_mumps_icntl_14 30 -beta_sub_pc_type lu

TEST*/
