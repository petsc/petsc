static char help[] = "Test MatNullSpaceTest() with options prefixes.\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat          mat;
  MatNullSpace nsp;
  PetscBool    prefix = PETSC_FALSE, flg;
  PetscInt     zero   = 0;
  PetscScalar  value  = 0;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-with_prefix", &prefix, NULL));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, 1, 1, 1, 1, NULL, &mat));
  PetscCall(MatSetOptionsPrefix(mat, prefix ? "prefix_" : NULL));
  PetscCall(MatSetUp(mat));
  PetscCall(MatSetValues(mat, 1, &zero, 1, &zero, &value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nsp));
  PetscCall(MatNullSpaceTest(nsp, mat, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Null space test failed!");
  PetscCall(MatNullSpaceDestroy(&nsp));
  PetscCall(MatDestroy(&mat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
       suffix: no_prefix
       output_file: output/ex227_no_prefix.out
       args: -mat_null_space_test_view -mat_view

   test:
       suffix: prefix
       output_file: output/ex227_prefix.out
       args: -prefix_mat_null_space_test_view -with_prefix -prefix_mat_view

TEST*/
