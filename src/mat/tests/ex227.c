static char help[] = "Test MatNullSpaceTest() with options prefixes.\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat mat;
  MatNullSpace nsp;
  PetscBool prefix = PETSC_FALSE, flg;
  PetscErrorCode ierr;
  PetscInt zero = 0;
  PetscScalar value = 0;
  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-with_prefix",&prefix,NULL));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD, 1, 1, 1, 1, NULL, &mat));
  CHKERRQ(MatSetOptionsPrefix(mat, prefix ? "prefix_" : NULL));
  CHKERRQ(MatSetUp(mat));
  CHKERRQ(MatSetValues(mat, 1, &zero, 1, &zero, &value, INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nsp));
  CHKERRQ(MatNullSpaceTest(nsp, mat, &flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Null space test failed!");
  CHKERRQ(MatNullSpaceDestroy(&nsp));
  CHKERRQ(MatDestroy(&mat));
  ierr = PetscFinalize();
  return ierr;
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
