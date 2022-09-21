
static char help[] = "Creates a matrix, inserts some values, and tests MatCreateSubMatrices() and MatZeroEntries().\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         mat, submat, submat1, *submatrices;
  PetscInt    m = 10, n = 10, i = 4, tmp, rstart, rend;
  IS          irow, icol;
  PetscScalar value = 1.0;
  PetscViewer sviewer;
  PetscBool   allA = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_COMMON));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF, PETSC_VIEWER_ASCII_COMMON));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &mat));
  PetscCall(MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetFromOptions(mat));
  PetscCall(MatSetUp(mat));
  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    value = (PetscReal)i + 1;
    tmp   = i % 5;
    PetscCall(MatSetValues(mat, 1, &tmp, 1, &i, &value, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Original matrix\n"));
  PetscCall(MatView(mat, PETSC_VIEWER_STDOUT_WORLD));

  /* Test MatCreateSubMatrix_XXX_All(), i.e., submatrix = A */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_all", &allA, NULL));
  if (allA) {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, m, 0, 1, &irow));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &icol));
    PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_INITIAL_MATRIX, &submatrices));
    PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_REUSE_MATRIX, &submatrices));
    submat = *submatrices;

    /* sviewer will cause the submatrices (one per processor) to be printed in the correct order */
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "\nSubmatrices with all\n"));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "--------------------\n"));
    PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
    PetscCall(MatView(submat, sviewer));
    PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
    PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(ISDestroy(&irow));
    PetscCall(ISDestroy(&icol));

    /* test getting a reference on a submat */
    PetscCall(PetscObjectReference((PetscObject)submat));
    PetscCall(MatDestroySubMatrices(1, &submatrices));
    PetscCall(MatDestroy(&submat));
  }

  /* Form submatrix with rows 2-4 and columns 4-8 */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 3, 2, 1, &irow));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 5, 4, 1, &icol));
  PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_INITIAL_MATRIX, &submatrices));
  submat = *submatrices;

  /* Test reuse submatrices */
  PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_REUSE_MATRIX, &submatrices));

  /* sviewer will cause the submatrices (one per processor) to be printed in the correct order */
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "\nSubmatrices\n"));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
  PetscCall(MatView(submat, sviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscObjectReference((PetscObject)submat));
  PetscCall(MatDestroySubMatrices(1, &submatrices));
  PetscCall(MatDestroy(&submat));

  /* Form submatrix with rows 2-4 and all columns */
  PetscCall(ISDestroy(&icol));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 10, 0, 1, &icol));
  PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_INITIAL_MATRIX, &submatrices));
  PetscCall(MatCreateSubMatrices(mat, 1, &irow, &icol, MAT_REUSE_MATRIX, &submatrices));
  submat = *submatrices;

  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "\nSubmatrices with allcolumns\n"));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
  PetscCall(MatView(submat, sviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sviewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));

  /* Test MatDuplicate */
  PetscCall(MatDuplicate(submat, MAT_COPY_VALUES, &submat1));
  PetscCall(MatDestroy(&submat1));

  /* Zero the original matrix */
  PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Original zeroed matrix\n"));
  PetscCall(MatZeroEntries(mat));
  PetscCall(MatView(mat, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(ISDestroy(&irow));
  PetscCall(ISDestroy(&icol));
  PetscCall(PetscObjectReference((PetscObject)submat));
  PetscCall(MatDestroySubMatrices(1, &submatrices));
  PetscCall(MatDestroy(&submat));
  PetscCall(MatDestroy(&mat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_type aij

   test:
      suffix: 2
      args: -mat_type dense

   test:
      suffix: 3
      nsize: 3
      args: -mat_type aij

   test:
      suffix: 4
      nsize: 3
      args: -mat_type dense

   test:
      suffix: 5
      nsize: 3
      args: -mat_type aij -test_all

TEST*/
