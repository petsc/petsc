
static char help[] = "Tests MatReorderForNonzeroDiagonal().\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat         mat, B, C;
  PetscInt    i, j;
  PetscMPIInt size;
  PetscScalar v;
  IS          isrow, iscol, identity;
  PetscViewer viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* ------- Assemble matrix, --------- */

  PetscCall(MatCreate(PETSC_COMM_WORLD, &mat));
  PetscCall(MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, 4, 4));
  PetscCall(MatSetFromOptions(mat));
  PetscCall(MatSetUp(mat));

  /* set anti-diagonal of matrix */
  v = 1.0;
  i = 0;
  j = 3;
  PetscCall(MatSetValues(mat, 1, &i, 1, &j, &v, INSERT_VALUES));
  v = 2.0;
  i = 1;
  j = 2;
  PetscCall(MatSetValues(mat, 1, &i, 1, &j, &v, INSERT_VALUES));
  v = 3.0;
  i = 2;
  j = 1;
  PetscCall(MatSetValues(mat, 1, &i, 1, &j, &v, INSERT_VALUES));
  v = 4.0;
  i = 3;
  j = 0;
  PetscCall(MatSetValues(mat, 1, &i, 1, &j, &v, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Original matrix\n"));
  PetscCall(MatView(mat, viewer));

  PetscCall(MatGetOrdering(mat, MATORDERINGNATURAL, &isrow, &iscol));

  PetscCall(MatPermute(mat, isrow, iscol, &B));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Original matrix permuted by identity\n"));
  PetscCall(MatView(B, viewer));
  PetscCall(MatDestroy(&B));

  PetscCall(MatReorderForNonzeroDiagonal(mat, 1.e-8, isrow, iscol));
  PetscCall(MatPermute(mat, isrow, iscol, &B));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Original matrix permuted by identity + NonzeroDiagonal()\n"));
  PetscCall(MatView(B, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Row permutation\n"));
  PetscCall(ISView(isrow, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Column permutation\n"));
  PetscCall(ISView(iscol, viewer));
  PetscCall(MatDestroy(&B));

  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));

  PetscCall(MatGetOrdering(mat, MATORDERINGND, &isrow, &iscol));
  PetscCall(MatPermute(mat, isrow, iscol, &B));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Original matrix permuted by ND\n"));
  PetscCall(MatView(B, viewer));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ND row permutation\n"));
  PetscCall(ISView(isrow, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ND column permutation\n"));
  PetscCall(ISView(iscol, viewer));

  PetscCall(MatReorderForNonzeroDiagonal(mat, 1.e-8, isrow, iscol));
  PetscCall(MatPermute(mat, isrow, iscol, &B));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Original matrix permuted by ND + NonzeroDiagonal()\n"));
  PetscCall(MatView(B, viewer));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ND + NonzeroDiagonal() row permutation\n"));
  PetscCall(ISView(isrow, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ND + NonzeroDiagonal() column permutation\n"));
  PetscCall(ISView(iscol, viewer));

  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));

  PetscCall(MatGetOrdering(mat, MATORDERINGRCM, &isrow, &iscol));
  PetscCall(MatPermute(mat, isrow, iscol, &B));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Original matrix permuted by RCM\n"));
  PetscCall(MatView(B, viewer));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscViewerASCIIPrintf(viewer, "RCM row permutation\n"));
  PetscCall(ISView(isrow, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "RCM column permutation\n"));
  PetscCall(ISView(iscol, viewer));

  PetscCall(MatReorderForNonzeroDiagonal(mat, 1.e-8, isrow, iscol));
  PetscCall(MatPermute(mat, isrow, iscol, &B));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Original matrix permuted by RCM + NonzeroDiagonal()\n"));
  PetscCall(MatView(B, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "RCM + NonzeroDiagonal() row permutation\n"));
  PetscCall(ISView(isrow, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "RCM + NonzeroDiagonal() column permutation\n"));
  PetscCall(ISView(iscol, viewer));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  if (size == 1) {
    PetscCall(MatSetOption(B, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 4, 0, 1, &identity));
    PetscCall(MatPermute(B, identity, identity, &C));
    PetscCall(MatConvert(C, MATSEQSBAIJ, MAT_INPLACE_MATRIX, &C));
    PetscCall(MatDestroy(&C));
    PetscCall(ISDestroy(&identity));
  }
  PetscCall(MatDestroy(&B));
  /* Test MatLUFactor(); set diagonal as zeros as requested by PETSc matrix factorization */
  for (i = 0; i < 4; i++) {
    v = 0.0;
    PetscCall(MatSetValues(mat, 1, &i, 1, &i, &v, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatLUFactor(mat, isrow, iscol, NULL));

  /* Free data structures */
  PetscCall(ISDestroy(&isrow));
  PetscCall(ISDestroy(&iscol));
  PetscCall(MatDestroy(&mat));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
