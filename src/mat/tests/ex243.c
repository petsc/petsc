static char help[] = "Test conversion of ScaLAPACK matrices.\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat             A, A_scalapack;
  PetscInt        i, j, M = 10, N = 5, nloc, mloc, nrows, ncols;
  PetscMPIInt     rank, size;
  IS              isrows, iscols;
  const PetscInt *rows, *cols;
  PetscScalar    *v;
  MatType         type;
  PetscBool       isDense, isAIJ, flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));

  /* Create a matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  mloc = PETSC_DECIDE;
  PetscCall(PetscSplitOwnershipEqual(PETSC_COMM_WORLD, &mloc, &M));
  nloc = PETSC_DECIDE;
  PetscCall(PetscSplitOwnershipEqual(PETSC_COMM_WORLD, &nloc, &N));
  PetscCall(MatSetSizes(A, mloc, nloc, M, N));
  PetscCall(MatSetType(A, MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /* Set local matrix entries */
  PetscCall(MatGetOwnershipIS(A, &isrows, &iscols));
  PetscCall(ISGetLocalSize(isrows, &nrows));
  PetscCall(ISGetIndices(isrows, &rows));
  PetscCall(ISGetLocalSize(iscols, &ncols));
  PetscCall(ISGetIndices(iscols, &cols));
  PetscCall(PetscMalloc1(nrows * ncols, &v));

  for (i = 0; i < nrows; i++) {
    for (j = 0; j < ncols; j++) {
      if (size == 1) {
        v[i * ncols + j] = (PetscScalar)(i + j);
      } else {
        v[i * ncols + j] = (PetscScalar)rank + j * 0.1;
      }
    }
  }
  PetscCall(MatSetValues(A, nrows, rows, ncols, cols, v, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* Test MatSetValues() by converting A to A_scalapack */
  PetscCall(MatGetType(A, &type));
  if (size == 1) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isDense));
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &isAIJ));
  } else {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIDENSE, &isDense));
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &isAIJ));
  }

  if (isDense || isAIJ) {
    Mat Aexplicit;
    PetscCall(MatConvert(A, MATSCALAPACK, MAT_INITIAL_MATRIX, &A_scalapack));
    PetscCall(MatComputeOperator(A_scalapack, isAIJ ? MATAIJ : MATDENSE, &Aexplicit));
    PetscCall(MatMultEqual(Aexplicit, A_scalapack, 5, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Aexplicit != A_scalapack.");
    PetscCall(MatDestroy(&Aexplicit));

    /* Test MAT_REUSE_MATRIX which is only supported for inplace conversion */
    PetscCall(MatConvert(A, MATSCALAPACK, MAT_INPLACE_MATRIX, &A));
    PetscCall(MatMultEqual(A_scalapack, A, 5, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "A_scalapack != A.");
    PetscCall(MatDestroy(&A_scalapack));
  }

  PetscCall(ISRestoreIndices(isrows, &rows));
  PetscCall(ISRestoreIndices(iscols, &cols));
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&iscols));
  PetscCall(PetscFree(v));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: scalapack

   test:
      nsize: 6

   test:
      suffix: 2
      nsize: 6
      args: -mat_type aij
      output_file: output/ex243_1.out

   test:
      suffix: 3
      nsize: 6
      args: -mat_type scalapack
      output_file: output/ex243_1.out

TEST*/
