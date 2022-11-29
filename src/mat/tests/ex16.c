static char help[] = "Tests MatDenseGetArray() and MatView()/MatLoad() with binary viewers.\n\n";

#include <petscmat.h>
#include <petscviewer.h>

static PetscErrorCode CheckValues(Mat A, PetscBool one)
{
  const PetscScalar *array;
  PetscInt           M, N, rstart, rend, lda, i, j;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(A, &array));
  PetscCall(MatDenseGetLDA(A, &lda));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    for (j = 0; j < N; j++) {
      PetscInt  ii = i - rstart, jj = j;
      PetscReal v = (PetscReal)(one ? 1 : (1 + i + j * M));
      PetscReal w = PetscRealPart(array[ii + jj * lda]);
      PetscCheck(PetscAbsReal(v - w) <= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Matrix entry (%" PetscInt_FMT ",%" PetscInt_FMT ") should be %g, got %g", i, j, (double)v, (double)w);
    }
  }
  PetscCall(MatDenseRestoreArrayRead(A, &array));
  PetscFunctionReturn(0);
}

#define CheckValuesIJ(A)  CheckValues(A, PETSC_FALSE)
#define CheckValuesOne(A) CheckValues(A, PETSC_TRUE)

int main(int argc, char **args)
{
  Mat          A;
  PetscInt     i, j, M = 4, N = 3, rstart, rend;
  PetscScalar *array;
  char         mattype[256];
  PetscViewer  view;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscStrcpy(mattype, MATMPIDENSE));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-mat_type", mattype, sizeof(mattype), NULL));
  /*
      Create a parallel dense matrix shared by all processors
  */
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &A));
  PetscCall(MatConvert(A, mattype, MAT_INPLACE_MATRIX, &A));
  /*
     Set values into the matrix
  */
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      PetscScalar v = (PetscReal)(1 + i + j * M);
      PetscCall(MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(A, 2.0));
  PetscCall(MatScale(A, 1.0 / 2.0));

  /*
      Store the binary matrix to a file
  */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_WRITE, &view));
  for (i = 0; i < 2; i++) {
    PetscCall(MatView(A, view));
    PetscCall(PetscViewerPushFormat(view, PETSC_VIEWER_NATIVE));
    PetscCall(MatView(A, view));
    PetscCall(PetscViewerPopFormat(view));
  }
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(MatDestroy(&A));

  /*
      Now reload the matrix and check its values
  */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_READ, &view));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, mattype));
  for (i = 0; i < 4; i++) {
    if (i > 0) PetscCall(MatZeroEntries(A));
    PetscCall(MatLoad(A, view));
    PetscCall(CheckValuesIJ(A));
  }
  PetscCall(PetscViewerDestroy(&view));

  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(PetscMalloc1((rend - rstart) * N, &array));
  for (i = 0; i < (rend - rstart) * N; i++) array[i] = (PetscReal)1;
  PetscCall(MatDensePlaceArray(A, array));
  PetscCall(MatScale(A, 2.0));
  PetscCall(MatScale(A, 1.0 / 2.0));
  PetscCall(CheckValuesOne(A));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_WRITE, &view));
  PetscCall(MatView(A, view));
  PetscCall(MatDenseResetArray(A));
  PetscCall(PetscFree(array));
  PetscCall(CheckValuesIJ(A));
  PetscCall(PetscViewerBinarySetSkipHeader(view, PETSC_TRUE));
  PetscCall(MatView(A, view));
  PetscCall(PetscViewerBinarySetSkipHeader(view, PETSC_FALSE));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(MatDestroy(&A));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, mattype));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_READ, &view));
  PetscCall(MatLoad(A, view));
  PetscCall(CheckValuesOne(A));
  PetscCall(MatZeroEntries(A));
  PetscCall(PetscViewerBinarySetSkipHeader(view, PETSC_TRUE));
  PetscCall(MatLoad(A, view));
  PetscCall(PetscViewerBinarySetSkipHeader(view, PETSC_FALSE));
  PetscCall(CheckValuesIJ(A));
  PetscCall(PetscViewerDestroy(&view));
  PetscCall(MatDestroy(&A));

  {
    PetscInt m = PETSC_DECIDE, n = PETSC_DECIDE;
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &m, &M));
    PetscCall(PetscSplitOwnership(PETSC_COMM_WORLD, &n, &N));
    /* TODO: MatCreateDense requires data!=NULL at all processes! */
    PetscCall(PetscMalloc1(m * N + 1, &array));

    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matrix.dat", FILE_MODE_READ, &view));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, n, M, N, array, &A));
    PetscCall(MatLoad(A, view));
    PetscCall(CheckValuesOne(A));
    PetscCall(PetscViewerBinarySetSkipHeader(view, PETSC_TRUE));
    PetscCall(MatLoad(A, view));
    PetscCall(PetscViewerBinarySetSkipHeader(view, PETSC_FALSE));
    PetscCall(CheckValuesIJ(A));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscViewerDestroy(&view));

    PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, n, M, N, array, &A));
    PetscCall(CheckValuesIJ(A));
    PetscCall(MatDestroy(&A));

    PetscCall(PetscFree(array));
  }

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      args: -viewer_binary_mpiio 0
      output_file: output/ex16.out
      test:
        suffix: stdio_1
        nsize: 1
        args: -mat_type seqdense
      test:
        suffix: stdio_2
        nsize: 2
      test:
        suffix: stdio_3
        nsize: 3
      test:
        suffix: stdio_4
        nsize: 4
      test:
        suffix: stdio_5
        nsize: 5
      test:
        requires: cuda
        args: -mat_type seqdensecuda
        suffix: stdio_cuda_1
        nsize: 1
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: stdio_cuda_2
        nsize: 2
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: stdio_cuda_3
        nsize: 3
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: stdio_cuda_4
        nsize: 4
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: stdio_cuda_5
        nsize: 5

   testset:
      requires: mpiio
      args: -viewer_binary_mpiio 1
      output_file: output/ex16.out
      test:
        suffix: mpiio_1
        nsize: 1
      test:
        suffix: mpiio_2
        nsize: 2
      test:
        suffix: mpiio_3
        nsize: 3
      test:
        suffix: mpiio_4
        nsize: 4
      test:
        suffix: mpiio_5
        nsize: 5
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_1
        nsize: 1
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_2
        nsize: 2
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_3
        nsize: 3
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_4
        nsize: 4
      test:
        requires: cuda
        args: -mat_type mpidensecuda
        suffix: mpiio_cuda_5
        nsize: 5

TEST*/
