
static char help[] = "Tests MatGetColumnNorms()/Sums()/Means() for matrix read from file.";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat          A;
  PetscReal   *reductions_real;
  PetscScalar *reductions_scalar;
  char         file[PETSC_MAX_PATH_LEN];
  PetscBool    flg;
  PetscViewer  fd;
  PetscInt     n;
  PetscMPIInt  rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate binary file with the -f option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));

  PetscCall(MatGetSize(A, NULL, &n));
  PetscCall(PetscMalloc1(n, &reductions_real));
  PetscCall(PetscMalloc1(n, &reductions_scalar));

  PetscCall(MatGetColumnNorms(A, NORM_2, reductions_real));
  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "NORM_2:\n"));
    PetscCall(PetscRealView(n, reductions_real, PETSC_VIEWER_STDOUT_SELF));
  }

  PetscCall(MatGetColumnNorms(A, NORM_1, reductions_real));
  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "NORM_1:\n"));
    PetscCall(PetscRealView(n, reductions_real, PETSC_VIEWER_STDOUT_SELF));
  }

  PetscCall(MatGetColumnNorms(A, NORM_INFINITY, reductions_real));
  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "NORM_INFINITY:\n"));
    PetscCall(PetscRealView(n, reductions_real, PETSC_VIEWER_STDOUT_SELF));
  }

  PetscCall(MatGetColumnSums(A, reductions_scalar));
  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "REDUCTION_SUM:\n"));
    PetscCall(PetscScalarView(n, reductions_scalar, PETSC_VIEWER_STDOUT_SELF));
  }

  PetscCall(MatGetColumnMeans(A, reductions_scalar));
  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "REDUCTION_MEAN:\n"));
    PetscCall(PetscScalarView(n, reductions_scalar, PETSC_VIEWER_STDOUT_SELF));
  }

  PetscCall(PetscFree(reductions_real));
  PetscCall(PetscFree(reductions_scalar));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type aij
      output_file: output/ex138.out

   test:
      suffix: 2
      nsize: {{1 2}}
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/small -mat_type baij -matload_block_size {{2 3}}
      output_file: output/ex138.out

   test:
      suffix: complex
      nsize: 2
      requires: datafilespath complex double !defined(PETSC_USE_64BIT_INDICES)
      args: -f ${DATAFILESPATH}/matrices/nimrod/small_112905 -mat_type aij
      output_file: output/ex138_complex.out
      filter: grep -E "\ 0:|1340:|1344:"

TEST*/
