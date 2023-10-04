static char help[] = "Tests that MatView() and MatLoad() work for MPIAIJ matrix with total nz > PETSC_MAX_INT\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  PetscInt     n = 1000000, nzr = (PetscInt)((double)PETSC_MAX_INT) / (3.8 * n);
  Mat          A;
  PetscScalar *a;
  PetscInt    *ii, *jd, *jo;
  PetscMPIInt  rank, size;
  PetscViewer  viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with two MPI ranks");
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscMalloc4(n + 1, &ii, n * nzr, &jd, n * nzr, &jo, n * nzr, &a));
  ii[0] = 0;
  for (PetscInt i = 0; i < n; i++) {
    ii[i + 1] = ii[i] + nzr;
    for (PetscInt j = 0; j < nzr; j++) jd[ii[i] + j] = j;
    if (rank == 0) {
      for (PetscInt j = 0; j < nzr; j++) jo[ii[i] + j] = n + j - 1;
    } else {
      for (PetscInt j = 0; j < nzr; j++) jo[ii[i] + j] = j;
    }
  }
  PetscCall(MatCreateMPIAIJWithSplitArrays(PETSC_COMM_WORLD, n, n, PETSC_DETERMINE, PETSC_DETERMINE, ii, jd, a, ii, jo, a, &A));
  PetscCall(MatView(A, PETSC_VIEWER_BINARY_WORLD));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree4(ii, jd, jo, a));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "binaryoutput", FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      TODO: requires to much memory to run in the CI
      nsize: 2

TEST*/
