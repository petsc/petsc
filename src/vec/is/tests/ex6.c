static char help[] = "Tests ISRenumber.\n\n";

#include <petscis.h>

PetscErrorCode TestRenumber(IS is, IS mult)
{
  IS       nis;
  PetscInt N;

  PetscFunctionBegin;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)is), "\n-----------------\n"));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)is), "\nInitial\n"));
  PetscCall(ISView(is, NULL));
  if (mult) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)is), "\nMult\n"));
    PetscCall(ISView(mult, NULL));
  }
  PetscCall(ISRenumber(is, mult, &N, NULL));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)is), "\nRenumbered, unique entries %" PetscInt_FMT "\n", N));
  PetscCall(ISRenumber(is, mult, NULL, &nis));
  PetscCall(ISView(nis, NULL));
  PetscCall(ISDestroy(&nis));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  IS          is;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  for (PetscInt c = 0; c < 3; c++) {
    IS mult = NULL;

    PetscCall(ISCreateStride(PETSC_COMM_WORLD, 0, 0, 0, &is));
    if (c) {
      PetscInt n;
      PetscCall(ISGetLocalSize(is, &n));
      PetscCall(ISCreateStride(PETSC_COMM_WORLD, n, c - 2, 0, &mult));
    }
    PetscCall(TestRenumber(is, mult));
    PetscCall(ISDestroy(&is));
    PetscCall(ISDestroy(&mult));

    PetscCall(ISCreateStride(PETSC_COMM_WORLD, 2, -rank - 1, -4, &is));
    if (c) {
      PetscInt n;
      PetscCall(ISGetLocalSize(is, &n));
      PetscCall(ISCreateStride(PETSC_COMM_WORLD, n, c - 2, 0, &mult));
    }
    PetscCall(TestRenumber(is, mult));
    PetscCall(ISDestroy(&is));
    PetscCall(ISDestroy(&mult));

    PetscCall(ISCreateStride(PETSC_COMM_WORLD, 10, 4 + rank, 2, &is));
    if (c) {
      PetscInt n;
      PetscCall(ISGetLocalSize(is, &n));
      PetscCall(ISCreateStride(PETSC_COMM_WORLD, n, c - 2, 1, &mult));
    }
    PetscCall(TestRenumber(is, mult));
    PetscCall(ISDestroy(&is));
    PetscCall(ISDestroy(&mult));

    PetscCall(ISCreateStride(PETSC_COMM_WORLD, 10, -rank - 1, 2, &is));
    if (c) {
      PetscInt n;
      PetscCall(ISGetLocalSize(is, &n));
      PetscCall(ISCreateStride(PETSC_COMM_WORLD, n, c - 2, 1, &mult));
    }
    PetscCall(TestRenumber(is, mult));
    PetscCall(ISDestroy(&is));
    PetscCall(ISDestroy(&mult));
  }
  /* Finalize */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    nsize: {{1 2}separate output}

TEST*/
