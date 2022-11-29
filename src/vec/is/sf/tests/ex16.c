static char help[] = "Test PetscSFCreateByMatchingIndices\n\n";

#include <petsc.h>
#include <petscsf.h>

/* Test PetscSFCreateByMatchingIndices.

testnum 0:

  rank             : 0            1            2
  numRootIndices   : 3            1            1
  rootIndices      : [1 0 2]      [3]          [3]
  rootLocalOffset  : 100          200          300
  layout           : [0 1]        [2]          [3]
  numLeafIndices   : 1            1            2
  leafIndices      : [0]          [2]          [0 3]
  leafLocalOffset  : 400          500          600

would build the following SF:

  [0] 400 <- (0,101)
  [1] 500 <- (0,102)
  [2] 600 <- (0,101)
  [2] 601 <- (2,300)

testnum 1:

  rank             : 0               1               2
  numRootIndices   : 3               1               1
  rootIndices      : [1 0 2]         [3]             [3]
  rootLocalOffset  : 100             200             300
  layout           : [0 1]           [2]             [3]
  numLeafIndices   : numRootIndices  numRootIndices  numRootIndices
  leafIndices      : rootIndices     rootIndices     rootIndices
  leafLocalOffset  : rootLocalOffset rootLocalOffset rootLocalOffset

would build the following SF:

  [1] 200 <- (2,300)

testnum 2:

  No one claims ownership of global index 1, but no one needs it.

  rank             : 0            1            2
  numRootIndices   : 2            1            1
  rootIndices      : [0 2]        [3]          [3]
  rootLocalOffset  : 100          200          300
  layout           : [0 1]        [2]          [3]
  numLeafIndices   : 1            1            2
  leafIndices      : [0]          [2]          [0 3]
  leafLocalOffset  : 400          500          600

would build the following SF:

  [0] 400 <- (0,100)
  [1] 500 <- (0,101)
  [2] 600 <- (0,100)
  [2] 601 <- (2,300)

*/

int main(int argc, char **argv)
{
  PetscSF     sf;
  PetscLayout layout;
  PetscInt    N, n;
  PetscInt    nA = -1, *A, offsetA = -1;
  PetscInt    nB = -1, *B, offsetB = -1;
  PetscMPIInt size, rank;
  PetscInt    testnum;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-testnum", &testnum, NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 3, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 3 MPI processes");

  switch (testnum) {
  case 0:
    N = 4;
    n = PETSC_DECIDE;
    switch (rank) {
    case 0:
      nA      = 3;
      offsetA = 100;
      nB      = 1;
      offsetB = 400;
      break;
    case 1:
      nA      = 1;
      offsetA = 200;
      nB      = 1;
      offsetB = 500;
      break;
    case 2:
      nA      = 1;
      offsetA = 300;
      nB      = 2;
      offsetB = 600;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 3 MPI processes");
    }
    PetscCall(PetscMalloc1(nA, &A));
    PetscCall(PetscMalloc1(nB, &B));
    switch (rank) {
    case 0:
      A[0] = 1;
      A[1] = 0;
      A[2] = 2;
      B[0] = 0;
      break;
    case 1:
      A[0] = 3;
      B[0] = 2;
      break;
    case 2:
      A[0] = 3;
      B[0] = 0;
      B[1] = 3;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 3 MPI processes");
    }
    break;
  case 1:
    N = 4;
    n = PETSC_DECIDE;
    switch (rank) {
    case 0:
      nA      = 3;
      offsetA = 100;
      break;
    case 1:
      nA      = 1;
      offsetA = 200;
      break;
    case 2:
      nA      = 1;
      offsetA = 300;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 3 MPI processes");
    }
    PetscCall(PetscMalloc1(nA, &A));
    switch (rank) {
    case 0:
      A[0] = 1;
      A[1] = 0;
      A[2] = 2;
      break;
    case 1:
      A[0] = 3;
      break;
    case 2:
      A[0] = 3;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 3 MPI processes");
    }
    nB      = nA;
    B       = A;
    offsetB = offsetA;
    break;
  case 2:
    N = 4;
    n = PETSC_DECIDE;
    switch (rank) {
    case 0:
      nA      = 2;
      offsetA = 100;
      nB      = 1;
      offsetB = 400;
      break;
    case 1:
      nA      = 1;
      offsetA = 200;
      nB      = 1;
      offsetB = 500;
      break;
    case 2:
      nA      = 1;
      offsetA = 300;
      nB      = 2;
      offsetB = 600;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 3 MPI processes");
    }
    PetscCall(PetscMalloc1(nA, &A));
    PetscCall(PetscMalloc1(nB, &B));
    switch (rank) {
    case 0:
      A[0] = 0;
      A[1] = 2;
      B[0] = 0;
      break;
    case 1:
      A[0] = 3;
      B[0] = 2;
      break;
    case 2:
      A[0] = 3;
      B[0] = 0;
      B[1] = 3;
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 3 MPI processes");
    }
    break;
  }
  PetscCall(PetscLayoutCreate(PETSC_COMM_WORLD, &layout));
  PetscCall(PetscLayoutSetSize(layout, N));
  PetscCall(PetscLayoutSetLocalSize(layout, n));
  PetscCall(PetscLayoutSetBlockSize(layout, 1));
  PetscCall(PetscSFCreateByMatchingIndices(layout, nA, A, NULL, offsetA, nB, B, NULL, offsetB, NULL, &sf));
  PetscCall(PetscLayoutDestroy(&layout));
  PetscCall(PetscFree(A));
  if (testnum != 1) PetscCall(PetscFree(B));
  PetscCall(PetscObjectSetName((PetscObject)sf, "sf"));
  PetscCall(PetscSFView(sf, NULL));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: 3
    args: -testnum 0

  test:
    suffix: 1
    nsize: 3
    args: -testnum 1

  test:
    suffix: 2
    nsize: 3
    args: -testnum 2

TEST*/
