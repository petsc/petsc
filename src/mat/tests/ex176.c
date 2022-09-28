
static char help[] = "Tests MatCreateMPIAIJWithArrays() abd MatUpdateMPIAIJWithArray()\n";

#include <petscmat.h>

/*
 * This is an extremely simple example to test MatUpdateMPIAIJWithArrays()
 *
 * A =

   1    2   0   3  0  0
   0    4   5   0  0  6
   7    0   8   0  9  0
   0   10  11  12  0  13
   0   14  15   0  0  16
  17    0   0   0  0  18
 *
 * */

int main(int argc, char **argv)
{
  Mat      A, B;
  PetscInt i[3][3] = {
    {0, 3, 6},
    {0, 3, 7},
    {0, 3, 5}
  };
  PetscInt j[3][7] = {
    {0, 1, 3, 1, 2, 5,  -1},
    {0, 2, 4, 1, 2, 3,  5 },
    {1, 2, 5, 0, 5, -1, -1}
  };
  PetscScalar a[3][7] = {
    {1,  2,  3,  4,  5,  6,  -1},
    {7,  8,  9,  10, 11, 12, 13},
    {14, 15, 16, 17, 18, -1, -1}
  };
  PetscScalar anew[3][7] = {
    {10,  20,  30,  40,  50,  60,  -1 },
    {70,  80,  90,  100, 110, 120, 130},
    {140, 150, 160, 170, 180, -1,  -1 }
  };
  MPI_Comm    comm;
  PetscMPIInt rank;
  PetscBool   equal = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(MatCreateMPIAIJWithArrays(comm, 2, PETSC_DETERMINE, PETSC_DETERMINE, 6, i[rank], j[rank], a[rank], &B));

  PetscCall(MatCreateMPIAIJWithArrays(comm, 2, PETSC_DETERMINE, PETSC_DETERMINE, 6, i[rank], j[rank], a[rank], &A));
  PetscCall(MatSetFromOptions(A)); /* might change A's type */

  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "wrong results");

  PetscCall(MatUpdateMPIAIJWithArray(A, anew[rank]));
  PetscCall(MatUpdateMPIAIJWithArray(B, anew[rank]));
  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "wrong results");

  PetscCall(MatUpdateMPIAIJWithArray(A, a[rank]));
  PetscCall(MatUpdateMPIAIJWithArray(B, a[rank]));
  PetscCall(MatEqual(A, B, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "wrong results");

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   testset:
     nsize: {{1 3}}
     output_file: output/empty.out

     test:
       suffix: aij

     test:
       requires: cuda
       suffix: cuda
       # since the matrices are created with MatCreateMPIxxx(), users are allowed to pass 'mpiaijcusparse' even with one rank
       args: -mat_type {{aijcusparse mpiaijcusparse}}

     test:
       requires: kokkos_kernels
       suffix: kok
       args: -mat_type aijkokkos
TEST*/
