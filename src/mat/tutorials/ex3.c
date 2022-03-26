static char help[] = "Illustration of MatIS using a 1D Laplacian assembly\n\n";

/*
  MatIS means that the matrix is not assembled. The easiest way to think of this (for me) is that processes do not have
  to hold full matrix rows. One process can hold part of row i, and another processes can hold another part. However, there
  are still the same number of global rows. The local size here is not the size of the local IS block, which we call the
  overlap size, since that is a property only of MatIS. It is the size of the local piece of the vector you multiply in
  MatMult(). This allows PETSc to understand the parallel layout of the Vec, and how it matches the Mat. If you only know
  the overlap size when assembling, it is best to use PETSC_DECIDE for the local size in the creation routine, so that PETSc
  automatically partitions the unknowns.

  Each P_1 element matrix for a cell will be

    /  1 -1 \
    \ -1  1 /

  so that the assembled matrix has a tridiagonal [-1, 2, -1] pattern. We will use 1 cell per process for illustration,
  and allow PETSc to decide the ownership.
*/

#include <petscmat.h>

int main(int argc, char **argv) {
  MPI_Comm               comm;
  Mat                    A;
  Vec                    x, y;
  ISLocalToGlobalMapping map;
  PetscScalar            elemMat[4] = {1.0, -1.0, -1.0, 1.0};
  PetscReal              error;
  PetscInt               overlapSize = 2, globalIdx[2];
  PetscMPIInt            rank, size;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  /* Create local-to-global map */
  globalIdx[0] = rank;
  globalIdx[1] = rank+1;
  PetscCall(ISLocalToGlobalMappingCreate(comm, 1, overlapSize, globalIdx, PETSC_COPY_VALUES, &map));
  /* Create matrix */
  PetscCall(MatCreateIS(comm, 1, PETSC_DECIDE, PETSC_DECIDE, size+1, size+1, map, map, &A));
  PetscCall(PetscObjectSetName((PetscObject) A, "A"));
  PetscCall(ISLocalToGlobalMappingDestroy(&map));
  PetscCall(MatISSetPreallocation(A, overlapSize, NULL, overlapSize, NULL));
  PetscCall(MatSetValues(A, 2, globalIdx, 2, globalIdx, elemMat, ADD_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  /* Check that the constant vector is in the nullspace */
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecSet(x, 1.0));
  PetscCall(PetscObjectSetName((PetscObject) x, "x"));
  PetscCall(VecViewFromOptions(x, NULL, "-x_view"));
  PetscCall(MatMult(A, x, y));
  PetscCall(PetscObjectSetName((PetscObject) y, "y"));
  PetscCall(VecViewFromOptions(y, NULL, "-y_view"));
  PetscCall(VecNorm(y, NORM_2, &error));
  PetscCheckFalse(error > PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Invalid output, x should be in the nullspace of A");
  /* Check that an interior unit vector gets mapped to something of 1-norm 4 */
  if (size > 1) {
    PetscCall(VecSet(x, 0.0));
    PetscCall(VecSetValue(x, 1, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
    PetscCall(MatMult(A, x, y));
    PetscCall(VecNorm(y, NORM_1, &error));
    PetscCheckFalse(PetscAbsReal(error - 4) > PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Invalid output for matrix multiply");
  }
  /* Cleanup */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires:
    args:

  test:
    suffix: 1
    nsize: 3
    args:

  test:
    suffix: 2
    nsize: 7
    args:

TEST*/
