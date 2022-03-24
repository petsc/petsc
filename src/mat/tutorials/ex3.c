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

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  /* Create local-to-global map */
  globalIdx[0] = rank;
  globalIdx[1] = rank+1;
  CHKERRQ(ISLocalToGlobalMappingCreate(comm, 1, overlapSize, globalIdx, PETSC_COPY_VALUES, &map));
  /* Create matrix */
  CHKERRQ(MatCreateIS(comm, 1, PETSC_DECIDE, PETSC_DECIDE, size+1, size+1, map, map, &A));
  CHKERRQ(PetscObjectSetName((PetscObject) A, "A"));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&map));
  CHKERRQ(MatISSetPreallocation(A, overlapSize, NULL, overlapSize, NULL));
  CHKERRQ(MatSetValues(A, 2, globalIdx, 2, globalIdx, elemMat, ADD_VALUES));
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  /* Check that the constant vector is in the nullspace */
  CHKERRQ(MatCreateVecs(A, &x, &y));
  CHKERRQ(VecSet(x, 1.0));
  CHKERRQ(PetscObjectSetName((PetscObject) x, "x"));
  CHKERRQ(VecViewFromOptions(x, NULL, "-x_view"));
  CHKERRQ(MatMult(A, x, y));
  CHKERRQ(PetscObjectSetName((PetscObject) y, "y"));
  CHKERRQ(VecViewFromOptions(y, NULL, "-y_view"));
  CHKERRQ(VecNorm(y, NORM_2, &error));
  PetscCheckFalse(error > PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Invalid output, x should be in the nullspace of A");
  /* Check that an interior unit vector gets mapped to something of 1-norm 4 */
  if (size > 1) {
    CHKERRQ(VecSet(x, 0.0));
    CHKERRQ(VecSetValue(x, 1, 1.0, INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(x));
    CHKERRQ(VecAssemblyEnd(x));
    CHKERRQ(MatMult(A, x, y));
    CHKERRQ(VecNorm(y, NORM_1, &error));
    PetscCheckFalse(PetscAbsReal(error - 4) > PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Invalid output for matrix multiply");
  }
  /* Cleanup */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(PetscFinalize());
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
