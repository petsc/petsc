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
  PetscErrorCode         ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  /* Create local-to-global map */
  globalIdx[0] = rank;
  globalIdx[1] = rank+1;
  ierr = ISLocalToGlobalMappingCreate(comm, 1, overlapSize, globalIdx, PETSC_COPY_VALUES, &map);CHKERRQ(ierr);
  /* Create matrix */
  ierr = MatCreateIS(comm, 1, PETSC_DECIDE, PETSC_DECIDE, size+1, size+1, map, map, &A);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) A, "A");CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&map);CHKERRQ(ierr);
  ierr = MatISSetPreallocation(A, overlapSize, NULL, overlapSize, NULL);CHKERRQ(ierr);
  ierr = MatSetValues(A, 2, globalIdx, 2, globalIdx, elemMat, ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* Check that the constant vector is in the nullspace */
  ierr = MatCreateVecs(A, &x, &y);CHKERRQ(ierr);
  ierr = VecSet(x, 1.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "x");CHKERRQ(ierr);
  ierr = VecViewFromOptions(x, NULL, "-x_view");CHKERRQ(ierr);
  ierr = MatMult(A, x, y);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) y, "y");CHKERRQ(ierr);
  ierr = VecViewFromOptions(y, NULL, "-y_view");CHKERRQ(ierr);
  ierr = VecNorm(y, NORM_2, &error);CHKERRQ(ierr);
  PetscCheckFalse(error > PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Invalid output, x should be in the nullspace of A");
  /* Check that an interior unit vector gets mapped to something of 1-norm 4 */
  if (size > 1) {
    ierr = VecSet(x, 0.0);CHKERRQ(ierr);
    ierr = VecSetValue(x, 1, 1.0, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    ierr = MatMult(A, x, y);CHKERRQ(ierr);
    ierr = VecNorm(y, NORM_1, &error);CHKERRQ(ierr);
    PetscCheckFalse(PetscAbsReal(error - 4) > PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Invalid output for matrix multiply");
  }
  /* Cleanup */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
