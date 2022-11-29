/*
 * ex193.c
 *
 *  Created on: Jul 29, 2015
 *      Author: Fande Kong fdkong.jd@gmail.com
 */
/*
 * An example demonstrates how to use hierarchical partitioning approach
 */

#include <petscmat.h>

static char help[] = "Illustrates use of hierarchical partitioning.\n";

int main(int argc, char **args)
{
  Mat             A;    /* matrix */
  PetscInt        m, n; /* mesh dimensions in x- and y- directions */
  PetscInt        i, j, Ii, J, Istart, Iend;
  PetscMPIInt     size;
  PetscScalar     v;
  MatPartitioning part;
  IS              coarseparts, fineparts;
  IS              is, isn, isrows;
  MPI_Comm        comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscOptionsBegin(comm, NULL, "ex193", "hierarchical partitioning");
  m = 15;
  PetscCall(PetscOptionsInt("-M", "Number of mesh points in the x-direction", "partitioning", m, &m, NULL));
  n = 15;
  PetscCall(PetscOptionsInt("-N", "Number of mesh points in the y-direction", "partitioning", n, &n, NULL));
  PetscOptionsEnd();

  /*
     Assemble the matrix for the five point stencil (finite difference), YET AGAIN
  */
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  for (Ii = Istart; Ii < Iend; Ii++) {
    v = -1.0;
    i = Ii / n;
    j = Ii - i * n;
    if (i > 0) {
      J = Ii - n;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    }
    if (i < m - 1) {
      J = Ii + n;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    }
    if (j > 0) {
      J = Ii - 1;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    }
    if (j < n - 1) {
      J = Ii + 1;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    }
    v = 4.0;
    PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &v, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  /*
   Partition the graph of the matrix
  */
  PetscCall(MatPartitioningCreate(comm, &part));
  PetscCall(MatPartitioningSetAdjacency(part, A));
  PetscCall(MatPartitioningSetType(part, MATPARTITIONINGHIERARCH));
  PetscCall(MatPartitioningHierarchicalSetNcoarseparts(part, 2));
  PetscCall(MatPartitioningHierarchicalSetNfineparts(part, 4));
  PetscCall(MatPartitioningSetFromOptions(part));
  /* get new processor owner number of each vertex */
  PetscCall(MatPartitioningApply(part, &is));
  /* coarse parts */
  PetscCall(MatPartitioningHierarchicalGetCoarseparts(part, &coarseparts));
  PetscCall(ISView(coarseparts, PETSC_VIEWER_STDOUT_WORLD));
  /* fine parts */
  PetscCall(MatPartitioningHierarchicalGetFineparts(part, &fineparts));
  PetscCall(ISView(fineparts, PETSC_VIEWER_STDOUT_WORLD));
  /* partitioning */
  PetscCall(ISView(is, PETSC_VIEWER_STDOUT_WORLD));
  /* get new global number of each old global number */
  PetscCall(ISPartitioningToNumbering(is, &isn));
  PetscCall(ISView(isn, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISBuildTwoSided(is, NULL, &isrows));
  PetscCall(ISView(isrows, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISDestroy(&is));
  PetscCall(ISDestroy(&coarseparts));
  PetscCall(ISDestroy(&fineparts));
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&isn));
  PetscCall(MatPartitioningDestroy(&part));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4
      args: -mat_partitioning_hierarchical_Nfineparts 2
      requires: parmetis
      TODO: cannot run because parmetis does reproduce across all machines, probably due to nonportable random number generator

TEST*/
