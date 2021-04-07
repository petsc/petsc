
static char help[] = "Partition a tiny grid using hierarchical partitioning.\n\n";

/*T
   Concepts: partitioning
   Processors: 4
T*/



/*
  Include "petscmat.h" so that we can use matrices.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets
     petscviewer.h - viewers
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat             A;
  PetscErrorCode  ierr;
  PetscMPIInt     rank,size;
  PetscInt        *ia,*ja;
  MatPartitioning part;
  IS              is,isn,isrows;
  IS              coarseparts,fineparts;
  MPI_Comm        comm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size != 4) SETERRQ(comm,PETSC_ERR_WRONG_MPI_SIZE,"Must run with 4 processors");
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  ierr = PetscMalloc1(5,&ia);CHKERRQ(ierr);
  ierr = PetscMalloc1(16,&ja);CHKERRQ(ierr);
  if (!rank) {
    ja[0] = 1; ja[1] = 4; ja[2] = 0; ja[3] = 2; ja[4] = 5; ja[5] = 1; ja[6] = 3; ja[7] = 6;
    ja[8] = 2; ja[9] = 7;
    ia[0] = 0; ia[1] = 2; ia[2] = 5; ia[3] = 8; ia[4] = 10;
  } else if (rank == 1) {
    ja[0] = 0; ja[1] = 5; ja[2] = 8; ja[3] = 1; ja[4] = 4; ja[5] = 6; ja[6] = 9; ja[7] = 2;
    ja[8] = 5; ja[9] = 7; ja[10] = 10; ja[11] = 3; ja[12] = 6; ja[13] = 11;
    ia[0] = 0; ia[1] = 3; ia[2] = 7; ia[3] = 11; ia[4] = 14;
  } else if (rank == 2) {
    ja[0] = 4; ja[1] = 9; ja[2] = 12; ja[3] = 5; ja[4] = 8; ja[5] = 10; ja[6] = 13; ja[7] = 6;
    ja[8] = 9; ja[9] = 11; ja[10] = 14; ja[11] = 7; ja[12] = 10; ja[13] = 15;
    ia[0] = 0; ia[1] = 3; ia[2] = 7; ia[3] = 11; ia[4] = 14;
  } else {
    ja[0] = 8; ja[1] = 13; ja[2] = 9; ja[3] = 12; ja[4] = 14; ja[5] = 10; ja[6] = 13; ja[7] = 15;
    ja[8] = 11; ja[9] = 14;
    ia[0] = 0; ia[1] = 2; ia[2] = 5; ia[3] = 8; ia[4] = 10;
  }
  ierr = MatCreateMPIAdj(comm,4,16,ia,ja,NULL,&A);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /*
   Partition the graph of the matrix
  */
  ierr = MatPartitioningCreate(comm,&part);CHKERRQ(ierr);
  ierr = MatPartitioningSetAdjacency(part,A);CHKERRQ(ierr);
  ierr = MatPartitioningSetType(part,MATPARTITIONINGHIERARCH);CHKERRQ(ierr);
  ierr = MatPartitioningHierarchicalSetNcoarseparts(part,2);CHKERRQ(ierr);
  ierr = MatPartitioningHierarchicalSetNfineparts(part,2);CHKERRQ(ierr);
  ierr = MatPartitioningSetFromOptions(part);CHKERRQ(ierr);
  /* get new processor owner number of each vertex */
  ierr = MatPartitioningApply(part,&is);CHKERRQ(ierr);
  /* coarse parts */
  ierr = MatPartitioningHierarchicalGetCoarseparts(part,&coarseparts);CHKERRQ(ierr);
  ierr = ISView(coarseparts,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* fine parts */
  ierr = MatPartitioningHierarchicalGetFineparts(part,&fineparts);CHKERRQ(ierr);
  ierr = ISView(fineparts,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* partitioning */
  ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* get new global number of each old global number */
  ierr = ISPartitioningToNumbering(is,&isn);CHKERRQ(ierr);
  ierr = ISView(isn,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISBuildTwoSided(is,NULL,&isrows);CHKERRQ(ierr);
  ierr = ISView(isrows,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&coarseparts);CHKERRQ(ierr);
  ierr = ISDestroy(&fineparts);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&isn);CHKERRQ(ierr);
  ierr = MatPartitioningDestroy(&part);CHKERRQ(ierr);
  /*
    Free work space.  All PETSc objects should be destroyed when they
    are no longer needed.
  */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 4
      requires: parmetis
      TODO: tests cannot use parmetis because it produces different results on different machines

TEST*/
