
static char help[] = "Partition tiny grid using hierarchical partitioning and increase overlap using MatIncreaseOverlapSplit.\n\n";

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
  Mat             A,B;
  PetscErrorCode  ierr;
  PetscMPIInt     rank,size,membershipKey;
  PetscInt        *ia,*ja,*indices_sc,isrows_localsize;
  const PetscInt  *indices;
  MatPartitioning part;
  IS              is,isrows,isrows_sc;
  IS              coarseparts,fineparts;
  MPI_Comm        comm,scomm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  PetscCheckFalse(size != 4,comm,PETSC_ERR_WRONG_MPI_SIZE,"Must run with 4 processors ");
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  /*set a small matrix */
  CHKERRQ(PetscMalloc1(5,&ia));
  CHKERRQ(PetscMalloc1(16,&ja));
  if (rank == 0) {
    ja[0] = 1; ja[1] = 4; ja[2] = 0; ja[3] = 2; ja[4] = 5; ja[5] = 1; ja[6] = 3; ja[7] = 6;
    ja[8] = 2; ja[9] = 7;
    ia[0] = 0; ia[1] = 2; ia[2] = 5; ia[3] = 8; ia[4] = 10;
    membershipKey = 0;
  } else if (rank == 1) {
    ja[0] = 0; ja[1] = 5; ja[2] = 8; ja[3] = 1; ja[4] = 4; ja[5] = 6; ja[6] = 9; ja[7] = 2;
    ja[8] = 5; ja[9] = 7; ja[10] = 10; ja[11] = 3; ja[12] = 6; ja[13] = 11;
    ia[0] = 0; ia[1] = 3; ia[2] = 7; ia[3] = 11; ia[4] = 14;
    membershipKey = 0;
  } else if (rank == 2) {
    ja[0] = 4; ja[1] = 9; ja[2] = 12; ja[3] = 5; ja[4] = 8; ja[5] = 10; ja[6] = 13; ja[7] = 6;
    ja[8] = 9; ja[9] = 11; ja[10] = 14; ja[11] = 7; ja[12] = 10; ja[13] = 15;
    ia[0] = 0; ia[1] = 3; ia[2] = 7; ia[3] = 11; ia[4] = 14;
    membershipKey = 1;
  } else {
    ja[0] = 8; ja[1] = 13; ja[2] = 9; ja[3] = 12; ja[4] = 14; ja[5] = 10; ja[6] = 13; ja[7] = 15;
    ja[8] = 11; ja[9] = 14;
    ia[0] = 0; ia[1] = 2; ia[2] = 5; ia[3] = 8; ia[4] = 10;
    membershipKey = 1;
  }
  CHKERRQ(MatCreateMPIAdj(comm,4,16,ia,ja,NULL,&A));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  /*
   Partition the graph of the matrix
  */
  CHKERRQ(MatPartitioningCreate(comm,&part));
  CHKERRQ(MatPartitioningSetAdjacency(part,A));
  CHKERRQ(MatPartitioningSetType(part,MATPARTITIONINGHIERARCH));
  CHKERRQ(MatPartitioningHierarchicalSetNcoarseparts(part,2));
  CHKERRQ(MatPartitioningHierarchicalSetNfineparts(part,2));
  CHKERRQ(MatPartitioningSetFromOptions(part));
  /* get new processor owner number of each vertex */
  CHKERRQ(MatPartitioningApply(part,&is));
  /* coarse parts */
  CHKERRQ(MatPartitioningHierarchicalGetCoarseparts(part,&coarseparts));
  CHKERRQ(ISView(coarseparts,PETSC_VIEWER_STDOUT_WORLD));
  /* fine parts */
  CHKERRQ(MatPartitioningHierarchicalGetFineparts(part,&fineparts));
  CHKERRQ(ISView(fineparts,PETSC_VIEWER_STDOUT_WORLD));
  /* partitioning */
  CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  /* compute coming rows */
  CHKERRQ(ISBuildTwoSided(is,NULL,&isrows));
  CHKERRQ(ISView(isrows,PETSC_VIEWER_STDOUT_WORLD));
  /*create a sub-communicator */
  CHKERRMPI(MPI_Comm_split(comm, membershipKey,rank,&scomm));
  CHKERRQ(ISGetLocalSize(isrows,&isrows_localsize));
  CHKERRQ(PetscMalloc1(isrows_localsize,&indices_sc));
  CHKERRQ(ISGetIndices(isrows,&indices));
  CHKERRQ(PetscArraycpy(indices_sc,indices,isrows_localsize));
  CHKERRQ(ISRestoreIndices(isrows,&indices));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(ISDestroy(&coarseparts));
  CHKERRQ(ISDestroy(&fineparts));
  CHKERRQ(ISDestroy(&isrows));
  CHKERRQ(MatPartitioningDestroy(&part));
  /*create a sub-IS on the sub communicator  */
  CHKERRQ(ISCreateGeneral(scomm,isrows_localsize,indices_sc,PETSC_OWN_POINTER,&isrows_sc));
  CHKERRQ(MatConvert(A,MATMPIAIJ,MAT_INITIAL_MATRIX,&B));
#if 1
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
#endif
  /*increase overlap */
  CHKERRQ(MatIncreaseOverlapSplit(B,1,&isrows_sc,1));
  CHKERRQ(ISView(isrows_sc,NULL));
  CHKERRQ(ISDestroy(&isrows_sc));
  /*
    Free work space.  All PETSc objects should be destroyed when they
    are no longer needed.
  */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = PetscFinalize();
  return ierr;
}
