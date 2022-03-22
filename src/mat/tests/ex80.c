
static char help[] = "Partition tiny grid.\n\n";

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
  PetscMPIInt     rank,size;
  PetscInt        *ia,*ja;
  MatPartitioning part;
  IS              is,isn;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 4,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run with 4 processors");
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscMalloc1(5,&ia));
  PetscCall(PetscMalloc1(16,&ja));
  if (rank == 0) {
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

  PetscCall(MatCreateMPIAdj(PETSC_COMM_WORLD,4,16,ia,ja,NULL,&A));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  /*
       Partition the graph of the matrix
  */
  PetscCall(MatPartitioningCreate(PETSC_COMM_WORLD,&part));
  PetscCall(MatPartitioningSetAdjacency(part,A));
  PetscCall(MatPartitioningSetFromOptions(part));
  /* get new processor owner number of each vertex */
  PetscCall(MatPartitioningApply(part,&is));
  /* get new global number of each old global number */
  PetscCall(ISPartitioningToNumbering(is,&isn));
  PetscCall(ISView(isn,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISDestroy(&is));

  PetscCall(ISDestroy(&isn));
  PetscCall(MatPartitioningDestroy(&part));

  /*
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
  */
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}
