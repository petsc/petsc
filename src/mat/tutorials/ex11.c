
static char help[] = "Tests MatMeshToDual()\n\n";

/*T
   Concepts: Mat^mesh partitioning
   Processors: n
T*/

/*
  Include "petscmat.h" so that we can use matrices.
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h    - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscviewer.h - viewers
*/
#include <petscmat.h>

int main(int argc,char **args)
{
  Mat             mesh,dual;
  PetscErrorCode  ierr;
  PetscInt        Nvertices = 6;       /* total number of vertices */
  PetscInt        ncells    = 2;       /* number cells on this process */
  PetscInt        *ii,*jj;
  PetscMPIInt     size,rank;
  MatPartitioning part;
  IS              is;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(MPI_COMM_WORLD,&size));
  PetscCheckFalse(size != 2,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example is for exactly two processes");
  CHKERRMPI(MPI_Comm_rank(MPI_COMM_WORLD,&rank));

  CHKERRQ(PetscMalloc1(3,&ii));
  CHKERRQ(PetscMalloc1(6,&jj));
  ii[0] = 0; ii[1] = 3; ii[2] = 6;
  if (rank == 0) {
    jj[0] = 0; jj[1] = 1; jj[2] = 2; jj[3] = 1; jj[4] = 2; jj[5] = 3;
  } else {
    jj[0] = 1; jj[1] = 4; jj[2] = 5; jj[3] = 1; jj[4] = 3; jj[5] = 5;
  }
  CHKERRQ(MatCreateMPIAdj(MPI_COMM_WORLD,ncells,Nvertices,ii,jj,NULL,&mesh));
  CHKERRQ(MatMeshToCellGraph(mesh,2,&dual));
  CHKERRQ(MatView(dual,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatPartitioningCreate(MPI_COMM_WORLD,&part));
  CHKERRQ(MatPartitioningSetAdjacency(part,dual));
  CHKERRQ(MatPartitioningSetFromOptions(part));
  CHKERRQ(MatPartitioningApply(part,&is));
  CHKERRQ(ISView(is,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(MatPartitioningDestroy(&part));

  CHKERRQ(MatDestroy(&mesh));
  CHKERRQ(MatDestroy(&dual));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: parmetis

   test:
      nsize: 2
      requires: parmetis

TEST*/
