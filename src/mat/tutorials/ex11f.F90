! Test MatCreateMPIAdj() with NULL argument 'values'

program main
#include <petsc/finclude/petscmat.h>
  use petscmat
  implicit none

  Mat                   :: mesh, dual
  MatPartitioning       :: part
  IS                    :: is
  PetscInt, parameter    :: Nvertices = 6, ncells = 2, two = 2
  PetscInt              :: ii(3), jj(6)
  PetscMPIInt           :: sz, rnk
  PetscErrorCode        :: ierr

  PetscCallA(PetscInitialize(ierr))

  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, sz, ierr))
  PetscCheckA(sz == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, 'This example is for exactly two processes')
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rnk, ierr))
  ii(1) = 0
  ii(2) = 3
  ii(3) = 6
  if (rnk == 0) then
    jj(1) = 0
    jj(2) = 1
    jj(3) = 2
    jj(4) = 1
    jj(5) = 2
    jj(6) = 3
  else
    jj(1) = 1
    jj(2) = 4
    jj(3) = 5
    jj(4) = 1
    jj(5) = 3
    jj(6) = 5
  end if

  PetscCallA(MatCreateMPIAdj(PETSC_COMM_WORLD, ncells, Nvertices, ii, jj, PETSC_NULL_INTEGER_ARRAY, mesh, ierr))
  PetscCallA(MatMeshToCellGraph(mesh, two, dual, ierr))
  PetscCallA(MatView(dual, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(MatPartitioningCreate(PETSC_COMM_WORLD, part, ierr))
  PetscCallA(MatPartitioningSetAdjacency(part, dual, ierr))
  PetscCallA(MatPartitioningSetFromOptions(part, ierr))
  PetscCallA(MatPartitioningApply(part, is, ierr))
  PetscCallA(ISView(is, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(ISDestroy(is, ierr))
  PetscCallA(MatPartitioningDestroy(part, ierr))

  PetscCallA(MatDestroy(mesh, ierr))
  PetscCallA(MatDestroy(dual, ierr))
  PetscCallA(PetscFinalize(ierr))

end program

!/*TEST
!
!   build:
!     requires: parmetis
!
!   test:
!      nsize: 2
!
!TEST*/
