!
!  Test VecGetSubVector()
!  Contributed-by: Adrian Croucher <gitlab@mg.gitlab.com>

      program main
#include <petsc/finclude/petsc.h>
      use petsc
      implicit none

      PetscMPIInt :: rank
      PetscErrorCode :: ierr
      PetscInt :: num_cells, subsize, i
      PetscInt, parameter :: blocksize = 3, field = 0
      Vec :: v, subv
      IS :: index_set
      PetscInt, allocatable :: subindices(:)

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_COMM_RANK(PETSC_COMM_WORLD, rank, ierr))

      if (rank .eq. 0) then
         num_cells = 1
      else
         num_cells = 0
      end if

      PetscCallA(VecCreate(PETSC_COMM_WORLD, v, ierr))
      PetscCallA(VecSetSizes(v, num_cells * blocksize, PETSC_DECIDE, ierr))
      PetscCallA(VecSetBlockSize(v, blocksize, ierr))
      PetscCallA(VecSetFromOptions(v, ierr))

      subsize = num_cells
      allocate(subindices(0: subsize - 1))
      subindices = [(i, i = 0, subsize - 1)] * blocksize + field
      PetscCallA(ISCreateGeneral(PETSC_COMM_WORLD, subsize, subindices,PETSC_COPY_VALUES, index_set, ierr))
      deallocate(subindices)

      PetscCallA(VecGetSubVector(v, index_set, subv, ierr))
      PetscCallA(VecRestoreSubVector(v, index_set, subv, ierr))
      PetscCallA(ISDestroy(index_set, ierr))

      PetscCallA(VecDestroy(v, ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      nsize: 2
!      filter: sort -b
!      filter_output: sort -b
!
!TEST*/
