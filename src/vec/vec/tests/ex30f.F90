!
!
!  Tests parallel to parallel scatter where a to from index are
!  duplicated
#include <petsc/finclude/petscvec.h>
program main
  use petscvec
  implicit none

  PetscErrorCode ierr
  PetscInt, parameter :: nlocal = 2, n = 8
  PetscInt row
  PetscMPIInt rank, size
  PetscInt, parameter, dimension(8) :: &
    from = [1, 5, 9, 13, 3, 7, 11, 15], &
    to = [0, 0, 0, 0, 7, 7, 7, 7]
  PetscScalar num
  Vec v1, v2, v3
  VecScatter scat1, scat2
  IS fromis, tois
  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_COMM_RANK(PETSC_COMM_WORLD, rank, ierr))
  PetscCallMPIA(MPI_COMM_SIZE(PETSC_COMM_WORLD, size, ierr))
  if (size /= 4) then
    print *, 'Four processor test'
    stop
  end if

  PetscCallA(VecCreateFromOptions(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, 1_PETSC_INT_KIND, nlocal*2, n*2, v1, ierr))
  PetscCallA(VecCreateFromOptions(PETSC_COMM_WORLD, PETSC_NULL_CHARACTER, 1_PETSC_INT_KIND, nlocal, n, v2, ierr))
  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, n, v3, ierr))

  num = 2.0
  row = 1
  PetscCallA(VecSetValue(v1, row, num, INSERT_VALUES, ierr))
  row = 5
  PetscCallA(VecSetValue(v1, row, num, INSERT_VALUES, ierr))
  row = 9
  PetscCallA(VecSetValue(v1, row, num, INSERT_VALUES, ierr))
  row = 13
  PetscCallA(VecSetValue(v1, row, num, INSERT_VALUES, ierr))
  num = 1.0
  row = 15
  PetscCallA(VecSetValue(v1, row, num, INSERT_VALUES, ierr))
  row = 3
  PetscCallA(VecSetValue(v1, row, num, INSERT_VALUES, ierr))
  row = 7
  PetscCallA(VecSetValue(v1, row, num, INSERT_VALUES, ierr))
  row = 11
  PetscCallA(VecSetValue(v1, row, num, INSERT_VALUES, ierr))

  PetscCallA(VecAssemblyBegin(v1, ierr))
  PetscCallA(VecAssemblyEnd(v1, ierr))

  num = 0.0
  PetscCallA(VecScale(v2, num, ierr))
  PetscCallA(VecScale(v3, num, ierr))

  PetscCallA(ISCreateGeneral(PETSC_COMM_SELF, 8_PETSC_INT_KIND, from, PETSC_COPY_VALUES, fromis, ierr))
  PetscCallA(ISCreateGeneral(PETSC_COMM_SELF, 8_PETSC_INT_KIND, to, PETSC_COPY_VALUES, tois, ierr))
  PetscCallA(VecScatterCreate(v1, fromis, v2, tois, scat1, ierr))
  PetscCallA(VecScatterCreate(v1, fromis, v3, tois, scat2, ierr))
  PetscCallA(ISDestroy(fromis, ierr))
  PetscCallA(ISDestroy(tois, ierr))

  PetscCallA(VecScatterBegin(scat1, v1, v2, ADD_VALUES, SCATTER_FORWARD, ierr))
  PetscCallA(VecScatterEnd(scat1, v1, v2, ADD_VALUES, SCATTER_FORWARD, ierr))

  PetscCallA(VecScatterBegin(scat2, v1, v3, ADD_VALUES, SCATTER_FORWARD, ierr))
  PetscCallA(VecScatterEnd(scat2, v1, v3, ADD_VALUES, SCATTER_FORWARD, ierr))

  PetscCallA(PetscObjectSetName(v1, 'V1', ierr))
  PetscCallA(VecView(v1, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(PetscObjectSetName(v2, 'V2', ierr))
  PetscCallA(VecView(v2, PETSC_VIEWER_STDOUT_WORLD, ierr))

  if (rank == 0) then
    PetscCallA(PetscObjectSetName(v3, 'V3', ierr))
    PetscCallA(VecView(v3, PETSC_VIEWER_STDOUT_SELF, ierr))
  end if

  PetscCallA(VecScatterDestroy(scat1, ierr))
  PetscCallA(VecScatterDestroy(scat2, ierr))
  PetscCallA(VecDestroy(v1, ierr))
  PetscCallA(VecDestroy(v2, ierr))
  PetscCallA(VecDestroy(v3, ierr))

  PetscCallA(PetscFinalize(ierr))

end

!/*TEST
!
!     test:
!       nsize: 4
!
!TEST*/
