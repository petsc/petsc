#include <petsc/finclude/petscdmplex.h>
program main
  use petscdmplex
  implicit none
!
!
  DM dm
  PetscInt, dimension(4) :: EC
  PetscInt, pointer :: pEC(:), pES(:)
  PetscInt, parameter :: firstCell = 0, numCells = 2, numVertices = 6, numPoints = numCells + numVertices
  PetscInt c, v
  PetscErrorCode ierr

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(DMPlexCreate(PETSC_COMM_WORLD, dm, ierr))
  PetscCallA(DMPlexSetChart(dm, 0_PETSC_INT_KIND, numPoints, ierr))
  do c = firstCell, numCells - 1
    PetscCallA(DMPlexSetConeSize(dm, c, 4_PETSC_INT_KIND, ierr))
  end do
  PetscCallA(DMSetUp(dm, ierr))

  EC = [2, 3, 4, 5]
  c = 0
  write (*, 1000) 'cell EC 0', c, EC
1000 format(a, i4, 50i4)
  PetscCallA(DMPlexSetCone(dm, c, EC, ierr))
  PetscCallA(DMPlexGetCone(dm, c, pEC, ierr))
  write (*, 1000) 'cell pEC 0', c, pEC
  PetscCallA(DMPlexRestoreCone(dm, c, pEC, ierr))
  EC = [4, 5, 6, 7]
  c = 1
  write (*, 1000) 'cell EC 1', c, EC
  PetscCallA(DMPlexSetCone(dm, c, EC, ierr))
  PetscCallA(DMPlexGetCone(dm, c, pEC, ierr))
  write (*, 1000) 'cell pEC 1', c, pEC
  PetscCallA(DMPlexRestoreCone(dm, c, pEC, ierr))
  CHKMEMQ

  PetscCallA(DMPlexSymmetrize(dm, ierr))
  PetscCallA(DMPlexStratify(dm, ierr))
  PetscCallA(DMPlexGetCone(dm, c, pEC, ierr))
  write (*, 1000) 'cell pEC 3', c, pEC
  PetscCallA(DMPlexRestoreCone(dm, c, pEC, ierr))

  v = 4
  PetscCallA(DMPlexGetSupport(dm, v, pES, ierr))
  write (*, 1000) 'vertex', v, pES
  PetscCallA(DMPlexRestoreSupport(dm, v, pES, ierr))

  PetscCallA(DMDestroy(dm, ierr))
  PetscCallA(PetscFinalize(ierr))
end

! /*TEST
!
! test:
!   suffix: 0
!
! TEST*/
