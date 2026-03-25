#include <petsc/finclude/petscdmplex.h>
program main
  use petscdm
  use petscdmplex
  implicit none

  DM dm
  PetscInt, target, dimension(3) :: EC
  PetscInt, target, dimension(2) :: VE
  PetscInt, pointer, dimension(:) :: pEC, pVE, nClosure, nJoin, nMeet
  PetscInt, parameter :: dim = 2
  PetscInt cell, size, nC
  PetscErrorCode ierr

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(DMPlexCreate(PETSC_COMM_WORLD, dm, ierr))
  PetscCallA(PetscObjectSetName(dm, 'Mesh', ierr))
  PetscCallA(DMSetDimension(dm, dim, ierr))

! Make Doublet Mesh from Fig 2 of Flexible Representation of Computational Meshes,
! except indexing is from 0 instead of 1 and we obey the new restrictions on
! numbering: cells, vertices, faces, edges
  PetscCallA(DMPlexSetChart(dm, 0_PETSC_INT_KIND, 11_PETSC_INT_KIND, ierr))
!     cells
  PetscCallA(DMPlexSetConeSize(dm, 0_PETSC_INT_KIND, 3_PETSC_INT_KIND, ierr))
  PetscCallA(DMPlexSetConeSize(dm, 1_PETSC_INT_KIND, 3_PETSC_INT_KIND, ierr))
!     edges
  PetscCallA(DMPlexSetConeSize(dm, 6_PETSC_INT_KIND, 2_PETSC_INT_KIND, ierr))
  PetscCallA(DMPlexSetConeSize(dm, 7_PETSC_INT_KIND, 2_PETSC_INT_KIND, ierr))
  PetscCallA(DMPlexSetConeSize(dm, 8_PETSC_INT_KIND, 2_PETSC_INT_KIND, ierr))
  PetscCallA(DMPlexSetConeSize(dm, 9_PETSC_INT_KIND, 2_PETSC_INT_KIND, ierr))
  PetscCallA(DMPlexSetConeSize(dm, 10_PETSC_INT_KIND, 2_PETSC_INT_KIND, ierr))

  PetscCallA(DMSetUp(dm, ierr))

  EC = [6, 7, 8]
  pEC => EC
  PetscCallA(DMPlexSetCone(dm, 0_PETSC_INT_KIND, pEC, ierr))
  EC = [7, 9, 10]
  pEC => EC
  PetscCallA(DMPlexSetCone(dm, 1_PETSC_INT_KIND, pEC, ierr))

  VE = [2, 3]
  pVE => VE
  PetscCallA(DMPlexSetCone(dm, 6_PETSC_INT_KIND, pVE, ierr))
  VE = [3, 4]
  pVE => VE
  PetscCallA(DMPlexSetCone(dm, 7_PETSC_INT_KIND, pVE, ierr))
  VE = [4, 2]
  pVE => VE
  PetscCallA(DMPlexSetCone(dm, 8_PETSC_INT_KIND, pVE, ierr))
  VE = [3, 5]
  pVE => VE
  PetscCallA(DMPlexSetCone(dm, 9_PETSC_INT_KIND, pVE, ierr))
  VE = [5, 4]
  pVE => VE
  PetscCallA(DMPlexSetCone(dm, 10_PETSC_INT_KIND, pVE, ierr))

  PetscCallA(DMPlexSymmetrize(dm, ierr))
  PetscCallA(DMPlexStratify(dm, ierr))
  PetscCallA(DMView(dm, PETSC_VIEWER_STDOUT_WORLD, ierr))

! Test Closure
  do cell = 0, 1
    PetscCallA(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, nC, nClosure, ierr))
!   Different Fortran compilers print a different number of columns
!   per row producing different outputs in the test runs hence
!   do not print the nClosure
    write (*, 1000) 'nClosure ', nClosure
1000 format(a, 30i4)
    PetscCallA(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, nC, nClosure, ierr))
  end do

! Test Join
  size = 2
  VE = [6, 7]
  pVE => VE
  PetscCallA(DMPlexGetJoin(dm, size, pVE, PETSC_NULL_INTEGER, nJoin, ierr))
  write (*, 1001) 'Join of', pVE
  write (*, 1002) '  is', nJoin
  PetscCallA(DMPlexRestoreJoin(dm, size, pVE, PETSC_NULL_INTEGER, nJoin, ierr))
  size = 2
  VE = [9, 7]
  pVE => VE
  PetscCallA(DMPlexGetJoin(dm, size, pVE, PETSC_NULL_INTEGER, nJoin, ierr))
  write (*, 1001) 'Join of', pVE
1001 format(a, 10i5)
  write (*, 1002) '  is', nJoin
1002 format(a, 10i5)
  PetscCallA(DMPlexRestoreJoin(dm, size, pVE, PETSC_NULL_INTEGER, nJoin, ierr))
! Test Full Join
  size = 3
  EC = [3, 4, 5]
  pEC => EC
  PetscCallA(DMPlexGetFullJoin(dm, size, pEC, PETSC_NULL_INTEGER, nJoin, ierr))
  write (*, 1001) 'Full Join of', pEC
  write (*, 1002) '  is', nJoin
  PetscCallA(DMPlexRestoreJoin(dm, size, pEC, PETSC_NULL_INTEGER, nJoin, ierr))
! Test Meet
  size = 2
  VE = [0, 1]
  pVE => VE
  PetscCallA(DMPlexGetMeet(dm, size, pVE, PETSC_NULL_INTEGER, nMeet, ierr))
  write (*, 1001) 'Meet of', pVE
  write (*, 1002) '  is', nMeet
  PetscCallA(DMPlexRestoreMeet(dm, size, pVE, PETSC_NULL_INTEGER, nMeet, ierr))
  size = 2
  VE = [6, 7]
  pVE => VE
  PetscCallA(DMPlexGetMeet(dm, size, pVE, PETSC_NULL_INTEGER, nMeet, ierr))
  write (*, 1001) 'Meet of', pVE
  write (*, 1002) '  is', nMeet
  PetscCallA(DMPlexRestoreMeet(dm, size, pVE, PETSC_NULL_INTEGER, nMeet, ierr))

  PetscCallA(DMDestroy(dm, ierr))
  PetscCallA(PetscFinalize(ierr))
end
!
!/*TEST
!
!   test:
!     suffix: 0
!
!TEST*/
