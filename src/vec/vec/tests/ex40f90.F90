#include <petsc/finclude/petscis.h>
program main
  use petscis
  implicit none

  type(tPetscSection) section
  PetscInt, parameter :: pStart = 0, pEnd = 5
  PetscInt :: p
  PetscErrorCode ierr

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(PetscSectionCreate(PETSC_COMM_WORLD, section, ierr))
  PetscCallA(PetscSectionSetChart(section, pStart, pEnd, ierr))
  do p = pStart, pEnd - 1
    PetscCallA(PetscSectionSetDof(section, p, 3_PETSC_INT_KIND, ierr))
  end do
  PetscCallA(PetscSectionSetUp(section, ierr))
  PetscCallA(PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(PetscSectionDestroy(section, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!     test:
!
!TEST*/
