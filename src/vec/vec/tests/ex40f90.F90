      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

      type(tPetscSection)   section
      PetscInt       pStart, pEnd, p,three
      PetscErrorCode ierr

      three = 3
      PetscCallA(PetscInitialize(ierr))

      PetscCallA(PetscSectionCreate(PETSC_COMM_WORLD, section, ierr))
      pStart = 0
      pEnd   = 5
      PetscCallA(PetscSectionSetChart(section, pStart, pEnd, ierr))
      do p=pStart,pEnd-1
         PetscCallA(PetscSectionSetDof(section, p, three, ierr))
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
