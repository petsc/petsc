      program main
      implicit none
!
#include <petsc/finclude/petsc.h90>
!
      PetscSection   section
      PetscInt       pStart, pEnd, p,three
      PetscErrorCode ierr

      three = 3
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call PetscSectionCreate(PETSC_COMM_WORLD, section, ierr);CHKERRQ(ierr)
      pStart = 0
      pEnd   = 5
      call PetscSectionSetChart(section, pStart, pEnd, ierr);CHKERRQ(ierr)
      do p=pStart,pEnd-1
         call PetscSectionSetDof(section, p, three, ierr);CHKERRQ(ierr)
      end do
      call PetscSectionSetUp(section, ierr);CHKERRQ(ierr)
      call PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD, ierr);CHKERRQ(ierr)
      call PetscSectionDestroy(section, ierr);CHKERRQ(ierr)
      call PetscFinalize(ierr)
      end
