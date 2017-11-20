      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

      type(tPetscSection)   section
      PetscInt       pStart, pEnd, p,three
      PetscErrorCode ierr

      three = 3
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call PetscSectionCreate(PETSC_COMM_WORLD, section, ierr);CHKERRA(ierr)
      pStart = 0
      pEnd   = 5
      call PetscSectionSetChart(section, pStart, pEnd, ierr);CHKERRA(ierr)
      do p=pStart,pEnd-1
         call PetscSectionSetDof(section, p, three, ierr);CHKERRA(ierr)
      end do
      call PetscSectionSetUp(section, ierr);CHKERRA(ierr)
      call PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD, ierr);CHKERRA(ierr)
      call PetscSectionDestroy(section, ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!     test:
!
!TEST*/
