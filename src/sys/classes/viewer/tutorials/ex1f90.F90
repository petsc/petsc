
      program ex5f90

#include <petsc/finclude/petscsys.h>
      use petscsys
      implicit none

      PetscViewer viewer
      PetscErrorCode ierr

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
      endif
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'binaryoutput',FILE_MODE_READ,viewer,ierr);CHKERRA(ierr)
      call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!      output_file: output/ex1_1.out
!
!TEST*/
