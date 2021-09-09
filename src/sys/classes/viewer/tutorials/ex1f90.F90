
      program ex1f90

#include <petsc/finclude/petscsys.h>
      use petscsys
      use iso_c_binding
      implicit none

      PetscViewer viewer
      PetscErrorCode ierr
      call PetscInitialize(PETSC_NULL_CHARACTER,"ex1f90 test"//c_new_line,ierr)
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
!   build:
!      requires: defined(PETSC_USING_F2003) defined(PETSC_USING_F90FREEFORM)
!
!   test:
!      output_file: output/ex1_1.out
!
!TEST*/
