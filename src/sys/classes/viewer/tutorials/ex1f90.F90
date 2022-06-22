
      program ex1f90

#include <petsc/finclude/petscsys.h>
      use petscsys
      use,intrinsic :: iso_c_binding
      implicit none

      PetscViewer viewer
      PetscErrorCode ierr
      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER,"ex1f90 test"//c_new_line,ierr))

      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,'binaryoutput',FILE_MODE_READ,viewer,ierr))
      PetscCallA(PetscViewerDestroy(viewer,ierr))
      PetscCallA(PetscFinalize(ierr))
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
