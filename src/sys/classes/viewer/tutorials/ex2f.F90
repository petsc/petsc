program main

#include <petsc/finclude/petscsys.h>
  use petscsys
  implicit none

  PetscErrorCode ierr

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, 'ex2f90 test'//c_new_line, ierr))
  PetscCallA(PetscViewerView(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      args:
!
!TEST*/
