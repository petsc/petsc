#include <petsc/finclude/petscsys.h>
program main
  use petscsys
  implicit none
  character(len=256), parameter :: filename = 'filename'
  character(len=1), parameter :: mode = 'r'
  PetscBool          :: exists
  PetscErrorCode     :: ierr

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(PetscTestFile(filename, mode, exists, ierr))
  write (*, '(A, A, A, I0)') "File ", trim(filename), " doesn't exist = ", merge(1, 0, exists)
  PetscCallA(PetscFinalize(ierr))
end
!/*TEST
!
!   test:
!
!TEST*/
