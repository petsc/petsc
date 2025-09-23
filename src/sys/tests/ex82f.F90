program main
#include <petsc/finclude/petscsys.h>
  use petscsys
  implicit none
  character(len=256) :: filename
  character(len=1)   :: mode
  PetscBool          :: exists
  PetscErrorCode     :: ierr
  PetscCallA(PetscInitialize(ierr))
  filename = 'filename'
  mode = 'r'
  PetscCallA(PetscTestFile(filename, mode, exists, ierr))
  write (*, '(A, A, A, I0)') "File ", trim(filename), " doesn't exist = ", merge(1, 0, exists)
  PetscCallA(PetscFinalize(ierr))
end
!/*TEST
!
!   test:
!
!TEST*/
