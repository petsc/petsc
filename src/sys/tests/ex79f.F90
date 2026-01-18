!
!  PETSc Program to test PetscReal2d
!
#include <petsc/finclude/petscsys.h>
program main
  use petscsys
  implicit none
  PetscReal2d, pointer :: dbleptr(:)
  PetscInt, parameter :: n = 10
  PetscInt i
  PetscErrorCode ierr

  PetscCallA(PetscInitialize(ierr))

  allocate (dbleptr(n))
  do i = 1, n
    allocate (dbleptr(i)%ptr(20))
  end do
  do i = 1, n
    deallocate (dbleptr(i)%ptr)
  end do
  deallocate (dbleptr)

  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      output_file: output/empty.out
!
!TEST*/
