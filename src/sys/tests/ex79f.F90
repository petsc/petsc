!
!  PETSc Program to test PetscReal2d
!
program main
#include <petsc/finclude/petscsys.h>
  use petscsys
  implicit none
  PetscReal2d, pointer :: dbleptr(:)
  PetscInt i
  PetscErrorCode ierr

  PetscCallA(PetscInitialize(ierr))

  allocate (dbleptr(10))
  do i = 1, 10
    allocate (dbleptr(i)%ptr(20))
  end do
  do i = 1, 10
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
