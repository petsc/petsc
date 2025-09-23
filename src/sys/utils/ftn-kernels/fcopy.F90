!
!
!    Fortran kernel for the copy vector routine
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranCopy(n, x, y)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) :: x(*)
  PetscScalar, intent(inout) :: y(*)
  PetscInt, intent(in) :: n

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y(1))

  y(1:n) = x(1:n)
end subroutine FortranCopy

pure subroutine FortranZero(n, x)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(inout) :: x(*)
  PetscInt, intent(in) :: n

  PETSC_AssertAlignx(16, x(1))

  x(1:n) = 0.0
end subroutine FortranZero
