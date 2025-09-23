!
!
!    Fortran kernel for the WAXPY() vector routine
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranWAXPY(n, a, x, y, w)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) :: a
  PetscScalar, intent(in) :: x(*), y(*)
  PetscScalar, intent(inout) :: w(*)
  PetscInt, intent(in) :: n

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y(1))
  PETSC_AssertAlignx(16, w(1))

  w(1:n) = a*x(1:n) + y(1:n)
end subroutine FortranWAXPY
