#include <petsc/finclude/petscsys.h>
!

pure subroutine Fortranxtimesy(x, y, z, n)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) :: x(*), y(*)
  PetscScalar, intent(inout) :: z(*)
  PetscInt, intent(in) :: n

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y(1))
  PETSC_AssertAlignx(16, z(1))

  z(1:n) = x(1:n)*y(1:n)
end subroutine Fortranxtimesy
