!
!
!    Fortran kernel for the MAXPY() vector routine
!
#include <petsc/finclude/petscsys.h>
!

pure subroutine FortranMAXPY4(x, a0, a1, a2, a3, y0, y1, y2, y3, n)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) ::  a0, a1, a2, a3
  PetscScalar, intent(inout) :: x(*)
  PetscScalar, intent(in) :: y0(*), y1(*), y2(*), y3(*)
  PetscInt, intent(in) :: n

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y0(1))
  PETSC_AssertAlignx(16, y1(1))
  PETSC_AssertAlignx(16, y2(1))
  PETSC_AssertAlignx(16, y3(1))

  x(1:n) = x(1:n) + (a0*y0(1:n) + a1*y1(1:n) + a2*y2(1:n) + a3*y3(1:n))
end subroutine FortranMAXPY4

pure subroutine FortranMAXPY3(x, a0, a1, a2, y0, y1, y2, n)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) ::  a0, a1, a2
  PetscScalar, intent(inout) :: x(*)
  PetscScalar, intent(in) :: y0(*), y1(*), y2(*)
  PetscInt, intent(in) :: n

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y0(1))
  PETSC_AssertAlignx(16, y1(1))
  PETSC_AssertAlignx(16, y2(1))

  x(1:n) = x(1:n) + (a0*y0(1:n) + a1*y1(1:n) + a2*y2(1:n))
end subroutine FortranMAXPY3

pure subroutine FortranMAXPY2(x, a0, a1, y0, y1, n)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) ::  a0, a1
  PetscScalar, intent(inout) :: x(*)
  PetscScalar, intent(in) :: y0(*), y1(*)
  PetscInt, intent(in) :: n

  PETSC_AssertAlignx(16, x(1))
  PETSC_AssertAlignx(16, y0(1))
  PETSC_AssertAlignx(16, y1(1))

  x(1:n) = x(1:n) + (a0*y0(1:n) + a1*y1(1:n))
end subroutine FortranMAXPY2
