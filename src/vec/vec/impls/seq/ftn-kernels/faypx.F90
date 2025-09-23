!
!
!    Fortran kernel for the AYPX() vector routine
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranAYPX(n, a, x, y)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) :: a
  PetscScalar, intent(in) :: x(*)
  PetscScalar, intent(inout) :: y(*)
  PetscInt, intent(in) :: n

  y(1:n) = x(1:n) + a*y(1:n)
end subroutine FortranAYPX
