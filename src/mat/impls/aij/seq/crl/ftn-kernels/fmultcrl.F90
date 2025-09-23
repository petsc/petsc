!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ/CRL format
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranMultCRL(m, rmax, x, y, icols, acols)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscInt, intent(in) :: m, rmax, icols(m, rmax)
  PetscScalar, intent(in) :: x(0:m - 1), acols(m, rmax)
  PetscScalar, intent(out) :: y(m)

  PetscInt :: i

  y(1:m) = acols(1:m, 1)*x(icols(1:m, 1))
  do i = 2, rmax
    y(1:m) = y(1:m) + acols(1:m, i)*x(icols(1:m, i))
  end do
end subroutine FortranMultCRL
