!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ format
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranMultAddAIJ(n, x, ii, jj, a, y, z)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) :: x(0:*), a(0:*), y(*)
  PetscScalar, intent(inout) :: z(*)
  PetscInt, intent(in) :: n, ii(*), jj(0:*)

  PetscInt :: i, jstart, jend

  jend = ii(1)
  do i = 1, n
    jstart = jend
    jend = ii(i + 1)
    z(i) = y(i) + sum(a(jstart:jend - 1)*x(jj(jstart:jend - 1)))
  end do
end subroutine FortranMultAddAIJ
