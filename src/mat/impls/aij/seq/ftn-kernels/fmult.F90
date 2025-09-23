!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ matrix format
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranMultTransposeAddAIJ(n, x, ii, jj, a, y)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) :: x(0:*), a(0:*)
  PetscScalar, intent(inout) :: y(0:*)
  PetscInt, intent(in) :: n, ii(*), jj(0:*)

  PetscInt :: i, jstart, jend

  jend = ii(1)
  do i = 1, n
    jstart = jend
    jend = ii(i + 1)
    y(jj(jstart:jend - 1)) = y(jj(jstart:jend - 1)) + x(i - 1)*a(jstart:jend - 1)
  end do
end subroutine FortranMultTransposeAddAIJ

pure subroutine FortranMultAIJ(n, x, ii, jj, a, y)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) :: x(0:*), a(0:*)
  PetscScalar, intent(inout) :: y(*)
  PetscInt, intent(in) :: n, ii(*), jj(0:*)

  PetscInt :: i, jstart, jend

#ifdef PETSC_USE_OPENMP_KERNELS
  !omp parallel do private(jstart,jend)
#endif
  do i = 1, n
    jstart = ii(i)
    jend = ii(i + 1)
    y(i) = sum(a(jstart:jend - 1)*x(jj(jstart:jend - 1)))
  end do
end subroutine FortranMultAIJ
