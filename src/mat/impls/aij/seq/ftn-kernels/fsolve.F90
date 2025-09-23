!
!
!    Fortran kernel for sparse triangular solve in the AIJ matrix format
! This ONLY works for factorizations in the NATURAL ORDERING, i.e.
! with MatSolve_SeqAIJ_NaturalOrdering()
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranSolveAIJ(n, x, ai, aj, adiag, aa, b)
  use, intrinsic :: ISO_C_binding
  implicit none(type, external)
  PetscScalar, intent(in) :: aa(0:*), b(0:*)
  PetscInt, intent(in) :: n, ai(0:*), aj(0:*), adiag(0:*)
  PetscScalar, intent(inout) :: x(0:*)

  PetscInt i, jstart, jend
  !
  ! Forward Solve
  !
  x(0) = b(0)
  do i = 1, n - 1
    jstart = ai(i)
    jend = adiag(i) - 1
    x(i) = b(i) - sum(aa(jstart:jend)*x(aj(jstart:jend)))
  end do
  !
  ! Backward solve the upper triangular
  !
  do i = n - 1, 0, -1
    jstart = adiag(i) + 1
    jend = ai(i + 1) - 1
    x(i) = x(i) - sum(aa(jstart:jend)*x(aj(jstart:jend)))*aa(adiag(i))
  end do
end subroutine FortranSolveAIJ
