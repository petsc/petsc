!
!
!    Fortran kernel for sparse triangular solve in the AIJ matrix format
! This ONLY works for factorizations in the NATURAL ORDERING, i.e.
! with MatSolve_SeqAIJ_NaturalOrdering()
!
#include <petsc/finclude/petscsys.h>
!
subroutine FortranSolveAIJ(n,x,ai,aj,adiag,aa,b)
  implicit none
  PetscScalar x(0:*),aa(0:*),b(0:*)
  PetscInt n,ai(0:*)
  PetscInt aj(0:*),adiag(0:*)

  PetscInt i,j,jstart,jend
  PetscScalar sum
  !
  ! Forward Solve
  !
  x(0) = b(0)
  do i=1,n-1
    jstart = ai(i)
    jend   = adiag(i) - 1
    sum    = b(i)
    do j=jstart,jend
      sum  = sum -  aa(j) * x(aj(j))
    end do
    x(i) = sum
  end do
  !
  ! Backward solve the upper triangular
  !
  do i=n-1,0,-1
    jstart = adiag(i) + 1
    jend   = ai(i+1) - 1
    sum    = x(i)
    do j=jstart,jend
      sum = sum - aa(j)* x(aj(j))
    end do
    x(i) = sum * aa(adiag(i))
  end do

end subroutine FortranSolveAIJ
