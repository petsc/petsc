!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ matrix format
!
#include <petsc/finclude/petscsys.h>
!
pure subroutine FortranMultTransposeAddAIJ(n,x,ii,jj,a,y)
  implicit none (type, external)
  PetscScalar, intent(in) :: x(0:*),a(0:*)
  PetscScalar, intent(inout) :: y(0:*)
  PetscScalar :: alpha
  PetscInt, intent(in) :: n,ii(*),jj(0:*)

  PetscInt    i,j,jstart,jend

  jend  = ii(1)
  do i=1,n
    jstart = jend
    jend   = ii(i+1)
    alpha  = x(i-1)
    do j=jstart,jend-1
      y(jj(j)) = y(jj(j)) + alpha*a(j)
    end do
  end do
end subroutine FortranMultTransposeAddAIJ

pure subroutine FortranMultAIJ(n,x,ii,jj,a,y)
  implicit none (type, external)
  PetscScalar, intent(in) :: x(0:*),a(0:*)
  PetscScalar, intent(inout) :: y(*)
  PetscInt, intent(in) :: n,ii(*),jj(0:*)

  PetscInt i,j,jstart,jend
  PetscScalar  sum

#ifdef PETSC_USE_OPENMP_KERNELS
  !omp parallel do private(j,jstart,jend,sum)
#endif
  do i=1,n
    jstart = ii(i)
    jend   = ii(i+1)
    sum    = 0.d0
    do j=jstart,jend-1
      sum = sum + a(j)*x(jj(j))
    end do
    y(i) = sum
  end do
end subroutine FortranMultAIJ
