!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ matrix format
!
#include <petsc/finclude/petscsys.h>
!
subroutine FortranMultTransposeAddAIJ(n,x,ii,jj,a,y)
  implicit none
  PetscScalar x(0:*),a(0:*),y(0:*)
  PetscScalar alpha
  PetscInt    n,ii(*),jj(0:*)

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

subroutine FortranMultAIJ(n,x,ii,jj,a,y)
  implicit none
  PetscScalar x(0:*),a(0:*),y(*)
  PetscInt    n,ii(*),jj(0:*)

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
