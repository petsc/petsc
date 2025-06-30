!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ format
!
#include <petsc/finclude/petscsys.h>
!
subroutine FortranMultAddAIJ(n,x,ii,jj,a,y,z)
  implicit none
  PetscScalar      x(0:*),a(0:*),y(*),z(*)
  PetscInt          n,ii(*),jj(0:*)

  PetscInt i,j,jstart,jend
  PetscScalar  sum

  jend  = ii(1)
  do i=1,n
    jstart = jend
    jend   = ii(i+1)
    sum    = y(i)
    do j=jstart,jend-1
      sum = sum + a(j)*x(jj(j))
    end do
    z(i) = sum
  end do

end subroutine FortranMultAddAIJ
