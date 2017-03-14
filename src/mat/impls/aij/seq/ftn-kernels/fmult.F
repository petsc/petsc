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
      do 10,i=1,n
        jstart = jend
        jend   = ii(i+1)
        alpha  = x(i-1)
        do 20 j=jstart,jend-1
          y(jj(j)) = y(jj(j)) + alpha*a(j)
 20     continue
 10   continue

      return
      end

      subroutine FortranMultAIJ(n,x,ii,jj,a,y)
      implicit none
      PetscScalar x(0:*),a(0:*),y(*)
      PetscInt    n,ii(*),jj(0:*)

      PetscInt i,j,jstart,jend
      PetscScalar  sum

      jend  = ii(1)
      do 10,i=1,n
        jstart = jend
        jend   = ii(i+1)
        sum    = 0.d0
        do 20 j=jstart,jend-1
          sum = sum + a(j)*x(jj(j))
 20     continue
        y(i) = sum
 10   continue

      return
      end

