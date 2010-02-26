!
!
!    Fortran kernel for sparse matrix-vector product in the AIJ format
!
#include "finclude/petscsysdef.h"
!
      subroutine FortranMultAddAIJ(n,x,ii,jj,a,y,z)
      implicit none
      PetscScalar      x(0:*),a(0:*),y(*),z(*)
      PetscInt          n,ii(*),jj(0:*)

      PetscInt i,j,jstart,jend
      PetscScalar  sum

      jend  = ii(1)
      do 10,i=1,n
        jstart = jend
        jend   = ii(i+1)
        sum    = y(i)
        do 20 j=jstart,jend-1
          sum = sum + a(j)*x(jj(j))
 20     continue
        z(i) = sum
 10   continue

      return 
      end

