!
!  "$Id: fmultadd.F,v 1.10 1999/03/24 18:13:04 balay Exp bsmith $";
!
!    Fortran kernel for sparse matrix-vector product in the AIJ format
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranMultAddAIJ(n,x,ii,jj,a,y,z)
      implicit none
      Scalar           x(0:*),a(0:*),y(*),z(*)
      integer          n,ii(*),jj(0:*)

      integer i,j,jstart,jend
      Scalar  sum

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

