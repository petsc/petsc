!
!  "$Id: fmultadd.F,v 1.8 1998/04/15 18:03:24 balay Exp balay $";
!
!    Fortran kernel for sparse matrix-vector product in the AIJ format
!
!
      subroutine FortranMultAddAIJ(n,x,ii,jj,a,y,z)
      implicit none
#include "include/finclude/petscdef.h"

      Scalar           x(0:*),a(0:*),y(*),z(*)
      integer          n, ii(*), jj(0:*)

      integer i,j,jstart,jend
      Scalar  sum

      jend  = ii(1)
      do 10, i=1,n
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

