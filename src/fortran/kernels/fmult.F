!
!  "$Id: fmult.F,v 1.4 1997/12/07 04:35:22 bsmith Exp balay $";
!
!    Fortran kernel for sparse matrix-vector product in the AIJ matrix format
!
!
      subroutine FortranMultAIJ(n,x,ii,jj,a,y)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar           x(0:*),a(0:*),y(*)
      integer          n, ii(*), jj(0:*)

      integer i,j,jstart,jend
      Scalar  sum

      jend  = ii(1)
      do 10, i=1,n
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

