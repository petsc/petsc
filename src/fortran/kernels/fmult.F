
C
C  "$Id: fmult.F,v 1.1 1997/09/17 21:22:19 bsmith Exp bsmith $";
C
C    Fortran kernel for sparse matrix vector product in AIJ format
C
C
      subroutine FortranMultAIJ(n,x,ii,jj,a,y)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar           x(0:*),a(0:*),y(*)
      integer          n, ii(*), jj(0:*)

      integer i,j,jstart,jend,cnt
      Scalar  sum

      cnt   = 0
      jend  = ii(1)
      do 10, i=1,n
        jstart = jend
        jend   = ii(i+1)
        sum    = 0.d0
        do 20 j=jstart,jend-1
          sum = sum + a(cnt)*x(jj(cnt))
          cnt = cnt + 1
 20     continue
        y(i) = sum
 10   continue

      return 
      end

