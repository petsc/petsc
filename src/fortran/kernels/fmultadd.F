C
C  "$Id: fmultadd.F,v 1.2 1997/09/18 03:07:08 balay Exp curfman $";
C
C    Fortran kernel for sparse matrix-vector product in AIJ format
C
C
      subroutine FortranMultAddAIJ(n,x,ii,jj,a,y,z)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar           x(0:*),a(0:*),y(0:*),z(*)
      integer          n, ii(*), jj(0:*)

      integer i,j,jstart,jend,cnt
      Scalar  sum

      cnt   = 0
      jend  = ii(1)
      do 10, i=1,n
        jstart = jend
        jend   = ii(i+1)
        sum    = y(i)
        do 20 j=jstart,jend-1
          sum = sum + a(cnt)*x(jj(cnt))
          cnt = cnt + 1
 20     continue
        z(i) = sum
 10   continue

      return 
      end

