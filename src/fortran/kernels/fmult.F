
C
C  "$Id: fnorm.F,v 1.1 1997/09/17 20:37:29 bsmith Exp $";
C
C    Fortran kernel for the Norm() vector routine
C
C
      subroutine FortranNormSqr(x, n,sum1)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar           x(*)
      double precision sum1
      integer n

      integer i

      do 10, i=1,n
        sum1 = sum1 + dreal(x(i)*dconjg(x(i)))
 10   continue

      return 
      end

