!
!  "$Id: fnorm.F,v 1.7 1999/03/24 18:13:06 balay Exp bsmith $";
!
!    Fortran kernel for the Norm() vector routine
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranNormSqr(x,n,sum1)
      implicit none
      Scalar           x(*)
      double precision sum1
      integer n

      integer i

      do 10,i=1,n
        sum1 = sum1 + PetscRealPart(x(i)*PetscConj(x(i)))
 10   continue

      return 
      end

