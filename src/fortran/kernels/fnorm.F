!
!  "$Id: fnorm.F,v 1.6 1998/07/13 18:14:51 balay Exp balay $";
!
!    Fortran kernel for the Norm() vector routine
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranNormSqr(x, n,sum1)
      implicit none
      Scalar           x(*)
      double precision sum1
      integer n

      integer i

      do 10, i=1,n
        sum1 = sum1 + PetscReal(x(i)*PetscConj(x(i)))
 10   continue

      return 
      end

