!
!  "$Id: fnorm.F,v 1.5 1998/04/15 18:03:20 balay Exp balay $";
!
!    Fortran kernel for the Norm() vector routine
!
!
      subroutine FortranNormSqr(x, n,sum1)
      implicit none
#include "include/finclude/petscdef.h"

      Scalar           x(*)
      double precision sum1
      integer n

      integer i

      do 10, i=1,n
        sum1 = sum1 + PetscReal(x(i)*PetscConj(x(i)))
 10   continue

      return 
      end

