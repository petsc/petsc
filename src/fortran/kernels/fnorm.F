!
!  "$Id: fnorm.F,v 1.8 2000/01/11 21:04:00 bsmith Exp balay $";
!
!    Fortran kernel for the Norm() vector routine
!
#include "include/finclude/petscdef.h"
!
      subroutine FortranNormSqr(x,n,sum1)
      implicit none
      PetscScalar      x(*)
      double precision sum1
      integer n

      integer i

      do 10,i=1,n
        sum1 = sum1 + PetscRealPart(x(i)*PetscConj(x(i)))
 10   continue

      return 
      end

