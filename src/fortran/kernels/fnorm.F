!
!  "$Id: fnorm.F,v 1.4 1998/03/25 17:13:06 balay Exp balay $";
!
!    Fortran kernel for the Norm() vector routine
!
!
      subroutine FortranNormSqr(x, n,sum1)
      implicit none
#include "include/finclude/petsc.h"

      Scalar           x(*)
      double precision sum1
      integer n

      integer i

      do 10, i=1,n
        sum1 = sum1 + PetscReal(x(i)*PetscConj(x(i)))
 10   continue

      return 
      end

