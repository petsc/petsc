!
!  "$Id: fnorm.F,v 1.3 1997/09/18 01:50:16 bsmith Exp balay $";
!
!    Fortran kernel for the Norm() vector routine
!
!
      subroutine FortranNormSqr(x, n,sum1)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar           x(*)
      double precision sum1
      integer n

      integer i

      do 10, i=1,n
        sum1 = sum1 + PetscReal(x(i)*PetscConj(x(i)))
 10   continue

      return 
      end

