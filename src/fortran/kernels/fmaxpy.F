!
!  "$Id: fmaxpy.F,v 1.4 1998/04/15 18:03:35 balay Exp balay $";
!
!    Fortran kernel for the MAXPY() vector routine
!
!
      subroutine FortranMAXPY4(x, a1, a2, a3, a4, y1, y2, y3, y4, n)
      implicit none
#include "include/finclude/petsc.h"

      Scalar  a1, a2, a3, a4
      Scalar  x(*), y1(*), y2(*), y3(*), y4(*)
      integer n

      integer i

      do 10, i=1,n
        x(i) = x(i) + a1*y1(i) + a2*y2(i) + a3*y3(i) + a4*y4(i)
 10   continue

      return 
      end

      subroutine FortranMAXPY3(x, a1, a2, a3, y1, y2, y3, n)
      implicit none
#include "include/finclude/petsc.h"

      Scalar  a1, a2, a3, x(*), y1(*), y2(*), y3(*)
      integer n

      integer i

      do 10, i=1,n
        x(i) = x(i) + a1*y1(i) + a2*y2(i) + a3*y3(i)
 10   continue

      return 
      end



      subroutine FortranMAXPY2(x, a1, a2, y1, y2, n)
      implicit none
#include "include/finclude/petsc.h"

      Scalar  a1, a2, x(*), y1(*), y2(*)
      integer n

      integer i

      do 10, i=1,n
        x(i) = x(i) + a1*y1(i) + a2*y2(i)
 10   continue

      return 
      end
