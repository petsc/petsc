
C
C  "$Id: fmdot.F,v 1.1 1997/09/17 02:30:33 bsmith Exp bsmith $";
C
C    Fortran kernel for the MDot() vector routine
C
C
      subroutine FortranMDot4(x, y1, y2, y3, y4, n,sum1,sum2,sum3,sum4)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar  sum1, sum2, sum3, sum4, x(*), y1(*), y2(*)
      Scalar  y3(*), y4(*)
      integer n

      integer i

      do 10, i=1,n
        sum1 = sum1 + x(i)*dconjg(y1(i))
        sum2 = sum2 + x(i)*dconjg(y2(i))
        sum3 = sum3 + x(i)*dconjg(y3(i))
        sum4 = sum4 + x(i)*dconjg(y4(i))
 10   continue

      return 
      end

      subroutine FortranMDot3(x, y1, y2, y3, n,sum1,sum2,sum3)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar  sum1, sum2, sum3, x(*), y1(*), y2(*), y3(*)
      integer n

      integer i

      do 10, i=1,n
        sum1 = sum1 + x(i)*dconjg(y1(i))
        sum2 = sum2 + x(i)*dconjg(y2(i))
        sum3 = sum3 + x(i)*dconjg(y3(i))
 10   continue

      return 
      end

      subroutine FortranMDot2(x, y1, y2, n,sum1,sum2)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar  sum1, sum2, x(*), y1(*), y2(*)
      integer n

      integer i

      do 10, i=1,n
        sum1 = sum1 + x(i)*dconjg(y1(i))
        sum2 = sum2 + x(i)*dconjg(y2(i))
 10   continue

      return 
      end


      subroutine FortranMDot1(x, y1, n,sum1)
      implicit none
#include "include/FINCLUDE/petsc.h"

      Scalar  sum1, x(*), y1(*)
      integer n

      integer i

      do 10, i=1,n
        sum1 = sum1 + x(i)*dconjg(y1(i))
 10   continue

      return 
      end
