!
!  "$Id: sgemv.F,v 1.1 1998/12/14 16:25:12 bsmith Exp bsmith $";
!
!    Fortran kernel for gemv() BLAS operation. This version supports
!  matrix array stored in single precision but vectors in double
!
!
      subroutine MSGemv(bs,ncols,A,x,y)
      implicit none
#include "include/finclude/petscdef.h"

      integer          bs,ncols
      MatScalar        a(bs,ncols)
      Scalar           x(ncols),y(bs)

      integer          i,j

      do 5, j=1,bs
        y(j) = 0.0d0
 5    continue

      do 10, i=1,ncols
        do 20, j=1,bs
          y(j) = y(j) + A(j,i)*x(i)
 20     continue
 10   continue

      return 
      end

      subroutine MSGemvp(bs,ncols,A,x,y)
      implicit none
#include "include/finclude/petscdef.h"

      integer          bs,ncols
      MatScalar        a(bs,ncols)
      Scalar           x(ncols),y(bs)

      integer          i,j

      do 10, i=1,ncols
        do 20, j=1,bs
          y(j) = y(j) + A(j,i)*x(i)
 20     continue
 10   continue

      return 
      end

      subroutine MSGemvm(bs,ncols,A,x,y)
      implicit none
#include "include/finclude/petscdef.h"

      integer          bs,ncols
      MatScalar        a(bs,ncols)
      Scalar           x(ncols),y(bs)

      integer          i,j

      do 10, i=1,ncols
        do 20, j=1,bs
          y(j) = y(j) - A(j,i)*x(i)
 20     continue
 10   continue

      return 
      end

      subroutine MSGemvt(bs,ncols,A,x,y)
      implicit none
#include "include/finclude/petscdef.h"

      integer          bs,ncols
      MatScalar        a(bs,ncols)
      Scalar           x(bs),y(ncols)

      integer          i,j
      Scalar           sum

      do 10, i=1,ncols
        sum = y(i)
        do 20, j=1,bs
          sum = sum + A(j,i)*x(j)
 20     continue
        y(i) = sum
 10   continue

      return 
      end

      subroutine MSGemm(bs,A,B,C)
      implicit none
#include "include/finclude/petscdef.h"

      integer          bs
      MatScalar        A(bs,bs),B(bs,bs),C(bs,bs)
      Scalar           sum

      integer          i,j,k

      do 10, i=1,bs
        do 20, j=1,bs
          sum = A(i,j)
          do 30, k=1,bs
            sum = sum - B(i,k)*C(k,j)
 30       continue
          A(i,j) = sum
 20     continue
 10   continue

      return 
      end

      subroutine MSGemmi(bs,A,C,B)
      implicit none
#include "include/finclude/petscdef.h"

      integer          bs
      MatScalar        A(bs,bs),B(bs,bs),C(bs,bs)
      Scalar           sum

      integer          i,j,k

      do 10, i=1,bs
        do 20, j=1,bs
          sum = 0.0d0
          do 30, k=1,bs
            sum = sum + B(i,k)*C(k,j)
 30       continue
          A(i,j) = sum
 20     continue
 10   continue

      return 
      end
