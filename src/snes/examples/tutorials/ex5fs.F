C "$Id: ex5fs.F,v 1.1 1997/11/24 23:07:57 bsmith Exp bsmith $";
C

C ---------------------------------------------------------------------
C
C  Fortran version of the user function based on shared memory
C  this routine is called only by MPI process 0 in the computation
C  but uses threads to run the loops in parallel.

C
C  Input Parameter:
C  x - local vector data
C
C  Output Parameters:
C  f - local vector data, f(x)
C  ierr - error code 
C
C  Notes:
C  This routine uses standard Fortran-style computations over a 2-dim array.
C
      subroutine ApplicationFunction(m,n,x,f,ierr)

      implicit none

#include "ex5f.h"

C  Input/output variables:
      Scalar   x(m,n), f(m,n)
      integer  ierr

C  Local variables:
      Scalar   two, one, hx, hy, hxdhy, hydhx, sc
      Scalar   u, uxx, uyy
      integer  i, j

      one    = 1.0
      two    = 2.0
      hx     = one/dble(mx-1)
      hy     = one/dble(my-1)
      sc     = hx*hy*lambda
      hxdhy  = hx/hy
      hydhx  = hy/hx

C  Compute function over the locally owned part of the grid

      do 20 j=ys,ye
         do 10 i=xs,xe
            if (i .eq. 1 .or. j .eq. 1
     &             .or. i .eq. mx .or. j .eq. my) then
               f(i,j) = x(i,j)
            else
               u = x(i,j) 
               uxx = hydhx * (two*u
     &                - x(i-1,j) - x(i+1,j))
               uyy = hxdhy * (two*u - x(i,j-1) - x(i,j+1))
               f(i,j) = uxx + uyy - sc*exp(u)
            endif
 10      continue
 20   continue

      return
      end

