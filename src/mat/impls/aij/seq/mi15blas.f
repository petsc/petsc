*********************************************************************
*
*     File  mi15blas fortran.
*
*     dasum    daxpy    dcopy    ddot     dnrm2    dscal    idamax
*
*     These correspond to members of the BLAS package (Basic Linear
*     Algebra Subprograms, Lawson et al. (1979), ACM TOMS 5, 3).
*     If possible, they should be replaced by authentic BLAS routines
*     tuned to the machine being used.
*
*     The following utilities are also used:
*
*     dddiv    ddscl    dload    dnorm1
*     hcopy    hload    icopy    iload    iload1
*
*     These could be tuned to the machine being used.
*     dload  is used the most.
*
*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

*** dasum thru idamax taken
*** from netlib, Thu May 16 21:00:13 EDT 1991 ***
*** Declarations of the form dx(1) changed to dx(*).
*** dabs(*), dsqrt(*) changed to generic intrinsics abs(*), sqrt(*).
*** Constants 0.0d0, 1.0d0 changed to 0.0d+0, 1.0d+0.

      double precision function dasum(n,dx,incx)
c
c     takes the sum of the absolute values.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(*),dtemp
      integer i,incx,m,mp1,n,nincx
c
      dasum = 0.0d+0
      dtemp = 0.0d+0
      if(n.le.0)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      nincx = n*incx
      do 10 i = 1,nincx,incx
        dtemp = dtemp + abs(dx(i))
   10 continue
      dasum = dtemp
      return
c
c        code for increment equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,6)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dtemp = dtemp + abs(dx(i))
   30 continue
      if( n .lt. 6 ) go to 60
   40 mp1 = m + 1
      do 50 i = mp1,n,6
        dtemp = dtemp + abs(dx(i)) + abs(dx(i + 1)) + abs(dx(i + 2))
     *  + abs(dx(i + 3)) + abs(dx(i + 4)) + abs(dx(i + 5))
   50 continue
   60 dasum = dtemp
      return
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine daxpy(n,da,dx,incx,dy,incy)
c
c     constant times a vector plus a vector.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(*),dy(*),da
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if (da .eq. 0.0d+0) return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dy(iy) + da*dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,4)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dy(i) = dy(i) + da*dx(i)
   30 continue
      if( n .lt. 4 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,4
        dy(i) = dy(i) + da*dx(i)
        dy(i + 1) = dy(i + 1) + da*dx(i + 1)
        dy(i + 2) = dy(i + 2) + da*dx(i + 2)
        dy(i + 3) = dy(i + 3) + da*dx(i + 3)
   50 continue
      return
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine  dcopy(n,dx,incx,dy,incy)
c
c     copies a vector, x, to a vector, y.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(*),dy(*)
      integer i,incx,incy,ix,iy,m,mp1,n
c
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dy(iy) = dx(ix)
        ix = ix + incx
        iy = iy + incy
   10 continue
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,7)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dy(i) = dx(i)
   30 continue
      if( n .lt. 7 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,7
        dy(i) = dx(i)
        dy(i + 1) = dx(i + 1)
        dy(i + 2) = dx(i + 2)
        dy(i + 3) = dx(i + 3)
        dy(i + 4) = dx(i + 4)
        dy(i + 5) = dx(i + 5)
        dy(i + 6) = dx(i + 6)
   50 continue
      return
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      double precision function ddot(n,dx,incx,dy,incy)
c
c     forms the dot product of two vectors.
c     uses unrolled loops for increments equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(*),dy(*),dtemp
      integer i,incx,incy,ix,iy,m,mp1,n
c
      ddot = 0.0d+0
      dtemp = 0.0d+0
      if(n.le.0)return
      if(incx.eq.1.and.incy.eq.1)go to 20
c
c        code for unequal increments or equal increments
c          not equal to 1
c
      ix = 1
      iy = 1
      if(incx.lt.0)ix = (-n+1)*incx + 1
      if(incy.lt.0)iy = (-n+1)*incy + 1
      do 10 i = 1,n
        dtemp = dtemp + dx(ix)*dy(iy)
        ix = ix + incx
        iy = iy + incy
   10 continue
      ddot = dtemp
      return
c
c        code for both increments equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,5)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dtemp = dtemp + dx(i)*dy(i)
   30 continue
      if( n .lt. 5 ) go to 60
   40 mp1 = m + 1
      do 50 i = mp1,n,5
        dtemp = dtemp + dx(i)*dy(i) + dx(i + 1)*dy(i + 1) +
     *   dx(i + 2)*dy(i + 2) + dx(i + 3)*dy(i + 3) + dx(i + 4)*dy(i + 4)
   50 continue
   60 ddot = dtemp
      return
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      double precision function dnrm2 ( n, dx, incx)
      integer          next
      double precision   dx(*), cutlo, cuthi, hitest, sum, xmax,zero,one
      data   zero, one /0.0d+0, 1.0d+0/
c
c     euclidean norm of the n-vector stored in dx() with storage
c     increment incx .
c     if    n .le. 0 return with result = 0.
c     if n .ge. 1 then incx must be .ge. 1
c
c           c.l.lawson, 1978 jan 08
c
c     four phase method     using two built-in constants that are
c     hopefully applicable to all machines.
c         cutlo = maximum of  sqrt(u/eps)  over all known machines.
c         cuthi = minimum of  sqrt(v)      over all known machines.
c     where
c         eps = smallest no. such that eps + 1. .gt. 1.
c         u   = smallest positive no.   (underflow limit)
c         v   = largest  no.            (overflow  limit)
c
c     brief outline of algorithm..
c
c     phase 1    scans zero components.
c     move to phase 2 when a component is nonzero and .le. cutlo
c     move to phase 3 when a component is .gt. cutlo
c     move to phase 4 when a component is .ge. cuthi/m
c     where m = n for x() real and m = 2*n for complex.
c
c     values for cutlo and cuthi..
c     from the environmental parameters listed in the imsl converter
c     document the limiting values are as follows..
c     cutlo, s.p.   u/eps = 2**(-102) for  honeywell.  close seconds are
c                   univac and dec at 2**(-103)
c                   thus cutlo = 2**(-51) = 4.44089e-16
c     cuthi, s.p.   v = 2**127 for univac, honeywell, and dec.
c                   thus cuthi = 2**(63.5) = 1.30438e19
c     cutlo, d.p.   u/eps = 2**(-67) for honeywell and dec.
c                   thus cutlo = 2**(-33.5) = 8.23181d-11
c     cuthi, d.p.   same as s.p.  cuthi = 1.30438d19
c     data cutlo, cuthi / 8.232d-11,  1.304d19 /
c     data cutlo, cuthi / 4.441e-16,  1.304e19 /
      data cutlo, cuthi / 8.232d-11,  1.304d19 /
c
      if(n .gt. 0) go to 10
         dnrm2  = zero
         go to 300
c
   10 assign 30 to next
      sum = zero
      nn = n * incx
c                                                 begin main loop
      i = 1
   20    go to next,(30, 50, 70, 110)
   30 if( abs(dx(i)) .gt. cutlo) go to 85
      assign 50 to next
      xmax = zero
c
c                        phase 1.  sum is zero
c
   50 if( dx(i) .eq. zero) go to 200
      if( abs(dx(i)) .gt. cutlo) go to 85
c
c                                prepare for phase 2.
      assign 70 to next
      go to 105
c
c                                prepare for phase 4.
c
  100 i = j
      assign 110 to next
      sum = (sum / dx(i)) / dx(i)
  105 xmax = abs(dx(i))
      go to 115
c
c                   phase 2.  sum is small.
c                             scale to avoid destructive underflow.
c
   70 if( abs(dx(i)) .gt. cutlo ) go to 75
c
c                     common code for phases 2 and 4.
c                     in phase 4 sum is large.  scale to avoid overflow.
c
  110 if( abs(dx(i)) .le. xmax ) go to 115
         sum = one + sum * (xmax / dx(i))**2
         xmax = abs(dx(i))
         go to 200
c
  115 sum = sum + (dx(i)/xmax)**2
      go to 200
c
c
c                  prepare for phase 3.
c
   75 sum = (sum * xmax) * xmax
c
c
c     for real or d.p. set hitest = cuthi/n
c     for complex      set hitest = cuthi/(2*n)
c
   85 hitest = cuthi/float( n )
c
c                   phase 3.  sum is mid-range.  no scaling.
c
      do 95 j =i,nn,incx
      if(abs(dx(j)) .ge. hitest) go to 100
   95    sum = sum + dx(j)**2
      dnrm2 = sqrt( sum )
      go to 300
c
  200 continue
      i = i + incx
      if ( i .le. nn ) go to 20
c
c              end of main loop.
c
c              compute square root and adjust for scaling.
c
      dnrm2 = xmax * sqrt(sum)
  300 continue
      return
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine  dscal(n,da,dx,incx)
c
c     scales a vector by a constant.
c     uses unrolled loops for increment equal to one.
c     jack dongarra, linpack, 3/11/78.
c
      double precision da,dx(*)
      integer i,incx,m,mp1,n,nincx
c
      if(n.le.0)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      nincx = n*incx
      do 10 i = 1,nincx,incx
        dx(i) = da*dx(i)
   10 continue
      return
c
c        code for increment equal to 1
c
c
c        clean-up loop
c
   20 m = mod(n,5)
      if( m .eq. 0 ) go to 40
      do 30 i = 1,m
        dx(i) = da*dx(i)
   30 continue
      if( n .lt. 5 ) return
   40 mp1 = m + 1
      do 50 i = mp1,n,5
        dx(i) = da*dx(i)
        dx(i + 1) = da*dx(i + 1)
        dx(i + 2) = da*dx(i + 2)
        dx(i + 3) = da*dx(i + 3)
        dx(i + 4) = da*dx(i + 4)
   50 continue
      return
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      integer function idamax(n,dx,incx)
c
c     finds the index of element having max. absolute value.
c     jack dongarra, linpack, 3/11/78.
c
      double precision dx(*),dmax
      integer i,incx,ix,n
c
      idamax = 0
      if( n .lt. 1 ) return
      idamax = 1
      if(n.eq.1)return
      if(incx.eq.1)go to 20
c
c        code for increment not equal to 1
c
      ix = 1
      dmax = abs(dx(1))
      ix = ix + incx
      do 10 i = 2,n
         if(abs(dx(ix)).le.dmax) go to 5
         idamax = i
         dmax = abs(dx(ix))
    5    ix = ix + incx
   10 continue
      return
c
c        code for increment equal to 1
c
   20 dmax = abs(dx(1))
      do 30 i = 2,n
         if(abs(dx(i)).le.dmax) go to 30
         idamax = i
         dmax = abs(dx(i))
   30 continue
      return
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine dddiv ( n, d, incd, x, incx )

      implicit           double precision (a-h,o-z)
      double precision   d(*), x(*)

*     dddiv  performs the diagonal scaling  x  =  x / d.

      integer            i, id, ix
      parameter        ( one = 1.0d+0 )

      if (n .gt. 0) then
         if (incd .eq. 0  .and.  incx .ne. 0) then
            call dscal ( n, one/d(1), x, abs(incx) )
         else if (incd .eq. incx  .and.  incd .gt. 0) then
            do 10 id = 1, 1 + (n - 1)*incd, incd
               x(id) = x(id) / d(id)
   10       continue
         else
            if (incx .ge. 0) then
               ix = 1
            else
               ix = 1 - (n - 1)*incx
            end if
            if (incd .gt. 0) then
               do 20 id = 1, 1 + (n - 1)*incd, incd
                  x(ix) = x(ix) / d(id)
                  ix    = ix   + incx
   20          continue
            else
               id = 1 - (n - 1)*incd
               do 30  i = 1, n
                  x(ix) = x(ix) / d(id)
                  id    = id + incd
                  ix    = ix + incx
   30          continue
            end if
         end if
      end if

*     end of dddiv
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine ddscl ( n, d, incd, x, incx )

      integer            incd, incx, n
      double precision   d(*), x(*)

*     ddscl  performs the diagonal scaling  x  =  d * x.

      integer            i, id, ix

      if (n .gt. 0) then
         if (incd .eq. 0  .and.  incx .ne. 0) then
            call dscal ( n, d(1), x, abs(incx) )
         else if (incd .eq. incx  .and.  incd .gt. 0) then
            do 10 id = 1, 1 + (n - 1)*incd, incd
               x(id) = d(id)*x(id)
   10       continue
         else
            if (incx .ge. 0) then
               ix = 1
            else
               ix = 1 - (n - 1)*incx
            end if
            if (incd .gt. 0) then
               do 20 id = 1, 1 + (n - 1)*incd, incd
                  x(ix) = d(id)*x(ix)
                  ix    = ix + incx
   20          continue
            else
               id = 1 - (n - 1)*incd
               do 30  i = 1, n
                  x(ix) = d(id)*x(ix)
                  id    = id + incd
                  ix    = ix + incx
   30          continue
            end if
         end if
      end if

*     end of ddscl
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine dload ( n, const, x, incx )

      double precision   const
      integer            incx, n
      double precision   x(*)

*     dload loads elements of x with const.

      double precision   zero
      parameter        ( zero = 0.0d+0 )
      integer            ix

      if (n .gt. 0) then
         if (incx .eq. 1  .and.  const .eq. zero) then
            do 10 ix = 1, n
               x(ix) = zero
   10       continue
         else
            do 20 ix = 1, 1 + (n - 1)*incx, incx
               x(ix) = const
   20       continue
         end if
      end if

*     end of dload
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      function   dnorm1( n, x, incx )

      implicit           double precision (a-h,o-z)
      double precision   x(*)

*     dnorm1  returns the 1-norm of the vector  x,  scaled by root(n).
*     This approximates an "average" element of x with some allowance
*     for x being sparse.

      d      = n
      if (n .gt. 0) d = dasum ( n, x, incx ) / sqrt(d)
      dnorm1 = d

*     end of dnorm1
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine hcopy ( n, hx, incx, hy, incy )

      integer*4          hx(*), hy(*)
      integer            incx, incy

*     hcopy  is the half-integer version of dcopy.
*     In this version of MINOS we no longer use half integers.

      integer            ix, iy

      if (n .gt. 0) then
         if (incx .eq. incy  .and.  incy .gt. 0) then
            do 10 iy  = 1, 1 + (n - 1)*incy, incy
               hy(iy) = hx(iy)
   10       continue
         else
            if (incx .ge. 0) then
               ix = 1
            else
               ix = 1 - (n - 1)*incx
            end if
            if (incy .gt. 0) then
               do 20 iy  = 1, 1 + ( n - 1 )*incy, incy
                  hy(iy) = hx(ix)
                  ix     = ix + incx
   20          continue
            else
               iy = 1 - (n - 1)*incy
               do 30  i  = 1, n
                  hy(iy) = hx(ix)
                  iy    = iy + incy
                  ix    = ix + incx
   30          continue
            end if
         end if
      end if

*     end of hcopy
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine hload ( n, const, hx, incx )

      integer            incx, n
      integer            const
      integer*4          hx(*)

*     hload loads elements of hx with const.
*     Beware that const is INTEGER, not half integer.

      integer            ix

      if (n .gt. 0) then
         if (incx .eq. 1  .and.  const .eq. 0) then
            do 10 ix  = 1, n
               hx(ix) = 0
   10       continue
         else
            do 20 ix  = 1, 1 + (n - 1)*incx, incx
               hx(ix) = const
   20       continue
         end if
      end if

*     end of hload
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine icopy ( n, x, incx, y, incy )

      integer            x(*), y(*)
      integer            incx, incy

*     icopy  is the integer version of dcopy.

      integer            ix, iy

      if (n .gt. 0) then
         if (incx .eq. incy  .and.  incy .gt. 0) then
            do 10 iy = 1, 1 + (n - 1)*incy, incy
               y(iy) = x(iy)
   10       continue
         else
            if (incx .ge. 0) then
               ix = 1
            else
               ix = 1 - (n - 1)*incx
            end if
            if (incy .gt. 0) then
               do 20 iy = 1, 1 + ( n - 1 )*incy, incy
                  y(iy) = x(ix)
                  ix    = ix + incx
   20          continue
            else
               iy = 1 - (n - 1)*incy
               do 30 i  = 1, n
                  y(iy) = x(ix)
                  iy    = iy + incy
                  ix    = ix + incx
   30          continue
            end if
         end if
      end if

*     end of icopy
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine iload ( n, const, x, incx )

      integer            incx, n
      integer            const
      integer            x(*)

*     iload  loads elements of x with const.

      integer            ix

      if (n .gt. 0) then
         if (incx .eq. 1  .and.  const .eq. 0) then
            do 10 ix = 1, n
               x(ix) = 0
   10       continue
         else
            do 20 ix = 1, 1 + (n - 1)*incx, incx
               x(ix) = const
   20       continue
         end if
      end if

*     end of iload
      end

*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      subroutine iload1( n, const, x, incx )

      integer            incx, n
      integer            const
      integer            x(*)

*     iload1 loads elements of x with const, by calling iload.
*     iload1 is needed in MINOS because iload  is a file number.

      call iload ( n, const, x, incx )

*     end of iload1
      end
