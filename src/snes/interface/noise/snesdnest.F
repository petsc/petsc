!
!  "$Id: snesdnest.F,v 1.5 1997/07/21 13:25:18 curfman Exp balay $";
!
!  Noise estimation routine, written by Jorge More'.  Details are below.
!

      subroutine dnest(nf,fval,h,fnoise,fder2,hopt,info,eps)
      implicit none

      integer nf, info
      double precision h, fnoise, fder2, hopt
      double precision fval(nf), eps(nf)
!     **********
!
!     Subroutine dnest
!
!     This subroutine estimates the noise in a function
!     and provides estimates of the optimal difference parameter
!     for a forward-difference approximation.
!
!     The user must provide a difference parameter h, and the
!     function value at nf points centered around the current point.
!     For example, if nf = 7, the user must provide
!
!        f(x-2*h), f(x-h), f(x), f(x+h),  f(x+2*h),
!
!     in the array fval. The use of nf = 7 function evaluations is 
!     recommended.
!
!     The noise in the function is roughly defined as the variance in
!     the computed value of the function. The noise in the function
!     provides valuable information. For example, function values
!     smaller than the noise should be considered to be zero.
!
!     This subroutine requires an initial estimate for h. Under estimates
!     are usually preferred. If noise is not detected, the user should
!     increase or decrease h according to the ouput value of info.
!     In most cases, the subroutine detects noise with the initial
!     value of h.
!
!     The subroutine statement is
!
!       subroutine dnest(nf,fval,h,hopt,fnoise,info,eps)
!
!     where
!
!       nf is an integer variable.
!         On entry nf is the number of function values.
!         On exit nf is unchanged.
!
!       f is a double precision array of dimension nf.
!         On entry f contains the function values.
!         On exit f is overwritten.
!
!       h is a double precision variable.
!         On entry h is an estimate of the optimal difference parameter.
!         On exit h is unchanged.
!
!       fnoise is a double precision variable.
!         On entry fnoise need not be specified.
!         On exit fnoise is set to an estimate of the function noise
!            if noise is detected; otherwise fnoise is set to zero.
!
!       hopt is a double precision variable.
!         On entry hopt need not be specified.
!         On exit hopt is set to an estimate of the optimal difference
!            parameter if noise is detected; otherwise hopt is set to zero.
!
!       info is an integer variable.
!         On entry info need not be specified.
!         On exit info is set as follows:
!
!            info = 1  Noise has been detected.
!
!            info = 2  Noise has not been detected; h is too small.
!                      Try 100*h for the next value of h.
!
!            info = 3  Noise has not been detected; h is too large.
!                      Try h/100 for the next value of h.
!
!            info = 4  Noise has been detected but the estimate of hopt
!                      is not reliable; h is too small.
!
!       eps is a double precision work array of dimension nf.
!
!     MINPACK-2 Project. April 1997.
!     Argonne National Laboratory.
!     Jorge J. More'.
!
!     **********
      double precision zero
      parameter(zero=0.0d0)

      integer i, j, mh
      logical dnoise
      logical cancel(6), dsgn(6)
      double precision err2, est1, est2, est3, est4
      double precision emin, emax, fmax, fmin, scale, stdv
      double precision const(15)
      data const/0.71d0, 0.41d0, 0.23d0, 0.12d0, 0.63d-1, 0.33d-1,
     +     0.18d-1, 0.89d-2, 0.46d-2, 0.24d-2, 0.12d-2, 0.61d-3,
     +     0.31d-3, 0.16d-3, 0.80d-4/

      fnoise = zero
      fder2 = zero
      hopt = zero

!     Compute an estimate of the second derivative and
!     determine a bound on the error.

      mh = (nf+1)/2
      est1 = ((fval(mh+1)-2*fval(mh)+fval(mh-1))/h)/h
      est2 = ((fval(mh+2)-2*fval(mh)+fval(mh-2))/(2*h))/(2*h)
      est3 = ((fval(mh+3)-2*fval(mh)+fval(mh-3))/(3*h))/(3*h)
      est4 = (est1+est2+est3)/3
      err2 = max(max(est1,est2,est3)-est4,est4-min(est1,est2,est3))
      write (2,123) est1, est2, est3
 123  format ('Second derivative estimates', 3d12.2)
      if (err2 .le. 0.1*abs(est4)) then
         fder2 = est4
      else if (err2 .lt. abs(est4)) then
         fder2 = est3
      else
         fder2 = zero
      end if         

!     Compute the range of function values.

      fmin = fval(1)
      fmax = fval(1)
      do i = 2, nf
         fmin = min(fmin,fval(i))
         fmax = max(fmax,fval(i))
      end do

!     Construct the difference table.

      dnoise = .false.
      do j = 1, 6
         dsgn(j) = .false.
         cancel(j) = .false.
         scale = zero
         do i = 1, nf - j
            fval(i) = fval(i+1) - fval(i)
            if (fval(i) .eq. zero) cancel(j) = .true.
            scale = max(scale,abs(fval(i)))
         end do

!        Compute the estimates for the noise level.

         if (scale .eq. zero) then
            stdv = zero
         else
            stdv = zero
            do i = 1, nf - j
               stdv = stdv + (fval(i)/scale)**2
            end do
            stdv = scale*sqrt(stdv/(nf-j))
         end if
         eps(j) = const(j)*stdv

!        Determine differences in sign.

         do i = 1, nf - j - 1
            if (min(fval(i),fval(i+1)) .lt. zero .and.
     +          max(fval(i),fval(i+1)) .gt. zero) dsgn(j) = .true.
         end do
      end do

!     First requirement for detection of noise.

      dnoise = dsgn(4)

!     Check for h too small or too large.

      info = 0
      if (fmax .eq. fmin) then
         info = 2
      else if (fmax-fmin .gt. 0.1*min(abs(fmax),abs(fmin))) then
         info = 3
      end if

      if (info .ne. 0) return

!     Determine the noise level.

      emin = min(eps(4),eps(5),eps(6))
      emax = max(eps(4),eps(5),eps(6))
      if (emax .le. 4*emin .and. dnoise) then
         fnoise = (eps(4)+eps(5)+eps(6))/3
         if (fder2 .ne. zero) then
            info = 1
            hopt = 1.68*sqrt(fnoise/abs(fder2))
         else
            info = 4
            hopt = 10*h
         end if
         return
      end if

      emin = min(eps(3),eps(4),eps(5))
      emax = max(eps(3),eps(4),eps(5))
      if (emax .le. 4*emin .and. dnoise) then
         fnoise = (eps(3)+eps(4)+eps(5))/3
         if (fder2 .ne. zero) then
            info = 1
            hopt = 1.68*sqrt(fnoise/abs(fder2))
         else
            info = 4
            hopt = 10*h
         end if
         return
      end if

!     Noise not detected; decide if h is too small or too large.

      if (.not. cancel(4)) then
         if (dsgn(4)) then
            info = 2
         else
            info = 3
         end if
         return
      end if
      if (.not. cancel(3)) then
         if (dsgn(3)) then
            info = 2
         else
            info = 3
         end if
         return
      end if

!     If there is cancelllation on the third and fourth column
!     then h is too small

      info = 2

!      if (cancel .or. dsgn(3)) then
!         info = 2
!      else
!         info = 3
!      end if



      end
