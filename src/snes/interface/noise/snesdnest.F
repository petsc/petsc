      subroutine dnest(nf,fval,h,fnoise,fder2,hopt,info,eps)

#include "../../../include/FINCLUDE/petsc.h"

      integer nf, info
      Double h, fnoise, fder2, hopt
      Double fval(nf), eps(nf)
c     **********
c
c     Changes made by Lois:
c
c     Modified the code so that it can run on machines using single precision
c     (e.g., the Cray t3e).  Introduced the include file petsc.h, and
c        - changed "double precision" -> "Double"
c        - changed constants of the form 0.0d0 -> PetscDoubleExp(0.0,0)
c     These are simply the usual double precision representations on machines
c     other than Crays.
c    
c     **********
c
c
c     Subroutine dnest
c
c     This subroutine estimates the noise in a function
c     and provides estimates of the optimal difference parameter
c     for a forward-difference approximation.
c
c     The user must provide a difference parameter h, and the
c     function value at nf points centered around the current point.
c     For example, if nf = 7, the user must provide
c
c        f(x-2*h), f(x-h), f(x), f(x+h),  f(x+2*h),
c
c     in the array fval. The use of nf = 7 function evaluations is 
c     recommended.
c
c     The noise in the function is roughly defined as the variance in
c     the computed value of the function. The noise in the function
c     provides valuable information. For example, function values
c     smaller than the noise should be considered to be zero.
c
c     This subroutine requires an initial estimate for h. Under estimates
c     are usually preferred. If noise is not detected, the user should
c     increase or decrease h according to the ouput value of info.
c     In most cases, the subroutine detects noise with the initial
c     value of h.
c
c     The subroutine statement is
c
c       subroutine dnest(nf,fval,h,hopt,fnoise,info,eps)
c
c     where
c
c       nf is an integer variable.
c         On entry nf is the number of function values.
c         On exit nf is unchanged.
c
c       f is a double precision array of dimension nf.
c         On entry f contains the function values.
c         On exit f is overwritten.
c
c       h is a double precision variable.
c         On entry h is an estimate of the optimal difference parameter.
c         On exit h is unchanged.
c
c       fnoise is a double precision variable.
c         On entry fnoise need not be specified.
c         On exit fnoise is set to an estimate of the function noise
c            if noise is detected; otherwise fnoise is set to zero.
c
c       hopt is a double precision variable.
c         On entry hopt need not be specified.
c         On exit hopt is set to an estimate of the optimal difference
c            parameter if noise is detected; otherwise hopt is set to zero.
c
c       info is an integer variable.
c         On entry info need not be specified.
c         On exit info is set as follows:
c
c            info = 1  Noise has been detected.
c
c            info = 2  Noise has not been detected; h is too small.
c                      Try 100*h for the next value of h.
c
c            info = 3  Noise has not been detected; h is too large.
c                      Try h/100 for the next value of h.
c
c            info = 4  Noise has been detected but the estimate of hopt
c                      is not reliable; h is too small.
c
c       eps is a double precision work array of dimension nf.
c
c     MINPACK-2 Project. April 1997.
c     Argonne National Laboratory.
c     Jorge J. More'.
c
c     **********
      Double zero
      parameter(zero=PetscDoubleExp(0.0,0))

      integer i, j, mh
      logical dnoise
      logical cancel(6), dsgn(6)
      Double err2, est1, est2, est3, est4
      Double emin, emax, fmax, fmin, scale, stdv
      Double const(15)
      data const/PetscDoubleExp(0.71,0), PetscDoubleExp(0.41,0), 
     +           PetscDoubleExp(0.23,0), PetscDoubleExp(0.12,0),
     +           PetscDoubleExp(0.63,-1), PetscDoubleExp(0.33,-1),
     +           PetscDoubleExp(0.18,-1), PetscDoubleExp(0.89,-2),
     +           PetscDoubleExp(0.46,-2), PetscDoubleExp(0.24,-2),
     +           PetscDoubleExp(0.12,-2), PetscDoubleExp(0.61,-3),
     +           PetscDoubleExp(0.31,-3), PetscDoubleExp(0.16,-3),
     +           PetscDoubleExp(0.80,-4)/

c      data const/0.71d0, 0.41d0, 0.23d0, 0.12d0, 0.63d-1, 0.33d-1,
c     +     0.18d-1, 0.89d-2, 0.46d-2, 0.24d-2, 0.12d-2, 0.61d-3,
c     +     0.31d-3, 0.16d-3, 0.80d-4/

      fnoise = zero
      fder2 = zero
      hopt = zero

c     Compute an estimate of the second derivative and
c     determine a bound on the error.

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

c     Compute the range of function values.

      fmin = fval(1)
      fmax = fval(1)
      do i = 2, nf
         fmin = min(fmin,fval(i))
         fmax = max(fmax,fval(i))
      end do

c     Construct the difference table.

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

c        Compute the estimates for the noise level.

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

c        Determine differences in sign.

         do i = 1, nf - j - 1
            if (min(fval(i),fval(i+1)) .lt. zero .and.
     +          max(fval(i),fval(i+1)) .gt. zero) dsgn(j) = .true.
         end do
      end do

c     First requirement for detection of noise.

      dnoise = dsgn(4)

c     Check for h too small or too large.

      info = 0
      if (fmax .eq. fmin) then
         info = 2
      else if (fmax-fmin .gt. 0.1*min(abs(fmax),abs(fmin))) then
         info = 3
      end if

      if (info .ne. 0) return

c     Determine the noise level.

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

c     Noise not detected; decide if h is too small or too large.

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

c     If there is cancelllation on the third and fourth column
c     then h is too small

      info = 2

c      if (cancel .or. dsgn(3)) then
c         info = 2
c      else
c         info = 3
c      end if



      end
