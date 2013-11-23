      subroutine nlsfun(n,x,m,fvec)
      integer n, m
      double precision fvec(m), x(n)
c     *********
c
c     Subroutine nlsfun
c
c     This subroutine returns a vector of residuals simular to the
c     smooth problem class defined in:
c
c     Benchmarking Derivative-Free Optimization Algorithms
c     Jorge J. More' and Stefan M. Wild
c     SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009.
c
c     The subroutine returns the residuals fvec(x)
c
c     The subroutine statement is
c
c       subroutine calfun(n,x,m,fvec)
c
c     where
c
c       n is a positive integer input variable.
c
c       x is an input array of length n.
c
c       m is a positive integer input variable.
c
c       fvec is an output array of length m that contains 
c         the function value at x.
c
c
c     Additional problem descriptors are passed through the common block
c     calfun_int containing:
c       nprob is a positive integer that defines the number of the problem.
c          nprob must not exceed 22.
c
c     To store the evaluation history, additional variables are passed 
c     through the common blocks calfun_int and calfun_fevals. These
c     may be commented out if a user desires. They are:
c       nfev is a non-negative integer containing the number of function 
c          evaluations done so far (nfev=0 is a good default).
c          nfev should be less than 1500 unless fevals is modified.
c          after calling calfun, nfev will be incremented by one.
c       np is a counter for the test problem number. np=1 is a good
c          default if only a single problem/run will be done.
c          np should be no bigger than 100 unless fevals is modified.
c       fevals(1500,100) is an array containing the history of function
c          values, the entry fevals(nfev+1,np) being updated here.
c
c     Argonne National Laboratory
c     Jorge More' and Stefan Wild. December 2009.
c     **********

      double precision dasum, dnrm2

c     Function value array
      double precision f

      integer nprob, nfev, np
      common /calfun_int/ nprob, nfev, np
      double precision fevals(1500,100)
      common /calfun_fevals/ fevals

      call dfovec(m,n,x,fvec,nprob)
      f = dnrm2(m,fvec,1)**2

      call wallclock(wctime2)
     
      
      if (f .ge. 1.0d64 ) then
          f = 1.0d64
      end if

      if (f .ne. f ) then
          f = 1.0d64
      end if
      
      print *,'nfev=',nfev,' f=',f
      nfev = nfev + 1
      fevals(nfev,np) = f

      end
