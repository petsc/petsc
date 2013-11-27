
      program driver

      integer mmax, nmax, nfmax, npmax
      parameter(mmax = 1000, nmax = 100, nfmax = 1500, npmax=100)
      integer lmax
      parameter (lmax = (npmax+11)*(npmax+nmax)+nmax*(3*nmax+11)/2)
      integer nread, nwrite
      parameter(nread=1,nwrite=2)
c     **********
c
c     Driver for POUNDERS
c
c     Subprograms called
c
c       USER ........... dfovec, dfoxs, dfonm
c
c       MINPACK-2 ...... wallclock
c
c       Level 1 BLAS ... dnrm2, surn01
c
c     Benchmarking DFO algorithms. July 2007.
c     Argonne National Laboratory.
c     Jorge J. More' and Stefan Wild.
c
c     **********
      double precision zero, ten
      parameter(zero=0.0d0,ten=1.0d1)

      integer n, m
      double precision f
      double precision x(nmax)
      double precision fvec(mmax)
      double precision wa(lmax)

c     Parameters for POUNDERS.
      integer npmax, maxfev
      double precision delta, gtol

c     Function value array
      integer np
      double precision fevals(nfmax,npmax)

c     Summary information.
      integer  nfev
      character *30 inform

c     Test problems.
      character*30 dfonm
      double precision factor
      double precision dasum, dnrm2

      common /calfun_int/ nprob, nfev, np
      common /calfun_fevals/ fevals

c     Seed for random number generator
      integer nseed

c     Timing.
      double precision wctime1, wctime2, time1, time2
      double precision wctime

      external dfovec, dfoxs, wallclock, dnrm2

      open (nread,file='dfo.dat',status='old')
      open (nwrite,file='pounders.info')

c     Performance file
      open (10,file='pounders.dat')

c     Define maxfev.
      maxfev = 1500

c     Variable np tracks the number of problems in the benchmark.
      np = 0

      do while (1. eq. 1)
      
c        Read problem data.
         read (nread,*) nprob, n, m, nstart
         
         if (nprob .eq. 0) then
            do i = 1, maxfev
                write(10,5000) (fevals(i,j),j= 1,np)
            end do
            close(nread)
            stop
         end if

         np = np + 1

c        Start random number generator, if needed
         nseed = 1

c        Generate the starting point.
         factor = ten**(nstart)
         call dfoxs(n,x,nprob,factor)

c        Compute the function at the starting point.
         nfev = 0 
         call nlsfun(n,x,m,fvec)
         f = dnrm2(m,fvec,1)**2
         nfev = 0 

c        Start of search.
         write (nwrite,1000) np, dfonm(nprob), m, n
         write (*,1000) np, dfonm(nprob), m, n

         write (nwrite,2000) f
         write (*,2000) f

c        Set parameters.
         nprint = 0
         delta = 0.0d0
         do i = 1, n
            delta = max(delta,abs(x(i)))
         end do
         delta = max(delta,1.0)
         gtol = 1.0d-16
         
c        Call POUNDERS.
         call wallclock(wctime1)
         nq = ((n+1)*(n+2))/2
         npmax = 2*n+1
         npmax = min(npmax,nq)
         call pounders(nlsfun,x,n,npmax,maxfev,gtol,delta,m)
         call wallclock(wctime2)

c        Summary information.
         call nlsfun(n,x,m,fvec)
         f = dnrm2(m,fvec,1)**2
 
         write (nwrite,3000) nfev, nfev/dble(n), f
         write (*,3000) nfev, nfev/dble(n), f


c        Determine exit status.
         if (nfev .ge. maxfev) then
            inform = '           nfev >= maxfev'
         else 
            inform = '           nfev <  maxfev'
         end if

c        Test for NaN's in the function values.
         nanf = 0
         do i = maxfev, 1, -1
            if (fevals(i,np) .ne. fevals(i,np)) then 
               nanf = i
            end if
         end do

         if (nanf .gt. 0) then
            inform = '      NaNs in the function values'
c            do i = nanf, maxfev
c               fevals(i,np) = 0.0
c            end do
         end if

c        Fill the fevals array if the algorithm terminates early.
         if (nfev .lt. maxfev) then
            do i = nfev+1, maxfev
               fevals(i,np) = 0.0
            end do
         end if
         
c        Timing information.
         wctime = wctime2 - wctime1
         write (nwrite,4000) inform
         write (*,4000) inform

      end do

 1000 format (' Problem ', i2, ': ', a30,                      //,
     +        ' Number of components                        ',i12,/,
     +        ' Number of variables                         ',i12)

 2000 format (
     +        ' Function value at initial iterate        '   ,d15.8)

 3000 format (
     +        ' Number of function evaluations              ',i12,/,
     +        ' Number of gradient evaluations              ',f12.2,/,
     +        ' Function value at final iterate          '   ,d15.8)

 4000 format (' Exit message                   '             ,a30/)

 5000 format(100e25.16)

      end
