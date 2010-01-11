      subroutine dfoxs(n,x,nprob,factor)
      integer n, nprob
      double precision factor
      double precision x(n)
c     **********
c
c     Subroutine dfoxs
c
c     This subroutine specifies the standard starting points for the
c     functions defined by subroutine dfovec as used in:
c
c     Benchmarking Derivative-Free Optimization Algorithms
c     Jorge J. More' and Stefan M. Wild
c     SIAM J. Optimization, Vol. 20 (1), pp.172-191, 2009.
c
c     The latest version of this subroutine is always available at
c     http://www.mcs.anl.gov/~more/dfo/
c     The authors would appreciate feedback and experiences from numerical
c     studies conducted using this subroutine.
c
c     The subroutine returns
c     in x a multiple (factor) of the standard starting point.
c
c     The subroutine statement is
c
c       subroutine dfoxs(n,x,nprob,factor)
c
c     where
c
c       n is a positive integer input variable.
c
c       x is an output array of length n which contains the standard
c         starting point for problem nprob multiplied by factor.
c
c       nprob is a positive integer input variable which defines the
c         number of the problem. nprob must not exceed 22.
c
c       factor is an input variable which specifies the multiple of
c         the standard starting point.
c
c     Argonne National Laboratory. 
c     Jorge More' and Stefan Wild. September 2007.
c
c     **********
      integer i, j
      double precision sum, temp

c     Selection of initial point.

      if (nprob .le. 3) then

c        Linear function - full rank or rank 1.

         do j = 1, n
            x(j) = 1.0d0
         end do
      
      else if (nprob .eq. 4) then

c        Rosenbrock function.

         x(1) = -1.2d0
         x(2) = 1.0d0

      else if (nprob .eq. 5) then

c        Helical valley function.

         x(1) = -1.0d0
         x(2) = 0.0d0
         x(3) = 0.0d0

      else if (nprob .eq. 6) then

c        Powell singular function.

         x(1) = 3.0d0
         x(2) = -1.0d0
         x(3) = 0.0d0
         x(4) = 1.0d0

      else if (nprob .eq. 7) then

c        Freudenstein and Roth function.

         x(1) = 0.5d0
         x(2) = -2.0d0

      else if (nprob .eq. 8) then

c        Bard function.

         x(1) = 1.0d0
         x(2) = 1.0d0
         x(3) = 1.0d0

      else if (nprob .eq. 9) then

c        Kowalik and Osborne function.

         x(1) = 0.25d0
         x(2) = 0.39d0
         x(3) = 0.415d0
         x(4) = 0.39d0

      else if (nprob .eq. 10) then

c        Meyer function.

        x(1) = 0.02d0
        x(2) = 4000.0d0
        x(3) = 250.0d0

      else if (nprob .eq. 11) then

c        Watson function.

         do j = 1, n
            x(j) = 0.5d0
         end do

      else if (nprob .eq. 12) then

c        Box 3-dimensional function.

         x(1) = 0.0d0
         x(2) = 10.0d0
         x(3) = 20.0d0

      else if (nprob .eq. 13) then
 
c        Jennrich and Sampson function.

         x(1) = 0.3d0
         x(2) = 0.4d0

      else if (nprob .eq. 14) then

c        Brown and Dennis function.

         x(1) = 25.0d0
         x(2) = 5.0d0
         x(3) = -5.0d0
         x(4) = -1.0d0

      else if (nprob .eq. 15) then

c        Chebyquad function.

         do j = 1, n
            x(j) = j/dble(n+1)
         end do

      else if (nprob .eq. 16) then

c        Brown almost-linear function.

         do j = 1, n
            x(j) = 0.5d0
         end do

      else if (nprob .eq. 17) then

c        Osborne 1 function.

         x(1) = 0.5d0
         x(2) = 1.5d0
         x(3) = 1.0d0
         x(4) = 0.01d0
         x(5) = 0.02d0

      else if (nprob .eq. 18) then

c        Osborne 2 function.

         x(1) = 1.3d0
         x(2) = 0.65d0
         x(3) = 0.65d0
         x(4) = 0.7d0
         x(5) = 0.6d0
         x(6) = 3.0d0
         x(7) = 5.0d0
         x(8) = 7.0d0
         x(9) = 2.0d0
         x(10) = 4.5d0
         x(11) = 5.5d0

      else if (nprob .eq. 19) then

c        Bdqrtic function.

         do j = 1, n
            x(j) = 1.0d0
         end do
         
      else if (nprob .eq. 20) then

c        Cube function.

         do j = 1, n
            x(j) = 0.5d0
         end do
                  
      else if (nprob .eq. 21) then

c        Mancino function.

         do i = 1, n
            sum = 0.0d0 
            do j = 1, n
               temp = sqrt(dble(i)/dble(j))
               sum = sum + 
     *               temp*((sin(log(temp)))**5+(cos(log(temp)))**5)
            end do
            x(i) = -8.710996D-4*((i-50)**3 + sum)
         end do
         
      else if (nprob .eq. 22) then

c        Heart8 function.

         x(1) = -0.3d0
         x(2) = -0.39d0
         x(3) = 0.3d0
         x(4) = -0.344d0
         x(5) = -1.2d0
         x(6) = 2.69d0
         x(7) = 1.59d0
         x(8) = -1.5d0
                           
      else
         write (*,*) "Parameter nprob > 22 in subroutine dfoxs"
      end if

c     Compute multiple of initial point.

      do j = 1, n
         x(j) = factor*x(j)
      end do

      return

      end
