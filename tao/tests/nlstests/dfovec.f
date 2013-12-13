      subroutine dfovec(m,n,x,fvec,nprob)
      integer m, n, nprob
      double precision x(n), fvec(m)
c     **********
c
c     Subroutine dfovec
c
c     This subroutine specifies the nonlinear benchmark problems in
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
c     The data file dfo.dat defines suitable values of m and n
c     for each problem number nprob.
c
c     The code for the first 18 functions in dfovec is derived 
c     from the MINPACK-1 subroutine ssqfcn
c
c     The subroutine statement is
c
c       subroutine dfovec(m,n,x,fvec,nprob)
c
c     where
c
c       m and n are positive integer input variables. 
c         n must not exceed m.
c
c       x is an input array of length n.
c
c       fvec is an output array of length m that contains the nprob
c         function evaluated at x.
c
c       nprob is a positive integer input variable which defines the
c         number of the problem. nprob must not exceed 22.
c
c     Argonne National Laboratory
c     Jorge More' and Stefan Wild. January 2008.
c
c     **********
      integer i,j
      double precision dx,one,prod,sum,temp,tmp1,tmp2,tmp3,tmp4,zero
      double precision y(11),y1(15),y2(11),y3(16),y4(33),y5(65),y6(8)
      data zero,one /0.0d0,1.0d0/
      data y(1),y(2),y(3),y(4),y(5),y(6),y(7),y(8),y(9),y(10),y(11)
     *     /4.0d0,2.0d0,1.0d0,5.0d-1,2.5d-1,1.67d-1,1.25d-1,1.0d-1,
     *      8.33d-2,7.14d-2,6.25d-2/
      data y1(1),y1(2),y1(3),y1(4),y1(5),y1(6),y1(7),y1(8),y1(9),
     *     y1(10),y1(11),y1(12),y1(13),y1(14),y1(15)
     *     /1.4d-1,1.8d-1,2.2d-1,2.5d-1,2.9d-1,3.2d-1,3.5d-1,3.9d-1,
     *      3.7d-1,5.8d-1,7.3d-1,9.6d-1,1.34d0,2.1d0,4.39d0/
      data y2(1),y2(2),y2(3),y2(4),y2(5),y2(6),y2(7),y2(8),y2(9),
     *     y2(10),y2(11)
     *     /1.957d-1,1.947d-1,1.735d-1,1.6d-1,8.44d-2,6.27d-2,4.56d-2,
     *      3.42d-2,3.23d-2,2.35d-2,2.46d-2/
      data y3(1),y3(2),y3(3),y3(4),y3(5),y3(6),y3(7),y3(8),y3(9),
     *     y3(10),y3(11),y3(12),y3(13),y3(14),y3(15),y3(16)
     *     /3.478d4,2.861d4,2.365d4,1.963d4,1.637d4,1.372d4,1.154d4,
     *      9.744d3,8.261d3,7.03d3,6.005d3,5.147d3,4.427d3,3.82d3,
     *      3.307d3,2.872d3/
      data y4(1),y4(2),y4(3),y4(4),y4(5),y4(6),y4(7),y4(8),y4(9),
     *     y4(10),y4(11),y4(12),y4(13),y4(14),y4(15),y4(16),y4(17),
     *     y4(18),y4(19),y4(20),y4(21),y4(22),y4(23),y4(24),y4(25),
     *     y4(26),y4(27),y4(28),y4(29),y4(30),y4(31),y4(32),y4(33)
     *     /8.44d-1,9.08d-1,9.32d-1,9.36d-1,9.25d-1,9.08d-1,8.81d-1,
     *      8.5d-1,8.18d-1,7.84d-1,7.51d-1,7.18d-1,6.85d-1,6.58d-1,
     *      6.28d-1,6.03d-1,5.8d-1,5.58d-1,5.38d-1,5.22d-1,5.06d-1,
     *      4.9d-1,4.78d-1,4.67d-1,4.57d-1,4.48d-1,4.38d-1,4.31d-1,
     *      4.24d-1,4.2d-1,4.14d-1,4.11d-1,4.06d-1/
      data y5(1),y5(2),y5(3),y5(4),y5(5),y5(6),y5(7),y5(8),y5(9),
     *     y5(10),y5(11),y5(12),y5(13),y5(14),y5(15),y5(16),y5(17),
     *     y5(18),y5(19),y5(20),y5(21),y5(22),y5(23),y5(24),y5(25),
     *     y5(26),y5(27),y5(28),y5(29),y5(30),y5(31),y5(32),y5(33),
     *     y5(34),y5(35),y5(36),y5(37),y5(38),y5(39),y5(40),y5(41),
     *     y5(42),y5(43),y5(44),y5(45),y5(46),y5(47),y5(48),y5(49),
     *     y5(50),y5(51),y5(52),y5(53),y5(54),y5(55),y5(56),y5(57),
     *     y5(58),y5(59),y5(60),y5(61),y5(62),y5(63),y5(64),y5(65)
     *     /1.366d0,1.191d0,1.112d0,1.013d0,9.91d-1,8.85d-1,8.31d-1,
     *      8.47d-1,7.86d-1,7.25d-1,7.46d-1,6.79d-1,6.08d-1,6.55d-1,
     *      6.16d-1,6.06d-1,6.02d-1,6.26d-1,6.51d-1,7.24d-1,6.49d-1,
     *      6.49d-1,6.94d-1,6.44d-1,6.24d-1,6.61d-1,6.12d-1,5.58d-1,
     *      5.33d-1,4.95d-1,5.0d-1,4.23d-1,3.95d-1,3.75d-1,3.72d-1,
     *      3.91d-1,3.96d-1,4.05d-1,4.28d-1,4.29d-1,5.23d-1,5.62d-1,
     *      6.07d-1,6.53d-1,6.72d-1,7.08d-1,6.33d-1,6.68d-1,6.45d-1,
     *      6.32d-1,5.91d-1,5.59d-1,5.97d-1,6.25d-1,7.39d-1,7.1d-1,
     *      7.29d-1,7.2d-1,6.36d-1,5.81d-1,4.28d-1,2.92d-1,1.62d-1,
     *      9.8d-2,5.4d-2/
      data y6(1),y6(2),y6(3),y6(4),y6(5),y6(6),y6(7),y6(8)
     *     /-6.9d-1,-4.4d-2,-1.57d0,-1.31d0,-2.65d0,2.0d0,
     *      -1.26d1,9.48d0/

      if (nprob .eq. 1) then

c        Linear function - full rank.

         sum = zero
         do j = 1, n
            sum = sum + x(j)
         end do
         temp = 2*sum/dble(m) + one
         do i = 1, m
            fvec(i) = -temp
            if (i .le. n) fvec(i) = fvec(i) + x(i)
         end do
         
      else if (nprob .eq. 2) then

c        Linear function - rank 1.

         sum = zero
         do j = 1, n
            sum = sum + dble(j)*x(j)
         end do
         do i = 1, m
            fvec(i) = dble(i)*sum - one
         end do
         
      else if (nprob .eq. 3) then

c        Linear function - rank 1 with zero columns and rows.

         sum = zero
         do j = 2, n-1
            sum = sum + dble(j)*x(j)
         end do
         do i = 1, m-1
            fvec(i) = dble(i-1)*sum - one
         end do
         fvec(m) = -one
         
      else if (nprob .eq. 4) then

c        Rosenbrock function.

         fvec(1) = 10*(x(2) - x(1)**2)
         fvec(2) = one - x(1)
         
      else if (nprob .eq. 5) then

c        Helical valley function.

         temp = 8*atan(one)
         tmp1 = sign(0.25d0,x(2))
         if (x(1) .gt. zero) tmp1 = atan(x(2)/x(1))/temp
         if (x(1) .lt. zero) tmp1 = atan(x(2)/x(1))/temp + 0.5d0
         tmp2 = sqrt(x(1)**2+x(2)**2)
         fvec(1) = 10*(x(3) - 10*tmp1)
         fvec(2) = 10*(tmp2 - one)
         fvec(3) = x(3)
         
      else if (nprob .eq. 6) then

c        Powell singular function.

         fvec(1) = x(1) + 10*x(2)
         fvec(2) = sqrt(5.0d0)*(x(3) - x(4))
         fvec(3) = (x(2) - 2*x(3))**2
         fvec(4) = sqrt(10.0d0)*(x(1) - x(4))**2
         
      else if (nprob .eq. 7) then

c        Freudenstein and Roth function.

         fvec(1) = -13 + x(1) + ((5 - x(2))*x(2) - 2)*x(2)
         fvec(2) = -29 + x(1) + ((one + x(2))*x(2) - 14)*x(2)
         
      else if (nprob .eq. 8) then

c        Bard function.

         do i = 1, 15
            tmp1 = dble(i)
            tmp2 = dble(16-i)
            tmp3 = tmp1
            if (i .gt. 8) tmp3 = tmp2
            fvec(i) = y1(i) - (x(1) + tmp1/(x(2)*tmp2 + x(3)*tmp3))
          end do
         
      else if (nprob .eq. 9) then

c        Kowalik and Osborne function.

         do i = 1, 11
            tmp1 = y(i)*(y(i) + x(2))
            tmp2 = y(i)*(y(i) + x(3)) + x(4)
            fvec(i) = y2(i) - x(1)*tmp1/tmp2
         end do
         
      else if (nprob .eq. 10) then

c        Meyer function.

         do i = 1, 16
            temp = 5*dble(i) + 45 + x(3)
            tmp1 = x(2)/temp
            tmp2 = exp(tmp1)
            fvec(i) = x(1)*tmp2 - y3(i)
         end do
         
      else if (nprob .eq. 11) then

c        Watson function.

         do i = 1, 29
            temp = dble(i)/29
            sum = zero
            dx = one
            do j = 2, n
               sum = sum + dble(j-1)*dx*x(j)
               dx = temp*dx
            end do
            fvec(i) = sum
            sum = zero
            dx = one
            do j = 1, n
               sum = sum + dx*x(j)
               dx = temp*dx
            end do
            fvec(i) = fvec(i) - sum**2 - one
         end do
         fvec(30) = x(1)
         fvec(31) = x(2) - x(1)**2 - one
         
      else if (nprob .eq. 12) then

c        Box 3-dimensional function.

         do i = 1, m
            temp = dble(i)
            tmp1 = temp/10
            fvec(i) = exp(-tmp1*x(1)) - exp(-tmp1*x(2))
     *                + (exp(-temp) - exp(-tmp1))*x(3)
         end do
         
      else if (nprob .eq. 13) then

c        Jennrich and Sampson function.

         do i = 1, m
            temp = dble(i)
            fvec(i) = 2*(one + temp) - (exp(temp*x(1)) + exp(temp*x(2)))
         end do
         
      else if (nprob .eq. 14) then

c        Brown and Dennis function.

         do i = 1, m
            temp = dble(i)/5
            tmp1 = x(1) + temp*x(2) - exp(temp)
            tmp2 = x(3) + sin(temp)*x(4) - cos(temp)
            fvec(i) = tmp1**2 + tmp2**2
         end do
         
      else if (nprob .eq. 15) then

c        Chebyquad function.

         do i = 1, m
            fvec(i) = zero
         end do
         do j = 1, n
            tmp1 = one
            tmp2 = 2*x(j) - one
            temp = 2*tmp2
            do i = 1, m
               fvec(i) = fvec(i) + tmp2
               tmp3 = temp*tmp2 - tmp1
               tmp1 = tmp2
               tmp2 = tmp3
            end do
         end do
         do i = 1, m
            fvec(i) = fvec(i)/n
            if (mod(i,2) .eq. 0) fvec(i) = fvec(i) + one/(i**2 - one)
         end do
         
      else if (nprob .eq. 16) then

c        Brown almost-linear function.

         sum = -dble(n+1)
         prod = one
         do j = 1, n
            sum = sum + x(j)
            prod = x(j)*prod
         end do
         do i = 1, n-1
            fvec(i) = x(i) + sum
         end do
         fvec(n) = prod - one
         
      else if (nprob .eq. 17) then

c        Osborne 1 function.

         do i = 1, 33
            temp = 10*dble(i-1)
            tmp1 = exp(-x(4)*temp)
            tmp2 = exp(-x(5)*temp)
            fvec(i) = y4(i) - (x(1) + x(2)*tmp1 + x(3)*tmp2)
         end do
         
      else if (nprob .eq. 18) then

c        Osborne 2 function.

         do i = 1, 65
            temp = dble(i-1)/10
            tmp1 = exp(-x(5)*temp)
            tmp2 = exp(-x(6)*(temp-x(9))**2)
            tmp3 = exp(-x(7)*(temp-x(10))**2)
            tmp4 = exp(-x(8)*(temp-x(11))**2)
            fvec(i) = y5(i) -
     *                (x(1)*tmp1 + x(2)*tmp2 + x(3)*tmp3 + x(4)*tmp4)
         end do

      else if (nprob .eq. 19) then

c        Bdqrtic function.
         
         do i = 1, n-4
            fvec(i) = -4*x(i) + 3
            fvec(n-4+i) = x(i)**2 + 2*x(i+1)**2 + 3*x(i+2)**2 +
     *                    4*x(i+3)**2 + 5*x(n)**2
         end do
         
      else if (nprob .eq. 20) then

c        Cube function.
         
         fvec(1) = x(1) - one
         do i = 2, n
            fvec(i) = 10*(x(i) - x(i-1)**3)
         end do  
                  
      else if (nprob .eq. 21) then

c        Mancino function.

         do i = 1, n
            sum = zero 
            do j = 1, n
               temp = sqrt(x(i)**2 + dble(i)/dble(j))
               sum = sum + 
     *               temp*((sin(log(temp)))**5 + (cos(log(temp)))**5)
            end do
            fvec(i) = 1400*x(i) + (i-50)**3 + sum
         end do
                  
      else if (nprob .eq. 22) then

c       Heart 8 function.

        fvec(1) = x(1) + x(2) - y6(1)
        fvec(2) = x(3) + x(4) - y6(2)
        fvec(3) = x(5)*x(1) + x(6)*x(2) - x(7)*x(3) - x(8)*x(4) - y6(3)
        fvec(4) = x(7)*x(1) + x(8)*x(2) + x(5)*x(3) + x(6)*x(4) - y6(4)
        fvec(5) = x(1)*(x(5)**2-x(7)**2) - 2*x(3)*x(5)*x(7) +
     *            x(2)*(x(6)**2-x(8)**2) - 2*x(4)*x(6)*x(8) - y6(5)
        fvec(6) = x(3)*(x(5)**2-x(7)**2) + 2*x(1)*x(5)*x(7) +
     *            x(4)*(x(6)**2-x(8)**2) + 2*x(2)*x(6)*x(8) - y6(6)
        fvec(7) = x(1)*x(5)*(x(5)**2-3*x(7)**2) +
     *            x(3)*x(7)*(x(7)**2-3*x(5)**2) +
     *            x(2)*x(6)*(x(6)**2-3*x(8)**2) +
     *            x(4)*x(8)*(x(8)**2-3*x(6)**2) - y6(7)
        fvec(8) = x(3)*x(5)*(x(5)**2-3*x(7)**2) -
     *            x(1)*x(7)*(x(7)**2-3*x(5)**2) + 
     *            x(4)*x(6)*(x(6)**2-3*x(8)**2) -
     *            x(2)*x(8)*(x(8)**2-3*x(6)**2) - y6(8)
      else
         write (*,*) "Parameter nprob > 22 in subroutine dfovec"
      end if

c            do i=1,m
c       if (abs(fvec(i)) .gt. 1.0d64) then
c         fvec(i) = 1.0d64
c      end if
c	      end do      
      


      return

      end
