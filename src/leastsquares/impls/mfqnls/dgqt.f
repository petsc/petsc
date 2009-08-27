      subroutine dgqt(n,a,lda,b,delta,rtol,atol,itmax,par,f,x,info,
     +                iter,z,wa1,wa2)
      integer n, lda, itmax, info
      double precision delta, rtol, atol, par, f
      double precision a(lda,n), b(n), x(n)
      double precision z(n), wa1(n), wa2(n)      
c      double precision z(100), wa1(100), wa2(100)
c     ***********
c
c     Subroutine dgqt
c
c     Given an n by n symmetric matrix A, an n-vector b, and a
c     positive number delta, this subroutine determines a vector
c     x which approximately minimizes the quadratic function
c
c           f(x) = (1/2)*x'*A*x + b'*x
c
c     subject to the Euclidean norm constraint
c
c           norm(x) <= delta.
c
c     This subroutine computes an approximation x and a Lagrange
c     multiplier par such that either par is zero and
c
c            norm(x) <= (1+rtol)*delta,
c
c     or par is positive and
c
c            abs(norm(x) - delta) <= rtol*delta.
c
c     If xsol is the solution to the problem, the approximation x
c     satisfies
c
c            f(x) <= ((1 - rtol)**2)*f(xsol)
c
c     The subroutine statement is
c
c       subroutine dgqt(n,a,lda,b,delta,rtol,atol,itmax,
c                        par,f,x,info,z,wa1,wa2)
c
c     where
c
c       n is an integer variable.
c         On entry n is the order of A.
c         On exit n is unchanged.
c
c       a is a double precision array of dimension (lda,n).
c         On entry the full upper triangle of a must contain the
c            full upper triangle of the symmetric matrix A.
c         On exit the array contains the matrix A.
c
c       lda is an integer variable.
c         On entry lda is the leading dimension of the array a.
c         On exit lda is unchanged.
c
c       b is an double precision array of dimension n.
c         On entry b specifies the linear term in the quadratic.
c         On exit b is unchanged.
c
c       delta is a double precision variable.
c         On entry delta is a bound on the Euclidean norm of x.
c         On exit delta is unchanged.
c
c       rtol is a double precision variable.
c         On entry rtol is the relative accuracy desired in the
c            solution. Convergence occurs if
c
c              f(x) <= ((1 - rtol)**2)*f(xsol)
c
c         On exit rtol is unchanged.
c
c       atol is a double precision variable.
c         On entry atol is the absolute accuracy desired in the
c            solution. Convergence occurs when
c
c              norm(x) <= (1 + rtol)*delta
c
c              max(-f(x),-f(xsol)) <= atol
c
c         On exit atol is unchanged.
c
c       itmax is an integer variable.
c         On entry itmax specifies the maximum number of iterations.
c         On exit itmax is unchanged.
c
c       par is a double precision variable.
c         On entry par is an initial estimate of the Lagrange
c            multiplier for the constraint norm(x) <= delta.
c         On exit par contains the final estimate of the multiplier.
c
c       f is a double precision variable.
c         On entry f need not be specified.
c         On exit f is set to f(x) at the output x.
c
c       x is a double precision array of dimension n.
c         On entry x need not be specified.
c         On exit x is set to the final estimate of the solution.
c
c       info is an integer variable.
c         On entry info need not be specified.
c         On exit info is set as follows:
c
c            info = 1  The function value f(x) has the relative
c                      accuracy specified by rtol.
c
c            info = 2  The function value f(x) has the absolute
c                      accuracy specified by atol.
c
c            info = 3  Rounding errors prevent further progress.
c                      On exit x is the best available approximation.
c
c            info = 4  Failure to converge after itmax iterations.
c                      On exit x is the best available approximation.
c
c       z is a double precision work array of dimension n.
c
c       wa1 is a double precision work array of dimension n.
c
c       wa2 is a double precision work array of dimension n.
c
c     Subprograms called
c
c       MINPACK-2  ......  destsv
c
c       LAPACK  .........  dpotrf
c
c       Level 1 BLAS  ...  daxpy, dcopy, ddot, dnrm2, dscal
c
c       Level 2 BLAS  ...  dtrmv, dtrsv
c
c     MINPACK-2 Project. October 1993.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick, Richard Carter, and Jorge J. More'
c
c     ***********
      double precision one, p001, p5, zero
      parameter (zero=0.0d0,p001=1.0d-3,p5=0.5d0,one=1.0d0)

      logical rednc
      integer indef, iter, j
      double precision alpha, anorm, bnorm, parc, parf, parl, pars,
     +                 paru, prod, rxnorm, rznorm, temp, xnorm

      double precision dasum, ddot, dnrm2
      external destsv, daxpy, dcopy, ddot, dnrm2, dscal, dtrmv, dtrsv

c     Initialization.

      iter = 0
      parf = zero
      xnorm = zero
      rxnorm = zero
      rednc = .false.
      do j = 1, n
         x(j) = zero
         z(j) = zero
      end do

c     Copy the diagonal and save A in its lower triangle.

      call dcopy(n,a,lda+1,wa1,1)
      do j = 1, n - 1
         call dcopy(n-j,a(j,j+1),lda,a(j+1,j),1)
      end do

c     Calculate the l1-norm of A, the Gershgorin row sums,
c     and the l2-norm of b.

      anorm = zero
      do j = 1, n
         wa2(j) = dasum(n,a(1,j),1)
         anorm = max(anorm,wa2(j))
      end do
      do j = 1, n
         wa2(j) = wa2(j) - abs(wa1(j))
      end do
      bnorm = dnrm2(n,b,1)

c     Calculate a lower bound, pars, for the domain of the problem.
c     Also calculate an upper bound, paru, and a lower bound, parl,
c     for the Lagrange multiplier.

      pars = -anorm
      parl = -anorm
      paru = -anorm
      do j = 1, n
         pars = max(pars,-wa1(j))
         parl = max(parl,wa1(j)+wa2(j))
         paru = max(paru,-wa1(j)+wa2(j))
      end do
      parl = max(zero,bnorm/delta-parl,pars)
      paru = max(zero,bnorm/delta+paru)

c     If the input par lies outside of the interval (parl,paru),
c     set par to the closer endpoint.

      par = max(par,parl)
      par = min(par,paru)

c     Special case: parl = paru.

      paru = max(paru,(one+rtol)*parl)

c     Beginning of an iteration.

      info = 0
      do iter = 1, itmax

c        Safeguard par.

         if (par .le. pars .and. paru .gt. zero) par = max(p001,
     +       sqrt(parl/paru))*paru

c        Copy the lower triangle of A into its upper triangle and
c        compute A + par*I.

         do j = 1, n - 1
            call dcopy(n-j,a(j+1,j),1,a(j,j+1),lda)
         end do
         do j = 1, n
            a(j,j) = wa1(j) + par
         end do

c        Attempt the  Cholesky factorization of A without referencing
c        the lower triangular part.

         call dpotrf('U',n,a,lda,indef)

c        Case 1: A + par*I is positive definite.

         if (indef .eq. 0) then

c           Compute an approximate solution x and save the
c           last value of par with A + par*I positive definite.

            parf = par
            call dcopy(n,b,1,wa2,1)
            call dtrsv('U','T','N',n,a,lda,wa2,1)
            rxnorm = dnrm2(n,wa2,1)
            call dtrsv('U','N','N',n,a,lda,wa2,1)
            call dcopy(n,wa2,1,x,1)
            call dscal(n,-one,x,1)
            xnorm = dnrm2(n,x,1)

c           Test for convergence.

            if (abs(xnorm-delta) .le. rtol*delta .or.
     +          (par .eq. zero .and. xnorm .le. (one+rtol)*delta))
     +          info = 1

c           Compute a direction of negative curvature and use this
c           information to improve pars.

            call destsv(n,a,lda,rznorm,z)
            pars = max(pars,par-rznorm**2)

c           Compute a negative curvature solution of the form
c           x + alpha*z where norm(x+alpha*z) = delta.

            rednc = .false.
            if (xnorm .lt. delta) then

c              Compute alpha

               prod = ddot(n,z,1,x,1)/delta
               temp = (delta-xnorm)*((delta+xnorm)/delta)
               alpha = temp/(abs(prod)+sqrt(prod**2+temp/delta))
               alpha = sign(alpha,prod)

c              Test to decide if the negative curvature step
c              produces a larger reduction than with z = 0.

               rznorm = abs(alpha)*rznorm
               if ((rznorm/delta)**2+par*(xnorm/delta)**2 .le.
     +             par) rednc = .true.

c              Test for convergence.

               if (p5*(rznorm/delta)**2 .le.
     +             rtol*(one-p5*rtol)*(par+(rxnorm/delta)**2)) then
                  info = 1
               else if (p5*(par+(rxnorm/delta)**2) .le.
     +                  (atol/delta)/delta .and. info .eq. 0) then
                  info = 2
               end if
            end if

c           Compute the Newton correction parc to par.

            if (xnorm .eq. zero) then
               parc = -par
            else
               call dcopy(n,x,1,wa2,1)
               temp = one/xnorm
               call dscal(n,temp,wa2,1)
               call dtrsv('U','T','N',n,a,lda,wa2,1)
               temp = dnrm2(n,wa2,1)
               parc = (((xnorm-delta)/delta)/temp)/temp
            end if

c           Update parl or paru.

            if (xnorm .gt. delta) parl = max(parl,par)
            if (xnorm .lt. delta) paru = min(paru,par)
         else

c           Case 2: A + par*I is not positive definite.

c           Use the rank information from the Cholesky
c           decomposition to update par.

            if (indef .gt. 1) then

c              Restore column indef to A + par*I.

               call dcopy(indef-1,a(indef,1),lda,a(1,indef),1)
               a(indef,indef) = wa1(indef) + par

c              Compute parc.

               call dcopy(indef-1,a(1,indef),1,wa2,1)
               call dtrsv('U','T','N',indef-1,a,lda,wa2,1)
               call dcopy(indef-1,wa2,1,a(1,indef),1)
               temp = dnrm2(indef-1,a(1,indef),1)
               a(indef,indef) = a(indef,indef) - temp**2
               call dtrsv('U','N','N',indef-1,a,lda,wa2,1)
            end if
            wa2(indef) = -one
            temp = dnrm2(indef,wa2,1)
            parc = -(a(indef,indef)/temp)/temp
            pars = max(pars,par+parc)

c           If necessary, increase paru slightly.
c           This is needed because in some exceptional situations
c           paru is the optimal value of par.

            paru = max(paru,(one+rtol)*pars)
         end if

c        Use pars to update parl.

         parl = max(parl,pars)

c        Test for termination.

         if (info .eq. 0) then
            if (iter .eq. itmax) info = 4
            if (paru .le. (one+p5*rtol)*pars) info = 3
            if (paru .eq. zero) info = 2
         end if

c        If exiting, store the best approximation and restore
c        the upper triangle of A.

         if (info .ne. 0) then

c           Compute the best current estimates for x and f.

            par = parf
            f = -p5*(rxnorm**2+par*xnorm**2)
            if (rednc) then
               f = -p5*((rxnorm**2+par*delta**2)-rznorm**2)
               call daxpy(n,alpha,z,1,x,1)
            end if

c           Restore the upper triangle of A.

            do j = 1, n - 1
               call dcopy(n-j,a(j+1,j),1,a(j,j+1),lda)
            end do
            call dcopy(n,wa1,1,a,lda+1)
            return
         end if

c        Compute an improved estimate for par.

         par = max(parl,par+parc)

c        End of an iteration.

      end do

      end
