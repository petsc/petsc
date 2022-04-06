#include <petscsys.h>
#include <petscblaslapack.h>

static PetscErrorCode estsv(PetscInt n, PetscReal *r, PetscInt ldr, PetscReal *svmin, PetscReal *z)
{
  PetscBLASInt   blas1=1, blasn, blasnmi, blasj, blasldr;
  PetscInt       i,j;
  PetscReal      e,temp,w,wm,ynorm,znorm,s,sm;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n,&blasn));
  PetscCall(PetscBLASIntCast(ldr,&blasldr));
  for (i=0;i<n;i++) {
    z[i]=0.0;
  }
  e = PetscAbs(r[0]);
  if (e == 0.0) {
    *svmin = 0.0;
    z[0] = 1.0;
  } else {
    /* Solve R'*y = e */
    for (i=0;i<n;i++) {
      /* Scale y. The scaling factor (0.01) reduces the number of scalings */
      if (z[i] >= 0.0) e =-PetscAbs(e);
      else             e = PetscAbs(e);

      if (PetscAbs(e - z[i]) > PetscAbs(r[i + ldr*i])) {
        temp = PetscMin(0.01,PetscAbs(r[i + ldr*i]))/PetscAbs(e-z[i]);
        PetscStackCallBLAS("BLASscal",BLASscal_(&blasn, &temp, z, &blas1));
        e = temp*e;
      }

      /* Determine the two possible choices of y[i] */
      if (r[i + ldr*i] == 0.0) {
        w = wm = 1.0;
      } else {
        w  = (e - z[i]) / r[i + ldr*i];
        wm = - (e + z[i]) / r[i + ldr*i];
      }

      /*  Chose y[i] based on the predicted value of y[j] for j>i */
      s  = PetscAbs(e - z[i]);
      sm = PetscAbs(e + z[i]);
      for (j=i+1;j<n;j++) {
        sm += PetscAbs(z[j] + wm * r[i + ldr*j]);
      }
      if (i < n-1) {
        PetscCall(PetscBLASIntCast(n-i-1,&blasnmi));
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasnmi, &w, &r[i + ldr*(i+1)], &blasldr, &z[i+1], &blas1));
        PetscStackCallBLAS("BLASasum",s += BLASasum_(&blasnmi, &z[i+1], &blas1));
      }
      if (s < sm) {
        temp = wm - w;
        w = wm;
        if (i < n-1) {
          PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasnmi, &temp, &r[i + ldr*(i+1)], &blasldr, &z[i+1], &blas1));
        }
      }
      z[i] = w;
    }

    PetscStackCallBLAS("BLASnrm2",ynorm = BLASnrm2_(&blasn, z, &blas1));

    /* Solve R*z = y */
    for (j=n-1; j>=0; j--) {
      /* Scale z */
      if (PetscAbs(z[j]) > PetscAbs(r[j + ldr*j])) {
        temp = PetscMin(0.01, PetscAbs(r[j + ldr*j] / z[j]));
        PetscStackCallBLAS("BLASscal",BLASscal_(&blasn, &temp, z, &blas1));
        ynorm *=temp;
      }
      if (r[j + ldr*j] == 0) {
        z[j] = 1.0;
      } else {
        z[j] = z[j] / r[j + ldr*j];
      }
      temp = -z[j];
      PetscCall(PetscBLASIntCast(j,&blasj));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasj,&temp,&r[0+ldr*j],&blas1,z,&blas1));
    }

    /* Compute svmin and normalize z */
    PetscStackCallBLAS("BLASnrm2",znorm = 1.0 / BLASnrm2_(&blasn, z, &blas1));
    *svmin = ynorm*znorm;
    PetscStackCallBLAS("BLASscal",BLASscal_(&blasn, &znorm, z, &blas1));
  }
  PetscFunctionReturn(0);
}

/*
c     ***********
c
c     Subroutine gqt
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
c       subroutine gqt(n,a,lda,b,delta,rtol,atol,itmax,
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
*/
PetscErrorCode gqt(PetscInt n, PetscReal *a, PetscInt lda, PetscReal *b,
                   PetscReal delta, PetscReal rtol, PetscReal atol,
                   PetscInt itmax, PetscReal *retpar, PetscReal *retf,
                   PetscReal *x, PetscInt *retinfo, PetscInt *retits,
                   PetscReal *z, PetscReal *wa1, PetscReal *wa2)
{
  PetscReal      f=0.0,p001=0.001,p5=0.5,minusone=-1,delta2=delta*delta;
  PetscInt       iter, j, rednc,info;
  PetscBLASInt   indef;
  PetscBLASInt   blas1=1, blasn, iblas, blaslda, blasldap1, blasinfo;
  PetscReal      alpha, anorm, bnorm, parc, parf, parl, pars, par=*retpar,paru, prod, rxnorm, rznorm=0.0, temp, xnorm;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n,&blasn));
  PetscCall(PetscBLASIntCast(lda,&blaslda));
  PetscCall(PetscBLASIntCast(lda+1,&blasldap1));
  parf   = 0.0;
  xnorm  = 0.0;
  rxnorm = 0.0;
  rednc  = 0;
  for (j=0; j<n; j++) {
    x[j] = 0.0;
    z[j] = 0.0;
  }

  /* Copy the diagonal and save A in its lower triangle */
  PetscStackCallBLAS("BLAScopy",BLAScopy_(&blasn,a,&blasldap1, wa1, &blas1));
  for (j=0;j<n-1;j++) {
    PetscCall(PetscBLASIntCast(n - j - 1,&iblas));
    PetscStackCallBLAS("BLAScopy",BLAScopy_(&iblas,&a[j + lda*(j+1)], &blaslda, &a[j+1 + lda*j], &blas1));
  }

  /* Calculate the l1-norm of A, the Gershgorin row sums, and the
   l2-norm of b */
  anorm = 0.0;
  for (j=0;j<n;j++) {
    PetscStackCallBLAS("BLASasum",wa2[j] = BLASasum_(&blasn, &a[0 + lda*j], &blas1));CHKMEMQ;
    anorm = PetscMax(anorm,wa2[j]);
  }
  for (j=0;j<n;j++) {
    wa2[j] = wa2[j] - PetscAbs(wa1[j]);
  }
  PetscStackCallBLAS("BLASnrm2",bnorm = BLASnrm2_(&blasn,b,&blas1));CHKMEMQ;
  /* Calculate a lower bound, pars, for the domain of the problem.
   Also calculate an upper bound, paru, and a lower bound, parl,
   for the Lagrange multiplier. */
  pars = parl = paru = -anorm;
  for (j=0;j<n;j++) {
    pars = PetscMax(pars, -wa1[j]);
    parl = PetscMax(parl, wa1[j] + wa2[j]);
    paru = PetscMax(paru, -wa1[j] + wa2[j]);
  }
  parl = PetscMax(bnorm/delta - parl,pars);
  parl = PetscMax(0.0,parl);
  paru = PetscMax(0.0, bnorm/delta + paru);

  /* If the input par lies outside of the interval (parl, paru),
   set par to the closer endpoint. */

  par = PetscMax(par,parl);
  par = PetscMin(par,paru);

  /* Special case: parl == paru */
  paru = PetscMax(paru, (1.0 + rtol)*parl);

  /* Beginning of an iteration */

  info = 0;
  for (iter=1;iter<=itmax;iter++) {
    /* Safeguard par */
    if (par <= pars && paru > 0) {
      par = PetscMax(p001, PetscSqrtScalar(parl/paru)) * paru;
    }

    /* Copy the lower triangle of A into its upper triangle and  compute A + par*I */

    for (j=0;j<n-1;j++) {
      PetscCall(PetscBLASIntCast(n - j - 1,&iblas));
      PetscStackCallBLAS("BLAScopy",BLAScopy_(&iblas,&a[j+1 + j*lda], &blas1,&a[j + (j+1)*lda], &blaslda));
    }
    for (j=0;j<n;j++) {
      a[j + j*lda] = wa1[j] + par;
    }

    /* Attempt the Cholesky factorization of A without referencing the lower triangular part. */
    PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("U",&blasn,a,&blaslda,&indef));

    /* Case 1: A + par*I is pos. def. */
    if (indef == 0) {

      /* Compute an approximate solution x and save the last value of par with A + par*I pos. def. */

      parf = par;
      PetscStackCallBLAS("BLAScopy",BLAScopy_(&blasn, b, &blas1, wa2, &blas1));
      PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U","T","N",&blasn,&blas1,a,&blaslda,wa2,&blasn,&blasinfo));
      PetscCheck(!blasinfo,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACKtrtrs() returned info %" PetscBLASInt_FMT,blasinfo);
      PetscStackCallBLAS("BLASnrm2",rxnorm = BLASnrm2_(&blasn, wa2, &blas1));
      PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U","N","N",&blasn,&blas1,a,&blaslda,wa2,&blasn,&blasinfo));
      PetscCheck(!blasinfo,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACKtrtrs() returned info %" PetscBLASInt_FMT,blasinfo);

      PetscStackCallBLAS("BLAScopy",BLAScopy_(&blasn, wa2, &blas1, x, &blas1));
      PetscStackCallBLAS("BLASscal",BLASscal_(&blasn, &minusone, x, &blas1));
      PetscStackCallBLAS("BLASnrm2",xnorm = BLASnrm2_(&blasn, x, &blas1));CHKMEMQ;

      /* Test for convergence */
      if (PetscAbs(xnorm - delta) <= rtol*delta || (par == 0  && xnorm <= (1.0+rtol)*delta)) {
        info = 1;
      }

      /* Compute a direction of negative curvature and use this information to improve pars. */
      PetscCall(estsv(n,a,lda,&rznorm,z));CHKMEMQ;
      pars = PetscMax(pars, par-rznorm*rznorm);

      /* Compute a negative curvature solution of the form x + alpha*z,  where norm(x+alpha*z)==delta */

      rednc = 0;
      if (xnorm < delta) {
        /* Compute alpha */
        PetscStackCallBLAS("BLASdot",prod = BLASdot_(&blasn, z, &blas1, x, &blas1)/delta);
        temp = (delta - xnorm)*((delta + xnorm)/delta);
        alpha = temp/(PetscAbs(prod) + PetscSqrtScalar(prod*prod + temp/delta));
        if (prod >= 0) alpha = PetscAbs(alpha);
        else alpha =-PetscAbs(alpha);

        /* Test to decide if the negative curvature step produces a larger reduction than with z=0 */
        rznorm = PetscAbs(alpha) * rznorm;
        if ((rznorm*rznorm + par*xnorm*xnorm)/(delta2) <= par) {
          rednc = 1;
        }
        /* Test for convergence */
        if (p5 * rznorm*rznorm / delta2 <= rtol*(1.0-p5*rtol)*(par + rxnorm*rxnorm/delta2)) {
          info = 1;
        } else if (info == 0 && (p5*(par + rxnorm*rxnorm/delta2) <= atol/delta2)) {
          info = 2;
        }
      }

      /* Compute the Newton correction parc to par. */
      if (xnorm == 0) {
        parc = -par;
      } else {
        PetscStackCallBLAS("BLAScopy",BLAScopy_(&blasn, x, &blas1, wa2, &blas1));
        temp = 1.0/xnorm;
        PetscStackCallBLAS("BLASscal",BLASscal_(&blasn, &temp, wa2, &blas1));
        PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U","T","N",&blasn, &blas1, a, &blaslda, wa2, &blasn, &blasinfo));
        PetscCheck(!blasinfo,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACKtrtrs() returned info %" PetscBLASInt_FMT,blasinfo);
        PetscStackCallBLAS("BLASnrm2",temp = BLASnrm2_(&blasn, wa2, &blas1));
        parc = (xnorm - delta)/(delta*temp*temp);
      }

      /* update parl or paru */
      if (xnorm > delta) {
        parl = PetscMax(parl, par);
      } else if (xnorm < delta) {
        paru = PetscMin(paru, par);
      }
    } else {
      /* Case 2: A + par*I is not pos. def. */

      /* Use the rank information from the Cholesky decomposition to update par. */

      if (indef > 1) {
        /* Restore column indef to A + par*I. */
        iblas = indef - 1;
        PetscStackCallBLAS("BLAScopy",BLAScopy_(&iblas,&a[indef-1 + 0*lda],&blaslda,&a[0 + (indef-1)*lda],&blas1));
        a[indef-1 + (indef-1)*lda] = wa1[indef-1] + par;

        /* compute parc. */
        PetscStackCallBLAS("BLAScopy",BLAScopy_(&iblas,&a[0 + (indef-1)*lda], &blas1, wa2, &blas1));
        PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U","T","N",&iblas,&blas1,a,&blaslda,wa2,&blasn,&blasinfo));
        PetscCheck(!blasinfo,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACKtrtrs() returned info %" PetscBLASInt_FMT,blasinfo);
        PetscStackCallBLAS("BLAScopy",BLAScopy_(&iblas,wa2,&blas1,&a[0 + (indef-1)*lda],&blas1));
        PetscStackCallBLAS("BLASnrm2",temp = BLASnrm2_(&iblas,&a[0 + (indef-1)*lda],&blas1));CHKMEMQ;
        a[indef-1 + (indef-1)*lda] -= temp*temp;
        PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U","N","N",&iblas,&blas1,a,&blaslda,wa2,&blasn,&blasinfo));
        PetscCheck(!blasinfo,PETSC_COMM_SELF,PETSC_ERR_LIB,"LAPACKtrtrs() returned info %" PetscBLASInt_FMT,blasinfo);
      }

      wa2[indef-1] = -1.0;
      iblas = indef;
      PetscStackCallBLAS("BLASnrm2",temp = BLASnrm2_(&iblas,wa2,&blas1));
      parc = - a[indef-1 + (indef-1)*lda]/(temp*temp);
      pars = PetscMax(pars,par+parc);

      /* If necessary, increase paru slightly.
       This is needed because in some exceptional situations
       paru is the optimal value of par. */

      paru = PetscMax(paru, (1.0+rtol)*pars);
    }

    /* Use pars to update parl */
    parl = PetscMax(parl,pars);

    /* Test for converged. */
    if (info == 0) {
      if (iter == itmax) info=4;
      if (paru <= (1.0+p5*rtol)*pars) info=3;
      if (paru == 0.0) info = 2;
    }

    /* If exiting, store the best approximation and restore
     the upper triangle of A. */

    if (info != 0) {
      /* Compute the best current estimates for x and f. */
      par = parf;
      f = -p5 * (rxnorm*rxnorm + par*xnorm*xnorm);
      if (rednc) {
        f = -p5 * (rxnorm*rxnorm + par*delta*delta - rznorm*rznorm);
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&blasn, &alpha, z, &blas1, x, &blas1));
      }
      /* Restore the upper triangle of A */
      for (j = 0; j<n; j++) {
        PetscCall(PetscBLASIntCast(n - j - 1,&iblas));
        PetscStackCallBLAS("BLAScopy",BLAScopy_(&iblas,&a[j+1 + j*lda],&blas1, &a[j + (j+1)*lda],&blaslda));
      }
      PetscCall(PetscBLASIntCast(lda+1,&iblas));
      PetscStackCallBLAS("BLAScopy",BLAScopy_(&blasn,wa1,&blas1,a,&iblas));
      break;
    }
    par = PetscMax(parl,par+parc);
  }
  *retpar  = par;
  *retf    = f;
  *retinfo = info;
  *retits  = iter;
  CHKMEMQ;
  PetscFunctionReturn(0);
}
