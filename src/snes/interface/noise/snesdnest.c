
/* fnoise/snesdnest.F -- translated by f2c (version 20020314).
*/
#include <petscsys.h>
#define FALSE_ 0
#define TRUE_ 1

/*  Noise estimation routine, written by Jorge More'.  Details are below. */

PETSC_INTERN PetscErrorCode SNESNoise_dnest_(PetscInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*);

PetscErrorCode SNESNoise_dnest_(PetscInt *nf, double *fval,double *h__,double *fnoise, double *fder2, double *hopt, PetscInt *info, double *eps)
{
  /* Initialized data */

  static double const__[15] = { .71,.41,.23,.12,.063,.033,.018,.0089,
                                .0046,.0024,.0012,6.1e-4,3.1e-4,1.6e-4,8e-5 };

  /* System generated locals */
  PetscInt i__1;
  double   d__1, d__2, d__3, d__4;

  /* Local variables */
  static double   emin, emax;
  static PetscInt dsgn[6];
  static double   f_max, f_min, stdv;
  static PetscInt i__, j;
  static double   scale;
  static PetscInt mh;
  static PetscInt cancel[6], dnoise;
  static double   err2, est1, est2, est3, est4;

/*     ********** */

/*     Subroutine dnest */

/*     This subroutine estimates the noise in a function */
/*     and provides estimates of the optimal difference parameter */
/*     for a forward-difference approximation. */

/*     The user must provide a difference parameter h, and the */
/*     function value at nf points centered around the current point. */
/*     For example, if nf = 7, the user must provide */

/*        f(x-2*h), f(x-h), f(x), f(x+h),  f(x+2*h), */

/*     in the array fval. The use of nf = 7 function evaluations is */
/*     recommended. */

/*     The noise in the function is roughly defined as the variance in */
/*     the computed value of the function. The noise in the function */
/*     provides valuable information. For example, function values */
/*     smaller than the noise should be considered to be zero. */

/*     This subroutine requires an initial estimate for h. Under estimates */
/*     are usually preferred. If noise is not detected, the user should */
/*     increase or decrease h according to the ouput value of info. */
/*     In most cases, the subroutine detects noise with the initial */
/*     value of h. */

/*     The subroutine statement is */

/*       subroutine dnest(nf,fval,h,hopt,fnoise,info,eps) */

/*     where */

/*       nf is a PetscInt variable. */
/*         On entry nf is the number of function values. */
/*         On exit nf is unchanged. */

/*       f is a double precision array of dimension nf. */
/*         On entry f contains the function values. */
/*         On exit f is overwritten. */

/*       h is a double precision variable. */
/*         On entry h is an estimate of the optimal difference parameter. */
/*         On exit h is unchanged. */

/*       fnoise is a double precision variable. */
/*         On entry fnoise need not be specified. */
/*         On exit fnoise is set to an estimate of the function noise */
/*            if noise is detected; otherwise fnoise is set to zero. */

/*       hopt is a double precision variable. */
/*         On entry hopt need not be specified. */
/*         On exit hopt is set to an estimate of the optimal difference */
/*            parameter if noise is detected; otherwise hopt is set to zero. */

/*       info is a PetscInt variable. */
/*         On entry info need not be specified. */
/*         On exit info is set as follows: */

/*            info = 1  Noise has been detected. */

/*            info = 2  Noise has not been detected; h is too small. */
/*                      Try 100*h for the next value of h. */

/*            info = 3  Noise has not been detected; h is too large. */
/*                      Try h/100 for the next value of h. */

/*            info = 4  Noise has been detected but the estimate of hopt */
/*                      is not reliable; h is too small. */

/*       eps is a double precision work array of dimension nf. */

/*     MINPACK-2 Project. April 1997. */
/*     Argonne National Laboratory. */
/*     Jorge J. More'. */

/*     ********** */
  /* Parameter adjustments */
  --eps;
  --fval;

  /* Function Body */
  *fnoise = 0.;
  *fder2  = 0.;
  *hopt   = 0.;
/*     Compute an estimate of the second derivative and */
/*     determine a bound on the error. */
  mh   = (*nf + 1) / 2;
  est1 = (fval[mh + 1] - fval[mh] * 2 + fval[mh - 1]) / *h__ / *h__;
  est2 = (fval[mh + 2] - fval[mh] * 2 + fval[mh - 2]) / (*h__ * 2) / (*h__ * 2);
  est3 = (fval[mh + 3] - fval[mh] * 2 + fval[mh - 3]) / (*h__ * 3) / (*h__ * 3);
  est4 = (est1 + est2 + est3) / 3;
/* Computing MAX */
/* Computing PETSCMAX */
  d__3 = PetscMax(est1,est2);
/* Computing MIN */
  d__4 = PetscMin(est1,est2);
  d__1 = PetscMax(d__3,est3) - est4;
  d__2 = est4 - PetscMin(d__4,est3);
  err2 = PetscMax(d__1,d__2);
/*      write (2,123) est1, est2, est3 */
/* 123  format ('Second derivative estimates', 3d12.2) */
  if (err2 <= PetscAbsScalar(est4) * .1) *fder2 = est4;
  else if (err2 < PetscAbsScalar(est4))  *fder2 = est3;
  else *fder2 = 0.;

/*     Compute the range of function values. */
  f_min = fval[1];
  f_max = fval[1];
  i__1  = *nf;
  for (i__ = 2; i__ <= i__1; ++i__) {
    /* Computing MIN */
    d__1 = f_min;
    d__2 = fval[i__];
    f_min = PetscMin(d__1,d__2);

    /* Computing MAX */
    d__1 = f_max;
    d__2 = fval[i__];
    f_max = PetscMax(d__1,d__2);
  }
/*     Construct the difference table. */
  dnoise = FALSE_;
  for (j = 1; j <= 6; ++j) {
    dsgn[j - 1]   = FALSE_;
    cancel[j - 1] = FALSE_;
    scale         = 0.;
    i__1          = *nf - j;
    for (i__ = 1; i__ <= i__1; ++i__) {
      fval[i__] = fval[i__ + 1] - fval[i__];
      if (fval[i__] == 0.) cancel[j - 1] = TRUE_;

      /* Computing MAX */
      d__1 = fval[i__];
      d__2 = scale;
      d__3 = PetscAbsScalar(d__1);
      scale = PetscMax(d__2,d__3);
    }

    /*        Compute the estimates for the noise level. */
    if (scale == 0.) stdv = 0.;
    else {
      stdv = 0.;
      i__1 = *nf - j;
      for (i__ = 1; i__ <= i__1; ++i__) {
        /* Computing 2nd power */
        d__1 = fval[i__] / scale;
        stdv += d__1 * d__1;
      }
      stdv = scale * PetscSqrtScalar(stdv / (*nf - j));
    }
    eps[j] = const__[j - 1] * stdv;
/*        Determine differences in sign. */
    i__1 = *nf - j - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
      /* Computing MIN */
      d__1 = fval[i__];
      d__2 = fval[i__ + 1];
      /* Computing MAX */
      d__3 = fval[i__];
      d__4 = fval[i__ + 1];
      if (PetscMin(d__1,d__2) < 0. && PetscMax(d__3,d__4) > 0.) dsgn[j - 1] = TRUE_;
    }
  }
  /*     First requirement for detection of noise. */
  dnoise = dsgn[3];
  /*     Check for h too small or too large. */
  *info = 0;
  if (f_max == f_min) *info = 2;
  else /* if (complicated condition) */ {
    /* Computing MIN */
    d__1 = PetscAbsScalar(f_max);
    d__2 = PetscAbsScalar(f_min);
    if (f_max - f_min > PetscMin(d__1,d__2) * .1) *info = 3;
  }
  if (*info != 0) PetscFunctionReturn(0);

  /*     Determine the noise level. */
  /* Computing MIN */
  d__1 = PetscMin(eps[4],eps[5]);
  emin = PetscMin(d__1,eps[6]);

  /* Computing MAX */
  d__1 = PetscMax(eps[4],eps[5]);
  emax = PetscMax(d__1,eps[6]);

  if (emax <= emin * 4 && dnoise) {
    *fnoise = (eps[4] + eps[5] + eps[6]) / 3;
    if (*fder2 != 0.) {
      *info = 1;
      *hopt = PetscSqrtScalar(*fnoise / PetscAbsScalar(*fder2)) * 1.68;
    } else {
      *info = 4;
      *hopt = *h__ * 10;
    }
    PetscFunctionReturn(0);
  }

  /* Computing MIN */
  d__1 = PetscMin(eps[3],eps[4]);
  emin = PetscMin(d__1,eps[5]);

  /* Computing MAX */
  d__1 = PetscMax(eps[3],eps[4]);
  emax = PetscMax(d__1,eps[5]);

  if (emax <= emin * 4 && dnoise) {
    *fnoise = (eps[3] + eps[4] + eps[5]) / 3;
    if (*fder2 != 0.) {
      *info = 1;
      *hopt = PetscSqrtScalar(*fnoise / PetscAbsScalar(*fder2)) * 1.68;
    } else {
      *info = 4;
      *hopt = *h__ * 10;
    }
    PetscFunctionReturn(0);
  }
/*     Noise not detected; decide if h is too small or too large. */
  if (!cancel[3]) {
    if (dsgn[3]) *info = 2;
    else *info = 3;
    PetscFunctionReturn(0);
  }
  if (!cancel[2]) {
    if (dsgn[2]) *info = 2;
    else *info = 3;
    PetscFunctionReturn(0);
  }
/*     If there is cancelllation on the third and fourth column */
/*     then h is too small */
  *info = 2;
  PetscFunctionReturn(0);
/*      if (cancel .or. dsgn(3)) then */
/*         info = 2 */
/*      else */
/*         info = 3 */
/*      end if */
} /* dnest_ */

