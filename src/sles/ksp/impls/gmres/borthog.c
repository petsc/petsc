#ifndef lint
static char vcid[] = "$Id: borthog.c,v 1.19 1996/04/04 22:03:06 bsmith Exp curfman $";
#endif
/*
    Routines used for the orthogonalization of the Hessenberg matrix.

    Note that for the complex numbers version, the VecDot() and
    VecMDot() arguments within the code MUST remain in the order
    given for correct computation of inner products.
*/
#include "gmresp.h"
#include <math.h>

/*
    This is the basic orthogonalization routine using modified Gram-Schmidt.
 */
int KSPGMRESModifiedGramSchmidtOrthogonalization( KSP ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j;
  Scalar    *hh, *hes, tmp;

  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* update Hessenberg matrix and do Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);
  for (j=0; j<=it; j++) {
    /* ( vv(it+1), vv(j) ) */
    VecDot( VEC_VV(it+1), VEC_VV(j), hh );
    *hes++   = *hh;
    /* vv(j) <- vv(j) - hh[j][it] vv(it) */
    tmp = - (*hh++);  VecAXPY(&tmp , VEC_VV(j), VEC_VV(it+1) );
  }
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

/*
  This version uses UNMODIFIED Gram-Schmidt.  It is NOT recommended, 
  but it can give better performance when running in a parallel 
  environment

  Multiple applications of this can be used to provide a better 
  orthogonalization (but be careful of the HH and HES values).
 */
int KSPGMRESUnmodifiedGramSchmidtOrthogonalization(KSP  ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j;
  Scalar    *hh, *hes;

  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* update hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* 
   This is really a matrix-vector product, with the matrix stored
   as pointer to rows 
  */
  VecMDot( it+1, VEC_VV(it+1), &(VEC_VV(0)), hes );

  /*
    This is really a matrix-vector product: 
        [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it].
  */
  for (j=0; j<=it; j++) hh[j] = -hes[j];
  VecMAXPY(it+1, hh, VEC_VV(it+1),&VEC_VV(0) );
  for (j=0; j<=it; j++) hh[j] = -hh[j];
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

/*
  This version uses iterative refinement of UNMODIFIED Gram-Schmidt.  
  It can give better performance when running in a parallel 
  environment and in some cases even in a sequential environment (because
  MAXPY has more data reuse).

  Care is taken to accumulate the updated HH/HES values.
 */
int KSPGMRESIROrthogonalization(KSP  ksp,int it )
{
  KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);
  int       j,ncnt;
  Scalar    *hh, *hes,shh[20], *lhh;
  double    dnorm;

  PLogEventBegin(KSP_GMRESOrthogonalization,ksp,0,0,0);
  /* Don't allocate small arrays */
  if (it < 20) lhh = shh;
  else {
    lhh = (Scalar *)PetscMalloc((it+1) * sizeof(Scalar)); CHKPTRQ(lhh);
  }
  
  /* update Hessenberg matrix and do unmodified Gram-Schmidt */
  hh  = HH(0,it);
  hes = HES(0,it);

  /* Clear hh and hes since we will accumulate values into them */
  for (j=0; j<=it; j++) {
    hh[j]  = 0.0;
    hes[j] = 0.0;
  }

  ncnt = 0;
  do {
    /* 
	 This is really a matrix-vector product, with the matrix stored
	 as pointer to rows 
    */
    VecMDot( it+1, VEC_VV(it+1), &(VEC_VV(0)), lhh ); /* <v,vnew> */

    /*
	 This is really a matrix vector product: 
	 [h[0],h[1],...]*[ v[0]; v[1]; ...] subtracted from v[it].
    */
    for (j=0; j<=it; j++) lhh[j] = - lhh[j];
    VecMAXPY(it+1, lhh, VEC_VV(it+1),&VEC_VV(0) );
    for (j=0; j<=it; j++) {
      hh[j]  -= lhh[j];     /* hh += <v,vnew> */
      hes[j] += lhh[j];     /* hes += - <v,vnew> */
    }

    /* Note that dnorm = (norm(d))**2 */
    dnorm = 0.0;
#if defined(PETSC_COMPLEX)
    for (j=0; j<=it; j++) dnorm += real(lhh[j] * conj(lhh[j]));
#else
    for (j=0; j<=it; j++) dnorm += lhh[j] * lhh[j];
#endif

    /* Continue until either we have only small corrections or we've done
	 as much work as a full orthogonalization (in terms of Mdots) */
  } while (dnorm > 1.0e-16 && ncnt++ < it);

  /* It would be nice to put ncnt somewhere.... */

  if (it >= 20) PetscFree( lhh );
  PLogEventEnd(KSP_GMRESOrthogonalization,ksp,0,0,0);
  return 0;
}

#if !defined(PETSC_COMPLEX)

/* hqr.f -- translated by f2c (version of 25 March 1992  12:58:56).*/

/* Cannot find d_sign in hpux libraries */
#if defined(PARCH_hpux)
double d_sign(double *x,double *y)
{
  if (*y >= 0.0) return *x;
  return -*x;
}
#endif

int KSPhqr_Private(int *nm, int *n, int *low, int *igh, double *h, 
                   double *wr, double *wi, int *ierr)
{
    /* System generated locals */
    int    h_dim1, h_offset, i__1, i__2, i__3;
    double d__1, d__2;
 
    /* Builtin functions */
    double d_sign();

    /* Local variables */
    double norm;
    int    i, j, k, l, m;
    double p, q, r, s, t, w, x, y;
    int    na, en, ll, mm;
    double zz;
    int    notlas;
    int    mp2, itn, its, enm2;
    double tst1, tst2;

/*     this subroutine is a translation of the algol procedure hqr, */
/*     num. math. 14, 219-231(1970) by martin, peters, and wilkinson. */
/*     handbook for auto. comp., vol.ii-linear algebra, 359-371(1971). */

/*     this subroutine finds the eigenvalues of a real */
/*     upper hessenberg matrix by the qr method. */

/*     on input */

/*        nm must be set to the row dimension of two-dimensional */
/*          array parameters as declared in the calling program */
/*          dimension statement. */

/*        n is the order of the matrix. */

/*        low and igh are integers determined by the balancing */
/*          subroutine  balanc.  if  balanc  has not been used, */
/*          set low=1, igh=n. */

/*        h contains the upper hessenberg matrix.  information about */
/*          the transformations used in the reduction to hessenberg */
/*          form by  elmhes  or  orthes, if performed, is stored */
/*          in the remaining triangle under the hessenberg matrix. */

/*     on output */

/*        h has been destroyed.  therefore, it must be saved */
/*          before calling  hqr  if subsequent calculation and */
/*          back transformation of eigenvectors is to be performed. */

/*        wr and wi contain the real and imaginary parts, */
/*          respectively, of the eigenvalues.  the eigenvalues */
/*          are unordered except that complex conjugate pairs */
/*          of values appear consecutively with the eigenvalue */
/*          having the positive imaginary part first.  if an */
/*          error exit is made, the eigenvalues should be correct */
/*          for indices ierr+1,...,n. */

/*        ierr is set to */
/*          zero       for normal return, */
/*          j          if the limit of 30*n iterations is exhausted */
/*                     while the j-th eigenvalue is being sought. */

/*     questions and comments should be directed to burton s. garbow, */
/*     mathematics and computer science div, argonne national laboratory 
*/

/*     this version dated september 1989. */

/*     ------------------------------------------------------------------ 
*/

    /* Parameter adjustments */
    --wi;
    --wr;
    h_dim1 = *nm;
    h_offset = h_dim1 + 1;
    h -= h_offset;

    /* Function Body */
    *ierr = 0;
    norm = 0.;
    k = 1;
/*     .......... store roots isolated by balanc */
/*                and compute matrix norm .......... */
    i__1 = *n;
    for (i = 1; i <= i__1; ++i) {

	i__2 = *n;
	for (j = k; j <= i__2; ++j) {
/* L40: */
	    norm += (d__1 = h[i + j * h_dim1], PetscAbsScalar(d__1));
	}

	k = i;
	if (i >= *low && i <= *igh) {
	    goto L50;
	}
	wr[i] = h[i + i * h_dim1];
	wi[i] = 0.;
L50:
	;
    }

    en = *igh;
    t = 0.;
    itn = *n * 30;
/*     .......... search for next eigenvalues .......... */
L60:
    if (en < *low) {
	goto L1001;
    }
    its = 0;
    na = en - 1;
    enm2 = na - 1;
/*     .......... look for single small sub-diagonal element */
/*                for l=en step -1 until low do -- .......... */
L70:
    i__1 = en;
    for (ll = *low; ll <= i__1; ++ll) {
	l = en + *low - ll;
	if (l == *low) {
	    goto L100;
	}
	s = (d__1 = h[l - 1 + (l - 1) * h_dim1], 
                PetscAbsScalar(d__1)) + (d__2 = h[l + l 
		* h_dim1], PetscAbsScalar(d__2));
	if (s == 0.) {
	    s = norm;
	}
	tst1 = s;
	tst2 = tst1 + (d__1 = h[l + (l - 1) * h_dim1], PetscAbsScalar(d__1));
	if (tst2 == tst1) {
	    goto L100;
	}
/* L80: */
    }
/*     .......... form shift .......... */
L100:
    x = h[en + en * h_dim1];
    if (l == en) {
	goto L270;
    }
    y = h[na + na * h_dim1];
    w = h[en + na * h_dim1] * h[na + en * h_dim1];
    if (l == na) {
	goto L280;
    }
    if (itn == 0) {
	goto L1000;
    }
    if (its != 10 && its != 20) {
	goto L130;
    }
/*     .......... form exceptional shift .......... */
    t += x;

    i__1 = en;
    for (i = *low; i <= i__1; ++i) {
/* L120: */
	h[i + i * h_dim1] -= x;
    }

    s = (d__1 = h[en + na * h_dim1], PetscAbsScalar(d__1)) + 
           (d__2 = h[na + enm2 * h_dim1], PetscAbsScalar(d__2));
    x = s * .75;
    y = x;
    w = s * -.4375 * s;
L130:
    ++its;
    --itn;
/*     .......... look for two consecutive small */
/*                sub-diagonal elements. */
/*                for m=en-2 step -1 until l do -- .......... */
    i__1 = enm2;
    for (mm = l; mm <= i__1; ++mm) {
	m = enm2 + l - mm;
	zz = h[m + m * h_dim1];
	r = x - zz;
	s = y - zz;
	p = (r * s - w) / h[m + 1 + m * h_dim1] + h[m + (m + 1) * h_dim1];
	q = h[m + 1 + (m + 1) * h_dim1] - zz - r - s;
	r = h[m + 2 + (m + 1) * h_dim1];
	s = PetscAbsScalar(p) + PetscAbsScalar(q) + PetscAbsScalar(r);
	p /= s;
	q /= s;
	r /= s;
	if (m == l) {
	    goto L150;
	}
	tst1 = PetscAbsScalar(p) * ((d__1 = h[m - 1 + (m - 1) * h_dim1],
                PetscAbsScalar(d__1)) + 
		PetscAbsScalar(zz) + (d__2 = h[m + 1 + (m + 1) * h_dim1],
                PetscAbsScalar(d__2)));
	tst2 = tst1 + (d__1 = h[m + (m - 1) * h_dim1], 
                PetscAbsScalar(d__1)) * (PetscAbsScalar(q) + 
		PetscAbsScalar(r));
	if (tst2 == tst1) {
	    goto L150;
	}
/* L140: */
    }

L150:
    mp2 = m + 2;

    i__1 = en;
    for (i = mp2; i <= i__1; ++i) {
	h[i + (i - 2) * h_dim1] = 0.;
	if (i == mp2) {
	    goto L160;
	}
	h[i + (i - 3) * h_dim1] = 0.;
L160:
	;
    }
/*     .......... double qr step involving rows l to en and */
/*                columns m to en .......... */
    i__1 = na;
    for (k = m; k <= i__1; ++k) {
	notlas = k != na;
	if (k == m) {
	    goto L170;
	}
	p = h[k + (k - 1) * h_dim1];
	q = h[k + 1 + (k - 1) * h_dim1];
	r = 0.;
	if (notlas) {
	    r = h[k + 2 + (k - 1) * h_dim1];
	}
	x = PetscAbsScalar(p) + PetscAbsScalar(q) + PetscAbsScalar(r);
	if (x == 0.) {
	    goto L260;
	}
	p /= x;
	q /= x;
	r /= x;
L170:
	d__1 = sqrt(p * p + q * q + r * r);
	s = d_sign(&d__1, &p);
	if (k == m) {
	    goto L180;
	}
	h[k + (k - 1) * h_dim1] = -s * x;
	goto L190;
L180:
	if (l != m) {
	    h[k + (k - 1) * h_dim1] = -h[k + (k - 1) * h_dim1];
	}
L190:
	p += s;
	x = p / s;
	y = q / s;
	zz = r / s;
	q /= p;
	r /= p;
	if (notlas) {
	    goto L225;
	}
/*     .......... row modification .......... */
	i__2 = en;
	for (j = k; j <= i__2; ++j) {
	    p = h[k + j * h_dim1] + q * h[k + 1 + j * h_dim1];
	    h[k + j * h_dim1] -= p * x;
	    h[k + 1 + j * h_dim1] -= p * y;
/* L200: */
	}

/* Computing MIN */
	i__2 = en, i__3 = k + 3;
	j = PetscMin(i__2,i__3);
/*     .......... column modification .......... */
	i__2 = j;
	for (i = l; i <= i__2; ++i) {
	    p = x * h[i + k * h_dim1] + y * h[i + (k + 1) * h_dim1];
	    h[i + k * h_dim1] -= p;
	    h[i + (k + 1) * h_dim1] -= p * q;
/* L210: */
	}
	goto L255;
L225:
/*     .......... row modification .......... */
	i__2 = en;
	for (j = k; j <= i__2; ++j) {
	    p = h[k + j * h_dim1] + q * h[k + 1 + j * h_dim1] + r * h[k + 2 + 
		    j * h_dim1];
	    h[k + j * h_dim1] -= p * x;
	    h[k + 1 + j * h_dim1] -= p * y;
	    h[k + 2 + j * h_dim1] -= p * zz;
/* L230: */
	}

/* Computing MIN */
	i__2 = en, i__3 = k + 3;
	j = PetscMin(i__2,i__3);
/*     .......... column modification .......... */
	i__2 = j;
	for (i = l; i <= i__2; ++i) {
	    p = x * h[i + k * h_dim1] + y * h[i + (k + 1) * h_dim1] + zz * h[
		    i + (k + 2) * h_dim1];
	    h[i + k * h_dim1] -= p;
	    h[i + (k + 1) * h_dim1] -= p * q;
	    h[i + (k + 2) * h_dim1] -= p * r;
/* L240: */
	}
L255:

L260:
	;
    }

    goto L70;
/*     .......... one root found .......... */
L270:
    wr[en] = x + t;
    wi[en] = 0.;
    en = na;
    goto L60;
/*     .......... two roots found .......... */
L280:
    p = (y - x) / 2.;
    q = p * p + w;
    zz = sqrt((PetscAbsScalar(q)));
    x += t;
    if (q < 0.) {
	goto L320;
    }
/*     .......... real pair .......... */
    zz = p + d_sign(&zz, &p);
    wr[na] = x + zz;
    wr[en] = wr[na];
    if (zz != 0.) {
	wr[en] = x - w / zz;
    }
    wi[na] = 0.;
    wi[en] = 0.;
    goto L330;
/*     .......... complex pair .......... */
L320:
    wr[na] = x + p;
    wr[en] = x + p;
    wi[na] = zz;
    wi[en] = -zz;
L330:
    en = enm2;
    goto L60;
/*     .......... set error -- all eigenvalues have not */
/*                converged after 30*n iterations .......... */
L1000:
    *ierr = en;
L1001:
    return 0;
}

#include "pinclude/plapack.h"
/*
    The Hessenberg matrix is factored at H = QR by the 
  GMRESUpdateHessenberg() routine. This routine computes the 
  singular values of R as estimates for the singular values of 
  the preconditioned operator.
*/
int KSPComputeExtremeSingularvalues_GMRES(KSP ksp,Scalar *emax,Scalar *emin)
{
  KSP_GMRES *gmres = (KSP_GMRES *) ksp->data;
  int       n = gmres->it + 1, N = gmres->max_k + 2, ierr, lwork = 5*N;
  int       idummy = N, i;
  Scalar    *R = gmres->Rsvd;
  Scalar    *realpart = R + N*N, *work = realpart + N;
  Scalar    sdummy;

  if (n == 0) {
    *emax = *emin = 1.0;
    return 0;
  }
  /* copy R matrix to work space */
  PetscMemcpy(R,gmres->hh_origin,N*N*sizeof(Scalar));

  /* zero below diagonal garbage */
  for ( i=0; i<n; i++ ) {
    R[i*N+i+1] = 0.0;
  }
  
  /*
    DoubleView(N*N,R,0);  
    DoubleView((N-1)*(N-1),gmres->hes_origin);
  */
 
  /* compute Singularvalues */
  /*
     KSPhqr_Private(&N,&n,&one,&n,R,realpart,imagpart,&ierr); 
     combine real and imaginary parts and sort 
     for ( i=0; i<n; i++ ) {
       realpart[i] = sqrt(realpart[i]*realpart[i] + imagpart[i]*imagpart[i]);
     }
     PetscSortDouble(n,realpart);
  */

  LAgesvd_("N","N",&n,&n,R,&N,realpart,&sdummy,&idummy,&sdummy,
           &idummy,work,&lwork,&ierr);

  if (ierr) SETERRQ(1,"KSPComputeExtremeSingularvalues_GMRES:Error in SVD");


  *emin = realpart[n-1];
  *emax = realpart[0];

  return 0;
}

#else

int KSPComputeExtremeSingularvalues_GMRES(KSP ksp,Scalar *emax,Scalar *emin)
{
  fprintf(stderr,"KSPComputeExtremeSingularvalues_GMRES:No code for complex case\n");
  return 0;
}

#endif


