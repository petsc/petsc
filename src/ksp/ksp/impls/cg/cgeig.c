
/*
      Code for calculating extreme eigenvalues via the Lanczo method
   running with CG. Note this only works for symmetric real and Hermitian
   matrices (not complex matrices that are symmetric).
*/
#include <../src/ksp/ksp/impls/cg/cgimpl.h>
static PetscErrorCode LINPACKcgtql1(PetscInt*,PetscReal*,PetscReal*,PetscInt*);

PetscErrorCode KSPComputeEigenvalues_CG(KSP ksp,PetscInt nmax,PetscReal *r,PetscReal *c,PetscInt *neig)
{
  KSP_CG         *cgP = (KSP_CG*)ksp->data;
  PetscScalar    *d,*e;
  PetscReal      *ee;
  PetscInt       j,n = ksp->its;

  PetscFunctionBegin;
  PetscCheck(nmax >= n,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_SIZ,"Not enough room in work space r and c for eigenvalues");
  *neig = n;

  PetscCall(PetscArrayzero(c,nmax));
  if (!n) {
    PetscFunctionReturn(0);
  }
  d = cgP->d; e = cgP->e; ee = cgP->ee;

  /* copy tridiagonal matrix to work space */
  for (j=0; j<n; j++) {
    r[j]  = PetscRealPart(d[j]);
    ee[j] = PetscRealPart(e[j]);
  }

  LINPACKcgtql1(&n,r,ee,&j);
  PetscCheck(j == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from tql1(); eispack eigenvalue routine");
  PetscCall(PetscSortReal(n,r));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP ksp,PetscReal *emax,PetscReal *emin)
{
  KSP_CG      *cgP = (KSP_CG*)ksp->data;
  PetscScalar *d,*e;
  PetscReal   *dd,*ee;
  PetscInt    j,n = ksp->its;

  PetscFunctionBegin;
  if (!n) {
    *emax = *emin = 1.0;
    PetscFunctionReturn(0);
  }
  d = cgP->d; e = cgP->e; dd = cgP->dd; ee = cgP->ee;

  /* copy tridiagonal matrix to work space */
  for (j=0; j<n; j++) {
    dd[j] = PetscRealPart(d[j]);
    ee[j] = PetscRealPart(e[j]);
  }

  LINPACKcgtql1(&n,dd,ee,&j);
  PetscCheck(j == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from tql1(); eispack eigenvalue routine");
  *emin = dd[0]; *emax = dd[n-1];
  PetscFunctionReturn(0);
}

/* tql1.f -- translated by f2c (version of 25 March 1992  12:58:56).
   By Barry Smith on March 27, 1994.
   Eispack routine to determine eigenvalues of symmetric
   tridiagonal matrix

  Note that this routine always uses real numbers (not complex) even if the underlying
  matrix is Hermitian. This is because the Lanczos process applied to Hermitian matrices
  always produces a real, symmetric tridiagonal matrix.
*/

static PetscReal LINPACKcgpthy(PetscReal*,PetscReal*);

static PetscErrorCode LINPACKcgtql1(PetscInt *n,PetscReal *d,PetscReal *e,PetscInt *ierr)
{
  /* System generated locals */
  PetscInt  i__1,i__2;
  PetscReal d__1,d__2,c_b10 = 1.0;

  /* Local variables */
  PetscReal c,f,g,h;
  PetscInt  i,j,l,m;
  PetscReal p,r,s,c2,c3 = 0.0;
  PetscInt  l1,l2;
  PetscReal s2 = 0.0;
  PetscInt  ii;
  PetscReal dl1,el1;
  PetscInt  mml;
  PetscReal tst1,tst2;

/*     THIS SUBROUTINE IS A TRANSLATION OF THE ALGOL PROCEDURE TQL1, */
/*     NUM. MATH. 11, 293-306(1968) BY BOWDLER, MARTIN, REINSCH, AND */
/*     WILKINSON. */
/*     HANDBOOK FOR AUTO. COMP., VOL.II-LINEAR ALGEBRA, 227-240(1971). */

/*     THIS SUBROUTINE FINDS THE EIGENVALUES OF A SYMMETRIC */
/*     TRIDIAGONAL MATRIX BY THE QL METHOD. */

/*     ON INPUT */

/*        N IS THE ORDER OF THE MATRIX. */

/*        D CONTAINS THE DIAGONAL ELEMENTS OF THE INPUT MATRIX. */

/*        E CONTAINS THE SUBDIAGONAL ELEMENTS OF THE INPUT MATRIX */
/*          IN ITS LAST N-1 POSITIONS.  E(1) IS ARBITRARY. */

/*      ON OUTPUT */

/*        D CONTAINS THE EIGENVALUES IN ASCENDING ORDER.  IF AN */
/*          ERROR EXIT IS MADE, THE EIGENVALUES ARE CORRECT AND */
/*          ORDERED FOR INDICES 1,2,...IERR-1, BUT MAY NOT BE */
/*          THE SMALLEST EIGENVALUES. */

/*        E HAS BEEN DESTROYED. */

/*        IERR IS SET TO */
/*          ZERO       FOR NORMAL RETURN, */
/*          J          IF THE J-TH EIGENVALUE HAS NOT BEEN */
/*                     DETERMINED AFTER 30 ITERATIONS. */

/*     CALLS CGPTHY FOR  DSQRT(A*A + B*B) . */

/*     QUESTIONS AND COMMENTS SHOULD BE DIRECTED TO BURTON S. GARBOW, */
/*     MATHEMATICS AND COMPUTER SCIENCE DIV, ARGONNE NATIONAL LABORATORY
*/

/*     THIS VERSION DATED AUGUST 1983. */

/*     ------------------------------------------------------------------
*/
  PetscReal ds;

  PetscFunctionBegin;
  --e;
  --d;

  *ierr = 0;
  if (*n == 1) goto L1001;

  i__1 = *n;
  for (i = 2; i <= i__1; ++i) e[i - 1] = e[i];

  f     = 0.;
  tst1  = 0.;
  e[*n] = 0.;

  i__1 = *n;
  for (l = 1; l <= i__1; ++l) {
    j = 0;
    d__1 = d[l];
    d__2 = e[l];
    h = PetscAbsReal(d__1) + PetscAbsReal(d__2);
    if (tst1 < h) tst1 = h;
/*     .......... LOOK FOR SMALL SUB-DIAGONAL ELEMENT .......... */
    i__2 = *n;
    for (m = l; m <= i__2; ++m) {
      d__1 = e[m];
      tst2 = tst1 + PetscAbsReal(d__1);
      if (tst2 == tst1) goto L120;
/*     .......... E(N) IS ALWAYS ZERO,SO THERE IS NO EXIT */
/*                THROUGH THE BOTTOM OF THE LOOP .......... */
    }
L120:
    if (m == l) goto L210;
L130:
    if (j == 30) goto L1000;
    ++j;
/*     .......... FORM SHIFT .......... */
    l1    = l + 1;
    l2    = l1 + 1;
    g     = d[l];
    p     = (d[l1] - g) / (e[l] * 2.);
    r     = LINPACKcgpthy(&p,&c_b10);
    ds    = 1.0; if (p < 0.0) ds = -1.0;
    d[l]  = e[l] / (p + ds*r);
    d[l1] = e[l] * (p + ds*r);
    dl1   = d[l1];
    h     = g - d[l];
    if (l2 > *n) goto L145;

    i__2 = *n;
    for (i = l2; i <= i__2; ++i) d[i] -= h;

L145:
    f += h;
/*     .......... QL TRANSFORMATION .......... */
    p   = d[m];
    c   = 1.;
    c2  = c;
    el1 = e[l1];
    s   = 0.;
    mml = m - l;
/*     .......... FOR I=M-1 STEP -1 UNTIL L DO -- .......... */
    i__2 = mml;
    for (ii = 1; ii <= i__2; ++ii) {
      c3       = c2;
      c2       = c;
      s2       = s;
      i        = m - ii;
      g        = c * e[i];
      h        = c * p;
      r        = LINPACKcgpthy(&p,&e[i]);
      e[i + 1] = s * r;
      s        = e[i] / r;
      c        = p / r;
      p        = c * d[i] - s * g;
      d[i + 1] = h + s * (c * g + s * d[i]);
    }

    p    = -s * s2 * c3 * el1 * e[l] / dl1;
    e[l] = s * p;
    d[l] = c * p;
    d__1 = e[l];
    tst2 = tst1 + PetscAbsReal(d__1);
    if (tst2 > tst1) goto L130;
L210:
    p = d[l] + f;
/*     .......... ORDER EIGENVALUES .......... */
    if (l == 1) goto L250;
/*     .......... FOR I=L STEP -1 UNTIL 2 DO -- .......... */
    i__2 = l;
    for (ii = 2; ii <= i__2; ++ii) {
      i = l + 2 - ii;
      if (p >= d[i - 1]) goto L270;
      d[i] = d[i - 1];
    }

L250:
    i = 1;
L270:
    d[i] = p;
  }

  goto L1001;
/*     .......... SET ERROR -- NO CONVERGENCE TO AN */
/*                EIGENVALUE AFTER 30 ITERATIONS .......... */
L1000:
  *ierr = l;
L1001:
  PetscFunctionReturn(0);
} /* cgtql1_ */

static PetscReal LINPACKcgpthy(PetscReal *a,PetscReal *b)
{
  /* System generated locals */
  PetscReal ret_val,d__1,d__2,d__3;

  /* Local variables */
  PetscReal p,r,s,t,u;

  PetscFunctionBegin;
/*     FINDS DSQRT(A**2+B**2) WITHOUT OVERFLOW OR DESTRUCTIVE UNDERFLOW */

/* Computing MAX */
  d__1 = PetscAbsReal(*a);
  d__2 = PetscAbsReal(*b);
  p    = PetscMax(d__1,d__2);
  if (!p) goto L20;
/* Computing MIN */
  d__2 = PetscAbsReal(*a);
  d__3 = PetscAbsReal(*b);
/* Computing 2nd power */
  d__1 = PetscMin(d__2,d__3) / p;
  r    = d__1 * d__1;
L10:
  t = r + 4.;
  if (t == 4.) goto L20;
  s = r / t;
  u = s * 2. + 1.;
  p = u * p;
/* Computing 2nd power */
  d__1 = s / u;
  r    = d__1 * d__1 * r;
  goto L10;
L20:
  ret_val = p;
  PetscFunctionReturn(ret_val);
} /* cgpthy_ */
