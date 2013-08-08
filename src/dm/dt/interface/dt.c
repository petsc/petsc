/* Discretization tools */

#include <petscconf.h>
#if defined(PETSC_HAVE_MATHIMF_H)
#include <mathimf.h>           /* this needs to be included before math.h */
#endif

#include <petscdt.h>            /*I "petscdt.h" I*/ /*I "petscfe.h" I*/
#include <petscblaslapack.h>
#include <petsc-private/petscimpl.h>
#include <petscviewer.h>

#undef __FUNCT__
#define __FUNCT__ "PetscDTLegendreEval"
/*@
   PetscDTLegendreEval - evaluate Legendre polynomial at points

   Not Collective

   Input Arguments:
+  npoints - number of spatial points to evaluate at
.  points - array of locations to evaluate at
.  ndegree - number of basis degrees to evaluate
-  degrees - sorted array of degrees to evaluate

   Output Arguments:
+  B - row-oriented basis evaluation matrix B[point*ndegree + degree] (dimension npoints*ndegrees, allocated by caller) (or NULL)
.  D - row-oriented derivative evaluation matrix (or NULL)
-  D2 - row-oriented second derivative evaluation matrix (or NULL)

   Level: intermediate

.seealso: PetscDTGaussQuadrature()
@*/
PetscErrorCode PetscDTLegendreEval(PetscInt npoints,const PetscReal *points,PetscInt ndegree,const PetscInt *degrees,PetscReal *B,PetscReal *D,PetscReal *D2)
{
  PetscInt i,maxdegree;

  PetscFunctionBegin;
  if (!npoints || !ndegree) PetscFunctionReturn(0);
  maxdegree = degrees[ndegree-1];
  for (i=0; i<npoints; i++) {
    PetscReal pm1,pm2,pd1,pd2,pdd1,pdd2,x;
    PetscInt  j,k;
    x    = points[i];
    pm2  = 0;
    pm1  = 1;
    pd2  = 0;
    pd1  = 0;
    pdd2 = 0;
    pdd1 = 0;
    k    = 0;
    if (degrees[k] == 0) {
      if (B) B[i*ndegree+k] = pm1;
      if (D) D[i*ndegree+k] = pd1;
      if (D2) D2[i*ndegree+k] = pdd1;
      k++;
    }
    for (j=1; j<=maxdegree; j++,k++) {
      PetscReal p,d,dd;
      p    = ((2*j-1)*x*pm1 - (j-1)*pm2)/j;
      d    = pd2 + (2*j-1)*pm1;
      dd   = pdd2 + (2*j-1)*pd1;
      pm2  = pm1;
      pm1  = p;
      pd2  = pd1;
      pd1  = d;
      pdd2 = pdd1;
      pdd1 = dd;
      if (degrees[k] == j) {
        if (B) B[i*ndegree+k] = p;
        if (D) D[i*ndegree+k] = d;
        if (D2) D2[i*ndegree+k] = dd;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTGaussQuadrature"
/*@
   PetscDTGaussQuadrature - create Gauss quadrature

   Not Collective

   Input Arguments:
+  npoints - number of points
.  a - left end of interval (often-1)
-  b - right end of interval (often +1)

   Output Arguments:
+  x - quadrature points
-  w - quadrature weights

   Level: intermediate

   References:
   Golub and Welsch, Calculation of Quadrature Rules, Math. Comp. 23(106), 221--230, 1969.

.seealso: PetscDTLegendreEval()
@*/
PetscErrorCode PetscDTGaussQuadrature(PetscInt npoints,PetscReal a,PetscReal b,PetscReal *x,PetscReal *w)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      *work;
  PetscScalar    *Z;
  PetscBLASInt   N,LDZ,info;

  PetscFunctionBegin;
  /* Set up the Golub-Welsch system */
  for (i=0; i<npoints; i++) {
    x[i] = 0;                   /* diagonal is 0 */
    if (i) w[i-1] = 0.5 / PetscSqrtReal(1 - 1./PetscSqr(2*i));
  }
  ierr = PetscRealView(npoints-1,w,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscMalloc2(npoints*npoints,PetscScalar,&Z,PetscMax(1,2*npoints-2),PetscReal,&work);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(npoints,&N);CHKERRQ(ierr);
  LDZ  = N;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKsteqr",LAPACKsteqr_("I",&N,x,w,Z,&LDZ,work,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"xSTEQR error");

  for (i=0; i<(npoints+1)/2; i++) {
    PetscReal y = 0.5 * (-x[i] + x[npoints-i-1]); /* enforces symmetry */
    x[i]           = (a+b)/2 - y*(b-a)/2;
    x[npoints-i-1] = (a+b)/2 + y*(b-a)/2;

    w[i] = w[npoints-1-i] = (b-a)*PetscSqr(0.5*PetscAbsScalar(Z[i*npoints] + Z[(npoints-i-1)*npoints]));
  }
  ierr = PetscFree2(Z,work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTFactorial_Internal"
/* Evaluates the nth jacobi polynomial with weight parameters a,b at a point x.
   Recurrence relations implemented from the pseudocode given in Karniadakis and Sherwin, Appendix B */
PETSC_STATIC_INLINE PetscErrorCode PetscDTFactorial_Internal(PetscInt n, PetscReal *factorial)
{
  PetscReal f = 1.0;
  PetscInt  i;

  PetscFunctionBegin;
  for (i = 1; i < n+1; ++i) f *= i;
  *factorial = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTComputeJacobi"
/* Evaluates the nth jacobi polynomial with weight parameters a,b at a point x.
   Recurrence relations implemented from the pseudocode given in Karniadakis and Sherwin, Appendix B */
PETSC_STATIC_INLINE PetscErrorCode PetscDTComputeJacobi(PetscReal a, PetscReal b, PetscInt n, PetscReal x, PetscReal *P)
{
  PetscReal apb, pn1, pn2;
  PetscInt  k;

  PetscFunctionBegin;
  if (!n) {*P = 1.0; PetscFunctionReturn(0);}
  if (n == 1) {*P = 0.5 * (a - b + (a + b + 2.0) * x); PetscFunctionReturn(0);}
  apb = a + b;
  pn2 = 1.0;
  pn1 = 0.5 * (a - b + (apb + 2.0) * x);
  *P  = 0.0;
  for (k = 2; k < n+1; ++k) {
    PetscReal a1 = 2.0 * k * (k + apb) * (2.0*k + apb - 2.0);
    PetscReal a2 = (2.0 * k + apb - 1.0) * (a*a - b*b);
    PetscReal a3 = (2.0 * k + apb - 2.0) * (2.0 * k + apb - 1.0) * (2.0 * k + apb);
    PetscReal a4 = 2.0 * (k + a - 1.0) * (k + b - 1.0) * (2.0 * k + apb);

    a2  = a2 / a1;
    a3  = a3 / a1;
    a4  = a4 / a1;
    *P  = (a2 + a3 * x) * pn1 - a4 * pn2;
    pn2 = pn1;
    pn1 = *P;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTComputeJacobiDerivative"
/* Evaluates the first derivative of P_{n}^{a,b} at a point x. */
PETSC_STATIC_INLINE PetscErrorCode PetscDTComputeJacobiDerivative(PetscReal a, PetscReal b, PetscInt n, PetscReal x, PetscReal *P)
{
  PetscReal      nP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!n) {*P = 0.0; PetscFunctionReturn(0);}
  ierr = PetscDTComputeJacobi(a+1, b+1, n-1, x, &nP);CHKERRQ(ierr);
  *P   = 0.5 * (a + b + n + 1) * nP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTMapSquareToTriangle_Internal"
/* Maps from [-1,1]^2 to the (-1,1) reference triangle */
PETSC_STATIC_INLINE PetscErrorCode PetscDTMapSquareToTriangle_Internal(PetscReal x, PetscReal y, PetscReal *xi, PetscReal *eta)
{
  PetscFunctionBegin;
  *xi  = 0.5 * (1.0 + x) * (1.0 - y) - 1.0;
  *eta = y;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTMapCubeToTetrahedron_Internal"
/* Maps from [-1,1]^2 to the (-1,1) reference triangle */
PETSC_STATIC_INLINE PetscErrorCode PetscDTMapCubeToTetrahedron_Internal(PetscReal x, PetscReal y, PetscReal z, PetscReal *xi, PetscReal *eta, PetscReal *zeta)
{
  PetscFunctionBegin;
  *xi   = 0.25 * (1.0 + x) * (1.0 - y) * (1.0 - z) - 1.0;
  *eta  = 0.5  * (1.0 + y) * (1.0 - z) - 1.0;
  *zeta = z;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTGaussJacobiQuadrature1D_Internal"
static PetscErrorCode PetscDTGaussJacobiQuadrature1D_Internal(PetscInt npoints, PetscReal a, PetscReal b, PetscReal *x, PetscReal *w)
{
  PetscInt       maxIter = 100;
  PetscReal      eps     = 1.0e-8;
  PetscReal      a1, a2, a3, a4, a5, a6;
  PetscInt       k;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  a1      = pow(2, a+b+1);
#if defined(PETSC_HAVE_TGAMMA)
  a2      = tgamma(a + npoints + 1);
  a3      = tgamma(b + npoints + 1);
  a4      = tgamma(a + b + npoints + 1);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"tgamma() - math routine is unavailable.");
#endif

  ierr = PetscDTFactorial_Internal(npoints, &a5);CHKERRQ(ierr);
  a6   = a1 * a2 * a3 / a4 / a5;
  /* Computes the m roots of P_{m}^{a,b} on [-1,1] by Newton's method with Chebyshev points as initial guesses.
   Algorithm implemented from the pseudocode given by Karniadakis and Sherwin and Python in FIAT */
  for (k = 0; k < npoints; ++k) {
    PetscReal r = -cos((2.0*k + 1.0) * PETSC_PI / (2.0 * npoints)), dP;
    PetscInt  j;

    if (k > 0) r = 0.5 * (r + x[k-1]);
    for (j = 0; j < maxIter; ++j) {
      PetscReal s = 0.0, delta, f, fp;
      PetscInt  i;

      for (i = 0; i < k; ++i) s = s + 1.0 / (r - x[i]);
      ierr = PetscDTComputeJacobi(a, b, npoints, r, &f);CHKERRQ(ierr);
      ierr = PetscDTComputeJacobiDerivative(a, b, npoints, r, &fp);CHKERRQ(ierr);
      delta = f / (fp - f * s);
      r     = r - delta;
      if (fabs(delta) < eps) break;
    }
    x[k] = r;
    ierr = PetscDTComputeJacobiDerivative(a, b, npoints, x[k], &dP);CHKERRQ(ierr);
    w[k] = a6 / (1.0 - PetscSqr(x[k])) / PetscSqr(dP);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTGaussJacobiQuadrature"
/*@C
  PetscDTGaussJacobiQuadrature - create Gauss-Jacobi quadrature for a simplex

  Not Collective

  Input Arguments:
+ dim - The simplex dimension
. npoints - number of points
. a - left end of interval (often-1)
- b - right end of interval (often +1)

  Output Arguments:
+ points - quadrature points
- weights - quadrature weights

  Level: intermediate

  References:
  Karniadakis and Sherwin.
  FIAT

.seealso: PetscDTGaussQuadrature()
@*/
PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt dim, PetscInt npoints, PetscReal a, PetscReal b, PetscReal *points[], PetscReal *weights[])
{
  PetscReal     *px, *wx, *py, *wy, *pz, *wz, *x, *w;
  PetscInt       i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((a != -1.0) || (b != 1.0)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must use default internal right now");
  switch (dim) {
  case 1:
    ierr = PetscMalloc(npoints * sizeof(PetscReal), &x);CHKERRQ(ierr);
    ierr = PetscMalloc(npoints * sizeof(PetscReal), &w);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(npoints, 0.0, 0.0, x, w);CHKERRQ(ierr);
    break;
  case 2:
    ierr = PetscMalloc(npoints*npoints*2 * sizeof(PetscReal), &x);CHKERRQ(ierr);
    ierr = PetscMalloc(npoints*npoints   * sizeof(PetscReal), &w);CHKERRQ(ierr);
    ierr = PetscMalloc4(npoints,PetscReal,&px,npoints,PetscReal,&wx,npoints,PetscReal,&py,npoints,PetscReal,&wy);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(npoints, 0.0, 0.0, px, wx);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(npoints, 1.0, 0.0, py, wy);CHKERRQ(ierr);
    for (i = 0; i < npoints; ++i) {
      for (j = 0; j < npoints; ++j) {
        ierr = PetscDTMapSquareToTriangle_Internal(px[i], py[j], &x[(i*npoints+j)*2+0], &x[(i*npoints+j)*2+1]);CHKERRQ(ierr);
        w[i*npoints+j] = 0.5 * wx[i] * wy[j];
      }
    }
    ierr = PetscFree4(px,wx,py,wy);CHKERRQ(ierr);
    break;
  case 3:
    ierr = PetscMalloc(npoints*npoints*3 * sizeof(PetscReal), &x);CHKERRQ(ierr);
    ierr = PetscMalloc(npoints*npoints   * sizeof(PetscReal), &w);CHKERRQ(ierr);
    ierr = PetscMalloc6(npoints,PetscReal,&px,npoints,PetscReal,&wx,npoints,PetscReal,&py,npoints,PetscReal,&wy,npoints,PetscReal,&pz,npoints,PetscReal,&wz);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(npoints, 0.0, 0.0, px, wx);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(npoints, 1.0, 0.0, py, wy);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(npoints, 2.0, 0.0, pz, wz);CHKERRQ(ierr);
    for (i = 0; i < npoints; ++i) {
      for (j = 0; j < npoints; ++j) {
        for (k = 0; k < npoints; ++k) {
          ierr = PetscDTMapCubeToTetrahedron_Internal(px[i], py[j], pz[k], &x[((i*npoints+j)*npoints+k)*3+0], &x[((i*npoints+j)*npoints+k)*3+1], &x[((i*npoints+j)*npoints+k)*3+2]);CHKERRQ(ierr);
          w[(i*npoints+j)*npoints+k] = 0.125 * wx[i] * wy[j] * wz[k];
        }
      }
    }
    ierr = PetscFree6(px,wx,py,wy,pz,wz);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot construct quadrature rule for dimension %d", dim);
  }
  if (points)  *points  = x;
  if (weights) *weights = w;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTPseudoInverseQR"
/* Overwrites A. Can only handle full-rank problems with m>=n
 * A in column-major format
 * Ainv in row-major format
 * tau has length m
 * worksize must be >= max(1,n)
 */
static PetscErrorCode PetscDTPseudoInverseQR(PetscInt m,PetscInt mstride,PetscInt n,PetscReal *A_in,PetscReal *Ainv_out,PetscScalar *tau,PetscInt worksize,PetscScalar *work)
{
  PetscErrorCode ierr;
  PetscBLASInt M,N,K,lda,ldb,ldwork,info;
  PetscScalar *A,*Ainv,*R,*Q,Alpha;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  {
    PetscInt i,j;
    ierr = PetscMalloc2(m*n,PetscScalar,&A,m*n,PetscScalar,&Ainv);CHKERRQ(ierr);
    for (j=0; j<n; j++) {
      for (i=0; i<m; i++) A[i+m*j] = A_in[i+mstride*j];
    }
    mstride = m;
  }
#else
  A = A_in;
  Ainv = Ainv_out;
#endif

  ierr = PetscBLASIntCast(m,&M);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&N);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(mstride,&lda);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(worksize,&ldwork);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  LAPACKgeqrf_(&M,&N,A,&lda,tau,work,&ldwork,&info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xGEQRF error");
  R = A; /* Upper triangular part of A now contains R, the rest contains the elementary reflectors */

  /* Extract an explicit representation of Q */
  Q = Ainv;
  ierr = PetscMemcpy(Q,A,mstride*n*sizeof(PetscScalar));CHKERRQ(ierr);
  K = N;                        /* full rank */
  LAPACKungqr_(&M,&N,&K,Q,&lda,tau,work,&ldwork,&info);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"xORGQR/xUNGQR error");

  /* Compute A^{-T} = (R^{-1} Q^T)^T = Q R^{-T} */
  Alpha = 1.0;
  ldb = lda;
  BLAStrsm_("Right","Upper","ConjugateTranspose","NotUnitTriangular",&M,&N,&Alpha,R,&lda,Q,&ldb);
  /* Ainv is Q, overwritten with inverse */

#if defined(PETSC_USE_COMPLEX)
  {
    PetscInt i;
    for (i=0; i<m*n; i++) Ainv_out[i] = PetscRealPart(Ainv[i]);
    ierr = PetscFree2(A,Ainv);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTLegendreIntegrate"
/* Computes integral of L_p' over intervals {(x0,x1),(x1,x2),...} */
static PetscErrorCode PetscDTLegendreIntegrate(PetscInt ninterval,const PetscReal *x,PetscInt ndegree,const PetscInt *degrees,PetscBool Transpose,PetscReal *B)
{
  PetscErrorCode ierr;
  PetscReal *Bv;
  PetscInt i,j;

  PetscFunctionBegin;
  ierr = PetscMalloc((ninterval+1)*ndegree*sizeof(PetscReal),&Bv);CHKERRQ(ierr);
  /* Point evaluation of L_p on all the source vertices */
  ierr = PetscDTLegendreEval(ninterval+1,x,ndegree,degrees,Bv,NULL,NULL);CHKERRQ(ierr);
  /* Integral over each interval: \int_a^b L_p' = L_p(b)-L_p(a) */
  for (i=0; i<ninterval; i++) {
    for (j=0; j<ndegree; j++) {
      if (Transpose) B[i+ninterval*j] = Bv[(i+1)*ndegree+j] - Bv[i*ndegree+j];
      else           B[i*ndegree+j]   = Bv[(i+1)*ndegree+j] - Bv[i*ndegree+j];
    }
  }
  ierr = PetscFree(Bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDTReconstructPoly"
/*@
   PetscDTReconstructPoly - create matrix representing polynomial reconstruction using cell intervals and evaluation at target intervals

   Not Collective

   Input Arguments:
+  degree - degree of reconstruction polynomial
.  nsource - number of source intervals
.  sourcex - sorted coordinates of source cell boundaries (length nsource+1)
.  ntarget - number of target intervals
-  targetx - sorted coordinates of target cell boundaries (length ntarget+1)

   Output Arguments:
.  R - reconstruction matrix, utarget = sum_s R[t*nsource+s] * usource[s]

   Level: advanced

.seealso: PetscDTLegendreEval()
@*/
PetscErrorCode PetscDTReconstructPoly(PetscInt degree,PetscInt nsource,const PetscReal *sourcex,PetscInt ntarget,const PetscReal *targetx,PetscReal *R)
{
  PetscErrorCode ierr;
  PetscInt i,j,k,*bdegrees,worksize;
  PetscReal xmin,xmax,center,hscale,*sourcey,*targety,*Bsource,*Bsinv,*Btarget;
  PetscScalar *tau,*work;

  PetscFunctionBegin;
  PetscValidRealPointer(sourcex,3);
  PetscValidRealPointer(targetx,5);
  PetscValidRealPointer(R,6);
  if (degree >= nsource) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Reconstruction degree %D must be less than number of source intervals %D",degree,nsource);
#if defined(PETSC_USE_DEBUG)
  for (i=0; i<nsource; i++) {
    if (sourcex[i] >= sourcex[i+1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Source interval %D has negative orientation (%G,%G)",i,sourcex[i],sourcex[i+1]);
  }
  for (i=0; i<ntarget; i++) {
    if (targetx[i] >= targetx[i+1]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Target interval %D has negative orientation (%G,%G)",i,targetx[i],targetx[i+1]);
  }
#endif
  xmin = PetscMin(sourcex[0],targetx[0]);
  xmax = PetscMax(sourcex[nsource],targetx[ntarget]);
  center = (xmin + xmax)/2;
  hscale = (xmax - xmin)/2;
  worksize = nsource;
  ierr = PetscMalloc4(degree+1,PetscInt,&bdegrees,nsource+1,PetscReal,&sourcey,nsource*(degree+1),PetscReal,&Bsource,worksize,PetscScalar,&work);CHKERRQ(ierr);
  ierr = PetscMalloc4(nsource,PetscScalar,&tau,nsource*(degree+1),PetscReal,&Bsinv,ntarget+1,PetscReal,&targety,ntarget*(degree+1),PetscReal,&Btarget);CHKERRQ(ierr);
  for (i=0; i<=nsource; i++) sourcey[i] = (sourcex[i]-center)/hscale;
  for (i=0; i<=degree; i++) bdegrees[i] = i+1;
  ierr = PetscDTLegendreIntegrate(nsource,sourcey,degree+1,bdegrees,PETSC_TRUE,Bsource);CHKERRQ(ierr);
  ierr = PetscDTPseudoInverseQR(nsource,nsource,degree+1,Bsource,Bsinv,tau,nsource,work);CHKERRQ(ierr);
  for (i=0; i<=ntarget; i++) targety[i] = (targetx[i]-center)/hscale;
  ierr = PetscDTLegendreIntegrate(ntarget,targety,degree+1,bdegrees,PETSC_FALSE,Btarget);CHKERRQ(ierr);
  for (i=0; i<ntarget; i++) {
    PetscReal rowsum = 0;
    for (j=0; j<nsource; j++) {
      PetscReal sum = 0;
      for (k=0; k<degree+1; k++) {
        sum += Btarget[i*(degree+1)+k] * Bsinv[k*nsource+j];
      }
      R[i*nsource+j] = sum;
      rowsum += sum;
    }
    for (j=0; j<nsource; j++) R[i*nsource+j] /= rowsum; /* normalize each row */
  }
  ierr = PetscFree4(bdegrees,sourcey,Bsource,work);CHKERRQ(ierr);
  ierr = PetscFree4(tau,Bsinv,targety,Btarget);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Basis Jet Tabulation

We would like to tabulate the nodal basis functions and derivatives at a set of points, usually quadrature points. We
follow here the derviation in http://www.math.ttu.edu/~kirby/papers/fiat-toms-2004.pdf. The nodal basis $\psi_i$ can
be expressed in terms of a prime basis $\phi_i$ which can be stably evaluated. In PETSc, we will use the Legendre basis
as a prime basis.

  \psi_i = \sum_k \alpha_{ki} \phi_k

Our nodal basis is defined in terms of the dual basis $n_j$

  n_j \cdot \psi_i = \delta_{ji}

and we may act on the first equation to obtain

  n_j \cdot \psi_i = \sum_k \alpha_{ki} n_j \cdot \phi_k
       \delta_{ji} = \sum_k \alpha_{ki} V_{jk}
                 I = V \alpha

so the coefficients of the nodal basis in the prime basis are

   \alpha = V^{-1}

We will define the dual basis vectors $n_j$ using a quadrature rule.

Right now, we will just use the polynomial spaces P^k. I know some elements use the space of symmetric polynomials
(I think Nedelec), but we will neglect this for now. Constraints in the space, e.g. Arnold-Winther elements, can
be implemented exactly as in FIAT using functionals $L_j$.

I will have to count the degrees correctly for the Legendre product when we are on simplices.

We will have three objects:
 - Space, P: this just need point evaluation I think
 - Dual Space, P'+K: This looks like a set of functionals that can act on members of P, each n is defined by a Q
 - FEM: This keeps {P, P', Q}
*/
#include <petsc-private/petscfeimpl.h>

PetscInt PETSCSPACE_CLASSID = 0;

PetscFunctionList PetscSpaceList              = NULL;
PetscBool         PetscSpaceRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceRegister"
/*@C
  PetscSpaceRegister - Adds a new PetscSpace implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscSpaceRegister() may be called multiple times to add several user-defined PetscSpaces

  Sample usage:
.vb
    PetscSpaceRegister("my_space", MyPetscSpaceCreate);
.ve

  Then, your PetscSpace type can be chosen with the procedural interface via
.vb
    PetscSpaceCreate(MPI_Comm, PetscSpace *);
    PetscSpaceSetType(PetscSpace, "my_space");
.ve
   or at runtime via the option
.vb
    -petscspace_type my_space
.ve

  Level: advanced

.keywords: PetscSpace, register
.seealso: PetscSpaceRegisterAll(), PetscSpaceRegisterDestroy()

@*/
PetscErrorCode PetscSpaceRegister(const char sname[], PetscErrorCode (*function)(PetscSpace))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscSpaceList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetType"
/*@C
  PetscSpaceSetType - Builds a particular PetscSpace

  Collective on PetscSpace

  Input Parameters:
+ sp   - The PetscSpace object
- name - The kind of space

  Options Database Key:
. -petscspace_type <type> - Sets the PetscSpace type; use -help for a list of available types

  Level: intermediate

.keywords: PetscSpace, set, type
.seealso: PetscSpaceGetType(), PetscSpaceCreate()
@*/
PetscErrorCode PetscSpaceSetType(PetscSpace sp, PetscSpaceType name)
{
  PetscErrorCode (*r)(PetscSpace);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) sp, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!PetscSpaceRegisterAllCalled) {ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);}
  ierr = PetscFunctionListFind(PetscSpaceList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscSpace type: %s", name);

  if (sp->ops->destroy) {
    ierr             = (*sp->ops->destroy)(sp);CHKERRQ(ierr);
    sp->ops->destroy = NULL;
  }
  ierr = (*r)(sp);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) sp, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceGetType"
/*@C
  PetscSpaceGetType - Gets the PetscSpace type name (as a string) from the object.

  Not Collective

  Input Parameter:
. dm  - The PetscSpace

  Output Parameter:
. name - The PetscSpace type name

  Level: intermediate

.keywords: PetscSpace, get, type, name
.seealso: PetscSpaceSetType(), PetscSpaceCreate()
@*/
PetscErrorCode PetscSpaceGetType(PetscSpace sp, PetscSpaceType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (!PetscSpaceRegisterAllCalled) {
    ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);
  }
  *name = ((PetscObject) sp)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceView"
/*@C
  PetscSpaceView - Views a PetscSpace

  Collective on PetscSpace

  Input Parameter:
+ sp - the PetscSpace object to view
- v  - the viewer

  Level: developer

.seealso PetscSpaceDestroy()
@*/
PetscErrorCode PetscSpaceView(PetscSpace sp, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (!v) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) sp), &v);CHKERRQ(ierr);
  }
  if (sp->ops->view) {
    ierr = (*sp->ops->view)(sp, v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceViewFromOptions"
/*
  PetscSpaceViewFromOptions - Processes command line options to determine if/how a PetscSpace is to be viewed.

  Collective on PetscSpace

  Input Parameters:
+ sp   - the PetscSpace
. prefix - prefix to use for viewing, or NULL to use prefix of 'rnd'
- optionname - option to activate viewing

  Level: intermediate

.keywords: PetscSpace, view, options, database
.seealso: VecViewFromOptions(), MatViewFromOptions()
*/
PetscErrorCode PetscSpaceViewFromOptions(PetscSpace sp, const char prefix[], const char optionname[])
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (prefix) {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) sp), prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) sp), ((PetscObject) sp)->prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  }
  if (flg) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscSpaceView(sp, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetFromOptions"
/*@
  PetscSpaceSetFromOptions - sets parameters in a PetscSpace from the options database

  Collective on PetscSpace

  Input Parameter:
. sp - the PetscSpace object to set options for

  Options Database:
. -petscspace_order the approximation order of the space

  Level: developer

.seealso PetscSpaceView()
@*/
PetscErrorCode PetscSpaceSetFromOptions(PetscSpace sp)
{
  const char    *defaultType;
  char           typename[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (!((PetscObject) sp)->type_name) {
    defaultType = PETSCSPACEPOLYNOMIAL;
  } else {
    defaultType = ((PetscObject) sp)->type_name;
  }
  if (!PetscSpaceRegisterAllCalled) {ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsList("-petscspace_type", "Linear space", "PetscSpaceSetType", PetscSpaceList, defaultType, typename, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscSpaceSetType(sp, typename);CHKERRQ(ierr);
  } else if (!((PetscObject) sp)->type_name) {
    ierr = PetscSpaceSetType(sp, defaultType);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-petscspace_order", "The approximation order", "PetscSpaceSetOrder", sp->order, &sp->order, NULL);CHKERRQ(ierr);
  if (sp->ops->setfromoptions) {
    ierr = (*sp->ops->setfromoptions)(sp);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscSpaceViewFromOptions(sp, NULL, "-petscspace_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetUp"
/*@C
  PetscSpaceSetUp - Construct data structures for the PetscSpace

  Collective on PetscSpace

  Input Parameter:
. sp - the PetscSpace object to setup

  Level: developer

.seealso PetscSpaceView(), PetscSpaceDestroy()
@*/
PetscErrorCode PetscSpaceSetUp(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (sp->ops->setup) {ierr = (*sp->ops->setup)(sp);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceDestroy"
/*@
  PetscSpaceDestroy - Destroys a PetscSpace object

  Collective on PetscSpace

  Input Parameter:
. sp - the PetscSpace object to destroy

  Level: developer

.seealso PetscSpaceView()
@*/
PetscErrorCode PetscSpaceDestroy(PetscSpace *sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sp), PETSCSPACE_CLASSID, 1);

  if (--((PetscObject)(*sp))->refct > 0) {*sp = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*sp))->refct = 0;
  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectAMSViewOff((PetscObject) *sp);CHKERRQ(ierr);

  ierr = DMDestroy(&(*sp)->dm);CHKERRQ(ierr);

  ierr = (*(*sp)->ops->destroy)(*sp);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceCreate"
/*@
  PetscSpaceCreate - Creates an empty PetscSpace object. The type can then be set with PetscSpaceSetType().

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the PetscSpace object

  Output Parameter:
. sp - The PetscSpace object

  Level: beginner

.seealso: PetscSpaceSetType(), PETSCSPACEPOLYNOMIAL
@*/
PetscErrorCode PetscSpaceCreate(MPI_Comm comm, PetscSpace *sp)
{
  PetscSpace     s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(sp, 2);
  *sp = NULL;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(s, _p_PetscSpace, struct _PetscSpaceOps, PETSCSPACE_CLASSID, "PetscSpace", "Linear Space", "PetscSpace", comm, PetscSpaceDestroy, PetscSpaceView);CHKERRQ(ierr);
  ierr = PetscMemzero(s->ops, sizeof(struct _PetscSpaceOps));CHKERRQ(ierr);

  s->order = 0;
  ierr = DMShellCreate(comm, &s->dm);CHKERRQ(ierr);

  *sp = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceGetDimension"
/* Dimension of the space, i.e. number of basis vectors */
PetscErrorCode PetscSpaceGetDimension(PetscSpace sp, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = 0;
  if (sp->ops->getdimension) {ierr = (*sp->ops->getdimension)(sp, dim);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceGetOrder"
PetscErrorCode PetscSpaceGetOrder(PetscSpace sp, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(order, 2);
  *order = sp->order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetOrder"
PetscErrorCode PetscSpaceSetOrder(PetscSpace sp, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  sp->order = order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceEvaluate"
PetscErrorCode PetscSpaceEvaluate(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(points, 3);
  if (B) PetscValidPointer(B, 4);
  if (D) PetscValidPointer(D, 5);
  if (H) PetscValidPointer(H, 6);
  if (sp->ops->evaluate) {ierr = (*sp->ops->evaluate)(sp, npoints, points, B, D, H);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetFromOptions_Polynomial"
PetscErrorCode PetscSpaceSetFromOptions_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscspace_num_variables", "The number of different variables, e.g. x and y", "PetscSpacePolynomialSetNumVariables", poly->numVariables, &poly->numVariables, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_poly_sym", "Use only symmetric polynomials", "PetscSpacePolynomialSetSymmetric", poly->symmetric, &poly->symmetric, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialView_Ascii"
PetscErrorCode PetscSpacePolynomialView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_Poly  *poly = (PetscSpace_Poly *) sp->data;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "Polynomial space in %d variables of order %d", poly->numVariables, sp->order);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "Polynomial space in %d variables of order %d", poly->numVariables, sp->order);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceView_Polynomial"
PetscErrorCode PetscSpaceView_Polynomial(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpacePolynomialView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceSetUp_Polynomial"
PetscErrorCode PetscSpaceSetUp_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  PetscInt         ndegree = sp->order+1;
  PetscInt         dim     = poly->numVariables;
  PetscInt         pdim, deg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(ndegree * sizeof(PetscInt), &poly->degrees);CHKERRQ(ierr);
  for (deg = 0; deg < ndegree; ++deg) poly->degrees[deg] = deg;
  ierr = PetscSpaceGetDimension(sp, &pdim);CHKERRQ(ierr);
  /* Create A (pdim x ndegree * dim) */
  ierr = PetscMalloc(pdim*ndegree*dim * sizeof(PetscReal), &poly->A);CHKERRQ(ierr);
  ierr = PetscMemzero(poly->A, pdim*ndegree*dim * sizeof(PetscReal));CHKERRQ(ierr);
  /* Hardcode P_1: Here we need a way to iterate through the basis */
  poly->A[(0*ndegree + 0)*dim + 0] = 1.0; /* 1 */
  poly->A[(1*ndegree + 1)*dim + 0] = 1.0; /* x */
  poly->A[(2*ndegree + 1)*dim + 1] = 1.0; /* y */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceDestroy_Polynomial"
PetscErrorCode PetscSpaceDestroy_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFree(poly->A);CHKERRQ(ierr);
  ierr = PetscFree(poly->degrees);CHKERRQ(ierr);
  ierr = PetscFree(poly);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceGetDimension_Polynomial"
PetscErrorCode PetscSpaceGetDimension_Polynomial(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscInt         deg  = sp->order;
  PetscInt         n    = poly->numVariables, i;
  PetscReal        D    = 1.0;

  PetscFunctionBegin;
  for (i = 1; i <= n; ++i) {
    D *= ((PetscReal) (deg+i))/i;
  }
  *dim = (PetscInt) (D + 0.5);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LatticePoint_Internal"
/*
  LatticePoint_Internal - Returns all tuples of size 'len' with nonnegative integers that sum up to 'sum'.

  Input Parameters:
+ len - The length of the tuple
. sum - The sum of all entries in the tuple
- ind - The current multi-index of the tuple, initialized to the 0 tuple

  Output Parameter:
+ ind - The multi-index of the tuple, -1 indicates the iteration has terminated
. tup - A tuple of len integers addig to sum

  Level: developer

.seealso: 
*/
static PetscErrorCode LatticePoint_Internal(PetscInt len, PetscInt sum, PetscInt ind[], PetscInt tup[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (len == 1) {
    ind[0] = -1;
    tup[0] = sum;
  } else if (sum == 0) {
    for (i = 0; i < len; ++i) {ind[0] = -1; tup[i] = 0;}
  } else {
    tup[0] = sum - ind[0];
    ierr = LatticePoint_Internal(len-1, ind[0], &ind[1], &tup[1]);CHKERRQ(ierr);
    if (ind[1] < 0) {
      if (ind[0] == sum) {ind[0] = -1;}
      else               {ind[1] = 0; ++ind[0];}
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceEvaluate_Polynomial"
PetscErrorCode PetscSpaceEvaluate_Polynomial(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Poly *poly    = (PetscSpace_Poly *) sp->data;
  DM               dm      = sp->dm;
  PetscInt         ndegree = sp->order+1;
  PetscInt        *degrees = poly->degrees;
  PetscInt         dim     = poly->numVariables;
  PetscReal       *A       = poly->A;
  PetscReal       *lpoints, *tmp, *LB, *LD, *LH;
  PetscInt        *ind, *tup;
  PetscInt         pdim, d, i, p, deg, o;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetDimension(sp, &pdim);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, npoints, PETSC_REAL, &lpoints);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, npoints*ndegree*3, PETSC_REAL, &tmp);CHKERRQ(ierr);
  if (B) {ierr = DMGetWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LB);CHKERRQ(ierr);}
  if (D) {ierr = DMGetWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LD);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LH);CHKERRQ(ierr);}
  for (d = 0; d < dim; ++d) {
    for (p = 0; p < npoints; ++p) {
      lpoints[p] = points[p*dim+d];
    }
    ierr = PetscDTLegendreEval(npoints, lpoints, ndegree, degrees, tmp, &tmp[1*npoints*ndegree], &tmp[2*npoints*ndegree]);CHKERRQ(ierr);
    /* LB, LD, LH (ndegree * dim x npoints) */
    for (deg = 0; deg < ndegree; ++deg) {
      for (p = 0; p < npoints; ++p) {
        if (B) LB[(deg*dim + d)*npoints + p] = tmp[(0*npoints + p)*ndegree+deg];
        if (D) LD[(deg*dim + d)*npoints + p] = tmp[(1*npoints + p)*ndegree+deg];
        if (H) LH[(deg*dim + d)*npoints + p] = tmp[(2*npoints + p)*ndegree+deg];
      }
    }
  }
  /* Multiply by A (pdim x ndegree * dim) */
  ierr = PetscMalloc2(dim,PetscInt,&ind,dim,PetscInt,&tup);CHKERRQ(ierr);
  if (B) {
    /* B (npoints x pdim) */
    i = 0;
    for (o = 0; o <= sp->order; ++o) {
      ierr = PetscMemzero(ind, dim * sizeof(PetscInt));CHKERRQ(ierr);
      while (ind[0] >= 0) {
        ierr = LatticePoint_Internal(dim, o, ind, tup);CHKERRQ(ierr);
        for (p = 0; p < npoints; ++p) {
          B[p*pdim + i] = 1.0;
          for (d = 0; d < dim; ++d) {
            B[p*pdim + i] *= LB[(tup[d]*dim + d)*npoints + p];
          }
        }
        ++i;
      }
    }
  }
  ierr = PetscFree2(ind,tup);CHKERRQ(ierr);
  if (D) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to code first derivatives");
  if (H) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to code second derivatives");
  if (B) {ierr = DMRestoreWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LB);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LD);CHKERRQ(ierr);}
  if (H) {ierr = DMRestoreWorkArray(dm, npoints*dim*ndegree, PETSC_REAL, &LH);CHKERRQ(ierr);}
  ierr = DMRestoreWorkArray(dm, npoints*ndegree*3, PETSC_REAL, &tmp);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, npoints, PETSC_REAL, &lpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceInitialize_Polynomial"
PetscErrorCode PetscSpaceInitialize_Polynomial(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions = PetscSpaceSetFromOptions_Polynomial;
  sp->ops->setup          = PetscSpaceSetUp_Polynomial;
  sp->ops->view           = PetscSpaceView_Polynomial;
  sp->ops->destroy        = PetscSpaceDestroy_Polynomial;
  sp->ops->getdimension   = PetscSpaceGetDimension_Polynomial;
  sp->ops->evaluate       = PetscSpaceEvaluate_Polynomial;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPOLYNOMIAL = "poly" - A PetscSpace object that encapsulates a polynomial space, e.g. P1 is the space of linear polynomials.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceCreate_Polynomial"
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp, PetscSpace_Poly, &poly);CHKERRQ(ierr);
  sp->data = poly;

  poly->numVariables = 0;
  poly->symmetric    = PETSC_FALSE;
  poly->degrees      = NULL;
  poly->A            = NULL;

  ierr = PetscSpaceInitialize_Polynomial(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialSetSymmetric"
PetscErrorCode PetscSpacePolynomialSetSymmetric(PetscSpace sp, PetscBool sym)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  poly->symmetric = sym;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialGetSymmetric"
PetscErrorCode PetscSpacePolynomialGetSymmetric(PetscSpace sp, PetscBool *sym)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(sym, 2);
  *sym = poly->symmetric;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialSetNumVariables"
PetscErrorCode PetscSpacePolynomialSetNumVariables(PetscSpace sp, PetscInt n)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  poly->numVariables = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpacePolynomialGetNumVariables"
PetscErrorCode PetscSpacePolynomialGetNumVariables(PetscSpace sp, PetscInt *n)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(n, 2);
  *n = poly->numVariables;
  PetscFunctionReturn(0);
}


PetscInt PETSCDUALSPACE_CLASSID = 0;

PetscFunctionList PetscDualSpaceList              = NULL;
PetscBool         PetscDualSpaceRegisterAllCalled = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceRegister"
/*@C
  PetscDualSpaceRegister - Adds a new PetscDualSpace implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscDualSpaceRegister() may be called multiple times to add several user-defined PetscDualSpaces

  Sample usage:
.vb
    PetscDualSpaceRegister("my_space", MyPetscDualSpaceCreate);
.ve

  Then, your PetscDualSpace type can be chosen with the procedural interface via
.vb
    PetscDualSpaceCreate(MPI_Comm, PetscDualSpace *);
    PetscDualSpaceSetType(PetscDualSpace, "my_dual_space");
.ve
   or at runtime via the option
.vb
    -petscdualspace_type my_dual_space
.ve

  Level: advanced

.keywords: PetscDualSpace, register
.seealso: PetscDualSpaceRegisterAll(), PetscDualSpaceRegisterDestroy()

@*/
PetscErrorCode PetscDualSpaceRegister(const char sname[], PetscErrorCode (*function)(PetscDualSpace))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscDualSpaceList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetType"
/*@C
  PetscDualSpaceSetType - Builds a particular PetscDualSpace

  Collective on PetscDualSpace

  Input Parameters:
+ sp   - The PetscDualSpace object
- name - The kind of space

  Options Database Key:
. -petscdualspace_type <type> - Sets the PetscDualSpace type; use -help for a list of available types

  Level: intermediate

.keywords: PetscDualSpace, set, type
.seealso: PetscDualSpaceGetType(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceSetType(PetscDualSpace sp, PetscDualSpaceType name)
{
  PetscErrorCode (*r)(PetscDualSpace);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) sp, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  if (!PetscDualSpaceRegisterAllCalled) {ierr = PetscDualSpaceRegisterAll();CHKERRQ(ierr);}
  ierr = PetscFunctionListFind(PetscDualSpaceList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscDualSpace type: %s", name);

  if (sp->ops->destroy) {
    ierr             = (*sp->ops->destroy)(sp);CHKERRQ(ierr);
    sp->ops->destroy = NULL;
  }
  ierr = (*r)(sp);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) sp, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetType"
/*@C
  PetscDualSpaceGetType - Gets the PetscDualSpace type name (as a string) from the object.

  Not Collective

  Input Parameter:
. dm  - The PetscDualSpace

  Output Parameter:
. name - The PetscDualSpace type name

  Level: intermediate

.keywords: PetscDualSpace, get, type, name
.seealso: PetscDualSpaceSetType(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceGetType(PetscDualSpace sp, PetscDualSpaceType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidCharPointer(name, 2);
  if (!PetscDualSpaceRegisterAllCalled) {
    ierr = PetscDualSpaceRegisterAll();CHKERRQ(ierr);
  }
  *name = ((PetscObject) sp)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceView"
/*@C
  PetscDualSpaceView - Views a PetscDualSpace

  Collective on PetscDualSpace

  Input Parameter:
+ sp - the PetscDualSpace object to view
- v  - the viewer

  Level: developer

.seealso PetscDualSpaceDestroy()
@*/
PetscErrorCode PetscDualSpaceView(PetscDualSpace sp, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (!v) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) sp), &v);CHKERRQ(ierr);
  }
  if (sp->ops->view) {
    ierr = (*sp->ops->view)(sp, v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceViewFromOptions"
/*
  PetscDualSpaceViewFromOptions - Processes command line options to determine if/how a PetscDualSpace is to be viewed.

  Collective on PetscDualSpace

  Input Parameters:
+ sp   - the PetscDualSpace
. prefix - prefix to use for viewing, or NULL to use prefix of 'rnd'
- optionname - option to activate viewing

  Level: intermediate

.keywords: PetscDualSpace, view, options, database
.seealso: VecViewFromOptions(), MatViewFromOptions()
*/
PetscErrorCode PetscDualSpaceViewFromOptions(PetscDualSpace sp, const char prefix[], const char optionname[])
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (prefix) {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) sp), prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject) sp), ((PetscObject) sp)->prefix, optionname, &viewer, &format, &flg);CHKERRQ(ierr);
  }
  if (flg) {
    ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
    ierr = PetscDualSpaceView(sp, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetFromOptions"
/*@
  PetscDualSpaceSetFromOptions - sets parameters in a PetscDualSpace from the options database

  Collective on PetscDualSpace

  Input Parameter:
. sp - the PetscDualSpace object to set options for

  Options Database:
. -petscspace_order the approximation order of the space

  Level: developer

.seealso PetscDualSpaceView()
@*/
PetscErrorCode PetscDualSpaceSetFromOptions(PetscDualSpace sp)
{
  const char    *defaultType;
  char           typename[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (!((PetscObject) sp)->type_name) {
    defaultType = PETSCDUALSPACELAGRANGE;
  } else {
    defaultType = ((PetscObject) sp)->type_name;
  }
  if (!PetscSpaceRegisterAllCalled) {ierr = PetscSpaceRegisterAll();CHKERRQ(ierr);}

  ierr = PetscObjectOptionsBegin((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsList("-petscdualspace_type", "Dual space", "PetscDualSpaceSetType", PetscDualSpaceList, defaultType, typename, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscDualSpaceSetType(sp, typename);CHKERRQ(ierr);
  } else if (!((PetscObject) sp)->type_name) {
    ierr = PetscDualSpaceSetType(sp, defaultType);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-petscdualspace_order", "The approximation order", "PetscDualSpaceSetOrder", sp->order, &sp->order, NULL);CHKERRQ(ierr);
  if (sp->ops->setfromoptions) {
    ierr = (*sp->ops->setfromoptions)(sp);CHKERRQ(ierr);
  }
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers((PetscObject) sp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscDualSpaceViewFromOptions(sp, NULL, "-petscdualspace_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetUp"
/*@C
  PetscDualSpaceSetUp - Construct a basis for the PetscDualSpace

  Collective on PetscDualSpace

  Input Parameter:
. sp - the PetscDualSpace object to setup

  Level: developer

.seealso PetscDualSpaceView(), PetscDualSpaceDestroy()
@*/
PetscErrorCode PetscDualSpaceSetUp(PetscDualSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (sp->ops->setup) {ierr = (*sp->ops->setup)(sp);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceDestroy"
/*@
  PetscDualSpaceDestroy - Destroys a PetscDualSpace object

  Collective on PetscDualSpace

  Input Parameter:
. sp - the PetscDualSpace object to destroy

  Level: developer

.seealso PetscDualSpaceView()
@*/
PetscErrorCode PetscDualSpaceDestroy(PetscDualSpace *sp)
{
  PetscInt       dim, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sp) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sp), PETSCDUALSPACE_CLASSID, 1);

  if (--((PetscObject)(*sp))->refct > 0) {*sp = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*sp))->refct = 0;
  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectAMSViewOff((PetscObject) *sp);CHKERRQ(ierr);

  ierr = DMDestroy(&(*sp)->dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(*sp, &dim);CHKERRQ(ierr);
  for (f = 0; f < dim; ++f) {
    /* ierr = PetscQuadratureDestroy((*sp)->functional[f]);CHKERRQ(ierr); */
  }
  ierr = PetscFree((*sp)->functional);CHKERRQ(ierr);

  ierr = (*(*sp)->ops->destroy)(*sp);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceCreate"
/*@
  PetscDualSpaceCreate - Creates an empty PetscDualSpace object. The type can then be set with PetscDualSpaceSetType().

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the PetscDualSpace object

  Output Parameter:
. sp - The PetscDualSpace object

  Level: beginner

.seealso: PetscDualSpaceSetType(), PETSCDUALSPACELAGRANGE
@*/
PetscErrorCode PetscDualSpaceCreate(MPI_Comm comm, PetscDualSpace *sp)
{
  PetscDualSpace s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(sp, 2);
  *sp = NULL;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(s, _p_PetscDualSpace, struct _PetscDualSpaceOps, PETSCDUALSPACE_CLASSID, "PetscDualSpace", "Dual Space", "PetscDualSpace", comm, PetscDualSpaceDestroy, PetscDualSpaceView);CHKERRQ(ierr);
  ierr = PetscMemzero(s->ops, sizeof(struct _PetscDualSpaceOps));CHKERRQ(ierr);

  s->order = 0;

  *sp = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDM"
PetscErrorCode PetscDualSpaceGetDM(PetscDualSpace sp, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(dm, 2);
  *dm = sp->dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetDM"
PetscErrorCode PetscDualSpaceSetDM(PetscDualSpace sp, DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  ierr = DMDestroy(&sp->dm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  sp->dm = dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetOrder"
PetscErrorCode PetscDualSpaceGetOrder(PetscDualSpace sp, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(order, 2);
  *order = sp->order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetOrder"
PetscErrorCode PetscDualSpaceSetOrder(PetscDualSpace sp, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  sp->order = order;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetFunctional"
PetscErrorCode PetscDualSpaceGetFunctional(PetscDualSpace sp, PetscInt i, PetscQuadrature *functional)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(functional, 3);
  ierr = PetscDualSpaceGetDimension(sp, &dim);CHKERRQ(ierr);
  if ((i < 0) || (i >= dim)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Functional index %d must be in [0, %d)", i, dim);
  *functional = sp->functional[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDimension"
/* Dimension of the space, i.e. number of basis vectors */
PetscErrorCode PetscDualSpaceGetDimension(PetscDualSpace sp, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  *dim = 0;
  if (sp->ops->getdimension) {ierr = (*sp->ops->getdimension)(sp, dim);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetUp_Lagrange"
PetscErrorCode PetscDualSpaceSetUp_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag   = (PetscDualSpace_Lag *) sp->data;
  DM                  dm    = sp->dm;
  PetscInt            order = sp->order;
  PetscSection        csection;
  Vec                 coordinates;
  PetscScalar        *coords;
  PetscInt            pdim, dim, vStart, vEnd, cStart, coneSize, f = 0;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(sp, &pdim);CHKERRQ(ierr);
  ierr = PetscMalloc(pdim * sizeof(PetscQuadrature), &sp->functional);CHKERRQ(ierr);
  /* Classify element type */
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &csection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (coneSize == dim+1) {
    PetscInt *closure = NULL, closureSize, c;

    /* Simplex */
    if (order > 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot handle order %d basis", order);
    ierr = DMPlexGetTransitiveClosure(dm, cStart, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (c = 0; c < closureSize*2; c += 2) {
      const PetscInt p = closure[c];
      /* Vertices */
      if ((p >= vStart) && (p < vEnd)) {
        PetscReal *qpoints, *qweights;
        PetscInt   dof, off, d;

        sp->functional[f].numQuadPoints = 1;
        ierr = PetscMalloc(sp->functional[f].numQuadPoints*dim * sizeof(PetscReal), &qpoints);CHKERRQ(ierr);
        ierr = PetscMalloc(sp->functional[f].numQuadPoints     * sizeof(PetscReal), &qweights);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(csection, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(csection, p, &off);CHKERRQ(ierr);
        if (dof != dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of coordinates %d does not match spatial dimension %d", dof, dim);
        for (d = 0; d < dof; ++d) {qpoints[d] = coords[off+d];}
        qweights[0] = 1.0;
        sp->functional[f].quadPoints  = qpoints;
        sp->functional[f].quadWeights = qweights;
        ++f;
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cStart, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle cells with cone size %d", coneSize);
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceGetDimension_Lagrange"
PetscErrorCode PetscDualSpaceGetDimension_Lagrange(PetscDualSpace sp, PetscInt *dim)
{
  PetscInt            deg = sp->order;
  PetscReal           D   = 1.0;
  PetscInt            n, i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* TODO: Assumes simplices */
  ierr = DMPlexGetDimension(sp->dm, &n);CHKERRQ(ierr);
  for (i = 1; i <= n; ++i) {
    D *= ((PetscReal) (deg+i))/i;
  }
  *dim = (PetscInt) (D + 0.5);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceInitialize_Lagrange"
PetscErrorCode PetscDualSpaceInitialize_Lagrange(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions = NULL;
  sp->ops->setup          = PetscDualSpaceSetUp_Lagrange;
  sp->ops->view           = NULL;
  sp->ops->destroy        = NULL;
  sp->ops->getdimension   = PetscDualSpaceGetDimension_Lagrange;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDUALSPACELAGRANGE = "lagrange" - A PetscDualSpace object that encapsulates a dual space of pointwise evaluation functionals

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceCreate_Lagrange"
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp, PetscDualSpace_Lag, &lag);CHKERRQ(ierr);
  sp->data = lag;

  /* lag->n = 0; */

  ierr = PetscDualSpaceInitialize_Lagrange(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscInt PETSCFE_CLASSID = 0;

#undef __FUNCT__
#define __FUNCT__ "PetscFEView"
/*@C
  PetscFEView - Views a PetscFE

  Collective on PetscFE

  Input Parameter:
+ sp - the PetscFE object to view
- v  - the viewer

  Level: developer

.seealso PetscFEDestroy()
@*/
PetscErrorCode PetscFEView(PetscFE fem, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  if (!v) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) fem), &v);CHKERRQ(ierr);
  }
  if (fem->ops->view) {
    ierr = (*fem->ops->view)(fem, v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEDestroy"
/*@
  PetscFEDestroy - Destroys a PetscFE object

  Collective on PetscFE

  Input Parameter:
. fem - the PetscFE object to destroy

  Level: developer

.seealso PetscFEView()
@*/
PetscErrorCode PetscFEDestroy(PetscFE *fem)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*fem) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*fem), PETSCFE_CLASSID, 1);

  if (--((PetscObject)(*fem))->refct > 0) {*fem = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*fem))->refct = 0;
  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectAMSViewOff((PetscObject) *fem);CHKERRQ(ierr);

  ierr = (*(*fem)->ops->destroy)(*fem);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFECreate"
/*@
  PetscFECreate - Creates an empty PetscFE object. The type can then be set with PetscFESetType().

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the PetscFE object

  Output Parameter:
. fem - The PetscFE object

  Level: beginner

.seealso: PetscFESetType(), PETSCFEGALERKIN
@*/
PetscErrorCode PetscFECreate(MPI_Comm comm, PetscFE *fem)
{
  PetscFE        f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(fem, 2);
  *fem = NULL;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = PetscFEInitializePackage();CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(f, _p_PetscFE, struct _PetscFEOps, PETSCFE_CLASSID, "PetscFE", "Finite Element", "PetscFE", comm, PetscFEDestroy, PetscFEView);CHKERRQ(ierr);
  ierr = PetscMemzero(f->ops, sizeof(struct _PetscFEOps));CHKERRQ(ierr);

  f->basisSpace = NULL;
  f->dualSpace  = NULL;

  *fem = f;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetDimension"
PetscErrorCode PetscFEGetDimension(PetscFE fem, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  ierr = PetscSpaceGetDimension(fem->basisSpace, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetNumComponents"
PetscErrorCode PetscFEGetNumComponents(PetscFE fem, PetscInt *comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(comp, 2);
  *comp = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetBasisSpace"
PetscErrorCode PetscFEGetBasisSpace(PetscFE fem, PetscSpace *sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(sp, 2);
  *sp = fem->basisSpace;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFESetBasisSpace"
PetscErrorCode PetscFESetBasisSpace(PetscFE fem, PetscSpace sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 2);
  fem->basisSpace = sp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetDualSpace"
PetscErrorCode PetscFEGetDualSpace(PetscFE fem, PetscDualSpace *sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(sp, 2);
  *sp = fem->dualSpace;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFESetDualSpace"
PetscErrorCode PetscFESetDualSpace(PetscFE fem, PetscDualSpace sp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 2);
  fem->dualSpace = sp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetTabulation"
PetscErrorCode PetscFEGetTabulation(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscReal **B, PetscReal **D, PetscReal **H)
{
  DM             dm;
  PetscReal     *tmpB, *invV;
  PetscInt       pdim; /* Dimension of FE space P */
  PetscInt       dim;  /* Spatial dimension */
  PetscInt       p, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(points, 3);
  if (B) PetscValidPointer(B, 4);
  if (D) PetscValidPointer(D, 5);
  if (H) PetscValidPointer(H, 6);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);

  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(fem->basisSpace, &pdim);CHKERRQ(ierr);
  /* if (nvalues%dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of coordinate values %d must be divisible by the spatial dimension %d", nvalues, dim); */

  if (pdim != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Code only works for linear elements right now");

  if (B) {
    ierr = DMGetWorkArray(dm, npoints*pdim, PETSC_REAL, B);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, npoints*pdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);
  }
  if (D) {ierr = DMGetWorkArray(dm, npoints*pdim*dim, PETSC_REAL, D);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm, npoints*pdim*dim*dim, PETSC_REAL, H);CHKERRQ(ierr);}
  ierr = PetscSpaceEvaluate(fem->basisSpace, npoints, points, B ? tmpB : NULL, D ? *D : NULL, H ? *H : NULL);CHKERRQ(ierr);

  ierr = DMGetWorkArray(dm, pdim*pdim, PETSC_REAL, &invV);CHKERRQ(ierr);
  for (j = 0; j < pdim; ++j) {
    PetscReal      *Bf;
    PetscQuadrature f;
    PetscInt        q;

    ierr = PetscDualSpaceGetFunctional(fem->dualSpace, j, &f);
    ierr = DMGetWorkArray(dm, f.numQuadPoints*pdim, PETSC_REAL, &Bf);CHKERRQ(ierr);
    ierr = PetscSpaceEvaluate(fem->basisSpace, f.numQuadPoints, f.quadPoints, Bf, NULL, NULL);CHKERRQ(ierr);
    for (k = 0; k < pdim; ++k) {
      /* n_j \cdot \phi_k */
      invV[j*pdim+k] = 0.0;
      for (q = 0; q < f.numQuadPoints; ++q) {
        invV[j*pdim+k] += Bf[q*pdim+k]*f.quadWeights[q];
      }
    }
    ierr = DMRestoreWorkArray(dm, f.numQuadPoints*pdim, PETSC_REAL, &Bf);CHKERRQ(ierr);
  }
  {
    PetscReal    *work;
    PetscBLASInt *pivots;
    PetscBLASInt  n = pdim, info;

    ierr = DMGetWorkArray(dm, pdim, PETSC_INT, &pivots);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, pdim, PETSC_REAL, &work);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&n, &n, invV, &n, pivots, &info));
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&n, invV, &n, pivots, work, &n, &info));
    ierr = DMRestoreWorkArray(dm, pdim, PETSC_INT, &pivots);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(dm, pdim, PETSC_REAL, &work);CHKERRQ(ierr);
  }
  for (p = 0; p < npoints; ++p) {
    if (B) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        const PetscInt i = p*pdim + j;

        (*B)[i] = 0.0;
        for (k = 0; k < pdim; ++k) {
          (*B)[i] += invV[k*pdim+j] * tmpB[p*pdim + k];
        }
      }
    }
  }
  ierr = DMRestoreWorkArray(dm, pdim*pdim, PETSC_REAL, &invV);CHKERRQ(ierr);
  if (B) {
    ierr = DMRestoreWorkArray(dm, npoints*pdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFERestoreTabulation"
PetscErrorCode PetscFERestoreTabulation(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscReal **B, PetscReal **D, PetscReal **D2)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  if (B)  {ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, B);CHKERRQ(ierr);}
  if (D)  {ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, D);CHKERRQ(ierr);}
  if (D2) {ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, D2);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}
