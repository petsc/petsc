/* Discretization tools */

#include <petscconf.h>
#if defined(PETSC_HAVE_MATHIMF_H)
#include <mathimf.h>           /* this needs to be included before math.h */
#endif

#include <petscdt.h>            /*I "petscdt.h" I*/ /*I "petscfe.h" I*/
#include <petscblaslapack.h>
#include <petsc-private/petscimpl.h>
#include <petscviewer.h>
#include <petscdmplex.h>
#include <petscdmshell.h>

#undef __FUNCT__
#define __FUNCT__ "PetscQuadratureDestroy"
PetscErrorCode PetscQuadratureDestroy(PetscQuadrature *q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(q->quadPoints);CHKERRQ(ierr);
  ierr = PetscFree(q->quadWeights);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
. order - The quadrature order
. a - left end of interval (often-1)
- b - right end of interval (often +1)

  Output Arguments:
. q - A PetscQuadrature object

  Level: intermediate

  References:
  Karniadakis and Sherwin.
  FIAT

.seealso: PetscDTGaussQuadrature()
@*/
PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt dim, PetscInt order, PetscReal a, PetscReal b, PetscQuadrature *q)
{
  PetscInt       npoints = dim > 1 ? dim > 2 ? order*PetscSqr(order) : PetscSqr(order) : order;
  PetscReal     *px, *wx, *py, *wy, *pz, *wz, *x, *w;
  PetscInt       i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((a != -1.0) || (b != 1.0)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must use default internal right now");
  ierr = PetscMalloc(npoints*dim * sizeof(PetscReal), &x);CHKERRQ(ierr);
  ierr = PetscMalloc(npoints     * sizeof(PetscReal), &w);CHKERRQ(ierr);
  switch (dim) {
  case 1:
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(order, 0.0, 0.0, x, w);CHKERRQ(ierr);
    break;
  case 2:
    ierr = PetscMalloc4(order,PetscReal,&px,order,PetscReal,&wx,order,PetscReal,&py,order,PetscReal,&wy);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(order, 0.0, 0.0, px, wx);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(order, 1.0, 0.0, py, wy);CHKERRQ(ierr);
    for (i = 0; i < order; ++i) {
      for (j = 0; j < order; ++j) {
        ierr = PetscDTMapSquareToTriangle_Internal(px[i], py[j], &x[(i*order+j)*2+0], &x[(i*order+j)*2+1]);CHKERRQ(ierr);
        w[i*order+j] = 0.5 * wx[i] * wy[j];
      }
    }
    ierr = PetscFree4(px,wx,py,wy);CHKERRQ(ierr);
    break;
  case 3:
    ierr = PetscMalloc6(order,PetscReal,&px,order,PetscReal,&wx,order,PetscReal,&py,order,PetscReal,&wy,order,PetscReal,&pz,order,PetscReal,&wz);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(order, 0.0, 0.0, px, wx);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(order, 1.0, 0.0, py, wy);CHKERRQ(ierr);
    ierr = PetscDTGaussJacobiQuadrature1D_Internal(order, 2.0, 0.0, pz, wz);CHKERRQ(ierr);
    for (i = 0; i < order; ++i) {
      for (j = 0; j < order; ++j) {
        for (k = 0; k < order; ++k) {
          ierr = PetscDTMapCubeToTetrahedron_Internal(px[i], py[j], pz[k], &x[((i*order+j)*order+k)*3+0], &x[((i*order+j)*order+k)*3+1], &x[((i*order+j)*order+k)*3+2]);CHKERRQ(ierr);
          w[(i*order+j)*order+k] = 0.125 * wx[i] * wy[j] * wz[k];
        }
      }
    }
    ierr = PetscFree6(px,wx,py,wy,pz,wz);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot construct quadrature rule for dimension %d", dim);
  }
  q->numQuadPoints = npoints;
  q->quadPoints    = x;
  q->quadWeights   = w;
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
  char           name[256];
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
  ierr = PetscOptionsList("-petscspace_type", "Linear space", "PetscSpaceSetType", PetscSpaceList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscSpaceSetType(sp, name);CHKERRQ(ierr);
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
  PetscInt         deg;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(ndegree * sizeof(PetscInt), &poly->degrees);CHKERRQ(ierr);
  for (deg = 0; deg < ndegree; ++deg) poly->degrees[deg] = deg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceDestroy_Polynomial"
PetscErrorCode PetscSpaceDestroy_Polynomial(PetscSpace sp)
{
  PetscSpace_Poly *poly = (PetscSpace_Poly *) sp->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
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
  PetscReal       *lpoints, *tmp, *LB, *LD, *LH;
  PetscInt        *ind, *tup;
  PetscInt         pdim, d, der, i, p, deg, o;
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
  if (D) {
    /* D (npoints x pdim x dim) */
    i = 0;
    for (o = 0; o <= sp->order; ++o) {
      ierr = PetscMemzero(ind, dim * sizeof(PetscInt));CHKERRQ(ierr);
      while (ind[0] >= 0) {
        ierr = LatticePoint_Internal(dim, o, ind, tup);CHKERRQ(ierr);
        for (p = 0; p < npoints; ++p) {
          for (der = 0; der < dim; ++der) {
            D[(p*pdim + i)*dim + der] = 1.0;
            for (d = 0; d < dim; ++d) {
              if (d == der) {
                D[(p*pdim + i)*dim + der] *= LD[(tup[d]*dim + d)*npoints + p];
              } else {
                D[(p*pdim + i)*dim + der] *= LB[(tup[d]*dim + d)*npoints + p];
              }
            }
          }
        }
        ++i;
      }
    }
  }
  if (H) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to code second derivatives");
  ierr = PetscFree2(ind,tup);CHKERRQ(ierr);
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
  char           name[256];
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
  ierr = PetscOptionsList("-petscdualspace_type", "Dual space", "PetscDualSpaceSetType", PetscDualSpaceList, defaultType, name, 256, &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscDualSpaceSetType(sp, name);CHKERRQ(ierr);
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

  ierr = PetscDualSpaceGetDimension(*sp, &dim);CHKERRQ(ierr);
  for (f = 0; f < dim; ++f) {
    ierr = PetscQuadratureDestroy(&(*sp)->functional[f]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*sp)->functional);CHKERRQ(ierr);
  ierr = DMDestroy(&(*sp)->dm);CHKERRQ(ierr);

  if ((*sp)->ops->destroy) {ierr = (*(*sp)->ops->destroy)(*sp);CHKERRQ(ierr);}
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
#define __FUNCT__ "PetscDualSpaceGetNumDof"
PetscErrorCode PetscDualSpaceGetNumDof(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(numDof, 2);
  *numDof = NULL;
  if (sp->ops->getnumdof) {ierr = (*sp->ops->getnumdof)(sp, numDof);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceCreateReferenceCell"
PetscErrorCode PetscDualSpaceCreateReferenceCell(PetscDualSpace sp, PetscInt dim, PetscBool simplex, DM *refdm)
{
  DM             rdm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(PetscObjectComm((PetscObject) sp), &rdm);CHKERRQ(ierr);
  ierr = DMSetType(rdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(rdm, dim);CHKERRQ(ierr);
  switch (dim) {
  case 2:
  {
    PetscInt    numPoints[2]        = {3, 1};
    PetscInt    coneSize[4]         = {3, 0, 0, 0};
    PetscInt    cones[3]            = {1, 2, 3};
    PetscInt    coneOrientations[3] = {0, 0, 0};
    PetscScalar vertexCoords[6]     = {-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0};

    ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  }
  break;
  case 3:
  {
    PetscInt    numPoints[2]        = {4, 1};
    PetscInt    coneSize[5]         = {4, 0, 0, 0, 0};
    PetscInt    cones[4]            = {1, 2, 3, 4};
    PetscInt    coneOrientations[4] = {0, 0, 0, 0};
    PetscScalar vertexCoords[12]    = {-1.0, -1.0, -1.0,  1.0, -1.0, -1.0,  -1.0, 1.0, -1.0,  -1.0, -1.0, 1.0};

    ierr = DMPlexCreateFromDAG(rdm, 1, numPoints, coneSize, cones, coneOrientations, vertexCoords);CHKERRQ(ierr);
  }
  break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_WRONG, "Cannot create reference cell for dimension %d", dim);
  }
  ierr = DMPlexInterpolate(rdm, refdm);CHKERRQ(ierr);
  ierr = DMPlexCopyCoordinates(rdm, *refdm);CHKERRQ(ierr);
  ierr = DMDestroy(&rdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceSetUp_Lagrange"
PetscErrorCode PetscDualSpaceSetUp_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  DM                  dm    = sp->dm;
  PetscInt            order = sp->order;
  PetscSection        csection;
  Vec                 coordinates;
  PetscReal          *qpoints, *qweights;
  PetscInt            depth, dim, pdim, *pStart, *pEnd, coneSize, d, n, f = 0;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(sp, &pdim);CHKERRQ(ierr);
  ierr = PetscMalloc(pdim * sizeof(PetscQuadrature), &sp->functional);CHKERRQ(ierr);
  /* Classify element type */
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc((dim+1) * sizeof(PetscInt), &lag->numDof);CHKERRQ(ierr);
  ierr = PetscMemzero(lag->numDof, (dim+1) * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc2(depth+1,PetscInt,&pStart,depth+1,PetscInt,&pEnd);CHKERRQ(ierr);
  for (d = 0; d < depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
  }
  ierr = DMPlexGetConeSize(dm, pStart[depth], &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &csection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  if (coneSize == dim+1) {
    PetscInt *closure = NULL, closureSize, c;

    /* Simplex */
    ierr = DMPlexGetTransitiveClosure(dm, pStart[depth], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (c = 0; c < closureSize*2; c += 2) {
      const PetscInt p = closure[c];

      if ((p >= pStart[0]) && (p < pEnd[0])) {
        /* Vertices */
        const PetscScalar *coords;
        PetscInt           dof, off, d;

        if (order < 1) continue;
        sp->functional[f].numQuadPoints = 1;
        ierr = PetscMalloc(sp->functional[f].numQuadPoints*dim * sizeof(PetscReal), &qpoints);CHKERRQ(ierr);
        ierr = PetscMalloc(sp->functional[f].numQuadPoints     * sizeof(PetscReal), &qweights);CHKERRQ(ierr);
        ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(csection, p, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(csection, p, &off);CHKERRQ(ierr);
        if (dof != dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of coordinates %d does not match spatial dimension %d", dof, dim);
        for (d = 0; d < dof; ++d) {qpoints[d] = coords[off+d];}
        qweights[0] = 1.0;
        sp->functional[f].quadPoints  = qpoints;
        sp->functional[f].quadWeights = qweights;
        ++f;
        ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
        lag->numDof[0] = 1;
      } else if ((p >= pStart[1]) && (p < pEnd[1])) {
        /* Edges */
        PetscScalar *coords;
        PetscInt     k;

        if (order < 2) continue;
        coords = NULL;
        ierr = DMPlexVecGetClosure(dm, csection, coordinates, p, &n, &coords);CHKERRQ(ierr);
        if (n != dim*2) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %d has %d coordinate values instead of %d", p, n, dim*2);
        for (k = 1; k < order; ++k) {
          sp->functional[f].numQuadPoints = 1;
          ierr = PetscMalloc(sp->functional[f].numQuadPoints*dim * sizeof(PetscReal), &qpoints);CHKERRQ(ierr);
          ierr = PetscMalloc(sp->functional[f].numQuadPoints     * sizeof(PetscReal), &qweights);CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) {qpoints[d] = k*(coords[1*dim+d] - coords[0*dim+d])/order + coords[0*dim+d];}
          qweights[0] = 1.0;
          sp->functional[f].quadPoints  = qpoints;
          sp->functional[f].quadWeights = qweights;
          ++f;
        }
        ierr = DMPlexVecRestoreClosure(dm, csection, coordinates, p, &n, &coords);CHKERRQ(ierr);
        lag->numDof[1] = order-1;
      } else if ((p >= pStart[depth-1]) && (p < pEnd[depth-1])) {
        /* Faces */

        if (order < 3) continue;
        lag->numDof[depth-1] = 0;
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to implement faces");
      } else if ((p >= pStart[depth]) && (p < pEnd[depth])) {
        /* Cells */

        if ((order > 0) && (order < 3)) continue;
        lag->numDof[depth] = 0;
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Too lazy to implement cells");
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, pStart[depth], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle cells with cone size %d", coneSize);
  ierr = PetscFree2(pStart,pEnd);CHKERRQ(ierr);
  if (f != pdim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vector %d not equal to dimension %d", f, pdim);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceDestroy_Lagrange"
PetscErrorCode PetscDualSpaceDestroy_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscFree(lag->numDof);CHKERRQ(ierr);
  ierr = PetscFree(lag);CHKERRQ(ierr);
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
#define __FUNCT__ "PetscDualSpaceGetNumDof_Lagrange"
PetscErrorCode PetscDualSpaceGetNumDof_Lagrange(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  *numDof = lag->numDof;
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
  sp->ops->destroy        = PetscDualSpaceDestroy_Lagrange;
  sp->ops->getdimension   = PetscDualSpaceGetDimension_Lagrange;
  sp->ops->getnumdof      = PetscDualSpaceGetNumDof_Lagrange;
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

  lag->numDof = NULL;

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

  ierr = PetscSpaceDestroy(&(*fem)->basisSpace);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&(*fem)->dualSpace);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&(*fem)->quadrature);CHKERRQ(ierr);
  ierr = PetscFree((*fem)->numDof);CHKERRQ(ierr);

  if ((*fem)->ops->destroy) {ierr = (*(*fem)->ops->destroy)(*fem);CHKERRQ(ierr);}
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

  f->basisSpace    = NULL;
  f->dualSpace     = NULL;
  f->numComponents = 1;
  f->numDof        = NULL;
  ierr = PetscMemzero(&f->quadrature, sizeof(PetscQuadrature));CHKERRQ(ierr);

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
#define __FUNCT__ "PetscFEGetSpatialDimension"
PetscErrorCode PetscFEGetSpatialDimension(PetscFE fem, PetscInt *dim)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(dim, 2);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFESetNumComponents"
PetscErrorCode PetscFESetNumComponents(PetscFE fem, PetscInt comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  fem->numComponents = comp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetNumComponents"
PetscErrorCode PetscFEGetNumComponents(PetscFE fem, PetscInt *comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(comp, 2);
  *comp = fem->numComponents;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 2);
  ierr = PetscSpaceDestroy(&fem->basisSpace);CHKERRQ(ierr);
  fem->basisSpace = sp;
  ierr = PetscObjectReference((PetscObject) fem->basisSpace);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 2);
  ierr = PetscDualSpaceDestroy(&fem->dualSpace);CHKERRQ(ierr);
  fem->dualSpace = sp;
  ierr = PetscObjectReference((PetscObject) fem->dualSpace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetQuadrature"
PetscErrorCode PetscFEGetQuadrature(PetscFE fem, PetscQuadrature *q)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(q, 2);
  *q = fem->quadrature;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFESetQuadrature"
PetscErrorCode PetscFESetQuadrature(PetscFE fem, PetscQuadrature q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscQuadratureDestroy(&fem->quadrature);CHKERRQ(ierr);
  fem->quadrature = q;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetNumDof"
PetscErrorCode PetscFEGetNumDof(PetscFE fem, const PetscInt **numDof)
{
  const PetscInt *numDofDual;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(numDof, 2);
  ierr = PetscDualSpaceGetNumDof(fem->dualSpace, &numDofDual);CHKERRQ(ierr);
  if (!fem->numDof) {
    DM       dm;
    PetscInt dim, d;

    ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
    ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = PetscMalloc((dim+1) * sizeof(PetscInt), &fem->numDof);CHKERRQ(ierr);
    for (d = 0; d <= dim; ++d) {
      fem->numDof[d] = fem->numComponents*numDofDual[d];
    }
  }
  *numDof = fem->numDof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFEGetTabulation"
PetscErrorCode PetscFEGetTabulation(PetscFE fem, PetscReal **B, PetscReal **D, PetscReal **H)
{
  DM               dm;
  PetscInt         pdim; /* Dimension of FE space P */
  PetscInt         dim;  /* Spatial dimension */
  PetscInt         comp; /* Field components */
  PetscInt         npoints = fem->quadrature.numQuadPoints;
  const PetscReal *points  = fem->quadrature.quadPoints;
  PetscReal       *tmpB, *tmpD, *invV;
  PetscInt         p, d, j, k;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  PetscValidPointer(points, 3);
  if (B) PetscValidPointer(B, 4);
  if (D) PetscValidPointer(D, 5);
  if (H) PetscValidPointer(H, 6);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);

  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(fem->basisSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &comp);CHKERRQ(ierr);
  /* if (nvalues%dim) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of coordinate values %d must be divisible by the spatial dimension %d", nvalues, dim); */

  if (B) {
    ierr = DMGetWorkArray(dm, npoints*pdim*comp, PETSC_REAL, B);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, npoints*pdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);
  }
  if (D) {
    ierr = DMGetWorkArray(dm, npoints*pdim*comp*dim, PETSC_REAL, D);CHKERRQ(ierr);
    ierr = DMGetWorkArray(dm, npoints*pdim*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);
  }
  if (H) {ierr = DMGetWorkArray(dm, npoints*pdim*dim*dim, PETSC_REAL, H);CHKERRQ(ierr);}
  ierr = PetscSpaceEvaluate(fem->basisSpace, npoints, points, B ? tmpB : NULL, D ? tmpD : NULL, H ? *H : NULL);CHKERRQ(ierr);

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
        const PetscInt i = (p*pdim + j)*comp;
        PetscInt       c;

        (*B)[i] = 0.0;
        for (k = 0; k < pdim; ++k) {
          (*B)[i] += invV[k*pdim+j] * tmpB[p*pdim + k];
        }
        for (c = 1; c < comp; ++c) {
          (*B)[i+c] = (*B)[i];
        }
      }
    }
    if (D) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        for (d = 0; d < dim; ++d) {
          const PetscInt i = ((p*pdim + j)*comp + 0)*dim + d;
          PetscInt       c;

          (*D)[i] = 0.0;
          for (k = 0; k < pdim; ++k) {
            (*D)[i] += invV[k*pdim+j] * tmpD[(p*pdim + k)*dim + d];
          }
          for (c = 1; c < comp; ++c) {
            (*D)[((p*pdim + j)*comp + c)*dim + d] = (*D)[i];
          }
        }
      }
    }
  }
  ierr = DMRestoreWorkArray(dm, pdim*pdim, PETSC_REAL, &invV);CHKERRQ(ierr);
  if (B) {ierr = DMRestoreWorkArray(dm, npoints*pdim, PETSC_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, npoints*pdim*dim, PETSC_REAL, &tmpD);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFERestoreTabulation"
PetscErrorCode PetscFERestoreTabulation(PetscFE fem, PetscReal **B, PetscReal **D, PetscReal **H)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  if (B) {ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, B);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, D);CHKERRQ(ierr);}
  if (H) {ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, H);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
Purpose: Compute element vector for chunk of elements

Input:
  Sizes:
     Ne:  number of elements
     Nf:  number of fields
     PetscFE
       dim: spatial dimension
       Nb:  number of basis functions
       Nc:  number of field components
       PetscQuadrature
         Nq:  number of quadrature points

  Geometry:
     PetscCellGeometry
       PetscReal v0s[Ne*dim]
       PetscReal jacobians[Ne*dim*dim]        possibly *Nq
       PetscReal jacobianInverses[Ne*dim*dim] possibly *Nq
       PetscReal jacobianDeterminants[Ne]     possibly *Nq
  FEM:
     PetscFE
       PetscQuadrature
         PetscReal   quadPoints[Nq*dim]
         PetscReal   quadWeights[Nq]
       PetscReal   basis[Nq*Nb*Nc]
       PetscReal   basisDer[Nq*Nb*Nc*dim]
     PetscScalar coefficients[Ne*Nb*Nc]
     PetscScalar elemVec[Ne*Nb*Nc]

  Problem:
     PetscInt f: the active field
     f0, f1

  Work Space:
     PetscFE
       PetscScalar f0[Nq*dim];
       PetscScalar f1[Nq*dim*dim];
       PetscScalar u[Nc];
       PetscScalar gradU[Nc*dim];
       PetscReal   x[dim];
       PetscScalar realSpaceDer[dim];

Purpose: Compute element vector for N_cb batches of elements

Input:
  Sizes:
     N_cb: Number of serial cell batches

  Geometry:
     PetscReal v0s[Ne*dim]
     PetscReal jacobians[Ne*dim*dim]        possibly *Nq
     PetscReal jacobianInverses[Ne*dim*dim] possibly *Nq
     PetscReal jacobianDeterminants[Ne]     possibly *Nq
  FEM:
     static PetscReal   quadPoints[Nq*dim]
     static PetscReal   quadWeights[Nq]
     static PetscReal   basis[Nq*Nb*Nc]
     static PetscReal   basisDer[Nq*Nb*Nc*dim]
     PetscScalar coefficients[Ne*Nb*Nc]
     PetscScalar elemVec[Ne*Nb*Nc]

ex62.c:
  PetscErrorCode PetscFEIntegrateResidualBatch(PetscInt Ne, PetscInt numFields, PetscInt field, PetscQuadrature quad[], const PetscScalar coefficients[],
                                               const PetscReal v0s[], const PetscReal jacobians[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[],
                                               void (*f0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[]),
                                               void (*f1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f1[]), PetscScalar elemVec[])

ex52.c:
  PetscErrorCode IntegrateLaplacianBatchCPU(PetscInt Ne, PetscInt Nb, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscInt Nq, const PetscReal quadPoints[], const PetscReal quadWeights[], const PetscReal basisTabulation[], const PetscReal basisDerTabulation[], PetscScalar elemVec[], AppCtx *user)
  PetscErrorCode IntegrateElasticityBatchCPU(PetscInt Ne, PetscInt Nb, PetscInt Ncomp, const PetscScalar coefficients[], const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscInt Nq, const PetscReal quadPoints[], const PetscReal quadWeights[], const PetscReal basisTabulation[], const PetscReal basisDerTabulation[], PetscScalar elemVec[], AppCtx *user)

ex52_integrateElement.cu
__global__ void integrateElementQuadrature(int N_cb, realType *coefficients, realType *jacobianInverses, realType *jacobianDeterminants, realType *elemVec)

PETSC_EXTERN PetscErrorCode IntegrateElementBatchGPU(PetscInt spatial_dim, PetscInt Ne, PetscInt Ncb, PetscInt Nbc, PetscInt Nbl, const PetscScalar coefficients[],
                                                     const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscScalar elemVec[],
                                                     PetscLogEvent event, PetscInt debug, PetscInt pde_op)

ex52_integrateElementOpenCL.c:
PETSC_EXTERN PetscErrorCode IntegrateElementBatchGPU(PetscInt spatial_dim, PetscInt Ne, PetscInt Ncb, PetscInt Nbc, PetscInt N_bl, const PetscScalar coefficients[],
                                                     const PetscReal jacobianInverses[], const PetscReal jacobianDeterminants[], PetscScalar elemVec[],
                                                     PetscLogEvent event, PetscInt debug, PetscInt pde_op)

__kernel void integrateElementQuadrature(int N_cb, __global float *coefficients, __global float *jacobianInverses, __global float *jacobianDeterminants, __global float *elemVec)
*/

#undef __FUNCT__
#define __FUNCT__ "PetscFEIntegrateResidualChunk"
/*C
  PetscFEIntegrateResidualChunk - Produce the element residual vector for a batch of elements by quadrature integration

  Not collective

  Input Parameters:
+ dim                  - The spatial dimension
. Nf                   - The number of physical fields
. Nc                   - The total number of fields components
. field                - The field being integrated
. Ne                   - The number of elements
. Nq                   - The number of quadrature points in this field
. Nb                   - The number of basis functions in this field
. v0s                  - The coordinates of the initial vertex for each element (the constant part of the transform from the reference element)
. jacobians            - The Jacobian for each element (the linear part of the transform from the reference element)
. jacobianInverses     - The Jacobian inverse for each element (the linear part of the transform to the reference element)
. jacobianDeterminants - The Jacobian determinant for each element

. quadPoints           - The quadrature points
. quadWeights          - The quadrature weights
. coefficients         - The array of FEM basis coefficients for the elements
. f0_func              - f_0 function from the first order FEM model
- f1_func              - f_1 function from the first order FEM model

  Output Parameter
. elemVec              - the element residual vectors from each element

   Calling sequence of f0_func and f1_func:
$    void f0(const PetscScalar u[], const PetscScalar gradU[], const PetscReal x[], PetscScalar f0[])

  Note:
$ Loop over batch of elements (e):
$   Loop over quadrature points (q):
$     Make u_q and gradU_q (loops over fields,Nb,Ncomp) and x_q
$     Call f_0 and f_1
$   Loop over element vector entries (f,fc --> i):
$     elemVec[i] += \psi^{fc}_f(q) f0_{fc}(u, \nabla u) + \nabla\psi^{fc}_f(q) \cdot f1_{fc,df}(u, \nabla u)
*/
PetscErrorCode PetscFEIntegrateResidualChunk(PetscInt Ne, PetscInt Nf, PetscFE fe[], PetscInt field, PetscCellGeometry geom, const PetscScalar coefficients[],
                                             void (*f0_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[]),
                                             void (*f1_func)(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[]),
                                             PetscScalar elemVec[])
{
  const PetscInt  debug = 0;
  PetscQuadrature quad;
  PetscScalar    *f0, *f1, *u, *gradU;
  PetscReal      *x, *realSpaceDer;
  PetscInt        dim, numComponents = 0, cOffset = 0, eOffset = 0, e, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fe[0], &dim);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscInt Nc;
    ierr = PetscFEGetNumComponents(fe[f], &Nc);CHKERRQ(ierr);
    numComponents += Nc;
  }
  ierr = PetscFEGetQuadrature(fe[field], &quad);CHKERRQ(ierr);
  ierr = PetscMalloc6(quad.numQuadPoints*dim,PetscScalar,&f0,quad.numQuadPoints*dim*dim,PetscScalar,&f1,numComponents,PetscScalar,&u,numComponents*dim,PetscScalar,&gradU,dim,PetscReal,&x,dim,PetscReal,&realSpaceDer);
  for (e = 0; e < Ne; ++e) {
    const PetscReal  detJ = geom.detJ[e];
    const PetscReal *v0   = &geom.v0[e*dim];
    const PetscReal *J    = &geom.J[e*dim*dim];
    const PetscReal *invJ = &geom.invJ[e*dim*dim];
    const PetscInt   Nq   = quad.numQuadPoints;
    PetscInt         q, f;

    if (debug > 1) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
      ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
    }
    for (q = 0; q < Nq; ++q) {
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      PetscInt         fOffset     = 0;
      PetscInt         dOffset     = cOffset;
      const PetscReal *quadPoints  = quad.quadPoints;
      const PetscReal *quadWeights = quad.quadWeights;
      PetscInt         Ncomp, d, d2, f, i;

      ierr = PetscFEGetNumComponents(fe[field], &Ncomp);CHKERRQ(ierr);
      for (d = 0; d < numComponents; ++d)       {u[d]     = 0.0;}
      for (d = 0; d < dim*(numComponents); ++d) {gradU[d] = 0.0;}
      for (d = 0; d < dim; ++d) {
        x[d] = v0[d];
        for (d2 = 0; d2 < dim; ++d2) {
          x[d] += J[d*dim+d2]*(quadPoints[q*dim+d2] + 1.0);
        }
      }
      for (f = 0; f < Nf; ++f) {
        PetscReal *basis, *basisDer;
        PetscInt   Nb, Ncomp, b, comp;

        ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
        ierr = PetscFEGetNumComponents(fe[f], &Ncomp);CHKERRQ(ierr);
        /* TODO: Hoist this tabulation out of the loops, maybe by memoizing */
        ierr = PetscFEGetTabulation(fe[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscInt       d, g;

            u[fOffset+comp] += coefficients[dOffset+cidx]*basis[q*Nb*Ncomp+cidx];
            for (d = 0; d < dim; ++d) {
              realSpaceDer[d] = 0.0;
              for (g = 0; g < dim; ++g) {
                realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
              }
              gradU[(fOffset+comp)*dim+d] += coefficients[dOffset+cidx]*realSpaceDer[d];
            }
          }
        }
        ierr = PetscFERestoreTabulation(fe[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        if (debug > 1) {
          PetscInt d;
          for (comp = 0; comp < Ncomp; ++comp) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    u[%d,%d]: %g\n", f, comp, u[fOffset+comp]);CHKERRQ(ierr);
            for (d = 0; d < dim; ++d) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    gradU[%d,%d]_%c: %g\n", f, comp, 'x'+d, gradU[(fOffset+comp)*dim+d]);CHKERRQ(ierr);
            }
          }
        }
        fOffset += Ncomp;
        dOffset += Nb*Ncomp;
      }

      f0_func(u, gradU, NULL, NULL, x, &f0[q*Ncomp]);
      for (i = 0; i < Ncomp; ++i) {
        f0[q*Ncomp+i] *= detJ*quadWeights[q];
      }
      f1_func(u, gradU, NULL, NULL, x, &f1[q*Ncomp*dim]);
      for (i = 0; i < Ncomp*dim; ++i) {
        f1[q*Ncomp*dim+i] *= detJ*quadWeights[q];
      }
      if (debug > 1) {
        PetscInt c,d;
        for (c = 0; c < Ncomp; ++c) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "    f0[%d]: %g\n", c, f0[q*Ncomp+c]);CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "    f1[%d]_%c: %g\n", c, 'x'+d, f1[(q*Ncomp + c)*dim+d]);CHKERRQ(ierr);
          }
        }
      }
      if (q == Nq-1) {cOffset = dOffset;}
    }
    for (f = 0; f < Nf; ++f) {
      PetscInt   Nb, Ncomp, b, comp;

      ierr = PetscFEGetDimension(fe[f], &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetNumComponents(fe[f], &Ncomp);CHKERRQ(ierr);
      if (f == field) {
        PetscReal *basis;
        PetscReal *basisDer;

        /* TODO: Hoist this tabulation out of the loops, maybe by memoizing */
        ierr = PetscFEGetTabulation(fe[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        for (b = 0; b < Nb; ++b) {
          for (comp = 0; comp < Ncomp; ++comp) {
            const PetscInt cidx = b*Ncomp+comp;
            PetscInt       q;

            elemVec[eOffset+cidx] = 0.0;
            for (q = 0; q < Nq; ++q) {
              PetscInt d, g;

              elemVec[eOffset+cidx] += basis[q*Nb*Ncomp+cidx]*f0[q*Ncomp+comp];
              for (d = 0; d < dim; ++d) {
                realSpaceDer[d] = 0.0;
                for (g = 0; g < dim; ++g) {
                  realSpaceDer[d] += invJ[g*dim+d]*basisDer[(q*Nb*Ncomp+cidx)*dim+g];
                }
                elemVec[eOffset+cidx] += realSpaceDer[d]*f1[(q*Ncomp+comp)*dim+d];
              }
            }
          }
        }
        ierr = PetscFERestoreTabulation(fe[f], &basis, &basisDer, NULL);CHKERRQ(ierr);
        if (debug > 1) {
          PetscInt b, comp;

          for (b = 0; b < Nb; ++b) {
            for (comp = 0; comp < Ncomp; ++comp) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemVec[%d,%d]: %g\n", b, comp, elemVec[eOffset+b*Ncomp+comp]);CHKERRQ(ierr);
            }
          }
        }
      }
      eOffset += Nb*Ncomp;
    }
  }
  ierr = PetscFree6(f0,f1,u,gradU,x,realSpaceDer);
  PetscFunctionReturn(0);
}
