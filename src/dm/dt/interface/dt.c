/* Discretization tools */

#include <petscconf.h>
#if defined(PETSC_HAVE_MATHIMF_H)
#include <mathimf.h>           /* this needs to be included before math.h */
#endif

#include <petscdt.h>            /*I "petscdt.h" I*/
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
  ierr = PetscFree(q->points);CHKERRQ(ierr);
  ierr = PetscFree(q->weights);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscQuadratureView"
PetscErrorCode PetscQuadratureView(PetscQuadrature quad, PetscViewer viewer)
{
  PetscInt       q, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer, "Quadrature on %d points\n  (", quad.numPoints);CHKERRQ(ierr);
  for (q = 0; q < quad.numPoints; ++q) {
    for (d = 0; d < quad.dim; ++d) {
      if (d) ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "%g\n", quad.points[q*quad.dim+d]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, ") %g\n", quad.weights[q]);CHKERRQ(ierr);
  }
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

  a1      = PetscPowReal(2, a+b+1);
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
    PetscReal r = -PetscCosReal((2.0*k + 1.0) * PETSC_PI / (2.0 * npoints)), dP;
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
      if (PetscAbsReal(delta) < eps) break;
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
  case 0:
    ierr = PetscFree(x);CHKERRQ(ierr);
    ierr = PetscFree(w);CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(PetscReal), &x);CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(PetscReal), &w);CHKERRQ(ierr);
    x[0] = 0.0;
    w[0] = 1.0;
    break;
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
  q->dim       = dim;
  q->numPoints = npoints;
  q->points    = x;
  q->weights   = w;
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
