static char help[] = "Tests 1D discretization tools.\n\n";

#include <petscdt.h>
#include <petscviewer.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dtimpl.h>

static PetscErrorCode CheckPoints(const char *name,PetscInt npoints,const PetscReal *points,PetscInt ndegrees,const PetscInt *degrees)
{
  PetscErrorCode ierr;
  PetscReal      *B,*D,*D2;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscMalloc3(npoints*ndegrees,&B,npoints*ndegrees,&D,npoints*ndegrees,&D2);CHKERRQ(ierr);
  ierr = PetscDTLegendreEval(npoints,points,ndegrees,degrees,B,D,D2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",name);CHKERRQ(ierr);
  for (i=0; i<npoints; i++) {
    for (j=0; j<ndegrees; j++) {
      PetscReal b,d,d2;
      b = B[i*ndegrees+j];
      d = D[i*ndegrees+j];
      d2 = D2[i*ndegrees+j];
      if (PetscAbsReal(b) < PETSC_SMALL) b   = 0;
      if (PetscAbsReal(d) < PETSC_SMALL) d   = 0;
      if (PetscAbsReal(d2) < PETSC_SMALL) d2 = 0;
      ierr = PetscPrintf(PETSC_COMM_WORLD,"degree %D at %12.4g: B=%12.4g  D=%12.4g  D2=%12.4g\n",degrees[j],(double)points[i],(double)b,(double)d,(double)d2);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree3(B,D,D2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef PetscErrorCode(*quadratureFunc)(PetscInt,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal[],PetscReal[]);

static PetscErrorCode CheckQuadrature_Basics(PetscInt npoints, PetscReal alpha, PetscReal beta, const PetscReal x[], const PetscReal w[])
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 1; i < npoints; i++) {
    if (x[i] <= x[i-1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Quadrature points not monotonically increasing, %D points, alpha = %g, beta = %g, i = %D, x[i] = %g, x[i-1] = %g",npoints, (double) alpha, (double) beta, i, x[i], x[i-1]);
  }
  for (i = 0; i < npoints; i++) {
    if (w[i] <= 0.) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Quadrature weight not positive, %D points, alpha = %g, beta = %g, i = %D, w[i] = %g",npoints, (double) alpha, (double) beta, i, w[i]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckQuadrature(PetscInt npoints, PetscReal alpha, PetscReal beta, const PetscReal x[], const PetscReal w[], PetscInt nexact)
{
  PetscInt i, j, k;
  PetscReal *Pi, *Pj;
  PetscReal eps;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  eps = PETSC_SMALL;
  ierr = PetscMalloc2(npoints, &Pi, npoints, &Pj);CHKERRQ(ierr);
  for (i = 0; i <= nexact; i++) {
    ierr = PetscDTJacobiEval(npoints, alpha, beta, x, 1, &i, Pi, NULL, NULL);CHKERRQ(ierr);
    for (j = i; j <= nexact - i; j++) {
      PetscReal I_quad = 0.;
      PetscReal I_exact = 0.;
      PetscReal err, tol;
      ierr = PetscDTJacobiEval(npoints, alpha, beta, x, 1, &j, Pj, NULL, NULL);CHKERRQ(ierr);

      tol = eps;
      if (i == j) {
        PetscReal norm, norm2diff;

        I_exact = PetscPowReal(2.0, alpha + beta + 1.) / (2.*i + alpha + beta + 1.);
#if defined(PETSC_HAVE_LGAMMA)
        I_exact *= PetscExpReal(PetscLGamma(i + alpha + 1.) + PetscLGamma(i + beta + 1.) - (PetscLGamma(i + alpha + beta + 1.) + PetscLGamma(i + 1.)));
#else
        {
          PetscInt ibeta = (PetscInt) beta;

          if ((PetscReal) ibeta != beta) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"lgamma() - math routine is unavailable.");
          for (k = 0; k < ibeta; k++) I_exact *= (i + 1. + k) / (i + alpha + 1. + k);
        }
#endif

        ierr = PetscDTJacobiNorm(alpha, beta, i, &norm);CHKERRQ(ierr);
        norm2diff = PetscAbsReal(norm*norm - I_exact);
        if (norm2diff > eps * I_exact) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Jacobi norm error %g", (double) norm2diff);

        tol = eps * I_exact;
      }
      for (k = 0; k < npoints; k++) I_quad += w[k] * (Pi[k] * Pj[k]);
      err = PetscAbsReal(I_exact - I_quad);
      ierr = PetscInfo(NULL,"npoints %D, alpha %g, beta %g, i %D, j %D, exact %g, err %g\n", npoints, (double) alpha, (double) beta, i, j, (double) I_exact, (double) err);CHKERRQ(ierr);
      if (err > tol) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrectly integrated P_%D * P_%D using %D point rule with alpha = %g, beta = %g: exact %g, err %g", i, j, npoints, (double) alpha, (double) beta, (double) I_exact, (double) err);
    }
  }
  ierr = PetscFree2(Pi, Pj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckJacobiQuadrature(PetscInt npoints, PetscReal alpha, PetscReal beta, quadratureFunc func, PetscInt nexact)
{
  PetscReal *x, *w;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(npoints, &x, npoints, &w);CHKERRQ(ierr);
  ierr = (*func)(npoints, -1., 1., alpha, beta, x, w);CHKERRQ(ierr);
  ierr = CheckQuadrature_Basics(npoints, alpha, beta, x, w);CHKERRQ(ierr);
  ierr = CheckQuadrature(npoints, alpha, beta, x, w, nexact);CHKERRQ(ierr);
#if defined(PETSCDTGAUSSIANQUADRATURE_EIG)
  /* compare methods of computing quadrature */
  PetscDTGaussQuadratureNewton_Internal = (PetscBool) !PetscDTGaussQuadratureNewton_Internal;
  {
    PetscReal *x2, *w2;
    PetscReal eps;
    PetscInt i;

    eps = PETSC_SMALL;
    ierr = PetscMalloc2(npoints, &x2, npoints, &w2);CHKERRQ(ierr);
    ierr = (*func)(npoints, -1., 1., alpha, beta, x2, w2);CHKERRQ(ierr);
    ierr = CheckQuadrature_Basics(npoints, alpha, beta, x2, w2);CHKERRQ(ierr);
    ierr = CheckQuadrature(npoints, alpha, beta, x2, w2, nexact);CHKERRQ(ierr);
    for (i = 0; i < npoints; i++) {
      PetscReal xdiff, xtol, wdiff, wtol;

      xdiff = PetscAbsReal(x[i] - x2[i]);
      wdiff = PetscAbsReal(w[i] - w2[i]);
      xtol = eps * (1. + PetscMin(PetscAbsReal(x[i]),1. - PetscAbsReal(x[i])));
      wtol = eps * (1. + w[i]);
      ierr = PetscInfo(NULL,"npoints %D, alpha %g, beta %g, i %D, xdiff/xtol %g, wdiff/wtol %g\n", npoints, (double) alpha, (double) beta, i, (double) xdiff/xtol, (double) wdiff/wtol);CHKERRQ(ierr);
      if (xdiff > xtol) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatch quadrature point: %D points, alpha = %g, beta = %g, i = %D, xdiff = %g", npoints, (double) alpha, (double) beta, i, (double) xdiff);
      if (wdiff > wtol) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatch quadrature weight: %D points, alpha = %g, beta = %g, i = %D, wdiff = %g", npoints, (double) alpha, (double) beta, i, (double) wdiff);
    }
    ierr = PetscFree2(x2, w2);CHKERRQ(ierr);
  }
  /* restore */
  PetscDTGaussQuadratureNewton_Internal = (PetscBool) !PetscDTGaussQuadratureNewton_Internal;
#endif
  ierr = PetscFree2(x, w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       degrees[1000],ndegrees,npoints,two;
  PetscReal      points[1000],weights[1000],interval[2];
  PetscInt       minpoints, maxpoints;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Discretization tools test options",NULL);CHKERRQ(ierr);
  {
    ndegrees   = 1000;
    degrees[0] = 0;
    degrees[1] = 1;
    degrees[2] = 2;
    ierr       = PetscOptionsIntArray("-degrees","list of degrees to evaluate","",degrees,&ndegrees,&flg);CHKERRQ(ierr);

    if (!flg) ndegrees = 3;
    npoints   = 1000;
    points[0] = 0.0;
    points[1] = -0.5;
    points[2] = 1.0;
    ierr      = PetscOptionsRealArray("-points","list of points at which to evaluate","",points,&npoints,&flg);CHKERRQ(ierr);

    if (!flg) npoints = 3;
    two         = 2;
    interval[0] = -1.;
    interval[1] = 1.;
    ierr        = PetscOptionsRealArray("-interval","interval on which to construct quadrature","",interval,&two,NULL);CHKERRQ(ierr);

    minpoints = 1;
    ierr = PetscOptionsInt("-minpoints","minimum points for thorough Gauss-Jacobi quadrature tests","",minpoints,&minpoints,NULL);CHKERRQ(ierr);
    maxpoints = 30;
#if defined(PETSC_USE_REAL_SINGLE)
    maxpoints = 5;
#elif defined(PETSC_USE_REAL___FLOAT128)
    maxpoints = 20; /* just to make test faster */
#endif
    ierr = PetscOptionsInt("-maxpoints","maximum points for thorough Gauss-Jacobi quadrature tests","",maxpoints,&maxpoints,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = CheckPoints("User-provided points",npoints,points,ndegrees,degrees);CHKERRQ(ierr);

  ierr = PetscDTGaussQuadrature(npoints,interval[0],interval[1],points,weights);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Quadrature weights\n");CHKERRQ(ierr);
  ierr = PetscRealView(npoints,weights,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  {
    PetscReal a = interval[0],b = interval[1],zeroth,first,second;
    PetscInt  i;
    zeroth = b - a;
    first  = (b*b - a*a)/2;
    second = (b*b*b - a*a*a)/3;
    for (i=0; i<npoints; i++) {
      zeroth -= weights[i];
      first  -= weights[i] * points[i];
      second -= weights[i] * PetscSqr(points[i]);
    }
    if (PetscAbs(zeroth) < 1e-10) zeroth = 0.;
    if (PetscAbs(first)  < 1e-10) first  = 0.;
    if (PetscAbs(second) < 1e-10) second = 0.;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Moment error: zeroth=%g, first=%g, second=%g\n",(double)(-zeroth),(double)(-first),(double)(-second));CHKERRQ(ierr);
  }
  ierr = CheckPoints("Gauss points",npoints,points,ndegrees,degrees);CHKERRQ(ierr);
  {
    PetscInt  i;

    for (i = minpoints; i <= maxpoints; i++) {
      PetscReal a1, b1, a2, b2;

#if defined(PETSC_HAVE_LGAMMA)
      a1 = -0.6;
      b1 = 1.1;
      a2 = 2.2;
      b2 = -0.6;
#else
      a1 = 0.;
      b1 = 1.;
      a2 = 2.;
      b2 = 0.;
#endif
      ierr = CheckJacobiQuadrature(i, 0., 0., PetscDTGaussJacobiQuadrature, 2*i-1);CHKERRQ(ierr);
      ierr = CheckJacobiQuadrature(i, a1, b1, PetscDTGaussJacobiQuadrature, 2*i-1);CHKERRQ(ierr);
      ierr = CheckJacobiQuadrature(i, a2, b2, PetscDTGaussJacobiQuadrature, 2*i-1);CHKERRQ(ierr);
      if (i >= 2) {
        ierr = CheckJacobiQuadrature(i, 0., 0., PetscDTGaussLobattoJacobiQuadrature, 2*i-3);CHKERRQ(ierr);
        ierr = CheckJacobiQuadrature(i, a1, b1, PetscDTGaussLobattoJacobiQuadrature, 2*i-3);CHKERRQ(ierr);
        ierr = CheckJacobiQuadrature(i, a2, b2, PetscDTGaussLobattoJacobiQuadrature, 2*i-3);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 1
    args: -degrees 1,2,3,4,5 -points 0,.2,-.5,.8,.9,1 -interval -.5,1
TEST*/
