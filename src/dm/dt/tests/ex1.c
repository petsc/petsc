static char help[] = "Tests 1D discretization tools.\n\n";

#include <petscdt.h>
#include <petscviewer.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dtimpl.h>

static PetscErrorCode CheckPoints(const char *name,PetscInt npoints,const PetscReal *points,PetscInt ndegrees,const PetscInt *degrees)
{
  PetscReal      *B,*D,*D2;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscCall(PetscMalloc3(npoints*ndegrees,&B,npoints*ndegrees,&D,npoints*ndegrees,&D2));
  PetscCall(PetscDTLegendreEval(npoints,points,ndegrees,degrees,B,D,D2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s\n",name));
  for (i=0; i<npoints; i++) {
    for (j=0; j<ndegrees; j++) {
      PetscReal b,d,d2;
      b = B[i*ndegrees+j];
      d = D[i*ndegrees+j];
      d2 = D2[i*ndegrees+j];
      if (PetscAbsReal(b) < PETSC_SMALL) b   = 0;
      if (PetscAbsReal(d) < PETSC_SMALL) d   = 0;
      if (PetscAbsReal(d2) < PETSC_SMALL) d2 = 0;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"degree %D at %12.4g: B=%12.4g  D=%12.4g  D2=%12.4g\n",degrees[j],(double)points[i],(double)b,(double)d,(double)d2));
    }
  }
  PetscCall(PetscFree3(B,D,D2));
  PetscFunctionReturn(0);
}

typedef PetscErrorCode(*quadratureFunc)(PetscInt,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal[],PetscReal[]);

static PetscErrorCode CheckQuadrature_Basics(PetscInt npoints, PetscReal alpha, PetscReal beta, const PetscReal x[], const PetscReal w[])
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 1; i < npoints; i++) {
    PetscCheck(x[i] > x[i-1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Quadrature points not monotonically increasing, %D points, alpha = %g, beta = %g, i = %D, x[i] = %g, x[i-1] = %g",npoints, (double) alpha, (double) beta, i, x[i], x[i-1]);
  }
  for (i = 0; i < npoints; i++) {
    PetscCheck(w[i] > 0.,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Quadrature weight not positive, %D points, alpha = %g, beta = %g, i = %D, w[i] = %g",npoints, (double) alpha, (double) beta, i, w[i]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckQuadrature(PetscInt npoints, PetscReal alpha, PetscReal beta, const PetscReal x[], const PetscReal w[], PetscInt nexact)
{
  PetscInt i, j, k;
  PetscReal *Pi, *Pj;
  PetscReal eps;

  PetscFunctionBegin;
  eps = PETSC_SMALL;
  PetscCall(PetscMalloc2(npoints, &Pi, npoints, &Pj));
  for (i = 0; i <= nexact; i++) {
    PetscCall(PetscDTJacobiEval(npoints, alpha, beta, x, 1, &i, Pi, NULL, NULL));
    for (j = i; j <= nexact - i; j++) {
      PetscReal I_quad = 0.;
      PetscReal I_exact = 0.;
      PetscReal err, tol;
      PetscCall(PetscDTJacobiEval(npoints, alpha, beta, x, 1, &j, Pj, NULL, NULL));

      tol = eps;
      if (i == j) {
        PetscReal norm, norm2diff;

        I_exact = PetscPowReal(2.0, alpha + beta + 1.) / (2.*i + alpha + beta + 1.);
#if defined(PETSC_HAVE_LGAMMA)
        I_exact *= PetscExpReal(PetscLGamma(i + alpha + 1.) + PetscLGamma(i + beta + 1.) - (PetscLGamma(i + alpha + beta + 1.) + PetscLGamma(i + 1.)));
#else
        {
          PetscInt ibeta = (PetscInt) beta;

          PetscCheck((PetscReal) ibeta == beta,PETSC_COMM_SELF,PETSC_ERR_SUP,"lgamma() - math routine is unavailable.");
          for (k = 0; k < ibeta; k++) I_exact *= (i + 1. + k) / (i + alpha + 1. + k);
        }
#endif

        PetscCall(PetscDTJacobiNorm(alpha, beta, i, &norm));
        norm2diff = PetscAbsReal(norm*norm - I_exact);
        PetscCheckFalse(norm2diff > eps * I_exact,PETSC_COMM_SELF,PETSC_ERR_PLIB, "Jacobi norm error %g", (double) norm2diff);

        tol = eps * I_exact;
      }
      for (k = 0; k < npoints; k++) I_quad += w[k] * (Pi[k] * Pj[k]);
      err = PetscAbsReal(I_exact - I_quad);
      PetscCall(PetscInfo(NULL,"npoints %D, alpha %g, beta %g, i %D, j %D, exact %g, err %g\n", npoints, (double) alpha, (double) beta, i, j, (double) I_exact, (double) err));
      PetscCheck(err <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrectly integrated P_%D * P_%D using %D point rule with alpha = %g, beta = %g: exact %g, err %g", i, j, npoints, (double) alpha, (double) beta, (double) I_exact, (double) err);
    }
  }
  PetscCall(PetscFree2(Pi, Pj));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckJacobiQuadrature(PetscInt npoints, PetscReal alpha, PetscReal beta, quadratureFunc func, PetscInt nexact)
{
  PetscReal *x, *w;

  PetscFunctionBegin;
  PetscCall(PetscMalloc2(npoints, &x, npoints, &w));
  PetscCall((*func)(npoints, -1., 1., alpha, beta, x, w));
  PetscCall(CheckQuadrature_Basics(npoints, alpha, beta, x, w));
  PetscCall(CheckQuadrature(npoints, alpha, beta, x, w, nexact));
#if defined(PETSCDTGAUSSIANQUADRATURE_EIG)
  /* compare methods of computing quadrature */
  PetscDTGaussQuadratureNewton_Internal = (PetscBool) !PetscDTGaussQuadratureNewton_Internal;
  {
    PetscReal *x2, *w2;
    PetscReal eps;
    PetscInt i;

    eps = PETSC_SMALL;
    PetscCall(PetscMalloc2(npoints, &x2, npoints, &w2));
    PetscCall((*func)(npoints, -1., 1., alpha, beta, x2, w2));
    PetscCall(CheckQuadrature_Basics(npoints, alpha, beta, x2, w2));
    PetscCall(CheckQuadrature(npoints, alpha, beta, x2, w2, nexact));
    for (i = 0; i < npoints; i++) {
      PetscReal xdiff, xtol, wdiff, wtol;

      xdiff = PetscAbsReal(x[i] - x2[i]);
      wdiff = PetscAbsReal(w[i] - w2[i]);
      xtol = eps * (1. + PetscMin(PetscAbsReal(x[i]),1. - PetscAbsReal(x[i])));
      wtol = eps * (1. + w[i]);
      PetscCall(PetscInfo(NULL,"npoints %D, alpha %g, beta %g, i %D, xdiff/xtol %g, wdiff/wtol %g\n", npoints, (double) alpha, (double) beta, i, (double) xdiff/xtol, (double) wdiff/wtol));
      PetscCheck(xdiff <= xtol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatch quadrature point: %D points, alpha = %g, beta = %g, i = %D, xdiff = %g", npoints, (double) alpha, (double) beta, i, (double) xdiff);
      PetscCheck(wdiff <= wtol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Mismatch quadrature weight: %D points, alpha = %g, beta = %g, i = %D, wdiff = %g", npoints, (double) alpha, (double) beta, i, (double) wdiff);
    }
    PetscCall(PetscFree2(x2, w2));
  }
  /* restore */
  PetscDTGaussQuadratureNewton_Internal = (PetscBool) !PetscDTGaussQuadratureNewton_Internal;
#endif
  PetscCall(PetscFree2(x, w));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscInt       degrees[1000],ndegrees,npoints,two;
  PetscReal      points[1000],weights[1000],interval[2];
  PetscInt       minpoints, maxpoints;
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Discretization tools test options",NULL);
  {
    ndegrees   = 1000;
    degrees[0] = 0;
    degrees[1] = 1;
    degrees[2] = 2;
    PetscCall(PetscOptionsIntArray("-degrees","list of degrees to evaluate","",degrees,&ndegrees,&flg));

    if (!flg) ndegrees = 3;
    npoints   = 1000;
    points[0] = 0.0;
    points[1] = -0.5;
    points[2] = 1.0;
    PetscCall(PetscOptionsRealArray("-points","list of points at which to evaluate","",points,&npoints,&flg));

    if (!flg) npoints = 3;
    two         = 2;
    interval[0] = -1.;
    interval[1] = 1.;
    PetscCall(PetscOptionsRealArray("-interval","interval on which to construct quadrature","",interval,&two,NULL));

    minpoints = 1;
    PetscCall(PetscOptionsInt("-minpoints","minimum points for thorough Gauss-Jacobi quadrature tests","",minpoints,&minpoints,NULL));
    maxpoints = 30;
#if defined(PETSC_USE_REAL_SINGLE)
    maxpoints = 5;
#elif defined(PETSC_USE_REAL___FLOAT128)
    maxpoints = 20; /* just to make test faster */
#endif
    PetscCall(PetscOptionsInt("-maxpoints","maximum points for thorough Gauss-Jacobi quadrature tests","",maxpoints,&maxpoints,NULL));
  }
  PetscOptionsEnd();
  PetscCall(CheckPoints("User-provided points",npoints,points,ndegrees,degrees));

  PetscCall(PetscDTGaussQuadrature(npoints,interval[0],interval[1],points,weights));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Quadrature weights\n"));
  PetscCall(PetscRealView(npoints,weights,PETSC_VIEWER_STDOUT_WORLD));
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
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Moment error: zeroth=%g, first=%g, second=%g\n",(double)(-zeroth),(double)(-first),(double)(-second)));
  }
  PetscCall(CheckPoints("Gauss points",npoints,points,ndegrees,degrees));
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
      PetscCall(CheckJacobiQuadrature(i, 0., 0., PetscDTGaussJacobiQuadrature, 2*i-1));
      PetscCall(CheckJacobiQuadrature(i, a1, b1, PetscDTGaussJacobiQuadrature, 2*i-1));
      PetscCall(CheckJacobiQuadrature(i, a2, b2, PetscDTGaussJacobiQuadrature, 2*i-1));
      if (i >= 2) {
        PetscCall(CheckJacobiQuadrature(i, 0., 0., PetscDTGaussLobattoJacobiQuadrature, 2*i-3));
        PetscCall(CheckJacobiQuadrature(i, a1, b1, PetscDTGaussLobattoJacobiQuadrature, 2*i-3));
        PetscCall(CheckJacobiQuadrature(i, a2, b2, PetscDTGaussLobattoJacobiQuadrature, 2*i-3));
      }
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 1
    args: -degrees 1,2,3,4,5 -points 0,.2,-.5,.8,.9,1 -interval -.5,1
TEST*/
