const char help[] = "Tests PetscDTPKDEvalJet()";

#include <petscdt.h>
#include <petscblaslapack.h>

static PetscErrorCode testOrthogonality(PetscInt dim, PetscInt deg)
{
  PetscQuadrature q;
  const PetscReal *points, *weights;
  PetscInt        Npoly, npoints, i, j, k;
  PetscReal       *p;

  PetscFunctionBegin;
  PetscCall(PetscDTStroudConicalQuadrature(dim, 1, deg + 1, -1., 1., &q));
  PetscCall(PetscQuadratureGetData(q, NULL, NULL, &npoints, &points, &weights));
  PetscCall(PetscDTBinomialInt(dim + deg, dim, &Npoly));
  PetscCall(PetscMalloc1(Npoly * npoints, &p));
  PetscCall(PetscDTPKDEvalJet(dim, npoints, points, deg, 0, p));
  for (i = 0; i < Npoly; i++) {
    for (j = i; j < Npoly; j++) {
      PetscReal integral = 0.;
      PetscReal exact = (i == j) ? 1. : 0.;

      for (k = 0; k < npoints; k++) integral += weights[k] * p[i * npoints + k] * p[j * npoints + k];
      PetscCheck(PetscAbsReal(integral - exact) <= PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_PLIB, "<P[%" PetscInt_FMT "], P[%" PetscInt_FMT "]> = %g != delta_{%" PetscInt_FMT ",%" PetscInt_FMT "}", i, j, (double) integral, i, j);
    }
  }
  PetscCall(PetscFree(p));
  PetscCall(PetscQuadratureDestroy(&q));
  PetscFunctionReturn(0);
}

static PetscErrorCode testDerivativesLegendre(PetscInt dim, PetscInt deg, PetscInt k)
{
  PetscInt       Np, Nk, i, j, l, d, npoints;
  PetscRandom    rand;
  PetscReal      *point;
  PetscReal      *lgndre_coeffs;
  PetscReal      *pkd_coeffs;
  PetscReal      *proj;
  PetscReal     **B;
  PetscQuadrature q;
  PetscReal       *points1d;
  PetscInt        *degrees;
  PetscInt        *degtup, *ktup;
  const PetscReal *points;
  const PetscReal *weights;
  PetscReal      *lgndre_jet;
  PetscReal     **D;
  PetscReal      *pkd_jet, *pkd_jet_basis;

  PetscFunctionBegin;
  PetscCall(PetscDTBinomialInt(dim + deg, dim, &Np));
  PetscCall(PetscDTBinomialInt(dim + k, dim, &Nk));

  /* create the projector (because it is an orthonormal basis, the projector is the moment integrals) */
  PetscCall(PetscDTStroudConicalQuadrature(dim, 1, deg + 1, -1., 1., &q));
  PetscCall(PetscQuadratureGetData(q, NULL, NULL, &npoints, &points, &weights));
  PetscCall(PetscMalloc1(npoints * Np, &proj));
  PetscCall(PetscDTPKDEvalJet(dim, npoints, points, deg, 0, proj));
  for (i = 0; i < Np; i++) for (j = 0; j < npoints; j++) proj[i * npoints + j] *= weights[j];

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rand));
  PetscCall(PetscRandomSetInterval(rand, -1., 1.));

  /* create a random coefficient vector */
  PetscCall(PetscMalloc2(Np, &lgndre_coeffs, Np, &pkd_coeffs));
  for (i = 0; i < Np; i++) {
    PetscCall(PetscRandomGetValueReal(rand, &lgndre_coeffs[i]));
  }

  PetscCall(PetscMalloc2(dim, &degtup, dim, &ktup));
  PetscCall(PetscMalloc1(deg + 1, &degrees));
  for (i = 0; i < deg + 1; i++) degrees[i] = i;

  /* project the lgndre_coeffs to pkd_coeffs */
  PetscCall(PetscArrayzero(pkd_coeffs, Np));
  PetscCall(PetscMalloc1(npoints, &points1d));
  PetscCall(PetscMalloc1(dim, &B));
  for (d = 0; d < dim; d++) {
    PetscCall(PetscMalloc1((deg + 1)*npoints, &(B[d])));
    /* get this coordinate */
    for (i = 0; i < npoints; i++) points1d[i] = points[i * dim + d];
    PetscCall(PetscDTLegendreEval(npoints, points1d, deg + 1, degrees, B[d], NULL, NULL));
  }
  PetscCall(PetscFree(points1d));
  for (i = 0; i < npoints; i++) {
    PetscReal val = 0.;

    for (j = 0; j < Np; j++) {
      PetscReal mul = lgndre_coeffs[j];
      PetscReal valj = 1.;

      PetscCall(PetscDTIndexToGradedOrder(dim, j, degtup));
      for (l = 0; l < dim; l++) {
        valj *= B[l][i * (deg + 1) + degtup[l]];
      }
      val += mul * valj;
    }
    for (j = 0; j < Np; j++) {
      pkd_coeffs[j] += proj[j * npoints + i] * val;
    }
  }
  for (i = 0; i < dim; i++) {
    PetscCall(PetscFree(B[i]));
  }
  PetscCall(PetscFree(B));

  /* create a random point in the biunit simplex */
  PetscCall(PetscMalloc1(dim, &point));
  for (i = 0; i < dim; i++) {
    PetscCall(PetscRandomGetValueReal(rand, &point[i]));
  }
  for (i = dim - 1; i > 0; i--) {
    PetscReal val = point[i];
    PetscInt  j;

    for (j = 0; j < i; j++) {
      point[j] = (point[j] + 1.)*(1. - val)*0.5 - 1.;
    }
  }

  PetscCall(PetscMalloc3(Nk*Np, &pkd_jet_basis, Nk, &lgndre_jet, Nk, &pkd_jet));
  /* evaluate the jet at the point with PKD polynomials */
  PetscCall(PetscDTPKDEvalJet(dim, 1, point, deg, k, pkd_jet_basis));
  for (i = 0; i < Nk; i++) {
    PetscReal val = 0.;
    for (j = 0; j < Np; j++) {
      val += pkd_coeffs[j] * pkd_jet_basis[j * Nk + i];
    }
    pkd_jet[i] = val;
  }

  /* evaluate the 1D jets of the Legendre polynomials */
  PetscCall(PetscMalloc1(dim, &D));
  for (i = 0; i < dim; i++) {
    PetscCall(PetscMalloc1((deg + 1) * (k+1), &(D[i])));
    PetscCall(PetscDTJacobiEvalJet(0., 0., 1, &(point[i]), deg, k, D[i]));
  }
  /* compile the 1D Legendre jets into the tensor Legendre jet */
  for (j = 0; j < Nk; j++) lgndre_jet[j] = 0.;
  for (i = 0; i < Np; i++) {
    PetscReal mul = lgndre_coeffs[i];

    PetscCall(PetscDTIndexToGradedOrder(dim, i, degtup));
    for (j = 0; j < Nk; j++) {
      PetscReal val = 1.;

      PetscCall(PetscDTIndexToGradedOrder(dim, j, ktup));
      for (l = 0; l < dim; l++) {
        val *= D[l][degtup[l]*(k+1) + ktup[l]];
      }
      lgndre_jet[j] += mul * val;
    }
  }
  for (i = 0; i < dim; i++) {
    PetscCall(PetscFree(D[i]));
  }
  PetscCall(PetscFree(D));

  for (i = 0; i < Nk; i++) {
    PetscReal diff = lgndre_jet[i] - pkd_jet[i];
    PetscReal scale = 1. + PetscAbsReal(lgndre_jet[i]) + PetscAbsReal(pkd_jet[i]);
    PetscReal tol = 10. * PETSC_SMALL * scale;

    PetscCheck(PetscAbsReal(diff) <= tol,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Jet mismatch between PKD and tensor Legendre bases: error %g at tolerance %g", (double) diff, (double) tol);
  }

  PetscCall(PetscFree2(degtup,ktup));
  PetscCall(PetscFree(degrees));
  PetscCall(PetscFree3(pkd_jet_basis, lgndre_jet, pkd_jet));
  PetscCall(PetscFree(point));
  PetscCall(PetscFree2(lgndre_coeffs, pkd_coeffs));
  PetscCall(PetscFree(proj));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscQuadratureDestroy(&q));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       dim, deg, k;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  dim = 3;
  deg = 4;
  k = 3;
  PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for PetscDTPKDEval() tests","none");
  PetscCall(PetscOptionsInt("-dim", "Dimension of the simplex","ex9.c",dim,&dim,NULL));
  PetscCall(PetscOptionsInt("-degree", "The degree of the polynomial space","ex9.c",deg,&deg,NULL));
  PetscCall(PetscOptionsInt("-k", "The number of derivatives to use in the taylor test","ex9.c",k,&k,NULL));
  PetscOptionsEnd();
  PetscCall(testOrthogonality(dim, deg));
  PetscCall(testDerivativesLegendre(dim, deg, k));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    args: -dim {{1 2 3 4}}

TEST*/
