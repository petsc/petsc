const char help[] = "Tests PetscDTPKDEvalJet()";

#include <petscdt.h>
#include <petscblaslapack.h>

static PetscErrorCode testOrthogonality(PetscInt dim, PetscInt deg)
{
  PetscQuadrature q;
  const PetscReal *points, *weights;
  PetscInt        Npoly, npoints, i, j, k;
  PetscReal       *p;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscDTStroudConicalQuadrature(dim, 1, deg + 1, -1., 1., &q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, NULL, NULL, &npoints, &points, &weights);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + deg, dim, &Npoly);CHKERRQ(ierr);
  ierr = PetscMalloc1(Npoly * npoints, &p);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, npoints, points, deg, 0, p);CHKERRQ(ierr);
  for (i = 0; i < Npoly; i++) {
    for (j = i; j < Npoly; j++) {
      PetscReal integral = 0.;
      PetscReal exact = (i == j) ? 1. : 0.;

      for (k = 0; k < npoints; k++) integral += weights[k] * p[i * npoints + k] * p[j * npoints + k];
      if (PetscAbsReal(integral - exact) > PETSC_SMALL) SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_PLIB, "<P[%D], P[%D]> = %g != delta_{%D,%D}", i, j, (double) integral, i, j);
    }
  }
  ierr = PetscFree(p);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTBinomialInt(dim + deg, dim, &Np);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + k, dim, &Nk);CHKERRQ(ierr);

  /* create the projector (because it is an orthonormal basis, the projector is the moment integrals) */
  ierr = PetscDTStroudConicalQuadrature(dim, 1, deg + 1, -1., 1., &q);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(q, NULL, NULL, &npoints, &points, &weights);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints * Np, &proj);CHKERRQ(ierr);
  ierr = PetscDTPKDEvalJet(dim, npoints, points, deg, 0, proj);CHKERRQ(ierr);
  for (i = 0; i < Np; i++) for (j = 0; j < npoints; j++) proj[i * npoints + j] *= weights[j];

  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand, -1., 1.);CHKERRQ(ierr);

  /* create a random coefficient vector */
  ierr = PetscMalloc2(Np, &lgndre_coeffs, Np, &pkd_coeffs);CHKERRQ(ierr);
  for (i = 0; i < Np; i++) {
    ierr = PetscRandomGetValueReal(rand, &lgndre_coeffs[i]);CHKERRQ(ierr);
  }

  ierr = PetscMalloc2(dim, &degtup, dim, &ktup);CHKERRQ(ierr);
  ierr = PetscMalloc1(deg + 1, &degrees);CHKERRQ(ierr);
  for (i = 0; i < deg + 1; i++) degrees[i] = i;

  /* project the lgndre_coeffs to pkd_coeffs */
  ierr = PetscArrayzero(pkd_coeffs, Np);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints, &points1d);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim, &B);CHKERRQ(ierr);
  for (d = 0; d < dim; d++) {
    ierr = PetscMalloc1((deg + 1)*npoints, &(B[d]));CHKERRQ(ierr);
    /* get this coordinate */
    for (i = 0; i < npoints; i++) points1d[i] = points[i * dim + d];
    ierr = PetscDTLegendreEval(npoints, points1d, deg + 1, degrees, B[d], NULL, NULL);CHKERRQ(ierr);
  }
  ierr = PetscFree(points1d);CHKERRQ(ierr);
  for (i = 0; i < npoints; i++) {
    PetscReal val = 0.;

    for (j = 0; j < Np; j++) {
      PetscReal mul = lgndre_coeffs[j];
      PetscReal valj = 1.;

      ierr = PetscDTIndexToGradedOrder(dim, j, degtup);CHKERRQ(ierr);
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
    ierr = PetscFree(B[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(B);CHKERRQ(ierr);

  /* create a random point in the biunit simplex */
  ierr = PetscMalloc1(dim, &point);CHKERRQ(ierr);
  for (i = 0; i < dim; i++) {
    ierr = PetscRandomGetValueReal(rand, &point[i]);CHKERRQ(ierr);
  }
  for (i = dim - 1; i > 0; i--) {
    PetscReal val = point[i];
    PetscInt  j;

    for (j = 0; j < i; j++) {
      point[j] = (point[j] + 1.)*(1. - val)*0.5 - 1.;
    }
  }

  ierr = PetscMalloc3(Nk*Np, &pkd_jet_basis, Nk, &lgndre_jet, Nk, &pkd_jet);CHKERRQ(ierr);
  /* evaluate the jet at the point with PKD polynomials */
  ierr = PetscDTPKDEvalJet(dim, 1, point, deg, k, pkd_jet_basis);CHKERRQ(ierr);
  for (i = 0; i < Nk; i++) {
    PetscReal val = 0.;
    for (j = 0; j < Np; j++) {
      val += pkd_coeffs[j] * pkd_jet_basis[j * Nk + i];
    }
    pkd_jet[i] = val;
  }

  /* evaluate the 1D jets of the Legendre polynomials */
  ierr = PetscMalloc1(dim, &D);CHKERRQ(ierr);
  for (i = 0; i < dim; i++) {
    ierr = PetscMalloc1((deg + 1) * (k+1), &(D[i]));CHKERRQ(ierr);
    ierr = PetscDTJacobiEvalJet(0., 0., 1, &(point[i]), deg, k, D[i]);CHKERRQ(ierr);
  }
  /* compile the 1D Legendre jets into the tensor Legendre jet */
  for (j = 0; j < Nk; j++) lgndre_jet[j] = 0.;
  for (i = 0; i < Np; i++) {
    PetscReal mul = lgndre_coeffs[i];

    ierr = PetscDTIndexToGradedOrder(dim, i, degtup);CHKERRQ(ierr);
    for (j = 0; j < Nk; j++) {
      PetscReal val = 1.;

      ierr = PetscDTIndexToGradedOrder(dim, j, ktup);CHKERRQ(ierr);
      for (l = 0; l < dim; l++) {
        val *= D[l][degtup[l]*(k+1) + ktup[l]];
      }
      lgndre_jet[j] += mul * val;
    }
  }
  for (i = 0; i < dim; i++) {
    ierr = PetscFree(D[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(D);CHKERRQ(ierr);

  for (i = 0; i < Nk; i++) {
    PetscReal diff = lgndre_jet[i] - pkd_jet[i];
    PetscReal scale = 1. + PetscAbsReal(lgndre_jet[i]) + PetscAbsReal(pkd_jet[i]);
    PetscReal tol = 10. * PETSC_SMALL * scale;

    if (PetscAbsReal(diff) > tol) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Jet mismatch between PKD and tensor Legendre bases: error %g at tolerance %g", (double) diff, (double) tol);
  }

  ierr = PetscFree2(degtup,ktup);CHKERRQ(ierr);
  ierr = PetscFree(degrees);CHKERRQ(ierr);
  ierr = PetscFree3(pkd_jet_basis, lgndre_jet, pkd_jet);CHKERRQ(ierr);
  ierr = PetscFree(point);CHKERRQ(ierr);
  ierr = PetscFree2(lgndre_coeffs, pkd_coeffs);CHKERRQ(ierr);
  ierr = PetscFree(proj);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       dim, deg, k;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  dim = 3;
  deg = 4;
  k = 3;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for PetscDTPKDEval() tests","none");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "Dimension of the simplex","ex9.c",dim,&dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-degree", "The degree of the polynomial space","ex9.c",deg,&deg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k", "The number of derivatives to use in the taylor test","ex9.c",k,&k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = testOrthogonality(dim, deg);CHKERRQ(ierr);
  ierr = testDerivativesLegendre(dim, deg, k);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    args: -dim {{1 2 3 4}}

TEST*/
