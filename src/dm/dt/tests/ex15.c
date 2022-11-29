const char help[] = "Test PetscDTSimplexQuadrature()";

#include <petscdt.h>

// if we trust the PKD polynomials (tested in ex9), then we can see how well the quadrature integrates
// the mass matrix, which should be the identity
static PetscErrorCode testQuadrature(PetscInt dim, PetscInt degree, PetscDTSimplexQuadratureType type)
{
  PetscInt         num_points;
  const PetscReal *points;
  const PetscReal *weights;
  PetscInt         p_degree     = (degree + 1) / 2;
  PetscInt         p_degree_min = degree - p_degree;
  PetscInt         Nb, Nb_min;
  PetscReal       *eval;
  PetscQuadrature  quad;

  PetscFunctionBegin;
  PetscCall(PetscDTSimplexQuadrature(dim, degree, type, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &num_points, &points, &weights));
  PetscCall(PetscDTBinomialInt(dim + p_degree, dim, &Nb));
  PetscCall(PetscDTBinomialInt(dim + p_degree_min, dim, &Nb_min));
  PetscCall(PetscMalloc1(num_points * Nb, &eval));
  PetscCall(PetscDTPKDEvalJet(dim, num_points, points, p_degree, 0, eval));
  for (PetscInt i = 0; i < Nb_min; i++) {
    for (PetscInt j = i; j < Nb; j++) {
      PetscReal I_exact = (i == j) ? 1.0 : 0.0;
      PetscReal I_quad  = 0.0;
      PetscReal err;

      for (PetscInt q = 0; q < num_points; q++) I_quad += weights[q] * eval[i * num_points + q] * eval[j * num_points + q];
      err = PetscAbsReal(I_exact - I_quad);
      PetscCall(PetscInfo(quad, "Dimension %d, degree %d, method %s, error in <P_PKD(%d),P_PKD(%d)> = %g\n", (int)dim, (int)degree, PetscDTSimplexQuadratureTypes[type], (int)i, (int)j, (double)err));
      PetscCheck(err < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Dimension %d, degree %d, method %s, error in <P_PKD(%d),P_PKD(%d)> = %g", (int)dim, (int)degree, PetscDTSimplexQuadratureTypes[type], (int)i, (int)j, (double)err);
    }
  }
  PetscCall(PetscFree(eval));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  const PetscInt dimdeg[] = {0, 20, 20, 20};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  for (PetscInt dim = 0; dim <= 3; dim++) {
    for (PetscInt deg = 0; deg <= dimdeg[dim]; deg++) {
      const PetscDTSimplexQuadratureType types[] = {PETSCDTSIMPLEXQUAD_DEFAULT, PETSCDTSIMPLEXQUAD_CONIC, PETSCDTSIMPLEXQUAD_MINSYM};

      for (PetscInt t = 0; t < 3; t++) PetscCall(testQuadrature(dim, deg, types[t]));
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
