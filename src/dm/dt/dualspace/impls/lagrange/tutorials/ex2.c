const char help[] = "Simple example to get equally space points in high-order elements (and XGC mirror)";

#include <petscfe.h>
#include <petscdmplex.h>
static PetscErrorCode x(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf_dummy, PetscScalar *u, void *actx)
{
  PetscFunctionBegin;
  u[0] = x[0];
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt       dim = 2, cells[] = {1, 1, 1};
  DM             K;
  PetscReal      radius = 2, lo[] = {-radius, -radius, -radius}, hi[] = {radius, radius, radius};
  DMBoundaryType periodicity[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscFE        fe;
  PetscErrorCode (*initu[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
  Vec X;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for PETSCDUALSPACELAGRANGE test", "none");
  PetscCall(PetscOptionsRangeInt("-dim", "The spatial dimension", "ex1.c", dim, &dim, NULL, 0, 3));
  PetscOptionsEnd();

  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_SELF, dim, PETSC_FALSE, cells, lo, hi, periodicity, PETSC_TRUE, &K));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, PETSC_DECIDE, &fe));
  PetscCall(DMSetField(K, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(K));

  initu[0] = x;
  PetscCall(DMCreateGlobalVector(K, &X));
  PetscCall(DMProjectFunction(K, 0.0, initu, NULL, INSERT_ALL_VALUES, X));
  PetscCall(DMViewFromOptions(K, NULL, "-dual_dm_view"));
  PetscCall(VecViewFromOptions(X, NULL, "-dual_vec_view"));
  PetscCall(DMDestroy(&K));
  PetscCall(VecDestroy(&X));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    filter: grep -v DM_
    diff_args: -j
    args: -petscspace_degree 4 -petscdualspace_lagrange_node_type equispaced -petscdualspace_lagrange_node_endpoints 1 -dual_dm_view -dual_vec_view
    test:
      requires: !single
      suffix: 0
    test:
      requires: !single
      suffix: 3d
      args: -dim 3

TEST*/
