const char help[] = "Test boundary condition insertion";

#include <petscdmplex.h>

static PetscErrorCode set_one(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar bcval[], void *ctx)
{
  PetscFunctionBegin;
  bcval[0] = 1.;
  PetscFunctionReturn(0);
}

static PetscErrorCode set_two(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar bcval[], void *ctx)
{
  PetscFunctionBegin;
  bcval[0] = 2.;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM dm;
  DMLabel label;
  PetscInt in_value = 1;
  PetscInt out_value = 3;
  PetscInt comps[] = {0};
  PetscFE  fe;
  Vec      localVec;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, NULL, NULL, NULL, NULL, PETSC_TRUE, &dm));
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, 2, 1, PETSC_FALSE, 1, PETSC_DETERMINE, &fe));
  PetscCall(DMAddField(dm, NULL, (PetscObject) fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "inflow condition", label, 1, &in_value, 0, 1, comps, (void (*) (void)) set_one, NULL, NULL, NULL));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "outflow condition", label, 1, &out_value, 0, 1, comps, (void (*) (void)) set_two, NULL, NULL, NULL));
  PetscCall(DMCreateLocalVector(dm, &localVec));
  PetscCall(VecSet(localVec, 0.));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, localVec, 0.0, NULL, NULL, NULL));
  PetscCall(VecView(localVec, NULL));
  PetscCall(VecDestroy(&localVec));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
