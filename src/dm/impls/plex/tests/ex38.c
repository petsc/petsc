const char help[] = "Test DMPlexInsertBoundaryValues with DMPlexSetClosurePermutationTensor.\n";

#include <petscdmplex.h>

static PetscErrorCode bc_func(PetscInt dim, PetscReal time, const PetscReal coords[], PetscInt num_comp_u, PetscScalar *u, void *ctx)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < num_comp_u; i++) u[i] = coords[i];
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM        dm;
  PetscFE   fe;
  Vec       U_loc;
  PetscInt  dim, order = 1;
  PetscBool tensorCoords = PETSC_TRUE;

  /* Initialize PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-tensor_coords", &tensorCoords, NULL));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));

  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscFECreateLagrange(PETSC_COMM_WORLD, dim, dim, PETSC_FALSE, order, order, &fe));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));

  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  DMLabel  label;
  PetscInt marker_ids[] = {1};
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "mms", label, 1, marker_ids, 0, 0, NULL, (void (*)(void))bc_func, NULL, NULL, NULL));
  PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
  {
    DM cdm;
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    if (tensorCoords) PetscCall(DMPlexSetClosurePermutationTensor(cdm, PETSC_DETERMINE, NULL));
  }

  PetscCall(DMCreateLocalVector(dm, &U_loc));
  PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, U_loc, 1., NULL, NULL, NULL));
  PetscCall(VecViewFromOptions(U_loc, NULL, "-u_loc_vec_view"));
  PetscCall(VecDestroy(&U_loc));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 2d
    args: -dm_plex_simplex 0 -dm_plex_dim 2 -dm_plex_box_faces 3,3 -u_loc_vec_view
  test:
    suffix: 3d
    args: -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 3,3,3 -u_loc_vec_view
TEST*/
