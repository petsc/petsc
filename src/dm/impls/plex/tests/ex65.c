static const char help[] = "Tests for mesh transformation using only options";

#include <petscdmplex.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));

  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, "phase_1_"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*dm, NULL));

  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM dm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # This verifies the correctness of an extruded coordinate space
  test:
    suffix: ext_coord_space
    args: -dm_plex_dim 1 -dm_plex_box_faces 1 -phase_1_dm_extrude 1 -phase_1_dm_plex_transform_extrude_use_tensor 0 -cdm_dm_petscds_view

TEST*/
