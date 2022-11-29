static char help[] = "Tests for coarsening\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscBool uninterpolate; /* Uninterpolate the mesh at the end */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->uninterpolate = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-uninterpolate", "Uninterpolate the mesh at the end", "ex14.c", options->uninterpolate, &options->uninterpolate, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-orig_dm_view"));
  if (user->uninterpolate) {
    DM udm = NULL;

    PetscCall(DMPlexUninterpolate(*dm, &udm));
    PetscCall(DMDestroy(dm));
    *dm = udm;
    PetscCall(DMViewFromOptions(*dm, NULL, "-un_dm_view"));
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM     dm;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 0
    requires: triangle
    args: -dm_view -dm_refine 1 -dm_coarsen -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_faces

TEST*/
