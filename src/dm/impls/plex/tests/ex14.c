static char help[] = "Tests for coarsening\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscBool uninterpolate;  /* Uninterpolate the mesh at the end */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->uninterpolate = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-uninterpolate", "Uninterpolate the mesh at the end", "ex14.c", options->uninterpolate, &options->uninterpolate, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-orig_dm_view"));
  if (user->uninterpolate) {
    DM udm = NULL;

    CHKERRQ(DMPlexUninterpolate(*dm, &udm));
    CHKERRQ(DMDestroy(dm));
    *dm  = udm;
    CHKERRQ(DMViewFromOptions(*dm, NULL, "-un_dm_view"));
  }
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
    requires: triangle
    args: -dm_view -dm_refine 1 -dm_coarsen -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_faces

TEST*/
