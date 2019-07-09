static char help[] = "Tests for coarsening\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt  debug;          /* The debugging level */
  PetscInt  dim;            /* The topological mesh dimension */
  PetscInt  testNum;        /* The particular mesh to test */
  PetscBool uninterpolate;  /* Uninterpolate the mesh at the end */
  char      filename[2048]; /* The optional mesh file */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug          = 0;
  options->dim            = 2;
  options->testNum        = 0;
  options->uninterpolate  = PETSC_FALSE;
  options->filename[0]    = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-debug", "The debugging level", "ex14.c", options->debug, &options->debug, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex14.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-test_num", "The particular mesh to test", "ex14.c", options->testNum, &options->testNum, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-uninterpolate", "Uninterpolate the mesh at the end", "ex14.c", options->uninterpolate, &options->uninterpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "Filename to read", "ex14.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const char    *filename = user->filename;
  PetscInt       dim      = user->dim;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) {ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_TRUE, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);}
  else      {ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);}
  ierr = DMViewFromOptions(*dm, NULL, "-orig_dm_view");CHKERRQ(ierr);
  {
    DM distributedMesh = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
      ierr = DMViewFromOptions(*dm, NULL, "-dist_dm_view");CHKERRQ(ierr);
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  if (user->uninterpolate) {
    DM udm = NULL;

    ierr = DMPlexUninterpolate(*dm, &udm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = udm;
    ierr = DMViewFromOptions(*dm, NULL, "-un_dm_view");CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Test Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
    requires: triangle
    args: -dm_view -dm_refine 1 -dm_coarsen -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_faces

TEST*/
