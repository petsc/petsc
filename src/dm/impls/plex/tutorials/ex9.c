static char help[] = "Evaluate the shape quality of a mesh\n\n";

#include <petscdmplex.h>

typedef struct {
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscBool report;                       /* Print a quality report */
  PetscReal condLimit, tol;               /* Condition number limit for cell output */
  PetscBool interpolate, distribute, simplex;
  PetscInt  dim, overlap;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->filename[0] = '\0';
  options->report      = PETSC_FALSE;
  options->interpolate = PETSC_FALSE;
  options->distribute  = PETSC_FALSE;
  options->simplex     = PETSC_FALSE;
  options->overlap     = 0;
  options->dim         = 2;
  options->tol         = 0.5;
  options->condLimit   = PETSC_DETERMINE;

  ierr = PetscOptionsBegin(comm, "", "Mesh Quality Evaluation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex9.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-report", "Output a mesh quality report", "ex9.c", options->report, &options->report, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cond_limit", "Condition number limit for cell output", "ex9.c", options->condLimit, &options->condLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-orth_qual_atol", "Absolute tolerance for Orthogonal Quality", "ex9.c", options->tol, &options->tol, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate mesh", "ex9.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute", "Distribute mesh", "ex9.c", options->distribute, &options->distribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Create simplex mesh elements", "ex9.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-overlap", "Number of overlap levels for dm", "ex9.c", options->overlap, &options->overlap, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "Dimension of mesh if generated", "ex9.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  if (user->filename[0]) {
    ierr = DMPlexCreateFromFile(comm, user->filename, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, NULL, NULL, NULL, NULL, user->interpolate, dm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  }
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  if (user->distribute) {
    DM  dmDist = NULL;
    ierr = DMPlexDistribute(*dm, user->overlap, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = dmDist;
    }
  }
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  DMLabel        OQLabel;
  Vec            OQ;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = DMPlexCheckCellShape(dm, ctx.report, ctx.condLimit);CHKERRQ(ierr);
  ierr = DMPlexComputeOrthogonalQuality(dm, NULL, ctx.tol, &OQ, &OQLabel);CHKERRQ(ierr);
  ierr = VecDestroy(&OQ);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: exodusii
    nsize: {{1 2}}
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -report

  test:
    suffix: 1
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.msh -report

  testset:
    requires: exodusii
    args: -dm_plex_orthogonal_quality_label_view -dm_plex_orthogonal_quality_vec_view

    test:
      suffix: box_1
      nsize: 1
      args: -interpolate -distribute -dim 2 -dm_plex_box_faces 2,2 -orth_qual_atol 1.0

    test:
      suffix: box_2
      nsize: 2
      args: -interpolate -distribute -dim 2 -dm_plex_box_faces 2,2 -orth_qual_atol 1.0

    test:
      suffix: mesh_1
      nsize: 1
      args: -interpolate -distribute -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -orth_qual_atol 0.95

    test:
      suffix: mesh_2
      nsize: 2
      args: -interpolate -distribute -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -orth_qual_atol 0.95
TEST*/
