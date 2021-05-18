static char help[] = "Evaluate the shape quality of a mesh\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscBool report;         /* Print a quality report */
  PetscReal condLimit, tol; /* Condition number limit for cell output */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->report      = PETSC_FALSE;
  options->tol         = 0.5;
  options->condLimit   = PETSC_DETERMINE;

  ierr = PetscOptionsBegin(comm, "", "Mesh Quality Evaluation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-report", "Output a mesh quality report", "ex9.c", options->report, &options->report, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cond_limit", "Condition number limit for cell output", "ex9.c", options->condLimit, &options->condLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-orth_qual_atol", "Absolute tolerance for Orthogonal Quality", "ex9.c", options->tol, &options->tol, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
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
    args: -dm_distribute -petscpartitioner_type simple -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -report

  test:
    suffix: 1
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.msh -report

  testset:
    requires: exodusii
    args: -dm_plex_orthogonal_quality_label_view -dm_plex_orthogonal_quality_vec_view

    test:
      suffix: box_1
      nsize: 1
      args: -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -orth_qual_atol 1.0

    test:
      suffix: box_2
      nsize: 2
      args: -dm_distribute -petscpartitioner_type simple -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -orth_qual_atol 1.0

    test:
      suffix: mesh_1
      nsize: 1
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -orth_qual_atol 0.95

    test:
      suffix: mesh_2
      nsize: 2
      args: -dm_distribute -petscpartitioner_type simple -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -orth_qual_atol 0.95
TEST*/
