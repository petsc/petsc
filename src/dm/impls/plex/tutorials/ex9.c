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
  CHKERRQ(PetscOptionsBool("-report", "Output a mesh quality report", "ex9.c", options->report, &options->report, NULL));
  CHKERRQ(PetscOptionsReal("-cond_limit", "Condition number limit for cell output", "ex9.c", options->condLimit, &options->condLimit, NULL));
  CHKERRQ(PetscOptionsReal("-orth_qual_atol", "Absolute tolerance for Orthogonal Quality", "ex9.c", options->tol, &options->tol, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  DMLabel        OQLabel;
  Vec            OQ;
  AppCtx         ctx;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  CHKERRQ(DMPlexCheckCellShape(dm, ctx.report, ctx.condLimit));
  CHKERRQ(DMPlexComputeOrthogonalQuality(dm, NULL, ctx.tol, &OQ, &OQLabel));
  CHKERRQ(VecDestroy(&OQ));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: exodusii
    nsize: {{1 2}}
    args: -petscpartitioner_type simple -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -report

  test:
    suffix: 1
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.msh -report

  testset:
    args: -dm_plex_orthogonal_quality_label_view -dm_plex_orthogonal_quality_vec_view

    test:
      suffix: box_1
      nsize: 1
      args: -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -orth_qual_atol 1.0

    test:
      suffix: box_2
      nsize: 2
      args: -petscpartitioner_type simple -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -orth_qual_atol 1.0

    test:
      suffix: mesh_1
      nsize: 1
      requires: exodusii
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -orth_qual_atol 0.95

    test:
      suffix: mesh_2
      nsize: 2
      requires: exodusii
      args: -petscpartitioner_type simple -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -orth_qual_atol 0.95
TEST*/
