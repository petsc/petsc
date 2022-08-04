static char help[] = "Evaluate the shape quality of a mesh\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscBool report;         /* Print a quality report */
  PetscReal condLimit, tol; /* Condition number limit for cell output */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->report      = PETSC_FALSE;
  options->tol         = 0.5;
  options->condLimit   = PETSC_DETERMINE;

  PetscOptionsBegin(comm, "", "Mesh Quality Evaluation Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-report", "Output a mesh quality report", "ex9.c", options->report, &options->report, NULL));
  PetscCall(PetscOptionsReal("-cond_limit", "Condition number limit for cell output", "ex9.c", options->condLimit, &options->condLimit, NULL));
  PetscCall(PetscOptionsReal("-orth_qual_atol", "Absolute tolerance for Orthogonal Quality", "ex9.c", options->tol, &options->tol, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  DMLabel        OQLabel;
  Vec            OQ;
  AppCtx         ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMPlexCheckCellShape(dm, ctx.report, ctx.condLimit));
  PetscCall(DMPlexComputeOrthogonalQuality(dm, NULL, ctx.tol, &OQ, &OQLabel));
  PetscCall(VecDestroy(&OQ));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
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
