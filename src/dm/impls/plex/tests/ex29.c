static char help[] = "Test scalable partitioning on distributed meshes\n\n";

#include <petscdmplex.h>

enum {STAGE_LOAD, STAGE_DISTRIBUTE, STAGE_REFINE, STAGE_OVERLAP};

typedef struct {
  PetscLogEvent createMeshEvent;
  PetscLogStage stages[4];
  /* Domain and mesh definition */
  PetscInt overlap; /* The cell overlap to use during partitioning */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->overlap = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBoundedInt("-overlap", "The cell overlap for partitioning", "ex29.c", options->overlap, &options->overlap, NULL,0));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent));
  CHKERRQ(PetscLogStageRegister("MeshLoad",       &options->stages[STAGE_LOAD]));
  CHKERRQ(PetscLogStageRegister("MeshDistribute", &options->stages[STAGE_DISTRIBUTE]));
  CHKERRQ(PetscLogStageRegister("MeshRefine",     &options->stages[STAGE_REFINE]));
  CHKERRQ(PetscLogStageRegister("MeshOverlap",    &options->stages[STAGE_OVERLAP]));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(user->createMeshEvent,0,0,0,0));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(PetscLogStagePush(user->stages[STAGE_LOAD]));
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(PetscLogStagePop());
  {
    DM               pdm = NULL;
    PetscPartitioner part;

    CHKERRQ(DMPlexGetPartitioner(*dm, &part));
    CHKERRQ(PetscPartitionerSetFromOptions(part));
    /* Distribute mesh over processes */
    CHKERRQ(PetscLogStagePush(user->stages[STAGE_DISTRIBUTE]));
    CHKERRQ(DMPlexDistribute(*dm, 0, NULL, &pdm));
    if (pdm) {
      CHKERRQ(DMDestroy(dm));
      *dm  = pdm;
    }
    CHKERRQ(PetscLogStagePop());
  }
  CHKERRQ(PetscLogStagePush(user->stages[STAGE_REFINE]));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) *dm, "post_"));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) *dm, ""));
  CHKERRQ(PetscLogStagePop());
  if (user->overlap) {
    DM odm = NULL;
    /* Add the level-1 overlap to refined mesh */
    CHKERRQ(PetscLogStagePush(user->stages[STAGE_OVERLAP]));
    CHKERRQ(DMPlexDistributeOverlap(*dm, 1, NULL, &odm));
    if (odm) {
      CHKERRQ(DMView(odm, PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(DMDestroy(dm));
      *dm = odm;
    }
    CHKERRQ(PetscLogStagePop());
  }
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  CHKERRQ(PetscLogEventEnd(user->createMeshEvent,0,0,0,0));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, pdm;
  AppCtx         user;                 /* user-defined work context */
  PetscPartitioner part;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(DMPlexGetPartitioner(dm, &part));
  CHKERRQ(PetscPartitionerSetFromOptions(part));
  CHKERRQ(DMPlexDistribute(dm, user.overlap, NULL, &pdm));
  if (pdm) CHKERRQ(DMViewFromOptions(pdm, NULL, "-pdm_view"));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(DMDestroy(&pdm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: ctetgen
    args: -dm_plex_dim 3 -post_dm_refine 2 -petscpartitioner_type simple -dm_view
  test:
    suffix: 1
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -post_dm_refine 2 -petscpartitioner_type simple -dm_view
  test:
    suffix: quad_0
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -post_dm_refine 2 -petscpartitioner_type simple -dm_view -pdm_view

TEST*/
