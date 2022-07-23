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
  PetscFunctionBegin;
  options->overlap = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-overlap", "The cell overlap for partitioning", "ex29.c", options->overlap, &options->overlap, NULL,0));
  PetscOptionsEnd();

  PetscCall(PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent));
  PetscCall(PetscLogStageRegister("MeshLoad",       &options->stages[STAGE_LOAD]));
  PetscCall(PetscLogStageRegister("MeshDistribute", &options->stages[STAGE_DISTRIBUTE]));
  PetscCall(PetscLogStageRegister("MeshRefine",     &options->stages[STAGE_REFINE]));
  PetscCall(PetscLogStageRegister("MeshOverlap",    &options->stages[STAGE_OVERLAP]));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscMPIInt    rank, size;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(user->createMeshEvent,0,0,0,0));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscLogStagePush(user->stages[STAGE_LOAD]));
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscLogStagePop());
  {
    DM               pdm = NULL;
    PetscPartitioner part;

    PetscCall(DMPlexGetPartitioner(*dm, &part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    /* Distribute mesh over processes */
    PetscCall(PetscLogStagePush(user->stages[STAGE_DISTRIBUTE]));
    PetscCall(DMPlexDistribute(*dm, 0, NULL, &pdm));
    if (pdm) {
      PetscCall(DMDestroy(dm));
      *dm  = pdm;
    }
    PetscCall(PetscLogStagePop());
  }
  PetscCall(PetscLogStagePush(user->stages[STAGE_REFINE]));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *dm, "post_"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *dm, ""));
  PetscCall(PetscLogStagePop());
  if (user->overlap) {
    DM odm = NULL;
    /* Add the level-1 overlap to refined mesh */
    PetscCall(PetscLogStagePush(user->stages[STAGE_OVERLAP]));
    PetscCall(DMPlexDistributeOverlap(*dm, 1, NULL, &odm));
    if (odm) {
      PetscCall(DMView(odm, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(DMDestroy(dm));
      *dm = odm;
    }
    PetscCall(PetscLogStagePop());
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscCall(PetscLogEventEnd(user->createMeshEvent,0,0,0,0));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, pdm;
  AppCtx         user;                 /* user-defined work context */
  PetscPartitioner part;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(DMPlexGetPartitioner(dm, &part));
  PetscCall(PetscPartitionerSetFromOptions(part));
  PetscCall(DMPlexDistribute(dm, user.overlap, NULL, &pdm));
  if (pdm) PetscCall(DMViewFromOptions(pdm, NULL, "-pdm_view"));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&pdm));
  PetscCall(PetscFinalize());
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
