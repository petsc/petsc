static const char help[] = "Tests for adaptive refinement";

#include <petscdmplex.h>

typedef struct {
  PetscBool metric;  /* Flag to use metric adaptation, instead of tagging */
  PetscInt *refcell; /* A cell to be refined on each process */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscMPIInt    size;
  PetscInt       n;

  PetscFunctionBeginUser;
  options->metric  = PETSC_FALSE;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscCalloc1(size, &options->refcell));
  n    = size;

  PetscOptionsBegin(comm, "", "Parallel Mesh Adaptation Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-metric", "Flag for metric refinement", "ex41.c", options->metric, &options->metric, NULL));
  PetscCall(PetscOptionsIntArray("-refcell", "The cell to be refined", "ex41.c", options->refcell, &n, NULL));
  PetscCheckFalse(n && n != size,comm, PETSC_ERR_ARG_SIZ, "Only gave %D cells to refine, must give one for all %D processes", n, size);
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAdaptLabel(DM dm, AppCtx *ctx, DMLabel *adaptLabel)
{
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", adaptLabel));
  if (ctx->refcell[rank] >= 0) PetscCall(DMLabelSetValue(*adaptLabel, ctx->refcell[rank], DM_ADAPT_REFINE));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dma;
  DMLabel        adaptLabel;
  AppCtx         ctx;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  PetscCall(CreateAdaptLabel(dm, &ctx, &adaptLabel));
  PetscCall(DMAdaptLabel(dm, adaptLabel, &dma));
  PetscCall(PetscObjectSetName((PetscObject) dma, "Adapted Mesh"));
  PetscCall(DMLabelDestroy(&adaptLabel));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMViewFromOptions(dma, NULL, "-adapt_dm_view"));
  PetscCall(DMDestroy(&dma));
  PetscCall(PetscFree(ctx.refcell));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_adaptor cellrefiner -dm_plex_transform_type refine_sbr

    test:
      suffix: 0
      requires: triangle
      args: -dm_view -adapt_dm_view

    test:
      suffix: 1
      requires: triangle
      args: -dm_coord_space 0 -refcell 2 -dm_view ::ascii_info_detail -adapt_dm_view ::ascii_info_detail

    test:
      suffix: 2
      requires: triangle
      nsize: 2
      args: -refcell 2,-1 -petscpartitioner_type simple -dm_view -adapt_dm_view

TEST*/
