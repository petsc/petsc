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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->metric  = PETSC_FALSE;
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(PetscCalloc1(size, &options->refcell));
  n    = size;

  ierr = PetscOptionsBegin(comm, "", "Parallel Mesh Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-metric", "Flag for metric refinement", "ex41.c", options->metric, &options->metric, NULL));
  CHKERRQ(PetscOptionsIntArray("-refcell", "The cell to be refined", "ex41.c", options->refcell, &n, NULL));
  PetscCheckFalse(n && n != size,comm, PETSC_ERR_ARG_SIZ, "Only gave %D cells to refine, must give one for all %D processes", n, size);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAdaptLabel(DM dm, AppCtx *ctx, DMLabel *adaptLabel)
{
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", adaptLabel));
  if (ctx->refcell[rank] >= 0) CHKERRQ(DMLabelSetValue(*adaptLabel, ctx->refcell[rank], DM_ADAPT_REFINE));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dma;
  DMLabel        adaptLabel;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  CHKERRQ(CreateAdaptLabel(dm, &ctx, &adaptLabel));
  CHKERRQ(DMAdaptLabel(dm, adaptLabel, &dma));
  CHKERRQ(PetscObjectSetName((PetscObject) dma, "Adapted Mesh"));
  CHKERRQ(DMLabelDestroy(&adaptLabel));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(DMViewFromOptions(dma, NULL, "-adapt_dm_view"));
  CHKERRQ(DMDestroy(&dma));
  CHKERRQ(PetscFree(ctx.refcell));
  ierr = PetscFinalize();
  return ierr;
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
