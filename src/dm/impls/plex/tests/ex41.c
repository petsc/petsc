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
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = PetscCalloc1(size, &options->refcell);CHKERRQ(ierr);
  n    = size;

  ierr = PetscOptionsBegin(comm, "", "Parallel Mesh Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-metric", "Flag for metric refinement", "ex41.c", options->metric, &options->metric, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-refcell", "The cell to be refined", "ex41.c", options->refcell, &n, NULL);CHKERRQ(ierr);
  PetscCheckFalse(n && n != size,comm, PETSC_ERR_ARG_SIZ, "Only gave %D cells to refine, must give one for all %D processes", n, size);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAdaptLabel(DM dm, AppCtx *ctx, DMLabel *adaptLabel)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRMPI(ierr);
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", adaptLabel);CHKERRQ(ierr);
  if (ctx->refcell[rank] >= 0) {ierr = DMLabelSetValue(*adaptLabel, ctx->refcell[rank], DM_ADAPT_REFINE);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dma;
  DMLabel        adaptLabel;
  AppCtx         ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = CreateAdaptLabel(dm, &ctx, &adaptLabel);CHKERRQ(ierr);
  ierr = DMAdaptLabel(dm, adaptLabel, &dma);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dma, "Adapted Mesh");CHKERRQ(ierr);
  ierr = DMLabelDestroy(&adaptLabel);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dma, NULL, "-adapt_dm_view");CHKERRQ(ierr);
  ierr = DMDestroy(&dma);CHKERRQ(ierr);
  ierr = PetscFree(ctx.refcell);CHKERRQ(ierr);
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
