static const char help[] = "Tests for mesh extrusion";

#include <petscdmplex.h>

typedef struct {
  char     bdLabel[PETSC_MAX_PATH_LEN]; /* The boundary label name */
  PetscInt Nbd;                         /* The number of boundary markers to extrude, 0 for all */
  PetscInt bd[64];                      /* The boundary markers to be extruded */
} AppCtx;

PETSC_EXTERN PetscErrorCode pyramidNormal(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);

/* The pyramid apex is at (0.5, 0.5, -1) */
PetscErrorCode pyramidNormal(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt r, PetscScalar u[], void *ctx)
{
  PetscReal apex[3] = {0.5, 0.5, -1.0};
  PetscInt  d;

  for (d = 0; d < dim; ++d) u[d] = x[d] - apex[d];
  for (d = dim; d < 3; ++d) u[d] = 0.0 - apex[d];
  return PETSC_SUCCESS;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt  n = 64;
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscCall(PetscStrncpy(options->bdLabel, "marker", sizeof(options->bdLabel)));
  PetscOptionsBegin(comm, "", "Parallel Mesh Adaptation Options", "DMPLEX");
  PetscCall(PetscOptionsString("-label", "The boundary label name", "ex44.c", options->bdLabel, options->bdLabel, sizeof(options->bdLabel), NULL));
  PetscCall(PetscOptionsIntArray("-bd", "The boundaries to be extruded", "ex44.c", options->bd, &n, &flg));
  options->Nbd = flg ? n : 0;
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateAdaptLabel(DM dm, AppCtx *ctx, DMLabel *adaptLabel)
{
  DMLabel  label;
  PetscInt b;

  PetscFunctionBegin;
  if (!ctx->Nbd) {
    *adaptLabel = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMGetLabel(dm, ctx->bdLabel, &label));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", adaptLabel));
  for (b = 0; b < ctx->Nbd; ++b) {
    IS              bdIS;
    const PetscInt *points;
    PetscInt        n, i;

    PetscCall(DMLabelGetStratumIS(label, ctx->bd[b], &bdIS));
    if (!bdIS) continue;
    PetscCall(ISGetLocalSize(bdIS, &n));
    PetscCall(ISGetIndices(bdIS, &points));
    for (i = 0; i < n; ++i) PetscCall(DMLabelSetValue(*adaptLabel, points[i], DM_ADAPT_REFINE));
    PetscCall(ISRestoreIndices(bdIS, &points));
    PetscCall(ISDestroy(&bdIS));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM      dm, dma;
  DMLabel adaptLabel;
  AppCtx  ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &ctx, &dm));
  PetscCall(CreateAdaptLabel(dm, &ctx, &adaptLabel));
  if (adaptLabel) {
    PetscCall(DMAdaptLabel(dm, adaptLabel, &dma));
  } else {
    PetscCall(DMExtrude(dm, 3, &dma));
  }
  PetscCall(PetscObjectSetName((PetscObject)dma, "Adapted Mesh"));
  PetscCall(DMLabelDestroy(&adaptLabel));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMViewFromOptions(dma, NULL, "-adapt_dm_view"));
  PetscCall(DMDestroy(&dma));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: tri_tensor_0
    requires: triangle
    args: -dm_plex_transform_extrude_use_tensor {{0 1}separate output} \
          -dm_view -adapt_dm_view -dm_plex_check_all

  test:
    suffix: quad_tensor_0
    args: -dm_plex_simplex 0 -dm_plex_transform_extrude_use_tensor {{0 1}separate output} \
          -dm_view -adapt_dm_view -dm_plex_check_all

  test:
    suffix: quad_normal_0
    args: -dm_plex_simplex 0 -dm_plex_transform_extrude_normal 0,1,1 \
          -dm_view -adapt_dm_view -dm_plex_check_all

  test:
    suffix: quad_normal_1
    args: -dm_plex_simplex 0 -dm_plex_transform_extrude_normal_function pyramidNormal \
          -dm_view -adapt_dm_view -dm_plex_check_all

  test:
    suffix: quad_symmetric_0
    args: -dm_plex_simplex 0 -dm_plex_transform_extrude_symmetric \
          -dm_view -adapt_dm_view -dm_plex_check_all

  testset:
    args: -dm_adaptor cellrefiner -dm_plex_transform_type extrude \
          -dm_view -adapt_dm_view

    test:
      suffix: quad_adapt_0
      args: -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_plex_separate_marker -bd 1,3 \
            -dm_plex_transform_extrude_thickness 0.5

    test:
      suffix: tet_adapt_0
      requires: ctetgen
      args: -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -dm_plex_separate_marker -bd 1,3 \
            -dm_plex_transform_extrude_thickness 0.5

    test:
      suffix: hex_adapt_0
      args: -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -dm_plex_separate_marker -bd 1,3 \
            -dm_plex_transform_extrude_thickness 0.5

TEST*/
