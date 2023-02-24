
static char help[] = "Test PetscSFConcatenate()\n\n";

#include <petscsf.h>

typedef struct {
  MPI_Comm                   comm;
  PetscMPIInt                rank, size;
  PetscInt                   leaveStep, nsfs, n;
  PetscBool                  sparseLeaves;
  PetscBool                  compare;
  PetscBool                  irregular;
  PetscSFConcatenateRootMode rootMode;
  PetscViewer                viewer;
} AppCtx;

static PetscErrorCode GetOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  ctx->comm         = comm;
  ctx->nsfs         = 3;
  ctx->n            = 1;
  ctx->leaveStep    = 1;
  ctx->sparseLeaves = PETSC_FALSE;
  ctx->compare      = PETSC_FALSE;
  ctx->irregular    = PETSC_FALSE;
  ctx->rootMode     = PETSCSF_CONCATENATE_ROOTMODE_LOCAL;
  ctx->viewer       = NULL;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nsfs", &ctx->nsfs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &ctx->n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-leave_step", &ctx->leaveStep, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-irregular", &ctx->irregular, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-compare_to_reference", &ctx->compare, NULL));
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-root_mode", PetscSFConcatenateRootModes, (PetscEnum *)&ctx->rootMode, NULL));
  PetscCall(PetscOptionsGetViewer(comm, NULL, NULL, "-sf_view", &ctx->viewer, &format, NULL));
  if (ctx->viewer) PetscCall(PetscViewerPushFormat(ctx->viewer, format));
  ctx->sparseLeaves = (PetscBool)(ctx->leaveStep != 1);
  PetscCallMPI(MPI_Comm_size(comm, &ctx->size));
  PetscCallMPI(MPI_Comm_rank(comm, &ctx->rank));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFCheckEqual_Private(PetscSF sf0, PetscSF sf1)
{
  PetscInt  nRoot, nLeave;
  Vec       vecRoot0, vecLeave0, vecRoot1, vecLeave1;
  MPI_Comm  comm;
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sf0, &comm));
  PetscCall(PetscSFGetGraph(sf0, &nRoot, NULL, NULL, NULL));
  PetscCall(PetscSFGetLeafRange(sf0, NULL, &nLeave));
  nLeave++;
  PetscCall(VecCreateMPI(comm, nRoot, PETSC_DECIDE, &vecRoot0));
  PetscCall(VecCreateMPI(comm, nLeave, PETSC_DECIDE, &vecLeave0));
  PetscCall(VecDuplicate(vecRoot0, &vecRoot1));
  PetscCall(VecDuplicate(vecLeave0, &vecLeave1));
  {
    PetscRandom rand;

    PetscCall(PetscRandomCreate(comm, &rand));
    PetscCall(PetscRandomSetFromOptions(rand));
    PetscCall(VecSetRandom(vecRoot0, rand));
    PetscCall(VecSetRandom(vecLeave0, rand));
    PetscCall(VecCopy(vecRoot0, vecRoot1));
    PetscCall(VecCopy(vecLeave0, vecLeave1));
    PetscCall(PetscRandomDestroy(&rand));
  }

  PetscCall(VecScatterBegin(sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecEqual(vecLeave0, vecLeave1, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "leave vectors differ");

  PetscCall(VecScatterBegin(sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecEqual(vecRoot0, vecRoot1, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "root vectors differ");

  PetscCall(VecDestroy(&vecRoot0));
  PetscCall(VecDestroy(&vecRoot1));
  PetscCall(VecDestroy(&vecLeave0));
  PetscCall(VecDestroy(&vecLeave1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFViewCustom(PetscSF sf, PetscViewer viewer)
{
  PetscMPIInt        rank;
  PetscInt           i, nroots, nleaves, nranks;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  PetscLayout        rootLayout;
  PetscInt          *gremote;

  PetscFunctionBegin;
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote));
  PetscCall(PetscSFGetRootRanks(sf, &nranks, NULL, NULL, NULL, NULL));
  PetscCall(PetscSFGetGraphLayout(sf, &rootLayout, NULL, NULL, &gremote));
  PetscCheck(nroots == rootLayout->n, PetscObjectComm((PetscObject)sf), PETSC_ERR_PLIB, "Assertion failed: nroots == rootLayout->n");
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)sf, viewer));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf), &rank));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  if (rank == 0) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "rank #leaves #roots\n"));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%2d] %7" PetscInt_FMT " %6" PetscInt_FMT "\n", rank, nleaves, nroots));
  PetscCall(PetscViewerFlush(viewer));
  if (rank == 0) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "leaves      roots       roots in global numbering\n"));
  for (i = 0; i < nleaves; i++)
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "(%2d, %2" PetscInt_FMT ") <- (%2" PetscInt_FMT ", %2" PetscInt_FMT ")  = %3" PetscInt_FMT "\n", rank, ilocal ? ilocal[i] : i, iremote[i].rank, iremote[i].index, gremote[i]));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscCall(PetscLayoutDestroy(&rootLayout));
  PetscCall(PetscFree(gremote));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateReferenceSF_Regular(AppCtx *ctx, PetscSF *refSF)
{
  PetscInt  j;
  PetscInt *ilocal  = NULL;
  PetscInt  nLeaves = ctx->nsfs * ctx->n * ctx->size;
  PetscInt  nroots  = ctx->n * ctx->nsfs;
  PetscSF   sf;

  PetscFunctionBegin;
  ilocal = NULL;
  if (ctx->sparseLeaves) PetscCall(PetscCalloc1(nLeaves + 1, &ilocal));
  PetscCall(PetscSFCreate(ctx->comm, &sf));
  for (j = 0; j < nLeaves; j++) {
    if (ctx->sparseLeaves) ilocal[j + 1] = ilocal[j] + ctx->leaveStep;
  }
  switch (ctx->rootMode) {
  case PETSCSF_CONCATENATE_ROOTMODE_SHARED:
  case PETSCSF_CONCATENATE_ROOTMODE_LOCAL: {
    PetscInt     i, k;
    PetscMPIInt  r;
    PetscSFNode *iremote;

    PetscCall(PetscCalloc1(nLeaves, &iremote));
    for (i = 0, j = 0; i < ctx->nsfs; i++) {
      for (r = 0; r < ctx->size; r++) {
        for (k = 0; k < ctx->n; k++, j++) {
          iremote[j].rank  = r;
          iremote[j].index = k + i * ctx->n;
        }
      }
    }
    PetscCall(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  } break;
  case PETSCSF_CONCATENATE_ROOTMODE_GLOBAL: {
    PetscLayout map = NULL;
    PetscInt   *gremote;

    PetscCall(PetscLayoutCreateFromSizes(ctx->comm, nroots, PETSC_DECIDE, 1, &map));
    PetscCall(PetscMalloc1(nLeaves, &gremote));
    for (j = 0; j < nLeaves; j++) gremote[j] = j;
    PetscCall(PetscSFSetGraphLayout(sf, map, nLeaves, ilocal, PETSC_OWN_POINTER, gremote));
    PetscCall(PetscFree(gremote));
    PetscCall(PetscLayoutDestroy(&map));
  } break;
  default:
    SETERRQ(ctx->comm, PETSC_ERR_SUP, "unsupported rootmode %d", ctx->rootMode);
  }
  PetscCall(PetscObjectSetName((PetscObject)sf, "reference_sf"));
  if (ctx->viewer) PetscCall(PetscSFViewCustom(sf, ctx->viewer));
  *refSF = sf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateSFs_Irregular(AppCtx *ctx, PetscSF *newSFs[], PetscInt *leafOffsets[])
{
  PetscInt  i;
  PetscInt *lOffsets = NULL;
  PetscSF  *sfs;
  PetscInt  nLeaves = ctx->n * ctx->size + (ctx->size - 1) * ctx->size / 2;
  PetscInt  nroots  = ctx->n + ctx->rank + ctx->nsfs - 1 + ctx->size - 1;

  PetscFunctionBegin;
  if (ctx->sparseLeaves) PetscCall(PetscCalloc1(ctx->nsfs + 1, &lOffsets));
  PetscCall(PetscMalloc1(ctx->nsfs, &sfs));
  for (i = 0; i < ctx->nsfs; i++) {
    PetscSF      sf;
    PetscInt     j, k;
    PetscMPIInt  r;
    PetscInt    *ilocal = NULL;
    PetscSFNode *iremote;
    char         name[32];

    if (ctx->sparseLeaves) PetscCall(PetscCalloc1(nLeaves + 1, &ilocal));
    PetscCall(PetscMalloc1(nLeaves, &iremote));
    for (r = ctx->size - 1, j = 0; r >= 0; r--) {
      for (k = 0; k < ctx->n + r; k++, j++) {
        if (ctx->sparseLeaves) ilocal[j + 1] = ilocal[j] + ctx->leaveStep;
        iremote[j].rank  = r;
        iremote[j].index = k + i + ctx->rank;
      }
    }
    if (ctx->sparseLeaves) lOffsets[i + 1] = lOffsets[i] + ilocal[j];

    PetscCall(PetscSFCreate(ctx->comm, &sf));
    PetscCall(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    PetscCall(PetscSNPrintf(name, sizeof(name), "sf_%" PetscInt_FMT, i));
    PetscCall(PetscObjectSetName((PetscObject)sf, name));
    if (ctx->viewer) PetscCall(PetscSFViewCustom(sf, ctx->viewer));
    sfs[i] = sf;
  }
  *newSFs      = sfs;
  *leafOffsets = lOffsets;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateSFs_Regular(AppCtx *ctx, PetscSF *newSFs[], PetscInt *leafOffsets[])
{
  PetscInt                   i;
  PetscInt                  *lOffsets = NULL;
  PetscInt                   nLeaves  = ctx->n * ctx->size;
  PetscSF                   *sfs;
  PetscSFConcatenateRootMode mode = ctx->compare ? ctx->rootMode : PETSCSF_CONCATENATE_ROOTMODE_LOCAL;

  PetscFunctionBegin;
  if (ctx->sparseLeaves) PetscCall(PetscCalloc1(ctx->nsfs + 1, &lOffsets));
  PetscCall(PetscCalloc1(ctx->nsfs, &sfs));
  for (i = 0; i < ctx->nsfs; i++) {
    PetscSF   sf;
    PetscInt  j;
    PetscInt *ilocal = NULL;
    char      name[32];

    PetscCall(PetscSFCreate(ctx->comm, &sf));
    if (ctx->sparseLeaves) {
      PetscCall(PetscCalloc1(nLeaves + 1, &ilocal));
      for (j = 0; j < nLeaves; j++) ilocal[j + 1] = ilocal[j] + ctx->leaveStep;
      lOffsets[i + 1] = lOffsets[i] + ilocal[nLeaves];
    }
    switch (mode) {
    case PETSCSF_CONCATENATE_ROOTMODE_LOCAL: {
      PetscInt     k, nroots = ctx->n;
      PetscMPIInt  r;
      PetscSFNode *iremote;

      PetscCall(PetscMalloc1(nLeaves, &iremote));
      for (r = 0, j = 0; r < ctx->size; r++) {
        for (k = 0; k < ctx->n; k++, j++) {
          iremote[j].rank  = r;
          iremote[j].index = k;
        }
      }
      PetscCall(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    } break;
    case PETSCSF_CONCATENATE_ROOTMODE_SHARED: {
      PetscInt     k, nroots = ctx->n * ctx->nsfs;
      PetscMPIInt  r;
      PetscSFNode *iremote;

      PetscCall(PetscMalloc1(nLeaves, &iremote));
      for (r = 0, j = 0; r < ctx->size; r++) {
        for (k = 0; k < ctx->n; k++, j++) {
          iremote[j].rank  = r;
          iremote[j].index = k + i * ctx->n;
        }
      }
      PetscCall(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    } break;
    case PETSCSF_CONCATENATE_ROOTMODE_GLOBAL: {
      PetscInt    nroots = ctx->n;
      PetscLayout map    = NULL;
      PetscInt   *gremote;

      PetscCall(PetscLayoutCreateFromSizes(ctx->comm, nroots, PETSC_DECIDE, 1, &map));
      PetscCall(PetscMalloc1(nLeaves, &gremote));
      for (j = 0; j < nLeaves; j++) gremote[j] = j;
      PetscCall(PetscSFSetGraphLayout(sf, map, nLeaves, ilocal, PETSC_OWN_POINTER, gremote));
      PetscCall(PetscFree(gremote));
      PetscCall(PetscLayoutDestroy(&map));
    } break;
    default:
      SETERRQ(ctx->comm, PETSC_ERR_SUP, "unsupported rootmode %d", ctx->rootMode);
    }
    PetscCall(PetscSNPrintf(name, sizeof(name), "sf_%" PetscInt_FMT, i));
    PetscCall(PetscObjectSetName((PetscObject)sf, name));
    if (ctx->viewer) PetscCall(PetscSFViewCustom(sf, ctx->viewer));
    sfs[i] = sf;
  }
  *newSFs      = sfs;
  *leafOffsets = lOffsets;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DestroySFs(AppCtx *ctx, PetscSF *sfs[])
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < ctx->nsfs; i++) PetscCall(PetscSFDestroy(&(*sfs)[i]));
  PetscCall(PetscFree(*sfs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx    ctx_;
  AppCtx   *ctx = &ctx_;
  PetscSF   sf;
  PetscSF  *sfs         = NULL;
  PetscInt *leafOffsets = NULL;
  MPI_Comm  comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(GetOptions(comm, ctx));

  if (ctx->irregular) {
    PetscCall(CreateSFs_Irregular(ctx, &sfs, &leafOffsets));
  } else {
    PetscCall(CreateSFs_Regular(ctx, &sfs, &leafOffsets));
  }
  PetscCall(PetscSFConcatenate(comm, ctx->nsfs, sfs, ctx->rootMode, leafOffsets, &sf));
  PetscCall(PetscObjectSetName((PetscObject)sf, "result_sf"));
  if (ctx->viewer) {
    PetscCall(PetscPrintf(comm, "rootMode = %s:\n", PetscSFConcatenateRootModes[ctx->rootMode]));
    PetscCall(PetscSFViewCustom(sf, ctx->viewer));
  }
  if (ctx->compare) {
    PetscSF sfRef;

    PetscAssert(!ctx->irregular, comm, PETSC_ERR_SUP, "Combination  -compare_to_reference true -irregular true  not implemented");
    PetscCall(CreateReferenceSF_Regular(ctx, &sfRef));
    PetscCall(PetscSFCheckEqual_Private(sf, sfRef));
    PetscCall(PetscSFDestroy(&sfRef));
  }
  PetscCall(DestroySFs(ctx, &sfs));
  PetscCall(PetscFree(leafOffsets));
  PetscCall(PetscSFDestroy(&sf));
  if (ctx->viewer) {
    PetscCall(PetscViewerPopFormat(ctx->viewer));
    PetscCall(PetscViewerDestroy(&ctx->viewer));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    nsize: {{1 3}}
    args: -compare_to_reference -nsfs {{1 3}} -n {{0 1 5}} -leave_step {{1 3}} -root_mode {{local shared global}}

  test:
    suffix: 2
    nsize: 2
    args: -irregular {{false true}separate output} -sf_view -nsfs 3 -n 1 -leave_step {{1 3}separate output} -root_mode {{local shared global}separate output}
TEST*/
