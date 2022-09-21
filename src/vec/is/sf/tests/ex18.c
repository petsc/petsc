
static char help[] = "Test PetscSFConcatenate()\n\n";

#include <petscsf.h>

typedef struct {
  MPI_Comm    comm;
  PetscMPIInt rank, size;
  PetscInt    leaveStep, nsfs, nLeavesPerRank;
  PetscBool   shareRoots, sparseLeaves;
} AppCtx;

static PetscErrorCode GetOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscFunctionBegin;
  ctx->comm           = comm;
  ctx->nsfs           = 3;
  ctx->nLeavesPerRank = 4;
  ctx->leaveStep      = 1;
  ctx->shareRoots     = PETSC_FALSE;
  ctx->sparseLeaves   = PETSC_FALSE;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nsfs", &ctx->nsfs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_leaves_per_rank", &ctx->nLeavesPerRank, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-leave_step", &ctx->leaveStep, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-share_roots", &ctx->shareRoots, NULL));
  ctx->sparseLeaves = (PetscBool)(ctx->leaveStep != 1);
  PetscCallMPI(MPI_Comm_size(comm, &ctx->size));
  PetscCallMPI(MPI_Comm_rank(comm, &ctx->rank));
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

PetscErrorCode CreateReferenceSF(AppCtx *ctx, PetscSF *refSF)
{
  PetscInt     i, j, k, r;
  PetscInt    *ilocal = NULL;
  PetscSFNode *iremote;
  PetscInt     nLeaves = ctx->nsfs * ctx->nLeavesPerRank * ctx->size;
  PetscInt     nroots  = ctx->nLeavesPerRank * ctx->nsfs;
  PetscSF      sf;

  PetscFunctionBegin;
  ilocal = NULL;
  if (ctx->sparseLeaves) PetscCall(PetscCalloc1(nLeaves + 1, &ilocal));
  PetscCall(PetscMalloc1(nLeaves, &iremote));
  PetscCall(PetscSFCreate(ctx->comm, &sf));
  for (i = 0, j = 0; i < ctx->nsfs; i++) {
    for (r = 0; r < ctx->size; r++) {
      for (k = 0; k < ctx->nLeavesPerRank; k++, j++) {
        if (ctx->sparseLeaves) ilocal[j + 1] = ilocal[j] + ctx->leaveStep;
        iremote[j].rank  = r;
        iremote[j].index = k + i * ctx->nLeavesPerRank;
      }
    }
  }
  PetscCall(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  *refSF = sf;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSFs(AppCtx *ctx, PetscSF *newSFs[], PetscInt *leafOffsets[])
{
  PetscInt  i;
  PetscInt *lOffsets = NULL;
  PetscSF  *sfs;
  PetscInt  nLeaves = ctx->nLeavesPerRank * ctx->size;
  PetscInt  nroots  = ctx->shareRoots ? ctx->nLeavesPerRank * ctx->nsfs : ctx->nLeavesPerRank;

  PetscFunctionBegin;
  if (ctx->sparseLeaves) PetscCall(PetscCalloc1(ctx->nsfs + 1, &lOffsets));
  PetscCall(PetscMalloc1(ctx->nsfs, &sfs));
  for (i = 0; i < ctx->nsfs; i++) {
    PetscInt     j, k;
    PetscMPIInt  r;
    PetscInt    *ilocal = NULL;
    PetscSFNode *iremote;

    if (ctx->sparseLeaves) PetscCall(PetscCalloc1(nLeaves + 1, &ilocal));
    PetscCall(PetscMalloc1(nLeaves, &iremote));
    for (r = 0, j = 0; r < ctx->size; r++) {
      for (k = 0; k < ctx->nLeavesPerRank; k++, j++) {
        if (ctx->sparseLeaves) ilocal[j + 1] = ilocal[j] + ctx->leaveStep;
        iremote[j].rank  = r;
        iremote[j].index = ctx->shareRoots ? k + i * ctx->nLeavesPerRank : k;
      }
    }
    if (ctx->sparseLeaves) lOffsets[i + 1] = lOffsets[i] + ilocal[j];

    PetscCall(PetscSFCreate(ctx->comm, &sfs[i]));
    PetscCall(PetscSFSetGraph(sfs[i], nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  }
  *newSFs      = sfs;
  *leafOffsets = lOffsets;
  PetscFunctionReturn(0);
}

PetscErrorCode DestroySFs(AppCtx *ctx, PetscSF *sfs[])
{
  PetscInt i;

  PetscFunctionBegin;
  for (i = 0; i < ctx->nsfs; i++) PetscCall(PetscSFDestroy(&(*sfs)[i]));
  PetscCall(PetscFree(*sfs));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx    ctx;
  PetscSF   sf, sfRef;
  PetscSF  *sfs;
  PetscInt *leafOffsets;
  MPI_Comm  comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(GetOptions(comm, &ctx));

  PetscCall(CreateSFs(&ctx, &sfs, &leafOffsets));
  PetscCall(PetscSFConcatenate(comm, ctx.nsfs, sfs, ctx.shareRoots, leafOffsets, &sf));
  PetscCall(CreateReferenceSF(&ctx, &sfRef));
  PetscCall(PetscSFCheckEqual_Private(sf, sfRef));

  PetscCall(DestroySFs(&ctx, &sfs));
  PetscCall(PetscFree(leafOffsets));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscSFDestroy(&sfRef));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    nsize: {{1 3}}
    args: -nsfs {{1 3}} -n_leaves_per_rank {{0 1 5}} -leave_step {{1 3}} -share_roots {{true false}}
TEST*/
