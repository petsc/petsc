
static char help[]= "Test PetscSFConcatenate()\n\n";

#include <petscsf.h>

typedef struct {
  MPI_Comm          comm;
  PetscMPIInt       rank, size;
  PetscInt          leaveStep, nsfs, nLeavesPerRank;
  PetscBool         shareRoots, sparseLeaves;
} AppCtx;

static PetscErrorCode GetOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscFunctionBegin;
  ctx->comm = comm;
  ctx->nsfs = 3;
  ctx->nLeavesPerRank = 4;
  ctx->leaveStep = 1;
  ctx->shareRoots = PETSC_FALSE;
  ctx->sparseLeaves = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-nsfs", &ctx->nsfs, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-n_leaves_per_rank", &ctx->nLeavesPerRank, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-leave_step", &ctx->leaveStep, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL, NULL, "-share_roots", &ctx->shareRoots, NULL));
  ctx->sparseLeaves = (PetscBool) (ctx->leaveStep != 1);
  CHKERRMPI(MPI_Comm_size(comm, &ctx->size));
  CHKERRMPI(MPI_Comm_rank(comm, &ctx->rank));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFCheckEqual_Private(PetscSF sf0, PetscSF sf1)
{
  PetscInt          nRoot, nLeave;
  Vec               vecRoot0, vecLeave0, vecRoot1, vecLeave1;
  MPI_Comm          comm;
  PetscBool         flg;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)sf0, &comm));
  CHKERRQ(PetscSFGetGraph(sf0, &nRoot, NULL, NULL, NULL));
  CHKERRQ(PetscSFGetLeafRange(sf0, NULL, &nLeave));
  nLeave++;
  CHKERRQ(VecCreateMPI(comm, nRoot, PETSC_DECIDE, &vecRoot0));
  CHKERRQ(VecCreateMPI(comm, nLeave, PETSC_DECIDE, &vecLeave0));
  CHKERRQ(VecDuplicate(vecRoot0, &vecRoot1));
  CHKERRQ(VecDuplicate(vecLeave0, &vecLeave1));
  {
    PetscRandom       rand;

    CHKERRQ(PetscRandomCreate(comm, &rand));
    CHKERRQ(PetscRandomSetFromOptions(rand));
    CHKERRQ(VecSetRandom(vecRoot0, rand));
    CHKERRQ(VecSetRandom(vecLeave0, rand));
    CHKERRQ(VecCopy(vecRoot0, vecRoot1));
    CHKERRQ(VecCopy(vecLeave0, vecLeave1));
    CHKERRQ(PetscRandomDestroy(&rand));
  }

  CHKERRQ(VecScatterBegin(sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(  sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD));
  CHKERRQ(VecScatterBegin(sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(  sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD));
  CHKERRQ(VecEqual(vecLeave0, vecLeave1, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "leave vectors differ");

  CHKERRQ(VecScatterBegin(sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(  sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(  sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE));
  CHKERRQ(VecEqual(vecRoot0, vecRoot1, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "root vectors differ");

  CHKERRQ(VecDestroy(&vecRoot0));
  CHKERRQ(VecDestroy(&vecRoot1));
  CHKERRQ(VecDestroy(&vecLeave0));
  CHKERRQ(VecDestroy(&vecLeave1));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateReferenceSF(AppCtx *ctx, PetscSF *refSF)
{
  PetscInt          i, j, k, r;
  PetscInt         *ilocal = NULL;
  PetscSFNode      *iremote;
  PetscInt          nLeaves = ctx->nsfs * ctx->nLeavesPerRank * ctx->size;
  PetscInt          nroots  = ctx->nLeavesPerRank * ctx->nsfs;
  PetscSF           sf;

  PetscFunctionBegin;
  ilocal = NULL;
  if (ctx->sparseLeaves) {
    CHKERRQ(PetscCalloc1(nLeaves+1, &ilocal));
  }
  CHKERRQ(PetscMalloc1(nLeaves, &iremote));
  CHKERRQ(PetscSFCreate(ctx->comm, &sf));
  for (i=0, j=0; i<ctx->nsfs; i++) {
    for (r=0; r<ctx->size; r++) {
      for (k=0; k<ctx->nLeavesPerRank; k++, j++) {
        if (ctx->sparseLeaves) {
          ilocal[j+1] = ilocal[j] + ctx->leaveStep;
        }
        iremote[j].rank = r;
        iremote[j].index = k + i * ctx->nLeavesPerRank;
      }
    }
  }
  CHKERRQ(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  *refSF = sf;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSFs(AppCtx *ctx, PetscSF *newSFs[], PetscInt *leafOffsets[])
{
  PetscInt          i;
  PetscInt         *lOffsets = NULL;
  PetscSF          *sfs;
  PetscInt          nLeaves = ctx->nLeavesPerRank * ctx->size;
  PetscInt          nroots  = ctx->shareRoots ? ctx->nLeavesPerRank * ctx->nsfs : ctx->nLeavesPerRank;

  PetscFunctionBegin;
  if (ctx->sparseLeaves) {
    CHKERRQ(PetscCalloc1(ctx->nsfs+1, &lOffsets));
  }
  CHKERRQ(PetscMalloc1(ctx->nsfs, &sfs));
  for (i=0; i<ctx->nsfs; i++) {
    PetscInt      j, k;
    PetscMPIInt   r;
    PetscInt     *ilocal = NULL;
    PetscSFNode  *iremote;

    if (ctx->sparseLeaves) {
      CHKERRQ(PetscCalloc1(nLeaves+1, &ilocal));
    }
    CHKERRQ(PetscMalloc1(nLeaves, &iremote));
    for (r=0, j=0; r<ctx->size; r++) {
      for (k=0; k<ctx->nLeavesPerRank; k++, j++) {
        if (ctx->sparseLeaves) {
          ilocal[j+1] = ilocal[j] + ctx->leaveStep;
        }
        iremote[j].rank = r;
        iremote[j].index = ctx->shareRoots ? k + i * ctx->nLeavesPerRank : k;
      }
    }
    if (ctx->sparseLeaves) lOffsets[i+1] = lOffsets[i] + ilocal[j];

    CHKERRQ(PetscSFCreate(ctx->comm, &sfs[i]));
    CHKERRQ(PetscSFSetGraph(sfs[i], nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  }
  *newSFs = sfs;
  *leafOffsets = lOffsets;
  PetscFunctionReturn(0);
}

PetscErrorCode DestroySFs(AppCtx *ctx, PetscSF *sfs[])
{
  PetscInt          i;

  PetscFunctionBegin;
  for (i=0; i<ctx->nsfs; i++) {
    CHKERRQ(PetscSFDestroy(&(*sfs)[i]));
  }
  CHKERRQ(PetscFree(*sfs));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx            ctx;
  PetscSF           sf, sfRef;
  PetscSF          *sfs;
  PetscInt         *leafOffsets;
  MPI_Comm          comm;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRQ(GetOptions(comm, &ctx));

  CHKERRQ(CreateSFs(&ctx, &sfs, &leafOffsets));
  CHKERRQ(PetscSFConcatenate(comm, ctx.nsfs, sfs, ctx.shareRoots, leafOffsets, &sf));
  CHKERRQ(CreateReferenceSF(&ctx, &sfRef));
  CHKERRQ(PetscSFCheckEqual_Private(sf, sfRef));

  CHKERRQ(DestroySFs(&ctx, &sfs));
  CHKERRQ(PetscFree(leafOffsets));
  CHKERRQ(PetscSFDestroy(&sf));
  CHKERRQ(PetscSFDestroy(&sfRef));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    nsize: {{1 3}}
    args: -nsfs {{1 3}} -n_leaves_per_rank {{0 1 5}} -leave_step {{1 3}} -share_roots {{true false}}
TEST*/
