
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ctx->comm = comm;
  ctx->nsfs = 3;
  ctx->nLeavesPerRank = 4;
  ctx->leaveStep = 1;
  ctx->shareRoots = PETSC_FALSE;
  ctx->sparseLeaves = PETSC_FALSE;
  ierr = PetscOptionsGetInt(NULL, NULL, "-nsfs", &ctx->nsfs, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-n_leaves_per_rank", &ctx->nLeavesPerRank, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-leave_step", &ctx->leaveStep, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-share_roots", &ctx->shareRoots, NULL);CHKERRQ(ierr);
  ctx->sparseLeaves = (PetscBool) (ctx->leaveStep != 1);
  ierr = MPI_Comm_size(comm, &ctx->size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &ctx->rank);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFCheckEqual_Private(PetscSF sf0, PetscSF sf1)
{
  PetscInt          nRoot, nLeave;
  Vec               vecRoot0, vecLeave0, vecRoot1, vecLeave1;
  MPI_Comm          comm;
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)sf0, &comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf0, &nRoot, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscSFGetLeafRange(sf0, NULL, &nLeave);CHKERRQ(ierr);
  nLeave++;
  ierr = VecCreateMPI(comm, nRoot, PETSC_DECIDE, &vecRoot0);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm, nLeave, PETSC_DECIDE, &vecLeave0);CHKERRQ(ierr);
  ierr = VecDuplicate(vecRoot0, &vecRoot1);CHKERRQ(ierr);
  ierr = VecDuplicate(vecLeave0, &vecLeave1);CHKERRQ(ierr);
  {
    PetscRandom       rand;

    ierr = PetscRandomCreate(comm, &rand);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
    ierr = VecSetRandom(vecRoot0, rand);CHKERRQ(ierr);
    ierr = VecSetRandom(vecLeave0, rand);CHKERRQ(ierr);
    ierr = VecCopy(vecRoot0, vecRoot1);CHKERRQ(ierr);
    ierr = VecCopy(vecLeave0, vecLeave1);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  }

  ierr = VecScatterBegin(sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(  sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(  sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecEqual(vecLeave0, vecLeave1, &flg);CHKERRQ(ierr);
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "leave vectors differ");

  ierr = VecScatterBegin(sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(  sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(  sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecEqual(vecRoot0, vecRoot1, &flg);CHKERRQ(ierr);
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "root vectors differ");

  ierr = VecDestroy(&vecRoot0);CHKERRQ(ierr);
  ierr = VecDestroy(&vecRoot1);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLeave0);CHKERRQ(ierr);
  ierr = VecDestroy(&vecLeave1);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ilocal = NULL;
  if (ctx->sparseLeaves) {
    ierr = PetscCalloc1(nLeaves+1, &ilocal);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(nLeaves, &iremote);CHKERRQ(ierr);
  ierr = PetscSFCreate(ctx->comm, &sf);CHKERRQ(ierr);
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
  ierr = PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (ctx->sparseLeaves) {
    ierr = PetscCalloc1(ctx->nsfs+1, &lOffsets);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(ctx->nsfs, &sfs);CHKERRQ(ierr);
  for (i=0; i<ctx->nsfs; i++) {
    PetscInt      j, k;
    PetscMPIInt   r;
    PetscInt     *ilocal = NULL;
    PetscSFNode  *iremote;

    if (ctx->sparseLeaves) {
      ierr = PetscCalloc1(nLeaves+1, &ilocal);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(nLeaves, &iremote);CHKERRQ(ierr);
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

    ierr = PetscSFCreate(ctx->comm, &sfs[i]);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfs[i], nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  *newSFs = sfs;
  *leafOffsets = lOffsets;
  PetscFunctionReturn(0);
}

PetscErrorCode DestroySFs(AppCtx *ctx, PetscSF *sfs[])
{
  PetscInt          i;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  for (i=0; i<ctx->nsfs; i++) {
    ierr = PetscSFDestroy(&(*sfs)[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(*sfs);CHKERRQ(ierr);
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
  ierr = GetOptions(comm, &ctx);CHKERRQ(ierr);

  ierr = CreateSFs(&ctx, &sfs, &leafOffsets);CHKERRQ(ierr);
  ierr = PetscSFConcatenate(comm, ctx.nsfs, sfs, ctx.shareRoots, leafOffsets, &sf);CHKERRQ(ierr);
  ierr = CreateReferenceSF(&ctx, &sfRef);CHKERRQ(ierr);
  ierr = PetscSFCheckEqual_Private(sf, sfRef);CHKERRQ(ierr);

  ierr = DestroySFs(&ctx, &sfs);CHKERRQ(ierr);
  ierr = PetscFree(leafOffsets);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfRef);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    nsize: {{1 3}}
    args: -nsfs {{1 3}} -n_leaves_per_rank {{0 1 5}} -leave_step {{1 3}} -share_roots {{true false}}
TEST*/
