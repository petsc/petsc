
static char help[]= "Test leaf sorting in PetscSFSetGraph()\n\n";

#include <petscsf.h>

typedef struct {
  MPI_Comm          comm;
  PetscMPIInt       rank, size;
  PetscInt          leaveStep, nLeavesPerRank;
  PetscBool         contiguousLeaves;
  PetscCopyMode     localmode, remotemode;
  PetscInt         *ilocal;
  PetscSFNode      *iremote;
} AppCtx;

static PetscErrorCode GetOptions(MPI_Comm comm, AppCtx *ctx)
{
  PetscFunctionBegin;
  ctx->comm = comm;
  ctx->nLeavesPerRank = 4;
  ctx->leaveStep = 1;
  ctx->contiguousLeaves = PETSC_FALSE;
  ctx->localmode = PETSC_OWN_POINTER;
  ctx->remotemode = PETSC_OWN_POINTER;
  ctx->ilocal = NULL;
  ctx->iremote = NULL;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_leaves_per_rank", &ctx->nLeavesPerRank, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-leave_step", &ctx->leaveStep, NULL));
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-localmode", PetscCopyModes, (PetscEnum*) &ctx->localmode, NULL));
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-remotemode", PetscCopyModes, (PetscEnum*) &ctx->remotemode, NULL));
  ctx->contiguousLeaves = (PetscBool) (ctx->leaveStep == 1);
  PetscCallMPI(MPI_Comm_size(comm, &ctx->size));
  PetscCallMPI(MPI_Comm_rank(comm, &ctx->rank));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFCheckEqual_Private(PetscSF sf0, PetscSF sf1)
{
  PetscInt          nRoot, nLeave;
  Vec               vecRoot0, vecLeave0, vecRoot1, vecLeave1;
  MPI_Comm          comm;
  PetscBool         flg;

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
    PetscRandom       rand;

    PetscCall(PetscRandomCreate(comm, &rand));
    PetscCall(PetscRandomSetFromOptions(rand));
    PetscCall(VecSetRandom(vecRoot0, rand));
    PetscCall(VecSetRandom(vecLeave0, rand));
    PetscCall(VecCopy(vecRoot0, vecRoot1));
    PetscCall(VecCopy(vecLeave0, vecLeave1));
    PetscCall(PetscRandomDestroy(&rand));
  }

  PetscCall(VecScatterBegin(sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(  sf0, vecRoot0, vecLeave0, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(  sf1, vecRoot1, vecLeave1, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecEqual(vecLeave0, vecLeave1, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "leave vectors differ");

  PetscCall(VecScatterBegin(sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(  sf0, vecLeave0, vecRoot0, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(  sf1, vecLeave1, vecRoot1, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecEqual(vecRoot0, vecRoot1, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "root vectors differ");

  PetscCall(VecDestroy(&vecRoot0));
  PetscCall(VecDestroy(&vecRoot1));
  PetscCall(VecDestroy(&vecLeave0));
  PetscCall(VecDestroy(&vecLeave1));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSF0(AppCtx *ctx, PetscSF *sf0)
{
  PetscInt          j, k, r;
  PetscInt          nLeaves = ctx->nLeavesPerRank * ctx->size;
  PetscInt          nroots  = ctx->nLeavesPerRank;
  PetscSF           sf;
  PetscInt         *ilocal;
  PetscSFNode      *iremote;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(nLeaves+1, &ctx->ilocal));
  PetscCall(PetscMalloc1(nLeaves, &ctx->iremote));
  ilocal = ctx->ilocal;
  iremote = ctx->iremote;
  ilocal[nLeaves] = -ctx->leaveStep;
  PetscCall(PetscSFCreate(ctx->comm, &sf));
  for (r=0, j=nLeaves-1; r<ctx->size; r++) {
    for (k=0; k<ctx->nLeavesPerRank; k++, j--) {
      ilocal[j] = ilocal[j+1] + ctx->leaveStep;
      iremote[j].rank = r;
      iremote[j].index = k;
    }
  }
  PetscCall(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, ctx->localmode, iremote, ctx->remotemode));
  {
    const PetscInt *tlocal;
    PetscBool       sorted;

    PetscCall(PetscSFGetGraph(sf, NULL, NULL, &tlocal, NULL));
    PetscCheck(!ctx->contiguousLeaves || !tlocal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ilocal=NULL expected for contiguous case");
    if (tlocal) {
      PetscCall(PetscSortedInt(nLeaves, tlocal, &sorted));
      PetscCheck(sorted,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ilocal expected to be sorted");
    }
  }
  *sf0 = sf;
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSF1(AppCtx *ctx, PetscSF *sf1)
{
  PetscInt          j, k, r;
  PetscInt         *ilocal = NULL;
  PetscSFNode      *iremote;
  PetscInt          nLeaves = ctx->nLeavesPerRank * ctx->size;
  PetscInt          nroots  = ctx->nLeavesPerRank;
  PetscSF           sf;

  PetscFunctionBegin;
  ilocal = NULL;
  if (!ctx->contiguousLeaves) {
    PetscCall(PetscCalloc1(nLeaves+1, &ilocal));
  }
  PetscCall(PetscMalloc1(nLeaves, &iremote));
  PetscCall(PetscSFCreate(ctx->comm, &sf));
  for (r=0, j=0; r<ctx->size; r++) {
    for (k=0; k<ctx->nLeavesPerRank; k++, j++) {
      if (!ctx->contiguousLeaves) {
        ilocal[j+1] = ilocal[j] + ctx->leaveStep;
      }
      iremote[j].rank = r;
      iremote[j].index = k;
    }
  }
  PetscCheck(j == nLeaves,PETSC_COMM_SELF,PETSC_ERR_PLIB,"j != nLeaves");
  PetscCall(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  if (ctx->contiguousLeaves) {
    const PetscInt *tlocal;

    PetscCall(PetscSFGetGraph(sf, NULL, NULL, &tlocal, NULL));
    PetscCheck(!tlocal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ilocal=NULL expected for contiguous case");
  }
  *sf1 = sf;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx   ctx;
  PetscSF  sf0, sf1;
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  PetscCall(GetOptions(comm, &ctx));

  PetscCall(CreateSF0(&ctx, &sf0));
  PetscCall(CreateSF1(&ctx, &sf1));
  PetscCall(PetscSFViewFromOptions(sf0, NULL, "-sf0_view"));
  PetscCall(PetscSFViewFromOptions(sf1, NULL, "-sf1_view"));
  PetscCall(PetscSFCheckEqual_Private(sf0, sf1));

  if (ctx.localmode != PETSC_OWN_POINTER)  PetscCall(PetscFree(ctx.ilocal));
  if (ctx.remotemode != PETSC_OWN_POINTER) PetscCall(PetscFree(ctx.iremote));
  PetscCall(PetscSFDestroy(&sf0));
  PetscCall(PetscSFDestroy(&sf1));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    suffix: 1
    nsize: {{1 3}}
    args: -n_leaves_per_rank {{0 5}} -leave_step {{1 3}}
    test:
      suffix: a
      args: -localmode {{COPY_VALUES OWN_POINTER}} -remotemode {{COPY_VALUES OWN_POINTER}}
    test:
      suffix: b
      args: -localmode USE_POINTER -remotemode {{COPY_VALUES OWN_POINTER USE_POINTER}}
TEST*/
