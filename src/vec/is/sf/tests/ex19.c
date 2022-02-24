
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
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-n_leaves_per_rank", &ctx->nLeavesPerRank, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL, NULL, "-leave_step", &ctx->leaveStep, NULL));
  CHKERRQ(PetscOptionsGetEnum(NULL, NULL, "-localmode", PetscCopyModes, (PetscEnum*) &ctx->localmode, NULL));
  CHKERRQ(PetscOptionsGetEnum(NULL, NULL, "-remotemode", PetscCopyModes, (PetscEnum*) &ctx->remotemode, NULL));
  ctx->contiguousLeaves = (PetscBool) (ctx->leaveStep == 1);
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

PetscErrorCode CreateSF0(AppCtx *ctx, PetscSF *sf0)
{
  PetscInt          j, k, r;
  PetscInt          nLeaves = ctx->nLeavesPerRank * ctx->size;
  PetscInt          nroots  = ctx->nLeavesPerRank;
  PetscSF           sf;
  PetscInt         *ilocal;
  PetscSFNode      *iremote;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(nLeaves+1, &ctx->ilocal));
  CHKERRQ(PetscMalloc1(nLeaves, &ctx->iremote));
  ilocal = ctx->ilocal;
  iremote = ctx->iremote;
  ilocal[nLeaves] = -ctx->leaveStep;
  CHKERRQ(PetscSFCreate(ctx->comm, &sf));
  for (r=0, j=nLeaves-1; r<ctx->size; r++) {
    for (k=0; k<ctx->nLeavesPerRank; k++, j--) {
      ilocal[j] = ilocal[j+1] + ctx->leaveStep;
      iremote[j].rank = r;
      iremote[j].index = k;
    }
  }
  CHKERRQ(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, ctx->localmode, iremote, ctx->remotemode));
  {
    const PetscInt *tlocal;
    PetscBool       sorted;

    CHKERRQ(PetscSFGetGraph(sf, NULL, NULL, &tlocal, NULL));
    PetscCheckFalse(ctx->contiguousLeaves && tlocal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ilocal=NULL expected for contiguous case");
    if (tlocal) {
      CHKERRQ(PetscSortedInt(nLeaves, tlocal, &sorted));
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
    CHKERRQ(PetscCalloc1(nLeaves+1, &ilocal));
  }
  CHKERRQ(PetscMalloc1(nLeaves, &iremote));
  CHKERRQ(PetscSFCreate(ctx->comm, &sf));
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
  CHKERRQ(PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  if (ctx->contiguousLeaves) {
    const PetscInt *tlocal;

    CHKERRQ(PetscSFGetGraph(sf, NULL, NULL, &tlocal, NULL));
    PetscCheckFalse(tlocal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ilocal=NULL expected for contiguous case");
  }
  *sf1 = sf;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx            ctx;
  PetscSF           sf0, sf1;
  MPI_Comm          comm;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRQ(GetOptions(comm, &ctx));

  CHKERRQ(CreateSF0(&ctx, &sf0));
  CHKERRQ(CreateSF1(&ctx, &sf1));
  ierr = PetscSFViewFromOptions(sf0, NULL, "-sf0_view");
  ierr = PetscSFViewFromOptions(sf1, NULL, "-sf1_view");
  CHKERRQ(PetscSFCheckEqual_Private(sf0, sf1));

  if (ctx.localmode != PETSC_OWN_POINTER) {
    CHKERRQ(PetscFree(ctx.ilocal));
  }
  if (ctx.remotemode != PETSC_OWN_POINTER) {
    CHKERRQ(PetscFree(ctx.iremote));
  }
  CHKERRQ(PetscSFDestroy(&sf0));
  CHKERRQ(PetscSFDestroy(&sf1));
  ierr = PetscFinalize();
  return ierr;
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
