
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ctx->comm = comm;
  ctx->nLeavesPerRank = 4;
  ctx->leaveStep = 1;
  ctx->contiguousLeaves = PETSC_FALSE;
  ctx->localmode = PETSC_OWN_POINTER;
  ctx->remotemode = PETSC_OWN_POINTER;
  ctx->ilocal = NULL;
  ctx->iremote = NULL;
  ierr = PetscOptionsGetInt(NULL, NULL, "-n_leaves_per_rank", &ctx->nLeavesPerRank, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-leave_step", &ctx->leaveStep, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum(NULL, NULL, "-localmode", PetscCopyModes, (PetscEnum*) &ctx->localmode, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum(NULL, NULL, "-remotemode", PetscCopyModes, (PetscEnum*) &ctx->remotemode, NULL);CHKERRQ(ierr);
  ctx->contiguousLeaves = (PetscBool) (ctx->leaveStep == 1);
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

PetscErrorCode CreateSF0(AppCtx *ctx, PetscSF *sf0)
{
  PetscInt          j, k, r;
  PetscInt          nLeaves = ctx->nLeavesPerRank * ctx->size;
  PetscInt          nroots  = ctx->nLeavesPerRank;
  PetscSF           sf;
  PetscInt         *ilocal;
  PetscSFNode      *iremote;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nLeaves+1, &ctx->ilocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(nLeaves, &ctx->iremote);CHKERRQ(ierr);
  ilocal = ctx->ilocal;
  iremote = ctx->iremote;
  ilocal[nLeaves] = -ctx->leaveStep;
  ierr = PetscSFCreate(ctx->comm, &sf);CHKERRQ(ierr);
  for (r=0, j=nLeaves-1; r<ctx->size; r++) {
    for (k=0; k<ctx->nLeavesPerRank; k++, j--) {
      ilocal[j] = ilocal[j+1] + ctx->leaveStep;
      iremote[j].rank = r;
      iremote[j].index = k;
    }
  }
  ierr = PetscSFSetGraph(sf, nroots, nLeaves, ilocal, ctx->localmode, iremote, ctx->remotemode);CHKERRQ(ierr);
  {
    const PetscInt *tlocal;
    PetscBool       sorted;

    ierr = PetscSFGetGraph(sf, NULL, NULL, &tlocal, NULL);CHKERRQ(ierr);
    PetscCheckFalse(ctx->contiguousLeaves && tlocal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ilocal=NULL expected for contiguous case");
    if (tlocal) {
      ierr = PetscSortedInt(nLeaves, tlocal, &sorted);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ilocal = NULL;
  if (!ctx->contiguousLeaves) {
    ierr = PetscCalloc1(nLeaves+1, &ilocal);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(nLeaves, &iremote);CHKERRQ(ierr);
  ierr = PetscSFCreate(ctx->comm, &sf);CHKERRQ(ierr);
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
  ierr = PetscSFSetGraph(sf, nroots, nLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  if (ctx->contiguousLeaves) {
    const PetscInt *tlocal;

    ierr = PetscSFGetGraph(sf, NULL, NULL, &tlocal, NULL);CHKERRQ(ierr);
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
  ierr = GetOptions(comm, &ctx);CHKERRQ(ierr);

  ierr = CreateSF0(&ctx, &sf0);CHKERRQ(ierr);
  ierr = CreateSF1(&ctx, &sf1);CHKERRQ(ierr);
  ierr = PetscSFViewFromOptions(sf0, NULL, "-sf0_view");
  ierr = PetscSFViewFromOptions(sf1, NULL, "-sf1_view");
  ierr = PetscSFCheckEqual_Private(sf0, sf1);CHKERRQ(ierr);

  if (ctx.localmode != PETSC_OWN_POINTER) {
    ierr = PetscFree(ctx.ilocal);CHKERRQ(ierr);
  }
  if (ctx.remotemode != PETSC_OWN_POINTER) {
    ierr = PetscFree(ctx.iremote);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sf0);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf1);CHKERRQ(ierr);
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
