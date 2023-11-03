#include <petsc/private/matimpl.h> /*I "petscmat.h"  I*/
#include "hpl.h"

/*@
  MatSetHPL - fills a `MATSEQDENSE` matrix using the HPL 2.3 random matrix generation routine

  Collective

  Input Parameters:
+ A     - the matrix
- iseed - the random number seed

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatCreate()`
@*/
PetscErrorCode MatSetHPL(Mat A, int iseed)
{
  PetscBool    isDense;
  PetscInt     M, N, LDA;
  PetscBLASInt bM, bN, bLDA;
  PetscScalar *values;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQDENSE, &isDense));
  PetscCheck(isDense, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only supports sequential dense matrices");
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(PetscBLASIntCast(M, &bM));
  PetscCall(PetscBLASIntCast(N, &bN));
  PetscCall(MatDenseGetLDA(A, &LDA));
  PetscCall(PetscBLASIntCast(LDA, &bLDA));
  PetscCall(MatDenseGetArrayWrite(A, &values));
  PetscStackCallExternalVoid("HPL_dmatgen", HPL_dmatgen(bM, bN, values, bLDA, iseed));
  PetscCall(MatDenseRestoreArrayWrite(A, &values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/bmimpl.h>

typedef struct {
  Mat A, F;
} PetscBench_HPL;

static PetscErrorCode PetscBenchSetUp_HPL(PetscBench bm)
{
  PetscBench_HPL *hp = (PetscBench_HPL *)bm->data;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)bm), &rank));
  if (rank > 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (bm->size == PETSC_DECIDE) bm->size = 2500;
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, bm->size, bm->size, NULL, &hp->A));
  PetscCall(MatSetHPL(hp->A, 0));
  PetscCall(MatGetFactor(hp->A, MATSOLVERPETSC, MAT_FACTOR_LU, &hp->F));
  PetscCall(MatLUFactorSymbolic(hp->F, hp->A, NULL, NULL, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscBenchRun_HPL(PetscBench bm)
{
  PetscBench_HPL *hp = (PetscBench_HPL *)bm->data;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)bm), &rank));
  if (rank > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatLUFactorNumeric(hp->F, hp->A, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscBenchView_HPL(PetscBench bm, PetscViewer viewer)
{
  PetscEventPerfInfo *info;
  PetscInt            numThreads;

  PetscFunctionBegin;
  PetscCall(PetscLogHandlerGetEventPerfInfo(bm->lhdlr, 0, MAT_LUFactor, &info)); // because symbolic is trivial it does not log numeric!
  PetscCall(PetscBLASGetNumThreads(&numThreads));
  PetscCall(PetscViewerASCIIPrintf(viewer, "HPL Benchmark, number of threads %d, matrix size %d, flop rate %g (megaflops)\n", (int)numThreads, (int)bm->size, (double)(info->flops / (1.e7 * info->time))));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscBenchReset_HPL(PetscBench bm)
{
  PetscBench_HPL *hp = (PetscBench_HPL *)bm->data;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)bm), &rank));
  if (rank > 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDestroy(&hp->F));
  PetscCall(MatDestroy(&hp->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscBenchDestroy_HPL(PetscBench bm)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(bm->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscBenchCreate_HPL(PetscBench bm)
{
  PetscBench_HPL *hp;

  PetscFunctionBegin;
  PetscCall(PetscNew(&hp));
  bm->data         = hp;
  bm->ops->setup   = PetscBenchSetUp_HPL;
  bm->ops->run     = PetscBenchRun_HPL;
  bm->ops->view    = PetscBenchView_HPL;
  bm->ops->reset   = PetscBenchReset_HPL;
  bm->ops->destroy = PetscBenchDestroy_HPL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
