#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <petsc/private/matimpl.h>

static PetscErrorCode MatFactorReset_NestHtool(Mat F)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)F, "MatFactorKSP", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorDestroy_NestHtool(Mat F)
{
  PetscFunctionBegin;
  PetscCall(MatFactorReset_NestHtool(F));
  PetscCall(PetscObjectComposeFunction((PetscObject)F, "MatFactorGetSolverType_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorGetSolverType_NestHtool(PETSC_UNUSED Mat F, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERHTOOL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorSymbolic_NestHtool_FieldSplit(Mat F, Mat A)
{
  KSP       ksp;
  PC        pc;
  Mat     **mats;
  IS       *rowis, *colis, *ises = NULL;
  IS        is_h, is_r = NULL;
  PetscBool found = PETSC_FALSE, same;
  PetscInt  nr, nc, hidx = -1, nrest = 0;

  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(A, &nr, &nc, &mats));
  PetscCheck(nr == nc, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only square MATNEST matrices are supported, got (%" PetscInt_FMT ",%" PetscInt_FMT ")", nr, nc);
  PetscCall(PetscMalloc2(nr, &rowis, nc, &colis));
  PetscCall(MatNestGetISs(A, rowis, colis));
  for (PetscInt i = 0; i < nr; ++i) {
    PetscCall(ISEqualUnsorted(rowis[i], colis[i], &same));
    PetscCheck(same, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only congruent MATNEST row/column layouts are supported");
  }
  for (PetscInt i = 0; i < nr; ++i) {
    PetscBool flg;

    PetscCall(PetscObjectTypeCompare((PetscObject)mats[i][i], MATHTOOL, &flg));
    if (flg) {
      PetscCheck(!found, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Exactly one diagonal MATHTOOL block is required");
      found = PETSC_TRUE;
      hidx  = i;
    } else {
      PetscCall(PetscObjectTypeCompareAny((PetscObject)mats[i][i], &flg, MATSEQDENSE, MATMPIDENSE, NULL));
      PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Diagonal block %" PetscInt_FMT " must be MATHTOOL or MATDENSE", i);
    }
  }
  PetscCheck(found, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "No diagonal MATHTOOL block found");
  PetscCall(ISDuplicate(rowis[hidx], &is_h));
  nrest = nr - 1;
  if (nrest > 0) {
    PetscCall(PetscMalloc1(nrest, &ises));
    for (PetscInt i = 0, j = 0; i < nr; ++i) {
      if (i == hidx) continue;
      ises[j++] = rowis[i];
    }
    PetscCall(ISConcatenate(PetscObjectComm((PetscObject)A), nrest, ises, &is_r));
    PetscCall(PetscFree(ises));
  }
  PetscCall(PetscFree2(rowis, colis));
  PetscCall(PetscObjectCompose((PetscObject)F, "MatFactorKSP", NULL));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)A), &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCFIELDSPLIT));
  PetscCall(PCFieldSplitSetIS(pc, NULL, is_h));
  PetscCall(ISDestroy(&is_h));
  if (nrest > 0) {
    PetscCall(PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR));
    PetscCall(PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_FULL, NULL));
    PetscCall(PCFieldSplitSetIS(pc, NULL, is_r));
    PetscCall(ISDestroy(&is_r));
  }
  PetscCall(PetscObjectCompose((PetscObject)F, "MatFactorKSP", (PetscObject)ksp));
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorNumeric_NestHtool(Mat F, Mat A, MatFactorType factor_type)
{
  KSP      ksp;
  KSP     *subksp;
  PC       pc;
  PetscInt nsplits;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)F, "MatFactorKSP", (PetscObject *)&ksp));
  PetscCheck(ksp, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "Outer PCFIELDSPLIT has not been created yet, call symbolic factorization first");
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCFieldSplitGetSubKSP(pc, &nsplits, &subksp));
  PetscCheck(nsplits == 1 || nsplits == 2, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Unexpected number of splits %" PetscInt_FMT " (!= 1 or 2)", nsplits);
  for (PetscInt i = 0; i < nsplits; ++i) {
    Mat       subA;
    PetscBool flg;

    PetscCall(KSPSetType(subksp[i], KSPPREONLY));
    PetscCall(KSPGetPC(subksp[i], &pc));
    PetscCall(KSPGetOperators(subksp[i], &subA, NULL));
    PetscCall(PetscObjectTypeCompare((PetscObject)subA, MATHTOOL, &flg));
    if (flg) flg = (PetscBool)(subA->symmetric == PETSC_BOOL3_TRUE || subA->hermitian == PETSC_BOOL3_TRUE);
    PetscCall(PCSetType(pc, factor_type == MAT_FACTOR_CHOLESKY || flg ? PCCHOLESKY : PCLU));
  }
  PetscCall(PetscFree(subksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_NestHtool(Mat F, Vec b, Vec x)
{
  KSP ksp;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)F, "MatFactorKSP", (PetscObject *)&ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveTranspose_NestHtool(Mat F, Vec b, Vec x)
{
  KSP ksp;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)F, "MatFactorKSP", (PetscObject *)&ksp));
  PetscCall(KSPSolveTranspose(ksp, b, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_NestHtool(Mat F, Mat B, Mat X)
{
  KSP ksp;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)F, "MatFactorKSP", (PetscObject *)&ksp));
  PetscCall(KSPMatSolve(ksp, B, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolveTranspose_NestHtool(Mat F, Mat B, Mat X)
{
  KSP ksp;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)F, "MatFactorKSP", (PetscObject *)&ksp));
  PetscCall(KSPMatSolveTranspose(ksp, B, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_NestHtool(Mat F, Mat A, PETSC_UNUSED const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatFactorNumeric_NestHtool(F, A, MAT_FACTOR_LU));
  F->ops->solve             = MatSolve_NestHtool;
  F->ops->matsolve          = MatMatSolve_NestHtool;
  F->ops->solvetranspose    = MatSolveTranspose_NestHtool;
  F->ops->matsolvetranspose = MatMatSolveTranspose_NestHtool;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_NestHtool(Mat F, Mat A, PETSC_UNUSED IS row, PETSC_UNUSED IS col, PETSC_UNUSED const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatFactorSymbolic_NestHtool_FieldSplit(F, A));
  F->ops->lufactornumeric = MatLUFactorNumeric_NestHtool;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorNumeric_NestHtool(Mat F, Mat A, PETSC_UNUSED const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatFactorNumeric_NestHtool(F, A, MAT_FACTOR_CHOLESKY));
  F->ops->solve             = MatSolve_NestHtool;
  F->ops->matsolve          = MatMatSolve_NestHtool;
  F->ops->solvetranspose    = MatSolveTranspose_NestHtool;
  F->ops->matsolvetranspose = MatMatSolveTranspose_NestHtool;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_NestHtool(Mat F, Mat A, PETSC_UNUSED IS row, PETSC_UNUSED const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatFactorSymbolic_NestHtool_FieldSplit(F, A));
  F->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_NestHtool;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_nest_htool(Mat A, MatFactorType ftype, Mat *F)
{
  Mat         B;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  PetscCheck(size == 1, PetscObjectComm((PetscObject)A), PETSC_ERR_WRONG_MPI_SIZE, "Unsupported parallel MatGetFactor()");
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(PetscStrallocpy(MATSOLVERHTOOL, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  B->ops->destroy    = MatFactorDestroy_NestHtool;
  B->ops->getinfo    = MatGetInfo_External;
  B->factortype      = ftype;
  B->trivialsymbolic = PETSC_FALSE;
  B->preallocated    = PETSC_TRUE;
  B->assembled       = PETSC_TRUE;

  PetscCheck(ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_CHOLESKY, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only MAT_FACTOR_LU and MAT_FACTOR_CHOLESKY are supported");
  if (ftype == MAT_FACTOR_LU) B->ops->lufactorsymbolic = MatLUFactorSymbolic_NestHtool;
  else B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_NestHtool;

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERHTOOL, &B->solvertype));

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_NestHtool));
  *F = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_NestHtool(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERHTOOL, MATNEST, MAT_FACTOR_LU, MatGetFactor_nest_htool));
  PetscCall(MatSolverTypeRegister(MATSOLVERHTOOL, MATNEST, MAT_FACTOR_CHOLESKY, MatGetFactor_nest_htool));
  PetscFunctionReturn(PETSC_SUCCESS);
}
