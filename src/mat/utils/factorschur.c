#include <petsc/private/matimpl.h>
#include <../src/mat/impls/dense/seq/dense.h>

PETSC_INTERN PetscErrorCode MatFactorSetUpInPlaceSchur_Private(Mat F)
{
  Mat              St, S = F->schur;
  MatFactorInfo    info;

  PetscFunctionBegin;
  CHKERRQ(MatSetUnfactored(S));
  CHKERRQ(MatGetFactor(S,S->solvertype ? S->solvertype : MATSOLVERPETSC,F->factortype,&St));
  if (St->factortype == MAT_FACTOR_CHOLESKY) { /* LDL^t regarded as Cholesky */
    CHKERRQ(MatCholeskyFactorSymbolic(St,S,NULL,&info));
  } else {
    CHKERRQ(MatLUFactorSymbolic(St,S,NULL,NULL,&info));
  }
  S->ops->solve             = St->ops->solve;
  S->ops->matsolve          = St->ops->matsolve;
  S->ops->solvetranspose    = St->ops->solvetranspose;
  S->ops->matsolvetranspose = St->ops->matsolvetranspose;
  S->ops->solveadd          = St->ops->solveadd;
  S->ops->solvetransposeadd = St->ops->solvetransposeadd;

  CHKERRQ(MatDestroy(&St));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatFactorUpdateSchurStatus_Private(Mat F)
{
  Mat            S = F->schur;

  PetscFunctionBegin;
  switch(F->schur_status) {
  case MAT_FACTOR_SCHUR_UNFACTORED:
  case MAT_FACTOR_SCHUR_INVERTED:
    if (S) {
      S->ops->solve             = NULL;
      S->ops->matsolve          = NULL;
      S->ops->solvetranspose    = NULL;
      S->ops->matsolvetranspose = NULL;
      S->ops->solveadd          = NULL;
      S->ops->solvetransposeadd = NULL;
      S->factortype             = MAT_FACTOR_NONE;
      CHKERRQ(PetscFree(S->solvertype));
    }
    break;
  case MAT_FACTOR_SCHUR_FACTORED:
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)F),PETSC_ERR_SUP,"Unhandled MatFactorSchurStatus %d",F->schur_status);
  }
  PetscFunctionReturn(0);
}

/* Schur status updated in the interface */
PETSC_INTERN PetscErrorCode MatFactorFactorizeSchurComplement_Private(Mat F)
{
  MatFactorInfo  info;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(MAT_FactorFactS,F,0,0,0));
  if (F->factortype == MAT_FACTOR_CHOLESKY) { /* LDL^t regarded as Cholesky */
    CHKERRQ(MatCholeskyFactor(F->schur,NULL,&info));
  } else {
    CHKERRQ(MatLUFactor(F->schur,NULL,NULL,&info));
  }
  CHKERRQ(PetscLogEventEnd(MAT_FactorFactS,F,0,0,0));
  PetscFunctionReturn(0);
}

/* Schur status updated in the interface */
PETSC_INTERN PetscErrorCode MatFactorInvertSchurComplement_Private(Mat F)
{
  Mat S = F->schur;

  PetscFunctionBegin;
  if (S) {
    PetscMPIInt    size;
    PetscBool      isdense,isdensecuda;

    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)S),&size));
    PetscCheckFalse(size > 1,PetscObjectComm((PetscObject)S),PETSC_ERR_SUP,"Not yet implemented");
    CHKERRQ(PetscObjectTypeCompare((PetscObject)S,MATSEQDENSE,&isdense));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)S,MATSEQDENSECUDA,&isdensecuda));
    CHKERRQ(PetscLogEventBegin(MAT_FactorInvS,F,0,0,0));
    if (isdense) {
      CHKERRQ(MatSeqDenseInvertFactors_Private(S));
#if defined(PETSC_HAVE_CUDA)
    } else if (isdensecuda) {
      CHKERRQ(MatSeqDenseCUDAInvertFactors_Private(S));
#endif
    } else SETERRQ(PetscObjectComm((PetscObject)S),PETSC_ERR_SUP,"Not implemented for type %s",((PetscObject)S)->type_name);
    CHKERRQ(PetscLogEventEnd(MAT_FactorInvS,F,0,0,0));
  }
  PetscFunctionReturn(0);
}
