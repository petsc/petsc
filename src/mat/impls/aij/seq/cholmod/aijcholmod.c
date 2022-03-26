
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/sbaij/seq/cholmod/cholmodimpl.h>

static PetscErrorCode MatWrapCholmod_seqaij(Mat A,PetscBool values,cholmod_sparse *C,PetscBool *aijalloc,PetscBool *valloc)
{
  Mat_SeqAIJ        *aij = (Mat_SeqAIJ*)A->data;
  const PetscScalar *aa;
  PetscScalar       *ca;
  const PetscInt    *ai = aij->i,*aj = aij->j,*adiag;
  PetscInt          m = A->rmap->n,i,j,k,nz,*ci,*cj;
  PetscBool         vain = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MatMarkDiagonal_SeqAIJ(A));
  adiag = aij->diag;
  for (i=0,nz=0; i<m; i++) nz += ai[i+1] - adiag[i];
  PetscCall(PetscMalloc2(m+1,&ci,nz,&cj));
  if (values) {
    vain = PETSC_TRUE;
    PetscCall(PetscMalloc1(nz,&ca));
    PetscCall(MatSeqAIJGetArrayRead(A,&aa));
  }
  for (i=0,k=0; i<m; i++) {
    ci[i] = k;
    for (j=adiag[i]; j<ai[i+1]; j++,k++) {
      cj[k] = aj[j];
      if (values) ca[k] = PetscConj(aa[j]);
    }
  }
  ci[i]     = k;
  *aijalloc = PETSC_TRUE;
  *valloc   = vain;
  if (values) {
    PetscCall(MatSeqAIJRestoreArrayRead(A,&aa));
  }

  PetscCall(PetscMemzero(C,sizeof(*C)));

  C->nrow   = (size_t)A->cmap->n;
  C->ncol   = (size_t)A->rmap->n;
  C->nzmax  = (size_t)nz;
  C->p      = ci;
  C->i      = cj;
  C->x      = values ? ca : 0;
  C->stype  = -1;
  C->itype  = CHOLMOD_INT_TYPE;
  C->xtype  = values ? CHOLMOD_SCALAR_TYPE : CHOLMOD_PATTERN;
  C->dtype  = CHOLMOD_DOUBLE;
  C->sorted = 1;
  C->packed = 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_seqaij_cholmod(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCHOLMOD;
  PetscFunctionReturn(0);
}

/* Almost a copy of MatGetFactor_seqsbaij_cholmod, yuck */
PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_cholmod(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_CHOLMOD    *chol;
  PetscInt       m=A->rmap->n,n=A->cmap->n;
  const char     *prefix;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCheck(A->hermitian,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only for hermitian matrices");
#endif
  /* Create the factorization matrix F */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(PetscStrallocpy("cholmod",&((PetscObject)B)->type_name));
  PetscCall(MatGetOptionsPrefix(A,&prefix));
  PetscCall(MatSetOptionsPrefix(B,prefix));
  PetscCall(MatSetUp(B));
  PetscCall(PetscNewLog(B,&chol));

  chol->Wrap = MatWrapCholmod_seqaij;
  B->data    = chol;

  B->ops->getinfo                = MatGetInfo_CHOLMOD;
  B->ops->view                   = MatView_CHOLMOD;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_CHOLMOD;
  B->ops->destroy                = MatDestroy_CHOLMOD;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_cholmod));

  B->factortype   = MAT_FACTOR_CHOLESKY;
  B->assembled    = PETSC_TRUE;
  B->preallocated = PETSC_TRUE;

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERCHOLMOD,&B->solvertype));
  B->canuseordering = PETSC_TRUE;
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_CHOLESKY]));
  PetscCall(CholmodStart(B));
  *F   = B;
  PetscFunctionReturn(0);
}
