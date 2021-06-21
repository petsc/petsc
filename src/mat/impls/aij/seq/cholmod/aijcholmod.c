
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr  = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);
  adiag = aij->diag;
  for (i=0,nz=0; i<m; i++) nz += ai[i+1] - adiag[i];
  ierr = PetscMalloc2(m+1,&ci,nz,&cj);CHKERRQ(ierr);
  if (values) {
    vain = PETSC_TRUE;
    ierr = PetscMalloc1(nz,&ca);CHKERRQ(ierr);
    ierr = MatSeqAIJGetArrayRead(A,&aa);CHKERRQ(ierr);
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
    ierr = MatSeqAIJRestoreArrayRead(A,&aa);CHKERRQ(ierr);
  }

  ierr = PetscMemzero(C,sizeof(*C));CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=A->cmap->n;
  const char     *prefix;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (!A->hermitian) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Only for hermitian matrices");
#endif
  /* Create the factorization matrix F */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = PetscStrallocpy("cholmod",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(B,prefix);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = PetscNewLog(B,&chol);CHKERRQ(ierr);

  chol->Wrap = MatWrapCholmod_seqaij;
  B->data    = chol;

  B->ops->getinfo                = MatGetInfo_CHOLMOD;
  B->ops->view                   = MatView_CHOLMOD;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_CHOLMOD;
  B->ops->destroy                = MatDestroy_CHOLMOD;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_cholmod);CHKERRQ(ierr);

  B->factortype   = MAT_FACTOR_CHOLESKY;
  B->assembled    = PETSC_TRUE;
  B->preallocated = PETSC_TRUE;

  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERCHOLMOD,&B->solvertype);CHKERRQ(ierr);
  B->canuseordering = PETSC_TRUE;
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&B->preferredordering[MAT_FACTOR_CHOLESKY]);CHKERRQ(ierr);
  ierr = CholmodStart(B);CHKERRQ(ierr);
  *F   = B;
  PetscFunctionReturn(0);
}
