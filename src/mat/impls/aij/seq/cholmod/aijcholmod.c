
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/sbaij/seq/cholmod/cholmodimpl.h>

#undef __FUNCT__
#define __FUNCT__ "MatWrapCholmod_seqaij"
static PetscErrorCode MatWrapCholmod_seqaij(Mat A,PetscBool  values,cholmod_sparse *C,PetscBool  *aijalloc)
{
  Mat_SeqAIJ      *aij = (Mat_SeqAIJ*)A->data;
  const PetscInt  *ai = aij->i,*aj = aij->j,*adiag;
  const MatScalar *aa = aij->a;
  PetscInt        m = A->rmap->n,i,j,k,nz,*ci,*cj;
  PetscScalar     *ca;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);
  adiag = aij->diag;
  for (i=0,nz=0; i<m; i++) nz += ai[i+1] - adiag[i];
  ierr = PetscMalloc3(m+1,PetscInt,&ci,nz,PetscInt,&cj,values?nz:0,PetscScalar,&ca);CHKERRQ(ierr);
  for (i=0,k=0; i<m; i++) {
    ci[i] = k;
    for (j=adiag[i]; j<ai[i+1]; j++,k++) {
      cj[k] = aj[j];
      if (values) ca[k] = aa[j];
    }
  }
  ci[i] = k;
  *aijalloc = PETSC_TRUE;

  ierr = PetscMemzero(C,sizeof(*C));CHKERRQ(ierr);
  C->nrow  = (size_t)A->cmap->n;
  C->ncol  = (size_t)A->rmap->n;
  C->nzmax = (size_t)nz;
  C->p     = ci;
  C->i     = cj;
  C->x     = values ? ca : 0;
  C->stype = -1;
  C->itype = CHOLMOD_INT_TYPE;
  C->xtype = values ? CHOLMOD_SCALAR_TYPE : CHOLMOD_PATTERN;
  C->dtype = CHOLMOD_DOUBLE;
  C->sorted = 1;
  C->packed = 1;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_seqaij_cholmod"
PetscErrorCode MatFactorGetSolverPackage_seqaij_cholmod(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCHOLMOD;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_seqaij_cholmod"
/* Almost a copy of MatGetFactor_seqsbaij_cholmod, yuck */
PetscErrorCode MatGetFactor_seqaij_cholmod(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_CHOLMOD    *chol;
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=A->cmap->n;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"CHOLMOD cannot do %s factorization with AIJ, only %s",
                                             MatFactorTypes[ftype],MatFactorTypes[MAT_FACTOR_CHOLESKY]);
  /* Create the factorization matrix F */
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscNewLog(B,Mat_CHOLMOD,&chol);CHKERRQ(ierr);
  chol->Wrap    = MatWrapCholmod_seqaij;
  chol->Destroy = MatDestroy_SeqAIJ;
  B->spptr      = chol;

  B->ops->view                   = MatView_CHOLMOD;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_CHOLMOD;
  B->ops->destroy                = MatDestroy_CHOLMOD;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_seqaij_cholmod",MatFactorGetSolverPackage_seqaij_cholmod);CHKERRQ(ierr);
  B->factortype   = MAT_FACTOR_CHOLESKY;
  B->assembled    = PETSC_TRUE;  /* required by -ksp_view */
  B->preallocated = PETSC_TRUE;

  ierr = CholmodStart(B);CHKERRQ(ierr);
  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END
