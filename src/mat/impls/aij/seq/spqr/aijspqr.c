
#include <petscsys.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/sbaij/seq/cholmod/cholmodimpl.h>

EXTERN_C_BEGIN
#include <SuiteSparseQR_C.h>
EXTERN_C_END

static PetscErrorCode MatWrapCholmod_SPQR_seqaij(Mat A,PetscBool values,cholmod_sparse *C,PetscBool *aijalloc,PetscBool *valloc)
{
  Mat_SeqAIJ        *aij;
  Mat                AT;
  const PetscScalar *aa;
  PetscScalar       *ca;
  const PetscInt    *ai, *aj;
  PetscInt          n = A->cmap->n, i,j,k,nz;
  SuiteSparse_long  *ci, *cj; /* SuiteSparse_long is the only choice for SPQR */
  PetscBool         vain = PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* cholmod_sparse is compressed sparse column */
  {
    PetscBool issym;

    ierr = MatGetOption(A, MAT_SYMMETRIC, &issym);CHKERRQ(ierr);
    if (issym) {
      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
      AT = A;
    } else {
      ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &AT);CHKERRQ(ierr);
    }
  }
  aij = (Mat_SeqAIJ*)AT->data;
  ai = aij->j;
  aj = aij->i;
  for (j=0,nz=0; j<n; j++) nz += aj[j+1] - aj[j];
  ierr = PetscMalloc2(n+1,&cj,nz,&ci);CHKERRQ(ierr);
  if (values) {
    vain = PETSC_TRUE;
    ierr = PetscMalloc1(nz,&ca);CHKERRQ(ierr);
    ierr = MatSeqAIJGetArrayRead(AT,&aa);CHKERRQ(ierr);
  }
  for (j=0,k=0; j<n; j++) {
    cj[j] = k;
    for (i=aj[j]; i<aj[j+1]; i++,k++) {
      ci[k] = ai[i];
      if (values) ca[k] = aa[i];
    }
  }
  cj[j]     = k;
  *aijalloc = PETSC_TRUE;
  *valloc   = vain;
  if (values) {
    ierr = MatSeqAIJRestoreArrayRead(A,&aa);CHKERRQ(ierr);
  }

  ierr = PetscMemzero(C,sizeof(*C));CHKERRQ(ierr);

  C->nrow   = (size_t)A->rmap->n;
  C->ncol   = (size_t)A->cmap->n;
  C->nzmax  = (size_t)nz;
  C->p      = cj;
  C->i      = ci;
  C->x      = values ? ca : 0;
  C->stype  = 0;
  C->itype  = CHOLMOD_LONG;
  C->xtype  = values ? CHOLMOD_SCALAR_TYPE : CHOLMOD_PATTERN;
  C->dtype  = CHOLMOD_DOUBLE;
  C->sorted = 1;
  C->packed = 1;

  ierr = MatDestroy(&AT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_seqaij_SPQR(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSPQR;
  PetscFunctionReturn(0);
}

#define GET_ARRAY_READ 0
#define GET_ARRAY_WRITE 1

static PetscErrorCode MatSolve_SPQR_Internal(Mat F, cholmod_dense *cholB, cholmod_dense **_Y_handle)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  *Y_handle = NULL, *QTB_handle = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  QTB_handle = SuiteSparseQR_C_qmult(SPQR_QTX, chol->spqrfact, cholB, chol->common);
  if (!QTB_handle) SETERRQ(PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_qmult failed");
  Y_handle = SuiteSparseQR_C_solve(SPQR_RETX_EQUALS_B, chol->spqrfact, QTB_handle, chol->common);
  if (!Y_handle) SETERRQ(PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_solve failed");
  *_Y_handle = Y_handle;
  ierr = !cholmod_l_free_dense(&QTB_handle, chol->common);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SPQR(Mat F,Vec B,Vec X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,*Y_handle = NULL;
  PetscInt       n;
  PetscScalar    *v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  ierr = MatSolve_SPQR_Internal(F, &cholB, &Y_handle);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(X, &v);CHKERRQ(ierr);
  ierr = PetscArraycpy(v, (PetscScalar *) (Y_handle->x), n);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(X, &v);CHKERRQ(ierr);
  ierr = !cholmod_l_free_dense(&Y_handle, chol->common);CHKERRQ(ierr);
  ierr = VecUnWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SPQR(Mat F,Mat B,Mat X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,*Y_handle = NULL;
  PetscScalar    *v;
  PetscInt       lda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  ierr = MatSolve_SPQR_Internal(F, &cholB, &Y_handle);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(X, &v);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(X, &lda);CHKERRQ(ierr);
  if ((size_t) lda == Y_handle->d) {
    ierr = PetscArraycpy(v, (PetscScalar *) (Y_handle->x), lda * Y_handle->ncol);CHKERRQ(ierr);
  } else {
    for (size_t j = 0; j < Y_handle->ncol; j++) {
      ierr = PetscArraycpy(&v[j*lda], &(((PetscScalar *) Y_handle->x)[j*Y_handle->d]), Y_handle->nrow);CHKERRQ(ierr);
    }
  }
  ierr = MatDenseRestoreArrayWrite(X, &v);CHKERRQ(ierr);
  ierr = !cholmod_l_free_dense(&Y_handle, chol->common);CHKERRQ(ierr);
  ierr = MatDenseUnWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SPQR_Internal(Mat F, cholmod_dense *cholB, cholmod_dense **_Y_handle)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  *Y_handle = NULL, *RTB_handle = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  RTB_handle = SuiteSparseQR_C_solve(SPQR_RTX_EQUALS_ETB, chol->spqrfact, cholB, chol->common);
  if (!RTB_handle) SETERRQ(PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_solve failed");
  Y_handle = SuiteSparseQR_C_qmult(SPQR_QX, chol->spqrfact, RTB_handle, chol->common);
  if (!Y_handle) SETERRQ(PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_qmult failed");
  *_Y_handle = Y_handle;
  ierr = !cholmod_l_free_dense(&RTB_handle, chol->common);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SPQR(Mat F,Vec B,Vec X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,*Y_handle = NULL;
  PetscInt       n;
  PetscScalar    *v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  ierr = MatSolveTranspose_SPQR_Internal(F, &cholB, &Y_handle);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(X, &v);CHKERRQ(ierr);
  ierr = PetscArraycpy(v, (PetscScalar *) Y_handle->x, n);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(X, &v);CHKERRQ(ierr);
  ierr = !cholmod_l_free_dense(&Y_handle, chol->common);CHKERRQ(ierr);
  ierr = VecUnWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SPQR(Mat F,Mat B,Mat X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,*Y_handle = NULL;
  PetscScalar    *v;
  PetscInt       lda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  ierr = MatSolveTranspose_SPQR_Internal(F, &cholB, &Y_handle);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(X, &v);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(X, &lda);CHKERRQ(ierr);
  if ((size_t) lda == Y_handle->d) {
    ierr = PetscArraycpy(v, (PetscScalar *) Y_handle->x, lda * Y_handle->ncol);CHKERRQ(ierr);
  } else {
    for (size_t j = 0; j < Y_handle->ncol; j++) {
      ierr = PetscArraycpy(&v[j*lda], &(((PetscScalar *) Y_handle->x)[j*Y_handle->d]), Y_handle->nrow);CHKERRQ(ierr);
    }
  }
  ierr = MatDenseRestoreArrayWrite(X, &v);CHKERRQ(ierr);
  ierr = !cholmod_l_free_dense(&Y_handle, chol->common);CHKERRQ(ierr);
  ierr = MatDenseUnWrapCholmod(B,GET_ARRAY_READ,&cholB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatQRFactorNumeric_SPQR(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_sparse cholA;
  PetscBool      aijalloc,valloc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*chol->Wrap)(A,PETSC_TRUE,&cholA,&aijalloc,&valloc);CHKERRQ(ierr);
  ierr = !SuiteSparseQR_C_numeric(SPQR_DEFAULT_TOL, &cholA, chol->spqrfact, chol->common);
  if (ierr) SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_LIB,"SPQR factorization failed with status %d",chol->common->status);

  if (aijalloc) {ierr = PetscFree2(cholA.p,cholA.i);CHKERRQ(ierr);}
  if (valloc) {ierr = PetscFree(cholA.x);CHKERRQ(ierr);}

  F->ops->solve             = MatSolve_SPQR;
  F->ops->matsolve          = MatMatSolve_SPQR;
  if (A->cmap->n == A->rmap->n) {
    F->ops->solvetranspose    = MatSolveTranspose_SPQR;
    F->ops->matsolvetranspose = MatMatSolveTranspose_SPQR;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode  MatQRFactorSymbolic_SPQR(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  PetscErrorCode ierr;
  cholmod_sparse cholA;
  PetscBool      aijalloc,valloc;

  PetscFunctionBegin;
  ierr = (*chol->Wrap)(A,PETSC_TRUE,&cholA,&aijalloc,&valloc);CHKERRQ(ierr);
  if (PetscDefined(USE_DEBUG)) {
    ierr = !cholmod_l_check_sparse(&cholA, chol->common);CHKERRQ(ierr);
  }
  if (chol->spqrfact) {
    ierr = !SuiteSparseQR_C_free(&chol->spqrfact, chol->common);CHKERRQ(ierr);
  }
  chol->spqrfact = SuiteSparseQR_C_symbolic(SPQR_ORDERING_FIXED, 0, &cholA, chol->common);
  if (!chol->spqrfact) SETERRQ1(PetscObjectComm((PetscObject)F),PETSC_ERR_LIB,"CHOLMOD analysis failed using internal ordering with status %d",chol->common->status);

  if (aijalloc) {ierr = PetscFree2(cholA.p,cholA.i);CHKERRQ(ierr);}
  if (valloc) {ierr = PetscFree(cholA.x);CHKERRQ(ierr);}

  ierr = PetscObjectComposeFunction((PetscObject)F,"MatQRFactorNumeric_C", MatQRFactorNumeric_SPQR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERSPQR

  A matrix type providing direct solvers (QR factorizations) for sequential matrices
  via the external package SPQR.

  Use ./configure --download-suitesparse to install PETSc to use CHOLMOD

  TODO: Use -pc_type qr -pc_factor_mat_solver_type spqr to use this direct solver (TODO: PCQR)

  Consult SPQR documentation for more information about the Common parameters
  which correspond to the options database keys below.

   Level: beginner

   Note: SPQR is part of SuiteSparse http://faculty.cse.tamu.edu/davis/suitesparse.html

.seealso: PCQR, PCFactorSetMatSolverType(), MatSolverType
M*/

PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_spqr(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_CHOLMOD    *chol;
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=A->cmap->n;
  const char     *prefix;

  PetscFunctionBegin;
  /* Create the factorization matrix F */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = PetscStrallocpy("spqr",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(B,prefix);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = PetscNewLog(B,&chol);CHKERRQ(ierr);

  chol->Wrap = MatWrapCholmod_SPQR_seqaij;
  B->data    = chol;

  B->ops->getinfo = MatGetInfo_CHOLMOD;
  B->ops->view    = MatView_CHOLMOD;
  B->ops->destroy = MatDestroy_CHOLMOD;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_SPQR);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatQRFactorSymbolic_C", MatQRFactorSymbolic_SPQR);CHKERRQ(ierr);

  B->factortype   = MAT_FACTOR_QR;
  B->assembled    = PETSC_TRUE;
  B->preallocated = PETSC_TRUE;

  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERCHOLMOD,&B->solvertype);CHKERRQ(ierr);
  B->canuseordering = PETSC_FALSE;
  ierr = CholmodStart(B);CHKERRQ(ierr);
  chol->common->itype = CHOLMOD_LONG;
  *F   = B;
  PetscFunctionReturn(0);
}

