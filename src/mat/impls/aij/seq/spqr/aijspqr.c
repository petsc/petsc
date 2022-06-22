
#include <petscsys.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/sbaij/seq/cholmod/cholmodimpl.h>

EXTERN_C_BEGIN
#include <SuiteSparseQR_C.h>
EXTERN_C_END

static PetscErrorCode MatWrapCholmod_SPQR_seqaij(Mat A,PetscBool values,cholmod_sparse *C,PetscBool *aijalloc,PetscBool *valloc)
{
  Mat_SeqAIJ        *aij;
  Mat               AT;
  const PetscScalar *aa;
  PetscScalar       *ca;
  const PetscInt    *ai, *aj;
  PetscInt          n = A->cmap->n, i,j,k,nz;
  SuiteSparse_long  *ci, *cj; /* SuiteSparse_long is the only choice for SPQR */
  PetscBool         vain = PETSC_FALSE,flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATNORMALHERMITIAN, &flg));
  if (flg) {
    PetscCall(MatNormalHermitianGetMat(A, &A));
  } else if (!PetscDefined(USE_COMPLEX)) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATNORMAL, &flg));
    if (flg) {
      PetscCall(MatNormalGetMat(A, &A));
    }
  }
  /* cholmod_sparse is compressed sparse column */
  PetscCall(MatGetOption(A, MAT_SYMMETRIC, &flg));
  if (flg) {
    PetscCall(PetscObjectReference((PetscObject)A));
    AT = A;
  } else {
    PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &AT));
  }
  aij = (Mat_SeqAIJ*)AT->data;
  ai = aij->j;
  aj = aij->i;
  for (j=0,nz=0; j<n; j++) nz += aj[j+1] - aj[j];
  PetscCall(PetscMalloc2(n+1,&cj,nz,&ci));
  if (values) {
    vain = PETSC_TRUE;
    PetscCall(PetscMalloc1(nz,&ca));
    PetscCall(MatSeqAIJGetArrayRead(AT,&aa));
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
    PetscCall(MatSeqAIJRestoreArrayRead(AT,&aa));
  }

  PetscCall(PetscMemzero(C,sizeof(*C)));

  C->nrow   = (size_t)AT->cmap->n;
  C->ncol   = (size_t)AT->rmap->n;
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

  PetscCall(MatDestroy(&AT));
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
  cholmod_dense  *Y_handle = NULL, *QTB_handle = NULL, *Z_handle = NULL;

  PetscFunctionBegin;
  if (!chol->normal) {
    QTB_handle = SuiteSparseQR_C_qmult(SPQR_QTX, chol->spqrfact, cholB, chol->common);
    PetscCheck(QTB_handle,PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_qmult failed");
    Y_handle = SuiteSparseQR_C_solve(SPQR_RETX_EQUALS_B, chol->spqrfact, QTB_handle, chol->common);
    PetscCheck(Y_handle,PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_solve failed");
  } else {
    Z_handle = SuiteSparseQR_C_solve(SPQR_RTX_EQUALS_ETB, chol->spqrfact, cholB, chol->common);
    PetscCheck(Z_handle,PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_solve failed");
    Y_handle = SuiteSparseQR_C_solve(SPQR_RETX_EQUALS_B, chol->spqrfact, Z_handle, chol->common);
    PetscCheck(Y_handle,PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_solve failed");
    PetscCall(!cholmod_l_free_dense(&Z_handle, chol->common));
  }
  *_Y_handle = Y_handle;
  PetscCall(!cholmod_l_free_dense(&QTB_handle, chol->common));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SPQR(Mat F,Vec B,Vec X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,*Y_handle = NULL;
  PetscInt       n;
  PetscScalar    *v;

  PetscFunctionBegin;
  PetscCall(VecWrapCholmod(B,GET_ARRAY_READ,&cholB));
  PetscCall(MatSolve_SPQR_Internal(F, &cholB, &Y_handle));
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetArrayWrite(X, &v));
  PetscCall(PetscArraycpy(v, (PetscScalar *) (Y_handle->x), n));
  PetscCall(VecRestoreArrayWrite(X, &v));
  PetscCall(!cholmod_l_free_dense(&Y_handle, chol->common));
  PetscCall(VecUnWrapCholmod(B,GET_ARRAY_READ,&cholB));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SPQR(Mat F,Mat B,Mat X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,*Y_handle = NULL;
  PetscScalar    *v;
  PetscInt       lda;

  PetscFunctionBegin;
  PetscCall(MatDenseWrapCholmod(B,GET_ARRAY_READ,&cholB));
  PetscCall(MatSolve_SPQR_Internal(F, &cholB, &Y_handle));
  PetscCall(MatDenseGetArrayWrite(X, &v));
  PetscCall(MatDenseGetLDA(X, &lda));
  if ((size_t) lda == Y_handle->d) {
    PetscCall(PetscArraycpy(v, (PetscScalar *) (Y_handle->x), lda * Y_handle->ncol));
  } else {
    for (size_t j = 0; j < Y_handle->ncol; j++) {
      PetscCall(PetscArraycpy(&v[j*lda], &(((PetscScalar *) Y_handle->x)[j*Y_handle->d]), Y_handle->nrow));
    }
  }
  PetscCall(MatDenseRestoreArrayWrite(X, &v));
  PetscCall(!cholmod_l_free_dense(&Y_handle, chol->common));
  PetscCall(MatDenseUnWrapCholmod(B,GET_ARRAY_READ,&cholB));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SPQR_Internal(Mat F, cholmod_dense *cholB, cholmod_dense **_Y_handle)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  *Y_handle = NULL, *RTB_handle = NULL;

  PetscFunctionBegin;
  RTB_handle = SuiteSparseQR_C_solve(SPQR_RTX_EQUALS_ETB, chol->spqrfact, cholB, chol->common);
  PetscCheck(RTB_handle,PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_solve failed");
  Y_handle = SuiteSparseQR_C_qmult(SPQR_QX, chol->spqrfact, RTB_handle, chol->common);
  PetscCheck(Y_handle,PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "SuiteSparseQR_C_qmult failed");
  *_Y_handle = Y_handle;
  PetscCall(!cholmod_l_free_dense(&RTB_handle, chol->common));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SPQR(Mat F,Vec B,Vec X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,*Y_handle = NULL;
  PetscInt       n;
  PetscScalar    *v;

  PetscFunctionBegin;
  PetscCall(VecWrapCholmod(B,GET_ARRAY_READ,&cholB));
  PetscCall(MatSolveTranspose_SPQR_Internal(F, &cholB, &Y_handle));
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetArrayWrite(X, &v));
  PetscCall(PetscArraycpy(v, (PetscScalar *) Y_handle->x, n));
  PetscCall(VecRestoreArrayWrite(X, &v));
  PetscCall(!cholmod_l_free_dense(&Y_handle, chol->common));
  PetscCall(VecUnWrapCholmod(B,GET_ARRAY_READ,&cholB));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SPQR(Mat F,Mat B,Mat X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_dense  cholB,*Y_handle = NULL;
  PetscScalar    *v;
  PetscInt       lda;

  PetscFunctionBegin;
  PetscCall(MatDenseWrapCholmod(B,GET_ARRAY_READ,&cholB));
  PetscCall(MatSolveTranspose_SPQR_Internal(F, &cholB, &Y_handle));
  PetscCall(MatDenseGetArrayWrite(X, &v));
  PetscCall(MatDenseGetLDA(X, &lda));
  if ((size_t) lda == Y_handle->d) {
    PetscCall(PetscArraycpy(v, (PetscScalar *) Y_handle->x, lda * Y_handle->ncol));
  } else {
    for (size_t j = 0; j < Y_handle->ncol; j++) {
      PetscCall(PetscArraycpy(&v[j*lda], &(((PetscScalar *) Y_handle->x)[j*Y_handle->d]), Y_handle->nrow));
    }
  }
  PetscCall(MatDenseRestoreArrayWrite(X, &v));
  PetscCall(!cholmod_l_free_dense(&Y_handle, chol->common));
  PetscCall(MatDenseUnWrapCholmod(B,GET_ARRAY_READ,&cholB));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatQRFactorNumeric_SPQR(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_sparse cholA;
  PetscBool      aijalloc,valloc;
  int            err;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATNORMALHERMITIAN, &chol->normal));
  if (!chol->normal && !PetscDefined(USE_COMPLEX)) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATNORMAL, &chol->normal));
  }
  PetscCall((*chol->Wrap)(A,PETSC_TRUE,&cholA,&aijalloc,&valloc));
  err = !SuiteSparseQR_C_numeric(PETSC_SMALL, &cholA, chol->spqrfact, chol->common);
  PetscCheck(!err,PetscObjectComm((PetscObject)F),PETSC_ERR_LIB,"SPQR factorization failed with status %d",chol->common->status);

  if (aijalloc) PetscCall(PetscFree2(cholA.p,cholA.i));
  if (valloc) PetscCall(PetscFree(cholA.x));

  F->ops->solve             = MatSolve_SPQR;
  F->ops->matsolve          = MatMatSolve_SPQR;
  if (chol->normal) {
    F->ops->solvetranspose    = MatSolve_SPQR;
    F->ops->matsolvetranspose = MatMatSolve_SPQR;
  } else if (A->cmap->n == A->rmap->n) {
    F->ops->solvetranspose    = MatSolveTranspose_SPQR;
    F->ops->matsolvetranspose = MatMatSolveTranspose_SPQR;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatQRFactorSymbolic_SPQR(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->data;
  cholmod_sparse cholA;
  PetscBool      aijalloc,valloc;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATNORMALHERMITIAN, &chol->normal));
  if (!chol->normal && !PetscDefined(USE_COMPLEX)) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATNORMAL, &chol->normal));
  }
  PetscCall((*chol->Wrap)(A,PETSC_TRUE,&cholA,&aijalloc,&valloc));
  if (PetscDefined(USE_DEBUG)) {
    PetscCall(!cholmod_l_check_sparse(&cholA, chol->common));
  }
  if (chol->spqrfact) {
    PetscCall(!SuiteSparseQR_C_free(&chol->spqrfact, chol->common));
  }
  chol->spqrfact = SuiteSparseQR_C_symbolic(SPQR_ORDERING_DEFAULT, 1, &cholA, chol->common);
  PetscCheck(chol->spqrfact,PetscObjectComm((PetscObject)F),PETSC_ERR_LIB,"CHOLMOD analysis failed using internal ordering with status %d",chol->common->status);

  if (aijalloc) PetscCall(PetscFree2(cholA.p,cholA.i));
  if (valloc) PetscCall(PetscFree(cholA.x));

  PetscCall(PetscObjectComposeFunction((PetscObject)F,"MatQRFactorNumeric_C", MatQRFactorNumeric_SPQR));
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERSPQR

  A matrix type providing direct solvers (QR factorizations) for sequential matrices
  via the external package SPQR.

  Use ./configure --download-suitesparse to install PETSc to use CHOLMOD

  Consult SPQR documentation for more information about the Common parameters
  which correspond to the options database keys below.

   Level: beginner

   Note: SPQR is part of SuiteSparse http://faculty.cse.tamu.edu/davis/suitesparse.html

.seealso: `PCQR`, `PCFactorSetMatSolverType()`, `MatSolverType`
M*/

PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_spqr(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_CHOLMOD    *chol;
  PetscInt       m=A->rmap->n,n=A->cmap->n;
  const char     *prefix;

  PetscFunctionBegin;
  /* Create the factorization matrix F */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(PetscStrallocpy("spqr",&((PetscObject)B)->type_name));
  PetscCall(MatGetOptionsPrefix(A,&prefix));
  PetscCall(MatSetOptionsPrefix(B,prefix));
  PetscCall(MatSetUp(B));
  PetscCall(PetscNewLog(B,&chol));

  chol->Wrap = MatWrapCholmod_SPQR_seqaij;
  B->data    = chol;

  B->ops->getinfo = MatGetInfo_CHOLMOD;
  B->ops->view    = MatView_CHOLMOD;
  B->ops->destroy = MatDestroy_CHOLMOD;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_SPQR));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatQRFactorSymbolic_C", MatQRFactorSymbolic_SPQR));

  B->factortype   = MAT_FACTOR_QR;
  B->assembled    = PETSC_TRUE;
  B->preallocated = PETSC_TRUE;

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERCHOLMOD,&B->solvertype));
  B->canuseordering = PETSC_FALSE;
  PetscCall(CholmodStart(B));
  chol->common->itype = CHOLMOD_LONG;
  *F   = B;
  PetscFunctionReturn(0);
}
