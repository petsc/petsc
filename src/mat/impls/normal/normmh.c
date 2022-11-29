
#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/

typedef struct {
  Mat         A;
  Mat         D; /* local submatrix for diagonal part */
  Vec         w, left, right, leftwork, rightwork;
  PetscScalar scale;
} Mat_Normal;

PetscErrorCode MatScale_NormalHermitian(Mat inA, PetscScalar scale)
{
  Mat_Normal *a = (Mat_Normal *)inA->data;

  PetscFunctionBegin;
  a->scale *= scale;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_NormalHermitian(Mat inA, Vec left, Vec right)
{
  Mat_Normal *a = (Mat_Normal *)inA->data;

  PetscFunctionBegin;
  if (left) {
    if (!a->left) {
      PetscCall(VecDuplicate(left, &a->left));
      PetscCall(VecCopy(left, a->left));
    } else {
      PetscCall(VecPointwiseMult(a->left, left, a->left));
    }
  }
  if (right) {
    if (!a->right) {
      PetscCall(VecDuplicate(right, &a->right));
      PetscCall(VecCopy(right, a->right));
    } else {
      PetscCall(VecPointwiseMult(a->right, right, a->right));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_NormalHermitian(Mat mat, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])
{
  Mat_Normal *a = (Mat_Normal *)mat->data;
  Mat         B = a->A, *suba;
  IS         *row;
  PetscInt    M;

  PetscFunctionBegin;
  PetscCheck(!a->left && !a->right && irow == icol, PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Not implemented");
  if (scall != MAT_REUSE_MATRIX) PetscCall(PetscCalloc1(n, submat));
  PetscCall(MatGetSize(B, &M, NULL));
  PetscCall(PetscMalloc1(n, &row));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, M, 0, 1, &row[0]));
  PetscCall(ISSetIdentity(row[0]));
  for (M = 1; M < n; ++M) row[M] = row[0];
  PetscCall(MatCreateSubMatrices(B, n, row, icol, MAT_INITIAL_MATRIX, &suba));
  for (M = 0; M < n; ++M) {
    PetscCall(MatCreateNormalHermitian(suba[M], *submat + M));
    ((Mat_Normal *)(*submat)[M]->data)->scale = a->scale;
  }
  PetscCall(ISDestroy(&row[0]));
  PetscCall(PetscFree(row));
  PetscCall(MatDestroySubMatrices(n, &suba));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPermute_NormalHermitian(Mat A, IS rowp, IS colp, Mat *B)
{
  Mat_Normal *a = (Mat_Normal *)A->data;
  Mat         C, Aa = a->A;
  IS          row;

  PetscFunctionBegin;
  PetscCheck(rowp == colp, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_INCOMP, "Row permutation and column permutation must be the same");
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)Aa), Aa->rmap->n, Aa->rmap->rstart, 1, &row));
  PetscCall(ISSetIdentity(row));
  PetscCall(MatPermute(Aa, row, colp, &C));
  PetscCall(ISDestroy(&row));
  PetscCall(MatCreateNormalHermitian(C, B));
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_NormalHermitian(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_Normal *a = (Mat_Normal *)A->data;
  Mat         C;

  PetscFunctionBegin;
  PetscCheck(!a->left && !a->right, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Not implemented");
  PetscCall(MatDuplicate(a->A, op, &C));
  PetscCall(MatCreateNormalHermitian(C, B));
  PetscCall(MatDestroy(&C));
  if (op == MAT_COPY_VALUES) ((Mat_Normal *)(*B)->data)->scale = a->scale;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_NormalHermitian(Mat A, Mat B, MatStructure str)
{
  Mat_Normal *a = (Mat_Normal *)A->data, *b = (Mat_Normal *)B->data;

  PetscFunctionBegin;
  PetscCheck(!a->left && !a->right, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Not implemented");
  PetscCall(MatCopy(a->A, b->A, str));
  b->scale = a->scale;
  PetscCall(VecDestroy(&b->left));
  PetscCall(VecDestroy(&b->right));
  PetscCall(VecDestroy(&b->leftwork));
  PetscCall(VecDestroy(&b->rightwork));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_NormalHermitian(Mat N, Vec x, Vec y)
{
  Mat_Normal *Na = (Mat_Normal *)N->data;
  Vec         in;

  PetscFunctionBegin;
  in = x;
  if (Na->right) {
    if (!Na->rightwork) PetscCall(VecDuplicate(Na->right, &Na->rightwork));
    PetscCall(VecPointwiseMult(Na->rightwork, Na->right, in));
    in = Na->rightwork;
  }
  PetscCall(MatMult(Na->A, in, Na->w));
  PetscCall(MatMultHermitianTranspose(Na->A, Na->w, y));
  if (Na->left) PetscCall(VecPointwiseMult(y, Na->left, y));
  PetscCall(VecScale(y, Na->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianAdd_Normal(Mat N, Vec v1, Vec v2, Vec v3)
{
  Mat_Normal *Na = (Mat_Normal *)N->data;
  Vec         in;

  PetscFunctionBegin;
  in = v1;
  if (Na->right) {
    if (!Na->rightwork) PetscCall(VecDuplicate(Na->right, &Na->rightwork));
    PetscCall(VecPointwiseMult(Na->rightwork, Na->right, in));
    in = Na->rightwork;
  }
  PetscCall(MatMult(Na->A, in, Na->w));
  PetscCall(VecScale(Na->w, Na->scale));
  if (Na->left) {
    PetscCall(MatMultHermitianTranspose(Na->A, Na->w, v3));
    PetscCall(VecPointwiseMult(v3, Na->left, v3));
    PetscCall(VecAXPY(v3, 1.0, v2));
  } else {
    PetscCall(MatMultHermitianTransposeAdd(Na->A, Na->w, v2, v3));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTranspose_Normal(Mat N, Vec x, Vec y)
{
  Mat_Normal *Na = (Mat_Normal *)N->data;
  Vec         in;

  PetscFunctionBegin;
  in = x;
  if (Na->left) {
    if (!Na->leftwork) PetscCall(VecDuplicate(Na->left, &Na->leftwork));
    PetscCall(VecPointwiseMult(Na->leftwork, Na->left, in));
    in = Na->leftwork;
  }
  PetscCall(MatMult(Na->A, in, Na->w));
  PetscCall(MatMultHermitianTranspose(Na->A, Na->w, y));
  if (Na->right) PetscCall(VecPointwiseMult(y, Na->right, y));
  PetscCall(VecScale(y, Na->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTransposeAdd_Normal(Mat N, Vec v1, Vec v2, Vec v3)
{
  Mat_Normal *Na = (Mat_Normal *)N->data;
  Vec         in;

  PetscFunctionBegin;
  in = v1;
  if (Na->left) {
    if (!Na->leftwork) PetscCall(VecDuplicate(Na->left, &Na->leftwork));
    PetscCall(VecPointwiseMult(Na->leftwork, Na->left, in));
    in = Na->leftwork;
  }
  PetscCall(MatMult(Na->A, in, Na->w));
  PetscCall(VecScale(Na->w, Na->scale));
  if (Na->right) {
    PetscCall(MatMultHermitianTranspose(Na->A, Na->w, v3));
    PetscCall(VecPointwiseMult(v3, Na->right, v3));
    PetscCall(VecAXPY(v3, 1.0, v2));
  } else {
    PetscCall(MatMultHermitianTransposeAdd(Na->A, Na->w, v2, v3));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_NormalHermitian(Mat N)
{
  Mat_Normal *Na = (Mat_Normal *)N->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&Na->A));
  PetscCall(MatDestroy(&Na->D));
  PetscCall(VecDestroy(&Na->w));
  PetscCall(VecDestroy(&Na->left));
  PetscCall(VecDestroy(&Na->right));
  PetscCall(VecDestroy(&Na->leftwork));
  PetscCall(VecDestroy(&Na->rightwork));
  PetscCall(PetscFree(N->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatNormalGetMatHermitian_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatConvert_normalh_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatConvert_normalh_mpiaij_C", NULL));
  PetscFunctionReturn(0);
}

/*
      Slow, nonscalable version
*/
PetscErrorCode MatGetDiagonal_NormalHermitian(Mat N, Vec v)
{
  Mat_Normal        *Na = (Mat_Normal *)N->data;
  Mat                A  = Na->A;
  PetscInt           i, j, rstart, rend, nnz;
  const PetscInt    *cols;
  PetscScalar       *diag, *work, *values;
  const PetscScalar *mvalues;

  PetscFunctionBegin;
  PetscCall(PetscMalloc2(A->cmap->N, &diag, A->cmap->N, &work));
  PetscCall(PetscArrayzero(work, A->cmap->N));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscCall(MatGetRow(A, i, &nnz, &cols, &mvalues));
    for (j = 0; j < nnz; j++) work[cols[j]] += mvalues[j] * PetscConj(mvalues[j]);
    PetscCall(MatRestoreRow(A, i, &nnz, &cols, &mvalues));
  }
  PetscCall(MPIU_Allreduce(work, diag, A->cmap->N, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)N)));
  rstart = N->cmap->rstart;
  rend   = N->cmap->rend;
  PetscCall(VecGetArray(v, &values));
  PetscCall(PetscArraycpy(values, diag + rstart, rend - rstart));
  PetscCall(VecRestoreArray(v, &values));
  PetscCall(PetscFree2(diag, work));
  PetscCall(VecScale(v, Na->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonalBlock_NormalHermitian(Mat N, Mat *D)
{
  Mat_Normal *Na = (Mat_Normal *)N->data;
  Mat         M, A = Na->A;

  PetscFunctionBegin;
  PetscCall(MatGetDiagonalBlock(A, &M));
  PetscCall(MatCreateNormalHermitian(M, &Na->D));
  *D = Na->D;
  PetscFunctionReturn(0);
}

PetscErrorCode MatNormalGetMat_NormalHermitian(Mat A, Mat *M)
{
  Mat_Normal *Aa = (Mat_Normal *)A->data;

  PetscFunctionBegin;
  *M = Aa->A;
  PetscFunctionReturn(0);
}

/*@
      MatNormalHermitianGetMat - Gets the `Mat` object stored inside a `MATNORMALHERMITIAN`

   Logically collective on A

   Input Parameter:
.   A  - the `MATNORMALHERMITIAN` matrix

   Output Parameter:
.   M - the matrix object stored inside A

   Level: intermediate

.seealso: `MATNORMALHERMITIAN`, `MatCreateNormalHermitian()`
@*/
PetscErrorCode MatNormalHermitianGetMat(Mat A, Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  PetscValidPointer(M, 2);
  PetscUseMethod(A, "MatNormalGetMatHermitian_C", (Mat, Mat *), (A, M));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_NormalHermitian_AIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat_Normal *Aa = (Mat_Normal *)A->data;
  Mat         B, conjugate;
  PetscInt    m, n, M, N;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(A, &m, &n));
  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
    PetscCall(MatProductReplaceMats(Aa->A, Aa->A, NULL, B));
  } else {
    PetscCall(MatProductCreate(Aa->A, Aa->A, NULL, &B));
    PetscCall(MatProductSetType(B, MATPRODUCT_AtB));
    PetscCall(MatProductSetFromOptions(B));
    PetscCall(MatProductSymbolic(B));
    PetscCall(MatSetOption(B, !PetscDefined(USE_COMPLEX) ? MAT_SYMMETRIC : MAT_HERMITIAN, PETSC_TRUE));
  }
  if (PetscDefined(USE_COMPLEX)) {
    PetscCall(MatDuplicate(Aa->A, MAT_COPY_VALUES, &conjugate));
    PetscCall(MatConjugate(conjugate));
    PetscCall(MatProductReplaceMats(conjugate, Aa->A, NULL, B));
  }
  PetscCall(MatProductNumeric(B));
  if (PetscDefined(USE_COMPLEX)) PetscCall(MatDestroy(&conjugate));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &B));
  } else if (reuse == MAT_INITIAL_MATRIX) *newmat = B;
  PetscCall(MatConvert(*newmat, MATAIJ, MAT_INPLACE_MATRIX, newmat));
  PetscFunctionReturn(0);
}

/*@
      MatCreateNormalHermitian - Creates a new matrix object `MATNORMALHERMITIAN` that behaves like (A*)'*A.

   Collective on A

   Input Parameter:
.   A  - the (possibly rectangular complex) matrix

   Output Parameter:
.   N - the matrix that represents (A*)'*A

   Level: intermediate

   Note:
    The product (A*)'*A is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product, `MatMult()`, by first multiplying by
          A and then (A*)'

.seealso: `MATNORMAL`, `MATNORMALHERMITIAN`, `MatNormalHermitianGetMat()`
@*/
PetscErrorCode MatCreateNormalHermitian(Mat A, Mat *N)
{
  PetscInt    m, n;
  Mat_Normal *Na;
  VecType     vtype;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), N));
  PetscCall(MatSetSizes(*N, n, n, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N, MATNORMALHERMITIAN));
  PetscCall(PetscLayoutReference(A->cmap, &(*N)->rmap));
  PetscCall(PetscLayoutReference(A->cmap, &(*N)->cmap));

  PetscCall(PetscNew(&Na));
  (*N)->data = (void *)Na;
  PetscCall(PetscObjectReference((PetscObject)A));
  Na->A     = A;
  Na->scale = 1.0;

  PetscCall(MatCreateVecs(A, NULL, &Na->w));

  (*N)->ops->destroy           = MatDestroy_NormalHermitian;
  (*N)->ops->mult              = MatMult_NormalHermitian;
  (*N)->ops->multtranspose     = MatMultHermitianTranspose_Normal;
  (*N)->ops->multtransposeadd  = MatMultHermitianTransposeAdd_Normal;
  (*N)->ops->multadd           = MatMultHermitianAdd_Normal;
  (*N)->ops->getdiagonal       = MatGetDiagonal_NormalHermitian;
  (*N)->ops->getdiagonalblock  = MatGetDiagonalBlock_NormalHermitian;
  (*N)->ops->scale             = MatScale_NormalHermitian;
  (*N)->ops->diagonalscale     = MatDiagonalScale_NormalHermitian;
  (*N)->ops->createsubmatrices = MatCreateSubMatrices_NormalHermitian;
  (*N)->ops->permute           = MatPermute_NormalHermitian;
  (*N)->ops->duplicate         = MatDuplicate_NormalHermitian;
  (*N)->ops->copy              = MatCopy_NormalHermitian;
  (*N)->assembled              = PETSC_TRUE;
  (*N)->preallocated           = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)(*N), "MatNormalGetMatHermitian_C", MatNormalGetMat_NormalHermitian));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N), "MatConvert_normalh_seqaij_C", MatConvert_NormalHermitian_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N), "MatConvert_normalh_mpiaij_C", MatConvert_NormalHermitian_AIJ));
  PetscCall(MatSetOption(*N, MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatSetVecType(*N, vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N, A->boundtocpu));
#endif
  PetscFunctionReturn(0);
}
