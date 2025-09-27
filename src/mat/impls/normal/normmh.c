#include <../src/mat/impls/shell/shell.h> /*I "petscmat.h" I*/

typedef struct {
  Mat A;
  Mat D; /* local submatrix for diagonal part */
  Vec w;
} Mat_NormalHermitian;

static PetscErrorCode MatCreateSubMatrices_NormalHermitian(Mat mat, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])
{
  Mat_NormalHermitian *a;
  Mat                  B, *suba;
  IS                  *row;
  PetscScalar          shift, scale;
  PetscInt             M;

  PetscFunctionBegin;
  PetscCheck(irow == icol, PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Not implemented");
  PetscCall(MatShellGetScalingShifts(mat, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(mat, &a));
  B = a->A;
  if (scall != MAT_REUSE_MATRIX) PetscCall(PetscCalloc1(n, submat));
  PetscCall(MatGetSize(B, &M, NULL));
  PetscCall(PetscMalloc1(n, &row));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, M, 0, 1, &row[0]));
  PetscCall(ISSetIdentity(row[0]));
  for (M = 1; M < n; ++M) row[M] = row[0];
  PetscCall(MatCreateSubMatrices(B, n, row, icol, MAT_INITIAL_MATRIX, &suba));
  for (M = 0; M < n; ++M) {
    PetscCall(MatCreateNormalHermitian(suba[M], *submat + M));
    PetscCall(MatShift((*submat)[M], shift));
    PetscCall(MatScale((*submat)[M], scale));
  }
  PetscCall(ISDestroy(&row[0]));
  PetscCall(PetscFree(row));
  PetscCall(MatDestroySubMatrices(n, &suba));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPermute_NormalHermitian(Mat A, IS rowp, IS colp, Mat *B)
{
  Mat_NormalHermitian *a;
  Mat                  C, Aa;
  IS                   row;
  PetscScalar          shift, scale;

  PetscFunctionBegin;
  PetscCheck(rowp == colp, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_INCOMP, "Row permutation and column permutation must be the same");
  PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(A, &a));
  Aa = a->A;
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)Aa), Aa->rmap->n, Aa->rmap->rstart, 1, &row));
  PetscCall(ISSetIdentity(row));
  PetscCall(MatPermute(Aa, row, colp, &C));
  PetscCall(ISDestroy(&row));
  PetscCall(MatCreateNormalHermitian(C, B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatShift(*B, shift));
  PetscCall(MatScale(*B, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_NormalHermitian(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_NormalHermitian *a;
  Mat                  C;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(MatDuplicate(a->A, op, &C));
  PetscCall(MatCreateNormalHermitian(C, B));
  PetscCall(MatDestroy(&C));
  if (op == MAT_COPY_VALUES) PetscCall(MatCopy(A, *B, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_NormalHermitian(Mat A, Mat B, MatStructure str)
{
  Mat_NormalHermitian *a, *b;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(MatShellGetContext(B, &b));
  PetscCall(MatCopy(a->A, b->A, str));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_NormalHermitian(Mat N, Vec x, Vec y)
{
  Mat_NormalHermitian *Na;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &Na));
  PetscCall(MatMult(Na->A, x, Na->w));
  PetscCall(MatMultHermitianTranspose(Na->A, Na->w, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_NormalHermitian(Mat N)
{
  Mat_NormalHermitian *Na;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &Na));
  PetscCall(MatDestroy(&Na->A));
  PetscCall(MatDestroy(&Na->D));
  PetscCall(VecDestroy(&Na->w));
  PetscCall(PetscFree(Na));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatNormalHermitianGetMat_C", NULL));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatNormalGetMat_C", NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatConvert_normalh_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatConvert_normalh_mpiaij_C", NULL));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatConvert_normalh_hypre_C", NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatShellSetContext_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
      Slow, nonscalable version
*/
static PetscErrorCode MatGetDiagonal_NormalHermitian(Mat N, Vec v)
{
  Mat_NormalHermitian *Na;
  Mat                  A;
  PetscInt             i, j, rstart, rend, nnz;
  const PetscInt      *cols;
  PetscScalar         *work, *values;
  const PetscScalar   *mvalues;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &Na));
  A = Na->A;
  PetscCall(PetscMalloc1(A->cmap->N, &work));
  PetscCall(PetscArrayzero(work, A->cmap->N));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscCall(MatGetRow(A, i, &nnz, &cols, &mvalues));
    for (j = 0; j < nnz; j++) work[cols[j]] += mvalues[j] * PetscConj(mvalues[j]);
    PetscCall(MatRestoreRow(A, i, &nnz, &cols, &mvalues));
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, work, A->cmap->N, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)N)));
  rstart = N->cmap->rstart;
  rend   = N->cmap->rend;
  PetscCall(VecGetArray(v, &values));
  PetscCall(PetscArraycpy(values, work + rstart, rend - rstart));
  PetscCall(VecRestoreArray(v, &values));
  PetscCall(PetscFree(work));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonalBlock_NormalHermitian(Mat N, Mat *D)
{
  Mat_NormalHermitian *Na;
  Mat                  M, A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &Na));
  A = Na->A;
  PetscCall(MatGetDiagonalBlock(A, &M));
  PetscCall(MatCreateNormalHermitian(M, &Na->D));
  *D = Na->D;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNormalHermitianGetMat_NormalHermitian(Mat A, Mat *M)
{
  Mat_NormalHermitian *Aa;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &Aa));
  *M = Aa->A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNormalHermitianGetMat - Gets the `Mat` object stored inside a `MATNORMALHERMITIAN`

  Logically Collective

  Input Parameter:
. A - the `MATNORMALHERMITIAN` matrix

  Output Parameter:
. M - the matrix object stored inside `A`

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MATNORMALHERMITIAN`, `MatCreateNormalHermitian()`
@*/
PetscErrorCode MatNormalHermitianGetMat(Mat A, Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  PetscAssertPointer(M, 2);
  PetscUseMethod(A, "MatNormalHermitianGetMat_C", (Mat, Mat *), (A, M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvert_NormalHermitian_AIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat_NormalHermitian *Aa;
  Mat                  B, conjugate;
  Vec                  left, right, dshift;
  PetscScalar          scale, shift;
  PetscInt             m, n, M, N;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &Aa));
  PetscCall(MatShellGetScalingShifts(A, &shift, &scale, &dshift, &left, &right, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
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
  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A, &B));
  else if (reuse == MAT_INITIAL_MATRIX) *newmat = B;
  PetscCall(MatConvert(*newmat, MATAIJ, MAT_INPLACE_MATRIX, newmat));
  PetscCall(MatDiagonalScale(*newmat, left, right));
  PetscCall(MatScale(*newmat, scale));
  PetscCall(MatShift(*newmat, shift));
  if (dshift) PetscCall(MatDiagonalSet(*newmat, dshift, ADD_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_HYPRE)
static PetscErrorCode MatConvert_NormalHermitian_HYPRE(Mat A, MatType type, MatReuse reuse, Mat *B)
{
  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatConvert(A, MATAIJ, reuse, B));
    PetscCall(MatConvert(*B, type, MAT_INPLACE_MATRIX, B));
  } else PetscCall(MatConvert_Basic(A, type, reuse, B)); /* fall back to basic convert */
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*MC
  MATNORMALHERMITIAN - a matrix that behaves like (A*)'*A for `MatMult()` while only containing A

  Level: intermediate

  Developer Notes:
  This is implemented on top of `MATSHELL` to get support for scaling and shifting without requiring duplicate code

  Users can not call `MatShellSetOperation()` operations on this class, there is some error checking for that incorrect usage

.seealso: [](ch_matrices), `Mat`, `MatCreateNormalHermitian()`, `MatMult()`, `MatNormalHermitianGetMat()`, `MATNORMAL`, `MatCreateNormal()`
M*/

/*@
  MatCreateNormalHermitian - Creates a new matrix object `MATNORMALHERMITIAN` that behaves like $A^* A$.

  Collective

  Input Parameter:
. A - the (possibly rectangular complex) matrix

  Output Parameter:
. N - the matrix that represents $ A^* A$

  Level: intermediate

  Note:
  The product $ A^* A$ is NOT actually formed! Rather the new matrix
  object performs the matrix-vector product, `MatMult()`, by first multiplying by
  $A$ and then $A^*$

  If `MatGetFactor()` is called on this matrix with `MAT_FACTOR_QR` then the inner matrix `A` is used for the factorization

.seealso: [](ch_matrices), `Mat`, `MATNORMAL`, `MATNORMALHERMITIAN`, `MatNormalHermitianGetMat()`
@*/
PetscErrorCode MatCreateNormalHermitian(Mat A, Mat *N)
{
  Mat_NormalHermitian *Na;
  VecType              vtype;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), N));
  PetscCall(PetscLayoutReference(A->cmap, &(*N)->rmap));
  PetscCall(PetscLayoutReference(A->cmap, &(*N)->cmap));
  PetscCall(MatSetType(*N, MATSHELL));
  PetscCall(PetscNew(&Na));
  PetscCall(MatShellSetContext(*N, Na));
  PetscCall(PetscObjectReference((PetscObject)A));
  Na->A = A;
  PetscCall(MatCreateVecs(A, NULL, &Na->w));

  PetscCall(MatSetBlockSize(*N, A->cmap->bs));
  PetscCall(MatShellSetOperation(*N, MATOP_DESTROY, (PetscErrorCodeFn *)MatDestroy_NormalHermitian));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT, (PetscErrorCodeFn *)MatMult_NormalHermitian));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT_HERMITIAN_TRANSPOSE, (PetscErrorCodeFn *)MatMult_NormalHermitian));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(MatShellSetOperation(*N, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMult_NormalHermitian));
#endif
  PetscCall(MatShellSetOperation(*N, MATOP_DUPLICATE, (PetscErrorCodeFn *)MatDuplicate_NormalHermitian));
  PetscCall(MatShellSetOperation(*N, MATOP_GET_DIAGONAL, (PetscErrorCodeFn *)MatGetDiagonal_NormalHermitian));
  PetscCall(MatShellSetOperation(*N, MATOP_GET_DIAGONAL_BLOCK, (PetscErrorCodeFn *)MatGetDiagonalBlock_NormalHermitian));
  PetscCall(MatShellSetOperation(*N, MATOP_COPY, (PetscErrorCodeFn *)MatCopy_NormalHermitian));
  (*N)->ops->createsubmatrices = MatCreateSubMatrices_NormalHermitian;
  (*N)->ops->permute           = MatPermute_NormalHermitian;

  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatNormalHermitianGetMat_C", MatNormalHermitianGetMat_NormalHermitian));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatNormalGetMat_C", MatNormalHermitianGetMat_NormalHermitian));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatConvert_normalh_seqaij_C", MatConvert_NormalHermitian_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatConvert_normalh_mpiaij_C", MatConvert_NormalHermitian_AIJ));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatConvert_normalh_hypre_C", MatConvert_NormalHermitian_HYPRE));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContext_C", MatShellSetContext_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContextDestroy_C", MatShellSetContextDestroy_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetManageScalingShifts_C", MatShellSetManageScalingShifts_Immutable));
  PetscCall(MatSetOption(*N, !PetscDefined(USE_COMPLEX) ? MAT_SYMMETRIC : MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatSetVecType(*N, vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N, A->boundtocpu));
#endif
  PetscCall(MatSetUp(*N));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N, MATNORMALHERMITIAN));
  PetscFunctionReturn(PETSC_SUCCESS);
}
