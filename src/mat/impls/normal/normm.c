#include <../src/mat/impls/shell/shell.h> /*I "petscmat.h" I*/

typedef struct {
  Mat A;
  Mat D; /* local submatrix for diagonal part */
  Vec w;
} Mat_Normal;

static PetscErrorCode MatIncreaseOverlap_Normal(Mat A, PetscInt is_max, IS is[], PetscInt ov)
{
  Mat_Normal *a;
  Mat         pattern;

  PetscFunctionBegin;
  PetscCheck(ov >= 0, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_OUTOFRANGE, "Negative overlap specified");
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(MatProductCreate(a->A, a->A, NULL, &pattern));
  PetscCall(MatProductSetType(pattern, MATPRODUCT_AtB));
  PetscCall(MatProductSetFromOptions(pattern));
  PetscCall(MatProductSymbolic(pattern));
  PetscCall(MatIncreaseOverlap(pattern, is_max, is, ov));
  PetscCall(MatDestroy(&pattern));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrices_Normal(Mat mat, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])
{
  Mat_Normal *a;
  Mat         B, *suba;
  IS         *row;
  PetscScalar shift, scale;
  PetscInt    M;

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
    PetscCall(MatCreateNormal(suba[M], *submat + M));
    PetscCall(MatShift((*submat)[M], shift));
    PetscCall(MatScale((*submat)[M], scale));
  }
  PetscCall(ISDestroy(&row[0]));
  PetscCall(PetscFree(row));
  PetscCall(MatDestroySubMatrices(n, &suba));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPermute_Normal(Mat A, IS rowp, IS colp, Mat *B)
{
  Mat_Normal *a;
  Mat         C, Aa;
  IS          row;
  PetscScalar shift, scale;

  PetscFunctionBegin;
  PetscCheck(rowp == colp, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_INCOMP, "Row permutation and column permutation must be the same");
  PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(A, &a));
  Aa = a->A;
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)Aa), Aa->rmap->n, Aa->rmap->rstart, 1, &row));
  PetscCall(ISSetIdentity(row));
  PetscCall(MatPermute(Aa, row, colp, &C));
  PetscCall(ISDestroy(&row));
  PetscCall(MatCreateNormal(C, B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatShift(*B, shift));
  PetscCall(MatScale(*B, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_Normal(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_Normal *a;
  Mat         C;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(MatDuplicate(a->A, op, &C));
  PetscCall(MatCreateNormal(C, B));
  PetscCall(MatDestroy(&C));
  if (op == MAT_COPY_VALUES) PetscCall(MatCopy(A, *B, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_Normal(Mat A, Mat B, MatStructure str)
{
  Mat_Normal *a, *b;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(MatShellGetContext(B, &b));
  PetscCall(MatCopy(a->A, b->A, str));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_Normal(Mat N, Vec x, Vec y)
{
  Mat_Normal *Na;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &Na));
  PetscCall(MatMult(Na->A, x, Na->w));
  PetscCall(MatMultTranspose(Na->A, Na->w, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Normal(Mat N)
{
  Mat_Normal *Na;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &Na));
  PetscCall(MatDestroy(&Na->A));
  PetscCall(MatDestroy(&Na->D));
  PetscCall(VecDestroy(&Na->w));
  PetscCall(PetscFree(Na));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatNormalGetMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatConvert_normal_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatConvert_normal_mpiaij_C", NULL));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatConvert_normal_hypre_C", NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatProductSetFromOptions_normal_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatProductSetFromOptions_normal_mpidense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatShellSetContext_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
      Slow, nonscalable version
*/
static PetscErrorCode MatGetDiagonal_Normal(Mat N, Vec v)
{
  Mat_Normal        *Na;
  Mat                A;
  PetscInt           i, j, rstart, rend, nnz;
  const PetscInt    *cols;
  PetscScalar       *work, *values;
  const PetscScalar *mvalues;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &Na));
  A = Na->A;
  PetscCall(PetscMalloc1(A->cmap->N, &work));
  PetscCall(PetscArrayzero(work, A->cmap->N));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscCall(MatGetRow(A, i, &nnz, &cols, &mvalues));
    for (j = 0; j < nnz; j++) work[cols[j]] += mvalues[j] * mvalues[j];
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

static PetscErrorCode MatGetDiagonalBlock_Normal(Mat N, Mat *D)
{
  Mat_Normal *Na;
  Mat         M, A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &Na));
  A = Na->A;
  PetscCall(MatGetDiagonalBlock(A, &M));
  PetscCall(MatCreateNormal(M, &Na->D));
  *D = Na->D;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNormalGetMat_Normal(Mat A, Mat *M)
{
  Mat_Normal *Aa;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &Aa));
  *M = Aa->A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatNormalGetMat - Gets the `Mat` object stored inside a `MATNORMAL`

  Logically Collective

  Input Parameter:
. A - the `MATNORMAL` matrix

  Output Parameter:
. M - the matrix object stored inside `A`

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MATNORMAL`, `MATNORMALHERMITIAN`, `MatCreateNormal()`
@*/
PetscErrorCode MatNormalGetMat(Mat A, Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  PetscAssertPointer(M, 2);
  PetscUseMethod(A, "MatNormalGetMat_C", (Mat, Mat *), (A, M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvert_Normal_AIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat_Normal *Aa;
  Mat         B;
  Vec         left, right, dshift;
  PetscScalar scale, shift;
  PetscInt    m, n, M, N;

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
    PetscCall(MatSetOption(B, MAT_SYMMETRIC, PETSC_TRUE));
  }
  PetscCall(MatProductNumeric(B));
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
static PetscErrorCode MatConvert_Normal_HYPRE(Mat A, MatType type, MatReuse reuse, Mat *B)
{
  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatConvert(A, MATAIJ, reuse, B));
    PetscCall(MatConvert(*B, type, MAT_INPLACE_MATRIX, B));
  } else PetscCall(MatConvert_Basic(A, type, reuse, B)); /* fall back to basic convert */
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

typedef struct {
  Mat work[2];
} Normal_Dense;

static PetscErrorCode MatProductNumeric_Normal_Dense(Mat C)
{
  Mat           A, B;
  Normal_Dense *contents;
  Mat_Normal   *a;
  Vec           right;
  PetscScalar  *array, scale;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  A = C->product->A;
  B = C->product->B;
  PetscCall(MatShellGetContext(A, &a));
  contents = (Normal_Dense *)C->product->data;
  PetscCheck(contents, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data empty");
  PetscCall(MatShellGetScalingShifts(A, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, &right, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  if (right) {
    PetscCall(MatCopy(B, C, SAME_NONZERO_PATTERN));
    PetscCall(MatDiagonalScale(C, right, NULL));
  }
  PetscCall(MatProductNumeric(contents->work[0]));
  PetscCall(MatDenseGetArrayWrite(C, &array));
  PetscCall(MatDensePlaceArray(contents->work[1], array));
  PetscCall(MatProductNumeric(contents->work[1]));
  PetscCall(MatDenseRestoreArrayWrite(C, &array));
  PetscCall(MatDenseResetArray(contents->work[1]));
  PetscCall(MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(C, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNormal_DenseDestroy(void *ctx)
{
  Normal_Dense *contents = (Normal_Dense *)ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(contents->work));
  PetscCall(MatDestroy(contents->work + 1));
  PetscCall(PetscFree(contents));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_Normal_Dense(Mat C)
{
  Mat           A, B;
  Normal_Dense *contents = NULL;
  Mat_Normal   *a;
  Vec           right;
  PetscScalar  *array, scale;
  PetscInt      n, N, m, M;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  A = C->product->A;
  B = C->product->B;
  PetscCall(MatShellGetScalingShifts(A, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, &right, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(MatGetLocalSize(C, &m, &n));
  PetscCall(MatGetSize(C, &M, &N));
  if (m == PETSC_DECIDE || n == PETSC_DECIDE || M == PETSC_DECIDE || N == PETSC_DECIDE) {
    PetscCall(MatGetLocalSize(B, NULL, &n));
    PetscCall(MatGetSize(B, NULL, &N));
    PetscCall(MatGetLocalSize(A, &m, NULL));
    PetscCall(MatGetSize(A, &M, NULL));
    PetscCall(MatSetSizes(C, m, n, M, N));
  }
  PetscCall(MatSetType(C, ((PetscObject)B)->type_name));
  PetscCall(MatSetUp(C));
  PetscCall(PetscNew(&contents));
  C->product->data    = contents;
  C->product->destroy = MatNormal_DenseDestroy;
  if (right) PetscCall(MatProductCreate(a->A, C, NULL, contents->work));
  else PetscCall(MatProductCreate(a->A, B, NULL, contents->work));
  PetscCall(MatProductSetType(contents->work[0], MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(contents->work[0]));
  PetscCall(MatProductSymbolic(contents->work[0]));
  PetscCall(MatProductCreate(a->A, contents->work[0], NULL, contents->work + 1));
  PetscCall(MatProductSetType(contents->work[1], MATPRODUCT_AtB));
  PetscCall(MatProductSetFromOptions(contents->work[1]));
  PetscCall(MatProductSymbolic(contents->work[1]));
  PetscCall(MatDenseGetArrayWrite(C, &array));
  PetscCall(MatSeqDenseSetPreallocation(contents->work[1], array));
  PetscCall(MatMPIDenseSetPreallocation(contents->work[1], array));
  PetscCall(MatDenseRestoreArrayWrite(C, &array));
  C->ops->productnumeric = MatProductNumeric_Normal_Dense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_Normal_Dense(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_AB) C->ops->productsymbolic = MatProductSymbolic_Normal_Dense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATNORMAL - a matrix that behaves like A'*A for `MatMult()` while only containing A

  Level: intermediate

  Developer Notes:
  This is implemented on top of `MATSHELL` to get support for scaling and shifting without requiring duplicate code

  Users can not call `MatShellSetOperation()` operations on this class, there is some error checking for that incorrect usage

.seealso: [](ch_matrices), `Mat`, `MatCreateNormal()`, `MatMult()`, `MatNormalGetMat()`, `MATNORMALHERMITIAN`, `MatCreateNormalHermitian()`
M*/

/*@
  MatCreateNormal - Creates a new `MATNORMAL` matrix object that behaves like $A^T A$.

  Collective

  Input Parameter:
. A - the (possibly rectangular) matrix

  Output Parameter:
. N - the matrix that represents $A^T A $

  Level: intermediate

  Notes:
  The product $A^T A$ is NOT actually formed! Rather the new matrix
  object performs the matrix-vector product, `MatMult()`, by first multiplying by
  $A$ and then $A^T$

  If `MatGetFactor()` is called on this matrix with `MAT_FACTOR_QR` then the inner matrix `A` is used for the factorization

.seealso: [](ch_matrices), `Mat`, `MATNORMAL`, `MatMult()`, `MatNormalGetMat()`, `MATNORMALHERMITIAN`, `MatCreateNormalHermitian()`
@*/
PetscErrorCode MatCreateNormal(Mat A, Mat *N)
{
  Mat_Normal *Na;
  VecType     vtype;

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
  PetscCall(MatShellSetOperation(*N, MATOP_DESTROY, (PetscErrorCodeFn *)MatDestroy_Normal));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT, (PetscErrorCodeFn *)MatMult_Normal));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMult_Normal));
  PetscCall(MatShellSetOperation(*N, MATOP_DUPLICATE, (PetscErrorCodeFn *)MatDuplicate_Normal));
  PetscCall(MatShellSetOperation(*N, MATOP_GET_DIAGONAL, (PetscErrorCodeFn *)MatGetDiagonal_Normal));
  PetscCall(MatShellSetOperation(*N, MATOP_GET_DIAGONAL_BLOCK, (PetscErrorCodeFn *)MatGetDiagonalBlock_Normal));
  PetscCall(MatShellSetOperation(*N, MATOP_COPY, (PetscErrorCodeFn *)MatCopy_Normal));
  (*N)->ops->increaseoverlap   = MatIncreaseOverlap_Normal;
  (*N)->ops->createsubmatrices = MatCreateSubMatrices_Normal;
  (*N)->ops->permute           = MatPermute_Normal;

  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatNormalGetMat_C", MatNormalGetMat_Normal));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatConvert_normal_seqaij_C", MatConvert_Normal_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatConvert_normal_mpiaij_C", MatConvert_Normal_AIJ));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatConvert_normal_hypre_C", MatConvert_Normal_HYPRE));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatProductSetFromOptions_normal_seqdense_C", MatProductSetFromOptions_Normal_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatProductSetFromOptions_normal_mpidense_C", MatProductSetFromOptions_Normal_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContext_C", MatShellSetContext_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContextDestroy_C", MatShellSetContextDestroy_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetManageScalingShifts_C", MatShellSetManageScalingShifts_Immutable));
  PetscCall(MatSetOption(*N, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatSetVecType(*N, vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N, A->boundtocpu));
#endif
  PetscCall(MatSetUp(*N));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N, MATNORMAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
