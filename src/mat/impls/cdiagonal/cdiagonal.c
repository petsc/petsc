#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/

typedef struct {
  PetscScalar diag;
} Mat_ConstantDiagonal;

static PetscErrorCode MatAXPY_ConstantDiagonal(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  Mat_ConstantDiagonal *yctx = (Mat_ConstantDiagonal *)Y->data;
  Mat_ConstantDiagonal *xctx = (Mat_ConstantDiagonal *)X->data;

  PetscFunctionBegin;
  yctx->diag += a * xctx->diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatEqual_ConstantDiagonal(Mat Y, Mat X, PetscBool *equal)
{
  Mat_ConstantDiagonal *yctx = (Mat_ConstantDiagonal *)Y->data;
  Mat_ConstantDiagonal *xctx = (Mat_ConstantDiagonal *)X->data;

  PetscFunctionBegin;
  *equal = (yctx->diag == xctx->diag) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRow_ConstantDiagonal(Mat A, PetscInt row, PetscInt *ncols, PetscInt *cols[], PetscScalar *vals[])
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)A->data;

  PetscFunctionBegin;
  if (ncols) *ncols = 1;
  if (cols) {
    PetscCall(PetscMalloc1(1, cols));
    (*cols)[0] = row;
  }
  if (vals) {
    PetscCall(PetscMalloc1(1, vals));
    (*vals)[0] = ctx->diag;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRestoreRow_ConstantDiagonal(Mat A, PetscInt row, PetscInt *ncols, PetscInt *cols[], PetscScalar *vals[])
{
  PetscFunctionBegin;
  if (cols) PetscCall(PetscFree(*cols));
  if (vals) PetscCall(PetscFree(*vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_ConstantDiagonal(Mat mat, Vec v1, Vec v2, Vec v3)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)mat->data;

  PetscFunctionBegin;
  if (v2 == v3) PetscCall(VecAXPBY(v3, ctx->diag, 1.0, v1));
  else PetscCall(VecAXPBYPCZ(v3, ctx->diag, 1.0, 0.0, v1, v2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTransposeAdd_ConstantDiagonal(Mat mat, Vec v1, Vec v2, Vec v3)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)mat->data;

  PetscFunctionBegin;
  if (v2 == v3) PetscCall(VecAXPBY(v3, PetscConj(ctx->diag), 1.0, v1));
  else PetscCall(VecAXPBYPCZ(v3, PetscConj(ctx->diag), 1.0, 0.0, v1, v2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNorm_ConstantDiagonal(Mat A, NormType type, PetscReal *nrm)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCheck(type == NORM_FROBENIUS || type == NORM_2 || type == NORM_1 || type == NORM_INFINITY, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Unsupported norm");
  *nrm = PetscAbsScalar(ctx->diag);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrices_ConstantDiagonal(Mat A, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatCreateSubMatrices(B, n, irow, icol, scall, submat));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_ConstantDiagonal(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_ConstantDiagonal *actx = (Mat_ConstantDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(MatSetBlockSizesFromMats(*B, A, A));
  PetscCall(MatSetType(*B, MATCONSTANTDIAGONAL));
  PetscCall(PetscLayoutReference(A->rmap, &(*B)->rmap));
  PetscCall(PetscLayoutReference(A->cmap, &(*B)->cmap));
  if (op == MAT_COPY_VALUES) {
    Mat_ConstantDiagonal *bctx = (Mat_ConstantDiagonal *)(*B)->data;
    bctx->diag                 = actx->diag;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_ConstantDiagonal(Mat mat)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(mat->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConstantDiagonalGetConstant_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatProductSetFromOptions_constantdiagonal_constantdiagonal_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatProductSetFromOptions_constantdiagonal_diagonal_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatProductSetFromOptions_diagonal_constantdiagonal_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatProductSetFromOptions_anytype_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_ConstantDiagonal(Mat J, PetscViewer viewer)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)J->data;
  PetscBool             isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) PetscFunctionReturn(PETSC_SUCCESS);
    if (PetscImaginaryPart(ctx->diag) == 0) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Diagonal value: %g\n", (double)PetscRealPart(ctx->diag)));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Diagonal value: %g + i %g\n", (double)PetscRealPart(ctx->diag), (double)PetscImaginaryPart(ctx->diag)));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_ConstantDiagonal(Mat J, Vec x, Vec y)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)J->data;

  PetscFunctionBegin;
  PetscCall(VecAXPBY(y, ctx->diag, 0.0, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_ConstantDiagonal(Mat J, Vec x, Vec y)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)J->data;

  PetscFunctionBegin;
  PetscCall(VecAXPBY(y, PetscConj(ctx->diag), 0.0, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_ConstantDiagonal(Mat J, Vec x)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)J->data;

  PetscFunctionBegin;
  PetscCall(VecSet(x, ctx->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatShift_ConstantDiagonal(Mat Y, PetscScalar a)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)Y->data;

  PetscFunctionBegin;
  ctx->diag += a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_ConstantDiagonal(Mat Y, PetscScalar a)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)Y->data;

  PetscFunctionBegin;
  ctx->diag *= a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_ConstantDiagonal(Mat Y)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)Y->data;

  PetscFunctionBegin;
  ctx->diag = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConjugate_ConstantDiagonal(Mat Y)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)Y->data;

  PetscFunctionBegin;
  ctx->diag = PetscConj(ctx->diag);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTranspose_ConstantDiagonal(Mat A, MatReuse reuse, Mat *matout)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)A->data;

  PetscFunctionBegin;
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscLayout tmplayout = A->rmap;

    A->rmap = A->cmap;
    A->cmap = tmplayout;
  } else {
    if (reuse == MAT_INITIAL_MATRIX) {
      PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)A), A->cmap->n, A->rmap->n, A->cmap->N, A->rmap->N, ctx->diag, matout));
    } else {
      PetscCall(MatZeroEntries(*matout));
      PetscCall(MatShift(*matout, ctx->diag));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetRandom_ConstantDiagonal(Mat A, PetscRandom rand)
{
  PetscMPIInt           rank;
  MPI_Comm              comm;
  PetscScalar           v   = 0.0;
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)A->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (!rank) PetscCall(PetscRandomGetValue(rand, &v));
  PetscCallMPI(MPI_Bcast(&v, 1, MPIU_SCALAR, 0, comm));
  ctx->diag = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_ConstantDiagonal(Mat matin, Vec b, Vec x)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)matin->data;

  PetscFunctionBegin;
  if (ctx->diag == 0.0) matin->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  else matin->factorerrortype = MAT_FACTOR_NOERROR;
  PetscCall(VecAXPBY(x, 1.0 / ctx->diag, 0.0, b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSOR_ConstantDiagonal(Mat matin, Vec x, PetscReal omega, MatSORType flag, PetscReal fshift, PetscInt its, PetscInt lits, Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_ConstantDiagonal(matin, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetInfo_ConstantDiagonal(Mat A, MatInfoType flag, MatInfo *info)
{
  PetscFunctionBegin;
  info->block_size   = 1.0;
  info->nz_allocated = 1.0;
  info->nz_used      = 1.0;
  info->nz_unneeded  = 0.0;
  info->assemblies   = A->num_ass;
  info->mallocs      = 0.0;
  info->memory       = 0; /* REVIEW ME */
  if (A->factortype) {
    info->fill_ratio_given  = 1.0;
    info->fill_ratio_needed = 1.0;
    info->factor_mallocs    = 0.0;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateConstantDiagonal - Creates a matrix with a uniform value along the diagonal

  Collective

  Input Parameters:
+ comm - MPI communicator
. m    - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
. n    - This value should be the same as the local size used in creating the
       x vector for the matrix-vector product y = Ax. (or `PETSC_DECIDE` to have
       calculated if `N` is given) For square matrices n is almost always `m`.
. M    - number of global rows (or `PETSC_DETERMINE` to have calculated if m is given)
. N    - number of global columns (or `PETSC_DETERMINE` to have calculated if n is given)
- diag - the diagonal value

  Output Parameter:
. J - the diagonal matrix

  Level: advanced

  Notes:
  Only supports square matrices with the same number of local rows and columns

.seealso: [](ch_matrices), `Mat`, `MatDestroy()`, `MATCONSTANTDIAGONAL`, `MatScale()`, `MatShift()`, `MatMult()`, `MatGetDiagonal()`, `MatGetFactor()`, `MatSolve()`
@*/
PetscErrorCode MatCreateConstantDiagonal(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscScalar diag, Mat *J)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, J));
  PetscCall(MatSetSizes(*J, m, n, M, N));
  PetscCall(MatSetType(*J, MATCONSTANTDIAGONAL));
  PetscCall(MatShift(*J, diag));
  PetscCall(MatSetUp(*J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatConstantDiagonalGetConstant - Get the scalar constant of a constant diagonal matrix

  Not collective

  Input Parameter:
. mat - a `MATCONSTANTDIAGONAL`

  Output Parameter:
. value - the scalar value

  Level: developer

.seealso: [](ch_matrices), `Mat`, `MatDestroy()`, `MATCONSTANTDIAGONAL`
@*/
PetscErrorCode MatConstantDiagonalGetConstant(Mat mat, PetscScalar *value)
{
  PetscFunctionBegin;
  PetscUseMethod(mat, "MatConstantDiagonalGetConstant_C", (Mat, PetscScalar *), (mat, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConstantDiagonalGetConstant_ConstantDiagonal(Mat mat, PetscScalar *value)
{
  Mat_ConstantDiagonal *ctx = (Mat_ConstantDiagonal *)mat->data;

  PetscFunctionBegin;
  *value = ctx->diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PtAP for constantdiagonal * constantdiagonal: C = alpha * beta^2 * I */
static PetscErrorCode MatProductNumeric_PtAP_ConstDiag_ConstDiag(Mat C)
{
  Mat                   A = C->product->A, P = C->product->B;
  Mat_ConstantDiagonal *a = (Mat_ConstantDiagonal *)A->data, *p = (Mat_ConstantDiagonal *)P->data, *c = (Mat_ConstantDiagonal *)C->data;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  c->diag = a->diag * p->diag * p->diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_PtAP_ConstDiag_ConstDiag(Mat C)
{
  Mat P = C->product->B;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatSetSizes(C, P->cmap->n, P->cmap->n, P->cmap->N, P->cmap->N));
  PetscCall(MatSetType(C, MATCONSTANTDIAGONAL));
  C->assembled           = PETSC_TRUE;
  C->ops->productnumeric = MatProductNumeric_PtAP_ConstDiag_ConstDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PtAP for constantdiagonal A and diagonal P: C_i = alpha * p_i^2 */
static PetscErrorCode MatProductNumeric_PtAP_ConstDiag_Diagonal(Mat C)
{
  Mat                   A = C->product->A, P = C->product->B;
  Mat_ConstantDiagonal *a = (Mat_ConstantDiagonal *)A->data;
  Vec                   pdiag, cdiag;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(MatDiagonalGetDiagonal(P, &pdiag));
  PetscCall(MatDiagonalGetDiagonal(C, &cdiag));
  PetscCall(VecPointwiseMult(cdiag, pdiag, pdiag));
  PetscCall(VecScale(cdiag, a->diag));
  PetscCall(MatDiagonalRestoreDiagonal(C, &cdiag));
  PetscCall(MatDiagonalRestoreDiagonal(P, &pdiag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_PtAP_ConstDiag_Diagonal(Mat C)
{
  Mat P = C->product->B;
  Vec pdiag, cdiag;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatSetSizes(C, P->cmap->n, P->cmap->n, P->cmap->N, P->cmap->N));
  PetscCall(MatSetType(C, MATDIAGONAL));
  PetscCall(MatDiagonalGetDiagonal(P, &pdiag));
  PetscCall(VecDuplicate(pdiag, &cdiag));
  PetscCall(MatDiagonalSetDiagonal(C, cdiag));
  PetscCall(VecDestroy(&cdiag));
  PetscCall(MatDiagonalRestoreDiagonal(P, &pdiag));
  C->assembled           = PETSC_TRUE;
  C->ops->productnumeric = MatProductNumeric_PtAP_ConstDiag_Diagonal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PtAP for diagonal A and constantdiagonal P: C_i = beta^2 * a_i */
static PetscErrorCode MatProductNumeric_PtAP_Diagonal_ConstDiag(Mat C)
{
  Mat                   A = C->product->A, P = C->product->B;
  Mat_ConstantDiagonal *p = (Mat_ConstantDiagonal *)P->data;
  Vec                   adiag, cdiag;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(MatDiagonalGetDiagonal(A, &adiag));
  PetscCall(MatDiagonalGetDiagonal(C, &cdiag));
  PetscCall(VecCopy(adiag, cdiag));
  PetscCall(VecScale(cdiag, p->diag * p->diag));
  PetscCall(MatDiagonalRestoreDiagonal(C, &cdiag));
  PetscCall(MatDiagonalRestoreDiagonal(A, &adiag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_PtAP_Diagonal_ConstDiag(Mat C)
{
  Mat A = C->product->A, P = C->product->B;
  Vec adiag, cdiag;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatSetSizes(C, P->cmap->n, P->cmap->n, P->cmap->N, P->cmap->N));
  PetscCall(MatSetType(C, MATDIAGONAL));
  /* Duplicate A's diagonal Vec so C inherits the correct VecType (e.g., Kokkos, CUDA, HIP) */
  PetscCall(MatDiagonalGetDiagonal(A, &adiag));
  PetscCall(VecDuplicate(adiag, &cdiag));
  PetscCall(MatDiagonalSetDiagonal(C, cdiag));
  PetscCall(VecDestroy(&cdiag));
  PetscCall(MatDiagonalRestoreDiagonal(A, &adiag));
  C->assembled           = PETSC_TRUE;
  C->ops->productnumeric = MatProductNumeric_PtAP_Diagonal_ConstDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PtAP for any (non-diagonal) A and constantdiagonal P: C = beta^2 * A */
static PetscErrorCode MatProductNumeric_PtAP_Anytype_ConstDiag(Mat C)
{
  Mat                   A = C->product->A, P = C->product->B;
  Mat_ConstantDiagonal *p = (Mat_ConstantDiagonal *)P->data;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(MatCopy(A, C, SAME_NONZERO_PATTERN));
  PetscCall(MatScale(C, p->diag * p->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_PtAP_Anytype_ConstDiag(Mat C)
{
  Mat          A       = C->product->A;
  Mat_Product *product = C->product;
  Mat          Cwork;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &Cwork));
  C->product = NULL;
  PetscCall(MatHeaderReplace(C, &Cwork));
  C->product             = product;
  C->ops->productnumeric = MatProductNumeric_PtAP_Anytype_ConstDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* PtAP for constantdiagonal A and any non-diagonal P: C = alpha * P^T * P */
typedef struct {
  Mat              PtP;       /* P^T * P result via MatProduct AtB */
  PetscObjectState pnnzstate; /* P's nonzero state when inner symbolic was last built */
} MatProductCtx_PtAP_ConstDiag_Anytype;

static PetscErrorCode MatProductCtxDestroy_PtAP_ConstDiag_Anytype(PetscCtxRt data)
{
  MatProductCtx_PtAP_ConstDiag_Anytype *ctx = *(MatProductCtx_PtAP_ConstDiag_Anytype **)data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&ctx->PtP));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_PtAP_ConstDiag_Anytype(Mat C)
{
  Mat_Product                          *product = C->product;
  Mat                                   A = product->A, P = product->B;
  MatProductCtx_PtAP_ConstDiag_Anytype *ctx = (MatProductCtx_PtAP_ConstDiag_Anytype *)product->data;
  Mat_ConstantDiagonal                 *a   = (Mat_ConstantDiagonal *)A->data;
  PetscObjectState                      pnnzstate;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  /* Rebuild inner symbolic if P's nonzero structure has changed */
  PetscCall(MatGetNonzeroState(P, &pnnzstate));
  if (pnnzstate != ctx->pnnzstate) {
    PetscCall(MatDestroy(&ctx->PtP));
    PetscCall(MatProductCreate(P, P, NULL, &ctx->PtP));
    PetscCall(MatProductSetType(ctx->PtP, MATPRODUCT_AtB));
    PetscCall(MatProductSetFill(ctx->PtP, product->fill));
    PetscCall(MatProductSetFromOptions(ctx->PtP));
    PetscCall(MatProductSymbolic(ctx->PtP));
    ctx->pnnzstate = pnnzstate;
  }
  /* Compute P^T * P */
  PetscCall(MatProductNumeric(ctx->PtP));
  PetscCall(MatCopy(ctx->PtP, C, SAME_NONZERO_PATTERN));
  PetscCall(MatScale(C, a->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_PtAP_ConstDiag_Anytype(Mat C)
{
  Mat_Product                          *product = C->product;
  Mat                                   P       = product->B;
  MatProductCtx_PtAP_ConstDiag_Anytype *ctx;
  Mat                                   Cwork;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(PetscNew(&ctx));

  /* PtP = P^T * P (symbolic) */
  PetscCall(MatProductCreate(P, P, NULL, &ctx->PtP));
  PetscCall(MatProductSetType(ctx->PtP, MATPRODUCT_AtB));
  PetscCall(MatProductSetFill(ctx->PtP, product->fill));
  PetscCall(MatProductSetFromOptions(ctx->PtP));
  PetscCall(MatProductSymbolic(ctx->PtP));

  /* Record P's nonzero state so numeric phase can detect structural changes */
  PetscCall(MatGetNonzeroState(P, &ctx->pnnzstate));

  /* Set up C with the same structure as PtP */
  PetscCall(MatDuplicate(ctx->PtP, MAT_DO_NOT_COPY_VALUES, &Cwork));
  C->product = NULL;
  PetscCall(MatHeaderReplace(C, &Cwork));
  C->product = product;
  PetscCall(MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  product->data          = ctx;
  product->destroy       = MatProductCtxDestroy_PtAP_ConstDiag_Anytype;
  C->ops->productnumeric = MatProductNumeric_PtAP_ConstDiag_Anytype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* AB for MATCONSTANTDIAGONAL A and any non-diagonal B: C = alpha * B */
static PetscErrorCode MatProductNumeric_AB_ConstDiag_Anytype(Mat C)
{
  Mat                   A = C->product->A, B = C->product->B;
  Mat_ConstantDiagonal *a = (Mat_ConstantDiagonal *)A->data;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(MatCopy(B, C, SAME_NONZERO_PATTERN));
  PetscCall(MatScale(C, a->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_AB_ConstDiag_Anytype(Mat C)
{
  Mat          B       = C->product->B;
  Mat_Product *product = C->product;
  Mat          Cwork;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &Cwork));
  C->product = NULL;
  PetscCall(MatHeaderReplace(C, &Cwork));
  C->product             = product;
  C->ops->productnumeric = MatProductNumeric_AB_ConstDiag_Anytype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* AB for any non-diagonal A and MATCONSTANTDIAGONAL B: C = beta * A */
static PetscErrorCode MatProductNumeric_AB_Anytype_ConstDiag(Mat C)
{
  Mat                   A = C->product->A, B = C->product->B;
  Mat_ConstantDiagonal *b = (Mat_ConstantDiagonal *)B->data;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(MatCopy(A, C, SAME_NONZERO_PATTERN));
  PetscCall(MatScale(C, b->diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_AB_Anytype_ConstDiag(Mat C)
{
  Mat          A       = C->product->A;
  Mat_Product *product = C->product;
  Mat          Cwork;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &Cwork));
  C->product = NULL;
  PetscCall(MatHeaderReplace(C, &Cwork));
  C->product             = product;
  C->ops->productnumeric = MatProductNumeric_AB_Anytype_ConstDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* AB for MATCONSTANTDIAGONAL * MATCONSTANTDIAGONAL: C = alpha * beta * I */
static PetscErrorCode MatProductNumeric_AB_ConstDiag_ConstDiag(Mat C)
{
  Mat                   A = C->product->A, B = C->product->B;
  Mat_ConstantDiagonal *a = (Mat_ConstantDiagonal *)A->data, *b = (Mat_ConstantDiagonal *)B->data, *c = (Mat_ConstantDiagonal *)C->data;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  c->diag = a->diag * b->diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_AB_ConstDiag_ConstDiag(Mat C)
{
  Mat A = C->product->A, B = C->product->B;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatSetSizes(C, A->rmap->n, B->cmap->n, A->rmap->N, B->cmap->N));
  PetscCall(MatSetType(C, MATCONSTANTDIAGONAL));
  C->assembled           = PETSC_TRUE;
  C->ops->productnumeric = MatProductNumeric_AB_ConstDiag_ConstDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* AB for MATCONSTANTDIAGONAL A and MATDIAGONAL B: C_i = alpha * b_i */
static PetscErrorCode MatProductNumeric_AB_ConstDiag_Diagonal(Mat C)
{
  Mat                   A = C->product->A, B = C->product->B;
  Mat_ConstantDiagonal *a = (Mat_ConstantDiagonal *)A->data;
  Vec                   bdiag, cdiag;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(MatDiagonalGetDiagonal(B, &bdiag));
  PetscCall(MatDiagonalGetDiagonal(C, &cdiag));
  PetscCall(VecCopy(bdiag, cdiag));
  PetscCall(VecScale(cdiag, a->diag));
  PetscCall(MatDiagonalRestoreDiagonal(C, &cdiag));
  PetscCall(MatDiagonalRestoreDiagonal(B, &bdiag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_AB_ConstDiag_Diagonal(Mat C)
{
  Mat A = C->product->A, B = C->product->B;
  Vec bdiag, cdiag;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatSetSizes(C, A->rmap->n, B->cmap->n, A->rmap->N, B->cmap->N));
  PetscCall(MatSetType(C, MATDIAGONAL));
  /* Duplicate B's diagonal Vec so C inherits the correct VecType (e.g., Kokkos, CUDA, HIP) */
  PetscCall(MatDiagonalGetDiagonal(B, &bdiag));
  PetscCall(VecDuplicate(bdiag, &cdiag));
  PetscCall(MatDiagonalSetDiagonal(C, cdiag));
  PetscCall(VecDestroy(&cdiag));
  PetscCall(MatDiagonalRestoreDiagonal(B, &bdiag));
  C->assembled           = PETSC_TRUE;
  C->ops->productnumeric = MatProductNumeric_AB_ConstDiag_Diagonal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* AB for MATDIAGONAL A and MATCONSTANTDIAGONAL B: C_i = beta * a_i */
static PetscErrorCode MatProductNumeric_AB_Diagonal_ConstDiag(Mat C)
{
  Mat                   A = C->product->A, B = C->product->B;
  Mat_ConstantDiagonal *b = (Mat_ConstantDiagonal *)B->data;
  Vec                   adiag, cdiag;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(MatDiagonalGetDiagonal(A, &adiag));
  PetscCall(MatDiagonalGetDiagonal(C, &cdiag));
  PetscCall(VecCopy(adiag, cdiag));
  PetscCall(VecScale(cdiag, b->diag));
  PetscCall(MatDiagonalRestoreDiagonal(C, &cdiag));
  PetscCall(MatDiagonalRestoreDiagonal(A, &adiag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_AB_Diagonal_ConstDiag(Mat C)
{
  Mat A = C->product->A, B = C->product->B;
  Vec adiag, cdiag;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(MatSetSizes(C, A->rmap->n, B->cmap->n, A->rmap->N, B->cmap->N));
  PetscCall(MatSetType(C, MATDIAGONAL));
  /* Duplicate A's diagonal Vec so C inherits the correct VecType (e.g., Kokkos, CUDA, HIP) */
  PetscCall(MatDiagonalGetDiagonal(A, &adiag));
  PetscCall(VecDuplicate(adiag, &cdiag));
  PetscCall(MatDiagonalSetDiagonal(C, cdiag));
  PetscCall(VecDestroy(&cdiag));
  PetscCall(MatDiagonalRestoreDiagonal(A, &adiag));
  C->assembled           = PETSC_TRUE;
  C->ops->productnumeric = MatProductNumeric_AB_Diagonal_ConstDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_ConstDiag_ConstDiag(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_PtAP) C->ops->productsymbolic = MatProductSymbolic_PtAP_ConstDiag_ConstDiag;
  else if (product->type == MATPRODUCT_AB) C->ops->productsymbolic = MatProductSymbolic_AB_ConstDiag_ConstDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_ConstDiag_Diagonal(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_PtAP) C->ops->productsymbolic = MatProductSymbolic_PtAP_ConstDiag_Diagonal;
  else if (product->type == MATPRODUCT_AB) C->ops->productsymbolic = MatProductSymbolic_AB_ConstDiag_Diagonal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_Diagonal_ConstDiag(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_PtAP) C->ops->productsymbolic = MatProductSymbolic_PtAP_Diagonal_ConstDiag;
  else if (product->type == MATPRODUCT_AB) C->ops->productsymbolic = MatProductSymbolic_AB_Diagonal_ConstDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_ConstDiag_Anytype(Mat C)
{
  Mat_Product *product = C->product;
  PetscBool    Acdiag, Bcdiag;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)product->A, MATCONSTANTDIAGONAL, &Acdiag));
  PetscCall(PetscObjectTypeCompare((PetscObject)product->B, MATCONSTANTDIAGONAL, &Bcdiag));
  if (Acdiag && !Bcdiag && (product->type == MATPRODUCT_PtAP)) C->ops->productsymbolic = MatProductSymbolic_PtAP_ConstDiag_Anytype;
  else if (Bcdiag && !Acdiag && (product->type == MATPRODUCT_PtAP)) C->ops->productsymbolic = MatProductSymbolic_PtAP_Anytype_ConstDiag;
  else if (Acdiag && !Bcdiag && (product->type == MATPRODUCT_AB)) C->ops->productsymbolic = MatProductSymbolic_AB_ConstDiag_Anytype;
  else if (Bcdiag && !Acdiag && (product->type == MATPRODUCT_AB)) C->ops->productsymbolic = MatProductSymbolic_AB_Anytype_ConstDiag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATCONSTANTDIAGONAL - "constant-diagonal" - A diagonal matrix type with a uniform value
   along the diagonal.

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatCreateConstantDiagonal()`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_ConstantDiagonal(Mat A)
{
  Mat_ConstantDiagonal *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  ctx->diag = 0.0;
  A->data   = (void *)ctx;

  A->assembled                   = PETSC_TRUE;
  A->preallocated                = PETSC_TRUE;
  A->structurally_symmetric      = PETSC_BOOL3_TRUE;
  A->structural_symmetry_eternal = PETSC_TRUE;
  A->symmetric                   = PETSC_BOOL3_TRUE;
  if (!PetscDefined(USE_COMPLEX)) A->hermitian = PETSC_BOOL3_TRUE;
  A->symmetry_eternal = PETSC_TRUE;

  A->ops->mult                      = MatMult_ConstantDiagonal;
  A->ops->multadd                   = MatMultAdd_ConstantDiagonal;
  A->ops->multtranspose             = MatMult_ConstantDiagonal;
  A->ops->multtransposeadd          = MatMultAdd_ConstantDiagonal;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_ConstantDiagonal;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_ConstantDiagonal;
  A->ops->solve                     = MatSolve_ConstantDiagonal;
  A->ops->solvetranspose            = MatSolve_ConstantDiagonal;
  A->ops->norm                      = MatNorm_ConstantDiagonal;
  A->ops->createsubmatrices         = MatCreateSubMatrices_ConstantDiagonal;
  A->ops->duplicate                 = MatDuplicate_ConstantDiagonal;
  A->ops->getrow                    = MatGetRow_ConstantDiagonal;
  A->ops->restorerow                = MatRestoreRow_ConstantDiagonal;
  A->ops->sor                       = MatSOR_ConstantDiagonal;
  A->ops->shift                     = MatShift_ConstantDiagonal;
  A->ops->scale                     = MatScale_ConstantDiagonal;
  A->ops->getdiagonal               = MatGetDiagonal_ConstantDiagonal;
  A->ops->view                      = MatView_ConstantDiagonal;
  A->ops->zeroentries               = MatZeroEntries_ConstantDiagonal;
  A->ops->destroy                   = MatDestroy_ConstantDiagonal;
  A->ops->getinfo                   = MatGetInfo_ConstantDiagonal;
  A->ops->equal                     = MatEqual_ConstantDiagonal;
  A->ops->axpy                      = MatAXPY_ConstantDiagonal;
  A->ops->setrandom                 = MatSetRandom_ConstantDiagonal;
  A->ops->conjugate                 = MatConjugate_ConstantDiagonal;
  A->ops->transpose                 = MatTranspose_ConstantDiagonal;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATCONSTANTDIAGONAL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConstantDiagonalGetConstant_C", MatConstantDiagonalGetConstant_ConstantDiagonal));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_constantdiagonal_constantdiagonal_C", MatProductSetFromOptions_ConstDiag_ConstDiag));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_constantdiagonal_diagonal_C", MatProductSetFromOptions_ConstDiag_Diagonal));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_diagonal_constantdiagonal_C", MatProductSetFromOptions_Diagonal_ConstDiag));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_anytype_C", MatProductSetFromOptions_ConstDiag_Anytype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorNumeric_ConstantDiagonal(Mat fact, Mat A, const MatFactorInfo *info)
{
  Mat_ConstantDiagonal *actx = (Mat_ConstantDiagonal *)A->data, *fctx = (Mat_ConstantDiagonal *)fact->data;

  PetscFunctionBegin;
  if (actx->diag == 0.0) fact->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  else fact->factorerrortype = MAT_FACTOR_NOERROR;
  fctx->diag       = 1.0 / actx->diag;
  fact->ops->solve = MatMult_ConstantDiagonal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorSymbolic_LU_ConstantDiagonal(Mat fact, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->ops->lufactornumeric = MatFactorNumeric_ConstantDiagonal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorSymbolic_Cholesky_ConstantDiagonal(Mat fact, Mat A, IS isrow, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->ops->choleskyfactornumeric = MatFactorNumeric_ConstantDiagonal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatGetFactor_constantdiagonal_petsc(Mat A, MatFactorType ftype, Mat *B)
{
  PetscInt n = A->rmap->n, N = A->rmap->N;

  PetscFunctionBegin;
  PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)A), n, n, N, N, 0, B));

  (*B)->factortype                  = ftype;
  (*B)->ops->ilufactorsymbolic      = MatFactorSymbolic_LU_ConstantDiagonal;
  (*B)->ops->lufactorsymbolic       = MatFactorSymbolic_LU_ConstantDiagonal;
  (*B)->ops->iccfactorsymbolic      = MatFactorSymbolic_Cholesky_ConstantDiagonal;
  (*B)->ops->choleskyfactorsymbolic = MatFactorSymbolic_Cholesky_ConstantDiagonal;

  (*B)->ops->shift       = NULL;
  (*B)->ops->scale       = NULL;
  (*B)->ops->mult        = NULL;
  (*B)->ops->sor         = NULL;
  (*B)->ops->zeroentries = NULL;

  PetscCall(PetscFree((*B)->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPETSC, &(*B)->solvertype));
  PetscFunctionReturn(PETSC_SUCCESS);
}
