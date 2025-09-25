#include <../src/mat/impls/shell/shell.h> /*I "petscmat.h" I*/

typedef struct {
  PetscErrorCode (*numeric)(Mat);
  PetscErrorCode (*destroy)(void *);
  Mat            B;
  PetscScalar    scale;
  PetscBool      conjugate;
  PetscContainer container;
  void          *stash;
} MatProductData;

static PetscErrorCode DestroyMatProductData(void *ptr)
{
  MatProductData *data = (MatProductData *)ptr;

  PetscFunctionBegin;
  if (data->stash) PetscCall((*data->destroy)(data->stash));
  if (data->conjugate) PetscCall(MatDestroy(&data->B));
  PetscCall(PetscContainerDestroy(&data->container));
  PetscCall(PetscFree(data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_HT(Mat D)
{
  Mat_Product    *product;
  Mat             B;
  MatProductData *data;
  PetscContainer  container;

  PetscFunctionBegin;
  MatCheckProduct(D, 1);
  PetscCheck(D->product->data, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "Product data empty");
  product = D->product;
  PetscCall(PetscObjectQuery((PetscObject)D, "MatProductData", (PetscObject *)&container));
  PetscCheck(container, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "MatProductData missing");
  PetscCall(PetscContainerGetPointer(container, (void **)&data));
  B    = product->B;
  data = (MatProductData *)product->data;
  if (data->conjugate) {
    PetscCall(MatCopy(product->B, data->B, SAME_NONZERO_PATTERN));
    PetscCall(MatConjugate(data->B));
    product->B = data->B;
  }
  product->data = data->stash;
  PetscCall((*data->numeric)(D));
  if (data->conjugate) {
    PetscCall(MatConjugate(D));
    product->B = B;
  }
  PetscCall(MatScale(D, data->scale));
  product->data = data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_HT(Mat D)
{
  Mat_Product    *product;
  Mat             B;
  MatProductData *data;
  PetscContainer  container;

  PetscFunctionBegin;
  MatCheckProduct(D, 1);
  product = D->product;
  B       = product->B;
  if (D->ops->productsymbolic == MatProductSymbolic_HT) {
    PetscCheck(!product->data, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "Product data not empty");
    PetscCall(PetscObjectQuery((PetscObject)D, "MatProductData", (PetscObject *)&container));
    PetscCheck(container, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "MatProductData missing");
    PetscCall(PetscContainerGetPointer(container, (void **)&data));
    PetscCall(MatProductSetFromOptions(D));
    if (data->conjugate) {
      PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &data->B));
      product->B = data->B;
    }
    PetscCall(MatProductSymbolic(D));
    data->numeric          = D->ops->productnumeric;
    data->destroy          = product->destroy;
    data->stash            = product->data;
    D->ops->productnumeric = MatProductNumeric_HT;
    product->destroy       = DestroyMatProductData;
    if (data->conjugate) product->B = B;
    product->data = data;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_HT(Mat D)
{
  Mat             A, B, C, Ain, Bin, Cin;
  PetscScalar     scale = 1.0, vscale;
  PetscBool       Aistrans, Bistrans, Cistrans, conjugate = PETSC_FALSE;
  PetscInt        Atrans, Btrans, Ctrans;
  PetscContainer  container = NULL;
  MatProductData *data;
  MatProductType  ptype;

  PetscFunctionBegin;
  MatCheckProduct(D, 1);
  A = D->product->A;
  B = D->product->B;
  C = D->product->C;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATHERMITIANTRANSPOSEVIRTUAL, &Aistrans));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATHERMITIANTRANSPOSEVIRTUAL, &Bistrans));
  PetscCall(PetscObjectTypeCompare((PetscObject)C, MATHERMITIANTRANSPOSEVIRTUAL, &Cistrans));
  PetscCheck(Aistrans || Bistrans || Cistrans, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "This should not happen");
  Atrans = 0;
  Ain    = A;
  while (Aistrans) {
    Atrans++;
    PetscCall(MatShellGetScalingShifts(Ain, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, &vscale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    conjugate = (PetscBool)!conjugate;
    scale *= vscale;
    PetscCall(MatHermitianTransposeGetMat(Ain, &Ain));
    PetscCall(PetscObjectTypeCompare((PetscObject)Ain, MATHERMITIANTRANSPOSEVIRTUAL, &Aistrans));
  }
  Btrans = 0;
  Bin    = B;
  while (Bistrans) {
    Btrans++;
    PetscCall(MatShellGetScalingShifts(Bin, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, &vscale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    scale *= vscale;
    PetscCall(MatHermitianTransposeGetMat(Bin, &Bin));
    PetscCall(PetscObjectTypeCompare((PetscObject)Bin, MATHERMITIANTRANSPOSEVIRTUAL, &Bistrans));
  }
  Ctrans = 0;
  Cin    = C;
  while (Cistrans) {
    Ctrans++;
    PetscCall(MatShellGetScalingShifts(Cin, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, &vscale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    scale *= vscale;
    PetscCall(MatHermitianTransposeGetMat(Cin, &Cin));
    PetscCall(PetscObjectTypeCompare((PetscObject)Cin, MATHERMITIANTRANSPOSEVIRTUAL, &Cistrans));
  }
  Atrans = Atrans % 2;
  Btrans = Btrans % 2;
  Ctrans = Ctrans % 2;
  ptype  = D->product->type; /* same product type by default */
  if (Ain->symmetric == PETSC_BOOL3_TRUE) Atrans = 0;
  if (Bin->symmetric == PETSC_BOOL3_TRUE) Btrans = 0;
  if (Cin && Cin->symmetric == PETSC_BOOL3_TRUE) Ctrans = 0;

  if (Atrans || Btrans || Ctrans) {
    PetscCheck(!PetscDefined(USE_COMPLEX) || (!Btrans && !Ctrans), PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "No support for complex Hermitian transpose matrices");
    if ((PetscDefined(USE_COMPLEX) && Atrans) || scale != 1.0) {
      PetscCall(PetscObjectQuery((PetscObject)D, "MatProductData", (PetscObject *)&container));
      if (!container) {
        PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)D), &container));
        PetscCall(PetscNew(&data));
        data->scale     = scale;
        data->conjugate = (PetscBool)Atrans;
        data->container = container;
        PetscCall(PetscContainerSetPointer(container, data));
        PetscCall(PetscObjectCompose((PetscObject)D, "MatProductData", (PetscObject)container));
      }
    }
    ptype = MATPRODUCT_UNSPECIFIED;
    switch (D->product->type) {
    case MATPRODUCT_AB:
      if (Atrans && Btrans) { /* At * Bt we do not have support for this */
        /* TODO custom implementation ? */
      } else if (Atrans) { /* At * B */
        ptype = MATPRODUCT_AtB;
      } else { /* A * Bt */
        ptype = MATPRODUCT_ABt;
      }
      break;
    case MATPRODUCT_AtB:
      if (Atrans && Btrans) { /* A * Bt */
        ptype = MATPRODUCT_ABt;
      } else if (Atrans) { /* A * B */
        ptype = MATPRODUCT_AB;
      } else { /* At * Bt we do not have support for this */
        /* TODO custom implementation ? */
      }
      break;
    case MATPRODUCT_ABt:
      if (Atrans && Btrans) { /* At * B */
        ptype = MATPRODUCT_AtB;
      } else if (Atrans) { /* At * Bt we do not have support for this */
        /* TODO custom implementation ? */
      } else { /* A * B */
        ptype = MATPRODUCT_AB;
      }
      break;
    case MATPRODUCT_PtAP:
      if (Atrans) { /* PtAtP */
        /* TODO custom implementation ? */
      } else { /* RARt */
        ptype = MATPRODUCT_RARt;
      }
      break;
    case MATPRODUCT_RARt:
      if (Atrans) { /* RAtRt */
        /* TODO custom implementation ? */
      } else { /* PtAP */
        ptype = MATPRODUCT_PtAP;
      }
      break;
    case MATPRODUCT_ABC:
      /* TODO custom implementation ? */
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)D), PETSC_ERR_SUP, "ProductType %s is not supported", MatProductTypes[D->product->type]);
    }
  }
  PetscCall(MatProductReplaceMats(Ain, Bin, Cin, D));
  PetscCall(MatProductSetType(D, ptype));
  if (container) D->ops->productsymbolic = MatProductSymbolic_HT;
  else PetscCall(MatProductSetFromOptions(D));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_HT(Mat N, Vec x, Vec y)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatMultHermitianTranspose(A, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultHermitianTranspose_HT(Mat N, Vec x, Vec y)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatMult(A, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_HT_LU(Mat N, Vec b, Vec x)
{
  Mat A;
  Vec w;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(VecDuplicate(b, &w));
  PetscCall(VecCopy(b, w));
  PetscCall(VecConjugate(w));
  PetscCall(MatSolveTranspose(A, w, x));
  PetscCall(VecConjugate(x));
  PetscCall(VecDestroy(&w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveAdd_HT_LU(Mat N, Vec b, Vec y, Vec x)
{
  Mat A;
  Vec v, w;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(VecDuplicate(b, &v));
  PetscCall(VecDuplicate(b, &w));
  PetscCall(VecCopy(y, v));
  PetscCall(VecCopy(b, w));
  PetscCall(VecConjugate(v));
  PetscCall(VecConjugate(w));
  PetscCall(MatSolveTransposeAdd(A, w, v, x));
  PetscCall(VecConjugate(x));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_HT_LU(Mat N, Mat B, Mat X)
{
  Mat A, W;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &W));
  PetscCall(MatConjugate(W));
  PetscCall(MatMatSolveTranspose(A, W, X));
  PetscCall(MatConjugate(X));
  PetscCall(MatDestroy(&W));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactor_HT(Mat N, IS row, IS col, const MatFactorInfo *minfo)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatLUFactor(A, col, row, minfo));
  PetscCall(MatShellSetOperation(N, MATOP_SOLVE, (PetscErrorCodeFn *)MatSolve_HT_LU));
  PetscCall(MatShellSetOperation(N, MATOP_SOLVE_ADD, (PetscErrorCodeFn *)MatSolveAdd_HT_LU));
  PetscCall(MatShellSetOperation(N, MATOP_MAT_SOLVE, (PetscErrorCodeFn *)MatMatSolve_HT_LU));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_HT_Cholesky(Mat N, Vec b, Vec x)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatSolve(A, b, x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveAdd_HT_Cholesky(Mat N, Vec b, Vec y, Vec x)
{
  Mat A;
  Vec v, w;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(VecDuplicate(b, &v));
  PetscCall(VecDuplicate(b, &w));
  PetscCall(VecCopy(y, v));
  PetscCall(VecCopy(b, w));
  PetscCall(VecConjugate(v));
  PetscCall(VecConjugate(w));
  PetscCall(MatSolveTransposeAdd(A, w, v, x));
  PetscCall(VecConjugate(x));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_HT_Cholesky(Mat N, Mat B, Mat X)
{
  Mat A, W;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &W));
  PetscCall(MatConjugate(W));
  PetscCall(MatMatSolveTranspose(A, W, X));
  PetscCall(MatConjugate(X));
  PetscCall(MatDestroy(&W));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactor_HT(Mat N, IS perm, const MatFactorInfo *minfo)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCheck(!PetscDefined(USE_COMPLEX) || A->hermitian == PETSC_BOOL3_TRUE, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Cholesky supported only if original matrix is Hermitian");
  PetscCall(MatCholeskyFactor(A, perm, minfo));
  PetscCall(MatShellSetOperation(N, MATOP_SOLVE, (PetscErrorCodeFn *)MatSolve_HT_Cholesky));
  PetscCall(MatShellSetOperation(N, MATOP_SOLVE_ADD, (PetscErrorCodeFn *)MatSolveAdd_HT_Cholesky));
  PetscCall(MatShellSetOperation(N, MATOP_MAT_SOLVE, (PetscErrorCodeFn *)MatMatSolve_HT_Cholesky));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_HT(Mat F, Mat N, const MatFactorInfo *info)
{
  Mat A, FA;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatShellGetContext(F, &FA));
  PetscCall(MatLUFactorNumeric(FA, A, info));
  PetscCall(MatShellSetOperation(F, MATOP_SOLVE, (PetscErrorCodeFn *)MatSolve_HT_LU));
  PetscCall(MatShellSetOperation(F, MATOP_SOLVE_ADD, (PetscErrorCodeFn *)MatSolveAdd_HT_LU));
  PetscCall(MatShellSetOperation(F, MATOP_MAT_SOLVE, (PetscErrorCodeFn *)MatMatSolve_HT_LU));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_HT(Mat F, Mat N, IS row, IS col, const MatFactorInfo *info)
{
  Mat A, FA;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatShellGetContext(F, &FA));
  PetscCall(MatLUFactorSymbolic(FA, A, row, col, info));
  PetscCall(MatShellSetOperation(F, MATOP_LUFACTOR_NUMERIC, (PetscErrorCodeFn *)MatLUFactorNumeric_HT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorNumeric_HT(Mat F, Mat N, const MatFactorInfo *info)
{
  Mat A, FA;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatShellGetContext(F, &FA));
  PetscCall(MatCholeskyFactorNumeric(FA, A, info));
  PetscCall(MatShellSetOperation(F, MATOP_SOLVE, (PetscErrorCodeFn *)MatSolve_HT_Cholesky));
  PetscCall(MatShellSetOperation(F, MATOP_SOLVE_ADD, (PetscErrorCodeFn *)MatSolveAdd_HT_Cholesky));
  PetscCall(MatShellSetOperation(F, MATOP_MAT_SOLVE, (PetscErrorCodeFn *)MatMatSolve_HT_Cholesky));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_HT(Mat F, Mat N, IS perm, const MatFactorInfo *info)
{
  Mat A, FA;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatShellGetContext(F, &FA));
  PetscCall(MatCholeskyFactorSymbolic(FA, A, perm, info));
  PetscCall(MatShellSetOperation(F, MATOP_CHOLESKY_FACTOR_NUMERIC, (PetscErrorCodeFn *)MatCholeskyFactorNumeric_HT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_HT(Mat N, MatSolverType type, MatFactorType ftype, Mat *F)
{
  Mat A, FA;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatGetFactor(A, type, ftype, &FA));
  PetscCall(MatCreateTranspose(FA, F));
  if (ftype == MAT_FACTOR_LU) PetscCall(MatShellSetOperation(*F, MATOP_LUFACTOR_SYMBOLIC, (PetscErrorCodeFn *)MatLUFactorSymbolic_HT));
  else if (ftype == MAT_FACTOR_CHOLESKY) {
    PetscCheck(!PetscDefined(USE_COMPLEX) || A->hermitian == PETSC_BOOL3_TRUE, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Cholesky supported only if original matrix is Hermitian");
    PetscCall(MatPropagateSymmetryOptions(A, FA));
    PetscCall(MatShellSetOperation(*F, MATOP_CHOLESKY_FACTOR_SYMBOLIC, (PetscErrorCodeFn *)MatCholeskyFactorSymbolic_HT));
  } else SETERRQ(PetscObjectComm((PetscObject)N), PETSC_ERR_SUP, "Support for factor type %s not implemented in MATTRANSPOSEVIRTUAL", MatFactorTypes[ftype]);
  (*F)->factortype = ftype;
  PetscCall(MatDestroy(&FA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_HT(Mat N)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatHermitianTransposeGetMat_C", NULL));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatTransposeGetMat_C", NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatProductSetFromOptions_anytype_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatShellSetContext_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatFactorGetSolverType_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetInfo_HT(Mat N, MatInfoType flag, MatInfo *info)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatGetInfo(A, flag, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorGetSolverType_HT(Mat N, MatSolverType *type)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatFactorGetSolverType(A, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_HT(Mat N, MatDuplicateOption op, Mat *m)
{
  Mat A, C;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatDuplicate(A, op, &C));
  PetscCall(MatCreateHermitianTranspose(C, m));
  if (op == MAT_COPY_VALUES) PetscCall(MatCopy(N, *m, SAME_NONZERO_PATTERN));
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHasOperation_HT(Mat mat, MatOperation op, PetscBool *has)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &A));
  *has = PETSC_FALSE;
  if (op == MATOP_MULT || op == MATOP_MULT_ADD) {
    PetscCall(MatHasOperation(A, MATOP_MULT_HERMITIAN_TRANSPOSE, has));
    if (!*has) PetscCall(MatHasOperation(A, MATOP_MULT_TRANSPOSE, has));
  } else if (op == MATOP_MULT_HERMITIAN_TRANSPOSE || op == MATOP_MULT_HERMITIAN_TRANS_ADD || op == MATOP_MULT_TRANSPOSE || op == MATOP_MULT_TRANSPOSE_ADD) {
    PetscCall(MatHasOperation(A, MATOP_MULT, has));
  } else if (((void **)mat->ops)[op]) *has = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHermitianTransposeGetMat_HT(Mat N, Mat *M)
{
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHermitianTransposeGetMat - Gets the `Mat` object stored inside a `MATHERMITIANTRANSPOSEVIRTUAL`

  Logically Collective

  Input Parameter:
. A - the `MATHERMITIANTRANSPOSEVIRTUAL` matrix

  Output Parameter:
. M - the matrix object stored inside A

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MATHERMITIANTRANSPOSEVIRTUAL`, `MatCreateHermitianTranspose()`
@*/
PetscErrorCode MatHermitianTransposeGetMat(Mat A, Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  PetscAssertPointer(M, 2);
  PetscUseMethod(A, "MatHermitianTransposeGetMat_C", (Mat, Mat *), (A, M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_HT(Mat N, Vec v)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatGetDiagonal(A, v));
  PetscCall(VecConjugate(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_HT(Mat A, Mat B, MatStructure str)
{
  Mat a, b;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(MatShellGetContext(B, &b));
  PetscCall(MatCopy(a, b, str));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvert_HT(Mat N, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat         A;
  PetscScalar vscale = 1.0, vshift = 0.0;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatHasOperation(A, MATOP_HERMITIAN_TRANSPOSE, &flg));
  if (flg || N->ops->getrow) { /* if this condition is false, MatConvert_Shell() will be called in MatConvert_Basic(), so the following checks are not needed */
    PetscCall(MatShellGetScalingShifts(N, &vshift, &vscale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  }
  if (flg) {
    Mat B;

    PetscCall(MatHermitianTranspose(A, MAT_INITIAL_MATRIX, &B));
    if (reuse != MAT_INPLACE_MATRIX) {
      PetscCall(MatConvert(B, newtype, reuse, newmat));
      PetscCall(MatDestroy(&B));
    } else {
      PetscCall(MatConvert(B, newtype, MAT_INPLACE_MATRIX, &B));
      PetscCall(MatHeaderReplace(N, &B));
    }
  } else { /* use basic converter as fallback */
    flg = (PetscBool)(N->ops->getrow != NULL);
    PetscCall(MatConvert_Basic(N, newtype, reuse, newmat));
  }
  if (flg) {
    PetscCall(MatScale(*newmat, vscale));
    PetscCall(MatShift(*newmat, vshift));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATHERMITIANTRANSPOSEVIRTUAL - "hermitiantranspose" - A matrix type that represents a virtual transpose of a matrix

  Level: advanced

  Developer Notes:
  This is implemented on top of `MATSHELL` to get support for scaling and shifting without requiring duplicate code

  Users can not call `MatShellSetOperation()` operations on this class, there is some error checking for that incorrect usage

.seealso: [](ch_matrices), `Mat`, `MATTRANSPOSEVIRTUAL`, `Mat`, `MatCreateHermitianTranspose()`, `MatCreateTranspose()`
M*/

/*@
  MatCreateHermitianTranspose - Creates a new matrix object of `MatType` `MATHERMITIANTRANSPOSEVIRTUAL` that behaves like A'*

  Collective

  Input Parameter:
. A - the (possibly rectangular) matrix

  Output Parameter:
. N - the matrix that represents A'*

  Level: intermediate

  Note:
  The Hermitian transpose A' is NOT actually formed! Rather the new matrix
  object performs the matrix-vector product, `MatMult()`, by using the `MatMultHermitianTranspose()` on
  the original matrix

.seealso: [](ch_matrices), `Mat`, `MatCreateNormal()`, `MatMult()`, `MatMultHermitianTranspose()`, `MatCreate()`,
          `MATTRANSPOSEVIRTUAL`, `MatCreateTranspose()`, `MatHermitianTransposeGetMat()`, `MATNORMAL`, `MATNORMALHERMITIAN`
@*/
PetscErrorCode MatCreateHermitianTranspose(Mat A, Mat *N)
{
  VecType vtype;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), N));
  PetscCall(PetscLayoutReference(A->rmap, &((*N)->cmap)));
  PetscCall(PetscLayoutReference(A->cmap, &((*N)->rmap)));
  PetscCall(MatSetType(*N, MATSHELL));
  PetscCall(MatShellSetContext(*N, A));
  PetscCall(PetscObjectReference((PetscObject)A));

  PetscCall(MatSetBlockSizes(*N, A->cmap->bs, A->rmap->bs));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatSetVecType(*N, vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N, A->boundtocpu));
#endif
  PetscCall(MatSetUp(*N));

  PetscCall(MatShellSetOperation(*N, MATOP_DESTROY, (PetscErrorCodeFn *)MatDestroy_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT, (PetscErrorCodeFn *)MatMult_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT_HERMITIAN_TRANSPOSE, (PetscErrorCodeFn *)MatMultHermitianTranspose_HT));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(MatShellSetOperation(*N, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMultHermitianTranspose_HT));
#endif
  PetscCall(MatShellSetOperation(*N, MATOP_LUFACTOR, (PetscErrorCodeFn *)MatLUFactor_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_CHOLESKYFACTOR, (PetscErrorCodeFn *)MatCholeskyFactor_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_GET_FACTOR, (PetscErrorCodeFn *)MatGetFactor_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_GETINFO, (PetscErrorCodeFn *)MatGetInfo_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_DUPLICATE, (PetscErrorCodeFn *)MatDuplicate_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_HAS_OPERATION, (PetscErrorCodeFn *)MatHasOperation_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_GET_DIAGONAL, (PetscErrorCodeFn *)MatGetDiagonal_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_COPY, (PetscErrorCodeFn *)MatCopy_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_CONVERT, (PetscErrorCodeFn *)MatConvert_HT));

  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatHermitianTransposeGetMat_C", MatHermitianTransposeGetMat_HT));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatTransposeGetMat_C", MatHermitianTransposeGetMat_HT));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatProductSetFromOptions_anytype_C", MatProductSetFromOptions_HT));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatFactorGetSolverType_C", MatFactorGetSolverType_HT));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContext_C", MatShellSetContext_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContextDestroy_C", MatShellSetContextDestroy_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetManageScalingShifts_C", MatShellSetManageScalingShifts_Immutable));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N, MATHERMITIANTRANSPOSEVIRTUAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
