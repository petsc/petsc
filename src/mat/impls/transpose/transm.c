#include <../src/mat/impls/shell/shell.h> /*I "petscmat.h" I*/

static PetscErrorCode MatMult_Transpose(Mat N, Vec x, Vec y)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatMultTranspose(A, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_Transpose(Mat N, Vec x, Vec y)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatMult(A, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Transpose(Mat N)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatTransposeGetMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatProductSetFromOptions_anytype_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatShellSetContext_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_Transpose(Mat N, MatDuplicateOption op, Mat *m)
{
  Mat A, C;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatDuplicate(A, op, &C));
  PetscCall(MatCreateTranspose(C, m));
  PetscCall(MatDestroy(&C));
  if (op == MAT_COPY_VALUES) PetscCall(MatCopy(N, *m, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHasOperation_Transpose(Mat mat, MatOperation op, PetscBool *has)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &A));
  *has = PETSC_FALSE;
  if (op == MATOP_MULT || op == MATOP_MULT_ADD) {
    PetscCall(MatHasOperation(A, MATOP_MULT_TRANSPOSE, has));
  } else if (op == MATOP_MULT_TRANSPOSE || op == MATOP_MULT_TRANSPOSE_ADD) {
    PetscCall(MatHasOperation(A, MATOP_MULT, has));
  } else if (((void **)mat->ops)[op]) *has = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_Transpose(Mat D)
{
  Mat            A, B, C, Ain, Bin, Cin;
  PetscBool      Aistrans, Bistrans, Cistrans;
  PetscInt       Atrans, Btrans, Ctrans;
  MatProductType ptype;

  PetscFunctionBegin;
  MatCheckProduct(D, 1);
  A = D->product->A;
  B = D->product->B;
  C = D->product->C;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATTRANSPOSEVIRTUAL, &Aistrans));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATTRANSPOSEVIRTUAL, &Bistrans));
  PetscCall(PetscObjectTypeCompare((PetscObject)C, MATTRANSPOSEVIRTUAL, &Cistrans));
  PetscCheck(Aistrans || Bistrans || Cistrans, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "This should not happen");
  Atrans = 0;
  Ain    = A;
  while (Aistrans) {
    Atrans++;
    PetscCall(MatTransposeGetMat(Ain, &Ain));
    PetscCall(PetscObjectTypeCompare((PetscObject)Ain, MATTRANSPOSEVIRTUAL, &Aistrans));
  }
  Btrans = 0;
  Bin    = B;
  while (Bistrans) {
    Btrans++;
    PetscCall(MatTransposeGetMat(Bin, &Bin));
    PetscCall(PetscObjectTypeCompare((PetscObject)Bin, MATTRANSPOSEVIRTUAL, &Bistrans));
  }
  Ctrans = 0;
  Cin    = C;
  while (Cistrans) {
    Ctrans++;
    PetscCall(MatTransposeGetMat(Cin, &Cin));
    PetscCall(PetscObjectTypeCompare((PetscObject)Cin, MATTRANSPOSEVIRTUAL, &Cistrans));
  }
  Atrans = Atrans % 2;
  Btrans = Btrans % 2;
  Ctrans = Ctrans % 2;
  ptype  = D->product->type; /* same product type by default */
  if (Ain->symmetric == PETSC_BOOL3_TRUE) Atrans = 0;
  if (Bin->symmetric == PETSC_BOOL3_TRUE) Btrans = 0;
  if (Cin && Cin->symmetric == PETSC_BOOL3_TRUE) Ctrans = 0;

  if (Atrans || Btrans || Ctrans) {
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
  PetscCall(MatProductSetFromOptions(D));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_Transpose(Mat N, Vec v)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatGetDiagonal(A, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_Transpose(Mat A, Mat B, MatStructure str)
{
  Mat a, b;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(MatShellGetContext(B, &b));
  PetscCall(MatCopy(a, b, str));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvert_Transpose(Mat N, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat       A;
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatHasOperation(A, MATOP_TRANSPOSE, &flg));
  if (flg) {
    Mat B;

    PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &B));
    if (reuse != MAT_INPLACE_MATRIX) {
      PetscCall(MatConvert(B, newtype, reuse, newmat));
      PetscCall(MatDestroy(&B));
    } else {
      PetscCall(MatConvert(B, newtype, MAT_INPLACE_MATRIX, &B));
      PetscCall(MatHeaderReplace(N, &B));
    }
  } else { /* use basic converter as fallback */
    PetscCall(MatConvert_Basic(N, newtype, reuse, newmat));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTransposeGetMat_Transpose(Mat N, Mat *M)
{
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatTransposeGetMat - Gets the `Mat` object stored inside a `MATTRANSPOSEVIRTUAL`

  Logically Collective

  Input Parameter:
. A - the `MATTRANSPOSEVIRTUAL` matrix

  Output Parameter:
. M - the matrix object stored inside `A`

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MATTRANSPOSEVIRTUAL`, `MatCreateTranspose()`
@*/
PetscErrorCode MatTransposeGetMat(Mat A, Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  PetscAssertPointer(M, 2);
  PetscUseMethod(A, "MatTransposeGetMat_C", (Mat, Mat *), (A, M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATTRANSPOSEVIRTUAL - "transpose" - A matrix type that represents a virtual transpose of a matrix

  Level: advanced

  Developer Notes:
  This is implemented on top of `MATSHELL` to get support for scaling and shifting without requiring duplicate code

  Users can not call `MatShellSetOperation()` operations on this class, there is some error checking for that incorrect usage

.seealso: [](ch_matrices), `Mat`, `MATHERMITIANTRANSPOSEVIRTUAL`, `Mat`, `MatCreateHermitianTranspose()`, `MatCreateTranspose()`,
          `MATNORMALHERMITIAN`, `MATNORMAL`
M*/

/*@
  MatCreateTranspose - Creates a new matrix `MATTRANSPOSEVIRTUAL` object that behaves like A'

  Collective

  Input Parameter:
. A - the (possibly rectangular) matrix

  Output Parameter:
. N - the matrix that represents A'

  Level: intermediate

  Note:
  The transpose A' is NOT actually formed! Rather the new matrix
  object performs the matrix-vector product by using the `MatMultTranspose()` on
  the original matrix

.seealso: [](ch_matrices), `Mat`, `MATTRANSPOSEVIRTUAL`, `MatCreateNormal()`, `MatMult()`, `MatMultTranspose()`, `MatCreate()`,
          `MATNORMALHERMITIAN`
@*/
PetscErrorCode MatCreateTranspose(Mat A, Mat *N)
{
  VecType vtype;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), N));
  PetscCall(PetscLayoutReference(A->rmap, &((*N)->cmap)));
  PetscCall(PetscLayoutReference(A->cmap, &((*N)->rmap)));
  PetscCall(MatSetType(*N, MATSHELL));
  PetscCall(MatShellSetContext(*N, A));
  PetscCall(PetscObjectReference((PetscObject)A));

  PetscCall(MatSetBlockSizes(*N, PetscAbs(A->cmap->bs), PetscAbs(A->rmap->bs)));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatSetVecType(*N, vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N, A->boundtocpu));
#endif
  PetscCall(MatSetUp(*N));

  PetscCall(MatShellSetOperation(*N, MATOP_DESTROY, (void (*)(void))MatDestroy_Transpose));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT, (void (*)(void))MatMult_Transpose));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT_TRANSPOSE, (void (*)(void))MatMultTranspose_Transpose));
  PetscCall(MatShellSetOperation(*N, MATOP_DUPLICATE, (void (*)(void))MatDuplicate_Transpose));
  PetscCall(MatShellSetOperation(*N, MATOP_HAS_OPERATION, (void (*)(void))MatHasOperation_Transpose));
  PetscCall(MatShellSetOperation(*N, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiagonal_Transpose));
  PetscCall(MatShellSetOperation(*N, MATOP_COPY, (void (*)(void))MatCopy_Transpose));
  PetscCall(MatShellSetOperation(*N, MATOP_CONVERT, (void (*)(void))MatConvert_Transpose));

  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatTransposeGetMat_C", MatTransposeGetMat_Transpose));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatProductSetFromOptions_anytype_C", MatProductSetFromOptions_Transpose));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContext_C", MatShellSetContext_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContextDestroy_C", MatShellSetContextDestroy_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetManageScalingShifts_C", MatShellSetManageScalingShifts_Immutable));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N, MATTRANSPOSEVIRTUAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
