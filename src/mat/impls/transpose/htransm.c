#include <../src/mat/impls/shell/shell.h> /*I "petscmat.h" I*/

static PetscErrorCode MatProductSetFromOptions_HT(Mat D)
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
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATHERMITIANTRANSPOSEVIRTUAL, &Aistrans));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATHERMITIANTRANSPOSEVIRTUAL, &Bistrans));
  PetscCall(PetscObjectTypeCompare((PetscObject)C, MATHERMITIANTRANSPOSEVIRTUAL, &Cistrans));
  PetscCheck(Aistrans || Bistrans || Cistrans, PetscObjectComm((PetscObject)D), PETSC_ERR_PLIB, "This should not happen");
  Atrans = 0;
  Ain    = A;
  while (Aistrans) {
    Atrans++;
    PetscCall(MatShellGetScalingShifts(Ain, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    PetscCall(MatHermitianTransposeGetMat(Ain, &Ain));
    PetscCall(PetscObjectTypeCompare((PetscObject)Ain, MATHERMITIANTRANSPOSEVIRTUAL, &Aistrans));
  }
  Btrans = 0;
  Bin    = B;
  while (Bistrans) {
    Btrans++;
    PetscCall(MatShellGetScalingShifts(Bin, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    PetscCall(MatHermitianTransposeGetMat(Bin, &Bin));
    PetscCall(PetscObjectTypeCompare((PetscObject)Bin, MATHERMITIANTRANSPOSEVIRTUAL, &Bistrans));
  }
  Ctrans = 0;
  Cin    = C;
  while (Cistrans) {
    Ctrans++;
    PetscCall(MatShellGetScalingShifts(Cin, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
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
    PetscCheck(!PetscDefined(USE_COMPLEX), PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "No support for complex Hermitian transpose matrices");
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_HT(Mat N, MatDuplicateOption op, Mat *m)
{
  Mat A, C;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(N, &A));
  PetscCall(MatDuplicate(A, op, &C));
  PetscCall(MatCreateHermitianTranspose(C, m));
  PetscCall(MatDestroy(&C));
  if (op == MAT_COPY_VALUES) PetscCall(MatCopy(N, *m, SAME_NONZERO_PATTERN));
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

  PetscCall(MatSetBlockSizes(*N, PetscAbs(A->cmap->bs), PetscAbs(A->rmap->bs)));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatSetVecType(*N, vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N, A->boundtocpu));
#endif
  PetscCall(MatSetUp(*N));

  PetscCall(MatShellSetOperation(*N, MATOP_DESTROY, (void (*)(void))MatDestroy_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT, (void (*)(void))MatMult_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_MULT_HERMITIAN_TRANSPOSE, (void (*)(void))MatMultHermitianTranspose_HT));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(MatShellSetOperation(*N, MATOP_MULT_TRANSPOSE, (void (*)(void))MatMultHermitianTranspose_HT));
#endif
  PetscCall(MatShellSetOperation(*N, MATOP_DUPLICATE, (void (*)(void))MatDuplicate_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_HAS_OPERATION, (void (*)(void))MatHasOperation_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiagonal_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_COPY, (void (*)(void))MatCopy_HT));
  PetscCall(MatShellSetOperation(*N, MATOP_CONVERT, (void (*)(void))MatConvert_HT));

  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatHermitianTransposeGetMat_C", MatHermitianTransposeGetMat_HT));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatTransposeGetMat_C", MatHermitianTransposeGetMat_HT));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatProductSetFromOptions_anytype_C", MatProductSetFromOptions_HT));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContext_C", MatShellSetContext_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetContextDestroy_C", MatShellSetContextDestroy_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)*N, "MatShellSetManageScalingShifts_C", MatShellSetManageScalingShifts_Immutable));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N, MATHERMITIANTRANSPOSEVIRTUAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
