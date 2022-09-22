
#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/

typedef struct {
  Mat A;
} Mat_HT;

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_HermitianTranspose(Mat D)
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
    PetscCall(MatHermitianTransposeGetMat(Ain, &Ain));
    PetscCall(PetscObjectTypeCompare((PetscObject)Ain, MATHERMITIANTRANSPOSEVIRTUAL, &Aistrans));
  }
  Btrans = 0;
  Bin    = B;
  while (Bistrans) {
    Btrans++;
    PetscCall(MatHermitianTransposeGetMat(Bin, &Bin));
    PetscCall(PetscObjectTypeCompare((PetscObject)Bin, MATHERMITIANTRANSPOSEVIRTUAL, &Bistrans));
  }
  Ctrans = 0;
  Cin    = C;
  while (Cistrans) {
    Ctrans++;
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
  PetscFunctionReturn(0);
}
PetscErrorCode MatMult_HT(Mat N, Vec x, Vec y)
{
  Mat_HT *Na = (Mat_HT *)N->data;

  PetscFunctionBegin;
  PetscCall(MatMultHermitianTranspose(Na->A, x, y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_HT(Mat N, Vec v1, Vec v2, Vec v3)
{
  Mat_HT *Na = (Mat_HT *)N->data;

  PetscFunctionBegin;
  PetscCall(MatMultHermitianTransposeAdd(Na->A, v1, v2, v3));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTranspose_HT(Mat N, Vec x, Vec y)
{
  Mat_HT *Na = (Mat_HT *)N->data;

  PetscFunctionBegin;
  PetscCall(MatMult(Na->A, x, y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTransposeAdd_HT(Mat N, Vec v1, Vec v2, Vec v3)
{
  Mat_HT *Na = (Mat_HT *)N->data;

  PetscFunctionBegin;
  PetscCall(MatMultAdd(Na->A, v1, v2, v3));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_HT(Mat N)
{
  Mat_HT *Na = (Mat_HT *)N->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&Na->A));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatHermitianTransposeGetMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatProductSetFromOptions_anytype_C", NULL));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscObjectComposeFunction((PetscObject)N, "MatTransposeGetMat_C", NULL));
#endif
  PetscCall(PetscFree(N->data));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_HT(Mat N, MatDuplicateOption op, Mat *m)
{
  Mat_HT *Na = (Mat_HT *)N->data;

  PetscFunctionBegin;
  if (op == MAT_COPY_VALUES) {
    PetscCall(MatHermitianTranspose(Na->A, MAT_INITIAL_MATRIX, m));
  } else if (op == MAT_DO_NOT_COPY_VALUES) {
    PetscCall(MatDuplicate(Na->A, MAT_DO_NOT_COPY_VALUES, m));
    PetscCall(MatHermitianTranspose(*m, MAT_INPLACE_MATRIX, m));
  } else SETERRQ(PetscObjectComm((PetscObject)N), PETSC_ERR_SUP, "MAT_SHARE_NONZERO_PATTERN not supported for this matrix type");
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateVecs_HT(Mat N, Vec *r, Vec *l)
{
  Mat_HT *Na = (Mat_HT *)N->data;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(Na->A, l, r));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_HT(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  Mat_HT *Ya = (Mat_HT *)Y->data;
  Mat_HT *Xa = (Mat_HT *)X->data;
  Mat     M  = Ya->A;
  Mat     N  = Xa->A;

  PetscFunctionBegin;
  PetscCall(MatAXPY(M, a, N, str));
  PetscFunctionReturn(0);
}

PetscErrorCode MatHermitianTransposeGetMat_HT(Mat N, Mat *M)
{
  Mat_HT *Na = (Mat_HT *)N->data;

  PetscFunctionBegin;
  *M = Na->A;
  PetscFunctionReturn(0);
}

/*@
      MatHermitianTransposeGetMat - Gets the `Mat` object stored inside a `MATHERMITIANTRANSPOSEVIRTUAL`

   Logically collective on Mat

   Input Parameter:
.   A  - the `MATHERMITIANTRANSPOSEVIRTUAL` matrix

   Output Parameter:
.   M - the matrix object stored inside A

   Level: intermediate

.seealso: `MATHERMITIANTRANSPOSEVIRTUAL`, `MatCreateHermitianTranspose()`
@*/
PetscErrorCode MatHermitianTransposeGetMat(Mat A, Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidType(A, 1);
  PetscValidPointer(M, 2);
  PetscUseMethod(A, "MatHermitianTransposeGetMat_C", (Mat, Mat *), (A, M));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_Transpose(Mat);

PetscErrorCode MatGetDiagonal_HT(Mat A, Vec v)
{
  Mat_HT *Na = (Mat_HT *)A->data;

  PetscFunctionBegin;
  PetscCall(MatGetDiagonal(Na->A, v));
  PetscCall(VecConjugate(v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_HT(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat_HT   *Na = (Mat_HT *)A->data;
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(MatHasOperation(Na->A, MATOP_HERMITIAN_TRANSPOSE, &flg));
  if (flg) {
    Mat B;

    PetscCall(MatHermitianTranspose(Na->A, MAT_INITIAL_MATRIX, &B));
    if (reuse != MAT_INPLACE_MATRIX) {
      PetscCall(MatConvert(B, newtype, reuse, newmat));
      PetscCall(MatDestroy(&B));
    } else {
      PetscCall(MatConvert(B, newtype, MAT_INPLACE_MATRIX, &B));
      PetscCall(MatHeaderReplace(A, &B));
    }
  } else { /* use basic converter as fallback */
    PetscCall(MatConvert_Basic(A, newtype, reuse, newmat));
  }
  PetscFunctionReturn(0);
}

/*MC
   MATHERMITIANTRANSPOSEVIRTUAL - "hermitiantranspose" - A matrix type that represents a virtual transpose of a matrix

  Level: advanced

.seealso: `MATTRANSPOSEVIRTUAL`, `Mat`, `MatCreateHermitianTranspose()`, `MatCreateTranspose()`
M*/

/*@
      MatCreateHermitianTranspose - Creates a new matrix object of `MatType` `MATHERMITIANTRANSPOSEVIRTUAL` that behaves like A'*

   Collective on A

   Input Parameter:
.   A  - the (possibly rectangular) matrix

   Output Parameter:
.   N - the matrix that represents A'*

   Level: intermediate

   Note:
    The Hermitian transpose A' is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product, `MatMult()`, by using the `MatMultHermitianTranspose()` on
          the original matrix

.seealso: `MatCreateNormal()`, `MatMult()`, `MatMultHermitianTranspose()`, `MatCreate()`,
          `MATTRANSPOSEVIRTUAL`, `MatCreateTranspose()`, `MatHermitianTransposeGetMat()`
@*/
PetscErrorCode MatCreateHermitianTranspose(Mat A, Mat *N)
{
  PetscInt m, n;
  Mat_HT  *Na;
  VecType  vtype;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), N));
  PetscCall(MatSetSizes(*N, n, m, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(PetscLayoutSetUp((*N)->rmap));
  PetscCall(PetscLayoutSetUp((*N)->cmap));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N, MATHERMITIANTRANSPOSEVIRTUAL));

  PetscCall(PetscNew(&Na));
  (*N)->data = (void *)Na;
  PetscCall(PetscObjectReference((PetscObject)A));
  Na->A = A;

  (*N)->ops->destroy                   = MatDestroy_HT;
  (*N)->ops->mult                      = MatMult_HT;
  (*N)->ops->multadd                   = MatMultAdd_HT;
  (*N)->ops->multhermitiantranspose    = MatMultHermitianTranspose_HT;
  (*N)->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_HT;
#if !defined(PETSC_USE_COMPLEX)
  (*N)->ops->multtranspose    = MatMultHermitianTranspose_HT;
  (*N)->ops->multtransposeadd = MatMultHermitianTransposeAdd_HT;
#endif
  (*N)->ops->duplicate = MatDuplicate_HT;
  (*N)->ops->getvecs   = MatCreateVecs_HT;
  (*N)->ops->axpy      = MatAXPY_HT;
#if !defined(PETSC_USE_COMPLEX)
  (*N)->ops->productsetfromoptions = MatProductSetFromOptions_Transpose;
#endif
  (*N)->ops->getdiagonal = MatGetDiagonal_HT;
  (*N)->ops->convert     = MatConvert_HT;
  (*N)->assembled        = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)(*N), "MatHermitianTransposeGetMat_C", MatHermitianTransposeGetMat_HT));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N), "MatProductSetFromOptions_anytype_C", MatProductSetFromOptions_HermitianTranspose));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N), "MatTransposeGetMat_C", MatHermitianTransposeGetMat_HT));
#endif
  PetscCall(MatSetBlockSizes(*N, PetscAbs(A->cmap->bs), PetscAbs(A->rmap->bs)));
  PetscCall(MatGetVecType(A, &vtype));
  PetscCall(MatSetVecType(*N, vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N, A->boundtocpu));
#endif
  PetscCall(MatSetUp(*N));
  PetscFunctionReturn(0);
}
