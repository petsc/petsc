
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat A;
} Mat_Transpose;

PetscErrorCode MatMult_Transpose(Mat N,Vec x,Vec y)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  PetscCall(MatMultTranspose(Na->A,x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Transpose(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  PetscCall(MatMultTransposeAdd(Na->A,v1,v2,v3));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Transpose(Mat N,Vec x,Vec y)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  PetscCall(MatMult(Na->A,x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Transpose(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  PetscCall(MatMultAdd(Na->A,v1,v2,v3));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Transpose(Mat N)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&Na->A));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatTransposeGetMat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_anytype_C",NULL));
  PetscCall(PetscFree(N->data));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_Transpose(Mat N, MatDuplicateOption op, Mat* m)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  if (op == MAT_COPY_VALUES) {
    PetscCall(MatTranspose(Na->A,MAT_INITIAL_MATRIX,m));
  } else if (op == MAT_DO_NOT_COPY_VALUES) {
    PetscCall(MatDuplicate(Na->A,MAT_DO_NOT_COPY_VALUES,m));
    PetscCall(MatTranspose(*m,MAT_INPLACE_MATRIX,m));
  } else SETERRQ(PetscObjectComm((PetscObject)N),PETSC_ERR_SUP,"MAT_SHARE_NONZERO_PATTERN not supported for this matrix type");
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateVecs_Transpose(Mat A,Vec *r, Vec *l)
{
  Mat_Transpose  *Aa = (Mat_Transpose*)A->data;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(Aa->A,l,r));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_Transpose(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_Transpose  *Ya = (Mat_Transpose*)Y->data;
  Mat_Transpose  *Xa = (Mat_Transpose*)X->data;
  Mat              M = Ya->A;
  Mat              N = Xa->A;

  PetscFunctionBegin;
  PetscCall(MatAXPY(M,a,N,str));
  PetscFunctionReturn(0);
}

PetscErrorCode MatHasOperation_Transpose(Mat mat,MatOperation op,PetscBool *has)
{
  Mat_Transpose  *X = (Mat_Transpose*)mat->data;
  PetscFunctionBegin;

  *has = PETSC_FALSE;
  if (op == MATOP_MULT) {
    PetscCall(MatHasOperation(X->A,MATOP_MULT_TRANSPOSE,has));
  } else if (op == MATOP_MULT_TRANSPOSE) {
    PetscCall(MatHasOperation(X->A,MATOP_MULT,has));
  } else if (op == MATOP_MULT_ADD) {
    PetscCall(MatHasOperation(X->A,MATOP_MULT_TRANSPOSE_ADD,has));
  } else if (op == MATOP_MULT_TRANSPOSE_ADD) {
    PetscCall(MatHasOperation(X->A,MATOP_MULT_ADD,has));
  } else if (((void**)mat->ops)[op]) *has = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* used by hermitian transpose */
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_Transpose(Mat D)
{
  Mat            A,B,C,Ain,Bin,Cin;
  PetscBool      Aistrans,Bistrans,Cistrans;
  PetscInt       Atrans,Btrans,Ctrans;
  MatProductType ptype;

  PetscFunctionBegin;
  MatCheckProduct(D,1);
  A = D->product->A;
  B = D->product->B;
  C = D->product->C;
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATTRANSPOSEMAT,&Aistrans));
  PetscCall(PetscObjectTypeCompare((PetscObject)B,MATTRANSPOSEMAT,&Bistrans));
  PetscCall(PetscObjectTypeCompare((PetscObject)C,MATTRANSPOSEMAT,&Cistrans));
  PetscCheck(Aistrans || Bistrans || Cistrans,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"This should not happen");
  Atrans = 0;
  Ain    = A;
  while (Aistrans) {
    Atrans++;
    PetscCall(MatTransposeGetMat(Ain,&Ain));
    PetscCall(PetscObjectTypeCompare((PetscObject)Ain,MATTRANSPOSEMAT,&Aistrans));
  }
  Btrans = 0;
  Bin    = B;
  while (Bistrans) {
    Btrans++;
    PetscCall(MatTransposeGetMat(Bin,&Bin));
    PetscCall(PetscObjectTypeCompare((PetscObject)Bin,MATTRANSPOSEMAT,&Bistrans));
  }
  Ctrans = 0;
  Cin    = C;
  while (Cistrans) {
    Ctrans++;
    PetscCall(MatTransposeGetMat(Cin,&Cin));
    PetscCall(PetscObjectTypeCompare((PetscObject)Cin,MATTRANSPOSEMAT,&Cistrans));
  }
  Atrans = Atrans%2;
  Btrans = Btrans%2;
  Ctrans = Ctrans%2;
  ptype = D->product->type; /* same product type by default */
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
      } else {  /* A * B */
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
    default: SETERRQ(PetscObjectComm((PetscObject)D),PETSC_ERR_SUP,"ProductType %s is not supported",MatProductTypes[D->product->type]);
    }
  }
  PetscCall(MatProductReplaceMats(Ain,Bin,Cin,D));
  PetscCall(MatProductSetType(D,ptype));
  PetscCall(MatProductSetFromOptions(D));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_Transpose(Mat A,Vec v)
{
  Mat_Transpose  *Aa = (Mat_Transpose*)A->data;

  PetscFunctionBegin;
  PetscCall(MatGetDiagonal(Aa->A,v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_Transpose(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_Transpose  *Aa = (Mat_Transpose*)A->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(MatHasOperation(Aa->A,MATOP_TRANSPOSE,&flg));
  if (flg) {
    Mat B;

    PetscCall(MatTranspose(Aa->A,MAT_INITIAL_MATRIX,&B));
    if (reuse != MAT_INPLACE_MATRIX) {
      PetscCall(MatConvert(B,newtype,reuse,newmat));
      PetscCall(MatDestroy(&B));
    } else {
      PetscCall(MatConvert(B,newtype,MAT_INPLACE_MATRIX,&B));
      PetscCall(MatHeaderReplace(A,&B));
    }
  } else { /* use basic converter as fallback */
    PetscCall(MatConvert_Basic(A,newtype,reuse,newmat));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeGetMat_Transpose(Mat A,Mat *M)
{
  Mat_Transpose  *Aa = (Mat_Transpose*)A->data;

  PetscFunctionBegin;
  *M = Aa->A;
  PetscFunctionReturn(0);
}

/*@
      MatTransposeGetMat - Gets the Mat object stored inside a MATTRANSPOSEMAT

   Logically collective on Mat

   Input Parameter:
.   A  - the MATTRANSPOSE matrix

   Output Parameter:
.   M - the matrix object stored inside A

   Level: intermediate

.seealso: `MatCreateTranspose()`

@*/
PetscErrorCode MatTransposeGetMat(Mat A,Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(M,2);
  PetscUseMethod(A,"MatTransposeGetMat_C",(Mat,Mat*),(A,M));
  PetscFunctionReturn(0);
}

/*@
      MatCreateTranspose - Creates a new matrix object that behaves like A'

   Collective on Mat

   Input Parameter:
.   A  - the (possibly rectangular) matrix

   Output Parameter:
.   N - the matrix that represents A'

   Level: intermediate

   Notes:
    The transpose A' is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product by using the MatMultTranspose() on
          the original matrix

.seealso: `MatCreateNormal()`, `MatMult()`, `MatMultTranspose()`, `MatCreate()`

@*/
PetscErrorCode  MatCreateTranspose(Mat A,Mat *N)
{
  PetscInt       m,n;
  Mat_Transpose  *Na;
  VecType        vtype;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),N));
  PetscCall(MatSetSizes(*N,n,m,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(PetscLayoutSetUp((*N)->rmap));
  PetscCall(PetscLayoutSetUp((*N)->cmap));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N,MATTRANSPOSEMAT));

  PetscCall(PetscNewLog(*N,&Na));
  (*N)->data = (void*) Na;
  PetscCall(PetscObjectReference((PetscObject)A));
  Na->A      = A;

  (*N)->ops->destroy               = MatDestroy_Transpose;
  (*N)->ops->mult                  = MatMult_Transpose;
  (*N)->ops->multadd               = MatMultAdd_Transpose;
  (*N)->ops->multtranspose         = MatMultTranspose_Transpose;
  (*N)->ops->multtransposeadd      = MatMultTransposeAdd_Transpose;
  (*N)->ops->duplicate             = MatDuplicate_Transpose;
  (*N)->ops->getvecs               = MatCreateVecs_Transpose;
  (*N)->ops->axpy                  = MatAXPY_Transpose;
  (*N)->ops->hasoperation          = MatHasOperation_Transpose;
  (*N)->ops->productsetfromoptions = MatProductSetFromOptions_Transpose;
  (*N)->ops->getdiagonal           = MatGetDiagonal_Transpose;
  (*N)->ops->convert               = MatConvert_Transpose;
  (*N)->assembled                  = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatTransposeGetMat_C",MatTransposeGetMat_Transpose));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_anytype_C",MatProductSetFromOptions_Transpose));
  PetscCall(MatSetBlockSizes(*N,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs)));
  PetscCall(MatGetVecType(A,&vtype));
  PetscCall(MatSetVecType(*N,vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N,A->boundtocpu));
#endif
  PetscCall(MatSetUp(*N));
  PetscFunctionReturn(0);
}
