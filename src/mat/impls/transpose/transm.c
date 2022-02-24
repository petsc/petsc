
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat A;
} Mat_Transpose;

PetscErrorCode MatMult_Transpose(Mat N,Vec x,Vec y)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  CHKERRQ(MatMultTranspose(Na->A,x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Transpose(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  CHKERRQ(MatMultTransposeAdd(Na->A,v1,v2,v3));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Transpose(Mat N,Vec x,Vec y)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  CHKERRQ(MatMult(Na->A,x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Transpose(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  CHKERRQ(MatMultAdd(Na->A,v1,v2,v3));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Transpose(Mat N)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&Na->A));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatTransposeGetMat_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_anytype_C",NULL));
  CHKERRQ(PetscFree(N->data));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_Transpose(Mat N, MatDuplicateOption op, Mat* m)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;

  PetscFunctionBegin;
  if (op == MAT_COPY_VALUES) {
    CHKERRQ(MatTranspose(Na->A,MAT_INITIAL_MATRIX,m));
  } else if (op == MAT_DO_NOT_COPY_VALUES) {
    CHKERRQ(MatDuplicate(Na->A,MAT_DO_NOT_COPY_VALUES,m));
    CHKERRQ(MatTranspose(*m,MAT_INPLACE_MATRIX,m));
  } else SETERRQ(PetscObjectComm((PetscObject)N),PETSC_ERR_SUP,"MAT_SHARE_NONZERO_PATTERN not supported for this matrix type");
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateVecs_Transpose(Mat A,Vec *r, Vec *l)
{
  Mat_Transpose  *Aa = (Mat_Transpose*)A->data;

  PetscFunctionBegin;
  CHKERRQ(MatCreateVecs(Aa->A,l,r));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_Transpose(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_Transpose  *Ya = (Mat_Transpose*)Y->data;
  Mat_Transpose  *Xa = (Mat_Transpose*)X->data;
  Mat              M = Ya->A;
  Mat              N = Xa->A;

  PetscFunctionBegin;
  CHKERRQ(MatAXPY(M,a,N,str));
  PetscFunctionReturn(0);
}

PetscErrorCode MatHasOperation_Transpose(Mat mat,MatOperation op,PetscBool *has)
{
  Mat_Transpose  *X = (Mat_Transpose*)mat->data;
  PetscFunctionBegin;

  *has = PETSC_FALSE;
  if (op == MATOP_MULT) {
    CHKERRQ(MatHasOperation(X->A,MATOP_MULT_TRANSPOSE,has));
  } else if (op == MATOP_MULT_TRANSPOSE) {
    CHKERRQ(MatHasOperation(X->A,MATOP_MULT,has));
  } else if (op == MATOP_MULT_ADD) {
    CHKERRQ(MatHasOperation(X->A,MATOP_MULT_TRANSPOSE_ADD,has));
  } else if (op == MATOP_MULT_TRANSPOSE_ADD) {
    CHKERRQ(MatHasOperation(X->A,MATOP_MULT_ADD,has));
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATTRANSPOSEMAT,&Aistrans));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)B,MATTRANSPOSEMAT,&Bistrans));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)C,MATTRANSPOSEMAT,&Cistrans));
  PetscCheckFalse(!Aistrans && !Bistrans && !Cistrans,PetscObjectComm((PetscObject)D),PETSC_ERR_PLIB,"This should not happen");
  Atrans = 0;
  Ain    = A;
  while (Aistrans) {
    Atrans++;
    CHKERRQ(MatTransposeGetMat(Ain,&Ain));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)Ain,MATTRANSPOSEMAT,&Aistrans));
  }
  Btrans = 0;
  Bin    = B;
  while (Bistrans) {
    Btrans++;
    CHKERRQ(MatTransposeGetMat(Bin,&Bin));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)Bin,MATTRANSPOSEMAT,&Bistrans));
  }
  Ctrans = 0;
  Cin    = C;
  while (Cistrans) {
    Ctrans++;
    CHKERRQ(MatTransposeGetMat(Cin,&Cin));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)Cin,MATTRANSPOSEMAT,&Cistrans));
  }
  Atrans = Atrans%2;
  Btrans = Btrans%2;
  Ctrans = Ctrans%2;
  ptype = D->product->type; /* same product type by default */
  if (Ain->symmetric) Atrans = 0;
  if (Bin->symmetric) Btrans = 0;
  if (Cin && Cin->symmetric) Ctrans = 0;

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
  CHKERRQ(MatProductReplaceMats(Ain,Bin,Cin,D));
  CHKERRQ(MatProductSetType(D,ptype));
  CHKERRQ(MatProductSetFromOptions(D));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_Transpose(Mat A,Vec v)
{
  Mat_Transpose  *Aa = (Mat_Transpose*)A->data;

  PetscFunctionBegin;
  CHKERRQ(MatGetDiagonal(Aa->A,v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_Transpose(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_Transpose  *Aa = (Mat_Transpose*)A->data;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(MatHasOperation(Aa->A,MATOP_TRANSPOSE,&flg));
  if (flg) {
    Mat B;

    CHKERRQ(MatTranspose(Aa->A,MAT_INITIAL_MATRIX,&B));
    if (reuse != MAT_INPLACE_MATRIX) {
      CHKERRQ(MatConvert(B,newtype,reuse,newmat));
      CHKERRQ(MatDestroy(&B));
    } else {
      CHKERRQ(MatConvert(B,newtype,MAT_INPLACE_MATRIX,&B));
      CHKERRQ(MatHeaderReplace(A,&B));
    }
  } else { /* use basic converter as fallback */
    CHKERRQ(MatConvert_Basic(A,newtype,reuse,newmat));
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

.seealso: MatCreateTranspose()

@*/
PetscErrorCode MatTransposeGetMat(Mat A,Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(M,2);
  CHKERRQ(PetscUseMethod(A,"MatTransposeGetMat_C",(Mat,Mat*),(A,M)));
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

.seealso: MatCreateNormal(), MatMult(), MatMultTranspose(), MatCreate()

@*/
PetscErrorCode  MatCreateTranspose(Mat A,Mat *N)
{
  PetscInt       m,n;
  Mat_Transpose  *Na;
  VecType        vtype;

  PetscFunctionBegin;
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),N));
  CHKERRQ(MatSetSizes(*N,n,m,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(PetscLayoutSetUp((*N)->rmap));
  CHKERRQ(PetscLayoutSetUp((*N)->cmap));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)*N,MATTRANSPOSEMAT));

  CHKERRQ(PetscNewLog(*N,&Na));
  (*N)->data = (void*) Na;
  CHKERRQ(PetscObjectReference((PetscObject)A));
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

  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatTransposeGetMat_C",MatTransposeGetMat_Transpose));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_anytype_C",MatProductSetFromOptions_Transpose));
  CHKERRQ(MatSetBlockSizes(*N,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs)));
  CHKERRQ(MatGetVecType(A,&vtype));
  CHKERRQ(MatSetVecType(*N,vtype));
#if defined(PETSC_HAVE_DEVICE)
  CHKERRQ(MatBindToCPU(*N,A->boundtocpu));
#endif
  CHKERRQ(MatSetUp(*N));
  PetscFunctionReturn(0);
}
