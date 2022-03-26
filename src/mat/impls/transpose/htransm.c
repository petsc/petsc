
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat A;
} Mat_HT;

PetscErrorCode MatMult_HT(Mat N,Vec x,Vec y)
{
  Mat_HT         *Na = (Mat_HT*)N->data;

  PetscFunctionBegin;
  PetscCall(MatMultHermitianTranspose(Na->A,x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_HT(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_HT         *Na = (Mat_HT*)N->data;

  PetscFunctionBegin;
  PetscCall(MatMultHermitianTransposeAdd(Na->A,v1,v2,v3));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTranspose_HT(Mat N,Vec x,Vec y)
{
  Mat_HT         *Na = (Mat_HT*)N->data;

  PetscFunctionBegin;
  PetscCall(MatMult(Na->A,x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTransposeAdd_HT(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_HT         *Na = (Mat_HT*)N->data;

  PetscFunctionBegin;
  PetscCall(MatMultAdd(Na->A,v1,v2,v3));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_HT(Mat N)
{
  Mat_HT         *Na = (Mat_HT*)N->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&Na->A));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatHermitianTransposeGetMat_C",NULL));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatTransposeGetMat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_anytype_C",NULL));
#endif
  PetscCall(PetscFree(N->data));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_HT(Mat N, MatDuplicateOption op, Mat* m)
{
  Mat_HT         *Na = (Mat_HT*)N->data;

  PetscFunctionBegin;
  if (op == MAT_COPY_VALUES) {
    PetscCall(MatHermitianTranspose(Na->A,MAT_INITIAL_MATRIX,m));
  } else if (op == MAT_DO_NOT_COPY_VALUES) {
    PetscCall(MatDuplicate(Na->A,MAT_DO_NOT_COPY_VALUES,m));
    PetscCall(MatHermitianTranspose(*m,MAT_INPLACE_MATRIX,m));
  } else SETERRQ(PetscObjectComm((PetscObject)N),PETSC_ERR_SUP,"MAT_SHARE_NONZERO_PATTERN not supported for this matrix type");
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateVecs_HT(Mat N,Vec *r, Vec *l)
{
  Mat_HT         *Na = (Mat_HT*)N->data;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(Na->A,l,r));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_HT(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_HT         *Ya = (Mat_HT*)Y->data;
  Mat_HT         *Xa = (Mat_HT*)X->data;
  Mat              M = Ya->A;
  Mat              N = Xa->A;

  PetscFunctionBegin;
  PetscCall(MatAXPY(M,a,N,str));
  PetscFunctionReturn(0);
}

PetscErrorCode MatHermitianTransposeGetMat_HT(Mat N,Mat *M)
{
  Mat_HT         *Na = (Mat_HT*)N->data;

  PetscFunctionBegin;
  *M = Na->A;
  PetscFunctionReturn(0);
}

/*@
      MatHermitianTransposeGetMat - Gets the Mat object stored inside a MATTRANSPOSEMAT

   Logically collective on Mat

   Input Parameter:
.   A  - the MATTRANSPOSE matrix

   Output Parameter:
.   M - the matrix object stored inside A

   Level: intermediate

.seealso: MatCreateHermitianTranspose()

@*/
PetscErrorCode MatHermitianTransposeGetMat(Mat A,Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(M,2);
  PetscCall(PetscUseMethod(A,"MatHermitianTransposeGetMat_C",(Mat,Mat*),(A,M)));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_Transpose(Mat);

PetscErrorCode MatGetDiagonal_HT(Mat A,Vec v)
{
  Mat_HT         *Na = (Mat_HT*)A->data;

  PetscFunctionBegin;
  PetscCall(MatGetDiagonal(Na->A,v));
  PetscCall(VecConjugate(v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_HT(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_HT         *Na = (Mat_HT*)A->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(MatHasOperation(Na->A,MATOP_HERMITIAN_TRANSPOSE,&flg));
  if (flg) {
    Mat B;

    PetscCall(MatHermitianTranspose(Na->A,MAT_INITIAL_MATRIX,&B));
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

/*@
      MatCreateHermitianTranspose - Creates a new matrix object that behaves like A'*

   Collective on Mat

   Input Parameter:
.   A  - the (possibly rectangular) matrix

   Output Parameter:
.   N - the matrix that represents A'*

   Level: intermediate

   Notes:
    The hermitian transpose A' is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product by using the MatMultHermitianTranspose() on
          the original matrix

.seealso: MatCreateNormal(), MatMult(), MatMultHermitianTranspose(), MatCreate()
@*/
PetscErrorCode  MatCreateHermitianTranspose(Mat A,Mat *N)
{
  PetscInt       m,n;
  Mat_HT         *Na;
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

  (*N)->ops->destroy                   = MatDestroy_HT;
  (*N)->ops->mult                      = MatMult_HT;
  (*N)->ops->multadd                   = MatMultAdd_HT;
  (*N)->ops->multhermitiantranspose    = MatMultHermitianTranspose_HT;
  (*N)->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_HT;
#if !defined(PETSC_USE_COMPLEX)
  (*N)->ops->multtranspose             = MatMultHermitianTranspose_HT;
  (*N)->ops->multtransposeadd          = MatMultHermitianTransposeAdd_HT;
#endif
  (*N)->ops->duplicate                 = MatDuplicate_HT;
  (*N)->ops->getvecs                   = MatCreateVecs_HT;
  (*N)->ops->axpy                      = MatAXPY_HT;
#if !defined(PETSC_USE_COMPLEX)
  (*N)->ops->productsetfromoptions     = MatProductSetFromOptions_Transpose;
#endif
  (*N)->ops->getdiagonal               = MatGetDiagonal_HT;
  (*N)->ops->convert                   = MatConvert_HT;
  (*N)->assembled                      = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatHermitianTransposeGetMat_C",MatHermitianTransposeGetMat_HT));
#if !defined(PETSC_USE_COMPLEX)
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatTransposeGetMat_C",MatHermitianTransposeGetMat_HT));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_anytype_C",MatProductSetFromOptions_Transpose));
#endif
  PetscCall(MatSetBlockSizes(*N,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs)));
  PetscCall(MatGetVecType(A,&vtype));
  PetscCall(MatSetVecType(*N,vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N,A->boundtocpu));
#endif
  PetscCall(MatSetUp(*N));
  PetscFunctionReturn(0);
}
