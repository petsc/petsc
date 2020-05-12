
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat A;
} Mat_Transpose;

PetscErrorCode MatMult_Transpose(Mat N,Vec x,Vec y)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultTranspose(Na->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Transpose(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultTransposeAdd(Na->A,v1,v2,v3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Transpose(Mat N,Vec x,Vec y)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(Na->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Transpose(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd(Na->A,v1,v2,v3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Transpose(Mat N)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatTransposeGetMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(N->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_Transpose(Mat N, MatDuplicateOption op, Mat* m)
{
  Mat_Transpose  *Na = (Mat_Transpose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (op == MAT_COPY_VALUES) {
    ierr = MatTranspose(Na->A,MAT_INITIAL_MATRIX,m);CHKERRQ(ierr);
  } else if (op == MAT_DO_NOT_COPY_VALUES) {
    ierr = MatDuplicate(Na->A,MAT_DO_NOT_COPY_VALUES,m);CHKERRQ(ierr);
    ierr = MatTranspose(*m,MAT_INPLACE_MATRIX,m);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)N),PETSC_ERR_SUP,"MAT_SHARE_NONZERO_PATTERN not supported for this matrix type");
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateVecs_Transpose(Mat A,Vec *r, Vec *l)
{
  Mat_Transpose  *Aa = (Mat_Transpose*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateVecs(Aa->A,l,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_Transpose(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_Transpose  *Ya = (Mat_Transpose*)Y->data;
  Mat_Transpose  *Xa = (Mat_Transpose*)X->data;
  Mat              M = Ya->A;
  Mat              N = Xa->A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAXPY(M,a,N,str);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatHasOperation_Transpose(Mat mat,MatOperation op,PetscBool *has)
{
  Mat_Transpose  *X = (Mat_Transpose*)mat->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;

  *has = PETSC_FALSE;
  if (op == MATOP_MULT) {
    ierr = MatHasOperation(X->A,MATOP_MULT_TRANSPOSE,has);CHKERRQ(ierr);
  } else if (op == MATOP_MULT_TRANSPOSE) {
    ierr = MatHasOperation(X->A,MATOP_MULT,has);CHKERRQ(ierr);
  } else if (op == MATOP_MULT_ADD) {
    ierr = MatHasOperation(X->A,MATOP_MULT_TRANSPOSE_ADD,has);CHKERRQ(ierr);
  } else if (op == MATOP_MULT_TRANSPOSE_ADD) {
    ierr = MatHasOperation(X->A,MATOP_MULT_ADD,has);CHKERRQ(ierr);
  } else if (((void**)mat->ops)[op]) *has = PETSC_TRUE;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(M,2);
  ierr = PetscUseMethod(A,"MatTransposeGetMat_C",(Mat,Mat*),(A,M));CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       m,n;
  Mat_Transpose  *Na;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,n,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp((*N)->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp((*N)->cmap);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATTRANSPOSEMAT);CHKERRQ(ierr);

  ierr       = PetscNewLog(*N,&Na);CHKERRQ(ierr);
  (*N)->data = (void*) Na;
  ierr       = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  Na->A      = A;

  (*N)->ops->destroy          = MatDestroy_Transpose;
  (*N)->ops->mult             = MatMult_Transpose;
  (*N)->ops->multadd          = MatMultAdd_Transpose;
  (*N)->ops->multtranspose    = MatMultTranspose_Transpose;
  (*N)->ops->multtransposeadd = MatMultTransposeAdd_Transpose;
  (*N)->ops->duplicate        = MatDuplicate_Transpose;
  (*N)->ops->getvecs          = MatCreateVecs_Transpose;
  (*N)->ops->axpy             = MatAXPY_Transpose;
  (*N)->ops->hasoperation     = MatHasOperation_Transpose;
  (*N)->assembled             = PETSC_TRUE;

  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatTransposeGetMat_C",MatTransposeGetMat_Transpose);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(*N,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs));CHKERRQ(ierr);
  ierr = MatSetUp(*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

