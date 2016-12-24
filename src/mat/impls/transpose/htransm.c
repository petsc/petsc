
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat A;
} Mat_HT;

PetscErrorCode MatMult_HT(Mat N,Vec x,Vec y)
{
  Mat_HT         *Na = (Mat_HT*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultHermitianTranspose(Na->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_HT(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_HT         *Na = (Mat_HT*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultHermitianTransposeAdd(Na->A,v1,v2,v3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTranspose_HT(Mat N,Vec x,Vec y)
{
  Mat_HT         *Na = (Mat_HT*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(Na->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTransposeAdd_HT(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_HT         *Na = (Mat_HT*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd(Na->A,v1,v2,v3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_HT(Mat N)
{
  Mat_HT         *Na = (Mat_HT*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = PetscFree(N->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_HT(Mat N, MatDuplicateOption op, Mat* m)
{
  Mat_HT         *Na = (Mat_HT*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (op == MAT_COPY_VALUES) {
    ierr = MatHermitianTranspose(Na->A,MAT_INITIAL_MATRIX,m);CHKERRQ(ierr);
  } else if (op == MAT_DO_NOT_COPY_VALUES) {
    ierr = MatDuplicate(Na->A,MAT_DO_NOT_COPY_VALUES,m);CHKERRQ(ierr);
    ierr = MatHermitianTranspose(*m,MAT_INPLACE_MATRIX,m);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)N),PETSC_ERR_SUP,"MAT_SHARE_NONZERO_PATTERN not supported for this matrix type");
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

   Notes: The hermitian transpose A' is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product by using the MatMultHermitianTranspose() on
          the original matrix

.seealso: MatCreateNormal(), MatMult(), MatMultHermitianTranspose(), MatCreate()

@*/
PetscErrorCode  MatCreateHermitianTranspose(Mat A,Mat *N)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  Mat_HT         *Na;

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

  (*N)->ops->destroy                    = MatDestroy_HT;
  (*N)->ops->mult                       = MatMult_HT;
  (*N)->ops->multadd                    = MatMultAdd_HT;
  (*N)->ops->multhermitiantranspose     = MatMultHermitianTranspose_HT;
  (*N)->ops->multhermitiantransposeadd  = MatMultHermitianTransposeAdd_HT;
  (*N)->ops->duplicate                  = MatDuplicate_HT;
  (*N)->assembled                       = PETSC_TRUE;

  ierr = MatSetBlockSizes(*N,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs));CHKERRQ(ierr);
  ierr = MatSetUp(*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
