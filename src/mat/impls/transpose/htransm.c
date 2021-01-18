
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
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatHermitianTransposeGetMat_C",NULL);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatTransposeGetMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_anytype_C",NULL);CHKERRQ(ierr);
#endif
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

PetscErrorCode MatCreateVecs_HT(Mat N,Vec *r, Vec *l)
{
  Mat_HT         *Na = (Mat_HT*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateVecs(Na->A,l,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_HT(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_HT         *Ya = (Mat_HT*)Y->data;
  Mat_HT         *Xa = (Mat_HT*)X->data;
  Mat              M = Ya->A;
  Mat              N = Xa->A;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAXPY(M,a,N,str);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(M,2);
  ierr = PetscUseMethod(A,"MatHermitianTransposeGetMat_C",(Mat,Mat*),(A,M));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_Transpose(Mat);

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
  PetscErrorCode ierr;
  PetscInt       m,n;
  Mat_HT         *Na;
  VecType        vtype;

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

  (*N)->ops->destroy                   = MatDestroy_HT;
  (*N)->ops->mult                      = MatMult_HT;
  (*N)->ops->multadd                   = MatMultAdd_HT;
  (*N)->ops->multhermitiantranspose    = MatMultHermitianTranspose_HT;
  (*N)->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_HT;
  (*N)->ops->duplicate                 = MatDuplicate_HT;
  (*N)->ops->getvecs                   = MatCreateVecs_HT;
  (*N)->ops->axpy                      = MatAXPY_HT;
#if !defined(PETSC_USE_COMPLEX)
  (*N)->ops->productsetfromoptions     = MatProductSetFromOptions_Transpose;
#endif
  (*N)->assembled                      = PETSC_TRUE;

  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatHermitianTransposeGetMat_C",MatHermitianTransposeGetMat_HT);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatTransposeGetMat_C",MatHermitianTransposeGetMat_HT);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_anytype_C",MatProductSetFromOptions_Transpose);CHKERRQ(ierr);
#endif
  ierr = MatSetBlockSizes(*N,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs));CHKERRQ(ierr);
  ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
  ierr = MatSetVecType(*N,vtype);CHKERRQ(ierr);
#if defined(PETSC_HAVE_DEVICE)
  ierr = MatBindToCPU(*N,A->boundtocpu);CHKERRQ(ierr);
#endif
  ierr = MatSetUp(*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
