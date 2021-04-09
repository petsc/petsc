
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat         A;
  Vec         w,left,right,leftwork,rightwork;
  PetscScalar scale;
} Mat_Normal;

PetscErrorCode MatScaleHermitian_Normal(Mat inA,PetscScalar scale)
{
  Mat_Normal *a = (Mat_Normal*)inA->data;

  PetscFunctionBegin;
  a->scale *= scale;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScaleHermitian_Normal(Mat inA,Vec left,Vec right)
{
  Mat_Normal     *a = (Mat_Normal*)inA->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (left) {
    if (!a->left) {
      ierr = VecDuplicate(left,&a->left);CHKERRQ(ierr);
      ierr = VecCopy(left,a->left);CHKERRQ(ierr);
    } else {
      ierr = VecPointwiseMult(a->left,left,a->left);CHKERRQ(ierr);
    }
  }
  if (right) {
    if (!a->right) {
      ierr = VecDuplicate(right,&a->right);CHKERRQ(ierr);
      ierr = VecCopy(right,a->right);CHKERRQ(ierr);
    } else {
      ierr = VecPointwiseMult(a->right,right,a->right);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitian_Normal(Mat N,Vec x,Vec y)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = x;
  if (Na->right) {
    if (!Na->rightwork) {
      ierr = VecDuplicate(Na->right,&Na->rightwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->rightwork,Na->right,in);CHKERRQ(ierr);
    in   = Na->rightwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = MatMultHermitianTranspose(Na->A,Na->w,y);CHKERRQ(ierr);
  if (Na->left) {
    ierr = VecPointwiseMult(y,Na->left,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,Na->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianAdd_Normal(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->right) {
    if (!Na->rightwork) {
      ierr = VecDuplicate(Na->right,&Na->rightwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->rightwork,Na->right,in);CHKERRQ(ierr);
    in   = Na->rightwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = VecScale(Na->w,Na->scale);CHKERRQ(ierr);
  if (Na->left) {
    ierr = MatMultHermitianTranspose(Na->A,Na->w,v3);CHKERRQ(ierr);
    ierr = VecPointwiseMult(v3,Na->left,v3);CHKERRQ(ierr);
    ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
  } else {
    ierr = MatMultHermitianTransposeAdd(Na->A,Na->w,v2,v3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTranspose_Normal(Mat N,Vec x,Vec y)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = x;
  if (Na->left) {
    if (!Na->leftwork) {
      ierr = VecDuplicate(Na->left,&Na->leftwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->leftwork,Na->left,in);CHKERRQ(ierr);
    in   = Na->leftwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = MatMultHermitianTranspose(Na->A,Na->w,y);CHKERRQ(ierr);
  if (Na->right) {
    ierr = VecPointwiseMult(y,Na->right,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,Na->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTransposeAdd_Normal(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->left) {
    if (!Na->leftwork) {
      ierr = VecDuplicate(Na->left,&Na->leftwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->leftwork,Na->left,in);CHKERRQ(ierr);
    in   = Na->leftwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = VecScale(Na->w,Na->scale);CHKERRQ(ierr);
  if (Na->right) {
    ierr = MatMultHermitianTranspose(Na->A,Na->w,v3);CHKERRQ(ierr);
    ierr = VecPointwiseMult(v3,Na->right,v3);CHKERRQ(ierr);
    ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
  } else {
    ierr = MatMultHermitianTransposeAdd(Na->A,Na->w,v2,v3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroyHermitian_Normal(Mat N)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->w);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->left);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->right);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->leftwork);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->rightwork);CHKERRQ(ierr);
  ierr = PetscFree(N->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Slow, nonscalable version
*/
PetscErrorCode MatGetDiagonalHermitian_Normal(Mat N,Vec v)
{
  Mat_Normal        *Na = (Mat_Normal*)N->data;
  Mat               A   = Na->A;
  PetscErrorCode    ierr;
  PetscInt          i,j,rstart,rend,nnz;
  const PetscInt    *cols;
  PetscScalar       *diag,*work,*values;
  const PetscScalar *mvalues;

  PetscFunctionBegin;
  ierr = PetscMalloc2(A->cmap->N,&diag,A->cmap->N,&work);CHKERRQ(ierr);
  ierr = PetscArrayzero(work,A->cmap->N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nnz,&cols,&mvalues);CHKERRQ(ierr);
    for (j=0; j<nnz; j++) {
      work[cols[j]] += mvalues[j]*PetscConj(mvalues[j]);
    }
    ierr = MatRestoreRow(A,i,&nnz,&cols,&mvalues);CHKERRQ(ierr);
  }
  ierr   = MPIU_Allreduce(work,diag,A->cmap->N,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)N));CHKERRMPI(ierr);
  rstart = N->cmap->rstart;
  rend   = N->cmap->rend;
  ierr   = VecGetArray(v,&values);CHKERRQ(ierr);
  ierr   = PetscArraycpy(values,diag+rstart,rend-rstart);CHKERRQ(ierr);
  ierr   = VecRestoreArray(v,&values);CHKERRQ(ierr);
  ierr   = PetscFree2(diag,work);CHKERRQ(ierr);
  ierr   = VecScale(v,Na->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
      MatCreateNormalHermitian - Creates a new matrix object that behaves like (A*)'*A.

   Collective on Mat

   Input Parameter:
.   A  - the (possibly rectangular complex) matrix

   Output Parameter:
.   N - the matrix that represents (A*)'*A

   Level: intermediate

   Notes:
    The product (A*)'*A is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product by first multiplying by
          A and then (A*)'
@*/
PetscErrorCode  MatCreateNormalHermitian(Mat A,Mat *N)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  Mat_Normal     *Na;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,n,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATNORMALHERMITIAN);CHKERRQ(ierr);

  ierr       = PetscNewLog(*N,&Na);CHKERRQ(ierr);
  (*N)->data = (void*) Na;
  ierr       = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  Na->A      = A;
  Na->scale  = 1.0;

  ierr = VecCreateMPI(PetscObjectComm((PetscObject)A),m,PETSC_DECIDE,&Na->w);CHKERRQ(ierr);

  (*N)->ops->destroy          = MatDestroyHermitian_Normal;
  (*N)->ops->mult             = MatMultHermitian_Normal;
  (*N)->ops->multtranspose    = MatMultHermitianTranspose_Normal;
  (*N)->ops->multtransposeadd = MatMultHermitianTransposeAdd_Normal;
  (*N)->ops->multadd          = MatMultHermitianAdd_Normal;
  (*N)->ops->getdiagonal      = MatGetDiagonalHermitian_Normal;
  (*N)->ops->scale            = MatScaleHermitian_Normal;
  (*N)->ops->diagonalscale    = MatDiagonalScaleHermitian_Normal;
  (*N)->assembled             = PETSC_TRUE;
  (*N)->cmap->N               = A->cmap->N;
  (*N)->rmap->N               = A->cmap->N;
  (*N)->cmap->n               = A->cmap->n;
  (*N)->rmap->n               = A->cmap->n;
  PetscFunctionReturn(0);
}

