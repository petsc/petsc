
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

  PetscFunctionBegin;
  if (left) {
    if (!a->left) {
      CHKERRQ(VecDuplicate(left,&a->left));
      CHKERRQ(VecCopy(left,a->left));
    } else {
      CHKERRQ(VecPointwiseMult(a->left,left,a->left));
    }
  }
  if (right) {
    if (!a->right) {
      CHKERRQ(VecDuplicate(right,&a->right));
      CHKERRQ(VecCopy(right,a->right));
    } else {
      CHKERRQ(VecPointwiseMult(a->right,right,a->right));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitian_Normal(Mat N,Vec x,Vec y)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  Vec            in;

  PetscFunctionBegin;
  in = x;
  if (Na->right) {
    if (!Na->rightwork) {
      CHKERRQ(VecDuplicate(Na->right,&Na->rightwork));
    }
    CHKERRQ(VecPointwiseMult(Na->rightwork,Na->right,in));
    in   = Na->rightwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(MatMultHermitianTranspose(Na->A,Na->w,y));
  if (Na->left) {
    CHKERRQ(VecPointwiseMult(y,Na->left,y));
  }
  CHKERRQ(VecScale(y,Na->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianAdd_Normal(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->right) {
    if (!Na->rightwork) {
      CHKERRQ(VecDuplicate(Na->right,&Na->rightwork));
    }
    CHKERRQ(VecPointwiseMult(Na->rightwork,Na->right,in));
    in   = Na->rightwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(VecScale(Na->w,Na->scale));
  if (Na->left) {
    CHKERRQ(MatMultHermitianTranspose(Na->A,Na->w,v3));
    CHKERRQ(VecPointwiseMult(v3,Na->left,v3));
    CHKERRQ(VecAXPY(v3,1.0,v2));
  } else {
    CHKERRQ(MatMultHermitianTransposeAdd(Na->A,Na->w,v2,v3));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTranspose_Normal(Mat N,Vec x,Vec y)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  Vec            in;

  PetscFunctionBegin;
  in = x;
  if (Na->left) {
    if (!Na->leftwork) {
      CHKERRQ(VecDuplicate(Na->left,&Na->leftwork));
    }
    CHKERRQ(VecPointwiseMult(Na->leftwork,Na->left,in));
    in   = Na->leftwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(MatMultHermitianTranspose(Na->A,Na->w,y));
  if (Na->right) {
    CHKERRQ(VecPointwiseMult(y,Na->right,y));
  }
  CHKERRQ(VecScale(y,Na->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultHermitianTransposeAdd_Normal(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->left) {
    if (!Na->leftwork) {
      CHKERRQ(VecDuplicate(Na->left,&Na->leftwork));
    }
    CHKERRQ(VecPointwiseMult(Na->leftwork,Na->left,in));
    in   = Na->leftwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(VecScale(Na->w,Na->scale));
  if (Na->right) {
    CHKERRQ(MatMultHermitianTranspose(Na->A,Na->w,v3));
    CHKERRQ(VecPointwiseMult(v3,Na->right,v3));
    CHKERRQ(VecAXPY(v3,1.0,v2));
  } else {
    CHKERRQ(MatMultHermitianTransposeAdd(Na->A,Na->w,v2,v3));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroyHermitian_Normal(Mat N)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&Na->A));
  CHKERRQ(VecDestroy(&Na->w));
  CHKERRQ(VecDestroy(&Na->left));
  CHKERRQ(VecDestroy(&Na->right));
  CHKERRQ(VecDestroy(&Na->leftwork));
  CHKERRQ(VecDestroy(&Na->rightwork));
  CHKERRQ(PetscFree(N->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatNormalGetMatHermitian_C",NULL));
  PetscFunctionReturn(0);
}

/*
      Slow, nonscalable version
*/
PetscErrorCode MatGetDiagonalHermitian_Normal(Mat N,Vec v)
{
  Mat_Normal        *Na = (Mat_Normal*)N->data;
  Mat               A   = Na->A;
  PetscInt          i,j,rstart,rend,nnz;
  const PetscInt    *cols;
  PetscScalar       *diag,*work,*values;
  const PetscScalar *mvalues;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc2(A->cmap->N,&diag,A->cmap->N,&work));
  CHKERRQ(PetscArrayzero(work,A->cmap->N));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatGetRow(A,i,&nnz,&cols,&mvalues));
    for (j=0; j<nnz; j++) {
      work[cols[j]] += mvalues[j]*PetscConj(mvalues[j]);
    }
    CHKERRQ(MatRestoreRow(A,i,&nnz,&cols,&mvalues));
  }
  CHKERRMPI(MPIU_Allreduce(work,diag,A->cmap->N,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)N)));
  rstart = N->cmap->rstart;
  rend   = N->cmap->rend;
  CHKERRQ(VecGetArray(v,&values));
  CHKERRQ(PetscArraycpy(values,diag+rstart,rend-rstart));
  CHKERRQ(VecRestoreArray(v,&values));
  CHKERRQ(PetscFree2(diag,work));
  CHKERRQ(VecScale(v,Na->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNormalGetMatHermitian_Normal(Mat A,Mat *M)
{
  Mat_Normal *Aa = (Mat_Normal*)A->data;

  PetscFunctionBegin;
  *M = Aa->A;
  PetscFunctionReturn(0);
}

/*@
      MatNormalHermitianGetMat - Gets the Mat object stored inside a MATNORMALHERMITIAN

   Logically collective on Mat

   Input Parameter:
.   A  - the MATNORMALHERMITIAN matrix

   Output Parameter:
.   M - the matrix object stored inside A

   Level: intermediate

.seealso: MatCreateNormalHermitian()

@*/
PetscErrorCode MatNormalHermitianGetMat(Mat A,Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(M,2);
  CHKERRQ(PetscUseMethod(A,"MatNormalGetMatHermitian_C",(Mat,Mat*),(A,M)));
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
  PetscInt       m,n;
  Mat_Normal     *Na;
  VecType        vtype;

  PetscFunctionBegin;
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),N));
  CHKERRQ(MatSetSizes(*N,n,n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)*N,MATNORMALHERMITIAN));
  CHKERRQ(PetscLayoutReference(A->cmap,&(*N)->rmap));
  CHKERRQ(PetscLayoutReference(A->cmap,&(*N)->cmap));

  CHKERRQ(PetscNewLog(*N,&Na));
  (*N)->data = (void*) Na;
  CHKERRQ(PetscObjectReference((PetscObject)A));
  Na->A      = A;
  Na->scale  = 1.0;

  CHKERRQ(MatCreateVecs(A,NULL,&Na->w));

  (*N)->ops->destroy          = MatDestroyHermitian_Normal;
  (*N)->ops->mult             = MatMultHermitian_Normal;
  (*N)->ops->multtranspose    = MatMultHermitianTranspose_Normal;
  (*N)->ops->multtransposeadd = MatMultHermitianTransposeAdd_Normal;
  (*N)->ops->multadd          = MatMultHermitianAdd_Normal;
  (*N)->ops->getdiagonal      = MatGetDiagonalHermitian_Normal;
  (*N)->ops->scale            = MatScaleHermitian_Normal;
  (*N)->ops->diagonalscale    = MatDiagonalScaleHermitian_Normal;
  (*N)->assembled             = PETSC_TRUE;
  (*N)->preallocated          = PETSC_TRUE;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatNormalGetMatHermitian_C",MatNormalGetMatHermitian_Normal));
  CHKERRQ(MatSetOption(*N,MAT_HERMITIAN,PETSC_TRUE));
  CHKERRQ(MatGetVecType(A,&vtype));
  CHKERRQ(MatSetVecType(*N,vtype));
#if defined(PETSC_HAVE_DEVICE)
  CHKERRQ(MatBindToCPU(*N,A->boundtocpu));
#endif
  PetscFunctionReturn(0);
}
