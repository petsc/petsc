
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
      PetscCall(VecDuplicate(left,&a->left));
      PetscCall(VecCopy(left,a->left));
    } else {
      PetscCall(VecPointwiseMult(a->left,left,a->left));
    }
  }
  if (right) {
    if (!a->right) {
      PetscCall(VecDuplicate(right,&a->right));
      PetscCall(VecCopy(right,a->right));
    } else {
      PetscCall(VecPointwiseMult(a->right,right,a->right));
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
      PetscCall(VecDuplicate(Na->right,&Na->rightwork));
    }
    PetscCall(VecPointwiseMult(Na->rightwork,Na->right,in));
    in   = Na->rightwork;
  }
  PetscCall(MatMult(Na->A,in,Na->w));
  PetscCall(MatMultHermitianTranspose(Na->A,Na->w,y));
  if (Na->left) {
    PetscCall(VecPointwiseMult(y,Na->left,y));
  }
  PetscCall(VecScale(y,Na->scale));
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
      PetscCall(VecDuplicate(Na->right,&Na->rightwork));
    }
    PetscCall(VecPointwiseMult(Na->rightwork,Na->right,in));
    in   = Na->rightwork;
  }
  PetscCall(MatMult(Na->A,in,Na->w));
  PetscCall(VecScale(Na->w,Na->scale));
  if (Na->left) {
    PetscCall(MatMultHermitianTranspose(Na->A,Na->w,v3));
    PetscCall(VecPointwiseMult(v3,Na->left,v3));
    PetscCall(VecAXPY(v3,1.0,v2));
  } else {
    PetscCall(MatMultHermitianTransposeAdd(Na->A,Na->w,v2,v3));
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
      PetscCall(VecDuplicate(Na->left,&Na->leftwork));
    }
    PetscCall(VecPointwiseMult(Na->leftwork,Na->left,in));
    in   = Na->leftwork;
  }
  PetscCall(MatMult(Na->A,in,Na->w));
  PetscCall(MatMultHermitianTranspose(Na->A,Na->w,y));
  if (Na->right) {
    PetscCall(VecPointwiseMult(y,Na->right,y));
  }
  PetscCall(VecScale(y,Na->scale));
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
      PetscCall(VecDuplicate(Na->left,&Na->leftwork));
    }
    PetscCall(VecPointwiseMult(Na->leftwork,Na->left,in));
    in   = Na->leftwork;
  }
  PetscCall(MatMult(Na->A,in,Na->w));
  PetscCall(VecScale(Na->w,Na->scale));
  if (Na->right) {
    PetscCall(MatMultHermitianTranspose(Na->A,Na->w,v3));
    PetscCall(VecPointwiseMult(v3,Na->right,v3));
    PetscCall(VecAXPY(v3,1.0,v2));
  } else {
    PetscCall(MatMultHermitianTransposeAdd(Na->A,Na->w,v2,v3));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroyHermitian_Normal(Mat N)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&Na->A));
  PetscCall(VecDestroy(&Na->w));
  PetscCall(VecDestroy(&Na->left));
  PetscCall(VecDestroy(&Na->right));
  PetscCall(VecDestroy(&Na->leftwork));
  PetscCall(VecDestroy(&Na->rightwork));
  PetscCall(PetscFree(N->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatNormalGetMatHermitian_C",NULL));
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
  PetscCall(PetscMalloc2(A->cmap->N,&diag,A->cmap->N,&work));
  PetscCall(PetscArrayzero(work,A->cmap->N));
  PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    PetscCall(MatGetRow(A,i,&nnz,&cols,&mvalues));
    for (j=0; j<nnz; j++) {
      work[cols[j]] += mvalues[j]*PetscConj(mvalues[j]);
    }
    PetscCall(MatRestoreRow(A,i,&nnz,&cols,&mvalues));
  }
  PetscCall(MPIU_Allreduce(work,diag,A->cmap->N,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)N)));
  rstart = N->cmap->rstart;
  rend   = N->cmap->rend;
  PetscCall(VecGetArray(v,&values));
  PetscCall(PetscArraycpy(values,diag+rstart,rend-rstart));
  PetscCall(VecRestoreArray(v,&values));
  PetscCall(PetscFree2(diag,work));
  PetscCall(VecScale(v,Na->scale));
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
  PetscCall(PetscUseMethod(A,"MatNormalGetMatHermitian_C",(Mat,Mat*),(A,M)));
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
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),N));
  PetscCall(MatSetSizes(*N,n,n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N,MATNORMALHERMITIAN));
  PetscCall(PetscLayoutReference(A->cmap,&(*N)->rmap));
  PetscCall(PetscLayoutReference(A->cmap,&(*N)->cmap));

  PetscCall(PetscNewLog(*N,&Na));
  (*N)->data = (void*) Na;
  PetscCall(PetscObjectReference((PetscObject)A));
  Na->A      = A;
  Na->scale  = 1.0;

  PetscCall(MatCreateVecs(A,NULL,&Na->w));

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

  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatNormalGetMatHermitian_C",MatNormalGetMatHermitian_Normal));
  PetscCall(MatSetOption(*N,MAT_HERMITIAN,PETSC_TRUE));
  PetscCall(MatGetVecType(A,&vtype));
  PetscCall(MatSetVecType(*N,vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N,A->boundtocpu));
#endif
  PetscFunctionReturn(0);
}
