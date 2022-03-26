#include <petscmat.h>              /*I  "mat.h"  I*/

PETSC_INTERN PetscErrorCode MatCreateADA(Mat,Vec, Vec, Mat*);

typedef struct {
  Mat      A;
  Vec      D1;
  Vec      D2;
  Vec      W;
  Vec      W2;
  Vec      ADADiag;
  PetscInt GotDiag;
} _p_TaoMatADACtx;
typedef  _p_TaoMatADACtx* TaoMatADACtx;

static PetscErrorCode MatMult_ADA(Mat mat,Vec a,Vec y)
{
  TaoMatADACtx   ctx;
  PetscReal      one = 1.0;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCall(MatMult(ctx->A,a,ctx->W));
  if (ctx->D1) {
    PetscCall(VecPointwiseMult(ctx->W,ctx->D1,ctx->W));
  }
  PetscCall(MatMultTranspose(ctx->A,ctx->W,y));
  if (ctx->D2) {
    PetscCall(VecPointwiseMult(ctx->W2, ctx->D2, a));
    PetscCall(VecAXPY(y, one, ctx->W2));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_ADA(Mat mat,Vec a,Vec y)
{
  PetscFunctionBegin;
  PetscCall(MatMult_ADA(mat,a,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalSet_ADA(Mat M,Vec D, InsertMode mode)
{
  TaoMatADACtx   ctx;
  PetscReal      zero=0.0,one = 1.0;

  PetscFunctionBegin;
  PetscCheck(mode != INSERT_VALUES,PetscObjectComm((PetscObject)M),PETSC_ERR_SUP,"Cannot insert diagonal entries of this matrix type, can only add");
  PetscCall(MatShellGetContext(M,&ctx));
  if (!ctx->D2) {
    PetscCall(VecDuplicate(D,&ctx->D2));
    PetscCall(VecSet(ctx->D2, zero));
  }
  PetscCall(VecAXPY(ctx->D2, one, D));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_ADA(Mat mat)
{
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCall(VecDestroy(&ctx->W));
  PetscCall(VecDestroy(&ctx->W2));
  PetscCall(VecDestroy(&ctx->ADADiag));
  PetscCall(MatDestroy(&ctx->A));
  PetscCall(VecDestroy(&ctx->D1));
  PetscCall(VecDestroy(&ctx->D2));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_ADA(Mat mat,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_ADA(Mat Y, PetscReal a)
{
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Y,&ctx));
  PetscCall(VecShift(ctx->D2,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_ADA(Mat mat,MatDuplicateOption op,Mat *M)
{
  TaoMatADACtx      ctx;
  Mat               A2;
  Vec               D1b=NULL,D2b;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCall(MatDuplicate(ctx->A,op,&A2));
  if (ctx->D1) {
    PetscCall(VecDuplicate(ctx->D1,&D1b));
    PetscCall(VecCopy(ctx->D1,D1b));
  }
  PetscCall(VecDuplicate(ctx->D2,&D2b));
  PetscCall(VecCopy(ctx->D2,D2b));
  PetscCall(MatCreateADA(A2,D1b,D2b,M));
  if (ctx->D1) {
    PetscCall(PetscObjectDereference((PetscObject)D1b));
  }
  PetscCall(PetscObjectDereference((PetscObject)D2b));
  PetscCall(PetscObjectDereference((PetscObject)A2));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatEqual_ADA(Mat A,Mat B,PetscBool *flg)
{
  TaoMatADACtx   ctx1,ctx2;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&ctx1));
  PetscCall(MatShellGetContext(B,&ctx2));
  PetscCall(VecEqual(ctx1->D2,ctx2->D2,flg));
  if (*flg==PETSC_TRUE) {
    PetscCall(VecEqual(ctx1->D1,ctx2->D1,flg));
  }
  if (*flg==PETSC_TRUE) {
    PetscCall(MatEqual(ctx1->A,ctx2->A,flg));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_ADA(Mat mat, PetscReal a)
{
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCall(VecScale(ctx->D1,a));
  if (ctx->D2) {
    PetscCall(VecScale(ctx->D2,a));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_ADA(Mat mat,MatReuse reuse,Mat *B)
{
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(mat,MAT_COPY_VALUES,B));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(mat,*B,SAME_NONZERO_PATTERN));
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Does not support inplace transpose");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatADAComputeDiagonal(Mat mat)
{
  PetscInt       i,m,n,low,high;
  PetscScalar    *dtemp,*dptr;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCall(MatGetOwnershipRange(mat, &low, &high));
  PetscCall(MatGetSize(mat,&m,&n));

  PetscCall(PetscMalloc1(n,&dtemp));
  for (i=0; i<n; i++) {
    PetscCall(MatGetColumnVector(ctx->A, ctx->W, i));
    PetscCall(VecPointwiseMult(ctx->W,ctx->W,ctx->W));
    PetscCall(VecDotBegin(ctx->D1, ctx->W,dtemp+i));
  }
  for (i=0; i<n; i++) {
    PetscCall(VecDotEnd(ctx->D1, ctx->W,dtemp+i));
  }

  PetscCall(VecGetArray(ctx->ADADiag,&dptr));
  for (i=low; i<high; i++) {
    dptr[i-low]= dtemp[i];
  }
  PetscCall(VecRestoreArray(ctx->ADADiag,&dptr));
  PetscCall(PetscFree(dtemp));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_ADA(Mat mat,Vec v)
{
  PetscReal       one=1.0;
  TaoMatADACtx    ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCall(MatADAComputeDiagonal(mat));
  PetscCall(VecCopy(ctx->ADADiag,v));
  if (ctx->D2) {
    PetscCall(VecAXPY(v, one, ctx->D2));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_ADA(Mat mat,IS isrow,IS iscol,MatReuse cll, Mat *newmat)
{
  PetscInt          low,high;
  IS                ISrow;
  Vec               D1,D2;
  Mat               Atemp;
  TaoMatADACtx      ctx;
  PetscBool         isequal;

  PetscFunctionBegin;
  PetscCall(ISEqual(isrow,iscol,&isequal));
  PetscCheck(isequal,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only for identical column and row indices");
  PetscCall(MatShellGetContext(mat,&ctx));

  PetscCall(MatGetOwnershipRange(ctx->A,&low,&high));
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)mat),high-low,low,1,&ISrow));
  PetscCall(MatCreateSubMatrix(ctx->A,ISrow,iscol,cll,&Atemp));
  PetscCall(ISDestroy(&ISrow));

  if (ctx->D1) {
    PetscCall(VecDuplicate(ctx->D1,&D1));
    PetscCall(VecCopy(ctx->D1,D1));
  } else {
    D1 = NULL;
  }

  if (ctx->D2) {
    Vec D2sub;

    PetscCall(VecGetSubVector(ctx->D2,isrow,&D2sub));
    PetscCall(VecDuplicate(D2sub,&D2));
    PetscCall(VecCopy(D2sub,D2));
    PetscCall(VecRestoreSubVector(ctx->D2,isrow,&D2sub));
  } else {
    D2 = NULL;
  }

  PetscCall(MatCreateADA(Atemp,D1,D2,newmat));
  PetscCall(MatShellGetContext(*newmat,&ctx));
  PetscCall(PetscObjectDereference((PetscObject)Atemp));
  if (ctx->D1) {
    PetscCall(PetscObjectDereference((PetscObject)D1));
  }
  if (ctx->D2) {
    PetscCall(PetscObjectDereference((PetscObject)D2));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_ADA(Mat A,PetscInt n, IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscCalloc1(n+1,B));
  }
  for (i=0; i<n; i++) {
    PetscCall(MatCreateSubMatrix_ADA(A,irow[i],icol[i],scall,&(*B)[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_ADA(Mat mat,Vec Y, PetscInt col)
{
  PetscInt       low,high;
  PetscScalar    zero=0.0,one=1.0;

  PetscFunctionBegin;
  PetscCall(VecSet(Y, zero));
  PetscCall(VecGetOwnershipRange(Y,&low,&high));
  if (col>=low && col<high) {
    PetscCall(VecSetValue(Y,col,one,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscCall(MatMult_ADA(mat,Y,Y));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_ADA(Mat mat,MatType newtype,Mat *NewMat)
{
  PetscMPIInt    size;
  PetscBool      sametype, issame, isdense, isseqdense;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));

  PetscCall(PetscObjectTypeCompare((PetscObject)mat,newtype,&sametype));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATSAME,&issame));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATMPIDENSE,&isdense));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATSEQDENSE,&isseqdense));

  if (sametype || issame) {
    PetscCall(MatDuplicate(mat,MAT_COPY_VALUES,NewMat));
  } else if (isdense) {
    PetscInt          i,j,low,high,m,n,M,N;
    const PetscScalar *dptr;
    Vec               X;

    PetscCall(VecDuplicate(ctx->D2,&X));
    PetscCall(MatGetSize(mat,&M,&N));
    PetscCall(MatGetLocalSize(mat,&m,&n));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)mat),m,m,N,N,NULL,NewMat));
    PetscCall(MatGetOwnershipRange(*NewMat,&low,&high));
    for (i=0;i<M;i++) {
      PetscCall(MatGetColumnVector_ADA(mat,X,i));
      PetscCall(VecGetArrayRead(X,&dptr));
      for (j=0; j<high-low; j++) {
        PetscCall(MatSetValue(*NewMat,low+j,i,dptr[j],INSERT_VALUES));
      }
      PetscCall(VecRestoreArrayRead(X,&dptr));
    }
    PetscCall(MatAssemblyBegin(*NewMat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*NewMat,MAT_FINAL_ASSEMBLY));
    PetscCall(VecDestroy(&X));
  } else if (isseqdense && size==1) {
    PetscInt          i,j,low,high,m,n,M,N;
    const PetscScalar *dptr;
    Vec               X;

    PetscCall(VecDuplicate(ctx->D2,&X));
    PetscCall(MatGetSize(mat,&M,&N));
    PetscCall(MatGetLocalSize(mat,&m,&n));
    PetscCall(MatCreateSeqDense(PetscObjectComm((PetscObject)mat),N,N,NULL,NewMat));
    PetscCall(MatGetOwnershipRange(*NewMat,&low,&high));
    for (i=0;i<M;i++) {
      PetscCall(MatGetColumnVector_ADA(mat,X,i));
      PetscCall(VecGetArrayRead(X,&dptr));
      for (j=0; j<high-low; j++) {
        PetscCall(MatSetValue(*NewMat,low+j,i,dptr[j],INSERT_VALUES));
      }
      PetscCall(VecRestoreArrayRead(X,&dptr));
    }
    PetscCall(MatAssemblyBegin(*NewMat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*NewMat,MAT_FINAL_ASSEMBLY));
    PetscCall(VecDestroy(&X));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"No support to convert objects to that type");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNorm_ADA(Mat mat,NormType type,PetscReal *norm)
{
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat,&ctx));
  if (type == NORM_FROBENIUS) {
    *norm = 1.0;
  } else if (type == NORM_1 || type == NORM_INFINITY) {
    *norm = 1.0;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No two norm");
  PetscFunctionReturn(0);
}

/*@C
   MatCreateADA - Creates a matrix M=A^T D1 A + D2 where D1, D2 are diagonal

   Collective on matrix

   Input Parameters:
+  mat - matrix of arbitrary type
.  d1 - A vector defining a diagonal matrix
-  d2 - A vector defining a diagonal matrix

   Output Parameters:
.  J - New matrix whose operations are defined in terms of mat, D1, and D2.

   Notes:
   The user provides the input data and is responsible for destroying
   this data after matrix J has been destroyed.

   Level: developer

.seealso: MatCreate()
@*/
PetscErrorCode MatCreateADA(Mat mat,Vec d1, Vec d2, Mat *J)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)mat);
  TaoMatADACtx   ctx;
  PetscInt       nloc,n;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  ctx->A=mat;
  ctx->D1=d1;
  ctx->D2=d2;
  if (d1) {
    PetscCall(VecDuplicate(d1,&ctx->W));
    PetscCall(PetscObjectReference((PetscObject)d1));
  } else {
    ctx->W = NULL;
  }
  if (d2) {
    PetscCall(VecDuplicate(d2,&ctx->W2));
    PetscCall(VecDuplicate(d2,&ctx->ADADiag));
    PetscCall(PetscObjectReference((PetscObject)d2));
  } else {
    ctx->W2      = NULL;
    ctx->ADADiag = NULL;
  }

  ctx->GotDiag = 0;
  PetscCall(PetscObjectReference((PetscObject)mat));

  PetscCall(VecGetLocalSize(d2,&nloc));
  PetscCall(VecGetSize(d2,&n));

  PetscCall(MatCreateShell(comm,nloc,nloc,n,n,ctx,J));
  PetscCall(MatShellSetManageScalingShifts(*J));
  PetscCall(MatShellSetOperation(*J,MATOP_MULT,(void(*)(void))MatMult_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_DESTROY,(void(*)(void))MatDestroy_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_VIEW,(void(*)(void))MatView_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_DIAGONAL_SET,(void(*)(void))MatDiagonalSet_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_SHIFT,(void(*)(void))MatShift_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_EQUAL,(void(*)(void))MatEqual_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_SCALE,(void(*)(void))MatScale_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_TRANSPOSE,(void(*)(void))MatTranspose_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_CREATE_SUBMATRICES,(void(*)(void))MatCreateSubMatrices_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_NORM,(void(*)(void))MatNorm_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_ADA));
  PetscCall(MatShellSetOperation(*J,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_ADA));

  PetscCall(PetscLogObjectParent((PetscObject)(*J),(PetscObject)ctx->W));
  PetscCall(PetscLogObjectParent((PetscObject)mat,(PetscObject)(*J)));

  PetscCall(MatSetOption(*J,MAT_SYMMETRIC,PETSC_TRUE));
  PetscFunctionReturn(0);
}
