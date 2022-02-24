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
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatMult(ctx->A,a,ctx->W));
  if (ctx->D1) {
    CHKERRQ(VecPointwiseMult(ctx->W,ctx->D1,ctx->W));
  }
  CHKERRQ(MatMultTranspose(ctx->A,ctx->W,y));
  if (ctx->D2) {
    CHKERRQ(VecPointwiseMult(ctx->W2, ctx->D2, a));
    CHKERRQ(VecAXPY(y, one, ctx->W2));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_ADA(Mat mat,Vec a,Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(MatMult_ADA(mat,a,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalSet_ADA(Mat M,Vec D, InsertMode mode)
{
  TaoMatADACtx   ctx;
  PetscReal      zero=0.0,one = 1.0;

  PetscFunctionBegin;
  PetscCheck(mode != INSERT_VALUES,PetscObjectComm((PetscObject)M),PETSC_ERR_SUP,"Cannot insert diagonal entries of this matrix type, can only add");
  CHKERRQ(MatShellGetContext(M,&ctx));
  if (!ctx->D2) {
    CHKERRQ(VecDuplicate(D,&ctx->D2));
    CHKERRQ(VecSet(ctx->D2, zero));
  }
  CHKERRQ(VecAXPY(ctx->D2, one, D));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_ADA(Mat mat)
{
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(VecDestroy(&ctx->W));
  CHKERRQ(VecDestroy(&ctx->W2));
  CHKERRQ(VecDestroy(&ctx->ADADiag));
  CHKERRQ(MatDestroy(&ctx->A));
  CHKERRQ(VecDestroy(&ctx->D1));
  CHKERRQ(VecDestroy(&ctx->D2));
  CHKERRQ(PetscFree(ctx));
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
  CHKERRQ(MatShellGetContext(Y,&ctx));
  CHKERRQ(VecShift(ctx->D2,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_ADA(Mat mat,MatDuplicateOption op,Mat *M)
{
  TaoMatADACtx      ctx;
  Mat               A2;
  Vec               D1b=NULL,D2b;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatDuplicate(ctx->A,op,&A2));
  if (ctx->D1) {
    CHKERRQ(VecDuplicate(ctx->D1,&D1b));
    CHKERRQ(VecCopy(ctx->D1,D1b));
  }
  CHKERRQ(VecDuplicate(ctx->D2,&D2b));
  CHKERRQ(VecCopy(ctx->D2,D2b));
  CHKERRQ(MatCreateADA(A2,D1b,D2b,M));
  if (ctx->D1) {
    CHKERRQ(PetscObjectDereference((PetscObject)D1b));
  }
  CHKERRQ(PetscObjectDereference((PetscObject)D2b));
  CHKERRQ(PetscObjectDereference((PetscObject)A2));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatEqual_ADA(Mat A,Mat B,PetscBool *flg)
{
  TaoMatADACtx   ctx1,ctx2;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx1));
  CHKERRQ(MatShellGetContext(B,&ctx2));
  CHKERRQ(VecEqual(ctx1->D2,ctx2->D2,flg));
  if (*flg==PETSC_TRUE) {
    CHKERRQ(VecEqual(ctx1->D1,ctx2->D1,flg));
  }
  if (*flg==PETSC_TRUE) {
    CHKERRQ(MatEqual(ctx1->A,ctx2->A,flg));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_ADA(Mat mat, PetscReal a)
{
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(VecScale(ctx->D1,a));
  if (ctx->D2) {
    CHKERRQ(VecScale(ctx->D2,a));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_ADA(Mat mat,MatReuse reuse,Mat *B)
{
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  if (reuse == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatDuplicate(mat,MAT_COPY_VALUES,B));
  } else if (reuse == MAT_REUSE_MATRIX) {
    CHKERRQ(MatCopy(mat,*B,SAME_NONZERO_PATTERN));
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Does not support inplace transpose");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatADAComputeDiagonal(Mat mat)
{
  PetscInt       i,m,n,low,high;
  PetscScalar    *dtemp,*dptr;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatGetOwnershipRange(mat, &low, &high));
  CHKERRQ(MatGetSize(mat,&m,&n));

  CHKERRQ(PetscMalloc1(n,&dtemp));
  for (i=0; i<n; i++) {
    CHKERRQ(MatGetColumnVector(ctx->A, ctx->W, i));
    CHKERRQ(VecPointwiseMult(ctx->W,ctx->W,ctx->W));
    CHKERRQ(VecDotBegin(ctx->D1, ctx->W,dtemp+i));
  }
  for (i=0; i<n; i++) {
    CHKERRQ(VecDotEnd(ctx->D1, ctx->W,dtemp+i));
  }

  CHKERRQ(VecGetArray(ctx->ADADiag,&dptr));
  for (i=low; i<high; i++) {
    dptr[i-low]= dtemp[i];
  }
  CHKERRQ(VecRestoreArray(ctx->ADADiag,&dptr));
  CHKERRQ(PetscFree(dtemp));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_ADA(Mat mat,Vec v)
{
  PetscReal       one=1.0;
  TaoMatADACtx    ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatADAComputeDiagonal(mat));
  CHKERRQ(VecCopy(ctx->ADADiag,v));
  if (ctx->D2) {
    CHKERRQ(VecAXPY(v, one, ctx->D2));
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
  CHKERRQ(ISEqual(isrow,iscol,&isequal));
  PetscCheck(isequal,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only for identical column and row indices");
  CHKERRQ(MatShellGetContext(mat,&ctx));

  CHKERRQ(MatGetOwnershipRange(ctx->A,&low,&high));
  CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)mat),high-low,low,1,&ISrow));
  CHKERRQ(MatCreateSubMatrix(ctx->A,ISrow,iscol,cll,&Atemp));
  CHKERRQ(ISDestroy(&ISrow));

  if (ctx->D1) {
    CHKERRQ(VecDuplicate(ctx->D1,&D1));
    CHKERRQ(VecCopy(ctx->D1,D1));
  } else {
    D1 = NULL;
  }

  if (ctx->D2) {
    Vec D2sub;

    CHKERRQ(VecGetSubVector(ctx->D2,isrow,&D2sub));
    CHKERRQ(VecDuplicate(D2sub,&D2));
    CHKERRQ(VecCopy(D2sub,D2));
    CHKERRQ(VecRestoreSubVector(ctx->D2,isrow,&D2sub));
  } else {
    D2 = NULL;
  }

  CHKERRQ(MatCreateADA(Atemp,D1,D2,newmat));
  CHKERRQ(MatShellGetContext(*newmat,&ctx));
  CHKERRQ(PetscObjectDereference((PetscObject)Atemp));
  if (ctx->D1) {
    CHKERRQ(PetscObjectDereference((PetscObject)D1));
  }
  if (ctx->D2) {
    CHKERRQ(PetscObjectDereference((PetscObject)D2));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_ADA(Mat A,PetscInt n, IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    CHKERRQ(PetscCalloc1(n+1,B));
  }
  for (i=0; i<n; i++) {
    CHKERRQ(MatCreateSubMatrix_ADA(A,irow[i],icol[i],scall,&(*B)[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_ADA(Mat mat,Vec Y, PetscInt col)
{
  PetscInt       low,high;
  PetscScalar    zero=0.0,one=1.0;

  PetscFunctionBegin;
  CHKERRQ(VecSet(Y, zero));
  CHKERRQ(VecGetOwnershipRange(Y,&low,&high));
  if (col>=low && col<high) {
    CHKERRQ(VecSetValue(Y,col,one,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(Y));
  CHKERRQ(VecAssemblyEnd(Y));
  CHKERRQ(MatMult_ADA(mat,Y,Y));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_ADA(Mat mat,MatType newtype,Mat *NewMat)
{
  PetscMPIInt    size;
  PetscBool      sametype, issame, isdense, isseqdense;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));

  CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,newtype,&sametype));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATSAME,&issame));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATMPIDENSE,&isdense));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATSEQDENSE,&isseqdense));

  if (sametype || issame) {
    CHKERRQ(MatDuplicate(mat,MAT_COPY_VALUES,NewMat));
  } else if (isdense) {
    PetscInt          i,j,low,high,m,n,M,N;
    const PetscScalar *dptr;
    Vec               X;

    CHKERRQ(VecDuplicate(ctx->D2,&X));
    CHKERRQ(MatGetSize(mat,&M,&N));
    CHKERRQ(MatGetLocalSize(mat,&m,&n));
    CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)mat),m,m,N,N,NULL,NewMat));
    CHKERRQ(MatGetOwnershipRange(*NewMat,&low,&high));
    for (i=0;i<M;i++) {
      CHKERRQ(MatGetColumnVector_ADA(mat,X,i));
      CHKERRQ(VecGetArrayRead(X,&dptr));
      for (j=0; j<high-low; j++) {
        CHKERRQ(MatSetValue(*NewMat,low+j,i,dptr[j],INSERT_VALUES));
      }
      CHKERRQ(VecRestoreArrayRead(X,&dptr));
    }
    CHKERRQ(MatAssemblyBegin(*NewMat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(*NewMat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(VecDestroy(&X));
  } else if (isseqdense && size==1) {
    PetscInt          i,j,low,high,m,n,M,N;
    const PetscScalar *dptr;
    Vec               X;

    CHKERRQ(VecDuplicate(ctx->D2,&X));
    CHKERRQ(MatGetSize(mat,&M,&N));
    CHKERRQ(MatGetLocalSize(mat,&m,&n));
    CHKERRQ(MatCreateSeqDense(PetscObjectComm((PetscObject)mat),N,N,NULL,NewMat));
    CHKERRQ(MatGetOwnershipRange(*NewMat,&low,&high));
    for (i=0;i<M;i++) {
      CHKERRQ(MatGetColumnVector_ADA(mat,X,i));
      CHKERRQ(VecGetArrayRead(X,&dptr));
      for (j=0; j<high-low; j++) {
        CHKERRQ(MatSetValue(*NewMat,low+j,i,dptr[j],INSERT_VALUES));
      }
      CHKERRQ(VecRestoreArrayRead(X,&dptr));
    }
    CHKERRQ(MatAssemblyBegin(*NewMat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(*NewMat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(VecDestroy(&X));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"No support to convert objects to that type");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNorm_ADA(Mat mat,NormType type,PetscReal *norm)
{
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
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
  CHKERRQ(PetscNew(&ctx));
  ctx->A=mat;
  ctx->D1=d1;
  ctx->D2=d2;
  if (d1) {
    CHKERRQ(VecDuplicate(d1,&ctx->W));
    CHKERRQ(PetscObjectReference((PetscObject)d1));
  } else {
    ctx->W = NULL;
  }
  if (d2) {
    CHKERRQ(VecDuplicate(d2,&ctx->W2));
    CHKERRQ(VecDuplicate(d2,&ctx->ADADiag));
    CHKERRQ(PetscObjectReference((PetscObject)d2));
  } else {
    ctx->W2      = NULL;
    ctx->ADADiag = NULL;
  }

  ctx->GotDiag = 0;
  CHKERRQ(PetscObjectReference((PetscObject)mat));

  CHKERRQ(VecGetLocalSize(d2,&nloc));
  CHKERRQ(VecGetSize(d2,&n));

  CHKERRQ(MatCreateShell(comm,nloc,nloc,n,n,ctx,J));
  CHKERRQ(MatShellSetManageScalingShifts(*J));
  CHKERRQ(MatShellSetOperation(*J,MATOP_MULT,(void(*)(void))MatMult_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_DESTROY,(void(*)(void))MatDestroy_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_VIEW,(void(*)(void))MatView_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_DIAGONAL_SET,(void(*)(void))MatDiagonalSet_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_SHIFT,(void(*)(void))MatShift_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_EQUAL,(void(*)(void))MatEqual_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_SCALE,(void(*)(void))MatScale_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_TRANSPOSE,(void(*)(void))MatTranspose_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_CREATE_SUBMATRICES,(void(*)(void))MatCreateSubMatrices_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_NORM,(void(*)(void))MatNorm_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_ADA));
  CHKERRQ(MatShellSetOperation(*J,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_ADA));

  CHKERRQ(PetscLogObjectParent((PetscObject)(*J),(PetscObject)ctx->W));
  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)(*J)));

  CHKERRQ(MatSetOption(*J,MAT_SYMMETRIC,PETSC_TRUE));
  PetscFunctionReturn(0);
}
