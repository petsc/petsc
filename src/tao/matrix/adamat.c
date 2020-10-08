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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,a,ctx->W);CHKERRQ(ierr);
  if (ctx->D1){
    ierr = VecPointwiseMult(ctx->W,ctx->D1,ctx->W);CHKERRQ(ierr);
  }
  ierr = MatMultTranspose(ctx->A,ctx->W,y);CHKERRQ(ierr);
  if (ctx->D2){
    ierr = VecPointwiseMult(ctx->W2, ctx->D2, a);CHKERRQ(ierr);
    ierr = VecAXPY(y, one, ctx->W2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_ADA(Mat mat,Vec a,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult_ADA(mat,a,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalSet_ADA(Mat M,Vec D, InsertMode mode)
{
  TaoMatADACtx   ctx;
  PetscReal      zero=0.0,one = 1.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mode == INSERT_VALUES) SETERRQ(PetscObjectComm((PetscObject)M),PETSC_ERR_SUP,"Cannot insert diagonal entries of this matrix type, can only add");
  ierr = MatShellGetContext(M,(void **)&ctx);CHKERRQ(ierr);
  if (!ctx->D2){
    ierr = VecDuplicate(D,&ctx->D2);CHKERRQ(ierr);
    ierr = VecSet(ctx->D2, zero);CHKERRQ(ierr);
  }
  ierr = VecAXPY(ctx->D2, one, D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_ADA(Mat mat)
{
  PetscErrorCode ierr;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  ierr=MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr=VecDestroy(&ctx->W);CHKERRQ(ierr);
  ierr=VecDestroy(&ctx->W2);CHKERRQ(ierr);
  ierr=VecDestroy(&ctx->ADADiag);CHKERRQ(ierr);
  ierr=MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr=VecDestroy(&ctx->D1);CHKERRQ(ierr);
  ierr=VecDestroy(&ctx->D2);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_ADA(Mat mat,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_ADA(Mat Y, PetscReal a)
{
  PetscErrorCode ierr;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(Y,(void **)&ctx);CHKERRQ(ierr);
  ierr = VecShift(ctx->D2,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_ADA(Mat mat,MatDuplicateOption op,Mat *M)
{
  PetscErrorCode    ierr;
  TaoMatADACtx      ctx;
  Mat               A2;
  Vec               D1b=NULL,D2b;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatDuplicate(ctx->A,op,&A2);CHKERRQ(ierr);
  if (ctx->D1){
    ierr = VecDuplicate(ctx->D1,&D1b);CHKERRQ(ierr);
    ierr = VecCopy(ctx->D1,D1b);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(ctx->D2,&D2b);CHKERRQ(ierr);
  ierr = VecCopy(ctx->D2,D2b);CHKERRQ(ierr);
  ierr = MatCreateADA(A2,D1b,D2b,M);CHKERRQ(ierr);
  if (ctx->D1){
    ierr = PetscObjectDereference((PetscObject)D1b);CHKERRQ(ierr);
  }
  ierr = PetscObjectDereference((PetscObject)D2b);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)A2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatEqual_ADA(Mat A,Mat B,PetscBool *flg)
{
  PetscErrorCode ierr;
  TaoMatADACtx   ctx1,ctx2;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&ctx1);CHKERRQ(ierr);
  ierr = MatShellGetContext(B,(void **)&ctx2);CHKERRQ(ierr);
  ierr = VecEqual(ctx1->D2,ctx2->D2,flg);CHKERRQ(ierr);
  if (*flg==PETSC_TRUE){
    ierr = VecEqual(ctx1->D1,ctx2->D1,flg);CHKERRQ(ierr);
  }
  if (*flg==PETSC_TRUE){
    ierr = MatEqual(ctx1->A,ctx2->A,flg);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_ADA(Mat mat, PetscReal a)
{
  PetscErrorCode ierr;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = VecScale(ctx->D1,a);CHKERRQ(ierr);
  if (ctx->D2){
    ierr = VecScale(ctx->D2,a);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_ADA(Mat mat,MatReuse reuse,Mat *B)
{
  PetscErrorCode ierr;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX){
    ierr = MatDuplicate(mat,MAT_COPY_VALUES,B);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX){
    ierr = MatCopy(mat,*B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Does not support inplace transpose");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatADAComputeDiagonal(Mat mat)
{
  PetscErrorCode ierr;
  PetscInt       i,m,n,low,high;
  PetscScalar    *dtemp,*dptr;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat, &low, &high);CHKERRQ(ierr);
  ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);

  ierr = PetscMalloc1(n,&dtemp);CHKERRQ(ierr);
  for (i=0; i<n; i++){
    ierr = MatGetColumnVector(ctx->A, ctx->W, i);CHKERRQ(ierr);
    ierr = VecPointwiseMult(ctx->W,ctx->W,ctx->W);CHKERRQ(ierr);
    ierr = VecDotBegin(ctx->D1, ctx->W,dtemp+i);CHKERRQ(ierr);
  }
  for (i=0; i<n; i++){
    ierr = VecDotEnd(ctx->D1, ctx->W,dtemp+i);CHKERRQ(ierr);
  }

  ierr = VecGetArray(ctx->ADADiag,&dptr);CHKERRQ(ierr);
  for (i=low; i<high; i++){
    dptr[i-low]= dtemp[i];
  }
  ierr = VecRestoreArray(ctx->ADADiag,&dptr);CHKERRQ(ierr);
  ierr = PetscFree(dtemp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_ADA(Mat mat,Vec v)
{
  PetscErrorCode  ierr;
  PetscReal       one=1.0;
  TaoMatADACtx    ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatADAComputeDiagonal(mat);CHKERRQ(ierr);
  ierr = VecCopy(ctx->ADADiag,v);CHKERRQ(ierr);
  if (ctx->D2){
    ierr = VecAXPY(v, one, ctx->D2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_ADA(Mat mat,IS isrow,IS iscol,MatReuse cll, Mat *newmat)
{
  PetscErrorCode    ierr;
  PetscInt          low,high;
  IS                ISrow;
  Vec               D1,D2;
  Mat               Atemp;
  TaoMatADACtx      ctx;
  PetscBool         isequal;

  PetscFunctionBegin;
  ierr = ISEqual(isrow,iscol,&isequal);CHKERRQ(ierr);
  if (!isequal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only for identical column and row indices");
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(ctx->A,&low,&high);CHKERRQ(ierr);
  ierr = ISCreateStride(PetscObjectComm((PetscObject)mat),high-low,low,1,&ISrow);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(ctx->A,ISrow,iscol,cll,&Atemp);CHKERRQ(ierr);
  ierr = ISDestroy(&ISrow);CHKERRQ(ierr);

  if (ctx->D1){
    ierr=VecDuplicate(ctx->D1,&D1);CHKERRQ(ierr);
    ierr=VecCopy(ctx->D1,D1);CHKERRQ(ierr);
  } else {
    D1 = NULL;
  }

  if (ctx->D2){
    Vec D2sub;

    ierr=VecGetSubVector(ctx->D2,isrow,&D2sub);CHKERRQ(ierr);
    ierr=VecDuplicate(D2sub,&D2);CHKERRQ(ierr);
    ierr=VecCopy(D2sub,D2);CHKERRQ(ierr);
    ierr=VecRestoreSubVector(ctx->D2,isrow,&D2sub);CHKERRQ(ierr);
  } else {
    D2 = NULL;
  }

  ierr = MatCreateADA(Atemp,D1,D2,newmat);CHKERRQ(ierr);
  ierr = MatShellGetContext(*newmat,(void **)&ctx);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)Atemp);CHKERRQ(ierr);
  if (ctx->D1){
    ierr = PetscObjectDereference((PetscObject)D1);CHKERRQ(ierr);
  }
  if (ctx->D2){
    ierr = PetscObjectDereference((PetscObject)D2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_ADA(Mat A,PetscInt n, IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscCalloc1(n+1,B);CHKERRQ(ierr);
  }
  for (i=0; i<n; i++) {
    ierr = MatCreateSubMatrix_ADA(A,irow[i],icol[i],scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_ADA(Mat mat,Vec Y, PetscInt col)
{
  PetscErrorCode ierr;
  PetscInt       low,high;
  PetscScalar    zero=0.0,one=1.0;

  PetscFunctionBegin;
  ierr=VecSet(Y, zero);CHKERRQ(ierr);
  ierr=VecGetOwnershipRange(Y,&low,&high);CHKERRQ(ierr);
  if (col>=low && col<high){
    ierr=VecSetValue(Y,col,one,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr=VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr=VecAssemblyEnd(Y);CHKERRQ(ierr);
  ierr=MatMult_ADA(mat,Y,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_ADA(Mat mat,MatType newtype,Mat *NewMat)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscBool      sametype, issame, isdense, isseqdense;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRMPI(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)mat,newtype,&sametype);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSAME,&issame);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIDENSE,&isdense);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSEQDENSE,&isseqdense);CHKERRQ(ierr);

  if (sametype || issame) {
    ierr=MatDuplicate(mat,MAT_COPY_VALUES,NewMat);CHKERRQ(ierr);
  } else if (isdense) {
    PetscInt          i,j,low,high,m,n,M,N;
    const PetscScalar *dptr;
    Vec               X;

    ierr = VecDuplicate(ctx->D2,&X);CHKERRQ(ierr);
    ierr=MatGetSize(mat,&M,&N);CHKERRQ(ierr);
    ierr=MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
    ierr = MatCreateDense(PetscObjectComm((PetscObject)mat),m,m,N,N,NULL,NewMat);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(*NewMat,&low,&high);CHKERRQ(ierr);
    for (i=0;i<M;i++){
      ierr = MatGetColumnVector_ADA(mat,X,i);CHKERRQ(ierr);
      ierr = VecGetArrayRead(X,&dptr);CHKERRQ(ierr);
      for (j=0; j<high-low; j++){
        ierr = MatSetValue(*NewMat,low+j,i,dptr[j],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayRead(X,&dptr);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(*NewMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*NewMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecDestroy(&X);CHKERRQ(ierr);
  } else if (isseqdense && size==1){
    PetscInt          i,j,low,high,m,n,M,N;
    const PetscScalar *dptr;
    Vec               X;

    ierr = VecDuplicate(ctx->D2,&X);CHKERRQ(ierr);
    ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(mat,&m,&n);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PetscObjectComm((PetscObject)mat),N,N,NULL,NewMat);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(*NewMat,&low,&high);CHKERRQ(ierr);
    for (i=0;i<M;i++){
      ierr = MatGetColumnVector_ADA(mat,X,i);CHKERRQ(ierr);
      ierr = VecGetArrayRead(X,&dptr);CHKERRQ(ierr);
      for (j=0; j<high-low; j++){
        ierr = MatSetValue(*NewMat,low+j,i,dptr[j],INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayRead(X,&dptr);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(*NewMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*NewMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecDestroy(&X);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"No support to convert objects to that type");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNorm_ADA(Mat mat,NormType type,PetscReal *norm)
{
  PetscErrorCode ierr;
  TaoMatADACtx   ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       nloc,n;

  PetscFunctionBegin;
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->A=mat;
  ctx->D1=d1;
  ctx->D2=d2;
  if (d1){
    ierr = VecDuplicate(d1,&ctx->W);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)d1);CHKERRQ(ierr);
  } else {
    ctx->W = NULL;
  }
  if (d2){
    ierr = VecDuplicate(d2,&ctx->W2);CHKERRQ(ierr);
    ierr = VecDuplicate(d2,&ctx->ADADiag);CHKERRQ(ierr);
    ierr =  PetscObjectReference((PetscObject)d2);CHKERRQ(ierr);
  } else {
    ctx->W2      = NULL;
    ctx->ADADiag = NULL;
  }

  ctx->GotDiag = 0;
  ierr =  PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);

  ierr=VecGetLocalSize(d2,&nloc);CHKERRQ(ierr);
  ierr=VecGetSize(d2,&n);CHKERRQ(ierr);

  ierr = MatCreateShell(comm,nloc,nloc,n,n,ctx,J);CHKERRQ(ierr);
  ierr = MatShellSetManageScalingShifts(*J);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT,(void(*)(void))MatMult_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DESTROY,(void(*)(void))MatDestroy_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_VIEW,(void(*)(void))MatView_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DIAGONAL_SET,(void(*)(void))MatDiagonalSet_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_SHIFT,(void(*)(void))MatShift_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_EQUAL,(void(*)(void))MatEqual_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_SCALE,(void(*)(void))MatScale_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_TRANSPOSE,(void(*)(void))MatTranspose_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_CREATE_SUBMATRICES,(void(*)(void))MatCreateSubMatrices_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_NORM,(void(*)(void))MatNorm_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_ADA);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_ADA);CHKERRQ(ierr);

  ierr = PetscLogObjectParent((PetscObject)(*J),(PetscObject)ctx->W);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)(*J));CHKERRQ(ierr);

  ierr = MatSetOption(*J,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
