#include "normalmat.h"                /*I  "mat.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "MatCreateADA"
/*@C
   MatCreateADA - Creates a matrix M=A^T D1 A + D2 where D1, D2 are diagonal
   
   Collective on matrix

   Input Parameters:
+  mat - matrix of arbitrary type
.  d1 - A vector with diagonal elements of D1
-  d2 - A vector

   Output Parameters:
.  J - New matrix whose operations are defined in terms of mat, D1, and D2.

   Notes: 
   The user provides the input data and is responsible for destroying
   this data after matrix J has been destroyed.  
   The operation MatMult(A,D2,D1) must be well defined.
   Before calling the operation MatGetDiagonal(), the function 
   MatADAComputeDiagonal() must be called.  The matrices A and D1 must
   be the same during calls to MatADAComputeDiagonal() and
   MatGetDiagonal().

   Level: developer

.seealso: MatCreate()
@*/
int MatCreateADA(Mat mat,Vec d1, Vec d2, Mat *J)
{
  MPI_Comm     comm=mat->comm;
  TaoMatADACtx ctx;
  int          info,nloc,n;

  PetscFunctionBegin;
  /*
  info=MatCheckVecs(mat,d1,D2,&flg);CHKERRQ(info);
  if (flg==PETSC_FALSE){
    SETERRQ(PETSC_ERR_SUP,"InCompatible matrix and vector for ADA^T matrix");
  }
  */
  info = PetscNew(_p_TaoMatADACtx,&ctx);CHKERRQ(info);

  ctx->A=mat;
  ctx->D1=d1;
  ctx->D2=d2;
  if (d1){
    info = VecDuplicate(d1,&ctx->W);CHKERRQ(info);
    info =  PetscObjectReference((PetscObject)d1);CHKERRQ(info);
  } else {
    ctx->W=0;
  }
  if (D2){
    info = VecDuplicate(d2,&ctx->W2);CHKERRQ(info);
    info = VecDuplicate(d2,&ctx->ADADiag);CHKERRQ(info);
    info =  PetscObjectReference((PetscObject)d2);CHKERRQ(info);
  } else {
    ctx->W2=0;
    ctx->ADADiag=0;
  }

  ctx->GotDiag=0;
  info =  PetscObjectReference((PetscObject)mat);CHKERRQ(info);

  info=VecGetLocalSize(d2,&nloc);CHKERRQ(info);
  info=VecGetSize(d2,&n);CHKERRQ(info);

  info = MatCreateShell(comm,nloc,nloc,n,n,ctx,J);CHKERRQ(info);

  info = MatShellSetOperation(*J,MATOP_MULT,(void(*)())MatMult_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_DESTROY,(void(*)())MatDestroy_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_VIEW,(void(*)())MatView_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_MULT_TRANSPOSE,(void(*)())MatMultTranspose_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_DIAGONAL_SHIFT,(void(*)())MatDiagonalShift_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_SHIFT,(void(*)())MatShift_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_EQUAL,(void(*)())MatEqual_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_SCALE,(void(*)())MatScale_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_TRANSPOSE,(void(*)())MatTranspose_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_GET_SUBMATRICES,(void(*)())MatGetSubMatrices_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_NORM,(void(*)())MatNorm_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_DUPLICATE,(void(*)())MatDuplicate_ADA);CHKERRQ(info);
  info = MatShellSetOperation(*J,MATOP_GET_SUBMATRIX,(void(*)())MatGetSubMatrix_ADA);CHKERRQ(info);

  info = PetscLogObjectParent(*J,ctx->W); CHKERRQ(info);
  info = PetscLogObjectParent(mat,*J); CHKERRQ(info);

  info = MatSetOption(*J,MAT_SYMMETRIC);CHKERRQ(info);
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_ADA"
int MatMult_ADA(Mat mat,Vec a,Vec y)
{
  TaoMatADACtx ctx;
  PetscScalar        one = 1.0;
  int           info;

  PetscFunctionBegin;
  info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);

  info = MatMult(ctx->A,a,ctx->W);CHKERRQ(info);
  if (ctx->d1){
    info = VecPointwiseMult(ctx->W,ctx->d1,ctx->W);CHKERRQ(info);
  }
  info = MatMultTranspose(ctx->A,ctx->W,y);CHKERRQ(info);
  if (ctx->D2){
    info = VecPointwiseMult(ctx->W2, ctx->D2, a);CHKERRQ(info);
    info = VecAXPY(y, one, ctx->W2);CHKERRQ(info);
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_ADA"
int MatMultTranspose_ADA(Mat mat,Vec a,Vec y)
{
  int info;

  PetscFunctionBegin;
  info = MatMult_ADA(mat,a,y);CHKERRQ(info);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalShift_ADA"
int MatDiagonalShift_ADA(Vec D, Mat M)
{
  TaoMatADACtx ctx;
  PetscScalar        zero=0.0,one = 1.0;
  int           info;

  PetscFunctionBegin;
  info = MatShellGetContext(M,(void **)&ctx);CHKERRQ(info);

  if (ctx->D2==PETSC_NULL){
    info = VecDuplicate(D,&ctx->D2);CHKERRQ(info);
    info = VecSet(ctx->D2, zero);CHKERRQ(info);
  }
  info = VecAXPY(ctx->D2, one, D);CHKERRQ(info);

  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_ADA"
int MatDestroy_ADA(Mat mat)
{
  int          info;
  TaoMatADACtx ctx;

  PetscFunctionBegin;
  info=MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);
  info=VecDestroy(ctx->W);CHKERRQ(info);
  info=VecDestroy(ctx->W2);CHKERRQ(info);
  info=VecDestroy(ctx->ADADiag);CHKERRQ(info);
  info=MatDestroy(ctx->A);CHKERRQ(info);
  info=VecDestroy(ctx->D1);CHKERRQ(info);
  info=VecDestroy(ctx->D2);CHKERRQ(info);
  info = PetscFree(ctx); CHKERRQ(info);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_ADA"
int MatView_ADA(Mat mat,PetscViewer viewer)
{

  PetscFunctionBegin;
  /*
  info = ViewerGetFormat(viewer,&format);CHKERRQ(info);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_LONG) {
    PetscFunctionReturn(0);  / * do nothing for now * /
  }
  info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);
  info = MatView(ctx->A,viewer);CHKERRQ(info);
  if (ctx->D1){
    info = VecView(ctx->D1,viewer);CHKERRQ(info);
  }
  if (ctx->D2){
    info = VecView(ctx->D2,viewer);CHKERRQ(info);
  }
  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShift_ADA"
int MatShift_ADA(Mat Y, PetscScalar a)
{
  int          info;
  TaoMatADACtx ctx;

  PetscFunctionBegin;
  info = MatShellGetContext(Y,(void **)&ctx);CHKERRQ(info);
  info = VecShift(ctx->D2,a);CHKERRQ(info);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_ADA"
int MatDuplicate_ADA(Mat mat,MatDuplicateOption op,Mat *M)
{
  int          info;
  TaoMatADACtx ctx;
  Mat          A2;
  Vec          D1b=NULL,D2b;

  PetscFunctionBegin;
  info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);
  info = MatDuplicate(ctx->A,op,&A2);CHKERRQ(info);
  if (ctx->D1){
    info = VecDuplicate(ctx->D1,&D1b);CHKERRQ(info);
    info = VecCopy(ctx->D1,D1b);CHKERRQ(info);
  }
  info = VecDuplicate(ctx->D2,&D2b);CHKERRQ(info);
  info = VecCopy(ctx->D2,D2b);CHKERRQ(info);
  info = MatCreateADA(A2,D1b,D2b,M);CHKERRQ(info);
  if (ctx->D1){
  info = PetscObjectDereference((PetscObject)D1b);CHKERRQ(info);
  }
  info = PetscObjectDereference((PetscObject)D2b);CHKERRQ(info);
  info = PetscObjectDereference((PetscObject)A2);CHKERRQ(info);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_ADA"
int MatEqual_ADA(Mat A,Mat B,PetscTruth *flg)
{
  int          info;
  TaoMatADACtx  ctx1,ctx2;

  PetscFunctionBegin;
  info = MatShellGetContext(A,(void **)&ctx1);CHKERRQ(info);
  info = MatShellGetContext(B,(void **)&ctx2);CHKERRQ(info);
  info = VecEqual(ctx1->D2,ctx2->D2,flg);CHKERRQ(info);
  if (*flg==PETSC_TRUE){
    info = VecEqual(ctx1->D1,ctx2->D1,flg);CHKERRQ(info);
  }
  if (*flg==PETSC_TRUE){
    info = MatEqual(ctx1->A,ctx2->A,flg);CHKERRQ(info);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_ADA"
int MatScale_ADA(Mat mat, PetscScalar a)
{
  int          info;
  TaoMatADACtx ctx;

  PetscFunctionBegin;
  info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);
  info = VecScale(ctx->D1,a);CHKERRQ(info);
  if (ctx->D2){
    info = VecScale(ctx->D2,a);CHKERRQ(info);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_ADA"
int MatTranspose_ADA(Mat mat,Mat *B)
{
  int          info;
  TaoMatADACtx ctx;

  PetscFunctionBegin;
  if (*B){
    info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);
    info = MatDuplicate(mat,MAT_COPY_VALUES,B);CHKERRQ(info);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatADAComputeDiagonal"
int MatADAComputeDiagonal(Mat mat)
{
  int          i,m,n,low,high,info;
  PetscScalar       *dtemp,*dptr;
  TaoMatADACtx ctx;

  PetscFunctionBegin;
  info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);

  info = MatGetOwnershipRange(mat, &low, &high);CHKERRQ(info);
  info = MatGetSize(mat,&m,&n);CHKERRQ(info);
  
  info = PetscMalloc( n*sizeof(PetscScalar),&dtemp ); CHKERRQ(info);

  for (i=0; i<n; i++){
    info = MatGetColumnVector(ctx->A, ctx->W, i);CHKERRQ(info);
    info = VecPointwiseMult(ctx->W,ctx->W,ctx->W);CHKERRQ(info);
    info = VecDotBegin(ctx->D1, ctx->W,dtemp+i);CHKERRQ(info);
  }
  for (i=0; i<n; i++){
    info = VecDotEnd(ctx->D1, ctx->W,dtemp+i);CHKERRQ(info);
  } 

  info = VecGetArray(ctx->ADADiag,&dptr);CHKERRQ(info);
  for (i=low; i<high; i++){
    dptr[i-low]= dtemp[i];
  }
  info = VecRestoreArray(ctx->ADADiag,&dptr);CHKERRQ(info);
  if (dtemp) {
    info = PetscFree(dtemp); CHKERRQ(info);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_ADA"
int MatGetDiagonal_ADA(Mat mat,Vec v)
{
  int          info;
  PetscScalar       one=1.0;
  TaoMatADACtx ctx;

  PetscFunctionBegin;
  info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);
  info = MatADAComputeDiagonal(mat);
  info=VecCopy(ctx->ADADiag,v);CHKERRQ(info);
  if (ctx->D2){
    info=VecAXPY(v, one, ctx->D2);CHKERRQ(info);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_ADA"
int MatGetSubMatrices_ADA(Mat A,int n, IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  int info,i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    info = PetscMalloc( (n+1)*sizeof(Mat),B );CHKERRQ(info);
  }

  for ( i=0; i<n; i++ ) {
    info = MatGetSubMatrix_ADA(A,irow[i],icol[i],PETSC_DECIDE,scall,&(*B)[i]);CHKERRQ(info);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_ADA"
int MatGetSubMatrix_ADA(Mat mat,IS isrow,IS iscol,int csize,MatReuse cll,
			Mat *newmat)
{
  int          info,low,high;
  int          n,nlocal,i;
  int          *iptr;
  PetscScalar       *dptr,*ddptr,zero=0.0;
  VecType type_name;
  IS           ISrow;
  Vec          D1,D2;
  Mat          Atemp;
  TaoMatADACtx ctx;

  PetscFunctionBegin;

  info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);

  info = MatGetOwnershipRange(ctx->A,&low,&high);CHKERRQ(info);
  info = ISCreateStride(mat->comm,high-low,low,1,&ISrow);CHKERRQ(info);
  info = MatGetSubMatrix(ctx->A,ISrow,iscol,csize,cll,&Atemp);CHKERRQ(info);
  info = ISDestroy(ISrow);CHKERRQ(info);

  if (ctx->D1){
    info=VecDuplicate(ctx->D1,&D1);CHKERRQ(info);
    info=VecCopy(ctx->D1,D1);CHKERRQ(info);
  } else {
    D1=PETSC_NULL;
  }

  if (ctx->D2){
    info=ISGetSize(isrow,&n);CHKERRQ(info);
    info=ISGetLocalSize(isrow,&nlocal);CHKERRQ(info);
    info=VecCreate(ctx->D2->comm,&D2);CHKERRQ(info);
    info=VecGetType(ctx->D2,&type_name);CHKERRQ(info);
    info=VecSetSizes(D2,nlocal,n);CHKERRQ(info);
    info=VecSetType(D2,type_name);CHKERRQ(info);
    info=VecSet(D2, zero);CHKERRQ(info);
    info=VecGetArray(ctx->D2, &dptr); CHKERRQ(info);
    info=VecGetArray(D2, &ddptr); CHKERRQ(info);
    info=ISGetIndices(isrow,&iptr); CHKERRQ(info);
    for (i=0;i<nlocal;i++){
      ddptr[i] = dptr[iptr[i]-low];
    }
    info=ISRestoreIndices(isrow,&iptr); CHKERRQ(info);
    info=VecRestoreArray(D2, &ddptr); CHKERRQ(info);
    info=VecRestoreArray(ctx->D2, &dptr); CHKERRQ(info);
   
  } else {
    D2=PETSC_NULL;
  }

  info = MatCreateADA(Atemp,D1,D2,newmat);CHKERRQ(info);
  info = MatShellGetContext(*newmat,(void **)&ctx);CHKERRQ(info);
  info = PetscObjectDereference((PetscObject)Atemp);CHKERRQ(info);
  if (ctx->D1){
    info = PetscObjectDereference((PetscObject)D1);CHKERRQ(info);
  }
  if (ctx->D2){
    info = PetscObjectDereference((PetscObject)D2);CHKERRQ(info);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowADA"
int MatGetRowADA(Mat mat,int row,int *ncols,int **cols,PetscScalar **vals)
{
  int info,m,n;

  PetscFunctionBegin;
  info = MatGetSize(mat,&m,&n);CHKERRQ(info);

  if (*ncols>0){
    info = PetscMalloc( (*ncols)*sizeof(int),cols );CHKERRQ(info);
    info = PetscMalloc( (*ncols)*sizeof(PetscScalar),vals );CHKERRQ(info);
  } else {
    *cols=PETSC_NULL;
    *vals=PETSC_NULL;
  }
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowADA"
int MatRestoreRowADA(Mat mat,int row,int *ncols,int **cols,PetscScalar **vals)
{
  int info;
  PetscFunctionBegin;
  if (*ncols>0){
    info = PetscFree(*cols);  CHKERRQ(info);
    info = PetscFree(*vals);  CHKERRQ(info);
  }
  *cols=PETSC_NULL;
  *vals=PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnVectorADA"
int MatGetColumnVector_ADA(Mat mat,Vec Y, int col)
{
  int    info,low,high;
  PetscScalar zero=0.0,one=1.0;

  PetscFunctionBegin;
  info=VecSet(Y, zero);CHKERRQ(info);
  info=VecGetOwnershipRange(Y,&low,&high);CHKERRQ(info);
  if (col>=low && col<high){
    info=VecSetValue(Y,col,one,INSERT_VALUES);CHKERRQ(info);
  }
  info=VecAssemblyBegin(Y);CHKERRQ(info);
  info=VecAssemblyEnd(Y);CHKERRQ(info);
  info=MatMult_ADA(mat,Y,Y);CHKERRQ(info);

  PetscFunctionReturn(0);
}

int MatConvert_ADA(Mat mat,MatType newtype,Mat *NewMat)
{
  int info,size;
  TaoMatADACtx  ctx;

  PetscFunctionBegin;
  info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);
  MPI_Comm_size(mat->comm,&size);

  if (newtype==MATSAME){

    info=MatDuplicate(mat,MAT_COPY_VALUES,NewMat);CHKERRQ(info);

  } else if (newtype==MATMPIDENSE){

    int i,j,low,high,m,n,M,N;
    PetscScalar *dptr;
    Vec X;

    info = VecDuplicate(ctx->D2,&X);CHKERRQ(info);
    info=MatGetSize(mat,&M,&N);CHKERRQ(info);
    info=MatGetLocalSize(mat,&m,&n);CHKERRQ(info);
    info = MatCreateMPIDense(mat->comm,m,m,N,N,PETSC_NULL,NewMat);
    CHKERRQ(info);
    info = MatGetOwnershipRange(*NewMat,&low,&high);CHKERRQ(info);
    for (i=0;i<M;i++){
      info = MatGetColumnVector_ADA(mat,X,i);CHKERRQ(info);
      info = VecGetArray(X,&dptr);CHKERRQ(info);
      for (j=0; j<high-low; j++){
	info = MatSetValue(*NewMat,low+j,i,dptr[j],INSERT_VALUES);CHKERRQ(info);
      }
      info=VecRestoreArray(X,&dptr);CHKERRQ(info);
    }
    info=MatAssemblyBegin(*NewMat,MAT_FINAL_ASSEMBLY);CHKERRQ(info);
    info=MatAssemblyEnd(*NewMat,MAT_FINAL_ASSEMBLY);CHKERRQ(info);
    info = VecDestroy(X);CHKERRQ(info);

  } else if (newtype==MATSEQDENSE && size==1){

    int i,j,low,high,m,n,M,N;
    PetscScalar *dptr;
    Vec X;

    info = VecDuplicate(ctx->D2,&X);CHKERRQ(info);
    info = MatGetSize(mat,&M,&N);CHKERRQ(info);
    info = MatGetLocalSize(mat,&m,&n);CHKERRQ(info);
    info = MatCreateSeqDense(mat->comm,N,N,PETSC_NULL,NewMat);
    CHKERRQ(info);
    info = MatGetOwnershipRange(*NewMat,&low,&high);CHKERRQ(info);
    for (i=0;i<M;i++){
      info = MatGetColumnVector_ADA(mat,X,i);CHKERRQ(info);
      info = VecGetArray(X,&dptr);CHKERRQ(info);
      for (j=0; j<high-low; j++){
	info = MatSetValue(*NewMat,low+j,i,dptr[j],INSERT_VALUES);CHKERRQ(info);
      }
      info=VecRestoreArray(X,&dptr);CHKERRQ(info);
    }
    info=MatAssemblyBegin(*NewMat,MAT_FINAL_ASSEMBLY);CHKERRQ(info);
    info=MatAssemblyEnd(*NewMat,MAT_FINAL_ASSEMBLY);CHKERRQ(info);
    info=VecDestroy(X);CHKERRQ(info);

  } else {
    SETERRQ(1,"No support to convert objects to that type");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_ADA"
int MatNorm_ADA(Mat mat,NormType type,PetscReal *norm)
{
  int info;
  TaoMatADACtx  ctx;

  PetscFunctionBegin;
  info = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(info);

  if (type == NORM_FROBENIUS) {
    *norm = 1.0;
  } else if (type == NORM_1 || type == NORM_INFINITY) {
    *norm = 1.0;
  } else {
    SETERRQ(PETSC_ERR_SUP,"No two norm");
  }
  PetscFunctionReturn(0);
}
