#include <petsctao.h>   /*I "petsctao.h" I*/
#include <../src/tao/matrix/submatfree.h> /*I "submatfree.h" I*/

/*@C
  MatCreateSubMatrixFree - Creates a reduced matrix by masking a
  full matrix.

   Collective on matrix

   Input Parameters:
+  mat - matrix of arbitrary type
.  Rows - the rows that will be in the submatrix
-  Cols - the columns that will be in the submatrix

   Output Parameters:
.  J - New matrix

   Notes:
   The caller is responsible for destroying the input objects after matrix J has been destroyed.

   Level: developer

.seealso: MatCreate()
@*/
PetscErrorCode MatCreateSubMatrixFree(Mat mat,IS Rows, IS Cols, Mat *J)
{
  MPI_Comm         comm=PetscObjectComm((PetscObject)mat);
  MatSubMatFreeCtx ctx;
  PetscErrorCode   ierr;
  PetscInt         mloc,nloc,m,n;

  PetscFunctionBegin;
  ierr    = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->A  = mat;
  ierr    = MatGetSize(mat,&m,&n);CHKERRQ(ierr);
  ierr    = MatGetLocalSize(mat,&mloc,&nloc);CHKERRQ(ierr);
  ierr    = MatCreateVecs(mat,NULL,&ctx->VC);CHKERRQ(ierr);
  ctx->VR = ctx->VC;
  ierr    =  PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);

  ctx->Rows = Rows;
  ctx->Cols = Cols;
  ierr = PetscObjectReference((PetscObject)Rows);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Cols);CHKERRQ(ierr);
  ierr = MatCreateShell(comm,mloc,nloc,m,n,ctx,J);CHKERRQ(ierr);
  ierr = MatShellSetManageScalingShifts(*J);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT,(void(*)(void))MatMult_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DESTROY,(void(*)(void))MatDestroy_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_VIEW,(void(*)(void))MatView_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DIAGONAL_SET,(void(*)(void))MatDiagonalSet_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_SHIFT,(void(*)(void))MatShift_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_EQUAL,(void(*)(void))MatEqual_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_SCALE,(void(*)(void))MatScale_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_TRANSPOSE,(void(*)(void))MatTranspose_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_CREATE_SUBMATRICES,(void(*)(void))MatCreateSubMatrices_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_NORM,(void(*)(void))MatNorm_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_GET_ROW_MAX,(void(*)(void))MatDuplicate_SMF);CHKERRQ(ierr);

  ierr = PetscLogObjectParent((PetscObject)mat,(PetscObject)(*J));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSMFResetRowColumn(Mat mat,IS Rows,IS Cols){
  MatSubMatFreeCtx ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx->Rows);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx->Cols);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Rows);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)Cols);CHKERRQ(ierr);
  ctx->Cols=Cols;
  ctx->Rows=Rows;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SMF(Mat mat,Vec a,Vec y)
{
  MatSubMatFreeCtx ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = VecCopy(a,ctx->VR);CHKERRQ(ierr);
  ierr = VecISSet(ctx->VR,ctx->Cols,0.0);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,ctx->VR,y);CHKERRQ(ierr);
  ierr = VecISSet(y,ctx->Rows,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SMF(Mat mat,Vec a,Vec y)
{
  MatSubMatFreeCtx ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = VecCopy(a,ctx->VC);CHKERRQ(ierr);
  ierr = VecISSet(ctx->VC,ctx->Rows,0.0);CHKERRQ(ierr);
  ierr = MatMultTranspose(ctx->A,ctx->VC,y);CHKERRQ(ierr);
  ierr = VecISSet(y,ctx->Cols,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalSet_SMF(Mat M, Vec D,InsertMode is)
{
  MatSubMatFreeCtx ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatDiagonalSet(ctx->A,D,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SMF(Mat mat)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx->Rows);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx->Cols);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->VC);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SMF(Mat mat,PetscViewer viewer)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatView(ctx->A,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SMF(Mat Y, PetscReal a)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(Y,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatShift(ctx->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SMF(Mat mat,MatDuplicateOption op,Mat *M)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatCreateSubMatrixFree(ctx->A,ctx->Rows,ctx->Cols,M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatEqual_SMF(Mat A,Mat B,PetscBool *flg)
{
  PetscErrorCode    ierr;
  MatSubMatFreeCtx  ctx1,ctx2;
  PetscBool         flg1,flg2,flg3;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&ctx1);CHKERRQ(ierr);
  ierr = MatShellGetContext(B,(void **)&ctx2);CHKERRQ(ierr);
  ierr = ISEqual(ctx1->Rows,ctx2->Rows,&flg2);CHKERRQ(ierr);
  ierr = ISEqual(ctx1->Cols,ctx2->Cols,&flg3);CHKERRQ(ierr);
  if (flg2==PETSC_FALSE || flg3==PETSC_FALSE){
    *flg=PETSC_FALSE;
  } else {
    ierr = MatEqual(ctx1->A,ctx2->A,&flg1);CHKERRQ(ierr);
    if (flg1==PETSC_FALSE){ *flg=PETSC_FALSE;}
    else { *flg=PETSC_TRUE;}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_SMF(Mat mat, PetscReal a)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatScale(ctx->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_SMF(Mat mat,Mat *B)
{
  PetscFunctionBegin;
  PetscFunctionReturn(1);
}

PetscErrorCode MatGetDiagonal_SMF(Mat mat,Vec v)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->A,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRowMax_SMF(Mat M, Vec D)
{
  MatSubMatFreeCtx ctx;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatGetRowMax(ctx->A,D,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_SMF(Mat A,PetscInt n, IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscCalloc1(n+1,B);CHKERRQ(ierr);
  }

  for (i=0; i<n; i++) {
    ierr = MatCreateSubMatrix_SMF(A,irow[i],icol[i],scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrix_SMF(Mat mat,IS isrow,IS iscol,MatReuse cll,
                        Mat *newmat)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (newmat){
    ierr = MatDestroy(&*newmat);CHKERRQ(ierr);
  }
  ierr = MatCreateSubMatrixFree(ctx->A,isrow,iscol, newmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_SMF(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt **cols,const PetscScalar **vals)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatGetRow(ctx->A,row,ncols,cols,vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_SMF(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt **cols,const PetscScalar **vals)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatRestoreRow(ctx->A,row,ncols,cols,vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetColumnVector_SMF(Mat mat,Vec Y, PetscInt col)
{
  PetscErrorCode   ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatGetColumnVector(ctx->A,Y,col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatNorm_SMF(Mat mat,NormType type,PetscReal *norm)
{
  PetscErrorCode    ierr;
  MatSubMatFreeCtx  ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (type == NORM_FROBENIUS) {
    *norm = 1.0;
  } else if (type == NORM_1 || type == NORM_INFINITY) {
    *norm = 1.0;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No two norm");
  PetscFunctionReturn(0);
}

