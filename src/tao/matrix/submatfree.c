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
  PetscInt         mloc,nloc,m,n;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&ctx));
  ctx->A  = mat;
  CHKERRQ(MatGetSize(mat,&m,&n));
  CHKERRQ(MatGetLocalSize(mat,&mloc,&nloc));
  CHKERRQ(MatCreateVecs(mat,NULL,&ctx->VC));
  ctx->VR = ctx->VC;
  CHKERRQ(PetscObjectReference((PetscObject)mat));

  ctx->Rows = Rows;
  ctx->Cols = Cols;
  CHKERRQ(PetscObjectReference((PetscObject)Rows));
  CHKERRQ(PetscObjectReference((PetscObject)Cols));
  CHKERRQ(MatCreateShell(comm,mloc,nloc,m,n,ctx,J));
  CHKERRQ(MatShellSetManageScalingShifts(*J));
  CHKERRQ(MatShellSetOperation(*J,MATOP_MULT,(void(*)(void))MatMult_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_DESTROY,(void(*)(void))MatDestroy_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_VIEW,(void(*)(void))MatView_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_DIAGONAL_SET,(void(*)(void))MatDiagonalSet_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_SHIFT,(void(*)(void))MatShift_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_EQUAL,(void(*)(void))MatEqual_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_SCALE,(void(*)(void))MatScale_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_TRANSPOSE,(void(*)(void))MatTranspose_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_CREATE_SUBMATRICES,(void(*)(void))MatCreateSubMatrices_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_NORM,(void(*)(void))MatNorm_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_CREATE_SUBMATRIX,(void(*)(void))MatCreateSubMatrix_SMF));
  CHKERRQ(MatShellSetOperation(*J,MATOP_GET_ROW_MAX,(void(*)(void))MatDuplicate_SMF));

  CHKERRQ(PetscLogObjectParent((PetscObject)mat,(PetscObject)(*J)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSMFResetRowColumn(Mat mat,IS Rows,IS Cols)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(ISDestroy(&ctx->Rows));
  CHKERRQ(ISDestroy(&ctx->Cols));
  CHKERRQ(PetscObjectReference((PetscObject)Rows));
  CHKERRQ(PetscObjectReference((PetscObject)Cols));
  ctx->Cols=Cols;
  ctx->Rows=Rows;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SMF(Mat mat,Vec a,Vec y)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(VecCopy(a,ctx->VR));
  CHKERRQ(VecISSet(ctx->VR,ctx->Cols,0.0));
  CHKERRQ(MatMult(ctx->A,ctx->VR,y));
  CHKERRQ(VecISSet(y,ctx->Rows,0.0));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SMF(Mat mat,Vec a,Vec y)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(VecCopy(a,ctx->VC));
  CHKERRQ(VecISSet(ctx->VC,ctx->Rows,0.0));
  CHKERRQ(MatMultTranspose(ctx->A,ctx->VC,y));
  CHKERRQ(VecISSet(y,ctx->Cols,0.0));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalSet_SMF(Mat M, Vec D,InsertMode is)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(MatDiagonalSet(ctx->A,D,is));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SMF(Mat mat)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatDestroy(&ctx->A));
  CHKERRQ(ISDestroy(&ctx->Rows));
  CHKERRQ(ISDestroy(&ctx->Cols));
  CHKERRQ(VecDestroy(&ctx->VC));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SMF(Mat mat,PetscViewer viewer)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatView(ctx->A,viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SMF(Mat Y, PetscReal a)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(Y,&ctx));
  CHKERRQ(MatShift(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SMF(Mat mat,MatDuplicateOption op,Mat *M)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatCreateSubMatrixFree(ctx->A,ctx->Rows,ctx->Cols,M));
  PetscFunctionReturn(0);
}

PetscErrorCode MatEqual_SMF(Mat A,Mat B,PetscBool *flg)
{
  MatSubMatFreeCtx  ctx1,ctx2;
  PetscBool         flg1,flg2,flg3;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx1));
  CHKERRQ(MatShellGetContext(B,&ctx2));
  CHKERRQ(ISEqual(ctx1->Rows,ctx2->Rows,&flg2));
  CHKERRQ(ISEqual(ctx1->Cols,ctx2->Cols,&flg3));
  if (flg2==PETSC_FALSE || flg3==PETSC_FALSE) {
    *flg=PETSC_FALSE;
  } else {
    CHKERRQ(MatEqual(ctx1->A,ctx2->A,&flg1));
    if (flg1==PETSC_FALSE) { *flg=PETSC_FALSE;}
    else { *flg=PETSC_TRUE;}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_SMF(Mat mat, PetscReal a)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatScale(ctx->A,a));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_SMF(Mat mat,Mat *B)
{
  PetscFunctionBegin;
  PetscFunctionReturn(1);
}

PetscErrorCode MatGetDiagonal_SMF(Mat mat,Vec v)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatGetDiagonal(ctx->A,v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRowMax_SMF(Mat M, Vec D)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(M,&ctx));
  CHKERRQ(MatGetRowMax(ctx->A,D,NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_SMF(Mat A,PetscInt n, IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    CHKERRQ(PetscCalloc1(n+1,B));
  }

  for (i=0; i<n; i++) {
    CHKERRQ(MatCreateSubMatrix_SMF(A,irow[i],icol[i],scall,&(*B)[i]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrix_SMF(Mat mat,IS isrow,IS iscol,MatReuse cll,
                        Mat *newmat)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  if (newmat) {
    CHKERRQ(MatDestroy(&*newmat));
  }
  CHKERRQ(MatCreateSubMatrixFree(ctx->A,isrow,iscol, newmat));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_SMF(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt **cols,const PetscScalar **vals)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatGetRow(ctx->A,row,ncols,cols,vals));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_SMF(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt **cols,const PetscScalar **vals)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatRestoreRow(ctx->A,row,ncols,cols,vals));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetColumnVector_SMF(Mat mat,Vec Y, PetscInt col)
{
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  CHKERRQ(MatGetColumnVector(ctx->A,Y,col));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNorm_SMF(Mat mat,NormType type,PetscReal *norm)
{
  MatSubMatFreeCtx  ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(mat,&ctx));
  if (type == NORM_FROBENIUS) {
    *norm = 1.0;
  } else if (type == NORM_1 || type == NORM_INFINITY) {
    *norm = 1.0;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No two norm");
  PetscFunctionReturn(0);
}
