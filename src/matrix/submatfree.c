#include "submatfree.h"

PetscErrorCode ISCreateComplement(IS, Vec, IS *);
PetscErrorCode VecISSetToConstant(IS, PetscScalar, Vec);


#undef __FUNCT__  
#define __FUNCT__ "MatCreateSubMatrixFree"
/*@C
  MatCreateSubMatrixFree - Creates a reduced matrix by masking a
  full matrix.

   Collective on matrix

   Input Parameters:
+  mat - matrix of arbitrary type
.  RowMask - the rows that will be zero
-  ColMask - the columns that will be zero

   Output Parameters:
.  J - New matrix

   Notes: 
   The user provides the input data and is responsible for destroying
   this data after matrix J has been destroyed.  
 
   Level: developer

.seealso: MatCreate()
@*/
PetscErrorCode MatCreateSubMatrixFree(Mat mat,IS RowMask, IS ColMask, Mat *J)
{
  MPI_Comm     comm=((PetscObject)mat)->comm;
  MatSubMatFreeCtx ctx;
  PetscErrorCode ierr;
  PetscInt mloc,nloc,m,n;
  PetscFunctionBegin;

  ierr = PetscNew(_p_MatSubMatFreeCtx,&ctx);CHKERRQ(ierr);

  ctx->A=mat;
  //  ctx->Row=Row;
  //  ctx->Col=Col;

  ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&mloc,&nloc);CHKERRQ(ierr);

  ierr = VecCreateMPI(comm,nloc,n,&ctx->VC);CHKERRQ(ierr);
  //  ierr = ISCreateComplement(Col, ctx->VC, &ctx->ColComplement);CHKERRQ(ierr);
  //  ctx->RowComplement=ctx->ColComplement;
  ctx->VR=ctx->VC;
  ierr =  PetscObjectReference((PetscObject)mat);CHKERRQ(ierr);

  ierr =ISCreateComplement(RowMask,ctx->VC,&ctx->RowComplement);CHKERRQ(ierr);
  ierr =ISCreateComplement(ColMask,ctx->VC,&ctx->ColComplement);CHKERRQ(ierr);
  /*
  ierr =  PetscObjectReference((PetscObject)ctx->RowComplement);CHKERRQ(ierr);
  ierr =  PetscObjectReference((PetscObject)ctx->ColComplement);CHKERRQ(ierr);
  */
  ierr = MatCreateShell(comm,mloc,nloc,m,n,ctx,J);CHKERRQ(ierr);

  //  ierr = MatShellSetOperation(*J,MATOP_GET_ROW,(void(*)())MatGetRow_SMF);CHKERRQ(ierr);
  //  ierr = MatShellSetOperation(*J,MATOP_RESTORE_ROW,(void(*)())MatRestoreRow_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT,(void(*)())MatMult_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DESTROY,(void(*)())MatDestroy_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_VIEW,(void(*)())MatView_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_MULT_TRANSPOSE,(void(*)())MatMultTranspose_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DIAGONAL_SET,(void(*)())MatDiagonalSet_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_SHIFT,(void(*)())MatShift_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_EQUAL,(void(*)())MatEqual_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_SCALE,(void(*)())MatScale_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_TRANSPOSE,(void(*)())MatTranspose_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_GET_SUBMATRICES,(void(*)())MatGetSubMatrices_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_NORM,(void(*)())MatNorm_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_DUPLICATE,(void(*)())MatDuplicate_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_GET_SUBMATRIX,(void(*)())MatGetSubMatrix_SMF);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*J,MATOP_GET_ROW_MAX,(void(*)())MatDuplicate_SMF);CHKERRQ(ierr);

  ierr = PetscLogObjectParent(mat,*J); CHKERRQ(ierr);

  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "MatSMFResetRowColumn"
PetscErrorCode MatSMFResetRowColumn(Mat mat,IS RowMask,IS ColMask){
  MatSubMatFreeCtx ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)RowMask);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)ColMask);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx->RowComplement);CHKERRQ(ierr);
  ierr = ISDestroy(&ctx->ColComplement);CHKERRQ(ierr);
  ctx->ColComplement=ColMask;
  ctx->RowComplement=RowMask;
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SMF"
PetscErrorCode MatMult_SMF(Mat mat,Vec a,Vec y)
{
  MatSubMatFreeCtx ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = VecCopy(a,ctx->VR);CHKERRQ(ierr);
  ierr = VecISSetToConstant(ctx->ColComplement,0.0,ctx->VR);CHKERRQ(ierr);
  ierr = MatMult(ctx->A,ctx->VR,y);CHKERRQ(ierr);
  ierr = VecISSetToConstant(ctx->RowComplement,0.0,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SMF"
PetscErrorCode MatMultTranspose_SMF(Mat mat,Vec a,Vec y)
{
  MatSubMatFreeCtx ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = VecCopy(a,ctx->VC);CHKERRQ(ierr);
  ierr = VecISSetToConstant(ctx->RowComplement,0.0,ctx->VC);CHKERRQ(ierr);
  ierr = MatMultTranspose(ctx->A,ctx->VC,y);CHKERRQ(ierr);
  ierr = VecISSetToConstant(ctx->ColComplement,0.0,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalSet_SMF"
PetscErrorCode MatDiagonalSet_SMF(Mat M, Vec D,InsertMode is)
{
  MatSubMatFreeCtx ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatDiagonalSet(ctx->A,D,is);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SMF"
PetscErrorCode MatDestroy_SMF(Mat mat)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr =MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  //  ierr =ISDestroy(&ctx->Row);CHKERRQ(ierr);
  //  ierr =ISDestroy(&ctx->Col);CHKERRQ(ierr);
  if (ctx->A) {
    ierr =MatDestroy(&ctx->A);CHKERRQ(ierr);
  }
  if (ctx->RowComplement) {
    ierr =ISDestroy(&ctx->RowComplement);CHKERRQ(ierr);
  }
  if (ctx->ColComplement) {
    ierr =ISDestroy(&ctx->ColComplement);CHKERRQ(ierr);
  }
  if (ctx->VC) {
    ierr =VecDestroy(&ctx->VC);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "MatView_SMF"
PetscErrorCode MatView_SMF(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatView(ctx->A,viewer);CHKERRQ(ierr);
  //  ierr = ISView(ctx->Row,viewer);CHKERRQ(ierr);
  //  ierr = ISView(ctx->Col,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatShift_SMF"
PetscErrorCode MatShift_SMF(Mat Y, PetscScalar a)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(Y,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatShift(ctx->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_SMF"
PetscErrorCode MatDuplicate_SMF(Mat mat,MatDuplicateOption op,Mat *M)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatCreateSubMatrixFree(ctx->A,ctx->RowComplement,ctx->ColComplement,M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_SMF"
PetscErrorCode MatEqual_SMF(Mat A,Mat B,PetscBool *flg)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx  ctx1,ctx2;
  PetscBool flg1,flg2,flg3;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void **)&ctx1);CHKERRQ(ierr);
  ierr = MatShellGetContext(B,(void **)&ctx2);CHKERRQ(ierr);
  ierr = ISEqual(ctx1->RowComplement,ctx2->RowComplement,&flg2);CHKERRQ(ierr);
  ierr = ISEqual(ctx1->ColComplement,ctx2->ColComplement,&flg3);CHKERRQ(ierr);
  if (flg2==PETSC_FALSE || flg3==PETSC_FALSE){
    *flg=PETSC_FALSE;
  } else {
    ierr = MatEqual(ctx1->A,ctx2->A,&flg1);CHKERRQ(ierr);
    if (flg1==PETSC_FALSE){ *flg=PETSC_FALSE;} 
    else { *flg=PETSC_TRUE;} 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_SMF"
PetscErrorCode MatScale_SMF(Mat mat, PetscScalar a)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatScale(ctx->A,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_SMF"
PetscErrorCode MatTranspose_SMF(Mat mat,Mat *B)
{
  PetscFunctionBegin;
  PetscFunctionReturn(1);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_SMF"
PetscErrorCode MatGetDiagonal_SMF(Mat mat,Vec v)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatGetDiagonal(ctx->A,v);CHKERRQ(ierr);
  //  ierr = VecISSetToConstant(ctx->RowComplement,0.0,v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalSet_SMF"
PetscErrorCode MatGetRowMax_SMF(Mat M, Vec D)
{
  MatSubMatFreeCtx ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatGetRowMax(ctx->A,D,PETSC_NULL);
  //  ierr = VecISSetToConstant(ctx->RowComplement,0.0,D);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 


#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_SMF"
PetscErrorCode MatGetSubMatrices_SMF(Mat A,PetscInt n, IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  PetscErrorCode ierr;
  PetscInt i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscMalloc( (n+1)*sizeof(Mat),B );CHKERRQ(ierr);
  }

  for ( i=0; i<n; i++ ) {
    ierr = MatGetSubMatrix_SMF(A,irow[i],icol[i],scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_SMF"
PetscErrorCode MatGetSubMatrix_SMF(Mat mat,IS isrow,IS iscol,MatReuse cll,
			Mat *newmat)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (newmat){
    ierr =MatDestroy(&*newmat);CHKERRQ(ierr);
  }
  ierr = MatCreateSubMatrixFree(ctx->A,isrow,iscol, newmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_SMF"
PetscErrorCode MatGetRow_SMF(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt **cols,const PetscScalar **vals)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatGetRow(ctx->A,row,ncols,cols,vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_SMF"
PetscErrorCode MatRestoreRow_SMF(Mat mat,PetscInt row,PetscInt *ncols,const PetscInt **cols,const PetscScalar **vals)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatRestoreRow(ctx->A,row,ncols,cols,vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnVector_SMF"
PetscErrorCode MatGetColumnVector_SMF(Mat mat,Vec Y, PetscInt col)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx  ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  ierr = MatGetColumnVector(ctx->A,Y,col);CHKERRQ(ierr);
  //  ierr = VecISSetToConstant(ctx->RowComplement,0.0,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatConvert_SMF"
PetscErrorCode MatConvert_SMF(Mat mat,MatType newtype,Mat *NewMat)
{
  PetscErrorCode ierr;
  PetscMPIInt size;
  MatSubMatFreeCtx  ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  MPI_Comm_size(((PetscObject)mat)->comm,&size);
  PetscFunctionReturn(1);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_SMF"
PetscErrorCode MatNorm_SMF(Mat mat,NormType type,PetscReal *norm)
{
  PetscErrorCode ierr;
  MatSubMatFreeCtx  ctx;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);

  if (type == NORM_FROBENIUS) {
    *norm = 1.0;
  } else if (type == NORM_1 || type == NORM_INFINITY) {
    *norm = 1.0;
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No two norm");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecISSetToConstant"
/*@C
   VecISSetToConstant - Sets the elements of a vector, specified by an index set, to a constant

   Input Parameter:
+  S -  a PETSc IS
.  c - the constant
-  V - a Vec

.seealso VecSet()

   Level: advanced
@*/
PetscErrorCode VecISSetToConstant(IS S, PetscScalar c, Vec V){
  PetscErrorCode ierr;
  PetscInt nloc,low,high,i;
  const PetscInt *s;
  PetscScalar *v;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,VEC_CLASSID,3); 
  PetscValidHeaderSpecific(S,IS_CLASSID,1); 
  PetscValidType(V,3);
  PetscCheckSameComm(V,3,S,1);

  ierr = VecGetOwnershipRange(V, &low, &high); CHKERRQ(ierr);
  ierr = ISGetLocalSize(S,&nloc);CHKERRQ(ierr);

  ierr = ISGetIndices(S, &s); CHKERRQ(ierr);
  ierr = VecGetArray(V,&v); CHKERRQ(ierr);
  for (i=0; i<nloc; i++){
    v[s[i]-low] = c;
  }
  
  ierr = ISRestoreIndices(S, &s); CHKERRQ(ierr);
  ierr = VecRestoreArray(V,&v); CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

#undef __FUNCT__  
#define __FUNCT__ "ISCreateComplement"
/*@C
   ISCreateComplement - Creates the complement of the the index set

   Input Parameter:
+  S -  a PETSc IS
-  V - the reference vector space

   Output Parameter:
.  T -  the complement of S


.seealso ISCreateGeneral()

   Level: advanced
@*/
PetscErrorCode ISCreateComplement(IS S, Vec V, IS *T){
  PetscErrorCode ierr;
  PetscInt i,nis,nloc,high,low,n=0;
  const PetscInt *s;
  PetscInt *tt,*ss;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(S,IS_CLASSID,1); 
  PetscValidHeaderSpecific(V,VEC_CLASSID,2); 

  ierr = VecGetOwnershipRange(V,&low,&high); CHKERRQ(ierr);
  ierr = VecGetLocalSize(V,&nloc); CHKERRQ(ierr);
  ierr = ISGetLocalSize(S,&nis); CHKERRQ(ierr);
  ierr = ISGetIndices(S, &s); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscInt),&tt ); CHKERRQ(ierr);
  ierr = PetscMalloc( nloc*sizeof(PetscInt),&ss ); CHKERRQ(ierr);

  for (i=low; i<high; i++){ tt[i-low]=i; }

  for (i=0; i<nis; i++){ tt[s[i]-low] = -2; }
  
  for (i=0; i<nloc; i++){
    if (tt[i]>-1){ ss[n]=tt[i]; n++; }
  }

  ierr = ISRestoreIndices(S, &s); CHKERRQ(ierr);
  
  ierr = PetscObjectGetComm((PetscObject)S,&comm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n,ss,PETSC_COPY_VALUES,T);CHKERRQ(ierr);
  
  if (tt) {
    ierr = PetscFree(tt); CHKERRQ(ierr);
  }
  if (ss) {
    ierr = PetscFree(ss); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
