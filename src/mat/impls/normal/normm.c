
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat         A;
  Vec         w,left,right,leftwork,rightwork;
  PetscScalar scale;
} Mat_Normal;

PetscErrorCode MatScale_Normal(Mat inA,PetscScalar scale)
{
  Mat_Normal *a = (Mat_Normal*)inA->data;

  PetscFunctionBegin;
  a->scale *= scale;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_Normal(Mat inA,Vec left,Vec right)
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

PetscErrorCode MatIncreaseOverlap_Normal(Mat A,PetscInt is_max,IS is[],PetscInt ov)
{
  Mat_Normal     *a = (Mat_Normal*)A->data;
  Mat            pattern;

  PetscFunctionBegin;
  PetscCheckFalse(ov < 0,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");
  PetscCall(MatProductCreate(a->A,a->A,NULL,&pattern));
  PetscCall(MatProductSetType(pattern,MATPRODUCT_AtB));
  PetscCall(MatProductSetFromOptions(pattern));
  PetscCall(MatProductSymbolic(pattern));
  PetscCall(MatIncreaseOverlap(pattern,is_max,is,ov));
  PetscCall(MatDestroy(&pattern));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_Normal(Mat mat,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  Mat_Normal     *a = (Mat_Normal*)mat->data;
  Mat            B = a->A, *suba;
  IS             *row;
  PetscInt       M;

  PetscFunctionBegin;
  PetscCheckFalse(a->left || a->right || irow != icol,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Not implemented");
  if (scall != MAT_REUSE_MATRIX) {
    PetscCall(PetscCalloc1(n,submat));
  }
  PetscCall(MatGetSize(B,&M,NULL));
  PetscCall(PetscMalloc1(n,&row));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,M,0,1,&row[0]));
  PetscCall(ISSetIdentity(row[0]));
  for (M = 1; M < n; ++M) row[M] = row[0];
  PetscCall(MatCreateSubMatrices(B,n,row,icol,MAT_INITIAL_MATRIX,&suba));
  for (M = 0; M < n; ++M) {
    PetscCall(MatCreateNormal(suba[M],*submat+M));
    ((Mat_Normal*)(*submat)[M]->data)->scale = a->scale;
  }
  PetscCall(ISDestroy(&row[0]));
  PetscCall(PetscFree(row));
  PetscCall(MatDestroySubMatrices(n,&suba));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPermute_Normal(Mat A,IS rowp,IS colp,Mat *B)
{
  Mat_Normal     *a = (Mat_Normal*)A->data;
  Mat            C,Aa = a->A;
  IS             row;

  PetscFunctionBegin;
  PetscCheckFalse(rowp != colp,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Row permutation and column permutation must be the same");
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)Aa),Aa->rmap->n,Aa->rmap->rstart,1,&row));
  PetscCall(ISSetIdentity(row));
  PetscCall(MatPermute(Aa,row,colp,&C));
  PetscCall(ISDestroy(&row));
  PetscCall(MatCreateNormal(C,B));
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_Normal(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_Normal     *a = (Mat_Normal*)A->data;
  Mat            C;

  PetscFunctionBegin;
  PetscCheckFalse(a->left || a->right,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented");
  PetscCall(MatDuplicate(a->A,op,&C));
  PetscCall(MatCreateNormal(C,B));
  PetscCall(MatDestroy(&C));
  if (op == MAT_COPY_VALUES) ((Mat_Normal*)(*B)->data)->scale = a->scale;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_Normal(Mat A,Mat B,MatStructure str)
{
  Mat_Normal     *a = (Mat_Normal*)A->data,*b = (Mat_Normal*)B->data;

  PetscFunctionBegin;
  PetscCheckFalse(a->left || a->right,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented");
  PetscCall(MatCopy(a->A,b->A,str));
  b->scale = a->scale;
  PetscCall(VecDestroy(&b->left));
  PetscCall(VecDestroy(&b->right));
  PetscCall(VecDestroy(&b->leftwork));
  PetscCall(VecDestroy(&b->rightwork));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_Normal(Mat N,Vec x,Vec y)
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
  PetscCall(MatMultTranspose(Na->A,Na->w,y));
  if (Na->left) {
    PetscCall(VecPointwiseMult(y,Na->left,y));
  }
  PetscCall(VecScale(y,Na->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Normal(Mat N,Vec v1,Vec v2,Vec v3)
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
    PetscCall(MatMultTranspose(Na->A,Na->w,v3));
    PetscCall(VecPointwiseMult(v3,Na->left,v3));
    PetscCall(VecAXPY(v3,1.0,v2));
  } else {
    PetscCall(MatMultTransposeAdd(Na->A,Na->w,v2,v3));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Normal(Mat N,Vec x,Vec y)
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
  PetscCall(MatMultTranspose(Na->A,Na->w,y));
  if (Na->right) {
    PetscCall(VecPointwiseMult(y,Na->right,y));
  }
  PetscCall(VecScale(y,Na->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Normal(Mat N,Vec v1,Vec v2,Vec v3)
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
    PetscCall(MatMultTranspose(Na->A,Na->w,v3));
    PetscCall(VecPointwiseMult(v3,Na->right,v3));
    PetscCall(VecAXPY(v3,1.0,v2));
  } else {
    PetscCall(MatMultTransposeAdd(Na->A,Na->w,v2,v3));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Normal(Mat N)
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
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatNormalGetMat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatConvert_normal_seqaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatConvert_normal_mpiaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_normal_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_normal_mpidense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_normal_dense_C",NULL));
  PetscFunctionReturn(0);
}

/*
      Slow, nonscalable version
*/
PetscErrorCode MatGetDiagonal_Normal(Mat N,Vec v)
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
      work[cols[j]] += mvalues[j]*mvalues[j];
    }
    PetscCall(MatRestoreRow(A,i,&nnz,&cols,&mvalues));
  }
  PetscCallMPI(MPIU_Allreduce(work,diag,A->cmap->N,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)N)));
  rstart = N->cmap->rstart;
  rend   = N->cmap->rend;
  PetscCall(VecGetArray(v,&values));
  PetscCall(PetscArraycpy(values,diag+rstart,rend-rstart));
  PetscCall(VecRestoreArray(v,&values));
  PetscCall(PetscFree2(diag,work));
  PetscCall(VecScale(v,Na->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNormalGetMat_Normal(Mat A,Mat *M)
{
  Mat_Normal *Aa = (Mat_Normal*)A->data;

  PetscFunctionBegin;
  *M = Aa->A;
  PetscFunctionReturn(0);
}

/*@
      MatNormalGetMat - Gets the Mat object stored inside a MATNORMAL

   Logically collective on Mat

   Input Parameter:
.   A  - the MATNORMAL matrix

   Output Parameter:
.   M - the matrix object stored inside A

   Level: intermediate

.seealso: MatCreateNormal()

@*/
PetscErrorCode MatNormalGetMat(Mat A,Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(M,2);
  PetscCall(PetscUseMethod(A,"MatNormalGetMat_C",(Mat,Mat*),(A,M)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_Normal_AIJ(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_Normal     *Aa = (Mat_Normal*)A->data;
  Mat            B;
  PetscInt       m,n,M,N;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetLocalSize(A,&m,&n));
  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
    PetscCall(MatProductReplaceMats(Aa->A,Aa->A,NULL,B));
  } else {
    PetscCall(MatProductCreate(Aa->A,Aa->A,NULL,&B));
    PetscCall(MatProductSetType(B,MATPRODUCT_AtB));
    PetscCall(MatProductSetFromOptions(B));
    PetscCall(MatProductSymbolic(B));
    PetscCall(MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE));
  }
  PetscCall(MatProductNumeric(B));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&B));
  } else if (reuse == MAT_INITIAL_MATRIX) *newmat = B;
  PetscCall(MatConvert(*newmat,MATAIJ,MAT_INPLACE_MATRIX,newmat));
  PetscFunctionReturn(0);
}

typedef struct {
  Mat work[2];
} Normal_Dense;

PetscErrorCode MatProductNumeric_Normal_Dense(Mat C)
{
  Mat            A,B;
  Normal_Dense   *contents;
  Mat_Normal     *a;
  PetscScalar    *array;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  A = C->product->A;
  a = (Mat_Normal*)A->data;
  B = C->product->B;
  contents = (Normal_Dense*)C->product->data;
  PetscCheck(contents,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  if (a->right) {
    PetscCall(MatCopy(B,C,SAME_NONZERO_PATTERN));
    PetscCall(MatDiagonalScale(C,a->right,NULL));
  }
  PetscCall(MatProductNumeric(contents->work[0]));
  PetscCall(MatDenseGetArrayWrite(C,&array));
  PetscCall(MatDensePlaceArray(contents->work[1],array));
  PetscCall(MatProductNumeric(contents->work[1]));
  PetscCall(MatDenseRestoreArrayWrite(C,&array));
  PetscCall(MatDenseResetArray(contents->work[1]));
  PetscCall(MatSetOption(C,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(C,a->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNormal_DenseDestroy(void *ctx)
{
  Normal_Dense   *contents = (Normal_Dense*)ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(contents->work));
  PetscCall(MatDestroy(contents->work+1));
  PetscCall(PetscFree(contents));
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_Normal_Dense(Mat C)
{
  Mat            A,B;
  Normal_Dense   *contents = NULL;
  Mat_Normal     *a;
  PetscScalar    *array;
  PetscInt       n,N,m,M;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheck(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A = C->product->A;
  a = (Mat_Normal*)A->data;
  PetscCheck(!a->left,PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"Not implemented");
  B = C->product->B;
  PetscCall(MatGetLocalSize(C,&m,&n));
  PetscCall(MatGetSize(C,&M,&N));
  if (m == PETSC_DECIDE || n == PETSC_DECIDE || M == PETSC_DECIDE || N == PETSC_DECIDE) {
    PetscCall(MatGetLocalSize(B,NULL,&n));
    PetscCall(MatGetSize(B,NULL,&N));
    PetscCall(MatGetLocalSize(A,&m,NULL));
    PetscCall(MatGetSize(A,&M,NULL));
    PetscCall(MatSetSizes(C,m,n,M,N));
  }
  PetscCall(MatSetType(C,((PetscObject)B)->type_name));
  PetscCall(MatSetUp(C));
  PetscCall(PetscNew(&contents));
  C->product->data = contents;
  C->product->destroy = MatNormal_DenseDestroy;
  if (a->right) {
    PetscCall(MatProductCreate(a->A,C,NULL,contents->work));
  } else {
    PetscCall(MatProductCreate(a->A,B,NULL,contents->work));
  }
  PetscCall(MatProductSetType(contents->work[0],MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(contents->work[0]));
  PetscCall(MatProductSymbolic(contents->work[0]));
  PetscCall(MatProductCreate(a->A,contents->work[0],NULL,contents->work+1));
  PetscCall(MatProductSetType(contents->work[1],MATPRODUCT_AtB));
  PetscCall(MatProductSetFromOptions(contents->work[1]));
  PetscCall(MatProductSymbolic(contents->work[1]));
  PetscCall(MatDenseGetArrayWrite(C,&array));
  PetscCall(MatSeqDenseSetPreallocation(contents->work[1],array));
  PetscCall(MatMPIDenseSetPreallocation(contents->work[1],array));
  PetscCall(MatDenseRestoreArrayWrite(C,&array));
  C->ops->productnumeric = MatProductNumeric_Normal_Dense;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_Normal_Dense_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_Normal_Dense;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_Normal_Dense(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_AB) {
    PetscCall(MatProductSetFromOptions_Normal_Dense_AB(C));
  }
  PetscFunctionReturn(0);
}

/*@
      MatCreateNormal - Creates a new matrix object that behaves like A'*A.

   Collective on Mat

   Input Parameter:
.   A  - the (possibly rectangular) matrix

   Output Parameter:
.   N - the matrix that represents A'*A

   Level: intermediate

   Notes:
    The product A'*A is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product by first multiplying by
          A and then A'
@*/
PetscErrorCode  MatCreateNormal(Mat A,Mat *N)
{
  PetscInt       n,nn;
  Mat_Normal     *Na;
  VecType        vtype;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,NULL,&nn));
  PetscCall(MatGetLocalSize(A,NULL,&n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),N));
  PetscCall(MatSetSizes(*N,n,n,nn,nn));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*N,MATNORMAL));
  PetscCall(PetscLayoutReference(A->cmap,&(*N)->rmap));
  PetscCall(PetscLayoutReference(A->cmap,&(*N)->cmap));

  PetscCall(PetscNewLog(*N,&Na));
  (*N)->data = (void*) Na;
  PetscCall(PetscObjectReference((PetscObject)A));
  Na->A      = A;
  Na->scale  = 1.0;

  PetscCall(MatCreateVecs(A,NULL,&Na->w));

  (*N)->ops->destroy           = MatDestroy_Normal;
  (*N)->ops->mult              = MatMult_Normal;
  (*N)->ops->multtranspose     = MatMultTranspose_Normal;
  (*N)->ops->multtransposeadd  = MatMultTransposeAdd_Normal;
  (*N)->ops->multadd           = MatMultAdd_Normal;
  (*N)->ops->getdiagonal       = MatGetDiagonal_Normal;
  (*N)->ops->scale             = MatScale_Normal;
  (*N)->ops->diagonalscale     = MatDiagonalScale_Normal;
  (*N)->ops->increaseoverlap   = MatIncreaseOverlap_Normal;
  (*N)->ops->createsubmatrices = MatCreateSubMatrices_Normal;
  (*N)->ops->permute           = MatPermute_Normal;
  (*N)->ops->duplicate         = MatDuplicate_Normal;
  (*N)->ops->copy              = MatCopy_Normal;
  (*N)->assembled              = PETSC_TRUE;
  (*N)->preallocated           = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatNormalGetMat_C",MatNormalGetMat_Normal));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatConvert_normal_seqaij_C",MatConvert_Normal_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatConvert_normal_mpiaij_C",MatConvert_Normal_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_normal_seqdense_C",MatProductSetFromOptions_Normal_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_normal_mpidense_C",MatProductSetFromOptions_Normal_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_normal_dense_C",MatProductSetFromOptions_Normal_Dense));
  PetscCall(MatSetOption(*N,MAT_SYMMETRIC,PETSC_TRUE));
  PetscCall(MatGetVecType(A,&vtype));
  PetscCall(MatSetVecType(*N,vtype));
#if defined(PETSC_HAVE_DEVICE)
  PetscCall(MatBindToCPU(*N,A->boundtocpu));
#endif
  PetscFunctionReturn(0);
}
