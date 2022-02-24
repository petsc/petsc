
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

PetscErrorCode MatIncreaseOverlap_Normal(Mat A,PetscInt is_max,IS is[],PetscInt ov)
{
  Mat_Normal     *a = (Mat_Normal*)A->data;
  Mat            pattern;

  PetscFunctionBegin;
  PetscCheckFalse(ov < 0,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");
  CHKERRQ(MatProductCreate(a->A,a->A,NULL,&pattern));
  CHKERRQ(MatProductSetType(pattern,MATPRODUCT_AtB));
  CHKERRQ(MatProductSetFromOptions(pattern));
  CHKERRQ(MatProductSymbolic(pattern));
  CHKERRQ(MatIncreaseOverlap(pattern,is_max,is,ov));
  CHKERRQ(MatDestroy(&pattern));
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
    CHKERRQ(PetscCalloc1(n,submat));
  }
  CHKERRQ(MatGetSize(B,&M,NULL));
  CHKERRQ(PetscMalloc1(n,&row));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,M,0,1,&row[0]));
  CHKERRQ(ISSetIdentity(row[0]));
  for (M = 1; M < n; ++M) row[M] = row[0];
  CHKERRQ(MatCreateSubMatrices(B,n,row,icol,MAT_INITIAL_MATRIX,&suba));
  for (M = 0; M < n; ++M) {
    CHKERRQ(MatCreateNormal(suba[M],*submat+M));
    ((Mat_Normal*)(*submat)[M]->data)->scale = a->scale;
  }
  CHKERRQ(ISDestroy(&row[0]));
  CHKERRQ(PetscFree(row));
  CHKERRQ(MatDestroySubMatrices(n,&suba));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPermute_Normal(Mat A,IS rowp,IS colp,Mat *B)
{
  Mat_Normal     *a = (Mat_Normal*)A->data;
  Mat            C,Aa = a->A;
  IS             row;

  PetscFunctionBegin;
  PetscCheckFalse(rowp != colp,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Row permutation and column permutation must be the same");
  CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)Aa),Aa->rmap->n,Aa->rmap->rstart,1,&row));
  CHKERRQ(ISSetIdentity(row));
  CHKERRQ(MatPermute(Aa,row,colp,&C));
  CHKERRQ(ISDestroy(&row));
  CHKERRQ(MatCreateNormal(C,B));
  CHKERRQ(MatDestroy(&C));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_Normal(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_Normal     *a = (Mat_Normal*)A->data;
  Mat            C;

  PetscFunctionBegin;
  PetscCheckFalse(a->left || a->right,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented");
  CHKERRQ(MatDuplicate(a->A,op,&C));
  CHKERRQ(MatCreateNormal(C,B));
  CHKERRQ(MatDestroy(&C));
  if (op == MAT_COPY_VALUES) ((Mat_Normal*)(*B)->data)->scale = a->scale;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_Normal(Mat A,Mat B,MatStructure str)
{
  Mat_Normal     *a = (Mat_Normal*)A->data,*b = (Mat_Normal*)B->data;

  PetscFunctionBegin;
  PetscCheckFalse(a->left || a->right,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented");
  CHKERRQ(MatCopy(a->A,b->A,str));
  b->scale = a->scale;
  CHKERRQ(VecDestroy(&b->left));
  CHKERRQ(VecDestroy(&b->right));
  CHKERRQ(VecDestroy(&b->leftwork));
  CHKERRQ(VecDestroy(&b->rightwork));
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
      CHKERRQ(VecDuplicate(Na->right,&Na->rightwork));
    }
    CHKERRQ(VecPointwiseMult(Na->rightwork,Na->right,in));
    in   = Na->rightwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(MatMultTranspose(Na->A,Na->w,y));
  if (Na->left) {
    CHKERRQ(VecPointwiseMult(y,Na->left,y));
  }
  CHKERRQ(VecScale(y,Na->scale));
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
      CHKERRQ(VecDuplicate(Na->right,&Na->rightwork));
    }
    CHKERRQ(VecPointwiseMult(Na->rightwork,Na->right,in));
    in   = Na->rightwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(VecScale(Na->w,Na->scale));
  if (Na->left) {
    CHKERRQ(MatMultTranspose(Na->A,Na->w,v3));
    CHKERRQ(VecPointwiseMult(v3,Na->left,v3));
    CHKERRQ(VecAXPY(v3,1.0,v2));
  } else {
    CHKERRQ(MatMultTransposeAdd(Na->A,Na->w,v2,v3));
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
      CHKERRQ(VecDuplicate(Na->left,&Na->leftwork));
    }
    CHKERRQ(VecPointwiseMult(Na->leftwork,Na->left,in));
    in   = Na->leftwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(MatMultTranspose(Na->A,Na->w,y));
  if (Na->right) {
    CHKERRQ(VecPointwiseMult(y,Na->right,y));
  }
  CHKERRQ(VecScale(y,Na->scale));
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
      CHKERRQ(VecDuplicate(Na->left,&Na->leftwork));
    }
    CHKERRQ(VecPointwiseMult(Na->leftwork,Na->left,in));
    in   = Na->leftwork;
  }
  CHKERRQ(MatMult(Na->A,in,Na->w));
  CHKERRQ(VecScale(Na->w,Na->scale));
  if (Na->right) {
    CHKERRQ(MatMultTranspose(Na->A,Na->w,v3));
    CHKERRQ(VecPointwiseMult(v3,Na->right,v3));
    CHKERRQ(VecAXPY(v3,1.0,v2));
  } else {
    CHKERRQ(MatMultTransposeAdd(Na->A,Na->w,v2,v3));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Normal(Mat N)
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
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatNormalGetMat_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatConvert_normal_seqaij_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatConvert_normal_mpiaij_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_normal_seqdense_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_normal_mpidense_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_normal_dense_C",NULL));
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
  CHKERRQ(PetscMalloc2(A->cmap->N,&diag,A->cmap->N,&work));
  CHKERRQ(PetscArrayzero(work,A->cmap->N));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatGetRow(A,i,&nnz,&cols,&mvalues));
    for (j=0; j<nnz; j++) {
      work[cols[j]] += mvalues[j]*mvalues[j];
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
  CHKERRQ(PetscUseMethod(A,"MatNormalGetMat_C",(Mat,Mat*),(A,M)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_Normal_AIJ(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_Normal     *Aa = (Mat_Normal*)A->data;
  Mat            B;
  PetscInt       m,n,M,N;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
    CHKERRQ(MatProductReplaceMats(Aa->A,Aa->A,NULL,B));
  } else {
    CHKERRQ(MatProductCreate(Aa->A,Aa->A,NULL,&B));
    CHKERRQ(MatProductSetType(B,MATPRODUCT_AtB));
    CHKERRQ(MatProductSetFromOptions(B));
    CHKERRQ(MatProductSymbolic(B));
    CHKERRQ(MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE));
  }
  CHKERRQ(MatProductNumeric(B));
  if (reuse == MAT_INPLACE_MATRIX) {
    CHKERRQ(MatHeaderReplace(A,&B));
  } else if (reuse == MAT_INITIAL_MATRIX) *newmat = B;
  CHKERRQ(MatConvert(*newmat,MATAIJ,MAT_INPLACE_MATRIX,newmat));
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
  PetscCheckFalse(!contents,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  if (a->right) {
    CHKERRQ(MatCopy(B,C,SAME_NONZERO_PATTERN));
    CHKERRQ(MatDiagonalScale(C,a->right,NULL));
  }
  CHKERRQ(MatProductNumeric(contents->work[0]));
  CHKERRQ(MatDenseGetArrayWrite(C,&array));
  CHKERRQ(MatDensePlaceArray(contents->work[1],array));
  CHKERRQ(MatProductNumeric(contents->work[1]));
  CHKERRQ(MatDenseRestoreArrayWrite(C,&array));
  CHKERRQ(MatDenseResetArray(contents->work[1]));
  CHKERRQ(MatSetOption(C,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatScale(C,a->scale));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNormal_DenseDestroy(void *ctx)
{
  Normal_Dense   *contents = (Normal_Dense*)ctx;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(contents->work));
  CHKERRQ(MatDestroy(contents->work+1));
  CHKERRQ(PetscFree(contents));
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
  PetscCheckFalse(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A = C->product->A;
  a = (Mat_Normal*)A->data;
  PetscCheckFalse(a->left,PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"Not implemented");
  B = C->product->B;
  CHKERRQ(MatGetLocalSize(C,&m,&n));
  CHKERRQ(MatGetSize(C,&M,&N));
  if (m == PETSC_DECIDE || n == PETSC_DECIDE || M == PETSC_DECIDE || N == PETSC_DECIDE) {
    CHKERRQ(MatGetLocalSize(B,NULL,&n));
    CHKERRQ(MatGetSize(B,NULL,&N));
    CHKERRQ(MatGetLocalSize(A,&m,NULL));
    CHKERRQ(MatGetSize(A,&M,NULL));
    CHKERRQ(MatSetSizes(C,m,n,M,N));
  }
  CHKERRQ(MatSetType(C,((PetscObject)B)->type_name));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(PetscNew(&contents));
  C->product->data = contents;
  C->product->destroy = MatNormal_DenseDestroy;
  if (a->right) {
    CHKERRQ(MatProductCreate(a->A,C,NULL,contents->work));
  } else {
    CHKERRQ(MatProductCreate(a->A,B,NULL,contents->work));
  }
  CHKERRQ(MatProductSetType(contents->work[0],MATPRODUCT_AB));
  CHKERRQ(MatProductSetFromOptions(contents->work[0]));
  CHKERRQ(MatProductSymbolic(contents->work[0]));
  CHKERRQ(MatProductCreate(a->A,contents->work[0],NULL,contents->work+1));
  CHKERRQ(MatProductSetType(contents->work[1],MATPRODUCT_AtB));
  CHKERRQ(MatProductSetFromOptions(contents->work[1]));
  CHKERRQ(MatProductSymbolic(contents->work[1]));
  CHKERRQ(MatDenseGetArrayWrite(C,&array));
  CHKERRQ(MatSeqDenseSetPreallocation(contents->work[1],array));
  CHKERRQ(MatMPIDenseSetPreallocation(contents->work[1],array));
  CHKERRQ(MatDenseRestoreArrayWrite(C,&array));
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
    CHKERRQ(MatProductSetFromOptions_Normal_Dense_AB(C));
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
  CHKERRQ(MatGetSize(A,NULL,&nn));
  CHKERRQ(MatGetLocalSize(A,NULL,&n));
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)A),N));
  CHKERRQ(MatSetSizes(*N,n,n,nn,nn));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)*N,MATNORMAL));
  CHKERRQ(PetscLayoutReference(A->cmap,&(*N)->rmap));
  CHKERRQ(PetscLayoutReference(A->cmap,&(*N)->cmap));

  CHKERRQ(PetscNewLog(*N,&Na));
  (*N)->data = (void*) Na;
  CHKERRQ(PetscObjectReference((PetscObject)A));
  Na->A      = A;
  Na->scale  = 1.0;

  CHKERRQ(MatCreateVecs(A,NULL,&Na->w));

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

  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatNormalGetMat_C",MatNormalGetMat_Normal));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatConvert_normal_seqaij_C",MatConvert_Normal_AIJ));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatConvert_normal_mpiaij_C",MatConvert_Normal_AIJ));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_normal_seqdense_C",MatProductSetFromOptions_Normal_Dense));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_normal_mpidense_C",MatProductSetFromOptions_Normal_Dense));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_normal_dense_C",MatProductSetFromOptions_Normal_Dense));
  CHKERRQ(MatSetOption(*N,MAT_SYMMETRIC,PETSC_TRUE));
  CHKERRQ(MatGetVecType(A,&vtype));
  CHKERRQ(MatSetVecType(*N,vtype));
#if defined(PETSC_HAVE_DEVICE)
  CHKERRQ(MatBindToCPU(*N,A->boundtocpu));
#endif
  PetscFunctionReturn(0);
}
