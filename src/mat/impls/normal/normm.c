
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (left) {
    if (!a->left) {
      ierr = VecDuplicate(left,&a->left);CHKERRQ(ierr);
      ierr = VecCopy(left,a->left);CHKERRQ(ierr);
    } else {
      ierr = VecPointwiseMult(a->left,left,a->left);CHKERRQ(ierr);
    }
  }
  if (right) {
    if (!a->right) {
      ierr = VecDuplicate(right,&a->right);CHKERRQ(ierr);
      ierr = VecCopy(right,a->right);CHKERRQ(ierr);
    } else {
      ierr = VecPointwiseMult(a->right,right,a->right);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatIncreaseOverlap_Normal(Mat A,PetscInt is_max,IS is[],PetscInt ov)
{
  Mat_Normal     *a = (Mat_Normal*)A->data;
  Mat            pattern;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(ov < 0,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap specified");
  ierr = MatProductCreate(a->A,a->A,NULL,&pattern);CHKERRQ(ierr);
  ierr = MatProductSetType(pattern,MATPRODUCT_AtB);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(pattern);CHKERRQ(ierr);
  ierr = MatProductSymbolic(pattern);CHKERRQ(ierr);
  ierr = MatIncreaseOverlap(pattern,is_max,is,ov);CHKERRQ(ierr);
  ierr = MatDestroy(&pattern);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices_Normal(Mat mat,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  Mat_Normal     *a = (Mat_Normal*)mat->data;
  Mat            B = a->A, *suba;
  IS             *row;
  PetscInt       M;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(a->left || a->right || irow != icol,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Not implemented");
  if (scall != MAT_REUSE_MATRIX) {
    ierr = PetscCalloc1(n,submat);CHKERRQ(ierr);
  }
  ierr = MatGetSize(B,&M,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&row);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&row[0]);CHKERRQ(ierr);
  ierr = ISSetIdentity(row[0]);CHKERRQ(ierr);
  for (M = 1; M < n; ++M) row[M] = row[0];
  ierr = MatCreateSubMatrices(B,n,row,icol,MAT_INITIAL_MATRIX,&suba);CHKERRQ(ierr);
  for (M = 0; M < n; ++M) {
    ierr = MatCreateNormal(suba[M],*submat+M);CHKERRQ(ierr);
    ((Mat_Normal*)(*submat)[M]->data)->scale = a->scale;
  }
  ierr = ISDestroy(&row[0]);CHKERRQ(ierr);
  ierr = PetscFree(row);CHKERRQ(ierr);
  ierr = MatDestroySubMatrices(n,&suba);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPermute_Normal(Mat A,IS rowp,IS colp,Mat *B)
{
  Mat_Normal     *a = (Mat_Normal*)A->data;
  Mat            C,Aa = a->A;
  IS             row;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(rowp != colp,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Row permutation and column permutation must be the same");
  ierr = ISCreateStride(PetscObjectComm((PetscObject)Aa),Aa->rmap->n,Aa->rmap->rstart,1,&row);CHKERRQ(ierr);
  ierr = ISSetIdentity(row);CHKERRQ(ierr);
  ierr = MatPermute(Aa,row,colp,&C);CHKERRQ(ierr);
  ierr = ISDestroy(&row);CHKERRQ(ierr);
  ierr = MatCreateNormal(C,B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_Normal(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat_Normal     *a = (Mat_Normal*)A->data;
  Mat            C;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(a->left || a->right,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented");
  ierr = MatDuplicate(a->A,op,&C);CHKERRQ(ierr);
  ierr = MatCreateNormal(C,B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  if (op == MAT_COPY_VALUES) ((Mat_Normal*)(*B)->data)->scale = a->scale;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_Normal(Mat A,Mat B,MatStructure str)
{
  Mat_Normal     *a = (Mat_Normal*)A->data,*b = (Mat_Normal*)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(a->left || a->right,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented");
  ierr = MatCopy(a->A,b->A,str);CHKERRQ(ierr);
  b->scale = a->scale;
  ierr = VecDestroy(&b->left);CHKERRQ(ierr);
  ierr = VecDestroy(&b->right);CHKERRQ(ierr);
  ierr = VecDestroy(&b->leftwork);CHKERRQ(ierr);
  ierr = VecDestroy(&b->rightwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_Normal(Mat N,Vec x,Vec y)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = x;
  if (Na->right) {
    if (!Na->rightwork) {
      ierr = VecDuplicate(Na->right,&Na->rightwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->rightwork,Na->right,in);CHKERRQ(ierr);
    in   = Na->rightwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = MatMultTranspose(Na->A,Na->w,y);CHKERRQ(ierr);
  if (Na->left) {
    ierr = VecPointwiseMult(y,Na->left,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,Na->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_Normal(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->right) {
    if (!Na->rightwork) {
      ierr = VecDuplicate(Na->right,&Na->rightwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->rightwork,Na->right,in);CHKERRQ(ierr);
    in   = Na->rightwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = VecScale(Na->w,Na->scale);CHKERRQ(ierr);
  if (Na->left) {
    ierr = MatMultTranspose(Na->A,Na->w,v3);CHKERRQ(ierr);
    ierr = VecPointwiseMult(v3,Na->left,v3);CHKERRQ(ierr);
    ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
  } else {
    ierr = MatMultTransposeAdd(Na->A,Na->w,v2,v3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Normal(Mat N,Vec x,Vec y)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = x;
  if (Na->left) {
    if (!Na->leftwork) {
      ierr = VecDuplicate(Na->left,&Na->leftwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->leftwork,Na->left,in);CHKERRQ(ierr);
    in   = Na->leftwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = MatMultTranspose(Na->A,Na->w,y);CHKERRQ(ierr);
  if (Na->right) {
    ierr = VecPointwiseMult(y,Na->right,y);CHKERRQ(ierr);
  }
  ierr = VecScale(y,Na->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_Normal(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;
  Vec            in;

  PetscFunctionBegin;
  in = v1;
  if (Na->left) {
    if (!Na->leftwork) {
      ierr = VecDuplicate(Na->left,&Na->leftwork);CHKERRQ(ierr);
    }
    ierr = VecPointwiseMult(Na->leftwork,Na->left,in);CHKERRQ(ierr);
    in   = Na->leftwork;
  }
  ierr = MatMult(Na->A,in,Na->w);CHKERRQ(ierr);
  ierr = VecScale(Na->w,Na->scale);CHKERRQ(ierr);
  if (Na->right) {
    ierr = MatMultTranspose(Na->A,Na->w,v3);CHKERRQ(ierr);
    ierr = VecPointwiseMult(v3,Na->right,v3);CHKERRQ(ierr);
    ierr = VecAXPY(v3,1.0,v2);CHKERRQ(ierr);
  } else {
    ierr = MatMultTransposeAdd(Na->A,Na->w,v2,v3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_Normal(Mat N)
{
  Mat_Normal     *Na = (Mat_Normal*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->w);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->left);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->right);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->leftwork);CHKERRQ(ierr);
  ierr = VecDestroy(&Na->rightwork);CHKERRQ(ierr);
  ierr = PetscFree(N->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatNormalGetMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatConvert_normal_seqaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatConvert_normal_mpiaij_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_normal_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_normal_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)N,"MatProductSetFromOptions_normal_dense_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      Slow, nonscalable version
*/
PetscErrorCode MatGetDiagonal_Normal(Mat N,Vec v)
{
  Mat_Normal        *Na = (Mat_Normal*)N->data;
  Mat               A   = Na->A;
  PetscErrorCode    ierr;
  PetscInt          i,j,rstart,rend,nnz;
  const PetscInt    *cols;
  PetscScalar       *diag,*work,*values;
  const PetscScalar *mvalues;

  PetscFunctionBegin;
  ierr = PetscMalloc2(A->cmap->N,&diag,A->cmap->N,&work);CHKERRQ(ierr);
  ierr = PetscArrayzero(work,A->cmap->N);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nnz,&cols,&mvalues);CHKERRQ(ierr);
    for (j=0; j<nnz; j++) {
      work[cols[j]] += mvalues[j]*mvalues[j];
    }
    ierr = MatRestoreRow(A,i,&nnz,&cols,&mvalues);CHKERRQ(ierr);
  }
  ierr   = MPIU_Allreduce(work,diag,A->cmap->N,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)N));CHKERRMPI(ierr);
  rstart = N->cmap->rstart;
  rend   = N->cmap->rend;
  ierr   = VecGetArray(v,&values);CHKERRQ(ierr);
  ierr   = PetscArraycpy(values,diag+rstart,rend-rstart);CHKERRQ(ierr);
  ierr   = VecRestoreArray(v,&values);CHKERRQ(ierr);
  ierr   = PetscFree2(diag,work);CHKERRQ(ierr);
  ierr   = VecScale(v,Na->scale);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(M,2);
  ierr = PetscUseMethod(A,"MatNormalGetMat_C",(Mat,Mat*),(A,M));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_Normal_AIJ(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_Normal     *Aa = (Mat_Normal*)A->data;
  Mat            B;
  PetscInt       m,n,M,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
    ierr = MatProductReplaceMats(Aa->A,Aa->A,NULL,B);CHKERRQ(ierr);
  } else {
    ierr = MatProductCreate(Aa->A,Aa->A,NULL,&B);CHKERRQ(ierr);
    ierr = MatProductSetType(B,MATPRODUCT_AtB);CHKERRQ(ierr);
    ierr = MatProductSetFromOptions(B);CHKERRQ(ierr);
    ierr = MatProductSymbolic(B);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = MatProductNumeric(B);CHKERRQ(ierr);
  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else if (reuse == MAT_INITIAL_MATRIX) *newmat = B;
  ierr = MatConvert(*newmat,MATAIJ,MAT_INPLACE_MATRIX,newmat);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  A = C->product->A;
  a = (Mat_Normal*)A->data;
  B = C->product->B;
  contents = (Normal_Dense*)C->product->data;
  PetscCheckFalse(!contents,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  if (a->right) {
    ierr = MatCopy(B,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDiagonalScale(C,a->right,NULL);CHKERRQ(ierr);
  }
  ierr = MatProductNumeric(contents->work[0]);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(C,&array);CHKERRQ(ierr);
  ierr = MatDensePlaceArray(contents->work[1],array);CHKERRQ(ierr);
  ierr = MatProductNumeric(contents->work[1]);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayWrite(C,&array);CHKERRQ(ierr);
  ierr = MatDenseResetArray(contents->work[1]);CHKERRQ(ierr);
  ierr = MatSetOption(C,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatScale(C,a->scale);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatNormal_DenseDestroy(void *ctx)
{
  Normal_Dense   *contents = (Normal_Dense*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(contents->work);CHKERRQ(ierr);
  ierr = MatDestroy(contents->work+1);CHKERRQ(ierr);
  ierr = PetscFree(contents);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSymbolic_Normal_Dense(Mat C)
{
  Mat            A,B;
  Normal_Dense   *contents = NULL;
  Mat_Normal     *a;
  PetscScalar    *array;
  PetscInt       n,N,m,M;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheckFalse(C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A = C->product->A;
  a = (Mat_Normal*)A->data;
  PetscCheckFalse(a->left,PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"Not implemented");
  B = C->product->B;
  ierr = MatGetLocalSize(C,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(C,&M,&N);CHKERRQ(ierr);
  if (m == PETSC_DECIDE || n == PETSC_DECIDE || M == PETSC_DECIDE || N == PETSC_DECIDE) {
    ierr = MatGetLocalSize(B,NULL,&n);CHKERRQ(ierr);
    ierr = MatGetSize(B,NULL,&N);CHKERRQ(ierr);
    ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
    ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
    ierr = MatSetSizes(C,m,n,M,N);CHKERRQ(ierr);
  }
  ierr = MatSetType(C,((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = PetscNew(&contents);CHKERRQ(ierr);
  C->product->data = contents;
  C->product->destroy = MatNormal_DenseDestroy;
  if (a->right) {
    ierr = MatProductCreate(a->A,C,NULL,contents->work);CHKERRQ(ierr);
  } else {
    ierr = MatProductCreate(a->A,B,NULL,contents->work);CHKERRQ(ierr);
  }
  ierr = MatProductSetType(contents->work[0],MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(contents->work[0]);CHKERRQ(ierr);
  ierr = MatProductSymbolic(contents->work[0]);CHKERRQ(ierr);
  ierr = MatProductCreate(a->A,contents->work[0],NULL,contents->work+1);CHKERRQ(ierr);
  ierr = MatProductSetType(contents->work[1],MATPRODUCT_AtB);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(contents->work[1]);CHKERRQ(ierr);
  ierr = MatProductSymbolic(contents->work[1]);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(C,&array);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(contents->work[1],array);CHKERRQ(ierr);
  ierr = MatMPIDenseSetPreallocation(contents->work[1],array);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayWrite(C,&array);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_AB) {
    ierr = MatProductSetFromOptions_Normal_Dense_AB(C);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       n,nn;
  Mat_Normal     *Na;
  VecType        vtype;

  PetscFunctionBegin;
  ierr = MatGetSize(A,NULL,&nn);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,NULL,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,n,n,nn,nn);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATNORMAL);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&(*N)->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&(*N)->cmap);CHKERRQ(ierr);

  ierr       = PetscNewLog(*N,&Na);CHKERRQ(ierr);
  (*N)->data = (void*) Na;
  ierr       = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  Na->A      = A;
  Na->scale  = 1.0;

  ierr = MatCreateVecs(A,NULL,&Na->w);CHKERRQ(ierr);

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

  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatNormalGetMat_C",MatNormalGetMat_Normal);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatConvert_normal_seqaij_C",MatConvert_Normal_AIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatConvert_normal_mpiaij_C",MatConvert_Normal_AIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_normal_seqdense_C",MatProductSetFromOptions_Normal_Dense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_normal_mpidense_C",MatProductSetFromOptions_Normal_Dense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*N),"MatProductSetFromOptions_normal_dense_C",MatProductSetFromOptions_Normal_Dense);CHKERRQ(ierr);
  ierr = MatSetOption(*N,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatGetVecType(A,&vtype);CHKERRQ(ierr);
  ierr = MatSetVecType(*N,vtype);CHKERRQ(ierr);
#if defined(PETSC_HAVE_DEVICE)
  ierr = MatBindToCPU(*N,A->boundtocpu);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
