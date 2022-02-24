#include <petsc/private/matimpl.h>        /*I "petscmat.h" I*/

PetscErrorCode MatConvert_Shell(Mat oldmat,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            mat;
  Vec            in,out;
  PetscScalar    *array;
  PetscInt       *dnnz,*onnz,*dnnzu,*onnzu;
  PetscInt       cst,Nbs,mbs,nbs,rbs,cbs;
  PetscInt       im,i,m,n,M,N,*rows,start;

  PetscFunctionBegin;
  CHKERRQ(MatGetOwnershipRange(oldmat,&start,NULL));
  CHKERRQ(MatGetOwnershipRangeColumn(oldmat,&cst,NULL));
  CHKERRQ(MatCreateVecs(oldmat,&in,&out));
  CHKERRQ(MatGetLocalSize(oldmat,&m,&n));
  CHKERRQ(MatGetSize(oldmat,&M,&N));
  CHKERRQ(PetscMalloc1(m,&rows));
  if (reuse != MAT_REUSE_MATRIX) {
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)oldmat),&mat));
    CHKERRQ(MatSetSizes(mat,m,n,M,N));
    CHKERRQ(MatSetType(mat,newtype));
    CHKERRQ(MatSetBlockSizesFromMats(mat,oldmat,oldmat));
    CHKERRQ(MatGetBlockSizes(mat,&rbs,&cbs));
    mbs  = m/rbs;
    nbs  = n/cbs;
    Nbs  = N/cbs;
    cst  = cst/cbs;
    CHKERRQ(PetscMalloc4(mbs,&dnnz,mbs,&onnz,mbs,&dnnzu,mbs,&onnzu));
    for (i=0; i<mbs; i++) {
      dnnz[i]  = nbs;
      onnz[i]  = Nbs - nbs;
      dnnzu[i] = PetscMax(nbs - i,0);
      onnzu[i] = PetscMax(Nbs - (cst + nbs),0);
    }
    CHKERRQ(MatXAIJSetPreallocation(mat,PETSC_DECIDE,dnnz,onnz,dnnzu,onnzu));
    CHKERRQ(PetscFree4(dnnz,onnz,dnnzu,onnzu));
    CHKERRQ(VecSetOption(in,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE));
    CHKERRQ(MatSetUp(mat));
  } else {
    mat = *newmat;
    CHKERRQ(MatZeroEntries(mat));
  }
  for (i=0; i<N; i++) {
    PetscInt j;

    CHKERRQ(VecZeroEntries(in));
    CHKERRQ(VecSetValue(in,i,1.,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(in));
    CHKERRQ(VecAssemblyEnd(in));
    CHKERRQ(MatMult(oldmat,in,out));
    CHKERRQ(VecGetArray(out,&array));
    for (j=0, im = 0; j<m; j++) {
      if (PetscAbsScalar(array[j]) == 0.0) continue;
      rows[im]  = j+start;
      array[im] = array[j];
      im++;
    }
    CHKERRQ(MatSetValues(mat,im,rows,1,&i,array,INSERT_VALUES));
    CHKERRQ(VecRestoreArray(out,&array));
  }
  CHKERRQ(PetscFree(rows));
  CHKERRQ(VecDestroy(&in));
  CHKERRQ(VecDestroy(&out));
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INPLACE_MATRIX) {
    CHKERRQ(MatHeaderReplace(oldmat,&mat));
  } else {
    *newmat = mat;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_CF(Mat A,Vec X)
{
  Mat            B;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&B));
  PetscCheckFalse(!B,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  CHKERRQ(MatGetDiagonal(B,X));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_CF(Mat A,Vec X,Vec Y)
{
  Mat            B;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&B));
  PetscCheckFalse(!B,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  CHKERRQ(MatMult(B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_CF(Mat A,Vec X,Vec Y)
{
  Mat            B;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&B));
  PetscCheckFalse(!B,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  CHKERRQ(MatMultTranspose(B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_CF(Mat A)
{
  Mat            B;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&B));
  PetscCheckFalse(!B,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_anytype_C",NULL));
  PetscFunctionReturn(0);
}

typedef struct {
  void           *userdata;
  PetscErrorCode (*userdestroy)(void*);
  PetscErrorCode (*numeric)(Mat);
  MatProductType ptype;
  Mat            Dwork;
} MatMatCF;

static PetscErrorCode MatProductDestroy_CF(void *data)
{
  MatMatCF       *mmcfdata = (MatMatCF*)data;

  PetscFunctionBegin;
  if (mmcfdata->userdestroy) {
    CHKERRQ((*mmcfdata->userdestroy)(mmcfdata->userdata));
  }
  CHKERRQ(MatDestroy(&mmcfdata->Dwork));
  CHKERRQ(PetscFree(mmcfdata));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumericPhase_CF(Mat A, Mat B, Mat C, void *data)
{
  MatMatCF       *mmcfdata = (MatMatCF*)data;

  PetscFunctionBegin;
  PetscCheckFalse(!mmcfdata,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing data");
  PetscCheckFalse(!mmcfdata->numeric,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing numeric operation");
  /* the MATSHELL interface allows us to play with the product data */
  CHKERRQ(PetscNew(&C->product));
  C->product->type  = mmcfdata->ptype;
  C->product->data  = mmcfdata->userdata;
  C->product->Dwork = mmcfdata->Dwork;
  CHKERRQ(MatShellGetContext(A,&C->product->A));
  C->product->B = B;
  CHKERRQ((*mmcfdata->numeric)(C));
  CHKERRQ(PetscFree(C->product));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolicPhase_CF(Mat A, Mat B, Mat C, void **data)
{
  MatMatCF       *mmcfdata;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&C->product->A));
  CHKERRQ(MatProductSetFromOptions(C));
  CHKERRQ(MatProductSymbolic(C));
  /* the MATSHELL interface does not allow non-empty product data */
  CHKERRQ(PetscNew(&mmcfdata));

  mmcfdata->numeric     = C->ops->productnumeric;
  mmcfdata->ptype       = C->product->type;
  mmcfdata->userdata    = C->product->data;
  mmcfdata->userdestroy = C->product->destroy;
  mmcfdata->Dwork       = C->product->Dwork;

  C->product->Dwork   = NULL;
  C->product->data    = NULL;
  C->product->destroy = NULL;
  C->product->A       = A;

  *data = mmcfdata;
  PetscFunctionReturn(0);
}

/* only for A of type shell, mainly used for MatMat operations of shells with AXPYs */
static PetscErrorCode MatProductSetFromOptions_CF(Mat D)
{
  Mat            A,B,Ain;
  void           (*Af)(void) = NULL;
  PetscBool      flg;

  PetscFunctionBegin;
  MatCheckProduct(D,1);
  if (D->product->type == MATPRODUCT_ABC) PetscFunctionReturn(0);
  A = D->product->A;
  B = D->product->B;
  CHKERRQ(MatIsShell(A,&flg));
  if (!flg) PetscFunctionReturn(0);
  CHKERRQ(PetscObjectQueryFunction((PetscObject)A,"MatProductSetFromOptions_anytype_C",&Af));
  if (Af == (void(*)(void))MatProductSetFromOptions_CF) {
    CHKERRQ(MatShellGetContext(A,&Ain));
  } else PetscFunctionReturn(0);
  D->product->A = Ain;
  CHKERRQ(MatProductSetFromOptions(D));
  D->product->A = A;
  if (D->ops->productsymbolic) { /* we have a symbolic match, now populate the MATSHELL operations */
    CHKERRQ(MatShellSetMatProductOperation(A,D->product->type,MatProductSymbolicPhase_CF,MatProductNumericPhase_CF,MatProductDestroy_CF,((PetscObject)B)->type_name,NULL));
    CHKERRQ(MatProductSetFromOptions(D));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertFrom_Shell(Mat A,MatType newtype,MatReuse reuse,Mat *B)
{
  Mat            M;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscStrcmp(newtype,MATSHELL,&flg));
  PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only conversion to MATSHELL");
  if (reuse == MAT_INITIAL_MATRIX) {
    CHKERRQ(PetscObjectReference((PetscObject)A));
    CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject)A),A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,A,&M));
    CHKERRQ(MatSetBlockSizesFromMats(M,A,A));
    CHKERRQ(MatShellSetOperation(M,MATOP_MULT,          (void (*)(void))MatMult_CF));
    CHKERRQ(MatShellSetOperation(M,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_CF));
    CHKERRQ(MatShellSetOperation(M,MATOP_GET_DIAGONAL,  (void (*)(void))MatGetDiagonal_CF));
    CHKERRQ(MatShellSetOperation(M,MATOP_DESTROY,       (void (*)(void))MatDestroy_CF));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)M,"MatProductSetFromOptions_anytype_C",MatProductSetFromOptions_CF));
    CHKERRQ(PetscFree(M->defaultvectype));
    CHKERRQ(PetscStrallocpy(A->defaultvectype,&M->defaultvectype));
#if defined(PETSC_HAVE_DEVICE)
    CHKERRQ(MatBindToCPU(M,A->boundtocpu));
#endif
    *B = M;
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented");
  PetscFunctionReturn(0);
}
