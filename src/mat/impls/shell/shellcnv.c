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
  PetscCall(MatGetOwnershipRange(oldmat,&start,NULL));
  PetscCall(MatGetOwnershipRangeColumn(oldmat,&cst,NULL));
  PetscCall(MatCreateVecs(oldmat,&in,&out));
  PetscCall(MatGetLocalSize(oldmat,&m,&n));
  PetscCall(MatGetSize(oldmat,&M,&N));
  PetscCall(PetscMalloc1(m,&rows));
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)oldmat),&mat));
    PetscCall(MatSetSizes(mat,m,n,M,N));
    PetscCall(MatSetType(mat,newtype));
    PetscCall(MatSetBlockSizesFromMats(mat,oldmat,oldmat));
    PetscCall(MatGetBlockSizes(mat,&rbs,&cbs));
    mbs  = m/rbs;
    nbs  = n/cbs;
    Nbs  = N/cbs;
    cst  = cst/cbs;
    PetscCall(PetscMalloc4(mbs,&dnnz,mbs,&onnz,mbs,&dnnzu,mbs,&onnzu));
    for (i=0; i<mbs; i++) {
      dnnz[i]  = nbs;
      onnz[i]  = Nbs - nbs;
      dnnzu[i] = PetscMax(nbs - i,0);
      onnzu[i] = PetscMax(Nbs - (cst + nbs),0);
    }
    PetscCall(MatXAIJSetPreallocation(mat,PETSC_DECIDE,dnnz,onnz,dnnzu,onnzu));
    PetscCall(PetscFree4(dnnz,onnz,dnnzu,onnzu));
    PetscCall(VecSetOption(in,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE));
    PetscCall(MatSetUp(mat));
  } else {
    mat = *newmat;
    PetscCall(MatZeroEntries(mat));
  }
  for (i=0; i<N; i++) {
    PetscInt j;

    PetscCall(VecZeroEntries(in));
    PetscCall(VecSetValue(in,i,1.,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(in));
    PetscCall(VecAssemblyEnd(in));
    PetscCall(MatMult(oldmat,in,out));
    PetscCall(VecGetArray(out,&array));
    for (j=0, im = 0; j<m; j++) {
      if (PetscAbsScalar(array[j]) == 0.0) continue;
      rows[im]  = j+start;
      array[im] = array[j];
      im++;
    }
    PetscCall(MatSetValues(mat,im,rows,1,&i,array,INSERT_VALUES));
    PetscCall(VecRestoreArray(out,&array));
  }
  PetscCall(PetscFree(rows));
  PetscCall(VecDestroy(&in));
  PetscCall(VecDestroy(&out));
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(oldmat,&mat));
  } else {
    *newmat = mat;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_CF(Mat A,Vec X)
{
  Mat            B;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&B));
  PetscCheck(B,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  PetscCall(MatGetDiagonal(B,X));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_CF(Mat A,Vec X,Vec Y)
{
  Mat            B;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&B));
  PetscCheck(B,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  PetscCall(MatMult(B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_CF(Mat A,Vec X,Vec Y)
{
  Mat            B;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&B));
  PetscCheck(B,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  PetscCall(MatMultTranspose(B,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_CF(Mat A)
{
  Mat            B;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&B));
  PetscCheck(B,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing user matrix");
  PetscCall(MatDestroy(&B));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_anytype_C",NULL));
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
    PetscCall((*mmcfdata->userdestroy)(mmcfdata->userdata));
  }
  PetscCall(MatDestroy(&mmcfdata->Dwork));
  PetscCall(PetscFree(mmcfdata));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumericPhase_CF(Mat A, Mat B, Mat C, void *data)
{
  MatMatCF       *mmcfdata = (MatMatCF*)data;

  PetscFunctionBegin;
  PetscCheck(mmcfdata,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing data");
  PetscCheck(mmcfdata->numeric,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Missing numeric operation");
  /* the MATSHELL interface allows us to play with the product data */
  PetscCall(PetscNew(&C->product));
  C->product->type  = mmcfdata->ptype;
  C->product->data  = mmcfdata->userdata;
  C->product->Dwork = mmcfdata->Dwork;
  PetscCall(MatShellGetContext(A,&C->product->A));
  C->product->B = B;
  PetscCall((*mmcfdata->numeric)(C));
  PetscCall(PetscFree(C->product));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolicPhase_CF(Mat A, Mat B, Mat C, void **data)
{
  MatMatCF       *mmcfdata;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&C->product->A));
  PetscCall(MatProductSetFromOptions(C));
  PetscCall(MatProductSymbolic(C));
  /* the MATSHELL interface does not allow non-empty product data */
  PetscCall(PetscNew(&mmcfdata));

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
  PetscCall(MatIsShell(A,&flg));
  if (!flg) PetscFunctionReturn(0);
  PetscCall(PetscObjectQueryFunction((PetscObject)A,"MatProductSetFromOptions_anytype_C",&Af));
  if (Af == (void(*)(void))MatProductSetFromOptions_CF) {
    PetscCall(MatShellGetContext(A,&Ain));
  } else PetscFunctionReturn(0);
  D->product->A = Ain;
  PetscCall(MatProductSetFromOptions(D));
  D->product->A = A;
  if (D->ops->productsymbolic) { /* we have a symbolic match, now populate the MATSHELL operations */
    PetscCall(MatShellSetMatProductOperation(A,D->product->type,MatProductSymbolicPhase_CF,MatProductNumericPhase_CF,MatProductDestroy_CF,((PetscObject)B)->type_name,NULL));
    PetscCall(MatProductSetFromOptions(D));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvertFrom_Shell(Mat A,MatType newtype,MatReuse reuse,Mat *B)
{
  Mat            M;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(newtype,MATSHELL,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only conversion to MATSHELL");
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(PetscObjectReference((PetscObject)A));
    PetscCall(MatCreateShell(PetscObjectComm((PetscObject)A),A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N,A,&M));
    PetscCall(MatSetBlockSizesFromMats(M,A,A));
    PetscCall(MatShellSetOperation(M,MATOP_MULT,          (void (*)(void))MatMult_CF));
    PetscCall(MatShellSetOperation(M,MATOP_MULT_TRANSPOSE,(void (*)(void))MatMultTranspose_CF));
    PetscCall(MatShellSetOperation(M,MATOP_GET_DIAGONAL,  (void (*)(void))MatGetDiagonal_CF));
    PetscCall(MatShellSetOperation(M,MATOP_DESTROY,       (void (*)(void))MatDestroy_CF));
    PetscCall(PetscObjectComposeFunction((PetscObject)M,"MatProductSetFromOptions_anytype_C",MatProductSetFromOptions_CF));
    PetscCall(PetscFree(M->defaultvectype));
    PetscCall(PetscStrallocpy(A->defaultvectype,&M->defaultvectype));
#if defined(PETSC_HAVE_DEVICE)
    PetscCall(MatBindToCPU(M,A->boundtocpu));
#endif
    *B = M;
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented");
  PetscFunctionReturn(0);
}
