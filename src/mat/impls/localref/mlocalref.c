
#include <petsc-private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat Top;
  PetscErrorCode (*SetValues)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
  PetscErrorCode (*SetValuesBlocked)(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
} Mat_LocalRef;

/* These need to be macros because they use sizeof */
#define IndexSpaceGet(buf,nrow,ncol,irowm,icolm) do {                   \
    if (nrow + ncol > (PetscInt)(sizeof(buf)/sizeof(buf[0]))) {         \
      ierr = PetscMalloc2(nrow,PetscInt,&irowm,ncol,PetscInt,&icolm);CHKERRQ(ierr); \
    } else {                                                            \
      irowm = &buf[0];                                                  \
      icolm = &buf[nrow];                                               \
    }                                                                   \
  } while (0)

#define IndexSpaceRestore(buf,nrow,ncol,irowm,icolm) do {       \
    if (nrow + ncol > (PetscInt)(sizeof(buf)/sizeof(buf[0]))) { \
      ierr = PetscFree2(irowm,icolm);CHKERRQ(ierr);             \
    }                                                           \
  } while (0)

static void BlockIndicesExpand(PetscInt n,const PetscInt idx[],PetscInt bs,PetscInt idxm[])
{
  PetscInt i,j;
  for (i=0; i<n; i++) {
    for (j=0; j<bs; j++) {
      idxm[i*bs+j] = idx[i]*bs + j;
    }
  }
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBlockedLocal_LocalRef_Block"
static PetscErrorCode MatSetValuesBlockedLocal_LocalRef_Block(Mat A,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
{
  Mat_LocalRef   *lr = (Mat_LocalRef*)A->data;
  PetscErrorCode ierr;
  PetscInt       buf[4096],*irowm,*icolm;

  PetscFunctionBegin;
  if (!nrow || !ncol) PetscFunctionReturn(0);
  IndexSpaceGet(buf,nrow,ncol,irowm,icolm);
  ierr = ISLocalToGlobalMappingApply(A->rmap->bmapping,nrow,irow,irowm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(A->cmap->bmapping,ncol,icol,icolm);CHKERRQ(ierr);
  ierr = (*lr->SetValuesBlocked)(lr->Top,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
  IndexSpaceRestore(buf,nrow,ncol,irowm,icolm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesBlockedLocal_LocalRef_Scalar"
static PetscErrorCode MatSetValuesBlockedLocal_LocalRef_Scalar(Mat A,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
{
  Mat_LocalRef   *lr = (Mat_LocalRef*)A->data;
  PetscErrorCode ierr;
  PetscInt       bs  = A->rmap->bs,buf[4096],*irowm,*icolm;

  PetscFunctionBegin;
  IndexSpaceGet(buf,nrow*bs,ncol*bs,irowm,icolm);
  BlockIndicesExpand(nrow,irow,bs,irowm);
  BlockIndicesExpand(ncol,icol,bs,icolm);
  ierr = ISLocalToGlobalMappingApply(A->rmap->mapping,nrow*bs,irowm,irowm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(A->cmap->mapping,ncol*bs,icolm,icolm);CHKERRQ(ierr);
  ierr = (*lr->SetValues)(lr->Top,nrow*bs,irowm,ncol*bs,icolm,y,addv);CHKERRQ(ierr);
  IndexSpaceRestore(buf,nrow*bs,ncol*bs,irowm,icolm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetValuesLocal_LocalRef_Scalar"
static PetscErrorCode MatSetValuesLocal_LocalRef_Scalar(Mat A,PetscInt nrow,const PetscInt irow[],PetscInt ncol,const PetscInt icol[],const PetscScalar y[],InsertMode addv)
{
  Mat_LocalRef   *lr = (Mat_LocalRef*)A->data;
  PetscErrorCode ierr;
  PetscInt       buf[4096],*irowm,*icolm;

  PetscFunctionBegin;
  IndexSpaceGet(buf,nrow,ncol,irowm,icolm);
  ierr = ISLocalToGlobalMappingApply(A->rmap->mapping,nrow,irow,irowm);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(A->cmap->mapping,ncol,icol,icolm);CHKERRQ(ierr);
  ierr = (*lr->SetValues)(lr->Top,nrow,irowm,ncol,icolm,y,addv);CHKERRQ(ierr);
  IndexSpaceRestore(buf,nrow,ncol,irowm,icolm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISL2GCompose"
/* Compose an IS with an ISLocalToGlobalMapping to map from IS source indices to global indices */
static PetscErrorCode ISL2GCompose(IS is,ISLocalToGlobalMapping ltog,ISLocalToGlobalMapping *cltog)
{
  PetscErrorCode ierr;
  const PetscInt *idx;
  PetscInt m,*idxm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,2);
  PetscValidPointer(cltog,3);
  ierr = ISGetLocalSize(is,&m);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idx);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt i;
    for (i=0; i<m; i++) {
      if (idx[i] < 0 || ltog->n <= idx[i]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"is[%D] = %D is not in the local range [0:%D]",i,idx[i],ltog->n);
    }
  }
#endif
  ierr = PetscMalloc(m*sizeof(PetscInt),&idxm);CHKERRQ(ierr);
  if (ltog) {
    ierr = ISLocalToGlobalMappingApply(ltog,m,idx,idxm);CHKERRQ(ierr);
  } else {
    ierr = PetscMemcpy(idxm,idx,m*sizeof(PetscInt));CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingCreate(((PetscObject)is)->comm,m,idxm,PETSC_OWN_POINTER,cltog);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISL2GComposeBlock"
static PetscErrorCode ISL2GComposeBlock(IS is,ISLocalToGlobalMapping ltog,ISLocalToGlobalMapping *cltog)
{
  PetscErrorCode ierr;
  const PetscInt *idx;
  PetscInt m,*idxm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,2);
  PetscValidPointer(cltog,3);
  ierr = ISBlockGetLocalSize(is,&m);CHKERRQ(ierr);
  ierr = ISBlockGetIndices(is,&idx);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  {
    PetscInt i;
    for (i=0; i<m; i++) {
      if (idx[i] < 0 || ltog->n <= idx[i]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"is[%D] = %D is not in the local range [0:%D]",i,idx[i],ltog->n);
    }
  }
#endif
  ierr = PetscMalloc(m*sizeof(PetscInt),&idxm);CHKERRQ(ierr);
  if (ltog) {
    ierr = ISLocalToGlobalMappingApply(ltog,m,idx,idxm);CHKERRQ(ierr);
  } else {
    ierr = PetscMemcpy(idxm,idx,m*sizeof(PetscInt));CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingCreate(((PetscObject)is)->comm,m,idxm,PETSC_OWN_POINTER,cltog);CHKERRQ(ierr);
  ierr = ISBlockRestoreIndices(is,&idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_LocalRef"
static PetscErrorCode MatDestroy_LocalRef(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(B->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCreateLocalRef"
/*@
   MatCreateLocalRef - Gets a logical reference to a local submatrix, for use in assembly

   Not Collective

   Input Arguments:
+ A - Full matrix, generally parallel
. isrow - Local index set for the rows
- iscol - Local index set for the columns

   Output Arguments:
. newmat - New serial Mat

   Level: developer

   Notes:
   Most will use MatGetLocalSubMatrix() which returns a real matrix corresponding to the local
   block if it available, such as with matrix formats that store these blocks separately.

   The new matrix forwards MatSetValuesLocal() and MatSetValuesBlockedLocal() to the global system.
   In general, it does not define MatMult() or any other functions.  Local submatrices can be nested.

.seealso: MatSetValuesLocal(), MatSetValuesBlockedLocal(), MatGetLocalSubMatrix(), MatCreateSubMatrix()
@*/
PetscErrorCode  MatCreateLocalRef(Mat A,IS isrow,IS iscol,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat_LocalRef   *lr;
  Mat            B;
  PetscInt       m,n;
  PetscBool      islr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(isrow,IS_CLASSID,2);
  PetscValidHeaderSpecific(iscol,IS_CLASSID,3);
  PetscValidPointer(newmat,4);
  *newmat = 0;

  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&m);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&n);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,m,n);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATLOCALREF);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  B->ops->destroy = MatDestroy_LocalRef;

  ierr = PetscNewLog(B,Mat_LocalRef,&lr);CHKERRQ(ierr);
  B->data = (void*)lr;

  ierr = PetscObjectTypeCompare((PetscObject)A,MATLOCALREF,&islr);CHKERRQ(ierr);
  if (islr) {
    Mat_LocalRef *alr = (Mat_LocalRef*)A->data;
    lr->Top = alr->Top;
  } else {
    /* This does not increase the reference count because MatLocalRef is not allowed to live longer than its parent */
    lr->Top = A;
  }
  {
    ISLocalToGlobalMapping rltog,cltog;
    PetscInt abs,rbs,cbs;

    /* We will translate directly to global indices for the top level */
    lr->SetValues        = MatSetValues;
    lr->SetValuesBlocked = MatSetValuesBlocked;

    B->ops->setvalueslocal = MatSetValuesLocal_LocalRef_Scalar;
    ierr = ISL2GCompose(isrow,A->rmap->mapping,&rltog);CHKERRQ(ierr);
    if (isrow == iscol && A->rmap->mapping == A->cmap->mapping) {
      ierr = PetscObjectReference((PetscObject)rltog);CHKERRQ(ierr);
      cltog = rltog;
    } else {
      ierr = ISL2GCompose(iscol,A->cmap->mapping,&cltog);CHKERRQ(ierr);
    }
    ierr = MatSetLocalToGlobalMapping(B,rltog,cltog);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&rltog);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&cltog);CHKERRQ(ierr);

    ierr = MatGetBlockSize(A,&abs);CHKERRQ(ierr);
    ierr = ISGetBlockSize(isrow,&rbs);CHKERRQ(ierr);
    ierr = ISGetBlockSize(iscol,&cbs);CHKERRQ(ierr);
    if (rbs == cbs) {           /* submatrix has block structure, so user can insert values with blocked interface */
      ierr = PetscLayoutSetBlockSize(B->rmap,rbs);CHKERRQ(ierr);
      ierr = PetscLayoutSetBlockSize(B->cmap,cbs);CHKERRQ(ierr);
      if (abs != rbs || abs == 1) {
        /* Top-level matrix has different block size, so we have to call its scalar insertion interface */
        B->ops->setvaluesblockedlocal = MatSetValuesBlockedLocal_LocalRef_Scalar;
      } else {
        /* Block sizes match so we can forward values to the top level using the block interface */
        B->ops->setvaluesblockedlocal = MatSetValuesBlockedLocal_LocalRef_Block;
        ierr = ISL2GComposeBlock(isrow,A->rmap->bmapping,&rltog);CHKERRQ(ierr);
        if (isrow == iscol && A->rmap->bmapping == A->cmap->bmapping) {
          ierr =  PetscObjectReference((PetscObject)rltog);CHKERRQ(ierr);
          cltog = rltog;
        } else {
          ierr = ISL2GComposeBlock(iscol,A->cmap->bmapping,&cltog);CHKERRQ(ierr);
        }
        ierr = MatSetLocalToGlobalMappingBlock(B,rltog,cltog);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy(&rltog);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy(&cltog);CHKERRQ(ierr);
      }
    }
  }
  *newmat = B;
  PetscFunctionReturn(0);
}
