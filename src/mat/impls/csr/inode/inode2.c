#include "src/mat/impls/csr/inode/inode.h"
EXTERN PetscErrorCode Mat_CheckInode(Mat,PetscTruth);
EXTERN_C_BEGIN
EXTERN PetscErrorCode MatInodeAdjustForInodes_Inode(Mat,IS*,IS*);
EXTERN PetscErrorCode MatInodeGetInodeSizes_Inode(Mat,PetscInt*,PetscInt*[],PetscInt*);
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatView_Inode"
PetscErrorCode MatView_Inode(Mat A,PetscViewer viewer)
{
  Mat_inode         *a=(Mat_inode*)A->data;
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL || format == PETSC_VIEWER_ASCII_INFO) {
      if (a->inode.size) {
        ierr = PetscViewerASCIIPrintf(viewer,"using I-node routines: found %D nodes, limit used is %D\n",
                                      a->inode.node_count,a->inode.limit);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"not using I-node routines\n");CHKERRQ(ierr);
      }
    }
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_Inode"
PetscErrorCode MatAssemblyEnd_Inode(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  PetscTruth     samestructure;

  PetscFunctionBegin;
  /* info.nz_unneeded of zero denotes no structural change was made to the matrix during Assembly */
  samestructure = (PetscTruth)(!A->info.nz_unneeded);
  /* check for identical nodes. If found, use inode functions */
  ierr = Mat_CheckInode(A,samestructure);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Inode"
PetscErrorCode MatDestroy_Inode(Mat A)
{
  PetscErrorCode ierr;
  Mat_inode      *a=(Mat_inode*)A->data;

  PetscFunctionBegin;
  if (a->inode.size) {
    ierr = PetscFree(a->inode.size);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatInodeAdjustForInodes_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatInodeGetInodeSizes_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreate_Inode"
PetscErrorCode MatCreate_Inode(Mat B)
{
  Mat_inode      *b=(Mat_inode*)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  b->inode.use         = PETSC_TRUE;
  b->inode.node_count  = 0;
  b->inode.size        = 0;
  b->inode.limit       = 5;
  b->inode.max_limit   = 5;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatInodeAdjustForInodes_C",
                                     "MatInodeAdjustForInodes_Inode",
                                      MatInodeAdjustForInodes_Inode);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatInodeGetInodeSizes_C",
                                     "MatInodeGetInodeSizes_Inode",
                                      MatInodeGetInodeSizes_Inode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_Inode"
PetscErrorCode MatSetOption_Inode(Mat A,MatOption op)
{
  Mat_inode *a=(Mat_inode*)A->data;
  PetscFunctionBegin;
  switch(op) {
    case MAT_USE_INODES:
      a->inode.use         = PETSC_TRUE;
      break;
    case MAT_DO_NOT_USE_INODES:
      a->inode.use         = PETSC_FALSE;
      break;
    case MAT_INODE_LIMIT_1:
      a->inode.limit  = 1;
      break;
    case MAT_INODE_LIMIT_2:
      a->inode.limit  = 2;
      break;
    case MAT_INODE_LIMIT_3:
      a->inode.limit  = 3;
      break;
    case MAT_INODE_LIMIT_4:
      a->inode.limit  = 4;
      break;
    case MAT_INODE_LIMIT_5:
      a->inode.limit  = 5;
      break;
    default:
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPrintHelp_Inode"
PetscErrorCode MatPrintHelp_Inode(Mat A)
{
  static PetscTruth called=PETSC_FALSE;
  MPI_Comm          comm=A->comm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!called) {
    called = PETSC_TRUE;
    ierr = (*PetscHelpPrintf)(comm," Inode related options (the defaults):\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -mat_inode_limit <limit>: Set inode limit (max limit=5)\n");CHKERRQ(ierr);
    ierr = (*PetscHelpPrintf)(comm,"  -mat_no_inode: Do not use inodes\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_Inode"
PetscErrorCode MatDuplicate_Inode(Mat A,MatDuplicateOption cpvalues,Mat *C)
{
  Mat            B=*C;
  Mat_inode      *c=(Mat_inode*)B->data,*a=(Mat_inode*)A->data;
  PetscErrorCode ierr;
  PetscInt       m=A->m;

  PetscFunctionBegin;

  c->inode.use          = a->inode.use;
  c->inode.limit        = a->inode.limit;
  c->inode.max_limit    = a->inode.max_limit;
  if (a->inode.size){
    ierr                = PetscMalloc((m+1)*sizeof(PetscInt),&c->inode.size);CHKERRQ(ierr);
    c->inode.node_count = a->inode.node_count;
    ierr                = PetscMemcpy(c->inode.size,a->inode.size,(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    c->inode.size       = 0;
    c->inode.node_count = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatILUDTFactor_Inode"
PetscErrorCode MatILUDTFactor_Inode(Mat A,IS isrow,IS iscol,MatFactorInfo *info,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
    /* check for identical nodes. If found, use inode functions */
  ierr = Mat_CheckInode(*fact,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_Inode"
PetscErrorCode MatLUFactorSymbolic_Inode(Mat A,IS isrow,IS iscol,MatFactorInfo *info,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
    /* check for identical nodes. If found, use inode functions */
  ierr = Mat_CheckInode(*fact,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatILUFactorSymbolic_Inode"
PetscErrorCode MatILUFactorSymbolic_Inode(Mat A,IS isrow,IS iscol,MatFactorInfo *info,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
    /* check for identical nodes. If found, use inode functions */
  ierr = Mat_CheckInode(*fact,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


