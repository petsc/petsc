#define PETSCMAT_DLL
#include "../src/mat/impls/aij/seq/aij.h"

EXTERN PetscErrorCode Mat_CheckInode(Mat,PetscTruth);
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatInodeAdjustForInodes_SeqAIJ_Inode(Mat,IS*,IS*);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatInodeGetInodeSizes_SeqAIJ_Inode(Mat,PetscInt*,PetscInt*[],PetscInt*);
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqAIJ_Inode"
PetscErrorCode MatView_SeqAIJ_Inode(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ         *a=(Mat_SeqAIJ*)A->data;
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
#define __FUNCT__ "MatAssemblyEnd_SeqAIJ_Inode"
PetscErrorCode MatAssemblyEnd_SeqAIJ_Inode(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscTruth     samestructure;

  PetscFunctionBegin;
  /* info.nz_unneeded of zero denotes no structural change was made to the matrix during Assembly */
  samestructure = (PetscTruth)(!A->info.nz_unneeded);
  /* check for identical nodes. If found, use inode functions */
  ierr = Mat_CheckInode(A,samestructure);CHKERRQ(ierr);
  a->inode.ibdiagvalid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJ_Inode"
PetscErrorCode MatDestroy_SeqAIJ_Inode(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  ierr = PetscFree(a->inode.size);CHKERRQ(ierr);
  ierr = PetscFree3(a->inode.ibdiag,a->inode.bdiag,a->inode.ssor_work);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatInodeAdjustForInodes_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatInodeGetInodeSizes_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* MatCreate_SeqAIJ_Inode is not DLLEXPORTed because it is not a constructor for a complete type.    */
/* It is also not registered as a type for use within MatSetType.                             */
/* It is intended as a helper for the MATSEQAIJ class, so classes which desire Inodes should  */
/*    inherit off of MATSEQAIJ instead by calling MatSetType(MATSEQAIJ) in their constructor. */
/* Maybe this is a bad idea. (?) */
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJ_Inode"
PetscErrorCode MatCreate_SeqAIJ_Inode(Mat B)
{
  Mat_SeqAIJ     *b=(Mat_SeqAIJ*)B->data;
  PetscErrorCode ierr;
  PetscTruth     no_inode,no_unroll;

  PetscFunctionBegin;
  no_inode             = PETSC_FALSE;
  no_unroll            = PETSC_FALSE;
  b->inode.node_count  = 0;
  b->inode.size        = 0;
  b->inode.limit       = 5;
  b->inode.max_limit   = 5;
  b->inode.ibdiagvalid = PETSC_FALSE;
  b->inode.ibdiag      = 0;
  b->inode.bdiag       = 0;

  ierr = PetscOptionsBegin(((PetscObject)B)->comm,((PetscObject)B)->prefix,"Options for SEQAIJ matrix","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-mat_no_unroll","Do not optimize for inodes (slower)",PETSC_NULL,no_unroll,&no_unroll,PETSC_NULL);CHKERRQ(ierr);
    if (no_unroll) {ierr = PetscInfo(B,"Not using Inode routines due to -mat_no_unroll\n");CHKERRQ(ierr);}
    ierr = PetscOptionsTruth("-mat_no_inode","Do not optimize for inodes (slower)",PETSC_NULL,no_inode,&no_inode,PETSC_NULL);CHKERRQ(ierr);
    if (no_inode) {ierr = PetscInfo(B,"Not using Inode routines due to -mat_no_inode\n");CHKERRQ(ierr);}
    ierr = PetscOptionsInt("-mat_inode_limit","Do not use inodes larger then this value",PETSC_NULL,b->inode.limit,&b->inode.limit,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  b->inode.use = (PetscTruth)(!(no_unroll || no_inode));
  if (b->inode.limit > b->inode.max_limit) b->inode.limit = b->inode.max_limit;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatInodeAdjustForInodes_C",
                                     "MatInodeAdjustForInodes_SeqAIJ_Inode",
                                      MatInodeAdjustForInodes_SeqAIJ_Inode);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatInodeGetInodeSizes_C",
                                     "MatInodeGetInodeSizes_SeqAIJ_Inode",
                                      MatInodeGetInodeSizes_SeqAIJ_Inode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_SeqAIJ_Inode"
PetscErrorCode MatSetOption_SeqAIJ_Inode(Mat A,MatOption op,PetscTruth flg)
{
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  switch(op) {
    case MAT_USE_INODES:
      a->inode.use         = flg;
      break;
    default:
      break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_SeqAIJ_Inode"
PetscErrorCode MatDuplicate_SeqAIJ_Inode(Mat A,MatDuplicateOption cpvalues,Mat *C)
{
  Mat            B=*C;
  Mat_SeqAIJ     *c=(Mat_SeqAIJ*)B->data,*a=(Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n;

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
  c->inode.ibdiagvalid = PETSC_FALSE;
  c->inode.ibdiag      = 0;
  c->inode.bdiag       = 0;
  PetscFunctionReturn(0);
}



