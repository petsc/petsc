
#include <../src/mat/impls/aij/seq/aij.h>

extern PetscErrorCode MatInodeAdjustForInodes_SeqAIJ_Inode(Mat,IS*,IS*);
extern PetscErrorCode MatInodeGetInodeSizes_SeqAIJ_Inode(Mat,PetscInt*,PetscInt*[],PetscInt*);

PetscErrorCode MatView_SeqAIJ_Inode(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ        *a=(Mat_SeqAIJ*)A->data;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL || format == PETSC_VIEWER_ASCII_INFO) {
      if (a->inode.size) {
        PetscCall(PetscViewerASCIIPrintf(viewer,"using I-node routines: found %" PetscInt_FMT " nodes, limit used is %" PetscInt_FMT "\n",a->inode.node_count,a->inode.limit));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"not using I-node routines\n"));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqAIJ_Inode(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJCheckInode(A));
  a->inode.ibdiagvalid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJ_Inode(Mat A)
{
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(a->inode.size));
  PetscCall(PetscFree3(a->inode.ibdiag,a->inode.bdiag,a->inode.ssor_work));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatInodeAdjustForInodes_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatInodeGetInodeSizes_C",NULL));
  PetscFunctionReturn(0);
}

/* MatCreate_SeqAIJ_Inode is not DLLEXPORTed because it is not a constructor for a complete type.    */
/* It is also not registered as a type for use within MatSetType.                             */
/* It is intended as a helper for the MATSEQAIJ class, so classes which desire Inodes should  */
/*    inherit off of MATSEQAIJ instead by calling MatSetType(MATSEQAIJ) in their constructor. */
/* Maybe this is a bad idea. (?) */
PetscErrorCode MatCreate_SeqAIJ_Inode(Mat B)
{
  Mat_SeqAIJ     *b=(Mat_SeqAIJ*)B->data;
  PetscBool      no_inode,no_unroll;

  PetscFunctionBegin;
  no_inode             = PETSC_FALSE;
  no_unroll            = PETSC_FALSE;
  b->inode.checked     = PETSC_FALSE;
  b->inode.node_count  = 0;
  b->inode.size        = NULL;
  b->inode.limit       = 5;
  b->inode.max_limit   = 5;
  b->inode.ibdiagvalid = PETSC_FALSE;
  b->inode.ibdiag      = NULL;
  b->inode.bdiag       = NULL;

  PetscOptionsBegin(PetscObjectComm((PetscObject)B),((PetscObject)B)->prefix,"Options for SEQAIJ matrix","Mat");
  PetscCall(PetscOptionsBool("-mat_no_unroll","Do not optimize for inodes (slower)",NULL,no_unroll,&no_unroll,NULL));
  if (no_unroll) {
    PetscCall(PetscInfo(B,"Not using Inode routines due to -mat_no_unroll\n"));
  }
  PetscCall(PetscOptionsBool("-mat_no_inode","Do not optimize for inodes -slower-",NULL,no_inode,&no_inode,NULL));
  if (no_inode) {
    PetscCall(PetscInfo(B,"Not using Inode routines due to -mat_no_inode\n"));
  }
  PetscCall(PetscOptionsInt("-mat_inode_limit","Do not use inodes larger then this value",NULL,b->inode.limit,&b->inode.limit,NULL));
  PetscOptionsEnd();

  b->inode.use = (PetscBool)(!(no_unroll || no_inode));
  if (b->inode.limit > b->inode.max_limit) b->inode.limit = b->inode.max_limit;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatInodeAdjustForInodes_C",MatInodeAdjustForInodes_SeqAIJ_Inode));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatInodeGetInodeSizes_C",MatInodeGetInodeSizes_SeqAIJ_Inode));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqAIJ_Inode(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqAIJ *a=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_USE_INODES:
    a->inode.use = flg;
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
