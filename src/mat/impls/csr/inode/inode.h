#if !defined(__INODE_H)
#define __INODE_H

#include "src/mat/impls/csr/csr.h"

/* Info about i-nodes (identical nodes) helper class */
typedef struct {
  PetscTruth use;
  PetscInt   node_count;                    /* number of inodes */
  PetscInt   *size;                         /* size of each inode */
  PetscInt   limit;                         /* inode limit */
  PetscInt   max_limit;                     /* maximum supported inode limit */
  PetscTruth checked;                       /* if inodes have been checked for */
} Mat_Inode;

#define MAT_INODE_HEADER   \
  MAT_CSR_HEADER;          \
  Mat_Inode  inode

typedef struct {
  MAT_INODE_HEADER;
} Mat_inode;

EXTERN PetscErrorCode MatView_Inode(Mat,PetscViewer);
EXTERN PetscErrorCode MatAssemblyEnd_Inode(Mat,MatAssemblyType);
EXTERN PetscErrorCode MatDestroy_Inode(Mat);
EXTERN PetscErrorCode MatCreate_Inode(Mat);
EXTERN PetscErrorCode MatSetOption_Inode(Mat,MatOption);
EXTERN PetscErrorCode MatPrintHelp_Inode(Mat);
EXTERN PetscErrorCode MatDuplicate_Inode(Mat A,MatDuplicateOption cpvalues,Mat *B);
EXTERN PetscErrorCode MatILUDTFactor_Inode(Mat A,IS isrow,IS iscol,MatFactorInfo *info,Mat *fact);
EXTERN PetscErrorCode MatLUFactorSymbolic_Inode(Mat A,IS isrow,IS iscol,MatFactorInfo *info,Mat *fact);
EXTERN PetscErrorCode MatILUFactorSymbolic_Inode(Mat A,IS isrow,IS iscol,MatFactorInfo *info,Mat *fact);

#endif
