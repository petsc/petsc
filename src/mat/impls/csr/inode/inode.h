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


#endif
