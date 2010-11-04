#if !defined(__MATDD_H)
#define __MATDD_H

#include "private/matimpl.h"
#include "../src/dm/dd/vecdd/ddlayout.h"

typedef struct {
  Mat mat;
} MatDD_Block;

#define MatDD_BlockGetMat(A,i,j,_block,_blockmat) 0; *_blockmat = _block->mat
#define MatDD_BlockSetMat(A,i,j,_block,blockmat) 0; _block->mat = blockmat
struct _MatDDOps {
  PetscErrorCode (*locateblock)(Mat M, PetscInt i, PetscInt j, PetscBool insert, MatDD_Block **block);
};

typedef struct {
  struct _MatDDOps *ops;
  DDLayout rmapdd, cmapdd;
  Vec outvec, invec;
  PetscBool setup;
  Mat scatter, gather;
  const MatType    default_block_type;
} Mat_DD; 


#endif
