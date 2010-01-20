#define PETSCVEC_DLL
/*
     Provides the functions for index sets (IS) defined by a list of integers.
   These are for blocks of data, each block is indicated with a single integer.
*/
#include "private/isimpl.h"               /*I  "petscis.h"     I*/
#include "petscvec.h"

typedef struct {
  PetscInt        N,n;            /* number of blocks */
  PetscTruth      sorted;       /* are the blocks sorted? */
  PetscInt        *idx;
  PetscInt        bs;           /* blocksize */
} IS_Block;

#undef __FUNCT__  
#define __FUNCT__ "ISDestroy_Block" 
PetscErrorCode ISDestroy_Block(IS is)
{
  IS_Block       *is_block = (IS_Block*)is->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(is_block->idx);CHKERRQ(ierr);
  ierr = PetscFree(is_block);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetIndices_Block" 
PetscErrorCode ISGetIndices_Block(IS in,const PetscInt *idx[])
{
  IS_Block       *sub = (IS_Block*)in->data;
  PetscErrorCode ierr;
  PetscInt       i,j,k,bs = sub->bs,n = sub->n,*ii,*jj;

  PetscFunctionBegin;
  if (sub->bs == 1) {
    *idx = sub->idx; 
  } else {
    ierr = PetscMalloc(sub->bs*sub->n*sizeof(PetscInt),&jj);CHKERRQ(ierr);
    *idx = jj;
    k    = 0;
    ii   = sub->idx;
    for (i=0; i<n; i++) {
      for (j=0; j<bs; j++) {
        jj[k++] = ii[i] + j;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISRestoreIndices_Block" 
PetscErrorCode ISRestoreIndices_Block(IS in,const PetscInt *idx[])
{
  IS_Block       *sub = (IS_Block*)in->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sub->bs != 1) {
    ierr = PetscFree(*(void**)idx);CHKERRQ(ierr);
  } else {
    if (*idx !=  sub->idx) {
      SETERRQ(PETSC_ERR_ARG_WRONG,"Must restore with value from ISGetIndices()");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetSize_Block" 
PetscErrorCode ISGetSize_Block(IS is,PetscInt *size)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *size = sub->bs*sub->N; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetLocalSize_Block" 
PetscErrorCode ISGetLocalSize_Block(IS is,PetscInt *size)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *size = sub->bs*sub->n; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISInvertPermutation_Block" 
PetscErrorCode ISInvertPermutation_Block(IS is,PetscInt nlocal,IS *isout)
{
  IS_Block       *sub = (IS_Block *)is->data;
  PetscInt       i,*ii,n = sub->n,*idx = sub->idx;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)is)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscMalloc(n*sizeof(PetscInt),&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ii[idx[i]] = i;
    }
    ierr = ISCreateBlock(PETSC_COMM_SELF,sub->bs,n,ii,isout);CHKERRQ(ierr);
    ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
    ierr = PetscFree(ii);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"No inversion written yet for block IS");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISView_Block" 
PetscErrorCode ISView_Block(IS is, PetscViewer viewer)
{
  IS_Block       *sub = (IS_Block *)is->data;
  PetscErrorCode ierr;
  PetscInt       i,n = sub->n,*idx = sub->idx;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) { 
    if (is->isperm) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block Index set is permutation\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block size %D\n",sub->bs);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Number of block indices in set %D\n",n);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"The first indices of each block are\n");CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block %D Index %D\n",i,idx[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSort_Block" 
PetscErrorCode ISSort_Block(IS is)
{
  IS_Block       *sub = (IS_Block *)is->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sub->sorted) PetscFunctionReturn(0);
  ierr = PetscSortInt(sub->n,sub->idx);CHKERRQ(ierr);
  sub->sorted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSorted_Block" 
PetscErrorCode ISSorted_Block(IS is,PetscTruth *flg)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *flg = sub->sorted;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISDuplicate_Block" 
PetscErrorCode ISDuplicate_Block(IS is,IS *newIS)
{
  PetscErrorCode ierr;
  IS_Block       *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  ierr = ISCreateBlock(((PetscObject)is)->comm,sub->bs,sub->n,sub->idx,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISIdentity_Block" 
PetscErrorCode ISIdentity_Block(IS is,PetscTruth *ident)
{
  IS_Block *is_block = (IS_Block*)is->data;
  PetscInt i,n = is_block->n,*idx = is_block->idx,bs = is_block->bs;

  PetscFunctionBegin;
  is->isidentity = PETSC_TRUE;
  *ident         = PETSC_TRUE;
  for (i=0; i<n; i++) {
    if (idx[i] != bs*i) {
      is->isidentity = PETSC_FALSE;
      *ident         = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISCopy_Block"
static PetscErrorCode ISCopy_Block(IS is,IS isy)
{
  IS_Block *is_block = (IS_Block*)is->data,*isy_block = (IS_Block*)isy->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is_block->n != isy_block->n || is_block->N != isy_block->N || is_block->bs != isy_block->bs) SETERRQ(PETSC_ERR_ARG_INCOMP,"Index sets incompatible");
  isy_block->sorted = is_block->sorted;
  ierr = PetscMemcpy(isy_block->idx,is_block->idx,is_block->n*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _ISOps myops = { ISGetSize_Block,
                               ISGetLocalSize_Block,
                               ISGetIndices_Block,
                               ISRestoreIndices_Block,
                               ISInvertPermutation_Block,
                               ISSort_Block,
                               ISSorted_Block,
                               ISDuplicate_Block,
                               ISDestroy_Block,
                               ISView_Block,
                               ISIdentity_Block,
                               ISCopy_Block };
#undef __FUNCT__  
#define __FUNCT__ "ISCreateBlock" 
/*@
   ISCreateBlock - Creates a data structure for an index set containing
   a list of integers. The indices are relative to entries, not blocks. 

   Collective on MPI_Comm

   Input Parameters:
+  n - the length of the index set (the number of blocks)
.  bs - number of elements in each block
.  idx - the list of integers
-  comm - the MPI communicator

   Output Parameter:
.  is - the new index set

   Notes:
   When the communicator is not MPI_COMM_SELF, the operations on the 
   index sets, IS, are NOT conceptually the same as MPI_Group operations. 
   The index sets are then distributed sets of indices and thus certain operations
   on them are collective. 

   Example:
   If you wish to index the values {0,1,4,5}, then use
   a block size of 2 and idx of {0,4}.

   Level: beginner

  Concepts: IS^block
  Concepts: index sets^block
  Concepts: block^index set

.seealso: ISCreateStride(), ISCreateGeneral(), ISAllGather()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISCreateBlock(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscInt idx[],IS *is)
{
  PetscErrorCode ierr;
  PetscInt       i,min,max;
  IS             Nindex;
  IS_Block       *sub;
  PetscTruth     sorted = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidPointer(is,5);
  if (n < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) {PetscValidIntPointer(idx,4);}
  *is = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = ISInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(Nindex,_p_IS,struct _ISOps,IS_COOKIE,IS_BLOCK,"IS",comm,ISDestroy,ISView);CHKERRQ(ierr);
  ierr = PetscNewLog(Nindex,IS_Block,&sub);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&sub->idx);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(Nindex,n*sizeof(PetscInt));CHKERRQ(ierr);
  sub->n = n;
  ierr = MPI_Allreduce(&n,&sub->N,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  for (i=1; i<n; i++) {
    if (idx[i] < idx[i-1]) {sorted = PETSC_FALSE; break;}
  }
  if (n) {min = max = idx[0];} else {min = max = 0;}
  for (i=1; i<n; i++) {
    if (idx[i] < min) min = idx[i];
    if (idx[i] > max) max = idx[i];
  }
  ierr = PetscMemcpy(sub->idx,idx,n*sizeof(PetscInt));CHKERRQ(ierr);
  sub->sorted     = sorted;
  sub->bs         = bs;
  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void*)sub;
  ierr = PetscMemcpy(Nindex->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  Nindex->isperm  = PETSC_FALSE;
  *is = Nindex; PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISBlockGetIndices" 
/*@C
   ISBlockGetIndices - Gets the indices associated with each block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  idx - the integer indices

   Level: intermediate

   Concepts: IS^block
   Concepts: index sets^getting indices
   Concepts: index sets^block

.seealso: ISGetIndices(), ISBlockRestoreIndices()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISBlockGetIndices(IS in,const PetscInt *idx[])
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(in,IS_COOKIE,1);
  PetscValidPointer(idx,2);
  if (((PetscObject)in)->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a block index set");

  sub = (IS_Block*)in->data;
  *idx = sub->idx; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISBlockRestoreIndices" 
/*@C
   ISBlockRestoreIndices - Restores the indices associated with each block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  idx - the integer indices

   Level: intermediate

   Concepts: IS^block
   Concepts: index sets^getting indices
   Concepts: index sets^block

.seealso: ISRestoreIndices(), ISBlockGetIndices()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISBlockRestoreIndices(IS is,const PetscInt *idx[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(idx,2);
  if (((PetscObject)is)->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a block index set");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISBlockGetBlockSize" 
/*@
   ISBlockGetBlockSize - Returns the number of elements in a block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the number of elements in a block

   Level: intermediate

   Concepts: IS^block size
   Concepts: index sets^block size

.seealso: ISBlockGetSize(), ISGetSize(), ISBlock(), ISCreateBlock()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISBlockGetBlockSize(IS is,PetscInt *size)
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(size,2);
  if (((PetscObject)is)->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a block index set");

  sub = (IS_Block *)is->data;
  *size = sub->bs; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISBlock" 
/*@
   ISBlock - Checks whether an index set is blocked.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  flag - PETSC_TRUE if a block index set, else PETSC_FALSE

   Level: intermediate

   Concepts: IS^block
   Concepts: index sets^block

.seealso: ISBlockGetSize(), ISGetSize(), ISBlockGetBlockSize(), ISCreateBlock()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISBlock(IS is,PetscTruth *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(flag,2);
  if (((PetscObject)is)->type != IS_BLOCK) *flag = PETSC_FALSE;
  else                          *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISBlockGetLocalSize" 
/*@
   ISBlockGetLocalSize - Returns the local number of blocks in the index set.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the local number of blocks

   Level: intermediate

   Concepts: IS^block sizes
   Concepts: index sets^block sizes

.seealso: ISBlockGetBlockSize(), ISBlockGetSize(), ISGetSize(), ISBlock(), ISCreateBlock()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISBlockGetLocalSize(IS is,PetscInt *size)
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(size,2);
  if (((PetscObject)is)->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a block index set");

  sub = (IS_Block *)is->data;
  *size = sub->n; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISBlockGetSize" 
/*@
   ISBlockGetSize - Returns the global number of blocks in the index set.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the global number of blocks

   Level: intermediate

   Concepts: IS^block sizes
   Concepts: index sets^block sizes

.seealso: ISBlockGetBlockSize(), ISBlockGetLocalSize(), ISGetSize(), ISBlock(), ISCreateBlock()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISBlockGetSize(IS is,PetscInt *size)
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(size,2);
  if (((PetscObject)is)->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a block index set");

  sub = (IS_Block *)is->data;
  *size = sub->N;
  PetscFunctionReturn(0);
}
