
/*
     Provides the functions for index sets (IS) defined by a list of integers.
   These are for blocks of data, each block is indicated with a single integer.
*/
#include <petsc-private/isimpl.h>               /*I  "petscis.h"     I*/
#include <petscvec.h>

typedef struct {
  PetscInt        N,n;            /* number of blocks */
  PetscBool       sorted;       /* are the blocks sorted? */
  PetscInt        *idx;
} IS_Block;

#undef __FUNCT__
#define __FUNCT__ "ISDestroy_Block"
PetscErrorCode ISDestroy_Block(IS is)
{
  IS_Block       *is_block = (IS_Block*)is->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(is_block->idx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockSetIndices_C","",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockGetIndices_C","",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockRestoreIndices_C","",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockGetSize_C","",0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockGetLocalSize_C","",0);CHKERRQ(ierr);
  ierr = PetscFree(is->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISGetIndices_Block"
PetscErrorCode ISGetIndices_Block(IS in,const PetscInt *idx[])
{
  IS_Block       *sub = (IS_Block*)in->data;
  PetscErrorCode ierr;
  PetscInt       i,j,k,bs = in->bs,n = sub->n,*ii,*jj;

  PetscFunctionBegin;
  if (bs == 1) {
    *idx = sub->idx;
  } else {
    ierr = PetscMalloc(bs*n*sizeof(PetscInt),&jj);CHKERRQ(ierr);
    *idx = jj;
    k    = 0;
    ii   = sub->idx;
    for (i=0; i<n; i++) {
      for (j=0; j<bs; j++) {
        jj[k++] = bs*ii[i] + j;
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
  if (in->bs != 1) {
    ierr = PetscFree(*(void**)idx);CHKERRQ(ierr);
  } else {
    if (*idx !=  sub->idx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from ISGetIndices()");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISGetSize_Block"
PetscErrorCode ISGetSize_Block(IS is,PetscInt *size)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *size = is->bs*sub->N;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISGetLocalSize_Block"
PetscErrorCode ISGetLocalSize_Block(IS is,PetscInt *size)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *size = is->bs*sub->n;
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
    ierr = ISCreateBlock(PETSC_COMM_SELF,is->bs,n,ii,PETSC_OWN_POINTER,isout);CHKERRQ(ierr);
    ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No inversion written yet for block IS");
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
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    if (is->isperm) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block Index set is permutation\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block size %D\n",is->bs);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Number of block indices in set %D\n",n);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"The first indices of each block are\n");CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block %D Index %D\n",i,idx[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
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
PetscErrorCode ISSorted_Block(IS is,PetscBool  *flg)
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
  ierr = ISCreateBlock(((PetscObject)is)->comm,is->bs,sub->n,sub->idx,PETSC_COPY_VALUES,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISIdentity_Block"
PetscErrorCode ISIdentity_Block(IS is,PetscBool  *ident)
{
  IS_Block *is_block = (IS_Block*)is->data;
  PetscInt i,n = is_block->n,*idx = is_block->idx,bs = is->bs;

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
  if (is_block->n != isy_block->n || is_block->N != isy_block->N || is->bs != isy->bs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Index sets incompatible");
  isy_block->sorted = is_block->sorted;
  ierr = PetscMemcpy(isy_block->idx,is_block->idx,is_block->n*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISOnComm_Block"
static PetscErrorCode ISOnComm_Block(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  PetscErrorCode ierr;
  IS_Block       *sub = (IS_Block*)is->data;

  PetscFunctionBegin;
  if (mode == PETSC_OWN_POINTER) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Cannot use PETSC_OWN_POINTER");
  ierr = ISCreateBlock(comm,is->bs,sub->n,sub->idx,mode,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISSetBlockSize_Block"
static PetscErrorCode ISSetBlockSize_Block(IS is,PetscInt bs)
{

  PetscFunctionBegin;
  if (is->bs != bs) SETERRQ2(((PetscObject)is)->comm,PETSC_ERR_ARG_SIZ,"Cannot change block size for ISBLOCK from %D to %D",is->bs,bs);
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
                               ISCopy_Block,
                               0,
                               ISOnComm_Block,
                               ISSetBlockSize_Block
};

#undef __FUNCT__
#define __FUNCT__ "ISBlockSetIndices"
/*@
   ISBlockSetIndices - The indices are relative to entries, not blocks.

   Collective on IS

   Input Parameters:
+  is - the index set
.  bs - number of elements in each block, one for each block and count of block not indices
.   n - the length of the index set (the number of blocks)
.  idx - the list of integers, these are by block, not by location
+  mode - see PetscCopyMode, only PETSC_COPY_VALUES and PETSC_OWN_POINTER are supported


   Notes:
   When the communicator is not MPI_COMM_SELF, the operations on the
   index sets, IS, are NOT conceptually the same as MPI_Group operations.
   The index sets are then distributed sets of indices and thus certain operations
   on them are collective.

   Example:
   If you wish to index the values {0,1,4,5}, then use
   a block size of 2 and idx of {0,2}.

   Level: beginner

  Concepts: IS^block
  Concepts: index sets^block
  Concepts: block^index set

.seealso: ISCreateStride(), ISCreateGeneral(), ISAllGather()
@*/
PetscErrorCode  ISBlockSetIndices(IS is,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISBlockSetIndices_C",(IS,PetscInt,PetscInt,const PetscInt[],PetscCopyMode),(is,bs,n,idx,mode));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "ISBlockSetIndices_Block"
PetscErrorCode  ISBlockSetIndices_Block(IS is,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscErrorCode ierr;
  PetscInt       i,min,max;
  IS_Block       *sub = (IS_Block*)is->data;
  PetscBool      sorted = PETSC_TRUE;

  PetscFunctionBegin;
  ierr = PetscFree(sub->idx);CHKERRQ(ierr);
  sub->n = n;
  ierr = MPI_Allreduce(&n,&sub->N,1,MPIU_INT,MPI_SUM,((PetscObject)is)->comm);CHKERRQ(ierr);
  for (i=1; i<n; i++) {
    if (idx[i] < idx[i-1]) {sorted = PETSC_FALSE; break;}
  }
  if (n) {min = max = idx[0];} else {min = max = 0;}
  for (i=1; i<n; i++) {
    if (idx[i] < min) min = idx[i];
    if (idx[i] > max) max = idx[i];
  }
  if (mode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc(n*sizeof(PetscInt),&sub->idx);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(is,n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(sub->idx,idx,n*sizeof(PetscInt));CHKERRQ(ierr);
  } else if (mode == PETSC_OWN_POINTER) {
    sub->idx = (PetscInt*) idx;
  } else SETERRQ(((PetscObject)is)->comm,PETSC_ERR_SUP,"Only supports PETSC_COPY_VALUES and PETSC_OWN_POINTER");
  sub->sorted = sorted;
  is->bs      = bs;
  is->min     = bs*min;
  is->max     = bs*max+bs-1;
  is->data    = (void*)sub;
  ierr = PetscMemcpy(is->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  is->isperm  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "ISCreateBlock"
/*@
   ISCreateBlock - Creates a data structure for an index set containing
   a list of integers. The indices are relative to entries, not blocks.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator
.  bs - number of elements in each block
.  n - the length of the index set (the number of blocks)
.  idx - the list of integers, one for each block and count of block not indices
-  mode - see PetscCopyMode, only PETSC_COPY_VALUES and PETSC_OWN_POINTER are supported in this routine

   Output Parameter:
.  is - the new index set

   Notes:
   When the communicator is not MPI_COMM_SELF, the operations on the
   index sets, IS, are NOT conceptually the same as MPI_Group operations.
   The index sets are then distributed sets of indices and thus certain operations
   on them are collective.

   Example:
   If you wish to index the values {0,1,6,7}, then use
   a block size of 2 and idx of {0,3}.

   Level: beginner

  Concepts: IS^block
  Concepts: index sets^block
  Concepts: block^index set

.seealso: ISCreateStride(), ISCreateGeneral(), ISAllGather()
@*/
PetscErrorCode  ISCreateBlock(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(is,5);
  if (n < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) {PetscValidIntPointer(idx,4);}

  ierr = ISCreate(comm,is);CHKERRQ(ierr);
  ierr = ISSetType(*is,ISBLOCK);CHKERRQ(ierr);
  ierr = ISBlockSetIndices(*is,bs,n,idx,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "ISBlockGetIndices_Block"
PetscErrorCode  ISBlockGetIndices_Block(IS is,const PetscInt *idx[])
{
  IS_Block       *sub = (IS_Block*)is->data;

  PetscFunctionBegin;
  *idx = sub->idx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "ISBlockRestoreIndices_Block"
PetscErrorCode  ISBlockRestoreIndices_Block(IS is,const PetscInt *idx[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "ISBlockGetIndices"
/*@C
   ISBlockGetIndices - Gets the indices associated with each block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  idx - the integer indices, one for each block and count of block not indices

   Level: intermediate

   Concepts: IS^block
   Concepts: index sets^getting indices
   Concepts: index sets^block

.seealso: ISGetIndices(), ISBlockRestoreIndices()
@*/
PetscErrorCode  ISBlockGetIndices(IS is,const PetscInt *idx[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISBlockGetIndices_C",(IS,const PetscInt*[]),(is,idx));CHKERRQ(ierr);
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
PetscErrorCode  ISBlockRestoreIndices(IS is,const PetscInt *idx[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISBlockRestoreIndices_C",(IS,const PetscInt*[]),(is,idx));CHKERRQ(ierr);
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

.seealso: ISGetBlockSize(), ISBlockGetSize(), ISGetSize(), ISCreateBlock()
@*/
PetscErrorCode  ISBlockGetLocalSize(IS is,PetscInt *size)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISBlockGetLocalSize_C",(IS,PetscInt*),(is,size));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "ISBlockGetLocalSize_Block"
PetscErrorCode  ISBlockGetLocalSize_Block(IS is,PetscInt *size)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *size = sub->n;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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

.seealso: ISGetBlockSize(), ISBlockGetLocalSize(), ISGetSize(), ISCreateBlock()
@*/
PetscErrorCode  ISBlockGetSize(IS is,PetscInt *size)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISBlockGetSize_C",(IS,PetscInt*),(is,size));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "ISBlockGetSize_Block"
PetscErrorCode  ISBlockGetSize_Block(IS is,PetscInt *size)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *size = sub->N;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "ISToGeneral_Block"
PetscErrorCode  ISToGeneral_Block(IS inis)
{
  PetscErrorCode ierr;
  const PetscInt *idx;
  PetscInt       n;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(inis,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(inis,&idx);CHKERRQ(ierr);
  ierr = ISSetType(inis,ISGENERAL);CHKERRQ(ierr);
  ierr = ISGeneralSetIndices(inis,n,idx,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "ISCreate_Block"
PetscErrorCode  ISCreate_Block(IS is)
{
  PetscErrorCode ierr;
  IS_Block       *sub;

  PetscFunctionBegin;
  ierr = PetscNewLog(is,IS_Block,&sub);CHKERRQ(ierr);
  is->data = sub;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockSetIndices_C","ISBlockSetIndices_Block",
					   ISBlockSetIndices_Block);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockGetIndices_C","ISBlockGetIndices_Block",
					   ISBlockGetIndices_Block);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockRestoreIndices_C","ISBlockRestoreIndices_Block",
					   ISBlockRestoreIndices_Block);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockGetSize_C","ISBlockGetSize_Block",
					   ISBlockGetSize_Block);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISBlockGetLocalSize_C","ISBlockGetLocalSize_Block",
					   ISBlockGetLocalSize_Block);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
