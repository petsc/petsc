
/*
     Provides the functions for index sets (IS) defined by a list of integers.
   These are for blocks of data, each block is indicated with a single integer.
*/
#include <petsc/private/isimpl.h>               /*I  "petscis.h"     I*/
#include <petscviewer.h>

typedef struct {
  PetscBool sorted;      /* are the blocks sorted? */
  PetscBool allocated;   /* did we allocate the index array ourselves? */
  PetscInt  *idx;
} IS_Block;

static PetscErrorCode ISDestroy_Block(IS is)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sub->allocated) {ierr = PetscFree(sub->idx);CHKERRQ(ierr);}
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockSetIndices_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockGetIndices_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockRestoreIndices_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockGetSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockGetLocalSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(is->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISLocate_Block(IS is,PetscInt key,PetscInt *location)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       numIdx, i, bs, bkey, mkey;
  PetscBool      sorted;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map,&bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(is->map,&numIdx);CHKERRQ(ierr);
  numIdx /= bs;
  bkey    = key / bs;
  mkey    = key % bs;
  if (mkey < 0) {
    bkey--;
    mkey += bs;
  }
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,&sorted);CHKERRQ(ierr);
  if (sorted) {
    ierr = PetscFindInt(bkey,numIdx,sub->idx,location);CHKERRQ(ierr);
  } else {
    const PetscInt *idx = sub->idx;

    *location = -1;
    for (i = 0; i < numIdx; i++) {
      if (idx[i] == bkey) {
        *location = i;
        break;
      }
    }
  }
  if (*location >= 0) {
    *location = *location * bs + mkey;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGetIndices_Block(IS in,const PetscInt *idx[])
{
  IS_Block       *sub = (IS_Block*)in->data;
  PetscErrorCode ierr;
  PetscInt       i,j,k,bs,n,*ii,*jj;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(in->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(in->map, &n);CHKERRQ(ierr);
  n   /= bs;
  if (bs == 1) *idx = sub->idx;
  else {
    ierr = PetscMalloc1(bs*n,&jj);CHKERRQ(ierr);
    *idx = jj;
    k    = 0;
    ii   = sub->idx;
    for (i=0; i<n; i++)
      for (j=0; j<bs; j++)
        jj[k++] = bs*ii[i] + j;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISRestoreIndices_Block(IS is,const PetscInt *idx[])
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  if (bs != 1) {
    ierr = PetscFree(*(void**)idx);CHKERRQ(ierr);
  } else {
    if (*idx != sub->idx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from ISGetIndices()");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISInvertPermutation_Block(IS is,PetscInt nlocal,IS *isout)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       i,*ii,bs,n,*idx = sub->idx;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)is),&size);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  n   /= bs;
  if (size == 1) {
    ierr = PetscMalloc1(n,&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) ii[idx[i]] = i;
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,n,ii,PETSC_OWN_POINTER,isout);CHKERRQ(ierr);
    ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No inversion written yet for block IS");
  PetscFunctionReturn(0);
}

static PetscErrorCode ISView_Block(IS is, PetscViewer viewer)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscErrorCode ierr;
  PetscInt       i,bs,n,*idx = sub->idx;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  n   /= bs;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerFormat fmt;

    ierr = PetscViewerGetFormat(viewer,&fmt);CHKERRQ(ierr);
    if (fmt == PETSC_VIEWER_ASCII_MATLAB) {
      IS             ist;
      const char     *name;
      const PetscInt *idx;
      PetscInt       n;

      ierr = PetscObjectGetName((PetscObject)is,&name);CHKERRQ(ierr);
      ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
      ierr = ISGetIndices(is,&idx);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is),n,idx,PETSC_USE_POINTER,&ist);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)ist,name);CHKERRQ(ierr);
      ierr = ISView(ist,viewer);CHKERRQ(ierr);
      ierr = ISDestroy(&ist);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is,&idx);CHKERRQ(ierr);
    } else {
      PetscBool isperm;

      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,&isperm);CHKERRQ(ierr);
      if (isperm) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block Index set is permutation\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block size %D\n",bs);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Number of block indices in set %D\n",n);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"The first indices of each block are\n");CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block %D Index %D\n",i,idx[i]);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSort_Block(IS is)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       bs, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscSortInt(n/bs,sub->idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSortRemoveDups_Block(IS is)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       bs, n, nb;
  PetscBool      sorted;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  nb   = n/bs;
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,&sorted);CHKERRQ(ierr);
  if (sorted) {
    ierr = PetscSortedRemoveDupsInt(&nb,sub->idx);CHKERRQ(ierr);
  } else {
    ierr = PetscSortRemoveDupsInt(&nb,sub->idx);CHKERRQ(ierr);
  }
  ierr = PetscLayoutDestroy(&is->map);CHKERRQ(ierr);
  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is), nb*bs, PETSC_DECIDE, bs, &is->map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSorted_Block(IS is,PetscBool  *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSortedLocal_Block(IS is,PetscBool *flg)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       n, bs, i, *idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  n   /= bs;
  idx  = sub->idx;
  for (i = 1; i < n; i++) if (idx[i] < (idx[i - 1] + bs - 1)) break;
  if (i < n) *flg = PETSC_FALSE;
  else       *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISUniqueLocal_Block(IS is,PetscBool *flg)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       n, bs, i, *idx, *idxcopy = NULL;
  PetscBool      sortedLocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  n   /= bs;
  idx  = sub->idx;
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,&sortedLocal);CHKERRQ(ierr);
  if (!sortedLocal) {
    ierr = PetscMalloc1(n, &idxcopy);CHKERRQ(ierr);
    ierr = PetscArraycpy(idxcopy, idx, n);CHKERRQ(ierr);
    ierr = PetscSortInt(n, idxcopy);CHKERRQ(ierr);
    idx = idxcopy;
  }
  for (i = 1; i < n; i++) if (idx[i] <= (idx[i - 1] + bs - 1)) break;
  if (i < n) *flg = PETSC_FALSE;
  else       *flg = PETSC_TRUE;
  ierr = PetscFree(idxcopy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISPermutationLocal_Block(IS is,PetscBool *flg)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       n, bs, i, *idx, *idxcopy = NULL;
  PetscBool      sortedLocal;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  n   /= bs;
  idx  = sub->idx;
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,&sortedLocal);CHKERRQ(ierr);
  if (!sortedLocal) {
    ierr = PetscMalloc1(n, &idxcopy);CHKERRQ(ierr);
    ierr = PetscArraycpy(idxcopy, idx, n);CHKERRQ(ierr);
    ierr = PetscSortInt(n, idxcopy);CHKERRQ(ierr);
    idx = idxcopy;
  }
  for (i = 0; i < n; i++) if (idx[i] != i*bs) break;
  if (i < n) *flg = PETSC_FALSE;
  else       *flg = PETSC_TRUE;
  ierr = PetscFree(idxcopy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISIntervalLocal_Block(IS is,PetscBool *flg)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       n, bs, i, *idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  n   /= bs;
  idx  = sub->idx;
  for (i = 1; i < n; i++) if (idx[i] != (idx[i - 1] + bs)) break;
  if (i < n) *flg = PETSC_FALSE;
  else       *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISDuplicate_Block(IS is,IS *newIS)
{
  PetscErrorCode ierr;
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt        bs, n;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  n   /= bs;
  ierr = ISCreateBlock(PetscObjectComm((PetscObject)is),bs,n,sub->idx,PETSC_COPY_VALUES,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISIdentity_Block(IS is,PetscBool  *ident)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetInfo(is,IS_IDENTITY,IS_GLOBAL,ident);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISCopy_Block(IS is,IS isy)
{
  IS_Block       *is_block = (IS_Block*)is->data,*isy_block = (IS_Block*)isy->data;
  PetscInt       bs, n, N, bsy, ny, Ny;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(is->map, &N);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(isy->map, &bsy);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(isy->map, &ny);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(isy->map, &Ny);CHKERRQ(ierr);
  if (n != ny || N != Ny || bs != bsy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Index sets incompatible");
  ierr = PetscArraycpy(isy_block->idx,is_block->idx,(n/bs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISOnComm_Block(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  PetscErrorCode ierr;
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       bs, n;

  PetscFunctionBegin;
  if (mode == PETSC_OWN_POINTER) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Cannot use PETSC_OWN_POINTER");
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = ISCreateBlock(comm,bs,n/bs,sub->idx,mode,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSetBlockSize_Block(IS is,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetBlockSize(is->map, bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISToGeneral_Block(IS inis)
{
  IS_Block       *sub   = (IS_Block*)inis->data;
  PetscInt       bs,n;
  const PetscInt *idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetBlockSize(inis,&bs);CHKERRQ(ierr);
  ierr = ISGetLocalSize(inis,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(inis,&idx);CHKERRQ(ierr);
  if (bs == 1) {
    PetscCopyMode mode = sub->allocated ? PETSC_OWN_POINTER : PETSC_USE_POINTER;
    sub->allocated = PETSC_FALSE; /* prevent deallocation when changing the subtype*/
    ierr = ISSetType(inis,ISGENERAL);CHKERRQ(ierr);
    ierr = ISGeneralSetIndices(inis,n,idx,mode);CHKERRQ(ierr);
  } else {
    ierr = ISSetType(inis,ISGENERAL);CHKERRQ(ierr);
    ierr = ISGeneralSetIndices(inis,n,idx,PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


static struct _ISOps myops = { ISGetIndices_Block,
                               ISRestoreIndices_Block,
                               ISInvertPermutation_Block,
                               ISSort_Block,
                               ISSortRemoveDups_Block,
                               ISSorted_Block,
                               ISDuplicate_Block,
                               ISDestroy_Block,
                               ISView_Block,
                               ISLoad_Default,
                               ISIdentity_Block,
                               ISCopy_Block,
                               ISToGeneral_Block,
                               ISOnComm_Block,
                               ISSetBlockSize_Block,
                               0,
                               ISLocate_Block,
                               /* we can have specialized local routines for determining properties,
                                * but unless the block size is the same on each process (which is not guaranteed at
                                * the moment), then trying to do something specialized for global properties is too
                                * complicated */
                               ISSortedLocal_Block,
                               NULL,
                               ISUniqueLocal_Block,
                               NULL,
                               ISPermutationLocal_Block,
                               NULL,
                               ISIntervalLocal_Block,
                               NULL};

/*@
   ISBlockSetIndices - The indices are relative to entries, not blocks.

   Collective on IS

   Input Parameters:
+  is - the index set
.  bs - number of elements in each block, one for each block and count of block not indices
.   n - the length of the index set (the number of blocks)
.  idx - the list of integers, these are by block, not by location
-  mode - see PetscCopyMode, only PETSC_COPY_VALUES and PETSC_OWN_POINTER are supported


   Notes:
   When the communicator is not MPI_COMM_SELF, the operations on the
   index sets, IS, are NOT conceptually the same as MPI_Group operations.
   The index sets are then distributed sets of indices and thus certain operations
   on them are collective.

   Example:
   If you wish to index the values {0,1,4,5}, then use
   a block size of 2 and idx of {0,2}.

   Level: beginner

.seealso: ISCreateStride(), ISCreateGeneral(), ISAllGather()
@*/
PetscErrorCode  ISBlockSetIndices(IS is,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISClearInfoCache(is,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscUseMethod(is,"ISBlockSetIndices_C",(IS,PetscInt,PetscInt,const PetscInt[],PetscCopyMode),(is,bs,n,idx,mode));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISBlockSetIndices_Block(IS is,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscErrorCode ierr;
  PetscInt       i,min,max;
  IS_Block       *sub = (IS_Block*)is->data;
  PetscLayout    map;

  PetscFunctionBegin;
  if (bs < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"block size < 1");
  if (n < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) PetscValidIntPointer(idx,3);

  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is),n*bs,is->map->N,bs,&map);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&is->map);CHKERRQ(ierr);
  is->map = map;

  if (sub->allocated) {ierr = PetscFree(sub->idx);CHKERRQ(ierr);}
  if (mode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc1(n,&sub->idx);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)is,n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(sub->idx,idx,n);CHKERRQ(ierr);
    sub->allocated = PETSC_TRUE;
  } else if (mode == PETSC_OWN_POINTER) {
    sub->idx = (PetscInt*) idx;
    ierr = PetscLogObjectMemory((PetscObject)is,n*sizeof(PetscInt));CHKERRQ(ierr);
    sub->allocated = PETSC_TRUE;
  } else if (mode == PETSC_USE_POINTER) {
    sub->idx = (PetscInt*) idx;
    sub->allocated = PETSC_FALSE;
  }

  if (n) {
    min = max = idx[0];
    for (i=1; i<n; i++) {
      if (idx[i] < min) min = idx[i];
      if (idx[i] > max) max = idx[i];
    }
    is->min = bs*min;
    is->max = bs*max+bs-1;
  } else {
    is->min = PETSC_MAX_INT;
    is->max = PETSC_MIN_INT;
  }
  PetscFunctionReturn(0);
}

/*@
   ISCreateBlock - Creates a data structure for an index set containing
   a list of integers. The indices are relative to entries, not blocks.

   Collective

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

.seealso: ISCreateStride(), ISCreateGeneral(), ISAllGather()
@*/
PetscErrorCode  ISCreateBlock(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(is,5);
  if (bs < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"block size < 1");
  if (n < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) PetscValidIntPointer(idx,4);

  ierr = ISCreate(comm,is);CHKERRQ(ierr);
  ierr = ISSetType(*is,ISBLOCK);CHKERRQ(ierr);
  ierr = ISBlockSetIndices(*is,bs,n,idx,mode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISBlockGetIndices_Block(IS is,const PetscInt *idx[])
{
  IS_Block *sub = (IS_Block*)is->data;

  PetscFunctionBegin;
  *idx = sub->idx;
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISBlockRestoreIndices_Block(IS is,const PetscInt *idx[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*@C
   ISBlockGetIndices - Gets the indices associated with each block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  idx - the integer indices, one for each block and count of block not indices

   Level: intermediate

.seealso: ISGetIndices(), ISBlockRestoreIndices()
@*/
PetscErrorCode  ISBlockGetIndices(IS is,const PetscInt *idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISBlockGetIndices_C",(IS,const PetscInt*[]),(is,idx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   ISBlockRestoreIndices - Restores the indices associated with each block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  idx - the integer indices

   Level: intermediate

.seealso: ISRestoreIndices(), ISBlockGetIndices()
@*/
PetscErrorCode  ISBlockRestoreIndices(IS is,const PetscInt *idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISBlockRestoreIndices_C",(IS,const PetscInt*[]),(is,idx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISBlockGetLocalSize - Returns the local number of blocks in the index set.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the local number of blocks

   Level: intermediate


.seealso: ISGetBlockSize(), ISBlockGetSize(), ISGetSize(), ISCreateBlock()
@*/
PetscErrorCode  ISBlockGetLocalSize(IS is,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISBlockGetLocalSize_C",(IS,PetscInt*),(is,size));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISBlockGetLocalSize_Block(IS is,PetscInt *size)
{
  PetscInt       bs, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  *size = n/bs;
  PetscFunctionReturn(0);
}

/*@
   ISBlockGetSize - Returns the global number of blocks in the index set.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the global number of blocks

   Level: intermediate


.seealso: ISGetBlockSize(), ISBlockGetLocalSize(), ISGetSize(), ISCreateBlock()
@*/
PetscErrorCode  ISBlockGetSize(IS is,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(is,"ISBlockGetSize_C",(IS,PetscInt*),(is,size));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISBlockGetSize_Block(IS is,PetscInt *size)
{
  PetscInt       bs, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(is->map, &N);CHKERRQ(ierr);
  *size = N/bs;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode ISCreate_Block(IS is)
{
  PetscErrorCode ierr;
  IS_Block       *sub;

  PetscFunctionBegin;
  ierr = PetscNewLog(is,&sub);CHKERRQ(ierr);
  is->data = (void *) sub;
  ierr = PetscMemcpy(is->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockSetIndices_C",ISBlockSetIndices_Block);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockGetIndices_C",ISBlockGetIndices_Block);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockRestoreIndices_C",ISBlockRestoreIndices_Block);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockGetSize_C",ISBlockGetSize_Block);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISBlockGetLocalSize_C",ISBlockGetLocalSize_Block);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
