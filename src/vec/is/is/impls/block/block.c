
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

  PetscFunctionBegin;
  if (sub->allocated) PetscCall(PetscFree(sub->idx));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockSetIndices_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockGetIndices_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockRestoreIndices_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockGetSize_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockGetLocalSize_C",NULL));
  PetscCall(PetscFree(is->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISLocate_Block(IS is,PetscInt key,PetscInt *location)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       numIdx, i, bs, bkey, mkey;
  PetscBool      sorted;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(is->map,&bs));
  PetscCall(PetscLayoutGetSize(is->map,&numIdx));
  numIdx /= bs;
  bkey    = key / bs;
  mkey    = key % bs;
  if (mkey < 0) {
    bkey--;
    mkey += bs;
  }
  PetscCall(ISGetInfo(is,IS_SORTED,IS_LOCAL,PETSC_TRUE,&sorted));
  if (sorted) {
    PetscCall(PetscFindInt(bkey,numIdx,sub->idx,location));
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
  PetscInt       i,j,k,bs,n,*ii,*jj;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(in->map, &bs));
  PetscCall(PetscLayoutGetLocalSize(in->map, &n));
  n   /= bs;
  if (bs == 1) *idx = sub->idx;
  else {
    if (n) {
      PetscCall(PetscMalloc1(bs*n,&jj));
      *idx = jj;
      k    = 0;
      ii   = sub->idx;
      for (i=0; i<n; i++)
        for (j=0; j<bs; j++)
          jj[k++] = bs*ii[i] + j;
    } else {
      /* do not malloc for zero size because F90Array1dCreate() inside ISRestoreArrayF90() does not keep array when zero length array */
      *idx = NULL;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISRestoreIndices_Block(IS is,const PetscInt *idx[])
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       bs;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  if (bs != 1) {
    PetscCall(PetscFree(*(void**)idx));
  } else {
    /* F90Array1dCreate() inside ISRestoreArrayF90() does not keep array when zero length array */
    PetscCheck(is->map->n <= 0 || *idx == sub->idx,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from ISGetIndices()");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISInvertPermutation_Block(IS is,PetscInt nlocal,IS *isout)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       i,*ii,bs,n,*idx = sub->idx;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)is),&size));
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  n   /= bs;
  if (size == 1) {
    PetscCall(PetscMalloc1(n,&ii));
    for (i=0; i<n; i++) ii[idx[i]] = i;
    PetscCall(ISCreateBlock(PETSC_COMM_SELF,bs,n,ii,PETSC_OWN_POINTER,isout));
    PetscCall(ISSetPermutation(*isout));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No inversion written yet for block IS");
  PetscFunctionReturn(0);
}

static PetscErrorCode ISView_Block(IS is, PetscViewer viewer)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       i,bs,n,*idx = sub->idx;
  PetscBool      iascii,ibinary;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  n   /= bs;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&ibinary));
  if (iascii) {
    PetscViewerFormat fmt;

    PetscCall(PetscViewerGetFormat(viewer,&fmt));
    if (fmt == PETSC_VIEWER_ASCII_MATLAB) {
      IS             ist;
      const char     *name;
      const PetscInt *idx;
      PetscInt       n;

      PetscCall(PetscObjectGetName((PetscObject)is,&name));
      PetscCall(ISGetLocalSize(is,&n));
      PetscCall(ISGetIndices(is,&idx));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is),n,idx,PETSC_USE_POINTER,&ist));
      PetscCall(PetscObjectSetName((PetscObject)ist,name));
      PetscCall(ISView(ist,viewer));
      PetscCall(ISDestroy(&ist));
      PetscCall(ISRestoreIndices(is,&idx));
    } else {
      PetscBool isperm;

      PetscCall(ISGetInfo(is,IS_PERMUTATION,IS_GLOBAL,PETSC_FALSE,&isperm));
      if (isperm) PetscCall(PetscViewerASCIIPrintf(viewer,"Block Index set is permutation\n"));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"Block size %" PetscInt_FMT "\n",bs));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"Number of block indices in set %" PetscInt_FMT "\n",n));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"The first indices of each block are\n"));
      for (i=0; i<n; i++) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"Block %" PetscInt_FMT " Index %" PetscInt_FMT "\n",i,idx[i]));
      }
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    }
  } else if (ibinary) PetscCall(ISView_Binary(is,viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSort_Block(IS is)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       bs, n;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(PetscIntSortSemiOrdered(n/bs,sub->idx));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSortRemoveDups_Block(IS is)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       bs, n, nb;
  PetscBool      sorted;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  nb   = n/bs;
  PetscCall(ISGetInfo(is,IS_SORTED,IS_LOCAL,PETSC_TRUE,&sorted));
  if (sorted) {
    PetscCall(PetscSortedRemoveDupsInt(&nb,sub->idx));
  } else {
    PetscCall(PetscSortRemoveDupsInt(&nb,sub->idx));
  }
  PetscCall(PetscLayoutDestroy(&is->map));
  PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is), nb*bs, PETSC_DECIDE, bs, &is->map));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSorted_Block(IS is,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscCall(ISGetInfo(is,IS_SORTED,IS_LOCAL,PETSC_TRUE,flg));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSortedLocal_Block(IS is,PetscBool *flg)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       n, bs, i, *idx;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  n   /= bs;
  idx  = sub->idx;
  for (i = 1; i < n; i++) if (idx[i] < idx[i - 1]) break;
  if (i < n) *flg = PETSC_FALSE;
  else       *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISUniqueLocal_Block(IS is,PetscBool *flg)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       n, bs, i, *idx, *idxcopy = NULL;
  PetscBool      sortedLocal;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  n   /= bs;
  idx  = sub->idx;
  PetscCall(ISGetInfo(is,IS_SORTED,IS_LOCAL,PETSC_TRUE,&sortedLocal));
  if (!sortedLocal) {
    PetscCall(PetscMalloc1(n, &idxcopy));
    PetscCall(PetscArraycpy(idxcopy, idx, n));
    PetscCall(PetscIntSortSemiOrdered(n, idxcopy));
    idx = idxcopy;
  }
  for (i = 1; i < n; i++) if (idx[i] == idx[i - 1]) break;
  if (i < n) *flg = PETSC_FALSE;
  else       *flg = PETSC_TRUE;
  PetscCall(PetscFree(idxcopy));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISPermutationLocal_Block(IS is,PetscBool *flg)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       n, bs, i, *idx, *idxcopy = NULL;
  PetscBool      sortedLocal;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  n   /= bs;
  idx  = sub->idx;
  PetscCall(ISGetInfo(is,IS_SORTED,IS_LOCAL,PETSC_TRUE,&sortedLocal));
  if (!sortedLocal) {
    PetscCall(PetscMalloc1(n, &idxcopy));
    PetscCall(PetscArraycpy(idxcopy, idx, n));
    PetscCall(PetscIntSortSemiOrdered(n, idxcopy));
    idx = idxcopy;
  }
  for (i = 0; i < n; i++) if (idx[i] != i) break;
  if (i < n) *flg = PETSC_FALSE;
  else       *flg = PETSC_TRUE;
  PetscCall(PetscFree(idxcopy));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISIntervalLocal_Block(IS is,PetscBool *flg)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       n, bs, i, *idx;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  n   /= bs;
  idx  = sub->idx;
  for (i = 1; i < n; i++) if (idx[i] != idx[i - 1] + 1) break;
  if (i < n) *flg = PETSC_FALSE;
  else       *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISDuplicate_Block(IS is,IS *newIS)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt        bs, n;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  n   /= bs;
  PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)is),bs,n,sub->idx,PETSC_COPY_VALUES,newIS));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISCopy_Block(IS is,IS isy)
{
  IS_Block       *is_block = (IS_Block*)is->data,*isy_block = (IS_Block*)isy->data;
  PetscInt       bs, n;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(PetscArraycpy(isy_block->idx, is_block->idx, n/bs));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISOnComm_Block(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  IS_Block       *sub = (IS_Block*)is->data;
  PetscInt       bs, n;

  PetscFunctionBegin;
  PetscCheck(mode != PETSC_OWN_POINTER,comm,PETSC_ERR_ARG_WRONG,"Cannot use PETSC_OWN_POINTER");
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(ISCreateBlock(comm,bs,n/bs,sub->idx,mode,newis));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSetBlockSize_Block(IS is,PetscInt bs)
{
  PetscFunctionBegin;
  PetscCheck(is->map->bs <= 0 || bs == is->map->bs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot change blocksize %" PetscInt_FMT " (to %" PetscInt_FMT ") if ISType is ISBLOCK",is->map->bs,bs);
  PetscCall(PetscLayoutSetBlockSize(is->map, bs));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISToGeneral_Block(IS inis)
{
  IS_Block       *sub   = (IS_Block*)inis->data;
  PetscInt       bs,n;
  const PetscInt *idx;

  PetscFunctionBegin;
  PetscCall(ISGetBlockSize(inis,&bs));
  PetscCall(ISGetLocalSize(inis,&n));
  PetscCall(ISGetIndices(inis,&idx));
  if (bs == 1) {
    PetscCopyMode mode = sub->allocated ? PETSC_OWN_POINTER : PETSC_USE_POINTER;
    sub->allocated = PETSC_FALSE; /* prevent deallocation when changing the subtype*/
    PetscCall(ISSetType(inis,ISGENERAL));
    PetscCall(ISGeneralSetIndices(inis,n,idx,mode));
  } else {
    PetscCall(ISSetType(inis,ISGENERAL));
    PetscCall(ISGeneralSetIndices(inis,n,idx,PETSC_OWN_POINTER));
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
                               ISCopy_Block,
                               ISToGeneral_Block,
                               ISOnComm_Block,
                               ISSetBlockSize_Block,
                               NULL,
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
   ISBlockSetIndices - Set integers representing blocks of indices in an index set.

   Collective on IS

   Input Parameters:
+  is - the index set
.  bs - number of elements in each block
.   n - the length of the index set (the number of blocks)
.  idx - the list of integers, one for each block, the integers contain the index of the first index of each block divided by the block size
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

.seealso: `ISCreateStride()`, `ISCreateGeneral()`, `ISAllGather()`, `ISCreateBlock()`, `ISBLOCK`, `ISGeneralSetIndices()`
@*/
PetscErrorCode  ISBlockSetIndices(IS is,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscFunctionBegin;
  PetscCall(ISClearInfoCache(is,PETSC_FALSE));
  PetscUseMethod(is,"ISBlockSetIndices_C",(IS,PetscInt,PetscInt,const PetscInt[],PetscCopyMode),(is,bs,n,idx,mode));
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISBlockSetIndices_Block(IS is,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscInt       i,min,max;
  IS_Block       *sub = (IS_Block*)is->data;
  PetscLayout    map;

  PetscFunctionBegin;
  PetscCheck(bs >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"block size < 1");
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) PetscValidIntPointer(idx,4);

  PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is),n*bs,is->map->N,bs,&map));
  PetscCall(PetscLayoutDestroy(&is->map));
  is->map = map;

  if (sub->allocated) PetscCall(PetscFree(sub->idx));
  if (mode == PETSC_COPY_VALUES) {
    PetscCall(PetscMalloc1(n,&sub->idx));
    PetscCall(PetscLogObjectMemory((PetscObject)is,n*sizeof(PetscInt)));
    PetscCall(PetscArraycpy(sub->idx,idx,n));
    sub->allocated = PETSC_TRUE;
  } else if (mode == PETSC_OWN_POINTER) {
    sub->idx = (PetscInt*) idx;
    PetscCall(PetscLogObjectMemory((PetscObject)is,n*sizeof(PetscInt)));
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
   a list of integers. Each integer represents a fixed block size set of indices.

   Collective

   Input Parameters:
+  comm - the MPI communicator
.  bs - number of elements in each block
.  n - the length of the index set (the number of blocks)
.  idx - the list of integers, one for each block, the integers contain the index of the first entry of each block divided by the block size
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

.seealso: `ISCreateStride()`, `ISCreateGeneral()`, `ISAllGather()`, `ISBlockSetIndices()`, `ISBLOCK`, `ISGENERAL`
@*/
PetscErrorCode  ISCreateBlock(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscInt idx[],PetscCopyMode mode,IS *is)
{
  PetscFunctionBegin;
  PetscValidPointer(is,6);
  PetscCheck(bs >= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"block size < 1");
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) PetscValidIntPointer(idx,4);

  PetscCall(ISCreate(comm,is));
  PetscCall(ISSetType(*is,ISBLOCK));
  PetscCall(ISBlockSetIndices(*is,bs,n,idx,mode));
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

.seealso: `ISGetIndices()`, `ISBlockRestoreIndices()`, `ISBLOCK`, `ISBlockSetIndices()`, `ISCreateBlock()`
@*/
PetscErrorCode  ISBlockGetIndices(IS is,const PetscInt *idx[])
{
  PetscFunctionBegin;
  PetscUseMethod(is,"ISBlockGetIndices_C",(IS,const PetscInt*[]),(is,idx));
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

.seealso: `ISRestoreIndices()`, `ISBlockGetIndices()`
@*/
PetscErrorCode  ISBlockRestoreIndices(IS is,const PetscInt *idx[])
{
  PetscFunctionBegin;
  PetscUseMethod(is,"ISBlockRestoreIndices_C",(IS,const PetscInt*[]),(is,idx));
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

.seealso: `ISGetBlockSize()`, `ISBlockGetSize()`, `ISGetSize()`, `ISCreateBlock()`, `ISBLOCK`
@*/
PetscErrorCode  ISBlockGetLocalSize(IS is,PetscInt *size)
{
  PetscFunctionBegin;
  PetscUseMethod(is,"ISBlockGetLocalSize_C",(IS,PetscInt*),(is,size));
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISBlockGetLocalSize_Block(IS is,PetscInt *size)
{
  PetscInt       bs, n;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
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

.seealso: `ISGetBlockSize()`, `ISBlockGetLocalSize()`, `ISGetSize()`, `ISCreateBlock()`, `ISBLOCK`
@*/
PetscErrorCode  ISBlockGetSize(IS is,PetscInt *size)
{
  PetscFunctionBegin;
  PetscUseMethod(is,"ISBlockGetSize_C",(IS,PetscInt*),(is,size));
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISBlockGetSize_Block(IS is,PetscInt *size)
{
  PetscInt       bs, N;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetBlockSize(is->map, &bs));
  PetscCall(PetscLayoutGetSize(is->map, &N));
  *size = N/bs;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode ISCreate_Block(IS is)
{
  IS_Block       *sub;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(is,&sub));
  is->data = (void *) sub;
  PetscCall(PetscMemcpy(is->ops,&myops,sizeof(myops)));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockSetIndices_C",ISBlockSetIndices_Block));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockGetIndices_C",ISBlockGetIndices_Block));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockRestoreIndices_C",ISBlockRestoreIndices_Block));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockGetSize_C",ISBlockGetSize_Block));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISBlockGetLocalSize_C",ISBlockGetLocalSize_Block));
  PetscFunctionReturn(0);
}
