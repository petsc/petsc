/*
     Provides the functions for index sets (IS) defined by a list of integers.
   These are for blocks of data, each block is indicated with a single integer.
*/
#include "src/vec/is/isimpl.h"               /*I  "petscis.h"     I*/
#include "petscsys.h"

EXTERN int VecInitializePackage(char *);

typedef struct {
  int        N,n;            /* number of blocks */
  PetscTruth sorted;       /* are the blocks sorted? */
  int        *idx;
  int        bs;           /* blocksize */
} IS_Block;

#undef __FUNCT__  
#define __FUNCT__ "ISDestroy_Block" 
int ISDestroy_Block(IS is)
{
  IS_Block *is_block = (IS_Block*)is->data;
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(is_block->idx);CHKERRQ(ierr);
  ierr = PetscFree(is_block);CHKERRQ(ierr);
  PetscLogObjectDestroy(is);
  PetscHeaderDestroy(is); PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetIndices_Block" 
int ISGetIndices_Block(IS in,int **idx)
{
  IS_Block *sub = (IS_Block*)in->data;
  int      i,j,k,bs = sub->bs,n = sub->n,*ii,*jj,ierr;

  PetscFunctionBegin;
  if (sub->bs == 1) {
    *idx = sub->idx; 
  } else {
    ierr = PetscMalloc(sub->bs*(1+sub->n)*sizeof(int),&jj);CHKERRQ(ierr);
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
int ISRestoreIndices_Block(IS in,int **idx)
{
  IS_Block *sub = (IS_Block*)in->data;
  int      ierr;

  PetscFunctionBegin;
  if (sub->bs != 1) {
    ierr = PetscFree(*idx);CHKERRQ(ierr);
  } else {
    if (*idx !=  sub->idx) {
      SETERRQ(PETSC_ERR_ARG_WRONG,"Must restore with value from ISGetIndices()");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetSize_Block" 
int ISGetSize_Block(IS is,int *size)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *size = sub->bs*sub->N; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetLocalSize_Block" 
int ISGetLocalSize_Block(IS is,int *size)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *size = sub->bs*sub->n; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISInvertPermutation_Block" 
int ISInvertPermutation_Block(IS is,int nlocal,IS *isout)
{
  IS_Block *sub = (IS_Block *)is->data;
  int      i,ierr,*ii,n = sub->n,*idx = sub->idx,size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(is->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscMalloc((n+1)*sizeof(int),&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ii[idx[i]] = i;
    }
    ierr = ISCreateBlock(PETSC_COMM_SELF,sub->bs,n,ii,isout);CHKERRQ(ierr);
    ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
    ierr = PetscFree(ii);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"No inversion written yet for block IS");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISView_Block" 
int ISView_Block(IS is, PetscViewer viewer)
{
  IS_Block    *sub = (IS_Block *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  PetscTruth  iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) { 
    if (is->isperm) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block Index set is permutation\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Block size %d\n",sub->bs);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Number of block indices in set %d\n",n);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"The first indices of each block are\n");CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d %d\n",i,idx[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSort_Block" 
int ISSort_Block(IS is)
{
  IS_Block *sub = (IS_Block *)is->data;
  int      ierr;

  PetscFunctionBegin;
  if (sub->sorted) PetscFunctionReturn(0);
  ierr = PetscSortInt(sub->n,sub->idx);CHKERRQ(ierr);
  sub->sorted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSorted_Block" 
int ISSorted_Block(IS is,PetscTruth *flg)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *flg = sub->sorted;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISDuplicate_Block" 
int ISDuplicate_Block(IS is,IS *newIS)
{
  int      ierr;
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  ierr = ISCreateBlock(is->comm,sub->bs,sub->n,sub->idx,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISIdentity_Block" 
int ISIdentity_Block(IS is,PetscTruth *ident)
{
  IS_Block *is_block = (IS_Block*)is->data;
  int      i,n = is_block->n,*idx = is_block->idx,bs = is_block->bs;

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
                               ISIdentity_Block };
#undef __FUNCT__  
#define __FUNCT__ "ISCreateBlock" 
/*@C
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
int ISCreateBlock(MPI_Comm comm,int bs,int n,const int idx[],IS *is)
{
  int        i,min,max,ierr;
  IS         Nindex;
  IS_Block   *sub;
  PetscTruth sorted = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidPointer(is,5);
  if (n < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) {PetscValidIntPointer(idx,4);}
  *is = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = VecInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  PetscHeaderCreate(Nindex,_p_IS,struct _ISOps,IS_COOKIE,IS_BLOCK,"IS",comm,ISDestroy,ISView); 
  PetscLogObjectCreate(Nindex);
  ierr = PetscNew(IS_Block,&sub);CHKERRQ(ierr);
  PetscLogObjectMemory(Nindex,sizeof(IS_Block)+n*sizeof(int)+sizeof(struct _p_IS));
  ierr   = PetscMalloc((n+1)*sizeof(int),&sub->idx);CHKERRQ(ierr);
  sub->n = n;
  ierr = MPI_Allreduce(&n,&sub->N,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  for (i=1; i<n; i++) {
    if (idx[i] < idx[i-1]) {sorted = PETSC_FALSE; break;}
  }
  if (n) {min = max = idx[0];} else {min = max = 0;}
  for (i=1; i<n; i++) {
    if (idx[i] < min) min = idx[i];
    if (idx[i] > max) max = idx[i];
  }
  ierr = PetscMemcpy(sub->idx,idx,n*sizeof(int));CHKERRQ(ierr);
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
int ISBlockGetIndices(IS in,int *idx[])
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(in,IS_COOKIE,1);
  PetscValidPointer(idx,2);
  if (in->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a block index set");

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
int ISBlockRestoreIndices(IS is,int *idx[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(idx,2);
  if (is->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a block index set");
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
int ISBlockGetBlockSize(IS is,int *size)
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(size,2);
  if (is->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a block index set");

  sub = (IS_Block *)is->data;
  *size = sub->bs; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISBlock" 
/*@C
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
int ISBlock(IS is,PetscTruth *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(flag,2);
  if (is->type != IS_BLOCK) *flag = PETSC_FALSE;
  else                          *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISBlockGetSize" 
/*@
   ISBlockGetSize - Returns the number of blocks in the index set.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the number of blocks

   Level: intermediate

   Concepts: IS^block sizes
   Concepts: index sets^block sizes

.seealso: ISBlockGetBlockSize(), ISGetSize(), ISBlock(), ISCreateBlock()
@*/
int ISBlockGetSize(IS is,int *size)
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(size,2);
  if (is->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a block index set");

  sub = (IS_Block *)is->data;
  *size = sub->n; 
  PetscFunctionReturn(0);
}
