#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: block.c,v 1.39 1999/10/01 21:20:52 bsmith Exp bsmith $";
#endif
/*
     Provides the functions for index sets (IS) defined by a list of integers.
   These are for blocks of data, each block is indicated with a single integer.
*/
#include "src/vec/is/isimpl.h"               /*I  "is.h"     I*/
#include "sys.h"

typedef struct {
  int n;            /* number of blocks */
  int sorted;       /* are the blocks sorted? */
  int *idx;
  int bs;           /* blocksize */
} IS_Block;

#undef __FUNC__  
#define __FUNC__ "ISDestroy_Block" 
int ISDestroy_Block(IS is)
{
  IS_Block *is_block = (IS_Block *) is->data;
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(is_block->idx);CHKERRQ(ierr);
  ierr = PetscFree(is_block);CHKERRQ(ierr);
  PLogObjectDestroy(is);
  PetscHeaderDestroy(is); PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISGetIndices_Block" 
int ISGetIndices_Block(IS in,int **idx)
{
  IS_Block *sub = (IS_Block *) in->data;
  int      i,j,k,bs = sub->bs,n = sub->n,*ii,*jj;

  PetscFunctionBegin;
  if (sub->bs == 1) {
    *idx = sub->idx; 
  } else {
    jj   = (int *) PetscMalloc(sub->bs*(1+sub->n)*sizeof(int));CHKPTRQ(jj)
    *idx = jj;
    k    = 0;
    ii   = sub->idx;
    for ( i=0; i<n; i++ ) {
      for ( j=0; j<bs; j++ ) {
        jj[k++] = ii[i] + j;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISRestoreIndices_Block" 
int ISRestoreIndices_Block(IS in,int **idx)
{
  IS_Block *sub = (IS_Block *) in->data;
  int      ierr;

  PetscFunctionBegin;
  if (sub->bs != 1) {
    ierr = PetscFree(*idx);CHKERRQ(ierr);
  } else {
    if (*idx !=  sub->idx) {
      SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must restore with value from ISGetIndices()");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISGetSize_Block" 
int ISGetSize_Block(IS is,int *size)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *size = sub->bs*sub->n; 
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "ISInvertPermutation_Block" 
int ISInvertPermutation_Block(IS is, IS *isout)
{
  IS_Block *sub = (IS_Block *)is->data;
  int      i,ierr, *ii,n = sub->n,*idx = sub->idx;

  PetscFunctionBegin;
  ii = (int *) PetscMalloc( (n+1)*sizeof(int) );CHKPTRQ(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  ierr = ISCreateBlock(PETSC_COMM_SELF,sub->bs,n,ii,isout);CHKERRQ(ierr);
  ISSetPermutation(*isout);
  ierr = PetscFree(ii);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISView_Block" 
int ISView_Block(IS is, Viewer viewer)
{
  IS_Block    *sub = (IS_Block *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  int         isascii;
  FILE        *fd;

  PetscFunctionBegin;
  isascii = PetscTypeCompare(viewer,ASCII_VIEWER);
  if (isascii) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd);CHKERRQ(ierr);
    if (is->isperm) {
      fprintf(fd,"Block Index set is permutation\n");
    }
    fprintf(fd,"Block size %d\n",sub->bs);
    fprintf(fd,"Number of block indices in set %d\n",n);
    fprintf(fd,"The first indices of each block are\n");
    for ( i=0; i<n; i++ ) {
      fprintf(fd,"%d %d\n",i,idx[i]);
    }
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSort_Block" 
int ISSort_Block(IS is)
{
  IS_Block *sub = (IS_Block *)is->data;
  int      ierr;

  PetscFunctionBegin;
  if (sub->sorted) PetscFunctionReturn(0);
  ierr = PetscSortInt(sub->n, sub->idx);CHKERRQ(ierr);
  sub->sorted = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSorted_Block" 
int ISSorted_Block(IS is, PetscTruth *flg)
{
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  *flg = (PetscTruth) sub->sorted;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISDuplicate_Block" 
int ISDuplicate_Block(IS is, IS *newIS)
{
  int      ierr;
  IS_Block *sub = (IS_Block *)is->data;

  PetscFunctionBegin;
  ierr = ISCreateBlock(is->comm, sub->bs, sub->n, sub->idx, newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISIdentity_Block" 
int ISIdentity_Block(IS is,PetscTruth *ident)
{
  IS_Block *is_block = (IS_Block *) is->data;
  int      i, n = is_block->n,*idx = is_block->idx, bs = is_block->bs;

  PetscFunctionBegin;
  is->isidentity = 1;
  *ident         = PETSC_TRUE;
  for (i=0; i<n; i++) {
    if (idx[i] != bs*i) {
      is->isidentity = 0;
      *ident         = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

static struct _ISOps myops = { ISGetSize_Block,
                               ISGetSize_Block,
                               ISGetIndices_Block,
                               ISRestoreIndices_Block,
                               ISInvertPermutation_Block,
                               ISSort_Block,
                               ISSorted_Block,
                               ISDuplicate_Block,
                               ISDestroy_Block,
                               ISView_Block,
                               ISIdentity_Block };
#undef __FUNC__  
#define __FUNC__ "ISCreateBlock" 
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
   The index sets are then distributed sets of indices. 

   Example:
   If you wish to index the values {0,1,4,5}, then use
   a block size of 2 and idx of {0,4}.

   Level: beginner

.keywords: IS, index set, create, block

.seealso: ISCreateStride(), ISCreateGeneral(), ISAllGather()
@*/
int ISCreateBlock(MPI_Comm comm,int bs,int n,const int idx[],IS *is)
{
  int      i, sorted = 1, min, max, ierr;
  IS       Nindex;
  IS_Block *sub;

  PetscFunctionBegin;
  *is = 0;
  PetscHeaderCreate(Nindex, _p_IS,struct _ISOps,IS_COOKIE,IS_BLOCK,"IS",comm,ISDestroy,ISView); 
  PLogObjectCreate(Nindex);
  sub            = PetscNew(IS_Block);CHKPTRQ(sub);
  PLogObjectMemory(Nindex,sizeof(IS_Block)+n*sizeof(int)+sizeof(struct _p_IS));
  sub->idx       = (int *) PetscMalloc((n+1)*sizeof(int));CHKPTRQ(sub->idx);
  sub->n         = n;
  for ( i=1; i<n; i++ ) {
    if (idx[i] < idx[i-1]) {sorted = 0; break;}
  }
  if (n) {min = max = idx[0];} else {min = max = 0;}
  for ( i=1; i<n; i++ ) {
    if (idx[i] < min) min = idx[i];
    if (idx[i] > max) max = idx[i];
  }
  ierr = PetscMemcpy(sub->idx,idx,n*sizeof(int));CHKERRQ(ierr);
  sub->sorted     = sorted;
  sub->bs         = bs;
  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void *) sub;
  ierr = PetscMemcpy(Nindex->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  Nindex->isperm  = 0;
  *is = Nindex; PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "ISBlockGetIndices" 
/*@C
   ISBlockGetIndices - Gets the indices associated with each block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  idx - the integer indices

   Level: intermediate

.keywords: IS, index set, block, get, indices

.seealso: ISGetIndices(), ISBlockRestoreIndices()
@*/
int ISBlockGetIndices(IS in,int *idx[])
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(in,IS_COOKIE);
  PetscValidPointer(idx);
  if (in->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Not a block index set");

  sub = (IS_Block *) in->data;
  *idx = sub->idx; 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISBlockRestoreIndices" 
/*@C
   ISBlockRestoreIndices - Restores the indices associated with each block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  idx - the integer indices

   Level: intermediate

.keywords: IS, index set, block, restore, indices

.seealso: ISRestoreIndices(), ISBlockGetIndices()
@*/
int ISBlockRestoreIndices(IS is,int *idx[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidPointer(idx);
  if (is->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Not a block index set");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISBlockGetBlockSize" 
/*@
   ISBlockGetBlockSize - Returns the number of elements in a block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the number of elements in a block

   Level: intermediate

.keywords: IS, index set, block, get, size

.seealso: ISBlockGetSize(), ISGetSize(), ISBlock(), ISCreateBlock()
@*/
int ISBlockGetBlockSize(IS is,int *size)
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(size);
  if (is->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Not a block index set");

  sub = (IS_Block *)is->data;
  *size = sub->bs; 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISBlock" 
/*@C
   ISBlock - Checks whether an index set is blocked.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  flag - PETSC_TRUE if a block index set, else PETSC_FALSE

   Level: intermediate

.keywords: IS, index set, block

.seealso: ISBlockGetSize(), ISGetSize(), ISBlockGetBlockSize(), ISCreateBlock()
@*/
int ISBlock(IS is,PetscTruth *flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(flag);
  if (is->type != IS_BLOCK) *flag = PETSC_FALSE;
  else                          *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISBlockGetSize" 
/*@
   ISBlockGetSize - Returns the number of blocks in the index set.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the number of blocks

   Level: intermediate

.keywords: IS, index set, block, get, size

.seealso: ISBlockGetBlockSize(), ISGetSize(), ISBlock(), ISCreateBlock()
@*/
int ISBlockGetSize(IS is,int *size)
{
  IS_Block *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(size);
  if (is->type != IS_BLOCK) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Not a block index set");

  sub = (IS_Block *)is->data;
  *size = sub->n; 
  PetscFunctionReturn(0);
}
