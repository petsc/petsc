
#ifndef lint
static char vcid[] = "$Id: block.c,v 1.1 1996/07/31 21:50:50 bsmith Exp bsmith $";
#endif
/*
     Provides the functions for index sets (IS) defined by a list of integers.
   These are for blocks of data, each block is indicated with a single integer.
*/
#include "isimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

typedef struct {
  int n;            /* number of blocks */
  int sorted;       /* are the blocks sorted? */
  int *idx;
  int bs;           /* blocksize */
} IS_Block;

static int ISDestroy_Block(PetscObject obj)
{
  IS is = (IS) obj;
  PetscFree(is->data); 
  PLogObjectDestroy(is);
  PetscHeaderDestroy(is); return 0;
}

static int ISGetIndices_Block(IS in,int **idx)
{
  IS_Block *sub = (IS_Block *) in->data;
  int      i,j,k,bs = sub->bs,n = sub->n,*ii,*jj;

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
  return 0;
}

static int ISRestoreIndices_Block(IS in,int **idx)
{
  IS_Block *sub = (IS_Block *) in->data;

  if (sub->bs != 1) {
    PetscFree(*idx);
  }
  return 0;
}

static int ISGetSize_Block(IS is,int *size)
{
  IS_Block *sub = (IS_Block *)is->data;
  *size = sub->bs*sub->n; 
  return 0;
}


static int ISInvertPermutation_Block(IS is, IS *isout)
{
  IS_Block *sub = (IS_Block *)is->data;
  int      i,ierr, *ii,n = sub->n,*idx = sub->idx;

  ii = (int *) PetscMalloc( (n+1)*sizeof(int) ); CHKPTRQ(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  ierr = ISCreateBlockSeq(MPI_COMM_SELF,sub->bs,n,ii,isout); CHKERRQ(ierr);
  ISSetPermutation(*isout);
  PetscFree(ii);
  return 0;
}

static int ISView_Block(PetscObject obj, Viewer viewer)
{
  IS          is = (IS) obj;
  IS_Block    *sub = (IS_Block *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  FILE        *fd;
  ViewerType  vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (is->isperm) {
      fprintf(fd,"Block Index set is permutation\n");
    }
    fprintf(fd,"Block size %d\n",sub->bs);
    fprintf(fd,"Number of block indices in set %d\n",n);
    for ( i=0; i<n; i++ ) {
      fprintf(fd,"%d %d\n",i,idx[i]);
    }
  }
  return 0;
}

static int ISSort_Block(IS is)
{
  IS_Block *sub = (IS_Block *)is->data;
  int        ierr;

  if (sub->sorted) return 0;
  ierr = PetscSortInt(sub->n, sub->idx); CHKERRQ(ierr);
  sub->sorted = 1;
  return 0;
}

static int ISSorted_Block(IS is, PetscTruth *flg)
{
  IS_Block *sub = (IS_Block *)is->data;
  *flg = (PetscTruth) sub->sorted;
  return 0;
}

static struct _ISOps myops = { ISGetSize_Block,
                               ISGetSize_Block,
                               ISGetIndices_Block,
                               ISRestoreIndices_Block,
                               ISInvertPermutation_Block,
                               ISSort_Block,
                               ISSorted_Block };
/*@C
   ISCreateBlockSeq - Creates a data structure for an index set 
   containing a list of integers.

   Input Parameters:
.  n - the length of the index set
.  bs - number of elements in each block
.  idx - the list of integers
.  comm - the MPI communicator

   Output Parameter:
.  is - the new index set

.keywords: IS, sequential, index set, create, blocks

.seealso: ISCreateStrideSeq(), ISCreateSeq()
@*/
int ISCreateBlockSeq(MPI_Comm comm,int bs,int n,int *idx,IS *is)
{
  int      i, sorted = 1, size = sizeof(IS_Block) + n*sizeof(int),min, max;
  IS       Nindex;
  IS_Block *sub;

  *is = 0;
  PetscHeaderCreate(Nindex, _IS,IS_COOKIE,IS_BLOCK_SEQ,comm); 
  PLogObjectCreate(Nindex);
  sub            = (IS_Block *) PetscMalloc(size); CHKPTRQ(sub);
  PLogObjectMemory(Nindex,size + sizeof(struct _IS));
  sub->idx       = (int *) (sub+1);
  sub->n         = n;
  for ( i=1; i<n; i++ ) {
    if (idx[i] < idx[i-1]) {sorted = 0; break;}
  }
  if (n) {min = max = idx[0];} else {min = max = 0;}
  for ( i=1; i<n; i++ ) {
    if (idx[i] < min) min = idx[i];
    if (idx[i] > max) max = idx[i];
  }
  PetscMemcpy(sub->idx,idx,n*sizeof(int));
  sub->sorted     = sorted;
  sub->bs         = bs;
  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void *) sub;
  PetscMemcpy(&Nindex->ops,&myops,sizeof(myops));
  Nindex->destroy = ISDestroy_Block;
  Nindex->view    = ISView_Block;
  Nindex->isperm  = 0;
  *is = Nindex; return 0;
}


int ISBlockGetIndices(IS in,int **idx)
{
  IS_Block *sub;

  PetscValidPointer(idx);
  PetscValidHeaderSpecific(in,IS_COOKIE);

  sub = (IS_Block *) in->data;
  *idx = sub->idx; 
  return 0;
}

int ISBlockRestoreIndices(IS in,int **idx)
{
  PetscValidPointer(idx);
  PetscValidHeaderSpecific(in,IS_COOKIE);
  return 0;
}

int ISGetBlockSize(IS is,int *size)
{
  IS_Block *sub = (IS_Block *)is->data;
  *size = sub->bs; 
  return 0;
}

