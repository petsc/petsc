
#ifndef lint
static char vcid[] = "$Id: general.c,v 1.44 1996/03/19 21:22:39 bsmith Exp bsmith $";
#endif
/*
     Provides the functions for index sets (IS) defined by a list of integers.
*/
#include "isimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

typedef struct {
  int n,sorted; 
  int *idx;
} IS_General;

static int ISDestroy_General(PetscObject obj)
{
  IS is = (IS) obj;
  PetscFree(is->data); 
  PLogObjectDestroy(is);
  PetscHeaderDestroy(is); return 0;
}

static int ISGetIndices_General(IS in,int **idx)
{
  IS_General *sub = (IS_General *) in->data;
  *idx = sub->idx; return 0;
}

static int ISGetSize_General(IS is,int *size)
{
  IS_General *sub = (IS_General *)is->data;
  *size = sub->n; return 0;
}


static int ISInvertPermutation_General(IS is, IS *isout)
{
  IS_General *sub = (IS_General *)is->data;
  int        i,ierr, *ii,n = sub->n,*idx = sub->idx;

  ii = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  ierr = ISCreateSeq(MPI_COMM_SELF,n,ii,isout); CHKERRQ(ierr);
  ISSetPermutation(*isout);
  PetscFree(ii);
  return 0;
}

static int ISView_General(PetscObject obj, Viewer viewer)
{
  IS          is = (IS) obj;
  IS_General  *sub = (IS_General *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  FILE        *fd;
  ViewerType  vtype;

  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF; 
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (is->isperm) {
      fprintf(fd,"Index set is permutation\n");
    }
    fprintf(fd,"Number of indices in set %d\n",n);
    for ( i=0; i<n; i++ ) {
      fprintf(fd,"%d %d\n",i,idx[i]);
    }
  }
  return 0;
}
static int ISSort_General(IS is)
{
  IS_General *sub = (IS_General *)is->data;
  int        ierr;

  if (sub->sorted) return 0;
  ierr = PetscSortInt(sub->n, sub->idx); CHKERRQ(ierr);
  sub->sorted = 1;
  return 0;
}

static int ISSorted_General(IS is, PetscTruth *flg)
{
  IS_General *sub = (IS_General *)is->data;
  *flg = (PetscTruth) sub->sorted;
  return 0;
}

static struct _ISOps myops = { ISGetSize_General,
                               ISGetSize_General,
                               ISGetIndices_General,0,
                               ISInvertPermutation_General,
                               ISSort_General,
                               ISSorted_General };
/*@C
   ISCreateSeq - Creates a data structure for an index set 
   containing a list of integers.

   Input Parameters:
.  n - the length of the index set
.  idx - the list of integers
.  comm - the MPI communicator

   Output Parameter:
.  is - the new index set

.keywords: IS, sequential, index set, create

.seealso: ISCreateStrideSeq()
@*/
int ISCreateSeq(MPI_Comm comm,int n,int *idx,IS *is)
{
  int        i, sorted = 1, size = sizeof(IS_General) + n*sizeof(int);
  int        min, max;
  IS         Nindex;
  IS_General *sub;

  *is = 0;
  PetscHeaderCreate(Nindex, _IS,IS_COOKIE,IS_SEQ,comm); 
  PLogObjectCreate(Nindex);
  sub            = (IS_General *) PetscMalloc(size); CHKPTRQ(sub);
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
  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void *) sub;
  PetscMemcpy(&Nindex->ops,&myops,sizeof(myops));
  Nindex->destroy = ISDestroy_General;
  Nindex->view    = ISView_General;
  Nindex->isperm  = 0;
  *is = Nindex; return 0;
}



