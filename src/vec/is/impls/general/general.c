#ifndef lint
static char vcid[] = "$Id: general.c,v 1.15 1995/03/27 22:56:22 bsmith Exp bsmith $";
#endif
/*
       General indices as a list of integers
*/
#include "isimpl.h"

typedef struct {
  int n,sorted; 
  int *idx;
} IS_General;

static int ISDestroy_General(PetscObject obj)
{
  IS is = (IS) obj;
  FREE(is->data); 
  PLogObjectDestroy(is);
  PETSCHEADERDESTROY(is); return 0;
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
  int           i,ierr, *ii,n = sub->n,*idx = sub->idx;
  ii = (int *) MALLOC( n*sizeof(int) ); CHKPTR(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  if ((ierr = ISCreateSequential(MPI_COMM_SELF,n,ii,isout))) SETERR(ierr,0);
  ISSetPermutation(*isout);
  FREE(ii);
  return 0;
}

static int ISView_General(PetscObject obj, Viewer viewer)
{
  IS          is = (IS) obj;
  IS_General  *sub = (IS_General *)is->data;
  int         i,n = sub->n,*idx = sub->idx;
  FILE        *fd;
  PetscObject vobj = (PetscObject) viewer;

  if (vobj->cookie == VIEWER_COOKIE && ((vobj->type == FILE_VIEWER) ||
                                       (vobj->type == FILES_VIEWER))) {
    fd = ViewerFileGetPointer(viewer);
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
  
static struct _ISOps myops = { ISGetSize_General,ISGetSize_General,
                               ISGetIndices_General,0,
                               ISInvertPermutation_General};
/*@
    ISCreateSequential - Creates a data structure for an index set 
    containing a list of integers.

  Input Parameters:
.   n - the length of the index set
.   idx - the list of integers
.   comm - the MPI communicator

  Output Parameter:
.   is - the new index set
@*/
int ISCreateSequential(MPI_Comm comm,int n,int *idx,IS *is)
{
  int        i, sorted = 1, size = sizeof(IS_General) + n*sizeof(int);
  int        min, max;
  IS         Nindex;
  IS_General *sub;

  *is = 0;
  PETSCHEADERCREATE(Nindex, _IS,IS_COOKIE,ISGENERALSEQUENTIAL,comm); 
  PLogObjectCreate(Nindex);
  sub            = (IS_General *) MALLOC(size); CHKPTR(sub);
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
  MEMCPY(sub->idx,idx,n*sizeof(int));
  sub->sorted     = sorted;
  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void *) sub;
  Nindex->ops     = &myops;
  Nindex->destroy = ISDestroy_General;
  Nindex->view    = ISView_General;
  Nindex->isperm  = 0;
  *is = Nindex; return 0;
}

