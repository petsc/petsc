#ifndef lint
static char vcid[] = "$Id: stride.c,v 1.7 1995/03/21 23:17:58 bsmith Exp bsmith $";
#endif
/*
       General indices as a list of integers
*/
#include "isimpl.h"

typedef struct {
  int n,first,step;
} IndexiStride;

/*
   Returns info on stride index set. This is a pseudo-public function.
*/
int ISStrideGetInfo(IS is,int *first,int *step)
{
  IndexiStride *sub = (IndexiStride *) is->data;
  *first = sub->first; *step = sub->step;
  return 0;
}

static int ISidestroy(PetscObject obj)
{
  IS is = (IS) obj;
  FREE(is->data); 
  PLogObjectDestroy(is);
  PETSCHEADERDESTROY(is); return 0;
}

static int ISiIndices(IS in,int **idx)
{
  IndexiStride *sub = (IndexiStride *) in->data;
  int          i;

  if (sub->n) {
    *idx = (int *) MALLOC(sub->n*sizeof(int)); CHKPTR(idx);
    (*idx)[0] = sub->first;
    for ( i=1; i<sub->n; i++ ) (*idx)[i] = (*idx)[i-1] + sub->step;
  }
  else *idx = 0;
  return 0;
}

static int ISiRestoreIndices(IS in,int **idx)
{
  if (*idx) FREE(*idx);
  return 0;
}

static int ISiSize(IS is,int *size)
{
  IndexiStride *sub = (IndexiStride *)is->data;
  *size = sub->n; return 0;
}



static int ISgview(PetscObject obj, Viewer viewer)
{
  IS            is = (IS) obj;
  IndexiStride *sub = (IndexiStride *)is->data;
  int           i,n = sub->n;
  if (is->isperm) {
    ViewerPrintf(viewer,"Index set is permutation\n");
  }
  ViewerPrintf(viewer,"Number of indices in set %d\n",n);
  for ( i=0; i<n; i++ ) {
    ViewerPrintf(viewer,"%d %d\n",i,sub->first + i*sub->step);
  }
  return 0;
}
  
static struct _ISOps myops = { ISiSize,ISiSize,
                               ISiIndices,ISiRestoreIndices,0};
/*@
    ISCreateStrideSequential - Creates data structure for 
     a index set containing a list of evenly spaced integers.

  Input Parameters:
.   n - the length of the index set
.   first - the first element in the index set
.   step - the change to the next index

  Output Parameters:
.   is - the location to stash the index set

  Keywords: index set,stride
@*/
int ISCreateStrideSequential(int n,int first,int step,IS *is)
{
  int          size = sizeof(IndexiStride);
  int          min, max;
  IS           Nindex;
  IndexiStride *sub;

  *is = 0;
 
  if (n < 0) SETERR(1,"Number of indices must be non-negative");
  if (step == 0) SETERR(1,"Step must be nonzero");

  PETSCHEADERCREATE(Nindex, _IS,IS_COOKIE,ISSTRIDESEQUENTIAL,MPI_COMM_SELF); 
  PLogObjectCreate(Nindex);
  sub            = (IndexiStride *) MALLOC(size); CHKPTR(sub);
  sub->n         = n;
  sub->first     = first;
  sub->step      = step;
  if (step > 0) {min = first; max = first + step*(n-1);}
  else          {max = first; min = first + step*(n-1);}

  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void *) sub;
  Nindex->ops     = &myops;
  Nindex->destroy = ISidestroy;
  Nindex->view    = ISgview;
  Nindex->isperm  = 0;
  *is = Nindex; return 0;
}

