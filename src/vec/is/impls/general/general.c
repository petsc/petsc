/*
       General indices as a list of integers
*/
#include "isimpl.h"

typedef struct {
  int n,sorted; 
  int *idx;
} IndexiGeneral;

static int ISidestroy(PetscObject obj)
{
  IS is = (IS) obj;
  FREE(is->data); FREE(is); return 0;
}

static int ISiIndices(IS in,int **idx)
{
  IndexiGeneral *sub = (IndexiGeneral *) in->data;
  *idx = sub->idx; return 0;
}

static int ISiSize(IS is,int *size)
{
  IndexiGeneral *sub = (IndexiGeneral *)is->data;
  *size = sub->n; return 0;
}


static int ISiInverse(IS is, IS *isout)
{
  IndexiGeneral *sub = (IndexiGeneral *)is->data;
  int           i,ierr, *ii,n = sub->n,*idx = sub->idx;
  ii = (int *) MALLOC( n*sizeof(int) ); CHKPTR(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  if (ierr = ISCreateSequential(n,ii,isout)) SETERR(ierr,0);
  ISSetPermutation(*isout);
  FREE(ii);
  return 0;
}

static int ISgview(PetscObject obj, Viewer viewer)
{
  IS            is = (IS) obj;
  IndexiGeneral *sub = (IndexiGeneral *)is->data;
  int           i,n = sub->n,*idx = sub->idx;
  if (is->isperm) {
    printf("Index set is permutation\n");
  }
  printf("Number of indices in set %d\n",n);
  for ( i=0; i<n; i++ ) {
    printf("%d %d\n",i,idx[i]);
  }
  return 0;
}
  
static struct _ISOps myops = { ISiSize,ISiSize,
                               ISiIndices,0,ISiInverse};
/*@
    ISCreateSequential - Creates data structure for 
     a index set containing a list of integers.

  Input Parameters:
.   n - the length of the index set
.   idx - the list of integers.

@*/
int ISCreateSequential(int n,int *idx,IS *is)
{
  int     i, sorted = 1, size = sizeof(IndexiGeneral) + n*sizeof(int);
  int     min, max;
  IS      Nindex;
  IndexiGeneral *sub;

  *is = 0;
  CREATEHEADER(Nindex, _IS); 
  sub            = (IndexiGeneral *) MALLOC(size); CHKPTR(sub);
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
  Nindex->cookie  = IS_COOKIE;
  Nindex->type    = ISGENERALSEQUENTIAL;
  Nindex->ops     = &myops;
  Nindex->destroy = ISidestroy;
  Nindex->view    = ISgview;
  Nindex->isperm  = 0;
  *is = Nindex; return 0;
}

