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

static int ISiPosition(IS is,int ii,int *pos)
{
  IndexiGeneral *sub = (IndexiGeneral *)is->data;
  int m,*idx = sub->idx, a, b, i, n = sub->n;
  if (!sub->sorted) {
    for (i=0; i<n; i++) {
      if (idx[i] = ii) {*pos = i; return 0;}
    }
    *pos = -1; return 0;
  }
  else {
    a = 0; b = n - 1;
    if (idx[a] > ii || idx[b] < ii) {*pos = -1; return 0;}
    while (b-a > 2) {
      m = (a+b)/2;
      if (idx[m] < ii) a = m;
      else if (idx[m] > ii) b = m;
      else {*pos = m; return 0;}
    }
    if (idx[a] == ii) {*pos = a; return 0;}
    if (idx[b] == ii) {*pos = b; return 0;}
    *pos = -1; return 0;
  }
}

static int ISiInverse(IS is, IS *isout)
{
  IndexiGeneral *sub = (IndexiGeneral *)is->data;
  int           i,ierr, *ii,n = sub->n,*idx = sub->idx;
  ii = (int *) MALLOC( n*sizeof(int) ); CHKPTR(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  if (ierr = ISCreateSequentialPermutation(n,ii,isout)) SETERR(ierr,0);
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
                               ISiPosition,ISiIndices,0,ISiInverse};
/*@
    ISCreateSequential - creates data structure for 
     a index set containing a list of integers.

  Input Parameters:
.   n - the length of the index set
.   idx - the list of integers.

@*/
int ISCreateSequential(int n,int *idx,IS *is)
{
  int     i, sorted = 1, size = sizeof(IndexiGeneral) + n*sizeof(int);
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
  MEMCPY(sub->idx,idx,n*sizeof(int));
  sub->sorted     = sorted;
  Nindex->data    = (void *) sub;
  Nindex->cookie  = IS_COOKIE;
  Nindex->type    = GENERALSEQUENTIAL;
  Nindex->ops     = &myops;
  Nindex->destroy = ISidestroy;
  Nindex->view    = ISgview;
  Nindex->isperm  = 0;
  *is = Nindex; return 0;
}

/*@
    ISCreateSequentialPermutation - creates data structure for 
     a index set containing a permutation. Special case of 
     ISCreateSequential().

  Input Parameters:
.   n - the length of the index set
.   idx - the list of integers.

@*/
int ISCreateSequentialPermutation(int n,int *idx,IS *is)
{
  int ierr;

  if (ierr = ISCreateSequential(n,idx,is)) SETERR(ierr,0);
  (*is)->isperm = 1;
  return 0;
}

