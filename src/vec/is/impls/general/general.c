
#ifndef lint
static char vcid[] = "$Id: general.c,v 1.34 1995/09/30 19:26:17 bsmith Exp curfman $";
#endif
/*
       General indices as a list of integers
*/
#include "isimpl.h"
#include "pinclude/pviewer.h"

typedef struct {
  int n,sorted; 
  int *idx;
} IS_General;

static int ISDestroy_General(PetscObject obj)
{
  IS is = (IS) obj;
  PETSCFREE(is->data); 
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
  int        i,ierr, *ii,n = sub->n,*idx = sub->idx;
  ii = (int *) PETSCMALLOC( n*sizeof(int) ); CHKPTRQ(ii);
  for ( i=0; i<n; i++ ) {
    ii[idx[i]] = i;
  }
  ierr = ISCreateSeq(MPI_COMM_SELF,n,ii,isout); CHKERRQ(ierr);
  ISSetPermutation(*isout);
  PETSCFREE(ii);
  return 0;
}

static int ISView_General(PetscObject obj, Viewer viewer)
{
  IS          is = (IS) obj;
  IS_General  *sub = (IS_General *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  FILE        *fd;
  PetscObject vobj = (PetscObject) viewer;

  if (!viewer) {
    viewer = STDOUT_VIEWER_SELF; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == VIEWER_COOKIE) {
    if ((vobj->type == ASCII_FILE_VIEWER) || (vobj->type == ASCII_FILES_VIEWER)) {
      ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
      if (is->isperm) {
        fprintf(fd,"Index set is permutation\n");
      }
      fprintf(fd,"Number of indices in set %d\n",n);
      for ( i=0; i<n; i++ ) {
        fprintf(fd,"%d %d\n",i,idx[i]);
      }
    }
  }
  return 0;
}
  
static struct _ISOps myops = { ISGetSize_General,
                               ISGetSize_General,
                               ISGetIndices_General,0,
                               ISInvertPermutation_General};
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
  PETSCHEADERCREATE(Nindex, _IS,IS_COOKIE,IS_SEQ,comm); 
  PLogObjectCreate(Nindex);
  sub            = (IS_General *) PETSCMALLOC(size); CHKPTRQ(sub);
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

/*@C
   ISAddStrideSeq - Adds additional entries to a sequential 
                    index set by a stride.

   Input Parameters:
.  n - the length of the new piece of the index set
.  first - the first additional entry
.  step - the step between entries
.  is - the index set

.keywords: IS, sequential, index set, create, adding to

.seealso: ISCreateStrideSeq(), ISCreate()
@*/
int ISAddStrideSeq(IS *is,int n,int first,int step)
{
  int        N, *old,size,min,max,*idx,i;
  IS         Newis;
  IS_General *sub;
  if (*is) PETSCVALIDHEADERSPECIFIC(*is,IS_COOKIE);

  PETSCHEADERCREATE(Newis, _IS,IS_COOKIE,IS_SEQ,MPI_COMM_SELF); 
  PLogObjectCreate(Newis);

  if (*is) {
    ISGetSize(*is,&N);
    ISGetIndices(*is,&old);
  }
  else {
    N = 0;
  }

  size = sizeof(IS_General) + n*sizeof(int) + N*sizeof(int);
  sub            = (IS_General *) PETSCMALLOC(size); CHKPTRQ(sub);
  PLogObjectMemory(Newis,size + sizeof(struct _IS));
  sub->idx = idx = (int *) (sub+1);
  sub->sorted    = 1;

  PetscMemcpy(sub->idx,old,N*sizeof(int));
  if (*is) {
    PLogObjectParent(Newis,*is);ISRestoreIndices(*is,&old); ISDestroy(*is);
  }
  for ( i=0; i<n; i++ ) {idx[N+i] = first + i*step;}

  sub->n         = n + N;
  for ( i=1; i<N+n; i++ ) {
    if (idx[i] < idx[i-1]) {sub->sorted = 0; break;}
  }
  if (n+N) {min = max = idx[0];} else {min = max = 0;}
  for ( i=1; i<n+N; i++ ) {
    if (idx[i] < min) min = idx[i];
    if (idx[i] > max) max = idx[i];
  }

  Newis->min     = min;
  Newis->max     = max;
  Newis->data    = (void *) sub;
  PetscMemcpy(&Newis->ops,&myops,sizeof(myops));
  Newis->destroy = ISDestroy_General;
  Newis->view    = ISView_General;
  Newis->isperm  = 0;
  *is = Newis; 
  return 0;
}
