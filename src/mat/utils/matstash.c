#ifndef lint
static char vcid[] = "$Id: stash.c,v 1.12 1996/11/19 16:31:54 bsmith Exp balay $";
#endif

#include "src/vec/vecimpl.h"
#include "src/mat/matimpl.h"

#define CHUNCKSIZE   5000
/*
   This stash is currently used for all the parallel matrix implementations.
   The stash is where elements of a matrix destined to be stored on other 
   processors are kept until matrix assembly is done.

   This is a simple minded stash. Simply add entry to end of stash.
*/

#undef __FUNCTION__  
#define __FUNCTION__ "StashInitialize_Private"
int StashInitialize_Private(Stash *stash)
{
  stash->nmax  = 0;
  stash->n     = 0;
  stash->array = 0;
  stash->idx   = 0;
  stash->idy   = 0;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "StashBuild_Private"
int StashBuild_Private(Stash *stash)
{
  stash->nmax  = CHUNCKSIZE; /* completely arbitrary number */
  stash->n     = 0;
  stash->array = (Scalar *) PetscMalloc( stash->nmax*(2*sizeof(int) +
                            sizeof(Scalar))); CHKPTRQ(stash->array);
  stash->idx   = (int *) (stash->array + stash->nmax); CHKPTRQ(stash->idx);
  stash->idy   = (int *) (stash->idx + stash->nmax); CHKPTRQ(stash->idy);
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "StashDestroy_Private"
int StashDestroy_Private(Stash *stash)
{
  stash->nmax = stash->n = 0;
  if (stash->array) {PetscFree(stash->array); stash->array = 0;}
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "StashInfo_Private"
int StashInfo_Private(Stash *stash)
{
  PLogInfo(0,"Stash size %d\n",stash->n);
  return 0;
}

/* 
    Should do this properly. With a sorted array.
*/
#undef __FUNCTION__  
#define __FUNCTION__ "StashValues_Private"
int StashValues_Private(Stash *stash,int row,int n, int *idxn,Scalar *values,InsertMode addv)
{
  int    i, found, *n_idx, *n_idy; 
  Scalar val, *n_array;

  for ( i=0; i<n; i++ ) {
    found = 0;
    val = *values++;
    if (!found) { /* not found so add to end */
      if ( stash->n == stash->nmax ) {
        /* allocate a larger stash */
        n_array = (Scalar *) PetscMalloc( (stash->nmax + CHUNCKSIZE)*(
                                     2*sizeof(int)+sizeof(Scalar)));CHKPTRQ(n_array);
        n_idx = (int *) (n_array + stash->nmax + CHUNCKSIZE);
        n_idy = (int *) (n_idx + stash->nmax + CHUNCKSIZE);
        PetscMemcpy(n_array,stash->array,stash->nmax*sizeof(Scalar));
        PetscMemcpy(n_idx,stash->idx,stash->nmax*sizeof(int));
        PetscMemcpy(n_idy,stash->idy,stash->nmax*sizeof(int));
        if (stash->array) PetscFree(stash->array);
        stash->array = n_array; stash->idx = n_idx; stash->idy = n_idy;
        stash->nmax += CHUNCKSIZE;
      }
      stash->array[stash->n] = val;
      stash->idx[stash->n]   = row;
      stash->idy[stash->n++] = idxn[i];
    }
  }
  return 0;
}



