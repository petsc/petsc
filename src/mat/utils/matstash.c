#include "vec/vecimpl.h"
#include "matimpl.h"

#define CHUNCKSIZE   100
/*
   This stash is currently used for all the parallel matrix implementations.
   Perhaps this code ultimately should be moved elsewhere.

   This is a simple minded stash. Do a linear search to determine if
   in stash, if not add to end.
*/

int StashInitialize_Private(Stash *stash)
{
  stash->nmax  = 0;
  stash->n     = 0;
  stash->array = 0;
  stash->idx   = 0;
  stash->idy   = 0;
  return 0;
}

int StashBuild_Private(Stash *stash)
{
  stash->nmax  = CHUNCKSIZE; /* completely arbitrary number */
  stash->n     = 0;
  stash->array = (Scalar *) PETSCMALLOC( stash->nmax*(2*sizeof(int) +
                            sizeof(Scalar))); CHKPTRQ(stash->array);
  stash->idx   = (int *) (stash->array + stash->nmax); CHKPTRQ(stash->idx);
  stash->idy   = (int *) (stash->idx + stash->nmax); CHKPTRQ(stash->idy);
  return 0;
}

int StashDestroy_Private(Stash *stash)
{
  stash->nmax = stash->n = 0;
  if (stash->array) {PETSCFREE(stash->array); stash->array = 0;}
  return 0;
}

int StashValues_Private(Stash *stash,int row,int n, int *idxn,
                        Scalar *values,InsertMode addv)
{
  int    i,j,N = stash->n,found,*n_idx, *n_idy;
  Scalar val,*n_array;

  for ( i=0; i<n; i++ ) {
    found = 0;
    val = *values++;
    for ( j=0; j<N; j++ ) {
      if ( stash->idx[j] == row && stash->idy[j] == idxn[i]) {
        /* found a match */
        if (addv == ADD_VALUES) stash->array[j] += val;
        else stash->array[j] = val;
        found = 1;
        break;
      }
    }
    if (!found) { /* not found so add to end */
      if ( stash->n == stash->nmax ) {
        /* allocate a larger stash */
        n_array = (Scalar *) PETSCMALLOC( (stash->nmax + CHUNCKSIZE)*(
                                     2*sizeof(int) + sizeof(Scalar)));
        CHKPTRQ(n_array);
        n_idx = (int *) (n_array + stash->nmax + CHUNCKSIZE);
        n_idy = (int *) (n_idx + stash->nmax + CHUNCKSIZE);
        PETSCMEMCPY(n_array,stash->array,stash->nmax*sizeof(Scalar));
        PETSCMEMCPY(n_idx,stash->idx,stash->nmax*sizeof(int));
        PETSCMEMCPY(n_idy,stash->idy,stash->nmax*sizeof(int));
        if (stash->array) PETSCFREE(stash->array);
        stash->array = n_array; stash->idx = n_idx; stash->idy = n_idy;
        stash->nmax += CHUNCKSIZE;
      }
      stash->array[stash->n]   = val;
      stash->idx[stash->n]     = row;
      stash->idy[stash->n++]   = idxn[i];
    }
  }
  return 0;
}
