#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: stash.c,v 1.19 1998/01/06 20:11:15 bsmith Exp balay $";
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

#undef __FUNC__  
#define __FUNC__ "StashInitialize_Private"
int StashInitialize_Private(Stash *stash)
{
  PetscFunctionBegin;
  stash->nmax    = 0;
  stash->oldnmax = 0;
  stash->n       = 0;
  stash->array   = 0;
  stash->idx     = 0;
  stash->idy     = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashBuild_Private"
int StashBuild_Private(Stash *stash)
{
  int ierr,flg,size,max;

  PetscFunctionBegin;
  ierr = OptionsGetInt(PETSC_NULL,"-stash_initial_size",&max,&flg);CHKERRQ(ierr);
  if (flg) {
    stash->nmax    = max; 
    stash->oldnmax = max;
  } else {
    stash->nmax    = CHUNCKSIZE; /* completely arbitrary number */
    stash->oldnmax = CHUNCKSIZE;
  }
  stash->n       = 0;
  stash->array   = (Scalar *) PetscMalloc( stash->nmax*(2*sizeof(int) +
                            sizeof(Scalar))); CHKPTRQ(stash->array);
  stash->idx     = (int *) (stash->array + stash->nmax); CHKPTRQ(stash->idx);
  stash->idy     = (int *) (stash->idx + stash->nmax); CHKPTRQ(stash->idy);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashDestroy_Private"
int StashDestroy_Private(Stash *stash)
{
  PetscFunctionBegin;
  /* Now update nmaxold to be app 10% more than nmax, this way the
     wastage of space is reduced the next time this stash is used */
  stash->oldnmax = (int)stash->nmax * 1.1;
  stash->nmax    = 0;
  stash->n       = 0;
  if (stash->array) {PetscFree(stash->array); stash->array = 0;}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "StashInfo_Private"
int StashInfo_Private(Stash *stash)
{
  PetscFunctionBegin;
  PLogInfo(0,"StashInfo_Private:Stash size %d\n",stash->n);
  PetscFunctionReturn(0);
}

/* 
    Should do this properly. With a sorted array.
*/
#undef __FUNC__  
#define __FUNC__ "StashValues_Private"
int StashValues_Private(Stash *stash,int row,int n, int *idxn,Scalar *values,InsertMode addv)
{
  int    i, found, *n_idx, *n_idy,newnmax; 
  Scalar val, *n_array;

  PetscFunctionBegin;
  for ( i=0; i<n; i++ ) {
    found = 0;
    val = *values++;
    if (!found) { /* not found so add to end */
      if ( stash->n == stash->nmax ) {
        /* allocate a larger stash */
        if (stash->nmax == 0) newnmax = stash->oldnmax;
        else                  newnmax = stash->nmax *2;

        n_array = (Scalar *)PetscMalloc((newnmax)*(2*sizeof(int)+sizeof(Scalar)));CHKPTRQ(n_array);
        n_idx = (int *) (n_array + newnmax);
        n_idy = (int *) (n_idx + newnmax);
        PetscMemcpy(n_array,stash->array,stash->nmax*sizeof(Scalar));
        PetscMemcpy(n_idx,stash->idx,stash->nmax*sizeof(int));
        PetscMemcpy(n_idy,stash->idy,stash->nmax*sizeof(int));
        if (stash->array) PetscFree(stash->array);
        stash->array   = n_array; 
        stash->idx     = n_idx; 
        stash->idy     = n_idy;
        stash->nmax    = newnmax;
        stash->oldnmax = newnmax;
      }
      stash->array[stash->n] = val;
      stash->idx[stash->n]   = row;
      stash->idy[stash->n++] = idxn[i];
    }
  }
  PetscFunctionReturn(0);
}
