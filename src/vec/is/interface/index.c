#ifndef lint
static char vcid[] = "$Id: index.c,v 1.6 1995/03/06 04:10:56 bsmith Exp curfman $";
#endif
/*  
   Defines the abstract operations on index sets 
*/
#include "isimpl.h"      /*I "is.h" I*/

/*@
    ISIsPermutation - Returns 1 if the index set is a permutation;
                      0 if not, -1 on error.

  InputParmeters:
.   is - the index set
@*/
int ISIsPermutation(IS is)
{
  if (!is) {SETERR(-1,"Null pointer");}
  if (is->cookie != IS_COOKIE) {SETERR(-1,"Not indexset");}
  return is->isperm;
}
/*@
    ISSetPermutation - Informs the index set that it is a permutation.

  InputParmeters:
.   is - the index set
@*/
int ISSetPermutation(IS is)
{
  VALIDHEADER(is,IS_COOKIE);
  is->isperm = 1;
  return 0;
}

/*@
    ISDestroy - Destroys an index set.

  Input Parameters:
.  is - the index set

@*/
int ISDestroy(IS is)
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->destroy)((PetscObject) is);
}

/*@
    ISInvertPermutation - Creates a new permutation that is the inverse of 
                          a given permutation.

  Input Parameters:
.  is - the index set

  Putput Parameters:
.  isout - the inverse permutation.
@*/
int ISInvertPermutation(IS is,IS *isout)
{
  VALIDHEADER(is,IS_COOKIE);
  if (!is->isperm) SETERR(1,"Cannot invert nonpermutation");
  return (*is->ops->invert)(is,isout);
}

/*@
    ISGetSize - Returns length of an index set. In a parallel 
     environment this returns the entire size. Use ISGetLocalSize()
     for length of local piece.

  Input Parameters:
.  is - the index set

  Output Parameters:
.  size - the size.
@*/
int ISGetSize(IS is,int *size)
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->ops->size)(is,size);
}
/*@
    ISGetLocalSize - Returns length of an index set. In a parallel 
     environment this returns the size in local memory. Use
     ISGetLocal() for length of total.

  Input Parameters:
.  is - the index set

  Output Parameters:
.  size - the local size.
@*/
int ISGetLocalSize(IS is,int *size)
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->ops->localsize)(is,size);
}


/*@ 

    ISGetIndices - Returns a pointer to the indices.
                   You should call ISRestoreIndices()
                   after you have looked at the indices. 
                   You should not change the indices.

  Input Parameters:
.  is - the index set

  Output Parameters:
.  ptr - the location to put the pointer to the indices

  Keywords: index set, indices

  Note:
  In a parallel enviroment this probably points to 
          only the local indices to that processor.
@*/
int ISGetIndices(IS is,int **ptr)
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->ops->indices)(is,ptr);
} 

/*@ 

    ISRestoreIndices - See ISGetIndices(). Restores index set to a usable
                       state after call to ISGetIndices().

  Input Parameters:
,  is - the index set
.  ptr - the pointer obtained with ISGetIndices

@*/
int ISRestoreIndices(IS is,int **ptr)
{
  VALIDHEADER(is,IS_COOKIE);
  if (is->ops->restoreindices) return (*is->ops->restoreindices)(is,ptr);
  else return 0;
}

/*@
   ISView - Displays an index set.

  Input Parameters:
.  is - the index set
.  viewer - location to display set
@*/
int ISView(IS is, Viewer viewer)
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->view)((PetscObject)is,viewer);
}
