
/*  
   Defines the abstract operations on index sets 
*/
#include "isimpl.h"      /*I "is.h" I*/


/*@
    ISDestroy - Destroy an index set.

  Input Parameters:
.  is - the index set

@*/
int ISDestroy(IS is)
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->destroy)((PetscObject) is);
}

/*@
    ISGetSize - returns length of an index set. In a parallel 
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
    ISGetLocalSize - returns length of an index set. In a parallel 
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
    ISGetPosition - determines if an integer is in a index set
         and returns its relative position. Returns -1 if not in.

  Input Parameters:
.   is - the index set
.   i - the index to check

  Output Parameters:
.   pos - relative position in list or negative -1 if not in list.

@*/
int ISGetPosition(IS is,int i,int *pos)
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->ops->position)(is,i,pos);
}

/*@ 

    ISGetIndices - returns a pointer to the indices.
                   You should call ISRestoreIndices()
                   after you have looked at the indices. 
                   You should not change the indices.

  Input Parameters:
,  is - the index set

  Output Parameters:
.  ptr - the location to put the pointer to the indices

    Note: in a parallel enviroment this probably points to 
          only the local indices to that processor.
@*/
int ISGetIndices(IS is,int **ptr)
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->ops->indices)(is,ptr);
} 

/*@ 

    ISRestoreIndices - See ISGetIndices. Restores and index to usable
                       state.

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
