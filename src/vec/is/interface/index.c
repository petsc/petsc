
/*  
   Defines the abstract operations on index sets 
*/
#include "petsc.h"
#include "isimpl.h"


/*@
    ISDestroy - Destroy an index set.

  Input Parameters:
.  is - the index set

@*/
int ISDestroy(is)
IS is;
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->destroy)(is);
}

/*@
    ISGetSize - returns length of an index set. In a parallel 
     environment this returns the entire size. Use ISGetLocalSize()
     for length of local piece.

  Input Parameters:
.  is - the index set

  Output Parameters:
.  returns the number of indices in index set.
@*/
int ISGetSize(is,size)
IS  is;
int *size;
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->ops->size)(is->data,size);
}
/*@
    ISGetLocalSize - returns length of an index set. In a parallel 
     environment this returns the size in local memory. Use
     ISGetLocal() for length of total.

  Input Parameters:
.  is - the index set

  Output Parameters:
.  returns the number of indices in index set.
@*/
int ISGetLocalSize(is,size)
IS is;
int *size;
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->ops->localsize)(is->data,size);
}

/*@
    ISGetPosition - determines if an integer is in a index set
         and returns its relative position. Returns -1 if not in.

  Input Parameters:
.   is - the index set
.   i - the index to check

  Output Parameters:
.   returns the relative position or -1
@*/
int ISGetPosition(is,i,pos)
IS is;
int i,*pos;
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->ops->position)(is->data,i,pos);
}

/*@ 

    ISGetIndices - returns a pointer to the indices.
                   You should call ISRestoreIndices()
                   after you have looked at the indices. 
                   You should not change the indices.

  Input Parameters:
,  is - the index set

  Output Parameters:
.   returns a pointer to the indices

    Note: in a parallel enviroment this probably points to 
          only the local indices to that processor.
@*/
int ISGetIndices(is,ptr)
IS is;
int **ptr;
{
  VALIDHEADER(is,IS_COOKIE);
  return (*is->ops->indices)(is->data,ptr);
} 

/*@ 

    ISRestroreIndices - See ISGetIndices

  Input Parameters:
,  is - the index set
.  ptr - the pointer obtained with ISGetIndices

@*/
int ISRestoreIndices(is,ptr)
IS  is;
int **ptr;
{
  VALIDHEADER(is,IS_COOKIE);
  if (is->ops->restoreindices) return (*is->ops->restoreindices)(is->data,ptr);
  else return 0;
}
