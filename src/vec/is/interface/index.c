#ifndef lint
static char vcid[] = "$Id: index.c,v 1.16 1995/08/22 19:28:47 curfman Exp bsmith $";
#endif
/*  
   Defines the abstract operations on index sets 
*/
#include "isimpl.h"      /*I "is.h" I*/

/*@
   ISIsPermutation - Returns 1 if the index set is a permutation;
                     0 if not; -1 on error.

   Input Parmeters:
.  is - the index set

.keywords: IS, index set, permutation

.seealso: ISSetPermutation()
@*/
int ISIsPermutation(IS is)
{
  if (!is) {SETERRQ(-1,"ISIsPermutation: Null pointer");}
  if (is->cookie != IS_COOKIE) {SETERRQ(-1,"ISIsPermutation: Not indexset");}
  return is->isperm;
}
/*@
   ISSetPermutation - Informs the index set that it is a permutation.

   Input Parmeters:
.  is - the index set

.keywords: IS, index set, permutation

.seealso: ISIsPermutation()
@*/
int ISSetPermutation(IS is)
{
  PETSCVALIDHEADERSPECIFIC(is,IS_COOKIE);
  is->isperm = 1;
  return 0;
}

/*@C
   ISDestroy - Destroys an index set.

   Input Parameters:
.  is - the index set

.keywords: IS, index set, destroy

.seealso: ISCreateSequential(), ISCreateStrideSequential()
@*/
int ISDestroy(IS is)
{
  PETSCVALIDHEADERSPECIFIC(is,IS_COOKIE);
  return (*is->destroy)((PetscObject) is);
}

/*@
   ISInvertPermutation - Creates a new permutation that is the inverse of 
                         a given permutation.

   Input Parameter:
.  is - the index set

   Output Parameter:
.  isout - the inverse permutation

.keywords: IS, index set, invert, inverse, permutation
@*/
int ISInvertPermutation(IS is,IS *isout)
{
  PETSCVALIDHEADERSPECIFIC(is,IS_COOKIE);
  if (!is->isperm) SETERRQ(1,"ISInvertPermutation: not permutation");
  return (*is->ops->invertpermutation)(is,isout);
}

/*@
   ISGetSize - Returns the global length of an index set. 

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the global size

.keywords: IS, index set, get, global, size

.seealso: ISGetLocalSize()
@*/
int ISGetSize(IS is,int *size)
{
  PETSCVALIDHEADERSPECIFIC(is,IS_COOKIE);
  return (*is->ops->getsize)(is,size);
}
/*@
   ISGetLocalSize - Returns local length of an index set.

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the local size

.keywords: IS, index set, get, local, size

.seealso: ISGetSize()
@*/
int ISGetLocalSize(IS is,int *size)
{
  PETSCVALIDHEADERSPECIFIC(is,IS_COOKIE);
  return (*is->ops->getlocalsize)(is,size);
}

/*@C
   ISGetIndices - Returns a pointer to the indices.  The user should call 
   ISRestoreIndices() after having looked at the indices.  The user should 
   NOT change the indices.

   Input Parameter:
.  is - the index set

   Output Parameter:
.  ptr - the location to put the pointer to the indices

   Notes:
   In a parallel enviroment this probably points to only the indices that 
   are local to a particular processor.

.keywords: IS, index set, get, indices

.seealso: ISRestoreIndices()
@*/
int ISGetIndices(IS is,int **ptr)
{
  PETSCVALIDHEADERSPECIFIC(is,IS_COOKIE);
  return (*is->ops->getindices)(is,ptr);
} 

/*@C
   ISRestoreIndices - Restores an index set to a usable state after a call 
   to ISGetIndices().

   Input Parameters:
.  is - the index set
.  ptr - the pointer obtained by ISGetIndices()

.keywords: IS, index set, restore, indices

.seealso: ISGetIndices()
@*/
int ISRestoreIndices(IS is,int **ptr)
{
  PETSCVALIDHEADERSPECIFIC(is,IS_COOKIE);
  if (is->ops->restoreindices) return (*is->ops->restoreindices)(is,ptr);
  else return 0;
}

/*@
   ISView - Displays an index set.

   Input Parameters:
.  is - the index set
.  viewer - viewer used to display the set, for example STDOUT_VIEWER_SELF.

.keywords: IS, index set, restore, indices

.seealso: ViewerFileOpen()
@*/
int ISView(IS is, Viewer viewer)
{
  PETSCVALIDHEADERSPECIFIC(is,IS_COOKIE);
  return (*is->view)((PetscObject)is,viewer);
}
