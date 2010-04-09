#define PETSCVEC_DLL

/*  
   Defines the abstract operations on index sets, i.e. the public interface. 
*/
#include "private/isimpl.h"      /*I "petscis.h" I*/

/* Logging support */
PetscCookie PETSCVEC_DLLEXPORT IS_COOKIE;

#undef __FUNCT__  
#define __FUNCT__ "ISIdentity" 
/*@
   ISIdentity - Determines whether index set is the identity mapping.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  ident - PETSC_TRUE if an identity, else PETSC_FALSE

   Level: intermediate

   Concepts: identity mapping
   Concepts: index sets^is identity

.seealso: ISSetIdentity()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISIdentity(IS is,PetscTruth *ident)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(ident,2);
  *ident = is->isidentity;
  if (*ident) PetscFunctionReturn(0);
  if (is->ops->identity) {
    ierr = (*is->ops->identity)(is,ident);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSetIdentity" 
/*@
   ISSetIdentity - Informs the index set that it is an identity.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Level: intermediate

   Concepts: identity mapping
   Concepts: index sets^is identity

.seealso: ISIdentity()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISSetIdentity(IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  is->isidentity = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISPermutation" 
/*@
   ISPermutation - PETSC_TRUE or PETSC_FALSE depending on whether the 
   index set has been declared to be a permutation.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  perm - PETSC_TRUE if a permutation, else PETSC_FALSE

   Level: intermediate

  Concepts: permutation
  Concepts: index sets^is permutation

.seealso: ISSetPermutation()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISPermutation(IS is,PetscTruth *perm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(perm,2);
  *perm = (PetscTruth) is->isperm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSetPermutation" 
/*@
   ISSetPermutation - Informs the index set that it is a permutation.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Level: intermediate

  Concepts: permutation
  Concepts: index sets^permutation

   The debug version of the libraries (config/configure.py --with-debugging=1) checks if the 
  index set is actually a permutation. The optimized version just believes you.

.seealso: ISPermutation()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISSetPermutation(IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
#if defined(PETSC_USE_DEBUG)
  {
    PetscMPIInt    size;
    PetscErrorCode ierr;

    ierr = MPI_Comm_size(((PetscObject)is)->comm,&size);CHKERRQ(ierr);
    if (size == 1) {
      PetscInt       i,n,*idx;
      const PetscInt *iidx;
    
      ierr = ISGetSize(is,&n);CHKERRQ(ierr);
      ierr = PetscMalloc(n*sizeof(PetscInt),&idx);CHKERRQ(ierr);
      ierr = ISGetIndices(is,&iidx);CHKERRQ(ierr);
      ierr = PetscMemcpy(idx,iidx,n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscSortInt(n,idx);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        if (idx[i] != i) SETERRQ(PETSC_ERR_ARG_WRONG,"Index set is not a permutation");
      }
      ierr = PetscFree(idx);CHKERRQ(ierr);
    }
  }
#endif
  is->isperm = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISDestroy" 
/*@
   ISDestroy - Destroys an index set.

   Collective on IS

   Input Parameters:
.  is - the index set

   Level: beginner

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlocked()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISDestroy(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  if (--((PetscObject)is)->refct > 0) PetscFunctionReturn(0);
  if (is->ops->destroy) {
    ierr = (*is->ops->destroy)(is);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISInvertPermutation" 
/*@
   ISInvertPermutation - Creates a new permutation that is the inverse of 
                         a given permutation.

   Collective on IS

   Input Parameter:
+  is - the index set
-  nlocal - number of indices on this processor in result (ignored for 1 proccessor) or
            use PETSC_DECIDE

   Output Parameter:
.  isout - the inverse permutation

   Level: intermediate

   Notes: For parallel index sets this does the complete parallel permutation, but the 
    code is not efficient for huge index sets (10,000,000 indices).

   Concepts: inverse permutation
   Concepts: permutation^inverse
   Concepts: index sets^inverting
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISInvertPermutation(IS is,PetscInt nlocal,IS *isout)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(isout,3);
  if (!is->isperm) SETERRQ(PETSC_ERR_ARG_WRONG,"Not a permutation, must call ISSetPermutation() on the IS first");
  ierr = (*is->ops->invertpermutation)(is,nlocal,isout);CHKERRQ(ierr);
  ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetSize" 
/*@
   ISGetSize - Returns the global length of an index set. 

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the global size

   Level: beginner

   Concepts: size^of index set
   Concepts: index sets^size

@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISGetSize(IS is,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(size,2);
  ierr = (*is->ops->getsize)(is,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetLocalSize" 
/*@
   ISGetLocalSize - Returns the local (processor) length of an index set. 

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the local size

   Level: beginner

   Concepts: size^of index set
   Concepts: local size^of index set
   Concepts: index sets^local size
  
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISGetLocalSize(IS is,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(size,2);
  ierr = (*is->ops->getlocalsize)(is,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetIndices" 
/*@C
   ISGetIndices - Returns a pointer to the indices.  The user should call 
   ISRestoreIndices() after having looked at the indices.  The user should 
   NOT change the indices.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  ptr - the location to put the pointer to the indices

   Fortran Note:
   This routine is used differently from Fortran
$    IS          is
$    integer     is_array(1)
$    PetscOffset i_is
$    int         ierr
$       call ISGetIndices(is,is_array,i_is,ierr)
$
$   Access first local entry in list
$      value = is_array(i_is + 1)
$
$      ...... other code
$       call ISRestoreIndices(is,is_array,i_is,ierr)

   See the Fortran chapter of the users manual and 
   petsc/src/is/examples/[tutorials,tests] for details.

   Level: intermediate

   Concepts: index sets^getting indices
   Concepts: indices of index set

.seealso: ISRestoreIndices(), ISGetIndicesF90()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISGetIndices(IS is,const PetscInt *ptr[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(ptr,2);
  ierr = (*is->ops->getindices)(is,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "ISRestoreIndices" 
/*@C
   ISRestoreIndices - Restores an index set to a usable state after a call 
                      to ISGetIndices().

   Not Collective

   Input Parameters:
+  is - the index set
-  ptr - the pointer obtained by ISGetIndices()

   Fortran Note:
   This routine is used differently from Fortran
$    IS          is
$    integer     is_array(1)
$    PetscOffset i_is
$    int         ierr
$       call ISGetIndices(is,is_array,i_is,ierr)
$
$   Access first local entry in list
$      value = is_array(i_is + 1)
$
$      ...... other code
$       call ISRestoreIndices(is,is_array,i_is,ierr)

   See the Fortran chapter of the users manual and 
   petsc/src/is/examples/[tutorials,tests] for details.

   Level: intermediate

.seealso: ISGetIndices(), ISRestoreIndicesF90()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISRestoreIndices(IS is,const PetscInt *ptr[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(ptr,2);
  if (is->ops->restoreindices) {
    ierr = (*is->ops->restoreindices)(is,ptr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISView" 
/*@C
   ISView - Displays an index set.

   Collective on IS

   Input Parameters:
+  is - the index set
-  viewer - viewer used to display the set, for example PETSC_VIEWER_STDOUT_SELF.

   Level: intermediate

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISView(IS is,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)is)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  PetscCheckSameComm(is,1,viewer,2);
  
  ierr = (*is->ops->view)(is,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSort" 
/*@
   ISSort - Sorts the indices of an index set.

   Collective on IS

   Input Parameters:
.  is - the index set

   Level: intermediate

   Concepts: index sets^sorting
   Concepts: sorting^index set

.seealso: ISSorted()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISSort(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  ierr = (*is->ops->sortindices)(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSorted" 
/*@
   ISSorted - Checks the indices to determine whether they have been sorted.

   Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  flg - output flag, either PETSC_TRUE if the index set is sorted, 
         or PETSC_FALSE otherwise.

   Notes: For parallel IS objects this only indicates if the local part of the IS
          is sorted. So some processors may return PETSC_TRUE while others may 
          return PETSC_FALSE.

   Level: intermediate

.seealso: ISSort()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISSorted(IS is,PetscTruth *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidIntPointer(flg,2);
  ierr = (*is->ops->sorted)(is,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISDuplicate" 
/*@
   ISDuplicate - Creates a duplicate copy of an index set.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  isnew - the copy of the index set

   Notes:
   ISDuplicate() does not copy the index set, but rather allocates storage
   for the new one.  Use ISCopy() to copy an index set.

   Level: beginner

   Concepts: index sets^duplicating

.seealso: ISCreateGeneral(), ISCopy()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISDuplicate(IS is,IS *newIS)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidPointer(newIS,2);
  ierr = (*is->ops->duplicate)(is,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISCopy"
/*@
   ISCopy - Copies an index set.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  isy - the copy of the index set

   Level: beginner

   Concepts: index sets^copying

.seealso: ISDuplicate()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT ISCopy(IS is,IS isy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE,1);
  PetscValidHeaderSpecific(isy,IS_COOKIE,2);
  PetscCheckSameComm(is,1,isy,2);
  if (is == isy) PetscFunctionReturn(0);
  ierr = (*is->ops->copy)(is,isy);CHKERRQ(ierr);
  isy->isperm     = is->isperm;
  isy->max        = is->max;
  isy->min        = is->min;
  isy->isidentity = is->isidentity;
  PetscFunctionReturn(0);
}

/*MC
    ISGetIndicesF90 - Accesses the elements of an index set from Fortran90.
    The users should call ISRestoreIndicesF90() after having looked at the
    indices.  The user should NOT change the indices.

    Synopsis:
    ISGetIndicesF90(IS x,{integer, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameter:
.   x - index set

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code

    Example of Usage: 
.vb
    PetscScalar, pointer xx_v(:)
    ....
    call ISGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISRestoreIndicesF90(x,xx_v,ierr)
.ve

    Notes:
    Not yet supported for all F90 compilers.

    Level: intermediate

.seealso:  ISRestoreIndicesF90(), ISGetIndices(), ISRestoreIndices()

  Concepts: index sets^getting indices in f90
  Concepts: indices of index set in f90

M*/

/*MC
    ISRestoreIndicesF90 - Restores an index set to a usable state after
    a call to ISGetIndicesF90().

    Synopsis:
    ISRestoreIndicesF90(IS x,{integer, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameters:
.   x - index set
.   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code


    Example of Usage: 
.vb
    PetscScalar, pointer xx_v(:)
    ....
    call ISGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISRestoreIndicesF90(x,xx_v,ierr)
.ve
   
    Notes:
    Not yet supported for all F90 compilers.

    Level: intermediate

.seealso:  ISGetIndicesF90(), ISGetIndices(), ISRestoreIndices()

M*/

/*MC
    ISBlockGetIndicesF90 - Accesses the elements of an index set from Fortran90.
    The users should call ISBlockRestoreIndicesF90() after having looked at the
    indices.  The user should NOT change the indices.

    Synopsis:
    ISBlockGetIndicesF90(IS x,{integer, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameter:
.   x - index set

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code
    Example of Usage: 
.vb
    PetscScalar, pointer xx_v(:)
    ....
    call ISBlockGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISBlockRestoreIndicesF90(x,xx_v,ierr)
.ve

    Notes:
    Not yet supported for all F90 compilers

    Level: intermediate

.seealso:  ISBlockRestoreIndicesF90(), ISGetIndices(), ISRestoreIndices(),
           ISRestoreIndices()

  Concepts: index sets^getting block indices in f90
  Concepts: indices of index set in f90
  Concepts: block^ indices of index set in f90

M*/

/*MC
    ISBlockRestoreIndicesF90 - Restores an index set to a usable state after
    a call to ISBlockGetIndicesF90().

    Synopsis:
    ISBlockRestoreIndicesF90(IS x,{integer, pointer :: xx_v(:)},integer ierr)

    Not Collective

    Input Parameters:
+   x - index set
-   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Example of Usage: 
.vb
    PetscScalar, pointer xx_v(:)
    ....
    call ISBlockGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISBlockRestoreIndicesF90(x,xx_v,ierr)
.ve
   
    Notes:
    Not yet supported for all F90 compilers

    Level: intermediate

.seealso:  ISBlockGetIndicesF90(), ISGetIndices(), ISRestoreIndices(), ISRestoreIndicesF90()

M*/


