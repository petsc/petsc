
/*  
   Defines the abstract operations on index sets, i.e. the public interface. 
*/
#include <petsc-private/isimpl.h>      /*I "petscis.h" I*/

/* Logging support */
PetscClassId  IS_CLASSID;

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
PetscErrorCode  ISIdentity(IS is,PetscBool  *ident)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
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

   Logically Collective on IS

   Input Parmeters:
.  is - the index set

   Level: intermediate

   Concepts: identity mapping
   Concepts: index sets^is identity

.seealso: ISIdentity()
@*/
PetscErrorCode  ISSetIdentity(IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  is->isidentity = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISContiguousLocal"
/*@
   ISContiguousLocal - Locates an index set with contiguous range within a global range, if possible

   Not Collective

   Input Parmeters:
+  is - the index set
.  gstart - global start
.  gend - global end

   Output Parameters:
+  start - start of contiguous block, as an offset from gstart
-  contig - PETSC_TRUE if the index set refers to contiguous entries on this process, else PETSC_FALSE

   Level: developer

   Concepts: index sets^is contiguous

.seealso: ISGetLocalSize(), VecGetOwnershipRange()
@*/
PetscErrorCode  ISContiguousLocal(IS is,PetscInt gstart,PetscInt gend,PetscInt *start,PetscBool *contig)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidIntPointer(start,5);
  PetscValidIntPointer(contig,5);
  if (is->ops->contiguous) {
    ierr = (*is->ops->contiguous)(is,gstart,gend,start,contig);CHKERRQ(ierr);
  } else {
    *start = -1;
    *contig = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISPermutation" 
/*@
   ISPermutation - PETSC_TRUE or PETSC_FALSE depending on whether the 
   index set has been declared to be a permutation.

   Logically Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  perm - PETSC_TRUE if a permutation, else PETSC_FALSE

   Level: intermediate

  Concepts: permutation
  Concepts: index sets^is permutation

.seealso: ISSetPermutation()
@*/
PetscErrorCode  ISPermutation(IS is,PetscBool  *perm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidIntPointer(perm,2);
  *perm = (PetscBool) is->isperm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSetPermutation" 
/*@
   ISSetPermutation - Informs the index set that it is a permutation.

   Logically Collective on IS

   Input Parmeters:
.  is - the index set

   Level: intermediate

  Concepts: permutation
  Concepts: index sets^permutation

   The debug version of the libraries (./configure --with-debugging=1) checks if the 
  index set is actually a permutation. The optimized version just believes you.

.seealso: ISPermutation()
@*/
PetscErrorCode  ISSetPermutation(IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
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
        if (idx[i] != i) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Index set is not a permutation");
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
PetscErrorCode  ISDestroy(IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*is) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*is),IS_CLASSID,1);
  if (--((PetscObject)(*is))->refct > 0) {*is = 0; PetscFunctionReturn(0);}
  if ((*is)->complement) {
    PetscInt refcnt;
    ierr = PetscObjectGetReference((PetscObject)((*is)->complement), &refcnt); CHKERRQ(ierr);
    if (refcnt > 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Nonlocal IS has not been restored");
    ierr = ISDestroy(&(*is)->complement); CHKERRQ(ierr);
  }
  if ((*is)->ops->destroy) {
    ierr = (*(*is)->ops->destroy)(*is);CHKERRQ(ierr);
  }
  /* Destroy local representations of offproc data. */
  ierr = PetscFree((*is)->total); CHKERRQ(ierr);
  ierr = PetscFree((*is)->nonlocal); CHKERRQ(ierr);
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
PetscErrorCode  ISInvertPermutation(IS is,PetscInt nlocal,IS *isout)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(isout,3);
  if (!is->isperm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a permutation, must call ISSetPermutation() on the IS first");
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
PetscErrorCode  ISGetSize(IS is,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
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
PetscErrorCode  ISGetLocalSize(IS is,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
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
PetscErrorCode  ISGetIndices(IS is,const PetscInt *ptr[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
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
PetscErrorCode  ISRestoreIndices(IS is,const PetscInt *ptr[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(ptr,2);
  if (is->ops->restoreindices) {
    ierr = (*is->ops->restoreindices)(is,ptr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGatherTotal_Private" 
static PetscErrorCode ISGatherTotal_Private(IS is)
{
  PetscErrorCode ierr;
  PetscInt       i,n,N;
  const PetscInt *lindices;
  MPI_Comm       comm;
  PetscMPIInt    rank,size,*sizes = PETSC_NULL,*offsets = PETSC_NULL,nn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = PetscMalloc2(size,PetscMPIInt,&sizes,size,PetscMPIInt,&offsets);CHKERRQ(ierr);
  
  nn   = PetscMPIIntCast(n);
  ierr = MPI_Allgather(&nn,1,MPIU_INT,sizes,1,MPIU_INT,comm);CHKERRQ(ierr);
  offsets[0] = 0;
  for (i=1;i<size; ++i) offsets[i] = offsets[i-1] + sizes[i-1];
  N = offsets[size-1] + sizes[size-1];
  
  ierr = PetscMalloc(N*sizeof(PetscInt),&(is->total));CHKERRQ(ierr);
  ierr = ISGetIndices(is,&lindices);CHKERRQ(ierr);
  ierr = MPI_Allgatherv((void*)lindices,nn,MPIU_INT,is->total,sizes,offsets,MPIU_INT,comm);CHKERRQ(ierr); 
  ierr = ISRestoreIndices(is,&lindices);CHKERRQ(ierr);
  is->local_offset = offsets[rank];
  ierr = PetscFree2(sizes,offsets);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetTotalIndices" 
/*@C
   ISGetTotalIndices - Retrieve an array containing all indices across the communicator.

   Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  indices - total indices with rank 0 indices first, and so on; total array size is 
             the same as returned with ISGetSize().

   Level: intermediate

   Notes: this is potentially nonscalable, but depends on the size of the total index set
     and the size of the communicator. This may be feasible for index sets defined on
     subcommunicators, such that the set size does not grow with PETSC_WORLD_COMM.
     Note also that there is no way to tell where the local part of the indices starts
     (use ISGetIndices() and ISGetNonlocalIndices() to retrieve just the local and just
      the nonlocal part (complement), respectively).

   Concepts: index sets^getting nonlocal indices
.seealso: ISRestoreTotalIndices(), ISGetNonlocalIndices(), ISGetSize()
@*/
PetscErrorCode ISGetTotalIndices(IS is, const PetscInt *indices[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  ierr = MPI_Comm_size(((PetscObject)is)->comm, &size); CHKERRQ(ierr);
  if(size == 1) {
    ierr = (*is->ops->getindices)(is,indices);CHKERRQ(ierr);
  }
  else {
    if(!is->total) {
      ierr = ISGatherTotal_Private(is); CHKERRQ(ierr);
    }
    *indices = is->total;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISRestoreTotalIndices" 
/*@C
   ISRestoreTotalIndices - Restore the index array obtained with ISGetTotalIndices().

   Not Collective.

   Input Parameter:
+  is - the index set
-  indices - index array; must be the array obtained with ISGetTotalIndices()

   Level: intermediate

   Concepts: index sets^getting nonlocal indices
   Concepts: index sets^restoring nonlocal indices
.seealso: ISRestoreTotalIndices(), ISGetNonlocalIndices()
@*/
PetscErrorCode  ISRestoreTotalIndices(IS is, const PetscInt *indices[])
{
  PetscErrorCode ierr;
  PetscMPIInt size;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  ierr = MPI_Comm_size(((PetscObject)is)->comm, &size); CHKERRQ(ierr);
  if(size == 1) {
    ierr = (*is->ops->restoreindices)(is,indices);CHKERRQ(ierr);
  }
  else {
    if(is->total != *indices) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index array pointer being restored does not point to the array obtained from the IS.");
    }
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "ISGetNonlocalIndices" 
/*@C
   ISGetNonlocalIndices - Retrieve an array of indices from remote processors
                       in this communicator.

   Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  indices - indices with rank 0 indices first, and so on,  omitting 
             the current rank.  Total number of indices is the difference
             total and local, obtained with ISGetSize() and ISGetLocalSize(),
	     respectively.

   Level: intermediate

   Notes: restore the indices using ISRestoreNonlocalIndices().   
          The same scalability considerations as those for ISGetTotalIndices
          apply here.

   Concepts: index sets^getting nonlocal indices
.seealso: ISGetTotalIndices(), ISRestoreNonlocalIndices(), ISGetSize(), ISGetLocalSize().
@*/
PetscErrorCode  ISGetNonlocalIndices(IS is, const PetscInt *indices[])
{
  PetscErrorCode ierr;
  PetscMPIInt size;
  PetscInt n, N;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  ierr = MPI_Comm_size(((PetscObject)is)->comm, &size); CHKERRQ(ierr);
  if(size == 1) {
      *indices = PETSC_NULL;
  }
  else {
    if(!is->total) {
      ierr = ISGatherTotal_Private(is); CHKERRQ(ierr);
    }
    ierr = ISGetLocalSize(is,&n); CHKERRQ(ierr);
    ierr = ISGetSize(is,&N);      CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*(N-n), &(is->nonlocal));   CHKERRQ(ierr);
    ierr = PetscMemcpy(is->nonlocal, is->total, sizeof(PetscInt)*is->local_offset); CHKERRQ(ierr);
    ierr = PetscMemcpy(is->nonlocal+is->local_offset, is->total+is->local_offset+n, sizeof(PetscInt)*(N - is->local_offset - n)); CHKERRQ(ierr);
    *indices = is->nonlocal;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISRestoreNonlocalIndices" 
/*@C
   ISRestoreTotalIndices - Restore the index array obtained with ISGetNonlocalIndices().

   Not Collective.

   Input Parameter:
+  is - the index set
-  indices - index array; must be the array obtained with ISGetNonlocalIndices()

   Level: intermediate

   Concepts: index sets^getting nonlocal indices
   Concepts: index sets^restoring nonlocal indices
.seealso: ISGetTotalIndices(), ISGetNonlocalIndices(), ISRestoreTotalIndices()
@*/
PetscErrorCode  ISRestoreNonlocalIndices(IS is, const PetscInt *indices[])
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  if(is->nonlocal != *indices) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index array pointer being restored does not point to the array obtained from the IS.");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetNonlocalIS" 
/*@
   ISGetNonlocalIS - Gather all nonlocal indices for this IS and present 
                     them as another sequential index set.  


   Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  complement - sequential IS with indices identical to the result of
                ISGetNonlocalIndices()

   Level: intermediate

   Notes: complement represents the result of ISGetNonlocalIndices as an IS.
          Therefore scalability issues similar to ISGetNonlocalIndices apply.
	  The resulting IS must be restored using ISRestoreNonlocalIS().

   Concepts: index sets^getting nonlocal indices
.seealso: ISGetNonlocalIndices(), ISRestoreNonlocalIndices(),  ISAllGather(), ISGetSize()
@*/
PetscErrorCode  ISGetNonlocalIS(IS is, IS *complement)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(complement,2);
  /* Check if the complement exists already. */
  if(is->complement) {
    *complement = is->complement;
    ierr = PetscObjectReference((PetscObject)(is->complement)); CHKERRQ(ierr);
  }
  else {
    PetscInt       N, n;
    const PetscInt *idx;
    ierr = ISGetSize(is, &N);              CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&n);          CHKERRQ(ierr);
    ierr = ISGetNonlocalIndices(is, &idx); CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, N-n,idx, PETSC_USE_POINTER, &(is->complement)); CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)is->complement); CHKERRQ(ierr);
    *complement = is->complement;
  }  
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "ISRestoreNonlocalIS" 
/*@
   ISRestoreNonlocalIS - Restore the IS obtained with ISGetNonlocalIS().

   Not collective.

   Input Parameter:
+  is         - the index set
-  complement - index set of is's nonlocal indices

   Level: intermediate


   Concepts: index sets^getting nonlocal indices
   Concepts: index sets^restoring nonlocal indices
.seealso: ISGetNonlocalIS(), ISGetNonlocalIndices(), ISRestoreNonlocalIndices()
@*/
PetscErrorCode  ISRestoreNonlocalIS(IS is, IS *complement)
{
  PetscErrorCode ierr;
  PetscInt       refcnt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(complement,2);
  if(*complement != is->complement) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Complement IS being restored was not obtained with ISGetNonlocalIS()");
  }
  ierr = PetscObjectGetReference((PetscObject)(is->complement), &refcnt); CHKERRQ(ierr);
  if(refcnt <= 1) {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Duplicate call to ISRestoreNonlocalIS() detected");
  }
  ierr = PetscObjectDereference((PetscObject)(is->complement));  CHKERRQ(ierr);
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
PetscErrorCode  ISView(IS is,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)is)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
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
PetscErrorCode  ISSort(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  ierr = (*is->ops->sort)(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISToGeneral" 
/*@
   ISToGeneral - Converts an IS object of any type to ISGENERAL type

   Collective on IS

   Input Parameters:
.  is - the index set

   Level: intermediate

   Concepts: index sets^sorting
   Concepts: sorting^index set

.seealso: ISSorted()
@*/
PetscErrorCode  ISToGeneral(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (is->ops->togeneral) {
    ierr = (*is->ops->togeneral)(is);CHKERRQ(ierr);
  } else SETERRQ1(((PetscObject)is)->comm,PETSC_ERR_SUP,"Not written for this type %s",((PetscObject)is)->type_name);
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
PetscErrorCode  ISSorted(IS is,PetscBool  *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
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
PetscErrorCode  ISDuplicate(IS is,IS *newIS)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
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
PetscErrorCode  ISCopy(IS is,IS isy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidHeaderSpecific(isy,IS_CLASSID,2);
  PetscCheckSameComm(is,1,isy,2);
  if (is == isy) PetscFunctionReturn(0);
  ierr = (*is->ops->copy)(is,isy);CHKERRQ(ierr);
  isy->isperm     = is->isperm;
  isy->max        = is->max;
  isy->min        = is->min;
  isy->isidentity = is->isidentity;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISOnComm"
/*@
   ISOnComm - Split a parallel IS on subcomms (usually self) or concatenate index sets on subcomms into a parallel index set

   Collective on IS and comm

   Input Arguments:
+ is - index set
. comm - communicator for new index set
- mode - copy semantics, PETSC_USE_POINTER for no-copy if possible, otherwise PETSC_COPY_VALUES

   Output Arguments:
. newis - new IS on comm

   Level: advanced

   Notes:
   It is usually desirable to create a parallel IS and look at the local part when necessary.

   This function is useful if serial ISs must be created independently, or to view many
   logically independent serial ISs.

   The input IS must have the same type on every process.

.seealso: ISSplit()
@*/
PetscErrorCode  ISOnComm(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  PetscErrorCode ierr;
  PetscMPIInt match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(newis,3);
  ierr = MPI_Comm_compare(((PetscObject)is)->comm,comm,&match);CHKERRQ(ierr);
  if (mode != PETSC_COPY_VALUES && (match == MPI_IDENT || match == MPI_CONGRUENT)) {
    ierr = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
    *newis = is;
  } else {
    ierr = (*is->ops->oncomm)(is,comm,mode,newis);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSetBlockSize"
/*@
   ISSetBlockSize - informs an index set that it has a given block size

   Logicall Collective on IS

   Input Arguments:
+ is - index set
- bs - block size

   Level: intermediate

.seealso: ISGetBlockSize(), ISCreateBlock()
@*/
PetscErrorCode  ISSetBlockSize(IS is,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidLogicalCollectiveInt(is,bs,2);
  if (bs < 1) SETERRQ1(((PetscObject)is)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Block size %D, must be positive",bs);
  ierr = (*is->ops->setblocksize)(is,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetBlockSize"
/*@
   ISGetBlockSize - Returns the number of elements in a block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the number of elements in a block

   Level: intermediate

   Concepts: IS^block size
   Concepts: index sets^block size

.seealso: ISBlockGetSize(), ISGetSize(), ISCreateBlock(), ISSetBlockSize()
@*/
PetscErrorCode  ISGetBlockSize(IS is,PetscInt *size)
{
  PetscFunctionBegin;
  *size = is->bs;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__ 
#define __FUNCT__ "ISGetIndicesCopy"
PetscErrorCode ISGetIndicesCopy(IS is, PetscInt idx[])
{
  PetscErrorCode ierr;
  PetscInt       len,i;
  const PetscInt *ptr;

  PetscFunctionBegin;
  ierr = ISGetSize(is,&len); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ptr); CHKERRQ(ierr);
  for(i=0;i<len;i++) idx[i] = ptr[i];
  ierr = ISRestoreIndices(is,&ptr); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

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


