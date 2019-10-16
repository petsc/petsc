/*
   Defines the abstract operations on index sets, i.e. the public interface.
*/
#include <petsc/private/isimpl.h>      /*I "petscis.h" I*/
#include <petscviewer.h>
#include <petscsf.h>

/* Logging support */
PetscClassId IS_CLASSID;

/*@
   ISRenumber - Renumbers an index set (with multiplicities) in a contiguous way.

   Collective on IS

   Input Parmeters:
+  subset - the index set
-  subset_mult - the multiplcity of each entry in subset (optional, can be NULL)

   Output Parameters:
+  N - the maximum entry of the new IS
-  subset_n - the new IS

   Level: intermediate

.seealso:
@*/
PetscErrorCode ISRenumber(IS subset, IS subset_mult, PetscInt *N, IS *subset_n)
{
  PetscSF        sf;
  PetscLayout    map;
  const PetscInt *idxs;
  PetscInt       *leaf_data,*root_data,*gidxs;
  PetscInt       N_n,n,i,lbounds[2],gbounds[2],Nl;
  PetscInt       n_n,nlocals,start,first_index;
  PetscMPIInt    commsize;
  PetscBool      first_found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(subset,IS_CLASSID,1);
  if (subset_mult) {
    PetscValidHeaderSpecific(subset_mult,IS_CLASSID,2);
  }
  if (!N && !subset_n) PetscFunctionReturn(0);
  ierr = ISGetLocalSize(subset,&n);CHKERRQ(ierr);
  if (subset_mult) {
    ierr = ISGetLocalSize(subset_mult,&i);CHKERRQ(ierr);
    if (i != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Local subset and multiplicity sizes don't match! %d != %d",n,i);
  }
  /* create workspace layout for computing global indices of subset */
  ierr = ISGetIndices(subset,&idxs);CHKERRQ(ierr);
  lbounds[0] = lbounds[1] = 0;
  for (i=0;i<n;i++) {
    if (idxs[i] < lbounds[0]) lbounds[0] = idxs[i];
    else if (idxs[i] > lbounds[1]) lbounds[1] = idxs[i];
  }
  lbounds[0] = -lbounds[0];
  ierr = MPIU_Allreduce(lbounds,gbounds,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)subset));CHKERRQ(ierr);
  gbounds[0] = -gbounds[0];
  N_n  = gbounds[1] - gbounds[0] + 1;

  ierr = PetscLayoutCreate(PetscObjectComm((PetscObject)subset),&map);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(map,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(map,N_n);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(map,&Nl);CHKERRQ(ierr);

  /* create sf : leaf_data == multiplicity of indexes, root data == global index in layout */
  ierr = PetscMalloc2(n,&leaf_data,Nl,&root_data);CHKERRQ(ierr);
  if (subset_mult) {
    const PetscInt* idxs_mult;

    ierr = ISGetIndices(subset_mult,&idxs_mult);CHKERRQ(ierr);
    ierr = PetscArraycpy(leaf_data,idxs_mult,n);CHKERRQ(ierr);
    ierr = ISRestoreIndices(subset_mult,&idxs_mult);CHKERRQ(ierr);
  } else {
    for (i=0;i<n;i++) leaf_data[i] = 1;
  }
  /* local size of new subset */
  n_n = 0;
  for (i=0;i<n;i++) n_n += leaf_data[i];

  /* global indexes in layout */
  ierr = PetscMalloc1(n_n,&gidxs);CHKERRQ(ierr); /* allocating possibly extra space in gidxs which will be used later */
  for (i=0;i<n;i++) gidxs[i] = idxs[i] - gbounds[0];
  ierr = ISRestoreIndices(subset,&idxs);CHKERRQ(ierr);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)subset),&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(sf,map,n,NULL,PETSC_COPY_VALUES,gidxs);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);

  /* reduce from leaves to roots */
  ierr = PetscArrayzero(root_data,Nl);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,leaf_data,root_data,MPI_MAX);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,leaf_data,root_data,MPI_MAX);CHKERRQ(ierr);

  /* count indexes in local part of layout */
  nlocals = 0;
  first_index = -1;
  first_found = PETSC_FALSE;
  for (i=0;i<Nl;i++) {
    if (!first_found && root_data[i]) {
      first_found = PETSC_TRUE;
      first_index = i;
    }
    nlocals += root_data[i];
  }

  /* cumulative of number of indexes and size of subset without holes */
#if defined(PETSC_HAVE_MPI_EXSCAN)
  start = 0;
  ierr  = MPI_Exscan(&nlocals,&start,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)subset));CHKERRQ(ierr);
#else
  ierr  = MPI_Scan(&nlocals,&start,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)subset));CHKERRQ(ierr);
  start = start-nlocals;
#endif

  if (N) { /* compute total size of new subset if requested */
    *N   = start + nlocals;
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)subset),&commsize);CHKERRQ(ierr);
    ierr = MPI_Bcast(N,1,MPIU_INT,commsize-1,PetscObjectComm((PetscObject)subset));CHKERRQ(ierr);
  }

  if (!subset_n) {
    ierr = PetscFree(gidxs);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    ierr = PetscFree2(leaf_data,root_data);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* adapt root data with cumulative */
  if (first_found) {
    PetscInt old_index;

    root_data[first_index] += start;
    old_index = first_index;
    for (i=first_index+1;i<Nl;i++) {
      if (root_data[i]) {
        root_data[i] += root_data[old_index];
        old_index = i;
      }
    }
  }

  /* from roots to leaves */
  ierr = PetscSFBcastBegin(sf,MPIU_INT,root_data,leaf_data);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,root_data,leaf_data);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);

  /* create new IS with global indexes without holes */
  if (subset_mult) {
    const PetscInt* idxs_mult;
    PetscInt        cum;

    cum = 0;
    ierr = ISGetIndices(subset_mult,&idxs_mult);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      PetscInt j;
      for (j=0;j<idxs_mult[i];j++) gidxs[cum++] = leaf_data[i] - idxs_mult[i] + j;
    }
    ierr = ISRestoreIndices(subset_mult,&idxs_mult);CHKERRQ(ierr);
  } else {
    for (i=0;i<n;i++) {
      gidxs[i] = leaf_data[i]-1;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)subset),n_n,gidxs,PETSC_OWN_POINTER,subset_n);CHKERRQ(ierr);
  ierr = PetscFree2(leaf_data,root_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@
   ISCreateSubIS - Create a sub index set from a global index set selecting some components.

   Collective on IS

   Input Parmeters:
+  is - the index set
-  comps - which components we will extract from is

   Output Parameters:
.  subis - the new sub index set

   Level: intermediate

   Example usage:
   We have an index set (is) living on 3 processes with the following values:
   | 4 9 0 | 2 6 7 | 10 11 1|
   and another index set (comps) used to indicate which components of is  we want to take,
   | 7 5  | 1 2 | 0 4|
   The output index set (subis) should look like:
   | 11 7 | 9 0 | 4 6|

.seealso: VecGetSubVector(), MatCreateSubMatrix()
@*/
PetscErrorCode ISCreateSubIS(IS is,IS comps,IS *subis)
{
  PetscSF         sf;
  const PetscInt  *is_indices,*comps_indices;
  PetscInt        *subis_indices,nroots,nleaves,*mine,i,owner,lidx;
  PetscSFNode     *remote;
  PetscErrorCode  ierr;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidHeaderSpecific(comps,IS_CLASSID,2);
  PetscValidPointer(subis,3);

  ierr = PetscObjectGetComm((PetscObject)is, &comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(comps,&nleaves);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&nroots);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves,&remote);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves,&mine);CHKERRQ(ierr);
  ierr = ISGetIndices(comps,&comps_indices);CHKERRQ(ierr);
  /*
   * Construct a PetscSF in which "is" data serves as roots and "subis" is leaves.
   * Root data are sent to leaves using PetscSFBcast().
   * */
  for (i=0; i<nleaves; i++) {
    mine[i] = i;
    /* Connect a remote root with the current leaf. The value on the remote root
     * will be received by the current local leaf.
     * */
    owner = -1;
    lidx =  -1;
    ierr = PetscLayoutFindOwnerIndex(is->map,comps_indices[i],&owner, &lidx);CHKERRQ(ierr);
    remote[i].rank = owner;
    remote[i].index = lidx;
  }
  ierr = ISRestoreIndices(comps,&comps_indices);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);\
  ierr = PetscSFSetGraph(sf,nroots,nleaves,mine,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);

  ierr = PetscMalloc1(nleaves,&subis_indices);CHKERRQ(ierr);
  ierr = ISGetIndices(is, &is_indices);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,is_indices,subis_indices);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,is_indices,subis_indices);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&is_indices);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,nleaves,subis_indices,PETSC_OWN_POINTER,subis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@
   ISIdentity - Determines whether index set is the identity mapping.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  ident - PETSC_TRUE if an identity, else PETSC_FALSE

   Level: intermediate


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

/*@
   ISSetIdentity - Informs the index set that it is an identity.

   Logically Collective on IS

   Input Parmeters:
.  is - the index set

   Level: intermediate


.seealso: ISIdentity()
@*/
PetscErrorCode  ISSetIdentity(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  is->isidentity = PETSC_TRUE;
  ierr = ISSetPermutation(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISContiguousLocal - Locates an index set with contiguous range within a global range, if possible

   Not Collective

   Input Parmeters:
+  is - the index set
.  gstart - global start
-  gend - global end

   Output Parameters:
+  start - start of contiguous block, as an offset from gstart
-  contig - PETSC_TRUE if the index set refers to contiguous entries on this process, else PETSC_FALSE

   Level: developer

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
    *start  = -1;
    *contig = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*@
   ISPermutation - PETSC_TRUE or PETSC_FALSE depending on whether the
   index set has been declared to be a permutation.

   Logically Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  perm - PETSC_TRUE if a permutation, else PETSC_FALSE

   Level: intermediate


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

/*@
   ISSetPermutation - Informs the index set that it is a permutation.

   Logically Collective on IS

   Input Parmeters:
.  is - the index set

   Level: intermediate


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

    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)is),&size);CHKERRQ(ierr);
    if (size == 1) {
      PetscInt       i,n,*idx;
      const PetscInt *iidx;

      ierr = ISGetSize(is,&n);CHKERRQ(ierr);
      ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
      ierr = ISGetIndices(is,&iidx);CHKERRQ(ierr);
      ierr = PetscArraycpy(idx,iidx,n);CHKERRQ(ierr);
      ierr = PetscSortInt(n,idx);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        if (idx[i] != i) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Index set is not a permutation");
      }
      ierr = PetscFree(idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is,&iidx);CHKERRQ(ierr);
    }
  }
#endif
  is->isperm = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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
    ierr = PetscObjectGetReference((PetscObject)((*is)->complement), &refcnt);CHKERRQ(ierr);
    if (refcnt > 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Nonlocal IS has not been restored");
    ierr = ISDestroy(&(*is)->complement);CHKERRQ(ierr);
  }
  if ((*is)->ops->destroy) {
    ierr = (*(*is)->ops->destroy)(*is);CHKERRQ(ierr);
  }
  ierr = PetscLayoutDestroy(&(*is)->map);CHKERRQ(ierr);
  /* Destroy local representations of offproc data. */
  ierr = PetscFree((*is)->total);CHKERRQ(ierr);
  ierr = PetscFree((*is)->nonlocal);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

   Notes:
    For parallel index sets this does the complete parallel permutation, but the
    code is not efficient for huge index sets (10,000,000 indices).

@*/
PetscErrorCode  ISInvertPermutation(IS is,PetscInt nlocal,IS *isout)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(isout,3);
  if (!is->isperm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a permutation, must call ISSetPermutation() on the IS first");
  if (is->isidentity) {
    ierr = ISDuplicate(is,isout);CHKERRQ(ierr);
  } else {
    ierr = (*is->ops->invertpermutation)(is,nlocal,isout);CHKERRQ(ierr);
    ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   ISGetSize - Returns the global length of an index set.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the global size

   Level: beginner


@*/
PetscErrorCode  ISGetSize(IS is,PetscInt *size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidIntPointer(size,2);
  *size = is->map->N;
  PetscFunctionReturn(0);
}

/*@
   ISGetLocalSize - Returns the local (processor) length of an index set.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the local size

   Level: beginner

@*/
PetscErrorCode  ISGetLocalSize(IS is,PetscInt *size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidIntPointer(size,2);
  *size = is->map->n;
  PetscFunctionReturn(0);
}

/*@
   ISGetLayout - get PetscLayout describing index set layout

   Not Collective

   Input Arguments:
.  is - the index set

   Output Arguments:
.  map - the layout

   Level: developer

.seealso: ISGetSize(), ISGetLocalSize()
@*/
PetscErrorCode ISGetLayout(IS is,PetscLayout *map)
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  *map = is->map;
  PetscFunctionReturn(0);
}

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
   This routine has two different interfaces from Fortran; the first is not recommend, it does not require Fortran 90
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
   The second Fortran interface is recommended.
$          use petscisdef
$          PetscInt, pointer :: array(:)
$          PetscErrorCode  ierr
$          IS       i
$          call ISGetIndicesF90(i,array,ierr)



   See the Fortran chapter of the users manual and
   petsc/src/is/examples/[tutorials,tests] for details.

   Level: intermediate


.seealso: ISRestoreIndices(), ISGetIndicesF90()
@*/
PetscErrorCode  ISGetIndices(IS is,const PetscInt *ptr[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidIntPointer(ptr,2);
  ierr = (*is->ops->getindices)(is,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   ISGetMinMax - Gets the minimum and maximum values in an IS

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
+   min - the minimum value
-   max - the maximum value

   Level: intermediate

   Notes:
    Empty index sets return min=PETSC_MAX_INT and max=PETSC_MIN_INT.
    In parallel, it returns the min and max of the local portion of the IS


.seealso: ISGetIndices(), ISRestoreIndices(), ISGetIndicesF90()
@*/
PetscErrorCode  ISGetMinMax(IS is,PetscInt *min,PetscInt *max)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (min) *min = is->min;
  if (max) *max = is->max;
  PetscFunctionReturn(0);
}

/*@
  ISLocate - determine the location of an index within the local component of an index set

  Not Collective

  Input Parameter:
+ is - the index set
- key - the search key

  Output Parameter:
. location - if >= 0, a location within the index set that is equal to the key, otherwise the key is not in the index set

  Level: intermediate
@*/
PetscErrorCode ISLocate(IS is, PetscInt key, PetscInt *location)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is->ops->locate) {
    ierr = (*is->ops->locate)(is,key,location);CHKERRQ(ierr);
  } else {
    PetscInt       numIdx;
    PetscBool      sorted;
    const PetscInt *idx;

    ierr = ISGetLocalSize(is,&numIdx);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&idx);CHKERRQ(ierr);
    ierr = ISSorted(is,&sorted);CHKERRQ(ierr);
    if (sorted) {
      ierr = PetscFindInt(key,numIdx,idx,location);CHKERRQ(ierr);
    } else {
      PetscInt i;

      *location = -1;
      for (i = 0; i < numIdx; i++) {
        if (idx[i] == key) {
          *location = i;
          break;
        }
      }
    }
    ierr = ISRestoreIndices(is,&idx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

   Note:
   This routine zeros out ptr. This is to prevent accidental us of the array after it has been restored.

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

static PetscErrorCode ISGatherTotal_Private(IS is)
{
  PetscErrorCode ierr;
  PetscInt       i,n,N;
  const PetscInt *lindices;
  MPI_Comm       comm;
  PetscMPIInt    rank,size,*sizes = NULL,*offsets = NULL,nn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = PetscMalloc2(size,&sizes,size,&offsets);CHKERRQ(ierr);

  ierr = PetscMPIIntCast(n,&nn);CHKERRQ(ierr);
  ierr = MPI_Allgather(&nn,1,MPI_INT,sizes,1,MPI_INT,comm);CHKERRQ(ierr);
  offsets[0] = 0;
  for (i=1; i<size; ++i) offsets[i] = offsets[i-1] + sizes[i-1];
  N = offsets[size-1] + sizes[size-1];

  ierr = PetscMalloc1(N,&(is->total));CHKERRQ(ierr);
  ierr = ISGetIndices(is,&lindices);CHKERRQ(ierr);
  ierr = MPI_Allgatherv((void*)lindices,nn,MPIU_INT,is->total,sizes,offsets,MPIU_INT,comm);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&lindices);CHKERRQ(ierr);
  is->local_offset = offsets[rank];
  ierr = PetscFree2(sizes,offsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   ISGetTotalIndices - Retrieve an array containing all indices across the communicator.

   Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  indices - total indices with rank 0 indices first, and so on; total array size is
             the same as returned with ISGetSize().

   Level: intermediate

   Notes:
    this is potentially nonscalable, but depends on the size of the total index set
     and the size of the communicator. This may be feasible for index sets defined on
     subcommunicators, such that the set size does not grow with PETSC_WORLD_COMM.
     Note also that there is no way to tell where the local part of the indices starts
     (use ISGetIndices() and ISGetNonlocalIndices() to retrieve just the local and just
      the nonlocal part (complement), respectively).

.seealso: ISRestoreTotalIndices(), ISGetNonlocalIndices(), ISGetSize()
@*/
PetscErrorCode ISGetTotalIndices(IS is, const PetscInt *indices[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)is), &size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = (*is->ops->getindices)(is,indices);CHKERRQ(ierr);
  } else {
    if (!is->total) {
      ierr = ISGatherTotal_Private(is);CHKERRQ(ierr);
    }
    *indices = is->total;
  }
  PetscFunctionReturn(0);
}

/*@C
   ISRestoreTotalIndices - Restore the index array obtained with ISGetTotalIndices().

   Not Collective.

   Input Parameter:
+  is - the index set
-  indices - index array; must be the array obtained with ISGetTotalIndices()

   Level: intermediate

.seealso: ISRestoreTotalIndices(), ISGetNonlocalIndices()
@*/
PetscErrorCode  ISRestoreTotalIndices(IS is, const PetscInt *indices[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)is), &size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = (*is->ops->restoreindices)(is,indices);CHKERRQ(ierr);
  } else {
    if (is->total != *indices) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index array pointer being restored does not point to the array obtained from the IS.");
  }
  PetscFunctionReturn(0);
}
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

   Notes:
    restore the indices using ISRestoreNonlocalIndices().
          The same scalability considerations as those for ISGetTotalIndices
          apply here.

.seealso: ISGetTotalIndices(), ISRestoreNonlocalIndices(), ISGetSize(), ISGetLocalSize().
@*/
PetscErrorCode  ISGetNonlocalIndices(IS is, const PetscInt *indices[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n, N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)is), &size);CHKERRQ(ierr);
  if (size == 1) *indices = NULL;
  else {
    if (!is->total) {
      ierr = ISGatherTotal_Private(is);CHKERRQ(ierr);
    }
    ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
    ierr = ISGetSize(is,&N);CHKERRQ(ierr);
    ierr = PetscMalloc1(N-n, &(is->nonlocal));CHKERRQ(ierr);
    ierr = PetscArraycpy(is->nonlocal, is->total, is->local_offset);CHKERRQ(ierr);
    ierr = PetscArraycpy(is->nonlocal+is->local_offset, is->total+is->local_offset+n,N - is->local_offset - n);CHKERRQ(ierr);
    *indices = is->nonlocal;
  }
  PetscFunctionReturn(0);
}

/*@C
   ISRestoreTotalIndices - Restore the index array obtained with ISGetNonlocalIndices().

   Not Collective.

   Input Parameter:
+  is - the index set
-  indices - index array; must be the array obtained with ISGetNonlocalIndices()

   Level: intermediate

.seealso: ISGetTotalIndices(), ISGetNonlocalIndices(), ISRestoreTotalIndices()
@*/
PetscErrorCode  ISRestoreNonlocalIndices(IS is, const PetscInt *indices[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  if (is->nonlocal != *indices) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index array pointer being restored does not point to the array obtained from the IS.");
  PetscFunctionReturn(0);
}

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

   Notes:
    complement represents the result of ISGetNonlocalIndices as an IS.
          Therefore scalability issues similar to ISGetNonlocalIndices apply.
          The resulting IS must be restored using ISRestoreNonlocalIS().

.seealso: ISGetNonlocalIndices(), ISRestoreNonlocalIndices(),  ISAllGather(), ISGetSize()
@*/
PetscErrorCode  ISGetNonlocalIS(IS is, IS *complement)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(complement,2);
  /* Check if the complement exists already. */
  if (is->complement) {
    *complement = is->complement;
    ierr = PetscObjectReference((PetscObject)(is->complement));CHKERRQ(ierr);
  } else {
    PetscInt       N, n;
    const PetscInt *idx;
    ierr = ISGetSize(is, &N);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
    ierr = ISGetNonlocalIndices(is, &idx);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, N-n,idx, PETSC_USE_POINTER, &(is->complement));CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)is->complement);CHKERRQ(ierr);
    *complement = is->complement;
  }
  PetscFunctionReturn(0);
}


/*@
   ISRestoreNonlocalIS - Restore the IS obtained with ISGetNonlocalIS().

   Not collective.

   Input Parameter:
+  is         - the index set
-  complement - index set of is's nonlocal indices

   Level: intermediate


.seealso: ISGetNonlocalIS(), ISGetNonlocalIndices(), ISRestoreNonlocalIndices()
@*/
PetscErrorCode  ISRestoreNonlocalIS(IS is, IS *complement)
{
  PetscErrorCode ierr;
  PetscInt       refcnt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(complement,2);
  if (*complement != is->complement) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Complement IS being restored was not obtained with ISGetNonlocalIS()");
  ierr = PetscObjectGetReference((PetscObject)(is->complement), &refcnt);CHKERRQ(ierr);
  if (refcnt <= 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Duplicate call to ISRestoreNonlocalIS() detected");
  ierr = PetscObjectDereference((PetscObject)(is->complement));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)is),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(is,1,viewer,2);

  ierr = PetscObjectPrintClassNamePrefixType((PetscObject)is,viewer);CHKERRQ(ierr);
  ierr = (*is->ops->view)(is,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  ISLoad - Loads a vector that has been stored in binary or HDF5 format with ISView().

  Collective on PetscViewer

  Input Parameters:
+ is - the newly loaded vector, this needs to have been created with ISCreate() or some related function before a call to ISLoad().
- viewer - binary file viewer, obtained from PetscViewerBinaryOpen() or HDF5 file viewer, obtained from PetscViewerHDF5Open()

  Level: intermediate

  Notes:
  IF using HDF5, you must assign the IS the same name as was used in the IS
  that was stored in the file using PetscObjectSetName(). Otherwise you will
  get the error message: "Cannot H5DOpen2() with Vec name NAMEOFOBJECT"

.seealso: PetscViewerBinaryOpen(), ISView(), MatLoad(), VecLoad()
@*/
PetscErrorCode ISLoad(IS is, PetscViewer viewer)
{
  PetscBool      isbinary, ishdf5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5);CHKERRQ(ierr);
  if (!isbinary && !ishdf5) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");
  if (!((PetscObject)is)->type_name) {ierr = ISSetType(is, ISGENERAL);CHKERRQ(ierr);}
  ierr = (*is->ops->load)(is, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISSort - Sorts the indices of an index set.

   Collective on IS

   Input Parameters:
.  is - the index set

   Level: intermediate


.seealso: ISSortRemoveDups(), ISSorted()
@*/
PetscErrorCode  ISSort(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  ierr = (*is->ops->sort)(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  ISSortRemoveDups - Sorts the indices of an index set, removing duplicates.

  Collective on IS

  Input Parameters:
. is - the index set

  Level: intermediate


.seealso: ISSort(), ISSorted()
@*/
PetscErrorCode ISSortRemoveDups(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  ierr = (*is->ops->sortremovedups)(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISToGeneral - Converts an IS object of any type to ISGENERAL type

   Collective on IS

   Input Parameters:
.  is - the index set

   Level: intermediate


.seealso: ISSorted()
@*/
PetscErrorCode  ISToGeneral(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (is->ops->togeneral) {
    ierr = (*is->ops->togeneral)(is);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)is),PETSC_ERR_SUP,"Not written for this type %s",((PetscObject)is)->type_name);
  PetscFunctionReturn(0);
}

/*@
   ISSorted - Checks the indices to determine whether they have been sorted.

   Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  flg - output flag, either PETSC_TRUE if the index set is sorted,
         or PETSC_FALSE otherwise.

   Notes:
    For parallel IS objects this only indicates if the local part of the IS
          is sorted. So some processors may return PETSC_TRUE while others may
          return PETSC_FALSE.

   Level: intermediate

.seealso: ISSort(), ISSortRemoveDups()
@*/
PetscErrorCode  ISSorted(IS is,PetscBool  *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  ierr = (*is->ops->sorted)(is,flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISDuplicate - Creates a duplicate copy of an index set.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  isnew - the copy of the index set

   Level: beginner

.seealso: ISCreateGeneral(), ISCopy()
@*/
PetscErrorCode  ISDuplicate(IS is,IS *newIS)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(newIS,2);
  ierr = (*is->ops->duplicate)(is,newIS);CHKERRQ(ierr);
  (*newIS)->isidentity = is->isidentity;
  (*newIS)->isperm     = is->isperm;
  PetscFunctionReturn(0);
}

/*@
   ISCopy - Copies an index set.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  isy - the copy of the index set

   Level: beginner

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

/*@
   ISOnComm - Split a parallel IS on subcomms (usually self) or concatenate index sets on subcomms into a parallel index set

   Collective on IS

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
  PetscMPIInt    match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(newis,3);
  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)is),comm,&match);CHKERRQ(ierr);
  if (mode != PETSC_COPY_VALUES && (match == MPI_IDENT || match == MPI_CONGRUENT)) {
    ierr   = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
    *newis = is;
  } else {
    ierr = (*is->ops->oncomm)(is,comm,mode,newis);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   ISSetBlockSize - informs an index set that it has a given block size

   Logicall Collective on IS

   Input Arguments:
+ is - index set
- bs - block size

   Level: intermediate
   
   Notes: 
   This is much like the block size for Vecs. It indicates that one can think of the indices as 
   being in a collection of equal size blocks. For ISBlock() these collections of blocks are all contiquous
   within a block but this is not the case for other IS.
   ISBlockGetIndices() only works for ISBlock IS, not others.

.seealso: ISGetBlockSize(), ISCreateBlock(), ISBlockGetIndices(), 
@*/
PetscErrorCode  ISSetBlockSize(IS is,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidLogicalCollectiveInt(is,bs,2);
  if (bs < 1) SETERRQ1(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_OUTOFRANGE,"Block size %D, must be positive",bs);
  ierr = (*is->ops->setblocksize)(is,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISGetBlockSize - Returns the number of elements in a block.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the number of elements in a block

   Level: intermediate

Notes: 
   This is much like the block size for Vecs. It indicates that one can think of the indices as 
   being in a collection of equal size blocks. For ISBlock() these collections of blocks are all contiquous
   within a block but this is not the case for other IS.
   ISBlockGetIndices() only works for ISBlock IS, not others.

.seealso: ISBlockGetSize(), ISGetSize(), ISCreateBlock(), ISSetBlockSize()
@*/
PetscErrorCode  ISGetBlockSize(IS is,PetscInt *size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetBlockSize(is->map, size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ISGetIndicesCopy(IS is, PetscInt idx[])
{
  PetscErrorCode ierr;
  PetscInt       len,i;
  const PetscInt *ptr;

  PetscFunctionBegin;
  ierr = ISGetSize(is,&len);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ptr);CHKERRQ(ierr);
  for (i=0; i<len; i++) idx[i] = ptr[i];
  ierr = ISRestoreIndices(is,&ptr);CHKERRQ(ierr);
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
    PetscInt, pointer xx_v(:)
    ....
    call ISGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISRestoreIndicesF90(x,xx_v,ierr)
.ve

    Level: intermediate

.seealso:  ISRestoreIndicesF90(), ISGetIndices(), ISRestoreIndices()


M*/

/*MC
    ISRestoreIndicesF90 - Restores an index set to a usable state after
    a call to ISGetIndicesF90().

    Synopsis:
    ISRestoreIndicesF90(IS x,{integer, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameters:
+   x - index set
-   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code


    Example of Usage:
.vb
    PetscInt, pointer xx_v(:)
    ....
    call ISGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISRestoreIndicesF90(x,xx_v,ierr)
.ve

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
    PetscInt, pointer xx_v(:)
    ....
    call ISBlockGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISBlockRestoreIndicesF90(x,xx_v,ierr)
.ve

    Level: intermediate

.seealso:  ISBlockRestoreIndicesF90(), ISGetIndices(), ISRestoreIndices(),
           ISRestoreIndices()

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
    PetscInt, pointer xx_v(:)
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


