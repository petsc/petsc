/*
   Defines the abstract operations on index sets, i.e. the public interface.
*/
#include <petsc/private/isimpl.h>      /*I "petscis.h" I*/
#include <petscviewer.h>
#include <petscsf.h>

/* Logging support */
PetscClassId IS_CLASSID;
/* TODO: Much more events are missing! */
PetscLogEvent IS_View;
PetscLogEvent IS_Load;

/*@
   ISRenumber - Renumbers the non-negative entries of an index set in a contiguous way, starting from 0.

   Collective on IS

   Input Parameters:
+  subset - the index set
-  subset_mult - the multiplicity of each entry in subset (optional, can be NULL)

   Output Parameters:
+  N - the maximum entry of the new IS
-  subset_n - the new IS

   Notes: All negative entries are mapped to -1. Indices with non positive multiplicities are skipped.

   Level: intermediate

.seealso:
@*/
PetscErrorCode ISRenumber(IS subset, IS subset_mult, PetscInt *N, IS *subset_n)
{
  PetscSF        sf;
  PetscLayout    map;
  const PetscInt *idxs, *idxs_mult = NULL;
  PetscInt       *leaf_data,*root_data,*gidxs,*ilocal,*ilocalneg;
  PetscInt       N_n,n,i,lbounds[2],gbounds[2],Nl,ibs;
  PetscInt       n_n,nlocals,start,first_index,npos,nneg;
  PetscMPIInt    commsize;
  PetscBool      first_found,isblock;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(subset,IS_CLASSID,1);
  if (subset_mult) PetscValidHeaderSpecific(subset_mult,IS_CLASSID,2);
  if (N) PetscValidIntPointer(N,3);
  else if (!subset_n) PetscFunctionReturn(0);
  CHKERRQ(ISGetLocalSize(subset,&n));
  if (subset_mult) {
    CHKERRQ(ISGetLocalSize(subset_mult,&i));
    PetscCheck(i == n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Local subset and multiplicity sizes don't match! %" PetscInt_FMT " != %" PetscInt_FMT,n,i);
  }
  /* create workspace layout for computing global indices of subset */
  CHKERRQ(PetscMalloc1(n,&ilocal));
  CHKERRQ(PetscMalloc1(n,&ilocalneg));
  CHKERRQ(ISGetIndices(subset,&idxs));
  CHKERRQ(ISGetBlockSize(subset,&ibs));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)subset,ISBLOCK,&isblock));
  if (subset_mult) CHKERRQ(ISGetIndices(subset_mult,&idxs_mult));
  lbounds[0] = PETSC_MAX_INT;
  lbounds[1] = PETSC_MIN_INT;
  for (i=0,npos=0,nneg=0;i<n;i++) {
    if (idxs[i] < 0) { ilocalneg[nneg++] = i; continue; }
    if (idxs[i] < lbounds[0]) lbounds[0] = idxs[i];
    if (idxs[i] > lbounds[1]) lbounds[1] = idxs[i];
    ilocal[npos++] = i;
  }
  if (npos == n) {
    CHKERRQ(PetscFree(ilocal));
    CHKERRQ(PetscFree(ilocalneg));
  }

  /* create sf : leaf_data == multiplicity of indexes, root data == global index in layout */
  CHKERRQ(PetscMalloc1(n,&leaf_data));
  for (i=0;i<n;i++) leaf_data[i] = idxs_mult ? PetscMax(idxs_mult[i],0) : 1;

  /* local size of new subset */
  n_n = 0;
  for (i=0;i<n;i++) n_n += leaf_data[i];
  if (ilocalneg) for (i=0;i<nneg;i++) leaf_data[ilocalneg[i]] = 0;
  CHKERRQ(PetscFree(ilocalneg));
  CHKERRQ(PetscMalloc1(PetscMax(n_n,n),&gidxs)); /* allocating extra space to reuse gidxs */
  /* check for early termination (all negative) */
  CHKERRQ(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)subset),lbounds,gbounds));
  if (gbounds[1] < gbounds[0]) {
    if (N) *N = 0;
    if (subset_n) { /* all negative */
      for (i=0;i<n_n;i++) gidxs[i] = -1;
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)subset),n_n,gidxs,PETSC_COPY_VALUES,subset_n));
    }
    CHKERRQ(PetscFree(leaf_data));
    CHKERRQ(PetscFree(gidxs));
    CHKERRQ(ISRestoreIndices(subset,&idxs));
    if (subset_mult) CHKERRQ(ISRestoreIndices(subset_mult,&idxs_mult));
    CHKERRQ(PetscFree(ilocal));
    CHKERRQ(PetscFree(ilocalneg));
    PetscFunctionReturn(0);
  }

  /* split work */
  N_n  = gbounds[1] - gbounds[0] + 1;
  CHKERRQ(PetscLayoutCreate(PetscObjectComm((PetscObject)subset),&map));
  CHKERRQ(PetscLayoutSetBlockSize(map,1));
  CHKERRQ(PetscLayoutSetSize(map,N_n));
  CHKERRQ(PetscLayoutSetUp(map));
  CHKERRQ(PetscLayoutGetLocalSize(map,&Nl));

  /* global indexes in layout */
  for (i=0;i<npos;i++) gidxs[i] = (ilocal ? idxs[ilocal[i]] : idxs[i]) - gbounds[0];
  CHKERRQ(ISRestoreIndices(subset,&idxs));
  CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)subset),&sf));
  CHKERRQ(PetscSFSetGraphLayout(sf,map,npos,ilocal,PETSC_USE_POINTER,gidxs));
  CHKERRQ(PetscLayoutDestroy(&map));

  /* reduce from leaves to roots */
  CHKERRQ(PetscCalloc1(Nl,&root_data));
  CHKERRQ(PetscSFReduceBegin(sf,MPIU_INT,leaf_data,root_data,MPI_MAX));
  CHKERRQ(PetscSFReduceEnd(sf,MPIU_INT,leaf_data,root_data,MPI_MAX));

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
  CHKERRMPI(MPI_Exscan(&nlocals,&start,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)subset)));
#else
  CHKERRMPI(MPI_Scan(&nlocals,&start,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)subset)));
  start = start-nlocals;
#endif

  if (N) { /* compute total size of new subset if requested */
    *N   = start + nlocals;
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)subset),&commsize));
    CHKERRMPI(MPI_Bcast(N,1,MPIU_INT,commsize-1,PetscObjectComm((PetscObject)subset)));
  }

  if (!subset_n) {
    CHKERRQ(PetscFree(gidxs));
    CHKERRQ(PetscSFDestroy(&sf));
    CHKERRQ(PetscFree(leaf_data));
    CHKERRQ(PetscFree(root_data));
    CHKERRQ(PetscFree(ilocal));
    if (subset_mult) CHKERRQ(ISRestoreIndices(subset_mult,&idxs_mult));
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
  CHKERRQ(PetscSFBcastBegin(sf,MPIU_INT,root_data,leaf_data,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf,MPIU_INT,root_data,leaf_data,MPI_REPLACE));
  CHKERRQ(PetscSFDestroy(&sf));

  /* create new IS with global indexes without holes */
  for (i=0;i<n_n;i++) gidxs[i] = -1;
  if (subset_mult) {
    PetscInt cum;

    isblock = PETSC_FALSE;
    for (i=0,cum=0;i<n;i++) for (PetscInt j=0;j<idxs_mult[i];j++) gidxs[cum++] = leaf_data[i] - idxs_mult[i] + j;
  } else for (i=0;i<n;i++) gidxs[i] = leaf_data[i]-1;

  if (isblock) {
    if (ibs > 1) for (i=0;i<n_n/ibs;i++) gidxs[i] = gidxs[i*ibs]/ibs;
    CHKERRQ(ISCreateBlock(PetscObjectComm((PetscObject)subset),ibs,n_n/ibs,gidxs,PETSC_COPY_VALUES,subset_n));
  } else {
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)subset),n_n,gidxs,PETSC_COPY_VALUES,subset_n));
  }
  if (subset_mult) CHKERRQ(ISRestoreIndices(subset_mult,&idxs_mult));
  CHKERRQ(PetscFree(gidxs));
  CHKERRQ(PetscFree(leaf_data));
  CHKERRQ(PetscFree(root_data));
  CHKERRQ(PetscFree(ilocal));
  PetscFunctionReturn(0);
}

/*@
   ISCreateSubIS - Create a sub index set from a global index set selecting some components.

   Collective on IS

   Input Parameters:
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
  PetscInt        *subis_indices,nroots,nleaves,*mine,i,lidx;
  PetscMPIInt     owner;
  PetscSFNode     *remote;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidHeaderSpecific(comps,IS_CLASSID,2);
  PetscValidPointer(subis,3);

  CHKERRQ(PetscObjectGetComm((PetscObject)is, &comm));
  CHKERRQ(ISGetLocalSize(comps,&nleaves));
  CHKERRQ(ISGetLocalSize(is,&nroots));
  CHKERRQ(PetscMalloc1(nleaves,&remote));
  CHKERRQ(PetscMalloc1(nleaves,&mine));
  CHKERRQ(ISGetIndices(comps,&comps_indices));
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
    CHKERRQ(PetscLayoutFindOwnerIndex(is->map,comps_indices[i],&owner,&lidx));
    remote[i].rank = owner;
    remote[i].index = lidx;
  }
  CHKERRQ(ISRestoreIndices(comps,&comps_indices));
  CHKERRQ(PetscSFCreate(comm,&sf));
  CHKERRQ(PetscSFSetFromOptions(sf));\
  CHKERRQ(PetscSFSetGraph(sf,nroots,nleaves,mine,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER));

  CHKERRQ(PetscMalloc1(nleaves,&subis_indices));
  CHKERRQ(ISGetIndices(is, &is_indices));
  CHKERRQ(PetscSFBcastBegin(sf,MPIU_INT,is_indices,subis_indices,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf,MPIU_INT,is_indices,subis_indices,MPI_REPLACE));
  CHKERRQ(ISRestoreIndices(is,&is_indices));
  CHKERRQ(PetscSFDestroy(&sf));
  CHKERRQ(ISCreateGeneral(comm,nleaves,subis_indices,PETSC_OWN_POINTER,subis));
  PetscFunctionReturn(0);
}

/*@
   ISClearInfoCache - clear the cache of computed index set properties

   Not collective

   Input Parameters:
+  is - the index set
-  clear_permanent_local - whether to remove the permanent status of local properties

   NOTE: because all processes must agree on the global permanent status of a property,
   the permanent status can only be changed with ISSetInfo(), because this routine is not collective

   Level: developer

.seealso:  ISInfo, ISInfoType, ISSetInfo(), ISClearInfoCache()

@*/
PetscErrorCode ISClearInfoCache(IS is, PetscBool clear_permanent_local)
{
  PetscInt i, j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidType(is,1);
  for (i = 0; i < IS_INFO_MAX; i++) {
    if (clear_permanent_local) is->info_permanent[IS_LOCAL][i] = PETSC_FALSE;
    for (j = 0; j < 2; j++) {
      if (!is->info_permanent[j][i]) is->info[j][i] = IS_INFO_UNKNOWN;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSetInfo_Internal(IS is, ISInfo info, ISInfoType type, ISInfoBool ipermanent, PetscBool flg)
{
  ISInfoBool     iflg = flg ? IS_INFO_TRUE : IS_INFO_FALSE;
  PetscInt       itype = (type == IS_LOCAL) ? 0 : 1;
  PetscBool      permanent_set = (ipermanent == IS_INFO_UNKNOWN) ? PETSC_FALSE : PETSC_TRUE;
  PetscBool      permanent = (ipermanent == IS_INFO_TRUE) ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;
  /* set this property */
  is->info[itype][(int)info] = iflg;
  if (permanent_set) is->info_permanent[itype][(int)info] = permanent;
  /* set implications */
  switch (info) {
  case IS_SORTED:
    if (flg && type == IS_GLOBAL) { /* an array that is globally sorted is also locally sorted */
      is->info[IS_LOCAL][(int)info] = IS_INFO_TRUE;
      /* global permanence implies local permanence */
      if (permanent_set && permanent) is->info_permanent[IS_LOCAL][(int)info] = PETSC_TRUE;
    }
    if (!flg) { /* if an array is not sorted, it cannot be an interval or the identity */
      is->info[itype][IS_INTERVAL] = IS_INFO_FALSE;
      is->info[itype][IS_IDENTITY] = IS_INFO_FALSE;
      if (permanent_set) {
        is->info_permanent[itype][IS_INTERVAL] = permanent;
        is->info_permanent[itype][IS_IDENTITY] = permanent;
      }
    }
    break;
  case IS_UNIQUE:
    if (flg && type == IS_GLOBAL) { /* an array that is globally unique is also locally unique */
      is->info[IS_LOCAL][(int)info] = IS_INFO_TRUE;
      /* global permanence implies local permanence */
      if (permanent_set && permanent) is->info_permanent[IS_LOCAL][(int)info] = PETSC_TRUE;
    }
    if (!flg) { /* if an array is not unique, it cannot be a permutation, and interval, or the identity */
      is->info[itype][IS_PERMUTATION] = IS_INFO_FALSE;
      is->info[itype][IS_INTERVAL]    = IS_INFO_FALSE;
      is->info[itype][IS_IDENTITY]    = IS_INFO_FALSE;
      if (permanent_set) {
        is->info_permanent[itype][IS_PERMUTATION] = permanent;
        is->info_permanent[itype][IS_INTERVAL]    = permanent;
        is->info_permanent[itype][IS_IDENTITY]    = permanent;
      }
    }
    break;
  case IS_PERMUTATION:
    if (flg) { /* an array that is a permutation is unique and is unique locally */
      is->info[itype][IS_UNIQUE] = IS_INFO_TRUE;
      is->info[IS_LOCAL][IS_UNIQUE] = IS_INFO_TRUE;
      if (permanent_set && permanent) {
        is->info_permanent[itype][IS_UNIQUE] = PETSC_TRUE;
        is->info_permanent[IS_LOCAL][IS_UNIQUE] = PETSC_TRUE;
      }
    } else { /* an array that is not a permutation cannot be the identity */
      is->info[itype][IS_IDENTITY] = IS_INFO_FALSE;
      if (permanent_set) is->info_permanent[itype][IS_IDENTITY] = permanent;
    }
    break;
  case IS_INTERVAL:
    if (flg) { /* an array that is an interval is sorted and unique */
      is->info[itype][IS_SORTED]         = IS_INFO_TRUE;
      is->info[IS_LOCAL][IS_SORTED]      = IS_INFO_TRUE;
      is->info[itype][IS_UNIQUE]         = IS_INFO_TRUE;
      is->info[IS_LOCAL][IS_UNIQUE]      = IS_INFO_TRUE;
      if (permanent_set && permanent) {
        is->info_permanent[itype][IS_SORTED]    = PETSC_TRUE;
        is->info_permanent[IS_LOCAL][IS_SORTED] = PETSC_TRUE;
        is->info_permanent[itype][IS_UNIQUE]    = PETSC_TRUE;
        is->info_permanent[IS_LOCAL][IS_UNIQUE] = PETSC_TRUE;
      }
    } else { /* an array that is not an interval cannot be the identity */
      is->info[itype][IS_IDENTITY] = IS_INFO_FALSE;
      if (permanent_set) is->info_permanent[itype][IS_IDENTITY] = permanent;
    }
    break;
  case IS_IDENTITY:
    if (flg) { /* an array that is the identity is sorted, unique, an interval, and a permutation */
      is->info[itype][IS_SORTED]         = IS_INFO_TRUE;
      is->info[IS_LOCAL][IS_SORTED]      = IS_INFO_TRUE;
      is->info[itype][IS_UNIQUE]         = IS_INFO_TRUE;
      is->info[IS_LOCAL][IS_UNIQUE]      = IS_INFO_TRUE;
      is->info[itype][IS_PERMUTATION]    = IS_INFO_TRUE;
      is->info[itype][IS_INTERVAL]       = IS_INFO_TRUE;
      is->info[IS_LOCAL][IS_INTERVAL]    = IS_INFO_TRUE;
      if (permanent_set && permanent) {
        is->info_permanent[itype][IS_SORTED]         = PETSC_TRUE;
        is->info_permanent[IS_LOCAL][IS_SORTED]      = PETSC_TRUE;
        is->info_permanent[itype][IS_UNIQUE]         = PETSC_TRUE;
        is->info_permanent[IS_LOCAL][IS_UNIQUE]      = PETSC_TRUE;
        is->info_permanent[itype][IS_PERMUTATION]    = PETSC_TRUE;
        is->info_permanent[itype][IS_INTERVAL]       = PETSC_TRUE;
        is->info_permanent[IS_LOCAL][IS_INTERVAL]    = PETSC_TRUE;
      }
    }
    break;
  default:
    PetscCheckFalse(type == IS_LOCAL,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown IS property");
    else SETERRQ(PetscObjectComm((PetscObject)is), PETSC_ERR_ARG_OUTOFRANGE, "Unknown IS property");
  }
  PetscFunctionReturn(0);
}

/*@
   ISSetInfo - Set known information about an index set.

   Logically Collective on IS if type is IS_GLOBAL

   Input Parameters:
+  is - the index set
.  info - describing a property of the index set, one of those listed below,
.  type - IS_LOCAL if the information describes the local portion of the index set,
          IS_GLOBAL if it describes the whole index set
.  permanent - PETSC_TRUE if it is known that the property will persist through changes to the index set, PETSC_FALSE otherwise
               If the user sets a property as permanently known, it will bypass computation of that property
-  flg - set the described property as true (PETSC_TRUE) or false (PETSC_FALSE)

  Info Describing IS Structure:
+    IS_SORTED - the [local part of the] index set is sorted in ascending order
.    IS_UNIQUE - each entry in the [local part of the] index set is unique
.    IS_PERMUTATION - the [local part of the] index set is a permutation of the integers {0, 1, ..., N-1}, where N is the size of the [local part of the] index set
.    IS_INTERVAL - the [local part of the] index set is equal to a contiguous range of integers {f, f + 1, ..., f + N-1}
-    IS_IDENTITY - the [local part of the] index set is equal to the integers {0, 1, ..., N-1}

   Notes:
   If type is IS_GLOBAL, all processes that share the index set must pass the same value in flg

   It is possible to set a property with ISSetInfo() that contradicts what would be previously computed with ISGetInfo()

   Level: advanced

.seealso:  ISInfo, ISInfoType, IS

@*/
PetscErrorCode ISSetInfo(IS is, ISInfo info, ISInfoType type, PetscBool permanent, PetscBool flg)
{
  MPI_Comm       comm, errcomm;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidType(is,1);
  comm = PetscObjectComm((PetscObject)is);
  if (type == IS_GLOBAL) {
    PetscValidLogicalCollectiveEnum(is,info,2);
    PetscValidLogicalCollectiveBool(is,permanent,4);
    PetscValidLogicalCollectiveBool(is,flg,5);
    errcomm = comm;
  } else {
    errcomm = PETSC_COMM_SELF;
  }

  PetscCheckFalse(((int) info) <= IS_INFO_MIN || ((int) info) >= IS_INFO_MAX,errcomm,PETSC_ERR_ARG_OUTOFRANGE,"Options %d is out of range",(int)info);

  CHKERRMPI(MPI_Comm_size(comm, &size));
  /* do not use global values if size == 1: it makes it easier to keep the implications straight */
  if (size == 1) type = IS_LOCAL;
  CHKERRQ(ISSetInfo_Internal(is, info, type, permanent ? IS_INFO_TRUE : IS_INFO_FALSE, flg));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGetInfo_Sorted(IS is, ISInfoType type, PetscBool *flg)
{
  MPI_Comm       comm;
  PetscMPIInt    size, rank;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)is);
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_size(comm, &rank));
  if (type == IS_GLOBAL && is->ops->sortedglobal) {
    CHKERRQ((*is->ops->sortedglobal)(is,flg));
  } else {
    PetscBool sortedLocal = PETSC_FALSE;

    /* determine if the array is locally sorted */
    if (type == IS_GLOBAL && size > 1) {
      /* call ISGetInfo so that a cached value will be used if possible */
      CHKERRQ(ISGetInfo(is, IS_SORTED, IS_LOCAL, PETSC_TRUE, &sortedLocal));
    } else if (is->ops->sortedlocal) {
      CHKERRQ((*is->ops->sortedlocal)(is,&sortedLocal));
    } else {
      /* default: get the local indices and directly check */
      const PetscInt *idx;
      PetscInt n;

      CHKERRQ(ISGetIndices(is, &idx));
      CHKERRQ(ISGetLocalSize(is, &n));
      CHKERRQ(PetscSortedInt(n, idx, &sortedLocal));
      CHKERRQ(ISRestoreIndices(is, &idx));
    }

    if (type == IS_LOCAL || size == 1) {
      *flg = sortedLocal;
    } else {
      CHKERRMPI(MPI_Allreduce(&sortedLocal, flg, 1, MPIU_BOOL, MPI_LAND, comm));
      if (*flg) {
        PetscInt  n, min = PETSC_MAX_INT, max = PETSC_MIN_INT;
        PetscInt  maxprev;

        CHKERRQ(ISGetLocalSize(is, &n));
        if (n) CHKERRQ(ISGetMinMax(is, &min, &max));
        maxprev = PETSC_MIN_INT;
        CHKERRMPI(MPI_Exscan(&max, &maxprev, 1, MPIU_INT, MPI_MAX, comm));
        if (rank && (maxprev > min)) sortedLocal = PETSC_FALSE;
        CHKERRMPI(MPI_Allreduce(&sortedLocal, flg, 1, MPIU_BOOL, MPI_LAND, comm));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ISGetIndicesCopy(IS is, PetscInt idx[]);

static PetscErrorCode ISGetInfo_Unique(IS is, ISInfoType type, PetscBool *flg)
{
  MPI_Comm       comm;
  PetscMPIInt    size, rank;
  PetscInt       i;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)is);
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_size(comm, &rank));
  if (type == IS_GLOBAL && is->ops->uniqueglobal) {
    CHKERRQ((*is->ops->uniqueglobal)(is,flg));
  } else {
    PetscBool uniqueLocal;
    PetscInt  n = -1;
    PetscInt  *idx = NULL;

    /* determine if the array is locally unique */
    if (type == IS_GLOBAL && size > 1) {
      /* call ISGetInfo so that a cached value will be used if possible */
      CHKERRQ(ISGetInfo(is, IS_UNIQUE, IS_LOCAL, PETSC_TRUE, &uniqueLocal));
    } else if (is->ops->uniquelocal) {
      CHKERRQ((*is->ops->uniquelocal)(is,&uniqueLocal));
    } else {
      /* default: get the local indices and directly check */
      uniqueLocal = PETSC_TRUE;
      CHKERRQ(ISGetLocalSize(is, &n));
      CHKERRQ(PetscMalloc1(n, &idx));
      CHKERRQ(ISGetIndicesCopy(is, idx));
      CHKERRQ(PetscIntSortSemiOrdered(n, idx));
      for (i = 1; i < n; i++) if (idx[i] == idx[i-1]) break;
      if (i < n) uniqueLocal = PETSC_FALSE;
    }

    CHKERRQ(PetscFree(idx));
    if (type == IS_LOCAL || size == 1) {
      *flg = uniqueLocal;
    } else {
      CHKERRMPI(MPI_Allreduce(&uniqueLocal, flg, 1, MPIU_BOOL, MPI_LAND, comm));
      if (*flg) {
        PetscInt  min = PETSC_MAX_INT, max = PETSC_MIN_INT, maxprev;

        if (!idx) {
          CHKERRQ(ISGetLocalSize(is, &n));
          CHKERRQ(PetscMalloc1(n, &idx));
          CHKERRQ(ISGetIndicesCopy(is, idx));
        }
        CHKERRQ(PetscParallelSortInt(is->map, is->map, idx, idx));
        if (n) {
          min = idx[0];
          max = idx[n - 1];
        }
        for (i = 1; i < n; i++) if (idx[i] == idx[i-1]) break;
        if (i < n) uniqueLocal = PETSC_FALSE;
        maxprev = PETSC_MIN_INT;
        CHKERRMPI(MPI_Exscan(&max, &maxprev, 1, MPIU_INT, MPI_MAX, comm));
        if (rank && (maxprev == min)) uniqueLocal = PETSC_FALSE;
        CHKERRMPI(MPI_Allreduce(&uniqueLocal, flg, 1, MPIU_BOOL, MPI_LAND, comm));
      }
    }
    CHKERRQ(PetscFree(idx));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGetInfo_Permutation(IS is, ISInfoType type, PetscBool *flg)
{
  MPI_Comm       comm;
  PetscMPIInt    size, rank;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)is);
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_size(comm, &rank));
  if (type == IS_GLOBAL && is->ops->permglobal) {
    CHKERRQ((*is->ops->permglobal)(is,flg));
  } else if (type == IS_LOCAL && is->ops->permlocal) {
    CHKERRQ((*is->ops->permlocal)(is,flg));
  } else {
    PetscBool permLocal;
    PetscInt  n, i, rStart;
    PetscInt  *idx;

    CHKERRQ(ISGetLocalSize(is, &n));
    CHKERRQ(PetscMalloc1(n, &idx));
    CHKERRQ(ISGetIndicesCopy(is, idx));
    if (type == IS_GLOBAL) {
      CHKERRQ(PetscParallelSortInt(is->map, is->map, idx, idx));
      CHKERRQ(PetscLayoutGetRange(is->map, &rStart, NULL));
    } else {
      CHKERRQ(PetscIntSortSemiOrdered(n, idx));
      rStart = 0;
    }
    permLocal = PETSC_TRUE;
    for (i = 0; i < n; i++) {
      if (idx[i] != rStart + i) break;
    }
    if (i < n) permLocal = PETSC_FALSE;
    if (type == IS_LOCAL || size == 1) {
      *flg = permLocal;
    } else {
      CHKERRMPI(MPI_Allreduce(&permLocal, flg, 1, MPIU_BOOL, MPI_LAND, comm));
    }
    CHKERRQ(PetscFree(idx));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGetInfo_Interval(IS is, ISInfoType type, PetscBool *flg)
{
  MPI_Comm       comm;
  PetscMPIInt    size, rank;
  PetscInt       i;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)is);
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_size(comm, &rank));
  if (type == IS_GLOBAL && is->ops->intervalglobal) {
    CHKERRQ((*is->ops->intervalglobal)(is,flg));
  } else {
    PetscBool intervalLocal;

    /* determine if the array is locally an interval */
    if (type == IS_GLOBAL && size > 1) {
      /* call ISGetInfo so that a cached value will be used if possible */
      CHKERRQ(ISGetInfo(is, IS_INTERVAL, IS_LOCAL, PETSC_TRUE, &intervalLocal));
    } else if (is->ops->intervallocal) {
      CHKERRQ((*is->ops->intervallocal)(is,&intervalLocal));
    } else {
      PetscInt        n;
      const PetscInt  *idx;
      /* default: get the local indices and directly check */
      intervalLocal = PETSC_TRUE;
      CHKERRQ(ISGetLocalSize(is, &n));
      CHKERRQ(PetscMalloc1(n, &idx));
      CHKERRQ(ISGetIndices(is, &idx));
      for (i = 1; i < n; i++) if (idx[i] != idx[i-1] + 1) break;
      if (i < n) intervalLocal = PETSC_FALSE;
      CHKERRQ(ISRestoreIndices(is, &idx));
    }

    if (type == IS_LOCAL || size == 1) {
      *flg = intervalLocal;
    } else {
      CHKERRMPI(MPI_Allreduce(&intervalLocal, flg, 1, MPIU_BOOL, MPI_LAND, comm));
      if (*flg) {
        PetscInt  n, min = PETSC_MAX_INT, max = PETSC_MIN_INT;
        PetscInt  maxprev;

        CHKERRQ(ISGetLocalSize(is, &n));
        if (n) CHKERRQ(ISGetMinMax(is, &min, &max));
        maxprev = PETSC_MIN_INT;
        CHKERRMPI(MPI_Exscan(&max, &maxprev, 1, MPIU_INT, MPI_MAX, comm));
        if (rank && n && (maxprev != min - 1)) intervalLocal = PETSC_FALSE;
        CHKERRMPI(MPI_Allreduce(&intervalLocal, flg, 1, MPIU_BOOL, MPI_LAND, comm));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGetInfo_Identity(IS is, ISInfoType type, PetscBool *flg)
{
  MPI_Comm       comm;
  PetscMPIInt    size, rank;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)is);
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_size(comm, &rank));
  if (type == IS_GLOBAL && is->ops->intervalglobal) {
    PetscBool isinterval;

    CHKERRQ((*is->ops->intervalglobal)(is,&isinterval));
    *flg = PETSC_FALSE;
    if (isinterval) {
      PetscInt  min;

      CHKERRQ(ISGetMinMax(is, &min, NULL));
      CHKERRMPI(MPI_Bcast(&min, 1, MPIU_INT, 0, comm));
      if (min == 0) *flg = PETSC_TRUE;
    }
  } else if (type == IS_LOCAL && is->ops->intervallocal) {
    PetscBool isinterval;

    CHKERRQ((*is->ops->intervallocal)(is,&isinterval));
    *flg = PETSC_FALSE;
    if (isinterval) {
      PetscInt  min;

      CHKERRQ(ISGetMinMax(is, &min, NULL));
      if (min == 0) *flg = PETSC_TRUE;
    }
  } else {
    PetscBool identLocal;
    PetscInt  n, i, rStart;
    const PetscInt *idx;

    CHKERRQ(ISGetLocalSize(is, &n));
    CHKERRQ(ISGetIndices(is, &idx));
    CHKERRQ(PetscLayoutGetRange(is->map, &rStart, NULL));
    identLocal = PETSC_TRUE;
    for (i = 0; i < n; i++) {
      if (idx[i] != rStart + i) break;
    }
    if (i < n) identLocal = PETSC_FALSE;
    if (type == IS_LOCAL || size == 1) {
      *flg = identLocal;
    } else {
      CHKERRMPI(MPI_Allreduce(&identLocal, flg, 1, MPIU_BOOL, MPI_LAND, comm));
    }
    CHKERRQ(ISRestoreIndices(is, &idx));
  }
  PetscFunctionReturn(0);
}

/*@
   ISGetInfo - Determine whether an index set satisfies a given property

   Collective or logically collective on IS if the type is IS_GLOBAL (logically collective if the value of the property has been permanently set with ISSetInfo())

   Input Parameters:
+  is - the index set
.  info - describing a property of the index set, one of those listed in the documentation of ISSetInfo()
.  compute - if PETSC_FALSE, the property will not be computed if it is not already known and the property will be assumed to be false
-  type - whether the property is local (IS_LOCAL) or global (IS_GLOBAL)

   Output Parameter:
.  flg - wheter the property is true (PETSC_TRUE) or false (PETSC_FALSE)

   Note: ISGetInfo uses cached values when possible, which will be incorrect if ISSetInfo() has been called with incorrect information.  To clear cached values, use ISClearInfoCache().

   Level: advanced

.seealso:  ISInfo, ISInfoType, ISSetInfo(), ISClearInfoCache()

@*/
PetscErrorCode ISGetInfo(IS is, ISInfo info, ISInfoType type, PetscBool compute, PetscBool *flg)
{
  MPI_Comm       comm, errcomm;
  PetscMPIInt    rank, size;
  PetscInt       itype;
  PetscBool      hasprop;
  PetscBool      infer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidType(is,1);
  comm = PetscObjectComm((PetscObject)is);
  if (type == IS_GLOBAL) {
    PetscValidLogicalCollectiveEnum(is,info,2);
    errcomm = comm;
  } else {
    errcomm = PETSC_COMM_SELF;
  }

  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));

  PetscCheckFalse(((int) info) <= IS_INFO_MIN || ((int) info) >= IS_INFO_MAX,errcomm,PETSC_ERR_ARG_OUTOFRANGE,"Options %d is out of range",(int)info);
  if (size == 1) type = IS_LOCAL;
  itype = (type == IS_LOCAL) ? 0 : 1;
  hasprop = PETSC_FALSE;
  infer = PETSC_FALSE;
  if (is->info_permanent[itype][(int)info]) {
    hasprop = (is->info[itype][(int)info] == IS_INFO_TRUE) ? PETSC_TRUE : PETSC_FALSE;
    infer = PETSC_TRUE;
  } else if ((itype == IS_LOCAL) && (is->info[IS_LOCAL][info] != IS_INFO_UNKNOWN)) {
    /* we can cache local properties as long as we clear them when the IS changes */
    /* NOTE: we only cache local values because there is no ISAssemblyBegin()/ISAssemblyEnd(),
     so we have no way of knowing when a cached value has been invalidated by changes on a different process */
    hasprop = (is->info[itype][(int)info] == IS_INFO_TRUE) ? PETSC_TRUE : PETSC_FALSE;
    infer = PETSC_TRUE;
  } else if (compute) {
    switch (info) {
    case IS_SORTED:
      CHKERRQ(ISGetInfo_Sorted(is, type, &hasprop));
      break;
    case IS_UNIQUE:
      CHKERRQ(ISGetInfo_Unique(is, type, &hasprop));
      break;
    case IS_PERMUTATION:
      CHKERRQ(ISGetInfo_Permutation(is, type, &hasprop));
      break;
    case IS_INTERVAL:
      CHKERRQ(ISGetInfo_Interval(is, type, &hasprop));
      break;
    case IS_IDENTITY:
      CHKERRQ(ISGetInfo_Identity(is, type, &hasprop));
      break;
    default:
      SETERRQ(errcomm, PETSC_ERR_ARG_OUTOFRANGE, "Unknown IS property");
    }
    infer = PETSC_TRUE;
  }
  /* call ISSetInfo_Internal to keep all of the implications straight */
  if (infer) CHKERRQ(ISSetInfo_Internal(is, info, type, IS_INFO_UNKNOWN, hasprop));
  *flg = hasprop;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISCopyInfo(IS source, IS dest)
{
  PetscFunctionBegin;
  CHKERRQ(PetscArraycpy(&dest->info[0], &source->info[0], 2));
  CHKERRQ(PetscArraycpy(&dest->info_permanent[0], &source->info_permanent[0], 2));
  PetscFunctionReturn(0);
}

/*@
   ISIdentity - Determines whether index set is the identity mapping.

   Collective on IS

   Input Parameters:
.  is - the index set

   Output Parameters:
.  ident - PETSC_TRUE if an identity, else PETSC_FALSE

   Level: intermediate

   Note: If ISSetIdentity() (or ISSetInfo() for a permanent property) has been called,
   ISIdentity() will return its answer without communication between processes, but
   otherwise the output ident will be computed from ISGetInfo(),
   which may require synchronization on the communicator of IS.  To avoid this computation,
   call ISGetInfo() directly with the compute flag set to PETSC_FALSE, and ident will be assumed false.

.seealso: ISSetIdentity(), ISGetInfo()
@*/
PetscErrorCode  ISIdentity(IS is,PetscBool  *ident)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidBoolPointer(ident,2);
  CHKERRQ(ISGetInfo(is,IS_IDENTITY,IS_GLOBAL,PETSC_TRUE,ident));
  PetscFunctionReturn(0);
}

/*@
   ISSetIdentity - Informs the index set that it is an identity.

   Logically Collective on IS

   Input Parameter:
.  is - the index set

   Level: intermediate

   Note: The IS will be considered the identity permanently, even if indices have been changes (for example, with
   ISGeneralSetIndices()).  It's a good idea to only set this property if the IS will not change in the future.
   To clear this property, use ISClearInfoCache().

.seealso: ISIdentity(), ISSetInfo(), ISClearInfoCache()
@*/
PetscErrorCode  ISSetIdentity(IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  CHKERRQ(ISSetInfo(is,IS_IDENTITY,IS_GLOBAL,PETSC_TRUE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@
   ISContiguousLocal - Locates an index set with contiguous range within a global range, if possible

   Not Collective

   Input Parameters:
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidIntPointer(start,4);
  PetscValidBoolPointer(contig,5);
  *start  = -1;
  *contig = PETSC_FALSE;
  if (is->ops->contiguous) {
    CHKERRQ((*is->ops->contiguous)(is,gstart,gend,start,contig));
  }
  PetscFunctionReturn(0);
}

/*@
   ISPermutation - PETSC_TRUE or PETSC_FALSE depending on whether the
   index set has been declared to be a permutation.

   Logically Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  perm - PETSC_TRUE if a permutation, else PETSC_FALSE

   Level: intermediate

   Note: If it is not alread known that the IS is a permutation (if ISSetPermutation()
   or ISSetInfo() has not been called), this routine will not attempt to compute
   whether the index set is a permutation and will assume perm is PETSC_FALSE.
   To compute the value when it is not already known, use ISGetInfo() with
   the compute flag set to PETSC_TRUE.

.seealso: ISSetPermutation(), ISGetInfo()
@*/
PetscErrorCode  ISPermutation(IS is,PetscBool  *perm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidBoolPointer(perm,2);
  CHKERRQ(ISGetInfo(is,IS_PERMUTATION,IS_GLOBAL,PETSC_FALSE,perm));
  PetscFunctionReturn(0);
}

/*@
   ISSetPermutation - Informs the index set that it is a permutation.

   Logically Collective on IS

   Input Parameter:
.  is - the index set

   Level: intermediate

   The debug version of the libraries (./configure --with-debugging=1) checks if the
  index set is actually a permutation. The optimized version just believes you.

   Note: The IS will be considered a permutation permanently, even if indices have been changes (for example, with
   ISGeneralSetIndices()).  It's a good idea to only set this property if the IS will not change in the future.
   To clear this property, use ISClearInfoCache().

.seealso: ISPermutation(), ISSetInfo(), ISClearInfoCache().
@*/
PetscErrorCode  ISSetPermutation(IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (PetscDefined(USE_DEBUG)) {
    PetscMPIInt    size;

    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)is),&size));
    if (size == 1) {
      PetscInt       i,n,*idx;
      const PetscInt *iidx;

      CHKERRQ(ISGetSize(is,&n));
      CHKERRQ(PetscMalloc1(n,&idx));
      CHKERRQ(ISGetIndices(is,&iidx));
      CHKERRQ(PetscArraycpy(idx,iidx,n));
      CHKERRQ(PetscIntSortSemiOrdered(n,idx));
      for (i=0; i<n; i++) {
        PetscCheckFalse(idx[i] != i,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Index set is not a permutation");
      }
      CHKERRQ(PetscFree(idx));
      CHKERRQ(ISRestoreIndices(is,&iidx));
    }
  }
  CHKERRQ(ISSetInfo(is,IS_PERMUTATION,IS_GLOBAL,PETSC_TRUE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@C
   ISDestroy - Destroys an index set.

   Collective on IS

   Input Parameters:
.  is - the index set

   Level: beginner

.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlocked()
@*/
PetscErrorCode  ISDestroy(IS *is)
{
  PetscFunctionBegin;
  if (!*is) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*is),IS_CLASSID,1);
  if (--((PetscObject)(*is))->refct > 0) {*is = NULL; PetscFunctionReturn(0);}
  if ((*is)->complement) {
    PetscInt refcnt;
    CHKERRQ(PetscObjectGetReference((PetscObject)((*is)->complement), &refcnt));
    PetscCheckFalse(refcnt > 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Nonlocal IS has not been restored");
    CHKERRQ(ISDestroy(&(*is)->complement));
  }
  if ((*is)->ops->destroy) {
    CHKERRQ((*(*is)->ops->destroy)(*is));
  }
  CHKERRQ(PetscLayoutDestroy(&(*is)->map));
  /* Destroy local representations of offproc data. */
  CHKERRQ(PetscFree((*is)->total));
  CHKERRQ(PetscFree((*is)->nonlocal));
  CHKERRQ(PetscHeaderDestroy(is));
  PetscFunctionReturn(0);
}

/*@
   ISInvertPermutation - Creates a new permutation that is the inverse of
                         a given permutation.

   Collective on IS

   Input Parameters:
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
  PetscBool      isperm, isidentity, issame;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(isout,3);
  CHKERRQ(ISGetInfo(is,IS_PERMUTATION,IS_GLOBAL,PETSC_TRUE,&isperm));
  PetscCheck(isperm,PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_WRONG,"Not a permutation");
  CHKERRQ(ISGetInfo(is,IS_IDENTITY,IS_GLOBAL,PETSC_TRUE,&isidentity));
  issame = PETSC_FALSE;
  if (isidentity) {
    PetscInt n;
    PetscBool isallsame;

    CHKERRQ(ISGetLocalSize(is, &n));
    issame = (PetscBool) (n == nlocal);
    CHKERRMPI(MPI_Allreduce(&issame, &isallsame, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)is)));
    issame = isallsame;
  }
  if (issame) {
    CHKERRQ(ISDuplicate(is,isout));
  } else {
    CHKERRQ((*is->ops->invertpermutation)(is,nlocal,isout));
    CHKERRQ(ISSetPermutation(*isout));
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

   Input Parameter:
.  is - the index set

   Output Parameter:
.  map - the layout

   Level: developer

.seealso: ISSetLayout(), ISGetSize(), ISGetLocalSize()
@*/
PetscErrorCode ISGetLayout(IS is,PetscLayout *map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(map,2);
  *map = is->map;
  PetscFunctionReturn(0);
}

/*@
   ISSetLayout - set PetscLayout describing index set layout

   Collective

   Input Arguments:
+  is - the index set
-  map - the layout

   Level: developer

   Notes:
   Users should typically use higher level functions such as ISCreateGeneral().

   This function can be useful in some special cases of constructing a new IS, e.g. after ISCreate() and before ISLoad().
   Otherwise, it is only valid to replace the layout with a layout known to be equivalent.

.seealso: ISCreate(), ISGetLayout(), ISGetSize(), ISGetLocalSize()
@*/
PetscErrorCode ISSetLayout(IS is,PetscLayout map)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(map,2);
  CHKERRQ(PetscLayoutReference(map,&is->map));
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
   petsc/src/is/[tutorials,tests] for details.

   Level: intermediate

.seealso: ISRestoreIndices(), ISGetIndicesF90()
@*/
PetscErrorCode  ISGetIndices(IS is,const PetscInt *ptr[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(ptr,2);
  CHKERRQ((*is->ops->getindices)(is,ptr));
  PetscFunctionReturn(0);
}

/*@C
   ISGetMinMax - Gets the minimum and maximum values in an IS

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameters:
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

  Input Parameters:
+ is - the index set
- key - the search key

  Output Parameter:
. location - if >= 0, a location within the index set that is equal to the key, otherwise the key is not in the index set

  Level: intermediate
@*/
PetscErrorCode ISLocate(IS is, PetscInt key, PetscInt *location)
{
  PetscFunctionBegin;
  if (is->ops->locate) {
    CHKERRQ((*is->ops->locate)(is,key,location));
  } else {
    PetscInt       numIdx;
    PetscBool      sorted;
    const PetscInt *idx;

    CHKERRQ(ISGetLocalSize(is,&numIdx));
    CHKERRQ(ISGetIndices(is,&idx));
    CHKERRQ(ISSorted(is,&sorted));
    if (sorted) {
      CHKERRQ(PetscFindInt(key,numIdx,idx,location));
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
    CHKERRQ(ISRestoreIndices(is,&idx));
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
   petsc/src/vec/is/tests for details.

   Level: intermediate

   Note:
   This routine zeros out ptr. This is to prevent accidental us of the array after it has been restored.

.seealso: ISGetIndices(), ISRestoreIndicesF90()
@*/
PetscErrorCode  ISRestoreIndices(IS is,const PetscInt *ptr[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(ptr,2);
  if (is->ops->restoreindices) {
    CHKERRQ((*is->ops->restoreindices)(is,ptr));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGatherTotal_Private(IS is)
{
  PetscInt       i,n,N;
  const PetscInt *lindices;
  MPI_Comm       comm;
  PetscMPIInt    rank,size,*sizes = NULL,*offsets = NULL,nn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);

  CHKERRQ(PetscObjectGetComm((PetscObject)is,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  CHKERRQ(ISGetLocalSize(is,&n));
  CHKERRQ(PetscMalloc2(size,&sizes,size,&offsets));

  CHKERRQ(PetscMPIIntCast(n,&nn));
  CHKERRMPI(MPI_Allgather(&nn,1,MPI_INT,sizes,1,MPI_INT,comm));
  offsets[0] = 0;
  for (i=1; i<size; ++i) offsets[i] = offsets[i-1] + sizes[i-1];
  N = offsets[size-1] + sizes[size-1];

  CHKERRQ(PetscMalloc1(N,&(is->total)));
  CHKERRQ(ISGetIndices(is,&lindices));
  CHKERRMPI(MPI_Allgatherv((void*)lindices,nn,MPIU_INT,is->total,sizes,offsets,MPIU_INT,comm));
  CHKERRQ(ISRestoreIndices(is,&lindices));
  is->local_offset = offsets[rank];
  CHKERRQ(PetscFree2(sizes,offsets));
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
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)is), &size));
  if (size == 1) {
    CHKERRQ((*is->ops->getindices)(is,indices));
  } else {
    if (!is->total) {
      CHKERRQ(ISGatherTotal_Private(is));
    }
    *indices = is->total;
  }
  PetscFunctionReturn(0);
}

/*@C
   ISRestoreTotalIndices - Restore the index array obtained with ISGetTotalIndices().

   Not Collective.

   Input Parameters:
+  is - the index set
-  indices - index array; must be the array obtained with ISGetTotalIndices()

   Level: intermediate

.seealso: ISRestoreTotalIndices(), ISGetNonlocalIndices()
@*/
PetscErrorCode  ISRestoreTotalIndices(IS is, const PetscInt *indices[])
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)is), &size));
  if (size == 1) {
    CHKERRQ((*is->ops->restoreindices)(is,indices));
  } else {
    PetscCheckFalse(is->total != *indices,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index array pointer being restored does not point to the array obtained from the IS.");
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
  PetscMPIInt    size;
  PetscInt       n, N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(indices,2);
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)is), &size));
  if (size == 1) *indices = NULL;
  else {
    if (!is->total) {
      CHKERRQ(ISGatherTotal_Private(is));
    }
    CHKERRQ(ISGetLocalSize(is,&n));
    CHKERRQ(ISGetSize(is,&N));
    CHKERRQ(PetscMalloc1(N-n, &(is->nonlocal)));
    CHKERRQ(PetscArraycpy(is->nonlocal, is->total, is->local_offset));
    CHKERRQ(PetscArraycpy(is->nonlocal+is->local_offset, is->total+is->local_offset+n,N - is->local_offset - n));
    *indices = is->nonlocal;
  }
  PetscFunctionReturn(0);
}

/*@C
   ISRestoreNonlocalIndices - Restore the index array obtained with ISGetNonlocalIndices().

   Not Collective.

   Input Parameters:
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
  PetscCheckFalse(is->nonlocal != *indices,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Index array pointer being restored does not point to the array obtained from the IS.");
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(complement,2);
  /* Check if the complement exists already. */
  if (is->complement) {
    *complement = is->complement;
    CHKERRQ(PetscObjectReference((PetscObject)(is->complement)));
  } else {
    PetscInt       N, n;
    const PetscInt *idx;
    CHKERRQ(ISGetSize(is, &N));
    CHKERRQ(ISGetLocalSize(is,&n));
    CHKERRQ(ISGetNonlocalIndices(is, &idx));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, N-n,idx, PETSC_USE_POINTER, &(is->complement)));
    CHKERRQ(PetscObjectReference((PetscObject)is->complement));
    *complement = is->complement;
  }
  PetscFunctionReturn(0);
}

/*@
   ISRestoreNonlocalIS - Restore the IS obtained with ISGetNonlocalIS().

   Not collective.

   Input Parameters:
+  is         - the index set
-  complement - index set of is's nonlocal indices

   Level: intermediate

.seealso: ISGetNonlocalIS(), ISGetNonlocalIndices(), ISRestoreNonlocalIndices()
@*/
PetscErrorCode  ISRestoreNonlocalIS(IS is, IS *complement)
{
  PetscInt       refcnt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(complement,2);
  PetscCheckFalse(*complement != is->complement,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Complement IS being restored was not obtained with ISGetNonlocalIS()");
  CHKERRQ(PetscObjectGetReference((PetscObject)(is->complement), &refcnt));
  PetscCheckFalse(refcnt <= 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Duplicate call to ISRestoreNonlocalIS() detected");
  CHKERRQ(PetscObjectDereference((PetscObject)(is->complement)));
  PetscFunctionReturn(0);
}

/*@C
   ISViewFromOptions - View from Options

   Collective on IS

   Input Parameters:
+  A - the index set
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  IS, ISView, PetscObjectViewFromOptions(), ISCreate()
@*/
PetscErrorCode  ISViewFromOptions(IS A,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,IS_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)A,obj,name));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (!viewer) CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)is),&viewer));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(is,1,viewer,2);

  CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)is,viewer));
  CHKERRQ(PetscLogEventBegin(IS_View,is,viewer,0,0));
  CHKERRQ((*is->ops->view)(is,viewer));
  CHKERRQ(PetscLogEventEnd(IS_View,is,viewer,0,0));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(is,1,viewer,2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5, &ishdf5));
  PetscCheckFalse(!isbinary && !ishdf5,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");
  if (!((PetscObject)is)->type_name) CHKERRQ(ISSetType(is, ISGENERAL));
  CHKERRQ(PetscLogEventBegin(IS_Load,is,viewer,0,0));
  CHKERRQ((*is->ops->load)(is, viewer));
  CHKERRQ(PetscLogEventEnd(IS_Load,is,viewer,0,0));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  CHKERRQ((*is->ops->sort)(is));
  CHKERRQ(ISSetInfo(is,IS_SORTED,IS_LOCAL,is->info_permanent[IS_LOCAL][IS_SORTED],PETSC_TRUE));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  CHKERRQ(ISClearInfoCache(is,PETSC_FALSE));
  CHKERRQ((*is->ops->sortremovedups)(is));
  CHKERRQ(ISSetInfo(is,IS_SORTED,IS_LOCAL,is->info_permanent[IS_LOCAL][IS_SORTED],PETSC_TRUE));
  CHKERRQ(ISSetInfo(is,IS_UNIQUE,IS_LOCAL,is->info_permanent[IS_LOCAL][IS_UNIQUE],PETSC_TRUE));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (is->ops->togeneral) {
    CHKERRQ((*is->ops->togeneral)(is));
  } else SETERRQ(PetscObjectComm((PetscObject)is),PETSC_ERR_SUP,"Not written for this type %s",((PetscObject)is)->type_name);
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  CHKERRQ(ISGetInfo(is,IS_SORTED,IS_LOCAL,PETSC_TRUE,flg));
  PetscFunctionReturn(0);
}

/*@
   ISDuplicate - Creates a duplicate copy of an index set.

   Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  isnew - the copy of the index set

   Level: beginner

.seealso: ISCreateGeneral(), ISCopy()
@*/
PetscErrorCode  ISDuplicate(IS is,IS *newIS)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(newIS,2);
  CHKERRQ((*is->ops->duplicate)(is,newIS));
  CHKERRQ(ISCopyInfo(is,*newIS));
  PetscFunctionReturn(0);
}

/*@
   ISCopy - Copies an index set.

   Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  isy - the copy of the index set

   Level: beginner

.seealso: ISDuplicate()
@*/
PetscErrorCode  ISCopy(IS is,IS isy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidHeaderSpecific(isy,IS_CLASSID,2);
  PetscCheckSameComm(is,1,isy,2);
  if (is == isy) PetscFunctionReturn(0);
  CHKERRQ(ISCopyInfo(is,isy));
  isy->max        = is->max;
  isy->min        = is->min;
  CHKERRQ((*is->ops->copy)(is,isy));
  PetscFunctionReturn(0);
}

/*@
   ISOnComm - Split a parallel IS on subcomms (usually self) or concatenate index sets on subcomms into a parallel index set

   Collective on IS

   Input Parameters:
+ is - index set
. comm - communicator for new index set
- mode - copy semantics, PETSC_USE_POINTER for no-copy if possible, otherwise PETSC_COPY_VALUES

   Output Parameter:
. newis - new IS on comm

   Level: advanced

   Notes:
   It is usually desirable to create a parallel IS and look at the local part when necessary.

   This function is useful if serial ISs must be created independently, or to view many
   logically independent serial ISs.

   The input IS must have the same type on every process.
@*/
PetscErrorCode  ISOnComm(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  PetscMPIInt    match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(newis,4);
  CHKERRMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)is),comm,&match));
  if (mode != PETSC_COPY_VALUES && (match == MPI_IDENT || match == MPI_CONGRUENT)) {
    CHKERRQ(PetscObjectReference((PetscObject)is));
    *newis = is;
  } else {
    CHKERRQ((*is->ops->oncomm)(is,comm,mode,newis));
  }
  PetscFunctionReturn(0);
}

/*@
   ISSetBlockSize - informs an index set that it has a given block size

   Logicall Collective on IS

   Input Parameters:
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidLogicalCollectiveInt(is,bs,2);
  PetscCheckFalse(bs < 1,PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_OUTOFRANGE,"Block size %" PetscInt_FMT ", must be positive",bs);
  if (PetscDefined(USE_DEBUG)) {
    const PetscInt *indices;
    PetscInt       length,i,j;
    CHKERRQ(ISGetIndices(is,&indices));
    if (indices) {
      CHKERRQ(ISGetLocalSize(is,&length));
      PetscCheckFalse(length%bs != 0,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size %D not compatible with block size %D",length,bs);
      for (i=0;i<length/bs;i+=bs) {
        for (j=0;j<bs-1;j++) {
          PetscCheckFalse(indices[i*bs+j] != indices[i*bs+j+1]-1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block size %" PetscInt_FMT " is incompatible with the indices: non consecutive indices %" PetscInt_FMT " %" PetscInt_FMT,bs,indices[i*bs+j],indices[i*bs+j+1]);
        }
      }
    }
    CHKERRQ(ISRestoreIndices(is,&indices));
  }
  CHKERRQ((*is->ops->setblocksize)(is,bs));
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
  PetscFunctionBegin;
  CHKERRQ(PetscLayoutGetBlockSize(is->map, size));
  PetscFunctionReturn(0);
}

PetscErrorCode ISGetIndicesCopy(IS is, PetscInt idx[])
{
  PetscInt       len,i;
  const PetscInt *ptr;

  PetscFunctionBegin;
  CHKERRQ(ISGetLocalSize(is,&len));
  CHKERRQ(ISGetIndices(is,&ptr));
  for (i=0; i<len; i++) idx[i] = ptr[i];
  CHKERRQ(ISRestoreIndices(is,&ptr));
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
