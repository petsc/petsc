
/*
     Provides the functions for index sets (IS) defined by a list of integers.
*/
#include <../src/vec/is/is/impls/general/general.h> /*I  "petscis.h"  I*/
#include <petsc/private/viewerhdf5impl.h>

static PetscErrorCode ISDuplicate_General(IS is,IS *newIS)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       n;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject) is), n, sub->idx, PETSC_COPY_VALUES, newIS));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISDestroy_General(IS is)
{
  IS_General     *is_general = (IS_General*)is->data;

  PetscFunctionBegin;
  if (is_general->allocated) PetscCall(PetscFree(is_general->idx));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISGeneralSetIndices_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISGeneralFilter_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISGeneralSetIndicesFromMask_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISShift_C",NULL));
  PetscCall(PetscFree(is->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISCopy_General(IS is,IS isy)
{
  IS_General     *is_general = (IS_General*)is->data,*isy_general = (IS_General*)isy->data;
  PetscInt       n;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(PetscArraycpy(isy_general->idx,is_general->idx,n));
  PetscFunctionReturn(0);
}

PetscErrorCode ISShift_General(IS is, PetscInt shift, IS isy)
{
  IS_General     *is_general = (IS_General*)is->data,*isy_general = (IS_General*)isy->data;
  PetscInt       i, n;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  for (i=0; i<n; i++) isy_general->idx[i] = is_general->idx[i] + shift;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISOnComm_General(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       n;

  PetscFunctionBegin;
  PetscCheck(mode != PETSC_OWN_POINTER,comm,PETSC_ERR_ARG_WRONG,"Cannot use PETSC_OWN_POINTER");
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(ISCreateGeneral(comm,n,sub->idx,mode,newis));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSetBlockSize_General(IS is,PetscInt bs)
{
  PetscFunctionBegin;
  PetscCall(PetscLayoutSetBlockSize(is->map, bs));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISContiguousLocal_General(IS is,PetscInt gstart,PetscInt gend,PetscInt *start,PetscBool *contig)
{
  IS_General *sub = (IS_General*)is->data;
  PetscInt   n,i,p;

  PetscFunctionBegin;
  *start  = 0;
  *contig = PETSC_TRUE;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  if (!n) PetscFunctionReturn(0);
  p = sub->idx[0];
  if (p < gstart) goto nomatch;
  *start = p - gstart;
  if (n > gend-p) goto nomatch;
  for (i=1; i<n; i++,p++) {
    if (sub->idx[i] != p+1) goto nomatch;
  }
  PetscFunctionReturn(0);
nomatch:
  *start  = -1;
  *contig = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISLocate_General(IS is,PetscInt key,PetscInt *location)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       numIdx, i;
  PetscBool      sorted;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map,&numIdx));
  PetscCall(ISGetInfo(is,IS_SORTED,IS_LOCAL,PETSC_TRUE,&sorted));
  if (sorted) PetscCall(PetscFindInt(key,numIdx,sub->idx,location));
  else {
    const PetscInt *idx = sub->idx;

    *location = -1;
    for (i = 0; i < numIdx; i++) {
      if (idx[i] == key) {
        *location = i;
        PetscFunctionReturn(0);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGetIndices_General(IS in,const PetscInt *idx[])
{
  IS_General *sub = (IS_General*)in->data;

  PetscFunctionBegin;
  *idx = sub->idx;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISRestoreIndices_General(IS in,const PetscInt *idx[])
{
  IS_General *sub = (IS_General*)in->data;

  PetscFunctionBegin;
   /* F90Array1dCreate() inside ISRestoreArrayF90() does not keep array when zero length array */
  PetscCheck(in->map->n <= 0 || *idx == sub->idx,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from ISGetIndices()");
  PetscFunctionReturn(0);
}

static PetscErrorCode ISInvertPermutation_General(IS is,PetscInt nlocal,IS *isout)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       i,*ii,n,nstart;
  const PetscInt *idx = sub->idx;
  PetscMPIInt    size;
  IS             istmp,nistmp;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)is),&size));
  if (size == 1) {
    PetscCall(PetscMalloc1(n,&ii));
    for (i=0; i<n; i++) ii[idx[i]] = i;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_OWN_POINTER,isout));
    PetscCall(ISSetPermutation(*isout));
  } else {
    /* crude, nonscalable get entire IS on each processor */
    PetscCall(ISAllGather(is,&istmp));
    PetscCall(ISSetPermutation(istmp));
    PetscCall(ISInvertPermutation(istmp,PETSC_DECIDE,&nistmp));
    PetscCall(ISDestroy(&istmp));
    /* get the part we need */
    if (nlocal == PETSC_DECIDE) nlocal = n;
    PetscCallMPI(MPI_Scan(&nlocal,&nstart,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)is)));
    if (PetscDefined(USE_DEBUG)) {
      PetscInt    N;
      PetscMPIInt rank;
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)is),&rank));
      PetscCall(PetscLayoutGetSize(is->map, &N));
      PetscCheck((rank != size-1) || (nstart == N),PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sum of nlocal lengths %" PetscInt_FMT " != total IS length %" PetscInt_FMT,nstart,N);
    }
    nstart -= nlocal;
    PetscCall(ISGetIndices(nistmp,&idx));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is),nlocal,idx+nstart,PETSC_COPY_VALUES,isout));
    PetscCall(ISRestoreIndices(nistmp,&idx));
    PetscCall(ISDestroy(&nistmp));
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HDF5)
static PetscErrorCode ISView_General_HDF5(IS is, PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  hid_t           filespace;  /* file dataspace identifier */
  hid_t           chunkspace; /* chunk dataset property identifier */
  hid_t           dset_id;    /* dataset identifier */
  hid_t           memspace;   /* memory dataspace identifier */
  hid_t           inttype;    /* int type (H5T_NATIVE_INT or H5T_NATIVE_LLONG) */
  hid_t           file_id, group;
  hsize_t         dim, maxDims[3], dims[3], chunkDims[3], count[3],offset[3];
  PetscBool       timestepping;
  PetscInt        bs, N, n, timestep=PETSC_MIN_INT, low;
  hsize_t         chunksize;
  const PetscInt *ind;
  const char     *isname;

  PetscFunctionBegin;
  PetscCall(ISGetBlockSize(is,&bs));
  bs   = PetscMax(bs, 1); /* If N = 0, bs  = 0 as well */
  PetscCall(PetscViewerHDF5OpenGroup(viewer, &file_id, &group));
  PetscCall(PetscViewerHDF5IsTimestepping(viewer, &timestepping));
  if (timestepping) {
    PetscCall(PetscViewerHDF5GetTimestep(viewer, &timestep));
  }

  /* Create the dataspace for the dataset.
   *
   * dims - holds the current dimensions of the dataset
   *
   * maxDims - holds the maximum dimensions of the dataset (unlimited
   * for the number of time steps with the current dimensions for the
   * other dimensions; so only additional time steps can be added).
   *
   * chunkDims - holds the size of a single time step (required to
   * permit extending dataset).
   */
  dim = 0;
  chunksize = 1;
  if (timestep >= 0) {
    dims[dim]      = timestep+1;
    maxDims[dim]   = H5S_UNLIMITED;
    chunkDims[dim] = 1;
    ++dim;
  }
  PetscCall(ISGetSize(is, &N));
  PetscCall(ISGetLocalSize(is, &n));
  PetscCall(PetscHDF5IntCast(N/bs,dims + dim));

  maxDims[dim]   = dims[dim];
  chunkDims[dim] = PetscMax(1,dims[dim]);
  chunksize      *= chunkDims[dim];
  ++dim;
  if (bs >= 1) {
    dims[dim]      = bs;
    maxDims[dim]   = dims[dim];
    chunkDims[dim] = dims[dim];
    chunksize      *= chunkDims[dim];
    ++dim;
  }
  /* hdf5 chunks must be less than 4GB */
  if (chunksize > PETSC_HDF5_MAX_CHUNKSIZE/64) {
    if (bs >= 1) {
      if (chunkDims[dim-2] > (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/64))) {
        chunkDims[dim-2] = (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/64));
      } if (chunkDims[dim-1] > (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/64))) {
        chunkDims[dim-1] = (PetscInt)PetscSqrtReal((PetscReal)(PETSC_HDF5_MAX_CHUNKSIZE/64));
      }
    } else {
      chunkDims[dim-1] = PETSC_HDF5_MAX_CHUNKSIZE/64;
    }
  }
  PetscCallHDF5Return(filespace,H5Screate_simple,(dim, dims, maxDims));

#if defined(PETSC_USE_64BIT_INDICES)
  inttype = H5T_NATIVE_LLONG;
#else
  inttype = H5T_NATIVE_INT;
#endif

  /* Create the dataset with default properties and close filespace */
  PetscCall(PetscObjectGetName((PetscObject) is, &isname));
  if (!H5Lexists(group, isname, H5P_DEFAULT)) {
    /* Create chunk */
    PetscCallHDF5Return(chunkspace,H5Pcreate,(H5P_DATASET_CREATE));
    PetscCallHDF5(H5Pset_chunk,(chunkspace, dim, chunkDims));

    PetscCallHDF5Return(dset_id,H5Dcreate2,(group, isname, inttype, filespace, H5P_DEFAULT, chunkspace, H5P_DEFAULT));
    PetscCallHDF5(H5Pclose,(chunkspace));
  } else {
    PetscCallHDF5Return(dset_id,H5Dopen2,(group, isname, H5P_DEFAULT));
    PetscCallHDF5(H5Dset_extent,(dset_id, dims));
  }
  PetscCallHDF5(H5Sclose,(filespace));

  /* Each process defines a dataset and writes it to the hyperslab in the file */
  dim = 0;
  if (timestep >= 0) {
    count[dim] = 1;
    ++dim;
  }
  PetscCall(PetscHDF5IntCast(n/bs,count + dim));
  ++dim;
  if (bs >= 1) {
    count[dim] = bs;
    ++dim;
  }
  if (n > 0 || H5_VERSION_GE(1,10,0)) {
    PetscCallHDF5Return(memspace,H5Screate_simple,(dim, count, NULL));
  } else {
    /* Can't create dataspace with zero for any dimension, so create null dataspace. */
    PetscCallHDF5Return(memspace,H5Screate,(H5S_NULL));
  }

  /* Select hyperslab in the file */
  PetscCall(PetscLayoutGetRange(is->map, &low, NULL));
  dim  = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  PetscCall(PetscHDF5IntCast(low/bs,offset + dim));
  ++dim;
  if (bs >= 1) {
    offset[dim] = 0;
    ++dim;
  }
  if (n > 0 || H5_VERSION_GE(1,10,0)) {
    PetscCallHDF5Return(filespace,H5Dget_space,(dset_id));
    PetscCallHDF5(H5Sselect_hyperslab,(filespace, H5S_SELECT_SET, offset, NULL, count, NULL));
  } else {
    /* Create null filespace to match null memspace. */
    PetscCallHDF5Return(filespace,H5Screate,(H5S_NULL));
  }

  PetscCall(ISGetIndices(is, &ind));
  PetscCallHDF5(H5Dwrite,(dset_id, inttype, memspace, filespace, hdf5->dxpl_id, ind));
  PetscCallHDF5(H5Fflush,(file_id, H5F_SCOPE_GLOBAL));
  PetscCall(ISRestoreIndices(is, &ind));

  /* Close/release resources */
  PetscCallHDF5(H5Gclose,(group));
  PetscCallHDF5(H5Sclose,(filespace));
  PetscCallHDF5(H5Sclose,(memspace));
  PetscCallHDF5(H5Dclose,(dset_id));

  if (timestepping) {
    PetscCall(PetscViewerHDF5WriteObjectAttribute(viewer,(PetscObject)is,"timestepping",PETSC_BOOL,&timestepping));
  }
  PetscCall(PetscInfo(is, "Wrote IS object with name %s\n", isname));
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode ISView_General(IS is,PetscViewer viewer)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       i,n,*idx = sub->idx;
  PetscBool      iascii,isbinary,ishdf5;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5));
  if (iascii) {
    MPI_Comm          comm;
    PetscMPIInt       rank,size;
    PetscViewerFormat fmt;
    PetscBool         isperm;

    PetscCall(PetscObjectGetComm((PetscObject)viewer,&comm));
    PetscCallMPI(MPI_Comm_rank(comm,&rank));
    PetscCallMPI(MPI_Comm_size(comm,&size));

    PetscCall(PetscViewerGetFormat(viewer,&fmt));
    PetscCall(ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,PETSC_FALSE,&isperm));
    if (isperm && fmt != PETSC_VIEWER_ASCII_MATLAB) PetscCall(PetscViewerASCIIPrintf(viewer,"Index set is permutation\n"));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    if (size > 1) {
      if (fmt == PETSC_VIEWER_ASCII_MATLAB) {
        const char* name;

        PetscCall(PetscObjectGetName((PetscObject)is,&name));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%s_%d = [...\n",name,rank));
        for (i=0; i<n; i++) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%" PetscInt_FMT "\n",idx[i]+1));
        }
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"];\n"));
      } else {
        PetscInt  st = 0;

        if (fmt == PETSC_VIEWER_ASCII_INDEX) st = is->map->rstart;
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of indices in set %" PetscInt_FMT "\n",rank,n));
        for (i=0; i<n; i++) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %" PetscInt_FMT " %" PetscInt_FMT "\n",rank,i + st,idx[i]));
        }
      }
    } else {
      if (fmt == PETSC_VIEWER_ASCII_MATLAB) {
        const char* name;

        PetscCall(PetscObjectGetName((PetscObject)is,&name));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%s = [...\n",name));
        for (i=0; i<n; i++) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%" PetscInt_FMT "\n",idx[i]+1));
        }
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"];\n"));
      } else {
        PetscInt  st = 0;

        if (fmt == PETSC_VIEWER_ASCII_INDEX) st = is->map->rstart;
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"Number of indices in set %" PetscInt_FMT "\n",n));
        for (i=0; i<n; i++) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT "\n",i + st,idx[i]));
        }
      }
    }
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  } else if (isbinary) {
    PetscCall(ISView_Binary(is,viewer));
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscCall(ISView_General_HDF5(is,viewer));
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSort_General(IS is)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       n;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(PetscIntSortSemiOrdered(n,sub->idx));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSortRemoveDups_General(IS is)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscLayout    map;
  PetscInt       n;
  PetscBool      sorted;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));
  PetscCall(ISGetInfo(is,IS_SORTED,IS_LOCAL,PETSC_TRUE,&sorted));
  if (sorted) {
    PetscCall(PetscSortedRemoveDupsInt(&n,sub->idx));
  } else {
    PetscCall(PetscSortRemoveDupsInt(&n,sub->idx));
  }
  PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is), n, PETSC_DECIDE, is->map->bs, &map));
  PetscCall(PetscLayoutDestroy(&is->map));
  is->map = map;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSorted_General(IS is,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscCall(ISGetInfo(is,IS_SORTED,IS_LOCAL,PETSC_TRUE,flg));
  PetscFunctionReturn(0);
}

PetscErrorCode  ISToGeneral_General(IS is)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static struct _ISOps myops = { ISGetIndices_General,
                               ISRestoreIndices_General,
                               ISInvertPermutation_General,
                               ISSort_General,
                               ISSortRemoveDups_General,
                               ISSorted_General,
                               ISDuplicate_General,
                               ISDestroy_General,
                               ISView_General,
                               ISLoad_Default,
                               ISCopy_General,
                               ISToGeneral_General,
                               ISOnComm_General,
                               ISSetBlockSize_General,
                               ISContiguousLocal_General,
                               ISLocate_General,
                               /* no specializations of {sorted,unique,perm,interval}{local,global}
                                * because the default checks in ISGetInfo_XXX in index.c are exactly
                                * what we would do for ISGeneral */
                               NULL,
                               NULL,
                               NULL,
                               NULL,
                               NULL,
                               NULL,
                               NULL,
                               NULL};

PETSC_INTERN PetscErrorCode ISSetUp_General(IS);

PetscErrorCode ISSetUp_General(IS is)
{
  IS_General     *sub = (IS_General*)is->data;
  const PetscInt *idx = sub->idx;
  PetscInt       n,i,min,max;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetLocalSize(is->map, &n));

  if (n) {
    min = max = idx[0];
    for (i=1; i<n; i++) {
      if (idx[i] < min) min = idx[i];
      if (idx[i] > max) max = idx[i];
    }
    is->min = min;
    is->max = max;
  } else {
    is->min = PETSC_MAX_INT;
    is->max = PETSC_MIN_INT;
  }
  PetscFunctionReturn(0);
}

/*@
   ISCreateGeneral - Creates a data structure for an index set
   containing a list of integers.

   Collective

   Input Parameters:
+  comm - the MPI communicator
.  n - the length of the index set
.  idx - the list of integers
-  mode - PETSC_COPY_VALUES, PETSC_OWN_POINTER, or PETSC_USE_POINTER; see PetscCopyMode for meaning of this flag.

   Output Parameter:
.  is - the new index set

   Notes:
   When the communicator is not MPI_COMM_SELF, the operations on IS are NOT
   conceptually the same as MPI_Group operations. The IS are then
   distributed sets of indices and thus certain operations on them are
   collective.

   Level: beginner

.seealso: `ISCreateStride()`, `ISCreateBlock()`, `ISAllGather()`, `PETSC_COPY_VALUES`, `PETSC_OWN_POINTER`, `PETSC_USE_POINTER`, `PetscCopyMode`
@*/
PetscErrorCode  ISCreateGeneral(MPI_Comm comm,PetscInt n,const PetscInt idx[],PetscCopyMode mode,IS *is)
{
  PetscFunctionBegin;
  PetscCall(ISCreate(comm,is));
  PetscCall(ISSetType(*is,ISGENERAL));
  PetscCall(ISGeneralSetIndices(*is,n,idx,mode));
  PetscFunctionReturn(0);
}

/*@
   ISGeneralSetIndices - Sets the indices for an ISGENERAL index set

   Collective on IS

   Input Parameters:
+  is - the index set
.  n - the length of the index set
.  idx - the list of integers
-  mode - see PetscCopyMode for meaning of this flag.

   Level: beginner

.seealso: `ISCreateGeneral()`, `ISGeneralSetIndicesFromMask()`, `ISBlockSetIndices()`, `ISGENERAL`, `PetscCopyMode`
@*/
PetscErrorCode  ISGeneralSetIndices(IS is,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (n) PetscValidIntPointer(idx,3);
  PetscCall(ISClearInfoCache(is,PETSC_FALSE));
  PetscUseMethod(is,"ISGeneralSetIndices_C",(IS,PetscInt,const PetscInt[],PetscCopyMode),(is,n,idx,mode));
  PetscFunctionReturn(0);
}

PetscErrorCode  ISGeneralSetIndices_General(IS is,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscLayout    map;
  IS_General     *sub = (IS_General*)is->data;

  PetscFunctionBegin;
  PetscCheck(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) PetscValidIntPointer(idx,3);

  PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is),n,PETSC_DECIDE,is->map->bs,&map));
  PetscCall(PetscLayoutDestroy(&is->map));
  is->map = map;

  if (sub->allocated) PetscCall(PetscFree(sub->idx));
  if (mode == PETSC_COPY_VALUES) {
    PetscCall(PetscMalloc1(n,&sub->idx));
    PetscCall(PetscLogObjectMemory((PetscObject)is,n*sizeof(PetscInt)));
    PetscCall(PetscArraycpy(sub->idx,idx,n));
    sub->allocated = PETSC_TRUE;
  } else if (mode == PETSC_OWN_POINTER) {
    sub->idx = (PetscInt*)idx;
    PetscCall(PetscLogObjectMemory((PetscObject)is,n*sizeof(PetscInt)));
    sub->allocated = PETSC_TRUE;
  } else {
    sub->idx = (PetscInt*)idx;
    sub->allocated = PETSC_FALSE;
  }

  PetscCall(ISSetUp_General(is));
  PetscCall(ISViewFromOptions(is,NULL,"-is_view"));
  PetscFunctionReturn(0);
}

/*@
   ISGeneralSetIndicesFromMask - Sets the indices for an ISGENERAL index set using a boolean mask

   Collective on IS

   Input Parameters:
+  is - the index set
.  rstart - the range start index (inclusive)
.  rend - the range end index (exclusive)
-  mask - the boolean mask array of length rend-rstart, indices will be set for each PETSC_TRUE value in the array

   Notes:
   The mask array may be freed by the user after this call.

   Example:
$  PetscBool mask[] = {PETSC_FALSE, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, PETSC_TRUE};
$  ISGeneralSetIndicesFromMask(is,10,15,mask);
   will feed the IS with indices
$  {11, 14}
   locally.

   Level: beginner

.seealso: `ISCreateGeneral()`, `ISGeneralSetIndices()`, `ISGENERAL`
@*/
PetscErrorCode ISGeneralSetIndicesFromMask(IS is,PetscInt rstart,PetscInt rend,const PetscBool mask[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (rend-rstart) PetscValidBoolPointer(mask,4);
  PetscCall(ISClearInfoCache(is,PETSC_FALSE));
  PetscUseMethod(is,"ISGeneralSetIndicesFromMask_C",(IS,PetscInt,PetscInt,const PetscBool[]),(is,rstart,rend,mask));
  PetscFunctionReturn(0);
}

PetscErrorCode ISGeneralSetIndicesFromMask_General(IS is,PetscInt rstart,PetscInt rend,const PetscBool mask[])
{
  PetscInt        i,nidx;
  PetscInt       *idx;

  PetscFunctionBegin;
  for (i=0,nidx=0; i<rend-rstart; i++) if (mask[i]) nidx++;
  PetscCall(PetscMalloc1(nidx,&idx));
  for (i=0,nidx=0; i<rend-rstart; i++) {
    if (mask[i]) {
      idx[nidx] = i+rstart;
      nidx++;
    }
  }
  PetscCall(ISGeneralSetIndices_General(is,nidx,idx,PETSC_OWN_POINTER));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGeneralFilter_General(IS is, PetscInt start, PetscInt end)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       *idx = sub->idx,*idxnew;
  PetscInt       i,n = is->map->n,nnew = 0,o;

  PetscFunctionBegin;
  for (i=0; i<n; ++i)
    if (idx[i] >= start && idx[i] < end)
      nnew++;
  PetscCall(PetscMalloc1(nnew, &idxnew));
  for (o=0, i=0; i<n; i++) {
    if (idx[i] >= start && idx[i] < end)
      idxnew[o++] = idx[i];
  }
  PetscCall(ISGeneralSetIndices_General(is,nnew,idxnew,PETSC_OWN_POINTER));
  PetscFunctionReturn(0);
}

/*@
   ISGeneralFilter - Remove all indices outside of [start, end)

   Collective on IS

   Input Parameters:
+  is - the index set
.  start - the lowest index kept
-  end - one more than the highest index kept

   Level: beginner

.seealso: `ISCreateGeneral()`, `ISGeneralSetIndices()`
@*/
PetscErrorCode ISGeneralFilter(IS is, PetscInt start, PetscInt end)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscCall(ISClearInfoCache(is,PETSC_FALSE));
  PetscUseMethod(is,"ISGeneralFilter_C",(IS,PetscInt,PetscInt),(is,start,end));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode ISCreate_General(IS is)
{
  IS_General     *sub;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(is,&sub));
  is->data = (void *) sub;
  PetscCall(PetscMemcpy(is->ops,&myops,sizeof(myops)));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISGeneralSetIndices_C",ISGeneralSetIndices_General));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISGeneralSetIndicesFromMask_C",ISGeneralSetIndicesFromMask_General));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISGeneralFilter_C",ISGeneralFilter_General));
  PetscCall(PetscObjectComposeFunction((PetscObject)is,"ISShift_C",ISShift_General));
  PetscFunctionReturn(0);
}
