
/*
     Provides the functions for index sets (IS) defined by a list of integers.
*/
#include <../src/vec/is/is/impls/general/general.h> /*I  "petscis.h"  I*/
#include <petsc/private/viewerimpl.h>
#include <petsc/private/viewerhdf5impl.h>

static PetscErrorCode ISDuplicate_General(IS is,IS *newIS)
{
  PetscErrorCode ierr;
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       n;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) is), n, sub->idx, PETSC_COPY_VALUES, newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISDestroy_General(IS is)
{
  IS_General     *is_general = (IS_General*)is->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is_general->allocated) {ierr = PetscFree(is_general->idx);CHKERRQ(ierr);}
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISGeneralSetIndices_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISGeneralFilter_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(is->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISIdentity_General(IS is, PetscBool *ident)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetInfo(is,IS_IDENTITY,IS_GLOBAL,ident);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISCopy_General(IS is,IS isy)
{
  IS_General     *is_general = (IS_General*)is->data,*isy_general = (IS_General*)isy->data;
  PetscInt       n, N, ny, Ny;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(is->map, &N);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(isy->map, &ny);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(isy->map, &Ny);CHKERRQ(ierr);
  if (n != ny || N != Ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Index sets incompatible");
  ierr = PetscArraycpy(isy_general->idx,is_general->idx,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISOnComm_General(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  PetscErrorCode ierr;
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       n;

  PetscFunctionBegin;
  if (mode == PETSC_OWN_POINTER) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Cannot use PETSC_OWN_POINTER");
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n,sub->idx,mode,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSetBlockSize_General(IS is,PetscInt bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetBlockSize(is->map, bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISContiguousLocal_General(IS is,PetscInt gstart,PetscInt gend,PetscInt *start,PetscBool *contig)
{
  IS_General *sub = (IS_General*)is->data;
  PetscInt   n,i,p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *start  = 0;
  *contig = PETSC_TRUE;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map,&numIdx);CHKERRQ(ierr);
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,&sorted);CHKERRQ(ierr);
  if (sorted) {ierr = PetscFindInt(key,numIdx,sub->idx,location);CHKERRQ(ierr);}
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
  if (*idx != sub->idx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must restore with value from ISGetIndices()");
  PetscFunctionReturn(0);
}

static PetscErrorCode ISInvertPermutation_General(IS is,PetscInt nlocal,IS *isout)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       i,*ii,n,nstart;
  const PetscInt *idx = sub->idx;
  PetscMPIInt    size;
  IS             istmp,nistmp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)is),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscMalloc1(n,&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) ii[idx[i]] = i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,PETSC_OWN_POINTER,isout);CHKERRQ(ierr);
    ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
  } else {
    /* crude, nonscalable get entire IS on each processor */
    ierr = ISAllGather(is,&istmp);CHKERRQ(ierr);
    ierr = ISSetPermutation(istmp);CHKERRQ(ierr);
    ierr = ISInvertPermutation(istmp,PETSC_DECIDE,&nistmp);CHKERRQ(ierr);
    ierr = ISDestroy(&istmp);CHKERRQ(ierr);
    /* get the part we need */
    if (nlocal == PETSC_DECIDE) nlocal = n;
    ierr = MPI_Scan(&nlocal,&nstart,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)is));CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    {
      PetscInt    N;
      PetscMPIInt rank;
      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)is),&rank);CHKERRQ(ierr);
      ierr = PetscLayoutGetSize(is->map, &N);CHKERRQ(ierr);
      if (rank == size-1) {
        if (nstart != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Sum of nlocal lengths %d != total IS length %d",nstart,N);
      }
    }
#endif
    nstart -= nlocal;
    ierr    = ISGetIndices(nistmp,&idx);CHKERRQ(ierr);
    ierr    = ISCreateGeneral(PetscObjectComm((PetscObject)is),nlocal,idx+nstart,PETSC_COPY_VALUES,isout);CHKERRQ(ierr);
    ierr    = ISRestoreIndices(nistmp,&idx);CHKERRQ(ierr);
    ierr    = ISDestroy(&nistmp);CHKERRQ(ierr);
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
  PetscInt        bs, N, n, timestep, low;
  const PetscInt *ind;
  const char     *isname;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = ISGetBlockSize(is,&bs);CHKERRQ(ierr);
  bs   = PetscMax(bs, 1); /* If N = 0, bs  = 0 as well */
  ierr = PetscViewerHDF5OpenGroup(viewer, &file_id, &group);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetTimestep(viewer, &timestep);CHKERRQ(ierr);

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
  if (timestep >= 0) {
    dims[dim]      = timestep+1;
    maxDims[dim]   = H5S_UNLIMITED;
    chunkDims[dim] = 1;
    ++dim;
  }
  ierr = ISGetSize(is, &N);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is, &n);CHKERRQ(ierr);
  ierr = PetscHDF5IntCast(N/bs,dims + dim);CHKERRQ(ierr);

  maxDims[dim]   = dims[dim];
  chunkDims[dim] = PetscMax(1,dims[dim]);
  ++dim;
  if (bs >= 1) {
    dims[dim]      = bs;
    maxDims[dim]   = dims[dim];
    chunkDims[dim] = dims[dim];
    ++dim;
  }
  PetscStackCallHDF5Return(filespace,H5Screate_simple,(dim, dims, maxDims));

#if defined(PETSC_USE_64BIT_INDICES)
  inttype = H5T_NATIVE_LLONG;
#else
  inttype = H5T_NATIVE_INT;
#endif

  /* Create the dataset with default properties and close filespace */
  ierr = PetscObjectGetName((PetscObject) is, &isname);CHKERRQ(ierr);
  if (!H5Lexists(group, isname, H5P_DEFAULT)) {
    /* Create chunk */
    PetscStackCallHDF5Return(chunkspace,H5Pcreate,(H5P_DATASET_CREATE));
    PetscStackCallHDF5(H5Pset_chunk,(chunkspace, dim, chunkDims));

    PetscStackCallHDF5Return(dset_id,H5Dcreate2,(group, isname, inttype, filespace, H5P_DEFAULT, chunkspace, H5P_DEFAULT));
    PetscStackCallHDF5(H5Pclose,(chunkspace));
  } else {
    PetscStackCallHDF5Return(dset_id,H5Dopen2,(group, isname, H5P_DEFAULT));
    PetscStackCallHDF5(H5Dset_extent,(dset_id, dims));
  }
  PetscStackCallHDF5(H5Sclose,(filespace));

  /* Each process defines a dataset and writes it to the hyperslab in the file */
  dim = 0;
  if (timestep >= 0) {
    count[dim] = 1;
    ++dim;
  }
  ierr = PetscHDF5IntCast(n/bs,count + dim);CHKERRQ(ierr);
  ++dim;
  if (bs >= 1) {
    count[dim] = bs;
    ++dim;
  }
  if (n > 0) {
    PetscStackCallHDF5Return(memspace,H5Screate_simple,(dim, count, NULL));
  } else {
    /* Can't create dataspace with zero for any dimension, so create null dataspace. */
    PetscStackCallHDF5Return(memspace,H5Screate,(H5S_NULL));
  }

  /* Select hyperslab in the file */
  ierr = PetscLayoutGetRange(is->map, &low, NULL);CHKERRQ(ierr);
  dim  = 0;
  if (timestep >= 0) {
    offset[dim] = timestep;
    ++dim;
  }
  ierr = PetscHDF5IntCast(low/bs,offset + dim);CHKERRQ(ierr);
  ++dim;
  if (bs >= 1) {
    offset[dim] = 0;
    ++dim;
  }
  if (n > 0) {
    PetscStackCallHDF5Return(filespace,H5Dget_space,(dset_id));
    PetscStackCallHDF5(H5Sselect_hyperslab,(filespace, H5S_SELECT_SET, offset, NULL, count, NULL));
  } else {
    /* Create null filespace to match null memspace. */
    PetscStackCallHDF5Return(filespace,H5Screate,(H5S_NULL));
  }

  ierr = ISGetIndices(is, &ind);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Dwrite,(dset_id, inttype, memspace, filespace, hdf5->dxpl_id, ind));
  PetscStackCallHDF5(H5Fflush,(file_id, H5F_SCOPE_GLOBAL));
  ierr = ISRestoreIndices(is, &ind);CHKERRQ(ierr);

  /* Close/release resources */
  PetscStackCallHDF5(H5Gclose,(group));
  PetscStackCallHDF5(H5Sclose,(filespace));
  PetscStackCallHDF5(H5Sclose,(memspace));
  PetscStackCallHDF5(H5Dclose,(dset_id));
  ierr = PetscInfo1(is, "Wrote IS object with name %s\n", isname);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode ISView_General_Binary(IS is,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      skipHeader,useMPIIO;
  IS_General     *isa = (IS_General*) is->data;
  PetscMPIInt    rank,size,mesgsize,tag = ((PetscObject)viewer)->tag, mesglen;
  PetscInt       n,N,len,j,tr[2];
  int            fdes;
  MPI_Status     status;
  PetscInt       message_count,flowcontrolcount,*values;

  PetscFunctionBegin;
  /* ierr = ISGetLayout(is,&map);CHKERRQ(ierr); */
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetSize(is->map, &N);CHKERRQ(ierr);

  tr[0] = IS_FILE_CLASSID;
  tr[1] = N;

  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);
  if (!skipHeader) {
    ierr  = PetscViewerBinaryWrite(viewer,tr,2,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  }

  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&useMPIIO);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  if (useMPIIO) {
    MPI_File       mfdes;
    MPI_Offset     off;
    PetscMPIInt    lsize;
    PetscInt       rstart;
    const PetscInt *iarray;

    ierr = PetscMPIIntCast(n,&lsize);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,&mfdes);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIOOffset(viewer,&off);CHKERRQ(ierr);
    ierr = PetscLayoutGetRange(is->map,&rstart,NULL);CHKERRQ(ierr);
    off += rstart*(MPI_Offset)sizeof(PetscInt); /* off is MPI_Offset, not PetscMPIInt */
    ierr = MPI_File_set_view(mfdes,off,MPIU_INT,MPIU_INT,(char*)"native",MPI_INFO_NULL);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&iarray);CHKERRQ(ierr);
    ierr = MPIU_File_write_all(mfdes,(void*)iarray,lsize,MPIU_INT,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&iarray);CHKERRQ(ierr);
    ierr = PetscViewerBinaryAddMPIIOOffset(viewer,N*(MPI_Offset)sizeof(PetscInt));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = PetscViewerBinaryGetDescriptor(viewer,&fdes);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)is),&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)is),&size);CHKERRQ(ierr);

  /* determine maximum message to arrive */
  ierr = MPI_Reduce(&n,&len,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)is));CHKERRQ(ierr);

  ierr = PetscViewerFlowControlStart(viewer,&message_count,&flowcontrolcount);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscBinaryWrite(fdes,isa->idx,n,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);

    ierr = PetscMalloc1(len,&values);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(len,&mesgsize);CHKERRQ(ierr);
    /* receive and save messages */
    for (j=1; j<size; j++) {
      ierr = PetscViewerFlowControlStepMaster(viewer,j,&message_count,flowcontrolcount);CHKERRQ(ierr);
      ierr = MPI_Recv(values,mesgsize,MPIU_INT,j,tag,PetscObjectComm((PetscObject)is),&status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status,MPIU_INT,&mesglen);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fdes,values,(PetscInt)mesglen,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlowControlEndMaster(viewer,&message_count);CHKERRQ(ierr);
    ierr = PetscFree(values);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerFlowControlStepWorker(viewer,rank,&message_count);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(n,&mesgsize);CHKERRQ(ierr);
    ierr = MPI_Send(isa->idx,mesgsize,MPIU_INT,0,tag,PetscObjectComm((PetscObject)is));CHKERRQ(ierr);
    ierr = PetscViewerFlowControlEndWorker(viewer,&message_count);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISView_General(IS is,PetscViewer viewer)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscErrorCode ierr;
  PetscInt       i,n,*idx = sub->idx;
  PetscBool      iascii,isbinary,ishdf5;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&ishdf5);CHKERRQ(ierr);
  if (iascii) {
    MPI_Comm          comm;
    PetscMPIInt       rank,size;
    PetscViewerFormat fmt;

    ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

    ierr = PetscViewerGetFormat(viewer,&fmt);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    if (size > 1) {
      if (fmt == PETSC_VIEWER_ASCII_MATLAB) {
        const char* name;

        ierr = PetscObjectGetName((PetscObject)is,&name);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%s_%d = [...\n",name,rank);CHKERRQ(ierr);
        for (i=0; i<n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%D\n",idx[i]+1);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"];\n");CHKERRQ(ierr);
      } else {
        PetscInt  st = 0;
        PetscBool isperm;

        if (fmt == PETSC_VIEWER_ASCII_INDEX) st = is->map->rstart;
        ierr = ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,&isperm);CHKERRQ(ierr);
        if (isperm) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Index set is permutation\n",rank);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of indices in set %D\n",rank,n);CHKERRQ(ierr);
        for (i=0; i<n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D %D\n",rank,i + st,idx[i]);CHKERRQ(ierr);
        }
      }
    } else {
      if (fmt == PETSC_VIEWER_ASCII_MATLAB) {
        const char* name;

        ierr = PetscObjectGetName((PetscObject)is,&name);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%s = [...\n",name);CHKERRQ(ierr);
        for (i=0; i<n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%D\n",idx[i]+1);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"];\n");CHKERRQ(ierr);
      } else {
        PetscInt  st = 0;
        PetscBool isperm;

        if (fmt == PETSC_VIEWER_ASCII_INDEX) st = is->map->rstart;
        ierr = ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,&isperm);CHKERRQ(ierr);
        if (isperm) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Index set is permutation\n");CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Number of indices in set %D\n",n);CHKERRQ(ierr);
        for (i=0; i<n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%D %D\n",i + st,idx[i]);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = ISView_General_Binary(is,viewer);CHKERRQ(ierr);
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = ISView_General_HDF5(is,viewer);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSort_General(IS is)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = PetscSortInt(n,sub->idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSortRemoveDups_General(IS is)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscLayout    map;
  PetscInt       n;
  PetscBool      sorted;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,&sorted);CHKERRQ(ierr);
  if (sorted) {
    ierr = PetscSortedRemoveDupsInt(&n,sub->idx);CHKERRQ(ierr);
  } else {
    ierr = PetscSortRemoveDupsInt(&n,sub->idx);CHKERRQ(ierr);
  }
  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is), n, PETSC_DECIDE, is->map->bs, &map);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&is->map);CHKERRQ(ierr);
  is->map = map;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSorted_General(IS is,PetscBool  *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,flg);CHKERRQ(ierr);
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
                               ISIdentity_General,
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
  PetscErrorCode ierr;
  IS_General     *sub = (IS_General*)is->data;
  const PetscInt *idx = sub->idx;
  PetscInt       n,i,min,max;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(is->map, &n);CHKERRQ(ierr);

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


.seealso: ISCreateStride(), ISCreateBlock(), ISAllGather(), PETSC_COPY_VALUES, PETSC_OWN_POINTER, PETSC_USE_POINTER, PetscCopyMode
@*/
PetscErrorCode  ISCreateGeneral(MPI_Comm comm,PetscInt n,const PetscInt idx[],PetscCopyMode mode,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISCreate(comm,is);CHKERRQ(ierr);
  ierr = ISSetType(*is,ISGENERAL);CHKERRQ(ierr);
  ierr = ISGeneralSetIndices(*is,n,idx,mode);CHKERRQ(ierr);
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


.seealso: ISCreateGeneral(), ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISGeneralSetIndices(IS is,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISClearInfoCache(is,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscUseMethod(is,"ISGeneralSetIndices_C",(IS,PetscInt,const PetscInt[],PetscCopyMode),(is,n,idx,mode));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  ISGeneralSetIndices_General(IS is,PetscInt n,const PetscInt idx[],PetscCopyMode mode)
{
  PetscLayout    map;
  PetscErrorCode ierr;
  IS_General     *sub = (IS_General*)is->data;

  PetscFunctionBegin;
  if (n < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) PetscValidIntPointer(idx,3);

  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is),n,PETSC_DECIDE,is->map->bs,&map);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&is->map);CHKERRQ(ierr);
  is->map = map;

  if (sub->allocated) {ierr = PetscFree(sub->idx);CHKERRQ(ierr);}
  if (mode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc1(n,&sub->idx);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)is,n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscArraycpy(sub->idx,idx,n);CHKERRQ(ierr);
    sub->allocated = PETSC_TRUE;
  } else if (mode == PETSC_OWN_POINTER) {
    sub->idx = (PetscInt*)idx;
    ierr = PetscLogObjectMemory((PetscObject)is,n*sizeof(PetscInt));CHKERRQ(ierr);
    sub->allocated = PETSC_TRUE;
  } else {
    sub->idx = (PetscInt*)idx;
    sub->allocated = PETSC_FALSE;
  }

  ierr = ISSetUp_General(is);CHKERRQ(ierr);
  ierr = ISViewFromOptions(is,NULL,"-is_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGeneralFilter_General(IS is, PetscInt start, PetscInt end)
{
  IS_General     *sub = (IS_General*)is->data;
  PetscInt       *idx = sub->idx,*idxnew;
  PetscInt       i,n = is->map->n,nnew = 0,o;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<n; ++i)
    if (idx[i] >= start && idx[i] < end)
      nnew++;
  ierr = PetscMalloc1(nnew, &idxnew);CHKERRQ(ierr);
  for (o=0, i=0; i<n; i++) {
    if (idx[i] >= start && idx[i] < end)
      idxnew[o++] = idx[i];
  }
  ierr = ISGeneralSetIndices_General(is,nnew,idxnew,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISGeneralFilter - Remove all points outside of [start, end)

   Collective on IS

   Input Parameters:
+  is - the index set
.  start - the lowest index kept
-  end - one more than the highest index kept

   Level: beginner

.seealso: ISCreateGeneral(), ISGeneralSetIndices()
@*/
PetscErrorCode ISGeneralFilter(IS is, PetscInt start, PetscInt end)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  ierr = ISClearInfoCache(is,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscUseMethod(is,"ISGeneralFilter_C",(IS,PetscInt,PetscInt),(is,start,end));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode ISCreate_General(IS is)
{
  PetscErrorCode ierr;
  IS_General     *sub;

  PetscFunctionBegin;
  ierr = PetscNewLog(is,&sub);CHKERRQ(ierr);
  is->data = (void *) sub;
  ierr = PetscMemcpy(is->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISGeneralSetIndices_C",ISGeneralSetIndices_General);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISGeneralFilter_C",ISGeneralFilter_General);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
