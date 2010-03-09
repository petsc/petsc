#define PETSCVEC_DLL
/* 
   This file contains simple binary input routines for vectors.  The
   analogous output routines are within each vector implementation's 
   VecView (with viewer types PETSC_VIEWER_BINARY)
 */

#include "petscsys.h"
#include "petscvec.h"         /*I  "petscvec.h"  I*/
#include "private/vecimpl.h"
#if defined(PETSC_HAVE_PNETCDF)
EXTERN_C_BEGIN
#include "pnetcdf.h"
EXTERN_C_END
#endif
EXTERN PetscErrorCode VecLoad_Binary(PetscViewer, const VecType, Vec*);
EXTERN PetscErrorCode VecLoad_Netcdf(PetscViewer, Vec*);
EXTERN PetscErrorCode VecLoad_HDF5(PetscViewer, Vec*);
EXTERN PetscErrorCode VecLoadIntoVector_Binary(PetscViewer, Vec);
EXTERN PetscErrorCode VecLoadIntoVector_Netcdf(PetscViewer, Vec);

#undef __FUNCT__  
#define __FUNCT__ "VecLoad"
/*@C 
  VecLoad - Loads a vector that has been stored in binary format
  with VecView().

  Collective on PetscViewer 

  Input Parameters:
+ viewer - binary file viewer, obtained from PetscViewerBinaryOpen() or
           NetCDF file viewer, obtained from PetscViewerNetcdfOpen()
- outtype - the type of vector VECSEQ or VECMPI or PETSC_NULL (which indicates
            using VECSEQ if the communicator in the Viewer is of size 1; otherwise
            use VECMPI).

  Output Parameter:
. newvec - the newly loaded vector

   Level: intermediate

  Notes:
  The input file must contain the full global vector, as
  written by the routine VecView().

  Notes for advanced users:
  Most users should not need to know the details of the binary storage
  format, since VecLoad() and VecView() completely hide these details.
  But for anyone who's interested, the standard binary matrix storage
  format is
.vb
     int    VEC_FILE_COOKIE
     int    number of rows
     PetscScalar *values of all entries
.ve

   In addition, PETSc automatically does the byte swapping for
machines that store the bytes reversed, e.g.  DEC alpha, freebsd,
linux, Windows and the paragon; thus if you write your own binary
read/write routines you have to swap the bytes; see PetscBinaryRead()
and PetscBinaryWrite() to see how this may be done.

  Concepts: vector^loading from file

.seealso: PetscViewerBinaryOpen(), VecView(), MatLoad(), VecLoadIntoVector() 
@*/  
PetscErrorCode PETSCVEC_DLLEXPORT VecLoad(PetscViewer viewer, const VecType outtype,Vec *newvec)
{
  PetscErrorCode ierr;
  PetscTruth     isbinary,flg;
  char           vtype[256];
  const char    *prefix;
#if defined(PETSC_HAVE_PNETCDF)
  PetscTruth     isnetcdf;
#endif
#if defined(PETSC_HAVE_HDF5)
  PetscTruth     ishdf5;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(newvec,3);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_HDF5,&ishdf5);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PNETCDF)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_NETCDF,&isnetcdf);CHKERRQ(ierr);
#endif

#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = VecInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscLogEventBegin(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PNETCDF)
  if (isnetcdf) {
    ierr = VecLoad_Netcdf(viewer,newvec);CHKERRQ(ierr);
  } else
#endif
#if defined(PETSC_HAVE_HDF5)
  if (ishdf5) {
    SETERRQ(PETSC_ERR_SUP,"Since HDF5 format gives ASCII name for each object in file; must use VecLoadIntoVector() after setting name of Vec with PetscObjectSetName()");
  } else 
#endif
  {
    Vec            factory;
    MPI_Comm       comm;
    PetscErrorCode (*r)(PetscViewer, const VecType,Vec*);
    PetscMPIInt    size;

    ierr = PetscObjectGetOptionsPrefix((PetscObject)viewer,(const char**)&prefix);CHKERRQ(ierr);
    ierr = PetscOptionsGetString(prefix,"-vec_type",vtype,256,&flg);CHKERRQ(ierr);
    if (flg) {
      outtype = vtype;
    }
    ierr = PetscOptionsGetString(prefix,"-vecload_type",vtype,256,&flg);CHKERRQ(ierr);
    if (flg) {
      outtype = vtype;
    }
    ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);  
    if (!outtype) {
      ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
      outtype = (size > 1) ? VECMPI : VECSEQ;
    }

    ierr = VecCreate(comm,&factory);CHKERRQ(ierr);
    ierr = VecSetSizes(factory,1,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetType(factory,outtype);CHKERRQ(ierr);
    r = factory->ops->load;
    ierr = VecDestroy(factory);
    if (!r) SETERRQ1(PETSC_ERR_SUP,"VecLoad is not supported for type: %s",outtype);
    ierr = (*r)(viewer,outtype,newvec);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PNETCDF)
#undef __FUNCT__  
#define __FUNCT__ "VecLoad_Netcdf"
PetscErrorCode VecLoad_Netcdf(PetscViewer viewer,Vec *newvec)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       N,n,bs;
  PetscInt       ncid,start;
  Vec            vec;
  PetscScalar    *avec;
  MPI_Comm       comm;
  PetscTruth     flag;
  char           name[NC_MAX_NAME];

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscViewerNetcdfGetID(viewer,&ncid);CHKERRQ(ierr);
  ierr = ncmpi_inq_dim(ncid,0,name,(MPI_Offset*)&N);CHKERRQ(ierr); /* N gets the global vector size */
  ierr = VecCreate(comm,&vec);CHKERRQ(ierr);
  ierr = VecSetSizes(vec,PETSC_DECIDE,N);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscOptionsGetInt(PETSC_NULL,"-vecload_block_size",&bs,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);
    }
  }
  ierr = VecSetFromOptions(vec);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(vec,&start,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
  ierr = ncmpi_get_vara_double_all(ncid,0,(const MPI_Offset*)&start,(const MPI_Offset*)&n,(double *)avec);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  *newvec = vec;
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#include "petscmat.h" /* so that MAT_FILE_COOKIE is defined */

#undef __FUNCT__  
#define __FUNCT__ "VecLoad_Binary"
PetscErrorCode VecLoad_Binary(PetscViewer viewer, const VecType itype,Vec *newvec)
{
  PetscMPIInt    size,rank,tag;
  int            fd;
  PetscInt       i,rows,type,n,*range,bs,tr[2];
  PetscErrorCode ierr;
  Vec            vec;
  PetscScalar    *avec,*avecwork;
  MPI_Comm       comm;
  MPI_Request    request;
  MPI_Status     status;
  PetscTruth     flag;
#if defined(PETSC_HAVE_MPIIO)
  PetscTruth     useMPIIO;
#endif

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* Read vector header. */
  ierr = PetscViewerBinaryRead(viewer,tr,2,PETSC_INT);CHKERRQ(ierr);
  type = tr[0];
  rows = tr[1];
  if (type != VEC_FILE_COOKIE) {
      ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
      if (type == MAT_FILE_COOKIE) {
        SETERRQ(PETSC_ERR_ARG_WRONG,"Matrix is next in file, not a vector as you requested");
      } else {
        SETERRQ(PETSC_ERR_ARG_WRONG,"Not a vector next in file");
      }
  }
  ierr = VecCreate(comm,&vec);CHKERRQ(ierr);
  ierr = VecSetSizes(vec,PETSC_DECIDE,rows);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-vecload_block_size",&bs,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(vec);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr); 
  ierr = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinaryGetMPIIO(viewer,&useMPIIO);CHKERRQ(ierr);
  if (!useMPIIO) {
#endif
    if (!rank) {
      ierr = PetscBinaryRead(fd,avec,n,PETSC_SCALAR);CHKERRQ(ierr);

      if (size > 1) {
	/* read in other chuncks and send to other processors */
	/* determine maximum chunck owned by other */
	range = vec->map->range;
	n = 1;
	for (i=1; i<size; i++) {
	  n = PetscMax(n,range[i+1] - range[i]);
	}
	ierr = PetscMalloc(n*sizeof(PetscScalar),&avecwork);CHKERRQ(ierr);
	for (i=1; i<size; i++) {
	  n    = range[i+1] - range[i];
	  ierr = PetscBinaryRead(fd,avecwork,n,PETSC_SCALAR);CHKERRQ(ierr);
	  ierr = MPI_Isend(avecwork,n,MPIU_SCALAR,i,tag,comm,&request);CHKERRQ(ierr);
	  ierr = MPI_Wait(&request,&status);CHKERRQ(ierr);
	}
	ierr = PetscFree(avecwork);CHKERRQ(ierr);
      }
    } else {
      ierr = MPI_Recv(avec,n,MPIU_SCALAR,0,tag,comm,&status);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_MPIIO)
  } else {
    PetscMPIInt  gsizes[1],lsizes[1],lstarts[1];
    MPI_Datatype view;
    MPI_File     mfdes;
    MPI_Aint     ub,ul;
    MPI_Offset   off;

    gsizes[0]  = PetscMPIIntCast(rows);
    lsizes[0]  = PetscMPIIntCast(n);
    lstarts[0] = PetscMPIIntCast(vec->map->rstart);CHKERRQ(ierr);
    ierr = MPI_Type_create_subarray(1,gsizes,lsizes,lstarts,MPI_ORDER_FORTRAN,MPIU_SCALAR,&view);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&view);CHKERRQ(ierr);

    ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,&mfdes);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIOOffset(viewer,&off);CHKERRQ(ierr);
    ierr = MPI_File_set_view(mfdes,off,MPIU_SCALAR,view,(char *)"native",MPI_INFO_NULL);CHKERRQ(ierr);
    ierr = MPIU_File_read_all(mfdes,avec,lsizes[0],MPIU_SCALAR,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Type_get_extent(view,&ul,&ub);CHKERRQ(ierr);
    ierr = PetscViewerBinaryAddMPIIOOffset(viewer,ub);CHKERRQ(ierr);
    ierr = MPI_Type_free(&view);CHKERRQ(ierr);
  }
#endif
  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  *newvec = vec;
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Load,viewer,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PNETCDF)
#undef __FUNCT__  
#define __FUNCT__ "VecLoadIntoVector_Netcdf"
PetscErrorCode VecLoadIntoVector_Netcdf(PetscViewer viewer,Vec vec)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       N,rows,n,bs;
  PetscInt       ncid,start;
  PetscScalar    *avec;
  MPI_Comm       comm;
  PetscTruth     flag;
  char           name[NC_MAX_NAME];

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(VEC_Load,viewer,vec,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscViewerNetcdfGetID(viewer,&ncid);CHKERRQ(ierr);
  ierr = ncmpi_inq_dim(ncid,0,name,(MPI_Offset*)&N);CHKERRQ(ierr); /* N gets the global vector size */
  if (!rank) {
    ierr = VecGetSize(vec,&rows);CHKERRQ(ierr);
    if (N != rows) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Vector in file different length then input vector");
    ierr = PetscOptionsGetInt(PETSC_NULL,"-vecload_block_size",&bs,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = VecSetBlockSize(vec,bs);CHKERRQ(ierr);
    }
  }
  ierr = VecSetFromOptions(vec);CHKERRQ(ierr);
  ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(vec,&start,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
  ierr = ncmpi_get_vara_double_all(ncid,0,(const MPI_Offset*)&start,(const MPI_Offset*)&n,(double *)avec);CHKERRQ(ierr);
  ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Load,viewer,vec,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#if defined(PETSC_HAVE_HDF5)
#undef __FUNCT__  
#define __FUNCT__ "VecLoadIntoVector_HDF5"
PetscErrorCode VecLoadIntoVector_HDF5(PetscViewer viewer, Vec xin)
{
  hsize_t        rdim,dim = 1; /* Could have dim 2 for blocked vectors */
  PetscInt       n, N, bs, low;
  PetscScalar   *x;
  PetscTruth     flag;
  hid_t          file_id, dset_id, filespace, memspace, plist_id;
  hsize_t        dims[2];
  hsize_t        count[2];
  hsize_t        offset[2];
  herr_t         status;
  PetscErrorCode ierr;
  const char     *vecname;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(VEC_Load,viewer,xin,0,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-vecload_block_size", &bs, &flag);CHKERRQ(ierr);
  if (flag) {
    ierr = VecSetBlockSize(xin, bs);CHKERRQ(ierr);
  }
  ierr = VecSetFromOptions(xin);CHKERRQ(ierr);

  ierr = PetscViewerHDF5GetFileId(viewer, &file_id);CHKERRQ(ierr);

  /* Create the dataset with default properties and close filespace */
  ierr = PetscObjectGetName((PetscObject)xin,&vecname);CHKERRQ(ierr);
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
  dset_id = H5Dopen2(file_id, vecname, H5P_DEFAULT);
#else
  dset_id = H5Dopen(file_id, vecname);
#endif
  if (dset_id == -1) SETERRQ1(PETSC_ERR_LIB,"Could not H5Dopen() with Vec named %s",vecname);

  /* Retrieve the dataspace for the dataset */
  ierr = VecGetSize(xin, &N);CHKERRQ(ierr);
  filespace = H5Dget_space(dset_id);
  if (filespace == -1) SETERRQ(PETSC_ERR_LIB,"Could not H5Dget_space()");
  rdim = H5Sget_simple_extent_dims(filespace, dims, PETSC_NULL);
#if defined(PETSC_USE_COMPLEX)
  if (rdim != 2) SETERRQ1(PETSC_ERR_FILE_UNEXPECTED, "Dimension of array in file %d not 2 (complex numbers) as expected",rdim);
#else
  if (rdim != 1) SETERRQ1(PETSC_ERR_FILE_UNEXPECTED, "Dimension of array in file %d not 1 as expected",rdim);
#endif
  if (N != (int) dims[0]) SETERRQ2(PETSC_ERR_FILE_UNEXPECTED, "Vector in file different length (%d) then input vector (%d)", (int) dims[0], N);

  /* Each process defines a dataset and reads it from the hyperslab in the file */
  ierr = VecGetLocalSize(xin, &n);CHKERRQ(ierr);
  count[0] = n;
#if defined(PETSC_USE_COMPLEX)
  count[1] = 2;
  dim++;
#endif
  memspace = H5Screate_simple(dim, count, NULL);
  if (memspace == -1) SETERRQ(PETSC_ERR_LIB,"Could not H5Screate_simple()");

  /* Select hyperslab in the file */
  ierr = VecGetOwnershipRange(xin, &low, PETSC_NULL);CHKERRQ(ierr);
  offset[0] = low;
#if defined(PETSC_USE_COMPLEX)
  offset[1] = 0;
#endif
  status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);CHKERRQ(status);

  /* Create property list for collective dataset read */
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  if (plist_id == -1) SETERRQ(PETSC_ERR_LIB,"Could not H5Pcreate()");
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);CHKERRQ(status);
  /* To write dataset independently use H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT) */
#endif

  ierr = VecGetArray(xin, &x);CHKERRQ(ierr);
  status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, plist_id, x);CHKERRQ(status);
  ierr = VecRestoreArray(xin, &x);CHKERRQ(ierr);

  /* Close/release resources */
  status = H5Pclose(plist_id);CHKERRQ(status);
  status = H5Sclose(filespace);CHKERRQ(status);
  status = H5Sclose(memspace);CHKERRQ(status);
  status = H5Dclose(dset_id);CHKERRQ(status);

  ierr = VecAssemblyBegin(xin);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xin);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Load,viewer,xin,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "VecLoadIntoVector_Binary"
PetscErrorCode VecLoadIntoVector_Binary(PetscViewer viewer,Vec vec)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag;
  PetscInt       i,rows,type,n,*range;
  int            fd;
  PetscScalar    *avec;
  MPI_Comm       comm;
  MPI_Request    request;
  MPI_Status     status;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(VEC_Load,viewer,vec,0,0);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (!rank) {
    /* Read vector header. */
    ierr = PetscBinaryRead(fd,&type,1,PETSC_INT);CHKERRQ(ierr);
    if (type != VEC_FILE_COOKIE) SETERRQ(PETSC_ERR_ARG_WRONG,"Non-vector object");
    ierr = PetscBinaryRead(fd,&rows,1,PETSC_INT);CHKERRQ(ierr);
    ierr = VecGetSize(vec,&n);CHKERRQ(ierr);
    if (n != rows) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Vector in file different length then input vector");
    ierr = MPI_Bcast(&rows,1,MPIU_INT,0,comm);CHKERRQ(ierr);

    ierr = VecSetFromOptions(vec);CHKERRQ(ierr);
    ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr);
    ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,avec,n,PETSC_SCALAR);CHKERRQ(ierr);
    ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);

    if (size > 1) {
      /* read in other chuncks and send to other processors */
      /* determine maximum chunck owned by other */
      range = vec->map->range;
      n = 1;
      for (i=1; i<size; i++) {
        n = PetscMax(n,range[i+1] - range[i]);
      }
      ierr = PetscMalloc(n*sizeof(PetscScalar),&avec);CHKERRQ(ierr);
      ierr = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
      for (i=1; i<size; i++) {
        n    = range[i+1] - range[i];
        ierr = PetscBinaryRead(fd,avec,n,PETSC_SCALAR);CHKERRQ(ierr);
        ierr = MPI_Isend(avec,n,MPIU_SCALAR,i,tag,comm,&request);CHKERRQ(ierr);
        ierr = MPI_Wait(&request,&status);CHKERRQ(ierr);
      }
      ierr = PetscFree(avec);CHKERRQ(ierr);
    }
  } else {
    ierr = MPI_Bcast(&rows,1,MPIU_INT,0,comm);CHKERRQ(ierr);
    ierr = VecSetFromOptions(vec);CHKERRQ(ierr);
    ierr = VecGetLocalSize(vec,&n);CHKERRQ(ierr); 
    ierr = PetscObjectGetNewTag((PetscObject)viewer,&tag);CHKERRQ(ierr);
    ierr = VecGetArray(vec,&avec);CHKERRQ(ierr);
    ierr = MPI_Recv(avec,n,MPIU_SCALAR,0,tag,comm,&status);CHKERRQ(ierr);
    ierr = VecRestoreArray(vec,&avec);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(vec);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vec);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(VEC_Load,viewer,vec,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecLoadIntoVector_Default"
PetscErrorCode VecLoadIntoVector_Default(PetscViewer viewer,Vec vec)
{
  PetscTruth     isbinary;
#if defined(PETSC_HAVE_PNETCDF)
  PetscTruth     isnetcdf;
#endif
#if defined(PETSC_HAVE_HDF5)
  PetscTruth     ishdf5;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PNETCDF)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_NETCDF,&isnetcdf);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_HDF5,&ishdf5);CHKERRQ(ierr);
#endif

  if (isbinary) {
    ierr = VecLoadIntoVector_Binary(viewer,vec);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PNETCDF)
  } else if (isnetcdf) {
    ierr = VecLoadIntoVector_Netcdf(viewer,vec);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_HDF5)
  } else if (ishdf5) {
    ierr = VecLoadIntoVector_HDF5(viewer,vec);CHKERRQ(ierr);
#endif
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for vector loading", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}
