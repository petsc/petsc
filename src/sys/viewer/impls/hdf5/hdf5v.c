#include "private/viewerimpl.h"    /*I   "petscsys.h"   I*/
#include <hdf5.h>

typedef struct {
  char         *filename;
  PetscFileMode btype;
  hid_t         file_id;
} PetscViewer_HDF5;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_HDF5" 
PetscErrorCode PetscViewerDestroy_HDF5(PetscViewer viewer)
{
 PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
 PetscErrorCode    ierr;

 PetscFunctionBegin;
 ierr = PetscFree(hdf5->filename);CHKERRQ(ierr);
 if (hdf5->file_id) {
   H5Fclose(hdf5->file_id);
 }
 ierr = PetscFree(hdf5);CHKERRQ(ierr);
 PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetMode_HDF5"
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetMode_HDF5(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 1);
  hdf5->btype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetName_HDF5"
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetName_HDF5(PetscViewer viewer, const char name[])
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  MPI_Info          info = MPI_INFO_NULL;
#endif
  hid_t             plist_id;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(name, &hdf5->filename);CHKERRQ(ierr);
  /* Set up file access property list with parallel I/O access */
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  H5Pset_fapl_mpio(plist_id, ((PetscObject)viewer)->comm, info);
#endif
  /* Create or open the file collectively */
  switch (hdf5->btype) {
    case FILE_MODE_READ:
      hdf5->file_id = H5Fopen(name, H5F_ACC_RDONLY, plist_id);
      break;
    case FILE_MODE_WRITE:
      hdf5->file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
      break;
    default:
      SETERRQ(PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  }
  if (hdf5->file_id < 0) SETERRQ1(PETSC_ERR_LIB, "H5Fcreate failed for %s", name);
  viewer->format = PETSC_VIEWER_NOFORMAT;
  H5Pclose(plist_id);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_HDF5" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerCreate_HDF5(PetscViewer v)
{  
  PetscViewer_HDF5 *hdf5;
  PetscErrorCode    ierr;
 
  PetscFunctionBegin;
  ierr = PetscNewLog(v, PetscViewer_HDF5, &hdf5);CHKERRQ(ierr);
  v->data         = (void *) hdf5;
  v->ops->destroy = PetscViewerDestroy_HDF5;
  v->ops->flush   = 0;
  v->iformat      = 0;
  hdf5->btype     = (PetscFileMode) -1; 
  hdf5->filename  = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerFileSetName_C","PetscViewerFileSetName_HDF5",
                                           PetscViewerFileSetName_HDF5);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerFileSetMode_C","PetscViewerFileSetMode_HDF5",
                                           PetscViewerFileSetMode_HDF5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF5Open" 
/*@C
   PetscViewerHDF5Open - Opens a file for HDF5 input/output.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  name - name of file 
-  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

   Output Parameter:
.  hdf5v - PetscViewer for HDF5 input/output to use with the specified file

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().

   Concepts: HDF5 files
   Concepts: PetscViewerHDF5^creating

.seealso: PetscViewerASCIIOpen(), PetscViewerSetFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(),
          PetscFileMode, PetscViewer
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerHDF5Open(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *hdf5v)
{
  PetscErrorCode ierr;
 
  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm, hdf5v);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*hdf5v, PETSC_VIEWER_HDF5);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*hdf5v, type);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*hdf5v, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF5GetFileId" 
/*@C
  PetscViewerHDF5GetFileId - Retrieve the file id, this file ID then can be used in direct HDF5 calls

  Not collective

  Input Parameter:
. viewer - the PetscViewer

  Output Parameter:
. file_id - The file id

  Level: intermediate

.seealso: PetscViewerHDF5Open()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerHDF5GetFileId(PetscViewer viewer, hid_t *file_id)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
 
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  if (file_id) *file_id = hdf5->file_id;
  PetscFunctionReturn(0);
}


#if defined(oldhdf4stuff)
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF5WriteSDS" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerHDF5WriteSDS(PetscViewer viewer, float *xf, int d, int *dims,int bs)
{
 int                   i;
 PetscViewer_HDF5      *vhdf5 = (PetscViewer_HDF5*)viewer->data;
 int32                 sds_id,zero32[3],dims32[3];

 PetscFunctionBegin;

 for (i = 0; i < d; i++) {
   zero32[i] = 0;
   dims32[i] = dims[i];
 }
 sds_id = SDcreate(vhdf5->sd_id, "Vec", DFNT_FLOAT32, d, dims32);
 if (sds_id < 0) {
   SETERRQ(PETSC_ERR_LIB,"SDcreate failed");
 }
 SDwritedata(sds_id, zero32, 0, dims32, xf);
 SDendaccess(sds_id);
 PetscFunctionReturn(0);
}

#endif
