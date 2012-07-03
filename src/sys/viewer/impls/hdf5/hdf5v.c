#include <petsc-private/viewerimpl.h>    /*I   "petscsys.h"   I*/
#include <hdf5.h>

typedef struct GroupList {
  const char       *name;
  struct GroupList *next;
} GroupList;

typedef struct {
  char         *filename;
  PetscFileMode btype;
  hid_t         file_id;
  PetscInt      timestep;
  GroupList    *groups;
} PetscViewer_HDF5;

#undef __FUNCT__
#define __FUNCT__ "PetscViewerFileClose_HDF5"
static PetscErrorCode PetscViewerFileClose_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*)viewer->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(hdf5->filename);CHKERRQ(ierr);
  if (hdf5->file_id) {
    H5Fclose(hdf5->file_id);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_HDF5" 
PetscErrorCode PetscViewerDestroy_HDF5(PetscViewer viewer)
{
 PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
 PetscErrorCode    ierr;

 PetscFunctionBegin;
 ierr = PetscViewerFileClose_HDF5(viewer);CHKERRQ(ierr);
 if (hdf5->groups) {
   while(hdf5->groups) {
     GroupList *tmp = hdf5->groups->next;

     ierr = PetscFree(hdf5->groups->name);CHKERRQ(ierr);
     ierr = PetscFree(hdf5->groups);CHKERRQ(ierr);
     hdf5->groups = tmp;
   }
 }
 ierr = PetscFree(hdf5);CHKERRQ(ierr);
 PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetMode_HDF5"
PetscErrorCode  PetscViewerFileSetMode_HDF5(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  hdf5->btype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetName_HDF5"
PetscErrorCode  PetscViewerFileSetName_HDF5(PetscViewer viewer, const char name[])
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  MPI_Info          info = MPI_INFO_NULL;
#endif
  hid_t             plist_id;
  herr_t            herr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(name, &hdf5->filename);CHKERRQ(ierr);
  /* Set up file access property list with parallel I/O access */
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  herr = H5Pset_fapl_mpio(plist_id, ((PetscObject)viewer)->comm, info);CHKERRQ(herr);
#endif
  /* Create or open the file collectively */
  switch (hdf5->btype) {
    case FILE_MODE_READ:
      hdf5->file_id = H5Fopen(name, H5F_ACC_RDONLY, plist_id);
      break;
    case FILE_MODE_APPEND:
      hdf5->file_id = H5Fopen(name, H5F_ACC_RDWR, plist_id);
      break;
    case FILE_MODE_WRITE:
      hdf5->file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  }
  if (hdf5->file_id < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "H5Fcreate failed for %s", name);
  viewer->format = PETSC_VIEWER_NOFORMAT;
  H5Pclose(plist_id);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_HDF5" 
PetscErrorCode  PetscViewerCreate_HDF5(PetscViewer v)
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
  hdf5->timestep  = -1;
  hdf5->groups    = PETSC_NULL;

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
PetscErrorCode  PetscViewerHDF5Open(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *hdf5v)
{
  PetscErrorCode ierr;
 
  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm, hdf5v);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*hdf5v, PETSCVIEWERHDF5);CHKERRQ(ierr);
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
PetscErrorCode  PetscViewerHDF5GetFileId(PetscViewer viewer, hid_t *file_id)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
 
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (file_id) *file_id = hdf5->file_id;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5PushGroup"
/*@C
  PetscViewerHDF5PushGroup - Set the current HDF5 group for output

  Not collective

  Input Parameters:
+ viewer - the PetscViewer
- name - The group name

  Level: intermediate

.seealso: PetscViewerHDF5Open(),PetscViewerHDF5PopGroup(),PetscViewerHDF5GetGroup()
@*/
PetscErrorCode  PetscViewerHDF5PushGroup(PetscViewer viewer, const char *name)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
  GroupList        *groupNode;
  PetscErrorCode    ierr;
 
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(name,2);
  ierr = PetscMalloc(sizeof(GroupList), &groupNode);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, (char **) &groupNode->name);CHKERRQ(ierr);
  groupNode->next = hdf5->groups;
  hdf5->groups    = groupNode;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5PopGroup"
/*@C
  PetscViewerHDF5PopGroup - Return the current HDF5 group for output to the previous value

  Not collective

  Input Parameter:
. viewer - the PetscViewer

  Level: intermediate

.seealso: PetscViewerHDF5Open(),PetscViewerHDF5PushGroup(),PetscViewerHDF5GetGroup()
@*/
PetscErrorCode  PetscViewerHDF5PopGroup(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
  GroupList        *groupNode;
  PetscErrorCode    ierr;
 
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (!hdf5->groups) SETERRQ(((PetscObject) viewer)->comm, PETSC_ERR_ARG_WRONGSTATE, "HDF5 group stack is empty, cannot pop");
  groupNode    = hdf5->groups;
  hdf5->groups = hdf5->groups->next;
  ierr = PetscFree(groupNode->name);CHKERRQ(ierr);
  ierr = PetscFree(groupNode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5GetGroup"
/*@C
  PetscViewerHDF5GetGroup - Get the current HDF5 group for output. If none has been assigned, returns PETSC_NULL.

  Not collective

  Input Parameter:
. viewer - the PetscViewer

  Output Parameter:
. name - The group name

  Level: intermediate

.seealso: PetscViewerHDF5Open(),PetscViewerHDF5PushGroup(),PetscViewerHDF5PopGroup()
@*/
PetscErrorCode  PetscViewerHDF5GetGroup(PetscViewer viewer, const char **name)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
 
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(name,2);
  if (hdf5->groups) {
    *name = hdf5->groups->name;
  } else {
    *name = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5IncrementTimestep"
/*@C
  PetscViewerHDF5IncrementTimestep - Increments the current timestep for the HDF5 output. Fields are stacked in time.

  Not collective

  Input Parameter:
. viewer - the PetscViewer

  Level: intermediate

.seealso: PetscViewerHDF5Open(), PetscViewerHDF5SetTimestep(), PetscViewerHDF5GetTimestep()
@*/
PetscErrorCode PetscViewerHDF5IncrementTimestep(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
 
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ++hdf5->timestep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5SetTimestep"
/*@C
  PetscViewerHDF5SetTimestep - Set the current timestep for the HDF5 output. Fields are stacked in time. A timestep
  of -1 disables blocking with timesteps.

  Not collective

  Input Parameters:
+ viewer - the PetscViewer
- timestep - The timestep number

  Level: intermediate

.seealso: PetscViewerHDF5Open(), PetscViewerHDF5IncrementTimestep(), PetscViewerHDF5GetTimestep()
@*/
PetscErrorCode  PetscViewerHDF5SetTimestep(PetscViewer viewer, PetscInt timestep)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
 
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  hdf5->timestep = timestep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscViewerHDF5GetTimestep"
/*@C
  PetscViewerHDF5GetTimestep - Get the current timestep for the HDF5 output. Fields are stacked in time.

  Not collective

  Input Parameter:
. viewer - the PetscViewer

  Output Parameter:
. timestep - The timestep number

  Level: intermediate

.seealso: PetscViewerHDF5Open(), PetscViewerHDF5IncrementTimestep(), PetscViewerHDF5SetTimestep()
@*/
PetscErrorCode  PetscViewerHDF5GetTimestep(PetscViewer viewer, PetscInt *timestep)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;
 
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(timestep,2);
  *timestep = hdf5->timestep;
  PetscFunctionReturn(0);
}

#if defined(oldhdf4stuff)
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerHDF5WriteSDS" 
PetscErrorCode  PetscViewerHDF5WriteSDS(PetscViewer viewer, float *xf, int d, int *dims,int bs)
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
 if (sds_id < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SDcreate failed");
 SDwritedata(sds_id, zero32, 0, dims32, xf);
 SDendaccess(sds_id);
 PetscFunctionReturn(0);
}

#endif
