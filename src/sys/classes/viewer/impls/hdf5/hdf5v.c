#include <petsc/private/viewerimpl.h>    /*I   "petscsys.h"   I*/
#include <petscviewerhdf5.h>    /*I   "petscviewerhdf5.h"   I*/

typedef struct GroupList {
  const char       *name;
  struct GroupList *next;
} GroupList;

typedef struct {
  char          *filename;
  PetscFileMode btype;
  hid_t         file_id;
  PetscInt      timestep;
  GroupList     *groups;
  PetscBool     basedimension2;  /* save vectors and DMDA vectors with a dimension of at least 2 even if the bs/dof is 1 */
  PetscBool     spoutput;  /* write data in single precision even if PETSc is compiled with double precision PetscReal */
} PetscViewer_HDF5;

struct _n_HDF5ReadCtx {
  hid_t file, group, dataset, dataspace, plist;
  PetscInt timestep;
  PetscBool complexVal, dim2;
};
typedef struct _n_HDF5ReadCtx* HDF5ReadCtx;

static PetscErrorCode PetscViewerSetFromOptions_HDF5(PetscOptionItems *PetscOptionsObject,PetscViewer v)
{
  PetscErrorCode   ierr;
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*)v->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HDF5 PetscViewer Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewer_hdf5_base_dimension2","1d Vectors get 2 dimensions in HDF5","PetscViewerHDF5SetBaseDimension2",hdf5->basedimension2,&hdf5->basedimension2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewer_hdf5_sp_output","Force data to be written in single precision","PetscViewerHDF5SetSPOutput",hdf5->spoutput,&hdf5->spoutput,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileClose_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*)viewer->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFree(hdf5->filename);CHKERRQ(ierr);
  if (hdf5->file_id) PetscStackCallHDF5(H5Fclose,(hdf5->file_id));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerDestroy_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscViewerFileClose_HDF5(viewer);CHKERRQ(ierr);
  while (hdf5->groups) {
    GroupList *tmp = hdf5->groups->next;

    ierr         = PetscFree(hdf5->groups->name);CHKERRQ(ierr);
    ierr         = PetscFree(hdf5->groups);CHKERRQ(ierr);
    hdf5->groups = tmp;
  }
  ierr = PetscFree(hdf5);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetMode_HDF5(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  hdf5->btype = type;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerHDF5SetBaseDimension2_HDF5(PetscViewer viewer, PetscBool flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  hdf5->basedimension2 = flg;
  PetscFunctionReturn(0);
}

/*@
     PetscViewerHDF5SetBaseDimension2 - Vectors of 1 dimension (i.e. bs/dof is 1) will be saved in the HDF5 file with a
       dimension of 2.

    Logically Collective on PetscViewer

  Input Parameters:
+  viewer - the PetscViewer; if it is not hdf5 then this command is ignored
-  flg - if PETSC_TRUE the vector will always have at least a dimension of 2 even if that first dimension is of size 1

  Options Database:
.  -viewer_hdf5_base_dimension2 - turns on (true) or off (false) using a dimension of 2 in the HDF5 file even if the bs/dof of the vector is 1


  Notes:
    Setting this option allegedly makes code that reads the HDF5 in easier since they do not have a "special case" of a bs/dof
         of one when the dimension is lower. Others think the option is crazy.

  Level: intermediate

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()

@*/
PetscErrorCode PetscViewerHDF5SetBaseDimension2(PetscViewer viewer,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscTryMethod(viewer,"PetscViewerHDF5SetBaseDimension2_C",(PetscViewer,PetscBool),(viewer,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     PetscViewerHDF5GetBaseDimension2 - Vectors of 1 dimension (i.e. bs/dof is 1) will be saved in the HDF5 file with a
       dimension of 2.

    Logically Collective on PetscViewer

  Input Parameter:
.  viewer - the PetscViewer, must be of type HDF5

  Output Parameter:
.  flg - if PETSC_TRUE the vector will always have at least a dimension of 2 even if that first dimension is of size 1

  Notes:
    Setting this option allegedly makes code that reads the HDF5 in easier since they do not have a "special case" of a bs/dof
         of one when the dimension is lower. Others think the option is crazy.

  Level: intermediate

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()

@*/
PetscErrorCode PetscViewerHDF5GetBaseDimension2(PetscViewer viewer,PetscBool *flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  *flg = hdf5->basedimension2;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerHDF5SetSPOutput_HDF5(PetscViewer viewer, PetscBool flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  hdf5->spoutput = flg;
  PetscFunctionReturn(0);
}

/*@
     PetscViewerHDF5SetSPOutput - Data is written to disk in single precision even if PETSc is
       compiled with double precision PetscReal.

    Logically Collective on PetscViewer

  Input Parameters:
+  viewer - the PetscViewer; if it is not hdf5 then this command is ignored
-  flg - if PETSC_TRUE the data will be written to disk with single precision

  Options Database:
.  -viewer_hdf5_sp_output - turns on (true) or off (false) output in single precision


  Notes:
    Setting this option does not make any difference if PETSc is compiled with single precision
         in the first place. It does not affect reading datasets (HDF5 handle this internally).

  Level: intermediate

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen(),
          PetscReal

@*/
PetscErrorCode PetscViewerHDF5SetSPOutput(PetscViewer viewer,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscTryMethod(viewer,"PetscViewerHDF5SetSPOutput_C",(PetscViewer,PetscBool),(viewer,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
     PetscViewerHDF5GetSPOutput - Data is written to disk in single precision even if PETSc is
       compiled with double precision PetscReal.

    Logically Collective on PetscViewer

  Input Parameter:
.  viewer - the PetscViewer, must be of type HDF5

  Output Parameter:
.  flg - if PETSC_TRUE the data will be written to disk with single precision

  Notes:
    Setting this option does not make any difference if PETSc is compiled with single precision
         in the first place. It does not affect reading datasets (HDF5 handle this internally).

  Level: intermediate

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen(),
          PetscReal

@*/
PetscErrorCode PetscViewerHDF5GetSPOutput(PetscViewer viewer,PetscBool *flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  *flg = hdf5->spoutput;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerFileSetName_HDF5(PetscViewer viewer, const char name[])
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  MPI_Info          info = MPI_INFO_NULL;
#endif
  hid_t             plist_id;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (hdf5->file_id) PetscStackCallHDF5(H5Fclose,(hdf5->file_id));
  if (hdf5->filename) {ierr = PetscFree(hdf5->filename);CHKERRQ(ierr);}
  ierr = PetscStrallocpy(name, &hdf5->filename);CHKERRQ(ierr);
  /* Set up file access property list with parallel I/O access */
  PetscStackCallHDF5Return(plist_id,H5Pcreate,(H5P_FILE_ACCESS));
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  PetscStackCallHDF5(H5Pset_fapl_mpio,(plist_id, PetscObjectComm((PetscObject)viewer), info));
#endif
  /* Create or open the file collectively */
  switch (hdf5->btype) {
  case FILE_MODE_READ:
    PetscStackCallHDF5Return(hdf5->file_id,H5Fopen,(name, H5F_ACC_RDONLY, plist_id));
    break;
  case FILE_MODE_APPEND:
    PetscStackCallHDF5Return(hdf5->file_id,H5Fopen,(name, H5F_ACC_RDWR, plist_id));
    break;
  case FILE_MODE_WRITE:
    PetscStackCallHDF5Return(hdf5->file_id,H5Fcreate,(name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  }
  if (hdf5->file_id < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "H5Fcreate failed for %s", name);
  PetscStackCallHDF5(H5Pclose,(plist_id));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetName_HDF5(PetscViewer viewer,const char **name)
{
  PetscViewer_HDF5 *vhdf5 = (PetscViewer_HDF5*)viewer->data;

  PetscFunctionBegin;
  *name = vhdf5->filename;
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWERHDF5 - A viewer that writes to an HDF5 file


.seealso:  PetscViewerHDF5Open(), PetscViewerStringSPrintf(), PetscViewerSocketOpen(), PetscViewerDrawOpen(), PETSCVIEWERSOCKET,
           PetscViewerCreate(), PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), PETSCVIEWERBINARY, PETSCVIEWERDRAW, PETSCVIEWERSTRING,
           PetscViewerMatlabOpen(), VecView(), DMView(), PetscViewerMatlabPutArray(), PETSCVIEWERASCII, PETSCVIEWERMATLAB,
           PetscViewerFileSetName(), PetscViewerFileSetMode(), PetscViewerFormat, PetscViewerType, PetscViewerSetType()

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_HDF5(PetscViewer v)
{
  PetscViewer_HDF5 *hdf5;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(v,&hdf5);CHKERRQ(ierr);

  v->data                = (void*) hdf5;
  v->ops->destroy        = PetscViewerDestroy_HDF5;
  v->ops->setfromoptions = PetscViewerSetFromOptions_HDF5;
  v->ops->flush          = 0;
  hdf5->btype            = (PetscFileMode) -1;
  hdf5->filename         = 0;
  hdf5->timestep         = -1;
  hdf5->groups           = NULL;

  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",PetscViewerFileSetName_HDF5);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",PetscViewerFileGetName_HDF5);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_HDF5);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerHDF5SetBaseDimension2_C",PetscViewerHDF5SetBaseDimension2_HDF5);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerHDF5SetSPOutput_C",PetscViewerHDF5SetSPOutput_HDF5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

  Options Database:
.  -viewer_hdf5_base_dimension2 - turns on (true) or off (false) using a dimension of 2 in the HDF5 file even if the bs/dof of the vector is 1
.  -viewer_hdf5_sp_output - forces (if true) the viewer to write data in single precision independent on the precision of PetscReal

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().

   Concepts: HDF5 files
   Concepts: PetscViewerHDF5^creating

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(), PetscViewerHDF5SetBaseDimension2(),
          PetscViewerHDF5SetSPOutput(), PetscViewerHDF5GetBaseDimension2(), VecView(), MatView(), VecLoad(),
          MatLoad(), PetscFileMode, PetscViewer, PetscViewerSetType(), PetscViewerFileSetMode(), PetscViewerFileSetName()
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
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (file_id) *file_id = hdf5->file_id;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerHDF5PushGroup - Set the current HDF5 group for output

  Not collective

  Input Parameters:
+ viewer - the PetscViewer
- name - The group name

  Level: intermediate

.seealso: PetscViewerHDF5Open(),PetscViewerHDF5PopGroup(),PetscViewerHDF5GetGroup(),PetscViewerHDF5OpenGroup()
@*/
PetscErrorCode  PetscViewerHDF5PushGroup(PetscViewer viewer, const char *name)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  GroupList        *groupNode;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(name,2);
  ierr = PetscNew(&groupNode);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, (char**) &groupNode->name);CHKERRQ(ierr);

  groupNode->next = hdf5->groups;
  hdf5->groups    = groupNode;
  PetscFunctionReturn(0);
}

/*@
  PetscViewerHDF5PopGroup - Return the current HDF5 group for output to the previous value

  Not collective

  Input Parameter:
. viewer - the PetscViewer

  Level: intermediate

.seealso: PetscViewerHDF5Open(),PetscViewerHDF5PushGroup(),PetscViewerHDF5GetGroup(),PetscViewerHDF5OpenGroup()
@*/
PetscErrorCode  PetscViewerHDF5PopGroup(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;
  GroupList        *groupNode;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (!hdf5->groups) SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "HDF5 group stack is empty, cannot pop");
  groupNode    = hdf5->groups;
  hdf5->groups = hdf5->groups->next;
  ierr         = PetscFree(groupNode->name);CHKERRQ(ierr);
  ierr         = PetscFree(groupNode);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerHDF5GetGroup - Get the current HDF5 group name (full path), set with PetscViewerHDF5PushGroup()/PetscViewerHDF5PopGroup().
  If none has been assigned, returns NULL.

  Not collective

  Input Parameter:
. viewer - the PetscViewer

  Output Parameter:
. name - The group name

  Level: intermediate

.seealso: PetscViewerHDF5Open(),PetscViewerHDF5PushGroup(),PetscViewerHDF5PopGroup(),PetscViewerHDF5OpenGroup()
@*/
PetscErrorCode  PetscViewerHDF5GetGroup(PetscViewer viewer, const char **name)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(name,2);
  if (hdf5->groups) *name = hdf5->groups->name;
  else *name = NULL;
  PetscFunctionReturn(0);
}

/*@
  PetscViewerHDF5OpenGroup - Open the HDF5 group with the name (full path) returned by PetscViewerHDF5GetGroup(),
  and return this group's ID and file ID.
  If PetscViewerHDF5GetGroup() yields NULL, then group ID is file ID.

  Not collective

  Input Parameter:
. viewer - the PetscViewer

  Output Parameter:
+ fileId - The HDF5 file ID
- groupId - The HDF5 group ID

  Level: intermediate

.seealso: PetscViewerHDF5Open(),PetscViewerHDF5PushGroup(),PetscViewerHDF5PopGroup(),PetscViewerHDF5GetGroup()
@*/
PetscErrorCode PetscViewerHDF5OpenGroup(PetscViewer viewer, hid_t *fileId, hid_t *groupId)
{
  hid_t          file_id, group;
  htri_t         found;
  const char     *groupName = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5GetFileId(viewer, &file_id);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetGroup(viewer, &groupName);CHKERRQ(ierr);
  /* Open group */
  if (groupName) {
    PetscBool root;

    ierr = PetscStrcmp(groupName, "/", &root);CHKERRQ(ierr);
    PetscStackCall("H5Lexists",found = H5Lexists(file_id, groupName, H5P_DEFAULT));
    if (!root && (found <= 0)) {
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
      PetscStackCallHDF5Return(group,H5Gcreate2,(file_id, groupName, 0, H5P_DEFAULT, H5P_DEFAULT));
#else /* deprecated HDF5 1.6 API */
      PetscStackCallHDF5Return(group,H5Gcreate,(file_id, groupName, 0));
#endif
      PetscStackCallHDF5(H5Gclose,(group));
    }
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
    PetscStackCallHDF5Return(group,H5Gopen2,(file_id, groupName, H5P_DEFAULT));
#else
    PetscStackCallHDF5Return(group,H5Gopen,(file_id, groupName));
#endif
  } else group = file_id;

  *fileId  = file_id;
  *groupId = group;
  PetscFunctionReturn(0);
}

/*@
  PetscViewerHDF5IncrementTimestep - Increments the current timestep for the HDF5 output. Fields are stacked in time.

  Not collective

  Input Parameter:
. viewer - the PetscViewer

  Level: intermediate

.seealso: PetscViewerHDF5Open(), PetscViewerHDF5SetTimestep(), PetscViewerHDF5GetTimestep()
@*/
PetscErrorCode PetscViewerHDF5IncrementTimestep(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ++hdf5->timestep;
  PetscFunctionReturn(0);
}

/*@
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
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  hdf5->timestep = timestep;
  PetscFunctionReturn(0);
}

/*@
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
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5*) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(timestep,2);
  *timestep = hdf5->timestep;
  PetscFunctionReturn(0);
}

/*@C
  PetscDataTypeToHDF5DataType - Converts the PETSc name of a datatype to its HDF5 name.

  Not collective

  Input Parameter:
. ptype - the PETSc datatype name (for example PETSC_DOUBLE)

  Output Parameter:
. mtype - the MPI datatype (for example MPI_DOUBLE, ...)

  Level: advanced

.seealso: PetscDataType, PetscHDF5DataTypeToPetscDataType()
@*/
PetscErrorCode PetscDataTypeToHDF5DataType(PetscDataType ptype, hid_t *htype)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT)
#if defined(PETSC_USE_64BIT_INDICES)
                                       *htype = H5T_NATIVE_LLONG;
#else
                                       *htype = H5T_NATIVE_INT;
#endif
  else if (ptype == PETSC_DOUBLE)      *htype = H5T_NATIVE_DOUBLE;
  else if (ptype == PETSC_LONG)        *htype = H5T_NATIVE_LONG;
  else if (ptype == PETSC_SHORT)       *htype = H5T_NATIVE_SHORT;
  else if (ptype == PETSC_ENUM)        *htype = H5T_NATIVE_DOUBLE;
  else if (ptype == PETSC_BOOL)        *htype = H5T_NATIVE_DOUBLE;
  else if (ptype == PETSC_FLOAT)       *htype = H5T_NATIVE_FLOAT;
  else if (ptype == PETSC_CHAR)        *htype = H5T_NATIVE_CHAR;
  else if (ptype == PETSC_BIT_LOGICAL) *htype = H5T_NATIVE_UCHAR;
  else if (ptype == PETSC_STRING)      *htype = H5Tcopy(H5T_C_S1);
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PETSc datatype");
  PetscFunctionReturn(0);
}

/*@C
  PetscHDF5DataTypeToPetscDataType - Finds the PETSc name of a datatype from its HDF5 name

  Not collective

  Input Parameter:
. htype - the HDF5 datatype (for example H5T_NATIVE_DOUBLE, ...)

  Output Parameter:
. ptype - the PETSc datatype name (for example PETSC_DOUBLE)

  Level: advanced

.seealso: PetscDataType, PetscHDF5DataTypeToPetscDataType()
@*/
PetscErrorCode PetscHDF5DataTypeToPetscDataType(hid_t htype, PetscDataType *ptype)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES)
  if      (htype == H5T_NATIVE_INT)    *ptype = PETSC_LONG;
  else if (htype == H5T_NATIVE_LLONG)  *ptype = PETSC_INT;
#else
  if      (htype == H5T_NATIVE_INT)    *ptype = PETSC_INT;
#endif
  else if (htype == H5T_NATIVE_DOUBLE) *ptype = PETSC_DOUBLE;
  else if (htype == H5T_NATIVE_LONG)   *ptype = PETSC_LONG;
  else if (htype == H5T_NATIVE_SHORT)  *ptype = PETSC_SHORT;
  else if (htype == H5T_NATIVE_FLOAT)  *ptype = PETSC_FLOAT;
  else if (htype == H5T_NATIVE_CHAR)   *ptype = PETSC_CHAR;
  else if (htype == H5T_NATIVE_UCHAR)  *ptype = PETSC_CHAR;
  else if (htype == H5T_C_S1)          *ptype = PETSC_STRING;
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported HDF5 datatype");
  PetscFunctionReturn(0);
}

/*@C
 PetscViewerHDF5WriteAttribute - Write a scalar attribute

  Input Parameters:
+ viewer - The HDF5 viewer
. parent - The parent name
. name   - The attribute name
. datatype - The attribute type
- value    - The attribute value

  Level: advanced

.seealso: PetscViewerHDF5Open(), PetscViewerHDF5ReadAttribute(), PetscViewerHDF5HasAttribute()
@*/
PetscErrorCode PetscViewerHDF5WriteAttribute(PetscViewer viewer, const char parent[], const char name[], PetscDataType datatype, const void *value)
{
  hid_t          h5, dataspace, obj, attribute, dtype;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(parent, 2);
  PetscValidPointer(name, 3);
  PetscValidPointer(value, 4);
  ierr = PetscDataTypeToHDF5DataType(datatype, &dtype);CHKERRQ(ierr);
  if (datatype == PETSC_STRING) {
    size_t len;
    ierr = PetscStrlen((const char *) value, &len);CHKERRQ(ierr);
    PetscStackCallHDF5(H5Tset_size,(dtype, len+1));
  }
  ierr = PetscViewerHDF5GetFileId(viewer, &h5);CHKERRQ(ierr);
  PetscStackCallHDF5Return(dataspace,H5Screate,(H5S_SCALAR));
  PetscStackCallHDF5Return(obj,H5Oopen,(h5, parent, H5P_DEFAULT));
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
  PetscStackCallHDF5Return(attribute,H5Acreate2,(obj, name, dtype, dataspace, H5P_DEFAULT, H5P_DEFAULT));
#else
  PetscStackCallHDF5Return(attribute,H5Acreate,(obj, name, dtype, dataspace, H5P_DEFAULT));
#endif
  PetscStackCallHDF5(H5Awrite,(attribute, dtype, value));
  if (datatype == PETSC_STRING) PetscStackCallHDF5(H5Tclose,(dtype));
  PetscStackCallHDF5(H5Aclose,(attribute));
  PetscStackCallHDF5(H5Oclose,(obj));
  PetscStackCallHDF5(H5Sclose,(dataspace));
  PetscFunctionReturn(0);
}

/*@C
 PetscViewerHDF5ReadAttribute - Read a scalar attribute

  Input Parameters:
+ viewer - The HDF5 viewer
. parent - The parent name
. name   - The attribute name
- datatype - The attribute type

  Output Parameter:
. value    - The attribute value

  Level: advanced

.seealso: PetscViewerHDF5Open(), PetscViewerHDF5WriteAttribute(), PetscViewerHDF5HasAttribute()
@*/
PetscErrorCode PetscViewerHDF5ReadAttribute(PetscViewer viewer, const char parent[], const char name[], PetscDataType datatype, void *value)
{
  hid_t          h5, obj, attribute, atype, dtype;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(parent, 2);
  PetscValidPointer(name, 3);
  PetscValidPointer(value, 4);
  ierr = PetscDataTypeToHDF5DataType(datatype, &dtype);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetFileId(viewer, &h5);CHKERRQ(ierr);
  PetscStackCallHDF5Return(obj,H5Oopen,(h5, parent, H5P_DEFAULT));
  PetscStackCallHDF5Return(attribute,H5Aopen_name,(obj, name));
  PetscStackCallHDF5Return(atype,H5Aget_type,(attribute));
  if (datatype == PETSC_STRING) {
    size_t len;

    PetscStackCallHDF5Return(len,H5Tget_size,(atype));
    PetscStackCallHDF5(H5Tclose,(atype));
    ierr = PetscMalloc((len+1) * sizeof(char *), &value);CHKERRQ(ierr);
  }
  PetscStackCallHDF5(H5Aread,(attribute, dtype, value));
  PetscStackCallHDF5(H5Aclose,(attribute));
  PetscStackCallHDF5(H5Dclose,(obj));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5HasObject(PetscViewer viewer, const char name[], H5O_type_t otype, PetscBool *has)
{
  hid_t          h5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(name, 2);
  PetscValidPointer(has, 3);
  *has = PETSC_FALSE;
  ierr = PetscViewerHDF5GetFileId(viewer, &h5);CHKERRQ(ierr);
  if (H5Lexists(h5, name, H5P_DEFAULT)) {
    H5O_info_t info;
    hid_t      obj;

    PetscStackCallHDF5Return(obj,H5Oopen,(h5, name, H5P_DEFAULT));
    PetscStackCallHDF5(H5Oget_info,(obj, &info));
    if (otype == info.type) *has = PETSC_TRUE;
    PetscStackCallHDF5(H5Oclose,(obj));
  }
  PetscFunctionReturn(0);
}

/*@C
 PetscViewerHDF5HasAttribute - Check whether a scalar attribute exists

  Input Parameters:
+ viewer - The HDF5 viewer
. parent - The parent name
- name   - The attribute name

  Output Parameter:
. has    - Flag for attribute existence

  Level: advanced

.seealso: PetscViewerHDF5Open(), PetscViewerHDF5WriteAttribute(), PetscViewerHDF5ReadAttribute()
@*/
PetscErrorCode PetscViewerHDF5HasAttribute(PetscViewer viewer, const char parent[], const char name[], PetscBool *has)
{
  hid_t          h5, dataset;
  htri_t         hhas;
  PetscBool      exists;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(parent, 2);
  PetscValidPointer(name, 3);
  PetscValidPointer(has, 4);
  *has = PETSC_FALSE;
  ierr = PetscViewerHDF5GetFileId(viewer, &h5);CHKERRQ(ierr);
  ierr = PetscViewerHDF5HasObject(viewer, parent, H5O_TYPE_DATASET, &exists);CHKERRQ(ierr);
  if (exists) {
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
    PetscStackCall("H5Dopen2",dataset = H5Dopen2(h5, parent, H5P_DEFAULT));
#else
    PetscStackCall("H5Dopen",dataset = H5Dopen(h5, parent));
#endif
    if (dataset < 0) PetscFunctionReturn(0);
    PetscStackCall("H5Aexists",hhas = H5Aexists(dataset, name));
    if (hhas < 0) {
      PetscStackCallHDF5(H5Dclose,(dataset));
      PetscFunctionReturn(0);
    }
    PetscStackCallHDF5(H5Dclose,(dataset));
    *has = hhas ? PETSC_TRUE : PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadInitialize_Private(PetscViewer viewer, const char name[], HDF5ReadCtx *ctx)
{
  HDF5ReadCtx    h;
  const char    *groupname;
  char           vecgroup[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&h);CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer, &h->file, &h->group);CHKERRQ(ierr);
#if (H5_VERS_MAJOR * 10000 + H5_VERS_MINOR * 100 + H5_VERS_RELEASE >= 10800)
  PetscStackCallHDF5Return(h->dataset,H5Dopen2,(h->group, name, H5P_DEFAULT));
#else
  PetscStackCallHDF5Return(h->dataset,H5Dopen,(h->group, name));
#endif
  PetscStackCallHDF5Return(h->dataspace,H5Dget_space,(h->dataset));
  ierr = PetscViewerHDF5GetTimestep(viewer, &h->timestep);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetGroup(viewer,&groupname);CHKERRQ(ierr);
  ierr = PetscSNPrintf(vecgroup,PETSC_MAX_PATH_LEN,"%s/%s",groupname ? groupname : "",name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5HasAttribute(viewer,vecgroup,"complex",&h->complexVal);CHKERRQ(ierr);
  /* Create property list for collective dataset read */
  PetscStackCallHDF5Return(h->plist,H5Pcreate,(H5P_DATASET_XFER));
#if defined(PETSC_HAVE_H5PSET_FAPL_MPIO)
  PetscStackCallHDF5(H5Pset_dxpl_mpio,(h->plist, H5FD_MPIO_COLLECTIVE));
#endif
  /* To write dataset independently use H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT) */
  *ctx = h;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadFinalize_Private(PetscViewer viewer, HDF5ReadCtx *ctx)
{
  HDF5ReadCtx    h;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  h = *ctx;
  PetscStackCallHDF5(H5Pclose,(h->plist));
  if (h->group != h->file) PetscStackCallHDF5(H5Gclose,(h->group));
  PetscStackCallHDF5(H5Sclose,(h->dataspace));
  PetscStackCallHDF5(H5Dclose,(h->dataset));
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadSizes_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscLayout *map_)
{
  int            rdim, dim;
  hsize_t        dims[4];
  PetscInt       bsInd, lenInd, bs, N;
  PetscLayout    map;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(*map_)) {
    ierr = PetscLayoutCreate(PetscObjectComm((PetscObject)viewer),map_);CHKERRQ(ierr);
  }
  map = *map_;
  /* calculate expected number of dimensions */
  dim = 0;
  if (ctx->timestep >= 0) ++dim;
  ++dim; /* length in blocks */
  if (ctx->complexVal) ++dim;
  /* get actual number of dimensions in dataset */
  PetscStackCallHDF5Return(rdim,H5Sget_simple_extent_dims,(ctx->dataspace, dims, NULL));
  /* calculate expected dimension indices */
  lenInd = 0;
  if (ctx->timestep >= 0) ++lenInd;
  bsInd = lenInd + 1;
  ctx->dim2 = PETSC_FALSE;
  if (rdim == dim) {
    bs = 1; /* support vectors stored as 1D array */
  } else if (rdim == dim+1) {
    bs = (PetscInt) dims[bsInd];
    if (bs == 1) ctx->dim2 = PETSC_TRUE; /* vector with blocksize of 1, still stored as 2D array */
  } else {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Dimension of array in file %d not %d as expected", rdim, dim);
  }
  N = (PetscInt) dims[lenInd]*bs;

  /* Set Vec sizes,blocksize,and type if not already set */
  if (map->bs < 0) {
    ierr = PetscLayoutSetBlockSize(map, bs);CHKERRQ(ierr);
  } else if (map->bs != bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Block size of array in file is %D, not %D as expected",bs,map->bs);
  if (map->N < 0) {
    ierr = PetscLayoutSetSize(map, N);CHKERRQ(ierr);
  } else if (map->N != N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Global size of array in file is %D, not %D as expected",N,map->N);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadSelectHyperslab_Private(PetscViewer viewer, HDF5ReadCtx ctx, PetscLayout map, hid_t *memspace)
{
  hsize_t        count[4], offset[4];
  int            dim;
  PetscInt       bs, n, low;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Compute local size and ownership range */
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(map, &bs);CHKERRQ(ierr);
  ierr = PetscLayoutGetLocalSize(map, &n);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(map, &low, NULL);CHKERRQ(ierr);

  /* Each process defines a dataset and reads it from the hyperslab in the file */
  dim  = 0;
  if (ctx->timestep >= 0) {
    count[dim]  = 1;
    offset[dim] = ctx->timestep;
    ++dim;
  }
  {
    ierr = PetscHDF5IntCast(n/bs, &count[dim]);CHKERRQ(ierr);
    ierr = PetscHDF5IntCast(low/bs, &offset[dim]);CHKERRQ(ierr);
    ++dim;
  }
  if (bs > 1 || ctx->dim2) {
    count[dim]  = bs;
    offset[dim] = 0;
    ++dim;
  }
  if (ctx->complexVal) {
    count[dim]  = 2;
    offset[dim] = 0;
    ++dim;
  }
  PetscStackCallHDF5Return(*memspace,H5Screate_simple,(dim, count, NULL));
  PetscStackCallHDF5(H5Sselect_hyperslab,(ctx->dataspace, H5S_SELECT_SET, offset, NULL, count, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerHDF5ReadArray_Private(PetscViewer viewer, HDF5ReadCtx h, hid_t datatype, hid_t memspace, void *arr)
{
  PetscFunctionBegin;
  PetscStackCallHDF5(H5Dread,(h->dataset, datatype, memspace, h->dataspace, h->plist, arr));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerHDF5Load_Internal(PetscViewer viewer, const char *name, PetscLayout map, hid_t datatype, void **newarr)
{
  HDF5ReadCtx     h;
  hid_t           memspace;
  size_t          unitsize;
  void            *arr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscViewerHDF5ReadInitialize_Private(viewer, name, &h);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSizes_Private(viewer, h, &map);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(map);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSelectHyperslab_Private(viewer, h, map, &memspace);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  if (!h->complexVal) {
    H5T_class_t class = H5Tget_class(datatype);
    if (class == H5T_FLOAT) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"File contains real numbers but PETSc is configured for complex. The conversion is not yet implemented. Configure with --with-scalar-type=real.");
  }
#else
  if (h->complexVal) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"File contains complex numbers but PETSc not configured for them. Configure with --with-scalar-type=complex.");
#endif
  unitsize = H5Tget_size(datatype);
  if (h->complexVal) unitsize *= 2;
  ierr = PetscMalloc(map->n*unitsize, &arr);CHKERRQ(ierr);

  ierr = PetscViewerHDF5ReadArray_Private(viewer, h, datatype, memspace, arr);CHKERRQ(ierr);
  PetscStackCallHDF5(H5Sclose,(memspace));
  ierr = PetscViewerHDF5ReadFinalize_Private(viewer, &h);CHKERRQ(ierr);
  *newarr = arr;
  PetscFunctionReturn(0);
}

/*@C
 PetscViewerHDF5ReadSizes - Read block size and global size of a vector (Vec or IS) stored in an HDF5 file.

  Input Parameters:
+ viewer - The HDF5 viewer
- name   - The vector name

  Output Parameter:
+ bs     - block size
- N      - global size

  Note:
  A vector is stored as an HDF5 dataspace with 1-4 dimensions in this order:
  1) # timesteps (optional), 2) # blocks, 3) # elements per block (optional), 4) real and imaginary part (only for complex).

  A vectors can be stored as a 2D dataspace even if its blocksize is 1; see PetscViewerHDF5SetBaseDimension2().

  Level: advanced

.seealso: PetscViewerHDF5Open(), VecLoad(), ISLoad(), VecGetSize(), ISGetSize(), PetscViewerHDF5SetBaseDimension2()
@*/
PetscErrorCode PetscViewerHDF5ReadSizes(PetscViewer viewer, const char name[], PetscInt *bs, PetscInt *N)
{
  HDF5ReadCtx    h;
  PetscLayout    map=NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscViewerHDF5ReadInitialize_Private(viewer, name, &h);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadSizes_Private(viewer, h, &map);CHKERRQ(ierr);
  ierr = PetscViewerHDF5ReadFinalize_Private(viewer, &h);CHKERRQ(ierr);
  if (bs) *bs = map->bs;
  if (N) *N = map->N;
  ierr = PetscLayoutDestroy(&map);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  The variable Petsc_Viewer_HDF5_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
PetscMPIInt Petsc_Viewer_HDF5_keyval = MPI_KEYVAL_INVALID;

/*@C
  PETSC_VIEWER_HDF5_ - Creates an HDF5 PetscViewer shared by all processors in a communicator.

  Collective on MPI_Comm

  Input Parameter:
. comm - the MPI communicator to share the HDF5 PetscViewer

  Level: intermediate

  Options Database Keys:
. -viewer_hdf5_filename <name>

  Environmental variables:
. PETSC_VIEWER_HDF5_FILENAME

  Notes:
  Unlike almost all other PETSc routines, PETSC_VIEWER_HDF5_ does not return
  an error code.  The HDF5 PetscViewer is usually used in the form
$       XXXView(XXX object, PETSC_VIEWER_HDF5_(comm));

.seealso: PetscViewerHDF5Open(), PetscViewerCreate(), PetscViewerDestroy()
@*/
PetscViewer PETSC_VIEWER_HDF5_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscViewer    viewer;
  char           fname[PETSC_MAX_PATH_LEN];
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&ncomm,NULL);if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_HDF5_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  if (Petsc_Viewer_HDF5_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Viewer_HDF5_keyval,0);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_HDF5_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  }
  ierr = MPI_Comm_get_attr(ncomm,Petsc_Viewer_HDF5_keyval,(void**)&viewer,(int*)&flg);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_HDF5_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscOptionsGetenv(ncomm,"PETSC_VIEWER_HDF5_FILENAME",fname,PETSC_MAX_PATH_LEN,&flg);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_HDF5_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
    if (!flg) {
      ierr = PetscStrcpy(fname,"output.h5");
      if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_HDF5_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
    }
    ierr = PetscViewerHDF5Open(ncomm,fname,FILE_MODE_WRITE,&viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_HDF5_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_HDF5_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
    ierr = MPI_Comm_set_attr(ncomm,Petsc_Viewer_HDF5_keyval,(void*)viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_HDF5_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  }
  ierr = PetscCommDestroy(&ncomm);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_HDF5_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  PetscFunctionReturn(viewer);
}
