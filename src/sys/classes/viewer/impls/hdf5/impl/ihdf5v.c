#include <petsc/private/viewerhdf5impl.h> /*I "petscviewerhdf5.h" I*/

static PetscErrorCode PetscViewerHDF5Traverse_Inner_Internal(hid_t h5, const char name[], PetscBool createGroup, PetscBool *exists_)
{
  htri_t exists;
  hid_t  group;

  PetscFunctionBegin;
  PetscCallHDF5Return(exists, H5Lexists, (h5, name, H5P_DEFAULT));
  if (exists) PetscCallHDF5Return(exists, H5Oexists_by_name, (h5, name, H5P_DEFAULT));
  if (!exists && createGroup) {
    hid_t plist_id;
    PetscCallHDF5Return(plist_id, H5Pcreate, (H5P_GROUP_CREATE));
    PetscCallHDF5(H5Pset_link_creation_order, (plist_id, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED));
    PetscCallHDF5Return(group, H5Gcreate2, (h5, name, H5P_DEFAULT, plist_id, H5P_DEFAULT));
    PetscCallHDF5(H5Pclose, (plist_id));
    PetscCallHDF5(H5Gclose, (group));
    exists = PETSC_TRUE;
  }
  *exists_ = (PetscBool)exists;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5Traverse_Internal(PetscViewer viewer, const char name[], PetscBool createGroup, PetscBool *has, H5O_type_t *otype)
{
  const char rootGroupName[] = "/";
  hid_t      h5;
  PetscBool  exists = PETSC_FALSE;
  PetscInt   i;
  int        n;
  char     **hierarchy;
  char       buf[PETSC_MAX_PATH_LEN] = "";

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (name) PetscAssertPointer(name, 2);
  else name = rootGroupName;
  if (has) {
    PetscAssertPointer(has, 4);
    *has = PETSC_FALSE;
  }
  if (otype) {
    PetscAssertPointer(otype, 5);
    *otype = H5O_TYPE_UNKNOWN;
  }
  PetscCall(PetscViewerHDF5GetFileId(viewer, &h5));

  /*
     Unfortunately, H5Oexists_by_name() fails if any object in hierarchy is missing.
     Hence, each of them needs to be tested separately:
     1) whether it's a valid link
     2) whether this link resolves to an object
     See H5Oexists_by_name() documentation.
  */
  PetscCall(PetscStrToArray(name, '/', &n, &hierarchy));
  if (!n) {
    /*  Assume group "/" always exists in accordance with HDF5 >= 1.10.0. See H5Lexists() documentation. */
    if (has) *has = PETSC_TRUE;
    if (otype) *otype = H5O_TYPE_GROUP;
    PetscCall(PetscStrToArrayDestroy(n, hierarchy));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (i = 0; i < n; i++) {
    PetscCall(PetscStrlcat(buf, "/", sizeof(buf)));
    PetscCall(PetscStrlcat(buf, hierarchy[i], sizeof(buf)));
    PetscCall(PetscViewerHDF5Traverse_Inner_Internal(h5, buf, createGroup, &exists));
    if (!exists) break;
  }
  PetscCall(PetscStrToArrayDestroy(n, hierarchy));

  /* If the object exists, get its type */
  if (exists && otype) {
    H5O_info_t info;

    /* We could use H5Iget_type() here but that would require opening the object. This way we only need its name. */
    PetscCallHDF5(H5Oget_info_by_name, (h5, name, &info, H5P_DEFAULT));
    *otype = info.type;
  }
  if (has) *has = exists;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5GetGroup_Internal(PetscViewer viewer, const char *name[])
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(name, 2);
  if (hdf5->groups) *name = hdf5->groups->name;
  else *name = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5HasAttribute_Internal(PetscViewer viewer, const char parent[], const char name[], PetscBool *has)
{
  hid_t  h5;
  htri_t hhas;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetFileId(viewer, &h5));
  PetscCallHDF5Return(hhas, H5Aexists_by_name, (h5, parent, name, H5P_DEFAULT));
  *has = hhas ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5GetGroup_HDF5(PetscViewer viewer, const char path[], const char *abspath[])
{
  size_t      len;
  PetscBool   relative = PETSC_FALSE;
  const char *group;
  char        buf[PETSC_MAX_PATH_LEN] = "";

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetGroup_Internal(viewer, &group));
  PetscCall(PetscStrlen(path, &len));
  relative = (PetscBool)(!len || path[0] != '/');
  if (relative) {
    PetscCall(PetscStrncpy(buf, group, sizeof(buf)));
    if (!group || len) PetscCall(PetscStrlcat(buf, "/", sizeof(buf)));
    PetscCall(PetscStrlcat(buf, path, sizeof(buf)));
    PetscCall(PetscStrallocpy(buf, (char **)abspath));
  } else {
    PetscCall(PetscStrallocpy(path, (char **)abspath));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerSetFromOptions_HDF5(PetscViewer v, PetscOptionItems PetscOptionsObject)
{
  PetscBool         flg  = PETSC_FALSE, set;
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)v->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "HDF5 PetscViewer Options");
  PetscCall(PetscOptionsBool("-viewer_hdf5_base_dimension2", "1d Vectors get 2 dimensions in HDF5", "PetscViewerHDF5SetBaseDimension2", hdf5->basedimension2, &hdf5->basedimension2, NULL));
  PetscCall(PetscOptionsBool("-viewer_hdf5_sp_output", "Force data to be written in single precision", "PetscViewerHDF5SetSPOutput", hdf5->spoutput, &hdf5->spoutput, NULL));
  PetscCall(PetscOptionsBool("-viewer_hdf5_collective", "Enable collective transfer mode", "PetscViewerHDF5SetCollective", flg, &flg, &set));
  if (set) PetscCall(PetscViewerHDF5SetCollective(v, flg));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-viewer_hdf5_default_timestepping", "Set default timestepping state", "PetscViewerHDF5SetDefaultTimestepping", flg, &flg, &set));
  if (set) PetscCall(PetscViewerHDF5SetDefaultTimestepping(v, flg));
  PetscCall(PetscOptionsBool("-viewer_hdf5_compress", "Enable compression", "PetscViewerHDF5SetCompress", hdf5->compress, &hdf5->compress, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerView_HDF5(PetscViewer v, PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)v->data;
  PetscBool         flg;

  PetscFunctionBegin;
  if (hdf5->filename) PetscCall(PetscViewerASCIIPrintf(viewer, "Filename: %s\n", hdf5->filename));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Vectors with blocksize 1 saved as 2D datasets: %s\n", PetscBools[hdf5->basedimension2]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Enforce single precision storage: %s\n", PetscBools[hdf5->spoutput]));
  PetscCall(PetscViewerHDF5GetCollective(v, &flg));
  PetscCall(PetscViewerASCIIPrintf(viewer, "MPI-IO transfer mode: %s\n", flg ? "collective" : "independent"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Default timestepping: %s\n", PetscBools[hdf5->defTimestepping]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Compression: %s\n", PetscBools[hdf5->compress]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileClose_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(hdf5->filename));
  if (hdf5->file_id) PetscCallHDF5(H5Fclose, (hdf5->file_id));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFlush_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  if (hdf5->file_id) PetscCallHDF5(H5Fflush, (hdf5->file_id, H5F_SCOPE_LOCAL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerDestroy_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscCallHDF5(H5Pclose, (hdf5->dxpl_id));
  PetscCall(PetscViewerFileClose_HDF5(viewer));
  while (hdf5->groups) {
    PetscViewerHDF5GroupList *tmp = hdf5->groups->next;

    PetscCall(PetscFree(hdf5->groups->name));
    PetscCall(PetscFree(hdf5->groups));
    hdf5->groups = tmp;
  }
  PetscCall(PetscFree(hdf5));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5SetBaseDimension2_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5GetBaseDimension2_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5SetSPOutput_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5SetCollective_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5GetCollective_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5GetDefaultTimestepping_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5SetDefaultTimestepping_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5SetCompress_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5GetCompress_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5WriteGroup_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5PushGroup_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5PopGroup_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5GetGroup_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5PushTimestepping_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5PopTimestepping_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5IsTimestepping_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5IncrementTimestep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5SetTimestep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5GetTimestep_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5WriteAttribute_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5ReadAttribute_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5HasGroup_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5HasDataset_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5HasAttribute_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerHDF5GetSPOutput_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetMode_HDF5(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  hdf5->btype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetMode_HDF5(PetscViewer viewer, PetscFileMode *type)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  *type = hdf5->btype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5SetBaseDimension2_HDF5(PetscViewer viewer, PetscBool flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  hdf5->basedimension2 = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5GetBaseDimension2_HDF5(PetscViewer viewer, PetscBool *flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  *flg = hdf5->basedimension2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5SetSPOutput_HDF5(PetscViewer viewer, PetscBool flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  hdf5->spoutput = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5GetSPOutput_HDF5(PetscViewer viewer, PetscBool *flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  *flg = hdf5->spoutput;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5SetCollective_HDF5(PetscViewer viewer, PetscBool flg)
{
  PetscFunctionBegin;
  /* H5FD_MPIO_COLLECTIVE is wrong in hdf5 1.10.2, and is the same as H5FD_MPIO_INDEPENDENT in earlier versions
     - see e.g. https://gitlab.cosma.dur.ac.uk/swift/swiftsim/issues/431 */
#if H5_VERSION_GE(1, 10, 3) && defined(H5_HAVE_PARALLEL)
  {
    PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;
    PetscCallHDF5(H5Pset_dxpl_mpio, (hdf5->dxpl_id, flg ? H5FD_MPIO_COLLECTIVE : H5FD_MPIO_INDEPENDENT));
  }
#else
  if (flg) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)viewer), "Warning: PetscViewerHDF5SetCollective(viewer,PETSC_TRUE) is ignored for HDF5 versions prior to 1.10.3 or if built without MPI support\n"));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5GetCollective_HDF5(PetscViewer viewer, PetscBool *flg)
{
#if defined(H5_HAVE_PARALLEL)
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;
  H5FD_mpio_xfer_t  mode;
#endif

  PetscFunctionBegin;
#if !defined(H5_HAVE_PARALLEL)
  *flg = PETSC_FALSE;
#else
  PetscCallHDF5(H5Pget_dxpl_mpio, (hdf5->dxpl_id, &mode));
  *flg = (mode == H5FD_MPIO_COLLECTIVE) ? PETSC_TRUE : PETSC_FALSE;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetName_HDF5(PetscViewer viewer, const char name[])
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;
  hid_t             plist_access_id;
  hid_t             plist_create_id;

  PetscFunctionBegin;
  if (hdf5->file_id) PetscCallHDF5(H5Fclose, (hdf5->file_id));
  if (hdf5->filename) PetscCall(PetscFree(hdf5->filename));
  PetscCall(PetscStrallocpy(name, &hdf5->filename));
  PetscCallHDF5Return(plist_create_id, H5Pcreate, (H5P_FILE_CREATE));
  PetscCallHDF5(H5Pset_link_creation_order, (plist_create_id, H5P_CRT_ORDER_TRACKED | H5P_CRT_ORDER_INDEXED));
  PetscCallHDF5Return(plist_access_id, H5Pcreate, (H5P_FILE_ACCESS));
#if defined(H5_HAVE_PARALLEL)
  PetscCallHDF5(H5Pset_fapl_mpio, (plist_access_id, PetscObjectComm((PetscObject)viewer), MPI_INFO_NULL));
#endif
  /* Create or open the file collectively */
  switch (hdf5->btype) {
  case FILE_MODE_READ:
    if (PetscDefined(USE_DEBUG)) {
      PetscMPIInt rank;
      PetscBool   flg;

      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
      if (rank == 0) {
        PetscCall(PetscTestFile(hdf5->filename, 'r', &flg));
        PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "File %s requested for reading does not exist", hdf5->filename);
      }
      PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)viewer)));
    }
    PetscCallHDF5Return(hdf5->file_id, H5Fopen, (name, H5F_ACC_RDONLY, plist_access_id));
    break;
  case FILE_MODE_APPEND:
  case FILE_MODE_UPDATE: {
    PetscBool flg;
    PetscCall(PetscTestFile(hdf5->filename, 'r', &flg));
    if (flg) PetscCallHDF5Return(hdf5->file_id, H5Fopen, (name, H5F_ACC_RDWR, plist_access_id));
    else PetscCallHDF5Return(hdf5->file_id, H5Fcreate, (name, H5F_ACC_EXCL, plist_create_id, plist_access_id));
    break;
  }
  case FILE_MODE_WRITE:
    PetscCallHDF5Return(hdf5->file_id, H5Fcreate, (name, H5F_ACC_TRUNC, plist_create_id, plist_access_id));
    break;
  case FILE_MODE_UNDEFINED:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  default:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Unsupported file mode %s", PetscFileModes[hdf5->btype]);
  }
  PetscCheck(hdf5->file_id >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "H5Fcreate failed for %s", name);
  PetscCallHDF5(H5Pclose, (plist_access_id));
  PetscCallHDF5(H5Pclose, (plist_create_id));
  PetscCall(PetscViewerHDF5ResetAttachedDMPlexStorageVersion(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetName_HDF5(PetscViewer viewer, const char **name)
{
  PetscViewer_HDF5 *vhdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  *name = vhdf5->filename;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerSetUp_HDF5(PETSC_UNUSED PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5SetDefaultTimestepping_HDF5(PetscViewer viewer, PetscBool flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  hdf5->defTimestepping = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5GetDefaultTimestepping_HDF5(PetscViewer viewer, PetscBool *flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  *flg = hdf5->defTimestepping;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5SetCompress_HDF5(PetscViewer viewer, PetscBool flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  hdf5->compress = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5GetCompress_HDF5(PetscViewer viewer, PetscBool *flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  *flg = hdf5->compress;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5GetFileId - Retrieve the file id, this file ID then can be used in direct HDF5 calls

  Not Collective

  Input Parameter:
. viewer - the `PetscViewer`

  Output Parameter:
. file_id - The file id

  Level: intermediate

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`
@*/
PetscErrorCode PetscViewerHDF5GetFileId(PetscViewer viewer, hid_t *file_id)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (file_id) *file_id = hdf5->file_id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5PushGroup_HDF5(PetscViewer viewer, const char name[])
{
  PetscViewer_HDF5         *hdf5 = (PetscViewer_HDF5 *)viewer->data;
  PetscViewerHDF5GroupList *groupNode;
  size_t                    i, len;
  char                      buf[PETSC_MAX_PATH_LEN];
  const char               *gname;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &len));
  gname = NULL;
  if (len) {
    if (len == 1 && name[0] == '.') {
      /* use current name */
      gname = (hdf5->groups && hdf5->groups->name) ? hdf5->groups->name : NULL;
    } else if (name[0] == '/') {
      /* absolute */
      for (i = 1; i < len; i++) {
        if (name[i] != '/') {
          gname = name;
          break;
        }
      }
    } else {
      /* relative */
      const char *parent = (hdf5->groups && hdf5->groups->name) ? hdf5->groups->name : "";
      PetscCall(PetscSNPrintf(buf, sizeof(buf), "%s/%s", parent, name));
      gname = buf;
    }
  }
  PetscCall(PetscNew(&groupNode));
  PetscCall(PetscStrallocpy(gname, (char **)&groupNode->name));
  groupNode->next = hdf5->groups;
  hdf5->groups    = groupNode;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5PopGroup_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5         *hdf5 = (PetscViewer_HDF5 *)viewer->data;
  PetscViewerHDF5GroupList *groupNode;

  PetscFunctionBegin;
  PetscCheck(hdf5->groups, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "HDF5 group stack is empty, cannot pop");
  groupNode    = hdf5->groups;
  hdf5->groups = hdf5->groups->next;
  PetscCall(PetscFree(groupNode->name));
  PetscCall(PetscFree(groupNode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5OpenGroup - Open the HDF5 group with the name (full path) returned by `PetscViewerHDF5GetGroup()`,
  and return this group's ID and file ID.
  If `PetscViewerHDF5GetGroup()` yields NULL, then group ID is file ID.

  Not Collective

  Input Parameters:
+ viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`
- path   - (Optional) The path relative to the pushed group

  Output Parameters:
+ fileId  - The HDF5 file ID
- groupId - The HDF5 group ID

  Level: intermediate

  Note:
  If path starts with '/', it is taken as an absolute path overriding currently pushed group, else path is relative to the current pushed group.
  `NULL` or empty path means the current pushed group.

  If the viewer is writable, the group is created if it doesn't exist yet.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`, `PetscViewerHDF5WriteGroup()`
@*/
PetscErrorCode PetscViewerHDF5OpenGroup(PetscViewer viewer, const char path[], hid_t *fileId, hid_t *groupId)
{
  hid_t       file_id;
  H5O_type_t  type;
  const char *fileName  = NULL;
  const char *groupName = NULL;
  PetscBool   writable, has;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (path) PetscAssertPointer(path, 2);
  PetscAssertPointer(fileId, 3);
  PetscAssertPointer(groupId, 4);
  PetscCall(PetscViewerWritable(viewer, &writable));
  PetscCall(PetscViewerHDF5GetFileId(viewer, &file_id));
  PetscCall(PetscViewerFileGetName(viewer, &fileName));
  PetscCall(PetscViewerHDF5GetGroup(viewer, path, &groupName));
  PetscCall(PetscViewerHDF5Traverse_Internal(viewer, groupName, writable, &has, &type));
  if (!has) {
    PetscCheck(writable, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Group %s does not exist and file %s is not open for writing", groupName, fileName);
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_LIB, "HDF5 failed to create group %s although file %s is open for writing", groupName, fileName);
  }
  PetscCheck(type == H5O_TYPE_GROUP, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Path %s in file %s resolves to something which is not a group", groupName, fileName);
  PetscCallHDF5Return(*groupId, H5Gopen2, (file_id, groupName, H5P_DEFAULT));
  PetscCall(PetscFree(groupName));
  *fileId = file_id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5WriteGroup_HDF5(PetscViewer viewer, const char path[])
{
  hid_t fileId, groupId;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5OpenGroup(viewer, path, &fileId, &groupId)); // make sure group is actually created
  PetscCallHDF5(H5Gclose, (groupId));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5PushTimestepping_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscCheck(!hdf5->timestepping, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Timestepping is already pushed");
  hdf5->timestepping = PETSC_TRUE;
  if (hdf5->timestep < 0) hdf5->timestep = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5PopTimestepping_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscCheck(hdf5->timestepping, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Timestepping has not been pushed yet. Call PetscViewerHDF5PushTimestepping() first");
  hdf5->timestepping = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5IsTimestepping_HDF5(PetscViewer viewer, PetscBool *flg)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  *flg = hdf5->timestepping;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5IncrementTimestep_HDF5(PetscViewer viewer)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscCheck(hdf5->timestepping, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Timestepping has not been pushed yet. Call PetscViewerHDF5PushTimestepping() first");
  ++hdf5->timestep;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5SetTimestep_HDF5(PetscViewer viewer, PetscInt timestep)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscCheck(timestep >= 0, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Timestep %" PetscInt_FMT " is negative", timestep);
  PetscCheck(hdf5->timestepping, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Timestepping has not been pushed yet. Call PetscViewerHDF5PushTimestepping() first");
  hdf5->timestep = timestep;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5GetTimestep_HDF5(PetscViewer viewer, PetscInt *timestep)
{
  PetscViewer_HDF5 *hdf5 = (PetscViewer_HDF5 *)viewer->data;

  PetscFunctionBegin;
  PetscCheck(hdf5->timestepping, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Timestepping has not been pushed yet. Call PetscViewerHDF5PushTimestepping() first");
  *timestep = hdf5->timestep;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDataTypeToHDF5DataType - Converts the PETSc name of a datatype to its HDF5 name.

  Not Collective

  Input Parameter:
. ptype - the PETSc datatype name (for example `PETSC_DOUBLE`)

  Output Parameter:
. htype - the HDF5 datatype

  Level: advanced

.seealso: [](sec_viewers), `PetscDataType`, `PetscHDF5DataTypeToPetscDataType()`
@*/
PetscErrorCode PetscDataTypeToHDF5DataType(PetscDataType ptype, hid_t *htype)
{
  PetscFunctionBegin;
  if (ptype == PETSC_INT) *htype = PetscDefined(USE_64BIT_INDICES) ? H5T_NATIVE_LLONG : H5T_NATIVE_INT;
  else if (ptype == PETSC_DOUBLE) *htype = H5T_NATIVE_DOUBLE;
  else if (ptype == PETSC_LONG) *htype = H5T_NATIVE_LONG;
  else if (ptype == PETSC_SHORT) *htype = H5T_NATIVE_SHORT;
  else if (ptype == PETSC_ENUM) *htype = H5T_NATIVE_INT;
  else if (ptype == PETSC_BOOL) *htype = H5T_NATIVE_HBOOL;
  else if (ptype == PETSC_FLOAT) *htype = H5T_NATIVE_FLOAT;
  else if (ptype == PETSC_CHAR) *htype = H5T_NATIVE_CHAR;
  else if (ptype == PETSC_BIT_LOGICAL) *htype = H5T_NATIVE_UCHAR;
  else if (ptype == PETSC_STRING) *htype = H5Tcopy(H5T_C_S1);
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PETSc datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscHDF5DataTypeToPetscDataType - Finds the PETSc name of a datatype from its HDF5 name

  Not Collective

  Input Parameter:
. htype - the HDF5 datatype (for example `H5T_NATIVE_DOUBLE`, ...)

  Output Parameter:
. ptype - the PETSc datatype name (for example `PETSC_DOUBLE`)

  Level: advanced

.seealso: [](sec_viewers), `PetscDataType`
@*/
PetscErrorCode PetscHDF5DataTypeToPetscDataType(hid_t htype, PetscDataType *ptype)
{
  PetscFunctionBegin;
  if (htype == H5T_NATIVE_INT) *ptype = PetscDefined(USE_64BIT_INDICES) ? PETSC_LONG : PETSC_INT;
#if defined(PETSC_USE_64BIT_INDICES)
  else if (htype == H5T_NATIVE_LLONG) *ptype = PETSC_INT;
#endif
  else if (htype == H5T_NATIVE_DOUBLE) *ptype = PETSC_DOUBLE;
  else if (htype == H5T_NATIVE_LONG) *ptype = PETSC_LONG;
  else if (htype == H5T_NATIVE_SHORT) *ptype = PETSC_SHORT;
  else if (htype == H5T_NATIVE_HBOOL) *ptype = PETSC_BOOL;
  else if (htype == H5T_NATIVE_FLOAT) *ptype = PETSC_FLOAT;
  else if (htype == H5T_NATIVE_CHAR) *ptype = PETSC_CHAR;
  else if (htype == H5T_NATIVE_UCHAR) *ptype = PETSC_CHAR;
  else if (htype == H5T_C_S1) *ptype = PETSC_STRING;
  else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported HDF5 datatype");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5WriteAttribute_HDF5(PetscViewer viewer, const char parent[], const char name[], PetscDataType datatype, const void *value)
{
  const char *parentAbsPath;
  hid_t       h5, dataspace, obj, attribute, dtype;
  PetscBool   has;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetGroup(viewer, parent, &parentAbsPath));
  PetscCall(PetscViewerHDF5Traverse_Internal(viewer, parentAbsPath, PETSC_TRUE, NULL, NULL));
  PetscCall(PetscViewerHDF5HasAttribute_Internal(viewer, parentAbsPath, name, &has));
  PetscCall(PetscDataTypeToHDF5DataType(datatype, &dtype));
  if (datatype == PETSC_STRING) {
    size_t len;
    PetscCall(PetscStrlen((const char *)value, &len));
    PetscCallHDF5(H5Tset_size, (dtype, len + 1));
  }
  PetscCall(PetscViewerHDF5GetFileId(viewer, &h5));
  PetscCallHDF5Return(dataspace, H5Screate, (H5S_SCALAR));
  PetscCallHDF5Return(obj, H5Oopen, (h5, parentAbsPath, H5P_DEFAULT));
  if (has) {
    PetscCallHDF5Return(attribute, H5Aopen_name, (obj, name));
  } else {
    PetscCallHDF5Return(attribute, H5Acreate2, (obj, name, dtype, dataspace, H5P_DEFAULT, H5P_DEFAULT));
  }
  PetscCallHDF5(H5Awrite, (attribute, dtype, value));
  if (datatype == PETSC_STRING) PetscCallHDF5(H5Tclose, (dtype));
  PetscCallHDF5(H5Aclose, (attribute));
  PetscCallHDF5(H5Oclose, (obj));
  PetscCallHDF5(H5Sclose, (dataspace));
  PetscCall(PetscFree(parentAbsPath));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5ReadAttribute_HDF5(PetscViewer viewer, const char parent[], const char name[], PetscDataType datatype, const void *defaultValue, void *value)
{
  const char *parentAbsPath;
  hid_t       h5, obj, attribute, dtype;
  PetscBool   has;

  PetscFunctionBegin;
  PetscCall(PetscDataTypeToHDF5DataType(datatype, &dtype));
  PetscCall(PetscViewerHDF5GetGroup(viewer, parent, &parentAbsPath));
  PetscCall(PetscViewerHDF5Traverse_Internal(viewer, parentAbsPath, PETSC_FALSE, &has, NULL));
  if (has) PetscCall(PetscViewerHDF5HasAttribute_Internal(viewer, parentAbsPath, name, &has));
  if (!has) {
    if (defaultValue) {
      if (defaultValue != value) {
        if (datatype == PETSC_STRING) {
          PetscCall(PetscStrallocpy(*(char **)defaultValue, (char **)value));
        } else {
          size_t len;
          PetscCallHDF5ReturnNoCheck(len, H5Tget_size, (dtype));
          PetscCall(PetscMemcpy(value, defaultValue, len));
        }
      }
      PetscCall(PetscFree(parentAbsPath));
      PetscFunctionReturn(PETSC_SUCCESS);
    } else SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Attribute %s/%s does not exist and default value not provided", parentAbsPath, name);
  }
  PetscCall(PetscViewerHDF5GetFileId(viewer, &h5));
  PetscCallHDF5Return(obj, H5Oopen, (h5, parentAbsPath, H5P_DEFAULT));
  PetscCallHDF5Return(attribute, H5Aopen_name, (obj, name));
  if (datatype == PETSC_STRING) {
    size_t len;
    hid_t  atype;
    PetscCallHDF5Return(atype, H5Aget_type, (attribute));
    PetscCallHDF5ReturnNoCheck(len, H5Tget_size, (atype));
    PetscCall(PetscMalloc((len + 1) * sizeof(char), value));
    PetscCallHDF5(H5Tset_size, (dtype, len + 1));
    PetscCallHDF5(H5Aread, (attribute, dtype, *(char **)value));
  } else {
    PetscCallHDF5(H5Aread, (attribute, dtype, value));
  }
  PetscCallHDF5(H5Aclose, (attribute));
  /* H5Oclose can be used to close groups, datasets, or committed datatypes */
  PetscCallHDF5(H5Oclose, (obj));
  PetscCall(PetscFree(parentAbsPath));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5HasGroup_HDF5(PetscViewer viewer, const char path[], PetscBool *has)
{
  H5O_type_t  type;
  const char *abspath;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetGroup(viewer, path, &abspath));
  PetscCall(PetscViewerHDF5Traverse_Internal(viewer, abspath, PETSC_FALSE, NULL, &type));
  *has = (PetscBool)(type == H5O_TYPE_GROUP);
  PetscCall(PetscFree(abspath));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5HasDataset_HDF5(PetscViewer viewer, const char path[], PetscBool *has)
{
  H5O_type_t  type;
  const char *abspath;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetGroup(viewer, path, &abspath));
  PetscCall(PetscViewerHDF5Traverse_Internal(viewer, abspath, PETSC_FALSE, NULL, &type));
  *has = (PetscBool)(type == H5O_TYPE_DATASET);
  PetscCall(PetscFree(abspath));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5HasAttribute_HDF5(PetscViewer viewer, const char parent[], const char name[], PetscBool *has)
{
  const char *parentAbsPath;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5GetGroup(viewer, parent, &parentAbsPath));
  PetscCall(PetscViewerHDF5Traverse_Internal(viewer, parentAbsPath, PETSC_FALSE, has, NULL));
  if (*has) PetscCall(PetscViewerHDF5HasAttribute_Internal(viewer, parentAbsPath, name, has));
  PetscCall(PetscFree(parentAbsPath));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCVIEWERHDF5 - A viewer that writes to an HDF5 file

  Level: beginner

.seealso: [](sec_viewers), `PetscViewerHDF5Open()`, `PetscViewerStringSPrintf()`, `PetscViewerSocketOpen()`, `PetscViewerDrawOpen()`, `PETSCVIEWERSOCKET`,
          `PetscViewerCreate()`, `PetscViewerASCIIOpen()`, `PetscViewerBinaryOpen()`, `PETSCVIEWERBINARY`, `PETSCVIEWERDRAW`, `PETSCVIEWERSTRING`,
          `PetscViewerMatlabOpen()`, `VecView()`, `DMView()`, `PetscViewerMatlabPutArray()`, `PETSCVIEWERASCII`, `PETSCVIEWERMATLAB`,
          `PetscViewerFileSetName()`, `PetscViewerFileSetMode()`, `PetscViewerFormat`, `PetscViewerType`, `PetscViewerSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_HDF5(PetscViewer v)
{
  PetscViewer_HDF5 *hdf5;

  PetscFunctionBegin;
#if !defined(H5_HAVE_PARALLEL)
  {
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)v), &size));
    PetscCheck(size <= 1, PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "Cannot use parallel HDF5 viewer since the given HDF5 does not support parallel I/O (H5_HAVE_PARALLEL is unset)");
  }
#endif

  PetscCall(PetscNew(&hdf5));

  v->data                = (void *)hdf5;
  v->ops->destroy        = PetscViewerDestroy_HDF5;
  v->ops->setfromoptions = PetscViewerSetFromOptions_HDF5;
  v->ops->setup          = PetscViewerSetUp_HDF5;
  v->ops->view           = PetscViewerView_HDF5;
  v->ops->flush          = PetscViewerFlush_HDF5;
  hdf5->btype            = FILE_MODE_UNDEFINED;
  hdf5->filename         = NULL;
  hdf5->timestep         = -1;
  hdf5->groups           = NULL;

  PetscCallHDF5Return(hdf5->dxpl_id, H5Pcreate, (H5P_DATASET_XFER));

  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetName_C", PetscViewerFileSetName_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetName_C", PetscViewerFileGetName_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetMode_C", PetscViewerFileSetMode_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetMode_C", PetscViewerFileGetMode_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5SetBaseDimension2_C", PetscViewerHDF5SetBaseDimension2_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5GetBaseDimension2_C", PetscViewerHDF5GetBaseDimension2_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5SetSPOutput_C", PetscViewerHDF5SetSPOutput_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5SetCollective_C", PetscViewerHDF5SetCollective_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5GetCollective_C", PetscViewerHDF5GetCollective_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5GetDefaultTimestepping_C", PetscViewerHDF5GetDefaultTimestepping_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5SetDefaultTimestepping_C", PetscViewerHDF5SetDefaultTimestepping_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5GetCompress_C", PetscViewerHDF5GetCompress_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5SetCompress_C", PetscViewerHDF5SetCompress_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5WriteGroup_C", PetscViewerHDF5WriteGroup_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5PushGroup_C", PetscViewerHDF5PushGroup_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5PopGroup_C", PetscViewerHDF5PopGroup_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5GetGroup_C", PetscViewerHDF5GetGroup_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5PushTimestepping_C", PetscViewerHDF5PushTimestepping_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5PopTimestepping_C", PetscViewerHDF5PopTimestepping_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5IsTimestepping_C", PetscViewerHDF5IsTimestepping_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5IncrementTimestep_C", PetscViewerHDF5IncrementTimestep_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5SetTimestep_C", PetscViewerHDF5SetTimestep_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5GetTimestep_C", PetscViewerHDF5GetTimestep_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5WriteAttribute_C", PetscViewerHDF5WriteAttribute_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5ReadAttribute_C", PetscViewerHDF5ReadAttribute_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5HasGroup_C", PetscViewerHDF5HasGroup_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5HasDataset_C", PetscViewerHDF5HasDataset_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5HasAttribute_C", PetscViewerHDF5HasAttribute_HDF5));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerHDF5GetSPOutput_C", PetscViewerHDF5GetSPOutput_HDF5));
  PetscFunctionReturn(PETSC_SUCCESS);
}
