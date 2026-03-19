#include <petsc/private/petscimpl.h>
#include <petscviewerhdf5.h>

/*@C
  PetscViewerHDF5GetGroup - Get the current HDF5 group name (full path), set with `PetscViewerHDF5PushGroup()`/`PetscViewerHDF5PopGroup()`.

  Not Collective

  Input Parameters:
+ viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`
- path   - (Optional) The path relative to the pushed group

  Output Parameter:
. abspath - The absolute HDF5 path (group)

  Level: intermediate

  Notes:
  If path starts with '/', it is taken as an absolute path overriding currently pushed group, else path is relative to the current pushed group.
  `NULL` or empty path means the current pushed group.

  The output abspath is newly allocated so needs to be freed.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5OpenGroup()`, `PetscViewerHDF5WriteGroup()`
@*/
PetscErrorCode PetscViewerHDF5GetGroup(PetscViewer viewer, const char path[], const char *abspath[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (path) PetscAssertPointer(path, 2);
  PetscAssertPointer(abspath, 3);
  PetscUseMethod(viewer, "PetscViewerHDF5GetGroup_C", (PetscViewer, const char[], const char *[]), (viewer, path, abspath));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerHDF5CheckNamedObject_Internal(PetscViewer viewer, PetscObject obj)
{
  PetscBool has;

  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5HasObject(viewer, obj, &has));
  if (!has) {
    const char *group;
    PetscCall(PetscViewerHDF5GetGroup(viewer, NULL, &group));
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Object (dataset) \"%s\" not stored in group %s", obj->name, group);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5SetBaseDimension2 - Vectors of 1 dimension (i.e. bs/dof is 1) will be saved in the HDF5 file with a
  dimension of 2.

  Logically Collective

  Input Parameters:
+ viewer - the `PetscViewer`; if it is a `PETSCVIEWERHDF5` then this command is ignored
- flg    - if `PETSC_TRUE` the vector will always have at least a dimension of 2 even if that first dimension is of size 1

  Options Database Key:
. -viewer_hdf5_base_dimension2 - turns on (true) or off (false) using a dimension of 2 in the HDF5 file even if the bs/dof of the vector is 1

  Level: intermediate

  Note:
  Setting this option allegedly makes code that reads the HDF5 in easier since they do not have a "special case" of a bs/dof
  of one when the dimension is lower. Others think the option is crazy.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerFileSetMode()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinaryOpen()`
@*/
PetscErrorCode PetscViewerHDF5SetBaseDimension2(PetscViewer viewer, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerHDF5SetBaseDimension2_C", (PetscViewer, PetscBool), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5GetBaseDimension2 - Vectors of 1 dimension (i.e. bs/dof is 1) will be saved in the HDF5 file with a
  dimension of 2.

  Logically Collective

  Input Parameter:
. viewer - the `PetscViewer`, must be `PETSCVIEWERHDF5`

  Output Parameter:
. flg - if `PETSC_TRUE` the vector will always have at least a dimension of 2 even if that first dimension is of size 1

  Level: intermediate

  Note:
  Setting this option allegedly makes code that reads the HDF5 in easier since they do not have a "special case" of a bs/dof
  of one when the dimension is lower. Others think the option is crazy.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerFileSetMode()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinaryOpen()`
@*/
PetscErrorCode PetscViewerHDF5GetBaseDimension2(PetscViewer viewer, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5GetBaseDimension2_C", (PetscViewer, PetscBool *), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5SetSPOutput - Data is written to disk in single precision even if PETSc is
  compiled with double precision `PetscReal`.

  Logically Collective

  Input Parameters:
+ viewer - the PetscViewer; if it is a `PETSCVIEWERHDF5` then this command is ignored
- flg    - if `PETSC_TRUE` the data will be written to disk with single precision

  Options Database Key:
. -viewer_hdf5_sp_output - turns on (true) or off (false) output in single precision

  Level: intermediate

  Note:
  Setting this option does not make any difference if PETSc is compiled with single precision
  in the first place. It does not affect reading datasets (HDF5 handle this internally).

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerFileSetMode()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinaryOpen()`,
          `PetscReal`, `PetscViewerHDF5GetSPOutput()`
@*/
PetscErrorCode PetscViewerHDF5SetSPOutput(PetscViewer viewer, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerHDF5SetSPOutput_C", (PetscViewer, PetscBool), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5GetSPOutput - Data is written to disk in single precision even if PETSc is
  compiled with double precision `PetscReal`.

  Logically Collective

  Input Parameter:
. viewer - the PetscViewer, must be of type `PETSCVIEWERHDF5`

  Output Parameter:
. flg - if `PETSC_TRUE` the data will be written to disk with single precision

  Level: intermediate

.seealso: [](sec_viewers), `PetscViewerFileSetMode()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinaryOpen()`,
          `PetscReal`, `PetscViewerHDF5SetSPOutput()`
@*/
PetscErrorCode PetscViewerHDF5GetSPOutput(PetscViewer viewer, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5GetSPOutput_C", (PetscViewer, PetscBool *), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5SetCollective - Use collective MPI-IO transfer mode for HDF5 reads and writes.

  Logically Collective; flg must contain common value

  Input Parameters:
+ viewer - the `PetscViewer`; if it is not `PETSCVIEWERHDF5` then this command is ignored
- flg    - `PETSC_TRUE` for collective mode; `PETSC_FALSE` for independent mode (default)

  Options Database Key:
. -viewer_hdf5_collective - turns on (true) or off (false) collective transfers

  Level: intermediate

  Note:
  Collective mode gives the MPI-IO layer underneath HDF5 a chance to do some additional collective optimizations and hence can perform better.
  However, this works correctly only since HDF5 1.10.3 and if HDF5 is installed for MPI; hence, we ignore this setting for older versions.

  Developer Notes:
  In the HDF5 layer, `PETSC_TRUE` / `PETSC_FALSE` means `H5Pset_dxpl_mpio()` is called with `H5FD_MPIO_COLLECTIVE` / `H5FD_MPIO_INDEPENDENT`, respectively.
  This in turn means use of MPI_File_{read,write}_all /  MPI_File_{read,write} in the MPI-IO layer, respectively.
  See HDF5 documentation and MPI-IO documentation for details.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5GetCollective()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerHDF5Open()`
@*/
PetscErrorCode PetscViewerHDF5SetCollective(PetscViewer viewer, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveBool(viewer, flg, 2);
  PetscTryMethod(viewer, "PetscViewerHDF5SetCollective_C", (PetscViewer, PetscBool), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5GetCollective - Return flag whether collective MPI-IO transfer mode is used for HDF5 reads and writes.

  Not Collective

  Input Parameter:
. viewer - the `PETSCVIEWERHDF5` `PetscViewer`

  Output Parameter:
. flg - the flag

  Level: intermediate

  Note:
  This setting works correctly only since HDF5 1.10.3 and if HDF5 was installed for MPI. For older versions, `PETSC_FALSE` will be always returned.
  For more details, see `PetscViewerHDF5SetCollective()`.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5SetCollective()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerHDF5Open()`
@*/
PetscErrorCode PetscViewerHDF5GetCollective(PetscViewer viewer, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  PetscUseMethod(viewer, "PetscViewerHDF5GetCollective_C", (PetscViewer, PetscBool *), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5SetDefaultTimestepping - Set the flag for default timestepping

  Logically Collective

  Input Parameters:
+ viewer - the `PetscViewer`; if it is not `PETSCVIEWERHDF5` then this command is ignored
- flg    - if `PETSC_TRUE` we will assume that timestepping is on

  Options Database Key:
. -viewer_hdf5_default_timestepping - turns on (true) or off (false) default timestepping

  Level: intermediate

  Note:
  If the timestepping attribute is not found for an object, then the default timestepping is used

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5GetDefaultTimestepping()`, `PetscViewerHDF5PushTimestepping()`, `PetscViewerHDF5GetTimestep()`
@*/
PetscErrorCode PetscViewerHDF5SetDefaultTimestepping(PetscViewer viewer, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerHDF5SetDefaultTimestepping_C", (PetscViewer, PetscBool), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5GetDefaultTimestepping - Get the flag for default timestepping

  Not Collective

  Input Parameter:
. viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`

  Output Parameter:
. flg - if `PETSC_TRUE` we will assume that timestepping is on

  Level: intermediate

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5SetDefaultTimestepping()`, `PetscViewerHDF5PushTimestepping()`, `PetscViewerHDF5GetTimestep()`
@*/
PetscErrorCode PetscViewerHDF5GetDefaultTimestepping(PetscViewer viewer, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5GetDefaultTimestepping_C", (PetscViewer, PetscBool *), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5SetCompress - Set the flag for compression

  Logically Collective

  Input Parameters:
+ viewer - the `PetscViewer`; if it is not `PETSCVIEWERHDF5` then this command is ignored
- flg    - if `PETSC_TRUE` we will turn on compression

  Options Database Key:
. -viewer_hdf5_compress - turns on (true) or off (false) compression

  Level: intermediate

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5GetCompress()`
@*/
PetscErrorCode PetscViewerHDF5SetCompress(PetscViewer viewer, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscTryMethod(viewer, "PetscViewerHDF5SetCompress_C", (PetscViewer, PetscBool), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5GetCompress - Get the flag for compression

  Not Collective

  Input Parameter:
. viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`

  Output Parameter:
. flg - if `PETSC_TRUE` we will turn on compression

  Level: intermediate

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5SetCompress()`
@*/
PetscErrorCode PetscViewerHDF5GetCompress(PetscViewer viewer, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5GetCompress_C", (PetscViewer, PetscBool *), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5Open - Opens a file for HDF5 input/output as a `PETSCVIEWERHDF5` `PetscViewer`

  Collective

  Input Parameters:
+ comm - MPI communicator
. name - name of file
- type - type of file

  Output Parameter:
. hdf5v - `PetscViewer` for HDF5 input/output to use with the specified file

  Options Database Keys:
+ -viewer_hdf5_base_dimension2 - turns on (true) or off (false) using a dimension of 2 in the HDF5 file even if the bs/dof of the vector is 1
- -viewer_hdf5_sp_output       - forces (if true) the viewer to write data in single precision independent on the precision of PetscReal

  Level: beginner

  Notes:
  Reading is always available, regardless of the mode. Available modes are
.vb
  FILE_MODE_READ - open existing HDF5 file for read only access, fail if file does not exist [H5Fopen() with H5F_ACC_RDONLY]
  FILE_MODE_WRITE - if file exists, fully overwrite it, else create new HDF5 file [H5FcreateH5Fcreate() with H5F_ACC_TRUNC]
  FILE_MODE_APPEND - if file exists, keep existing contents [H5Fopen() with H5F_ACC_RDWR], else create new HDF5 file [H5FcreateH5Fcreate() with H5F_ACC_EXCL]
  FILE_MODE_UPDATE - same as FILE_MODE_APPEND
.ve

  In case of `FILE_MODE_APPEND` / `FILE_MODE_UPDATE`, any stored object (dataset, attribute) can be selectively overwritten if the same fully qualified name (/group/path/to/object) is specified.

  This PetscViewer should be destroyed with PetscViewerDestroy().

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerASCIIOpen()`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`, `PetscViewerHDF5SetBaseDimension2()`,
          `PetscViewerHDF5SetSPOutput()`, `PetscViewerHDF5GetBaseDimension2()`, `VecView()`, `MatView()`, `VecLoad()`,
          `MatLoad()`, `PetscFileMode`, `PetscViewer`, `PetscViewerSetType()`, `PetscViewerFileSetMode()`, `PetscViewerFileSetName()`
@*/
PetscErrorCode PetscViewerHDF5Open(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *hdf5v)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm, hdf5v));
  PetscCall(PetscViewerSetType(*hdf5v, PETSCVIEWERHDF5));
  PetscCall(PetscViewerFileSetMode(*hdf5v, type));
  PetscCall(PetscViewerFileSetName(*hdf5v, name));
  PetscCall(PetscViewerSetFromOptions(*hdf5v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5PushGroup - Set the current HDF5 group for output

  Not Collective

  Input Parameters:
+ viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`
- name   - The group name

  Level: intermediate

  Notes:
  This is designed to mnemonically resemble the Unix cd command.
.vb
  If name begins with '/', it is interpreted as an absolute path fully replacing current group, otherwise it is taken as relative to the current group.
  `NULL`, empty string, or any sequence of all slashes (e.g. "///") is interpreted as the root group "/".
  "." means the current group is pushed again.
.ve

  Example:
  Suppose the current group is "/a".
.vb
  If name is `NULL`, empty string, or a sequence of all slashes (e.g. "///"), then the new group will be "/".
  If name is ".", then the new group will be "/a".
  If name is "b", then the new group will be "/a/b".
  If name is "/b", then the new group will be "/b".
.ve

  Developer Notes:
  The root group "/" is internally stored as `NULL`.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`, `PetscViewerHDF5OpenGroup()`, `PetscViewerHDF5WriteGroup()`
@*/
PetscErrorCode PetscViewerHDF5PushGroup(PetscViewer viewer, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (name) PetscAssertPointer(name, 2);
  PetscUseMethod(viewer, "PetscViewerHDF5PushGroup_C", (PetscViewer, const char[]), (viewer, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5PopGroup - Return the current HDF5 group for output to the previous value

  Not Collective

  Input Parameter:
. viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`

  Level: intermediate

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5GetGroup()`, `PetscViewerHDF5OpenGroup()`, `PetscViewerHDF5WriteGroup()`
@*/
PetscErrorCode PetscViewerHDF5PopGroup(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5PopGroup_C", (PetscViewer), (viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5WriteGroup - Ensure the HDF5 group exists in the HDF5 file

  Not Collective

  Input Parameters:
+ viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`
- path   - (Optional) The path relative to the pushed group

  Level: intermediate

  Note:
  If path starts with '/', it is taken as an absolute path overriding currently pushed group, else path is relative to the current pushed group.
  `NULL` or empty path means the current pushed group.

  This will fail if the viewer is not writable.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`, `PetscViewerHDF5OpenGroup()`
@*/
PetscErrorCode PetscViewerHDF5WriteGroup(PetscViewer viewer, const char path[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5WriteGroup_C", (PetscViewer, const char[]), (viewer, path));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5PushTimestepping - Activate timestepping mode for subsequent HDF5 reading and writing.

  Not Collective

  Input Parameter:
. viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`

  Level: intermediate

  Notes:
  On first `PetscViewerHDF5PushTimestepping()`, the initial time step is set to 0.
  Next timesteps can then be set using `PetscViewerHDF5IncrementTimestep()` or `PetscViewerHDF5SetTimestep()`.
  Current timestep value determines which timestep is read from or written to any dataset on the next HDF5 I/O operation [e.g. `VecView()`].
  Use `PetscViewerHDF5PopTimestepping()` to deactivate timestepping mode; calling it by the end of the program is NOT mandatory.
  Current timestep is remembered between `PetscViewerHDF5PopTimestepping()` and the next `PetscViewerHDF5PushTimestepping()`.

  If a dataset was stored with timestepping, it can be loaded only in the timestepping mode again.
  Loading a timestepped dataset with timestepping disabled, or vice-versa results in an error.

  Developer Notes:
  Timestepped HDF5 dataset has an extra dimension and attribute "timestepping" set to true.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PopTimestepping()`, `PetscViewerHDF5IsTimestepping()`, `PetscViewerHDF5SetTimestep()`, `PetscViewerHDF5IncrementTimestep()`, `PetscViewerHDF5GetTimestep()`
@*/
PetscErrorCode PetscViewerHDF5PushTimestepping(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5PushTimestepping_C", (PetscViewer), (viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5PopTimestepping - Deactivate timestepping mode for subsequent HDF5 reading and writing.

  Not Collective

  Input Parameter:
. viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`

  Level: intermediate

  Note:
  See `PetscViewerHDF5PushTimestepping()` for details.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushTimestepping()`, `PetscViewerHDF5IsTimestepping()`, `PetscViewerHDF5SetTimestep()`, `PetscViewerHDF5IncrementTimestep()`, `PetscViewerHDF5GetTimestep()`
@*/
PetscErrorCode PetscViewerHDF5PopTimestepping(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5PopTimestepping_C", (PetscViewer), (viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5IsTimestepping - Ask the viewer whether it is in timestepping mode currently.

  Not Collective

  Input Parameter:
. viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`

  Output Parameter:
. flg - is timestepping active?

  Level: intermediate

  Note:
  See `PetscViewerHDF5PushTimestepping()` for details.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushTimestepping()`, `PetscViewerHDF5PopTimestepping()`, `PetscViewerHDF5SetTimestep()`, `PetscViewerHDF5IncrementTimestep()`, `PetscViewerHDF5GetTimestep()`
@*/
PetscErrorCode PetscViewerHDF5IsTimestepping(PetscViewer viewer, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5IsTimestepping_C", (PetscViewer, PetscBool *), (viewer, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5IncrementTimestep - Increments current timestep for the HDF5 output. Fields are stacked in time.

  Not Collective

  Input Parameter:
. viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`

  Level: intermediate

  Note:
  This can be called only if the viewer is in timestepping mode. See `PetscViewerHDF5PushTimestepping()` for details.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushTimestepping()`, `PetscViewerHDF5SetTimestep()`, `PetscViewerHDF5GetTimestep()`
@*/
PetscErrorCode PetscViewerHDF5IncrementTimestep(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscUseMethod(viewer, "PetscViewerHDF5IncrementTimestep_C", (PetscViewer), (viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5SetTimestep - Set the current timestep for the HDF5 output. Fields are stacked in time.

  Logically Collective

  Input Parameters:
+ viewer   - the `PetscViewer` of type `PETSCVIEWERHDF5`
- timestep - The timestep

  Level: intermediate

  Note:
  This can be called only if the viewer is in timestepping mode. See `PetscViewerHDF5PushTimestepping()` for details.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushTimestepping()`, `PetscViewerHDF5IncrementTimestep()`, `PetscViewerHDF5GetTimestep()`
@*/
PetscErrorCode PetscViewerHDF5SetTimestep(PetscViewer viewer, PetscInt timestep)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(viewer, timestep, 2);
  PetscUseMethod(viewer, "PetscViewerHDF5SetTimestep_C", (PetscViewer, PetscInt), (viewer, timestep));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5GetTimestep - Get the current timestep for the HDF5 output. Fields are stacked in time.

  Not Collective

  Input Parameter:
. viewer - the `PetscViewer` of type `PETSCVIEWERHDF5`

  Output Parameter:
. timestep - The timestep

  Level: intermediate

  Note:
  This can be called only if the viewer is in the timestepping mode. See `PetscViewerHDF5PushTimestepping()` for details.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5PushTimestepping()`, `PetscViewerHDF5IncrementTimestep()`, `PetscViewerHDF5SetTimestep()`
@*/
PetscErrorCode PetscViewerHDF5GetTimestep(PetscViewer viewer, PetscInt *timestep)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(timestep, 2);
  PetscUseMethod(viewer, "PetscViewerHDF5GetTimestep_C", (PetscViewer, PetscInt *), (viewer, timestep));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5WriteAttribute - Write an attribute

  Collective

  Input Parameters:
+ viewer   - The `PETSCVIEWERHDF5` viewer
. parent   - The parent dataset/group name
. name     - The attribute name
. datatype - The attribute type
- value    - The attribute value

  Level: advanced

  Note:
  If parent starts with '/', it is taken as an absolute path overriding currently pushed group, else parent is relative to the current pushed group.
  `NULL` means the current pushed group.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5WriteObjectAttribute()`, `PetscViewerHDF5ReadAttribute()`, `PetscViewerHDF5HasAttribute()`,
          `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`
@*/
PetscErrorCode PetscViewerHDF5WriteAttribute(PetscViewer viewer, const char parent[], const char name[], PetscDataType datatype, const void *value)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (parent) PetscAssertPointer(parent, 2);
  PetscAssertPointer(name, 3);
  PetscValidLogicalCollectiveEnum(viewer, datatype, 4);
  PetscAssertPointer(value, 5);
  PetscUseMethod(viewer, "PetscViewerHDF5WriteAttribute_C", (PetscViewer, const char[], const char[], PetscDataType, const void *), (viewer, parent, name, datatype, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5WriteObjectAttribute - Write an attribute to the dataset matching the given `PetscObject` by name

  Collective

  Input Parameters:
+ viewer   - The `PETSCVIEWERHDF5` viewer
. obj      - The object whose name is used to lookup the parent dataset, relative to the current group.
. name     - The attribute name
. datatype - The attribute type
- value    - The attribute value

  Level: advanced

  Note:
  This fails if the path current_group/object_name doesn't resolve to a dataset (the path doesn't exist or is not a dataset).
  You might want to check first if it does using `PetscViewerHDF5HasObject()`.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5WriteAttribute()`, `PetscViewerHDF5ReadObjectAttribute()`, `PetscViewerHDF5HasObjectAttribute()`,
          `PetscViewerHDF5HasObject()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`
@*/
PetscErrorCode PetscViewerHDF5WriteObjectAttribute(PetscViewer viewer, PetscObject obj, const char name[], PetscDataType datatype, const void *value)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidHeader(obj, 2);
  PetscAssertPointer(name, 3);
  PetscValidLogicalCollectiveEnum(viewer, datatype, 4);
  PetscAssertPointer(value, 5);
  PetscCall(PetscViewerHDF5CheckNamedObject_Internal(viewer, obj));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, obj->name, name, datatype, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5ReadAttribute - Read an attribute

  Collective

  Input Parameters:
+ viewer       - The `PETSCVIEWERHDF5` viewer
. parent       - The parent dataset/group name
. name         - The attribute name
. datatype     - The attribute type
- defaultValue - The pointer to the default value

  Output Parameter:
. value - The pointer to the read HDF5 attribute value

  Level: advanced

  Notes:
  If defaultValue is `NULL` and the attribute is not found, an error occurs.

  If defaultValue is not `NULL` and the attribute is not found, `defaultValue` is copied to value.

  The pointers `defaultValue` and `value` can be the same; for instance
.vb
  flg = PETSC_FALSE;
  PetscCall(`PetscViewerHDF5ReadAttribute`(viewer,name,"attr",PETSC_BOOL,&flg,&flg));
.ve
  is valid, but make sure the default value is initialized.

  If the datatype is `PETSC_STRING`, the output string is newly allocated so one must `PetscFree()` it when no longer needed.

  If parent starts with '/', it is taken as an absolute path overriding currently pushed group, else parent is relative to the current pushed group. `NULL` means the current pushed group.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5ReadObjectAttribute()`, `PetscViewerHDF5WriteAttribute()`, `PetscViewerHDF5HasAttribute()`, `PetscViewerHDF5HasObject()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`
@*/
PetscErrorCode PetscViewerHDF5ReadAttribute(PetscViewer viewer, const char parent[], const char name[], PetscDataType datatype, const void *defaultValue, void *value)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (parent) PetscAssertPointer(parent, 2);
  PetscAssertPointer(name, 3);
  PetscValidLogicalCollectiveEnum(viewer, datatype, 4);
  if (defaultValue) PetscAssertPointer(defaultValue, 5);
  PetscAssertPointer(value, 6);
  PetscUseMethod(viewer, "PetscViewerHDF5ReadAttribute_C", (PetscViewer, const char[], const char[], PetscDataType, const void *, void *), (viewer, parent, name, datatype, defaultValue, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5ReadObjectAttribute - Read an attribute from the dataset matching the given `PetscObject` by name

  Collective

  Input Parameters:
+ viewer       - The `PETSCVIEWERHDF5` viewer
. obj          - The object whose name is used to lookup the parent dataset, relative to the current group.
. name         - The attribute name
. datatype     - The attribute type
- defaultValue - The default attribute value

  Output Parameter:
. value - The attribute value

  Level: advanced

  Note:
  This fails if current_group/object_name doesn't resolve to a dataset (the path doesn't exist or is not a dataset).
  You might want to check first if it does using `PetscViewerHDF5HasObject()`.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5ReadAttribute()`, `PetscViewerHDF5WriteObjectAttribute()`, `PetscViewerHDF5HasObjectAttribute()`,
          `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`
@*/
PetscErrorCode PetscViewerHDF5ReadObjectAttribute(PetscViewer viewer, PetscObject obj, const char name[], PetscDataType datatype, void *defaultValue, void *value)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidHeader(obj, 2);
  PetscAssertPointer(name, 3);
  PetscAssertPointer(value, 6);
  PetscCall(PetscViewerHDF5CheckNamedObject_Internal(viewer, obj));
  PetscCall(PetscViewerHDF5ReadAttribute(viewer, obj->name, name, datatype, defaultValue, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5HasGroup - Check whether the current (pushed) group exists in the HDF5 file

  Collective

  Input Parameters:
+ viewer - The `PETSCVIEWERHDF5` viewer
- path   - (Optional) The path relative to the pushed group

  Output Parameter:
. has - Flag for group existence

  Level: advanced

  Notes:
  If path starts with '/', it is taken as an absolute path overriding currently pushed group, else path is relative to the current pushed group.
  `NULL` or empty path means the current pushed group.

  If path exists but is not a group, `PETSC_FALSE` is returned.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5HasAttribute()`, `PetscViewerHDF5HasDataset()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`, `PetscViewerHDF5OpenGroup()`
@*/
PetscErrorCode PetscViewerHDF5HasGroup(PetscViewer viewer, const char path[], PetscBool *has)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (path) PetscAssertPointer(path, 2);
  PetscAssertPointer(has, 3);
  PetscUseMethod(viewer, "PetscViewerHDF5HasGroup_C", (PetscViewer, const char[], PetscBool *), (viewer, path, has));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5HasDataset - Check whether a given dataset exists in the HDF5 file

  Collective

  Input Parameters:
+ viewer - The `PETSCVIEWERHDF5` viewer
- path   - The dataset path

  Output Parameter:
. has - Flag whether dataset exists

  Level: advanced

  Notes:
  If path starts with '/', it is taken as an absolute path overriding currently pushed group, else path is relative to the current pushed group.

  If `path` is `NULL` or empty, has is set to `PETSC_FALSE`.

  If `path` exists but is not a dataset, has is set to `PETSC_FALSE` as well.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5HasObject()`, `PetscViewerHDF5HasAttribute()`, `PetscViewerHDF5HasGroup()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`
@*/
PetscErrorCode PetscViewerHDF5HasDataset(PetscViewer viewer, const char path[], PetscBool *has)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (path) PetscAssertPointer(path, 2);
  PetscAssertPointer(has, 3);
  PetscUseMethod(viewer, "PetscViewerHDF5HasDataset_C", (PetscViewer, const char[], PetscBool *), (viewer, path, has));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerHDF5HasObject - Check whether a dataset with the same name as given object exists in the HDF5 file under current group

  Collective

  Input Parameters:
+ viewer - The `PETSCVIEWERHDF5` viewer
- obj    - The named object

  Output Parameter:
. has - Flag for dataset existence

  Level: advanced

  Notes:
  If the object is unnamed, an error occurs.

  If the path current_group/object_name exists but is not a dataset, has is set to `PETSC_FALSE` as well.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5HasDataset()`, `PetscViewerHDF5HasAttribute()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`
@*/
PetscErrorCode PetscViewerHDF5HasObject(PetscViewer viewer, PetscObject obj, PetscBool *has)
{
  size_t len;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidHeader(obj, 2);
  PetscAssertPointer(has, 3);
  PetscCall(PetscStrlen(obj->name, &len));
  PetscCheck(len, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONG, "Object must be named");
  PetscCall(PetscViewerHDF5HasDataset(viewer, obj->name, has));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5HasAttribute - Check whether an attribute exists

  Collective

  Input Parameters:
+ viewer - The `PETSCVIEWERHDF5` viewer
. parent - The parent dataset/group name
- name   - The attribute name

  Output Parameter:
. has - Flag for attribute existence

  Level: advanced

  Note:
  If parent starts with '/', it is taken as an absolute path overriding currently pushed group, else parent is relative to the current pushed group. `NULL` means the current pushed group.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5HasObjectAttribute()`, `PetscViewerHDF5WriteAttribute()`, `PetscViewerHDF5ReadAttribute()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`
@*/
PetscErrorCode PetscViewerHDF5HasAttribute(PetscViewer viewer, const char parent[], const char name[], PetscBool *has)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (parent) PetscAssertPointer(parent, 2);
  PetscAssertPointer(name, 3);
  PetscAssertPointer(has, 4);
  PetscUseMethod(viewer, "PetscViewerHDF5HasAttribute_C", (PetscViewer, const char[], const char[], PetscBool *), (viewer, parent, name, has));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerHDF5HasObjectAttribute - Check whether an attribute is attached to the dataset matching the given `PetscObject` by name

  Collective

  Input Parameters:
+ viewer - The `PETSCVIEWERHDF5` viewer
. obj    - The object whose name is used to lookup the parent dataset, relative to the current group.
- name   - The attribute name

  Output Parameter:
. has - Flag for attribute existence

  Level: advanced

  Note:
  This fails if current_group/object_name doesn't resolve to a dataset (the path doesn't exist or is not a dataset).
  You might want to check first if it does using `PetscViewerHDF5HasObject()`.

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerHDF5HasAttribute()`, `PetscViewerHDF5WriteObjectAttribute()`, `PetscViewerHDF5ReadObjectAttribute()`, `PetscViewerHDF5HasObject()`, `PetscViewerHDF5PushGroup()`, `PetscViewerHDF5PopGroup()`, `PetscViewerHDF5GetGroup()`
@*/
PetscErrorCode PetscViewerHDF5HasObjectAttribute(PetscViewer viewer, PetscObject obj, const char name[], PetscBool *has)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidHeader(obj, 2);
  PetscAssertPointer(name, 3);
  PetscAssertPointer(has, 4);
  PetscCall(PetscViewerHDF5CheckNamedObject_Internal(viewer, obj));
  PetscCall(PetscViewerHDF5HasAttribute(viewer, obj->name, name, has));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The variable Petsc_Viewer_HDF5_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
PetscMPIInt Petsc_Viewer_HDF5_keyval = MPI_KEYVAL_INVALID;

/*@C
  PETSC_VIEWER_HDF5_ - Creates an `PETSCVIEWERHDF5` `PetscViewer` shared by all processors in a communicator.

  Collective

  Input Parameter:
. comm - the MPI communicator to share the `PETSCVIEWERHDF5` `PetscViewer`

  Options Database Key:
. -viewer_hdf5_filename name - name of the HDF5 file

  Environmental variable:
. `PETSC_VIEWER_HDF5_FILENAME` - name of the HDF5 file

  Level: intermediate

  Note:
  Unlike almost all other PETSc routines, `PETSC_VIEWER_HDF5_()` does not return
  an error code.  The HDF5 `PetscViewer` is usually used in the form
.vb
  XXXView(XXX object, PETSC_VIEWER_HDF5_(comm));
.ve

.seealso: [](sec_viewers), `PETSCVIEWERHDF5`, `PetscViewerHDF5Open()`, `PetscViewerCreate()`, `PetscViewerDestroy()`
@*/
PetscViewer PETSC_VIEWER_HDF5_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    mpi_ierr;
  PetscBool      flg;
  PetscMPIInt    iflg;
  PetscViewer    viewer;
  char           fname[PETSC_MAX_PATH_LEN];
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm, &ncomm, NULL);
  if (ierr) {
    ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_HDF5_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
    PetscFunctionReturn(NULL);
  }
  if (Petsc_Viewer_HDF5_keyval == MPI_KEYVAL_INVALID) {
    mpi_ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &Petsc_Viewer_HDF5_keyval, NULL);
    if (mpi_ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_HDF5_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
      PetscFunctionReturn(NULL);
    }
  }
  mpi_ierr = MPI_Comm_get_attr(ncomm, Petsc_Viewer_HDF5_keyval, (void **)&viewer, &iflg);
  if (mpi_ierr) {
    ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_HDF5_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
    PetscFunctionReturn(NULL);
  }
  if (!iflg) { /* PetscViewer not yet created */
    ierr = PetscOptionsGetenv(ncomm, "PETSC_VIEWER_HDF5_FILENAME", fname, PETSC_MAX_PATH_LEN, &flg);
    if (ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_HDF5_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
      PetscFunctionReturn(NULL);
    }
    if (!flg) {
      ierr = PetscStrncpy(fname, "output.h5", sizeof(fname));
      if (ierr) {
        ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_HDF5_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
        PetscFunctionReturn(NULL);
      }
    }
    ierr = PetscViewerHDF5Open(ncomm, fname, FILE_MODE_WRITE, &viewer);
    if (ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_HDF5_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
      PetscFunctionReturn(NULL);
    }
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_HDF5_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
      PetscFunctionReturn(NULL);
    }
    mpi_ierr = MPI_Comm_set_attr(ncomm, Petsc_Viewer_HDF5_keyval, (void *)viewer);
    if (mpi_ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_HDF5_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
      PetscFunctionReturn(NULL);
    }
  }
  ierr = PetscCommDestroy(&ncomm);
  if (ierr) {
    ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_HDF5_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
    PetscFunctionReturn(NULL);
  }
  PetscFunctionReturn(viewer);
}
