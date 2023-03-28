#include <petsc/private/viewerimpl.h> /*I   "petscviewer.h"   I*/

/*
   This needs to start the same as PetscViewer_Socket.
*/
typedef struct {
  int       fdes;        /* file descriptor, ignored if using MPI IO */
  PetscInt  flowcontrol; /* allow only <flowcontrol> messages outstanding at a time while doing IO */
  PetscBool skipheader;  /* don't write header, only raw data */
#if defined(PETSC_HAVE_MPIIO)
  PetscBool  usempiio;
  MPI_File   mfdes; /* ignored unless using MPI IO */
  MPI_File   mfsub; /* subviewer support */
  MPI_Offset moff;
#endif
  char         *filename;            /* file name */
  PetscFileMode filemode;            /* read/write/append mode */
  FILE         *fdes_info;           /* optional file containing info on binary file*/
  PetscBool     storecompressed;     /* gzip the write binary file when closing it*/
  char         *ogzfilename;         /* gzip can be run after the filename has been updated */
  PetscBool     skipinfo;            /* Don't create info file for writing; don't use for reading */
  PetscBool     skipoptions;         /* don't use PETSc options database when loading */
  PetscBool     matlabheaderwritten; /* if format is PETSC_VIEWER_BINARY_MATLAB has the MATLAB .info header been written yet */
  PetscBool     setfromoptionscalled;
} PetscViewer_Binary;

static PetscErrorCode PetscViewerBinaryClearFunctionList(PetscViewer v)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetFlowControl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetFlowControl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetSkipHeader_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetSkipHeader_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetSkipOptions_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetSkipOptions_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetSkipInfo_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetSkipInfo_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetInfoPointer_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetMode_C", NULL));
#if defined(PETSC_HAVE_MPIIO)
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetUseMPIIO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetUseMPIIO_C", NULL));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerBinarySyncMPIIO(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  if (vbinary->filemode == FILE_MODE_READ) PetscFunctionReturn(PETSC_SUCCESS);
  if (vbinary->mfsub != MPI_FILE_NULL) PetscCallMPI(MPI_File_sync(vbinary->mfsub));
  if (vbinary->mfdes != MPI_FILE_NULL) {
    PetscCallMPI(MPI_Barrier(PetscObjectComm((PetscObject)viewer)));
    PetscCallMPI(MPI_File_sync(vbinary->mfdes));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode PetscViewerGetSubViewer_Binary(PetscViewer viewer, MPI_Comm comm, PetscViewer *outviewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;
  PetscMPIInt         rank;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));

  /* Return subviewer in process zero */
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  if (rank == 0) {
    PetscMPIInt flg;

    PetscCallMPI(MPI_Comm_compare(PETSC_COMM_SELF, comm, &flg));
    PetscCheck(flg == MPI_IDENT || flg == MPI_CONGRUENT, PETSC_COMM_SELF, PETSC_ERR_SUP, "PetscViewerGetSubViewer() for PETSCVIEWERBINARY requires a singleton MPI_Comm");
    PetscCall(PetscViewerCreate(comm, outviewer));
    PetscCall(PetscViewerSetType(*outviewer, PETSCVIEWERBINARY));
    PetscCall(PetscMemcpy((*outviewer)->data, vbinary, sizeof(PetscViewer_Binary)));
    (*outviewer)->setupcalled = PETSC_TRUE;
  } else {
    *outviewer = NULL;
  }

#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio && *outviewer) {
    PetscViewer_Binary *obinary = (PetscViewer_Binary *)(*outviewer)->data;
    /* Parent viewer opens a new MPI file handle on PETSC_COMM_SELF and keeps track of it for future reuse */
    if (vbinary->mfsub == MPI_FILE_NULL) {
      int amode;
      switch (vbinary->filemode) {
      case FILE_MODE_READ:
        amode = MPI_MODE_RDONLY;
        break;
      case FILE_MODE_WRITE:
        amode = MPI_MODE_WRONLY;
        break;
      case FILE_MODE_APPEND:
        amode = MPI_MODE_WRONLY;
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported file mode %s", PetscFileModes[vbinary->filemode]);
      }
      PetscCallMPI(MPI_File_open(PETSC_COMM_SELF, vbinary->filename, amode, MPI_INFO_NULL, &vbinary->mfsub));
    }
    /* Subviewer gets the MPI file handle on PETSC_COMM_SELF */
    obinary->mfdes = vbinary->mfsub;
    obinary->mfsub = MPI_FILE_NULL;
    obinary->moff  = vbinary->moff;
  }
#endif

#if defined(PETSC_HAVE_MPIIO)
  PetscCall(PetscViewerBinarySyncMPIIO(viewer));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerRestoreSubViewer_Binary(PetscViewer viewer, MPI_Comm comm, PetscViewer *outviewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;
  PetscMPIInt         rank;
#if defined(PETSC_HAVE_MPIIO)
  MPI_Offset moff = 0;
#endif

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  PetscCheck(rank == 0 || !*outviewer, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Subviewer not obtained from viewer");

#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio && *outviewer) {
    PetscViewer_Binary *obinary = (PetscViewer_Binary *)(*outviewer)->data;
    PetscCheck(obinary->mfdes == vbinary->mfsub, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Subviewer not obtained from viewer");
    if (obinary->mfsub != MPI_FILE_NULL) PetscCallMPI(MPI_File_close(&obinary->mfsub));
    moff = obinary->moff;
  }
#endif

  if (*outviewer) {
    PetscViewer_Binary *obinary = (PetscViewer_Binary *)(*outviewer)->data;
    PetscCheck(obinary->fdes == vbinary->fdes, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Subviewer not obtained from viewer");
    PetscCall(PetscFree((*outviewer)->data));
    PetscCall(PetscViewerBinaryClearFunctionList(*outviewer));
    PetscCall(PetscHeaderDestroy(outviewer));
  }

#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio) {
    PetscInt64 ioff = (PetscInt64)moff; /* We could use MPI_OFFSET datatype (requires MPI 2.2) */
    PetscCallMPI(MPI_Bcast(&ioff, 1, MPIU_INT64, 0, PetscObjectComm((PetscObject)viewer)));
    vbinary->moff = (MPI_Offset)ioff;
  }
#endif

#if defined(PETSC_HAVE_MPIIO)
  PetscCall(PetscViewerBinarySyncMPIIO(viewer));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPIIO)
/*@C
    PetscViewerBinaryGetMPIIOOffset - Gets the current global offset that should be passed to `MPI_File_set_view()` or `MPI_File_{write|read}_at[_all]()`

    Not Collective; No Fortran Support

    Input Parameter:
.   viewer - PetscViewer context, obtained from `PetscViewerBinaryOpen()`

    Output Parameter:
.   off - the current global offset

    Level: advanced

    Note:
    Use `PetscViewerBinaryAddMPIIOOffset()` to increase this value after you have written a view.

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetInfoPointer()`, `PetscViewerBinaryGetUseMPIIO()`, `PetscViewerBinarySetUseMPIIO()`, `PetscViewerBinaryAddMPIIOOffset()`
@*/
PetscErrorCode PetscViewerBinaryGetMPIIOOffset(PetscViewer viewer, MPI_Offset *off)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERBINARY);
  PetscValidPointer(off, 2);
  vbinary = (PetscViewer_Binary *)viewer->data;
  *off    = vbinary->moff;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscViewerBinaryAddMPIIOOffset - Adds to the current global offset

    Logically Collective; No Fortran Support

    Input Parameters:
+   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`
-   off - the addition to the global offset

    Level: advanced

    Note:
    Use `PetscViewerBinaryGetMPIIOOffset()` to get the value that you should pass to `MPI_File_set_view()` or `MPI_File_{write|read}_at[_all]()`

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetInfoPointer()`, `PetscViewerBinaryGetUseMPIIO()`, `PetscViewerBinarySetUseMPIIO()`, `PetscViewerBinaryGetMPIIOOffset()`
@*/
PetscErrorCode PetscViewerBinaryAddMPIIOOffset(PetscViewer viewer, MPI_Offset off)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERBINARY);
  PetscValidLogicalCollectiveInt(viewer, (PetscInt)off, 2);
  vbinary = (PetscViewer_Binary *)viewer->data;
  vbinary->moff += off;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscViewerBinaryGetMPIIODescriptor - Extracts the MPI IO file descriptor from a `PetscViewer`.

    Not Collective; No Fortran Support

    Input Parameter:
.   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`

    Output Parameter:
.   fdes - file descriptor

    Level: advanced

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetInfoPointer()`, `PetscViewerBinaryGetUseMPIIO()`, `PetscViewerBinarySetUseMPIIO()`, `PetscViewerBinaryGetMPIIOOffset()`
@*/
PetscErrorCode PetscViewerBinaryGetMPIIODescriptor(PetscViewer viewer, MPI_File *fdes)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERBINARY);
  PetscValidPointer(fdes, 2);
  PetscCall(PetscViewerSetUp(viewer));
  vbinary = (PetscViewer_Binary *)viewer->data;
  *fdes   = vbinary->mfdes;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@
    PetscViewerBinarySetUseMPIIO - Sets a binary viewer to use MPI-IO for reading/writing. Must be called
        before `PetscViewerFileSetName()`

    Logically Collective

    Input Parameters:
+   viewer - the `PetscViewer`; must be a `PETSCVIEWERBINARY`
-   use - `PETSC_TRUE` means MPI-IO will be used

    Options Database Key:
    -viewer_binary_mpiio : Flag for using MPI-IO

    Level: advanced

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerFileSetMode()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinaryOpen()`,
          `PetscViewerBinaryGetUseMPIIO()`
@*/
PetscErrorCode PetscViewerBinarySetUseMPIIO(PetscViewer viewer, PetscBool use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveBool(viewer, use, 2);
  PetscTryMethod(viewer, "PetscViewerBinarySetUseMPIIO_C", (PetscViewer, PetscBool), (viewer, use));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerBinarySetUseMPIIO_Binary(PetscViewer viewer, PetscBool use)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;
  PetscFunctionBegin;
  PetscCheck(!viewer->setupcalled || vbinary->usempiio == use, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ORDER, "Cannot change MPIIO to %s after setup", PetscBools[use]);
  vbinary->usempiio = use;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@
    PetscViewerBinaryGetUseMPIIO - Returns `PETSC_TRUE` if the binary viewer uses MPI-IO.

    Not Collective

    Input Parameter:
.   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`; must be a `PETSCVIEWERBINARY`

    Output Parameter:
.   use - `PETSC_TRUE` if MPI-IO is being used

    Level: advanced

    Note:
    If MPI-IO is not available, this function will always return `PETSC_FALSE`

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetInfoPointer()`, `PetscViewerBinarySetUseMPIIO()`, `PetscViewerBinaryGetMPIIOOffset()`
@*/
PetscErrorCode PetscViewerBinaryGetUseMPIIO(PetscViewer viewer, PetscBool *use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidBoolPointer(use, 2);
  *use = PETSC_FALSE;
  PetscTryMethod(viewer, "PetscViewerBinaryGetUseMPIIO_C", (PetscViewer, PetscBool *), (viewer, use));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerBinaryGetUseMPIIO_Binary(PetscViewer viewer, PetscBool *use)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  *use = vbinary->usempiio;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@
    PetscViewerBinarySetFlowControl - Sets how many messages are allowed to outstanding at the same time during parallel IO reads/writes

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from `PetscViewerBinaryOpen()`
-   fc - the number of messages, defaults to 256 if this function was not called

    Level: advanced

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetInfoPointer()`, `PetscViewerBinaryGetFlowControl()`
@*/
PetscErrorCode PetscViewerBinarySetFlowControl(PetscViewer viewer, PetscInt fc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(viewer, fc, 2);
  PetscTryMethod(viewer, "PetscViewerBinarySetFlowControl_C", (PetscViewer, PetscInt), (viewer, fc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerBinarySetFlowControl_Binary(PetscViewer viewer, PetscInt fc)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  PetscCheck(fc > 1, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_OUTOFRANGE, "Flow control count must be greater than 1, %" PetscInt_FMT " was set", fc);
  vbinary->flowcontrol = fc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscViewerBinaryGetFlowControl - Returns how many messages are allowed to outstanding at the same time during parallel IO reads/writes

    Not Collective

    Input Parameter:
.   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`

    Output Parameter:
.   fc - the number of messages

    Level: advanced

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetInfoPointer()`, `PetscViewerBinarySetFlowControl()`
@*/
PetscErrorCode PetscViewerBinaryGetFlowControl(PetscViewer viewer, PetscInt *fc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidIntPointer(fc, 2);
  PetscUseMethod(viewer, "PetscViewerBinaryGetFlowControl_C", (PetscViewer, PetscInt *), (viewer, fc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscViewerBinaryGetFlowControl_Binary(PetscViewer viewer, PetscInt *fc)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  *fc = vbinary->flowcontrol;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscViewerBinaryGetDescriptor - Extracts the file descriptor from a `PetscViewer` of `PetscViewerType` `PETSCVIEWERBINARY`.

    Collective because it may trigger a `PetscViewerSetUp()` call; No Fortran Support

    Input Parameter:
.   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`

    Output Parameter:
.   fdes - file descriptor

    Level: advanced

    Note:
      For writable binary `PetscViewer`s, the descriptor will only be valid for the
    first processor in the communicator that shares the `PetscViewer`. For readable
    files it will only be valid on nodes that have the file. If node 0 does not
    have the file it generates an error even if another node does have the file.

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetInfoPointer()`
@*/
PetscErrorCode PetscViewerBinaryGetDescriptor(PetscViewer viewer, int *fdes)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERBINARY);
  PetscValidPointer(fdes, 2);
  PetscCall(PetscViewerSetUp(viewer));
  vbinary = (PetscViewer_Binary *)viewer->data;
  *fdes   = vbinary->fdes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscViewerBinarySkipInfo - Binary file will not have .info file created with it

    Not Collective

    Input Parameter:
.   viewer - `PetscViewer` context, obtained from `PetscViewerCreate()`

    Options Database Key:
.   -viewer_binary_skip_info - true indicates do not generate .info file

    Level: advanced

    Notes:
    This must be called after `PetscViewerSetType()`. If you use `PetscViewerBinaryOpen()` then
    you can only skip the info file with the `-viewer_binary_skip_info` flag. To use the function you must open the
    viewer with `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinarySkipInfo()`.

    The .info contains meta information about the data in the binary file, for example the block size if it was
    set for a vector or matrix.

    This routine is deprecated, use `PetscViewerBinarySetSkipInfo()`

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetDescriptor()`, `PetscViewerBinarySetSkipOptions()`,
          `PetscViewerBinaryGetSkipOptions()`, `PetscViewerBinaryGetSkipInfo()`
@*/
PetscErrorCode PetscViewerBinarySkipInfo(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerBinarySetSkipInfo(viewer, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscViewerBinarySetSkipInfo - Binary file will not have .info file created with it

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from `PetscViewerCreate()`
-   skip - `PETSC_TRUE` implies the .info file will not be generated

    Options Database Key:
.   -viewer_binary_skip_info - true indicates do not generate .info file

    Level: advanced

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetDescriptor()`, `PetscViewerBinarySetSkipOptions()`,
          `PetscViewerBinaryGetSkipOptions()`, `PetscViewerBinaryGetSkipInfo()`
@*/
PetscErrorCode PetscViewerBinarySetSkipInfo(PetscViewer viewer, PetscBool skip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveBool(viewer, skip, 2);
  PetscTryMethod(viewer, "PetscViewerBinarySetSkipInfo_C", (PetscViewer, PetscBool), (viewer, skip));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerBinarySetSkipInfo_Binary(PetscViewer viewer, PetscBool skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  vbinary->skipinfo = skip;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscViewerBinaryGetSkipInfo - check if viewer wrote a .info file

    Not Collective

    Input Parameter:
.   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`

    Output Parameter:
.   skip - `PETSC_TRUE` implies the .info file was not generated

    Level: advanced

    Note:
    This must be called after `PetscViewerSetType()`

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetDescriptor()`,
          `PetscViewerBinarySetSkipOptions()`, `PetscViewerBinarySetSkipInfo()`
@*/
PetscErrorCode PetscViewerBinaryGetSkipInfo(PetscViewer viewer, PetscBool *skip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidBoolPointer(skip, 2);
  PetscUseMethod(viewer, "PetscViewerBinaryGetSkipInfo_C", (PetscViewer, PetscBool *), (viewer, skip));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerBinaryGetSkipInfo_Binary(PetscViewer viewer, PetscBool *skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  *skip = vbinary->skipinfo;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscViewerBinarySetSkipOptions - do not use the PETSc options database when loading objects

    Not Collective

    Input Parameters:
+   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`
-   skip - `PETSC_TRUE` means do not use the options from the options database

    Options Database Key:
.   -viewer_binary_skip_options <true or false> - true means do not use the options from the options database

    Level: advanced

    Note:
    This must be called after `PetscViewerSetType()`

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetDescriptor()`, `PetscViewerBinarySkipInfo()`,
          `PetscViewerBinaryGetSkipOptions()`
@*/
PetscErrorCode PetscViewerBinarySetSkipOptions(PetscViewer viewer, PetscBool skip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveBool(viewer, skip, 2);
  PetscTryMethod(viewer, "PetscViewerBinarySetSkipOptions_C", (PetscViewer, PetscBool), (viewer, skip));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerBinarySetSkipOptions_Binary(PetscViewer viewer, PetscBool skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  vbinary->skipoptions = skip;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscViewerBinaryGetSkipOptions - checks if viewer uses the PETSc options database when loading objects

    Not Collective

    Input Parameter:
.   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`

    Output Parameter:
.   skip - `PETSC_TRUE` means do not use

    Level: advanced

    Note:
    This must be called after `PetscViewerSetType()`

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetDescriptor()`, `PetscViewerBinarySkipInfo()`,
          `PetscViewerBinarySetSkipOptions()`
@*/
PetscErrorCode PetscViewerBinaryGetSkipOptions(PetscViewer viewer, PetscBool *skip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidBoolPointer(skip, 2);
  PetscUseMethod(viewer, "PetscViewerBinaryGetSkipOptions_C", (PetscViewer, PetscBool *), (viewer, skip));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerBinaryGetSkipOptions_Binary(PetscViewer viewer, PetscBool *skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  *skip = vbinary->skipoptions;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscViewerBinarySetSkipHeader - do not write a header with size information on output, just raw data

    Not Collective

    Input Parameters:
+   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`
-   skip - `PETSC_TRUE` means do not write header

    Options Database Key:
.   -viewer_binary_skip_header <true or false> - true means do not write header

    Level: advanced

    Note:
      This must be called after `PetscViewerSetType()`

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetDescriptor()`, `PetscViewerBinarySkipInfo()`,
          `PetscViewerBinaryGetSkipHeader()`
@*/
PetscErrorCode PetscViewerBinarySetSkipHeader(PetscViewer viewer, PetscBool skip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveBool(viewer, skip, 2);
  PetscTryMethod(viewer, "PetscViewerBinarySetSkipHeader_C", (PetscViewer, PetscBool), (viewer, skip));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerBinarySetSkipHeader_Binary(PetscViewer viewer, PetscBool skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  vbinary->skipheader = skip;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    PetscViewerBinaryGetSkipHeader - checks whether to write a header with size information on output, or just raw data

    Not Collective

    Input Parameter:
.   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`

    Output Parameter:
.   skip - `PETSC_TRUE` means do not write header

    Level: advanced

    Notes:
    This must be called after PetscViewerSetType()

    Returns `PETSC_FALSE` for `PETSCSOCKETVIEWER`, you cannot skip the header for it.

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetDescriptor()`, `PetscViewerBinarySkipInfo()`,
          `PetscViewerBinarySetSkipHeader()`
@*/
PetscErrorCode PetscViewerBinaryGetSkipHeader(PetscViewer viewer, PetscBool *skip)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidBoolPointer(skip, 2);
  PetscUseMethod(viewer, "PetscViewerBinaryGetSkipHeader_C", (PetscViewer, PetscBool *), (viewer, skip));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerBinaryGetSkipHeader_Binary(PetscViewer viewer, PetscBool *skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  *skip = vbinary->skipheader;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscViewerBinaryGetInfoPointer - Extracts the file pointer for the ASCII
          info file associated with a binary file.

    Not Collective; No Fortran Support

    Input Parameter:
.   viewer - `PetscViewer` context, obtained from `PetscViewerBinaryOpen()`

    Output Parameter:
.   file - file pointer  Always returns NULL if not a binary viewer

    Level: advanced

    Note:
      For writable binary `PetscViewer`s, the file pointer will only be valid for the
    first processor in the communicator that shares the `PetscViewer`.

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinaryGetDescriptor()`
@*/
PetscErrorCode PetscViewerBinaryGetInfoPointer(PetscViewer viewer, FILE **file)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidPointer(file, 2);
  *file = NULL;
  PetscTryMethod(viewer, "PetscViewerBinaryGetInfoPointer_C", (PetscViewer, FILE **), (viewer, file));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerBinaryGetInfoPointer_Binary(PetscViewer viewer, FILE **file)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));
  *file = vbinary->fdes_info;
  if (viewer->format == PETSC_VIEWER_BINARY_MATLAB && !vbinary->matlabheaderwritten) {
    if (vbinary->fdes_info) {
      FILE *info = vbinary->fdes_info;
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, info, "#--- begin code written by PetscViewerBinary for MATLAB format ---#\n"));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, info, "#$$ Set.filename = '%s';\n", vbinary->filename));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, info, "#$$ fd = PetscOpenFile(Set.filename);\n"));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, info, "#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n"));
    }
    vbinary->matlabheaderwritten = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerFileClose_BinaryMPIIO(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)v->data;

  PetscFunctionBegin;
  if (vbinary->mfdes != MPI_FILE_NULL) PetscCallMPI(MPI_File_close(&vbinary->mfdes));
  if (vbinary->mfsub != MPI_FILE_NULL) PetscCallMPI(MPI_File_close(&vbinary->mfsub));
  vbinary->moff = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode PetscViewerFileClose_BinarySTDIO(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)v->data;

  PetscFunctionBegin;
  if (vbinary->fdes != -1) {
    PetscCall(PetscBinaryClose(vbinary->fdes));
    vbinary->fdes = -1;
    if (vbinary->storecompressed) {
      char        cmd[8 + PETSC_MAX_PATH_LEN], out[64 + PETSC_MAX_PATH_LEN] = "";
      const char *gzfilename = vbinary->ogzfilename ? vbinary->ogzfilename : vbinary->filename;
      /* compress the file */
      PetscCall(PetscStrncpy(cmd, "gzip -f ", sizeof(cmd)));
      PetscCall(PetscStrlcat(cmd, gzfilename, sizeof(cmd)));
#if defined(PETSC_HAVE_POPEN)
      {
        FILE *fp;
        PetscCall(PetscPOpen(PETSC_COMM_SELF, NULL, cmd, "r", &fp));
        PetscCheck(!fgets(out, (int)(sizeof(out) - 1), fp), PETSC_COMM_SELF, PETSC_ERR_LIB, "Error from command %s\n%s", cmd, out);
        PetscCall(PetscPClose(PETSC_COMM_SELF, fp));
      }
#endif
    }
  }
  PetscCall(PetscFree(vbinary->ogzfilename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileClose_BinaryInfo(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)v->data;

  PetscFunctionBegin;
  if (v->format == PETSC_VIEWER_BINARY_MATLAB && vbinary->matlabheaderwritten) {
    if (vbinary->fdes_info) {
      FILE *info = vbinary->fdes_info;
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, info, "#--- begin code written by PetscViewerBinary for MATLAB format ---#\n"));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, info, "#$$ close(fd);\n"));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, info, "#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n"));
    }
  }
  if (vbinary->fdes_info) {
    FILE *info         = vbinary->fdes_info;
    vbinary->fdes_info = NULL;
    PetscCheck(!fclose(info), PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileClose_Binary(PetscViewer v)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPIIO)
  PetscCall(PetscViewerFileClose_BinaryMPIIO(v));
#endif
  PetscCall(PetscViewerFileClose_BinarySTDIO(v));
  PetscCall(PetscViewerFileClose_BinaryInfo(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerDestroy_Binary(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)v->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerFileClose_Binary(v));
  PetscCall(PetscFree(vbinary->filename));
  PetscCall(PetscFree(vbinary));
  PetscCall(PetscViewerBinaryClearFunctionList(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerBinaryOpen - Opens a file for binary input/output.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  name - name of file
-  mode - open mode of file
.vb
    FILE_MODE_WRITE - create new file for binary output
    FILE_MODE_READ - open existing file for binary input
    FILE_MODE_APPEND - open existing file for binary output
.ve

   Output Parameter:
.  viewer - PetscViewer for binary input/output to use with the specified file

    Options Database Keys:
+    -viewer_binary_filename <name> - name of file to use
.    -viewer_binary_skip_info - true to skip opening an info file
.    -viewer_binary_skip_options - true to not use options database while creating viewer
.    -viewer_binary_skip_header - true to skip output object headers to the file
-    -viewer_binary_mpiio - true to use MPI-IO for input and output to the file (more scalable for large problems)

   Level: beginner

   Note:
   This `PetscViewer` should be destroyed with `PetscViewerDestroy()`.

    For reading files, the filename may begin with ftp:// or http:// and/or
    end with .gz; in this case file is brought over and uncompressed.

    For creating files, if the file name ends with .gz it is automatically
    compressed when closed.

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerASCIIOpen()`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `VecView()`, `MatView()`, `VecLoad()`, `MatLoad()`, `PetscViewerBinaryGetDescriptor()`,
          `PetscViewerBinaryGetInfoPointer()`, `PetscFileMode`, `PetscViewer`, `PetscViewerBinaryRead()`, `PetscViewerBinarySetUseMPIIO()`,
          `PetscViewerBinaryGetUseMPIIO()`, `PetscViewerBinaryGetMPIIOOffset()`
@*/
PetscErrorCode PetscViewerBinaryOpen(MPI_Comm comm, const char name[], PetscFileMode mode, PetscViewer *viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerCreate(comm, viewer));
  PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERBINARY));
  PetscCall(PetscViewerFileSetMode(*viewer, mode));
  PetscCall(PetscViewerFileSetName(*viewer, name));
  PetscCall(PetscViewerSetFromOptions(*viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerBinaryWriteReadMPIIO(PetscViewer viewer, void *data, PetscInt num, PetscInt *count, PetscDataType dtype, PetscBool write)
{
  MPI_Comm            comm    = PetscObjectComm((PetscObject)viewer);
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;
  MPI_File            mfdes   = vbinary->mfdes;
  MPI_Datatype        mdtype;
  PetscMPIInt         rank, cnt;
  MPI_Status          status;
  MPI_Aint            ul, dsize;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscMPIIntCast(num, &cnt));
  PetscCall(PetscDataTypeToMPIDataType(dtype, &mdtype));
  if (write) {
    if (rank == 0) PetscCall(MPIU_File_write_at(mfdes, vbinary->moff, data, cnt, mdtype, &status));
  } else {
    if (rank == 0) {
      PetscCall(MPIU_File_read_at(mfdes, vbinary->moff, data, cnt, mdtype, &status));
      if (cnt > 0) PetscCallMPI(MPI_Get_count(&status, mdtype, &cnt));
    }
    PetscCallMPI(MPI_Bcast(&cnt, 1, MPI_INT, 0, comm));
    PetscCallMPI(MPI_Bcast(data, cnt, mdtype, 0, comm));
  }
  PetscCallMPI(MPI_Type_get_extent(mdtype, &ul, &dsize));
  vbinary->moff += dsize * cnt;
  if (count) *count = cnt;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*@C
   PetscViewerBinaryRead - Reads from a binary file, all processors get the same result

   Collective

   Input Parameters:
+  viewer - the binary viewer
.  data - location of the data to be written
.  num - number of items of data to read
-  dtype - type of data to read

   Output Parameter:
.  count - number of items of data actually read, or `NULL`.

   Level: beginner

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerASCIIOpen()`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `VecView()`, `MatView()`, `VecLoad()`, `MatLoad()`, `PetscViewerBinaryGetDescriptor()`,
          `PetscViewerBinaryGetInfoPointer()`, `PetscFileMode`, `PetscViewer`, `PetscViewerBinaryRead()`
@*/
PetscErrorCode PetscViewerBinaryRead(PetscViewer viewer, void *data, PetscInt num, PetscInt *count, PetscDataType dtype)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERBINARY);
  PetscValidLogicalCollectiveInt(viewer, num, 3);
  PetscCall(PetscViewerSetUp(viewer));
  vbinary = (PetscViewer_Binary *)viewer->data;
#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio) {
    PetscCall(PetscViewerBinaryWriteReadMPIIO(viewer, data, num, count, dtype, PETSC_FALSE));
  } else {
#endif
    PetscCall(PetscBinarySynchronizedRead(PetscObjectComm((PetscObject)viewer), vbinary->fdes, data, num, count, dtype));
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerBinaryWrite - writes to a binary file, only from the first MPI rank

   Collective

   Input Parameters:
+  viewer - the binary viewer
.  data - location of data
.  count - number of items of data to write
-  dtype - type of data to write

   Level: beginner

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerASCIIOpen()`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `VecView()`, `MatView()`, `VecLoad()`, `MatLoad()`, `PetscViewerBinaryGetDescriptor()`, `PetscDataType`
          `PetscViewerBinaryGetInfoPointer()`, `PetscFileMode`, `PetscViewer`, `PetscViewerBinaryRead()`
@*/
PetscErrorCode PetscViewerBinaryWrite(PetscViewer viewer, const void *data, PetscInt count, PetscDataType dtype)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERBINARY);
  PetscValidLogicalCollectiveInt(viewer, count, 3);
  PetscCall(PetscViewerSetUp(viewer));
  vbinary = (PetscViewer_Binary *)viewer->data;
#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio) {
    PetscCall(PetscViewerBinaryWriteReadMPIIO(viewer, (void *)data, count, NULL, dtype, PETSC_TRUE));
  } else {
#endif
    PetscCall(PetscBinarySynchronizedWrite(PetscObjectComm((PetscObject)viewer), vbinary->fdes, data, count, dtype));
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerBinaryWriteReadAll(PetscViewer viewer, PetscBool write, void *data, PetscInt count, PetscInt start, PetscInt total, PetscDataType dtype)
{
  MPI_Comm              comm = PetscObjectComm((PetscObject)viewer);
  PetscMPIInt           size, rank;
  MPI_Datatype          mdtype;
  PETSC_UNUSED MPI_Aint lb;
  MPI_Aint              dsize;
  PetscBool             useMPIIO;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERBINARY);
  PetscValidLogicalCollectiveBool(viewer, ((start >= 0) || (start == PETSC_DETERMINE)), 5);
  PetscValidLogicalCollectiveBool(viewer, ((total >= 0) || (total == PETSC_DETERMINE)), 6);
  PetscValidLogicalCollectiveInt(viewer, total, 6);
  PetscCall(PetscViewerSetUp(viewer));

  PetscCall(PetscDataTypeToMPIDataType(dtype, &mdtype));
  PetscCallMPI(MPI_Type_get_extent(mdtype, &lb, &dsize));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCall(PetscViewerBinaryGetUseMPIIO(viewer, &useMPIIO));
#if defined(PETSC_HAVE_MPIIO)
  if (useMPIIO) {
    MPI_File    mfdes;
    MPI_Offset  off;
    PetscMPIInt cnt;

    if (start == PETSC_DETERMINE) {
      PetscCallMPI(MPI_Scan(&count, &start, 1, MPIU_INT, MPI_SUM, comm));
      start -= count;
    }
    if (total == PETSC_DETERMINE) {
      total = start + count;
      PetscCallMPI(MPI_Bcast(&total, 1, MPIU_INT, size - 1, comm));
    }
    PetscCall(PetscMPIIntCast(count, &cnt));
    PetscCall(PetscViewerBinaryGetMPIIODescriptor(viewer, &mfdes));
    PetscCall(PetscViewerBinaryGetMPIIOOffset(viewer, &off));
    off += (MPI_Offset)(start * dsize);
    if (write) {
      PetscCall(MPIU_File_write_at_all(mfdes, off, data, cnt, mdtype, MPI_STATUS_IGNORE));
    } else {
      PetscCall(MPIU_File_read_at_all(mfdes, off, data, cnt, mdtype, MPI_STATUS_IGNORE));
    }
    off = (MPI_Offset)(total * dsize);
    PetscCall(PetscViewerBinaryAddMPIIOOffset(viewer, off));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
  {
    int         fdes;
    char       *workbuf = NULL;
    PetscInt    tcount = rank == 0 ? 0 : count, maxcount = 0, message_count, flowcontrolcount;
    PetscMPIInt tag, cnt, maxcnt, scnt = 0, rcnt = 0, j;
    MPI_Status  status;

    PetscCall(PetscCommGetNewTag(comm, &tag));
    PetscCallMPI(MPI_Reduce(&tcount, &maxcount, 1, MPIU_INT, MPI_MAX, 0, comm));
    PetscCall(PetscMPIIntCast(maxcount, &maxcnt));
    PetscCall(PetscMPIIntCast(count, &cnt));

    PetscCall(PetscViewerBinaryGetDescriptor(viewer, &fdes));
    PetscCall(PetscViewerFlowControlStart(viewer, &message_count, &flowcontrolcount));
    if (rank == 0) {
      PetscCall(PetscMalloc(maxcnt * dsize, &workbuf));
      if (write) {
        PetscCall(PetscBinaryWrite(fdes, data, cnt, dtype));
      } else {
        PetscCall(PetscBinaryRead(fdes, data, cnt, NULL, dtype));
      }
      for (j = 1; j < size; j++) {
        PetscCall(PetscViewerFlowControlStepMain(viewer, j, &message_count, flowcontrolcount));
        if (write) {
          PetscCallMPI(MPI_Recv(workbuf, maxcnt, mdtype, j, tag, comm, &status));
          PetscCallMPI(MPI_Get_count(&status, mdtype, &rcnt));
          PetscCall(PetscBinaryWrite(fdes, workbuf, rcnt, dtype));
        } else {
          PetscCallMPI(MPI_Recv(&scnt, 1, MPI_INT, j, tag, comm, MPI_STATUS_IGNORE));
          PetscCall(PetscBinaryRead(fdes, workbuf, scnt, NULL, dtype));
          PetscCallMPI(MPI_Send(workbuf, scnt, mdtype, j, tag, comm));
        }
      }
      PetscCall(PetscFree(workbuf));
      PetscCall(PetscViewerFlowControlEndMain(viewer, &message_count));
    } else {
      PetscCall(PetscViewerFlowControlStepWorker(viewer, rank, &message_count));
      if (write) {
        PetscCallMPI(MPI_Send(data, cnt, mdtype, 0, tag, comm));
      } else {
        PetscCallMPI(MPI_Send(&cnt, 1, MPI_INT, 0, tag, comm));
        PetscCallMPI(MPI_Recv(data, cnt, mdtype, 0, tag, comm, MPI_STATUS_IGNORE));
      }
      PetscCall(PetscViewerFlowControlEndWorker(viewer, &message_count));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerBinaryReadAll - reads from a binary file from all MPI ranks, each rank receives its own portion of the data

   Collective

   Input Parameters:
+  viewer - the binary viewer
.  data - location of data
.  count - local number of items of data to read
.  start - local start, can be `PETSC_DETERMINE`
.  total - global number of items of data to read, can be `PETSC_DETERMINE`
-  dtype - type of data to read

   Level: advanced

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinarySetUseMPIIO()`, `PetscViewerBinaryRead()`, `PetscViewerBinaryWriteAll()`
@*/
PetscErrorCode PetscViewerBinaryReadAll(PetscViewer viewer, void *data, PetscInt count, PetscInt start, PetscInt total, PetscDataType dtype)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryWriteReadAll(viewer, PETSC_FALSE, data, count, start, total, dtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerBinaryWriteAll - writes to a binary file from all MPI ranks, each rank writes its own portion of the data

   Collective

   Input Parameters:
+  viewer - the binary viewer
.  data - location of data
.  count - local number of items of data to write
.  start - local start, can be `PETSC_DETERMINE`
.  total - global number of items of data to write, can be `PETSC_DETERMINE`
-  dtype - type of data to write

   Level: advanced

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerBinaryOpen()`, `PetscViewerBinarySetUseMPIIO()`, `PetscViewerBinaryWriteAll()`, `PetscViewerBinaryReadAll()`
@*/
PetscErrorCode PetscViewerBinaryWriteAll(PetscViewer viewer, const void *data, PetscInt count, PetscInt start, PetscInt total, PetscDataType dtype)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryWriteReadAll(viewer, PETSC_TRUE, (void *)data, count, start, total, dtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerBinaryWriteStringArray - writes to a binary file, only from the first MPI rank, an array of strings

   Collective

   Input Parameters:
+  viewer - the binary viewer
-  data - location of the array of strings

   Level: intermediate

    Note:
    The array of strings must be `NULL` terminated

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerASCIIOpen()`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `VecView()`, `MatView()`, `VecLoad()`, `MatLoad()`, `PetscViewerBinaryGetDescriptor()`,
          `PetscViewerBinaryGetInfoPointer()`, `PetscFileMode`, `PetscViewer`, `PetscViewerBinaryRead()`
@*/
PetscErrorCode PetscViewerBinaryWriteStringArray(PetscViewer viewer, const char *const *data)
{
  PetscInt i, n = 0, *sizes;
  size_t   len;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));
  /* count number of strings */
  while (data[n++])
    ;
  n--;
  PetscCall(PetscMalloc1(n + 1, &sizes));
  sizes[0] = n;
  for (i = 0; i < n; i++) {
    PetscCall(PetscStrlen(data[i], &len));
    sizes[i + 1] = (PetscInt)len + 1; /* size includes space for the null terminator */
  }
  PetscCall(PetscViewerBinaryWrite(viewer, sizes, n + 1, PETSC_INT));
  for (i = 0; i < n; i++) PetscCall(PetscViewerBinaryWrite(viewer, (void *)data[i], sizes[i + 1], PETSC_CHAR));
  PetscCall(PetscFree(sizes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerBinaryReadStringArray - reads a binary file an array of strings to all MPI ranks

   Collective

   Input Parameter:
.  viewer - the binary viewer

   Output Parameter:
.  data - location of the array of strings

   Level: intermediate

    Note:
    The array of strings must `NULL` terminated

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PetscViewerASCIIOpen()`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `VecView()`, `MatView()`, `VecLoad()`, `MatLoad()`, `PetscViewerBinaryGetDescriptor()`,
          `PetscViewerBinaryGetInfoPointer()`, `PetscFileMode`, `PetscViewer`, `PetscViewerBinaryRead()`
@*/
PetscErrorCode PetscViewerBinaryReadStringArray(PetscViewer viewer, char ***data)
{
  PetscInt i, n, *sizes, N = 0;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));
  /* count number of strings */
  PetscCall(PetscViewerBinaryRead(viewer, &n, 1, NULL, PETSC_INT));
  PetscCall(PetscMalloc1(n, &sizes));
  PetscCall(PetscViewerBinaryRead(viewer, sizes, n, NULL, PETSC_INT));
  for (i = 0; i < n; i++) N += sizes[i];
  PetscCall(PetscMalloc((n + 1) * sizeof(char *) + N * sizeof(char), data));
  (*data)[0] = (char *)((*data) + n + 1);
  for (i = 1; i < n; i++) (*data)[i] = (*data)[i - 1] + sizes[i - 1];
  PetscCall(PetscViewerBinaryRead(viewer, (*data)[0], N, NULL, PETSC_CHAR));
  (*data)[n] = NULL;
  PetscCall(PetscFree(sizes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     PetscViewerFileSetMode - Sets the open mode of file

    Logically Collective

  Input Parameters:
+  viewer - the `PetscViewer`; must be a a `PETSCVIEWERBINARY`, `PETSCVIEWERMATLAB`, `PETSCVIEWERHDF5`, or `PETSCVIEWERASCII`  `PetscViewer`
-  mode - open mode of file
.vb
    FILE_MODE_WRITE - create new file for output
    FILE_MODE_READ - open existing file for input
    FILE_MODE_APPEND - open existing file for output
.ve

  Level: advanced

.seealso: [](sec_viewers), `PetscViewerFileSetMode()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinaryOpen()`
@*/
PetscErrorCode PetscViewerFileSetMode(PetscViewer viewer, PetscFileMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(viewer, mode, 2);
  PetscCheck(mode != FILE_MODE_UNDEFINED, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Cannot set FILE_MODE_UNDEFINED");
  PetscCheck(mode >= FILE_MODE_UNDEFINED && mode <= FILE_MODE_APPEND_UPDATE, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_OUTOFRANGE, "Invalid file mode %d", (int)mode);
  PetscTryMethod(viewer, "PetscViewerFileSetMode_C", (PetscViewer, PetscFileMode), (viewer, mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetMode_Binary(PetscViewer viewer, PetscFileMode mode)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  PetscCheck(!viewer->setupcalled || vbinary->filemode == mode, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ORDER, "Cannot change mode to %s after setup", PetscFileModes[mode]);
  vbinary->filemode = mode;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     PetscViewerFileGetMode - Gets the open mode of file

    Not Collective

  Input Parameter:
.  viewer - the `PetscViewer`; must be a `PETSCVIEWERBINARY`, `PETSCVIEWERMATLAB`, `PETSCVIEWERHDF5`, or `PETSCVIEWERASCII`  `PetscViewer`

  Output Parameter:
.  mode - open mode of file
.vb
    FILE_MODE_WRITE - create new file for binary output
    FILE_MODE_READ - open existing file for binary input
    FILE_MODE_APPEND - open existing file for binary output
.ve

  Level: advanced

.seealso: [](sec_viewers), `PetscViewerFileSetMode()`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerBinaryOpen()`
@*/
PetscErrorCode PetscViewerFileGetMode(PetscViewer viewer, PetscFileMode *mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidPointer(mode, 2);
  PetscUseMethod(viewer, "PetscViewerFileGetMode_C", (PetscViewer, PetscFileMode *), (viewer, mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetMode_Binary(PetscViewer viewer, PetscFileMode *mode)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  *mode = vbinary->filemode;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetName_Binary(PetscViewer viewer, const char name[])
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  if (viewer->setupcalled && vbinary->filename) {
    /* gzip can be run after the file with the previous filename has been closed */
    PetscCall(PetscFree(vbinary->ogzfilename));
    PetscCall(PetscStrallocpy(vbinary->filename, &vbinary->ogzfilename));
  }
  PetscCall(PetscFree(vbinary->filename));
  PetscCall(PetscStrallocpy(name, &vbinary->filename));
  viewer->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetName_Binary(PetscViewer viewer, const char **name)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;

  PetscFunctionBegin;
  *name = vbinary->filename;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerFileSetUp_BinaryMPIIO(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;
  int                 amode;

  PetscFunctionBegin;
  vbinary->storecompressed = PETSC_FALSE;

  vbinary->moff = 0;
  switch (vbinary->filemode) {
  case FILE_MODE_READ:
    amode = MPI_MODE_RDONLY;
    break;
  case FILE_MODE_WRITE:
    amode = MPI_MODE_WRONLY | MPI_MODE_CREATE;
    break;
  case FILE_MODE_APPEND:
    amode = MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_APPEND;
    break;
  case FILE_MODE_UNDEFINED:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerSetUp()");
  default:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Unsupported file mode %s", PetscFileModes[vbinary->filemode]);
  }
  PetscCallMPI(MPI_File_open(PetscObjectComm((PetscObject)viewer), vbinary->filename, amode, MPI_INFO_NULL, &vbinary->mfdes));
  /*
      The MPI standard does not have MPI_MODE_TRUNCATE. We emulate this behavior by setting the file size to zero.
  */
  if (vbinary->filemode == FILE_MODE_WRITE) PetscCallMPI(MPI_File_set_size(vbinary->mfdes, 0));
  /*
      Initially, all processes view the file as a linear byte stream. Therefore, for files opened with MPI_MODE_APPEND,
      MPI_File_get_position[_shared](fh, &offset) returns the absolute byte position at the end of file.
      Otherwise, we would need to call MPI_File_get_byte_offset(fh, offset, &byte_offset) to convert
      the offset in etype units to an absolute byte position.
   */
  if (vbinary->filemode == FILE_MODE_APPEND) PetscCallMPI(MPI_File_get_position(vbinary->mfdes, &vbinary->moff));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode PetscViewerFileSetUp_BinarySTDIO(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;
  const char         *fname;
  char                bname[PETSC_MAX_PATH_LEN], *gz = NULL;
  PetscBool           found;
  PetscMPIInt         rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));

  /* if file name ends in .gz strip that off and note user wants file compressed */
  vbinary->storecompressed = PETSC_FALSE;
  if (vbinary->filemode == FILE_MODE_WRITE) {
    PetscCall(PetscStrstr(vbinary->filename, ".gz", &gz));
    if (gz && gz[3] == 0) {
      *gz                      = 0;
      vbinary->storecompressed = PETSC_TRUE;
    }
  }
#if !defined(PETSC_HAVE_POPEN)
  PetscCheck(!vbinary->storecompressed, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP_SYS, "Cannot run gzip on this machine");
#endif

  fname = vbinary->filename;
  if (vbinary->filemode == FILE_MODE_READ) { /* possibly get the file from remote site or compressed file */
    PetscCall(PetscFileRetrieve(PetscObjectComm((PetscObject)viewer), fname, bname, PETSC_MAX_PATH_LEN, &found));
    PetscCheck(found, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_OPEN, "Cannot locate file: %s", fname);
    fname = bname;
  }

  vbinary->fdes = -1;
  if (rank == 0) { /* only first processor opens file*/
    PetscFileMode mode = vbinary->filemode;
    if (mode == FILE_MODE_APPEND) {
      /* check if asked to append to a non-existing file */
      PetscCall(PetscTestFile(fname, '\0', &found));
      if (!found) mode = FILE_MODE_WRITE;
    }
    PetscCall(PetscBinaryOpen(fname, mode, &vbinary->fdes));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetUp_BinaryInfo(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;
  PetscMPIInt         rank;
  PetscBool           found;

  PetscFunctionBegin;
  vbinary->fdes_info = NULL;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
  if (!vbinary->skipinfo && (vbinary->filemode == FILE_MODE_READ || rank == 0)) {
    char infoname[PETSC_MAX_PATH_LEN], iname[PETSC_MAX_PATH_LEN], *gz;

    PetscCall(PetscStrncpy(infoname, vbinary->filename, sizeof(infoname)));
    /* remove .gz if it ends file name */
    PetscCall(PetscStrstr(infoname, ".gz", &gz));
    if (gz && gz[3] == 0) *gz = 0;

    PetscCall(PetscStrlcat(infoname, ".info", sizeof(infoname)));
    if (vbinary->filemode == FILE_MODE_READ) {
      PetscCall(PetscFixFilename(infoname, iname));
      PetscCall(PetscFileRetrieve(PetscObjectComm((PetscObject)viewer), iname, infoname, PETSC_MAX_PATH_LEN, &found));
      if (found) PetscCall(PetscOptionsInsertFile(PetscObjectComm((PetscObject)viewer), ((PetscObject)viewer)->options, infoname, PETSC_FALSE));
    } else if (rank == 0) { /* write or append */
      const char *omode  = (vbinary->filemode == FILE_MODE_APPEND) ? "a" : "w";
      vbinary->fdes_info = fopen(infoname, omode);
      PetscCheck(vbinary->fdes_info, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open .info file %s for writing", infoname);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerSetUp_Binary(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)viewer->data;
  PetscBool           usempiio;

  PetscFunctionBegin;
  if (!vbinary->setfromoptionscalled) PetscCall(PetscViewerSetFromOptions(viewer));
  PetscCheck(vbinary->filename, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Must call PetscViewerFileSetName()");
  PetscCheck(vbinary->filemode != (PetscFileMode)-1, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode()");
  PetscCall(PetscViewerFileClose_Binary(viewer));

  PetscCall(PetscViewerBinaryGetUseMPIIO(viewer, &usempiio));
  if (usempiio) {
#if defined(PETSC_HAVE_MPIIO)
    PetscCall(PetscViewerFileSetUp_BinaryMPIIO(viewer));
#endif
  } else {
    PetscCall(PetscViewerFileSetUp_BinarySTDIO(viewer));
  }
  PetscCall(PetscViewerFileSetUp_BinaryInfo(viewer));

  PetscCall(PetscLogObjectState((PetscObject)viewer, "File: %s", vbinary->filename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerView_Binary(PetscViewer v, PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary *)v->data;
  const char         *fname   = vbinary->filename ? vbinary->filename : "not yet set";
  const char         *fmode   = vbinary->filemode != (PetscFileMode)-1 ? PetscFileModes[vbinary->filemode] : "not yet set";
  PetscBool           usempiio;

  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryGetUseMPIIO(v, &usempiio));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Filename: %s\n", fname));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Mode: %s (%s)\n", fmode, usempiio ? "mpiio" : "stdio"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerSetFromOptions_Binary(PetscViewer viewer, PetscOptionItems *PetscOptionsObject)
{
  PetscViewer_Binary *binary = (PetscViewer_Binary *)viewer->data;
  char                defaultname[PETSC_MAX_PATH_LEN];
  PetscBool           flg;

  PetscFunctionBegin;
  if (viewer->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscOptionsHeadBegin(PetscOptionsObject, "Binary PetscViewer Options");
  PetscCall(PetscSNPrintf(defaultname, PETSC_MAX_PATH_LEN - 1, "binaryoutput"));
  PetscCall(PetscOptionsString("-viewer_binary_filename", "Specify filename", "PetscViewerFileSetName", defaultname, defaultname, sizeof(defaultname), &flg));
  if (flg) PetscCall(PetscViewerFileSetName_Binary(viewer, defaultname));
  PetscCall(PetscOptionsBool("-viewer_binary_skip_info", "Skip writing/reading .info file", "PetscViewerBinarySetSkipInfo", binary->skipinfo, &binary->skipinfo, NULL));
  PetscCall(PetscOptionsBool("-viewer_binary_skip_options", "Skip parsing Vec/Mat load options", "PetscViewerBinarySetSkipOptions", binary->skipoptions, &binary->skipoptions, NULL));
  PetscCall(PetscOptionsBool("-viewer_binary_skip_header", "Skip writing/reading header information", "PetscViewerBinarySetSkipHeader", binary->skipheader, &binary->skipheader, NULL));
#if defined(PETSC_HAVE_MPIIO)
  PetscCall(PetscOptionsBool("-viewer_binary_mpiio", "Use MPI-IO functionality to write/read binary file", "PetscViewerBinarySetUseMPIIO", binary->usempiio, &binary->usempiio, NULL));
#else
  PetscCall(PetscOptionsBool("-viewer_binary_mpiio", "Use MPI-IO functionality to write/read binary file (NOT AVAILABLE)", "PetscViewerBinarySetUseMPIIO", PETSC_FALSE, NULL, NULL));
#endif
  PetscOptionsHeadEnd();
  binary->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCVIEWERBINARY - A viewer that saves to binary files

  Level: beginner

.seealso: [](sec_viewers), `PetscViewerBinaryOpen()`, `PETSC_VIEWER_STDOUT_()`, `PETSC_VIEWER_STDOUT_SELF`, `PETSC_VIEWER_STDOUT_WORLD`, `PetscViewerCreate()`, `PetscViewerASCIIOpen()`,
          `PetscViewerMatlabOpen()`, `VecView()`, `DMView()`, `PetscViewerMatlabPutArray()`, `PETSCVIEWERASCII`, `PETSCVIEWERMATLAB`, `PETSCVIEWERDRAW`, `PETSCVIEWERSOCKET`
          `PetscViewerFileSetName()`, `PetscViewerFileSetMode()`, `PetscViewerFormat`, `PetscViewerType`, `PetscViewerSetType()`,
          `PetscViewerBinaryGetUseMPIIO()`, `PetscViewerBinarySetUseMPIIO()`
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_Binary(PetscViewer v)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscCall(PetscNew(&vbinary));
  v->data = (void *)vbinary;

  v->ops->setfromoptions   = PetscViewerSetFromOptions_Binary;
  v->ops->destroy          = PetscViewerDestroy_Binary;
  v->ops->view             = PetscViewerView_Binary;
  v->ops->setup            = PetscViewerSetUp_Binary;
  v->ops->flush            = NULL; /* Should we support Flush() ? */
  v->ops->getsubviewer     = PetscViewerGetSubViewer_Binary;
  v->ops->restoresubviewer = PetscViewerRestoreSubViewer_Binary;
  v->ops->read             = PetscViewerBinaryRead;

  vbinary->fdes = -1;
#if defined(PETSC_HAVE_MPIIO)
  vbinary->usempiio = PETSC_FALSE;
  vbinary->mfdes    = MPI_FILE_NULL;
  vbinary->mfsub    = MPI_FILE_NULL;
#endif
  vbinary->filename        = NULL;
  vbinary->filemode        = FILE_MODE_UNDEFINED;
  vbinary->fdes_info       = NULL;
  vbinary->skipinfo        = PETSC_FALSE;
  vbinary->skipoptions     = PETSC_TRUE;
  vbinary->skipheader      = PETSC_FALSE;
  vbinary->storecompressed = PETSC_FALSE;
  vbinary->ogzfilename     = NULL;
  vbinary->flowcontrol     = 256; /* seems a good number for Cray XT-5 */

  vbinary->setfromoptionscalled = PETSC_FALSE;

  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetFlowControl_C", PetscViewerBinaryGetFlowControl_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetFlowControl_C", PetscViewerBinarySetFlowControl_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetSkipHeader_C", PetscViewerBinaryGetSkipHeader_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetSkipHeader_C", PetscViewerBinarySetSkipHeader_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetSkipOptions_C", PetscViewerBinaryGetSkipOptions_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetSkipOptions_C", PetscViewerBinarySetSkipOptions_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetSkipInfo_C", PetscViewerBinaryGetSkipInfo_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetSkipInfo_C", PetscViewerBinarySetSkipInfo_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetInfoPointer_C", PetscViewerBinaryGetInfoPointer_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetName_C", PetscViewerFileGetName_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetName_C", PetscViewerFileSetName_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetMode_C", PetscViewerFileGetMode_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetMode_C", PetscViewerFileSetMode_Binary));
#if defined(PETSC_HAVE_MPIIO)
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinaryGetUseMPIIO_C", PetscViewerBinaryGetUseMPIIO_Binary));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerBinarySetUseMPIIO_C", PetscViewerBinarySetUseMPIIO_Binary));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    The variable Petsc_Viewer_Binary_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
PetscMPIInt Petsc_Viewer_Binary_keyval = MPI_KEYVAL_INVALID;

/*@C
     PETSC_VIEWER_BINARY_ - Creates a `PETSCVIEWERBINARY` `PetscViewer` shared by all processors
                     in a communicator.

     Collective

     Input Parameter:
.    comm - the MPI communicator to share the `PETSCVIEWERBINARY`

     Level: intermediate

   Options Database Keys:
+    -viewer_binary_filename <name> - filename in which to store the binary data, defaults to binaryoutput
.    -viewer_binary_skip_info - true means do not create .info file for this viewer
.    -viewer_binary_skip_options - true means do not use the options database for this viewer
.    -viewer_binary_skip_header - true means do not store the usual header information in the binary file
-    -viewer_binary_mpiio - true means use the file via MPI-IO, maybe faster for large files and many MPI ranks

   Environmental variable:
-   PETSC_VIEWER_BINARY_FILENAME - filename in which to store the binary data, defaults to binaryoutput

     Note:
     Unlike almost all other PETSc routines, `PETSC_VIEWER_BINARY_` does not return
     an error code.  The binary PetscViewer is usually used in the form
$       XXXView(XXX object,PETSC_VIEWER_BINARY_(comm));

.seealso: [](sec_viewers), `PETSCVIEWERBINARY`, `PETSC_VIEWER_BINARY_WORLD`, `PETSC_VIEWER_BINARY_SELF`, `PetscViewerBinaryOpen()`, `PetscViewerCreate()`,
          `PetscViewerDestroy()`
@*/
PetscViewer PETSC_VIEWER_BINARY_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscMPIInt    mpi_ierr;
  PetscBool      flg;
  PetscViewer    viewer;
  char           fname[PETSC_MAX_PATH_LEN];
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm, &ncomm, NULL);
  if (ierr) {
    ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_BINARY_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
    PetscFunctionReturn(NULL);
  }
  if (Petsc_Viewer_Binary_keyval == MPI_KEYVAL_INVALID) {
    mpi_ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &Petsc_Viewer_Binary_keyval, NULL);
    if (mpi_ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_BINARY_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
      PetscFunctionReturn(NULL);
    }
  }
  mpi_ierr = MPI_Comm_get_attr(ncomm, Petsc_Viewer_Binary_keyval, (void **)&viewer, (int *)&flg);
  if (mpi_ierr) {
    ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_BINARY_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
    PetscFunctionReturn(NULL);
  }
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscOptionsGetenv(ncomm, "PETSC_VIEWER_BINARY_FILENAME", fname, PETSC_MAX_PATH_LEN, &flg);
    if (ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_BINARY_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
      PetscFunctionReturn(NULL);
    }
    if (!flg) {
      ierr = PetscStrncpy(fname, "binaryoutput", sizeof(fname));
      if (ierr) {
        ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_BINARY_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
        PetscFunctionReturn(NULL);
      }
    }
    ierr = PetscViewerBinaryOpen(ncomm, fname, FILE_MODE_WRITE, &viewer);
    if (ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_BINARY_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
      PetscFunctionReturn(NULL);
    }
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_BINARY_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
      PetscFunctionReturn(NULL);
    }
    mpi_ierr = MPI_Comm_set_attr(ncomm, Petsc_Viewer_Binary_keyval, (void *)viewer);
    if (mpi_ierr) {
      ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_BINARY_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_INITIAL, " ");
      PetscFunctionReturn(NULL);
    }
  }
  ierr = PetscCommDestroy(&ncomm);
  if (ierr) {
    ierr = PetscError(PETSC_COMM_SELF, __LINE__, "PETSC_VIEWER_BINARY_", __FILE__, PETSC_ERR_PLIB, PETSC_ERROR_REPEAT, " ");
    PetscFunctionReturn(NULL);
  }
  PetscFunctionReturn(viewer);
}
