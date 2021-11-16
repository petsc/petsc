#include <petsc/private/viewerimpl.h>    /*I   "petscviewer.h"   I*/

typedef struct  {
  int           fdes;                 /* file descriptor, ignored if using MPI IO */
#if defined(PETSC_HAVE_MPIIO)
  PetscBool     usempiio;
  MPI_File      mfdes;                /* ignored unless using MPI IO */
  MPI_File      mfsub;                /* subviewer support */
  MPI_Offset    moff;
#endif
  char          *filename;            /* file name */
  PetscFileMode filemode;             /* read/write/append mode */
  FILE          *fdes_info;           /* optional file containing info on binary file*/
  PetscBool     storecompressed;      /* gzip the write binary file when closing it*/
  char          *ogzfilename;         /* gzip can be run after the filename has been updated */
  PetscBool     skipinfo;             /* Don't create info file for writing; don't use for reading */
  PetscBool     skipoptions;          /* don't use PETSc options database when loading */
  PetscInt      flowcontrol;          /* allow only <flowcontrol> messages outstanding at a time while doing IO */
  PetscBool     skipheader;           /* don't write header, only raw data */
  PetscBool     matlabheaderwritten;  /* if format is PETSC_VIEWER_BINARY_MATLAB has the MATLAB .info header been written yet */
  PetscBool     setfromoptionscalled;
} PetscViewer_Binary;

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerBinarySyncMPIIO(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (vbinary->filemode == FILE_MODE_READ) PetscFunctionReturn(0);
  if (vbinary->mfsub != MPI_FILE_NULL) {
    ierr = MPI_File_sync(vbinary->mfsub);CHKERRMPI(ierr);
  }
  if (vbinary->mfdes != MPI_FILE_NULL) {
    ierr = MPI_Barrier(PetscObjectComm((PetscObject)viewer));CHKERRMPI(ierr);
    ierr = MPI_File_sync(vbinary->mfdes);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode PetscViewerGetSubViewer_Binary(PetscViewer viewer,MPI_Comm comm,PetscViewer *outviewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

  /* Return subviewer in process zero */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRMPI(ierr);
  if (rank == 0) {
    PetscMPIInt flg;

    ierr = MPI_Comm_compare(PETSC_COMM_SELF,comm,&flg);CHKERRMPI(ierr);
    if (flg != MPI_IDENT && flg != MPI_CONGRUENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PetscViewerGetSubViewer() for PETSCVIEWERBINARY requires a singleton MPI_Comm");
    ierr = PetscViewerCreate(comm,outviewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(*outviewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
    ierr = PetscMemcpy((*outviewer)->data,vbinary,sizeof(PetscViewer_Binary));CHKERRQ(ierr);
    (*outviewer)->setupcalled = PETSC_TRUE;
  } else {
    *outviewer = NULL;
  }

#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio && *outviewer) {
    PetscViewer_Binary *obinary = (PetscViewer_Binary*)(*outviewer)->data;
    /* Parent viewer opens a new MPI file handle on PETSC_COMM_SELF and keeps track of it for future reuse */
    if (vbinary->mfsub == MPI_FILE_NULL) {
      int amode;
      switch (vbinary->filemode) {
      case FILE_MODE_READ:   amode = MPI_MODE_RDONLY; break;
      case FILE_MODE_WRITE:  amode = MPI_MODE_WRONLY; break;
      case FILE_MODE_APPEND: amode = MPI_MODE_WRONLY; break;
      default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported file mode %s",PetscFileModes[vbinary->filemode]);
      }
      ierr = MPI_File_open(PETSC_COMM_SELF,vbinary->filename,amode,MPI_INFO_NULL,&vbinary->mfsub);CHKERRMPI(ierr);
    }
    /* Subviewer gets the MPI file handle on PETSC_COMM_SELF */
    obinary->mfdes = vbinary->mfsub;
    obinary->mfsub = MPI_FILE_NULL;
    obinary->moff  = vbinary->moff;
  }
#endif

#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinarySyncMPIIO(viewer);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerRestoreSubViewer_Binary(PetscViewer viewer,MPI_Comm comm,PetscViewer *outviewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;
#if defined(PETSC_HAVE_MPIIO)
  MPI_Offset         moff = 0;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRMPI(ierr);
  if (rank && *outviewer) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Subviewer not obtained from viewer");

#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio && *outviewer) {
    PetscViewer_Binary *obinary = (PetscViewer_Binary*)(*outviewer)->data;
    if (obinary->mfdes != vbinary->mfsub) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Subviewer not obtained from viewer");
    if (obinary->mfsub != MPI_FILE_NULL) {ierr = MPI_File_close(&obinary->mfsub);CHKERRMPI(ierr);}
    moff = obinary->moff;
  }
#endif

  if (*outviewer) {
    PetscViewer_Binary *obinary = (PetscViewer_Binary*)(*outviewer)->data;
    if (obinary->fdes != vbinary->fdes) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Subviewer not obtained from viewer");
    ierr = PetscFree((*outviewer)->data);CHKERRQ(ierr);
    ierr = PetscHeaderDestroy(outviewer);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio) {
    PetscInt64 ioff = (PetscInt64)moff; /* We could use MPI_OFFSET datatype (requires MPI 2.2) */
    ierr = MPI_Bcast(&ioff,1,MPIU_INT64,0,PetscObjectComm((PetscObject)viewer));CHKERRMPI(ierr);
    vbinary->moff = (MPI_Offset)ioff;
  }
#endif

#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerBinarySyncMPIIO(viewer);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
/*@C
    PetscViewerBinaryGetMPIIOOffset - Gets the current global offset that should be passed to MPI_File_set_view() or MPI_File_{write|read}_at[_all]()

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   off - the current global offset

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

    Use PetscViewerBinaryAddMPIIOOffset() to increase this value after you have written a view.

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetInfoPointer(), PetscViewerBinaryGetUseMPIIO(), PetscViewerBinarySetUseMPIIO(), PetscViewerBinaryAddMPIIOOffset()
@*/
PetscErrorCode PetscViewerBinaryGetMPIIOOffset(PetscViewer viewer,MPI_Offset *off)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer,PETSC_VIEWER_CLASSID,1,PETSCVIEWERBINARY);
  PetscValidPointer(off,2);
  vbinary = (PetscViewer_Binary*)viewer->data;
  *off = vbinary->moff;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerBinaryAddMPIIOOffset - Adds to the current global offset

    Logically Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()
-   off - the addition to the global offset

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

    Use PetscViewerBinaryGetMPIIOOffset() to get the value that you should pass to MPI_File_set_view() or MPI_File_{write|read}_at[_all]()

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer(), PetscViewerBinaryGetUseMPIIO(), PetscViewerBinarySetUseMPIIO(), PetscViewerBinaryGetMPIIOOffset()
@*/
PetscErrorCode PetscViewerBinaryAddMPIIOOffset(PetscViewer viewer,MPI_Offset off)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer,PETSC_VIEWER_CLASSID,1,PETSCVIEWERBINARY);
  PetscValidLogicalCollectiveInt(viewer,(PetscInt)off,2);
  vbinary = (PetscViewer_Binary*)viewer->data;
  vbinary->moff += off;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerBinaryGetMPIIODescriptor - Extracts the MPI IO file descriptor from a PetscViewer.

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   fdes - file descriptor

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer(), PetscViewerBinaryGetUseMPIIO(), PetscViewerBinarySetUseMPIIO(), PetscViewerBinaryGetMPIIOOffset()
@*/
PetscErrorCode PetscViewerBinaryGetMPIIODescriptor(PetscViewer viewer,MPI_File *fdes)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer,PETSC_VIEWER_CLASSID,1,PETSCVIEWERBINARY);
  PetscValidPointer(fdes,2);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  vbinary = (PetscViewer_Binary*)viewer->data;
  *fdes = vbinary->mfdes;
  PetscFunctionReturn(0);
}
#endif

/*@
    PetscViewerBinarySetUseMPIIO - Sets a binary viewer to use MPI-IO for reading/writing. Must be called
        before PetscViewerFileSetName()

    Logically Collective on PetscViewer

    Input Parameters:
+   viewer - the PetscViewer; must be a binary
-   use - PETSC_TRUE means MPI-IO will be used

    Options Database:
    -viewer_binary_mpiio : Flag for using MPI-IO

    Level: advanced

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen(),
          PetscViewerBinaryGetUseMPIIO()

@*/
PetscErrorCode PetscViewerBinarySetUseMPIIO(PetscViewer viewer,PetscBool use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveBool(viewer,use,2);
  ierr = PetscTryMethod(viewer,"PetscViewerBinarySetUseMPIIO_C",(PetscViewer,PetscBool),(viewer,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerBinarySetUseMPIIO_Binary(PetscViewer viewer,PetscBool use)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscFunctionBegin;
  if (viewer->setupcalled && vbinary->usempiio != use) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ORDER,"Cannot change MPIIO to %s after setup",PetscBools[use]);
  vbinary->usempiio = use;
  PetscFunctionReturn(0);
}
#endif

/*@
    PetscViewerBinaryGetUseMPIIO - Returns PETSC_TRUE if the binary viewer uses MPI-IO.

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   use - PETSC_TRUE if MPI-IO is being used

    Options Database:
    -viewer_binary_mpiio : Flag for using MPI-IO

    Level: advanced

    Note:
    If MPI-IO is not available, this function will always return PETSC_FALSE

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetInfoPointer(), PetscViewerBinarySetUseMPIIO(), PetscViewerBinaryGetMPIIOOffset()
@*/
PetscErrorCode PetscViewerBinaryGetUseMPIIO(PetscViewer viewer,PetscBool *use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidBoolPointer(use,2);
  *use = PETSC_FALSE;
  ierr = PetscTryMethod(viewer,"PetscViewerBinaryGetUseMPIIO_C",(PetscViewer,PetscBool*),(viewer,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerBinaryGetUseMPIIO_Binary(PetscViewer viewer,PetscBool  *use)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *use = vbinary->usempiio;
  PetscFunctionReturn(0);
}
#endif

/*@
    PetscViewerBinarySetFlowControl - Sets how many messages are allowed to outstanding at the same time during parallel IO reads/writes

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()
-   fc - the number of messages, defaults to 256 if this function was not called

    Level: advanced

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer(), PetscViewerBinaryGetFlowControl()

@*/
PetscErrorCode  PetscViewerBinarySetFlowControl(PetscViewer viewer,PetscInt fc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveInt(viewer,fc,2);
  ierr = PetscTryMethod(viewer,"PetscViewerBinarySetFlowControl_C",(PetscViewer,PetscInt),(viewer,fc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinarySetFlowControl_Binary(PetscViewer viewer,PetscInt fc)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  if (fc <= 1) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_OUTOFRANGE,"Flow control count must be greater than 1, %" PetscInt_FMT " was set",fc);
  vbinary->flowcontrol = fc;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinaryGetFlowControl - Returns how many messages are allowed to outstanding at the same time during parallel IO reads/writes

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   fc - the number of messages

    Level: advanced

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer(), PetscViewerBinarySetFlowControl()

@*/
PetscErrorCode PetscViewerBinaryGetFlowControl(PetscViewer viewer,PetscInt *fc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidIntPointer(fc,2);
  ierr = PetscUseMethod(viewer,"PetscViewerBinaryGetFlowControl_C",(PetscViewer,PetscInt*),(viewer,fc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PetscViewerBinaryGetFlowControl_Binary(PetscViewer viewer,PetscInt *fc)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *fc = vbinary->flowcontrol;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerBinaryGetDescriptor - Extracts the file descriptor from a PetscViewer.

    Collective On PetscViewer

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   fdes - file descriptor

    Level: advanced

    Notes:
      For writable binary PetscViewers, the descriptor will only be valid for the
    first processor in the communicator that shares the PetscViewer. For readable
    files it will only be valid on nodes that have the file. If node 0 does not
    have the file it generates an error even if another node does have the file.

    Fortran Note:
    This routine is not supported in Fortran.

    Developer Notes:
    This must be called on all processes because Dave May changed
    the source code that this may be trigger a PetscViewerSetUp() call if it was not previously triggered.

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer()
@*/
PetscErrorCode PetscViewerBinaryGetDescriptor(PetscViewer viewer,int *fdes)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer,PETSC_VIEWER_CLASSID,1,PETSCVIEWERBINARY);
  PetscValidPointer(fdes,2);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  vbinary = (PetscViewer_Binary*)viewer->data;
  *fdes = vbinary->fdes;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinarySkipInfo - Binary file will not have .info file created with it

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerCreate()

    Options Database Key:
.   -viewer_binary_skip_info - true indicates do not generate .info file

    Level: advanced

    Notes:
    This must be called after PetscViewerSetType(). If you use PetscViewerBinaryOpen() then
    you can only skip the info file with the -viewer_binary_skip_info flag. To use the function you must open the
    viewer with PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinarySkipInfo().

    The .info contains meta information about the data in the binary file, for example the block size if it was
    set for a vector or matrix.

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySetSkipOptions(),
          PetscViewerBinaryGetSkipOptions(), PetscViewerBinaryGetSkipInfo()
@*/
PetscErrorCode PetscViewerBinarySkipInfo(PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinarySetSkipInfo(viewer,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinarySetSkipInfo - Binary file will not have .info file created with it

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from PetscViewerCreate()
-   skip - PETSC_TRUE implies the .info file will not be generated

    Options Database Key:
.   -viewer_binary_skip_info - true indicates do not generate .info file

    Level: advanced

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySetSkipOptions(),
          PetscViewerBinaryGetSkipOptions(), PetscViewerBinaryGetSkipInfo()
@*/
PetscErrorCode PetscViewerBinarySetSkipInfo(PetscViewer viewer,PetscBool skip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveBool(viewer,skip,2);
  ierr = PetscTryMethod(viewer,"PetscViewerBinarySetSkipInfo_C",(PetscViewer,PetscBool),(viewer,skip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinarySetSkipInfo_Binary(PetscViewer viewer,PetscBool skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  vbinary->skipinfo = skip;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinaryGetSkipInfo - check if viewer wrote a .info file

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   skip - PETSC_TRUE implies the .info file was not generated

    Level: advanced

    Notes:
    This must be called after PetscViewerSetType()

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySkipInfo(),
          PetscViewerBinarySetSkipOptions(), PetscViewerBinarySetSkipInfo()
@*/
PetscErrorCode PetscViewerBinaryGetSkipInfo(PetscViewer viewer,PetscBool *skip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidBoolPointer(skip,2);
  ierr = PetscUseMethod(viewer,"PetscViewerBinaryGetSkipInfo_C",(PetscViewer,PetscBool*),(viewer,skip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinaryGetSkipInfo_Binary(PetscViewer viewer,PetscBool *skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *skip  = vbinary->skipinfo;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinarySetSkipOptions - do not use the PETSc options database when loading objects

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()
-   skip - PETSC_TRUE means do not use the options from the options database

    Options Database Key:
.   -viewer_binary_skip_options - true means do not use the options from the options database

    Level: advanced

    Notes:
    This must be called after PetscViewerSetType()

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySkipInfo(),
          PetscViewerBinaryGetSkipOptions()
@*/
PetscErrorCode PetscViewerBinarySetSkipOptions(PetscViewer viewer,PetscBool skip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveBool(viewer,skip,2);
  ierr = PetscTryMethod(viewer,"PetscViewerBinarySetSkipOptions_C",(PetscViewer,PetscBool),(viewer,skip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinarySetSkipOptions_Binary(PetscViewer viewer,PetscBool skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  vbinary->skipoptions = skip;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinaryGetSkipOptions - checks if viewer uses the PETSc options database when loading objects

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   skip - PETSC_TRUE means do not use

    Level: advanced

    Notes:
    This must be called after PetscViewerSetType()

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySkipInfo(),
          PetscViewerBinarySetSkipOptions()
@*/
PetscErrorCode PetscViewerBinaryGetSkipOptions(PetscViewer viewer,PetscBool *skip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidBoolPointer(skip,2);
  ierr = PetscUseMethod(viewer,"PetscViewerBinaryGetSkipOptions_C",(PetscViewer,PetscBool*),(viewer,skip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinaryGetSkipOptions_Binary(PetscViewer viewer,PetscBool *skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *skip = vbinary->skipoptions;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinarySetSkipHeader - do not write a header with size information on output, just raw data

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()
-   skip - PETSC_TRUE means do not write header

    Options Database Key:
.   -viewer_binary_skip_header - PETSC_TRUE means do not write header

    Level: advanced

    Notes:
      This must be called after PetscViewerSetType()

      Is ignored on anything but a binary viewer

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySkipInfo(),
          PetscViewerBinaryGetSkipHeader()
@*/
PetscErrorCode PetscViewerBinarySetSkipHeader(PetscViewer viewer,PetscBool skip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveBool(viewer,skip,2);
  ierr = PetscTryMethod(viewer,"PetscViewerBinarySetSkipHeader_C",(PetscViewer,PetscBool),(viewer,skip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinarySetSkipHeader_Binary(PetscViewer viewer,PetscBool skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  vbinary->skipheader = skip;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinaryGetSkipHeader - checks whether to write a header with size information on output, or just raw data

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   skip - PETSC_TRUE means do not write header

    Level: advanced

    Notes:
    This must be called after PetscViewerSetType()

            Returns false for PETSCSOCKETVIEWER, you cannot skip the header for it.

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySkipInfo(),
          PetscViewerBinarySetSkipHeader()
@*/
PetscErrorCode PetscViewerBinaryGetSkipHeader(PetscViewer viewer,PetscBool  *skip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidBoolPointer(skip,2);
  ierr = PetscUseMethod(viewer,"PetscViewerBinaryGetSkipHeader_C",(PetscViewer,PetscBool*),(viewer,skip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinaryGetSkipHeader_Binary(PetscViewer viewer,PetscBool  *skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *skip = vbinary->skipheader;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerBinaryGetInfoPointer - Extracts the file pointer for the ASCII
          info file associated with a binary file.

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   file - file pointer  Always returns NULL if not a binary viewer

    Level: advanced

    Notes:
      For writable binary PetscViewers, the descriptor will only be valid for the
    first processor in the communicator that shares the PetscViewer.

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetDescriptor()
@*/
PetscErrorCode PetscViewerBinaryGetInfoPointer(PetscViewer viewer,FILE **file)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(file,2);
  *file = NULL;
  ierr = PetscTryMethod(viewer,"PetscViewerBinaryGetInfoPointer_C",(PetscViewer,FILE **),(viewer,file));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinaryGetInfoPointer_Binary(PetscViewer viewer,FILE **file)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  *file = vbinary->fdes_info;
  if (viewer->format == PETSC_VIEWER_BINARY_MATLAB && !vbinary->matlabheaderwritten) {
    if (vbinary->fdes_info) {
      FILE *info = vbinary->fdes_info;
      ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#$$ Set.filename = '%s';\n",vbinary->filename);CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#$$ fd = PetscOpenFile(Set.filename);\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n");CHKERRQ(ierr);
    }
    vbinary->matlabheaderwritten = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerFileClose_BinaryMPIIO(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (vbinary->mfdes != MPI_FILE_NULL) {
    ierr = MPI_File_close(&vbinary->mfdes);CHKERRMPI(ierr);
  }
  if (vbinary->mfsub != MPI_FILE_NULL) {
    ierr = MPI_File_close(&vbinary->mfsub);CHKERRMPI(ierr);
  }
  vbinary->moff = 0;
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode PetscViewerFileClose_BinarySTDIO(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (vbinary->fdes != -1) {
    ierr = PetscBinaryClose(vbinary->fdes);CHKERRQ(ierr);
    vbinary->fdes = -1;
    if (vbinary->storecompressed) {
      char cmd[8+PETSC_MAX_PATH_LEN],out[64+PETSC_MAX_PATH_LEN] = "";
      const char *gzfilename = vbinary->ogzfilename ? vbinary->ogzfilename : vbinary->filename;
      /* compress the file */
      ierr = PetscStrncpy(cmd,"gzip -f ",sizeof(cmd));CHKERRQ(ierr);
      ierr = PetscStrlcat(cmd,gzfilename,sizeof(cmd));CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)
      {
        FILE *fp;
        ierr = PetscPOpen(PETSC_COMM_SELF,NULL,cmd,"r",&fp);CHKERRQ(ierr);
        if (fgets(out,(int)(sizeof(out)-1),fp)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from command %s\n%s",cmd,out);
        ierr = PetscPClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
      }
#endif
    }
  }
  ierr = PetscFree(vbinary->ogzfilename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileClose_BinaryInfo(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (v->format == PETSC_VIEWER_BINARY_MATLAB && vbinary->matlabheaderwritten) {
    if (vbinary->fdes_info) {
      FILE *info = vbinary->fdes_info;
      ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#$$ close(fd);\n");CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,info,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n");CHKERRQ(ierr);
    }
  }
  if (vbinary->fdes_info) {
    FILE *info = vbinary->fdes_info;
    vbinary->fdes_info = NULL;
    if (fclose(info)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileClose_Binary(PetscViewer v)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscViewerFileClose_BinaryMPIIO(v);CHKERRQ(ierr);
#endif
  ierr = PetscViewerFileClose_BinarySTDIO(v);CHKERRQ(ierr);
  ierr = PetscViewerFileClose_BinaryInfo(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerDestroy_Binary(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerFileClose_Binary(v);CHKERRQ(ierr);
  ierr = PetscFree(vbinary->filename);CHKERRQ(ierr);
  ierr = PetscFree(vbinary);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetFlowControl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetFlowControl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetSkipHeader_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetSkipHeader_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetSkipOptions_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetSkipOptions_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetSkipInfo_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetSkipInfo_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetInfoPointer_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetMode_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetUseMPIIO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetUseMPIIO_C",NULL);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerBinaryOpen - Opens a file for binary input/output.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  name - name of file
-  mode - open mode of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

   Output Parameter:
.  viewer - PetscViewer for binary input/output to use with the specified file

    Options Database Keys:
+    -viewer_binary_filename <name> -
.    -viewer_binary_skip_info -
.    -viewer_binary_skip_options -
.    -viewer_binary_skip_header -
-    -viewer_binary_mpiio -

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().

    For reading files, the filename may begin with ftp:// or http:// and/or
    end with .gz; in this case file is brought over and uncompressed.

    For creating files, if the file name ends with .gz it is automatically
    compressed when closed.

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscViewerBinaryRead(), PetscViewerBinarySetUseMPIIO(),
          PetscViewerBinaryGetUseMPIIO(), PetscViewerBinaryGetMPIIOOffset()
@*/
PetscErrorCode PetscViewerBinaryOpen(MPI_Comm comm,const char name[],PetscFileMode mode,PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*viewer,mode);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*viewer,name);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(*viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerBinaryWriteReadMPIIO(PetscViewer viewer,void *data,PetscInt num,PetscInt *count,PetscDataType dtype,PetscBool write)
{
  MPI_Comm           comm = PetscObjectComm((PetscObject)viewer);
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  MPI_File           mfdes = vbinary->mfdes;
  PetscErrorCode     ierr;
  MPI_Datatype       mdtype;
  PetscMPIInt        rank,cnt;
  MPI_Status         status;
  MPI_Aint           ul,dsize;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = PetscMPIIntCast(num,&cnt);CHKERRQ(ierr);
  ierr = PetscDataTypeToMPIDataType(dtype,&mdtype);CHKERRQ(ierr);
  if (write) {
    if (rank == 0) {
      ierr = MPIU_File_write_at(mfdes,vbinary->moff,data,cnt,mdtype,&status);CHKERRQ(ierr);
    }
  } else {
    if (rank == 0) {
      ierr = MPIU_File_read_at(mfdes,vbinary->moff,data,cnt,mdtype,&status);CHKERRQ(ierr);
      if (cnt > 0) {ierr = MPI_Get_count(&status,mdtype,&cnt);CHKERRMPI(ierr);}
    }
    ierr = MPI_Bcast(&cnt,1,MPI_INT,0,comm);CHKERRMPI(ierr);
    ierr = MPI_Bcast(data,cnt,mdtype,0,comm);CHKERRMPI(ierr);
  }
  ierr = MPI_Type_get_extent(mdtype,&ul,&dsize);CHKERRMPI(ierr);
  vbinary->moff += dsize*cnt;
  if (count) *count = cnt;
  PetscFunctionReturn(0);
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

   Output Parameters:
.  count - number of items of data actually read, or NULL.

   Level: beginner

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscViewerBinaryRead()
@*/
PetscErrorCode PetscViewerBinaryRead(PetscViewer viewer,void *data,PetscInt num,PetscInt *count,PetscDataType dtype)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer,PETSC_VIEWER_CLASSID,1,PETSCVIEWERBINARY);
  PetscValidLogicalCollectiveInt(viewer,num,3);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  vbinary = (PetscViewer_Binary*)viewer->data;
#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio) {
    ierr = PetscViewerBinaryWriteReadMPIIO(viewer,data,num,count,dtype,PETSC_FALSE);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscBinarySynchronizedRead(PetscObjectComm((PetscObject)viewer),vbinary->fdes,data,num,count,dtype);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerBinaryWrite - writes to a binary file, only from the first process

   Collective

   Input Parameters:
+  viewer - the binary viewer
.  data - location of data
.  count - number of items of data to write
-  dtype - type of data to write

   Level: beginner

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(), PetscDataType
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscViewerBinaryRead()
@*/
PetscErrorCode PetscViewerBinaryWrite(PetscViewer viewer,const void *data,PetscInt count,PetscDataType dtype)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer,PETSC_VIEWER_CLASSID,1,PETSCVIEWERBINARY);
  PetscValidLogicalCollectiveInt(viewer,count,3);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  vbinary = (PetscViewer_Binary*)viewer->data;
#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio) {
    ierr = PetscViewerBinaryWriteReadMPIIO(viewer,(void*)data,count,NULL,dtype,PETSC_TRUE);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscBinarySynchronizedWrite(PetscObjectComm((PetscObject)viewer),vbinary->fdes,data,count,dtype);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinaryWriteReadAll(PetscViewer viewer,PetscBool write,void *data,PetscInt count,PetscInt start,PetscInt total,PetscDataType dtype)
{
  MPI_Comm              comm = PetscObjectComm((PetscObject)viewer);
  PetscMPIInt           size,rank;
  MPI_Datatype          mdtype;
  PETSC_UNUSED MPI_Aint lb;
  MPI_Aint              dsize;
  PetscBool             useMPIIO;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer,PETSC_VIEWER_CLASSID,1,PETSCVIEWERBINARY);
  PetscValidLogicalCollectiveBool(viewer,((start>=0)||(start==PETSC_DETERMINE)),5);
  PetscValidLogicalCollectiveBool(viewer,((total>=0)||(total==PETSC_DETERMINE)),6);
  PetscValidLogicalCollectiveInt(viewer,total,6);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

  ierr = PetscDataTypeToMPIDataType(dtype,&mdtype);CHKERRQ(ierr);
  ierr = MPI_Type_get_extent(mdtype,&lb,&dsize);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);

  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&useMPIIO);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  if (useMPIIO) {
    MPI_File       mfdes;
    MPI_Offset     off;
    PetscMPIInt    cnt;

    if (start == PETSC_DETERMINE) {
      ierr = MPI_Scan(&count,&start,1,MPIU_INT,MPI_SUM,comm);CHKERRMPI(ierr);
      start -= count;
    }
    if (total == PETSC_DETERMINE) {
      total = start + count;
      ierr = MPI_Bcast(&total,1,MPIU_INT,size-1,comm);CHKERRMPI(ierr);
    }
    ierr = PetscMPIIntCast(count,&cnt);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIODescriptor(viewer,&mfdes);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetMPIIOOffset(viewer,&off);CHKERRQ(ierr);
    off += (MPI_Offset)(start*dsize);
    if (write) {
      ierr = MPIU_File_write_at_all(mfdes,off,data,cnt,mdtype,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    } else {
      ierr = MPIU_File_read_at_all(mfdes,off,data,cnt,mdtype,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    }
    off  = (MPI_Offset)(total*dsize);
    ierr = PetscViewerBinaryAddMPIIOOffset(viewer,off);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif
  {
    int         fdes;
    char        *workbuf = NULL;
    PetscInt    tcount = rank == 0 ? 0 : count,maxcount=0,message_count,flowcontrolcount;
    PetscMPIInt tag,cnt,maxcnt,scnt=0,rcnt=0,j;
    MPI_Status  status;

    ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
    ierr = MPI_Reduce(&tcount,&maxcount,1,MPIU_INT,MPI_MAX,0,comm);CHKERRMPI(ierr);
    ierr = PetscMPIIntCast(maxcount,&maxcnt);CHKERRQ(ierr);
    ierr = PetscMPIIntCast(count,&cnt);CHKERRQ(ierr);

    ierr = PetscViewerBinaryGetDescriptor(viewer,&fdes);CHKERRQ(ierr);
    ierr = PetscViewerFlowControlStart(viewer,&message_count,&flowcontrolcount);CHKERRQ(ierr);
    if (rank == 0) {
      ierr = PetscMalloc(maxcnt*dsize,&workbuf);CHKERRQ(ierr);
      if (write) {
        ierr = PetscBinaryWrite(fdes,data,cnt,dtype);CHKERRQ(ierr);
      } else {
        ierr = PetscBinaryRead(fdes,data,cnt,NULL,dtype);CHKERRQ(ierr);
      }
      for (j=1; j<size; j++) {
        ierr = PetscViewerFlowControlStepMain(viewer,j,&message_count,flowcontrolcount);CHKERRQ(ierr);
        if (write) {
          ierr = MPI_Recv(workbuf,maxcnt,mdtype,j,tag,comm,&status);CHKERRMPI(ierr);
          ierr = MPI_Get_count(&status,mdtype,&rcnt);CHKERRMPI(ierr);
          ierr = PetscBinaryWrite(fdes,workbuf,rcnt,dtype);CHKERRQ(ierr);
        } else {
          ierr = MPI_Recv(&scnt,1,MPI_INT,j,tag,comm,MPI_STATUS_IGNORE);CHKERRMPI(ierr);
          ierr = PetscBinaryRead(fdes,workbuf,scnt,NULL,dtype);CHKERRQ(ierr);
          ierr = MPI_Send(workbuf,scnt,mdtype,j,tag,comm);CHKERRMPI(ierr);
        }
      }
      ierr = PetscFree(workbuf);CHKERRQ(ierr);
      ierr = PetscViewerFlowControlEndMain(viewer,&message_count);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerFlowControlStepWorker(viewer,rank,&message_count);CHKERRQ(ierr);
      if (write) {
        ierr = MPI_Send(data,cnt,mdtype,0,tag,comm);CHKERRMPI(ierr);
      } else {
        ierr = MPI_Send(&cnt,1,MPI_INT,0,tag,comm);CHKERRMPI(ierr);
        ierr = MPI_Recv(data,cnt,mdtype,0,tag,comm,MPI_STATUS_IGNORE);CHKERRMPI(ierr);
      }
      ierr = PetscViewerFlowControlEndWorker(viewer,&message_count);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerBinaryReadAll - reads from a binary file from all processes

   Collective

   Input Parameters:
+  viewer - the binary viewer
.  data - location of data
.  count - local number of items of data to read
.  start - local start, can be PETSC_DETERMINE
.  total - global number of items of data to read, can be PETSC_DETERMINE
-  dtype - type of data to read

   Level: advanced

.seealso: PetscViewerBinaryOpen(), PetscViewerBinarySetUseMPIIO(), PetscViewerBinaryRead(), PetscViewerBinaryWriteAll()
@*/
PetscErrorCode PetscViewerBinaryReadAll(PetscViewer viewer,void *data,PetscInt count,PetscInt start,PetscInt total,PetscDataType dtype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscViewerBinaryWriteReadAll(viewer,PETSC_FALSE,data,count,start,total,dtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerBinaryWriteAll - writes to a binary file from all processes

   Collective

   Input Parameters:
+  viewer - the binary viewer
.  data - location of data
.  count - local number of items of data to write
.  start - local start, can be PETSC_DETERMINE
.  total - global number of items of data to write, can be PETSC_DETERMINE
-  dtype - type of data to write

   Level: advanced

.seealso: PetscViewerBinaryOpen(), PetscViewerBinarySetUseMPIIO(), PetscViewerBinaryWriteAll(), PetscViewerBinaryReadAll()
@*/
PetscErrorCode PetscViewerBinaryWriteAll(PetscViewer viewer,const void *data,PetscInt count,PetscInt start,PetscInt total,PetscDataType dtype)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscViewerBinaryWriteReadAll(viewer,PETSC_TRUE,(void*)data,count,start,total,dtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerBinaryWriteStringArray - writes to a binary file, only from the first process an array of strings

   Collective

   Input Parameters:
+  viewer - the binary viewer
-  data - location of the array of strings

   Level: intermediate

    Notes:
    array of strings is null terminated

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscViewerBinaryRead()
@*/
PetscErrorCode PetscViewerBinaryWriteStringArray(PetscViewer viewer,const char * const *data)
{
  PetscErrorCode ierr;
  PetscInt       i,n = 0,*sizes;
  size_t         len;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  /* count number of strings */
  while (data[n++]);
  n--;
  ierr = PetscMalloc1(n+1,&sizes);CHKERRQ(ierr);
  sizes[0] = n;
  for (i=0; i<n; i++) {
    ierr = PetscStrlen(data[i],&len);CHKERRQ(ierr);
    sizes[i+1] = (PetscInt)len + 1; /* size includes space for the null terminator */
  }
  ierr = PetscViewerBinaryWrite(viewer,sizes,n+1,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscViewerBinaryWrite(viewer,(void*)data[i],sizes[i+1],PETSC_CHAR);CHKERRQ(ierr);
  }
  ierr = PetscFree(sizes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerBinaryReadStringArray - reads a binary file an array of strings

   Collective

   Input Parameter:
.  viewer - the binary viewer

   Output Parameter:
.  data - location of the array of strings

   Level: intermediate

    Notes:
    array of strings is null terminated

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscViewerBinaryRead()
@*/
PetscErrorCode PetscViewerBinaryReadStringArray(PetscViewer viewer,char ***data)
{
  PetscErrorCode ierr;
  PetscInt       i,n,*sizes,N = 0;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  /* count number of strings */
  ierr = PetscViewerBinaryRead(viewer,&n,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&sizes);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,sizes,n,NULL,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<n; i++) N += sizes[i];
  ierr = PetscMalloc((n+1)*sizeof(char*) + N*sizeof(char),data);CHKERRQ(ierr);
  (*data)[0] = (char*)((*data) + n + 1);
  for (i=1; i<n; i++) (*data)[i] = (*data)[i-1] + sizes[i-1];
  ierr = PetscViewerBinaryRead(viewer,(*data)[0],N,NULL,PETSC_CHAR);CHKERRQ(ierr);
  (*data)[n] = NULL;
  ierr = PetscFree(sizes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
     PetscViewerFileSetMode - Sets the open mode of file

    Logically Collective on PetscViewer

  Input Parameters:
+  viewer - the PetscViewer; must be a a PETSCVIEWERBINARY, PETSCVIEWERMATLAB, PETSCVIEWERHDF5, or PETSCVIEWERASCII  PetscViewer
-  mode - open mode of file
$    FILE_MODE_WRITE - create new file for output
$    FILE_MODE_READ - open existing file for input
$    FILE_MODE_APPEND - open existing file for output

  Level: advanced

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()

@*/
PetscErrorCode PetscViewerFileSetMode(PetscViewer viewer,PetscFileMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveEnum(viewer,mode,2);
  if (mode == FILE_MODE_UNDEFINED) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Cannot set FILE_MODE_UNDEFINED");
  else if (mode < FILE_MODE_UNDEFINED || mode > FILE_MODE_APPEND_UPDATE) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_OUTOFRANGE,"Invalid file mode %d",(int)mode);
  ierr = PetscTryMethod(viewer,"PetscViewerFileSetMode_C",(PetscViewer,PetscFileMode),(viewer,mode));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetMode_Binary(PetscViewer viewer,PetscFileMode mode)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  if (viewer->setupcalled && vbinary->filemode != mode) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ORDER,"Cannot change mode to %s after setup",PetscFileModes[mode]);
  vbinary->filemode = mode;
  PetscFunctionReturn(0);
}

/*@C
     PetscViewerFileGetMode - Gets the open mode of file

    Not Collective

  Input Parameter:
.  viewer - the PetscViewer; must be a PETSCVIEWERBINARY, PETSCVIEWERMATLAB, PETSCVIEWERHDF5, or PETSCVIEWERASCII  PetscViewer

  Output Parameter:
.  mode - open mode of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

  Level: advanced

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()

@*/
PetscErrorCode PetscViewerFileGetMode(PetscViewer viewer,PetscFileMode *mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(mode,2);
  ierr = PetscUseMethod(viewer,"PetscViewerFileGetMode_C",(PetscViewer,PetscFileMode*),(viewer,mode));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetMode_Binary(PetscViewer viewer,PetscFileMode *mode)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *mode = vbinary->filemode;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetName_Binary(PetscViewer viewer,const char name[])
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (viewer->setupcalled && vbinary->filename) {
    /* gzip can be run after the file with the previous filename has been closed */
    ierr = PetscFree(vbinary->ogzfilename);CHKERRQ(ierr);
    ierr = PetscStrallocpy(vbinary->filename,&vbinary->ogzfilename);CHKERRQ(ierr);
  }
  ierr = PetscFree(vbinary->filename);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&vbinary->filename);CHKERRQ(ierr);
  viewer->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetName_Binary(PetscViewer viewer,const char **name)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *name = vbinary->filename;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerFileSetUp_BinaryMPIIO(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  int                amode;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  vbinary->storecompressed = PETSC_FALSE;

  vbinary->moff = 0;
  switch (vbinary->filemode) {
  case FILE_MODE_READ:   amode = MPI_MODE_RDONLY; break;
  case FILE_MODE_WRITE:  amode = MPI_MODE_WRONLY | MPI_MODE_CREATE; break;
  case FILE_MODE_APPEND: amode = MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_APPEND; break;
  case FILE_MODE_UNDEFINED: SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerSetUp()");
  default: SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Unsupported file mode %s",PetscFileModes[vbinary->filemode]);
  }
  ierr = MPI_File_open(PetscObjectComm((PetscObject)viewer),vbinary->filename,amode,MPI_INFO_NULL,&vbinary->mfdes);CHKERRMPI(ierr);
  /*
      The MPI standard does not have MPI_MODE_TRUNCATE. We emulate this behavior by setting the file size to zero.
  */
  if (vbinary->filemode == FILE_MODE_WRITE) {ierr = MPI_File_set_size(vbinary->mfdes,0);CHKERRMPI(ierr);}
  /*
      Initially, all processes view the file as a linear byte stream. Therefore, for files opened with MPI_MODE_APPEND,
      MPI_File_get_position[_shared](fh, &offset) returns the absolute byte position at the end of file.
      Otherwise, we would need to call MPI_File_get_byte_offset(fh, offset, &byte_offset) to convert
      the offset in etype units to an absolute byte position.
   */
  if (vbinary->filemode == FILE_MODE_APPEND) {ierr = MPI_File_get_position(vbinary->mfdes,&vbinary->moff);CHKERRMPI(ierr);}
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode PetscViewerFileSetUp_BinarySTDIO(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  const char         *fname;
  char               bname[PETSC_MAX_PATH_LEN],*gz;
  PetscBool          found;
  PetscMPIInt        rank;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRMPI(ierr);

  /* if file name ends in .gz strip that off and note user wants file compressed */
  vbinary->storecompressed = PETSC_FALSE;
  if (vbinary->filemode == FILE_MODE_WRITE) {
    ierr = PetscStrstr(vbinary->filename,".gz",&gz);CHKERRQ(ierr);
    if (gz && gz[3] == 0) {*gz = 0; vbinary->storecompressed = PETSC_TRUE;}
  }
#if !defined(PETSC_HAVE_POPEN)
  if (vbinary->storecompressed) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP_SYS,"Cannot run gzip on this machine");
#endif

  fname = vbinary->filename;
  if (vbinary->filemode == FILE_MODE_READ) { /* possibly get the file from remote site or compressed file */
    ierr  = PetscFileRetrieve(PetscObjectComm((PetscObject)viewer),fname,bname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
    if (!found) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_OPEN,"Cannot locate file: %s",fname);
    fname = bname;
  }

  vbinary->fdes = -1;
  if (rank == 0) { /* only first processor opens file*/
    PetscFileMode mode = vbinary->filemode;
    if (mode == FILE_MODE_APPEND) {
      /* check if asked to append to a non-existing file */
      ierr = PetscTestFile(fname,'\0',&found);CHKERRQ(ierr);
      if (!found) mode = FILE_MODE_WRITE;
    }
    ierr = PetscBinaryOpen(fname,mode,&vbinary->fdes);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetUp_BinaryInfo(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscMPIInt        rank;
  PetscBool          found;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  vbinary->fdes_info = NULL;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRMPI(ierr);
  if (!vbinary->skipinfo && (vbinary->filemode == FILE_MODE_READ || rank == 0)) {
    char infoname[PETSC_MAX_PATH_LEN],iname[PETSC_MAX_PATH_LEN],*gz;

    ierr = PetscStrncpy(infoname,vbinary->filename,sizeof(infoname));CHKERRQ(ierr);
    /* remove .gz if it ends file name */
    ierr = PetscStrstr(infoname,".gz",&gz);CHKERRQ(ierr);
    if (gz && gz[3] == 0) *gz = 0;

    ierr = PetscStrlcat(infoname,".info",sizeof(infoname));CHKERRQ(ierr);
    if (vbinary->filemode == FILE_MODE_READ) {
      ierr = PetscFixFilename(infoname,iname);CHKERRQ(ierr);
      ierr = PetscFileRetrieve(PetscObjectComm((PetscObject)viewer),iname,infoname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
      if (found) {ierr = PetscOptionsInsertFile(PetscObjectComm((PetscObject)viewer),((PetscObject)viewer)->options,infoname,PETSC_FALSE);CHKERRQ(ierr);}
    } else if (rank == 0) { /* write or append */
      const char *omode = (vbinary->filemode == FILE_MODE_APPEND) ? "a" : "w";
      vbinary->fdes_info = fopen(infoname,omode);
      if (!vbinary->fdes_info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open .info file %s for writing",infoname);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerSetUp_Binary(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscBool          usempiio;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!vbinary->setfromoptionscalled) {ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);}
  if (!vbinary->filename) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call PetscViewerFileSetName()");
  if (vbinary->filemode == (PetscFileMode)-1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call PetscViewerFileSetMode()");
  ierr = PetscViewerFileClose_Binary(viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetUseMPIIO(viewer,&usempiio);CHKERRQ(ierr);
  if (usempiio) {
#if defined(PETSC_HAVE_MPIIO)
    ierr = PetscViewerFileSetUp_BinaryMPIIO(viewer);CHKERRQ(ierr);
#endif
  } else {
    ierr = PetscViewerFileSetUp_BinarySTDIO(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerFileSetUp_BinaryInfo(viewer);CHKERRQ(ierr);

  ierr = PetscLogObjectState((PetscObject)viewer,"File: %s",vbinary->filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerView_Binary(PetscViewer v,PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  const char         *fname = vbinary->filename ? vbinary->filename : "not yet set";
  const char         *fmode = vbinary->filemode != (PetscFileMode) -1 ? PetscFileModes[vbinary->filemode] : "not yet set";
  PetscBool          usempiio;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryGetUseMPIIO(v,&usempiio);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Filename: %s\n",fname);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Mode: %s (%s)\n",fmode,usempiio ? "mpiio" : "stdio");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerSetFromOptions_Binary(PetscOptionItems *PetscOptionsObject,PetscViewer viewer)
{
  PetscViewer_Binary *binary = (PetscViewer_Binary*)viewer->data;
  char               defaultname[PETSC_MAX_PATH_LEN];
  PetscBool          flg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (viewer->setupcalled) PetscFunctionReturn(0);
  ierr = PetscOptionsHead(PetscOptionsObject,"Binary PetscViewer Options");CHKERRQ(ierr);
  ierr = PetscSNPrintf(defaultname,PETSC_MAX_PATH_LEN-1,"binaryoutput");CHKERRQ(ierr);
  ierr = PetscOptionsString("-viewer_binary_filename","Specify filename","PetscViewerFileSetName",defaultname,defaultname,sizeof(defaultname),&flg);CHKERRQ(ierr);
  if (flg) { ierr = PetscViewerFileSetName_Binary(viewer,defaultname);CHKERRQ(ierr); }
  ierr = PetscOptionsBool("-viewer_binary_skip_info","Skip writing/reading .info file","PetscViewerBinarySetSkipInfo",binary->skipinfo,&binary->skipinfo,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewer_binary_skip_options","Skip parsing Vec/Mat load options","PetscViewerBinarySetSkipOptions",binary->skipoptions,&binary->skipoptions,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewer_binary_skip_header","Skip writing/reading header information","PetscViewerBinarySetSkipHeader",binary->skipheader,&binary->skipheader,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscOptionsBool("-viewer_binary_mpiio","Use MPI-IO functionality to write/read binary file","PetscViewerBinarySetUseMPIIO",binary->usempiio,&binary->usempiio,NULL);CHKERRQ(ierr);
#else
  ierr = PetscOptionsBool("-viewer_binary_mpiio","Use MPI-IO functionality to write/read binary file (NOT AVAILABLE)","PetscViewerBinarySetUseMPIIO",PETSC_FALSE,NULL,NULL);CHKERRQ(ierr);
#endif
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  binary->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWERBINARY - A viewer that saves to binary files

.seealso:  PetscViewerBinaryOpen(), PETSC_VIEWER_STDOUT_(),PETSC_VIEWER_STDOUT_SELF, PETSC_VIEWER_STDOUT_WORLD, PetscViewerCreate(), PetscViewerASCIIOpen(),
           PetscViewerMatlabOpen(), VecView(), DMView(), PetscViewerMatlabPutArray(), PETSCVIEWERASCII, PETSCVIEWERMATLAB, PETSCVIEWERDRAW,
           PetscViewerFileSetName(), PetscViewerFileSetMode(), PetscViewerFormat, PetscViewerType, PetscViewerSetType(),
           PetscViewerBinaryGetUseMPIIO(), PetscViewerBinarySetUseMPIIO()

  Level: beginner

M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_Binary(PetscViewer v)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  ierr    = PetscNewLog(v,&vbinary);CHKERRQ(ierr);
  v->data = (void*)vbinary;

  v->ops->setfromoptions   = PetscViewerSetFromOptions_Binary;
  v->ops->destroy          = PetscViewerDestroy_Binary;
  v->ops->view             = PetscViewerView_Binary;
  v->ops->setup            = PetscViewerSetUp_Binary;
  v->ops->flush            = NULL; /* Should we support Flush() ? */
  v->ops->getsubviewer     = PetscViewerGetSubViewer_Binary;
  v->ops->restoresubviewer = PetscViewerRestoreSubViewer_Binary;
  v->ops->read             = PetscViewerBinaryRead;

  vbinary->fdes            = -1;
#if defined(PETSC_HAVE_MPIIO)
  vbinary->usempiio        = PETSC_FALSE;
  vbinary->mfdes           = MPI_FILE_NULL;
  vbinary->mfsub           = MPI_FILE_NULL;
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

  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetFlowControl_C",PetscViewerBinaryGetFlowControl_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetFlowControl_C",PetscViewerBinarySetFlowControl_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetSkipHeader_C",PetscViewerBinaryGetSkipHeader_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetSkipHeader_C",PetscViewerBinarySetSkipHeader_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetSkipOptions_C",PetscViewerBinaryGetSkipOptions_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetSkipOptions_C",PetscViewerBinarySetSkipOptions_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetSkipInfo_C",PetscViewerBinaryGetSkipInfo_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetSkipInfo_C",PetscViewerBinarySetSkipInfo_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetInfoPointer_C",PetscViewerBinaryGetInfoPointer_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",PetscViewerFileGetName_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",PetscViewerFileSetName_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetMode_C",PetscViewerFileGetMode_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_Binary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetUseMPIIO_C",PetscViewerBinaryGetUseMPIIO_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetUseMPIIO_C",PetscViewerBinarySetUseMPIIO_Binary);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Binary_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
PetscMPIInt Petsc_Viewer_Binary_keyval = MPI_KEYVAL_INVALID;

/*@C
     PETSC_VIEWER_BINARY_ - Creates a binary PetscViewer shared by all processors
                     in a communicator.

     Collective

     Input Parameter:
.    comm - the MPI communicator to share the binary PetscViewer

     Level: intermediate

   Options Database Keys:
+    -viewer_binary_filename <name> - filename in which to store the binary data, defaults to binaryoutput
.    -viewer_binary_skip_info - true means do not create .info file for this viewer
.    -viewer_binary_skip_options - true means do not use the options database for this viewer
.    -viewer_binary_skip_header - true means do not store the usual header information in the binary file
-    -viewer_binary_mpiio - true means use the file via MPI-IO, maybe faster for large files and many MPI ranks

   Environmental variables:
-   PETSC_VIEWER_BINARY_FILENAME - filename in which to store the binary data, defaults to binaryoutput

     Notes:
     Unlike almost all other PETSc routines, PETSC_VIEWER_BINARY_ does not return
     an error code.  The binary PetscViewer is usually used in the form
$       XXXView(XXX object,PETSC_VIEWER_BINARY_(comm));

.seealso: PETSC_VIEWER_BINARY_WORLD, PETSC_VIEWER_BINARY_SELF, PetscViewerBinaryOpen(), PetscViewerCreate(),
          PetscViewerDestroy()
@*/
PetscViewer PETSC_VIEWER_BINARY_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PetscViewer    viewer;
  char           fname[PETSC_MAX_PATH_LEN];
  MPI_Comm       ncomm;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&ncomm,NULL);if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
  if (Petsc_Viewer_Binary_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Viewer_Binary_keyval,NULL);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
  }
  ierr = MPI_Comm_get_attr(ncomm,Petsc_Viewer_Binary_keyval,(void**)&viewer,(int*)&flg);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscOptionsGetenv(ncomm,"PETSC_VIEWER_BINARY_FILENAME",fname,PETSC_MAX_PATH_LEN,&flg);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT," ");PetscFunctionReturn(NULL);}
    if (!flg) {
      ierr = PetscStrcpy(fname,"binaryoutput");
      if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT," ");PetscFunctionReturn(NULL);}
    }
    ierr = PetscViewerBinaryOpen(ncomm,fname,FILE_MODE_WRITE,&viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT," ");PetscFunctionReturn(NULL);}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT," ");PetscFunctionReturn(NULL);}
    ierr = MPI_Comm_set_attr(ncomm,Petsc_Viewer_Binary_keyval,(void*)viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
  }
  ierr = PetscCommDestroy(&ncomm);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT," ");PetscFunctionReturn(NULL);}
  PetscFunctionReturn(viewer);
}
