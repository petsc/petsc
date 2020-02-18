
#include <petsc/private/viewerimpl.h>    /*I   "petscviewer.h"   I*/
#include <fcntl.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_IO_H)
#include <io.h>
#endif

typedef struct  {
  int           fdes;                 /* file descriptor, ignored if using MPI IO */
#if defined(PETSC_HAVE_MPIIO)
  PetscBool     usempiio;
  MPI_File      mfdes;                /* ignored unless using MPI IO */
  MPI_File      mfsub;                /* subviewer support */
  MPI_Offset    moff;
#endif
  PetscFileMode btype;                /* read or write? */
  FILE          *fdes_info;           /* optional file containing info on binary file*/
  PetscBool     storecompressed;      /* gzip the write binary file when closing it*/
  char          *filename;
  char          *ogzfilename;         /* gzip can be run after the filename has been updated */
  PetscBool     skipinfo;             /* Don't create info file for writing; don't use for reading */
  PetscBool     skipoptions;          /* don't use PETSc options database when loading */
  PetscInt      flowcontrol;          /* allow only <flowcontrol> messages outstanding at a time while doing IO */
  PetscBool     skipheader;           /* don't write header, only raw data */
  PetscBool     matlabheaderwritten;  /* if format is PETSC_VIEWER_BINARY_MATLAB has the MATLAB .info header been written yet */
  PetscBool     setfromoptionscalled;
} PetscViewer_Binary;

static PetscErrorCode PetscViewerGetSubViewer_Binary(PetscViewer viewer,MPI_Comm comm,PetscViewer *outviewer)
{
  int                rank;
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

  /* Return subviewer in process zero */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRQ(ierr);
  if (!rank) {
    PetscMPIInt flg;

    ierr = MPI_Comm_compare(PETSC_COMM_SELF,comm,&flg);CHKERRQ(ierr);
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
      switch (vbinary->btype) {
      case FILE_MODE_READ:  amode = MPI_MODE_RDONLY; break;
      case FILE_MODE_WRITE: amode = MPI_MODE_WRONLY; break;
      default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported file mode %s",PetscFileModes[vbinary->btype]);
      }
      ierr = MPI_File_open(PETSC_COMM_SELF,vbinary->filename,amode,MPI_INFO_NULL,&vbinary->mfsub);CHKERRQ(ierr);
    }
    /* Subviewer gets the MPI file handle on PETSC_COMM_SELF */
    obinary->mfdes = vbinary->mfsub;
    obinary->mfsub = MPI_FILE_NULL;
    obinary->moff  = vbinary->moff;
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerRestoreSubViewer_Binary(PetscViewer viewer,MPI_Comm comm,PetscViewer *outviewer)
{
  PetscMPIInt        rank;
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscErrorCode     ierr;
#if defined(PETSC_HAVE_MPIIO)
  MPI_Offset         moff = 0;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRQ(ierr);
  if (rank && *outviewer) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Subviewer not obtained from viewer");

#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio && *outviewer) {
    PetscViewer_Binary *obinary = (PetscViewer_Binary*)(*outviewer)->data;
    if (obinary->mfdes != vbinary->mfsub) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Subviewer not obtained from viewer");
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
    ierr = MPI_Bcast(&ioff,1,MPIU_INT64,0,PetscObjectComm((PetscObject)viewer));CHKERRQ(ierr);
    vbinary->moff = (MPI_Offset)ioff;
  }
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
/*@C
    PetscViewerBinaryGetMPIIOOffset - Gets the current offset that should be passed to MPI_File_set_view()

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.    off - the current offset

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

    Use PetscViewerBinaryAddMPIIOOffset() to increase this value after you have written a view.


.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetInfoPointer(), PetscViewerBinaryGetUseMPIIO(), PetscViewerBinarySetUseMPIIO(), PetscViewerBinaryAddMPIIOOffset()
@*/
PetscErrorCode PetscViewerBinaryGetMPIIOOffset(PetscViewer viewer,MPI_Offset *off)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *off = vbinary->moff;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerBinaryAddMPIIOOffset - Adds to the current offset that should be passed to MPI_File_set_view()

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()
-    off - the addition to the offset

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

    Use PetscViewerBinaryGetMPIIOOffset() to get the value that you should pass to MPI_File_set_view()


.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer(), PetscViewerBinaryGetUseMPIIO(), PetscViewerBinarySetUseMPIIO(), PetscViewerBinaryGetMPIIOOffset()
@*/
PetscErrorCode PetscViewerBinaryAddMPIIOOffset(PetscViewer viewer,MPI_Offset off)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
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
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  *fdes = vbinary->mfdes;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinaryGetUseMPIIO_Binary(PetscViewer viewer,PetscBool  *flg)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *flg = vbinary->usempiio;
  PetscFunctionReturn(0);
}
#endif


/*@C
    PetscViewerBinaryGetUseMPIIO - Returns PETSC_TRUE if the binary viewer uses MPI-IO.

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
-   flg - PETSC_TRUE if MPI-IO is being used

    Options Database:
    -viewer_binary_mpiio : Flag for using MPI-IO

    Level: advanced

    Note:
    If MPI-IO is not available, this function will always return PETSC_FALSE

    Fortran Note:
    This routine is not supported in Fortran.


.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetInfoPointer(), PetscViewerBinarySetUseMPIIO(), PetscViewerBinaryGetMPIIOOffset()
@*/
PetscErrorCode PetscViewerBinaryGetUseMPIIO(PetscViewer viewer,PetscBool *flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *flg = PETSC_FALSE;
  ierr = PetscTryMethod(viewer,"PetscViewerBinaryGetUseMPIIO_C",(PetscViewer,PetscBool*),(viewer,flg));CHKERRQ(ierr);
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
  ierr = PetscUseMethod(viewer,"PetscViewerBinaryGetFlowControl_C",(PetscViewer,PetscInt*),(viewer,fc));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinarySetFlowControl_Binary(PetscViewer viewer,PetscInt fc)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  if (fc <= 1) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ARG_OUTOFRANGE,"Flow control count must be greater than 1, %D was set",fc);
  vbinary->flowcontrol = fc;
  PetscFunctionReturn(0);
}

/*@C
    PetscViewerBinarySetFlowControl - Sets how many messages are allowed to outstanding at the same time during parallel IO reads/writes

    Not Collective

    Input Parameter:
+   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()
-   fc - the number of messages, defaults to 256 if this function was not called

    Level: advanced

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer(), PetscViewerBinaryGetFlowControl()

@*/
PetscErrorCode  PetscViewerBinarySetFlowControl(PetscViewer viewer,PetscInt fc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(viewer,"PetscViewerBinarySetFlowControl_C",(PetscViewer,PetscInt),(viewer,fc));CHKERRQ(ierr);
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
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  *fdes = vbinary->fdes;
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinarySkipInfo - Binary file will not have .info file created with it

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerCreate()

    Options Database Key:
.   -viewer_binary_skip_info

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
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  vbinary->skipinfo = PETSC_TRUE;
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
    PetscViewerBinarySetSkipInfo - Binary file will not have .info file created with it

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerCreate()

    Options Database Key:
.   -viewer_binary_skip_info

    Level: advanced

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySetSkipOptions(),
          PetscViewerBinaryGetSkipOptions(), PetscViewerBinaryGetSkipInfo()
@*/
PetscErrorCode PetscViewerBinarySetSkipInfo(PetscViewer viewer,PetscBool skip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(viewer,"PetscViewerBinarySetSkipInfo_C",(PetscViewer,PetscBool),(viewer,skip));CHKERRQ(ierr);
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
  ierr = PetscUseMethod(viewer,"PetscViewerBinaryGetSkipInfo_C",(PetscViewer,PetscBool*),(viewer,skip));CHKERRQ(ierr);
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
    PetscViewerBinarySetSkipOptions - do not use the PETSc options database when loading objects

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()
-   skip - PETSC_TRUE means do not use

    Options Database Key:
.   -viewer_binary_skip_options

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
  ierr = PetscUseMethod(viewer,"PetscViewerBinarySetSkipOptions_C",(PetscViewer,PetscBool),(viewer,skip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinaryGetSkipOptions_Binary(PetscViewer viewer,PetscBool *skip)
{
  PetscViewer_Binary *vbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  vbinary = (PetscViewer_Binary*)viewer->data;
  *skip   = vbinary->skipoptions;
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
  ierr = PetscUseMethod(viewer,"PetscViewerBinaryGetSkipOptions_C",(PetscViewer,PetscBool*),(viewer,skip));CHKERRQ(ierr);
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
    PetscViewerBinarySetSkipHeader - do not write a header with size information on output, just raw data

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()
-   skip - PETSC_TRUE means do not write header

    Options Database Key:
.   -viewer_binary_skip_header

    Level: advanced

    Notes:
    This must be called after PetscViewerSetType()

           Can ONLY be called on a binary viewer

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySkipInfo(),
          PetscViewerBinaryGetSkipHeader()
@*/
PetscErrorCode PetscViewerBinarySetSkipHeader(PetscViewer viewer,PetscBool skip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(viewer,"PetscViewerBinarySetSkipHeader_C",(PetscViewer,PetscBool),(viewer,skip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinaryGetSkipHeader_Binary(PetscViewer viewer,PetscBool  *skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *skip = vbinary->skipheader;
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
  *skip = PETSC_FALSE;
  ierr  = PetscUseMethod(viewer,"PetscViewerBinaryGetSkipHeader_C",(PetscViewer,PetscBool*),(viewer,skip));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinaryGetInfoPointer_Binary(PetscViewer viewer,FILE **file)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscErrorCode     ierr;
  MPI_Comm           comm;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  *file = vbinary->fdes_info;
  if (viewer->format == PETSC_VIEWER_BINARY_MATLAB && !vbinary->matlabheaderwritten) {
    vbinary->matlabheaderwritten = PETSC_TRUE;
    ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,*file,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,*file,"#$$ Set.filename = '%s';\n",vbinary->filename);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,*file,"#$$ fd = PetscOpenFile(Set.filename);\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,*file,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n");CHKERRQ(ierr);
  }
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
  *file = NULL;
  ierr  = PetscTryMethod(viewer,"PetscViewerBinaryGetInfoPointer_C",(PetscViewer,FILE **),(viewer,file));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileClose_Binary(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  PetscErrorCode     ierr;
  PetscMPIInt        rank;
  int                err;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)v),&rank);CHKERRQ(ierr);
  if ((!rank || vbinary->btype == FILE_MODE_READ) && vbinary->fdes) {
    close(vbinary->fdes);
    if (!rank && vbinary->storecompressed) {
      char par[PETSC_MAX_PATH_LEN],buf[PETSC_MAX_PATH_LEN];
      FILE *fp;
      /* compress the file */
      ierr = PetscStrncpy(par,"gzip -f ",sizeof(par));CHKERRQ(ierr);
      ierr = PetscStrlcat(par,vbinary->ogzfilename ? vbinary->ogzfilename : vbinary->filename,sizeof(par));CHKERRQ(ierr);
      ierr = PetscFree(vbinary->ogzfilename);CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)
      ierr = PetscPOpen(PETSC_COMM_SELF,NULL,par,"r",&fp);CHKERRQ(ierr);
      if (fgets(buf,1024,fp)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error from command %s\n%s",par,buf);
      ierr = PetscPClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
#else
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
    }
  }
  if (vbinary->fdes_info) {
    err = fclose(vbinary->fdes_info);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerFileClose_BinaryMPIIO(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  int                err;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (vbinary->mfdes != MPI_FILE_NULL) {
    ierr = MPI_File_close(&vbinary->mfdes);CHKERRQ(ierr);
  }
  if (vbinary->mfsub != MPI_FILE_NULL) {
    ierr = MPI_File_close(&vbinary->mfsub);CHKERRQ(ierr);
  }
  if (vbinary->fdes_info) {
    err = fclose(vbinary->fdes_info);
    if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
  }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode PetscViewerDestroy_Binary(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (v->format == PETSC_VIEWER_BINARY_MATLAB) {
    MPI_Comm comm;
    FILE     *info;

    ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);
    ierr = PetscViewerBinaryGetInfoPointer(v,&info);CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,info,"#--- begin code written by PetscViewerBinary for MATLAB format ---#\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,info,"#$$ close(fd);\n");CHKERRQ(ierr);
    ierr = PetscFPrintf(comm,info,"#--- end code written by PetscViewerBinary for MATLAB format ---#\n\n");CHKERRQ(ierr);
  }
#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio) {
    ierr = PetscViewerFileClose_BinaryMPIIO(v);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscViewerFileClose_Binary(v);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  ierr = PetscFree(vbinary->filename);CHKERRQ(ierr);
  ierr = PetscFree(vbinary->ogzfilename);CHKERRQ(ierr);
  ierr = PetscFree(vbinary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerBinaryOpen - Opens a file for binary input/output.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  name - name of file
-  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

   Output Parameter:
.  binv - PetscViewer for binary input/output to use with the specified file

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

    For writing files it only opens the file on processor 0 in the communicator.
    For readable files it opens the file on all nodes that have the file. If
    node 0 does not have the file it generates an error even if other nodes
    do have the file.

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscViewerBinaryRead(), PetscViewerBinarySetUseMPIIO(),
          PetscViewerBinaryGetUseMPIIO(), PetscViewerBinaryGetMPIIOOffset()
@*/
PetscErrorCode PetscViewerBinaryOpen(MPI_Comm comm,const char name[],PetscFileMode type,PetscViewer *binv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,binv);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*binv,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*binv,type);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*binv,name);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(*binv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerBinaryWriteReadMPIIO(PetscViewer viewer,void *data,PetscInt num,PetscInt *count,PetscDataType dtype,PetscBool write)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscErrorCode     ierr;
  MPI_File           mfdes;
  MPI_Datatype       mdtype;
  PetscMPIInt        cnt;
  MPI_Status         status;
  MPI_Aint           ul,dsize;

  PetscFunctionBegin;
  mfdes = vbinary->mfdes;
  ierr = PetscMPIIntCast(num,&cnt);CHKERRQ(ierr);
  ierr = PetscDataTypeToMPIDataType(dtype,&mdtype);CHKERRQ(ierr);
  ierr = MPI_File_set_view(mfdes,vbinary->moff,mdtype,mdtype,(char*)"native",MPI_INFO_NULL);CHKERRQ(ierr);
  if (write) {
    ierr = MPIU_File_write_all(mfdes,data,cnt,mdtype,&status);CHKERRQ(ierr);
  } else {
    ierr = MPIU_File_read_all(mfdes,data,cnt,mdtype,&status);CHKERRQ(ierr);
    if (cnt > 0) {ierr = MPI_Get_count(&status,mdtype,&cnt);CHKERRQ(ierr);}
  }
  ierr = MPI_Type_get_extent(mdtype,&ul,&dsize);CHKERRQ(ierr);

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
.  count - number of items of data actually read, or NULL. Unless an error is generated this is always set to the input parameter num.

   Level: beginner

   Developer Note: Since count is always set to num it is not clear what purpose the output argument count serves.

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscBinaryViewerRead()
@*/
PetscErrorCode PetscViewerBinaryRead(PetscViewer viewer,void *data,PetscInt num,PetscInt *count,PetscDataType dtype)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
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
.  dtype - type of data to write
-  istemp - data may be overwritten

   Level: beginner

   Notes:
    because byte-swapping may be done on the values in data it cannot be declared const

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(), PetscDataType
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscBinaryViewerRead()
@*/
PetscErrorCode PetscViewerBinaryWrite(PetscViewer viewer,void *data,PetscInt count,PetscDataType dtype,PetscBool istemp)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->usempiio) {
    ierr = PetscViewerBinaryWriteReadMPIIO(viewer,data,count,NULL,dtype,PETSC_TRUE);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscBinarySynchronizedWrite(PetscObjectComm((PetscObject)viewer),vbinary->fdes,data,count,dtype,istemp);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
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
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscBinaryViewerRead()
@*/
PetscErrorCode PetscViewerBinaryWriteStringArray(PetscViewer viewer,const char * const *data)
{
  PetscErrorCode ierr;
  PetscInt       i,n = 0,*sizes;

  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  /* count number of strings */
  while (data[n++]) ;
  n--;
  ierr     = PetscMalloc1(n+1,&sizes);CHKERRQ(ierr);
  sizes[0] = n;
  for (i=0; i<n; i++) {
    size_t tmp;
    ierr       = PetscStrlen(data[i],&tmp);CHKERRQ(ierr);
    sizes[i+1] = tmp + 1;   /* size includes space for the null terminator */
  }
  ierr = PetscViewerBinaryWrite(viewer,sizes,n+1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscViewerBinaryWrite(viewer,(void*)data[i],sizes[i+1],PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
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
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscBinaryViewerRead()
@*/
PetscErrorCode PetscViewerBinaryReadStringArray(PetscViewer viewer,char ***data)
{
  PetscErrorCode ierr;
  PetscInt       i,n,*sizes,N = 0;

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

static PetscErrorCode PetscViewerFileGetName_Binary(PetscViewer viewer,const char **name)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *name = vbinary->filename;
  PetscFunctionReturn(0);
}

/*@C
     PetscViewerFileGetMode - Gets the type of file to be open

    Not Collective

  Input Parameter:
.  viewer - the PetscViewer; must be a PETSCVIEWERBINARY, PETSCVIEWERMATLAB, PETSCVIEWERHDF5, or PETSCVIEWERASCII  PetscViewer

  Output Parameter:
.  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

  Level: advanced

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()

@*/
PetscErrorCode PetscViewerFileGetMode(PetscViewer viewer,PetscFileMode *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = PetscUseMethod(viewer,"PetscViewerFileGetMode_C",(PetscViewer,PetscFileMode*),(viewer,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscViewerBinarySetUseMPIIO - Sets a binary viewer to use MPI-IO for reading/writing. Must be called
        before PetscViewerFileSetName()

    Logically Collective on PetscViewer

    Input Parameters:
+   viewer - the PetscViewer; must be a binary
-   flg - PETSC_TRUE means MPI-IO will be used

    Options Database:
    -viewer_binary_mpiio : Flag for using MPI-IO

    Level: advanced

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen(),
          PetscViewerBinaryGetUseMPIIO()

@*/
PetscErrorCode PetscViewerBinarySetUseMPIIO(PetscViewer viewer,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscTryMethod(viewer,"PetscViewerBinarySetUseMPIIO_C",(PetscViewer,PetscBool),(viewer,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
     PetscViewerFileSetMode - Sets the type of file to be open

    Logically Collective on PetscViewer

  Input Parameters:
+  viewer - the PetscViewer; must be a a PETSCVIEWERBINARY, PETSCVIEWERMATLAB, PETSCVIEWERHDF5, or PETSCVIEWERASCII  PetscViewer
-  type - type of file
$    FILE_MODE_WRITE - create new file for output
$    FILE_MODE_READ - open existing file for input
$    FILE_MODE_APPEND - open existing file for output

  Level: advanced

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()

@*/
PetscErrorCode PetscViewerFileSetMode(PetscViewer viewer,PetscFileMode type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveEnum(viewer,type,2);
  ierr = PetscTryMethod(viewer,"PetscViewerFileSetMode_C",(PetscViewer,PetscFileMode),(viewer,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetMode_Binary(PetscViewer viewer,PetscFileMode *type)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *type = vbinary->btype;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetMode_Binary(PetscViewer viewer,PetscFileMode type)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  vbinary->btype = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetName_Binary(PetscViewer viewer,const char name[])
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (vbinary->filename) {
    /* gzip can be run after the file with the previous filename has been closed */
    ierr = PetscFree(vbinary->ogzfilename);CHKERRQ(ierr);
    ierr = PetscStrallocpy(vbinary->filename,&vbinary->ogzfilename);CHKERRQ(ierr);
    ierr = PetscFree(vbinary->filename);CHKERRQ(ierr);
    viewer->setupcalled = PETSC_FALSE;
  }
  ierr = PetscStrallocpy(name,&vbinary->filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
        Actually opens the file
*/
static PetscErrorCode PetscViewerFileSetUp_Binary(PetscViewer viewer)
{
  PetscMPIInt        rank;
  PetscErrorCode     ierr;
  size_t             len;
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  const char         *fname;
  char               bname[PETSC_MAX_PATH_LEN],*gz;
  PetscBool          found;
  PetscFileMode      type = vbinary->btype;

  PetscFunctionBegin;
  if (type == (PetscFileMode) -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call PetscViewerFileSetMode()");
  if (!vbinary->filename) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call PetscViewerFileSetName()");
  ierr = PetscViewerFileClose_Binary(viewer);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRQ(ierr);

  /* if ends in .gz strip that off and note user wants file compressed */
  vbinary->storecompressed = PETSC_FALSE;
  if (!rank && type == FILE_MODE_WRITE) {
    /* remove .gz if it ends library name */
    ierr = PetscStrstr(vbinary->filename,".gz",&gz);CHKERRQ(ierr);
    if (gz) {
      ierr = PetscStrlen(gz,&len);CHKERRQ(ierr);
      if (len == 3) {
        *gz = 0;
        vbinary->storecompressed = PETSC_TRUE;
      }
    }
  }

  /* only first processor opens file if writeable */
  if (!rank || type == FILE_MODE_READ) {

    if (type == FILE_MODE_READ) {
      /* possibly get the file from remote site or compressed file */
      ierr  = PetscFileRetrieve(PetscObjectComm((PetscObject)viewer),vbinary->filename,bname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
      fname = bname;
      /* comm below may be global as all ranks go here for FILE_MODE_READ and output 'found' of PetscFileRetrieve() is valid on all processes */
      if (!found) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_OPEN,"Cannot locate file: %s on node zero",vbinary->filename);
    } else fname = vbinary->filename;
    if (type == FILE_MODE_APPEND) { /* check if asked to append to a non-existing file */
      ierr = PetscTestFile(fname,'\0',&found);CHKERRQ(ierr);
    }

#if defined(PETSC_HAVE_O_BINARY)
    if (type == FILE_MODE_WRITE || (type == FILE_MODE_APPEND && !found) ) {
      if ((vbinary->fdes = open(fname,O_WRONLY|O_CREAT|O_TRUNC|O_BINARY,0666)) == -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot create file %s for writing",fname);
    } else if (type == FILE_MODE_READ && fname) {
      if ((vbinary->fdes = open(fname,O_RDONLY|O_BINARY,0)) == -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s for reading",fname);
    } else if (type == FILE_MODE_APPEND) {
      if ((vbinary->fdes = open(fname,O_WRONLY|O_APPEND|O_BINARY,0)) == -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s for writing",fname);
    } else if (fname) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown file type");
#else
    if (type == FILE_MODE_WRITE || (type == FILE_MODE_APPEND && !found) ) {
      if ((vbinary->fdes = creat(fname,0666)) == -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot create file %s for writing",fname);
    } else if (type == FILE_MODE_READ && fname) {
      if ((vbinary->fdes = open(fname,O_RDONLY,0)) == -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s for reading",fname);
    } else if (type == FILE_MODE_APPEND) {
      if ((vbinary->fdes = open(fname,O_WRONLY|O_APPEND,0)) == -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open file %s for writing",fname);
    } else if (fname) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Unknown file type");
#endif
  } else vbinary->fdes = -1;

  /*
      try to open info file: all processors open this file if read only
  */
  if (!vbinary->skipinfo && (!rank || type == FILE_MODE_READ)) {
    char infoname[PETSC_MAX_PATH_LEN],iname[PETSC_MAX_PATH_LEN];

    ierr = PetscStrncpy(infoname,vbinary->filename,sizeof(infoname));CHKERRQ(ierr);
    /* remove .gz if it ends library name */
    ierr = PetscStrstr(infoname,".gz",&gz);CHKERRQ(ierr);
    if (gz) {
      ierr = PetscStrlen(gz,&len);CHKERRQ(ierr);
      if (len == 3) *gz = 0;
    }

    ierr = PetscStrlcat(infoname,".info",sizeof(infoname));CHKERRQ(ierr);
    ierr = PetscFixFilename(infoname,iname);CHKERRQ(ierr);
    if (type == FILE_MODE_READ) {
      ierr = PetscFileRetrieve(PetscObjectComm((PetscObject)viewer),iname,infoname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
      if (found) {
        ierr = PetscOptionsInsertFile(PetscObjectComm((PetscObject)viewer),((PetscObject)viewer)->options,infoname,PETSC_FALSE);CHKERRQ(ierr);
      }
    } else {
      vbinary->fdes_info = fopen(infoname,"w");
      if (!vbinary->fdes_info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open .info file %s for writing",infoname);
    }
  }
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)viewer,"File: %s",vbinary->filename);
#endif
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
static PetscErrorCode PetscViewerFileSetUp_BinaryMPIIO(PetscViewer viewer)
{
  PetscMPIInt        rank;
  PetscErrorCode     ierr;
  size_t             len;
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  char               *gz;
  PetscBool          found;
  PetscFileMode      type = vbinary->btype;
  int                amode;

  PetscFunctionBegin;
  if (type == (PetscFileMode) -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call PetscViewerFileSetMode()");
  if (!vbinary->filename) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call PetscViewerFileSetName()");
  ierr = PetscViewerFileClose_BinaryMPIIO(viewer);CHKERRQ(ierr);

  vbinary->storecompressed = PETSC_FALSE;

  switch (type) {
  case FILE_MODE_READ:  amode = MPI_MODE_RDONLY; break;
  case FILE_MODE_WRITE: amode = MPI_MODE_WRONLY | MPI_MODE_CREATE; break;
  default: SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"Unsupported file mode %s",PetscFileModes[type]);
  }
  ierr = MPI_File_open(PetscObjectComm((PetscObject)viewer),vbinary->filename,amode,MPI_INFO_NULL,&vbinary->mfdes);CHKERRQ(ierr);

  /*
      try to open info file: all processors open this file if read only

      Below is identical code to the code for Binary above, should be put in separate routine
  */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)viewer),&rank);CHKERRQ(ierr);
  if (!vbinary->skipinfo && (!rank || type == FILE_MODE_READ)) {
    char infoname[PETSC_MAX_PATH_LEN],iname[PETSC_MAX_PATH_LEN];

    ierr = PetscStrncpy(infoname,vbinary->filename,sizeof(infoname));CHKERRQ(ierr);
    /* remove .gz if it ends library name */
    ierr = PetscStrstr(infoname,".gz",&gz);CHKERRQ(ierr);
    if (gz) {
      ierr = PetscStrlen(gz,&len);CHKERRQ(ierr);
      if (len == 3) *gz = 0;
    }

    ierr = PetscStrlcat(infoname,".info",sizeof(infoname));CHKERRQ(ierr);
    ierr = PetscFixFilename(infoname,iname);CHKERRQ(ierr);
    if (type == FILE_MODE_READ) {
      ierr = PetscFileRetrieve(PetscObjectComm((PetscObject)viewer),iname,infoname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
      ierr = PetscOptionsInsertFile(PetscObjectComm((PetscObject)viewer),((PetscObject)viewer)->options,infoname,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      vbinary->fdes_info = fopen(infoname,"w");
      if (!vbinary->fdes_info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot open .info file %s for writing",infoname);
    }
  }
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)viewer,"File: %s",vbinary->filename);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerBinarySetUseMPIIO_Binary(PetscViewer viewer,PetscBool flg)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscFunctionBegin;
  vbinary->usempiio = flg;
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode PetscViewerView_Binary(PetscViewer v,PetscViewer viewer)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *binary = (PetscViewer_Binary*)v->data;

  PetscFunctionBegin;
  if (binary->filename) {
    ierr = PetscViewerASCIIPrintf(viewer,"Filename: %s\n",binary->filename);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerSetUp_Binary(PetscViewer v)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *binary = (PetscViewer_Binary*)v->data;

  PetscFunctionBegin;
  if (!binary->setfromoptionscalled) { ierr = PetscViewerSetFromOptions(v);CHKERRQ(ierr); }
    
#if defined(PETSC_HAVE_MPIIO)
  if (binary->usempiio) {
    ierr = PetscViewerFileSetUp_BinaryMPIIO(v);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscViewerFileSetUp_Binary(v);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerSetFromOptions_Binary(PetscOptionItems *PetscOptionsObject,PetscViewer v)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *binary = (PetscViewer_Binary*)v->data;
  char               defaultname[PETSC_MAX_PATH_LEN];
  PetscBool          flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Binary PetscViewer Options");CHKERRQ(ierr);
  ierr = PetscSNPrintf(defaultname,PETSC_MAX_PATH_LEN-1,"binaryoutput");CHKERRQ(ierr);
  ierr = PetscOptionsString("-viewer_binary_filename","Specify filename","PetscViewerFileSetName",defaultname,defaultname,PETSC_MAX_PATH_LEN-1,&flg);CHKERRQ(ierr);
  if (flg) { ierr = PetscViewerFileSetName_Binary(v,defaultname);CHKERRQ(ierr); }
  ierr = PetscOptionsBool("-viewer_binary_skip_info","Skip writing/reading .info file","PetscViewerBinarySetSkipInfo",PETSC_FALSE,&binary->skipinfo,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewer_binary_skip_options","Skip parsing vec load options","PetscViewerBinarySetSkipOptions",PETSC_TRUE,&binary->skipoptions,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewer_binary_skip_header","Skip writing/reading header information","PetscViewerBinarySetSkipHeader",PETSC_FALSE,&binary->skipheader,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscOptionsBool("-viewer_binary_mpiio","Use MPI-IO functionality to write/read binary file","PetscViewerBinarySetUseMPIIO",PETSC_FALSE,&binary->usempiio,NULL);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_MPIUNI)
  ierr = PetscOptionsBool("-viewer_binary_mpiio","Use MPI-IO functionality to write/read binary file","PetscViewerBinarySetUseMPIIO",PETSC_FALSE,NULL,NULL);CHKERRQ(ierr);  
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
  ierr                     = PetscNewLog(v,&vbinary);CHKERRQ(ierr);
  v->data                  = (void*)vbinary;
  v->ops->setfromoptions   = PetscViewerSetFromOptions_Binary;
  v->ops->destroy          = PetscViewerDestroy_Binary;
  v->ops->view             = PetscViewerView_Binary;
  v->ops->setup            = PetscViewerSetUp_Binary;
  v->ops->flush            = NULL;
  vbinary->fdes            = 0;
#if defined(PETSC_HAVE_MPIIO)
  vbinary->mfdes           = MPI_FILE_NULL;
  vbinary->mfsub           = MPI_FILE_NULL;
#endif
  vbinary->fdes_info       = NULL;
  vbinary->skipinfo        = PETSC_FALSE;
  vbinary->skipoptions     = PETSC_TRUE;
  vbinary->skipheader      = PETSC_FALSE;
  vbinary->setfromoptionscalled = PETSC_FALSE;
  v->ops->getsubviewer     = PetscViewerGetSubViewer_Binary;
  v->ops->restoresubviewer = PetscViewerRestoreSubViewer_Binary;
  v->ops->read             = PetscViewerBinaryRead;
  vbinary->btype           = (PetscFileMode) -1;
  vbinary->storecompressed = PETSC_FALSE;
  vbinary->filename        = NULL;
  vbinary->ogzfilename     = NULL;
  vbinary->flowcontrol     = 256; /* seems a good number for Cray XT-5 */

  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetFlowControl_C",PetscViewerBinaryGetFlowControl_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetFlowControl_C",PetscViewerBinarySetFlowControl_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetSkipHeader_C",PetscViewerBinarySetSkipHeader_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetSkipHeader_C",PetscViewerBinaryGetSkipHeader_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetSkipOptions_C",PetscViewerBinaryGetSkipOptions_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetSkipOptions_C",PetscViewerBinarySetSkipOptions_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetSkipInfo_C",PetscViewerBinaryGetSkipInfo_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinarySetSkipInfo_C",PetscViewerBinarySetSkipInfo_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerBinaryGetInfoPointer_C",PetscViewerBinaryGetInfoPointer_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",PetscViewerFileSetName_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetMode_C",PetscViewerFileGetMode_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",PetscViewerFileGetName_Binary);CHKERRQ(ierr);
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
+    -viewer_binary_filename <name>
.    -viewer_binary_skip_info
.    -viewer_binary_skip_options
.    -viewer_binary_skip_header
-    -viewer_binary_mpiio

   Environmental variables:
-   PETSC_VIEWER_BINARY_FILENAME

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
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
    if (!flg) {
      ierr = PetscStrcpy(fname,"binaryoutput");
      if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
    }
    ierr = PetscViewerBinaryOpen(ncomm,fname,FILE_MODE_WRITE,&viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
    ierr = MPI_Comm_set_attr(ncomm,Petsc_Viewer_Binary_keyval,(void*)viewer);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
  }
  ierr = PetscCommDestroy(&ncomm);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
  PetscFunctionReturn(viewer);
}
