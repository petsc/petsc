#define PETSC_DLL
#include "private/viewerimpl.h"    /*I   "petscsys.h"   I*/
#include <fcntl.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (PETSC_HAVE_IO_H)
#include <io.h>
#endif

typedef struct  {
  int           fdes;            /* file descriptor, ignored if using MPI IO */
#if defined(PETSC_HAVE_MPIIO)
  PetscTruth    MPIIO;
  MPI_File      mfdes;           /* ignored unless using MPI IO */    
  MPI_Offset    moff;
#endif
  PetscFileMode btype;           /* read or write? */
  FILE          *fdes_info;      /* optional file containing info on binary file*/
  PetscTruth    storecompressed; /* gzip the write binary file when closing it*/
  char          *filename;
  PetscTruth    skipinfo;        /* Don't create info file for writing; don't use for reading */
  PetscTruth    skipoptions;     /* don't use PETSc options database when loading */
  PetscInt      flowcontrol;     /* allow only <flowcontrol> messages outstanding at a time while doing IO */
} PetscViewer_Binary;

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerGetSingleton_Binary" 
PetscErrorCode PetscViewerGetSingleton_Binary(PetscViewer viewer,PetscViewer *outviewer)
{
  int                rank;
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data,*obinary;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)viewer)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr    = PetscViewerCreate(PETSC_COMM_SELF,outviewer);CHKERRQ(ierr);
    ierr    = PetscViewerSetType(*outviewer,PETSC_VIEWER_BINARY);CHKERRQ(ierr);
    obinary = (PetscViewer_Binary*)(*outviewer)->data;
    ierr    = PetscMemcpy(obinary,vbinary,sizeof(PetscViewer_Binary));CHKERRQ(ierr);
  } else {
    *outviewer = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerRestoreSingleton_Binary" 
PetscErrorCode PetscViewerRestoreSingleton_Binary(PetscViewer viewer,PetscViewer *outviewer)
{
  PetscErrorCode ierr;
  PetscErrorCode rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)viewer)->comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscFree((*outviewer)->data);CHKERRQ(ierr);
    ierr = PetscHeaderDestroy(*outviewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryGetMPIIOOffset" 
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

  Concepts: file descriptor^getting
  Concepts: PetscViewerBinary^accessing file descriptor

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetMPIIOOffset(PetscViewer viewer,MPI_Offset *off)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *off = vbinary->moff;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryAddMPIIOOffset" 
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

  Concepts: file descriptor^getting
  Concepts: PetscViewerBinary^accessing file descriptor

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryAddMPIIOOffset(PetscViewer viewer,MPI_Offset off)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  vbinary->moff += off;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryGetMPIIODescriptor" 
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

  Concepts: file descriptor^getting
  Concepts: PetscViewerBinary^accessing file descriptor

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetMPIIODescriptor(PetscViewer viewer,MPI_File *fdes)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *fdes = vbinary->mfdes;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryGetMPIIO" 
/*@C
    PetscViewerBinaryGetMPIIO - Returns PETSC_TRUE if the binary viewer is an MPI viewer.

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
-   flg - PETSC_TRUE if MPI IO is being used

    Level: advanced

    Fortran Note:
    This routine is not supported in Fortran.

  Concepts: file descriptor^getting
  Concepts: PetscViewerBinary^accessing file descriptor

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetInfoPointer()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetMPIIO(PetscViewer viewer,PetscTruth *flg)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *flg = vbinary->MPIIO;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryGetFlowControl" 
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
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetFlowControl(PetscViewer viewer,PetscInt *fc)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *fc = vbinary->flowcontrol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryGetDescriptor" 
/*@C
    PetscViewerBinaryGetDescriptor - Extracts the file descriptor from a PetscViewer.

    Not Collective

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

  Concepts: file descriptor^getting
  Concepts: PetscViewerBinary^accessing file descriptor

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetInfoPointer()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetDescriptor(PetscViewer viewer,int *fdes)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *fdes = vbinary->fdes;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinarySkipInfo" 
/*@
    PetscViewerBinarySkipInfo - Binary file will not have .info file created with it

    Not Collective

    Input Paramter:
.   viewer - PetscViewer context, obtained from PetscViewerCreate()

    Options Database Key:
.   -viewer_binary_skip_info

    Level: advanced

    Notes: This must be called after PetscViewerSetType() but before PetscViewerFileSetName(). If you use PetscViewerBinaryOpen() then 
    you can only skip the info file with the -viewer_binary_skip_info flag. To use the function you must open the
    viewer with PetscViewerCreate(), PetscViewerSetType(), PetscViewerFileSetName().

    The .info contains meta information about the data in the binary file, for example the block size if it was
    set for a vector or matrix.

   Concepts: PetscViewerBinary^accessing info file

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySetSkipOptions(),
          PetscViewerBinaryGetSkipOptions()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinarySkipInfo(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  vbinary->skipinfo = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinarySetSkipOptions" 
/*@
    PetscViewerBinarySetSkipOptions - do not use the PETSc options database when loading objects

    Not Collective

    Input Parameters:
+   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()
-   skip - PETSC_TRUE means do not use

    Options Database Key:
.   -viewer_binary_skip_options

    Level: advanced

    Notes: This must be called after PetscViewerSetType()

   Concepts: PetscViewerBinary^accessing info file

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySkipInfo(),
          PetscViewerBinaryGetSkipOptions()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinarySetSkipOptions(PetscViewer viewer,PetscTruth skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  vbinary->skipoptions = skip;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryGetSkipOptions" 
/*@
    PetscViewerBinaryGetSkipOptions - checks if viewer uses the PETSc options database when loading objects

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   skip - PETSC_TRUE means do not use

    Level: advanced

    Notes: This must be called after PetscViewerSetType()

   Concepts: PetscViewerBinary^accessing info file

.seealso: PetscViewerBinaryOpen(), PetscViewerBinaryGetDescriptor(), PetscViewerBinarySkipInfo(),
          PetscViewerBinarySetSkipOptions()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetSkipOptions(PetscViewer viewer,PetscTruth *skip)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *skip = vbinary->skipoptions;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryGetInfoPointer" 
/*@C
    PetscViewerBinaryGetInfoPointer - Extracts the file pointer for the ASCII
          info file associated with a binary file.

    Not Collective

    Input Parameter:
.   viewer - PetscViewer context, obtained from PetscViewerBinaryOpen()

    Output Parameter:
.   file - file pointer

    Level: advanced

    Notes:
      For writable binary PetscViewers, the descriptor will only be valid for the 
    first processor in the communicator that shares the PetscViewer.
 
    Fortran Note:
    This routine is not supported in Fortran.

  Concepts: PetscViewerBinary^accessing info file

.seealso: PetscViewerBinaryOpen(),PetscViewerBinaryGetDescriptor()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryGetInfoPointer(PetscViewer viewer,FILE **file)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *file = vbinary->fdes_info;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_Binary" 
PetscErrorCode PetscViewerDestroy_Binary(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  PetscErrorCode     ierr;
  PetscMPIInt        rank;
  int                err;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)v)->comm,&rank);CHKERRQ(ierr);
  if ((!rank || vbinary->btype == FILE_MODE_READ) && vbinary->fdes) {
    close(vbinary->fdes);
    if (!rank && vbinary->storecompressed) {
      char par[PETSC_MAX_PATH_LEN],buf[PETSC_MAX_PATH_LEN];
      FILE *fp;
      /* compress the file */
      ierr = PetscStrcpy(par,"gzip -f ");CHKERRQ(ierr);
      ierr = PetscStrcat(par,vbinary->filename);CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)
      ierr = PetscPOpen(PETSC_COMM_SELF,PETSC_NULL,par,"r",&fp);CHKERRQ(ierr);
      if (fgets(buf,1024,fp)) {
        SETERRQ2(PETSC_ERR_LIB,"Error from command %s\n%s",par,buf);
      }
      ierr = PetscPClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
#else
      SETERRQ(PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
    }
  }
  if (vbinary->fdes_info) {
    err = fclose(vbinary->fdes_info);
    if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
  }
  ierr = PetscStrfree(vbinary->filename);CHKERRQ(ierr);
  ierr = PetscFree(vbinary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_MPIIO" 
PetscErrorCode PetscViewerDestroy_MPIIO(PetscViewer v)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)v->data;
  PetscErrorCode     ierr;
  int                err;

  PetscFunctionBegin;
  if (vbinary->mfdes) {
    ierr = MPI_File_close(&vbinary->mfdes);CHKERRQ(ierr);
  }
  if (vbinary->fdes_info) {
    err = fclose(vbinary->fdes_info);
    if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
  }
  ierr = PetscStrfree(vbinary->filename);CHKERRQ(ierr);
  ierr = PetscFree(vbinary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryCreate"
/*@
   PetscViewerBinaryCreate - Create a binary viewer.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI communicator

   Output Parameter:
.  binv - PetscViewer for binary input/output

   Level: beginner
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryCreate(MPI_Comm comm,PetscViewer *binv)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,binv);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*binv,PETSC_VIEWER_BINARY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryOpen" 
/*@C
   PetscViewerBinaryOpen - Opens a file for binary input/output.

   Collective on MPI_Comm

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
+    -viewer_binary_skip_info
-    -viewer_binary_skip_options

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

   Concepts: binary files
   Concepts: PetscViewerBinary^creating
   Concepts: gzip
   Concepts: accessing remote file
   Concepts: remote file

.seealso: PetscViewerASCIIOpen(), PetscViewerSetFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscViewerBinaryRead()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryOpen(MPI_Comm comm,const char name[],PetscFileMode type,PetscViewer *binv)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm,binv);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*binv,PETSC_VIEWER_BINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*binv,type);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*binv,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPIIO)
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryMPIIO" 
static PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryMPIIO(PetscViewer viewer,void *data,PetscInt count,PetscDataType dtype,PetscTruth write)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscErrorCode     ierr;
  MPI_Datatype       mdtype;
  PetscMPIInt        cnt = PetscMPIIntCast(count);
  MPI_Status         status;
  MPI_Aint           ul,dsize;

  PetscFunctionBegin;
  ierr = PetscDataTypeToMPIDataType(dtype,&mdtype);CHKERRQ(ierr);
  ierr = MPI_File_set_view(vbinary->mfdes,vbinary->moff,mdtype,mdtype,(char *)"native",MPI_INFO_NULL);CHKERRQ(ierr);
  if (write) {
    ierr = MPIU_File_write_all(vbinary->mfdes,data,cnt,mdtype,&status);CHKERRQ(ierr);
  } else {
    ierr = MPIU_File_read_all(vbinary->mfdes,data,cnt,mdtype,&status);CHKERRQ(ierr);
  }
  ierr = MPI_Type_get_extent(mdtype,&ul,&dsize);CHKERRQ(ierr);
  vbinary->moff += dsize*cnt;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryRead" 
/*@C
   PetscViewerBinaryRead - Reads from a binary file, all processors get the same result

   Collective on MPI_Comm

   Input Parameters:
+  viewer - the binary viewer
.  data - location to write the data
.  count - number of items of data to read
-  datatype - type of data to read

   Level: beginner

   Concepts: binary files

.seealso: PetscViewerASCIIOpen(), PetscViewerSetFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscBinaryViewerRead()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryRead(PetscViewer viewer,void *data,PetscInt count,PetscDataType dtype)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->MPIIO) {
    ierr = PetscViewerBinaryMPIIO(viewer,data,count,dtype,PETSC_FALSE);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscBinarySynchronizedRead(((PetscObject)viewer)->comm,vbinary->fdes,data,count,dtype);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryWrite" 
/*@C
   PetscViewerBinaryWrite - writes to a binary file, only from the first process

   Collective on MPI_Comm

   Input Parameters:
+  viewer - the binary viewer
.  data - location of data
.  count - number of items of data to read
.  istemp - data may be overwritten
-  datatype - type of data to read

   Level: beginner

   Notes: because byte-swapping may be done on the values in data it cannot be declared const

   Concepts: binary files

.seealso: PetscViewerASCIIOpen(), PetscViewerSetFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(), PetscDataType
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscBinaryViewerRead()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryWrite(PetscViewer viewer,void *data,PetscInt count,PetscDataType dtype,PetscTruth istemp)
{
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPIIO)
  if (vbinary->MPIIO) {
    ierr = PetscViewerBinaryMPIIO(viewer,data,count,dtype,PETSC_TRUE);CHKERRQ(ierr);
  } else {
#endif
    ierr = PetscBinarySynchronizedWrite(((PetscObject)viewer)->comm,vbinary->fdes,data,count,dtype,istemp);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinaryWriteStringArray" 
/*@C
   PetscViewerBinaryWriteStringArray - writes to a binary file, only from the first process an array of strings

   Collective on MPI_Comm

   Input Parameters:
+  viewer - the binary viewer
-  data - location of the array of strings


   Level: intermediate

   Concepts: binary files

    Notes: array of strings is null terminated

.seealso: PetscViewerASCIIOpen(), PetscViewerSetFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscBinaryViewerRead()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryWriteStringArray(PetscViewer viewer,char **data)
{
  PetscErrorCode     ierr;
  PetscInt           i,n = 0,*sizes;

  /* count number of strings */
  while (data[n++]);
  n--;
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&sizes);CHKERRQ(ierr);
  sizes[0] = n;
  for (i=0; i<n; i++) {
    size_t tmp;
    ierr       = PetscStrlen(data[i],&tmp);CHKERRQ(ierr);
    sizes[i+1] = tmp + 1;   /* size includes space for the null terminator */
  }
  ierr = PetscViewerBinaryWrite(viewer,sizes,n+1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscViewerBinaryWrite(viewer,data[i],sizes[i+1],PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = PetscFree(sizes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerBinaryReadStringArray - reads a binary file an array of strings

   Collective on MPI_Comm

   Input Parameter:
.  viewer - the binary viewer

   Output Parameter:
.  data - location of the array of strings

   Level: intermediate

   Concepts: binary files

    Notes: array of strings is null terminated

.seealso: PetscViewerASCIIOpen(), PetscViewerSetFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer, PetscBinaryViewerRead()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinaryReadStringArray(PetscViewer viewer,char ***data)
{
  PetscErrorCode     ierr;
  PetscInt           i,n,*sizes,N = 0;

  /* count number of strings */
  ierr = PetscViewerBinaryRead(viewer,&n,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&sizes);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,sizes,n,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    N += sizes[i];
  }
  ierr = PetscMalloc((n+1)*sizeof(char*) + N*sizeof(char),data);CHKERRQ(ierr);
  (*data)[0] = (char*)((*data) + n + 1);
  for (i=1; i<n; i++) {
    (*data)[i] = (*data)[i-1] + sizes[i-1];
  }
  ierr = PetscViewerBinaryRead(viewer,(*data)[0],N,PETSC_CHAR);CHKERRQ(ierr);
  (*data)[n] = 0;
  ierr = PetscFree(sizes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileGetMode" 
/*@C
     PetscViewerFileGetMode - Gets the type of file to be open

    Collective on PetscViewer

  Input Parameter:
.  viewer - the PetscViewer; must be a binary, Matlab, hdf, or netcdf PetscViewer

  Output Parameter:
.  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

  Level: advanced

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileGetMode(PetscViewer viewer,PetscFileMode *type)
{
  PetscErrorCode ierr,(*f)(PetscViewer,PetscFileMode*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(type,2);
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"PetscViewerFileGetMode_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(viewer,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinarySetMPIIO" 
/*@
     PetscViewerBinarySetMPIIO - Sets a binary viewer to use MPI IO for reading/writing. Must be called
        before PetscViewerFileSetName()

    Collective on PetscViewer

  Input Parameters:
.  viewer - the PetscViewer; must be a binary

   Notes: turns off the default usage of the .info file since that is not scalable

  Level: advanced

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinarySetMPIIO(PetscViewer viewer)
{
  PetscErrorCode ierr,(*f)(PetscViewer);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"PetscViewerBinarySetMPIIO_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@C
     PetscViewerFileSetMode - Sets the type of file to be open

    Collective on PetscViewer

  Input Parameters:
+  viewer - the PetscViewer; must be a binary, Matlab, hdf, or netcdf PetscViewer
-  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

  Level: advanced

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetMode(PetscViewer viewer,PetscFileMode type)
{
  PetscErrorCode ierr,(*f)(PetscViewer,PetscFileMode);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(viewer,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileGetMode_Binary" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileGetMode_Binary(PetscViewer viewer,PetscFileMode *type)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  *type = vbinary->btype;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetMode_Binary" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetMode_Binary(PetscViewer viewer,PetscFileMode type)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;

  PetscFunctionBegin;
  vbinary->btype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
        Actually opens the file 
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetName_Binary" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetName_Binary(PetscViewer viewer,const char name[])
{
  PetscMPIInt         rank;
  PetscErrorCode      ierr;
  size_t              len;
  PetscViewer_Binary  *vbinary = (PetscViewer_Binary*)viewer->data;
  const char          *fname;
  char                bname[PETSC_MAX_PATH_LEN],*gz;
  PetscTruth          found;
  PetscFileMode       type = vbinary->btype;

  PetscFunctionBegin;
  if (type == (PetscFileMode) -1) {
    SETERRQ(PETSC_ERR_ORDER,"Must call PetscViewerBinarySetFileType() before PetscViewerFileSetName()");
  }
  ierr = PetscOptionsGetTruth(((PetscObject)viewer)->prefix,"-viewer_binary_skip_info",&vbinary->skipinfo,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(((PetscObject)viewer)->prefix,"-viewer_binary_skip_options",&vbinary->skipoptions,PETSC_NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(((PetscObject)viewer)->comm,&rank);CHKERRQ(ierr);

  /* copy name so we can edit it */
  ierr = PetscStrallocpy(name,&vbinary->filename);CHKERRQ(ierr);

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

    if (type == FILE_MODE_READ){
      /* possibly get the file from remote site or compressed file */
      ierr  = PetscFileRetrieve(((PetscObject)viewer)->comm,vbinary->filename,bname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
      fname = bname;
      if (!rank && !found) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot locate file: %s on node zero",vbinary->filename);
      } else if (!found) {
        ierr = PetscInfo(viewer,"Nonzero processor did not locate readonly file\n");CHKERRQ(ierr);
        fname = 0;
      }
    } else {
      fname = vbinary->filename;
    }

#if defined(PETSC_HAVE_O_BINARY)
    if (type == FILE_MODE_WRITE) {
      if ((vbinary->fdes = open(fname,O_WRONLY|O_CREAT|O_TRUNC|O_BINARY,0666)) == -1) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot create file %s for writing",fname);
      }
    } else if (type == FILE_MODE_READ && fname) {
      if ((vbinary->fdes = open(fname,O_RDONLY|O_BINARY,0)) == -1) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open file %s for reading",fname);
      }
    } else if (type == FILE_MODE_APPEND) {
      if ((vbinary->fdes = open(fname,O_WRONLY|O_APPEND|O_BINARY,0)) == -1) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open file %s for writing",fname);
      }
    } else if (fname) {
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown file type");
    }
#else
    if (type == FILE_MODE_WRITE) {
      if ((vbinary->fdes = creat(fname,0666)) == -1) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot create file %s for writing",fname);
      }
    } else if (type == FILE_MODE_READ && fname) {
      if ((vbinary->fdes = open(fname,O_RDONLY,0)) == -1) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open file %s for reading",fname);
      }
    } else if (type == FILE_MODE_APPEND) {
      if ((vbinary->fdes = open(fname,O_WRONLY|O_APPEND,0)) == -1) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open file %s for writing",fname);
      }
    } else if (fname) {
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown file type");
    }
#endif
  } else vbinary->fdes = -1;
  viewer->format = PETSC_VIEWER_NOFORMAT;

  /* 
      try to open info file: all processors open this file if read only
  */
  if (!vbinary->skipinfo && (!rank || type == FILE_MODE_READ)) {
    char infoname[PETSC_MAX_PATH_LEN],iname[PETSC_MAX_PATH_LEN];
  
    ierr = PetscStrcpy(infoname,name);CHKERRQ(ierr);
    /* remove .gz if it ends library name */
    ierr = PetscStrstr(infoname,".gz",&gz);CHKERRQ(ierr);
    if (gz) {
      ierr = PetscStrlen(gz,&len);CHKERRQ(ierr);
      if (len == 3) {
        *gz = 0;
      } 
    }
    
    ierr = PetscStrcat(infoname,".info");CHKERRQ(ierr);
    ierr = PetscFixFilename(infoname,iname);CHKERRQ(ierr);
    if (type == FILE_MODE_READ) {
      ierr = PetscFileRetrieve(((PetscObject)viewer)->comm,iname,infoname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
      ierr = PetscOptionsInsertFile(((PetscObject)viewer)->comm,infoname,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      vbinary->fdes_info = fopen(infoname,"w");
      if (!vbinary->fdes_info) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open .info file %s for writing",infoname);
      }
    }
  }

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)viewer,"File: %s",name);
#endif
  PetscFunctionReturn(0);
}
EXTERN_C_END

#if defined(PETSC_HAVE_MPIIO)
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerFileSetName_MPIIO" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerFileSetName_MPIIO(PetscViewer viewer,const char name[])
{
  PetscMPIInt         rank;
  PetscErrorCode      ierr;
  size_t              len;
  PetscViewer_Binary  *vbinary = (PetscViewer_Binary*)viewer->data;
  char                *gz;
  PetscTruth          found;
  PetscFileMode       type = vbinary->btype;

  PetscFunctionBegin;
  if (type == (PetscFileMode) -1) {
    SETERRQ(PETSC_ERR_ORDER,"Must call PetscViewerBinarySetFileType() before PetscViewerFileSetName()");
  }
  ierr = PetscOptionsGetTruth(((PetscObject)viewer)->prefix,"-viewer_binary_skip_info",&vbinary->skipinfo,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(((PetscObject)viewer)->prefix,"-viewer_binary_skip_options",&vbinary->skipoptions,PETSC_NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(((PetscObject)viewer)->comm,&rank);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&vbinary->filename);CHKERRQ(ierr);
  vbinary->storecompressed = PETSC_FALSE;


  /* only first processor opens file if writeable */
  if (type == FILE_MODE_READ) {
    MPI_File_open(((PetscObject)viewer)->comm,vbinary->filename,MPI_MODE_RDONLY,MPI_INFO_NULL,&vbinary->mfdes);CHKERRQ(ierr);
  } else if (type == FILE_MODE_WRITE) {
    MPI_File_open(((PetscObject)viewer)->comm,vbinary->filename,MPI_MODE_WRONLY | MPI_MODE_CREATE,MPI_INFO_NULL,&vbinary->mfdes);CHKERRQ(ierr);
  }
  viewer->format = PETSC_VIEWER_NOFORMAT;

  /* 
      try to open info file: all processors open this file if read only

      Below is identical code to the code for Binary above, should be put in seperate routine
  */
  if (!vbinary->skipinfo && (!rank || type == FILE_MODE_READ)) {
    char infoname[PETSC_MAX_PATH_LEN],iname[PETSC_MAX_PATH_LEN];
  
    ierr = PetscStrcpy(infoname,name);CHKERRQ(ierr);
    /* remove .gz if it ends library name */
    ierr = PetscStrstr(infoname,".gz",&gz);CHKERRQ(ierr);
    if (gz) {
      ierr = PetscStrlen(gz,&len);CHKERRQ(ierr);
      if (len == 3) {
        *gz = 0;
      } 
    }
    
    ierr = PetscStrcat(infoname,".info");CHKERRQ(ierr);
    ierr = PetscFixFilename(infoname,iname);CHKERRQ(ierr);
    if (type == FILE_MODE_READ) {
      ierr = PetscFileRetrieve(((PetscObject)viewer)->comm,iname,infoname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
      ierr = PetscOptionsInsertFile(((PetscObject)viewer)->comm,infoname,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      vbinary->fdes_info = fopen(infoname,"w");
      if (!vbinary->fdes_info) {
        SETERRQ1(PETSC_ERR_FILE_OPEN,"Cannot open .info file %s for writing",infoname);
      }
    }
  }

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)viewer,"File: %s",name);
#endif
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerBinarySetMPIIO_Binary" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerBinarySetMPIIO_Binary(PetscViewer viewer)
{
  PetscViewer_Binary *vbinary = (PetscViewer_Binary*)viewer->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (vbinary->filename) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call before calling PetscViewerFileSetName()");
  viewer->ops->destroy = PetscViewerDestroy_MPIIO;
  vbinary->MPIIO       = PETSC_TRUE;
  /*  vbinary->skipinfo    = PETSC_TRUE; */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)viewer,"PetscViewerFileSetName_C","PetscViewerFileSetName_MPIIO",PetscViewerFileSetName_MPIIO);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_Binary" 
PetscErrorCode PETSC_DLLEXPORT PetscViewerCreate_Binary(PetscViewer v)
{  
  PetscErrorCode     ierr;
  PetscViewer_Binary *vbinary;
#if defined(PETSC_HAVE_MPIIO)
  PetscTruth         useMPIIO = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  ierr               = PetscNewLog(v,PetscViewer_Binary,&vbinary);CHKERRQ(ierr);
  v->data            = (void*)vbinary;
  v->ops->destroy    = PetscViewerDestroy_Binary;
  v->ops->flush      = 0;
  v->iformat         = 0;
  vbinary->fdes_info = 0;
  vbinary->fdes      = 0;
  vbinary->skipinfo        = PETSC_FALSE;
  vbinary->skipoptions     = PETSC_TRUE;
  v->ops->getsingleton     = PetscViewerGetSingleton_Binary;
  v->ops->restoresingleton = PetscViewerRestoreSingleton_Binary;
  vbinary->btype           = (PetscFileMode) -1; 
  vbinary->storecompressed = PETSC_FALSE;
  vbinary->filename        = 0;
  vbinary->flowcontrol     = 256; /* seems a good number for Cray XT-5 */

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerFileSetName_C",
                                    "PetscViewerFileSetName_Binary",
                                     PetscViewerFileSetName_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerFileSetMode_C",
                                    "PetscViewerFileSetMode_Binary",
                                     PetscViewerFileSetMode_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerFileGetMode_C",
                                    "PetscViewerFileGetMode_Binary",
                                     PetscViewerFileGetMode_Binary);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPIIO)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)v,"PetscViewerBinarySetMPIIO_C",
                                    "PetscViewerBinarySetMPIIO_Binary",
                                     PetscViewerBinarySetMPIIO_Binary);CHKERRQ(ierr);
 
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-viewer_binary_mpiio",&useMPIIO,PETSC_NULL);CHKERRQ(ierr);
  if (useMPIIO) {
    ierr = PetscViewerBinarySetMPIIO(v);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}
EXTERN_C_END


/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Binary_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscViewer.
*/
static int Petsc_Viewer_Binary_keyval = MPI_KEYVAL_INVALID;

#undef __FUNCT__  
#define __FUNCT__ "PETSC_VIEWER_BINARY_"  
/*@C
     PETSC_VIEWER_BINARY_ - Creates a binary PetscViewer shared by all processors 
                     in a communicator.

     Collective on MPI_Comm

     Input Parameter:
.    comm - the MPI communicator to share the binary PetscViewer
    
     Level: intermediate

   Options Database Keys:
+    -viewer_binary_filename <name>
.    -viewer_binary_skip_info
-    -viewer_binary_skip_options

   Environmental variables:
-   PETSC_VIEWER_BINARY_FILENAME

     Notes:
     Unlike almost all other PETSc routines, PETSC_VIEWER_BINARY_ does not return 
     an error code.  The binary PetscViewer is usually used in the form
$       XXXView(XXX object,PETSC_VIEWER_BINARY_(comm));

.seealso: PETSC_VIEWER_BINARY_WORLD, PETSC_VIEWER_BINARY_SELF, PetscViewerBinaryOpen(), PetscViewerCreate(),
          PetscViewerDestroy()
@*/
PetscViewer PETSC_DLLEXPORT PETSC_VIEWER_BINARY_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscTruth     flg;
  PetscViewer    viewer;
  char           fname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  if (Petsc_Viewer_Binary_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Binary_keyval,0);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
  }
  ierr = MPI_Attr_get(comm,Petsc_Viewer_Binary_keyval,(void **)&viewer,(int*)&flg);
  if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
  if (!flg) { /* PetscViewer not yet created */
    ierr = PetscOptionsGetenv(comm,"PETSC_VIEWER_BINARY_FILENAME",fname,PETSC_MAX_PATH_LEN,&flg);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
    if (!flg) {
      ierr = PetscStrcpy(fname,"binaryoutput");
      if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
    }
    ierr = PetscViewerBinaryOpen(comm,fname,FILE_MODE_WRITE,&viewer); 
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
    ierr = PetscObjectRegisterDestroy((PetscObject)viewer);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
    ierr = MPI_Attr_put(comm,Petsc_Viewer_Binary_keyval,(void*)viewer);
    if (ierr) {PetscError(__LINE__,"PETSC_VIEWER_BINARY_",__FILE__,__SDIR__,1,1," ");PetscFunctionReturn(0);}
  } 
  PetscFunctionReturn(viewer);
}







