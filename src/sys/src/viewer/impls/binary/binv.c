#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: binv.c,v 1.48 1998/10/29 04:03:07 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "sys.h"
#include "pinclude/pviewer.h"
#include <fcntl.h>
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (HAVE_IO_H)
#include <io.h>
#endif

struct _p_Viewer {
  VIEWERHEADER
  int          fdes;            /* file descriptor */
  FILE         *fdes_info;      /* optional file containing info on binary file*/
  PetscTruth   compressedfile;  /* file was compressed; should compress on close */
  char         filename[1024];
};

#undef __FUNC__  
#define __FUNC__ "ViewerBinaryGetDescriptor"
/*@C
    ViewerBinaryGetDescriptor - Extracts the file descriptor from a viewer.

    Not Collective

+   viewer - viewer context, obtained from ViewerFileOpenBinary()
-   fdes - file descriptor

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, descriptor

.seealso: ViewerFileOpenBinary(),ViewerBinaryGetInfoPointer()
@*/
int ViewerBinaryGetDescriptor(Viewer viewer,int *fdes)
{
  PetscFunctionBegin;
  *fdes = viewer->fdes;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerBinaryGetInfoPointer"
/*@C
    ViewerBinaryGetInfoPointer - Extracts the file pointer for the ASCII
          info file associated with a binary file.

    Not Collective

+   viewer - viewer context, obtained from ViewerFileOpenBinary()
-   file - file pointer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, descriptor

.seealso: ViewerFileOpenBinary(),ViewerBinaryGetDescriptor()
@*/
int ViewerBinaryGetInfoPointer(Viewer viewer,FILE **file)
{
  PetscFunctionBegin;
  *file = viewer->fdes_info;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_BinaryFile"
int ViewerDestroy_BinaryFile(Viewer v)
{
  int    rank,ierr;

  PetscFunctionBegin;
  MPI_Comm_rank(v->comm,&rank);
  if (!rank) close(v->fdes);
  if (!rank && v->fdes_info) fclose(v->fdes_info);

  if (v->compressedfile) {
    FILE *fp;
    char command[1024],buffer[1024];

    /* compress the file back up */
    PetscStrcpy(command,"\rm -f ");
    PetscStrcat(command,v->filename);
    fp = popen(command,"r");
    if (!fp) SETERRQ(1,1,"Cannot removed uncompressed file");
    fgets(buffer,1024,fp);
    ierr = pclose(fp);
    if (ierr) { 
      SETERRQ(1,1,"Unable to remove uncompressed binary file");
    }
  }
  PLogObjectDestroy((PetscObject)v);
  PetscHeaderDestroy((PetscObject)v);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerFileOpenBinary"
/*@C
   ViewerFileOpenBinary - Opens a file for binary input/output.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  name - name of file 
-  type - type of file
$    BINARY_CREATE - create new file for binary output
$    BINARY_RDONLY - open existing file for binary input
$    BINARY_WRONLY - open existing file for binary output

   Output Parameter:
.  binv - viewer for binary input/output to use with the specified file

   Note:
   This viewer can be destroyed with ViewerDestroy().

.keywords: binary, file, open, input, output

.seealso: ViewerFileOpenASCII(), ViewerSetFormat(), ViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), ViewerBinaryGetDescriptor(),
          ViewerBinaryGetInfoPointer()
@*/
int ViewerFileOpenBinary(MPI_Comm comm,const char name[],ViewerBinaryType type,Viewer *binv)
{  
  int        rank,ierr;
  Viewer     v;
  PetscTruth exists;
  const char *fname;
  static int id = 0;

  PetscFunctionBegin;
  PetscHeaderCreate(v,_p_Viewer,int,VIEWER_COOKIE,BINARY_FILE_VIEWER,comm,ViewerDestroy,0);
  PLogObjectCreate(v);
  v->destroy        = ViewerDestroy_BinaryFile;
  v->flush          = 0;
  v->iformat        = 0;
  v->compressedfile = PETSC_FALSE;
  *binv             = v;

  MPI_Comm_rank(comm,&rank);

  /* only first processor opens file if writeable */
  if (!rank || type == BINARY_RDONLY) {

    if (type == BINARY_RDONLY){
      /*
          Check if the file exists
      */
      ierr  = PetscTestFile(name,'r',&exists);CHKERRQ(ierr);
      if (!exists) {
        char *tname;
        int  size;

        MPI_Comm_size(comm,&size);
        if (size == 1) {
          tname = (char *) PetscMalloc((3+PetscStrlen(name))*sizeof(char));CHKPTRQ(tname);
          PetscStrcpy(tname,name);
          PetscStrcat(tname,".gz");
          ierr  = PetscTestFile(tname,'r',&exists);CHKERRQ(ierr);
          if (exists) { /* try to uncompress it */
            FILE *fp;
            char command[1024],buffer[1024],sid[16];

            /* should also include PID for uniqueness */
            PetscStrcpy(v->filename,"/tmp/petscbinary.tmp");
            sprintf(sid,"%d",id++);
            PetscStrcat(v->filename,sid);

            PetscStrcpy(command,"gunzip -c ");
            PetscStrcat(command,name);
            PetscStrcat(command," > ");
            PetscStrcat(command,v->filename);

            PLogInfo(*binv,"Uncompressing file %s into %s\n",name,v->filename);
            fp = popen(command,"r");
            if (!fp) SETERRQ(1,1,"Cannot uncompress file");
            fgets(buffer,1024,fp);
            ierr = pclose(fp);
            if (ierr) { 
              SETERRQ(1,1,"Unable to uncompress compressed binary file");
            }
            PLogInfo(*binv,"Uncompressed file %s\n %s\n",name,buffer);
            v->compressedfile = PETSC_TRUE;

          }
          PetscFree(tname);
          fname = v->filename;
        } else {
          SETERRQ(PETSC_ERR_FILE_OPEN,1,"Cannot open file for reading, does not exist");
        }
      } else {
        fname = name;
      }
  } else {
    fname = name;
  }

#if defined(PARCH_nt_gnu) || defined(PARCH_nt) 
    if (type == BINARY_CREATE) {
      if ((v->fdes = open(fname,O_WRONLY|O_CREAT|O_TRUNC|O_BINARY,0666 )) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot create file for writing");
      }
    } else if (type == BINARY_RDONLY) {
      if ((v->fdes = open(fname,O_RDONLY|O_BINARY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for reading");
      }
    } else if (type == BINARY_WRONLY) {
      if ((v->fdes = open(fname,O_WRONLY|O_BINARY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for writing");
      }
    } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown file type");
#else
    if (type == BINARY_CREATE) {
      if ((v->fdes = creat(fname,0666)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot create file for writing");
      }
    } else if (type == BINARY_RDONLY) {
      if ((v->fdes = open(fname,O_RDONLY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for reading");
      }
    } else if (type == BINARY_WRONLY) {
      if ((v->fdes = open(fname,O_WRONLY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for writing");
      }
    } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown file type");
#endif
  } else v->fdes = -1;
  v->format    = 0;

  /* 
      try to open info file: all processors open this file
  */
  if (type == BINARY_RDONLY) {
    char infoname[256],iname[256];
  
    ierr = PetscStrcpy(infoname,name);CHKERRQ(ierr);
    ierr = PetscStrcat(infoname,".info");CHKERRQ(ierr);
    ierr = PetscFixFilename(infoname,iname); CHKERRQ(ierr);
    v->fdes_info = fopen(iname,"r");
  }

#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  PetscFunctionReturn(0);
}





