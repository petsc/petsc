#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: binv.c,v 1.49 1998/10/29 23:17:27 bsmith Exp bsmith $";
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
  const char *fname,bname[1024];
  static int id = 0;
  PetscTruth found;

  PetscFunctionBegin;
  PetscHeaderCreate(v,_p_Viewer,int,VIEWER_COOKIE,BINARY_FILE_VIEWER,comm,ViewerDestroy,0);
  PLogObjectCreate(v);
  v->destroy        = ViewerDestroy_BinaryFile;
  v->flush          = 0;
  v->iformat        = 0;
  *binv             = v;

  MPI_Comm_rank(comm,&rank);

  /* only first processor opens file if writeable */
  if (!rank || type == BINARY_RDONLY) {

    if (type == BINARY_RDONLY){
      /* possibly get the file from remote site or compressed file */
      ierr  = PetscFileRetrieve(comm,name,bname,1024,&found);CHKERRQ(ierr);
      if (!found) {
        fprintf(stderr,"filename: %s\n",name);
        SETERRQ(1,1,"Cannot locate file");
      }
      fname = bname;
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
    char infoname[256],iname[256],*gz;
  
    ierr = PetscStrcpy(infoname,name);CHKERRQ(ierr);
    /* remove .gz if it ends library name */
    if ((gz = PetscStrstr(infoname,".gz")) && (PetscStrlen(gz) == 3)) {
      *gz = 0;
    }
    
    ierr = PetscStrcat(infoname,".info");CHKERRQ(ierr);
    ierr = PetscFixFilename(infoname,iname); CHKERRQ(ierr);
    ierr = PetscFileRetrieve(comm,iname,infoname,256,&found); CHKERRQ(ierr);
    if (found) {
      v->fdes_info = fopen(infoname,"r");
    }
  }

#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  PetscFunctionReturn(0);
}





