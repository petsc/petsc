#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: binv.c,v 1.44 1998/04/03 23:17:31 bsmith Exp bsmith $";
#endif

#include "petsc.h"
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
  int          fdes;         /* file descriptor */
  FILE         *fdes_info;   /* optional file containing info on binary file*/
};

#undef __FUNC__  
#define __FUNC__ "ViewerBinaryGetDescriptor"
/*@C
    ViewerBinaryGetDescriptor - Extracts the file descriptor from a viewer.

.   viewer - viewer context, obtained from ViewerFileOpenBinary()
.   fdes - file descriptor

    Not Collective

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

.   viewer - viewer context, obtained from ViewerFileOpenBinary()
.   file - file pointer

    Not Collective

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
  int    rank;

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

   Input Parameters:
.  comm - MPI communicator
.  name - name of file 
.  type - type of file
$    BINARY_CREATE - create new file for binary output
$    BINARY_RDONLY - open existing file for binary input
$    BINARY_WRONLY - open existing file for binary output

   Output Parameter:
.  binv - viewer for binary input/output to use with the specified file

   Collective on MPI_Comm

   Note:
   This viewer can be destroyed with ViewerDestroy().

.keywords: binary, file, open, input, output

.seealso: ViewerFileOpenASCII(), ViewerSetFormat(), ViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), ViewerBinaryGetDescriptor()
@*/
int ViewerFileOpenBinary(MPI_Comm comm,char *name,ViewerBinaryType type,Viewer *binv)
{  
  int    rank,ierr;
  Viewer v;

  PetscFunctionBegin;
  PetscHeaderCreate(v,_p_Viewer,int,VIEWER_COOKIE,BINARY_FILE_VIEWER,comm,ViewerDestroy,0);
  PLogObjectCreate(v);
  v->destroy = ViewerDestroy_BinaryFile;
  v->flush   = 0;
  v->iformat = 0;
  *binv = v;

  MPI_Comm_rank(comm,&rank);
  if (!rank || type == BINARY_RDONLY) {
#if defined(PARCH_nt_gnu) || defined(PARCH_nt) 
    if (type == BINARY_CREATE) {
      if ((v->fdes = open(name,O_WRONLY|O_CREAT|O_TRUNC|O_BINARY,0666 )) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot create file for writing");
      }
    } else if (type == BINARY_RDONLY) {
      if ((v->fdes = open(name,O_RDONLY|O_BINARY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for reading");
      }
    } else if (type == BINARY_WRONLY) {
      if ((v->fdes = open(name,O_WRONLY|O_BINARY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for writing");
      }
    } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown file type");
#else
    if (type == BINARY_CREATE) {
      if ((v->fdes = creat(name,0666)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot create file for writing");
      }
    } else if (type == BINARY_RDONLY) {
      if ((v->fdes = open(name,O_RDONLY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for reading");
      }
    } else if (type == BINARY_WRONLY) {
      if ((v->fdes = open(name,O_WRONLY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for writing");
      }
    } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown file type");
#endif
  } else v->fdes = -1;
  v->format    = 0;

  /* try to open info file */
  if (type == BINARY_RDONLY) {
    char *infoname;
    infoname = (char *)PetscMalloc(PetscStrlen(name)+6); CHKPTRQ(infoname);
    PetscStrcpy(infoname,name);
    PetscStrcat(infoname,".info");
    ierr = PetscFixFilename(infoname); CHKERRQ(ierr);
    v->fdes_info = fopen(infoname,"r");
    PetscFree(infoname);
  }
#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  PetscFunctionReturn(0);
}





