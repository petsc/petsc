#ifndef lint
static char vcid[] = "$Id: binv.c,v 1.23 1996/07/10 01:51:28 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "pinclude/pviewer.h"
#include <stdio.h>
#include <fcntl.h>
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif

struct _Viewer {
  VIEWERHEADER
  int          fdes;         /* file descriptor */
  FILE         *fdes_info;   /* optional file containing info on binary file*/
};

/*@C
    ViewerBinaryGetDescriptor - Extracts the file descriptor from a viewer.

.   viewer - viewer context, obtained from ViewerFileOpenBinary()
.   fdes - file descriptor

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, descriptor

.seealso: ViewerFileOpenBinary(),ViewerBinaryGetInfoPointer()
@*/
int ViewerBinaryGetDescriptor(Viewer viewer,int *fdes)
{
  *fdes = viewer->fdes;
  return 0;
}

/*@C
    ViewerBinaryGetInfoPointer - Extracts the file pointer for the ASCII
          info file associated with a binary file.

.   viewer - viewer context, obtained from ViewerFileOpenBinary()
.   file - file pointer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, descriptor

.seealso: ViewerFileOpenBinary(),ViewerBinaryGetDescriptor()
@*/
int ViewerBinaryGetInfoPointer(Viewer viewer,FILE **file)
{
  *file = viewer->fdes_info;
  return 0;
}

static int ViewerDestroy_BinaryFile(PetscObject obj)
{
  int    rank;
  Viewer v = (Viewer) obj;
  MPI_Comm_rank(v->comm,&rank);
  if (!rank) close(v->fdes);
  if (!rank && v->fdes_info) fclose(v->fdes_info);
  PLogObjectDestroy(obj);
  PetscHeaderDestroy(obj);
  return 0;
}

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

   Note:
   This viewer can be destroyed with ViewerDestroy().

.keywords: binary, file, open, input, output

.seealso: ViewerFileOpenASCII(), ViewerFileSetFormat(), ViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), ViewerBinaryGetDescriptor()
@*/
int ViewerFileOpenBinary(MPI_Comm comm,char *name,ViewerBinaryType type,Viewer *binv)
{  
  int    rank;
  Viewer v;

  PetscHeaderCreate(v,_Viewer,VIEWER_COOKIE,BINARY_FILE_VIEWER,comm);
  PLogObjectCreate(v);
  v->destroy = ViewerDestroy_BinaryFile;
  v->flush   = 0;
  v->iformat = 0;
  *binv = v;

  MPI_Comm_rank(comm,&rank);
  if (!rank) {
    if (type == BINARY_CREATE) {
      if ((v->fdes = creat(name,0666)) == -1)
        SETERRQ(1,"ViewerFileOpenBinary:Cannot create file for writing");
    } 
    else if (type == BINARY_RDONLY) {
      if ((v->fdes = open(name,O_RDONLY,0)) == -1) {
        SETERRQ(1,"ViewerFileOpenBinary:Cannot open file for reading");
      }
    }
    else if (type == BINARY_WRONLY) {
      if ((v->fdes = open(name,O_WRONLY,0)) == -1) {
        SETERRQ(1,"ViewerFileOpenBinary:Cannot open file for writing");
      }
    } else SETERRQ(1,"ViewerFileOpenBinary:Unknown file type");
  }
  else v->fdes = -1;
  v->format    = 0;

  /* try to open info file */
  if (type == BINARY_RDONLY) {
    char *infoname;
    infoname = (char *)PetscMalloc(PetscStrlen(name)+6); CHKPTRQ(infoname);
    PetscStrcpy(infoname,name);
    PetscStrcat(infoname,".info");
    v->fdes_info = fopen(infoname,"r");
    PetscFree(infoname);
  }
#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  return 0;
}





