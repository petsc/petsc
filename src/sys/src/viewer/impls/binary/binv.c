#ifndef lint
static char vcid[] = "$Id: binv.c,v 1.16 1996/03/14 22:26:48 curfman Exp bsmith $";
#endif

#include "petsc.h"
#include "pinclude/pviewer.h"
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

struct _Viewer {
  VIEWERHEADER
  int         fdes;            /* file descriptor */
};

int ViewerBinaryGetDescriptor(Viewer viewer,int *fdes)
{
  *fdes = viewer->fdes;
  return 0;
}

static int ViewerDestroy_BinaryFile(PetscObject obj)
{
  int    rank;
  Viewer v = (Viewer) obj;
  MPI_Comm_rank(v->comm,&rank);
  if (!rank) close(v->fdes);
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

.seealso: ViewerFileOpenASCII(), ViewerDestroy(), VecView(), MatView(),
          VecLoad(), MatLoad()
@*/
int ViewerFileOpenBinary(MPI_Comm comm,char *name,ViewerBinaryType type,Viewer *binv)
{  
  int    rank;
  Viewer v;

  PetscHeaderCreate(v,_Viewer,VIEWER_COOKIE,BINARY_FILE_VIEWER,comm);
  PLogObjectCreate(v);
  v->destroy = ViewerDestroy_BinaryFile;
  v->flush   = 0;
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
#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  return 0;
}





