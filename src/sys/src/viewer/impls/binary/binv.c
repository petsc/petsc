#ifndef lint
static char vcid[] = "$Id: binv.c,v 1.5 1995/09/05 19:24:57 curfman Exp bsmith $";
#endif

#include "petsc.h"
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

struct _Viewer {
  PETSCHEADER
  int         fdes;            /* file descriptor */
};

int ViewerFileGetDescriptor_Private(Viewer viewer,int *fdes)
{
  *fdes = viewer->fdes;
  return 0;
}

static int ViewerDestroy_BinaryFile(PetscObject obj)
{
  int    mytid;
  Viewer v = (Viewer) obj;
  MPI_Comm_rank(v->comm,&mytid);
  if (!mytid) close(v->fdes);
  PLogObjectDestroy(obj);
  PETSCHEADERDESTROY(obj);
  return 0;
}

/*@
   ViewerFileOpenBinary - Opens a file for binary input/output.

   Input Parameters:
.  comm - MPI communicator
.  name - name of file 
.  type - type of file
$    BIN_CREAT - create new file for binary output
$    BIN_RDONLY - open existing file for binary input
$    BIN_WRONLY - open existing file for binary output

   Output Parameter:
.  binv - viewer for binary input/output to use with the specified file

   Note:
   This viewer can be destroyed with ViewerDestroy().

.keywords: binary, file, open, input, output

.seealso: ViewerDestroy()
@*/
int ViewerFileOpenBinary(MPI_Comm comm,char *name,ViewerBinaryType type,
                         Viewer *binv)
{  
  int    mytid;
  Viewer v;
  PETSCHEADERCREATE(v,_Viewer,VIEWER_COOKIE,BIN_FILE_VIEWER,comm);
  PLogObjectCreate(v);
  v->destroy = ViewerDestroy_BinaryFile;
  *binv = v;

  MPI_Comm_rank(comm,&mytid);
  if (!mytid) {
    if (type == BIN_CREAT) {
      if ((v->fdes = creat(name,0666)) == -1)
        SETERRQ(1,"ViewerFileOpenBinary: Cannot create file for writing");
    } 
    else if (type == BIN_RDONLY) {
      if ((v->fdes = open(name,O_RDONLY,0)) == -1) {
        SETERRQ(1,"ViewerFileOpenBinary: Cannot open file for reading");
      }
    }
    else if (type == BIN_WRONLY) {
      if ((v->fdes = open(name,O_WRONLY,0)) == -1) {
        SETERRQ(1,"ViewerFileOpenBinary: Cannot open file for writing");
      }
    } else SETERRQ(1,"ViewerFileOpenBinary: File type not supported");
  }
  else v->fdes = -1;
#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  return 0;
}


