#ifndef lint
static char vcid[] = "$Id: binv.c,v 1.1 1995/08/22 19:26:33 curfman Exp curfman $";
#endif

#include "petsc.h"
#include <fcntl.h>
#include <unistd.h>

struct _Viewer {
  PETSCHEADER
  int         fdes;            /* file descriptor */
};

int ViewerFileGetDescriptor_Private(Viewer viewer)
{
  return viewer->fdes;
}

static int ViewerDestroy_BinaryFile(PetscObject obj)
{
  Viewer v = (Viewer) obj;
  int    mytid = 0;
  if (v->type == BIN_FILES_VIEWER) {MPI_Comm_rank(v->comm,&mytid);} 
  if (!mytid) close(v->fdes);
  PLogObjectDestroy(obj);
  PETSCHEADERDESTROY(obj);
  return 0;
}

/*@
   ViewerFileOpenBinary - Opens a file for binary input/output.

   Input Parameters:
.  name - name of file 
.  type - type of file
$    BIN_CREAT - create new file for binary output
$    BIN_RDONLY - open existing file for binary input
$    BIN_WRONLY - open existing file for binary output

   Output Parameter:
.  fd - file descriptor

   Notes:
   When a new file is created (type = BIN_CREAT), the file permissions
   are set to be 664,
$      user:  read and write
$      group: read and write
$      world: read

   This viewer can be destroyed with ViewerDestroy().

.keywords: binary, file, open, input, output

.seealso: ViewerDestroy()
@*/
ViewerFileOpenBinary(MPI_Comm comm,char *name,ViewerBinaryType type,
                     Viewer *binv)
{
  int otype;
  Viewer v;
  if (comm == MPI_COMM_SELF) {
    PETSCHEADERCREATE(v,_Viewer,VIEWER_COOKIE,BIN_FILE_VIEWER,comm);
  } else {
    PETSCHEADERCREATE(v,_Viewer,VIEWER_COOKIE,BIN_FILES_VIEWER,comm);
  }
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_BinaryFile;

  /* So far we're not using comm */
  if (type == BIN_CREAT) {
    if ((v->fdes = creat(name,664)) == -1)
      SETERRQ(1,"ViewerFileOpenBinary: Cannot create file for writing");
    return 0;
  } 
  switch ((int) type) {
    case (int)BIN_RDONLY: otype = O_RDONLY; break;
    case (int)BIN_WRONLY: otype = O_WRONLY; break;
    default: SETERRQ(1,"PetscBinaryFileOpen: File type not supported");
  }
  if ((v->fdes = open(name,otype,0)) == -1) {
    if (type == BIN_RDONLY) {
      SETERRQ(1,"ViewerBinaryFileOpenBinary: Cannot open file for reading");
    } else if (type == BIN_WRONLY) {
      SETERRQ(1,"ViewerFileOpenBinary: Cannot open file for writing");
    } else SETERRQ(1,"PetscBinaryFileOpen: Cannot open file");
  }

#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  *binv = v;
  return 0;
}


