#ifndef lint
static char vcid[] = "$Id: sysio.c,v 1.1 1995/08/17 20:46:21 curfman Exp curfman $";
#endif

#include "petsc.h"
#include <fcntl.h>

/*@
   ViewerFileOpenBinary - Opens a file for binary input/output.

   Input Parameters:
.  name - name of file 
.  type - type of file
$    SY_CREAT - create new file for binary output
$    SY_RDONLY - open existing file for binary input
$    SY_WRONLY - open existing file for binary output

   Output Parameter:
.  fd - file descriptor

   Notes:
   When a file is created with SY_CREAT, the file permissions are 
   set to be 664,
$      user:  read and write
$      group: read and write
$      world: read

   After using a file that is opened with PetscBinaryFileOpen, this file 
   should be closed with the command 
$      close(fd)

.keywords - binary, file, open, input, output
@*/
ViewerFileOpenBinary(MPI_Comm comm,char *name,ViewerBinaryType type,Viewer *fd)
{
  int otype;

  if (type == SY_CREAT) {
    if (permis == PETSC_DEFAULT) permis = 664;
    if ((*fd = creat(name,permis)) == -1)
      SETERRQ(1,"PetscBinaryFileOpen: Cannot open file for writing");
    return 0;
  } 
  permis = 0;  /* permis = 0 for all other file open statements */
  switch ((int) type) {
    case (int)SY_RDONLY: otype = O_RDONLY; break;
    case (int)SY_WRONLY: otype = O_WRONLY; break;
    case (int)SY_RDWR:   otype = O_RDWR;   break;
    default: SETERRQ(1,"PetscBinaryFileOpen: File type not supported");
  }
  if ((*fd = open(name,otype,permis)) == -1) {
    if (type == SY_RDONLY) {
      SETERRQ(1,"PetscBinaryFileOpen: Cannot open file for reading");
    } else if (type == SY_WRONLY) {
      SETERRQ(1,"PetscBinaryFileOpen: Cannot open file for writing");
    } else if (type == SY_RDWR) {
      SETERRQ(1,"PetscBinaryFileOpen: Cannot open file for reading/writing");
    } else SETERRQ(1,"PetscBinaryFileOpen: Cannot open file");
  }
  return 0;
}
