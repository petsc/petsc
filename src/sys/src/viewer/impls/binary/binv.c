#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: binv.c,v 1.53 1998/12/03 04:04:59 bsmith Exp bsmith $";
#endif

#include "sys.h"
#include "src/viewer/viewerimpl.h"    /*I   "petsc.h"   I*/
#include <fcntl.h>
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (HAVE_IO_H)
#include <io.h>
#endif

typedef struct  {
  int          fdes;            /* file descriptor */
  FILE         *fdes_info;      /* optional file containing info on binary file*/
} Viewer_Binary;

#undef __FUNC__  
#define __FUNC__ "ViewerBinaryGetDescriptor"
/*@C
    ViewerBinaryGetDescriptor - Extracts the file descriptor from a viewer.

    Not Collective

+   viewer - viewer context, obtained from ViewerBinaryOpen()
-   fdes - file descriptor

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, descriptor

.seealso: ViewerBinaryOpen(),ViewerBinaryGetInfoPointer()
@*/
int ViewerBinaryGetDescriptor(Viewer viewer,int *fdes)
{
  Viewer_Binary *vbinary = (Viewer_Binary *) viewer->data;

  PetscFunctionBegin;
  *fdes = vbinary->fdes;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerBinaryGetInfoPointer"
/*@C
    ViewerBinaryGetInfoPointer - Extracts the file pointer for the ASCII
          info file associated with a binary file.

    Not Collective

+   viewer - viewer context, obtained from ViewerBinaryOpen()
-   file - file pointer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, descriptor

.seealso: ViewerBinaryOpen(),ViewerBinaryGetDescriptor()
@*/
int ViewerBinaryGetInfoPointer(Viewer viewer,FILE **file)
{
  Viewer_Binary *vbinary = (Viewer_Binary *) viewer->data;

  PetscFunctionBegin;
  *file = vbinary->fdes_info;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_Binary"
int ViewerDestroy_Binary(Viewer v)
{
  Viewer_Binary *vbinary = (Viewer_Binary *) v->data;
  int           rank;

  PetscFunctionBegin;
  MPI_Comm_rank(v->comm,&rank);
  if (!rank) close(vbinary->fdes);
  if (vbinary->fdes_info) fclose(vbinary->fdes_info);
  PetscFree(vbinary);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerBinaryOpen"
/*@C
   ViewerBinaryOpen - Opens a file for binary input/output.

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

.seealso: ViewerASCIIOpen(), ViewerSetFormat(), ViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), ViewerBinaryGetDescriptor(),
          ViewerBinaryGetInfoPointer()
@*/
int ViewerBinaryOpen(MPI_Comm comm,const char name[],ViewerBinaryType type,Viewer *binv)
{  
  int           rank,ierr;
  Viewer        v;
  Viewer_Binary *vbinary;
  const char    *fname;
  char          bname[1024];
  PetscTruth    found;

  PetscFunctionBegin;
  PetscHeaderCreate(v,_p_Viewer,struct _ViewerOps,VIEWER_COOKIE,0,"Viewer",comm,ViewerDestroy,0);
  PLogObjectCreate(v);
  vbinary = PetscNew(Viewer_Binary);CHKPTRQ(vbinary);
  v->data            = (void *) vbinary;
  v->ops->destroy    = ViewerDestroy_Binary;
  v->ops->flush      = 0;
  v->iformat         = 0;
  *binv              = v;
  vbinary->fdes_info = 0;
 
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
      if ((vbinary->fdes = open(fname,O_WRONLY|O_CREAT|O_TRUNC|O_BINARY,0666 )) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot create file for writing");
      }
    } else if (type == BINARY_RDONLY) {
      if ((vbinary->fdes = open(fname,O_RDONLY|O_BINARY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for reading");
      }
    } else if (type == BINARY_WRONLY) {
      if ((vbinary->fdes = open(fname,O_WRONLY|O_BINARY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for writing");
      }
    } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown file type");
#else
    if (type == BINARY_CREATE) {
      if ((vbinary->fdes = creat(fname,0666)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot create file for writing");
      }
    } else if (type == BINARY_RDONLY) {
      if ((vbinary->fdes = open(fname,O_RDONLY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for reading");
      }
    } else if (type == BINARY_WRONLY) {
      if ((vbinary->fdes = open(fname,O_WRONLY,0)) == -1) {
        SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open file for writing");
      }
    } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown file type");
#endif
  } else vbinary->fdes = -1;
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
      vbinary->fdes_info = fopen(infoname,"r");
    }
  }
  v->type_name = (char *) PetscMalloc((1+PetscStrlen(BINARY_VIEWER))*sizeof(char));CHKPTRQ(v->type_name);
  PetscStrcpy(v->type_name,BINARY_VIEWER);

#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  PetscFunctionReturn(0);
}









