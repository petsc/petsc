#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: binv.c,v 1.69 1999/06/30 23:48:54 balay Exp bsmith $";
#endif

#include "sys.h"
#include "src/sys/src/viewer/viewerimpl.h"    /*I   "petsc.h"   I*/
#include <fcntl.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined (PETSC_HAVE_IO_H)
#include <io.h>
#endif

typedef struct  {
  int              fdes;            /* file descriptor */
  ViewerBinaryType btype;           /* read or write? */
  FILE             *fdes_info;      /* optional file containing info on binary file*/
} Viewer_Binary;

#undef __FUNC__  
#define __FUNC__ "ViewerBinaryGetDescriptor"
/*@C
    ViewerBinaryGetDescriptor - Extracts the file descriptor from a viewer.

    Not Collective

+   viewer - viewer context, obtained from ViewerBinaryOpen()
-   fdes - file descriptor

    Level: advanced

    Notes:
      For writable binary viewers, the descriptor will only be valid for the 
    first processor in the communicator that shares the viewer.
 
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

    Level: advanced

    Notes:
      For writable binary viewers, the descriptor will only be valid for the 
    first processor in the communicator that shares the viewer.
 
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
  int           ierr,rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(v->comm,&rank);CHKERRQ(ierr);
  if (!rank && vbinary->fdes) close(vbinary->fdes);
  if (vbinary->fdes_info) fclose(vbinary->fdes_info);
  ierr = PetscFree(vbinary);CHKERRQ(ierr);
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

   Level: beginner

   Note:
   This viewer can be destroyed with ViewerDestroy().

.keywords: binary, file, open, input, output

.seealso: ViewerASCIIOpen(), ViewerSetFormat(), ViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), ViewerBinaryGetDescriptor(),
          ViewerBinaryGetInfoPointer()
@*/
int ViewerBinaryOpen(MPI_Comm comm,const char name[],ViewerBinaryType type,Viewer *binv)
{
  int ierr;
  
  PetscFunctionBegin;
  ierr = ViewerCreate(comm,binv);CHKERRQ(ierr);
  ierr = ViewerSetType(*binv,BINARY_VIEWER);CHKERRQ(ierr);
  ierr = ViewerBinarySetType(*binv,type);CHKERRQ(ierr);
  ierr = ViewerSetFilename(*binv,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerBinarySetType"
/*@C
     ViewerBinarySetType - Sets the type of binary file to be open

    Collective on Viewer

  Input Parameters:
+  viewer - the viewer; must be a binary viewer
-  type - type of file
$    BINARY_CREATE - create new file for binary output
$    BINARY_RDONLY - open existing file for binary input
$    BINARY_WRONLY - open existing file for binary output

  Level: advanced

.seealso: ViewerCreate(), ViewerSetType(), ViewerBinaryOpen()

@*/
int ViewerBinarySetType(Viewer viewer,ViewerBinaryType type)
{
  int ierr, (*f)(Viewer,ViewerBinaryType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)viewer,"ViewerBinarySetType_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(viewer,type);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerBinarySetType_Binary"
int ViewerBinarySetType_Binary(Viewer viewer,ViewerBinaryType type)
{
  Viewer_Binary    *vbinary = (Viewer_Binary *) viewer->data;

  PetscFunctionBegin;
  vbinary->btype = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "ViewerBinaryLoadInfo"
/*
    ViewerBinaryLoadInfo options from the name.info file
    if it exists.
*/
int ViewerBinaryLoadInfo(Viewer viewer)
{
  FILE *file;
  char string[128],*first,*second,*final;
  int  len,ierr,flg;

  PetscFunctionBegin;
  ierr = OptionsHasName(PETSC_NULL,"-load_ignore_info",&flg);CHKERRQ(ierr);
  if (flg) PetscFunctionReturn(0);

  ierr = ViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
  if (!file) PetscFunctionReturn(0);

  /* read rows of the file adding them to options database */
  while (fgets(string,128,file)) {
    /* Comments are indicated by #, ! or % in the first column */
    if (string[0] == '#') continue;
    if (string[0] == '!') continue;
    if (string[0] == '%') continue;
    ierr = PetscStrtok(string," ",&first);CHKERRQ(ierr);
    ierr = PetscStrtok(0," ",&second);CHKERRQ(ierr);
    if (first && first[0] == '-') {

      /*
         Check for -mat_complex or -mat_double
      */
#if defined(PETSC_USE_COMPLEX)
      if (!PetscStrncmp(first,"-mat_double",11)) {
        SETERRQ(1,1,"Loading double number matrix with complex number code");
      }
#else
      if (!PetscStrncmp(first,"-mat_complex",12)) {
        SETERRQ(1,1,"Loading complex number matrix with double number code");
      }
#endif

      if (second) {final = second;} else {final = first;}
      len = PetscStrlen(final);
      while (len > 0 && (final[len-1] == ' ' || final[len-1] == '\n')) {
        len--; final[len] = 0;
      }
      ierr = OptionsSetValue(first,second);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);

}

/*
        Actually opens the file 
*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerSetFilename_Binary"
int ViewerSetFilename_Binary(Viewer viewer,const char name[])
{
  int              rank,ierr;
  Viewer_Binary    *vbinary = (Viewer_Binary *) viewer->data;
  const char       *fname;
  char             bname[1024];
  PetscTruth       found;
  ViewerBinaryType type = vbinary->btype;

  if (type == (ViewerBinaryType) -1) {
    SETERRQ(1,1,"Must call ViewerBinarySetType() before ViewerSetFilename()");
  }
  ierr = MPI_Comm_rank(viewer->comm,&rank);CHKERRQ(ierr);

  /* only first processor opens file if writeable */
  if (!rank || type == BINARY_RDONLY) {

    if (type == BINARY_RDONLY){
      /* possibly get the file from remote site or compressed file */
      ierr  = PetscFileRetrieve(viewer->comm,name,bname,1024,&found);CHKERRQ(ierr);
      if (!found) {
        SETERRQ1(1,1,"Cannot locate file: %s",name);
      }
      fname = bname;
    } else {
      fname = name;
    }

#if defined(PARCH_win32_gnu) || defined(PARCH_win32) 
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
  viewer->format    = 0;

  /* 
      try to open info file: all processors open this file
  */
  if (!rank || type == BINARY_RDONLY) {
    char infoname[256],iname[256],*gz;
  
    ierr = PetscStrcpy(infoname,name);CHKERRQ(ierr);
    /* remove .gz if it ends library name */
    ierr = PetscStrstr(infoname,".gz",&gz);CHKERRQ(ierr);
    if (gz && (PetscStrlen(gz) == 3)) {
      *gz = 0;
    }
    
    ierr = PetscStrcat(infoname,".info");CHKERRQ(ierr);
    ierr = PetscFixFilename(infoname,iname);CHKERRQ(ierr);
    if (type == BINARY_RDONLY) {
      ierr = PetscFileRetrieve(viewer->comm,iname,infoname,256,&found);CHKERRQ(ierr);
      if (found) {
        vbinary->fdes_info = fopen(infoname,"r");
        if (vbinary->fdes_info) {
          ierr = ViewerBinaryLoadInfo(viewer);CHKERRQ(ierr);
          fclose(vbinary->fdes_info);
        }
        vbinary->fdes_info = fopen(infoname,"r");
      }
    } else {
      vbinary->fdes_info = fopen(infoname,"w");
    }
  }

#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject)viewer,"File: %s",name);
#endif
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "ViewerCreate_Binary"
int ViewerCreate_Binary(Viewer v)
{  
  int           ierr;
  Viewer_Binary *vbinary;

  PetscFunctionBegin;
  vbinary            = PetscNew(Viewer_Binary);CHKPTRQ(vbinary);
  v->data            = (void *) vbinary;
  v->ops->destroy    = ViewerDestroy_Binary;
  v->ops->flush      = 0;
  v->iformat         = 0;
  vbinary->fdes_info = 0;
  vbinary->fdes      = 0;
  vbinary->btype     = (ViewerBinaryType) -1; 

  ierr = PetscObjectComposeFunction((PetscObject)v,"ViewerSetFilename_C",
                                    "ViewerSetFilename_Binary",
                                     (void*)ViewerSetFilename_Binary);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"ViewerBinarySetType_C",
                                    "ViewerBinarySetType_Binary",
                                     (void*)ViewerBinarySetType_Binary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END









