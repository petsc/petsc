
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: filev.c,v 1.74 1998/07/23 14:50:47 bsmith Exp balay $";
#endif

#include "petsc.h"
#include "pinclude/pviewer.h"
#include "pinclude/petscfix.h"
#include <stdarg.h>

struct _p_Viewer {
  VIEWERHEADER
  FILE          *fd;
  char          *outputname,*outputnames[10];
};

Viewer VIEWER_STDOUT_SELF, VIEWER_STDERR_SELF, VIEWER_STDOUT_WORLD, VIEWER_STDERR_WORLD;

/*
      This is called by PETScInitialize() to create the 
   default PETSc viewers.
*/
#undef __FUNC__  
#define __FUNC__ "ViewerInitialize_Private"
int ViewerInitialize_Private(void)
{
  PetscFunctionBegin;
  ViewerFileOpenASCII(PETSC_COMM_SELF,"stderr",&VIEWER_STDERR_SELF);
  ViewerFileOpenASCII(PETSC_COMM_SELF,"stdout",&VIEWER_STDOUT_SELF);
  ViewerFileOpenASCII(PETSC_COMM_WORLD,"stdout",&VIEWER_STDOUT_WORLD);
  ViewerFileOpenASCII(PETSC_COMM_WORLD,"stderr",&VIEWER_STDERR_WORLD);
  PetscFunctionReturn(0);
}

/*
      This is called in PetscFinalize() to destroy all
   traces of the default viewers.
*/
#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_Private"
int ViewerDestroy_Private(void)
{
  PetscFunctionBegin;
  ViewerDestroy(VIEWER_STDERR_SELF);
  ViewerDestroy(VIEWER_STDOUT_SELF);
  ViewerDestroy(VIEWER_STDOUT_WORLD);
  ViewerDestroy(VIEWER_STDERR_WORLD);
  VIEWER_STDOUT_Destroy(PETSC_COMM_SELF);
  VIEWER_STDERR_Destroy(PETSC_COMM_SELF);
  VIEWER_STDOUT_Destroy(PETSC_COMM_WORLD);
  VIEWER_STDERR_Destroy(PETSC_COMM_WORLD);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Stdout_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Stdout_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "VIEWER_STDOUT_" 
/*@C
   VIEWER_STDOUT_ - Creates a window viewer shared by all processors 
                    in a communicator.

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the window viewer

   Notes: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      XXXView(XXX object,VIEWER_STDOUT_(comm));

.seealso: VIEWER_DRAWX_(), ViewerFileOpenASCII()

@*/
Viewer VIEWER_STDOUT_(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stdout_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Stdout_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Stdout_keyval, (void **)&viewer, &flag );
  if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flag) { /* viewer not yet created */
    ierr = ViewerFileOpenASCII(comm,"stdout",&viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put( comm, Petsc_Viewer_Stdout_keyval, (void *) viewer );
    if (ierr) {PetscError(__LINE__,"VIEWER_STDOUT_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/*
       If there is a Viewer associated with this communicator it is destroyed.
*/
int VIEWER_STDOUT_Destroy(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stdout_keyval == MPI_KEYVAL_INVALID) {
    PetscFunctionReturn(0);
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Stdout_keyval, (void **)&viewer, &flag ); CHKERRQ(ierr);
  if (flag) { 
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_Stdout_keyval); CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------*/
/*
    The variable Petsc_Viewer_Stderr_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Viewer.
*/
static int Petsc_Viewer_Stderr_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ "VIEWER_STDERR_" 
/*@C
   VIEWER_STDERR_ - Creates a window viewer shared by all processors 
                    in a communicator.

   Collective on MPI_Comm

   Input Parameter:
.  comm - the MPI communicator to share the window viewer

   Note: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      XXXView(XXX object,VIEWER_STDERR_(comm));

.seealso: VIEWER_DRAWX_, ViewerFileOpenASCII(), 
@*/
Viewer VIEWER_STDERR_(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stderr_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Viewer_Stderr_keyval,0);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Stderr_keyval, (void **)&viewer, &flag );
  if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  if (!flag) { /* viewer not yet created */
    ierr = ViewerFileOpenASCII(comm,"stderr",&viewer);
    if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
    ierr = MPI_Attr_put( comm, Petsc_Viewer_Stderr_keyval, (void *) viewer );
    if (ierr) {PetscError(__LINE__,"VIEWER_STDERR_",__FILE__,__SDIR__,1,1,0); viewer = 0;}
  } 
  PetscFunctionReturn(viewer);
}

/*
       If there is a Viewer associated with this communicator it is destroyed.
*/
int VIEWER_STDERR_Destroy(MPI_Comm comm)
{
  int    ierr,flag;
  Viewer viewer;

  PetscFunctionBegin;
  if (Petsc_Viewer_Stderr_keyval == MPI_KEYVAL_INVALID) {
    PetscFunctionReturn(0);
  }
  ierr = MPI_Attr_get( comm, Petsc_Viewer_Stderr_keyval, (void **)&viewer, &flag );CHKERRQ(ierr);
  if (flag) { 
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    ierr = MPI_Attr_delete(comm,Petsc_Viewer_Stderr_keyval);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_File"
int ViewerDestroy_File(Viewer v)
{
  int    rank = 0;

  PetscFunctionBegin;
  if (v->type == ASCII_FILES_VIEWER) {MPI_Comm_rank(v->comm,&rank);} 
  if (!rank && v->fd != stderr && v->fd != stdout) fclose(v->fd);
  PLogObjectDestroy((PetscObject)v);
  PetscHeaderDestroy((PetscObject)v);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerFlush_File"
int ViewerFlush_File(Viewer v)
{
  int rank;

  PetscFunctionBegin;
  MPI_Comm_rank(v->comm,&rank);
  if (rank) PetscFunctionReturn(0);
  fflush(v->fd);
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ "ViewerASCIIGetPointer"
/*@C
    ViewerASCIIGetPointer - Extracts the file pointer from an ASCII viewer.

    Not Collective

+   viewer - viewer context, obtained from ViewerFileOpenASCII()
-   fd - file pointer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, pointer

.seealso: ViewerFileOpenASCII()
@*/
int ViewerASCIIGetPointer(Viewer viewer, FILE **fd)
{
  PetscFunctionBegin;
  *fd = viewer->fd;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerFileGetOutputname_Private"
int ViewerFileGetOutputname_Private(Viewer viewer, char **name)
{
  PetscFunctionBegin;
  *name = viewer->outputname;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerGetFormat"
int ViewerGetFormat(Viewer viewer,int *format)
{
  PetscFunctionBegin;
  *format =  viewer->format;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerFileOpenASCII"
/*@C
   ViewerFileOpenASCII - Opens an ASCII file as a viewer.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the communicator
-  name - the file name

   Output Parameter:
.  lab - the viewer to use with the specified file

   Notes:
   This viewer can be destroyed with ViewerDestroy().

   If a multiprocessor communicator is used (such as PETSC_COMM_WORLD), 
   then only the first processor in the group opens the file.  All other 
   processors send their data to the first processor to print. 

   Each processor can instead write its own independent output by
   specifying the communicator PETSC_COMM_SELF.

   As shown below, ViewerFileOpenASCII() is useful in conjunction with 
   MatView() and VecView()
.vb
     ViewerFileOpenASCII(PETSC_COMM_WORLD,"mat.output",&viewer);
     MatView(matrix,viewer);
.ve

.keywords: Viewer, file, open

.seealso: MatView(), VecView(), ViewerDestroy(), ViewerFileOpenBinary(),
          ViewerASCIIGetPointer()
@*/
int ViewerFileOpenASCII(MPI_Comm comm,const char name[],Viewer *lab)
{
  Viewer v;
  int    ierr;
  char   fname[256];

  PetscFunctionBegin;
  if (comm == PETSC_COMM_SELF) {
    PetscHeaderCreate(v,_p_Viewer,int,VIEWER_COOKIE,ASCII_FILE_VIEWER,comm,ViewerDestroy,0);
  } else {
    PetscHeaderCreate(v,_p_Viewer,int,VIEWER_COOKIE,ASCII_FILES_VIEWER,comm,ViewerDestroy,0);
  }
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_File;
  v->flush       = ViewerFlush_File;

  if (!PetscStrcmp(name,"stderr")) v->fd = stderr;
  else if (!PetscStrcmp(name,"stdout")) v->fd = stdout;
  else {
    ierr         = PetscFixFilename(name,fname);CHKERRQ(ierr);
    v->fd        = fopen(fname,"w"); 
    if (!v->fd) SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open viewer file");
  }
  v->format        = VIEWER_FORMAT_ASCII_DEFAULT;
  v->iformat       = 0;
  v->outputname    = 0;
#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  *lab           = v;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerSetFormat"
/*@C
   ViewerSetFormat - Sets the format for viewers.

   Collective on Viewer

   Input Parameters:
+  v - the viewer
.  format - the format
-  char - optional object name

   Notes:
   Available formats include
+    VIEWER_FORMAT_ASCII_DEFAULT - default format
.    VIEWER_FORMAT_ASCII_MATLAB - Matlab format
.    VIEWER_FORMAT_ASCII_IMPL - implementation-specific format
      (which is in many cases the same as the default)
.    VIEWER_FORMAT_ASCII_INFO - basic information about object
.    VIEWER_FORMAT_ASCII_INFO_LONG - more detailed info
       about object
.    VIEWER_FORMAT_ASCII_COMMON - identical output format for
       all objects of a particular type
.    VIEWER_FORMAT_ASCII_INDEX - (for vectors) prints the vector
       element number next to each vector entry
.    VIEWER_FORMAT_BINARY_NATIVE - store the object to the binary
      file in its native format (for example, dense
       matrices are stored as dense)
.    VIEWER_FORMAT_DRAW_BASIC - views the vector with a simple 1d plot
.    VIEWER_FORMAT_DRAW_LG - views the vector with a line graph
-    VIEWER_FORMAT_DRAW_CONTOUR - views the vector with a contour plot

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the Matlab format.

.keywords: Viewer, file, set, format

.seealso: ViewerFileOpenASCII(), ViewerFileOpenBinary(), MatView(), VecView(),
          ViewerPushFormat(), ViewerPopFormat(), ViewerDrawOpenX(),ViewerMatlabOpen()
@*/
int ViewerSetFormat(Viewer v,int format,char *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->type == ASCII_FILES_VIEWER || v->type == ASCII_FILE_VIEWER) {
    v->format     = format;
    v->outputname = name;
  } else {
    v->format     = format;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerPushFormat"
/*@C
   ViewerPushFormat - Sets the format for file viewers.

   Collective on Viewer

   Input Parameters:
+  v - the viewer
.  format - the format
-  char - optional object name

   Notes:
   Available formats include
+    VIEWER_FORMAT_ASCII_DEFAULT - default format
.    VIEWER_FORMAT_ASCII_MATLAB - Matlab format
.    VIEWER_FORMAT_ASCII_IMPL - implementation-specific format
      (which is in many cases the same as the default)
.    VIEWER_FORMAT_ASCII_INFO - basic information about object
.    VIEWER_FORMAT_ASCII_INFO_LONG - more detailed info
       about object
.    VIEWER_FORMAT_ASCII_COMMON - identical output format for
       all objects of a particular type
.    VIEWER_FORMAT_ASCII_INDEX - (for vectors) prints the vector
       element number next to each vector entry
.    VIEWER_FORMAT_BINARY_NATIVE - store the object to the binary
      file in its native format (for example, dense
       matrices are stored as dense)
.    VIEWER_FORMAT_DRAW_BASIC - views the vector with a simple 1d plot
.    VIEWER_FORMAT_DRAW_LG - views the vector with a line graph
-    VIEWER_FORMAT_DRAW_CONTOUR - views the vector with a contour plot

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the Matlab format.

.keywords: Viewer, file, set, format

.seealso: ViewerFileOpenASCII(), ViewerFileOpenBinary(), MatView(), VecView(),
          ViewerSetFormat(), ViewerPopFormat()
@*/
int ViewerPushFormat(Viewer v,int format,char *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->iformat > 9) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Too many pushes");

  if (v->type == ASCII_FILES_VIEWER || v->type == ASCII_FILE_VIEWER) {
    v->formats[v->iformat]       = v->format;
    v->outputnames[v->iformat++] = v->outputname;
    v->format                    = format;
    v->outputname                = name;
  } else {
    v->formats[v->iformat++]     = v->format;
    v->format                    = format;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerPopFormat"
/*@C
   ViewerPopFormat - Resets the format for file viewers.

   Collective on Viewer

   Input Parameters:
.  v - the viewer

.keywords: Viewer, file, set, format, push, pop

.seealso: ViewerFileOpenASCII(), ViewerFileOpenBinary(), MatView(), VecView(),
          ViewerSetFormat(), ViewerPushFormat()
@*/
int ViewerPopFormat(Viewer v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->iformat <= 0) PetscFunctionReturn(0);

  if (v->type == ASCII_FILES_VIEWER || v->type == ASCII_FILE_VIEWER) {
    v->format     = v->formats[--v->iformat];
    v->outputname = v->outputnames[v->iformat];
  } else if (v->type == BINARY_FILE_VIEWER) {
    v->format     = v->formats[--v->iformat];
  }
  PetscFunctionReturn(0);
}




