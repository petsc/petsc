#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: filev.c,v 1.58 1997/05/23 18:34:20 balay Exp balay $";
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
#define __FUNC__ "ViewerInitialize_Private" /* ADIC Ignore */
int ViewerInitialize_Private()
{
  ViewerFileOpenASCII(PETSC_COMM_SELF,"stderr",&VIEWER_STDERR_SELF);
  ViewerFileOpenASCII(PETSC_COMM_SELF,"stdout",&VIEWER_STDOUT_SELF);
  ViewerFileOpenASCII(PETSC_COMM_WORLD,"stdout",&VIEWER_STDOUT_WORLD);
  ViewerFileOpenASCII(PETSC_COMM_WORLD,"stderr",&VIEWER_STDERR_WORLD);
  return 0;
}

/*
      This is called in PetscFinalize() to destroy all
   traces of the default viewers.
*/
#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_Private" /* ADIC Ignore */
int ViewerDestroy_Private()
{
  ViewerDestroy(VIEWER_STDERR_SELF);
  ViewerDestroy(VIEWER_STDOUT_SELF);
  ViewerDestroy(VIEWER_STDOUT_WORLD);
  ViewerDestroy(VIEWER_STDERR_WORLD);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_File" /* ADIC Ignore */
int ViewerDestroy_File(PetscObject obj)
{
  Viewer v = (Viewer) obj;
  int    rank = 0;
  if (v->type == ASCII_FILES_VIEWER) {MPI_Comm_rank(v->comm,&rank);} 
  if (!rank && v->fd != stderr && v->fd != stdout) fclose(v->fd);
  PLogObjectDestroy(obj);
  PetscHeaderDestroy(obj);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ViewerFlush_File" /* ADIC Ignore */
int ViewerFlush_File(Viewer v)
{
  int rank;
  MPI_Comm_rank(v->comm,&rank);
  if (rank) return 0;
  fflush(v->fd);
  return 0;  
}

#undef __FUNC__  
#define __FUNC__ "ViewerASCIIGetPointer" /* ADIC Ignore */
/*@C
    ViewerASCIIGetPointer - Extracts the file pointer from an ASCII viewer.

.   viewer - viewer context, obtained from ViewerFileOpenASCII()
.   fd - file pointer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: Viewer, file, get, pointer

.seealso: ViewerFileOpenASCII()
@*/
int ViewerASCIIGetPointer(Viewer viewer, FILE **fd)
{
  *fd = viewer->fd;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ViewerFileGetOutputname_Private" /* ADIC Ignore */
int ViewerFileGetOutputname_Private(Viewer viewer, char **name)
{
  *name = viewer->outputname;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ViewerGetFormat" /* ADIC Ignore */
int ViewerGetFormat(Viewer viewer,int *format)
{
  *format =  viewer->format;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ViewerFileOpenASCII" /* ADIC Ignore */
/*@C
   ViewerFileOpenASCII - Opens an ASCII file as a viewer.

   Input Parameters:
.  comm - the communicator
.  name - the file name

   Output Parameter:
.  lab - the viewer to use with the specified file

   Notes:
   If a multiprocessor communicator is used (such as MPI_COMM_WORLD), 
   then only the first processor in the group opens the file.  All other 
   processors send their data to the first processor to print. 

   Each processor can instead write its own independent output by
   specifying the communicator PETSC_COMM_SELF.

   As shown below, ViewerFileOpenASCII() is useful in conjunction with 
   MatView() and VecView()
$
$    ViewerFileOpenASCII(MPI_COMM_WORLD,"mat.output",&viewer);
$    MatView(matrix,viewer);

   This viewer can be destroyed with ViewerDestroy().

.keywords: Viewer, file, open

.seealso: MatView(), VecView(), ViewerDestroy(), ViewerFileOpenBinary(),
          ViewerASCIIGetPointer()
@*/
int ViewerFileOpenASCII(MPI_Comm comm,char *name,Viewer *lab)
{
  Viewer v;
  if (comm == PETSC_COMM_SELF) {
    PetscHeaderCreate(v,_p_Viewer,VIEWER_COOKIE,ASCII_FILE_VIEWER,comm);
  } else {
    PetscHeaderCreate(v,_p_Viewer,VIEWER_COOKIE,ASCII_FILES_VIEWER,comm);
  }
  PLogObjectCreate(v);
  v->destroy     = ViewerDestroy_File;
  v->flush       = ViewerFlush_File;

  if (!PetscStrcmp(name,"stderr")) v->fd = stderr;
  else if (!PetscStrcmp(name,"stdout")) v->fd = stdout;
  else {
    v->fd        = fopen(name,"w"); 
    if (!v->fd) SETERRQ(PETSC_ERR_FILE_OPEN,0,"Cannot open viewer file");
  }
  v->format        = VIEWER_FORMAT_ASCII_DEFAULT;
  v->iformat       = 0;
  v->outputname    = 0;
#if defined(PETSC_LOG)
  PLogObjectState((PetscObject)v,"File: %s",name);
#endif
  *lab           = v;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ViewerSetFormat" /* ADIC Ignore */
/*@C
   ViewerSetFormat - Sets the format for viewers.

   Input Parameters:
.  v - the viewer
.  format - the format
.  char - optional object name

   Notes:
   Available formats include
$    VIEWER_FORMAT_ASCII_DEFAULT - default
$    VIEWER_FORMAT_ASCII_MATLAB - Matlab format
$    VIEWER_FORMAT_ASCII_IMPL - implementation-specific format
$      (which is in many cases the same as the default)
$    VIEWER_FORMAT_ASCII_INFO - basic information about object
$    VIEWER_FORMAT_ASCII_INFO_LONG - more detailed info
$       about object
$    VIEWER_FORMAT_ASCII_COMMON - identical output format for
$       all objects of a particular type
$    VIEWER_FORMAT_BINARY_NATIVE - store the object to the binary
$      file in its native format (for example, dense
$       matrices are stored as dense)
$    VIEWER_FORMAT_DRAW_BASIC - View the vector with a simple 1d plot
$    VIEWER_FORMAT_DRAW_LG - View the vector with a line graph
$    VIEWER_FORMAT_DRAW_CONTOUR - View the vector with a contour

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the Matlab format.

.keywords: Viewer, file, set, format

.seealso: ViewerFileOpenASCII(), ViewerFileOpenBinary(), MatView(), VecView(),
          ViewerPushFormat(), ViewerPopFormat(), ViewerDrawOpenX(),ViewerMatlabOpen()
@*/
int ViewerSetFormat(Viewer v,int format,char *name)
{
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->type == ASCII_FILES_VIEWER || v->type == ASCII_FILE_VIEWER) {
    v->format     = format;
    v->outputname = name;
  } else {
    v->format     = format;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ViewerPushFormat" /* ADIC Ignore */
/*@C
   ViewerPushFormat - Sets the format for file viewers.

   Input Parameters:
.  v - the viewer
.  format - the format
.  char - optional object name

   Notes:
   Available formats include
$    VIEWER_FORMAT_ASCII_DEFAULT - default
$    VIEWER_FORMAT_ASCII_MATLAB - Matlab format
$    VIEWER_FORMAT_ASCII_IMPL - implementation-specific format
$      (which is in many cases the same as the default)
$    VIEWER_FORMAT_ASCII_INFO - basic information about object
$    VIEWER_FORMAT_ASCII_INFO_LONG - more detailed info
$       about object
$    VIEWER_FORMAT_ASCII_COMMON - identical output format for
$       all objects of a particular type
$    VIEWER_FORMAT_BINARY_NATIVE - store the object to the binary
$      file in its native format (for example, dense
$       matrices are stored as dense)

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the Matlab format.

.keywords: Viewer, file, set, format

.seealso: ViewerFileOpenASCII(), ViewerFileOpenBinary(), MatView(), VecView(),
          ViewerSetFormat(), ViewerPopFormat()
@*/
int ViewerPushFormat(Viewer v,int format,char *name)
{
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->iformat > 9) SETERRQ(1,0,"Too many pushes");

  if (v->type == ASCII_FILES_VIEWER || v->type == ASCII_FILE_VIEWER) {
    v->formats[v->iformat]       = v->format;
    v->outputnames[v->iformat++] = v->outputname;
    v->format                    = format;
    v->outputname                = name;
  } else {
    v->formats[v->iformat++]     = v->format;
    v->format                    = format;
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ViewerPopFormat" /* ADIC Ignore */
/*@C
   ViewerPopFormat - Resets the format for file viewers.

   Input Parameters:
.  v - the viewer

.keywords: Viewer, file, set, format, push, pop

.seealso: ViewerFileOpenASCII(), ViewerFileOpenBinary(), MatView(), VecView(),
          ViewerSetFormat(), ViewerPushFormat()
@*/
int ViewerPopFormat(Viewer v)
{
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->iformat <= 0) return 0;

  if (v->type == ASCII_FILES_VIEWER || v->type == ASCII_FILE_VIEWER) {
    v->format     = v->formats[--v->iformat];
    v->outputname = v->outputnames[v->iformat];
  } else if (v->type == BINARY_FILE_VIEWER) {
    v->format     = v->formats[--v->iformat];
  }
  return 0;
}




