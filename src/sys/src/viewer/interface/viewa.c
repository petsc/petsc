/*$Id: viewa.c,v 1.15 2000/09/28 21:08:19 bsmith Exp bsmith $*/

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

#undef __FUNC__  
#define __FUNC__ "PetscViewerSetFormat" 
/*@C
   PetscViewerSetFormat - Sets the format for PetscViewers.

   Collective on PetscViewer

   Input Parameters:
+  PetscViewer - the PetscViewer
.  format - the format
-  char - optional object name

   Level: intermediate

   Notes:
   Available formats include
+    PETSC_VIEWER_FORMAT_ASCII_DEFAULT - default format
.    PETSC_VIEWER_FORMAT_ASCII_MATLAB - Matlab format
.    PETSC_VIEWER_FORMAT_ASCII_DENSE - print matrix as dense
.    PETSC_VIEWER_FORMAT_ASCII_IMPL - implementation-specific format
      (which is in many cases the same as the default)
.    PETSC_VIEWER_FORMAT_ASCII_INFO - basic information about object
.    PETSC_VIEWER_FORMAT_ASCII_INFO_LONG - more detailed info
       about object
.    PETSC_VIEWER_FORMAT_ASCII_COMMON - identical output format for
       all objects of a particular type
.    PETSC_VIEWER_FORMAT_ASCII_INDEX - (for vectors) prints the vector
       element number next to each vector entry
.    PETSC_VIEWER_FORMAT_BINARY_NATIVE - store the object to the binary
      file in its native format (for example, dense
       matrices are stored as dense)
.    PETSC_VIEWER_FORMAT_DRAW_BASIC - views the vector with a simple 1d plot
.    PETSC_VIEWER_FORMAT_DRAW_LG - views the vector with a line graph
-    PETSC_VIEWER_FORMAT_DRAW_CONTOUR - views the vector with a contour plot

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the Matlab format.

   Concepts: PetscViewer^setting format

.seealso: PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(),
          PetscViewerPushFormat(), PetscViewerPopFormat(), PetscViewerDrawOpenX(),PetscViewerSocketOpen()
@*/
int PetscViewerSetFormat(PetscViewer viewer,int format,char *name)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    viewer->format     = format;
    viewer->outputname = name;
  } else {
    viewer->format     = format;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscViewerPushFormat" 
/*@C
   PetscViewerPushFormat - Sets the format for file PetscViewers.

   Collective on PetscViewer

   Input Parameters:
+  PetscViewer - the PetscViewer
.  format - the format
-  char - optional object name

   Level: intermediate

   Notes:
   Available formats include
+    PETSC_VIEWER_FORMAT_ASCII_DEFAULT - default format
.    PETSC_VIEWER_FORMAT_ASCII_MATLAB - Matlab format
.    PETSC_VIEWER_FORMAT_ASCII_IMPL - implementation-specific format
      (which is in many cases the same as the default)
.    PETSC_VIEWER_FORMAT_ASCII_INFO - basic information about object
.    PETSC_VIEWER_FORMAT_ASCII_INFO_LONG - more detailed info
       about object
.    PETSC_VIEWER_FORMAT_ASCII_COMMON - identical output format for
       all objects of a particular type
.    PETSC_VIEWER_FORMAT_ASCII_INDEX - (for vectors) prints the vector
       element number next to each vector entry
.    PETSC_VIEWER_FORMAT_BINARY_NATIVE - store the object to the binary
      file in its native format (for example, dense
       matrices are stored as dense)
.    PETSC_VIEWER_FORMAT_DRAW_BASIC - views the vector with a simple 1d plot
.    PETSC_VIEWER_FORMAT_DRAW_LG - views the vector with a line graph
.    PETSC_VIEWER_FORMAT_DRAW_CONTOUR - views the vector with a contour plot
-    PETSC_VIEWER_FORMAT_NATIVE - for DA vectors displays vectors in DA ordering, not natural

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the Matlab format.

   Concepts: PetscViewer^setting format

.seealso: PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(),
          PetscViewerSetFormat(), PetscViewerPopFormat()
@*/
int PetscViewerPushFormat(PetscViewer viewer,int format,char *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  if (viewer->iformat > 9) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many pushes");

  viewer->formats[viewer->iformat]       = viewer->format;
  viewer->outputnames[viewer->iformat++] = viewer->outputname;
  viewer->format                    = format;
  viewer->outputname                = name;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscViewerPopFormat" 
/*@C
   PetscViewerPopFormat - Resets the format for file PetscViewers.

   Collective on PetscViewer

   Input Parameters:
.  PetscViewer - the PetscViewer

   Level: intermediate

   Concepts: PetscViewer^setting format

.seealso: PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(),
          PetscViewerSetFormat(), PetscViewerPushFormat()
@*/
int PetscViewerPopFormat(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  if (viewer->iformat <= 0) PetscFunctionReturn(0);

  viewer->format     = viewer->formats[--viewer->iformat];
  viewer->outputname = viewer->outputnames[viewer->iformat];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscViewerASCIIGetOutputname" 
int PetscViewerGetOutputname(PetscViewer viewer,char **name)
{
  PetscFunctionBegin;
  *name = viewer->outputname;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscViewerGetFormat" 
int PetscViewerGetFormat(PetscViewer viewer,int *format)
{
  PetscFunctionBegin;
  *format =  viewer->format;
  PetscFunctionReturn(0);
}



