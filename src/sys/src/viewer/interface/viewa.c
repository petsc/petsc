/*$Id: viewa.c,v 1.10 2000/01/11 20:59:04 bsmith Exp bsmith $*/

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"ViewerSetFormat" 
/*@C
   ViewerSetFormat - Sets the format for viewers.

   Collective on Viewer

   Input Parameters:
+  viewer - the viewer
.  format - the format
-  char - optional object name

   Level: intermediate

   Notes:
   Available formats include
+    VIEWER_FORMAT_ASCII_DEFAULT - default format
.    VIEWER_FORMAT_ASCII_MATLAB - Matlab format
.    VIEWER_FORMAT_ASCII_DENSE - print matrix as dense
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

.seealso: ViewerASCIIOpen(), ViewerBinaryOpen(), MatView(), VecView(),
          ViewerPushFormat(), ViewerPopFormat(), ViewerDrawOpenX(),ViewerSocketOpen()
@*/
int ViewerSetFormat(Viewer viewer,int format,char *name)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    viewer->format     = format;
    viewer->outputname = name;
  } else {
    viewer->format     = format;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"ViewerPushFormat" 
/*@C
   ViewerPushFormat - Sets the format for file viewers.

   Collective on Viewer

   Input Parameters:
+  viewer - the viewer
.  format - the format
-  char - optional object name

   Level: intermediate

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
.    VIEWER_FORMAT_DRAW_CONTOUR - views the vector with a contour plot
-    VIEWER_FORMAT_NATIVE - for DA vectors displays vectors in DA ordering, not natural

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the Matlab format.

.keywords: Viewer, file, set, format

.seealso: ViewerASCIIOpen(), ViewerBinaryOpen(), MatView(), VecView(),
          ViewerSetFormat(), ViewerPopFormat()
@*/
int ViewerPushFormat(Viewer viewer,int format,char *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  if (viewer->iformat > 9) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Too many pushes");

  viewer->formats[viewer->iformat]       = viewer->format;
  viewer->outputnames[viewer->iformat++] = viewer->outputname;
  viewer->format                    = format;
  viewer->outputname                = name;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"ViewerPopFormat" 
/*@C
   ViewerPopFormat - Resets the format for file viewers.

   Collective on Viewer

   Input Parameters:
.  viewer - the viewer

   Level: intermediate

.keywords: Viewer, file, set, format, push, pop

.seealso: ViewerASCIIOpen(), ViewerBinaryOpen(), MatView(), VecView(),
          ViewerSetFormat(), ViewerPushFormat()
@*/
int ViewerPopFormat(Viewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);
  if (viewer->iformat <= 0) PetscFunctionReturn(0);

  viewer->format     = viewer->formats[--viewer->iformat];
  viewer->outputname = viewer->outputnames[viewer->iformat];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"ViewerASCIIGetOutputname" 
int ViewerGetOutputname(Viewer viewer,char **name)
{
  PetscFunctionBegin;
  *name = viewer->outputname;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"ViewerGetFormat" 
int ViewerGetFormat(Viewer viewer,int *format)
{
  PetscFunctionBegin;
  *format =  viewer->format;
  PetscFunctionReturn(0);
}



