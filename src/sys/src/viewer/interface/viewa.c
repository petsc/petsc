#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: viewa.c,v 1.4 1999/03/17 23:21:09 bsmith Exp bsmith $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

#undef __FUNC__  
#define __FUNC__ "ViewerSetFormat"
/*@C
   ViewerSetFormat - Sets the format for viewers.

   Collective on Viewer

   Input Parameters:
+  v - the viewer
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
int ViewerSetFormat(Viewer v,int format,char *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (PetscTypeCompare(v->type_name,ASCII_VIEWER)) {
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
int ViewerPushFormat(Viewer v,int format,char *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->iformat > 9) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Too many pushes");

  v->formats[v->iformat]       = v->format;
  v->outputnames[v->iformat++] = v->outputname;
  v->format                    = format;
  v->outputname                = name;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerPopFormat"
/*@C
   ViewerPopFormat - Resets the format for file viewers.

   Collective on Viewer

   Input Parameters:
.  v - the viewer

   Level: intermediate

.keywords: Viewer, file, set, format, push, pop

.seealso: ViewerASCIIOpen(), ViewerBinaryOpen(), MatView(), VecView(),
          ViewerSetFormat(), ViewerPushFormat()
@*/
int ViewerPopFormat(Viewer v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->iformat <= 0) PetscFunctionReturn(0);

  v->format     = v->formats[--v->iformat];
  v->outputname = v->outputnames[v->iformat];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerASCIIGetOutputname"
int ViewerGetOutputname(Viewer viewer, char **name)
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



