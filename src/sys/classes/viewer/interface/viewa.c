
#include <petsc/private/viewerimpl.h>  /*I "petscsys.h" I*/

const char *const PetscViewerFormats[] = {
  "DEFAULT",
  "ASCII_MATLAB",
  "ASCII_MATHEMATICA",
  "ASCII_IMPL",
  "ASCII_INFO",
  "ASCII_INFO_DETAIL",
  "ASCII_COMMON",
  "ASCII_SYMMODU",
  "ASCII_INDEX",
  "ASCII_DENSE",
  "ASCII_MATRIXMARKET",
  "ASCII_VTK",
  "ASCII_VTK_CELL",
  "ASCII_VTK_COORDS",
  "ASCII_PCICE",
  "ASCII_PYTHON",
  "ASCII_FACTOR_INFO",
  "ASCII_LATEX",
  "ASCII_XML",
  "ASCII_FLAMEGRAPH",
  "ASCII_GLVIS",
  "ASCII_CSV",
  "DRAW_BASIC",
  "DRAW_LG",
  "DRAW_LG_XRANGE",
  "DRAW_CONTOUR",
  "DRAW_PORTS",
  "VTK_VTS",
  "VTK_VTR",
  "VTK_VTU",
  "BINARY_MATLAB",
  "NATIVE",
  "HDF5_PETSC",
  "HDF5_VIZ",
  "HDF5_XDMF",
  "HDF5_MAT",
  "NOFORMAT",
  "LOAD_BALANCE",
  "FAILED",
  "ALL",
  "PetscViewerFormat",
  "PETSC_VIEWER_",
  NULL
};

/*@C
   PetscViewerSetFormat - Sets the format for PetscViewers.

   Logically Collective on PetscViewer

   This routine is deprecated, you should use PetscViewerPushFormat()/PetscViewerPopFormat()

   Input Parameters:
+  viewer - the PetscViewer
-  format - the format

   Level: intermediate

   Notes:
   Available formats include
+    PETSC_VIEWER_DEFAULT - default format
.    PETSC_VIEWER_ASCII_MATLAB - MATLAB format
.    PETSC_VIEWER_ASCII_DENSE - print matrix as dense
.    PETSC_VIEWER_ASCII_IMPL - implementation-specific format
      (which is in many cases the same as the default)
.    PETSC_VIEWER_ASCII_INFO - basic information about object
.    PETSC_VIEWER_ASCII_INFO_DETAIL - more detailed info
       about object
.    PETSC_VIEWER_ASCII_COMMON - identical output format for
       all objects of a particular type
.    PETSC_VIEWER_ASCII_INDEX - (for vectors) prints the vector
       element number next to each vector entry
.    PETSC_VIEWER_ASCII_SYMMODU - print parallel vectors without
       indicating the processor ranges
.    PETSC_VIEWER_ASCII_VTK - outputs the object to a VTK file (deprecated since v3.14)
.    PETSC_VIEWER_NATIVE - store the object to the binary
       file in its native format (for example, dense
       matrices are stored as dense), DMDA vectors are dumped directly to the
       file instead of being first put in the natural ordering
.    PETSC_VIEWER_DRAW_BASIC - views the vector with a simple 1d plot
.    PETSC_VIEWER_DRAW_LG - views the vector with a line graph
-    PETSC_VIEWER_DRAW_CONTOUR - views the vector with a contour plot

   These formats are most often used for viewing matrices and vectors.

   If a format (for example PETSC_VIEWER_DRAW_CONTOUR) was applied to a viewer
  where it didn't apply (PETSC_VIEWER_STDOUT_WORLD) it cause the default behavior
  for that viewer to be used.

    Note: This supports passing in a NULL for the viewer for use in the debugger, but it should never be called in the code with a NULL viewer

.seealso: PetscViewerGetFormat(), PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(), PetscViewerType,
          PetscViewerPushFormat(), PetscViewerPopFormat(), PetscViewerDrawOpen(),PetscViewerSocketOpen()
@*/
PetscErrorCode  PetscViewerSetFormat(PetscViewer viewer,PetscViewerFormat format)
{
  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveEnum(viewer,format,2);
  viewer->format = format;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerPushFormat - Sets the format for file PetscViewers.

   Logically Collective on PetscViewer

   Input Parameters:
+  viewer - the PetscViewer
-  format - the format

   Level: intermediate

   Notes:
   Available formats include
+    PETSC_VIEWER_DEFAULT - default format
.    PETSC_VIEWER_ASCII_MATLAB - MATLAB format
.    PETSC_VIEWER_ASCII_IMPL - implementation-specific format
      (which is in many cases the same as the default)
.    PETSC_VIEWER_ASCII_INFO - basic information about object
.    PETSC_VIEWER_ASCII_INFO_DETAIL - more detailed info
       about object
.    PETSC_VIEWER_ASCII_COMMON - identical output format for
       all objects of a particular type
.    PETSC_VIEWER_ASCII_INDEX - (for vectors) prints the vector
       element number next to each vector entry
.    PETSC_VIEWER_NATIVE - store the object to the binary
       file in its native format (for example, dense
       matrices are stored as dense), for DMDA vectors displays vectors in DMDA ordering, not natural
.    PETSC_VIEWER_DRAW_BASIC - views the vector with a simple 1d plot
.    PETSC_VIEWER_DRAW_LG - views the vector with a line graph
.    PETSC_VIEWER_DRAW_CONTOUR - views the vector with a contour plot
-    PETSC_VIEWER_ASCII_XML - saves the data in XML format, needed for PetscLogView() when viewing with PetscLogNestedBegin()

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the MATLAB format.

.seealso: PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(),
          PetscViewerSetFormat(), PetscViewerPopFormat()
@*/
PetscErrorCode  PetscViewerPushFormat(PetscViewer viewer,PetscViewerFormat format)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveEnum(viewer,format,2);
  PetscCheck(viewer->iformat <= PETSCVIEWERFORMATPUSHESMAX-1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many PetscViewerPushFormat(), perhaps you forgot PetscViewerPopFormat()?");

  viewer->formats[viewer->iformat++] = viewer->format;
  viewer->format                     = format;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerPopFormat - Resets the format for file PetscViewers.

   Logically Collective on PetscViewer

   Input Parameters:
.  viewer - the PetscViewer

   Level: intermediate

.seealso: PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(),
          PetscViewerSetFormat(), PetscViewerPushFormat()
@*/
PetscErrorCode  PetscViewerPopFormat(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (viewer->iformat <= 0) PetscFunctionReturn(0);

  viewer->format = viewer->formats[--viewer->iformat];
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerGetFormat - Gets the format for PetscViewers.

   Not collective

   Input Parameter:
.  viewer - the PetscViewer

   Output Parameter:
.  format - the format

   Level: intermediate

   Notes:
   Available formats include
+    PETSC_VIEWER_DEFAULT - default format
.    PETSC_VIEWER_ASCII_MATLAB - MATLAB format
.    PETSC_VIEWER_ASCII_DENSE - print matrix as dense
.    PETSC_VIEWER_ASCII_IMPL - implementation-specific format
      (which is in many cases the same as the default)
.    PETSC_VIEWER_ASCII_INFO - basic information about object
.    PETSC_VIEWER_ASCII_INFO_DETAIL - more detailed info
       about object
.    PETSC_VIEWER_ASCII_COMMON - identical output format for
       all objects of a particular type
.    PETSC_VIEWER_ASCII_INDEX - (for vectors) prints the vector
       element number next to each vector entry
.    PETSC_VIEWER_ASCII_SYMMODU - print parallel vectors without
       indicating the processor ranges
.    PETSC_VIEWER_ASCII_VTK - outputs the object to a VTK file (deprecated since v3.14)
.    PETSC_VIEWER_NATIVE - store the object to the binary
       file in its native format (for example, dense
       matrices are stored as dense), DMDA vectors are dumped directly to the
       file instead of being first put in the natural ordering
.    PETSC_VIEWER_DRAW_BASIC - views the vector with a simple 1d plot
.    PETSC_VIEWER_DRAW_LG - views the vector with a line graph
-    PETSC_VIEWER_DRAW_CONTOUR - views the vector with a contour plot

   These formats are most often used for viewing matrices and vectors.

   If a format (for example PETSC_VIEWER_DRAW_CONTOUR) was applied to a viewer
  where it didn't apply (PETSC_VIEWER_STDOUT_WORLD) it cause the default behavior
  for that viewer to be used.

.seealso: PetscViewerSetFormat(), PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(), PetscViewerType,
          PetscViewerPushFormat(), PetscViewerPopFormat(), PetscViewerDrawOpen(),PetscViewerSocketOpen()
@*/
PetscErrorCode PetscViewerGetFormat(PetscViewer viewer,PetscViewerFormat *format)
{
  PetscFunctionBegin;
  *format =  viewer->format;
  PetscFunctionReturn(0);
}
