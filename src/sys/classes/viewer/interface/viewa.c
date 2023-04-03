
#include <petsc/private/viewerimpl.h> /*I "petscsys.h" I*/

const char *const PetscViewerFormats[] = {"DEFAULT", "ASCII_MATLAB", "ASCII_MATHEMATICA", "ASCII_IMPL", "ASCII_INFO", "ASCII_INFO_DETAIL", "ASCII_COMMON", "ASCII_SYMMODU", "ASCII_INDEX", "ASCII_DENSE", "ASCII_MATRIXMARKET", "ASCII_VTK", "ASCII_VTK_CELL", "ASCII_VTK_COORDS", "ASCII_PCICE", "ASCII_PYTHON", "ASCII_FACTOR_INFO", "ASCII_LATEX", "ASCII_XML", "ASCII_FLAMEGRAPH", "ASCII_GLVIS", "ASCII_CSV", "DRAW_BASIC", "DRAW_LG", "DRAW_LG_XRANGE", "DRAW_CONTOUR", "DRAW_PORTS", "VTK_VTS", "VTK_VTR", "VTK_VTU", "BINARY_MATLAB", "NATIVE", "HDF5_PETSC", "HDF5_VIZ", "HDF5_XDMF", "HDF5_MAT", "NOFORMAT", "LOAD_BALANCE", "FAILED", "ALL", "PetscViewerFormat", "PETSC_VIEWER_", NULL};

/*@C
   PetscViewerSetFormat - Sets the format for a `PetscViewer`.

   Logically Collective

   This routine is deprecated, you should use `PetscViewerPushFormat()`/`PetscViewerPopFormat()`

   Input Parameters:
+  viewer - the `PetscViewer`
-  format - the format

   Level: deprecated

   Note:
   See `PetscViewerFormat` for available values

.seealso: [](sec_viewers), `PetscViewerGetFormat()`, `PetscViewerASCIIOpen()`, `PetscViewerBinaryOpen()`, `MatView()`, `VecView()`, `PetscViewerType`,
          `PetscViewerPushFormat()`, `PetscViewerPopFormat()`, `PetscViewerDrawOpen()`, `PetscViewerSocketOpen()`
@*/
PetscErrorCode PetscViewerSetFormat(PetscViewer viewer, PetscViewerFormat format)
{
  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(viewer, format, 2);
  viewer->format = format;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerPushFormat - Sets the format for a `PetscViewer`.

   Logically Collective

   Input Parameters:
+  viewer - the `PetscViewer`
-  format - the format

   Level: intermediate

   Note:
   See `PetscViewerFormat` for available values

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerFormat`, `PetscViewerASCIIOpen()`, `PetscViewerBinaryOpen()`, `MatView()`, `VecView()`,
          `PetscViewerSetFormat()`, `PetscViewerPopFormat()`
@*/
PetscErrorCode PetscViewerPushFormat(PetscViewer viewer, PetscViewerFormat format)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(viewer, format, 2);
  PetscCheck(viewer->iformat <= PETSCVIEWERFORMATPUSHESMAX - 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many PetscViewerPushFormat(), perhaps you forgot PetscViewerPopFormat()?");

  viewer->formats[viewer->iformat++] = viewer->format;
  viewer->format                     = format;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerPopFormat - Resets the format for a `PetscViewer` to the value it had before the previous call to `PetscViewerPushFormat()`

   Logically Collective

   Input Parameter:
.  viewer - the `PetscViewer`

   Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerFormat`, `PetscViewerASCIIOpen()`, `PetscViewerBinaryOpen()`, `MatView()`, `VecView()`,
          `PetscViewerSetFormat()`, `PetscViewerPushFormat()`
@*/
PetscErrorCode PetscViewerPopFormat(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (viewer->iformat <= 0) PetscFunctionReturn(PETSC_SUCCESS);

  viewer->format = viewer->formats[--viewer->iformat];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerGetFormat - Gets the current format for `PetscViewer`.

   Not Collective

   Input Parameter:
.  viewer - the `PetscViewer`

   Output Parameter:
.  format - the format

   Level: intermediate

   Note:
   See `PetscViewerFormat` for available values

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerASCIIOpen()`, `PetscViewerBinaryOpen()`, `MatView()`, `VecView()`, `PetscViewerType`,
          `PetscViewerPushFormat()`, `PetscViewerPopFormat()`, `PetscViewerDrawOpen()`, `PetscViewerSocketOpen()`
@*/
PetscErrorCode PetscViewerGetFormat(PetscViewer viewer, PetscViewerFormat *format)
{
  PetscFunctionBegin;
  *format = viewer->format;
  PetscFunctionReturn(PETSC_SUCCESS);
}
