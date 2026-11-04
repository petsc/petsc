/*
     PetscViewers are objects where other objects can be looked at or stored.
*/

#pragma once

#include <petscsystypes.h>

/* MANSEC = Sys */
/* SUBMANSEC = Viewer */

/*S
  PetscViewer - Abstract PETSc object for displaying in ASCII, saving to a binary file, graphically displaying, etc.
                PETSc objects and their data

  Level: beginner

  Notes:
  Each PETSc class, for example `Vec`, has a viewer method associated with that class, for example `VecView()`, that can be used
  to view, display, store to a file information about that object, etc. Each class also has a method that uses
  the options database to view the object, for example `VecViewFromOptions()`.

  See `PetscViewerType` for a list of all `PetscViewer` types.

.seealso: [](sec_viewers), `PetscViewerType`, `PETSCVIEWERASCII`, `PetscViewerCreate()`, `PetscViewerSetType()`,
          `VecView()`, `VecViewFromOptions()`, `PetscObjectView()`
S*/
typedef struct _p_PetscViewer *PetscViewer;

/*J
   PetscViewerType - String with the name of a PETSc `PetscViewer` implementation

   Level: beginner

.seealso: [](sec_viewers), `PetscViewerSetType()`, `PetscViewer`, `PetscViewerRegister()`, `PetscViewerCreate()`
J*/
typedef const char *PetscViewerType;
#define PETSCVIEWERSOCKET      "socket"
#define PETSCVIEWERASCII       "ascii"
#define PETSCVIEWERBINARY      "binary"
#define PETSCVIEWERSTRING      "string"
#define PETSCVIEWERDRAW        "draw"
#define PETSCVIEWERVU          "vu"
#define PETSCVIEWERMATHEMATICA "mathematica"
#define PETSCVIEWERHDF5        "hdf5"
#define PETSCVIEWERVTK         "vtk"
#define PETSCVIEWERMATLAB      "matlab"
#define PETSCVIEWERSAWS        "saws"
#define PETSCVIEWERGLVIS       "glvis"
#define PETSCVIEWERADIOS       "adios"
#define PETSCVIEWEREXODUSII    "exodusii"
#define PETSCVIEWERCGNS        "cgns"
#define PETSCVIEWERPYTHON      "python"
#define PETSCVIEWERPYVISTA     "pyvista"

/*E
    PetscViewerGLVisType - indicates what type of `PETSCVIEWERGLVIS` viewer to use

    Values:
+   `PETSC_VIEWER_GLVIS_DUMP`   - save the data to a file
-   `PETSC_VIEWER_GLVIS_SOCKET` - communicate the data to another program via a socket

    Level: beginner

.seealso: [](sec_viewers), `PETSCVIEWERGLVIS`, `PetscViewerGLVisOpen()`
E*/
typedef enum {
  PETSC_VIEWER_GLVIS_DUMP,
  PETSC_VIEWER_GLVIS_SOCKET
} PetscViewerGLVisType;

/*E
   PetscViewerFormat - Way a viewer presents the object

   Values:
+    `PETSC_VIEWER_DEFAULT`           - default format for the specific object being viewed
.    `PETSC_VIEWER_ASCII_MATLAB`      - MATLAB format
.    `PETSC_VIEWER_ASCII_DENSE`       - print matrix as a dense two dimensiona array
.    `PETSC_VIEWER_ASCII_IMPL`        - implementation-specific format (which is in many cases the same as the default)
.    `PETSC_VIEWER_ASCII_INFO`        - basic information about object
.    `PETSC_VIEWER_ASCII_INFO_DETAIL` - more detailed info about object (but still not vector or matrix entries)
.    `PETSC_VIEWER_ASCII_COMMON`      - identical output format for all objects of a particular type
.    `PETSC_VIEWER_ASCII_INDEX`       - (for vectors) prints the vector  element number next to each vector entry
.    `PETSC_VIEWER_ASCII_SYMMODU`     - print parallel vectors without indicating the MPI process ranges that own the entries
.    `PETSC_VIEWER_ASCII_VTK`         - outputs the object to a VTK file (deprecated since v3.14)
.    `PETSC_VIEWER_NATIVE`            - store the object to the binary file in its native format (for example, dense
                                        matrices are stored as dense), `DMDA` vectors are dumped directly to the
                                        file instead of being first put in the natural ordering
.    `PETSC_VIEWER_ASCII_LATEX`       - output the data in LaTeX
.    `PETSC_VIEWER_BINARY_MATLAB`     - output additional information that can be used to read the data into MATLAB
.    `PETSC_VIEWER_DRAW_BASIC`        - views the vector with a simple 1d plot
.    `PETSC_VIEWER_DRAW_LG`           - views the vector with a line graph
-    `PETSC_VIEWER_DRAW_CONTOUR`      - views the vector with a contour plot

   Level: beginner

   Note:
   A variety of specialized formats also exist

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerType`, `PetscViewerPushFormat()`, `PetscViewerPopFormat()`
E*/
typedef enum {
  PETSC_VIEWER_DEFAULT,
  PETSC_VIEWER_ASCII_MATLAB,
  PETSC_VIEWER_ASCII_MATHEMATICA,
  PETSC_VIEWER_ASCII_IMPL,
  PETSC_VIEWER_ASCII_INFO,
  PETSC_VIEWER_ASCII_INFO_DETAIL,
  PETSC_VIEWER_ASCII_COMMON,
  PETSC_VIEWER_ASCII_SYMMODU,
  PETSC_VIEWER_ASCII_INDEX,
  PETSC_VIEWER_ASCII_DENSE,
  PETSC_VIEWER_ASCII_MATRIXMARKET,
  PETSC_VIEWER_ASCII_PCICE,
  PETSC_VIEWER_ASCII_PYTHON,
  PETSC_VIEWER_ASCII_FACTOR_INFO,
  PETSC_VIEWER_ASCII_LATEX,
  PETSC_VIEWER_ASCII_XML,
  PETSC_VIEWER_ASCII_FLAMEGRAPH,
  PETSC_VIEWER_ASCII_GLVIS,
  PETSC_VIEWER_ASCII_CSV,
  PETSC_VIEWER_DRAW_BASIC,
  PETSC_VIEWER_DRAW_LG,
  PETSC_VIEWER_DRAW_LG_XRANGE,
  PETSC_VIEWER_DRAW_CONTOUR,
  PETSC_VIEWER_DRAW_PORTS,
  PETSC_VIEWER_VTK_VTS,
  PETSC_VIEWER_VTK_VTR,
  PETSC_VIEWER_VTK_VTU,
  PETSC_VIEWER_BINARY_MATLAB,
  PETSC_VIEWER_NATIVE,
  PETSC_VIEWER_HDF5_PETSC,
  PETSC_VIEWER_HDF5_VIZ,
  PETSC_VIEWER_HDF5_XDMF,
  PETSC_VIEWER_HDF5_MAT,
  PETSC_VIEWER_NOFORMAT,
  PETSC_VIEWER_LOAD_BALANCE,
  PETSC_VIEWER_FAILED,
  PETSC_VIEWER_ALL
} PetscViewerFormat;
PETSC_EXTERN const char *const PetscViewerFormats[];

/*S
  PetscViewerAndFormat - A struct that contains both a `PetscViewer` and a `PetscViewerFormat` plus some optional situation-dependent data

  Level: beginner

  Note:
  Used by most monitor functions including, for example, `KSPMonitorResidual()`

.seealso: [](sec_viewers), `PetscViewerType`, `PETSCVIEWERASCII`, `PetscViewerCreate()`, `PetscViewerSetType()`,
          `VecView()`, `VecViewFromOptions()`, `PetscObjectView()`, `PetscViewerFormat`, `PetscViewer`,
          `PetscViewerAndFormatCreate()`, `PetscViewerAndFormatDestroy()`, `KSPMonitorResidual()`
S*/
typedef struct {
  PetscViewer        viewer;
  PetscViewerFormat  format;
  PetscInt           view_interval;
  void              *data;
  PetscCtxDestroyFn *data_destroy;
} PetscViewerAndFormat;

/*E
   PetscViewerVTKFieldType - Categorizes a field that is being written through a `PETSCVIEWERVTK` viewer so that the VTK writer can place it on the correct mesh entity and with the correct component layout

   Values:
+   `PETSC_VTK_INVALID`            - sentinel for an uninitialized entry
.   `PETSC_VTK_POINT_FIELD`        - scalar (or generic multi-component) field stored at mesh points
.   `PETSC_VTK_POINT_VECTOR_FIELD` - vector-valued field stored at mesh points
.   `PETSC_VTK_CELL_FIELD`         - scalar (or generic multi-component) field stored on mesh cells
-   `PETSC_VTK_CELL_VECTOR_FIELD`  - vector-valued field stored on mesh cells

   Level: developer

.seealso: `PETSCVIEWERVTK`, `PetscViewerVTKAddField()`, `PetscViewerVTKOpen()`, `PetscViewer`
E*/
typedef enum {
  PETSC_VTK_INVALID,
  PETSC_VTK_POINT_FIELD,
  PETSC_VTK_POINT_VECTOR_FIELD,
  PETSC_VTK_CELL_FIELD,
  PETSC_VTK_CELL_VECTOR_FIELD
} PetscViewerVTKFieldType;

/*S
   PetscViewers - Abstract collection of `PetscViewer`s. It is stored as an expandable array of viewers.

   Level: intermediate

.seealso: [](sec_viewers), `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerType`, `PetscViewer`, `PetscViewersCreate()`,
          `PetscViewersGetViewer()`
S*/
typedef struct _n_PetscViewers *PetscViewers;
