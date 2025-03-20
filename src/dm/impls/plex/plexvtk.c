#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>

PetscErrorCode DMPlexVTKGetCellType_Internal(DM dm, PetscInt dim, PetscInt corners, PetscInt *cellType)
{
  PetscFunctionBegin;
  *cellType = -1;
  switch (dim) {
  case 0:
    switch (corners) {
    case 1:
      *cellType = 1; /* VTK_VERTEX */
      break;
    default:
      break;
    }
    break;
  case 1:
    switch (corners) {
    case 2:
      *cellType = 3; /* VTK_LINE */
      break;
    case 3:
      *cellType = 21; /* VTK_QUADRATIC_EDGE */
      break;
    default:
      break;
    }
    break;
  case 2:
    switch (corners) {
    case 3:
      *cellType = 5; /* VTK_TRIANGLE */
      break;
    case 4:
      *cellType = 9; /* VTK_QUAD */
      break;
    case 6:
      *cellType = 22; /* VTK_QUADRATIC_TRIANGLE */
      break;
    case 9:
      *cellType = 23; /* VTK_QUADRATIC_QUAD */
      break;
    default:
      break;
    }
    break;
  case 3:
    switch (corners) {
    case 4:
      *cellType = 10; /* VTK_TETRA */
      break;
    case 5:
      *cellType = 14; /* VTK_PYRAMID */
      break;
    case 6:
      *cellType = 13; /* VTK_WEDGE */
      break;
    case 8:
      *cellType = 12; /* VTK_HEXAHEDRON */
      break;
    case 10:
      *cellType = 24; /* VTK_QUADRATIC_TETRA */
      break;
    case 27:
      *cellType = 29; /* VTK_QUADRATIC_HEXAHEDRON */
      break;
    default:
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexVTKWriteAll - Write a file containing all the fields that have been provided to the viewer

  Collective

  Input Parameters:
+ odm    - The `DMPLEX` specifying the mesh, passed as a `PetscObject`
- viewer - viewer of type `PETSCVIEWERVTK`

  Level: developer

  Note:
  This function is a callback used by the VTK viewer to actually write the file.
  The reason for this odd model is that the VTK file format does not provide any way to write one field at a time.
  Instead, metadata for the entire file needs to be available up-front before you can start writing the file.

.seealso: [](ch_unstructured), `DM`, `PETSCVIEWEREXODUSII`, `DMPLEX`, `PETSCVIEWERVTK`
@*/
PetscErrorCode DMPlexVTKWriteAll(PetscObject odm, PetscViewer viewer)
{
  DM dm = (DM)odm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 2, PETSCVIEWERVTK);
  switch (viewer->format) {
  case PETSC_VIEWER_DEFAULT:
  case PETSC_VIEWER_VTK_VTU:
    PetscCall(DMPlexVTKWriteAll_VTU(dm, viewer));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No support for format '%s'", PetscViewerFormats[viewer->format]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
