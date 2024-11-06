#include "petscviewer.h"
#include <petsc/private/viewerimpl.h> /*I     "petscsys.h"   I*/

/*@C
     PETSC_VIEWER_PYVISTA_ - Creates a PyVista `PetscViewer` shared by all MPI processes in a communicator.

     Collective

     Input Parameter:
.    comm - the MPI communicator to share the `PetscViewer`

     Level: developer

     Note:
     Unlike almost all other PETSc routines, `PETSC_VIEWER_PYVISTA_()` does not return
     an error code.  It is usually used in the form
   .vb
          XXXView(XXX object, PETSC_VIEWER_PYVISTA_(comm));
   .ve

.seealso: [](sec_viewers), `PetscViewer`
@*/
PetscViewer PETSC_VIEWER_PYVISTA_(MPI_Comm comm)
{
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscCallNull(PetscViewerCreate(comm, &viewer));
  PetscCallNull(PetscViewerSetType(viewer, PETSCVIEWERPYVISTA));
  PetscCallNull(PetscViewerSetFromOptions(viewer));
  PetscCallNull(PetscObjectRegisterDestroy((PetscObject)viewer));
  PetscFunctionReturn(viewer);
}

/*MC
   PETSCVIEWERPYVISTA - A PyVista viewer implemented using Python code

  Level: beginner

  Notes:
  Currently the `DM` viewer only supports `DMPLEX` meshes.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerCreate()`, `VecView()`, `DMView()`, `DMPLEX`
M*/
PETSC_EXTERN PetscErrorCode PetscViewerCreate_PyVista(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscPythonInitialize(NULL, NULL));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERPYTHON));
  PetscCall(PetscObjectChangeTypeName((PetscObject)viewer, PETSCVIEWERPYVISTA));
  PetscCall(PetscViewerPythonSetType(viewer, "petsc4py.lib._pytypes.viewer.petscpyvista.PetscPyVista"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
