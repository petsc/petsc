#include "petscsystypes.h"
#include <petsc/private/viewerimpl.h> /*I     "petscsys.h"   I*/

/*@
   PetscViewerPythonSetType - Initialize a `PetscViewer` object implemented in Python.

   Collective

   Input Parameters:
+  viewer - the viewer object.
-  pyname - full dotted Python name [package].module[.{class|function}]

   Options Database Key:
.  -viewer_python_type <pyname> - python class

   Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerType`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PETSCVIEWERPYTHON`, `PetscPythonInitialize()`
@*/
PetscErrorCode PetscViewerPythonSetType(PetscViewer viewer, const char pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(pyname, 2);
  PetscTryMethod(viewer, "PetscViewerPythonSetType_C", (PetscViewer, const char[]), (viewer, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscViewerPythonGetType - Get the Python name of a `PetscViewer` object implemented in Python.

   Not Collective

   Input Parameter:
.  viewer - the viewer

   Output Parameter:
.  pyname - full dotted Python name [package].module[.{class|function}]

   Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerType`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PETSCVIEWERPYTHON`, `PetscPythonInitialize()`, `PetscViewerPythonSetType()`
@*/
PetscErrorCode PetscViewerPythonGetType(PetscViewer viewer, const char *pyname[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(pyname, 2);
  PetscUseMethod(viewer, "PetscViewerPythonGetType_C", (PetscViewer, const char *[]), (viewer, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscViewerPythonViewObject - View a `PetscObject`.

   Collective

   Input Parameters:
+  viewer - the viewer object.
-  obj - the object to be viewed.

   Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerPythonCreate()`
@*/
PetscErrorCode PetscViewerPythonViewObject(PetscViewer viewer, PetscObject obj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidHeader(obj, 2);
  PetscTryMethod(viewer, "PetscViewerPythonViewObject_C", (PetscViewer, PetscObject), (viewer, obj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscViewerPythonCreate - Create a `PetscViewer` object implemented in Python.

   Collective

   Input Parameters:
+  comm - MPI communicator
-  pyname - full dotted Python name [package].module[.{class|function}]

   Output Parameter:
.  viewer - the viewer

   Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerType`, `PETSCVIEWERPYTHON`, `PetscViewerPythonSetType()`, `PetscPythonInitialize()`, `PetscViewerPythonViewObject()`
@*/
PetscErrorCode PetscViewerPythonCreate(MPI_Comm comm, const char pyname[], PetscViewer *viewer)
{
  PetscFunctionBegin;
  PetscAssertPointer(pyname, 2);
  PetscAssertPointer(viewer, 3);
  PetscCall(PetscViewerCreate(comm, viewer));
  PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERPYTHON));
  PetscCall(PetscViewerPythonSetType(*viewer, pyname));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     PETSC_VIEWER_PYTHON_ - Creates a Python `PetscViewer` shared by all MPI processes in a communicator.

     Collective

     Input Parameter:
.    comm - the MPI communicator to share the `PetscViewer`

     Level: developer

     Note:
     Unlike almost all other PETSc routines, `PETSC_VIEWER_PYTHON_()` does not return
     an error code.  It is usually used in the form
   .vb
          XXXView(XXX object, PETSC_VIEWER_PYTHON_(comm));
   .ve

.seealso: [](sec_viewers), `PetscViewer`
@*/
PetscViewer PETSC_VIEWER_PYTHON_(MPI_Comm comm)
{
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscCallNull(PetscViewerCreate(comm, &viewer));
  PetscCallNull(PetscViewerSetType(viewer, PETSCVIEWERPYTHON));
  PetscCallNull(PetscObjectRegisterDestroy((PetscObject)viewer));
  PetscFunctionReturn(viewer);
}

/*MC
   PETSCVIEWERPYTHON - A viewer implemented using Python code

  Level: beginner

  Notes:
  This is the parent viewer for any implemented in Python.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerCreate()`, `VecView()`, `DMView()`, `DMPLEX`, `PETSCVIEWERPYVISTA`
M*/
