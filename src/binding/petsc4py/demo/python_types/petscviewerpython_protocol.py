from petsc4py.PETSc import Object
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by PETSCVIEWERPYTHON


class PetscViewerPythonProtocol:
    def create(self, viewer: Viewer) -> None:
        """Initialize resources when the context is attached to the viewer."""
        ...

    def destroy(self, viewer: Viewer) -> None:
        """Release resources when the context is detached from the viewer."""
        ...

    def viewObject(self, viewer: Viewer, obj: Object) -> None:
        """View a generic object."""
        ...

    def setUp(self, viewer: Viewer) -> None:
        """Set up the viewer."""
        ...

    def setFromOptions(self, viewer: Viewer) -> None:
        """Process options from the options database."""
        ...

    def flush(self, viewer: Viewer) -> None:
        """Flush the viewer."""
        ...

    def view(self, viewer: Viewer, outviewer: Viewer) -> None:
        """View the viewer."""
        ...
