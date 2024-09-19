from petsc4py.PETSc import Object
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by PETSCVIEWERPYTHON


class PetscViewerPythonProtocol:
    def viewObject(self, viewer: Viewer, obj: Object) -> None:
        """View a generic object."""
        ...

    def setUp(self, viewer: Viewer) -> None:
        """Setup the viewer."""
        ...

    def setFromOptions(self, viewer: Viewer) -> None:
        """Process command line for customization."""
        ...

    def flush(self, viewer: Viewer) -> None:
        """Flush the viewer."""
        ...

    def view(self, viewer: Viewer, outviewer: Viewer) -> None:
        """View the viewer."""
        ...
