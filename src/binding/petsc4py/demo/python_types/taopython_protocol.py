from petsc4py.PETSc import TAO
from petsc4py.PETSc import Vec
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by TAOPYTHON


class TAOPythonProtocol:
    def create(self, tao: TAO) -> None:
        """Initialize resources when the context is attached to the TAO."""
        ...

    def destroy(self, tao: TAO) -> None:
        """Release resources when the context is detached from the TAO."""
        ...

    def setFromOptions(self, tao: TAO) -> None:
        """Process options from the options database."""
        ...

    def setUp(self, tao: TAO) -> None:
        """Set up the optimizer."""
        ...

    def solve(self, tao: TAO) -> None:
        """Solve the optimization problem with a user-defined routine.

        Implement this method to control the complete solve and set its
        convergence reason. Omit it to use step(), preStep(), and postStep()
        to customize the default solve.

        """
        ...

    def step(self, tao: TAO, x: Vec, g: Vec | None, s: Vec | None) -> None:
        """Compute gradient g and search direction s at x.

        Both g and s are None when the optimizer has no gradient routine.

        """
        ...

    def preStep(self, tao: TAO) -> None:
        """Process the optimizer state before a default step."""
        ...

    def postStep(self, tao: TAO) -> None:
        """Process the optimizer state after a default step."""
        ...

    def view(self, tao: TAO, viewer: Viewer) -> None:
        """View the optimizer."""
        ...
