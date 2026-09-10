from petsc4py.PETSc import SNES
from petsc4py.PETSc import Vec
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by SNESPYTHON


class SNESPythonProtocol:
    def solve(self, snes: SNES, b: Vec | None, x: Vec) -> None:
        """Solve the nonlinear system with a user-defined routine.

        Implement this method to control the complete solve. Use step, preStep,
        and postStep to customize the default solve routine instead. Return the
        solution in x.

        """
        ...

    def step(self, snes: SNES, x: Vec, f: Vec, y: Vec) -> None:
        """Compute update y from solution x and residual f in the default solve."""
        ...

    def preStep(self, snes: SNES) -> None:
        """Process the solver state before each step in the default solve."""
        ...

    def postStep(self, snes: SNES) -> None:
        """Process the solver state after each step in the default solve."""
        ...

    def view(self, snes: SNES, viewer: Viewer) -> None:
        """View the nonlinear solver."""
        ...

    def setFromOptions(self, snes: SNES) -> None:
        """Process command line for customization."""
        ...

    def setUp(self, snes: SNES) -> None:
        """Perform the required setup."""
        ...

    def reset(self, snes: SNES) -> None:
        """Reset the nonlinear solver."""
        ...
