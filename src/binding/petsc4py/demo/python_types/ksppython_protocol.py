from petsc4py.PETSc import KSP
from petsc4py.PETSc import Vec
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by KSPPYTHON


class KSPPythonProtocol:
    def create(self, ksp: KSP) -> None:
        """Initialize resources when the context is attached to the KSP."""
        ...

    def destroy(self, ksp: KSP) -> None:
        """Release resources when the context is detached from the KSP."""
        ...

    def solve(self, ksp: KSP, b: Vec, x: Vec) -> None:
        """Solve with right-hand side b, storing the solution in x.

        Implement this method to control the complete solve and set its
        convergence reason. Omit it to use the default iteration with step(),
        preStep(), and postStep().

        """
        ...

    def solveTranspose(self, ksp: KSP, b: Vec, x: Vec) -> None:
        """Solve the transposed system with right-hand side b and solution x.

        Implement this method to control the complete solve and set its
        convergence reason. Omit it to use the default iteration with
        stepTranspose(), preStep(), and postStep().

        """
        ...

    def step(self, ksp: KSP, b: Vec, x: Vec) -> None:
        """Update x for right-hand side b in the default iteration."""
        ...

    def stepTranspose(self, ksp: KSP, b: Vec, x: Vec) -> None:
        """Update x for the transposed system in the default iteration."""
        ...

    def preStep(self, ksp: KSP) -> None:
        """Process the solver state before a default iteration."""
        ...

    def postStep(self, ksp: KSP) -> None:
        """Process the solver state after a default iteration."""
        ...

    def view(self, ksp: KSP, viewer: Viewer) -> None:
        """View the Krylov solver."""
        ...

    def setFromOptions(self, ksp: KSP) -> None:
        """Process options from the options database."""
        ...

    def setUp(self, ksp: KSP) -> None:
        """Perform the required setup."""
        ...

    def buildSolution(self, ksp: KSP, x: Vec) -> None:
        """Compute the solution vector."""
        ...

    def buildResidual(self, ksp: KSP, t: Vec, r: Vec) -> None:
        """Compute the residual in r, using t as a work vector."""
        ...

    def reset(self, ksp: KSP) -> None:
        """Reset the Krylov solver."""
        ...
