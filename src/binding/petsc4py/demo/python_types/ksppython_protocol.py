from petsc4py.PETSc import KSP
from petsc4py.PETSc import Vec
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by KSPPYTHON


class KSPPythonProtocol:
    def solve(self, ksp: KSP, b: Vec, x: Vec) -> None:
        """Solve the linear system with right-hand side b. Return solution in x."""
        ...

    def solveTranspose(self, ksp: KSP, b: Vec, x: Vec) -> None:
        """Solve the transposed linear system with right-hand side b. Return solution in x."""
        ...

    def view(self, ksp: KSP, viewer: Viewer) -> None:
        """View the Krylov solver."""
        ...

    def setFromOptions(self, ksp: KSP) -> None:
        """Process command line for customization."""
        ...

    def setUp(self, ksp: KSP) -> None:
        """Perform the required setup."""
        ...

    def buildSolution(self, ksp: KSP, x: Vec) -> None:
        """Compute the solution vector."""
        ...

    def buildResidual(self, ksp: KSP, t: Vec, r: Vec) -> None:
        """Compute the residual vector, return it in r. t is a scratch working vector."""
        ...

    def reset(self, ksp: KSP) -> None:
        """Reset the Krylov solver."""
        ...
