from petsc4py.PETSc import KSP
from petsc4py.PETSc import PC
from petsc4py.PETSc import Mat
from petsc4py.PETSc import Vec
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by PCPYTHON

class PCPythonProtocol:

    def apply(self, pc: PC, b: Vec, x: Vec) -> None:
        """Apply the preconditioner on vector b, return in x."""
        ...

    def applySymmetricLeft(self, pc: PC, b: Vec, x: Vec) -> None:
        """Apply the symmetric left part of the preconditioner on vector b, return in x."""
        ...

    def applySymmetricRight(self, pc: PC, b: Vec, x: Vec) -> None:
        """Apply the symmetric right part of the preconditioner on vector b, return in x."""
        ...

    def applyTranspose(self, pc: PC, b: Vec, x: Vec) -> None:
        """Apply the transposed preconditioner on vector b, return in x."""
        ...

    def applyMat(self, pc: PC, B: Mat, X: Mat) -> None:
        """Apply the preconditioner on a block of right-hand sides B, return in X."""
        ...

    def preSolve(self, pc: PC, ksp: KSP, b: Vec, x: Vec) -> None:
        """Callback called at the beginning of a Krylov method.

        This method is allowed to modify the right-hand side b and the initial guess x.

        """
        ...

    def postSolve(self, pc: PC, ksp: KSP, b: Vec, x: Vec) -> None:
        """Callback called at the end of a Krylov method.

        This method is allowed to modify the right-hand side b and the solution x.

        """
    def view(self, pc: PC, viewer: Viewer) -> None:
        """View the preconditioner."""
        ...

    def setFromOptions(self, pc: PC) -> None:
        """Process command line for customization."""
        ...

    def setUp(self, pc: PC) -> None:
        """Perform the required setup."""
        ...

    def reset(self, pc: PC) -> None:
        """Reset the preconditioner."""
        ...

