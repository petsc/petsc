from petsc4py.PETSc import KSP
from petsc4py.PETSc import PC
from petsc4py.PETSc import Mat
from petsc4py.PETSc import Vec
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by PCPYTHON


class PCPythonProtocol:
    def create(self, pc: PC) -> None:
        """Initialize resources when the context is attached to the PC."""
        ...

    def destroy(self, pc: PC) -> None:
        """Release resources when the context is detached from the PC."""
        ...

    def apply(self, pc: PC, b: Vec, x: Vec) -> None:
        """Apply the preconditioner to b, storing the result in x."""
        ...

    def applySymmetricLeft(self, pc: PC, b: Vec, x: Vec) -> None:
        """Apply the symmetric left part to b, storing the result in x."""
        ...

    def applySymmetricRight(self, pc: PC, b: Vec, x: Vec) -> None:
        """Apply the symmetric right part to b, storing the result in x."""
        ...

    def applyTranspose(self, pc: PC, b: Vec, x: Vec) -> None:
        """Apply the transpose to b, storing the result in x."""
        ...

    def matApply(self, pc: PC, B: Mat, X: Mat) -> None:
        """Apply the preconditioner to B, storing the result in X."""
        ...

    def preSolve(self, pc: PC, ksp: KSP, b: Vec, x: Vec) -> None:
        """Prepare for a Krylov solve.

        This method may modify the right-hand side b and initial guess x.

        """
        ...

    def postSolve(self, pc: PC, ksp: KSP, b: Vec, x: Vec) -> None:
        """Postprocess a Krylov solve.

        This method may modify the right-hand side b and solution x.

        """
        ...

    def view(self, pc: PC, viewer: Viewer) -> None:
        """View the preconditioner."""
        ...

    def setFromOptions(self, pc: PC) -> None:
        """Process options from the options database."""
        ...

    def setUp(self, pc: PC) -> None:
        """Perform the required setup."""
        ...

    def reset(self, pc: PC) -> None:
        """Reset the preconditioner."""
        ...
