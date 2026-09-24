from petsc4py.PETSc import Mat
from petsc4py.PETSc import SNES
from petsc4py.PETSc import TS
from petsc4py.PETSc import Vec
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by TSPYTHON


class TSPythonProtocol:
    def create(self, ts: TS) -> None:
        """Initialize resources when the context is attached to the TS."""
        ...

    def destroy(self, ts: TS) -> None:
        """Release resources when the context is detached from the TS."""
        ...

    def step(self, ts: TS) -> None:
        """Advance the solution with a user-defined time-stepping routine.

        Implement this method to control the complete time step. Omit it to
        use solveStep() and adaptStep() to customize the default routine.

        """
        ...

    def rollback(self, ts: TS) -> None:
        """Roll back the time integrator's internal state by one step."""
        ...

    def interpolate(self, ts: TS, t: float, x: Vec) -> None:
        """Interpolate the solution at time t. Return the solution in x."""
        ...

    def evaluatestep(self, ts: TS, order: int, x: Vec) -> bool:
        """Evaluate the current step at a given order.

        Return the solution in x and whether the evaluation was available.

        """
        ...

    def formSNESFunction(self, snes: SNES, x: Vec, f: Vec, ts: TS) -> None:
        """Form the SNES residual f at the candidate solution x."""
        ...

    def formSNESJacobian(self, snes: SNES, x: Vec, A: Mat, B: Mat, ts: TS) -> None:
        """Form the Jacobian matrices A and B for the SNES solve at x."""
        ...

    def solveStep(self, ts: TS, t: float, x: Vec) -> None:
        """Solve for the candidate solution x at time t."""
        ...

    def adaptStep(
        self, ts: TS, t: float, x: Vec
    ) -> tuple[float, bool] | float | bool | None:
        """Choose the next time-step size and whether to accept x.

        Return the next time-step size and a flag: True to accept x, or False
        to reject it. A float accepts x and sets the next time-step size.
        A bool accepts or rejects x without changing the time-step size.
        None accepts x without changing the time-step size.

        """
        ...

    def view(self, ts: TS, viewer: Viewer) -> None:
        """View the time integrator."""
        ...

    def setFromOptions(self, ts: TS) -> None:
        """Process options from the options database."""
        ...

    def setUp(self, ts: TS) -> None:
        """Perform the required setup."""
        ...

    def reset(self, ts: TS) -> None:
        """Reset the time integrator."""
        ...
