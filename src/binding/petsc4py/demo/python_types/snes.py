import sys

import numpy as np
import petsc4py
from scipy.optimize import root

petsc4py.init(sys.argv)

from petsc4py import PETSc


# The user-defined Python class implementing the nonlinear solve with SciPy.
class SciPyRoot:
    # Solve the complete nonlinear system.
    def solve(self, snes, b, x):
        # Create PETSc vectors used to evaluate the user-provided function.
        work_x = x.duplicate()
        work_f = x.duplicate()

        # Adapt the PETSc function callback to the interface expected by SciPy.
        def function(values):
            work_x.setArray(values)
            snes.computeFunction(work_x, work_f)
            return work_f.getArray(readonly=True)

        # Report an inner-solver failure unless SciPy completes successfully.
        reason = PETSc.SNES.ConvergedReason.DIVERGED_INNER
        message = None
        try:
            # Solve the system and return the solution to PETSc.
            result = root(function, x.getArray(readonly=True))
            x[:] = result.x[:]

            # Record the solver state in the PETSc SNES object.
            snes.setIterationNumber(result.nfev)
            snes.setFunctionNorm(float(np.linalg.norm(result.fun)))
            if result.success:
                reason = PETSc.SNES.ConvergedReason.CONVERGED_ITS
            else:
                message = str(result.message)
        except Exception as error:
            message = str(error) or type(error).__name__
        finally:
            # Clean up work vectors.
            work_x.destroy()
            work_f.destroy()

        snes.setConvergedReason(reason)

        # Handle -snes_error_if_not_converged here instead of relying on PETSc's
        # standard nonconvergence error. Raising a Python exception lets PETSc preserve
        # SciPy's failure message when it converts the exception to a PETSc error.
        if reason < 0 and snes.getErrorIfNotConverged():
            if message:
                msg = f'SciPy solve failed: {message}'
                raise RuntimeError(msg)
            raise RuntimeError('SciPy solve failed')


# Define the nonlinear equation x**2 - 2 = 0.
def function(snes, x, f):
    f[0] = x[0] ** 2 - 2.0


# Create the initial guess and residual vector.
x = PETSc.Vec().createSeq(1, comm=PETSc.COMM_SELF)
x[0] = 1.0
f = x.duplicate()

# Create the Python SNES, set the equation, and solve it.
snes = PETSc.SNES().createPython(SciPyRoot(), comm=PETSc.COMM_SELF)
snes.setFunction(function, f)
snes.setErrorIfNotConverged()
snes.setFromOptions()
snes.solve(None, x)

PETSc.Sys.Print(f'sqrt(2) = {x[0]:.12f}')
