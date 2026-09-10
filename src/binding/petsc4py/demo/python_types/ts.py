import sys

import petsc4py
from scipy.integrate import odeint

petsc4py.init(sys.argv)

from petsc4py import PETSc


# The user-defined Python class implementing each time step with SciPy.
class SciPyODEInt:
    def __init__(self):
        self.solution = PETSc.Vec()
        self.function = PETSc.Vec()

    # Create PETSc vectors used to evaluate the user-provided ODE function.
    def setUp(self, ts):
        self.solution = ts.getSolution().duplicate()
        self.function = ts.getSolution().duplicate()

    # Advance the solution from the current time to t.
    def solveStep(self, ts, t, x):

        # Adapt the PETSc ODE callback to the interface expected by SciPy.
        def rhs(values, time):
            self.solution.setArray(values)
            ts.computeRHSFunction(time, self.solution, self.function)
            return self.function.getArray(readonly=True)

        # Integrate over the current PETSc time step and return the result.
        initial = ts.getSolution().getArray(readonly=True)
        solution = odeint(rhs, initial, [ts.getTime(), t])
        x[0] = solution[-1]

    # Destroy the work vectors when PETSc resets the integrator.
    def reset(self, ts):
        self.solution.destroy()
        self.function.destroy()


# Define the ODE du/dt = -u.
def rhs(ts, t, x, f):
    x.copy(f)
    f.scale(-1.0)


# Create the initial condition.
u = PETSc.Vec().createSeq(1, comm=PETSc.COMM_SELF)
u[0] = 1.0

# Create the Python TS and configure the time integration.
ts = PETSc.TS().createPython(SciPyODEInt(), comm=PETSc.COMM_SELF)
ts.setRHSFunction(rhs)
ts.setSolution(u)
ts.setTime(0.0)
ts.setTimeStep(0.125)
ts.setMaxTime(1.0)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
ts.setFromOptions()
ts.solve(u)

PETSc.Sys.Print(f'u({ts.getTime():.1f}) = {u[0]:.12f}')
