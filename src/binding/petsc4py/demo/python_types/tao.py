import sys

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc


# The user-defined Python class implementing gradient descent.
class myGradientDescent:
    def create(self, tao):
        # Create a line search type with constant step size.
        self._ls = PETSc.TAOLineSearch().create(comm=PETSc.COMM_SELF)
        self._ls.useTAORoutine(tao)
        self._ls.setType(PETSc.TAOLineSearch.Type.UNIT)
        self._ls.setInitialStepLength(0.2)

    def destroy(self, tao):
        self._ls.destroy()

    def solve(self, tao):
        # Evaluate the objective and gradient at the initial guess.
        x = tao.getSolution()
        gradient = tao.getGradient()[0]
        f = tao.computeObjectiveGradient(x, gradient)
        tao.monitor(f=f, res=gradient.norm(), step=1.0)
        if tao.checkConverged() != PETSc.TAO.ConvergedReason.CONTINUE_ITERATING:
            return

        # Prepare search direction for line search.
        search_direction = gradient.duplicate()

        # Optimization loop.
        for it in range(1, tao.getMaximumIterations() + 1):
            # Search in the negative gradient direction.
            gradient.copy(search_direction)
            search_direction.scale(-1)

            # Update x and evaluate the new objective and gradient.
            f, step, reason = self._ls.apply(x, gradient, search_direction)
            if reason < 0:
                tao.setConvergedReason(PETSc.TAO.ConvergedReason.DIVERGED_LS_FAILURE)
                break

            # Report the completed step and check for convergence or divergence.
            tao.setIterationNumber(it)
            tao.monitor(f=f, res=gradient.norm(), step=step)
            if tao.checkConverged() != PETSc.TAO.ConvergedReason.CONTINUE_ITERATING:
                break

        if tao.getConvergedReason() == PETSc.TAO.ConvergedReason.CONTINUE_ITERATING:
            tao.setConvergedReason(PETSc.TAO.ConvergedReason.DIVERGED_MAXITS)
        search_direction.destroy()


# Minimize f(x) = (x[0] - 1)^2 + (x[1] - 2)^2.
def objective(tao, x):
    return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2


def gradient(tao, x, g):
    g[0] = 2.0 * (x[0] - 1.0)
    g[1] = 2.0 * (x[1] - 2.0)
    g.assemble()


# Create the initial guess and gradient vector.
x = PETSc.Vec().createSeq(2, comm=PETSc.COMM_SELF)
x.set(0.5)
g = x.duplicate()

# Create the Python optimizer and configure the minimization.
tao = PETSc.TAO().createPython(myGradientDescent(), comm=PETSc.COMM_SELF)
tao.setObjective(objective)
tao.setGradient(gradient, g)
tao.setSolution(x)
tao.setTolerances(gatol=1e-6)
tao.setMaximumIterations(100)
tao.setFromOptions()
tao.solve()

PETSc.Sys.Print(f'x = ({x[0]:.6f}, {x[1]:.6f})')
