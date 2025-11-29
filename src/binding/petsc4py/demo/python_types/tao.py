# The user-defined Python class implementing the gradient descent.
from petsc4py import PETSc

class myGradientDescent:
    def create(self, tao):
        # Create a line search type with constant step size.
        self._ls = PETSc.TAOLineSearch().create(comm=PETSc.COMM_SELF)
        self._ls.useTAORoutine(tao)
        self._ls.setType(PETSc.TAOLineSearch.Type.UNIT)
        self._ls.setInitialStepLength(0.2)

    def solve(self, tao):
        # Get solution and Jacobian vector.
        x = tao.getSolution()
        gradient = tao.getGradient()[0]

        # Prepare search direction for line search.
        search_direction = gradient.copy()

        # Optimization loop.
        for it in range(tao.getMaximumIterations()):
            tao.setIterationNumber(it)

            # Compute search_direction.
            #   search_direction = -gradient
            tao.computeGradient(x, gradient)
            gradient.copy(search_direction)
            search_direction.scale(-1)

            # Apply line search:
            #   x += .2 search_direction
            f, s, reason = self._ls.apply(x, gradient, search_direction)

            if reason < 0:
                raise RuntimeError('LS failed.')

            # Log and update internal state.
            tao.monitor(f=f, res=gradient.norm())

            # Convergence check.
            if tao.checkConverged() > 0:
                break
