from petsc4py.PETSc import TAO
from petsc4py.PETSc import Vec
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by TAOPYTHON


class TAOPythonProtocol:
    def setFromOptions(self, tao: TAO):
        """Parse the command line options."""
        ...

    def setUp(self, tao: TAO):
        """Set up the optimizer."""
        ...

    def solve(self, tao):
        """Solve the optimisation problem.

        Note:
            Do not override if you want to rely on the default solve routine, using step, preStep and postStep.
        """
        ...

    def step(self, tao: TAO, x: Vec, g: Vec, s: Vec):
        """Given current iterate x compute gradient g and step s."""
        ...

    def preStep(self, tao: TAO):
        """Invoked before step."""
        ...

    def postStep(self, tao: TAO):
        """Invoked after step."""
        ...

    def view(self, tao: TAO, viewer: Viewer):
        """View the optimizer."""
        ...
