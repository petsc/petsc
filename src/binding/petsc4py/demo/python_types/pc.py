# The user-defined Python class implementing the Jacobi method.
class myJacobi:

    # Setup the internal data. In this case, we access the matrix diagonal.
    def setUp(self, pc):
        _, P = pc.getOperators()
        self.D = P.getDiagonal()

    # Apply the preconditioner
    def apply(self, pc, x, y):
        y.pointwiseDivide(x, self.D)
