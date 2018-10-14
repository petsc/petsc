import sys, petsc4py
petsc4py.init(sys.argv)

import numpy as np
from petsc4py import PETSc

class Chwirut(object):

    """
    Finds the nonlinear least-squares solution to the model
    y = exp(-b1*x)/(b2+b3*x)  +  e
    """

    def __init__(self):
        BETA = [0.2, 0.12, 0.08]
        NOBSERVATIONS = 100
        NPARAMETERS = 3

        np.random.seed(456)
        x = np.random.rand(NOBSERVATIONS)
        e = np.random.rand(NOBSERVATIONS)

        y = np.exp(-BETA[0]*x)/(BETA[1] + BETA[2]*x) + e

        self.NOBSERVATIONS = NOBSERVATIONS
        self.NPARAMETERS = NPARAMETERS
        self.x = x
        self.y = y

    def createVecs(self):
        X = PETSc.Vec().create(PETSc.COMM_SELF)
        X.setSizes(self.NPARAMETERS)
        F = PETSc.Vec().create(PETSc.COMM_SELF)
        F.setSizes(self.NOBSERVATIONS)
        return X, F

    def formInitialGuess(self, X):
        X[0] = 0.15
        X[1] = 0.08
        X[2] = 0.05

    def formResidual(self, tao, X, F):
        x, y = self.x, self.y
        b1, b2, b3 = X.array
        F.array = y - np.exp(-b1*x)/(b2 + b3*x)

    def plotSolution(self, X):
        try:
            from matplotlib import pylab
        except ImportError:
            return
        b1, b2, b3 = X.array
        x, y = self.x, self.y
        u = np.linspace(x.min(), x.max(), 100)
        v = np.exp(-b1*u)/(b2+b3*u)
        pylab.plot(x, y, 'ro')
        pylab.plot(u, v, 'b-')
        pylab.show()

OptDB = PETSc.Options()

user = Chwirut()

x, f = user.createVecs()
x.setFromOptions()
f.setFromOptions()

tao = PETSc.TAO().create(PETSc.COMM_SELF)
tao.setType(PETSc.TAO.Type.POUNDERS)
tao.setResidual(user.formResidual, f)
tao.setFromOptions()

user.formInitialGuess(x)
tao.solve(x)

plot = OptDB.getBool('plot', False)
if plot: user.plotSolution(x)

x.destroy()
f.destroy()
tao.destroy()
