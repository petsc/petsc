# Stiff scalar valued ODE problem with an exact solution

import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
from math import sin, cos, exp


class CE:
    n = 1
    comm = PETSc.COMM_SELF

    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_

    def evalSolution(self, t, x):
        lam = self.lambda_
        x[0] = lam / (lam * lam + 1) * (lam * cos(t) + sin(t)) - lam * lam / (
            lam * lam + 1
        ) * exp(-lam * t)
        x.assemble()

    def evalFunction(self, ts, t, x, xdot, f):
        lam = self.lambda_
        f[0] = xdot[0] + lam * (x[0] - cos(t))
        f.assemble()

    def evalJacobian(self, ts, t, x, xdot, a, A, B):
        J = B
        lam = self.lambda_
        J[0, 0] = a + lam
        J.assemble()
        if A != B:
            A.assemble()


OptDB = PETSc.Options()

lambda_ = OptDB.getScalar('lambda', 10.0)
ode = CE(lambda_)

J = PETSc.Mat().createDense([ode.n, ode.n], comm=ode.comm)
J.setUp()
x = PETSc.Vec().createSeq(ode.n, comm=ode.comm)
f = x.duplicate()

ts = PETSc.TS().create(comm=ode.comm)
ts.setProblemType(ts.ProblemType.NONLINEAR)
ts.setType(ts.Type.GLLE)

ts.setIFunction(ode.evalFunction, f)
ts.setIJacobian(ode.evalJacobian, J)

ts.setTime(0.0)
ts.setTimeStep(0.001)
ts.setMaxTime(10)
ts.setMaxSteps(10000)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)


class Monitor:
    def __init__(self, ode):
        self.ode = ode
        self.x = PETSc.Vec().createSeq(ode.n, comm=ode.comm)

    def __call__(self, ts, k, t, x):
        self.ode.evalSolution(t, self.x)
        self.x.axpy(-1, x)
        e = self.x.norm()
        h = ts.getTimeStep()
        PETSc.Sys.Print(
            'step %3d t=%8.2e h=%8.2e error=%8.2e' % (k, t, h, e), comm=self.ode.comm
        )


ts.setMonitor(Monitor(ode))

ts.setFromOptions()
ode.evalSolution(0.0, x)
ts.solve(x)

del ode, J, x, f, ts
