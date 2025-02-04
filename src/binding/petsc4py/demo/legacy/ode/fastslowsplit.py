# \begin{eqnarray}
#                  ys' = -2.0*\frac{-1.0+ys^2.0-\cos(t)}{2.0*ys}+0.05*\frac{-2.0+yf^2-\cos(5.0*t)}{2.0*yf}-\frac{\sin(t)}{2.0*ys}\\
#                  yf' = 0.05*\frac{-1.0+ys^2-\cos(t)}{2.0*ys}-\frac{-2.0+yf^2-\cos(5.0*t)}{2.0*yf}-5.0*\frac{\sin(5.0*t)}{2.0*yf}\\
# \end{eqnarray}

# This example demonstrates how to use ARKIMEX for solving a fast-slow system. The system is partitioned additively and component-wise at the same time.
# ys stands for the slow component and yf stands for the fast component. On the RHS for yf, only the term -\frac{-2.0+yf^2-\cos(5.0*t)}{2.0*yf} is treated implicitly while the rest is treated explicitly.

import sys
import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np


class FSS:
    n = 2
    comm = PETSc.COMM_SELF

    def __init__(self):
        pass

    def initialCondition(self, u):
        u[0] = np.sqrt(2.0)
        u[1] = np.sqrt(3.0)
        u.assemble()

    def evalRHSFunction(self, ts, t, u, f):
        f[0] = (
            -2.0 * (-1.0 + u[0] * u[0] - np.cos(t)) / (2.0 * u[0])
            + 0.05 * (-2.0 + u[1] * u[1] - np.cos(5.0 * t)) / (2.0 * u[1])
            - np.sin(t) / (2.0 * u[0])
        )
        f[1] = (
            0.05 * (-1.0 + u[0] * u[0] - np.cos(t)) / (2.0 * u[0])
            - (-2.0 + u[1] * u[1] - np.cos(5.0 * t)) / (2.0 * u[1])
            - 5.0 * np.sin(5.0 * t) / (2.0 * u[1])
        )
        f.assemble()

    def evalRHSFunctionSlow(self, ts, t, u, f):
        f[0] = (
            -2.0 * (-1.0 + u[0] * u[0] - np.cos(t)) / (2.0 * u[0])
            + 0.05 * (-2.0 + u[1] * u[1] - np.cos(5.0 * t)) / (2.0 * u[1])
            - np.sin(t) / (2.0 * u[0])
        )
        f.assemble()

    def evalRHSFunctionFast(self, ts, t, u, f):
        f[0] = 0.05 * (-1.0 + u[0] * u[0] - np.cos(t)) / (2.0 * u[0]) - 5.0 * np.sin(
            5.0 * t
        ) / (2.0 * u[1])
        f.assemble()

    def evalIFunctionFast(self, ts, t, u, udot, f):
        f[0] = udot[0] + (-2.0 + u[1] * u[1] - np.cos(5.0 * t)) / (2.0 * u[1])
        f.assemble()

    def evalIJacobianFast(self, ts, t, u, udot, a, A, B):
        A[0, 0] = a + (2.0 + np.cos(5.0 * t)) / (2.0 * u[1] * u[1]) + 0.5
        A.assemble()
        if A != B:
            B.assemble()


OptDB = PETSc.Options()
explicitform_ = OptDB.getBool('explicitform', False)

ode = FSS()

Jim = PETSc.Mat().createDense([1, 1], comm=ode.comm)
Jim.setUp()

u = PETSc.Vec().createSeq(ode.n, comm=ode.comm)

ts = PETSc.TS().create(comm=ode.comm)
ts.setProblemType(ts.ProblemType.NONLINEAR)

if not explicitform_:
    iss = PETSc.IS().createGeneral([0], comm=ode.comm)
    isf = PETSc.IS().createGeneral([1], comm=ode.comm)
    ts.setType(ts.Type.ARKIMEX)
    ts.setARKIMEXFastSlowSplit(True)
    ts.setRHSSplitIS('slow', iss)
    ts.setRHSSplitIS('fast', isf)
    ts.setRHSSplitRHSFunction('slow', ode.evalRHSFunctionSlow, None)
    ts.setRHSSplitRHSFunction('fast', ode.evalRHSFunctionFast, None)
    ts.setRHSSplitIFunction('fast', ode.evalIFunctionFast, None)
    ts.setRHSSplitIJacobian('fast', ode.evalIJacobianFast, Jim, Jim)
else:
    f = u.duplicate()
    ts.setType(ts.Type.RK)
    ts.setRHSFunction(ode.evalRHSFunction, f)

ts.setTimeStep(0.01)
ts.setMaxTime(0.3)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

ts.setFromOptions()
ode.initialCondition(u)
ts.solve(u)
u.view()

del ode, Jim, u, ts
