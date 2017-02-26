# Oregonator: stiff 3-variable oscillatory ODE system from chemical reactions,
# problem OREGO in Hairer&Wanner volume 2
# See also http://www.scholarpedia.org/article/Oregonator

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

class Orego(object):
    n = 3
    comm = PETSc.COMM_SELF
    def evalSolution(self, t, x):
        assert t == 0.0, "only for t=0.0"
        x.setArray([1, 2, 3])
    def evalFunction(self, ts, t, x, xdot, f):
        f.setArray([xdot[0] - 77.27*(x[1] + x[0]*(1 - 8.375e-6*x[0] - x[1])),
                    xdot[1] - 1/77.27*(x[2] - (1 + x[0])*x[1]),
                    xdot[2] - 0.161*(x[0] - x[2])])
    def evalJacobian(self, ts, t, x, xdot, a, A, B):
        B[:,:] = [[a - 77.27*((1 - 8.375e-6*x[0] - x[1]) - 8.375e-6*x[0]),   -77.27*(1 - x[0]),               0],
                  [1/77.27*x[1],                                             a + 1/77.27*(1 + x[0]),   -1/77.27],
                  [-0.161,                                                           0,               a + 0.161]]
        B.assemble()
        if A != B: A.assemble()
        return True # same nonzero pattern

OptDB = PETSc.Options()
ode = Orego()

J = PETSc.Mat().createDense([ode.n, ode.n], comm=ode.comm)
J.setUp()
x = PETSc.Vec().createSeq(ode.n, comm=ode.comm)
f = x.duplicate()

ts = PETSc.TS().create(comm=ode.comm)
ts.setType(ts.Type.ROSW)        # Rosenbrock-W. ARKIMEX is a nonlinearly implicit alternative.

ts.setIFunction(ode.evalFunction, f)
ts.setIJacobian(ode.evalJacobian, J)

history = []
def monitor(ts, i, t, x):
    xx = x[:].tolist()
    history.append((i, t, xx))
ts.setMonitor(monitor)

ts.setTime(0.0)
ts.setTimeStep(0.1)
ts.setMaxTime(360)
ts.setMaxSteps(2000)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.INTERPOLATE)
ts.setMaxSNESFailures(-1)       # allow an unlimited number of failures (step will be rejected and retried)

# Set a different tolerance on each variable. Can use a scalar or a vector for either or both atol and rtol.
vatol = x.duplicate(array=[1e-2, 1e-1, 1e-4])
ts.setTolerances(atol=vatol,rtol=1e-3) # adaptive controller attempts to match this tolerance

snes = ts.getSNES()             # Nonlinear solver
snes.setTolerances(max_it=10)   # Stop nonlinear solve after 10 iterations (TS will retry with shorter step)
ksp = snes.getKSP()             # Linear solver
ksp.setType(ksp.Type.PREONLY)   # Just use the preconditioner without a Krylov method
pc = ksp.getPC()                # Preconditioner
pc.setType(pc.Type.LU)          # Use a direct solve

ts.setFromOptions()             # Apply run-time options, e.g. -ts_adapt_monitor -ts_type arkimex -snes_converged_reason
ode.evalSolution(0.0, x)
ts.solve(x)
print('steps %d (%d rejected, %d SNES fails), nonlinear its %d, linear its %d'
      % (ts.getStepNumber(), ts.getStepRejections(), ts.getSNESFailures(),
         ts.getSNESIterations(), ts.getKSPIterations()))

if OptDB.getBool('plot_history', True):
    try:
        from matplotlib import pylab
        from matplotlib import rc
    except ImportError:
        print("matplotlib not available")
        raise SystemExit

    import numpy as np
    ii = np.asarray([v[0] for v in history])
    tt = np.asarray([v[1] for v in history])
    xx = np.asarray([v[2] for v in history])

    rc('text', usetex=True)
    pylab.suptitle('Oregonator: TS \\texttt{%s}' % ts.getType())
    pylab.subplot(2,2,1)
    pylab.subplots_adjust(wspace=0.3)
    pylab.semilogy(ii[:-1], np.diff(tt), )
    pylab.xlabel('step number')
    pylab.ylabel('timestep')

    for i in range(0,3):
        pylab.subplot(2,2,i+2)
        pylab.semilogy(tt, xx[:,i], "rgb"[i])
        pylab.xlabel('time')
        pylab.ylabel('$x_%d$' % i)

    # pylab.savefig('orego-history.png')
    pylab.show()
