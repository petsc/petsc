import unittest
from petsc4py import PETSc
from sys import getrefcount
import gc

# --------------------------------------------------------------------

class MyODE:
    """
    du/dt + u**2 = 0;
    u0 = 1
    """

    def __init__(self):
        self.function_calls = 0
        self.jacobian_calls = 0

    def function(self, ts,t,u,F):
        #print 'MyODE.function()'
        self.function_calls += 1
        dt = ts.getTimeStep()
        u0 = ts.getSolution()
        f = (u - u0)/dt + u * u
        f.copy(F)

    def jacobian(self,ts,t,u,J,P):
        #print 'MyODE.jacobian()'
        self.jacobian_calls += 1
        u0 = ts.getSolution()
        dt = ts.getTimeStep()
        P.zeroEntries()
        diag = 1/dt + 2 * u
        P.setDiagonal(diag)
        P.assemble()
        if J != P: J.assemble()
        return False # same_nz

class MyTS:
    def __init__(self):
        self.log = {}
    def _log(self, method, *args):
        self.log.setdefault(method, 0)
        self.log[method] += 1

    def create(self, *args):
        self._log('create', *args)

    def destroy(self,*args):
        self._log('destroy', *args)

    def setFromOptions(self, ts, *args):
        self._log('setFromOptions', *args)

    def setUp(self, ts, *args):
        self._log('setUp', ts, *args)

    def reset(self, ts, *args):
        self._log('reset', ts, *args)

    def computeRHSFunction(self, ts, *args):
        self._log('computeRHSFunction', ts, *args)
        return ts.computeRHSFunction(*args)

    def computeRHSJacobian(self, ts, *args):
        self._log('computeRHSJacobian', *args)
        return ts.computeRHSJacobian(*args)

    def preSolve(self, ts, *args):
        self._log('preSolve', ts, args)

    def postSolve(self, ts, *args):
        self._log('postSolve', ts, args)

    def preStep(self, ts, *args):
        self._log('preStep', ts, args)

    def postStep(self, ts, *args):
        self._log('postStep', ts, args)

    def startStep(self, ts, *args):
        self._log('startStep', ts, args)

    def verifyStep(self, ts, *args):
        self._log('verifyStep', ts, args)
        return (True, ts.getTimeStep())

    def monitor(self, ts, s, t, u):
        self._log('monitor', ts, s, t, u)
        dt = ts.time_step
        ut  = ts.vec_sol.norm()
        #prn = PETSc.Sys.Print
        #prn('TS: step %2d, T:%f, dT:%f, u:%f' % (s,t,dt,ut))


class TestTSPython(unittest.TestCase):

    def setUp(self):
        self.ts = PETSc.TS()
        self.ts.createPython(MyTS(), comm=PETSc.COMM_SELF)
        ctx = self.ts.getPythonContext()
        self.assertEqual(getrefcount(ctx),  3)
        self.assertEqual(ctx.log['create'], 1)

    def tearDown(self):
        ctx = self.ts.getPythonContext()
        self.assertEqual(getrefcount(ctx), 3)
        self.assertTrue('destroy' not in ctx.log)
        self.ts.destroy() # XXX
        self.ts = None
        self.assertEqual(ctx.log['destroy'], 1)
        self.assertEqual(getrefcount(ctx),   2)

    def testSolve(self, nsolve=1):
        ts = self.ts
        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(3);
        J.setFromOptions()
        u, f = J.createVecs()

        ts.setAppCtx(ode)
        ts.setRHSFunction(ode.function, f)
        ts.setRHSJacobian(ode.jacobian, J, J)
        ts.snes.ksp.pc.setType('none')

        T0, dT, nT = 0.00, 0.1, 10
        T = T0 + nT*dT
        ts.setTime(T0)
        ts.setTimeStep(dT)
        ts.setDuration(T, nT)
        ts.setFromOptions()
        u[0], u[1], u[2] = 1, 2, 3
        ts.solve(u)

        self.assertTrue(ode.function_calls > 0)
        self.assertTrue(ode.jacobian_calls > 0)

        ctx = self.ts.getPythonContext()
        self.assertEqual(getrefcount(ctx), 3)
        self.assertTrue(ctx.log['preSolve']  ==  nsolve)
        self.assertTrue(ctx.log['postSolve'] ==  nsolve)
        self.assertTrue(ctx.log['preStep']    >  1)
        self.assertTrue(ctx.log['postStep']   >  1)
        self.assertTrue(ctx.log['startStep']  >  1)
        self.assertTrue(ctx.log['verifyStep'] >  1)
        self.assertTrue(ctx.log['monitor']    >  1)
        del ctx

        dct = self.ts.getDict()
        self.assertTrue('__appctx__'   in dct)
        self.assertTrue('__rhsfunction__' in dct)
        self.assertTrue('__rhsjacobian__' in dct)

    def testFDColor(self):
        ts = self.ts
        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(5); J.setType('aij');
        J.setPreallocationNNZ(1)
        J.setFromOptions()
        u, f = J.createVecs()

        ts.setAppCtx(ode)
        ts.setRHSFunction(ode.function, f)
        ts.setRHSJacobian(ode.jacobian, J, J)

        T0, dT, nT = 0.00, 0.1, 10
        T = T0 + nT*dT
        ts.setTime(T0)
        ts.setTimeStep(dT)
        ts.setDuration(T, nT)
        ts.setFromOptions()
        u[0], u[1], u[2] = 1, 2, 3

        ts.setSolution(u)
        ode.jacobian(ts, 0,u,J,J)
        ts.snes.setUseFD(True)
        ts.solve(u)

    def testResetAndSolve(self):
        self.ts.reset()
        self.testSolve(nsolve=1)
        self.ts.reset()
        self.testFDColor()#self.testSolve(nsolve=2)
        self.ts.reset()
        self.testSolve(nsolve=3)
        self.ts.reset()

# --------------------------------------------------------------------

PETSC_VERSION = PETSc.Sys.getVersion()

i = PETSc.Sys.getVersionInfo()
if (PETSC_VERSION == (3, 1, 0) and
    not i['release']):
    PETSC_VERSION = (3, 2, 0)

if  PETSC_VERSION < (3, 2, 0):
    del TestTSPython.testResetAndSolve

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
