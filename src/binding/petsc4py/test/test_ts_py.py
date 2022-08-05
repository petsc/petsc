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

    def function(self,ts,t,u,du,F):
        #print 'MyODE.function()'
        self.function_calls += 1
        f = du + u * u
        f.copy(F)

    def jacobian(self,ts,t,u,du,a,J,P):
        #print 'MyODE.jacobian()'
        self.jacobian_calls += 1
        P.zeroEntries()
        diag = a + 2 * u
        P.setDiagonal(diag)
        P.assemble()
        if J != P: J.assemble()
        return False # same_nz

class MyTS(object):

    def __init__(self):
        self.log = {}

    def _log(self, method, *args):
        self.log.setdefault(method, 0)
        self.log[method] += 1

    def create(self, ts, *args):
        self._log('create', *args)
        self.vec_update = PETSc.Vec()

    def destroy(self, ts, *args):
        self._log('destroy', *args)
        self.vec_update.destroy()

    def setFromOptions(self, ts, *args):
        self._log('setFromOptions', *args)

    def setUp(self, ts, *args):
        self._log('setUp', ts, *args)
        self.vec_update = ts.getSolution().duplicate()

    def reset(self, ts, *args):
        self._log('reset', ts, *args)

    def solveStep(self, ts, t, u, *args):
        self._log('solveStep', ts, t, u, *args)
        ts.snes.solve(None, u)

    def adaptStep(self, ts, t, u, *args):
        self._log('adaptStep', ts, t, u, *args)
        return (ts.getTimeStep(), True)


class TestTSPython(unittest.TestCase):

    def setUp(self):
        self.ts = PETSc.TS()
        self.ts.createPython(MyTS(), comm=PETSc.COMM_SELF)
        eft = PETSc.TS.ExactFinalTime.STEPOVER
        self.ts.setExactFinalTime(eft)
        ctx = self.ts.getPythonContext()
        self.assertEqual(getrefcount(ctx),  3)
        self.assertEqual(ctx.log['create'], 1)
        self.nsolve = 0

    def tearDown(self):
        ctx = self.ts.getPythonContext()
        self.assertEqual(getrefcount(ctx), 3)
        self.assertTrue('destroy' not in ctx.log)
        self.ts.destroy() # XXX
        self.ts = None
        self.assertEqual(ctx.log['destroy'], 1)
        self.assertEqual(getrefcount(ctx),   2)

    def testGetType(self):
        ctx = self.ts.getPythonContext()
        pytype = "{0}.{1}".format(ctx.__module__, type(ctx).__name__)
        self.assertTrue(self.ts.getPythonType() == pytype)

    def testSolve(self):
        ts = self.ts
        ts.setProblemType(ts.ProblemType.NONLINEAR)
        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(3);
        J.setFromOptions()
        J.setUp()
        u, f = J.createVecs()

        ts.setAppCtx(ode)
        ts.setIFunction(ode.function, f)
        ts.setIJacobian(ode.jacobian, J, J)
        ts.snes.ksp.pc.setType('none')
        
        T0, dT, nT = 0.0, 0.1, 10
        T = T0 + nT*dT
        ts.setTime(T0)
        ts.setTimeStep(dT)
        ts.setMaxTime(T)
        ts.setMaxSteps(nT)
        ts.setFromOptions()
        u[0], u[1], u[2] = 1, 2, 3
        ts.solve(u)
        self.nsolve +=1

        self.assertTrue(ode.function_calls > 0)
        self.assertTrue(ode.jacobian_calls > 0)
        
        ctx = self.ts.getPythonContext()
        ncalls = self.nsolve * ts.step_number
        self.assertTrue(ctx.log['solveStep'] == ncalls)
        self.assertTrue(ctx.log['adaptStep'] == ncalls)
        del ctx

        dct = self.ts.getDict()
        self.assertTrue('__appctx__'    in dct)
        self.assertTrue('__ifunction__' in dct)
        self.assertTrue('__ijacobian__' in dct)

    def testFDColor(self):
        #
        ts = self.ts
        ts.setProblemType(ts.ProblemType.NONLINEAR)
        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(5); J.setType('aij');
        J.setPreallocationNNZ(1)
        J.setFromOptions()
        u, f = J.createVecs()

        ts.setAppCtx(ode)
        ts.setIFunction(ode.function, f)
        ts.setIJacobian(ode.jacobian, J, J)

        T0, dT, nT = 0.00, 0.1, 10
        T = T0 + nT*dT
        ts.setTime(T0)
        ts.setTimeStep(dT)
        ts.setMaxTime(T)
        ts.setMaxSteps(nT)
        ts.setFromOptions()
        u[:] = 1, 2, 3, 4, 5

        ts.setSolution(u)
        ode.jacobian(ts,0.0,u,u,1.0,J,J)
        ts.snes.setUseFD(True)
        ts.solve(u)
        self.nsolve +=1

    def testResetAndSolve(self):
        self.ts.reset()
        self.ts.setStepNumber(0)
        self.testSolve()
        self.ts.reset()
        self.ts.setStepNumber(0)
        self.testFDColor()
        self.ts.reset()
        self.ts.setStepNumber(0)
        self.testSolve()
        self.ts.reset()

    def testSetAdaptLimits(self):
        self.ts.setStepLimits(1.0, 2.0)
        hmin, hmax = self.ts.getStepLimits()
        self.assertEqual(1.0, hmin)
        self.assertEqual(2.0, hmax)

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
