import unittest
from petsc4py import PETSc
from sys import getrefcount

# --------------------------------------------------------------------

class MyODE:
    """
    du/dt + u**2 = 0;
    u0 = 1
    """
    def __init__(self):
        self.rhsfunction_calls = 0
        self.rhsjacobian_calls = 0
        self.presolve_calls = 0
        self.update_calls = 0
        self.postsolve_calls = 0
        self.monitor_calls = 0
    def rhsfunction(self, ts,t,u,F):
        # print 'MyODE.function()'
        self.rhsfunction_calls += 1
        dt = ts.getTimeStep()
        u0 = ts.getSolution()
        f = (u - u0)/dt + u * u
        f.copy(F)
    def rhsjacobian(self,ts,t,u,J,P):
        # print 'MyODE.jacobian()'
        self.rhsjacobian_calls += 1
        u0 = ts.getSolution()
        dt = ts.getTimeStep()
        P.zeroEntries()
        diag = 1/dt + 2 * u
        P.setDiagonal(diag)
        P.assemble()
        if J != P: J.assemble()
        return True # same_nz
    def monitor(self, ts, s, t, u):
        self.monitor_calls += 1
        dt = ts.time_step
        ut  = ts.vec_sol.norm()
        #prn = PETSc.Sys.Print
        #prn('TS: step %2d, T:%f, dT:%f, u:%f' % (s,t,dt,ut))


class TestTSNonlinear(unittest.TestCase):

    def setUp(self):
        self.ts = PETSc.TS().create(PETSc.COMM_SELF)
        ptype = PETSc.TS.ProblemType.NONLINEAR
        self.ts.setProblemType(ptype)
        self.ts.setType('python')

    def tearDown(self):
        self.ts = None

    def testSolve(self):
        ts = self.ts
        dct = self.ts.getDict()
        self.assertTrue(dct is not None)
        self.assertTrue(type(dct) is dict)

        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(3);
        J.setFromOptions()
        u, f = J.createVecs()

        ts.setAppCtx(ode)
        ts.setRHSFunction(ode.rhsfunction, f)
        ts.setRHSJacobian(ode.rhsjacobian, J, J)
        ts.setMonitor(ode.monitor)

        ts.snes.ksp.pc.setType('none')

        T0, dT, nT = 0.00, 0.1, 10
        T = T0 + nT*dT
        ts.setTime(T0)
        ts.setTimeStep(dT)
        ts.setDuration(T, nT)
        ts.setFromOptions()
        u[0], u[1], u[2] = 1, 2, 3
        ts.solve(u)

        self.assertTrue(ode.rhsfunction_calls > 0)
        self.assertTrue(ode.rhsjacobian_calls > 0)

        dct = self.ts.getDict()
        self.assertTrue('__appctx__' in dct)
        self.assertTrue('__rhsfunction__' in dct)
        self.assertTrue('__rhsjacobian__' in dct)
        self.assertTrue('__monitor__' in dct)

        n = ode.monitor_calls
        ts.callMonitor(ts.step_number, ts.time)
        self.assertEqual(ode.monitor_calls, n+1)
        n = ode.monitor_calls
        ts.cancelMonitor()
        ts.callMonitor(ts.step_number, ts.time)
        self.assertEqual(ode.monitor_calls, n)

    def testFDColor(self):
        ts = self.ts
        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(5); J.setType('aij')
        J.setPreallocationNNZ(nnz=1)
        J.setFromOptions()
        u, f = J.createVecs()

        ts.setAppCtx(ode)
        ts.setRHSFunction(ode.rhsfunction, f)
        ts.setRHSJacobian(ode.rhsjacobian, J, J)
        ts.setMonitor(ode.monitor)

        T0, dT, nT = 0.00, 0.1, 10
        T = T0 + nT*dT
        ts.setTime(T0)
        ts.setTimeStep(dT)
        ts.setDuration(T, nT)
        ts.setFromOptions()
        u[0], u[1], u[2] = 1, 2, 3

        ts.setSolution(u)
        ode.rhsjacobian(ts,0,u,J,J)
        ts.setUseFD(True)
        ts.solve(u)

    def testResetAndSolve(self):
        self.ts.reset()
        self.testSolve()
        self.ts.reset()
        self.testSolve()
        self.ts.reset()

# --------------------------------------------------------------------

v = PETSc.Sys.getVersion()
i = PETSc.Sys.getVersionInfo()
petsc_dev = v <  (3, 1, 0) or (v == (3, 1, 0) and i['release'])
if petsc_dev:
    del TestTSNonlinear.testResetAndSolve

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
