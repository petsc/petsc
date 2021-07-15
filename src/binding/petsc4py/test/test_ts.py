import unittest
from petsc4py import PETSc
from sys import getrefcount

# --------------------------------------------------------------------

class MyODE:
    """
    du/dt + u**2 = 0;
    u0,u1,u2 = 1,2,3
    """
    def __init__(self):
        self.rhsfunction_calls = 0
        self.rhsjacobian_calls = 0
        self.ifunction_calls = 0
        self.ijacobian_calls = 0
        self.presolve_calls = 0
        self.update_calls = 0
        self.postsolve_calls = 0
        self.monitor_calls = 0

    def rhsfunction(self,ts,t,u,F):
        # print ('MyODE.rhsfunction()')
        self.rhsfunction_calls += 1
        f = -(u * u)
        f.copy(F)

    def rhsjacobian(self,ts,t,u,J,P):
        # print ('MyODE.rhsjacobian()')
        self.rhsjacobian_calls += 1
        P.zeroEntries()
        diag = -2 * u
        P.setDiagonal(diag)
        P.assemble()
        if J != P: J.assemble()
        return True # same_nz

    def ifunction(self,ts,t,u,du,F):
        # print ('MyODE.ifunction()')
        self.ifunction_calls += 1
        f = du + u * u
        f.copy(F)

    def ijacobian(self,ts,t,u,du,a,J,P):
        # print ('MyODE.ijacobian()')
        self.ijacobian_calls += 1
        P.zeroEntries()
        diag = a + 2 * u
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


class BaseTestTSNonlinear(object):

    TYPE = None

    def setUp(self):
        self.ts = PETSc.TS().create(PETSc.COMM_SELF)
        eft = PETSc.TS.ExactFinalTime.STEPOVER
        self.ts.setExactFinalTime(eft)
        ptype = PETSc.TS.ProblemType.NONLINEAR
        self.ts.setProblemType(ptype)
        self.ts.setType(self.TYPE)
        if PETSc.ScalarType().dtype.char in 'fF':
            snes = self.ts.getSNES()
            snes.setTolerances(rtol=1e-6)

    def tearDown(self):
        self.ts = None


class BaseTestTSNonlinearRHS(BaseTestTSNonlinear):

    def testSolveRHS(self):
        ts = self.ts
        dct = self.ts.getDict()
        self.assertTrue(dct is not None)
        self.assertTrue(type(dct) is dict)

        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(3);
        J.setFromOptions()
        J.setUp()
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
        ts.setMaxTime(T)
        ts.setMaxSteps(nT)
        ts.setFromOptions()
        u[0], u[1], u[2] = 1, 2, 3
        ts.solve(u)

        self.assertTrue(ode.rhsfunction_calls > 0)
        self.assertTrue(ode.rhsjacobian_calls > 0)

        dct = self.ts.getDict()
        self.assertTrue('__appctx__'      in dct)
        self.assertTrue('__rhsfunction__' in dct)
        self.assertTrue('__rhsjacobian__' in dct)
        self.assertTrue('__monitor__'     in dct)

        n = ode.monitor_calls
        ts.monitor(ts.step_number, ts.time)
        self.assertEqual(ode.monitor_calls, n+1)
        n = ode.monitor_calls
        ts.monitorCancel()
        ts.monitor(ts.step_number, ts.time)
        self.assertEqual(ode.monitor_calls, n)

    def testFDColorRHS(self):
        ts = self.ts
        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(5); J.setType('aij')
        J.setPreallocationNNZ(nnz=1)
        u, f = J.createVecs()

        ts.setAppCtx(ode)
        ts.setRHSFunction(ode.rhsfunction, f)
        ts.setRHSJacobian(ode.rhsjacobian, J, J)
        ts.setMonitor(ode.monitor)

        T0, dT, nT = 0.00, 0.1, 10
        T = T0 + nT*dT
        ts.setTime(T0)
        ts.setTimeStep(dT)
        ts.setMaxTime(T)
        ts.setMaxSteps(nT)
        ts.setFromOptions()
        u[0], u[1], u[2] = 1, 2, 3

        ts.setSolution(u)
        ode.rhsjacobian(ts,0,u,J,J)
        ts.setUp()
        ts.snes.setUseFD(True)
        ts.solve(u)

    def testResetAndSolveRHS(self):
        self.ts.reset()
        self.ts.setStepNumber(0)
        self.testSolveRHS()
        self.ts.reset()
        self.ts.setStepNumber(0)
        self.testSolveRHS()
        self.ts.reset()

class BaseTestTSNonlinearI(BaseTestTSNonlinear):

    def testSolveI(self):
        ts = self.ts
        dct = self.ts.getDict()
        self.assertTrue(dct is not None)
        self.assertTrue(type(dct) is dict)

        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(3);
        J.setFromOptions()
        J.setUp()
        u, f = J.createVecs()

        ts.setAppCtx(ode)
        ts.setIFunction(ode.ifunction, f)
        ts.setIJacobian(ode.ijacobian, J, J)
        ts.setMonitor(ode.monitor)

        ts.snes.ksp.pc.setType('none')

        T0, dT, nT = 0.00, 0.1, 10
        T = T0 + nT*dT
        ts.setTime(T0)
        ts.setTimeStep(dT)
        ts.setMaxTime(T)
        ts.setMaxSteps(nT)
        ts.setFromOptions()
        u[0], u[1], u[2] = 1, 2, 3
        ts.solve(u)

        self.assertTrue(ode.ifunction_calls > 0)
        self.assertTrue(ode.ijacobian_calls > 0)

        dct = self.ts.getDict()
        self.assertTrue('__appctx__'      in dct)
        self.assertTrue('__ifunction__' in dct)
        self.assertTrue('__ijacobian__' in dct)
        self.assertTrue('__monitor__'     in dct)

        n = ode.monitor_calls
        ts.monitor(ts.step_number, ts.time)
        self.assertEqual(ode.monitor_calls, n+1)
        n = ode.monitor_calls
        ts.monitorCancel()
        ts.monitor(ts.step_number, ts.time)
        self.assertEqual(ode.monitor_calls, n)

    def testFDColorI(self):
        ts = self.ts
        ode = MyODE()
        J = PETSc.Mat().create(ts.comm)
        J.setSizes(5); J.setType('aij')
        J.setPreallocationNNZ(nnz=1)
        J.setFromOptions()
        u, f = J.createVecs()

        ts.setAppCtx(ode)
        ts.setIFunction(ode.ifunction, f)
        ts.setIJacobian(ode.ijacobian, J, J)
        ts.setMonitor(ode.monitor)

        T0, dT, nT = 0.00, 0.1, 10
        T = T0 + nT*dT
        ts.setTime(T0)
        ts.setTimeStep(dT)
        ts.setMaxTime(T)
        ts.setMaxSteps(nT)
        ts.setFromOptions()
        u[0], u[1], u[2] = 1, 2, 3

        ts.setSolution(u)
        ode.ijacobian(ts,0,u,0*u,1,J,J)
        ts.setUp()
        ts.snes.setUseFD(True)
        ts.solve(u)

    def testResetAndSolveI(self):
        self.ts.reset()
        self.ts.setStepNumber(0)
        self.testSolveI()
        self.ts.reset()
        self.ts.setStepNumber(0)
        self.testSolveI()
        self.ts.reset()

class TestTSBeuler(BaseTestTSNonlinearRHS,BaseTestTSNonlinearI,
                   unittest.TestCase):
    TYPE = PETSc.TS.Type.BEULER

class TestTSCN(BaseTestTSNonlinearRHS,BaseTestTSNonlinearI,
               unittest.TestCase):
    TYPE = PETSc.TS.Type.CN

class TestTSTheta(BaseTestTSNonlinearRHS, BaseTestTSNonlinearI,
                  unittest.TestCase):
    TYPE = PETSc.TS.Type.THETA

class TestTSAlpha(BaseTestTSNonlinearRHS, BaseTestTSNonlinearI,
                  unittest.TestCase):
    TYPE = PETSc.TS.Type.ALPHA

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
