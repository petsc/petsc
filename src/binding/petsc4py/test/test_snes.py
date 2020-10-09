# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
from sys import getrefcount

# --------------------------------------------------------------------

class Function:
    def __call__(self, snes, x, f):
        f[0] = (x[0]*x[0] + x[0]*x[1] - 3.0).item()
        f[1] = (x[0]*x[1] + x[1]*x[1] - 6.0).item()
        f.assemble()

class Jacobian:
    def __call__(self, snes, x, J, P):
        P[0,0] = (2.0*x[0] + x[1]).item()
        P[0,1] = (x[0]).item()
        P[1,0] = (x[1]).item()
        P[1,1] = (x[0] + 2.0*x[1]).item()
        P.assemble()
        if J != P: J.assemble()

# --------------------------------------------------------------------

class BaseTestSNES(object):

    SNES_TYPE = None

    def setUp(self):
        snes = PETSc.SNES()
        snes.create(PETSc.COMM_SELF)
        if self.SNES_TYPE:
            snes.setType(self.SNES_TYPE)
        self.snes = snes

    def tearDown(self):
        self.snes = None

    def testGetSetType(self):
        self.assertEqual(self.snes.getType(), self.SNES_TYPE)
        self.snes.setType(self.SNES_TYPE)
        self.assertEqual(self.snes.getType(), self.SNES_TYPE)

    def testTols(self):
        tols = self.snes.getTolerances()
        self.snes.setTolerances(*tols)
        tnames = ('rtol', 'atol','stol', 'max_it')
        tolvals = [getattr(self.snes, t) for t in  tnames]
        self.assertEqual(tuple(tols), tuple(tolvals))

    def testProperties(self):
        snes = self.snes
        #
        snes.appctx = (1,2,3)
        self.assertEqual(snes.appctx, (1,2,3))
        snes.appctx = None
        self.assertEqual(snes.appctx, None)
        #
        snes.its = 1
        self.assertEqual(snes.its, 1)
        snes.its = 0
        self.assertEqual(snes.its, 0)
        #
        snes.norm = 1
        self.assertEqual(snes.norm, 1)
        snes.norm = 0
        self.assertEqual(snes.norm, 0)
        #
        rh, ih = snes.history
        self.assertTrue(len(rh)==0)
        self.assertTrue(len(ih)==0)
        #
        reason = PETSc.SNES.ConvergedReason.CONVERGED_ITS
        snes.reason = reason
        self.assertEqual(snes.reason, reason)
        self.assertTrue(snes.converged)
        self.assertFalse(snes.diverged)
        self.assertFalse(snes.iterating)
        reason = PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT
        snes.reason = reason
        self.assertEqual(snes.reason, reason)
        self.assertFalse(snes.converged)
        self.assertTrue(snes.diverged)
        self.assertFalse(snes.iterating)
        reason = PETSc.SNES.ConvergedReason.CONVERGED_ITERATING
        snes.reason = reason
        self.assertEqual(snes.reason, reason)
        self.assertFalse(snes.converged)
        self.assertFalse(snes.diverged)
        self.assertTrue(snes.iterating)
        #
        self.assertFalse(snes.use_ew)
        self.assertFalse(snes.use_mf)
        self.assertFalse(snes.use_fd)

    def testGetSetFunc(self):
        r, func = self.snes.getFunction()
        self.assertFalse(r)
        self.assertTrue(func is None)
        r = PETSc.Vec().createSeq(2)
        func = Function()
        refcnt = getrefcount(func)
        self.snes.setFunction(func, r)
        self.snes.setFunction(func, r)
        self.assertEqual(getrefcount(func), refcnt + 1)
        r2, func2 = self.snes.getFunction()
        self.assertEqual(r, r2)
        self.assertEqual(func, func2[0])
        self.assertEqual(getrefcount(func), refcnt + 1)
        r3, func3 = self.snes.getFunction()
        self.assertEqual(r, r3)
        self.assertEqual(func, func3[0])
        self.assertEqual(getrefcount(func), refcnt + 1)

    def testCompFunc(self):
        r = PETSc.Vec().createSeq(2)
        func = Function()
        self.snes.setFunction(func, r)
        x, y = r.duplicate(), r.duplicate()
        x[0], x[1] = [1, 2]
        self.snes.computeFunction(x, y)
        self.assertAlmostEqual(abs(y[0]), 0.0)
        self.assertAlmostEqual(abs(y[1]), 0.0)

    def testGetSetJac(self):
        A, P, jac = self.snes.getJacobian()
        self.assertFalse(A)
        self.assertFalse(P)
        self.assertTrue(jac is None)
        J = PETSc.Mat().create(PETSc.COMM_SELF)
        J.setSizes([2,2])
        J.setType(PETSc.Mat.Type.SEQAIJ)
        J.setUp()
        jac = Jacobian()
        refcnt = getrefcount(jac)
        self.snes.setJacobian(jac, J)
        self.snes.setJacobian(jac, J)
        self.assertEqual(getrefcount(jac), refcnt + 1)
        J2, P2, jac2 = self.snes.getJacobian()
        self.assertEqual(J, J2)
        self.assertEqual(J2, P2)
        self.assertEqual(jac, jac2[0])
        self.assertEqual(getrefcount(jac), refcnt + 1)
        J3, P3, jac3 = self.snes.getJacobian()
        self.assertEqual(J, J3)
        self.assertEqual(J3, P3)
        self.assertEqual(jac, jac3[0])
        self.assertEqual(getrefcount(jac), refcnt + 1)

    def testCompJac(self):
        J = PETSc.Mat().create(PETSc.COMM_SELF)
        J.setSizes([2,2])
        J.setType(PETSc.Mat.Type.SEQAIJ)
        J.setUp()
        jac = Jacobian()
        self.snes.setJacobian(jac, J)
        x = PETSc.Vec().createSeq(2)
        x[0], x[1] = [1, 2]
        self.snes.getKSP().getPC()
        self.snes.computeJacobian(x, J)

    def testGetSetUpd(self):
        self.assertTrue(self.snes.getUpdate() is None)
        upd = lambda snes, it: None
        refcnt = getrefcount(upd)
        self.snes.setUpdate(upd)
        self.assertEqual(getrefcount(upd), refcnt + 1)
        self.snes.setUpdate(upd)
        self.assertEqual(getrefcount(upd), refcnt + 1)
        self.snes.setUpdate(None)
        self.assertTrue(self.snes.getUpdate() is None)
        self.assertEqual(getrefcount(upd), refcnt)
        self.snes.setUpdate(upd)
        self.assertEqual(getrefcount(upd), refcnt + 1)
        upd2 = lambda snes, it: None
        refcnt2 = getrefcount(upd2)
        self.snes.setUpdate(upd2)
        self.assertEqual(getrefcount(upd),  refcnt)
        self.assertEqual(getrefcount(upd2), refcnt2 + 1)
        tmp = self.snes.getUpdate()[0]
        self.assertTrue(tmp is upd2)
        self.assertEqual(getrefcount(upd2), refcnt2 + 2)
        del tmp
        self.snes.setUpdate(None)
        self.assertTrue(self.snes.getUpdate() is None)
        self.assertEqual(getrefcount(upd2), refcnt2)

    def testGetKSP(self):
        ksp = self.snes.getKSP()
        self.assertEqual(ksp.getRefCount(), 2)

    def testSolve(self):
        J = PETSc.Mat().create(PETSc.COMM_SELF)
        J.setSizes([2,2])
        J.setType(PETSc.Mat.Type.SEQAIJ)
        J.setUp()
        r = PETSc.Vec().createSeq(2)
        x = PETSc.Vec().createSeq(2)
        b = PETSc.Vec().createSeq(2)
        self.snes.setFunction(Function(), r)
        self.snes.setJacobian(Jacobian(), J)
        x.setArray([2,3])
        b.set(0)
        self.snes.setConvergenceHistory()
        self.snes.setFromOptions()
        self.snes.solve(b, x)
        rh, ih = self.snes.getConvergenceHistory()
        self.snes.setConvergenceHistory(0, reset=True)
        rh, ih = self.snes.getConvergenceHistory()
        self.assertEqual(len(rh), 0)
        self.assertEqual(len(ih), 0)
        self.assertAlmostEqual(abs(x[0]), 1.0)
        self.assertAlmostEqual(abs(x[1]), 2.0)
        # XXX this test should not be here !
        reason = self.snes.callConvergenceTest(1, 0, 0, 0)
        self.assertTrue(reason > 0)

    def testResetAndSolve(self):
        self.snes.reset()
        self.testSolve()
        self.snes.reset()
        self.testSolve()
        self.snes.reset()

    def testSetMonitor(self):
        reshist = {}
        def monitor(snes, its, fgnorm):
            reshist[its] = fgnorm
        refcnt = getrefcount(monitor)
        self.snes.setMonitor(monitor)
        self.assertEqual(getrefcount(monitor), refcnt + 1)
        self.testSolve()
        self.assertTrue(len(reshist) > 0)
        reshist = {}
        self.snes.cancelMonitor()
        self.assertEqual(getrefcount(monitor), refcnt)
        self.testSolve()
        self.assertTrue(len(reshist) == 0)
        self.snes.setMonitor(monitor)
        self.snes.monitor(1, 7)
        self.assertTrue(reshist[1] == 7)
        ## Monitor = PETSc.SNES.Monitor
        ## self.snes.setMonitor(Monitor())
        ## self.snes.setMonitor(Monitor.DEFAULT)
        ## self.snes.setMonitor(Monitor.SOLUTION)
        ## self.snes.setMonitor(Monitor.RESIDUAL)
        ## self.snes.setMonitor(Monitor.SOLUTION_UPDATE)

    def testSetGetStepFails(self):
        its = self.snes.getIterationNumber()
        self.assertEqual(its, 0)
        fails = self.snes.getNonlinearStepFailures()
        self.assertEqual(fails, 0)
        fails = self.snes.getMaxNonlinearStepFailures()
        self.assertEqual(fails, 1)
        self.snes.setMaxNonlinearStepFailures(5)
        fails = self.snes.getMaxNonlinearStepFailures()
        self.assertEqual(fails, 5)
        self.snes.setMaxNonlinearStepFailures(1)
        fails = self.snes.getMaxNonlinearStepFailures()
        self.assertEqual(fails, 1)

    def testSetGetLinFails(self):
        its = self.snes.getLinearSolveIterations()
        self.assertEqual(its, 0)
        fails = self.snes.getLinearSolveFailures()
        self.assertEqual(fails, 0)
        fails = self.snes.getMaxLinearSolveFailures()
        self.assertEqual(fails, 1)
        self.snes.setMaxLinearSolveFailures(5)
        fails = self.snes.getMaxLinearSolveFailures()
        self.assertEqual(fails, 5)
        self.snes.setMaxLinearSolveFailures(1)
        fails = self.snes.getMaxLinearSolveFailures()
        self.assertEqual(fails, 1)

    def testEW(self):
        self.snes.setUseEW(False)
        self.assertFalse(self.snes.getUseEW())
        self.snes.setUseEW(True)
        self.assertTrue(self.snes.getUseEW())
        params = self.snes.getParamsEW()
        params['version'] = 1
        self.snes.setParamsEW(**params)
        params = self.snes.getParamsEW()
        self.assertEqual(params['version'], 1)
        params['version'] = PETSc.DEFAULT
        self.snes.setParamsEW(**params)
        params = self.snes.getParamsEW()
        self.assertEqual(params['version'], 1)

    def testMF(self):
        #self.snes.setOptionsPrefix('MF-')
        #opts = PETSc.Options(self.snes)
        #opts['mat_mffd_type'] = 'ds'
        #opts['snes_monitor']  = 'stdout'
        #opts['ksp_monitor']   = 'stdout'
        #opts['snes_view']     = 'stdout'
        J = PETSc.Mat().create(PETSc.COMM_SELF)
        J.setSizes([2,2])
        J.setType(PETSc.Mat.Type.SEQAIJ)
        J.setUp()
        r = PETSc.Vec().createSeq(2)
        x = PETSc.Vec().createSeq(2)
        b = PETSc.Vec().createSeq(2)
        fun = Function()
        jac = Jacobian()
        self.snes.setFunction(fun, r)
        self.snes.setJacobian(jac, J)
        self.assertFalse(self.snes.getUseMF())
        self.snes.setUseMF(False)
        self.assertFalse(self.snes.getUseMF())
        self.snes.setUseMF(True)
        self.assertTrue(self.snes.getUseMF())
        self.snes.setFromOptions()
        x.setArray([2,3])
        b.set(0)
        self.snes.solve(b, x)
        self.assertAlmostEqual(abs(x[0]), 1.0)
        self.assertAlmostEqual(abs(x[1]), 2.0)

    def testFDColor(self):
        J = PETSc.Mat().create(PETSc.COMM_SELF)
        J.setSizes([2,2])
        J.setType(PETSc.Mat.Type.SEQAIJ)
        J.setUp()
        r = PETSc.Vec().createSeq(2)
        x = PETSc.Vec().createSeq(2)
        b = PETSc.Vec().createSeq(2)
        fun = Function()
        jac = Jacobian()
        self.snes.setFunction(fun, r)
        self.snes.setJacobian(jac, J)
        self.assertFalse(self.snes.getUseFD())
        jac(self.snes, x, J, J)
        self.snes.setUseFD(False)
        self.assertFalse(self.snes.getUseFD())
        self.snes.setUseFD(True)
        self.assertTrue(self.snes.getUseFD())
        self.snes.setFromOptions()
        x.setArray([2,3])
        b.set(0)
        self.snes.solve(b, x)
        self.assertAlmostEqual(abs(x[0]), 1.0)
        self.assertAlmostEqual(abs(x[1]), 2.0)

# --------------------------------------------------------------------

class TestSNESLS(BaseTestSNES, unittest.TestCase):
    SNES_TYPE = PETSc.SNES.Type.NEWTONLS

class TestSNESTR(BaseTestSNES, unittest.TestCase):
    SNES_TYPE = PETSc.SNES.Type.NEWTONTR

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
