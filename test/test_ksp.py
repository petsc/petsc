# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
from sys import getrefcount

# --------------------------------------------------------------------

class TestKSPBase(object):

    KSP_TYPE = None
    PC_TYPE  = None
    
    def setUp(self):
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_SELF)
        if self.KSP_TYPE:
            ksp.setType(self.KSP_TYPE)
        if self.PC_TYPE:
            pc = ksp.getPC()
            pc.setType(self.PC_TYPE)
        self.ksp = ksp
        
    def tearDown(self):
        self.ksp = None

    def testGetSetType(self):
        self.assertEqual(self.ksp.getType(), self.KSP_TYPE)
        self.ksp.setType(self.KSP_TYPE)
        self.assertEqual(self.ksp.getType(), self.KSP_TYPE)
        
    def testTols(self):
        tols = self.ksp.getTolerances()
        self.ksp.setTolerances(*tols)
        tnames = ('rtol', 'atol', 'divtol', 'max_it')
        tolvals = [getattr(self.ksp, t) for t in  tnames]
        self.assertEqual(tuple(tols), tuple(tolvals))
        
    def testGetSetPC(self):
        oldpc = self.ksp.getPC()
        self.assertEqual(oldpc.getRefCount(), 2)
        newpc = PETSc.PC()
        newpc.create(self.ksp.getComm())
        self.assertEqual(newpc.getRefCount(), 1)
        self.ksp.setPC(newpc)
        self.assertEqual(newpc.getRefCount(), 2)
        self.assertEqual(oldpc.getRefCount(), 1)
        oldpc.destroy()
        self.assertFalse(bool(oldpc))
        pc = self.ksp.getPC()
        self.assertTrue(bool(pc))
        self.assertEqual(pc, newpc)
        self.assertEqual(pc.getRefCount(), 3)
        newpc.destroy()
        self.assertFalse(bool(newpc))
        self.assertEqual(pc.getRefCount(), 2)
        
    def testSolve(self):
        A = PETSc.Mat().create(PETSc.COMM_SELF)
        A.setSizes([3,3]); A.setType(PETSc.Mat.Type.SEQAIJ)
        A.assemble()
        A.shift(10)
        x, b = A.getVecs()
        b.set(10)
        x.setRandom()
        self.ksp.setOperators(A)
        self.ksp.setConvergenceHistory()
        self.ksp.solve(b, x)
        rh = self.ksp.getConvergenceHistory()
        self.ksp.setConvergenceHistory(0)
        rh = self.ksp.getConvergenceHistory()
        self.assertEqual(len(rh), 0)
        del A, x, b

    def testSetMonitor(self):
        reshist = {}
        def monitor(ksp, its, rnorm):
            reshist[its] = rnorm
        refcnt = getrefcount(monitor)
        self.ksp.setMonitor(monitor)
        self.assertEqual(getrefcount(monitor), refcnt + 1)
        ## self.testSolve()
        reshist = {}
        self.ksp.cancelMonitor()
        self.assertEqual(getrefcount(monitor), refcnt)
        self.testSolve()
        self.assertEqual(len(reshist), 0)
        ## Monitor = PETSc.KSP.Monitor
        ## self.ksp.setMonitor(Monitor())
        ## self.ksp.setMonitor(Monitor.DEFAULT)
        ## self.ksp.setMonitor(Monitor.TRUE_RESIDUAL_NORM)
        ## self.ksp.setMonitor(Monitor.SOLUTION)
            
    def testSetConvergenceTest(self):
        def converged(ksp, its, rnorm):
            if its > 10: return True
            return False
        refcnt = getrefcount(converged)
        self.ksp.setConvergenceTest(converged)
        self.assertEqual(getrefcount(converged), refcnt + 1)
        self.ksp.setConvergenceTest(None)
        self.assertEqual(getrefcount(converged), refcnt)

# --------------------------------------------------------------------

class TestKSPPREONLY(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.PREONLY
    PC_TYPE = PETSc.PC.Type.LU

class TestKSPRICHARDSON(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.RICHARDSON

class TestKSPCHEBYCHEV(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CHEBYCHEV

class TestKSPCG(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CG

class TestKSPCGNE(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CGNE

class TestKSPSTCG(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.STCG

class TestKSPBCGS(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.BCGS

class TestKSPBCGSL(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.BCGSL

class TestKSPCGS(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.CGS

class TestKSPQCG(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.QCG
    PC_TYPE  = PETSc.PC.Type.JACOBI

class TestKSPBICG(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.BICG

class TestKSPGMRES(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.GMRES

class TestKSPFGMRES(TestKSPBase, unittest.TestCase):
    KSP_TYPE = PETSc.KSP.Type.FGMRES

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
