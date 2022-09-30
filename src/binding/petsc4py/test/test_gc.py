from petsc4py import PETSc
import unittest
import gc, weakref
import warnings

# --------------------------------------------------------------------

## gc.set_debug((gc.DEBUG_STATS |
##               gc.DEBUG_LEAK) &
##              ~gc.DEBUG_SAVEALL)

# --------------------------------------------------------------------

class BaseTestGC(object):

    def setUp(self):
        self.obj = self.CLASS().create(comm=PETSc.COMM_SELF)

    def tearDown(self):
        wref = self.make_weakref()
        self.assertTrue(wref() is self.obj)
        self.obj = None
        gc.collect()
        self.assertTrue(wref() is None)
        PETSc.garbage_cleanup()

    def make_weakref(self):
        wref = weakref.ref(self.obj)
        return wref

    def testCycleInSelf(self):
        self.obj.setAttr('myself', self.obj)

    def testCycleInMethod(self):
        self.obj.setAttr('mymeth', self.obj.view)

    def testCycleInInstance(self):
        class A: pass
        a = A()
        a.obj = self.obj
        self.obj.setAttr('myinst', a)

    def testCycleInAllWays(self):
        self.testCycleInSelf()
        self.testCycleInMethod()
        self.testCycleInInstance()

# --------------------------------------------------------------------

class TestGCVec(BaseTestGC, unittest.TestCase):
    CLASS = PETSc.Vec

class TestGCVecSubType(TestGCVec):
    CLASS = type('_Vec', (PETSc.Vec,), {})

class TestGCMat(BaseTestGC, unittest.TestCase):
    CLASS = PETSc.Mat

class TestGCMatSubType(TestGCMat):
    CLASS = type('_Mat', (PETSc.Mat,), {})

class TestGCPC(BaseTestGC, unittest.TestCase):
    CLASS = PETSc.PC

class TestGCPCSubType(TestGCPC):
    CLASS = type('_PC', (PETSc.PC,), {})

class TestGCKSP(BaseTestGC, unittest.TestCase):
    CLASS = PETSc.KSP

class TestGCKSPSubType(TestGCKSP):
    CLASS = type('_KSP', (PETSc.KSP,), {})

class TestGCSNES(BaseTestGC, unittest.TestCase):
    CLASS = PETSc.SNES
    def testCycleInAppCtx(self):
        self.obj.setAppCtx(self.obj)

class TestGCSNESSubType(TestGCSNES):
    CLASS = type('_SNES', (PETSc.SNES,), {})

class TestGCTS(BaseTestGC, unittest.TestCase):
    CLASS = PETSc.TS
    def testCycleInAppCtx(self):
        self.obj.setAppCtx(self.obj)

class TestGCTSSubType(TestGCTS):
    CLASS = type('_TS', (PETSc.TS,), {})
    def testCycleInAppCtx(self):
        self.obj.setAppCtx(self.obj)

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
