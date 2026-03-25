# --------------------------------------------------------------------

from petsc4py import PETSc
import unittest
import numpy


# --------------------------------------------------------------------

class BaseTestTAOTerm:
    COMM = None

    def setUp(self):
        self.taoterm = PETSc.TAOTerm().create(comm=self.COMM)

    def tearDown(self):
        self.taoterm = None
        PETSc.garbage_cleanup()

    def testL1(self):
        if self.taoterm.getComm().Get_size() > 1:
            return
        taoterm = self.taoterm
        taoterm.setType(PETSc.TAOTerm.Type.L1)
        taoterm.setFromOptions()

    def testL2(self):
        if self.taoterm.getComm().Get_size() > 1:
            return
        taoterm = self.taoterm
        taoterm.setType(PETSc.TAOTerm.Type.HALFL2SQUARED)
        taoterm.setFromOptions()

# --------------------------------------------------------------------


class TestTAOTermSelf(BaseTestTAOTerm, unittest.TestCase):
    COMM = PETSc.COMM_SELF


class TestTAOTermWorld(BaseTestTAOTerm, unittest.TestCase):
    COMM = PETSc.COMM_WORLD


# --------------------------------------------------------------------


if numpy.iscomplexobj(PETSc.ScalarType()):
    del BaseTestTAOTerm
    del TestTAOTermSelf
    del TestTAOTermWorld

if __name__ == '__main__':
    unittest.main()
