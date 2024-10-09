from petsc4py import PETSc
import unittest

class BaseTestPC:
    KSP_TYPE = None
    PC_TYPE = None
    def setUp(self):
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_SELF)
        pc = ksp.getPC()
        if self.KSP_TYPE:
            ksp.setType(self.KSP_TYPE)
        if self.PC_TYPE:
            pc.setType(self.PC_TYPE)
        self.ksp = ksp
        self.pc = pc

    def tearDown(self):
        self.ksp = None
        self.pc = None
        PETSc.garbage_cleanup()

class TestFIELDSPLITPC(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.FIELDSPLIT

    def testISoperations(self):
        test_index = [0,1,2]
        pc = self.pc
        is_u = PETSc.IS().createGeneral(test_index, comm=PETSc.COMM_SELF)
        pc.setFieldSplitIS(("u", is_u))

        self.assertTrue((pc.getFieldSplitSubIS("u").getIndices() == test_index).all())
        is_u = None


if __name__ == '__main__':
    unittest.main()
