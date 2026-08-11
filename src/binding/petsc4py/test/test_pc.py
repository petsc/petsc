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

    def testAmatDef(self):
        pc_type = self.PC_TYPE
        for test_pc_type in ['mg']:
            pc = PETSc.PC().create()
            new_pc_type = 'none' if pc_type == test_pc_type else test_pc_type
            amat = []
            pc.setType(pc_type)
            amat.append(pc.getUseAmat())
            pc.setType(new_pc_type)
            amat.append(pc.getUseAmat())
            pc.setType(pc_type)
            amat.append(pc.getUseAmat())
            self.assertTrue(amat[0] == amat[2])
            self.assertFalse(amat[0] == amat[1])
            pc.destroy()

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


class TestMG(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.MG


class TestASMPC(BaseTestPC, unittest.TestCase):
    PC_TYPE = PETSc.PC.Type.ASM

    def checkLocalSubdomains(self, pc, indices, local_indices=None):
        # Set the subdomains on pc and check that they are returned unchanged.
        is_sub = [PETSc.IS().createGeneral(idx, comm=PETSc.COMM_SELF)
                  for idx in indices]
        is_local = None
        if local_indices is not None:
            is_local = [PETSc.IS().createGeneral(idx, comm=PETSc.COMM_SELF)
                        for idx in local_indices]
        pc.setASMLocalSubdomains(len(is_sub), is_sub, is_local)

        got_nsd, got_sub, got_local = pc.getASMLocalSubdomains()
        self.assertEqual(got_nsd, len(indices))
        self.assertEqual(len(got_sub), len(indices))
        for got, idx in zip(got_sub, indices):
            self.assertTrue((got.getIndices() == idx).all())

        if local_indices is None:
            self.assertEqual(len(got_local), 0)
        else:
            self.assertEqual(len(got_local), len(local_indices))
            for got, idx in zip(got_local, local_indices):
                self.assertTrue((got.getIndices() == idx).all())

        for iset in is_sub:
            iset.destroy()
        if is_local is not None:
            for iset in is_local:
                iset.destroy()

    def testLocalSubdomains(self):
        self.checkLocalSubdomains(self.pc, [[0, 1, 2], [3, 4, 5]])

    def testLocalSubdomainsWithLocalPart(self):
        self.checkLocalSubdomains(self.pc,
                                  [[0, 1, 2, 3], [2, 3, 4, 5]],
                                  [[0, 1], [4, 5]])

    def testLocalSubdomainsParallel(self):
        # The index sets are in the global numbering of the vector, so give
        # each process a distinct block of it.
        rank = PETSc.COMM_WORLD.getRank()
        pc = PETSc.PC().create(PETSc.COMM_WORLD)
        pc.setType(PETSc.PC.Type.ASM)
        self.checkLocalSubdomains(pc, [[3 * rank, 3 * rank + 1, 3 * rank + 2]])
        pc.destroy()

    def testLocalSubdomainsCountOnly(self):
        # A subdomain count without index sets: nsd is set, the lists are not.
        self.pc.setASMLocalSubdomains(3)
        got_nsd, got_sub, got_local = self.pc.getASMLocalSubdomains()
        self.assertEqual(got_nsd, 3)
        self.assertEqual(got_sub, [])
        self.assertEqual(got_local, [])

    def testLocalSubdomainsUnset(self):
        # Nothing has been set and the preconditioner has not been set up, so
        # PETSc holds no subdomains yet.
        got_nsd, got_sub, got_local = self.pc.getASMLocalSubdomains()
        self.assertEqual(got_nsd, PETSc.DECIDE)
        self.assertEqual(got_sub, [])
        self.assertEqual(got_local, [])


if __name__ == '__main__':
    unittest.main()
