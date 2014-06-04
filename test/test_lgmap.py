from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class BaseTestLGMap(object):

    def _mk_idx(self, comm):
        comm_size = comm.getSize()
        comm_rank = comm.getRank()
        lsize = 10
        first = lsize * comm_rank
        last  = first + lsize
        if comm_rank > 0:
            first -= 1
        if comm_rank < (comm_size-1):
            last += 1
        return list(range(first, last))

    def tearDown(self):
        self.lgmap = None

    def testGetSize(self):
        size = self.lgmap.getSize()
        self.assertTrue(size >= 0)

    def testGetIndices(self):
        size = self.lgmap.getSize()
        idx  = self.lgmap.getIndices()
        self.assertEqual(len(idx), size)
        for i, val in enumerate(self.idx):
            self.assertEqual(idx[i], val)

    def testGetInfo(self):
        info = self.lgmap.getInfo()
        self.assertEqual(type(info), dict)
        if self.lgmap.getComm().getSize() == 1:
            self.assertEqual(info, {})
        else:
            self.assertTrue(len(info) > 1)
            self.assertTrue(len(info) < 4)

    def testApply(self):
        idxin  = list(range(self.lgmap.getSize()))
        idxout = self.lgmap.apply(idxin)
        self.lgmap.apply(idxin, idxout)
        invmap = self.lgmap.applyInverse(idxout)


    def testApplyIS(self):
        is_in  = PETSc.IS().createStride(self.lgmap.getSize())
        is_out = self.lgmap.apply(is_in)

    def testProperties(self):
        for prop in ('size', 'indices', 'info'):
            self.assertTrue(hasattr(self.lgmap, prop))

# --------------------------------------------------------------------

class TestLGMap(BaseTestLGMap, unittest.TestCase):

    def setUp(self):
        self.idx   = self._mk_idx(PETSc.COMM_WORLD)
        self.lgmap = PETSc.LGMap().create(self.idx, comm=PETSc.COMM_WORLD)

class TestLGMapIS(BaseTestLGMap, unittest.TestCase):

    def setUp(self):
        self.idx   = self._mk_idx(PETSc.COMM_WORLD)
        self.iset  = PETSc.IS().createGeneral(self.idx, comm=PETSc.COMM_WORLD)
        self.lgmap = PETSc.LGMap().create(self.iset)

    def tearDown(self):
        self.iset  = None
        self.lgmap = None

    def testSameComm(self):
        comm1 = self.lgmap.getComm()
        comm2 = self.iset.getComm()
        self.assertEqual(comm1, comm2)

# --------------------------------------------------------------------

class TestLGMapBlock(unittest.TestCase):

    BS = 3

    def setUp(self):
        comm = PETSc.COMM_WORLD
        comm_size = comm.getSize()
        comm_rank = comm.getRank()
        lsize = 10
        first = lsize * comm_rank
        last  = first + lsize
        if comm_rank > 0:
            first -= 1
        if comm_rank < (comm_size-1):
            last += 1
        self.idx = list(range(first, last))
        bs = self.BS
        self.lgmap = PETSc.LGMap().create(self.idx, bs, comm=PETSc.COMM_WORLD)

    def tearDown(self):
        self.lgmap = None

    def testGetSize(self):
        size = self.lgmap.getSize()
        self.assertTrue(size >= 0)

    def testGetBlockSize(self):
        bs = self.lgmap.getBlockSize()
        self.assertEqual(bs, self.BS)

    def testGetBlockIndices(self):
        size = self.lgmap.getSize()
        bs = self.lgmap.getBlockSize()
        idx = self.lgmap.getBlockIndices()
        self.assertEqual(len(idx), size//bs)
        for i, val in enumerate(self.idx):
            self.assertEqual(idx[i], val)

    def testGetIndices(self):
        size = self.lgmap.getSize()
        bs = self.lgmap.getBlockSize()
        idx = self.lgmap.getIndices()
        self.assertEqual(len(idx), size)
        for i, val in enumerate(self.idx):
            for j in range(bs):
                self.assertEqual(idx[i*bs+j], val*bs+j)

    def testGetBlockInfo(self):
        info = self.lgmap.getBlockInfo()
        self.assertEqual(type(info), dict)
        if self.lgmap.getComm().getSize() == 1:
            self.assertEqual(info, {})
        else:
            self.assertTrue(len(info) > 1)
            self.assertTrue(len(info) < 4)

    def testGetInfo(self):
        info = self.lgmap.getInfo()
        self.assertEqual(type(info), dict)
        if self.lgmap.getComm().getSize() == 1:
            self.assertEqual(info, {})
        else:
            self.assertTrue(len(info) > 1)
            self.assertTrue(len(info) < 4)

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
