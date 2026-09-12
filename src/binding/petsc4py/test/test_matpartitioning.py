import copy
import unittest

from petsc4py import PETSc


class TestMatPartitioning(unittest.TestCase):
    def setUp(self):
        comm = PETSc.COMM_WORLD
        self.adj = PETSc.Mat().createAIJ(((3, None), (3, None)), nnz=0, comm=comm)
        self.adj.assemble()
        self.part = PETSc.MatPartitioning().create(comm=comm)
        self.part.setType(PETSc.MatPartitioning.Type.PARTITIONINGAVERAGE)
        self.part.setAdjacency(self.adj)
        self.expected = [comm.getRank()] * 3

    def tearDown(self):
        self.part.destroy()
        self.adj.destroy()

    def testApply(self):
        result = PETSc.IS()
        self.part.apply(result)
        self.assertEqual(result.getIndices().tolist(), self.expected)
        old = copy.copy(result)
        self.part.apply(result)
        self.assertEqual(old.getRefCount(), 1)
        self.assertEqual(result.getRefCount(), 1)
        self.assertEqual(result.getIndices().tolist(), self.expected)
        old.destroy()
        result.destroy()


if __name__ == '__main__':
    unittest.main()
