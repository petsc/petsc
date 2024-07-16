from petsc4py import PETSc
import unittest

class TestMatSchur(unittest.TestCase):
    def test(self):
        COMM = PETSc.COMM_WORLD
        TYPE = PETSc.Mat.Type.AIJ
        comm_size = COMM.getSize()
        A00 = PETSc.Mat().create(comm=COMM)
        A00.setType(TYPE)
        A00.setSizes([[2, 2*comm_size], [2, 2*comm_size]])
        A01 = PETSc.Mat().create(comm=COMM)
        A01.setType(TYPE)
        A01.setSizes([[2, 2*comm_size], [3, 3*comm_size]])
        A10 = PETSc.Mat().create(comm=COMM)
        A10.setType(TYPE)
        A10.setSizes([[3, 3*comm_size], [2, 2*comm_size]])
        A11 = PETSc.Mat().create(comm=COMM)
        A11.setType(TYPE)
        A11.setSizes([[3, 3*comm_size], [3, 3*comm_size]])
        S = PETSc.Mat().createSchurComplement(A00, A00, A01, A10, A11)
        M, N = S.getSize()
        self.assertEqual(M, 3*comm_size)
        self.assertEqual(N, 3*comm_size)
        m, n = S.getLocalSize()
        self.assertEqual(m, 3)
        self.assertEqual(n, 3)
        A00_dup, A00p_dup, A01_dup, A10_dup, A11_dup = S.getSchurComplementSubMatrices()
        self.assertEqual(A00_dup.id, A00.id)
        self.assertEqual(A00p_dup.id, A00.id)
        self.assertEqual(A01_dup.id, A01.id)
        self.assertEqual(A10_dup.id, A10.id)
        self.assertEqual(A11_dup.id, A11.id)

if __name__ == '__main__':
    unittest.main()
