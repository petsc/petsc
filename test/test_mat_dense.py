from petsc4py import PETSc
import unittest

import numpy as np

def mkdata(comm, m, N, bs):
    start = m * comm.rank
    end   = start + m
    idt = PETSc.IntType
    sdt = PETSc.ScalarType
    rows = np.arange(start, end, dtype=idt)
    cols = np.arange(0, N, dtype=idt)
    vals = np.arange(0, m*N*bs*bs, dtype=sdt)
    return rows, cols, vals


class TestMatAnyDenseBase(object):

    COMM  = PETSc.COMM_NULL
    GRID  = 0, 0
    BSIZE = None
    TYPE  = PETSc.Mat.Type.DENSE
    
    def setUp(self):
        COMM   = self.COMM
        GM, GN = self.GRID
        BS     = self.BSIZE or 1
        #
        sdt = dtype=PETSc.ScalarType
        self.rows, self.cols, self.vals = mkdata(COMM, GM, GN, BS)
        self.vals.shape = (-1, BS, BS)
        #
        self.A = PETSc.Mat().create(comm=COMM)
        bs = BS; m, N = GM, GN;
        rowsz = (m*bs, None)
        colsz = (None, N*bs)
        self.A.setSizes([rowsz, colsz], bs)
        self.A.setType(self.TYPE)
        
    def tearDown(self):
        self.A = None

    def testSetValues(self):
        self._preallocate()
        r, c, v = self._set_values()
        self.A.assemble()
        self._chk_array(self.A, r, c, v)
        r, c, v = self._set_values()
        self.A.assemble()
        self._chk_array(self.A, r, c, v)

    def _preallocate(self):
        self.A.setPreallocationDense(None)

    def _set_values(self):
        rows, cols, vals = self.rows, self.cols, self.vals
        if not self.BSIZE:
            setvalues = self.A.setValues
        else:
            setvalues = self.A.setValuesBlocked
        setvalues(rows, cols, vals)
        return rows, cols, vals

    def _chk_bs(self, A, bs):
        self.assertEqual(A.getBlockSize(), bs or 1)

    def _chk_array(self, A, r, c, v):
        return # XXX
        vals = self.A.getValues(r, c)
        vals.shape = v.shape
        self.assertTrue(np.allclose(vals, v))


# -- Dense ---------------------

class TestMatDenseBase(TestMatAnyDenseBase, unittest.TestCase):
    COMM  = PETSc.COMM_WORLD
    GRID  = 0, 0
    BSIZE = None

# -- Seq Dense --

class TestMatSeqDense(TestMatDenseBase):
    COMM = PETSc.COMM_SELF
    TYPE = PETSc.Mat.Type.SEQDENSE
class TestMatSeqDense_G23(TestMatSeqDense):
    GRID  = 2, 3
class TestMatSeqDense_G45(TestMatSeqDense):
    GRID  = 4, 5
class TestMatSeqDense_G89(TestMatSeqDense):
    GRID  = 8, 9

# -- MPI Dense --

class TestMatMPIDense(TestMatDenseBase):
    COMM = PETSc.COMM_WORLD
    TYPE = PETSc.Mat.Type.MPIDENSE
class TestMatMPIDense_G23(TestMatMPIDense):
    GRID  = 2, 3
class TestMatMPIDense_G45(TestMatMPIDense):
    GRID  = 4, 5
class TestMatMPIDense_G89(TestMatMPIDense):
    GRID  = 8, 9


# -- Dense + Block ---------------

class TestMatDense_B_Base(TestMatAnyDenseBase, unittest.TestCase):
    COMM  = PETSc.COMM_WORLD
    GRID  = 0, 0
    BSIZE = 1

    def testSetValues(self):
        self._preallocate()
        r, c, v = self._set_values()
        self.A.assemble()
        self._chk_array(self.A, r, c, v)
        r, c, v = self._set_values()
        self.A.assemble()
        self._chk_array(self.A, r, c, v)
    def _preallocate(self):
        self.A.setPreallocationDense(None, self.BSIZE)
        self.A.setBlockSize(self.BSIZE)
        self._chk_bs(self.A, self.BSIZE)

# -- Seq Dense + Block --

class TestMatSeqDense_B(TestMatDense_B_Base):
    COMM = PETSc.COMM_SELF
    TYPE = PETSc.Mat.Type.SEQDENSE
# bs = 1
class TestMatSeqDense_B_G23(TestMatSeqDense_B):
    GRID  = 2, 3
class TestMatSeqDense_B_G45(TestMatSeqDense_B):
    GRID  = 4, 5
class TestMatSeqDense_B_G89(TestMatSeqDense_B):
    GRID  = 8, 9
# bs = 2
class TestMatSeqDense_B_G23_B2(TestMatSeqDense_B_G23):
    BSIZE = 2
class TestMatSeqDense_B_G45_B2(TestMatSeqDense_B_G45):
    BSIZE = 2
class TestMatSeqDense_B_G89_B2(TestMatSeqDense_B_G89):
    BSIZE = 2
# bs = 3
class TestMatSeqDense_B_G23_B3(TestMatSeqDense_B_G23):
    BSIZE = 3
class TestMatSeqDense_B_G45_B3(TestMatSeqDense_B_G45):
    BSIZE = 3
class TestMatSeqDense_B_G89_B3(TestMatSeqDense_B_G89):
    BSIZE = 3
# bs = 4
class TestMatSeqDense_B_G23_B4(TestMatSeqDense_B_G23):
    BSIZE = 4
class TestMatSeqDense_B_G45_B4(TestMatSeqDense_B_G45):
    BSIZE = 4
class TestMatSeqDense_B_G89_B4(TestMatSeqDense_B_G89):
    BSIZE = 4
# bs = 5
class TestMatSeqDense_B_G23_B5(TestMatSeqDense_B_G23):
    BSIZE = 5
class TestMatSeqDense_B_G45_B5(TestMatSeqDense_B_G45):
    BSIZE = 5
class TestMatSeqDense_B_G89_B5(TestMatSeqDense_B_G89):
    BSIZE = 5


# -- MPI Dense + Block --

class TestMatMPIDense_B(TestMatDense_B_Base):
    COMM = PETSc.COMM_WORLD
    TYPE = PETSc.Mat.Type.MPIDENSE
# bs = 1
class TestMatMPIDense_B_G23(TestMatMPIDense_B):
    GRID  = 2, 3
class TestMatMPIDense_B_G45(TestMatMPIDense_B):
    GRID  = 4, 5
class TestMatMPIDense_B_G89(TestMatMPIDense_B):
    GRID  = 8, 9
# bs = 2
class TestMatMPIDense_B_G23_B2(TestMatMPIDense_B_G23):
    BSIZE = 2
class TestMatMPIDense_B_G45_B2(TestMatMPIDense_B_G45):
    BSIZE = 2
class TestMatMPIDense_B_G89_B2(TestMatMPIDense_B_G89):
    BSIZE = 2
# bs = 3
class TestMatMPIDense_B_G23_B3(TestMatMPIDense_B_G23):
    BSIZE = 3
class TestMatMPIDense_B_G45_B3(TestMatMPIDense_B_G45):
    BSIZE = 3
class TestMatMPIDense_B_G89_B3(TestMatMPIDense_B_G89):
    BSIZE = 3
# bs = 4
class TestMatMPIDense_B_G23_B4(TestMatMPIDense_B_G23):
    BSIZE = 4
class TestMatMPIDense_B_G45_B4(TestMatMPIDense_B_G45):
    BSIZE = 4
class TestMatMPIDense_B_G89_B4(TestMatMPIDense_B_G89):
    BSIZE = 4
# bs = 5
class TestMatMPIDense_B_G23_B5(TestMatMPIDense_B_G23):
    BSIZE = 5
class TestMatMPIDense_B_G45_B5(TestMatMPIDense_B_G45):
    BSIZE = 5
class TestMatMPIDense_B_G89_B5(TestMatMPIDense_B_G89):
    BSIZE = 5

# -----




if __name__ == '__main__':
    unittest.main()
