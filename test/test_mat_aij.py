from petsc4py import PETSc
import unittest

import numpy as N

def mkgraph(comm, m, n):
    start = m*n * comm.rank
    end   = start + m*n
    idt = PETSc.IntType
    rows = []
    for I in range(start, end) :
        rows.append([])
        adj = rows[-1]
        i = I//n; j = I - i*n
        if i> 0  : J = I-n; adj.append(J)
        if j> 0  : J = I-1; adj.append(J)
        adj.append(I)
        if j< n-1: J = I+1; adj.append(J)
        if i< m-1: J = I+n; adj.append(J)
    nods = N.array(range(start, end), dtype=idt)
    xadj = N.array([0]*(len(rows)+1), dtype=idt)
    xadj[0] = 0
    xadj[1:] = N.cumsum([len(r) for r in rows], dtype=idt)
    if not rows: adjy = N.array([],dtype=idt)
    else:        adjy = N.concatenate(rows).astype(idt)
    return nods, xadj, adjy


class BaseTestMatAnyAIJ(object):

    COMM  = PETSc.COMM_NULL
    TYPE  = None
    GRID  = 0, 0
    BSIZE = None

    def setUp(self):
        COMM   = self.COMM
        GM, GN = self.GRID
        BS     = self.BSIZE or 1
        #
        sdt = dtype=PETSc.ScalarType
        self.rows, self.xadj, self.adjy = mkgraph(COMM, GM, GN)
        self.vals = N.array(range(1, 1 + len(self.adjy)* BS**2), dtype=sdt)
        self.vals.shape = (-1, BS, BS)
        #
        self.A = A = PETSc.Mat().create(comm=COMM)
        bs = BS; m, n = GM, GN; cs = COMM.getSize()
        rowsz = colsz = (m*n*bs, None)
        A.setSizes([rowsz, colsz], bs)
        A.setType(self.TYPE)

    def tearDown(self):
        self.A.destroy()
        self.A = None

    def testSetPreallocNNZ(self):
        nnz = [5, 2]
        self.A.setPreallocationNNZ(nnz, self.BSIZE)
        self._chk_bs(self.A, self.BSIZE)
        opt = PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
        opt = PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values_ijv()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)

    def testSetPreallocNNZ_2(self):
        _, ai, _, _ =self._get_aijv()
        d_nnz = N.diff(ai)
        nnz = [d_nnz, 3]
        self.A.setPreallocationNNZ(nnz, self.BSIZE)
        self._chk_bs(self.A, self.BSIZE)
        opt = PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
        opt = PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av =self._set_values_ijv()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)

    def testSetPreallocCSR(self):
        _, ai, aj, _ =self._get_aijv()
        csr = [ai, aj]
        self.A.setPreallocationCSR(csr, self.BSIZE)
        self._chk_bs(self.A, self.BSIZE)
        self._chk_aij(self.A, ai, aj)
        opt = PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR
        self.A.setOption(opt, True)
        self._set_values()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
        self._set_values_ijv()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)

    def testSetPreallocCSR_2(self):
        _, ai, aj, av =self._get_aijv()
        csr = [ai, aj, av]
        self.A.setPreallocationCSR(csr, self.BSIZE)
        self._chk_bs(self.A, self.BSIZE)
        self._chk_aij(self.A, ai, aj)
        opt = PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR
        self.A.setOption(opt, True)
        self._set_values()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
        self._set_values_ijv()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)

    def testSetValues(self):
        self._preallocate()
        opt = PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
        opt = PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)

    def testSetValuesIJV(self):
        self._preallocate()
        opt = PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values_ijv()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
        opt = PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values_ijv()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)

    def testGetValuesCSR(self):
        self._preallocate()
        self._set_values_ijv()
        A = self.A
        A.assemble()
        ai, aj, av = A.getValuesCSR()
        if not ('mpibaij' == self.A.type and
                self.A.comm.size == 1):
            B = PETSc.Mat()
            A.convert('aij', B)
            bi, bj, bv = B.getValuesCSR()
            self.assertTrue(N.allclose(ai, bi))
            self.assertTrue(N.allclose(aj, bj))
            self.assertTrue(N.allclose(av, bv))
            B.destroy()
        if 'crl' not in self.A.getType():
            C = A.duplicate()
            C.setValuesCSR(ai, aj, av)
            C.assemble()
            ci, cj, cv = C.getValuesCSR()
            self.assertTrue(N.allclose(ai, ci))
            self.assertTrue(N.allclose(aj, cj))
            self.assertTrue(N.allclose(av, cv))
            eq = A.equal(C)
            self.assertTrue(eq)
            C.destroy()

    def testGetDiagonalBlock(self):
        self._preallocate()
        self._set_values_ijv()
        self.A.assemble()
        B = self.A.getDiagonalBlock()
        self.assertEqual(self.A.getLocalSize(), B.getSize())
        B.destroy()

    def testGetSubMatrix(self):
        if 'baij' in self.A.getType(): return # XXX
        self._preallocate()
        self._set_values_ijv()
        self.A.assemble()
        #
        rank = self.A.getComm().getRank()
        rs, re = self.A.getOwnershipRange()
        cs, ce = self.A.getOwnershipRangeColumn()
        rows = N.array(range(rs, re), dtype=PETSc.IntType)
        cols = N.array(range(cs, ce), dtype=PETSc.IntType)
        rows = PETSc.IS().createGeneral(rows, comm=self.A.getComm())
        cols = PETSc.IS().createGeneral(cols, comm=self.A.getComm())
        #
        S = self.A.getSubMatrix(rows, None)
        S.zeroEntries()
        self.A.getSubMatrix(rows, None, S)
        S.destroy()
        #
        S = self.A.getSubMatrix(rows, cols)
        S.zeroEntries()
        self.A.getSubMatrix(rows, cols, S)
        S.destroy()

    def testCreateTranspose(self):
        self._preallocate()
        self._set_values_ijv()
        self.A.assemble()
        A = self.A
        AT = PETSc.Mat().createTranspose(A)
        x, y = A.createVecs()
        xt, yt = AT.createVecs()
        #
        y.setRandom()
        A.multTranspose(y, x)
        y.copy(xt)
        AT.mult(xt, yt)
        self.assertTrue(yt.equal(x))
        #
        x.setRandom()
        A.mult(x, y)
        x.copy(yt)
        AT.multTranspose(yt, xt)
        self.assertTrue(xt.equal(y))

    def _get_aijv(self):
        return (self.rows, self.xadj, self.adjy, self.vals,)

    def _preallocate(self):
        self.A.setPreallocationNNZ([5, 2], self.BSIZE)

    def _set_values(self):
        import sys
        if hasattr(sys, 'gettotalrefcount'):
            return self._set_values_ijv()
        # XXX Why the code below leak refs as a beast ???
        row, ai, aj, av =self._get_aijv()
        if not self.BSIZE:
            setvalues = self.A.setValues
        else:
            setvalues = self.A.setValuesBlocked
        for i, r in enumerate(row):
            s, e = ai[i], ai[i+1]
            setvalues(r, aj[s:e], av[s:e])
        return ai, aj, av

    def _set_values_ijv(self):
        row, ai, aj, av =self._get_aijv()
        if not self.BSIZE:
            setvalues = self.A.setValuesIJV
        else:
            setvalues = self.A.setValuesBlockedIJV
        setvalues(ai, aj, av, rowmap=row)
        setvalues(ai, aj, av, rowmap=None)
        return ai, aj, av

    def _chk_bs(self, A, bs):
        self.assertEqual(A.getBlockSize(), bs or 1)

    def _chk_aij(self, A, i, j):
        ai, aj = A.getRowIJ(compressed=bool(self.BSIZE))
        if None not in (ai, aj):
            self.assertTrue(N.all(i==ai))
            self.assertTrue(N.all(j==aj))
        ai, aj = A.getColumnIJ(compressed=bool(self.BSIZE))
        if None not in (ai, aj):
            self.assertTrue(N.all(i==ai))
            self.assertTrue(N.all(j==aj))

# -- AIJ ---------------------

class BaseTestMatAIJ(BaseTestMatAnyAIJ, unittest.TestCase):
    COMM  = PETSc.COMM_WORLD
    TYPE  = PETSc.Mat.Type.AIJ
    GRID  = 0, 0
    BSIZE = None

# -- Seq AIJ --

class TestMatSeqAIJ(BaseTestMatAIJ):
    COMM = PETSc.COMM_SELF
    TYPE = PETSc.Mat.Type.SEQAIJ
class TestMatSeqAIJ_G23(TestMatSeqAIJ):
    GRID  = 2, 3
class TestMatSeqAIJ_G45(TestMatSeqAIJ):
    GRID  = 4, 5
class TestMatSeqAIJ_G89(TestMatSeqAIJ):
    GRID  = 8, 9

# -- MPI AIJ --

class TestMatMPIAIJ(BaseTestMatAIJ):
    COMM = PETSc.COMM_WORLD
    TYPE = PETSc.Mat.Type.MPIAIJ
class TestMatMPIAIJ_G23(TestMatMPIAIJ):
    GRID  = 2, 3
class TestMatMPIAIJ_G45(TestMatMPIAIJ):
    GRID  = 4, 5
class TestMatMPIAIJ_G89(TestMatMPIAIJ):
    GRID  = 8, 9


# -- Block AIJ ---------------

class BaseTestMatBAIJ(BaseTestMatAnyAIJ, unittest.TestCase):
    COMM  = PETSc.COMM_WORLD
    TYPE  = PETSc.Mat.Type.BAIJ
    GRID  = 0, 0
    BSIZE = 1

# -- Seq Block AIJ --

class TestMatSeqBAIJ(BaseTestMatBAIJ):
    COMM = PETSc.COMM_SELF
    TYPE = PETSc.Mat.Type.SEQBAIJ
# bs = 1
class TestMatSeqBAIJ_G23(TestMatSeqBAIJ):
    GRID  = 2, 3
class TestMatSeqBAIJ_G45(TestMatSeqBAIJ):
    GRID  = 4, 5
class TestMatSeqBAIJ_G89(TestMatSeqBAIJ):
    GRID  = 8, 9
# bs = 2
class TestMatSeqBAIJ_G23_B2(TestMatSeqBAIJ_G23):
    BSIZE = 2
class TestMatSeqBAIJ_G45_B2(TestMatSeqBAIJ_G45):
    BSIZE = 2
class TestMatSeqBAIJ_G89_B2(TestMatSeqBAIJ_G89):
    BSIZE = 2
# bs = 3
class TestMatSeqBAIJ_G23_B3(TestMatSeqBAIJ_G23):
    BSIZE = 3
class TestMatSeqBAIJ_G45_B3(TestMatSeqBAIJ_G45):
    BSIZE = 3
class TestMatSeqBAIJ_G89_B3(TestMatSeqBAIJ_G89):
    BSIZE = 3
# bs = 4
class TestMatSeqBAIJ_G23_B4(TestMatSeqBAIJ_G23):
    BSIZE = 4
class TestMatSeqBAIJ_G45_B4(TestMatSeqBAIJ_G45):
    BSIZE = 4
class TestMatSeqBAIJ_G89_B4(TestMatSeqBAIJ_G89):
    BSIZE = 4
# bs = 5
class TestMatSeqBAIJ_G23_B5(TestMatSeqBAIJ_G23):
    BSIZE = 5
class TestMatSeqBAIJ_G45_B5(TestMatSeqBAIJ_G45):
    BSIZE = 5
class TestMatSeqBAIJ_G89_B5(TestMatSeqBAIJ_G89):
    BSIZE = 5


# -- MPI Block AIJ --

class TestMatMPIBAIJ(BaseTestMatBAIJ):
    COMM = PETSc.COMM_WORLD
    TYPE = PETSc.Mat.Type.MPIBAIJ
# bs = 1
class TestMatMPIBAIJ_G23(TestMatMPIBAIJ):
    GRID  = 2, 3
class TestMatMPIBAIJ_G45(TestMatMPIBAIJ):
    GRID  = 4, 5
class TestMatMPIBAIJ_G89(TestMatMPIBAIJ):
    GRID  = 8, 9
# bs = 2
class TestMatMPIBAIJ_G23_B2(TestMatMPIBAIJ_G23):
    BSIZE = 2
class TestMatMPIBAIJ_G45_B2(TestMatMPIBAIJ_G45):
    BSIZE = 2
class TestMatMPIBAIJ_G89_B2(TestMatMPIBAIJ_G89):
    BSIZE = 2
# bs = 3
class TestMatMPIBAIJ_G23_B3(TestMatMPIBAIJ_G23):
    BSIZE = 3
class TestMatMPIBAIJ_G45_B3(TestMatMPIBAIJ_G45):
    BSIZE = 3
class TestMatMPIBAIJ_G89_B3(TestMatMPIBAIJ_G89):
    BSIZE = 3
# bs = 4
class TestMatMPIBAIJ_G23_B4(TestMatMPIBAIJ_G23):
    BSIZE = 4
class TestMatMPIBAIJ_G45_B4(TestMatMPIBAIJ_G45):
    BSIZE = 4
class TestMatMPIBAIJ_G89_B4(TestMatMPIBAIJ_G89):
    BSIZE = 4
# bs = 5
class TestMatMPIBAIJ_G23_B5(TestMatMPIBAIJ_G23):
    BSIZE = 5
class TestMatMPIBAIJ_G45_B5(TestMatMPIBAIJ_G45):
    BSIZE = 5
class TestMatMPIBAIJ_G89_B5(TestMatMPIBAIJ_G89):
    BSIZE = 5


# -- AIJ + Block ---------------

class BaseTestMatAIJ_B(BaseTestMatAnyAIJ, unittest.TestCase):
    COMM  = PETSc.COMM_WORLD
    TYPE  = PETSc.Mat.Type.AIJ
    GRID  = 0, 0
    BSIZE = 1

    def testSetPreallocNNZ(self):pass
    def testSetPreallocNNZ_2(self):pass
    def testSetPreallocCSR(self):pass
    def testSetPreallocCSR_2(self):pass
    def testSetValues(self):
        self._preallocate()
        opt = PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
        opt = PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
    def testSetValuesIJV(self):
        self._preallocate()
        opt = PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values_ijv()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
        opt = PETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR
        self.A.setOption(opt, True)
        ai, aj, av = self._set_values_ijv()
        self.A.assemble()
        self._chk_aij(self.A, ai, aj)
    def _preallocate(self):
        self.A.setPreallocationNNZ([5*self.BSIZE, 3*self.BSIZE])
        self.A.setBlockSize(self.BSIZE)
        self._chk_bs(self.A, self.BSIZE)
    def _chk_aij(self, A, i, j):
        bs = self.BSIZE or 1
        ai, aj = A.getRowIJ()
        if None not in (ai, aj):  ## XXX map and check !!
            #self.assertTrue(N.all(i==ai))
            #self.assertTrue(N.all(j==aj))
            pass
        ai, aj = A.getColumnIJ(compressed=bool(self.BSIZE))
        if None not in (ai, aj): ## XXX map and check !!
            #self.assertTrue(N.all(i==ai))
            #self.assertTrue(N.all(j==aj))
            pass

# -- Seq AIJ + Block --

class TestMatSeqAIJ_B(BaseTestMatAIJ_B):
    COMM = PETSc.COMM_SELF
    TYPE = PETSc.Mat.Type.SEQAIJ
# bs = 1
class TestMatSeqAIJ_B_G23(TestMatSeqAIJ_B):
    GRID  = 2, 3
class TestMatSeqAIJ_B_G45(TestMatSeqAIJ_B):
    GRID  = 4, 5
class TestMatSeqAIJ_B_G89(TestMatSeqAIJ_B):
    GRID  = 8, 9
# bs = 2
class TestMatSeqAIJ_B_G23_B2(TestMatSeqAIJ_B_G23):
    BSIZE = 2
class TestMatSeqAIJ_B_G45_B2(TestMatSeqAIJ_B_G45):
    BSIZE = 2
class TestMatSeqAIJ_B_G89_B2(TestMatSeqAIJ_B_G89):
    BSIZE = 2
# bs = 3
class TestMatSeqAIJ_B_G23_B3(TestMatSeqAIJ_B_G23):
    BSIZE = 3
class TestMatSeqAIJ_B_G45_B3(TestMatSeqAIJ_B_G45):
    BSIZE = 3
class TestMatSeqAIJ_B_G89_B3(TestMatSeqAIJ_B_G89):
    BSIZE = 3
# bs = 4
class TestMatSeqAIJ_B_G23_B4(TestMatSeqAIJ_B_G23):
    BSIZE = 4
class TestMatSeqAIJ_B_G45_B4(TestMatSeqAIJ_B_G45):
    BSIZE = 4
class TestMatSeqAIJ_B_G89_B4(TestMatSeqAIJ_B_G89):
    BSIZE = 4
# bs = 5
class TestMatSeqAIJ_B_G23_B5(TestMatSeqAIJ_B_G23):
    BSIZE = 5
class TestMatSeqAIJ_B_G45_B5(TestMatSeqAIJ_B_G45):
    BSIZE = 5
class TestMatSeqAIJ_B_G89_B5(TestMatSeqAIJ_B_G89):
    BSIZE = 5


# -- MPI AIJ + Block --

class TestMatMPIAIJ_B(BaseTestMatAIJ_B):
    COMM = PETSc.COMM_WORLD
    TYPE = PETSc.Mat.Type.MPIAIJ
# bs = 1
class TestMatMPIAIJ_B_G23(TestMatMPIAIJ_B):
    GRID  = 2, 3
class TestMatMPIAIJ_B_G45(TestMatMPIAIJ_B):
    GRID  = 4, 5
class TestMatMPIAIJ_B_G89(TestMatMPIAIJ_B):
    GRID  = 8, 9
# bs = 2
class TestMatMPIAIJ_B_G23_B2(TestMatMPIAIJ_B_G23):
    BSIZE = 2
class TestMatMPIAIJ_B_G45_B2(TestMatMPIAIJ_B_G45):
    BSIZE = 2
class TestMatMPIAIJ_B_G89_B2(TestMatMPIAIJ_B_G89):
    BSIZE = 2
# bs = 3
class TestMatMPIAIJ_B_G23_B3(TestMatMPIAIJ_B_G23):
    BSIZE = 3
class TestMatMPIAIJ_B_G45_B3(TestMatMPIAIJ_B_G45):
    BSIZE = 3
class TestMatMPIAIJ_B_G89_B3(TestMatMPIAIJ_B_G89):
    BSIZE = 3
# bs = 4
class TestMatMPIAIJ_B_G23_B4(TestMatMPIAIJ_B_G23):
    BSIZE = 4
class TestMatMPIAIJ_B_G45_B4(TestMatMPIAIJ_B_G45):
    BSIZE = 4
class TestMatMPIAIJ_B_G89_B4(TestMatMPIAIJ_B_G89):
    BSIZE = 4
# bs = 5
class TestMatMPIAIJ_B_G23_B5(TestMatMPIAIJ_B_G23):
    BSIZE = 5
class TestMatMPIAIJ_B_G45_B5(TestMatMPIAIJ_B_G45):
    BSIZE = 5
class TestMatMPIAIJ_B_G89_B5(TestMatMPIAIJ_B_G89):
    BSIZE = 5

# -----

if PETSc.Sys.getVersion() >= (3,1,0):
    # -- AIJCRL ---------------------

    class BaseTestMatAIJCRL(BaseTestMatAIJ, unittest.TestCase):
        TYPE  = PETSc.Mat.Type.AIJCRL

    # -- Seq AIJCRL --

    class TestMatSeqAIJCRL(BaseTestMatAIJCRL):
        COMM = PETSc.COMM_SELF
        TYPE = PETSc.Mat.Type.SEQAIJCRL
    class TestMatSeqAIJCRL_G23(TestMatSeqAIJCRL):
        GRID  = 2, 3
    class TestMatSeqAIJCRL_G45(TestMatSeqAIJCRL):
        GRID  = 4, 5
    class TestMatSeqAIJCRL_G89(TestMatSeqAIJCRL):
        GRID  = 8, 9

    # -- MPI AIJCRL --

    class TestMatMPIAIJCRL(BaseTestMatAIJCRL):
        COMM = PETSc.COMM_WORLD
        TYPE = PETSc.Mat.Type.MPIAIJCRL
    class TestMatMPIAIJCRL_G23(TestMatMPIAIJCRL):
        GRID  = 2, 3
    class TestMatMPIAIJCRL_G45(TestMatMPIAIJCRL):
        GRID  = 4, 5
    class TestMatMPIAIJCRL_G89(TestMatMPIAIJCRL):
        GRID  = 8, 9

    # -- AIJCRL + Block -------------

    class BaseTestMatAIJCRL_B(BaseTestMatAIJ_B, unittest.TestCase):
        TYPE  = PETSc.Mat.Type.AIJ

    # -- Seq AIJCRL + Block --

    class TestMatSeqAIJCRL_B(BaseTestMatAIJCRL_B):
        COMM = PETSc.COMM_SELF
        TYPE = PETSc.Mat.Type.SEQAIJCRL
    # bs = 1
    class TestMatSeqAIJCRL_B_G23(TestMatSeqAIJCRL_B):
        GRID  = 2, 3
    class TestMatSeqAIJCRL_B_G45(TestMatSeqAIJCRL_B):
        GRID  = 4, 5
    class TestMatSeqAIJCRL_B_G89(TestMatSeqAIJCRL_B):
        GRID  = 8, 9
    # bs = 2
    class TestMatSeqAIJCRL_B_G23_B2(TestMatSeqAIJCRL_B_G23):
        BSIZE = 2
    class TestMatSeqAIJCRL_B_G45_B2(TestMatSeqAIJCRL_B_G45):
        BSIZE = 2
    class TestMatSeqAIJCRL_B_G89_B2(TestMatSeqAIJCRL_B_G89):
        BSIZE = 2
    # bs = 3
    class TestMatSeqAIJCRL_B_G23_B3(TestMatSeqAIJCRL_B_G23):
        BSIZE = 3
    class TestMatSeqAIJCRL_B_G45_B3(TestMatSeqAIJCRL_B_G45):
        BSIZE = 3
    class TestMatSeqAIJCRL_B_G89_B3(TestMatSeqAIJCRL_B_G89):
        BSIZE = 3
    # bs = 4
    class TestMatSeqAIJCRL_B_G23_B4(TestMatSeqAIJCRL_B_G23):
        BSIZE = 4
    class TestMatSeqAIJCRL_B_G45_B4(TestMatSeqAIJCRL_B_G45):
        BSIZE = 4
    class TestMatSeqAIJCRL_B_G89_B4(TestMatSeqAIJCRL_B_G89):
        BSIZE = 4
    # bs = 5
    class TestMatSeqAIJCRL_B_G23_B5(TestMatSeqAIJCRL_B_G23):
        BSIZE = 5
    class TestMatSeqAIJCRL_B_G45_B5(TestMatSeqAIJCRL_B_G45):
        BSIZE = 5
    class TestMatSeqAIJCRL_B_G89_B5(TestMatSeqAIJCRL_B_G89):
        BSIZE = 5


    # -- MPI AIJCRL + Block --

    class TestMatMPIAIJCRL_B(BaseTestMatAIJCRL_B):
        COMM = PETSc.COMM_WORLD
        TYPE = PETSc.Mat.Type.MPIAIJCRL
    # bs = 1
    class TestMatMPIAIJCRL_B_G23(TestMatMPIAIJCRL_B):
        GRID  = 2, 3
    class TestMatMPIAIJCRL_B_G45(TestMatMPIAIJCRL_B):
        GRID  = 4, 5
    class TestMatMPIAIJCRL_B_G89(TestMatMPIAIJCRL_B):
        GRID  = 8, 9
    # bs = 2
    class TestMatMPIAIJCRL_B_G23_B2(TestMatMPIAIJCRL_B_G23):
        BSIZE = 2
    class TestMatMPIAIJCRL_B_G45_B2(TestMatMPIAIJCRL_B_G45):
        BSIZE = 2
    class TestMatMPIAIJCRL_B_G89_B2(TestMatMPIAIJCRL_B_G89):
        BSIZE = 2
    # bs = 3
    class TestMatMPIAIJCRL_B_G23_B3(TestMatMPIAIJCRL_B_G23):
        BSIZE = 3
    class TestMatMPIAIJCRL_B_G45_B3(TestMatMPIAIJCRL_B_G45):
        BSIZE = 3
    class TestMatMPIAIJCRL_B_G89_B3(TestMatMPIAIJCRL_B_G89):
        BSIZE = 3
    # bs = 4
    class TestMatMPIAIJCRL_B_G23_B4(TestMatMPIAIJCRL_B_G23):
        BSIZE = 4
    class TestMatMPIAIJCRL_B_G45_B4(TestMatMPIAIJCRL_B_G45):
        BSIZE = 4
    class TestMatMPIAIJCRL_B_G89_B4(TestMatMPIAIJCRL_B_G89):
        BSIZE = 4
    # bs = 5
    class TestMatMPIAIJCRL_B_G23_B5(TestMatMPIAIJCRL_B_G23):
        BSIZE = 5
    class TestMatMPIAIJCRL_B_G45_B5(TestMatMPIAIJCRL_B_G45):
        BSIZE = 5
    class TestMatMPIAIJCRL_B_G89_B5(TestMatMPIAIJCRL_B_G89):
        BSIZE = 5

    # -----



if __name__ == '__main__':
    unittest.main()
