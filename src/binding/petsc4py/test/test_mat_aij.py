from petsc4py import PETSc
import unittest

import numpy as N
import numpy as np

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
        BS     = self.BSIZE
        #
        try:
            rbs, cbs = BS
            rbs = rbs or 1
            cbs = cbs or 1
        except (TypeError, ValueError):
            rbs = cbs = BS or 1
        sdt = dtype = PETSc.ScalarType
        self.rows, self.xadj, self.adjy = mkgraph(COMM, GM, GN)
        self.vals = N.array(range(1, 1 + len(self.adjy)*rbs*cbs), dtype=sdt)
        self.vals.shape = (-1, rbs, cbs)
        #
        m, n = GM, GN
        rowsz = (m*n*rbs, None)
        colsz = (m*n*cbs, None)
        A = self.A = PETSc.Mat().create(comm=COMM)
        A.setType(self.TYPE)
        A.setSizes([rowsz, colsz], BS)

    def tearDown(self):
        self.A.destroy()
        self.A = None

    def testSetPreallocNNZ(self):
        nnz = [5, 2]
        self.A.setPreallocationNNZ(nnz)
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
        self.A.setPreallocationNNZ(nnz)
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
        if 'is' in self.A.getType(): return # XXX
        _, ai, aj, _ = self._get_aijv()
        csr = [ai, aj]
        self.A.setPreallocationCSR(csr)
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
        if 'is' in self.A.getType(): return # XXX
        _, ai, aj, av =self._get_aijv()
        csr = [ai, aj, av]
        self.A.setPreallocationCSR(csr)
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
        if 'is' in self.A.getType(): return # XXX
        self._preallocate()
        self._set_values_ijv()
        A = self.A
        A.assemble()
        if 'sbaij' in A.getType():
            opt = PETSc.Mat.Option.GETROW_UPPERTRIANGULAR
            self.A.setOption(opt, True)
        ai, aj, av = A.getValuesCSR()
        rstart, rend = A.getOwnershipRange()
        for row in range(rstart, rend):
            cols, vals = A.getRow(row)
            i = row - rstart
            self.assertTrue(N.allclose(aj[ai[i]:ai[i+1]], cols))
            self.assertTrue(N.allclose(av[ai[i]:ai[i+1]], vals))

    def testConvertToSAME(self):
        self._preallocate()
        self._set_values_ijv()
        A = self.A
        A.assemble()
        A.convert('same')

    def testConvertToDENSE(self):
        self._preallocate()
        self._set_values_ijv()
        A = self.A
        A.assemble()
        x, y = A.getVecs()
        x.setRandom()
        z = y.duplicate()
        A.mult(x, y)
        if A.type.endswith('sbaij'): return
        B = PETSc.Mat()
        A.convert('dense', B)  # initial
        B.mult(x, z)
        self.assertTrue(np.allclose(y.array, z.array))
        A.convert('dense', B)  # reuse
        B.mult(x, z)
        self.assertTrue(np.allclose(y.array, z.array))
        A.convert('dense')     # inplace
        A.mult(x, z)
        self.assertTrue(np.allclose(y.array, z.array))

    def testConvertToAIJ(self):
        self._preallocate()
        self._set_values_ijv()
        A = self.A
        A.assemble()
        x, y = A.getVecs()
        x.setRandom()
        z = y.duplicate()
        A.mult(x, y)
        if A.type.endswith('sbaij'): return
        B = PETSc.Mat()
        A.convert('aij', B)  # initial
        B.mult(x, z)
        self.assertTrue(np.allclose(y.array, z.array))
        A.convert('aij', B)  # reuse
        B.mult(x, z)
        self.assertTrue(np.allclose(y.array, z.array))
        A.convert('aij')     # inplace
        A.mult(x, z)
        self.assertTrue(np.allclose(y.array, z.array))

    def testGetDiagonalBlock(self):
        if 'is' in self.A.getType(): return # XXX
        self._preallocate()
        self._set_values_ijv()
        self.A.assemble()
        B = self.A.getDiagonalBlock()
        self.assertEqual(self.A.getLocalSize(), B.getSize())
        B.destroy()

    def testInvertBlockDiagonal(self):
        if 'is' in self.A.getType(): return # XXX
        try:
            _ = len(self.BSIZE)
            return
        except (TypeError, ValueError):
            pass
        self._preallocate()
        rbs, cbs = self.A.getBlockSizes()
        if rbs != cbs: return
        self._set_values_ijv()
        self.A.assemble()
        self.A.shift(1000) # Make nonsingular
        ibdiag = self.A.invertBlockDiagonal()
        bs = self.A.getBlockSize()
        m, _ = self.A.getLocalSize()
        self.assertEqual(ibdiag.shape, (m//bs, bs, bs))
        tmp = N.empty((m//bs, bs, bs), dtype=PETSc.ScalarType)
        rstart, rend = self.A.getOwnershipRange()
        s, e = rstart//bs, rend//bs
        for i in range(s, e):
            rows = cols = N.arange(i*bs,(i+1)*bs, dtype=PETSc.IntType)
            vals = self.A.getValues(rows,cols)
            tmp[i-s,:,:] = N.linalg.inv(vals)
        self.assertTrue(N.allclose(ibdiag, tmp))

    def testCreateSubMatrix(self):
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
        S = self.A.createSubMatrix(rows, None)
        S.zeroEntries()
        self.A.createSubMatrix(rows, None, S)
        S.destroy()
        #
        S = self.A.createSubMatrix(rows, cols)
        S.zeroEntries()
        self.A.createSubMatrix(rows, cols, S)
        S.destroy()

    def testCreateSubMatrices(self):
        if 'baij' in self.A.getType(): return # XXX
        if 'is' in self.A.getType(): return # XXX
        self._preallocate()
        self._set_values_ijv()
        self.A.assemble()
        #
        rs, re = self.A.getOwnershipRange()
        cs, ce = self.A.getOwnershipRangeColumn()
        rows = N.array(range(rs, re), dtype=PETSc.IntType)
        cols = N.array(range(cs, ce), dtype=PETSc.IntType)
        rows = PETSc.IS().createGeneral(rows, comm=self.A.getComm())
        cols = PETSc.IS().createGeneral(cols, comm=self.A.getComm())
        #
        (S,) = self.A.createSubMatrices(rows, cols)
        S.zeroEntries()
        self.A.createSubMatrices(rows, cols, submats=[S])
        S.destroy()
        #
        (S1,) = self.A.createSubMatrices([rows], [cols])
        (S2,) = self.A.createSubMatrices([rows], [cols])
        self.assertTrue(S1.equal(S2))
        S2.zeroEntries()
        self.A.createSubMatrices([rows], [cols], [S2])
        self.assertTrue(S1.equal(S2))
        S1.destroy()
        S2.destroy()
        #
        if 'seq' not in self.A.getType(): return # XXX
        S1, S2 = self.A.createSubMatrices([rows, rows], [cols, cols])
        self.assertTrue(S1.equal(S2))
        S1.zeroEntries()
        S2.zeroEntries()
        self.A.createSubMatrices([rows, rows], [cols, cols], [S1, S2])
        self.assertTrue(S1.equal(S2))
        S1.destroy()
        S2.destroy()

    def testGetRedundantMatrix(self):
        if 'aijcrl' in self.A.getType(): return # duplicate not supported
        if 'mpisbaij' in self.A.getType(): return # not working
        if 'is' in self.A.getType(): return # XXX
        self._preallocate()
        self._set_values_ijv()
        self.A.assemble()
        #Test the most simple case
        sizecommA = self.A.getComm().getSize()
        Ared = self.A.getRedundantMatrix(sizecommA)
        sizecommAred = Ared.getComm().getSize()
        self.assertEqual(1, sizecommAred)
        Ared.destroy()

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
        self.A.setPreallocationNNZ([5, 2])

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

    def _chk_bsizes(self, A, bsizes):
        try:
            rbs, cbs = bsizes
        except (TypeError, ValueError):
            rbs = cbs = bsizes
        self.assertEqual(A.getBlockSizes(), (rbs, cbs))

    def _chk_aij(self, A, i, j):
        compressed = bool(self.BSIZE)
        ai, aj = A.getRowIJ(compressed=compressed)
        if ai is not None and aj is not None:
            self.assertTrue(N.all(i==ai))
            self.assertTrue(N.all(j==aj))
        ai, aj = A.getColumnIJ(compressed=compressed)
        if ai is not None and aj is not None:
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

# -- SymmBlock AIJ ---------------

class BaseTestMatSBAIJ(BaseTestMatAnyAIJ, unittest.TestCase):
    COMM  = PETSc.COMM_WORLD
    TYPE  = PETSc.Mat.Type.SBAIJ
    GRID  = 0, 0
    BSIZE = 1
    def testInvertBlockDiagonal(self): pass
    def _chk_aij(self, A, i, j):
        ai, aj = A.getRowIJ(compressed=True)
        if ai is not None and aj is not None:
            if 0: # XXX Implement
                self.assertTrue(N.all(i==ai))
                self.assertTrue(N.all(j==aj))
        ai, aj = A.getColumnIJ(compressed=True)
        if ai is not None and aj is not None:
            if 0: # XXX Implement
                self.assertTrue(N.all(i==ai))
                self.assertTrue(N.all(j==aj))

# -- Seq SymmBlock AIJ --

class TestMatSeqSBAIJ(BaseTestMatSBAIJ):
    COMM = PETSc.COMM_SELF
    TYPE = PETSc.Mat.Type.SEQSBAIJ
# bs = 1
class TestMatSeqSBAIJ_G23(TestMatSeqSBAIJ):
    GRID  = 2, 3
class TestMatSeqSBAIJ_G45(TestMatSeqSBAIJ):
    GRID  = 4, 5
class TestMatSeqSBAIJ_G89(TestMatSeqSBAIJ):
    GRID  = 8, 9
# bs = 2
class TestMatSeqSBAIJ_G23_B2(TestMatSeqSBAIJ_G23):
    BSIZE = 2
class TestMatSeqSBAIJ_G45_B2(TestMatSeqSBAIJ_G45):
    BSIZE = 2
class TestMatSeqSBAIJ_G89_B2(TestMatSeqSBAIJ_G89):
    BSIZE = 2
# bs = 3
class TestMatSeqSBAIJ_G23_B3(TestMatSeqSBAIJ_G23):
    BSIZE = 3
class TestMatSeqSBAIJ_G45_B3(TestMatSeqSBAIJ_G45):
    BSIZE = 3
class TestMatSeqSBAIJ_G89_B3(TestMatSeqSBAIJ_G89):
    BSIZE = 3
# bs = 4
class TestMatSeqSBAIJ_G23_B4(TestMatSeqSBAIJ_G23):
    BSIZE = 4
class TestMatSeqSBAIJ_G45_B4(TestMatSeqSBAIJ_G45):
    BSIZE = 4
class TestMatSeqSBAIJ_G89_B4(TestMatSeqSBAIJ_G89):
    BSIZE = 4
# bs = 5
class TestMatSeqSBAIJ_G23_B5(TestMatSeqSBAIJ_G23):
    BSIZE = 5
class TestMatSeqSBAIJ_G45_B5(TestMatSeqSBAIJ_G45):
    BSIZE = 5
class TestMatSeqSBAIJ_G89_B5(TestMatSeqSBAIJ_G89):
    BSIZE = 5


# -- MPI SymmBlock AIJ --

class TestMatMPISBAIJ(BaseTestMatSBAIJ):
    COMM = PETSc.COMM_WORLD
    TYPE = PETSc.Mat.Type.MPISBAIJ
# bs = 1
class TestMatMPISBAIJ_G23(TestMatMPISBAIJ):
    GRID  = 2, 3
class TestMatMPISBAIJ_G45(TestMatMPISBAIJ):
    GRID  = 4, 5
class TestMatMPISBAIJ_G89(TestMatMPISBAIJ):
    GRID  = 8, 9
# bs = 2
class TestMatMPISBAIJ_G23_B2(TestMatMPISBAIJ_G23):
    BSIZE = 2
class TestMatMPISBAIJ_G45_B2(TestMatMPISBAIJ_G45):
    BSIZE = 2
class TestMatMPISBAIJ_G89_B2(TestMatMPISBAIJ_G89):
    BSIZE = 2
# bs = 3
class TestMatMPISBAIJ_G23_B3(TestMatMPISBAIJ_G23):
    BSIZE = 3
class TestMatMPISBAIJ_G45_B3(TestMatMPISBAIJ_G45):
    BSIZE = 3
class TestMatMPISBAIJ_G89_B3(TestMatMPISBAIJ_G89):
    BSIZE = 3
# bs = 4
class TestMatMPISBAIJ_G23_B4(TestMatMPISBAIJ_G23):
    BSIZE = 4
class TestMatMPISBAIJ_G45_B4(TestMatMPISBAIJ_G45):
    BSIZE = 4
class TestMatMPISBAIJ_G89_B4(TestMatMPISBAIJ_G89):
    BSIZE = 4
# bs = 5
class TestMatMPISBAIJ_G23_B5(TestMatMPISBAIJ_G23):
    BSIZE = 5
class TestMatMPISBAIJ_G45_B5(TestMatMPISBAIJ_G45):
    BSIZE = 5
class TestMatMPISBAIJ_G89_B5(TestMatMPISBAIJ_G89):
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
        self._chk_bs(self.A, self.BSIZE)
    def _chk_aij(self, A, i, j):
        bs = self.BSIZE or 1
        ai, aj = A.getRowIJ()
        if ai is not None and aj is not None:  ## XXX map and check !!
            #self.assertTrue(N.all(i==ai))
            #self.assertTrue(N.all(j==aj))
            pass
        ai, aj = A.getColumnIJ(compressed=bool(self.BSIZE))
        if ai is not None and aj is not None: ## XXX map and check !!
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

# -- Non-square blocks --
class BaseTestMatAIJ_B(BaseTestMatAnyAIJ, unittest.TestCase):
    COMM  = PETSc.COMM_WORLD
    TYPE  = PETSc.Mat.Type.AIJ
    GRID  = 0, 0
    BSIZE = 4, 2

    def _preallocate(self):
        try:
            rbs, cbs = self.BSIZE
        except (TypeError, ValueError):
            rbs = cbs = self.BSIZE
        self.A.setPreallocationNNZ([5*rbs, 3*cbs])
        self._chk_bsizes(self.A, self.BSIZE)
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
    def _chk_aij(self, A, i, j):
        bs = self.BSIZE or 1
        ai, aj = A.getRowIJ()
        if ai is not None and aj is not None:  ## XXX map and check !!
            #self.assertTrue(N.all(i==ai))
            #self.assertTrue(N.all(j==aj))
            pass
        ai, aj = A.getColumnIJ()
        if ai is not None and aj is not None: ## XXX map and check !!
            #self.assertTrue(N.all(i==ai))
            #self.assertTrue(N.all(j==aj))
            pass

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
    TYPE  = PETSc.Mat.Type.AIJCRL

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

# -- MATIS --

class TestMatIS(BaseTestMatAIJ):
    COMM = PETSc.COMM_WORLD
    TYPE = PETSc.Mat.Type.IS
class TestMatIS_G23(TestMatIS):
    GRID  = 2, 3
class TestMatIS_G45(TestMatIS):
    GRID  = 4, 5
class TestMatIS_G89(TestMatIS):
    GRID  = 8, 9

# -----



if __name__ == '__main__':
    unittest.main()
