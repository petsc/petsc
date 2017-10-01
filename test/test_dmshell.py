from petsc4py import PETSc
import unittest
import numpy as np


class TestDMShell(unittest.TestCase):

    COMM = PETSc.COMM_WORLD

    def setUp(self):
        self.dm = PETSc.DMShell().create(comm=self.COMM)

    def tearDown(self):
        self.dm = None

    def testSetGlobalVector(self):
        vec = PETSc.Vec().create(comm=self.COMM)
        vec.setSizes((10, None))
        vec.setUp()
        self.dm.setGlobalVector(vec)
        gvec = self.dm.createGlobalVector()
        self.assertEqual(vec.getSizes(), gvec.getSizes())
        self.assertEqual(vec.comm, gvec.comm)

    def testSetCreateGlobalVector(self):
        def create_vec(dm):
            v = PETSc.Vec().create(comm=dm.comm)
            v.setSizes((10, None))
            v.setUp()
            return v
        self.dm.setCreateGlobalVector(create_vec)
        gvec = self.dm.createGlobalVector()
        self.assertEqual(gvec.comm, self.dm.comm)
        self.assertEqual(gvec.getLocalSize(), 10)

    def testSetLocalVector(self):
        vec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        vec.setSizes((1 + 10*self.COMM.rank, None))
        vec.setUp()
        self.dm.setLocalVector(vec)
        lvec = self.dm.createLocalVector()
        self.assertEqual(vec.getSizes(), lvec.getSizes())
        lsize, gsize = lvec.getSizes()
        self.assertEqual(lsize, gsize)
        self.assertEqual(lvec.comm, PETSc.COMM_SELF)

    def testSetCreateLocalVector(self):
        def create_vec(dm):
            v = PETSc.Vec().create(comm=PETSc.COMM_SELF)
            v.setSizes((1 + 10*dm.comm.rank, None))
            v.setUp()
            return v
        self.dm.setCreateLocalVector(create_vec)
        lvec = self.dm.createLocalVector()
        lsize, gsize = lvec.getSizes()
        self.assertEqual(lsize, gsize)
        self.assertEqual(lsize, 1 + 10*self.dm.comm.rank)
        self.assertEqual(lvec.comm, PETSc.COMM_SELF)

    def testSetMatrix(self):
        mat = PETSc.Mat().create(comm=self.COMM)
        mat.setSizes(((10, None), (2, None)))
        mat.setUp()
        mat.assemble()
        self.dm.setMatrix(mat)
        nmat = self.dm.createMatrix()
        self.assertEqual(nmat.getSizes(), mat.getSizes())

    def testSetCreateMatrix(self):
        def create_mat(dm):
            mat = PETSc.Mat().create(comm=self.COMM)
            mat.setSizes(((10, None), (2, None)))
            mat.setUp()
            return mat
        self.dm.setCreateMatrix(create_mat)
        nmat = self.dm.createMatrix()
        self.assertEqual(nmat.getSizes(), create_mat(self.dm).getSizes())

    def testGlobalToLocal(self):
        def begin(dm, ivec, mode, ovec):
            if mode == PETSc.InsertMode.INSERT_VALUES:
                ovec[...] = ivec[...]
            elif mode == PETSc.InsertMode.ADD_VALUES:
                ovec[...] += ivec[...]
        def end(dm, ivec, mode, ovec):
            pass
        vec = PETSc.Vec().create(comm=self.COMM)
        vec.setSizes((10, None))
        vec.setUp()
        vec[...] = self.dm.comm.rank + 1
        ovec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        ovec.setSizes((10, None))
        ovec.setUp()
        self.dm.setGlobalToLocal(begin, end)
        self.dm.globalToLocal(vec, ovec, addv=PETSc.InsertMode.INSERT_VALUES)
        self.assertTrue(np.allclose(vec.getArray(), ovec.getArray()))
        self.dm.globalToLocal(vec, ovec, addv=PETSc.InsertMode.ADD_VALUES)
        self.assertTrue(np.allclose(2*vec.getArray(), ovec.getArray()))

    def testLocalToGlobal(self):
        def begin(dm, ivec, mode, ovec):
            if mode == PETSc.InsertMode.INSERT_VALUES:
                ovec[...] = ivec[...]
            elif mode == PETSc.InsertMode.ADD_VALUES:
                ovec[...] += ivec[...]
        def end(dm, ivec, mode, ovec):
            pass
        vec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        vec.setSizes((10, None))
        vec.setUp()
        vec[...] = self.dm.comm.rank + 1
        ovec = PETSc.Vec().create(comm=self.COMM)
        ovec.setSizes((10, None))
        ovec.setUp()
        self.dm.setLocalToGlobal(begin, end)
        self.dm.localToGlobal(vec, ovec, addv=PETSc.InsertMode.INSERT_VALUES)
        self.assertTrue(np.allclose(vec.getArray(), ovec.getArray()))
        self.dm.localToGlobal(vec, ovec, addv=PETSc.InsertMode.ADD_VALUES)
        self.assertTrue(np.allclose(2*vec.getArray(), ovec.getArray()))

    def testLocalToLocal(self):
        def begin(dm, ivec, mode, ovec):
            if mode == PETSc.InsertMode.INSERT_VALUES:
                ovec[...] = ivec[...]
            elif mode == PETSc.InsertMode.ADD_VALUES:
                ovec[...] += ivec[...]
        def end(dm, ivec, mode, ovec):
            pass
        vec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        vec.setSizes((10, None))
        vec.setUp()
        vec[...] = self.dm.comm.rank + 1
        ovec = vec.duplicate()
        self.dm.setLocalToLocal(begin, end)
        self.dm.localToLocal(vec, ovec, addv=PETSc.InsertMode.INSERT_VALUES)
        self.assertTrue(np.allclose(vec.getArray(), ovec.getArray()))
        self.dm.localToLocal(vec, ovec, addv=PETSc.InsertMode.ADD_VALUES)
        self.assertTrue(np.allclose(2*vec.getArray(), ovec.getArray()))

    def testGlobalToLocalVecScatter(self):
        vec = PETSc.Vec().create()
        vec.setSizes((10, None))
        vec.setUp()
        sct, ovec = PETSc.Scatter.toAll(vec)
        self.dm.setGlobalToLocalVecScatter(sct)

        self.dm.globalToLocal(vec, ovec, addv=PETSc.InsertMode.INSERT_VALUES)

        self.assertTrue(np.allclose(vec.getArray(), ovec.getArray()))

    def testGlobalToLocalVecScatter(self):
        vec = PETSc.Vec().create()
        vec.setSizes((10, None))
        vec.setUp()
        sct, ovec = PETSc.Scatter.toAll(vec)
        self.dm.setGlobalToLocalVecScatter(sct)
        self.dm.globalToLocal(vec, ovec, addv=PETSc.InsertMode.INSERT_VALUES)

    def testLocalToGlobalVecScatter(self):
        vec = PETSc.Vec().create()
        vec.setSizes((10, None))
        vec.setUp()
        sct, ovec = PETSc.Scatter.toAll(vec)
        self.dm.setLocalToGlobalVecScatter(sct)
        self.dm.localToGlobal(vec, ovec, addv=PETSc.InsertMode.INSERT_VALUES)

    def testLocalToLocalVecScatter(self):
        vec = PETSc.Vec().create()
        vec.setSizes((10, None))
        vec.setUp()
        sct, ovec = PETSc.Scatter.toAll(vec)
        self.dm.setLocalToLocalVecScatter(sct)
        self.dm.localToLocal(vec, ovec, addv=PETSc.InsertMode.INSERT_VALUES)

    def testCoarsenRefine(self):
        cdm = PETSc.DMShell().create(comm=self.COMM)
        def coarsen(dm, comm):
            return cdm
        def refine(dm, comm):
            return self.dm
        cdm.setRefine(refine)
        self.dm.setCoarsen(coarsen)
        coarsened = self.dm.coarsen()
        self.assertEqual(coarsened, cdm)
        refined = coarsened.refine()
        self.assertEqual(refined, self.dm)

    def testCreateInterpolation(self):
        mat = PETSc.Mat().create()
        mat.setSizes(((10, None), (10, None)))
        mat.setUp()
        vec = PETSc.Vec().create()
        vec.setSizes((10, None))
        vec.setUp()
        def create_interp(dm, dmf):
            return mat, vec
        self.dm.setCreateInterpolation(create_interp)
        m, v = self.dm.createInterpolation(self.dm)
        self.assertEqual(m, mat)
        self.assertEqual(v, vec)

    def testCreateInjection(self):
        mat = PETSc.Mat().create()
        mat.setSizes(((10, None), (10, None)))
        mat.setUp()
        def create_inject(dm, dmf):
            return mat
        self.dm.setCreateInjection(create_inject)
        m = self.dm.createInjection(self.dm)
        self.assertEqual(m, mat)


if __name__ == '__main__':
    unittest.main()
