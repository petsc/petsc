from petsc4py import PETSc
import unittest
import numpy as np

# --------------------------------------------------------------------

ERR_SUP = 56

class BaseTestPlex(object):

    COMM = PETSc.COMM_WORLD
    DIM = 1
    CELLS = [[0, 1], [1, 2]]
    COORDS = [[0.], [0.5], [1.]]
    COMP = 1
    DOFS = [1, 0]

    def setUp(self):
        self.plex = PETSc.DMPlex().createFromCellList(self.DIM,
                                                      self.CELLS,
                                                      self.COORDS,
                                                      comm=self.COMM)

    def tearDown(self):
        self.plex.destroy()
        self.plex = None

    def testTopology(self):
        dim = self.plex.getDimension()
        pStart, pEnd = self.plex.getChart()
        cStart, cEnd = self.plex.getHeightStratum(0)
        vStart, vEnd = self.plex.getDepthStratum(0)
        numDepths = self.plex.getLabelSize("depth")
        coords_raw = self.plex.getCoordinates().getArray()
        coords = np.reshape(coords_raw, (vEnd - vStart, dim))
        self.assertEqual(dim, self.DIM)
        self.assertEqual(numDepths, self.DIM+1)
        if self.CELLS is not None:
            self.assertEqual(cEnd-cStart, len(self.CELLS))
        if self.COORDS is not None:
            self.assertEqual(vEnd-vStart, len(self.COORDS))
            self.assertTrue((coords == self.COORDS).all())

    def testClosure(self):
        pStart, pEnd = self.plex.getChart()
        for p in range(pStart, pEnd):
            closure = self.plex.getTransitiveClosure(p)[0]
            for c in closure:
                cone = self.plex.getCone(c)
                self.assertEqual(self.plex.getConeSize(c), len(cone))
                for i in cone:
                    self.assertIn(i, closure)
            star = self.plex.getTransitiveClosure(p, useCone=False)[0]
            for s in star:
                support = self.plex.getSupport(s)
                self.assertEqual(self.plex.getSupportSize(s), len(support))
                for i in support:
                    self.assertIn(i, star)

    def testAdjacency(self):
        PETSc.DMPlex.setAdjacencyUseAnchors(self.plex, False)
        flag = PETSc.DMPlex.getAdjacencyUseAnchors(self.plex)
        self.assertFalse(flag)
        PETSc.DMPlex.setAdjacencyUseAnchors(self.plex, True)
        flag = PETSc.DMPlex.getAdjacencyUseAnchors(self.plex)
        self.assertTrue(flag)
        PETSc.DMPlex.setBasicAdjacency(self.plex, False, False)
        flagA, flagB = PETSc.DMPlex.getBasicAdjacency(self.plex)
        self.assertFalse(flagA)
        self.assertFalse(flagB)
        PETSc.DMPlex.setBasicAdjacency(self.plex, True, True)
        flagA, flagB = PETSc.DMPlex.getBasicAdjacency(self.plex)
        self.assertTrue(flagA)
        self.assertTrue(flagB)
        pStart, pEnd = self.plex.getChart()
        for p in range(pStart, pEnd):
            adjacency = self.plex.getAdjacency(p)
            self.assertTrue(p in adjacency)
            self.assertTrue(len(adjacency) > 1)

    def testSectionDofs(self):
        self.plex.setNumFields(1)
        section = self.plex.createSection([self.COMP], [self.DOFS])
        size = section.getStorageSize()
        entity_dofs = [self.plex.getStratumSize("depth", d) *
                       self.DOFS[d] for d in range(self.DIM+1)]
        self.assertEqual(sum(entity_dofs), size)

    def testSectionClosure(self):
        section = self.plex.createSection([self.COMP], [self.DOFS])
        self.plex.setSection(section)
        vec = self.plex.createLocalVec()
        pStart, pEnd = self.plex.getChart()
        for p in range(pStart, pEnd):
            for i in range(section.getDof(p)):
                off = section.getOffset(p)
                vec.setValue(off+i, p)

        for p in range(pStart, pEnd):
            point_closure = self.plex.getTransitiveClosure(p)[0]
            dof_closure = self.plex.vecGetClosure(section, vec, p)
            for p in dof_closure:
                self.assertIn(p, point_closure)

    def testBoundaryLabel(self):
        self.assertFalse(self.plex.hasLabel("boundary"))
        self.plex.markBoundaryFaces("boundary")
        self.assertTrue(self.plex.hasLabel("boundary"))

        faces = self.plex.getStratumIS("boundary", 1)
        for f in faces.getIndices():
            points, orient = self.plex.getTransitiveClosure(f, useCone=True)
            for p in points:
                self.plex.setLabelValue("boundary", p, 1)

        pStart, pEnd = self.plex.getChart()
        for p in range(pStart, pEnd):
            if self.plex.getLabelValue("boundary", p) != 1:
                self.plex.setLabelValue("boundary", p, 2)

        numBoundary = self.plex.getStratumSize("boundary", 1)
        numInterior = self.plex.getStratumSize("boundary", 2)
        self.assertNotEqual(numBoundary, pEnd - pStart)
        self.assertNotEqual(numInterior, pEnd - pStart)
        self.assertEqual(numBoundary + numInterior, pEnd - pStart)


    def testAdapt(self):
        dim = self.plex.getDimension()
        if dim == 1: return
        vStart, vEnd = self.plex.getDepthStratum(0)
        numVertices = vEnd-vStart
        metric_array = np.zeros([numVertices,dim,dim])
        for met in metric_array:
            met[:,:] = np.diag([9]*dim)
        metric = PETSc.Vec().createWithArray(metric_array)
        try:
            newplex = self.plex.adaptMetric(metric,"")
        except PETSc.Error as exc:
            if exc.ierr != ERR_SUP: raise


# --------------------------------------------------------------------

class BaseTestPlex_2D(BaseTestPlex):
    DIM = 2
    CELLS = [[0, 1, 3], [1, 3, 4], [1, 2, 4], [2, 4, 5],
             [3, 4, 6], [4, 6, 7], [4, 5, 7], [5, 7, 8]]
    COORDS = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
              [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
              [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]
    DOFS = [1, 0, 0]

class BaseTestPlex_3D(BaseTestPlex):
    DIM = 3
    CELLS = [[0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 6, 7],
             [0, 1, 3, 7], [0, 1, 5, 7], [0, 4, 5, 7]]
    COORDS = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
              [0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]]
    DOFS = [1, 0, 0, 0]

# --------------------------------------------------------------------

class TestPlex_1D(BaseTestPlex, unittest.TestCase):
    pass

class TestPlex_2D(BaseTestPlex_2D, unittest.TestCase):
    pass

class TestPlex_3D(BaseTestPlex_3D, unittest.TestCase):
    pass

class TestPlex_2D_P3(BaseTestPlex_2D, unittest.TestCase):
    DOFS = [1, 2, 1]

class TestPlex_3D_P3(BaseTestPlex_3D, unittest.TestCase):
    DOFS = [1, 2, 1, 0]

class TestPlex_3D_P4(BaseTestPlex_3D, unittest.TestCase):
    DOFS = [1, 3, 3, 1]

class TestPlex_2D_BoxTensor(BaseTestPlex_2D, unittest.TestCase):
    CELLS = None
    COORDS = None
    def setUp(self):
        self.plex = PETSc.DMPlex().createBoxMesh([3,3], simplex=False)

class TestPlex_3D_BoxTensor(BaseTestPlex_3D, unittest.TestCase):
    CELLS = None
    COORDS = None
    def setUp(self):
        self.plex = PETSc.DMPlex().createBoxMesh([3,3,3], simplex=False)

import sys
try:
    raise PETSc.Error
    PETSc.DMPlex().createBoxMesh([2,2], simplex=True, comm=PETSc.COMM_SELF).destroy()
except PETSc.Error:
    pass
else:
    class TestPlex_2D_Box(BaseTestPlex_2D, unittest.TestCase):
        CELLS = None
        COORDS = None
        def setUp(self):
            self.plex = PETSc.DMPlex().createBoxMesh([1,1], simplex=True)

    class TestPlex_2D_Boundary(BaseTestPlex_2D, unittest.TestCase):
        CELLS = None
        COORDS = None
        def setUp(self):
            boundary = PETSc.DMPlex().create(self.COMM)
            boundary.createSquareBoundary([0., 0.], [1., 1.], [2, 2])
            boundary.setDimension(self.DIM-1)
            self.plex = PETSc.DMPlex().generate(boundary)

    class TestPlex_3D_Box(BaseTestPlex_3D, unittest.TestCase):
        CELLS = None
        COORDS = None
        def setUp(self):
            self.plex = PETSc.DMPlex().createBoxMesh([1,1,1], simplex=True)

    class TestPlex_3D_Boundary(BaseTestPlex_3D, unittest.TestCase):
        CELLS = None
        COORDS = None
        def setUp(self):
            boundary = PETSc.DMPlex().create(self.COMM)
            boundary.createCubeBoundary([0., 0., 0.], [1., 1., 1.], [1, 1, 1])
            boundary.setDimension(self.DIM-1)
            self.plex = PETSc.DMPlex().generate(boundary)

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
