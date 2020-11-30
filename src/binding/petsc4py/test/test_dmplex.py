import petsc4py
from petsc4py import PETSc
import unittest
import os
import filecmp
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

PETSC_DIR = petsc4py.get_config()['PETSC_DIR']

class BaseTestPlexHDF5(object):
    NSIZE = 4
    NTIMES = 3

    def setUp(self):
        self.txtvwr = PETSc.Viewer()

    def tearDown(self):
        if not PETSc.COMM_WORLD.rank:
            if os.path.exists(self.outfile()):
                os.remove(self.outfile())
            if os.path.exists(self.tmp_output_file()):
                os.remove(self.tmp_output_file())
        self.txtvwr = None

    def _name(self):
        return "%s_outformat-%s_%s" % (self.SUFFIX,
                                       self.OUTFORMAT,
                                       self.PARTITIONERTYPE)

    def infile(self):
        return os.path.join(PETSC_DIR, "share/petsc/datafiles/",
                            "meshes/blockcylinder-50.h5")

    def outfile(self):
        return os.path.join("./temp_test_dmplex_%s.h5" % self._name())

    def informat(self):
        return PETSc.Viewer.Format.HDF5_XDMF

    def outformat(self):
        d = {"hdf5_petsc": PETSc.Viewer.Format.HDF5_PETSC,
             "hdf5_xdmf": PETSc.Viewer.Format.HDF5_XDMF}
        return d[self.OUTFORMAT]

    def partitionerType(self):
        d = {"simple": PETSc.Partitioner.Type.SIMPLE,
             "ptscotch": PETSc.Partitioner.Type.PTSCOTCH,
             "parmetis": PETSc.Partitioner.Type.PARMETIS}
        return d[self.PARTITIONERTYPE]

    def ref_output_file(self):
        return os.path.join(PETSC_DIR, "src/dm/impls/plex/tutorials/",
                            "output/ex5_%s.out" % self._name())

    def tmp_output_file(self):
        return os.path.join("./temp_test_dmplex_%s.out" % self._name())

    def outputText(self, msg, comm):
        if not comm.rank:
            with open(self.tmp_output_file(), 'a') as f:
                f.write(msg)

    def outputPlex(self, plex):
        self.txtvwr.createASCII(self.tmp_output_file(),
                                mode='a', comm=plex.comm)
        plex.view(viewer=self.txtvwr)
        self.txtvwr.destroy()

    def testViewLoadCycle(self):
        if PETSc.COMM_WORLD.size != self.NSIZE:
            return
        grank = PETSc.COMM_WORLD.rank
        for i in range(self.NTIMES):
            if i == 0:
                infname = self.infile()
                informt = self.informat()
            else:
                infname = self.outfile()
                informt = self.outformat()
            if self.HETEROGENEOUS:
                mycolor = (grank > self.NTIMES - i)
            else:
                mycolor = 0
            mpicomm = PETSc.COMM_WORLD.tompi4py()
            comm = PETSc.Comm(comm=mpicomm.Split(color=mycolor, key=grank))
            if mycolor == 0:
                self.outputText("Begin cycle %d\n" % i, comm)
                plex = PETSc.DMPlex()
                vwr = PETSc.ViewerHDF5()
                # Create plex
                plex.create(comm=comm)
                plex.setName("DMPlex Object")
                # Load data from XDMF into dm in parallel
                vwr.create(infname, mode='r', comm=comm)
                vwr.pushFormat(format=informt)
                plex.load(viewer=vwr)
                plex.setOptionsPrefix("loaded_")
                plex.setFromOptions()
                vwr.popFormat()
                vwr.destroy()
                self.outputPlex(plex)
                # Test DM is indeed distributed
                flg = plex.isDistributed()
                self.outputText("Loaded mesh distributed? %s\n" %
                                str(flg).upper(), comm)
                # Interpolate
                plex.interpolate()
                plex.setOptionsPrefix("interpolated_")
                plex.setFromOptions()
                self.outputPlex(plex)
                # Redistribute
                part = plex.getPartitioner()
                part.setType(self.partitionerType())
                _ = plex.distribute(overlap=0)
                plex.setOptionsPrefix("redistributed_")
                plex.setFromOptions()
                self.outputPlex(plex)
                # Save redistributed dm to XDMF in parallel
                vwr.create(self.outfile(), mode='w', comm=comm)
                vwr.pushFormat(format=self.outformat())
                plex.view(viewer=vwr)
                vwr.popFormat()
                vwr.destroy()
                # Destroy plex
                plex.destroy()
                self.outputText("End   cycle %d\n--------\n" % i, comm)
            PETSc.COMM_WORLD.Barrier()
        # Check that the output is identical to that of plex/tutorial/ex5.c.
        self.assertTrue(filecmp.cmp(self.tmp_output_file(),
                                    self.ref_output_file(), shallow=False),
                        'Contents of the files not the same.')

class BaseTestPlexHDF5Homogeneous(BaseTestPlexHDF5):
    """Test save on N / load on N."""
    SUFFIX = 0
    HETEROGENEOUS = False

class BaseTestPlexHDF5Heterogeneous(BaseTestPlexHDF5):
    """Test save on N / load on M."""
    SUFFIX = 1
    HETEROGENEOUS = True

class TestPlexHDF5PETSCSimpleHomogeneous(BaseTestPlexHDF5Homogeneous,
                                         unittest.TestCase):
    OUTFORMAT = "hdf5_petsc"
    PARTITIONERTYPE = "simple"

"""
Skipping. PTScotch produces different distributions when run
in a sequence in a single session.

class TestPlexHDF5PETSCPTScotchHomogeneous(BaseTestPlexHDF5Homogeneous,
                                           unittest.TestCase):
    OUTFORMAT = "hdf5_petsc"
    PARTITIONERTYPE = "ptscotch"
"""

class TestPlexHDF5PETSCParmetisHomogeneous(BaseTestPlexHDF5Homogeneous,
                                           unittest.TestCase):
    OUTFORMAT = "hdf5_petsc"
    PARTITIONERTYPE = "parmetis"

class TestPlexHDF5XDMFSimpleHomogeneous(BaseTestPlexHDF5Homogeneous,
                                        unittest.TestCase):
    OUTFORMAT = "hdf5_xdmf"
    PARTITIONERTYPE = "simple"

"""
Skipping. PTScotch produces different distributions when run
in a sequence in a single session.

class TestPlexHDF5XDMFPTScotchHomogeneous(BaseTestPlexHDF5Homogeneous,
                                          unittest.TestCase):
    OUTFORMAT = "hdf5_xdmf"
    PARTITIONERTYPE = "ptscotch"
"""

class TestPlexHDF5XDMFParmetisHomogeneous(BaseTestPlexHDF5Homogeneous,
                                          unittest.TestCase):
    OUTFORMAT = "hdf5_xdmf"
    PARTITIONERTYPE = "parmetis"

class TestPlexHDF5PETSCSimpleHeterogeneous(BaseTestPlexHDF5Heterogeneous,
                                           unittest.TestCase):
    OUTFORMAT = "hdf5_petsc"
    PARTITIONERTYPE = "simple"

"""
Skipping. PTScotch produces different distributions when run
in a sequence in a single session.

class TestPlexHDF5PETSCPTScotchHeterogeneous(BaseTestPlexHDF5Heterogeneous,
                                             unittest.TestCase):
    OUTFORMAT = "hdf5_petsc"
    PARTITIONERTYPE = "ptscotch"
"""

class TestPlexHDF5PETSCParmetisHeterogeneous(BaseTestPlexHDF5Heterogeneous,
                                             unittest.TestCase):
    OUTFORMAT = "hdf5_petsc"
    PARTITIONERTYPE = "parmetis"

class TestPlexHDF5XDMFSimpleHeterogeneous(BaseTestPlexHDF5Heterogeneous,
                                          unittest.TestCase):
    OUTFORMAT = "hdf5_xdmf"
    PARTITIONERTYPE = "simple"

class TestPlexHDF5XDMFPTScotchHeterogeneous(BaseTestPlexHDF5Heterogeneous,
                                            unittest.TestCase):
    OUTFORMAT = "hdf5_xdmf"
    PARTITIONERTYPE = "ptscotch"

class TestPlexHDF5XDMFParmetisHeterogeneous(BaseTestPlexHDF5Heterogeneous,
                                            unittest.TestCase):
    OUTFORMAT = "hdf5_xdmf"
    PARTITIONERTYPE = "parmetis"

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
