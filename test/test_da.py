from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class BaseTestDA(object):

    COMM = PETSc.COMM_WORLD
    SIZES = None
    PERIODIC = PETSc.DA.PeriodicType.NONE
    STENCIL = PETSc.DA.StencilType.STAR
    NDOF = 1
    SWIDTH = 1

    def setUp(self):
        self.da = PETSc.DA().create(self.SIZES,
                                    periodic=self.PERIODIC,
                                    stencil=self.STENCIL,
                                    ndof=self.NDOF,
                                    width=self.SWIDTH,
                                    comm=self.COMM)

    def tearDown(self):
        self.da.destroy()
        self.da = None

    def testGetInfo(self):
        dim = self.da.getDim()
        sizes = self.da.getSizes()
        psizes = self.da.getProcSizes()
        stencil_type = self.da.getStencilType()
        periodic_type = self.da.getPeriodicType()
        ndof = self.da.getNDof()
        width = self.da.getWidth()
        self.assertEqual(dim, len(self.SIZES))
        self.assertEqual(sizes, tuple(self.SIZES))
        self.assertEqual(periodic_type, self.PERIODIC)
        self.assertEqual(stencil_type, self.STENCIL)
        self.assertEqual(ndof, self.NDOF)
        self.assertEqual(width, self.SWIDTH)

    def testRangesCorners(self):
        dim = self.da.getDim()
        ranges = self.da.getRanges()
        starts, lsizes  = self.da.getCorners()
        self.assertEqual(dim, len(ranges))
        self.assertEqual(dim, len(starts))
        self.assertEqual(dim, len(lsizes))
        for i in range(dim):
            s, e = ranges[i]
            self.assertEqual(s, starts[i])
            self.assertEqual(e-s, lsizes[i])

    def testGhostRangesCorners(self):
        dim = self.da.getDim()
        ranges = self.da.getGhostRanges()
        starts, lsizes  = self.da.getGhostCorners()
        self.assertEqual(dim, len(ranges))
        self.assertEqual(dim, len(starts))
        self.assertEqual(dim, len(lsizes))
        for i in range(dim):
            s, e = ranges[i]
            self.assertEqual(s, starts[i])
            self.assertEqual(e-s, lsizes[i])

    def testGetVector(self):
        vn = self.da.createNaturalVector()
        vg = self.da.createGlobalVector()
        vl = self.da.createLocalVector()

    def testGetMatrix(self):
        mat = self.da.getMatrix()
        self.assertTrue(mat.getType() in ('aij', 'seqaij', 'mpiaij'))

    def testGetScatter(self):
        l2g, g2l, l2l = self.da.getScatter()

    def testGetLGMap(self):
        lgmap = self.da.getLGMap()

    def testGetAO(self):
        ao = self.da.getAO()


class BaseTestDA_1D(BaseTestDA):
    SIZES = [100]

class BaseTestDA_2D(BaseTestDA):
    SIZES = [9,11]

class BaseTestDA_3D(BaseTestDA):
    SIZES = [4,5,6]

# --------------------------------------------------------------------

class TestDA_1D(BaseTestDA_1D, unittest.TestCase):
    pass
class TestDA_1D_W0(TestDA_1D):
    SWIDTH = 0
class TestDA_1D_W2(TestDA_1D):
    SWIDTH = 2

class TestDA_2D(BaseTestDA_2D, unittest.TestCase):
    pass
class TestDA_2D_W0(TestDA_2D):
    SWIDTH = 0
class TestDA_2D_W0_N2(TestDA_2D):
    SWIDTH = 0
    NDOF = 2
class TestDA_2D_W2(TestDA_2D):
    SWIDTH = 2
class TestDA_2D_W2_N2(TestDA_2D):
    SWIDTH = 2
    NDOF = 2

class TestDA_3D(BaseTestDA_3D, unittest.TestCase):
    pass
class TestDA_3D_W0(TestDA_3D):
    SWIDTH = 0
class TestDA_3D_W0_N2(TestDA_3D):
    SWIDTH = 0
    NDOF = 2

# The two below fails in 5 procs ...
## class TestDA_3D_W2(TestDA_3D):
##     SWIDTH = 2
## class TestDA_3D_W2_N2(TestDA_3D):
##     NDOF = 2
##     SWIDTH = 2

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
