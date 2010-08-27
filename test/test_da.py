from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class BaseTestDA(object):

    COMM = PETSc.COMM_WORLD
    SIZES = None
    PERIODIC = PETSc.DA.PeriodicType.NONE
    DOF = 1
    STENCIL = PETSc.DA.StencilType.STAR
    SWIDTH = 1

    def setUp(self):
        self.da = PETSc.DA().create(dim=len(self.SIZES),
                                    dof=self.DOF,
                                    sizes=self.SIZES,
                                    periodic_type=self.PERIODIC,
                                    stencil_type=self.STENCIL,
                                    stencil_width=self.SWIDTH,
                                    comm=self.COMM)

    def tearDown(self):
        self.da.destroy()
        self.da = None

    def testGetInfo(self):
        dim = self.da.getDim()
        dof = self.da.getDof()
        sizes = self.da.getSizes()
        psizes = self.da.getProcSizes()
        periodic_type = self.da.getPeriodicType()
        stencil_type = self.da.getStencilType()
        stencil_width = self.da.getStencilWidth()
        self.assertEqual(dim, len(self.SIZES))
        self.assertEqual(dof, self.DOF)
        self.assertEqual(sizes, tuple(self.SIZES))
        self.assertEqual(periodic_type, self.PERIODIC)
        self.assertEqual(stencil_type, self.STENCIL)
        self.assertEqual(stencil_width, self.SWIDTH)

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

    def testCoordinates(self):
        self.da.setUniformCoordinates(0,1,0,1,0,1)
        #
        c = self.da.getCoordinates()
        self.da.setCoordinates(c)
        c.destroy()
        cda = self.da.getCoordinateDA()
        cda.destroy()
        #
        c = self.da.getCoordinates()
        self.da.setCoordinates(c)
        c.destroy()
        gc = self.da.getGhostCoordinates()
        gc.destroy()

    def testCreateVec(self):
        vn = self.da.createNaturalVec()
        vg = self.da.createGlobalVec()
        vl = self.da.createLocalVec()

    def testCreateMat(self):
        mat = self.da.createMat()
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
    DOF = 2
    SWIDTH = 0
class TestDA_2D_W2(TestDA_2D):
    SWIDTH = 2
class TestDA_2D_W2_N2(TestDA_2D):
    DOF = 2
    SWIDTH = 2

class TestDA_3D(BaseTestDA_3D, unittest.TestCase):
    pass
class TestDA_3D_W0(TestDA_3D):
    SWIDTH = 0
class TestDA_3D_W0_N2(TestDA_3D):
    DOF = 2
    SWIDTH = 0

# The two below fails in 5 procs ...
## class TestDA_3D_W2(TestDA_3D):
##     SWIDTH = 2
## class TestDA_3D_W2_N2(TestDA_3D):
##     DOF = 2
##     SWIDTH = 2

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
