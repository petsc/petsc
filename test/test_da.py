from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class TestDABase(object):

    COMM = PETSc.COMM_WORLD
    SIZES = None
    STENCIL = PETSc.DA.StencilType.STAR
    PERIODIC = PETSc.DA.PeriodicType.NONE

    def setUp(self):
        self.da = PETSc.DA().create(self.SIZES,
                                    periodic=self.PERIODIC,
                                    stencil=self.STENCIL,
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
        corners = self.da.getCorners()
        ghostcorners = self.da.getGhostCorners()
        self.assertEqual(dim, len(self.SIZES))
        self.assertEqual(sizes, tuple(self.SIZES))
        self.assertEqual(stencil_type, self.STENCIL)
        self.assertEqual(periodic_type, self.PERIODIC)
        self.assertEqual(ndof, 1)
        self.assertEqual(width, 1)

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


class TestDABase_1D(TestDABase):
    SIZES = [100]

class TestDABase_2D(TestDABase):
    SIZES = [9,11]

class TestDABase_3D(TestDABase):
    SIZES = [4,5,6]

# --------------------------------------------------------------------

class TestDA_1D(TestDABase_1D, unittest.TestCase):
    pass

class TestDA_2D(TestDABase_2D, unittest.TestCase):
    pass

class TestDA_3D(TestDABase_3D, unittest.TestCase):
    pass

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
