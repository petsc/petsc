from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class BaseTestDA(object):

    COMM = PETSc.COMM_WORLD
    SIZES = None
    BOUNDARY = None
    DOF = 1
    STENCIL = PETSc.DA.StencilType.STAR
    SWIDTH = 1

    def setUp(self):
        self.da = PETSc.DA().create(dim=len(self.SIZES),
                                    dof=self.DOF,
                                    sizes=self.SIZES,
                                    boundary_type=self.BOUNDARY,
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
        boundary = self.da.getBoundary()
        stencil_type = self.da.getStencilType()
        stencil_width = self.da.getStencilWidth()
        self.assertEqual(dim, len(self.SIZES))
        self.assertEqual(dof, self.DOF)
        self.assertEqual(sizes, tuple(self.SIZES))
        self.assertEqual(boundary, self.BOUNDARY or (0,)*dim)
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

    def testGetLGMapBlock(self):
        lgmap = self.da.getLGMapBlock()

    def testGetAO(self):
        ao = self.da.getAO()

    def testRefineCoarsen(self):
        da = self.da
        rda = da.refine()
        self.assertEqual(da.getDim(), rda.getDim())
        self.assertEqual(da.getDof(), rda.getDof())
        if da.dim != 1:
            self.assertEqual(da.getStencilType(),  rda.getStencilType())
        self.assertEqual(da.getStencilWidth(), rda.getStencilWidth())
        cda = rda.coarsen()
        for n1, n2 in zip(self.da.getSizes(), cda.getSizes()):
            self.assertTrue(abs(n1-n2)<=1)

    def testCoarsenRefine(self):
        da = self.da
        cda = self.da.coarsen()
        self.assertEqual(da.getDim(), cda.getDim())
        self.assertEqual(da.getDof(), cda.getDof())
        if da.dim != 1:
            self.assertEqual(da.getStencilType(),  cda.getStencilType())
        self.assertEqual(da.getStencilWidth(), cda.getStencilWidth())
        rda = cda.refine()
        for n1, n2 in zip(self.da.getSizes(), rda.getSizes()):
            self.assertTrue(abs(n1-n2)<=1)

    def testRefineHierarchy(self):
        levels = self.da.refineHierarchy(2)
        self.assertTrue(isinstance(levels, list))
        self.assertEqual(len(levels), 2)
        for item in levels:
            self.assertTrue(isinstance(item, PETSc.DM))

    def testCoarsenHierarchy(self):
        levels = self.da.coarsenHierarchy(2)
        self.assertTrue(isinstance(levels, list))
        self.assertEqual(len(levels), 2)
        for item in levels:
            self.assertTrue(isinstance(item, PETSc.DM))

    def testGetInterpolation(self):
        da = self.da
        if da.dim == 1: return
        rda = da.refine()
        mat, vec = da.getInterpolation(rda)

    def testGetInjection(self):
        da = self.da
        if da.dim == 1: return
        if (da.dim == 3 and 
            PETSc.Sys.getVersion() < (3, 2)): return
        rda = da.refine()
        scatter = da.getInjection(rda)

    def testGetAggregates(self):
        da = self.da
        if da.dim == 1: return
        rda = da.refine()
        mat = da.getAggregates(rda)

SCALE = 4

class BaseTestDA_1D(BaseTestDA):
    SIZES = [100*SCALE]

class BaseTestDA_2D(BaseTestDA):
    SIZES = [9*SCALE,11*SCALE]

class BaseTestDA_3D(BaseTestDA):
    SIZES = [6*SCALE,7*SCALE,8*SCALE]

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
class TestDA_2D_PXY(TestDA_2D):
    DOF = 2
    SWIDTH = 5
    BOUNDARY = (1,1)
class TestDA_2D_GXY(TestDA_2D):
    DOF = 2
    SWIDTH = 5
    BOUNDARY = (2,2)

class TestDA_3D(BaseTestDA_3D, unittest.TestCase):
    pass
class TestDA_3D_W0(TestDA_3D):
    SWIDTH = 0
class TestDA_3D_W0_N2(TestDA_3D):
    DOF = 2
    SWIDTH = 0
class TestDA_3D_W2(TestDA_3D):
    SWIDTH = 2
class TestDA_3D_W2_N2(TestDA_3D):
    DOF = 2
    SWIDTH = 2
class TestDA_3D_PXYZ(TestDA_3D):
    DOF = 2
    SWIDTH = 3
    BOUNDARY = (1,1,1)
class TestDA_3D_GXYZ(TestDA_3D):
    DOF = 2
    SWIDTH = 3
    BOUNDARY = (2,2,2)

# --------------------------------------------------------------------

class TestDACreate(unittest.TestCase):

    def testCreate(self):
        da = PETSc.DA()
        for dim in (1,2,3):
            for dof in (1,2,3,4,5):
                for boundary in (None, "periodic", "ghosted",
                                 (False,)*dim, (True,)*dim,
                                 (0,)*dim,(1,)*dim,(2,)*dim,):
                    for stencil_type in (None, "box", "star"):
                        da.create([8*SCALE]*dim, dof=dof,
                                  boundary_type=boundary,
                                  stencil_type=stencil_type)
                        da.destroy()

    def testDuplicate(self):
        da = PETSc.DA()
        for dim in (1,2,3):
            da.create([8*SCALE]*dim)
            for dof in (None, 1,2,3,4,5):
                for boundary in (None, "periodic", "ghosted",
                                 (0,)*dim,(1,)*dim,(2,)*dim,):
                    for stencil in (None, "box", "star"):
                        for width in (None, 1,2,3,4,5):
                            newda = da.duplicate(
                                dof=dof,
                                boundary_type=boundary,
                                stencil_type=stencil,
                                stencil_width=width)
                            self.assertEqual(newda.dim, da.dim)
                            self.assertEqual(newda.sizes, da.sizes)
                            self.assertEqual(newda.proc_sizes, da.proc_sizes)
                            self.assertEqual(newda.ranges, da.ranges)
                            self.assertEqual(newda.corners, 
                                             da.corners)
                            if (newda.boundary == da.boundary and
                                newda.stencil_width == da.stencil_width):
                                self.assertEqual(newda.ghost_ranges,
                                                 da.ghost_ranges)
                                self.assertEqual(newda.ghost_corners,
                                                 da.ghost_corners)
                            if dof is None: 
                                dof = da.dof
                            if boundary is None:
                                boundary = da.boundary
                            elif boundary == "periodic":
                                boundary = (1,) * dim
                            elif boundary == "ghosted":
                                boundary = (2,) * dim
                            if stencil is None:
                                stencil = da.stencil
                            if width is None: 
                                width = da.stencil_width
                            self.assertEqual(newda.dof, dof)
                            self.assertEqual(newda.boundary, boundary)
                            self.assertEqual(newda.stencil, stencil)
                            self.assertEqual(newda.stencil_width, width)
                            newda.destroy()
        da.destroy()
        

# --------------------------------------------------------------------

if PETSc.COMM_WORLD.getSize() > 1:
    del TestDA_1D_W0
    del TestDA_2D_W0, TestDA_2D_W0_N2
    del TestDA_3D_W0, TestDA_3D_W0_N2
                        
# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
