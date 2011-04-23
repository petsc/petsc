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
        self.da = None

    def testGetInfo(self):
        dim = self.da.getDim()
        dof = self.da.getDof()
        sizes = self.da.getSizes()
        psizes = self.da.getProcSizes()
        boundary = self.da.getBoundaryType()
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

    def testCreateVecMat(self):
        vn = self.da.createNaturalVec()
        vg = self.da.createGlobalVec()
        vl = self.da.createLocalVec()
        mat = self.da.createMat()
        self.assertTrue(mat.getType() in ('aij', 'seqaij', 'mpiaij'))

    def testGetOther(self):
        ao = self.da.getAO()
        lgmap = self.da.getLGMap()
        lgmap = self.da.getLGMapBlock()
        l2g, g2l, l2l = self.da.getScatter()

    def testRefineCoarsen(self):
        da = self.da
        rda = da.refine()
        self.assertEqual(da.getDim(), rda.getDim())
        self.assertEqual(da.getDof(), rda.getDof())
        if da.dim != 1:
            self.assertEqual(da.getStencilType(),  rda.getStencilType())
        self.assertEqual(da.getStencilWidth(), rda.getStencilWidth())
        cda = rda.coarsen()
        self.assertEqual(rda.getDim(), cda.getDim())
        self.assertEqual(rda.getDof(), cda.getDof())
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


MIRROR   = PETSc.DA.BoundaryType.MIRROR
GHOSTED  = PETSc.DA.BoundaryType.GHOSTED
PERIODIC = PETSc.DA.BoundaryType.PERIODIC

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
    BOUNDARY = (PERIODIC,)*2
class TestDA_2D_GXY(TestDA_2D):
    DOF = 2
    SWIDTH = 5
    BOUNDARY = (GHOSTED,)*2

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
    BOUNDARY = (PERIODIC,)*3
class TestDA_3D_GXYZ(TestDA_3D):
    DOF = 2
    SWIDTH = 3
    BOUNDARY = (GHOSTED,)*3

# --------------------------------------------------------------------

DIM = (1,2,3,)
DOF = (None,1,2,3,4,5,)
BOUNDARY_TYPE = (
    None,
    "none",     (0,)*3,        0,
    "ghosted",  (GHOSTED,)*3,  GHOSTED,
    "periodic", (PERIODIC,)*3, PERIODIC,
    )
STENCIL_TYPE  = (None,"star","box")
STENCIL_WIDTH = (None,0,1,2,3)


DIM           = (1,2,3)
DOF           = (None,2,5)
BOUNDARY_TYPE = (None,"periodic", "ghosted")
STENCIL_TYPE  = (None,"box")
STENCIL_WIDTH = (None,1,2)

class TestDACreate(unittest.TestCase):
    pass
counter = 0
for dim in DIM:
    for dof in DOF:
        for boundary in BOUNDARY_TYPE:
            if isinstance(boundary, tuple):
                boundary = boundary[:dim]
            for stencil in STENCIL_TYPE:
                for width in STENCIL_WIDTH:
                    kargs = dict(sizes=[8*SCALE]*dim,
                                 dim=dim, dof=dof,
                                 boundary_type=boundary,
                                 stencil_type=stencil,
                                 stencil_width=width)
                    def testCreate(self, kargs=kargs):
                        da = PETSc.DA().create(**kargs)
                        da.destroy()
                    setattr(TestDACreate,
                            "testCreate%04d"%counter,
                            testCreate)
                    del testCreate, kargs
                    counter += 1
del counter, dim, dof, boundary, stencil, width

class TestDADuplicate(unittest.TestCase):
    pass
counter = 0
for dim in DIM:
    for dof in DOF:
        for boundary in BOUNDARY_TYPE:
            if isinstance(boundary, tuple):
                boundary = boundary[:dim]
            for stencil in STENCIL_TYPE:
                for width in STENCIL_WIDTH:
                    kargs = dict(dim=dim, dof=dof,
                                 boundary_type=boundary,
                                 stencil_type=stencil,
                                 stencil_width=width)
                    def testDuplicate(self, kargs=kargs):
                        dim = kargs.pop('dim')
                        dof = kargs['dof']
                        boundary = kargs['boundary_type']
                        stencil = kargs['stencil_type']
                        width = kargs['stencil_width']
                        da = PETSc.DA().create([8*SCALE]*dim)
                        newda = da.duplicate(**kargs)
                        self.assertEqual(newda.dim, da.dim)
                        self.assertEqual(newda.sizes, da.sizes)
                        self.assertEqual(newda.proc_sizes,
                                         da.proc_sizes)
                        self.assertEqual(newda.ranges, da.ranges)
                        self.assertEqual(newda.corners, da.corners)
                        if (newda.boundary_type == da.boundary_type
                            and
                            newda.stencil_width == da.stencil_width):
                            self.assertEqual(newda.ghost_ranges,
                                             da.ghost_ranges)
                            self.assertEqual(newda.ghost_corners,
                                             da.ghost_corners)
                        if dof is None:
                            dof = da.dof
                        if boundary is None:
                            boundary = da.boundary_type
                        elif boundary == "none":
                            boundary = (0,) * dim
                        elif boundary == "ghosted":
                            boundary = (GHOSTED,) * dim
                        elif boundary == "periodic":
                            boundary = (PERIODIC,) * dim
                        elif isinstance(boundary, int):
                            boundary = (boundary,) * dim
                        if stencil is None:
                            stencil = da.stencil[0]
                        if width is None:
                            width = da.stencil_width
                        self.assertEqual(newda.dof, dof)
                        self.assertEqual(newda.boundary_type,
                                         boundary)
                        if dim == 1:
                            if PETSc.Sys.getVersion() > (3,0,0):
                                self.assertEqual(newda.stencil,
                                                 (stencil, width))
                        newda.destroy()
                        da.destroy()
                    setattr(TestDADuplicate,
                            "testDuplicate%04d"%counter,
                            testDuplicate)
                    del testDuplicate, kargs
                    counter += 1
del counter, dim, dof, boundary, stencil, width

# --------------------------------------------------------------------

if PETSc.Sys.getVersion() <= (3,0,0):
    del BaseTestDA.testRefineHierarchy
    del BaseTestDA.testCoarsenHierarchy

if PETSc.COMM_WORLD.getSize() > 1:
    del TestDA_1D_W0
    del TestDA_2D_W0, TestDA_2D_W0_N2
    del TestDA_3D_W0, TestDA_3D_W0_N2

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
