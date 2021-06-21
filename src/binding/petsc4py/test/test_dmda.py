from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class BaseTestDA(object):

    COMM = PETSc.COMM_WORLD
    SIZES = None
    BOUNDARY = None
    DOF = 1
    STENCIL = PETSc.DMDA.StencilType.STAR
    SWIDTH = 1

    def setUp(self):
        self.da = PETSc.DMDA().create(dim=len(self.SIZES),
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

    def testOwnershipRanges(self):
        dim = self.da.getDim()
        ownership_ranges = self.da.getOwnershipRanges()
        procsizes = self.da.getProcSizes()
        self.assertEqual(len(procsizes), len(ownership_ranges))
        for i,m in enumerate(procsizes):
            self.assertEqual(m, len(ownership_ranges[i]))

    def testFieldName(self):
        for i in range(self.da.getDof()):
            self.da.setFieldName(i, "field%d" % i)
        for i in range(self.da.getDof()):
            name = self.da.getFieldName(i)
            self.assertEqual(name, "field%d" % i)

    def testCoordinates(self):
        self.da.setUniformCoordinates(0,1,0,1,0,1)
        #
        c = self.da.getCoordinates()
        self.da.setCoordinates(c)
        c.destroy()
        cda = self.da.getCoordinateDM()
        cda.destroy()
        #
        c = self.da.getCoordinates()
        self.da.setCoordinates(c)
        c.destroy()
        gc = self.da.getCoordinatesLocal()
        gc.destroy()

    def testCreateVecMat(self):
        vn = self.da.createNaturalVec()
        vg = self.da.createGlobalVec()
        vl = self.da.createLocalVec()
        mat = self.da.createMat()
        self.assertTrue(mat.getType() in ('aij', 'seqaij', 'mpiaij'))
        vn.set(1.0)
        self.da.naturalToGlobal(vn,vg)
        self.assertEqual(vg.max()[1], 1.0)
        self.assertEqual(vg.min()[1], 1.0)
        self.da.globalToLocal(vg,vl)
        self.assertEqual(vl.max()[1], 1.0)
        self.assertTrue (vl.min()[1] in (1.0, 0.0))
        vn.set(0.0)
        self.da.globalToNatural(vg,vn)
        self.assertEqual(vn.max()[1], 1.0)
        self.assertEqual(vn.min()[1], 1.0)
        vl2 = self.da.createLocalVec()
        self.da.localToLocal(vl,vl2)
        self.assertEqual(vl2.max()[1], 1.0)
        self.assertTrue (vl2.min()[1] in (1.0, 0.0))
        NONE = PETSc.DM.BoundaryType.NONE
        s = self.da.stencil_width
        btype = self.da.boundary_type
        psize = self.da.proc_sizes
        for b, p in zip(btype, psize):
            if b != NONE and p == 1: return
        vg2 = self.da.createGlobalVec()
        self.da.localToGlobal(vl2,vg2)

    def testGetVec(self):
        vg = self.da.getGlobalVec()
        vl = self.da.getLocalVec()
        try:
            vg.set(1.0)
            self.assertEqual(vg.max()[1], 1.0)
            self.assertEqual(vg.min()[1], 1.0)
            self.da.globalToLocal(vg,vl)
            self.assertEqual(vl.max()[1], 1.0)
            self.assertTrue (vl.min()[1] in (1.0, 0.0))
            vl.set(2.0)
            NONE = PETSc.DM.BoundaryType.NONE
            s = self.da.stencil_width
            btype = self.da.boundary_type
            psize = self.da.proc_sizes
            for b, p in zip(btype, psize):
                if b != NONE and p == 1: return
            self.da.localToGlobal(vl,vg)
            self.assertEqual(vg.max()[1], 2.0)
            self.assertTrue (vg.min()[1] in (2.0, 0.0))
        finally:
            self.da.restoreGlobalVec(vg)
            self.da.restoreLocalVec(vl)

    def testGetOther(self):
        ao = self.da.getAO()
        lgmap = self.da.getLGMap()
        l2g, g2l = self.da.getScatter()

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

    def testCreateInterpolation(self):
        da = self.da
        if da.dim == 1: return
        rda = da.refine()
        mat, vec = da.createInterpolation(rda)

    def testCreateInjection(self):
        da = self.da
        if da.dim == 1: return
        rda = da.refine()
        scatter = da.createInjection(rda)

    def testzeroRowsColumnsStencil(self):
        da = self.da
        A = da.createMatrix()
        x = da.createGlobalVector()
        x.set(2.0)
        A.setDiagonal(x)
        diag1 = x.duplicate()
        A.getDiagonal(diag1)
        if self.SIZES != 2: #only coded test for 2D case
        	return
        istart,iend, jstart, jend = da.getRanges()
        self.assertTrue(x.equal(diag1))
        zeroidx = []
        for i in range(istart,iend):
            for j in range(jstart,jend):
                row = PETSc.Mat.Stencil()
                row.index = (i,j)
                zeroidx = zeroidx + [row]
        diag2 = x.duplicate()
        diag2.set(1.0)
        A.zeroRowsColumnsStencil(zeroidx, 1.0, x, diag2)
        ans = x.duplicate()
        ans.set(2.0)
        self.assertTrue(ans.equal(diag2))


MIRROR   = PETSc.DMDA.BoundaryType.MIRROR
GHOSTED  = PETSc.DMDA.BoundaryType.GHOSTED
PERIODIC = PETSc.DMDA.BoundaryType.PERIODIC
TWIST    = PETSc.DMDA.BoundaryType.TWIST

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
    SIZES = [13*SCALE,17*SCALE]
    DOF = 2
    SWIDTH = 5
    BOUNDARY = (PERIODIC,)*2
class TestDA_2D_GXY(TestDA_2D):
    SIZES = [13*SCALE,17*SCALE]
    DOF = 2
    SWIDTH = 5
    BOUNDARY = (GHOSTED,)*2
class TestDA_2D_TXY(TestDA_2D):
    SIZES = [13*SCALE,17*SCALE]
    DOF = 2
    SWIDTH = 5
    BOUNDARY = (TWIST,)*2

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
    SIZES = [11*SCALE,13*SCALE,17*SCALE]
    DOF = 2
    SWIDTH = 3
    BOUNDARY = (PERIODIC,)*3
class TestDA_3D_GXYZ(TestDA_3D):
    SIZES = [11*SCALE,13*SCALE,17*SCALE]
    DOF = 2
    SWIDTH = 3
    BOUNDARY = (GHOSTED,)*3
class TestDA_3D_TXYZ(TestDA_3D):
    SIZES = [11*SCALE,13*SCALE,17*SCALE]
    DOF = 2
    SWIDTH = 3
    BOUNDARY = (TWIST,)*3

# --------------------------------------------------------------------

DIM = (1,2,3,)
DOF = (None,1,2,3,4,5,)
BOUNDARY_TYPE = (
    None,
    "none",     (0,)*3,        0,
    "ghosted",  (GHOSTED,)*3,  GHOSTED,
    "periodic", (PERIODIC,)*3, PERIODIC,
    "twist",    (TWIST,)*3,    TWIST,
    )
STENCIL_TYPE  = (None,"star","box")
STENCIL_WIDTH = (None,0,1,2,3)


DIM           = (1,2,3)
DOF           = (None,2,5)
BOUNDARY_TYPE = (None,"none","periodic","ghosted","twist")
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
                        kargs = dict(kargs)
                        da = PETSc.DMDA().create(**kargs)
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
                        kargs = dict(kargs)
                        dim = kargs.pop('dim')
                        dof = kargs['dof']
                        boundary = kargs['boundary_type']
                        stencil = kargs['stencil_type']
                        width = kargs['stencil_width']
                        da = PETSc.DMDA().create([8*SCALE]*dim)
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
                        elif boundary == "mirror":
                            boundary = (MIRROR,) * dim
                        elif boundary == "ghosted":
                            boundary = (GHOSTED,) * dim
                        elif boundary == "periodic":
                            boundary = (PERIODIC,) * dim
                        elif boundary == "twist":
                            boundary = (TWIST,) * dim
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

if PETSc.COMM_WORLD.getSize() > 1:
    del TestDA_1D_W0
    del TestDA_2D_W0, TestDA_2D_W0_N2
    del TestDA_3D_W0, TestDA_3D_W0_N2

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
