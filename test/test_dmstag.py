from petsc4py import PETSc
import unittest

# --------------------------------------------------------------------

class BaseTestDMStag(object):

    COMM = PETSc.COMM_WORLD
    STENCIL = PETSc.DMStag.StencilType.BOX
    SWIDTH = 1
    PROC_SIZES = None
    OWNERSHIP_RANGES = None

    def setUp(self):
        dim = len(self.SIZES)
        self.da = PETSc.DMStag().create(dim,
                                        dofs=self.DOFS, sizes=self.SIZES, boundary_types=self.BOUNDARY,
                                        stencil_type=self.STENCIL, stencil_width=self.SWIDTH,
                                        comm=self.COMM, proc_sizes=self.PROC_SIZES, ownership_ranges=self.OWNERSHIP_RANGES, setUp=True)
        
        self.directda = PETSc.DMStag().create(dim)
        self.directda.setStencilType(self.STENCIL)
        self.directda.setStencilWidth(self.SWIDTH)
        self.directda.setBoundaryTypes(self.BOUNDARY)
        self.directda.setDof(self.DOFS)
        self.directda.setGlobalSizes(self.SIZES)
        if self.PROC_SIZES is not None:
            self.directda.setProcSizes(self.PROC_SIZES)
        if self.OWNERSHIP_RANGES is not None:
            self.directda.setOwnershipRanges(self.OWNERSHIP_RANGES)
        self.directda.setUp()
                                    
    def tearDown(self):
        self.da = None
        self.directda = None

    def testCoordinates(self):
        self.da.setCoordinateDMType('stag')
        self.da.setUniformCoordinates(0,1,0,1,0,1)
        self.da.setUniformCoordinatesExplicit(0,1,0,1,0,1)
        cda = self.da.getCoordinateDM()
        datype = cda.getType()
        self.assertEqual(datype,'stag')
        cda.destroy()
        
        c = self.da.getCoordinatesLocal()
        self.da.setCoordinatesLocal(c)
        gc = self.da.getCoordinatesLocal()
        self.assertEqual(c.max()[1], gc.max()[1])
        self.assertEqual(c.min()[1], gc.min()[1])

        c = self.da.getCoordinates()
        self.da.setCoordinates(c)
        gc = self.da.getCoordinates()
        self.assertEqual(c.max()[1], gc.max()[1])
        self.assertEqual(c.min()[1], gc.min()[1])
        
        self.directda.setCoordinateDMType('product')
        self.directda.setUniformCoordinates(0,1,0,1,0,1)
        self.directda.setUniformCoordinatesProduct(0,1,0,1,0,1)
        cda = self.directda.getCoordinateDM()
        datype = cda.getType()
        self.assertEqual(datype,'product')
        cda.destroy()

    def testGetVec(self):
        vg = self.da.getGlobalVec()
        vl = self.da.getLocalVec()
        
        vg.set(1.0)
        self.assertEqual(vg.max()[1], 1.0)
        self.assertEqual(vg.min()[1], 1.0)
        self.da.globalToLocal(vg,vl)
        self.assertEqual(vl.max()[1], 1.0)
        self.assertTrue (vl.min()[1] in (1.0, 0.0))
        
        vl.set(2.0)
        self.da.localToGlobal(vl,vg)
        self.assertEqual(vg.max()[1], 2.0)
        self.assertTrue (vg.min()[1] in (2.0, 0.0))
        
        self.da.restoreGlobalVec(vg)
        self.da.restoreLocalVec(vl)

    def testGetOther(self):
        lgmap = self.da.getLGMap()
        dlgmap = self.directda.getLGMap()

    def testDof(self):
        dim = self.da.getDim()
        dofs = self.da.getDof()
        if dim == 1:
            dof0 = self.da.getLocationDof('left')
            dof1 = self.da.getLocationDof('element')
            self.assertEqual(dofs[0],dof0)
            self.assertEqual(dofs[1],dof1)
        if dim == 2:
            dof0 = self.da.getLocationDof('down_left')
            dof1 = self.da.getLocationDof('left')
            dof2 = self.da.getLocationDof('element')
            self.assertEqual(dofs[0],dof0)
            self.assertEqual(dofs[1],dof1)
            self.assertEqual(dofs[2],dof2)
        if dim == 3:
            dof0 = self.da.getLocationDof('back_down_right')
            dof1 = self.da.getLocationDof('down_left')
            dof2 = self.da.getLocationDof('left')
            dof3 = self.da.getLocationDof('element')
            self.assertEqual(dofs[0],dof0)
            self.assertEqual(dofs[1],dof1)
            self.assertEqual(dofs[2],dof2)
            self.assertEqual(dofs[3],dof3)

    def testMigrateVec(self):
        vec = self.da.createGlobalVec()
        dmTo = self.da.createCompatibleDMStag(self.NEWDOF)
        vecTo = dmTo.createGlobalVec()
        self.da.migrateVec(vec, dmTo, vecTo)

    def testDMDAInterface(self):
        return
        self.da.setCoordinateDMType('stag')
        self.da.setUniformCoordinates(0,1,0,1,0,1)
        dim = self.da.getDim()
        dofs = self.da.getDof()
        vec = self.da.createGlobalVec()
        if dim == 1:
            da,davec = self.da.VecSplitToDMDA(vec,'left',-dofs[0])
            da,davec = self.da.VecSplitToDMDA(vec,'element',-dofs[1])
        if dim == 2:
            da,davec = self.da.VecSplitToDMDA(vec,'down_left',-dofs[0])
            da,davec = self.da.VecSplitToDMDA(vec,'down_left',-dofs[1])
            da,davec = self.da.VecSplitToDMDA(vec,'down_left',-dofs[2])
        if dim == 3:
            da,davec = self.da.VecSplitToDMDA(vec,'back_down_right',-dofs[0])
            da,davec = self.da.VecSplitToDMDA(vec,'down_left',-dofs[1])
            da,davec = self.da.VecSplitToDMDA(vec,'left',-dofs[2])
            da,davec = self.da.VecSplitToDMDA(vec,'element',-dofs[3])

GHOSTED  = PETSc.DM.BoundaryType.GHOSTED
PERIODIC = PETSc.DM.BoundaryType.PERIODIC
NONE = PETSc.DM.BoundaryType.NONE

SCALE = 4

class BaseTestDMStag_1D(BaseTestDMStag):
    SIZES = [100*SCALE,]
    BOUNDARY = [NONE,]

class BaseTestDMStag_2D(BaseTestDMStag):
    SIZES = [9*SCALE, 11*SCALE]
    BOUNDARY = [NONE, NONE]

class BaseTestDMStag_3D(BaseTestDMStag):
    SIZES = [6*SCALE, 7*SCALE, 8*SCALE]
    BOUNDARY = [NONE, NONE, NONE]

# --------------------------------------------------------------------

class TestDMStag_1D_W0_N11(BaseTestDMStag_1D, unittest.TestCase):
    SWIDTH = 0
    DOFS = (1,1)
    NEWDOF = (2,1)
class TestDMStag_1D_W0_N21(BaseTestDMStag_1D, unittest.TestCase):
    SWIDTH = 0
    DOFS = (2,1)
    NEWDOF = (2,2)
class TestDMStag_1D_W0_N12(BaseTestDMStag_1D, unittest.TestCase):
    SWIDTH = 0
    DOFS = (1,2)
    NEWDOF = (2,2)
class TestDMStag_1D_W2_N11(BaseTestDMStag_1D, unittest.TestCase):
    SWIDTH = 2
    DOFS = (1,1)
    NEWDOF = (2,1)
class TestDMStag_1D_W2_N21(BaseTestDMStag_1D, unittest.TestCase):
    SWIDTH = 2
    DOFS = (2,1)
    NEWDOF = (2,2)
class TestDMStag_1D_W2_N12(BaseTestDMStag_1D, unittest.TestCase):
    SWIDTH = 2
    DOFS = (1,2)
    NEWDOF = (2,2)

class TestDMStag_2D_W0_N112(BaseTestDMStag_2D, unittest.TestCase):
    DOFS = (1,1,2)
    SWIDTH = 0
    NEWDOF = (2,2,2)
class TestDMStag_2D_W2_N112(BaseTestDMStag_2D, unittest.TestCase):
    DOFS = (1,1,2)
    SWIDTH = 2
    NEWDOF = (2,2,2)
class TestDMStag_2D_PXY(BaseTestDMStag_2D, unittest.TestCase):
    SIZES = [13*SCALE,17*SCALE]
    DOFS = (1,1,2)
    SWIDTH = 5
    BOUNDARY = (PERIODIC,)*2
    NEWDOF = (2,2,2)
class TestDMStag_2D_GXY(BaseTestDMStag_2D, unittest.TestCase):
    SIZES = [13*SCALE,17*SCALE]
    DOFS = (1,1,2)
    SWIDTH = 5
    BOUNDARY = (GHOSTED,)*2
    NEWDOF = (2,2,2)

class TestDMStag_3D_W0_N1123(BaseTestDMStag_3D, unittest.TestCase):
    DOFS = (1,1,2,3)
    SWIDTH = 0
    NEWDOF = (2,2,3,3)
class TestDMStag_3D_W2_N1123(BaseTestDMStag_3D, unittest.TestCase):
    DOFS = (1,1,2,3)
    SWIDTH = 2
    NEWDOF = (2,2,3,3)
class TestDMStag_3D_PXYZ(BaseTestDMStag_3D, unittest.TestCase):
    SIZES = [11*SCALE,13*SCALE,17*SCALE]
    DOFS = (1,1,2,3)
    NEWDOF = (2,2,3,3)
    SWIDTH = 3
    BOUNDARY = (PERIODIC,)*3
class TestDMStag_3D_GXYZ(BaseTestDMStag_3D, unittest.TestCase):
    SIZES = [11*SCALE,13*SCALE,17*SCALE]
    DOFS = (1,1,2,3)
    NEWDOF = (2,2,3,3)
    SWIDTH = 3
    BOUNDARY = (GHOSTED,)*3

# --------------------------------------------------------------------

DIM = (1,2,3)
DOF0 = (0,1,2)
DOF1 = (0,1,2)
DOF2 = (0,1,2)
DOF3 = (0,1,2)
BOUNDARY_TYPE = ('none', 'ghosted', 'periodic')
STENCIL_TYPE  = ('none', 'star', 'box')
STENCIL_WIDTH = (0,1,2,3)

class TestDMStagCreate(unittest.TestCase):
    pass
counter = 0
for dim in DIM:
    for dof0 in DOF0:
        for dof1 in DOF1:
            for dof2 in DOF2:
                if dim == 1 and dof2 > 0: continue
                for dof3 in DOF3:
                    if dim == 2 and dof3 > 0: continue
                    if dof0==0 and dof1==0 and dof2==0 and dof3==0: continue
                    dofs = [dof0,dof1,dof2,dof3][:dim+1]
                    for boundary in BOUNDARY_TYPE:
                        if boundary == "periodic": continue  # XXX broken
                        for stencil in STENCIL_TYPE:
                            if stencil == 'none' and boundary != 'none': continue
                            for width in STENCIL_WIDTH:
                                if stencil == 'none' and width > 0: continue
                                if stencil in ['star','box'] and width == 0: continue
                                kargs = dict(dim=dim, dofs=dofs, boundary_type=boundary, 
                                stencil_type=stencil, stencil_width=width)

                                def testCreate(self,kargs=kargs):
                                    kargs = dict(kargs)
                                    cda = PETSc.DMStag().create(kargs['dim'],
                                    dofs = kargs['dofs'], 
                                    sizes = [8*SCALE,]*kargs['dim'], 
                                    boundary_types = [kargs['boundary_type'],]*kargs['dim'], 
                                    stencil_type = kargs['stencil_type'], 
                                    stencil_width = kargs['stencil_width'], 
                                    setUp=True)

                                    dda = PETSc.DMStag().create(kargs['dim'])
                                    dda.setStencilType(kargs['stencil_type'])
                                    dda.setStencilWidth(kargs['stencil_width'])
                                    dda.setBoundaryTypes([kargs['boundary_type'],]*kargs['dim'])
                                    dda.setDof(kargs['dofs'])
                                    dda.setGlobalSizes([8*SCALE,]*kargs['dim'])
                                    dda.setUp()

                                    cdim = cda.getDim()
                                    cdof = cda.getDof()
                                    cgsizes = cda.getGlobalSizes()
                                    clsizes = cda.getLocalSizes()
                                    cboundary = cda.getBoundaryTypes()
                                    cstencil_type = cda.getStencilType()
                                    cstencil_width = cda.getStencilWidth()
                                    centries_per_element = cda.getEntriesPerElement()
                                    cstarts, csizes, cnextra  = cda.getCorners()
                                    cisLastRank = cda.getIsLastRank()
                                    cisFirstRank = cda.getIsFirstRank()
                                    cownershipranges = cda.getOwnershipRanges()
                                    cprocsizes = cda.getProcSizes()
            
                                    ddim = dda.getDim()
                                    ddof = dda.getDof()
                                    dgsizes = dda.getGlobalSizes()
                                    dlsizes = dda.getLocalSizes()
                                    dboundary = dda.getBoundaryTypes()
                                    dstencil_type = dda.getStencilType()
                                    dstencil_width = dda.getStencilWidth()
                                    dentries_per_element = dda.getEntriesPerElement()
                                    dstarts, dsizes, dnextra  = dda.getCorners()
                                    disLastRank = dda.getIsLastRank()
                                    disFirstRank = dda.getIsFirstRank()
                                    downershipranges = dda.getOwnershipRanges()
                                    dprocsizes = dda.getProcSizes()            
                                    
                                    self.assertEqual(cdim,kargs['dim'])
                                    self.assertEqual(cdof,tuple(kargs['dofs']))
                                    self.assertEqual(cboundary,tuple([kargs['boundary_type'],]*kargs['dim']))
                                    self.assertEqual(cstencil_type,kargs['stencil_type'])
                                    self.assertEqual(cstencil_width,kargs['stencil_width'])
                                    self.assertEqual(cgsizes,tuple([8*SCALE,]*kargs['dim']))
                                    
                                    self.assertEqual(cdim,ddim)
                                    self.assertEqual(cdof,ddof)
                                    self.assertEqual(cgsizes,dgsizes)
                                    self.assertEqual(clsizes,dlsizes)
                                    self.assertEqual(cboundary,dboundary)
                                    self.assertEqual(cstencil_type,dstencil_type)
                                    self.assertEqual(cstencil_width,dstencil_width)
                                    self.assertEqual(centries_per_element,dentries_per_element)
                                    self.assertEqual(cstarts,dstarts)
                                    self.assertEqual(csizes,dsizes)
                                    self.assertEqual(cnextra,dnextra)
                                    self.assertEqual(cisLastRank,disLastRank)
                                    self.assertEqual(cisFirstRank,disFirstRank)
                                    self.assertEqual(cprocsizes, dprocsizes)
                                    for co,do in zip(cownershipranges, downershipranges):
                                        for i,j in zip(co,do):
                                            self.assertEqual(i,j)
            
                                    self.assertEqual(cdim+1,len(cdof))
                                    self.assertEqual(cdim,len(cgsizes))
                                    self.assertEqual(cdim,len(clsizes))
                                    self.assertEqual(cdim,len(cboundary))
                                    self.assertEqual(cdim,len(cstarts))
                                    self.assertEqual(cdim,len(csizes))
                                    self.assertEqual(cdim,len(cnextra))
                                    self.assertEqual(cdim,len(cisLastRank))
                                    self.assertEqual(cdim,len(cisLastRank))
                                    if cdim == 1: self.assertEqual(centries_per_element, cdof[0] + cdof[1])
                                    if cdim == 2: self.assertEqual(centries_per_element, cdof[0] + 2*cdof[1] + cdof[2])
                                    if cdim == 3: self.assertEqual(centries_per_element, cdof[0] + 3*cdof[1] + 3*cdof[2] + cdof[3])
                                    for i in range(cdim):
                                        self.assertEqual(csizes[i], clsizes[i])
                                        if cisLastRank[i]: self.assertEqual(cnextra[i],1)
                                        if (cnextra[i]==1): self.assertTrue(cisLastRank[i])
                                        if (cisFirstRank[i]): self.assertEqual(cstarts[i],0)
                                    self.assertEqual(len(cprocsizes), len(cownershipranges))
                                    self.assertEqual(len(cprocsizes), cdim)
                                    for i,m in enumerate(cprocsizes):
                                        self.assertEqual(m, len(cownershipranges[i]))    
                                    dda.destroy()
                                    cda.destroy()

                                setattr(TestDMStagCreate,
                                        "testCreate%05d"%counter,
                                        testCreate)

                                del testCreate
                                counter += 1

del counter, dim, dofs, dof0, dof1, dof2, dof3, boundary, stencil, width

# --------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
