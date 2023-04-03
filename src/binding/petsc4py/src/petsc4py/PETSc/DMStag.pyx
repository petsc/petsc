# --------------------------------------------------------------------

class DMStagStencilType(object):
    STAR = DMSTAG_STENCIL_STAR
    BOX  = DMSTAG_STENCIL_BOX
    NONE = DMSTAG_STENCIL_NONE

class DMStagStencilLocation(object):
    NULLLOC          = DMSTAG_NULL_LOCATION
    BACK_DOWN_LEFT   = DMSTAG_BACK_DOWN_LEFT
    BACK_DOWN        = DMSTAG_BACK_DOWN
    BACK_DOWN_RIGHT  = DMSTAG_BACK_DOWN_RIGHT
    BACK_LEFT        = DMSTAG_BACK_LEFT
    BACK             = DMSTAG_BACK
    BACK_RIGHT       = DMSTAG_BACK_RIGHT
    BACK_UP_LEFT     = DMSTAG_BACK_UP_LEFT
    BACK_UP          = DMSTAG_BACK_UP
    BACK_UP_RIGHT    = DMSTAG_BACK_UP_RIGHT
    DOWN_LEFT        = DMSTAG_DOWN_LEFT
    DOWN             = DMSTAG_DOWN
    DOWN_RIGHT       = DMSTAG_DOWN_RIGHT
    LEFT             = DMSTAG_LEFT
    ELEMENT          = DMSTAG_ELEMENT
    RIGHT            = DMSTAG_RIGHT
    UP_LEFT          = DMSTAG_UP_LEFT
    UP               = DMSTAG_UP
    UP_RIGHT         = DMSTAG_UP_RIGHT
    FRONT_DOWN_LEFT  = DMSTAG_FRONT_DOWN_LEFT
    FRONT_DOWN       = DMSTAG_FRONT_DOWN
    FRONT_DOWN_RIGHT = DMSTAG_FRONT_DOWN_RIGHT
    FRONT_LEFT       = DMSTAG_FRONT_LEFT
    FRONT            = DMSTAG_FRONT
    FRONT_RIGHT      = DMSTAG_FRONT_RIGHT
    FRONT_UP_LEFT    = DMSTAG_FRONT_UP_LEFT
    FRONT_UP         = DMSTAG_FRONT_UP
    FRONT_UP_RIGHT   = DMSTAG_FRONT_UP_RIGHT

# --------------------------------------------------------------------

cdef class DMStag(DM):

    StencilType       = DMStagStencilType
    StencilLocation   = DMStagStencilLocation

    def create(self, dim, dofs=None, sizes=None, boundary_types=None, stencil_type=None, stencil_width=None, proc_sizes=None, ownership_ranges=None, comm=None, setUp=False):
        
        # ndim
        cdef PetscInt ndim = asInt(dim)
        
        # sizes
        cdef object gsizes = sizes
        cdef PetscInt nsizes=PETSC_DECIDE, M=1, N=1, P=1
        if sizes is not None:
            nsizes = asStagDims(gsizes, &M, &N, &P)
            assert(nsizes==ndim)
           
        # dofs
        cdef object cdofs = dofs
        cdef PetscInt ndofs=PETSC_DECIDE, dof0=1, dof1=0, dof2=0, dof3=0
        if dofs is not None:
            ndofs = asDofs(cdofs, &dof0, &dof1, &dof2, &dof3)
            assert(ndofs==ndim+1)

        # boundary types
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        asBoundary(boundary_types, &btx, &bty, &btz)
        
        # stencil
        cdef PetscInt swidth = 0
        if stencil_width is not None:
            swidth = asInt(stencil_width)
        cdef PetscDMStagStencilType stype = DMSTAG_STENCIL_NONE
        if stencil_type is not None:
            stype = asStagStencil(stencil_type)

        # comm
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)

        # proc sizes
        cdef object psizes = proc_sizes
        cdef PetscInt nprocs=PETSC_DECIDE, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        if proc_sizes is not None:
            nprocs = asStagDims(psizes, &m, &n, &p)
            assert(nprocs==ndim)

        # ownership ranges
        cdef PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        if ownership_ranges is not None:
            nranges = asStagOwnershipRanges(ownership_ranges, ndim, &m, &n, &p, &lx, &ly, &lz)       
            assert(nranges==ndim)
            
        # create
        cdef PetscDM newda = NULL
        if dim == 1:
            CHKERR( DMStagCreate1d(ccomm, btx, M, dof0, dof1, stype, swidth, lx, &newda) )
        if dim == 2:
            CHKERR( DMStagCreate2d(ccomm, btx, bty, M, N, m, n, dof0, dof1, dof2, stype, swidth, lx, ly, &newda) )
        if dim == 3:
            CHKERR( DMStagCreate3d(ccomm, btx, bty, btz, M, N, P, m, n, p, dof0, dof1, dof2, dof3, stype, swidth, lx, ly, lz, &newda) )
        PetscCLEAR(self.obj); self.dm = newda
        if setUp:
            CHKERR( DMSetUp(self.dm) )
        return self

    # Setters

    def setStencilWidth(self,swidth):
        cdef PetscInt sw = asInt(swidth)
        CHKERR( DMStagSetStencilWidth(self.dm, sw) )

    def setStencilType(self, stenciltype):
        cdef PetscDMStagStencilType stype = asStagStencil(stenciltype)
        CHKERR( DMStagSetStencilType(self.dm, stype) )

    def setBoundaryTypes(self, boundary_types):
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        asBoundary(boundary_types, &btx, &bty, &btz)
        CHKERR( DMStagSetBoundaryTypes(self.dm, btx, bty, btz) )    
        
    def setDof(self, dofs):
        cdef tuple gdofs = tuple(dofs)
        cdef PetscInt gdim=PETSC_DECIDE, dof0=1, dof1=0, dof2=0, dof3=0
        gdim = asDofs(gdofs, &dof0, &dof1, &dof2, &dof3)
        CHKERR( DMStagSetDOF(self.dm, dof0, dof1, dof2, dof3) )
        
    def setGlobalSizes(self, sizes):
        cdef tuple gsizes = tuple(sizes)
        cdef PetscInt gdim=PETSC_DECIDE, M=1, N=1, P=1
        gdim = asStagDims(gsizes, &M, &N, &P)
        CHKERR( DMStagSetGlobalSizes(self.dm, M, N, P) )
        
    def setProcSizes(self, sizes):
        cdef tuple psizes = tuple(sizes)
        cdef PetscInt pdim=PETSC_DECIDE, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        pdim = asStagDims(psizes, &m, &n, &p)
        CHKERR( DMStagSetNumRanks(self.dm, m, n, p) )

    def setOwnershipRanges(self, ranges):
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        cdef PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetNumRanks(self.dm, &m, &n, &p) )
        ownership_ranges = asStagOwnershipRanges(ranges, dim, &m, &n, &p, &lx, &ly, &lz)
        CHKERR( DMStagSetOwnershipRanges(self.dm, lx, ly, lz) )

    # Getters

    def getDim(self):
        return self.getDimension()
        
    def getEntriesPerElement(self):
        cdef PetscInt epe=0
        CHKERR( DMStagGetEntriesPerElement(self.dm, &epe) )
        return toInt(epe)
    
    def getStencilWidth(self):
        cdef PetscInt swidth=0
        CHKERR( DMStagGetStencilWidth(self.dm, &swidth) )
        return toInt(swidth)

    def getDof(self):
        cdef PetscInt dim=0, dof0=0, dof1=0, dof2=0, dof3=0
        CHKERR( DMStagGetDOF(self.dm, &dof0, &dof1, &dof2, &dof3) )
        CHKERR( DMGetDimension(self.dm, &dim) )
        return toDofs(dim+1,dof0,dof1,dof2,dof3)

    def getCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0, nExtrax=0, nExtray=0, nExtraz=0
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetCorners(self.dm, &x, &y, &z, &m, &n, &p, &nExtrax, &nExtray, &nExtraz) )
        return (asInt(x), asInt(y), asInt(z))[:<Py_ssize_t>dim], (asInt(m), asInt(n), asInt(p))[:<Py_ssize_t>dim], (asInt(nExtrax), asInt(nExtray), asInt(nExtraz))[:<Py_ssize_t>dim]

    def getGhostCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetGhostCorners(self.dm, &x, &y, &z, &m, &n, &p) )
        return (asInt(x), asInt(y), asInt(z))[:<Py_ssize_t>dim], (asInt(m), asInt(n), asInt(p))[:<Py_ssize_t>dim]

    def getLocalSizes(self):
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetLocalSizes(self.dm, &m, &n, &p) )
        return toStagDims(dim, m, n, p)

    def getGlobalSizes(self):
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetGlobalSizes(self.dm, &m, &n, &p) )
        return toStagDims(dim, m, n, p)
        
    def getProcSizes(self):
        cdef PetscInt dim=0, m=PETSC_DECIDE, n=PETSC_DECIDE, p=PETSC_DECIDE
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetNumRanks(self.dm, &m, &n, &p) )
        return toStagDims(dim, m, n, p)
        
    def getStencilType(self):
        cdef PetscDMStagStencilType stype = DMSTAG_STENCIL_BOX
        CHKERR( DMStagGetStencilType(self.dm, &stype) )
        return toStagStencil(stype)

    def getOwnershipRanges(self):
        cdef PetscInt dim=0, m=0, n=0, p=0
        cdef const PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetNumRanks(self.dm, &m, &n, &p) )
        CHKERR( DMStagGetOwnershipRanges(self.dm, &lx, &ly, &lz) )
        return toStagOwnershipRanges(dim, m, n, p, lx, ly, lz)

    def getBoundaryTypes(self):
        cdef PetscInt dim=0
        cdef PetscDMBoundaryType btx = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType bty = DM_BOUNDARY_NONE
        cdef PetscDMBoundaryType btz = DM_BOUNDARY_NONE
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetBoundaryTypes(self.dm, &btx, &bty, &btz) )
        return toStagBoundaryTypes(dim, btx, bty, btz)

    def getIsFirstRank(self):
        cdef PetscBool rank0=PETSC_FALSE, rank1=PETSC_FALSE, rank2=PETSC_FALSE
        cdef PetscInt dim=0
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetIsFirstRank(self.dm, &rank0, &rank1, &rank2) )
        return toStagDims(dim, rank0, rank1, rank2)
        
    def getIsLastRank(self):
        cdef PetscBool rank0=PETSC_FALSE, rank1=PETSC_FALSE, rank2=PETSC_FALSE
        cdef PetscInt dim=0
        CHKERR( DMGetDimension(self.dm, &dim) )
        CHKERR( DMStagGetIsLastRank(self.dm, &rank0, &rank1, &rank2) )
        return toStagDims(dim, rank0, rank1, rank2)

    # Coordinate-related functions

    def setUniformCoordinatesExplicit(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1):
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR( DMStagSetUniformCoordinatesExplicit(self.dm, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax) )        
        
    def setUniformCoordinatesProduct(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1):
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR( DMStagSetUniformCoordinatesProduct(self.dm, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax) )        
        
    def setUniformCoordinates(self, xmin=0, xmax=1, ymin=0, ymax=1, zmin=0, zmax=1):
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR( DMStagSetUniformCoordinates(self.dm, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax) )        

    def setCoordinateDMType(self, dmtype):
        cdef PetscDMType cval = NULL
        dmtype = str2bytes(dmtype, &cval)
        CHKERR( DMStagSetCoordinateDMType(self.dm, cval) )

    # Location slot related functions

    def getLocationSlot(self, loc, c):
        cdef PetscInt slot=0
        cdef PetscInt comp=asInt(c)
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        CHKERR( DMStagGetLocationSlot(self.dm, sloc, comp, &slot) ) 
        return toInt(slot)

    def getProductCoordinateLocationSlot(self, loc):
        cdef PetscInt slot=0
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        CHKERR( DMStagGetProductCoordinateLocationSlot(self.dm, sloc, &slot) )
        return toInt(slot)
        
    def getLocationDof(self, loc):
        cdef PetscInt dof=0
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        CHKERR( DMStagGetLocationDOF(self.dm, sloc, &dof) ) 
        return toInt(dof)

    # Random other functions

    def migrateVec(self, Vec vec, DM dmTo, Vec vecTo):
        CHKERR( DMStagMigrateVec(self.dm, vec.vec, dmTo.dm, vecTo.vec ) )
        
    def createCompatibleDMStag(self, dofs):
        cdef tuple gdofs = tuple(dofs)
        cdef PetscInt gdim=PETSC_DECIDE, dof0=1, dof1=0, dof2=0, dof3=0
        gdim = asDofs(gdofs, &dof0, &dof1, &dof2, &dof3)
        cdef PetscDM newda = NULL
        CHKERR( DMStagCreateCompatibleDMStag(self.dm, dof0, dof1, dof2, dof3, &newda) )
        cdef DM newdm = type(self)()
        PetscCLEAR(newdm.obj); newdm.dm = newda
        return newdm
        
    def VecSplitToDMDA(self, Vec vec, loc, c):
        cdef PetscInt pc = asInt(c)
        cdef PetscDMStagStencilLocation sloc = asStagStencilLocation(loc)
        cdef PetscDM pda = NULL
        cdef PetscVec pdavec = NULL
        CHKERR( DMStagVecSplitToDMDA(self.dm, vec.vec, sloc, pc, &pda, &pdavec) )
        cdef DM da = DMDA()
        PetscCLEAR(da.obj); da.dm = pda
        cdef Vec davec = Vec()
        PetscCLEAR(davec.obj); davec.vec = pdavec
        return (da,davec)

    def getVecArray(self, Vec vec):
        raise NotImplementedError('getVecArray for DMStag not yet implemented in petsc4py')

    def get1dCoordinatecArrays(self):
        raise NotImplementedError('get1dCoordinatecArrays for DMStag not yet implemented in petsc4py')

    property dim:
        def __get__(self):
            return self.getDim()

    property dofs:
        def __get__(self):
            return self.getDof()
    
    property entries_per_element:
        def __get__(self):
            return self.getEntriesPerElement()
            
    property global_sizes:
        def __get__(self):
            return self.getGlobalSizes()
            
    property local_sizes:
        def __get__(self):
            return self.getLocalSizes()

    property proc_sizes:
        def __get__(self):
            return self.getProcSizes()

    property boundary_types:
        def __get__(self):
            return self.getBoundaryTypes()

    property stencil_type:
        def __get__(self):
            return self.getStencilType()

    property stencil_width:
        def __get__(self):
            return self.getStencilWidth()

    property corners:
        def __get__(self):
            return self.getCorners()

    property ghost_corners:
        def __get__(self):
            return self.getGhostCorners()


# --------------------------------------------------------------------

del DMStagStencilType
del DMStagStencilLocation

# --------------------------------------------------------------------
