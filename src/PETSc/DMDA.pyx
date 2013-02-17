# --------------------------------------------------------------------

class DMDABoundaryType(object):
    NONE     = DMDA_BOUNDARY_NONE
    GHOSTED  = DMDA_BOUNDARY_GHOSTED
    MIRROR   = DMDA_BOUNDARY_MIRROR
    PERIODIC = DMDA_BOUNDARY_PERIODIC

class DMDAStencilType(object):
    STAR = DMDA_STENCIL_STAR
    BOX  = DMDA_STENCIL_BOX

class DMDAInterpolationType(object):
    Q0 = DMDA_INTERPOLATION_Q0
    Q1 = DMDA_INTERPOLATION_Q1

class DMDAElementType(object):
    P1 = DMDA_ELEMENT_P1
    Q1 = DMDA_ELEMENT_Q1

# --------------------------------------------------------------------

cdef class DMDA(DM):

    BoundaryType      = DMDABoundaryType
    StencilType       = DMDAStencilType
    InterpolationType = DMDAInterpolationType
    ElementType       = DMDAElementType

    #

    def create(self, dim=None, dof=None,
               sizes=None, proc_sizes=None, boundary_type=None,
               stencil_type=None, stencil_width=None,
               bint setup=True, ownership_ranges=None, comm=None):
        #
        cdef object arg = None
        try: arg = tuple(dim)
        except TypeError: pass
        else: dim, sizes = None, arg
        #
        cdef PetscInt ndim = PETSC_DECIDE
        cdef PetscInt ndof = PETSC_DECIDE
        cdef PetscInt M = 1, m = PETSC_DECIDE, *lx = NULL
        cdef PetscInt N = 1, n = PETSC_DECIDE, *ly = NULL
        cdef PetscInt P = 1, p = PETSC_DECIDE, *lz = NULL
        cdef PetscDMDABoundaryType btx = DMDA_BOUNDARY_NONE
        cdef PetscDMDABoundaryType bty = DMDA_BOUNDARY_NONE
        cdef PetscDMDABoundaryType btz = DMDA_BOUNDARY_NONE
        cdef PetscDMDAStencilType  stype = DMDA_STENCIL_BOX
        cdef PetscInt            swidth = 0
        # grid and proc sizes
        cdef object gsizes = sizes
        cdef object psizes = proc_sizes
        cdef PetscInt gdim = PETSC_DECIDE
        cdef PetscInt pdim = PETSC_DECIDE
        if sizes is not None:
            gdim = asDims(gsizes, &M, &N, &P)
        if psizes is not None:
            pdim = asDims(psizes, &m, &n, &p)
        if gdim>=0 and pdim>=0:
            assert gdim == pdim
        # dim and dof
        if dim is not None: ndim = asInt(dim)
        if dof is not None: ndof = asInt(dof)
        if ndim==PETSC_DECIDE: ndim = gdim
        if ndof==PETSC_DECIDE: ndof = 1
        # vertex distribution
        if ownership_ranges is not None:
            ranges_mem = asOwnershipRanges(ownership_ranges, &lx, &ly, &lz)
        # periodicity, stencil type & width
        if boundary_type is not None:
            asBoundary(boundary_type, &btx, &bty, &btz)
        if stencil_type is not None:
            stype = asStencil(stencil_type)
        if stencil_width is not None:
            swidth = asInt(stencil_width)
        # create the DMDA object
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newda = NULL
        CHKERR( DMDACreateND(ccomm, ndim, ndof,
                             M, N, P, m, n, p, lx, ly, lz,
                             btx, bty, btz, stype, swidth,
                             &newda) )
        if setup and ndim > 0: CHKERR( DMSetUp(newda) )
        PetscCLEAR(self.obj); self.dm = newda
        return self

    def duplicate(self, dof=None, boundary_type=None,
                  stencil_type=None, stencil_width=None):
        cdef PetscInt ndim = 0, ndof = 0
        cdef PetscInt M = 1, N = 1, P = 1
        cdef PetscInt m = 1, n = 1, p = 1
        cdef PetscDMDABoundaryType btx = DMDA_BOUNDARY_NONE
        cdef PetscDMDABoundaryType bty = DMDA_BOUNDARY_NONE
        cdef PetscDMDABoundaryType btz = DMDA_BOUNDARY_NONE
        cdef PetscDMDAStencilType  stype = DMDA_STENCIL_STAR
        cdef PetscInt            swidth = 0
        CHKERR( DMDAGetInfo(self.dm,
                          &ndim,
                          &M, &N, &P,
                          &m, &n, &p,
                          &ndof, &swidth,
                          &btx, &bty, &btz,
                          &stype) )
        cdef const_PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        CHKERR( DMDAGetOwnershipRanges(self.dm, &lx, &ly, &lz) )
        cdef MPI_Comm comm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.dm, &comm) )
        #
        if dof is not None:
            ndof = asInt(dof)
        if boundary_type is not None:
            asBoundary(boundary_type, &btx, &bty, &btz)
        if stencil_type  is not None:
            stype = asStencil(stencil_type)
        if stencil_width is not None:
            swidth = asInt(stencil_width)
        #
        cdef DMDA da = DMDA()
        CHKERR( DMDACreateND(comm, ndim, ndof,
                             M, N, P, m, n, p, lx, ly, lz,
                             btx, bty, btz, stype, swidth,
                             &da.dm) )
        CHKERR( DMSetUp(da.dm) )
        return da

    #

    def setDim(self, dim):
        cdef PetscInt ndim = asInt(dim)
        CHKERR( DMDASetDim(self.dm, ndim) )
        
    def getDim(self):
        cdef PetscInt dim = 0
        CHKERR( DMDAGetInfo(self.dm,
                            &dim,
                            NULL, NULL, NULL,
                            NULL, NULL, NULL,
                            NULL, NULL,
                            NULL, NULL, NULL,
                            NULL) )
        return toInt(dim)

    def setDof(self, dof):
        cdef PetscInt ndof = asInt(dof)
        CHKERR( DMDASetDof(self.dm, ndof) )

    def getDof(self):
        cdef PetscInt dof = 0
        CHKERR( DMDAGetInfo(self.dm,
                            NULL,
                            NULL, NULL, NULL,
                            NULL, NULL, NULL,
                            &dof, NULL,
                            NULL, NULL, NULL,
                            NULL) )
        return toInt(dof)

    def setSizes(self, sizes):
        cdef tuple gsizes = tuple(sizes)
        cdef PetscInt gdim = PETSC_DECIDE
        cdef PetscInt M = 1
        cdef PetscInt N = 1
        cdef PetscInt P = 1
        gdim = asDims(gsizes, &M, &N, &P)
        cdef PetscInt dim = PETSC_DECIDE
        CHKERR( DMDAGetDim(self.dm, &dim) )
        if dim == PETSC_DECIDE:
            CHKERR( DMDASetDim(self.dm, gdim) )
        CHKERR( DMDASetSizes(self.dm, M, N, P) )

    def getSizes(self):
        cdef PetscInt dim = 0
        cdef PetscInt M = PETSC_DECIDE
        cdef PetscInt N = PETSC_DECIDE
        cdef PetscInt P = PETSC_DECIDE
        CHKERR( DMDAGetInfo(self.dm,
                            &dim,
                            &M, &N, &P,
                            NULL, NULL, NULL,
                            NULL, NULL,
                            NULL, NULL, NULL,
                            NULL) )
        return toDims(dim, M, N, P)

    def setProcSizes(self, proc_sizes):
        cdef tuple psizes = tuple(proc_sizes)
        cdef PetscInt pdim = PETSC_DECIDE
        cdef PetscInt m = PETSC_DECIDE
        cdef PetscInt n = PETSC_DECIDE
        cdef PetscInt p = PETSC_DECIDE
        pdim = asDims(psizes, &m, &n, &p)
        cdef PetscInt dim = PETSC_DECIDE
        CHKERR( DMDAGetDim(self.dm, &dim) )
        if dim == PETSC_DECIDE:
            CHKERR( DMDASetDim(self.dm, pdim) )
        CHKERR( DMDASetNumProcs(self.dm, m, n, p) )

    def getProcSizes(self):
        cdef PetscInt dim = 0
        cdef PetscInt m = PETSC_DECIDE
        cdef PetscInt n = PETSC_DECIDE
        cdef PetscInt p = PETSC_DECIDE
        CHKERR( DMDAGetInfo(self.dm,
                            &dim,
                            NULL, NULL, NULL,
                            &m, &n, &p,
                            NULL, NULL,
                            NULL, NULL, NULL,
                            NULL) )
        return toDims(dim, m, n, p)

    def setBoundaryType(self, boundary_type):
        cdef PetscDMDABoundaryType btx = DMDA_BOUNDARY_NONE
        cdef PetscDMDABoundaryType bty = DMDA_BOUNDARY_NONE
        cdef PetscDMDABoundaryType btz = DMDA_BOUNDARY_NONE
        asBoundary(boundary_type, &btx, &bty, &btz)
        CHKERR( DMDASetBoundaryType(self.dm, btx, bty, btz) )
        
    def getBoundaryType(self):
        cdef PetscInt dim = 0
        cdef PetscDMDABoundaryType btx = DMDA_BOUNDARY_NONE
        cdef PetscDMDABoundaryType bty = DMDA_BOUNDARY_NONE
        cdef PetscDMDABoundaryType btz = DMDA_BOUNDARY_NONE
        CHKERR( DMDAGetInfo(self.dm,
                          &dim,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          &btx, &bty, &btz,
                          NULL) )
        return toDims(dim, btx, bty, btz)

    def setStencilType(self, stencil_type):
        cdef PetscDMDAStencilType stype = asStencil(stencil_type)
        CHKERR( DMDASetStencilType(self.dm, stype) )
        
    def getStencilType(self):
        cdef PetscDMDAStencilType stype = DMDA_STENCIL_BOX
        CHKERR( DMDAGetInfo(self.dm,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, NULL, NULL,
                          &stype) )
        return stype

    def setStencilWidth(self, stencil_width):
        cdef PetscInt swidth = asInt(stencil_width)
        CHKERR( DMDASetStencilWidth(self.dm, swidth) )

    def getStencilWidth(self):
        cdef PetscInt swidth = 0
        CHKERR( DMDAGetInfo(self.dm,
                            NULL,
                            NULL, NULL, NULL,
                            NULL, NULL, NULL,
                            NULL, &swidth,
                            NULL, NULL, NULL,
                            NULL) )
        return toInt(swidth)

    def setStencil(self, stencil_type, stencil_width):
        cdef PetscDMDAStencilType stype = asStencil(stencil_type)
        cdef PetscInt swidth = asInt(stencil_width)
        CHKERR( DMDASetStencilType(self.dm, stype) )
        CHKERR( DMDASetStencilWidth(self.dm, swidth) )
        
    def getStencil(self):
        cdef PetscDMDAStencilType stype = DMDA_STENCIL_BOX
        cdef PetscInt swidth = 0
        CHKERR( DMDAGetInfo(self.dm,
                            NULL,
                            NULL, NULL, NULL,
                            NULL, NULL, NULL,
                            NULL, &swidth,
                            NULL, NULL, NULL,
                            &stype) )
        return (toStencil(stype), toInt(swidth))

    #

    def getRanges(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DMDAGetDim(self.dm, &dim) )
        CHKERR( DMDAGetCorners(self.dm,
                               &x, &y, &z,
                               &m, &n, &p) )
        return ((toInt(x), toInt(x+m)),
                (toInt(y), toInt(y+n)),
                (toInt(z), toInt(z+p)))[:<Py_ssize_t>dim]

    def getGhostRanges(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DMDAGetDim(self.dm, &dim) )
        CHKERR( DMDAGetGhostCorners(self.dm,
                                    &x, &y, &z,
                                    &m, &n, &p) )
        return ((toInt(x), toInt(x+m)),
                (toInt(y), toInt(y+n)),
                (toInt(z), toInt(z+p)))[:<Py_ssize_t>dim]

    def getOwnershipRanges(self):
        cdef PetscInt dim=0, m, n, p
        cdef const_PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        CHKERR( DMDAGetInfo(self.dm, &dim,
                            NULL, NULL, NULL,
                            &m, &n, &p,
                            NULL, NULL,
                            NULL, NULL, NULL,
                            NULL) )
        CHKERR( DMDAGetOwnershipRanges(self.dm, &lx, &ly, &lz) )
        return toOwnershipRanges(dim, m, n, p, lx, ly, lz)

    def getCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DMDAGetDim(self.dm, &dim) )
        CHKERR( DMDAGetCorners(self.dm,
                               &x, &y, &z,
                               &m, &n, &p) )
        return ((toInt(x), toInt(y), toInt(z))[:<Py_ssize_t>dim],
                (toInt(m), toInt(n), toInt(p))[:<Py_ssize_t>dim])

    def getGhostCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DMDAGetDim(self.dm, &dim) )
        CHKERR( DMDAGetGhostCorners(self.dm,
                                    &x, &y, &z,
                                    &m, &n, &p) )
        return ((toInt(x), toInt(y), toInt(z))[:<Py_ssize_t>dim],
                (toInt(m), toInt(n), toInt(p))[:<Py_ssize_t>dim])

    #

    def getVecArray(self, Vec vec not None):
        return _DMDA_Vec_array(self, vec)

    #

    def setUniformCoordinates(self,
                              xmin=0, xmax=1,
                              ymin=0, ymax=1,
                              zmin=0, zmax=1):
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR( DMDASetUniformCoordinates(self.dm,
                                          _xmin, _xmax,
                                          _ymin, _ymax,
                                          _zmin, _zmax) )

    def getBoundingBox(self):
        cdef PetscInt i,dim=0
        CHKERR( DMDAGetDim(self.dm, &dim) )
        cdef PetscReal gmin[3], gmax[3]
        CHKERR( DMDAGetBoundingBox(self.dm, gmin, gmax) )
        return tuple([(toReal(gmin[i]), toReal(gmax[i]))
                      for i from 0 <= i < dim])

    def getLocalBoundingBox(self):
        cdef PetscInt i,dim=0
        CHKERR( DMDAGetDim(self.dm, &dim) )
        cdef PetscReal lmin[3], lmax[3]
        CHKERR( DMDAGetLocalBoundingBox(self.dm, lmin, lmax) )
        return tuple([(toReal(lmin[i]), toReal(lmax[i]))
                      for i from 0 <= i < dim])

    #

    def createNaturalVec(self):
        cdef Vec vn = Vec()
        CHKERR( DMDACreateNaturalVector(self.dm, &vn.vec) )
        return vn

    def globalToNatural(self, Vec vg not None, Vec vn not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMDAGlobalToNaturalBegin(self.dm, vg.vec, im, vn.vec) )
        CHKERR( DMDAGlobalToNaturalEnd  (self.dm, vg.vec, im, vn.vec) )

    def naturalToGlobal(self, Vec vn not None, Vec vg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMDANaturalToGlobalBegin(self.dm, vn.vec, im, vg.vec) )
        CHKERR( DMDANaturalToGlobalEnd  (self.dm, vn.vec, im, vg.vec) )

    def localToLocal(self, Vec vl not None, Vec vlg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DMDALocalToLocalBegin(self.dm, vl.vec, im, vlg.vec) )
        CHKERR( DMDALocalToLocalEnd  (self.dm, vl.vec, im, vlg.vec) )

    #

    def getAO(self):
        cdef AO ao = AO()
        CHKERR( DMDAGetAO(self.dm, &ao.ao) )
        PetscINCREF(ao.obj)
        return ao

    def getScatter(self):
        cdef Scatter l2g = Scatter()
        cdef Scatter g2l = Scatter()
        cdef Scatter l2l = Scatter()
        CHKERR( DMDAGetScatter(self.dm, &l2g.sct, &g2l.sct, &l2l.sct) )
        PetscINCREF(l2g.obj)
        PetscINCREF(g2l.obj)
        PetscINCREF(l2l.obj)
        return (l2g, g2l, l2l)

    #

    def setRefinementFactor(self,
                            refine_x=2,
                            refine_y=2,
                            refine_z=2):
        cdef PetscInt refine[3]
        refine[0] = asInt(refine_x)
        refine[1] = asInt(refine_y)
        refine[2] = asInt(refine_z)
        CHKERR( DMDASetRefinementFactor(self.dm,
                                      refine[0],
                                      refine[1],
                                      refine[2]) )

    def getRefinementFactor(self):
        cdef PetscInt i, dim = 0, refine[3]
        CHKERR( DMDAGetDim(self.dm, &dim) )
        CHKERR( DMDAGetRefinementFactor(self.dm,
                                      &refine[0],
                                      &refine[1],
                                      &refine[2]) )
        return tuple([toInt(refine[i]) for 0 <= i < dim])

    def setInterpolationType(self, interp_type):
        cdef PetscDMDAInterpolationType ival = dainterpolationtype(interp_type)
        CHKERR( DMDASetInterpolationType(self.dm, ival) )

    def getInterpolationType(self):
        cdef PetscDMDAInterpolationType ival = DMDA_INTERPOLATION_Q0
        CHKERR( DMDAGetInterpolationType(self.dm, &ival) )
        return <long>ival

    #

    def setElementType(self, elem_type):
        cdef PetscDMDAElementType ival = daelementtype(elem_type)
        CHKERR( DMDASetElementType(self.dm, ival) )

    def getElementType(self):
        cdef PetscDMDAElementType ival = DMDA_ELEMENT_Q1
        CHKERR( DMDAGetElementType(self.dm, &ival) )
        return <long>ival

    def getElements(self, elem_type=None):
        cdef PetscInt dim=0
        cdef PetscDMDAElementType etype
        cdef PetscInt nel=0, nen=0
        cdef const_PetscInt *elems=NULL
        cdef object elements
        CHKERR( DMDAGetDim(self.dm, &dim) )
        if elem_type is not None:
            etype = daelementtype(elem_type)
            CHKERR( DMDASetElementType(self.dm, etype) )
        try:
            CHKERR( DMDAGetElements(self.dm, &nel, &nen, &elems) )
            elements = array_i(nel*nen, elems)
            elements.shape = (toInt(nel), toInt(nen))
        finally:
            CHKERR( DMDARestoreElements(self.dm, &nel, &nen, &elems) )
        return elements

    #

    property dim:
        def __get__(self):
            return self.getDim()

    property dof:
        def __get__(self):
            return self.getDof()

    property sizes:
        def __get__(self):
            return self.getSizes()

    property proc_sizes:
        def __get__(self):
            return self.getProcSizes()

    property boundary_type:
        def __get__(self):
            return self.getBoundaryType()

    property stencil:
        def __get__(self):
            return self.getStencil()

    property stencil_type:
        def __get__(self):
            return self.getStencilType()

    property stencil_width:
        def __get__(self):
            return self.getStencilWidth()

    property ranges:
        def __get__(self):
            return self.getRanges()

    property ghost_ranges:
        def __get__(self):
            return self.getGhostRanges()

    property corners:
        def __get__(self):
            return self.getCorners()

    property ghost_corners:
        def __get__(self):
            return self.getGhostCorners()

    # backward compatibility
    createNaturalVector = createNaturalVec


# backward compatibility alias
DA = DMDA

# --------------------------------------------------------------------

del DMDABoundaryType
del DMDAStencilType
del DMDAInterpolationType
del DMDAElementType

# --------------------------------------------------------------------
