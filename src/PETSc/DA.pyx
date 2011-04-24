# --------------------------------------------------------------------

class DABoundaryType(object):
    NONE     = DA_BOUNDARY_NONE
    GHOSTED  = DA_BOUNDARY_GHOSTED
    MIRROR   = DA_BOUNDARY_MIRROR
    PERIODIC = DA_BOUNDARY_PERIODIC

class DAStencilType(object):
    STAR = DA_STENCIL_STAR
    BOX  = DA_STENCIL_BOX

class DAInterpolationType(object):
    Q0 = DA_INTERPOLATION_Q0
    Q1 = DA_INTERPOLATION_Q1

class DAElementType(object):
    P1 = DA_ELEMENT_P1
    Q1 = DA_ELEMENT_Q1

# --------------------------------------------------------------------

cdef class DA(DM):

    BoundaryType      = DABoundaryType
    StencilType       = DAStencilType
    InterpolationType = DAInterpolationType
    ElementType       = DAElementType

    #

    def create(self, dim=None, dof=1,
               sizes=None, proc_sizes=None, boundary_type=None,
               stencil_type=None, stencil_width=0, comm=None):
        #
        cdef object arg = None
        try: arg = tuple(dim)
        except TypeError: pass
        else: dim, sizes = None, arg
        #
        cdef PetscInt ndim = PETSC_DECIDE
        cdef PetscInt ndof = 1
        cdef PetscInt M = 1, m = PETSC_DECIDE, *lx = NULL
        cdef PetscInt N = 1, n = PETSC_DECIDE, *ly = NULL
        cdef PetscInt P = 1, p = PETSC_DECIDE, *lz = NULL
        cdef PetscDABoundaryType btx = DA_BOUNDARY_NONE
        cdef PetscDABoundaryType bty = DA_BOUNDARY_NONE
        cdef PetscDABoundaryType btz = DA_BOUNDARY_NONE
        cdef PetscDAStencilType  stype = DA_STENCIL_STAR
        cdef PetscInt            swidth = 0
        # global grid sizes
        cdef object gsizes = sizes
        if gsizes is None: gsizes = ()
        else: gsizes = tuple(gsizes)
        cdef PetscInt gdim = asDims(gsizes, &M, &N, &P)
        assert gdim <= 3
        # processor sizes
        cdef object psizes = proc_sizes
        if psizes is None: psizes = ()
        else: psizes = tuple(psizes)
        cdef PetscInt pdim = asDims(psizes, &m, &n, &p)
        assert pdim <= 3
        # vertex distribution
        lx = NULL # XXX implement!
        ly = NULL # XXX implement!
        lz = NULL # XXX implement!
        # dim and dof, periodicity, stencil type & width
        if dim is not None: ndim = asInt(dim)
        if dof is not None: ndof = asInt(dof)
        if ndim==PETSC_DECIDE and gdim>0: ndim = gdim
        if ndof==PETSC_DECIDE: ndof = 1
        if boundary_type is not None:
            asBoundary(ndim, boundary_type, &btx, &bty, &btz)
        if stencil_type is not None:
            stype = asStencil(stencil_type)
        if stencil_width is not None:
            swidth = asInt(stencil_width)
        # create the DA object
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDA newda = NULL
        CHKERR( DACreateND(ccomm, ndim, ndof,
                           M, N, P, m, n, p, lx, ly, lz,
                           btx, bty, btz, stype, swidth, &newda) )
        PetscCLEAR(self.obj); self.dm = newda
        return self

    def duplicate(self, dof=None, boundary_type=None,
                  stencil_type=None, stencil_width=None):
        cdef PetscInt dim = 0
        cdef PetscInt M = 1, N = 1, P = 1
        cdef PetscInt m = 1, n = 1, p = 1
        cdef PetscInt ndof = 1, swidth = 1
        cdef PetscDABoundaryType btx = DA_BOUNDARY_NONE
        cdef PetscDABoundaryType bty = DA_BOUNDARY_NONE
        cdef PetscDABoundaryType btz = DA_BOUNDARY_NONE
        cdef PetscDAStencilType  stype = DA_STENCIL_BOX
        CHKERR( DAGetInfo(self.dm,
                          &dim,
                          &M, &N, &P,
                          &m, &n, &p,
                          &ndof, &swidth,
                          &btx, &bty, &btz,
                          &stype) )
        cdef const_PetscInt *lx = NULL, *ly = NULL, *lz = NULL
        CHKERR( DAGetOwnershipRanges(self.dm, &lx, &ly, &lz) )
        cdef MPI_Comm comm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.dm, &comm) )
        #
        if dof is not None:
            ndof = asInt(dof)
        if boundary_type is not None:
            asBoundary(dim, boundary_type, &btx, &bty, &btz)
        if stencil_type  is not None:
            stype = asStencil(stencil_type)
        if stencil_width is not None:
            swidth = asInt(stencil_width)
        #
        cdef DA da = DA()
        CHKERR( DACreateND(comm, dim, ndof,
                           M, N, P, m, n, p, lx, ly, lz,
                           btx, bty, btz, stype, swidth, &da.dm) )
        return da

    #

    def getDim(self):
        cdef PetscInt dim = 0
        CHKERR( DAGetInfo(self.dm,
                          &dim,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, NULL, NULL,
                          NULL) )
        return toInt(dim)

    def getDof(self):
        cdef PetscInt dof = 0
        CHKERR( DAGetInfo(self.dm,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          &dof, NULL,
                          NULL, NULL, NULL,
                          NULL) )
        return toInt(dof)

    def getSizes(self):
        cdef PetscInt dim = 0
        cdef PetscInt M = PETSC_DECIDE
        cdef PetscInt N = PETSC_DECIDE
        cdef PetscInt P = PETSC_DECIDE
        CHKERR( DAGetInfo(self.dm,
                          &dim,
                          &M, &N, &P,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, NULL, NULL,
                          NULL) )
        return toDims(dim, M, N, P)

    def getProcSizes(self):
        cdef PetscInt dim = 0
        cdef PetscInt m = PETSC_DECIDE
        cdef PetscInt n = PETSC_DECIDE
        cdef PetscInt p = PETSC_DECIDE
        CHKERR( DAGetInfo(self.dm,
                          &dim,
                          NULL, NULL, NULL,
                          &m, &n, &p,
                          NULL, NULL,
                          NULL, NULL, NULL,
                          NULL) )
        return toDims(dim, m, n, p)

    def getBoundaryType(self):
        cdef PetscInt dim = 0
        cdef PetscDABoundaryType btx = DA_BOUNDARY_NONE
        cdef PetscDABoundaryType bty = DA_BOUNDARY_NONE
        cdef PetscDABoundaryType btz = DA_BOUNDARY_NONE
        CHKERR( DAGetInfo(self.dm,
                          &dim,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          &btx, &bty, &btz,
                          NULL) )
        return toDims(dim, btx, bty, btz)

    def getStencil(self):
        cdef PetscDAStencilType stype = DA_STENCIL_BOX
        cdef PetscInt swidth = 0
        CHKERR( DAGetInfo(self.dm,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, &swidth,
                          NULL, NULL, NULL,
                          &stype) )
        return (toStencil(stype), toInt(swidth))

    def getStencilType(self):
        cdef PetscDAStencilType stype = DA_STENCIL_BOX
        CHKERR( DAGetInfo(self.dm,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, NULL, NULL,
                          &stype) )
        return stype

    def getStencilWidth(self):
        cdef PetscInt swidth = 0
        CHKERR( DAGetInfo(self.dm,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, &swidth,
                          NULL, NULL, NULL,
                          NULL) )
        return toInt(swidth)

    #

    def getRanges(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.dm, &dim) )
        CHKERR( DAGetCorners(self.dm,
                             &x, &y, &z,
                             &m, &n, &p) )
        return ((toInt(x), toInt(x+m)),
                (toInt(y), toInt(y+n)),
                (toInt(z), toInt(z+p)))[:<Py_ssize_t>dim]

    def getGhostRanges(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.dm, &dim) )
        CHKERR( DAGetGhostCorners(self.dm,
                                  &x, &y, &z,
                                  &m, &n, &p) )
        return ((toInt(x), toInt(x+m)),
                (toInt(y), toInt(y+n)),
                (toInt(z), toInt(z+p)))[:<Py_ssize_t>dim]

    def getCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.dm, &dim) )
        CHKERR( DAGetCorners(self.dm,
                             &x, &y, &z,
                             &m, &n, &p) )
        return ((toInt(x), toInt(y), toInt(z))[:<Py_ssize_t>dim],
                (toInt(m), toInt(n), toInt(p))[:<Py_ssize_t>dim])

    def getGhostCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.dm, &dim) )
        CHKERR( DAGetGhostCorners(self.dm,
                                  &x, &y, &z,
                                  &m, &n, &p) )
        return ((toInt(x), toInt(y), toInt(z))[:<Py_ssize_t>dim],
                (toInt(m), toInt(n), toInt(p))[:<Py_ssize_t>dim])

    #

    def getVecArray(self, Vec vec not None):
        return _DA_Vec_array(self, vec)

    #

    def setUniformCoordinates(self,
                              xmin=0, xmax=1,
                              ymin=0, ymax=1,
                              zmin=0, zmax=1):
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR( DASetUniformCoordinates(self.dm,
                                        _xmin, _xmax,
                                        _ymin, _ymax,
                                        _zmin, _zmax) )

    def setCoordinates(self, Vec c not None):
        CHKERR( DASetCoordinates(self.dm, c.vec) )

    def getCoordinates(self):
        cdef Vec c = Vec()
        CHKERR( DAGetCoordinates(self.dm, &c.vec) )
        PetscINCREF(c.obj)
        return c

    def getCoordinateDA(self):
        cdef DA cda = DA()
        CHKERR( DAGetCoordinateDA(self.dm, &cda.dm) )
        PetscINCREF(cda.obj)
        return cda

    def getGhostCoordinates(self):
        cdef Vec gc = Vec()
        CHKERR( DAGetGhostedCoordinates(self.dm, &gc.vec) )
        PetscINCREF(gc.obj)
        return gc

    def getBoundingBox(self):
        cdef PetscInt i,dim=0
        CHKERR( DAGetDim(self.dm, &dim) )
        cdef PetscReal gmin[3], gmax[3]
        CHKERR( DAGetBoundingBox(self.dm, gmin, gmax) )
        return tuple([(toReal(gmin[i]), toReal(gmax[i]))
                      for i from 0 <= i < dim])

    def getLocalBoundingBox(self):
        cdef PetscInt i,dim=0
        CHKERR( DAGetDim(self.dm, &dim) )
        cdef PetscReal lmin[3], lmax[3]
        CHKERR( DAGetLocalBoundingBox(self.dm, lmin, lmax) )
        return tuple([(toReal(lmin[i]), toReal(lmax[i]))
                      for i from 0 <= i < dim])

    #

    def createNaturalVec(self):
        cdef Vec vn = Vec()
        CHKERR( DACreateNaturalVector(self.dm, &vn.vec) )
        return vn

    def globalToNatural(self, Vec vg not None, Vec vn not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DAGlobalToNaturalBegin(self.dm, vg.vec, im, vn.vec) )
        CHKERR( DAGlobalToNaturalEnd  (self.dm, vg.vec, im, vn.vec) )

    def naturalToGlobal(self, Vec vn not None, Vec vg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DANaturalToGlobalBegin(self.dm, vn.vec, im, vg.vec) )
        CHKERR( DANaturalToGlobalEnd  (self.dm, vn.vec, im, vg.vec) )

    def localToLocal(self, Vec vl not None, Vec vlg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DALocalToLocalBegin(self.dm, vl.vec, im, vlg.vec) )
        CHKERR( DALocalToLocalEnd  (self.dm, vl.vec, im, vlg.vec) )

    #

    def getAO(self):
        cdef AO ao = AO()
        CHKERR( DAGetAO(self.dm, &ao.ao) )
        PetscINCREF(ao.obj)
        return ao

    def getScatter(self):
        cdef Scatter l2g = Scatter()
        cdef Scatter g2l = Scatter()
        cdef Scatter l2l = Scatter()
        CHKERR( DAGetScatter(self.dm, &l2g.sct, &g2l.sct, &l2l.sct) )
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
        CHKERR( DASetRefinementFactor(self.dm,
                                      refine[0],
                                      refine[1],
                                      refine[2]) )

    def getRefinementFactor(self):
        cdef PetscInt i, dim, refine[3]
        CHKERR( DAGetDim(self.dm, &dim) )
        CHKERR( DAGetRefinementFactor(self.dm,
                                      &refine[0],
                                      &refine[1],
                                      &refine[2]) )
        return tuple([toInt(refine[i]) for 0 <= i < dim])

    def setInterpolationType(self, interp_type):
        cdef PetscDAInterpolationType ival = dainterpolationtype(interp_type)
        CHKERR( DASetInterpolationType(self.dm, ival) )

    def getInterpolationType(self, interp_type):
        cdef PetscDAInterpolationType ival = DA_INTERPOLATION_Q0
        CHKERR( DAGetInterpolationType(self.dm, &ival) )
        return <long>ival

    #

    def setElementType(self, elem_type):
        cdef PetscDAElementType ival = daelementtype(elem_type)
        CHKERR( DASetElementType(self.dm, ival) )

    def getElementType(self, elem_type):
        cdef PetscDAElementType ival = DA_ELEMENT_Q1
        CHKERR( DAGetElementType(self.dm, &ival) )
        return <long>ival

    def getElements(self, elem_type=None):
        cdef PetscInt dim=0
        cdef PetscDAElementType etype
        cdef PetscInt nel=0, nen=0
        cdef const_PetscInt *elems=NULL
        cdef object elements
        CHKERR( DAGetDim(self.dm, &dim) )
        if elem_type is not None:
            etype = daelementtype(elem_type)
            CHKERR( DASetElementType(self.dm, etype) )
        try:
            CHKERR( DAGetElements(self.dm, &nel, &nen, &elems) )
            elements = array_i(nel*nen, elems)
            elements.shape = (toInt(nel), toInt(nen))
        finally:
            CHKERR( DARestoreElements(self.dm, &nel, &nen, &elems) )
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

# --------------------------------------------------------------------

del DABoundaryType
del DAStencilType
del DAInterpolationType
del DAElementType

# --------------------------------------------------------------------
