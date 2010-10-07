# --------------------------------------------------------------------

class DAPeriodicType(object):
    NONE = DA_NONPERIODIC
    #
    X    = DA_XPERIODIC
    Y    = DA_YPERIODIC
    Z    = DA_ZPERIODIC
    XY   = DA_XYPERIODIC
    XZ   = DA_XZPERIODIC
    YZ   = DA_YZPERIODIC
    XYZ  = DA_XYZPERIODIC
    #
    PERIODIC_XYZ = DA_XYZPERIODIC
    GHOSTED_XYZ  = DA_XYZGHOSTED

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

cdef class DA(Object):

    PeriodicType      = DAPeriodicType
    StencilType       = DAStencilType
    InterpolationType = DAInterpolationType
    ElementType       = DAElementType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.da
        self.da = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( DAView(self.da, cviewer) )

    def destroy(self):
        CHKERR( DADestroy(self.da) )
        self.da = NULL
        return self

    def create(self, dim=None, dof=1,
               sizes=None, proc_sizes=None, periodic=None,
               stencil_type=None, stencil_width=1, comm=None):
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
        cdef PetscDAPeriodicType ptype = DA_NONPERIODIC
        cdef PetscDAStencilType  stype = DA_STENCIL_BOX
        cdef PetscInt            swidth = 1
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
        ptype = asPeriodic(ndim, periodic)
        stype = asStencil(stencil_type)
        swidth = asInt(stencil_width)
        # create the DA object
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDA newda = NULL
        CHKERR( DACreateND(ccomm, ndim, ndof,
                           M, N, P, m, n, p, lx, ly, lz,
                           ptype, stype, swidth, &newda) )
        PetscCLEAR(self.obj); self.da = newda
        return self

    def setOptionsPrefix(self, prefix):
        cdef const_char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( DASetOptionsPrefix(self.da, cval) )

    def setFromOptions(self):
        CHKERR( DASetFromOptions(self.da) )

    #

    def getDim(self):
        cdef PetscInt dim = 0
        CHKERR( DAGetInfo(self.da,
                          &dim,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, NULL) )
        return toInt(dim)

    def getDof(self):
        cdef PetscInt dof = 0
        CHKERR( DAGetInfo(self.da,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          &dof, NULL,
                          NULL, NULL) )
        return toInt(dof)

    def getSizes(self):
        cdef PetscInt dim = 0
        cdef PetscInt M = PETSC_DECIDE
        cdef PetscInt N = PETSC_DECIDE
        cdef PetscInt P = PETSC_DECIDE
        CHKERR( DAGetInfo(self.da,
                          &dim,
                          &M, &N, &P,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, NULL) )
        return (toInt(M), toInt(N), toInt(P))[:<Py_ssize_t>dim]

    def getProcSizes(self):
        cdef PetscInt dim = 0
        cdef PetscInt m = PETSC_DECIDE
        cdef PetscInt n = PETSC_DECIDE
        cdef PetscInt p = PETSC_DECIDE
        CHKERR( DAGetInfo(self.da,
                          &dim,
                          NULL, NULL, NULL,
                          &m, &n, &p,
                          NULL, NULL,
                          NULL, NULL) )
        return (toInt(m), toInt(n), toInt(p))[:<Py_ssize_t>dim]

    def getPeriodic(self):
        cdef PetscInt dim = 0
        cdef PetscDAPeriodicType ptype = DA_NONPERIODIC
        CHKERR( DAGetInfo(self.da,
                          &dim,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          &ptype, NULL) )
        return toPeriodic(dim, ptype)

    def getPeriodicType(self):
        cdef PetscDAPeriodicType ptype = DA_NONPERIODIC
        CHKERR( DAGetInfo(self.da,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          &ptype, NULL) )
        return ptype

    def getStencil(self):
        cdef PetscDAStencilType  stype = DA_STENCIL_BOX
        CHKERR( DAGetInfo(self.da,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, &stype) )
        return toStencil(stype)

    def getStencilType(self):
        cdef PetscDAStencilType  stype = DA_STENCIL_BOX
        CHKERR( DAGetInfo(self.da,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, &stype) )
        return stype

    def getStencilWidth(self):
        cdef PetscInt swidth = 0
        CHKERR( DAGetInfo(self.da,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, &swidth,
                          NULL, NULL) )
        return toInt(swidth)

    #

    def getRanges(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.da, &dim) )
        CHKERR( DAGetCorners(self.da,
                             &x, &y, &z,
                             &m, &n, &p) )
        return ((toInt(x), toInt(x+m)),
                (toInt(y), toInt(y+n)),
                (toInt(z), toInt(z+p)))[:<Py_ssize_t>dim]

    def getGhostRanges(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.da, &dim) )
        CHKERR( DAGetGhostCorners(self.da,
                                  &x, &y, &z,
                                  &m, &n, &p) )
        return ((toInt(x), toInt(x+m)),
                (toInt(y), toInt(y+n)),
                (toInt(z), toInt(z+p)))[:<Py_ssize_t>dim]

    def getCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.da, &dim) )
        CHKERR( DAGetCorners(self.da,
                             &x, &y, &z,
                             &m, &n, &p) )
        return ((toInt(x), toInt(y), toInt(z))[:<Py_ssize_t>dim],
                (toInt(m), toInt(n), toInt(p))[:<Py_ssize_t>dim])

    def getGhostCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.da, &dim) )
        CHKERR( DAGetGhostCorners(self.da,
                                  &x, &y, &z,
                                  &m, &n, &p) )
        return ((toInt(x), toInt(y), toInt(z))[:<Py_ssize_t>dim],
                (toInt(m), toInt(n), toInt(p))[:<Py_ssize_t>dim])

    #

    def setUniformCoordinates(self,
                              xmin=0, xmax=1,
                              ymin=0, ymax=1,
                              zmin=0, zmax=1):
        cdef PetscReal _xmin = asReal(xmin), _xmax = asReal(xmax)
        cdef PetscReal _ymin = asReal(ymin), _ymax = asReal(ymax)
        cdef PetscReal _zmin = asReal(zmin), _zmax = asReal(zmax)
        CHKERR( DASetUniformCoordinates(self.da,
                                        _xmin, _xmax,
                                        _ymin, _ymax,
                                        _zmin, _zmax) )

    def setCoordinates(self, Vec c not None):
        CHKERR( DASetCoordinates(self.da, c.vec) )

    def getCoordinates(self):
        cdef Vec c = Vec()
        CHKERR( DAGetCoordinates(self.da, &c.vec) )
        PetscIncref(<PetscObject>c.vec)
        return c

    def getCoordinateDA(self):
        cdef DA cda = DA()
        CHKERR( DAGetCoordinateDA(self.da, &cda.da) )
        PetscIncref(<PetscObject>cda.da)
        return cda

    def getGhostCoordinates(self):
        cdef Vec gc = Vec()
        CHKERR( DAGetGhostedCoordinates(self.da, &gc.vec) )
        PetscIncref(<PetscObject>gc.vec)
        return gc

    #

    def createNaturalVec(self):
        cdef Vec vn = Vec()
        CHKERR( DACreateNaturalVector(self.da, &vn.vec) )
        return vn

    def createGlobalVec(self):
        cdef Vec vg = Vec()
        CHKERR( DACreateGlobalVector(self.da, &vg.vec) )
        return vg

    def createLocalVec(self):
        cdef Vec vl = Vec()
        CHKERR( DACreateLocalVector(self.da, &vl.vec) )
        return vl

    def createMat(self, mat_type=None):
        cdef PetscMatType mtype = MATAIJ
        mat_type = str2bytes(mat_type, &mtype)
        if mtype == NULL: mtype = MATAIJ
        cdef Mat mat = Mat()
        CHKERR( DAGetMatrix(self.da, mtype, &mat.mat) )
        return mat

    createNaturalVector = createNaturalVec
    createGlobalVector = createGlobalVec
    createLocalVector = createLocalVec
    getMatrix = createMatrix = createMat

    #

    def globalToNatural(self, Vec vg not None, Vec vn not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DAGlobalToNaturalBegin(self.da, vg.vec, im, vn.vec) )
        CHKERR( DAGlobalToNaturalEnd  (self.da, vg.vec, im, vn.vec) )

    def naturalToGlobal(self, Vec vn not None, Vec vg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DANaturalToGlobalBegin(self.da, vn.vec, im, vg.vec) )
        CHKERR( DANaturalToGlobalEnd  (self.da, vn.vec, im, vg.vec) )

    def globalToLocal(self, Vec vg not None, Vec vl not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DAGlobalToLocalBegin(self.da, vg.vec, im, vl.vec) )
        CHKERR( DAGlobalToLocalEnd  (self.da, vg.vec, im, vl.vec) )

    def localToGlobalAdd(self, Vec vl not None, Vec vg not None):
        CHKERR( DALocalToGlobalBegin(self.da, vl.vec, vg.vec) )
        CHKERR( DALocalToGlobalEnd  (self.da, vl.vec, vg.vec) )

    def localToGlobal(self, Vec vl not None, Vec vg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DALocalToGlobal(self.da, vl.vec, im, vg.vec) )

    def localToLocal(self, Vec vl not None, Vec vlg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DALocalToLocalBegin(self.da, vl.vec, im, vlg.vec) )
        CHKERR( DALocalToLocalEnd  (self.da, vl.vec, im, vlg.vec) )

    #

    def getVecArray(self, Vec vec not None):
        return _DA_Vec_array(self, vec)

    #

    def getAO(self):
        cdef AO ao = AO()
        CHKERR( DAGetAO(self.da, &ao.ao) )
        PetscIncref(<PetscObject>ao.ao)
        return ao

    def getLGMap(self):
        cdef LGMap lgm = LGMap()
        CHKERR( DAGetISLocalToGlobalMapping(self.da, &lgm.lgm) )
        PetscIncref(<PetscObject>lgm.lgm)
        return lgm

    def getLGMapBlock(self):
        cdef LGMap lgm = LGMap()
        CHKERR( DAGetISLocalToGlobalMappingBlck(self.da, &lgm.lgm) )
        PetscIncref(<PetscObject>lgm.lgm)
        return lgm

    def getScatter(self):
        cdef Scatter l2g = Scatter()
        cdef Scatter g2l = Scatter()
        cdef Scatter l2l = Scatter()
        CHKERR( DAGetScatter(self.da, &l2g.sct, &g2l.sct, &l2l.sct) )
        PetscIncref(<PetscObject>l2g.sct)
        PetscIncref(<PetscObject>g2l.sct)
        PetscIncref(<PetscObject>l2l.sct)
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
        CHKERR( DASetRefinementFactor(self.da,
                                      refine[0],
                                      refine[1],
                                      refine[2]) )

    def getRefinementFactor(self):
        cdef PetscInt i, dim, refine[3]
        CHKERR( DAGetDim(self.da, &dim) )
        CHKERR( DAGetRefinementFactor(self.da,
                                      &refine[0],
                                      &refine[1],
                                      &refine[2]) )
        return tuple([toInt(refine[i]) for 0 <= i < dim])

    def refine(self, comm=None):
        cdef MPI_Comm dacomm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.da, &dacomm) )
        dacomm = def_Comm(comm, dacomm)
        cdef DA da = DA()
        CHKERR( DARefine(self.da, dacomm, &da.da) )
        return da

    def coarsen(self, comm=None):
        cdef MPI_Comm dacomm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.da, &dacomm) )
        dacomm = def_Comm(comm, dacomm)
        cdef DA da = DA()
        CHKERR( DACoarsen(self.da, dacomm, &da.da) )
        return da

    def setInterpolationType(self, interp_type):
        cdef PetscDAInterpolationType ival = dainterpolationtype(interp_type)
        CHKERR( DASetInterpolationType(self.da, ival) )

    def getInterpolation(self, DA da not None):
        cdef Mat A = Mat()
        cdef Vec scale = Vec()
        CHKERR( DAGetInterpolation(self.da, da.da, 
                                   &A.mat, &scale.vec))
        return(A, scale)

    #

    def setElementType(self, elem_type):
        cdef PetscDAElementType ival = daelementtype(elem_type)
        CHKERR( DASetElementType(self.da, ival) )

    def getElements(self, elem_type=None):
        cdef PetscInt dim=0
        cdef PetscDAElementType etype
        cdef PetscInt nel=0, nen=0
        cdef const_PetscInt *elems=NULL
        cdef object elements
        CHKERR( DAGetDim(self.da, &dim) )
        if elem_type is not None:
            etype = daelementtype(elem_type)
            CHKERR( DASetElementType(self.da, etype) )
        try:
            CHKERR( DAGetElements(self.da, &nel, &nen, &elems) )
            elements = array_i(nel*nen, elems)
            elements.shape = (toInt(nel), toInt(nen))
        finally:
            CHKERR( DARestoreElements(self.da, &nel, &nen, &elems) )
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

    property periodic:
        def __get__(self):
            return self.getPeriodic()

    property periodic_type:
        def __get__(self):
            return self.getPeriodicType()

    property stencil:
        def __get__(self):
            return self.getStencil()

    property stencil_type:
        def __get__(self):
            return self.getStencilType()

    property stencil_width:
        def __get__(self):
            return self.getStencilWidth()

    #

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

del DAPeriodicType
del DAStencilType
del DAInterpolationType
del DAElementType

# --------------------------------------------------------------------
