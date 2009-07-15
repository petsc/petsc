# --------------------------------------------------------------------

class DAPeriodicType(object):
    NONE  = DA_PERIODIC_NONE
    X     = DA_PERIODIC_X
    Y     = DA_PERIODIC_Y
    XY    = DA_PERIODIC_XY
    XZ    = DA_PERIODIC_XZ
    YZ    = DA_PERIODIC_YZ
    XYZ   = DA_PERIODIC_XYZ
    Z     = DA_PERIODIC_Z
    #XYZ_G = DA_PERIODIC_XYZ_G

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

    def create(self, sizes, proc_sizes=None,
               periodic=None, stencil=None,
               ndof=1, width=1, comm=None):
        cdef PetscDAPeriodicType ptype = DA_PERIODIC_NONE
        cdef PetscDAStencilType  stype = DA_STENCIL_BOX
        if periodic is not None: ptype = periodic
        if stencil  is not None: stype = stencil
        cdef PetscInt M = PETSC_DECIDE, m = PETSC_DECIDE, *lx=NULL
        cdef PetscInt N = PETSC_DECIDE, n = PETSC_DECIDE, *ly=NULL
        cdef PetscInt P = PETSC_DECIDE, p = PETSC_DECIDE, *lz=NULL
        sizes = tuple(sizes)
        cdef PetscInt dim = len(sizes)
        assert dim==1 or dim==2 or dim==3
        if   dim == 1: M, = sizes
        elif dim == 2: M, N = sizes
        elif dim == 3: M, N, P = sizes
        if proc_sizes is not None:
            proc_sizes = tuple(proc_sizes)
            if   dim == 1: m, = proc_sizes
            elif dim == 2: m, n = proc_sizes
            elif dim == 3: m, n, p = proc_sizes
        cdef PetscInt nd = ndof
        cdef PetscInt sw = width
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDA newda = NULL
        CHKERR( DACreate(ccomm,
                         dim,
                         ptype, stype,
                         M, N, P,
                         m, n, p,
                         ndof, width,
                         lx, ly, lz,
                         &newda) )
        PetscCLEAR(self.obj); self.da = newda
        return self

    #

    def getDim(self):
        cdef PetscInt dim = 0
        CHKERR( DAGetInfo(self.da,
                          &dim,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, NULL) )
        return dim

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
        return (M,N,P)[:dim]

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
        return (m,n,p)[:dim]

    def getNDof(self):
        cdef PetscInt ndof = 0
        CHKERR( DAGetInfo(self.da,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          &ndof, NULL,
                          NULL, NULL) )
        return ndof

    def getWidth(self):
        cdef PetscInt width = 0
        CHKERR( DAGetInfo(self.da,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, &width,
                          NULL, NULL) )
        return width

    def getPeriodicType(self):
        cdef PetscDAPeriodicType ptype = DA_PERIODIC_NONE
        CHKERR( DAGetInfo(self.da,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          &ptype, NULL) )
        return ptype

    def getStencilType(self):
        cdef PetscDAStencilType  stype = DA_STENCIL_BOX
        CHKERR( DAGetInfo(self.da,
                          NULL,
                          NULL, NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,
                          NULL, &stype) )
        return stype
    #

    def getRanges(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.da, &dim) )
        CHKERR( DAGetCorners(self.da,
                             &x, &y, &z,
                             &m, &n, &p) )
        return ((x, x+m),
                (y, y+n),
                (z, z+p))[:dim]

    def getGhostRanges(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.da, &dim) )
        CHKERR( DAGetGhostCorners(self.da,
                                  &x, &y, &z,
                                  &m, &n, &p) )
        return ((x, x+m),
                (y, y+n),
                (z, z+p))[:dim]

    def getCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.da, &dim) )
        CHKERR( DAGetCorners(self.da,
                             &x, &y, &z,
                             &m, &n, &p) )
        return ((x, y, z)[:dim],
                (m, n, p)[:dim])

    def getGhostCorners(self):
        cdef PetscInt dim=0, x=0, y=0, z=0, m=0, n=0, p=0
        CHKERR( DAGetDim(self.da, &dim) )
        CHKERR( DAGetGhostCorners(self.da,
                                  &x, &y, &z,
                                  &m, &n, &p) )
        return ((x, y, z)[:dim],
                (m, n, p)[:dim])

    #

    def createNaturalVector(self):
        cdef Vec vn = Vec()
        CHKERR( DACreateNaturalVector(self.da, &vn.vec) )
        return vn

    def createGlobalVector(self):
        cdef Vec vg = Vec()
        CHKERR( DACreateGlobalVector(self.da, &vg.vec) )
        return vg

    def createLocalVector(self):
        cdef Vec vl = Vec()
        CHKERR( DACreateLocalVector(self.da, &vl.vec) )
        return vl

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

    def localToGlobal(self, Vec vl not None, Vec vg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DALocalToGlobal(self.da, vl.vec, im, vg.vec) )

    def localToGlobalAdd(self, Vec vl not None, Vec vg not None):
        CHKERR( DALocalToGlobalBegin(self.da, vl.vec, vg.vec) )
        CHKERR( DALocalToGlobalEnd  (self.da, vl.vec, vg.vec) )

    def localToLocal(self, Vec vl not None, Vec vlg not None, addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DALocalToLocalBegin(self.da, vl.vec, im, vlg.vec) )
        CHKERR( DALocalToLocalEnd  (self.da, vl.vec, im, vlg.vec) )

    def getMatrix(self, mat_type=None):
        cdef PetscMatType mtype = MATAIJ
        if mat_type is not None: mtype = str2cp(mat_type)
        cdef Mat mat = Mat()
        CHKERR( DAGetMatrix(self.da, mtype, &mat.mat) )
        return mat

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

    property dim:
        def __get__(self):
            return self.getDim()

    property sizes:
        def __get__(self):
            return self.getSizes

    property proc_sizes:
        def __get__(self):
            return self.getProcSizes()

    property ndof:
        def __get__(self):
            return self.getNDof()

    property width:
        def __get__(self):
            return self.getWidth()

    property periodic_type:
        def __get__(self):
            return self.getPeriodicType()

    property stencil_type:
        def __get__(self):
            return self.getStencilType()

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
