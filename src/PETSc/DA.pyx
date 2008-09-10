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
    BOX = DA_STENCIL_BOX

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

    def create(self, sizes,
               periodic=None, stencil=None,
               dof=1, sw=1, comm=None):
        cdef PetscDAPeriodicType ptype = DA_PERIODIC_NONE
        cdef PetscDAStencilType  stype = DA_STENCIL_BOX
        if periodic is not None: ptype = periodic
        if stencil  is not None: stype = stencil
        cdef PetscInt M = PETSC_DECIDE, m = PETSC_DECIDE, *lx=NULL
        cdef PetscInt N = PETSC_DECIDE, n = PETSC_DECIDE, *ly=NULL
        cdef PetscInt P = PETSC_DECIDE, p = PETSC_DECIDE, *lz=NULL
        sizes = tuple(sizes)
        cdef PetscInt dim = len(sizes)
        if   dim == 1: M, = sizes
        elif dim == 2: M, N = sizes
        elif dim == 3: M, N, P = sizes
        cdef PetscInt idof = dof
        cdef PetscInt isw = sw
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDA newda = NULL
        CHKERR( DACreate(ccomm,
                         dim,
                         ptype, stype,
                         M, N, P,
                         m, n, p,
                         idof, isw,
                         lx, ly, lz,
                         &newda) )
        PetscCLEAR(self.obj); self.da = newda
        return self


    def getInfo(self):
        cdef PetscInt dim = 0
        cdef PetscInt M = PETSC_DECIDE, m = PETSC_DECIDE, *lx=NULL
        cdef PetscInt N = PETSC_DECIDE, n = PETSC_DECIDE, *ly=NULL
        cdef PetscInt P = PETSC_DECIDE, p = PETSC_DECIDE, *lz=NULL
        cdef PetscInt dof = 1
        cdef PetscInt sw = 1
        cdef PetscDAPeriodicType ptype = DA_PERIODIC_NONE
        cdef PetscDAStencilType  stype = DA_STENCIL_BOX
        CHKERR( DAGetInfo(self.da,
                          &dim,
                          &M, &N, &P,
                          &m, &n, &p,
                          &dof, &sw,
                          &ptype, &stype) )
        return (dim,
                (M,N,P)[:dim],
                (m,n,p)[:dim],
                dof, sw,
                ptype, stype)

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

    def globalToNatural(self, Vec vg not None, Vec vn not None,
                        addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DAGlobalToNaturalBegin(self.da, vg.vec, im, vn.vec) )
        CHKERR( DAGlobalToNaturalEnd  (self.da, vg.vec, im, vn.vec) )

    def naturalToGlobal(self, Vec vn not None, Vec vg not None,
                        addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DANaturalToGlobalBegin(self.da, vn.vec, im, vg.vec) )
        CHKERR( DANaturalToGlobalEnd  (self.da, vn.vec, im, vg.vec) )

    def globalToLocal(self, Vec vg not None, Vec vl not None,
                      addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DAGlobalToLocalBegin(self.da, vg.vec, im, vl.vec) )
        CHKERR( DAGlobalToLocalEnd  (self.da, vg.vec, im, vl.vec) )

    def localToGlobal(self, Vec vl not None, Vec vg not None,
                      addv=None):
        cdef PetscInsertMode im = insertmode(addv)
        CHKERR( DALocalToGlobal(self.da, vl.vec, im, vg.vec) )

    def localToGlobalAdd(self, Vec vl not None, Vec vg not None):
        CHKERR( DALocalToGlobalBegin(self.da, vl.vec, vg.vec) )
        CHKERR( DALocalToGlobalEnd  (self.da, vl.vec, vg.vec) )

    def getMatrix(self, mat_type=None):
        cdef PetscMatType mtype = <PetscMatType>MATAIJ
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
