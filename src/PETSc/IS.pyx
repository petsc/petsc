# --------------------------------------------------------------------

class ISType(object):
    GENERAL = S_(ISGENERAL)
    BLOCK   = S_(ISBLOCK)
    STRIDE  = S_(ISSTRIDE)

# --------------------------------------------------------------------

cdef class IS(Object):

    Type = ISType

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.iset
        self.iset = NULL

    #

    def __getbuffer__(self, Py_buffer *view, int flags):
        cdef _IS_buffer buf = _IS_buffer(self)
        buf.acquirebuffer(view, flags)
   
    def __releasebuffer__(self, Py_buffer *view):
        cdef _IS_buffer buf = <_IS_buffer>(view.obj)
        buf.releasebuffer(view)

    #

    def view(self, Viewer viewer=None):
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( ISView(self.iset, cviewer) )

    def destroy(self):
        CHKERR( ISDestroy(self.iset) )
        self.iset = NULL
        return self

    def create(self, indices, bsize=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs = PETSC_DECIDE, nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        cdef PetscIS newiset = NULL
        indices = iarray_i(indices, &nidx, &idx)
        if bsize is not None: bs = asInt(bsize)
        if bs == PETSC_DECIDE:
            CHKERR( ISCreateGeneral(ccomm, nidx, idx, cm, &newiset) )
        else:
            CHKERR( ISCreateBlock(ccomm, bs, nidx, idx, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def createGeneral(self, indices, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        cdef PetscIS newiset = NULL
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISCreateGeneral(ccomm, nidx, idx, cm, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def createBlock(self, bsize, indices, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs = asInt(bsize)
        cdef PetscInt nidx = 0, *idx = NULL
        cdef PetscIS newiset = NULL
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISCreateBlock(ccomm, bs, nidx, idx, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def createStride(self, size, first=None, step=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt csize = asInt(size), cfirst = 0, cstep = 1
        if first is not None: cfirst = asInt(first)
        if step  is not None: cstep  = asInt(step)
        cdef PetscIS newiset = NULL
        CHKERR( ISCreateStride(ccomm, csize, cfirst, cstep, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def getType(self):
        cdef PetscISType cval = NULL
        CHKERR( ISGetType(self.iset, &cval) )
        return bytes2str(cval)

    def duplicate(self, copy=False):
        cdef IS iset = IS()
        CHKERR( ISDuplicate(self.iset, &iset.iset) )
        if copy: CHKERR( ISCopy(self.iset, iset.iset) )
        return iset

    def copy(self, IS result=None):
        if result is None:
            result = IS()
        if result.iset == NULL:
            CHKERR( ISDuplicate(self.iset, &result.iset) )
        CHKERR( ISCopy(self.iset, result.iset) )
        return result

    def allGather(self):
        cdef IS iset = IS()
        CHKERR( ISAllGather(self.iset, &iset.iset) )
        return iset

    def toGeneral(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISStride(self.iset, &flag) )
        if flag == PETSC_FALSE: return self
        CHKERR( ISStrideToGeneral(self.iset) )
        return self # XXX IS_BLOCK ?

    def invertPermutation(self, nlocal=None):
        cdef PetscInt cnlocal = PETSC_DECIDE
        if nlocal is not None: cnlocal = asInt(nlocal)
        cdef IS iset = IS()
        CHKERR( ISInvertPermutation(self.iset, cnlocal, &iset.iset) )
        return iset

    def getSize(self):
        cdef PetscInt N = 0
        CHKERR( ISGetSize(self.iset, &N) )
        return toInt(N)

    def getLocalSize(self):
        cdef PetscInt n = 0
        CHKERR( ISGetLocalSize(self.iset, &n) )
        return toInt(n)

    def getSizes(self):
        cdef PetscInt n = 0, N = 0
        CHKERR( ISGetLocalSize(self.iset, &n) )
        CHKERR( ISGetSize(self.iset, &N) )
        return (toInt(n), toInt(N))

    def getBlockSize(self):
        cdef PetscInt bs = 1
        cdef PetscBool block = PETSC_FALSE
        CHKERR( ISBlock(self.iset, &block) )
        if block != PETSC_FALSE:
            CHKERR( ISBlockGetBlockSize(self.iset, &bs) )
        return toInt(bs)

    def getIndices(self):
        cdef PetscInt size = 0
        cdef const_PetscInt *indices = NULL
        CHKERR( ISGetLocalSize(self.iset, &size) )
        CHKERR( ISGetIndices(self.iset, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size, indices)
        finally:
            CHKERR( ISRestoreIndices(self.iset, &indices) )
        return oindices

    def getIndicesBlock(self):
        cdef PetscBool block = PETSC_FALSE
        CHKERR( ISBlock(self.iset, &block) )
        if block == PETSC_FALSE: return self.getIndices()
        cdef PetscInt size = 0, bs = 0
        cdef const_PetscInt *indices = NULL
        CHKERR( ISGetLocalSize(self.iset, &size) )
        CHKERR( ISBlockGetBlockSize(self.iset, &bs) )
        CHKERR( ISBlockGetIndices(self.iset, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size//bs, indices)
        finally:
            CHKERR( ISBlockRestoreIndices(self.iset, &indices) )
        return oindices

    def getInfo(self):
        cdef PetscBool stride = PETSC_FALSE
        CHKERR( ISStride(self.iset, &stride) )
        if stride == PETSC_FALSE: return None
        cdef PetscInt first = 0, step = 0
        CHKERR( ISStrideGetInfo(self.iset, &first, &step) )
        return (toInt(first), toInt(step))

    def sort(self):
        CHKERR( ISSort(self.iset) )
        return self

    def isSorted(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISSorted(self.iset, &flag) )
        return <bint> flag

    def setPermutation(self):
        CHKERR( ISSetPermutation(self.iset) )
        return self

    def isPermutation(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISPermutation(self.iset, &flag) )
        return <bint> flag

    def setIdentity(self):
        CHKERR( ISSetIdentity(self.iset) )
        return self

    def isIdentity(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISIdentity(self.iset, &flag) )
        return <bint> flag

    def equal(self, IS iset not None):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISEqual(self.iset, iset.iset, &flag) )
        return <bint> flag

    def sum(self, IS iset not None):
        cdef IS out = IS()
        CHKERR( ISSum(self.iset, iset.iset, &out.iset) )
        return out

    def expand(self, IS iset not None):
        cdef IS out = IS()
        CHKERR( ISExpand(self.iset, iset.iset, &out.iset) )
        return out

    def union(self, IS iset not None): # XXX review this
        cdef PetscBool flag1, flag2
        CHKERR( ISSorted(self.iset, &flag1) )
        CHKERR( ISSorted(iset.iset, &flag2) )
        cdef IS out = IS()
        if flag1==PETSC_TRUE and flag2==PETSC_TRUE:
            CHKERR( ISSum(self.iset, iset.iset, &out.iset) )
        else:
            CHKERR( ISExpand(self.iset, iset.iset, &out.iset) )
        return out

    def difference(self, IS iset not None):
        cdef IS out = IS()
        CHKERR( ISDifference(self.iset, iset.iset, &out.iset) )
        return out

    def complement(self, nmin, nmax):
        cdef PetscInt ival1 = asInt(nmin)
        cdef PetscInt ival2 = asInt(nmax)
        cdef IS out = IS()
        CHKERR( ISComplement(self.iset, ival1, ival2, &out.iset) )
        return out

    #

    property permutation:
        def __get__(self):
            return self.isPermutation()

    property identity:
        def __get__(self):
            return self.isIdentity()

    property sorted:
        def __get__(self):
            return self.isSorted()

    #

    property sizes:
        def __get__(self):
            return self.getSizes()

    property size:
        def __get__(self):
            return self.getSize()

    property local_size:
        def __get__(self):
            return self.getLocalSize()

    property block_size:
        def __get__(self):
            return self.getBlockSize()
    
    property array:
        def __get__(self):
            return asarray(self)

    # --- NumPy array interface (legacy) ---

    property __array_interface__:
        def __get__(self):
            cdef _IS_buffer buf = _IS_buffer(self)
            return buf.__array_interface__

# --------------------------------------------------------------------


class GLMapType(object):
    MASK = IS_GTOLM_MASK
    DROP = IS_GTOLM_DROP


# --------------------------------------------------------------------

cdef class LGMap(Object):

    MapType = GLMapType

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.lgm
        self.lgm = NULL

    #

    def view(self, Viewer viewer=None):
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( ISLocalToGlobalMappingView(self.lgm, cviewer) )

    def destroy(self):
        CHKERR( ISLocalToGlobalMappingDestroy(self.lgm) )
        self.lgm = NULL
        return self

    def create(self, indices, comm=None):
        cdef IS iset
        cdef MPI_Comm ccomm = MPI_COMM_NULL
        cdef PetscInt nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        cdef PetscLGMap newlgm = NULL
        if isinstance(indices, IS):
            iset = indices
            CHKERR( ISLocalToGlobalMappingCreateIS(iset.iset, &newlgm) )
        else:
            ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
            indices = iarray_i(indices, &nidx, &idx)
            CHKERR( ISLocalToGlobalMappingCreate(ccomm, nidx, idx, 
                                                 cm, &newlgm) )
        PetscCLEAR(self.obj); self.lgm = newlgm
        return self

    def getSize(self):
        cdef PetscInt n = 0
        CHKERR( ISLocalToGlobalMappingGetSize(self.lgm, &n) )
        return toInt(n)

    def getInfo(self):
        cdef PetscInt i, nproc = 0, *procs = NULL,
        cdef PetscInt *numprocs = NULL, **indices = NULL
        cdef object neighs = { }
        CHKERR( ISLocalToGlobalMappingGetInfo(
                self.lgm, &nproc, &procs, &numprocs, &indices) )
        try:
            for i from 0 <= i < nproc:
                neighs[toInt(procs[i])] = array_i(numprocs[i], indices[i])
        finally:
            ISLocalToGlobalMappingRestoreInfo(
                self.lgm, &nproc, &procs, &numprocs, &indices)
        return neighs

    def apply(self, indices, result=None):
        cdef IS isetin, iset
        cdef PetscInt niidx = 0, *iidx = NULL
        cdef PetscInt noidx = 0, *oidx = NULL
        if isinstance(indices, IS):
            isetin = indices; iset = IS()
            CHKERR( ISLocalToGlobalMappingApplyIS(
                    self.lgm, isetin.iset, &iset.iset) )
            return iset
        else:
            indices = iarray_i(indices, &niidx, &iidx)
            if result is None: result = empty_i(niidx)
            result  = oarray_i(result,  &noidx, &oidx)
            assert niidx == noidx, "incompatible array sizes"
            CHKERR( ISLocalToGlobalMappingApply(self.lgm, niidx, iidx, oidx) )
        return result

    def applyInverse(self, indices, map_type=None):
        cdef PetscGLMapType cmtype = IS_GTOLM_MASK
        if map_type is not None: cmtype = map_type
        cdef PetscInt n = 0, *idx = NULL
        indices = iarray_i(indices, &n, &idx)
        cdef PetscInt nout = n, *idxout = NULL
        if cmtype != IS_GTOLM_MASK:
            CHKERR( ISGlobalToLocalMappingApply(
                    self.lgm, cmtype, n, idx, &nout, NULL) )
        result = oarray_i(empty_i(nout), &nout, &idxout)
        CHKERR( ISGlobalToLocalMappingApply(
                self.lgm, cmtype, n, idx, &nout, idxout) )
        return result

    #

    property size:
        def __get__(self):
            return self.getSize()

    property info:
        def __get__(self):
            return self.getInfo()

# --------------------------------------------------------------------

del ISType
del GLMapType

# --------------------------------------------------------------------
