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

    # buffer interface (PEP 3118)

    def __getbuffer__(self, Py_buffer *view, int flags):
        cdef _IS_buffer buf = _IS_buffer(self)
        buf.acquirebuffer(view, flags)

    def __releasebuffer__(self, Py_buffer *view):
        cdef _IS_buffer buf = <_IS_buffer>(view.obj)
        buf.releasebuffer(view)
        <void>self # unused


    # 'with' statement (PEP 343)

    def __enter__(self):
        cdef _IS_buffer buf = _IS_buffer(self)
        self.set_attr('__buffer__', buf)
        return buf.enter()

    def __exit__(self, *exc):
        cdef _IS_buffer buf = self.get_attr('__buffer__')
        self.set_attr('__buffer__', None)
        return buf.exit()
    #

    def view(self, Viewer viewer=None):
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( ISView(self.iset, cviewer) )

    def destroy(self):
        CHKERR( ISDestroy(&self.iset) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscIS newiset = NULL
        CHKERR( ISCreate(ccomm, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def setType(self, is_type):
        cdef PetscISType cval = NULL
        is_type = str2bytes(is_type, &cval)
        CHKERR( ISSetType(self.iset, cval) )

    def getType(self):
        cdef PetscISType cval = NULL
        CHKERR( ISGetType(self.iset, &cval) )
        return bytes2str(cval)

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
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        cdef PetscIS newiset = NULL
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISCreateBlock(ccomm, bs, nidx, idx, cm, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def createStride(self, size, first=0, step=0, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt csize  = asInt(size)
        cdef PetscInt cfirst = asInt(first)
        cdef PetscInt cstep  = asInt(step)
        cdef PetscIS newiset = NULL
        CHKERR( ISCreateStride(ccomm, csize, cfirst, cstep, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def duplicate(self):
        cdef IS iset = type(self)()
        CHKERR( ISDuplicate(self.iset, &iset.iset) )
        return iset

    def copy(self, IS result=None):
        if result is None:
            result = type(self)()
        if result.iset == NULL:
            CHKERR( ISDuplicate(self.iset, &result.iset) )
        CHKERR( ISCopy(self.iset, result.iset) )
        return result

    def load(self, Viewer viewer):
        cdef MPI_Comm comm = MPI_COMM_NULL
        cdef PetscObject obj = <PetscObject>(viewer.vwr)
        if self.iset == NULL:
            CHKERR( PetscObjectGetComm(obj, &comm) )
            CHKERR( ISCreate(comm, &self.iset) )
        CHKERR( ISLoad(self.iset, viewer.vwr) )
        return self

    def allGather(self):
        cdef IS iset = IS()
        CHKERR( ISAllGather(self.iset, &iset.iset) )
        return iset

    def toGeneral(self):
        CHKERR( ISToGeneral(self.iset) )
        return self

    def buildTwoSided(self, IS toindx=None):
        cdef PetscIS ctoindx = NULL
        if toindx is not None: ctoindx = toindx.iset
        cdef IS result = IS()
        CHKERR( ISBuildTwoSided(self.iset, ctoindx, &result.iset) )
        return result

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
        CHKERR( ISGetBlockSize(self.iset, &bs) )
        return toInt(bs)

    def setBlockSize(self, bs):
        cdef PetscInt cbs = asInt(bs)
        CHKERR( ISSetBlockSize(self.iset, cbs) )

    def sort(self):
        CHKERR( ISSort(self.iset) )
        return self

    def isSorted(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISSorted(self.iset, &flag) )
        return toBool(flag)

    def setPermutation(self):
        CHKERR( ISSetPermutation(self.iset) )
        return self

    def isPermutation(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISPermutation(self.iset, &flag) )
        return toBool(flag)

    def setIdentity(self):
        CHKERR( ISSetIdentity(self.iset) )
        return self

    def isIdentity(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISIdentity(self.iset, &flag) )
        return toBool(flag)

    def equal(self, IS iset):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISEqual(self.iset, iset.iset, &flag) )
        return toBool(flag)

    def sum(self, IS iset):
        cdef IS out = IS()
        CHKERR( ISSum(self.iset, iset.iset, &out.iset) )
        return out

    def expand(self, IS iset):
        cdef IS out = IS()
        CHKERR( ISExpand(self.iset, iset.iset, &out.iset) )
        return out

    def union(self, IS iset): # XXX review this
        cdef PetscBool flag1=PETSC_FALSE, flag2=PETSC_FALSE
        CHKERR( ISSorted(self.iset, &flag1) )
        CHKERR( ISSorted(iset.iset, &flag2) )
        cdef IS out = IS()
        if flag1==PETSC_TRUE and flag2==PETSC_TRUE:
            CHKERR( ISSum(self.iset, iset.iset, &out.iset) )
        else:
            CHKERR( ISExpand(self.iset, iset.iset, &out.iset) )
        return out

    def difference(self, IS iset):
        cdef IS out = IS()
        CHKERR( ISDifference(self.iset, iset.iset, &out.iset) )
        return out

    def complement(self, nmin, nmax):
        cdef PetscInt cnmin = asInt(nmin)
        cdef PetscInt cnmax = asInt(nmax)
        cdef IS out = IS()
        CHKERR( ISComplement(self.iset, cnmin, cnmax, &out.iset) )
        return out

    def embed(self, IS iset, drop):
        cdef PetscBool bval = drop
        cdef IS out = IS()
        CHKERR( ISEmbed(self.iset, iset.iset, bval, &out.iset) )
        return out

    def renumber(self, IS mult=None):
        cdef PetscIS mlt = NULL
        if mult is not None: mlt = mult.iset
        cdef IS out = IS()
        cdef PetscInt n = 0
        CHKERR( ISRenumber(self.iset, mlt, &n, &out.iset) )
        return (toInt(n), out)
    #

    def setIndices(self, indices):
        cdef PetscInt nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISGeneralSetIndices(self.iset, nidx, idx, cm) )

    def getIndices(self):
        cdef PetscInt size = 0
        cdef const PetscInt *indices = NULL
        CHKERR( ISGetLocalSize(self.iset, &size) )
        CHKERR( ISGetIndices(self.iset, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size, indices)
        finally:
            CHKERR( ISRestoreIndices(self.iset, &indices) )
        return oindices

    def setBlockIndices(self, bsize, indices):
        cdef PetscInt bs = asInt(bsize)
        cdef PetscInt nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISBlockSetIndices(self.iset, bs, nidx, idx, cm) )

    def getBlockIndices(self):
        cdef PetscInt size = 0, bs = 1
        cdef const PetscInt *indices = NULL
        CHKERR( ISGetLocalSize(self.iset, &size) )
        CHKERR( ISGetBlockSize(self.iset, &bs) )
        CHKERR( ISBlockGetIndices(self.iset, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size//bs, indices)
        finally:
            CHKERR( ISBlockRestoreIndices(self.iset, &indices) )
        return oindices

    def setStride(self, size, first=0, step=1):
        cdef PetscInt csize = asInt(size)
        cdef PetscInt cfirst = asInt(first)
        cdef PetscInt cstep = asInt(step)
        CHKERR( ISStrideSetStride(self.iset, csize, cfirst, cstep) )

    def getStride(self):
        cdef PetscInt size=0, first=0, step=0
        CHKERR( ISGetLocalSize(self.iset, &size) )
        CHKERR( ISStrideGetInfo(self.iset, &first, &step) )
        return (toInt(size), toInt(first), toInt(step))

    def getInfo(self):
        cdef PetscInt first = 0, step = 0
        CHKERR( ISStrideGetInfo(self.iset, &first, &step) )
        return (toInt(first), toInt(step))

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

    property indices:
        def __get__(self):
            return self.getIndices()

    property array:
        def __get__(self):
            return asarray(self)

    # --- NumPy array interface (legacy) ---

    property __array_interface__:
        def __get__(self):
            cdef _IS_buffer buf = _IS_buffer(self)
            return buf.__array_interface__

# --------------------------------------------------------------------


class GLMapMode(object):
    MASK = PETSC_IS_GTOLM_MASK
    DROP = PETSC_IS_GTOLM_DROP


class LGMapType(object):
    BASIC = S_(ISLOCALTOGLOBALMAPPINGBASIC)
    HASH  = S_(ISLOCALTOGLOBALMAPPINGHASH)


# --------------------------------------------------------------------

cdef class LGMap(Object):

    MapMode = GLMapMode

    Type = LGMapType
    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.lgm
        self.lgm = NULL

    def __call__(self, indices, result=None):
        self.apply(indices, result)

    #

    def setType(self, lgmap_type):
        cdef PetscISLocalToGlobalMappingType cval = NULL
        lgmap_type = str2bytes(lgmap_type, &cval)
        CHKERR( ISLocalToGlobalMappingSetType(self.lgm, cval) )

    def setFromOptions(self):
        CHKERR( ISLocalToGlobalMappingSetFromOptions(self.lgm) )

    def view(self, Viewer viewer=None):
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( ISLocalToGlobalMappingView(self.lgm, cviewer) )

    def destroy(self):
        CHKERR( ISLocalToGlobalMappingDestroy(&self.lgm) )
        return self

    def create(self, indices, bsize=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs = 1, nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        cdef PetscLGMap newlgm = NULL
        if bsize is not None: bs = asInt(bsize)
        if bs == PETSC_DECIDE: bs = 1
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISLocalToGlobalMappingCreate(
                ccomm, bs, nidx, idx, cm, &newlgm) )
        PetscCLEAR(self.obj); self.lgm = newlgm
        return self

    def createIS(self, IS iset):
        cdef PetscLGMap newlgm = NULL
        CHKERR( ISLocalToGlobalMappingCreateIS(
            iset.iset, &newlgm) )
        PetscCLEAR(self.obj); self.lgm = newlgm
        return self

    def createSF(self, SF sf, start):
        cdef PetscLGMap newlgm = NULL
        cdef PetscInt cstart = asInt(start)
        CHKERR( ISLocalToGlobalMappingCreateSF(sf.sf, cstart, &newlgm) )
        PetscCLEAR(self.obj); self.lgm = newlgm
        return self

    def getSize(self):
        cdef PetscInt n = 0
        CHKERR( ISLocalToGlobalMappingGetSize(self.lgm, &n) )
        return toInt(n)

    def getBlockSize(self):
        cdef PetscInt bs = 1
        CHKERR( ISLocalToGlobalMappingGetBlockSize(self.lgm, &bs) )
        return toInt(bs)

    def getIndices(self):
        cdef PetscInt size = 0
        cdef const PetscInt *indices = NULL
        CHKERR( ISLocalToGlobalMappingGetSize(
                self.lgm, &size) )
        CHKERR( ISLocalToGlobalMappingGetIndices(
                self.lgm, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size, indices)
        finally:
            CHKERR( ISLocalToGlobalMappingRestoreIndices(
                    self.lgm, &indices) )
        return oindices

    def getBlockIndices(self):
        cdef PetscInt size = 0, bs = 1
        cdef const PetscInt *indices = NULL
        CHKERR( ISLocalToGlobalMappingGetSize(
                self.lgm, &size) )
        CHKERR( ISLocalToGlobalMappingGetBlockSize(
                self.lgm, &bs) )
        CHKERR( ISLocalToGlobalMappingGetBlockIndices(
                self.lgm, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size//bs, indices)
        finally:
            CHKERR( ISLocalToGlobalMappingRestoreBlockIndices(
                    self.lgm, &indices) )
        return oindices

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

    def getBlockInfo(self):
        cdef PetscInt i, nproc = 0, *procs = NULL,
        cdef PetscInt *numprocs = NULL, **indices = NULL
        cdef object neighs = { }
        CHKERR( ISLocalToGlobalMappingGetBlockInfo(
                self.lgm, &nproc, &procs, &numprocs, &indices) )
        try:
            for i from 0 <= i < nproc:
                neighs[toInt(procs[i])] = array_i(numprocs[i], indices[i])
        finally:
            ISLocalToGlobalMappingRestoreBlockInfo(
                self.lgm, &nproc, &procs, &numprocs, &indices)
        return neighs

    #

    def apply(self, indices, result=None):
        cdef PetscInt niidx = 0, *iidx = NULL
        cdef PetscInt noidx = 0, *oidx = NULL
        indices = iarray_i(indices, &niidx, &iidx)
        if result is None: result = empty_i(niidx)
        result  = oarray_i(result,  &noidx, &oidx)
        assert niidx == noidx, "incompatible array sizes"
        CHKERR( ISLocalToGlobalMappingApply(
            self.lgm, niidx, iidx, oidx) )
        return result

    def applyBlock(self, indices, result=None):
        cdef PetscInt niidx = 0, *iidx = NULL
        cdef PetscInt noidx = 0, *oidx = NULL
        indices = iarray_i(indices, &niidx, &iidx)
        if result is None: result = empty_i(niidx)
        result  = oarray_i(result,  &noidx, &oidx)
        assert niidx == noidx, "incompatible array sizes"
        CHKERR( ISLocalToGlobalMappingApplyBlock(
            self.lgm, niidx, iidx, oidx) )
        return result

    def applyIS(self, IS iset):
        cdef IS result = IS()
        CHKERR( ISLocalToGlobalMappingApplyIS(
            self.lgm, iset.iset, &result.iset) )
        return result

    def applyInverse(self, indices, mode=None):
        cdef PetscGLMapMode cmode = PETSC_IS_GTOLM_MASK
        if mode is not None: cmode = mode
        cdef PetscInt n = 0, *idx = NULL
        indices = iarray_i(indices, &n, &idx)
        cdef PetscInt nout = n, *idxout = NULL
        if cmode != PETSC_IS_GTOLM_MASK:
            CHKERR( ISGlobalToLocalMappingApply(
                    self.lgm, cmode, n, idx, &nout, NULL) )
        result = oarray_i(empty_i(nout), &nout, &idxout)
        CHKERR( ISGlobalToLocalMappingApply(
                self.lgm, cmode, n, idx, &nout, idxout) )
        return result

    def applyBlockInverse(self, indices, mode=None):
        cdef PetscGLMapMode cmode = PETSC_IS_GTOLM_MASK
        if mode is not None: cmode = mode
        cdef PetscInt n = 0, *idx = NULL
        indices = iarray_i(indices, &n, &idx)
        cdef PetscInt nout = n, *idxout = NULL
        if cmode != PETSC_IS_GTOLM_MASK:
            CHKERR( ISGlobalToLocalMappingApply(
                    self.lgm, cmode, n, idx, &nout, NULL) )
        result = oarray_i(empty_i(nout), &nout, &idxout)
        CHKERR( ISGlobalToLocalMappingApplyBlock(
                self.lgm, cmode, n, idx, &nout, idxout) )
        return result
    #

    property size:
        def __get__(self):
            return self.getSize()

    property block_size:
        def __get__(self):
            return self.getBlockSize()

    property indices:
        def __get__(self):
            return self.getIndices()

    property block_indices:
        def __get__(self):
            return self.getBlockIndices()

    property info:
        def __get__(self):
            return self.getInfo()

    property block_info:
        def __get__(self):
            return self.getBlockInfo()

# --------------------------------------------------------------------

del ISType
del GLMapMode
del LGMapType
# --------------------------------------------------------------------
