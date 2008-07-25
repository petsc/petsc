# --------------------------------------------------------------------

class ISType:
    GENERAL = "general"
    BLOCK   = "block"
    STRIDE  = "stride"

# --------------------------------------------------------------------

cdef class IS(Object):

    Type = ISType

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.iset
        self.iset = NULL

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
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
        cdef PetscInt bs=PETSC_DECIDE, nidx=0, *idx=NULL
        cdef PetscIS newiset = NULL
        indices = iarray_i(indices, &nidx, &idx)
        if bsize is not None: bs = bsize
        if bs == PETSC_DECIDE:
            CHKERR( ISCreateGeneral(ccomm, nidx, idx, &newiset) )
        else:
            CHKERR( ISCreateBlock(ccomm, bs, nidx, idx, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def createGeneral(self, indices, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
        cdef PetscInt nidx=0, *idx=NULL
        cdef PetscIS newiset = NULL
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISCreateGeneral(ccomm, nidx, idx, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def createBlock(self, bsize, indices, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
        cdef PetscInt bs = bsize
        cdef PetscInt nidx=0, *idx=NULL
        cdef PetscIS newiset = NULL
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISCreateBlock(ccomm, bs, nidx, idx, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def createStride(self, size, first=None, step=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
        cdef PetscInt csize = size, cfirst = 0, cstep = 1
        cdef PetscIS newiset = NULL
        if first is not None: cfist = first
        if step  is not None: cstep = step
        CHKERR( ISCreateStride(ccomm, csize, cfirst, cstep, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def getType(self):
        cdef object otype = None
        cdef PetscISType istype
        CHKERR( ISGetType(self.iset, &istype) )
        if   istype == IS_GENERAL: otype = cp2str("general")
        elif istype == IS_BLOCK:   otype = cp2str("block")
        elif istype == IS_STRIDE:  otype = cp2str("stride")
        return otype

    def duplicate(self):
        cdef IS iset = IS()
        CHKERR( ISDuplicate(self.iset, &iset.iset) )
        return iset

    def allGather(self):
        cdef IS iset = IS()
        CHKERR( ISAllGather(self.iset, &iset.iset) )
        return iset

    def toGeneral(self):
        cdef PetscTruth flag = PETSC_FALSE
        CHKERR( ISStride(self.iset, &flag) )
        if flag == PETSC_FALSE: return self
        CHKERR( ISStrideToGeneral(self.iset) )
        return self # XXX IS_BLOCK ?

    def invertPermutation(self, nlocal=None):
        cdef PetscInt cnlocal = PETSC_DECIDE
        if nlocal is not None: cnlocal = nlocal
        cdef IS iset = IS()
        CHKERR( ISInvertPermutation(self.iset, cnlocal, &iset.iset) )
        return iset

    def getSize(self):
        cdef PetscInt N = 0
        CHKERR( ISGetSize(self.iset, &N) )
        return N

    def getLocalSize(self):
        cdef PetscInt n = 0
        CHKERR( ISGetLocalSize(self.iset, &n) )
        return n

    def getSizes(self):
        cdef PetscInt n = 0, N = 0
        CHKERR( ISGetLocalSize(self.iset, &n) )
        CHKERR( ISGetSize(self.iset, &N) )
        return (n, N)

    def getBlockSize(self):
        cdef PetscTruth flag = PETSC_FALSE
        CHKERR( ISBlock(self.iset, &flag) )
        if flag == PETSC_FALSE: return 1
        cdef PetscInt bs = 0
        CHKERR( ISBlockGetBlockSize(self.iset, &bs) )
        return bs

    def getIndices(self, out=None):
        array = asarray(self)
        if out is None: out = array
        else: out[:] = array
        return out

    def getInfo(self):
        cdef PetscTruth flag = PETSC_FALSE
        CHKERR( ISStride(self.iset, &flag) )
        if flag == PETSC_FALSE: return None
        cdef PetscInt first = 0, step = 0
        CHKERR( ISStrideGetInfo(self.iset, &first, &step) )
        return (first, step)

    def sort(self):
        CHKERR( ISSort(self.iset) )
        return self

    def isSorted(self):
        cdef PetscTruth flag = PETSC_FALSE
        CHKERR( ISSorted(self.iset, &flag) )
        return <bint> flag

    def setPermutation(self):
        CHKERR( ISSetPermutation(self.iset) )
        return self

    def isPermutation(self):
        cdef PetscTruth flag = PETSC_FALSE
        CHKERR( ISPermutation(self.iset, &flag) )
        return <bint> flag

    def setIdentity(self):
        CHKERR( ISSetIdentity(self.iset) )
        return self

    def isIdentity(self):
        cdef PetscTruth flag = PETSC_FALSE
        CHKERR( ISIdentity(self.iset, &flag) )
        return <bint> flag

    def equal(self, IS iset not None):
        cdef PetscTruth flag = PETSC_FALSE
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
        cdef PetscTruth flag1, flag2
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

    # Array Interface V3

    property __array_struct__:
        def __get__(self):
            return PetscIS_array_struct(self, self.iset)

    property array:
        def __get__(self):
            return asarray(self)

# --------------------------------------------------------------------


class GLMapType:
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
        cdef PetscInt nidx = 0
        cdef PetscInt *idx = NULL
        cdef PetscLGMap newlgm = NULL
        if typecheck(indices, IS):
            iset = indices
            CHKERR( ISLocalToGlobalMappingCreateIS(iset.iset, &newlgm) )
        else:
            ccomm = def_Comm(comm, PETSC_COMM_WORLD)
            indices = iarray_i(indices, &nidx, &idx)
            CHKERR( ISLocalToGlobalMappingCreate(ccomm, nidx, idx, &newlgm) )
        PetscCLEAR(self.obj); self.lgm = newlgm
        return self

    def getSize(self):
        cdef PetscInt n = 0
        CHKERR( ISLocalToGlobalMappingGetSize(self.lgm, &n) )
        return n

    def getInfo(self):
        cdef PetscInt i, nproc = 0, *procs = NULL,
        cdef PetscInt *numprocs = NULL, **indices = NULL
        cdef object neighs = { }
        CHKERR( ISLocalToGlobalMappingGetInfo(self.lgm, &nproc, &procs, &numprocs, &indices) )
        try:
            for i from 0 <= i < nproc:
                neighs[ procs[i] ] = array_i(numprocs[i], indices[i])
        finally:
            ISLocalToGlobalMappingRestoreInfo(self.lgm, &nproc, &procs, &numprocs, &indices)
        return neighs

    def apply(self, indices, result=None):
        cdef IS isetin, iset
        cdef PetscInt niidx = 0, *iidx = NULL
        cdef PetscInt noidx = 0, *oidx = NULL
        if typecheck(indices, IS):
            isetin = indices; iset = IS()
            CHKERR( ISLocalToGlobalMappingApplyIS(self.lgm, isetin.iset, &iset.iset) )
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
        cdef PetscInt n    = 0, *idx    = NULL
        cdef PetscInt nout = 0, *idxout = NULL
        if map_type is not None: cmtype = map_type
        indices = iarray_i(indices, &n, &idx)
        if cmtype != IS_GTOLM_MASK:
            CHKERR( ISGlobalToLocalMappingApply(self.lgm, cmtype, n, idx, &nout, NULL) )
        else: nout = n
        result = empty_i(nout); result = oarray_i(result, &nout, &idxout)
        CHKERR( ISGlobalToLocalMappingApply(self.lgm, cmtype, n, idx, &nout, idxout) )
        return result

    #

    property size:
        def __get__(self):
            return self.getSize()

    property info:
        def __get__(self):
            return self.getInfo()

# --------------------------------------------------------------------
