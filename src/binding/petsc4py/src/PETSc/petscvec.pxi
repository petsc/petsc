# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef const char* PetscVecType "VecType"
    PetscVecType VECSEQ
    PetscVecType VECMPI
    PetscVecType VECSTANDARD
    PetscVecType VECSHARED
    PetscVecType VECSEQVIENNACL
    PetscVecType VECMPIVIENNACL
    PetscVecType VECVIENNACL
    PetscVecType VECSEQCUDA
    PetscVecType VECMPICUDA
    PetscVecType VECCUDA
    PetscVecType VECSEQHIP
    PetscVecType VECMPIHIP
    PetscVecType VECHIP
    PetscVecType VECNEST
    PetscVecType VECSEQKOKKOS
    PetscVecType VECMPIKOKKOS
    PetscVecType VECKOKKOS

    ctypedef enum PetscVecOption "VecOption":
        VEC_IGNORE_OFF_PROC_ENTRIES
        VEC_IGNORE_NEGATIVE_INDICES

    int VecView(PetscVec,PetscViewer)
    int VecDestroy(PetscVec*)
    int VecCreate(MPI_Comm,PetscVec*)

    int VecSetOptionsPrefix(PetscVec,char[])
    int VecGetOptionsPrefix(PetscVec,char*[])
    int VecSetFromOptions(PetscVec)
    int VecSetUp(PetscVec)

    int VecCreateSeq(MPI_Comm,PetscInt,PetscVec*)
    int VecCreateSeqWithArray(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscVec*)
    int VecCreateSeqCUDAWithArrays(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    int VecCreateSeqHIPWithArrays(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    int VecCreateSeqViennaCLWithArrays(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    int VecCreateMPI(MPI_Comm,PetscInt,PetscInt,PetscVec*)
    int VecCreateMPIWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscVec*)
    int VecCreateMPICUDAWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    int VecCreateMPIHIPWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    int VecCreateMPIViennaCLWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    int VecCreateGhost(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscVec*)
    int VecCreateGhostWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscScalar[],PetscVec*)
    int VecCreateGhostBlock(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscVec*)
    int VecCreateGhostBlockWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscScalar[],PetscVec*)
    int VecCreateShared(MPI_Comm,PetscInt,PetscInt,PetscVec*)
    int VecCreateNest(MPI_Comm,PetscInt,PetscIS[],PetscVec[],PetscVec*)
    int VecGetType(PetscVec,PetscVecType*)
    int VecSetType(PetscVec,PetscVecType)
    int VecSetOption(PetscVec,PetscVecOption,PetscBool)
    int VecSetSizes(PetscVec,PetscInt,PetscInt)
    int VecGetSize(PetscVec,PetscInt*)
    int VecGetLocalSize(PetscVec,PetscInt*)
    int VecSetBlockSize(PetscVec,PetscInt)
    int VecGetBlockSize(PetscVec,PetscInt*)
    int VecGetOwnershipRange(PetscVec,PetscInt*,PetscInt*)
    int VecGetOwnershipRanges(PetscVec,const PetscInt*[])

    int VecGetArrayWrite(PetscVec,PetscScalar*[])
    int VecRestoreArrayWrite(PetscVec,PetscScalar*[])
    int VecGetArrayRead(PetscVec,const PetscScalar*[])
    int VecRestoreArrayRead(PetscVec,const PetscScalar*[])
    int VecGetArray(PetscVec,PetscScalar*[])
    int VecRestoreArray(PetscVec,PetscScalar*[])
    int VecPlaceArray(PetscVec,PetscScalar[])
    int VecResetArray(PetscVec)

    int VecEqual(PetscVec,PetscVec,PetscBool*)
    int VecLoad(PetscVec,PetscViewer)

    int VecDuplicate(PetscVec,PetscVec*)
    int VecCopy(PetscVec,PetscVec)
    int VecChop(PetscVec,PetscReal)

    int VecDuplicateVecs(PetscVec,PetscInt,PetscVec*[])
    int VecDestroyVecs(PetscInt,PetscVec*[])

    int VecGetValues(PetscVec,PetscInt,PetscInt[],PetscScalar[])

    int VecSetValue(PetscVec,PetscInt,PetscScalar,PetscInsertMode)
    int VecSetValues(PetscVec,PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)
    int VecSetValuesBlocked(PetscVec,PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)

    int VecSetLocalToGlobalMapping(PetscVec,PetscLGMap)
    int VecGetLocalToGlobalMapping(PetscVec,PetscLGMap*)
    int VecSetValueLocal(PetscVec,PetscInt,PetscScalar,PetscInsertMode)
    int VecSetValuesLocal(PetscVec,PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)
    int VecSetValuesBlockedLocal(PetscVec,PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)

    int VecDot(PetscVec,PetscVec,PetscScalar*)
    int VecDotBegin(PetscVec,PetscVec,PetscScalar*)
    int VecDotEnd(PetscVec,PetscVec,PetscScalar*)
    int VecTDot(PetscVec,PetscVec,PetscScalar*)
    int VecTDotBegin(PetscVec,PetscVec,PetscScalar*)
    int VecTDotEnd(PetscVec,PetscVec,PetscScalar*)
    int VecMDot(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    int VecMDotBegin(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    int VecMDotEnd(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    int VecMTDot(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    int VecMTDotBegin(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    int VecMTDotEnd(PetscVec,PetscInt,PetscVec[],PetscScalar*)

    int VecNorm(PetscVec,PetscNormType,PetscReal*)
    int VecNormBegin(PetscVec,PetscNormType,PetscReal*)
    int VecNormEnd(PetscVec,PetscNormType,PetscReal*)

    int VecAssemblyBegin(PetscVec)
    int VecAssemblyEnd(PetscVec)

    int VecZeroEntries(PetscVec)
    int VecConjugate(PetscVec)
    int VecNormalize(PetscVec,PetscReal*)
    int VecSum(PetscVec,PetscScalar*)
    int VecMax(PetscVec,PetscInt*,PetscReal*)
    int VecMin(PetscVec,PetscInt*,PetscReal*)
    int VecScale(PetscVec,PetscScalar)
    int VecCopy(PetscVec,PetscVec)
    int VecSetRandom(PetscVec,PetscRandom)
    int VecSet(PetscVec,PetscScalar)
    int VecSwap(PetscVec,PetscVec)
    int VecAXPY(PetscVec,PetscScalar,PetscVec)
    int VecAXPBY(PetscVec,PetscScalar,PetscScalar,PetscVec)
    int VecAYPX(PetscVec,PetscScalar,PetscVec)
    int VecWAXPY(PetscVec,PetscScalar,PetscVec,PetscVec)
    int VecMAXPY(PetscVec,PetscInt,PetscScalar[],PetscVec[])
    int VecPointwiseMax(PetscVec,PetscVec,PetscVec)
    int VecPointwiseMaxAbs(PetscVec,PetscVec,PetscVec)
    int VecPointwiseMin(PetscVec,PetscVec,PetscVec)
    int VecPointwiseMult(PetscVec,PetscVec,PetscVec)
    int VecPointwiseDivide(PetscVec,PetscVec,PetscVec)
    int VecMaxPointwiseDivide(PetscVec,PetscVec,PetscReal*)
    int VecShift(PetscVec,PetscScalar)
    int VecChop(PetscVec,PetscReal)
    int VecReciprocal(PetscVec)
    int VecPermute(PetscVec,PetscIS,PetscBool)
    int VecExp(PetscVec)
    int VecLog(PetscVec)
    int VecSqrtAbs(PetscVec)
    int VecAbs(PetscVec)

    int VecStrideSum(PetscVec,PetscInt,PetscScalar*)
    int VecStrideMin(PetscVec,PetscInt,PetscInt*,PetscReal*)
    int VecStrideMax(PetscVec,PetscInt,PetscInt*,PetscReal*)
    int VecStrideScale(PetscVec,PetscInt,PetscScalar)
    int VecStrideGather(PetscVec,PetscInt,PetscVec,PetscInsertMode)
    int VecStrideScatter(PetscVec,PetscInt,PetscVec,PetscInsertMode)
    int VecStrideNorm(PetscVec,PetscInt,PetscNormType,PetscReal*)

    int VecGhostGetLocalForm(PetscVec,PetscVec*)
    int VecGhostRestoreLocalForm(PetscVec,PetscVec*)
    int VecGhostUpdateBegin(PetscVec,PetscInsertMode,PetscScatterMode)
    int VecGhostUpdateEnd(PetscVec,PetscInsertMode,PetscScatterMode)
    int VecMPISetGhost(PetscVec,PetscInt,const PetscInt*)

    int VecGetSubVector(PetscVec,PetscIS,PetscVec*)
    int VecRestoreSubVector(PetscVec,PetscIS,PetscVec*)

    int VecNestGetSubVecs(PetscVec,PetscInt*,PetscVec**)
    int VecNestSetSubVecs(PetscVec,PetscInt,PetscInt*,PetscVec*)

    int VecISAXPY(PetscVec,PetscIS,PetscScalar,PetscVec)
    int VecISSet(PetscVec,PetscIS,PetscScalar)

    int VecCUDAGetArrayRead(PetscVec,const PetscScalar*[])
    int VecCUDAGetArrayWrite(PetscVec,PetscScalar*[])
    int VecCUDAGetArray(PetscVec,PetscScalar*[])
    int VecCUDARestoreArrayRead(PetscVec,const PetscScalar*[])
    int VecCUDARestoreArrayWrite(PetscVec,PetscScalar*[])
    int VecCUDARestoreArray(PetscVec,PetscScalar*[])

    int VecHIPGetArrayRead(PetscVec,const PetscScalar*[])
    int VecHIPGetArrayWrite(PetscVec,PetscScalar*[])
    int VecHIPGetArray(PetscVec,PetscScalar*[])
    int VecHIPRestoreArrayRead(PetscVec,const PetscScalar*[])
    int VecHIPRestoreArrayWrite(PetscVec,PetscScalar*[])
    int VecHIPRestoreArray(PetscVec,PetscScalar*[])

    int VecBindToCPU(PetscVec,PetscBool)
    int VecGetOffloadMask(PetscVec,PetscOffloadMask*)

    int VecViennaCLGetCLContext(PetscVec,Py_uintptr_t*)
    int VecViennaCLGetCLQueue(PetscVec,Py_uintptr_t*)
    int VecViennaCLGetCLMemRead(PetscVec,Py_uintptr_t*)
    int VecViennaCLGetCLMemWrite(PetscVec,Py_uintptr_t*)
    int VecViennaCLRestoreCLMemWrite(PetscVec)
    int VecViennaCLGetCLMem(PetscVec,Py_uintptr_t*)
    int VecViennaCLRestoreCLMem(PetscVec)

    int VecCreateSeqCUDAWithArray(MPI_Comm,PetscInt,PetscInt,const PetscScalar*,PetscVec*)
    int VecCreateMPICUDAWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscScalar*,PetscVec*)
    int VecCreateSeqHIPWithArray(MPI_Comm,PetscInt,PetscInt,const PetscScalar*,PetscVec*)
    int VecCreateMPIHIPWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscScalar*,PetscVec*)

# --------------------------------------------------------------------

cdef inline Vec ref_Vec(PetscVec vec):
    cdef Vec ob = <Vec> Vec()
    ob.vec = vec
    PetscINCREF(ob.obj)
    return ob

# --------------------------------------------------------------------

# unary operations

cdef Vec vec_pos(Vec self):
    cdef Vec vec = type(self)()
    CHKERR( VecDuplicate(self.vec, &vec.vec) )
    CHKERR( VecCopy(self.vec, vec.vec) )
    return vec

cdef Vec vec_neg(Vec self):
    cdef Vec vec = <Vec> vec_pos(self)
    CHKERR( VecScale(vec.vec, -1) )
    return vec

cdef Vec vec_abs(Vec self):
    cdef Vec vec = <Vec> vec_pos(self)
    CHKERR( VecAbs(vec.vec) )
    return vec

# inplace binary operations

cdef Vec vec_iadd(Vec self, other):
    cdef PetscScalar alpha = 1
    cdef Vec vec
    if isinstance(other, Vec):
        alpha = 1; vec = other
        CHKERR( VecAXPY(self.vec, alpha, vec.vec) )
    elif isinstance(other, tuple) or isinstance(other, list):
        other, vec = other
        alpha = asScalar(other)
        CHKERR( VecAXPY(self.vec, alpha, vec.vec) )
    else:
        alpha = asScalar(other)
        CHKERR( VecShift(self.vec, alpha) )
    return self

cdef Vec vec_isub(Vec self, other):
    cdef PetscScalar alpha = 1
    cdef Vec vec
    if isinstance(other, Vec):
        alpha = 1; vec = other
        CHKERR( VecAXPY(self.vec, -alpha, vec.vec) )
    elif isinstance(other, tuple) or isinstance(other, list):
        other, vec = other
        alpha = asScalar(other)
        CHKERR( VecAXPY(self.vec, -alpha, vec.vec) )
    else:
        alpha = asScalar(other)
        CHKERR( VecShift(self.vec, -alpha) )
    return self

cdef Vec vec_imul(Vec self, other):
    cdef PetscScalar alpha = 1
    cdef Vec vec
    if isinstance(other, Vec):
        vec = other
        CHKERR( VecPointwiseMult(self.vec, self.vec, vec.vec) )
    else:
        alpha = asScalar(other)
        CHKERR( VecScale(self.vec, alpha) )
    return self

cdef Vec vec_idiv(Vec self, other):
    cdef PetscScalar one = 1
    cdef PetscScalar alpha = 1
    cdef Vec vec
    if isinstance(other, Vec):
        vec = other
        CHKERR( VecPointwiseDivide(self.vec, self.vec, vec.vec) )
    else:
        alpha = asScalar(other)
        CHKERR( VecScale(self.vec, one/alpha) )
    return self

# binary operations

cdef Vec vec_add(Vec self, other):
    return vec_iadd(vec_pos(self), other)

cdef Vec vec_sub(Vec self, other):
    return vec_isub(vec_pos(self), other)

cdef Vec vec_mul(Vec self, other):
    return vec_imul(vec_pos(self), other)

cdef Vec vec_div(Vec self, other):
    return vec_idiv(vec_pos(self), other)

# reflected binary operations

cdef Vec vec_radd(Vec self, other):
    return vec_add(self, other)

cdef Vec vec_rsub(Vec self, other):
    cdef Vec vec = <Vec> vec_sub(self, other)
    CHKERR( VecScale(vec.vec, -1) )
    return vec

cdef Vec vec_rmul(Vec self, other):
    return vec_mul(self, other)

cdef Vec vec_rdiv(Vec self, other):
    cdef Vec vec = <Vec> vec_div(self, other)
    CHKERR( VecReciprocal(vec.vec) )
    return vec

# --------------------------------------------------------------------

cdef inline int Vec_Sizes(object size, object bsize,
                          PetscInt *b, PetscInt *n, PetscInt *N) except -1:
    Sys_Sizes(size, bsize, b, n, N)
    return 0

# --------------------------------------------------------------------

ctypedef int VecSetValuesFcn(PetscVec,
                             PetscInt,const PetscInt*,
                             const PetscScalar*,PetscInsertMode)

cdef inline VecSetValuesFcn* vecsetvalues_fcn(int blocked, int local):
    cdef VecSetValuesFcn *setvalues = NULL
    if blocked and local: setvalues = VecSetValuesBlockedLocal
    elif blocked:         setvalues = VecSetValuesBlocked
    elif local:           setvalues = VecSetValuesLocal
    else:                 setvalues = VecSetValues
    return setvalues

cdef inline int vecsetvalues(PetscVec V,
                             object oi, object ov, object oim,
                             int blocked, int local) except -1:
    # block size
    cdef PetscInt bs=1
    if blocked:
        CHKERR( VecGetBlockSize(V, &bs) )
        if bs < 1: bs = 1
    # indices and values
    cdef PetscInt ni=0, nv=0
    cdef PetscInt    *i=NULL
    cdef PetscScalar *v=NULL
    cdef object tmp1 = iarray_i(oi, &ni, &i)
    cdef object tmp2 = iarray_s(ov, &nv, &v)
    if ni*bs != nv: raise ValueError(
        "incompatible array sizes: ni=%d, nv=%d, bs=%d" %
        (toInt(ni), toInt(nv), toInt(bs)) )
    # VecSetValuesXXX function and insert mode
    cdef VecSetValuesFcn *setvalues = vecsetvalues_fcn(blocked, local)
    cdef PetscInsertMode addv = insertmode(oim)
    # actual call
    CHKERR( setvalues(V, ni, i, v, addv) )
    return 0

cdef object vecgetvalues(PetscVec vec, object oindices, object values):
    cdef PetscInt ni=0, nv=0
    cdef PetscInt    *i=NULL
    cdef PetscScalar *v=NULL
    cdef object indices = iarray_i(oindices, &ni, &i)
    if values is None:
        values = empty_s(ni)
        values.shape = indices.shape
    values = oarray_s(values, &nv, &v)
    if (ni != nv): raise ValueError(
        ("incompatible array sizes: "
         "ni=%d, nv=%d") % (toInt(ni), toInt(nv)))
    CHKERR( VecGetValues(vec, ni, i, v) )
    return values

# --------------------------------------------------------------------

cdef inline _Vec_buffer vec_getbuffer_r(Vec self):
    cdef _Vec_buffer buf = _Vec_buffer(self)
    buf.readonly = 1
    return buf

cdef inline _Vec_buffer vec_getbuffer_w(Vec self):
    cdef _Vec_buffer buf = _Vec_buffer(self)
    buf.readonly = 0
    return buf

cdef inline ndarray vec_getarray_r(Vec self):
    return asarray(vec_getbuffer_r(self))

cdef inline ndarray vec_getarray_w(Vec self):
    return asarray(vec_getbuffer_w(self))

cdef inline int vec_setarray(Vec self, object o) except -1:
    cdef PetscInt na=0, nv=0, i=0
    cdef PetscScalar *va=NULL, *vv=NULL
    cdef ndarray ary = iarray_s(o, &na, &va)
    CHKERR( VecGetLocalSize(self.vec, &nv) )
    if (na != nv) and PyArray_NDIM(ary) > 0: raise ValueError(
        "array size %d incompatible with vector local size %d" %
        (toInt(na), toInt(nv)) )
    CHKERR( VecGetArray(self.vec, &vv) )
    try:
        if PyArray_NDIM(ary) == 0:
            for i from 0 <= i < nv:
                vv[i] = va[0]
        else:
            CHKERR( PetscMemcpy(vv, va, <size_t>nv*sizeof(PetscScalar)) )
    finally:
        CHKERR( VecRestoreArray(self.vec, &vv) )
    return 0

cdef object vec_getitem(Vec self, object i):
    cdef PetscInt N=0
    if i is Ellipsis:
        return asarray(self)
    if isinstance(i, slice):
        CHKERR( VecGetSize(self.vec, &N) )
        start, stop, stride = i.indices(toInt(N))
        i = arange(start, stop, stride)
    return vecgetvalues(self.vec, i, None)

cdef int vec_setitem(Vec self, object i, object v) except -1:
    cdef PetscInt N=0
    if i is Ellipsis:
        return vec_setarray(self, v)
    if isinstance(i, slice):
        CHKERR( VecGetSize(self.vec, &N) )
        start, stop, stride = i.indices(toInt(N))
        i = arange(start, stop, stride)
    vecsetvalues(self.vec, i, v, None, 0, 0)
    return 0

# --------------------------------------------------------------------

cdef extern from "pep3118.h":
    int  PyPetscBuffer_FillInfo(Py_buffer*,
                                void*,PetscInt,char,
                                int,int) except -1
    void PyPetscBuffer_Release(Py_buffer*)

# --------------------------------------------------------------------

cdef int Vec_AcquireArray(PetscVec v, PetscScalar *a[], int ro) nogil except -1:
    if ro: CHKERR( VecGetArrayRead(v, <const PetscScalar**>a) )
    else:  CHKERR( VecGetArray(v, a) )
    return 0

cdef int Vec_ReleaseArray(PetscVec v, PetscScalar *a[], int ro) nogil except -1:
    if ro: CHKERR( VecRestoreArrayRead(v, <const PetscScalar**>a) )
    else:  CHKERR( VecRestoreArray(v, a) )
    return 0

cdef class _Vec_buffer:

    cdef PetscVec vec
    cdef PetscInt size
    cdef PetscScalar *data
    cdef bint readonly
    cdef bint hasarray

    def __cinit__(self, Vec vec, bint readonly=0):
        cdef PetscVec v = vec.vec
        CHKERR( PetscINCREF(<PetscObject*>&v) )
        self.vec = v
        self.size = 0
        self.data = NULL
        self.readonly = 1 if readonly else 0
        self.hasarray = 0

    def __dealloc__(self):
        if self.hasarray and self.vec != NULL:
            Vec_ReleaseArray(self.vec, &self.data, self.readonly)
        CHKERR( VecDestroy(&self.vec) )

    #

    cdef int acquire(self) nogil except -1:
        if not self.hasarray and self.vec != NULL:
            CHKERR( VecGetLocalSize(self.vec, &self.size) )
            Vec_AcquireArray(self.vec, &self.data, self.readonly)
            self.hasarray = 1
        return 0

    cdef int release(self) nogil except -1:
        if self.hasarray and self.vec != NULL:
            self.size = 0
            Vec_ReleaseArray(self.vec, &self.data, self.readonly)
            self.hasarray = 0
        return 0

    # buffer interface (PEP 3118)

    cdef int acquirebuffer(self, Py_buffer *view, int flags) except -1:
        self.acquire()
        PyPetscBuffer_FillInfo(view, <void*>self.data, self.size,
                               c's', self.readonly, flags)
        view.obj = self
        return 0

    cdef int releasebuffer(self, Py_buffer *view) except -1:
        PyPetscBuffer_Release(view)
        self.release()
        return 0

    def __getbuffer__(self, Py_buffer *view, int flags):
        self.acquirebuffer(view, flags)

    def __releasebuffer__(self, Py_buffer *view):
        self.releasebuffer(view)

    # 'with' statement (PEP 343)

    cdef object enter(self):
        self.acquire()
        return asarray(self)

    cdef object exit(self):
        self.release()
        return None

    def __enter__(self):
        return self.enter()

    def __exit__(self, *exc):
        return self.exit()

    # buffer interface (legacy)

    cdef Py_ssize_t getbuffer(self, void **p) except -1:
        cdef PetscInt n = 0
        if p != NULL:
            self.acquire()
            p[0] = <void*>self.data
            n = self.size
        elif self.vec != NULL:
            CHKERR( VecGetLocalSize(self.vec, &n) )
        return <Py_ssize_t>(<size_t>n*sizeof(PetscScalar))

    def __getsegcount__(self, Py_ssize_t *lenp):
        if lenp != NULL:
            lenp[0] = self.getbuffer(NULL)
        return 1

    def __getreadbuffer__(self, Py_ssize_t idx, void **p):
        if idx != 0: raise SystemError(
            "accessing non-existent buffer segment")
        return self.getbuffer(p)

    def __getwritebuffer__(self, Py_ssize_t idx, void **p):
        if idx != 0: raise SystemError(
            "accessing non-existent buffer segment")
        if self.readonly: raise TypeError(
            "Object is not writable.")
        return self.getbuffer(p)

    # NumPy array interface (legacy)

    property __array_interface__:
        def __get__(self):
            cdef PetscInt n = 0
            if self.vec != NULL:
                CHKERR( VecGetLocalSize(self.vec, &n) )
            cdef object size = toInt(n)
            cdef dtype descr = PyArray_DescrFromType(NPY_PETSC_SCALAR)
            cdef str typestr = "=%c%d" % (descr.kind, descr.itemsize)
            return dict(version=3,
                        data=self,
                        shape=(size,),
                        typestr=typestr)

# --------------------------------------------------------------------

cdef class _Vec_LocalForm:

    "Context manager for `Vec` local form"

    cdef Vec gvec
    cdef Vec lvec

    def __init__(self, Vec gvec):
        self.gvec = gvec
        self.lvec = Vec()

    def __enter__(self):
        cdef PetscVec gvec = self.gvec.vec
        CHKERR( VecGhostGetLocalForm(gvec, &self.lvec.vec) )
        return self.lvec

    def __exit__(self, *exc):
        cdef PetscVec gvec = self.gvec.vec
        CHKERR( VecGhostRestoreLocalForm(gvec, &self.lvec.vec) )
        self.lvec.vec = NULL

# --------------------------------------------------------------------

cdef extern from "Python.h":
    ctypedef void (*PyCapsule_Destructor)(object)
    bint PyCapsule_IsValid(object, const char*)
    void* PyCapsule_GetPointer(object, const char*) except? NULL
    int PyCapsule_SetName(object, const char*) except -1
    object PyCapsule_New(void*, const char*, PyCapsule_Destructor)
    int PyCapsule_CheckExact(object)

cdef extern from "stdlib.h" nogil:
   ctypedef signed long int64_t
   ctypedef unsigned long long uint64_t
   ctypedef unsigned char uint8_t
   ctypedef unsigned short uint16_t
   void free(void* ptr)
   void* malloc(size_t size)

cdef struct DLDataType:
    uint8_t code
    uint8_t bits
    uint16_t lanes

ctypedef struct DLContext:
    int device_type
    int device_id

cdef enum DLDataTypeCode:
    kDLInt = <unsigned int>0
    kDLUInt = <unsigned int>1
    kDLFloat = <unsigned int>2

cdef struct DLTensor:
    void* data
    DLContext ctx
    int ndim
    DLDataType dtype
    int64_t* shape
    int64_t* strides
    uint64_t byte_offset

cdef struct DLManagedTensor:
    DLTensor dl_tensor
    void* manager_ctx
    void (*manager_deleter)(DLManagedTensor*) nogil

cdef void pycapsule_deleter(object dltensor):
    cdef DLManagedTensor* dlm_tensor = NULL
    try:
        dlm_tensor = <DLManagedTensor *>PyCapsule_GetPointer(dltensor, 'used_dltensor')
        return             # we do not call a used capsule's deleter
    except Exception:
        dlm_tensor = <DLManagedTensor *>PyCapsule_GetPointer(dltensor, 'dltensor')
    manager_deleter(dlm_tensor)

cdef void manager_deleter(DLManagedTensor* tensor) nogil:
    if tensor.manager_ctx is NULL:
        return
    free(tensor.dl_tensor.shape)
    CHKERR( PetscDEALLOC(<PetscObject*>&tensor.manager_ctx) )
    free(tensor)
    tensor.manager_ctx = NULL

# --------------------------------------------------------------------
