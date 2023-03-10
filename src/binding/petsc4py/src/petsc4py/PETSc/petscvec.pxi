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

    PetscErrorCode VecView(PetscVec,PetscViewer)
    PetscErrorCode VecDestroy(PetscVec*)
    PetscErrorCode VecCreate(MPI_Comm,PetscVec*)

    PetscErrorCode VecSetOptionsPrefix(PetscVec,char[])
    PetscErrorCode VecAppendOptionsPrefix(PetscVec,char[])
    PetscErrorCode VecGetOptionsPrefix(PetscVec,char*[])
    PetscErrorCode VecSetFromOptions(PetscVec)
    PetscErrorCode VecSetUp(PetscVec)

    PetscErrorCode VecCreateSeq(MPI_Comm,PetscInt,PetscVec*)
    PetscErrorCode VecCreateSeqWithArray(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateSeqCUDAWithArrays(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateSeqHIPWithArrays(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateSeqViennaCLWithArrays(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateMPI(MPI_Comm,PetscInt,PetscInt,PetscVec*)
    PetscErrorCode VecCreateMPIWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateMPICUDAWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateMPIHIPWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateMPIViennaCLWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateGhost(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscVec*)
    PetscErrorCode VecCreateGhostWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateGhostBlock(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscVec*)
    PetscErrorCode VecCreateGhostBlockWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscScalar[],PetscVec*)
    PetscErrorCode VecCreateShared(MPI_Comm,PetscInt,PetscInt,PetscVec*)
    PetscErrorCode VecCreateNest(MPI_Comm,PetscInt,PetscIS[],PetscVec[],PetscVec*)
    PetscErrorCode VecGetType(PetscVec,PetscVecType*)
    PetscErrorCode VecSetType(PetscVec,PetscVecType)
    PetscErrorCode VecSetOption(PetscVec,PetscVecOption,PetscBool)
    PetscErrorCode VecSetSizes(PetscVec,PetscInt,PetscInt)
    PetscErrorCode VecGetSize(PetscVec,PetscInt*)
    PetscErrorCode VecGetLocalSize(PetscVec,PetscInt*)
    PetscErrorCode VecSetBlockSize(PetscVec,PetscInt)
    PetscErrorCode VecGetBlockSize(PetscVec,PetscInt*)
    PetscErrorCode VecGetOwnershipRange(PetscVec,PetscInt*,PetscInt*)
    PetscErrorCode VecGetOwnershipRanges(PetscVec,const PetscInt*[])

    PetscErrorCode VecCreateLocalVector(PetscVec,PetscVec*)
    PetscErrorCode VecGetLocalVector(PetscVec,PetscVec)
    PetscErrorCode VecRestoreLocalVector(PetscVec,PetscVec)
    PetscErrorCode VecGetLocalVectorRead(PetscVec,PetscVec)
    PetscErrorCode VecRestoreLocalVectorRead(PetscVec,PetscVec)

    PetscErrorCode VecGetArrayWrite(PetscVec,PetscScalar*[])
    PetscErrorCode VecRestoreArrayWrite(PetscVec,PetscScalar*[])
    PetscErrorCode VecGetArrayRead(PetscVec,const PetscScalar*[])
    PetscErrorCode VecRestoreArrayRead(PetscVec,const PetscScalar*[])
    PetscErrorCode VecGetArray(PetscVec,PetscScalar*[])
    PetscErrorCode VecRestoreArray(PetscVec,PetscScalar*[])
    PetscErrorCode VecPlaceArray(PetscVec,PetscScalar[])
    PetscErrorCode VecResetArray(PetscVec)
    PetscErrorCode VecGetArrayWriteAndMemType(PetscVec,PetscScalar*[],PetscMemType*)
    PetscErrorCode VecRestoreArrayWriteAndMemType(PetscVec,PetscScalar*[])
    PetscErrorCode VecGetArrayReadAndMemType(PetscVec,const PetscScalar*[],PetscMemType*)
    PetscErrorCode VecRestoreArrayReadAndMemType(PetscVec,const PetscScalar*[])
    PetscErrorCode VecGetArrayAndMemType(PetscVec,PetscScalar*[],PetscMemType*)
    PetscErrorCode VecRestoreArrayAndMemType(PetscVec,PetscScalar*[])

    PetscErrorCode VecEqual(PetscVec,PetscVec,PetscBool*)
    PetscErrorCode VecLoad(PetscVec,PetscViewer)

    PetscErrorCode VecDuplicate(PetscVec,PetscVec*)
    PetscErrorCode VecCopy(PetscVec,PetscVec)
    PetscErrorCode VecChop(PetscVec,PetscReal)

    PetscErrorCode VecDuplicateVecs(PetscVec,PetscInt,PetscVec*[])
    PetscErrorCode VecDestroyVecs(PetscInt,PetscVec*[])

    PetscErrorCode VecGetValues(PetscVec,PetscInt,PetscInt[],PetscScalar[])

    PetscErrorCode VecSetValue(PetscVec,PetscInt,PetscScalar,PetscInsertMode)
    PetscErrorCode VecSetValues(PetscVec,PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)
    PetscErrorCode VecSetValuesBlocked(PetscVec,PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)

    PetscErrorCode VecSetLocalToGlobalMapping(PetscVec,PetscLGMap)
    PetscErrorCode VecGetLocalToGlobalMapping(PetscVec,PetscLGMap*)
    PetscErrorCode VecSetValueLocal(PetscVec,PetscInt,PetscScalar,PetscInsertMode)
    PetscErrorCode VecSetValuesLocal(PetscVec,PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)
    PetscErrorCode VecSetValuesBlockedLocal(PetscVec,PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)

    PetscErrorCode VecDot(PetscVec,PetscVec,PetscScalar*)
    PetscErrorCode VecDotBegin(PetscVec,PetscVec,PetscScalar*)
    PetscErrorCode VecDotEnd(PetscVec,PetscVec,PetscScalar*)
    PetscErrorCode VecTDot(PetscVec,PetscVec,PetscScalar*)
    PetscErrorCode VecTDotBegin(PetscVec,PetscVec,PetscScalar*)
    PetscErrorCode VecTDotEnd(PetscVec,PetscVec,PetscScalar*)
    PetscErrorCode VecMDot(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    PetscErrorCode VecMDotBegin(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    PetscErrorCode VecMDotEnd(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    PetscErrorCode VecMTDot(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    PetscErrorCode VecMTDotBegin(PetscVec,PetscInt,PetscVec[],PetscScalar*)
    PetscErrorCode VecMTDotEnd(PetscVec,PetscInt,PetscVec[],PetscScalar*)

    PetscErrorCode VecNorm(PetscVec,PetscNormType,PetscReal*)
    PetscErrorCode VecNormBegin(PetscVec,PetscNormType,PetscReal*)
    PetscErrorCode VecNormEnd(PetscVec,PetscNormType,PetscReal*)

    PetscErrorCode VecAssemblyBegin(PetscVec)
    PetscErrorCode VecAssemblyEnd(PetscVec)

    PetscErrorCode VecZeroEntries(PetscVec)
    PetscErrorCode VecConjugate(PetscVec)
    PetscErrorCode VecNormalize(PetscVec,PetscReal*)
    PetscErrorCode VecSum(PetscVec,PetscScalar*)
    PetscErrorCode VecMax(PetscVec,PetscInt*,PetscReal*)
    PetscErrorCode VecMin(PetscVec,PetscInt*,PetscReal*)
    PetscErrorCode VecScale(PetscVec,PetscScalar)
    PetscErrorCode VecCopy(PetscVec,PetscVec)
    PetscErrorCode VecSetRandom(PetscVec,PetscRandom)
    PetscErrorCode VecSet(PetscVec,PetscScalar)
    PetscErrorCode VecSwap(PetscVec,PetscVec)
    PetscErrorCode VecAXPY(PetscVec,PetscScalar,PetscVec)
    PetscErrorCode VecAXPBY(PetscVec,PetscScalar,PetscScalar,PetscVec)
    PetscErrorCode VecAYPX(PetscVec,PetscScalar,PetscVec)
    PetscErrorCode VecWAXPY(PetscVec,PetscScalar,PetscVec,PetscVec)
    PetscErrorCode VecMAXPY(PetscVec,PetscInt,PetscScalar[],PetscVec[])
    PetscErrorCode VecPointwiseMax(PetscVec,PetscVec,PetscVec)
    PetscErrorCode VecPointwiseMaxAbs(PetscVec,PetscVec,PetscVec)
    PetscErrorCode VecPointwiseMin(PetscVec,PetscVec,PetscVec)
    PetscErrorCode VecPointwiseMult(PetscVec,PetscVec,PetscVec)
    PetscErrorCode VecPointwiseDivide(PetscVec,PetscVec,PetscVec)
    PetscErrorCode VecMaxPointwiseDivide(PetscVec,PetscVec,PetscReal*)
    PetscErrorCode VecShift(PetscVec,PetscScalar)
    PetscErrorCode VecChop(PetscVec,PetscReal)
    PetscErrorCode VecReciprocal(PetscVec)
    PetscErrorCode VecPermute(PetscVec,PetscIS,PetscBool)
    PetscErrorCode VecExp(PetscVec)
    PetscErrorCode VecLog(PetscVec)
    PetscErrorCode VecSqrtAbs(PetscVec)
    PetscErrorCode VecAbs(PetscVec)

    PetscErrorCode VecStrideMin(PetscVec,PetscInt,PetscInt*,PetscReal*)
    PetscErrorCode VecStrideMax(PetscVec,PetscInt,PetscInt*,PetscReal*)
    PetscErrorCode VecStrideScale(PetscVec,PetscInt,PetscScalar)
    PetscErrorCode VecStrideGather(PetscVec,PetscInt,PetscVec,PetscInsertMode)
    PetscErrorCode VecStrideScatter(PetscVec,PetscInt,PetscVec,PetscInsertMode)
    PetscErrorCode VecStrideNorm(PetscVec,PetscInt,PetscNormType,PetscReal*)

    PetscErrorCode VecGhostGetLocalForm(PetscVec,PetscVec*)
    PetscErrorCode VecGhostRestoreLocalForm(PetscVec,PetscVec*)
    PetscErrorCode VecGhostUpdateBegin(PetscVec,PetscInsertMode,PetscScatterMode)
    PetscErrorCode VecGhostUpdateEnd(PetscVec,PetscInsertMode,PetscScatterMode)
    PetscErrorCode VecMPISetGhost(PetscVec,PetscInt,const PetscInt*)

    PetscErrorCode VecGetSubVector(PetscVec,PetscIS,PetscVec*)
    PetscErrorCode VecRestoreSubVector(PetscVec,PetscIS,PetscVec*)

    PetscErrorCode VecNestGetSubVecs(PetscVec,PetscInt*,PetscVec**)
    PetscErrorCode VecNestSetSubVecs(PetscVec,PetscInt,PetscInt*,PetscVec*)

    PetscErrorCode VecISAXPY(PetscVec,PetscIS,PetscScalar,PetscVec)
    PetscErrorCode VecISSet(PetscVec,PetscIS,PetscScalar)

    PetscErrorCode VecCUDAGetArrayRead(PetscVec,const PetscScalar*[])
    PetscErrorCode VecCUDAGetArrayWrite(PetscVec,PetscScalar*[])
    PetscErrorCode VecCUDAGetArray(PetscVec,PetscScalar*[])
    PetscErrorCode VecCUDARestoreArrayRead(PetscVec,const PetscScalar*[])
    PetscErrorCode VecCUDARestoreArrayWrite(PetscVec,PetscScalar*[])
    PetscErrorCode VecCUDARestoreArray(PetscVec,PetscScalar*[])

    PetscErrorCode VecHIPGetArrayRead(PetscVec,const PetscScalar*[])
    PetscErrorCode VecHIPGetArrayWrite(PetscVec,PetscScalar*[])
    PetscErrorCode VecHIPGetArray(PetscVec,PetscScalar*[])
    PetscErrorCode VecHIPRestoreArrayRead(PetscVec,const PetscScalar*[])
    PetscErrorCode VecHIPRestoreArrayWrite(PetscVec,PetscScalar*[])
    PetscErrorCode VecHIPRestoreArray(PetscVec,PetscScalar*[])

    PetscErrorCode VecBindToCPU(PetscVec,PetscBool)
    PetscErrorCode VecBoundToCPU(PetscVec,PetscBool*)
    PetscErrorCode VecGetOffloadMask(PetscVec,PetscOffloadMask*)

    PetscErrorCode VecViennaCLGetCLContext(PetscVec,Py_uintptr_t*)
    PetscErrorCode VecViennaCLGetCLQueue(PetscVec,Py_uintptr_t*)
    PetscErrorCode VecViennaCLGetCLMemRead(PetscVec,Py_uintptr_t*)
    PetscErrorCode VecViennaCLGetCLMemWrite(PetscVec,Py_uintptr_t*)
    PetscErrorCode VecViennaCLRestoreCLMemWrite(PetscVec)
    PetscErrorCode VecViennaCLGetCLMem(PetscVec,Py_uintptr_t*)
    PetscErrorCode VecViennaCLRestoreCLMem(PetscVec)

    PetscErrorCode VecCreateSeqCUDAWithArray(MPI_Comm,PetscInt,PetscInt,const PetscScalar*,PetscVec*)
    PetscErrorCode VecCreateMPICUDAWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscScalar*,PetscVec*)
    PetscErrorCode VecCreateSeqHIPWithArray(MPI_Comm,PetscInt,PetscInt,const PetscScalar*,PetscVec*)
    PetscErrorCode VecCreateMPIHIPWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,const PetscScalar*,PetscVec*)

cdef extern from * nogil: # custom.h
    PetscErrorCode VecStrideSum(PetscVec,PetscInt,PetscScalar*)
    PetscErrorCode VecGetCurrentMemType(PetscVec,PetscMemType*)

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

ctypedef PetscErrorCode VecSetValuesFcn(PetscVec,
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

cdef vec_get_dlpack_ctx(Vec self):
    cdef object ctx0 = self.get_attr('__dltensor_ctx__')
    cdef PetscInt n = 0
    cdef int64_t ndim = 1
    cdef int64_t* shape_arr = NULL
    cdef int64_t* strides_arr = NULL
    cdef object s1 = None
    cdef object s2 = None
    cdef PetscInt devId = 0
    cdef PetscMemType mtype = PETSC_MEMTYPE_HOST
    if ctx0 is None: # First time in, create a linear memory view
        s1 = oarray_p(empty_p(ndim), NULL, <void**>&shape_arr)
        s2 = oarray_p(empty_p(ndim), NULL, <void**>&strides_arr)
        CHKERR( VecGetLocalSize(self.vec, &n) )
        shape_arr[0] = <int64_t>n
        strides_arr[0] = 1
    else:
        (_, _, ndim, s1, s2) = ctx0

    devType_ = { PETSC_MEMTYPE_HOST : kDLCPU, PETSC_MEMTYPE_CUDA : kDLCUDA, PETSC_MEMTYPE_HIP : kDLROCM }
    CHKERR( VecGetCurrentMemType(self.vec, &mtype) )
    dtype = devType_.get(mtype, kDLCPU)
    if dtype != kDLCPU:
        CHKERR( PetscObjectGetDeviceId(<PetscObject>self.vec, &devId) )
    ctx0 = (dtype, devId, ndim, s1, s2)
    self.set_attr('__dltensor_ctx__', ctx0)
    return ctx0

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
