# --------------------------------------------------------------------

cdef extern from "petscvec.h" nogil:

    ctypedef char* PetscVecType "const char*"
    PetscVecType VECSEQ
    PetscVecType VECMPI
    PetscVecType VECFETI
    PetscVecType VECSHARED
    PetscVecType VECSIEVE

    ctypedef enum PetscVecOption "VecOption":
        VEC_IGNORE_OFF_PROC_ENTRIES
        VEC_IGNORE_NEGATIVE_INDICES

    int VecView(PetscVec,PetscViewer)
    int VecDestroy(PetscVec)
    int VecCreate(MPI_Comm,PetscVec*)

    int VecSetOptionsPrefix(PetscVec,char[])
    int VecGetOptionsPrefix(PetscVec,char*[])
    int VecSetFromOptions(PetscVec)
    int VecSetUp(PetscVec)

    int VecCreateSeq(MPI_Comm,PetscInt,PetscVec*)
    int VecCreateSeqWithArray(MPI_Comm,PetscInt,PetscScalar[],PetscVec*)
    int VecCreateMPI(MPI_Comm,PetscInt,PetscInt,PetscVec*)
    int VecCreateMPIWithArray(MPI_Comm,PetscInt,PetscInt,PetscScalar[],PetscVec*)
    int VecCreateGhost(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscVec*)
    int VecCreateGhostWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt[],PetscScalar[],PetscVec*)
    int VecCreateGhostBlock(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscVec*)
    int VecCreateGhostBlockWithArray(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscScalar[],PetscVec*)
    int VecCreateShared(MPI_Comm,PetscInt,PetscInt,PetscVec*)
    int VecGetType(PetscVec,PetscVecType*)
    int VecSetType(PetscVec,PetscVecType)
    int VecSetOption(PetscVec,PetscVecOption,PetscTruth)
    int VecSetSizes(PetscVec,PetscInt,PetscInt)
    int VecGetSize(PetscVec,PetscInt*)
    int VecGetLocalSize(PetscVec,PetscInt*)
    int VecSetBlockSize(PetscVec,PetscInt)
    int VecGetBlockSize(PetscVec,PetscInt*)
    int VecGetOwnershipRange(PetscVec,PetscInt*,PetscInt*)
    int VecGetOwnershipRanges(PetscVec,const_PetscInt*[])

    int VecGetArray(PetscVec,PetscScalar*[])
    int VecRestoreArray(PetscVec,PetscScalar*[])
    int VecGetArrayC(PetscVec,PetscScalar*[])
    int VecRestoreArrayC(PetscVec,PetscScalar*[])
    int VecPlaceArray(PetscVec,PetscScalar[])
    int VecResetArray(PetscVec)

    int VecEqual(PetscVec,PetscVec,PetscTruth*)
    int VecLoad(PetscViewer,PetscVec)

    int VecDuplicate(PetscVec,PetscVec*)
    int VecCopy(PetscVec,PetscVec)

    int VecDuplicateVecs(PetscVec,PetscInt,PetscVec*[])
    int VecDestroyVecs(PetscVec[],PetscInt)

    int VecGetValues(PetscVec,PetscInt,PetscInt[],PetscScalar[])

    int VecSetValue(PetscVec,PetscInt,PetscScalar,PetscInsertMode)
    int VecSetValues(PetscVec,PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)
    int VecSetValuesBlocked(PetscVec,PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)

    int VecSetLocalToGlobalMapping(PetscVec,PetscLGMap)
    int VecSetLocalToGlobalMappingBlock(PetscVec,PetscLGMap)

    int VecSetValueLocal(PetscVec,PetscInt,PetscScalar,PetscInsertMode)
    int VecSetValuesLocal(PetscVec,PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)
    int VecSetValuesBlockedLocal(PetscVec,PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)

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
    int VecReciprocal(PetscVec)
    int VecPermute(PetscVec,PetscIS,PetscTruth)
    int VecExp(PetscVec)
    int VecLog(PetscVec)
    int VecSqrtAbs(PetscVec)
    int VecAbs(PetscVec)

    int VecStrideMin(PetscVec,PetscInt,PetscInt*,PetscReal*)
    int VecStrideMax(PetscVec,PetscInt,PetscInt*,PetscReal*)
    int VecStrideScale(PetscVec,PetscInt,PetscScalar)
    int VecStrideGather(PetscVec,PetscInt,PetscVec,PetscInsertMode)
    int VecStrideScatter(PetscVec,PetscInt,PetscVec,PetscInsertMode)
    int VecStrideNorm(PetscVec,PetscInt,PetscNormType,PetscReal*)

    int VecGhostGetLocalForm(PetscVec,PetscVec*)
    int VecGhostUpdateBegin(PetscVec,PetscInsertMode,PetscScatterMode)
    int VecGhostUpdateEnd(PetscVec,PetscInsertMode,PetscScatterMode)

# --------------------------------------------------------------------

cdef extern from "petscvec.h" nogil:

    int VecScatterView(PetscScatter, PetscViewer)
    int VecScatterDestroy(PetscScatter)
    int VecScatterCreate(PetscVec,PetscIS,PetscVec,PetscIS,PetscScatter*)
    int VecScatterCopy(PetscScatter, PetscScatter*)
    int VecScatterCreateToAll(PetscVec,PetscScatter*,PetscVec*)
    int VecScatterCreateToZero(PetscVec,PetscScatter*,PetscVec*)
    int VecScatterBegin(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)
    int VecScatterEnd(PetscScatter,PetscVec,PetscVec,PetscInsertMode,PetscScatterMode)

# --------------------------------------------------------------------

cdef inline Vec ref_Vec(PetscVec vec):
    cdef Vec ob = <Vec> Vec()
    PetscIncref(<PetscObject>vec)
    ob.vec = vec
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

cdef inline int Vec_SplitSizes(MPI_Comm comm,
                               object size, object bsize,
                               PetscInt *b,
                               PetscInt *n, PetscInt *N) except -1:
    CHKERR( Sys_SplitSizes(comm, size, bsize, b, n, N) )
    return 0

# --------------------------------------------------------------------

ctypedef int VecSetValuesFcn(PetscVec,PetscInt,const_PetscInt[],
                             const_PetscScalar[],PetscInsertMode)

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
    cdef object ai = iarray_i(oi, &ni, &i)
    cdef object av = iarray_s(ov, &nv, &v)
    if ni*bs != nv: raise ValueError(
        "incompatible array sizes: ni=%d, nv=%d, bs=%d" %
        (toInt(ni), toInt(nv), toInt(bs)) )
    # insert mode
    cdef PetscInsertMode addv = insertmode(oim)
    # VecSetValuesXXX function
    cdef VecSetValuesFcn *setvalues = NULL
    if blocked and local: setvalues = VecSetValuesBlockedLocal
    elif blocked:         setvalues = VecSetValuesBlocked
    elif local:           setvalues = VecSetValuesLocal
    else:                 setvalues = VecSetValues
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
            CHKERR( PetscMemcpy(vv, va, nv*sizeof(PetscScalar)) )
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

cdef extern from "arraynpy.h":
    object PetscIS_array_struct(object,PetscIS)
    object PetscVec_array_struct(object,PetscVec)

cdef extern from "pep3118.h":
    int  PyPetscBuffer_FillInfo(Py_buffer*,object,char,int) except -1
    void PyPetscBuffer_Release(Py_buffer*)

# --------------------------------------------------------------------
