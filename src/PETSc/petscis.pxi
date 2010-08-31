cdef extern from "petscis.h" nogil:

    ctypedef enum PetscISType "ISType":
        IS_GENERAL
        IS_STRIDE
        IS_BLOCK

    int ISView(PetscIS,PetscViewer)
    int ISDestroy(PetscIS)
    int ISCreateGeneral(MPI_Comm,PetscInt,PetscInt[],PetscIS*)
    int ISCreateBlock(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscIS*)
    int ISCreateStride(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscIS*)
    int ISGetType(PetscIS,PetscISType*)

    int ISDuplicate(PetscIS,PetscIS*)
    int ISCopy(PetscIS,PetscIS)
    int ISAllGather(PetscIS,PetscIS*)
    int ISInvertPermutation(PetscIS,PetscInt,PetscIS*)

    int ISGetSize(PetscIS,PetscInt*)
    int ISGetLocalSize(PetscIS,PetscInt*)
    int ISGetIndices(PetscIS,const_PetscInt*[])
    int ISRestoreIndices(PetscIS,const_PetscInt*[])

    int ISEqual(PetscIS,PetscIS,PetscTruth*)

    int ISSetPermutation(PetscIS)
    int ISPermutation(PetscIS,PetscTruth*)
    int ISSetIdentity(PetscIS)
    int ISIdentity(PetscIS,PetscTruth*)

    int ISSort(PetscIS)
    int ISSorted(PetscIS,PetscTruth*)

    int ISSum(PetscIS,PetscIS,PetscIS*)
    int ISExpand(PetscIS,PetscIS,PetscIS*)
    int ISDifference(PetscIS,PetscIS,PetscIS*)
    int ISComplement(PetscIS,PetscInt,PetscInt,PetscIS*)

    int ISBlock(PetscIS,PetscTruth*)
    int ISBlockGetIndices(PetscIS,const_PetscInt*[])
    int ISBlockRestoreIndices(PetscIS,const_PetscInt*[])
    int ISBlockGetSize(PetscIS,PetscInt*)
    int ISBlockGetLocalSize(PetscIS,PetscInt*)
    int ISBlockGetBlockSize(PetscIS,PetscInt*)
    int ISStride(PetscIS,PetscTruth*)
    int ISStrideGetInfo(PetscIS,PetscInt*,PetscInt*)
    int ISStrideToGeneral(PetscIS)


cdef extern from "petscis.h" nogil:

    ctypedef enum PetscGLMapType "ISGlobalToLocalMappingType":
        IS_GTOLM_MASK
        IS_GTOLM_DROP

    int ISLocalToGlobalMappingCreate(MPI_Comm,PetscInt,PetscInt[],PetscLGMap*)
    int ISLocalToGlobalMappingCreateNC(MPI_Comm,PetscInt,PetscInt[],PetscLGMap*)
    int ISLocalToGlobalMappingCreateIS(PetscIS,PetscLGMap*)
    int ISLocalToGlobalMappingBlock(PetscLGMap,PetscInt,PetscLGMap*)
    int ISLocalToGlobalMappingView(PetscLGMap,PetscViewer)
    int ISLocalToGlobalMappingDestroy(PetscLGMap)
    int ISLocalToGlobalMappingApplyIS(PetscLGMap,PetscIS,PetscIS*)
    int ISLocalToGlobalMappingGetSize(PetscLGMap,PetscInt*)
    int ISLocalToGlobalMappingGetInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    int ISLocalToGlobalMappingRestoreInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    int ISLocalToGlobalMappingBlock(PetscLGMap,PetscInt,PetscLGMap*)
    int ISLocalToGlobalMappingApply(PetscLGMap mapping,PetscInt,PetscInt[],PetscInt[])
    int ISGlobalToLocalMappingApply(PetscLGMap,PetscGLMapType,PetscInt,PetscInt[],PetscInt*,PetscInt[])


# --------------------------------------------------------------------

cdef extern from "pep3118.h":
    int  PyPetscBuffer_FillInfo(Py_buffer*,
                                void*,PetscInt,char,
                                int,int) except -1
    void PyPetscBuffer_Release(Py_buffer*)

# --------------------------------------------------------------------

cdef class _IS_buffer:

    cdef PetscIS iset
    cdef PetscInt size
    cdef const_PetscInt *data

    def __cinit__(self, IS iset not None):
        CHKERR( PetscIncref(<PetscObject>iset.iset) )
        self.iset = iset.iset
        self.size = -1
        self.data = NULL

    def __dealloc__(self):
        if self.data != NULL and self.iset != NULL:
            CHKERR( ISRestoreIndices(self.iset, &self.data) )
        if self.iset != NULL:
            CHKERR( ISDestroy(self.iset) )

    #

    cdef int acquirebuffer(self, Py_buffer *view, int flags) except -1:
        cdef PetscInt size = 0
        cdef const_PetscInt *data = NULL
        CHKERR( ISGetLocalSize(self.iset, &size) )
        CHKERR( ISGetIndices(self.iset, &data) )
        PyPetscBuffer_FillInfo(view, <void*>data,
                               size, 'i', 0, flags)
        return 0

    cdef int releasebuffer(self, Py_buffer *view) except -1:
        cdef const_PetscInt *data = <PetscInt*>view.buf
        PyPetscBuffer_Release(view)
        CHKERR( ISRestoreIndices(self.iset, &data) )
        return 0

    def __getbuffer__(self, Py_buffer *view, int flags):
        self.acquirebuffer(view, flags)
        view.obj = self

    def __releasebuffer__(self, Py_buffer *view):
        self.releasebuffer(view)

    #

    cdef Py_ssize_t getbuffer(self, Py_ssize_t idx, void **p) except -1:
        if idx != 0: raise SystemError(
            "accessing non-existent buffer segment")
        if self.size < 0:
            CHKERR( ISGetLocalSize(self.iset, &self.size) )
        if p != NULL:
            if self.data == NULL:
                CHKERR( ISGetIndices(self.iset, &self.data) )
            p[0] = <void*>self.data
        return <Py_ssize_t> (self.size*sizeof(PetscInt))

    def __getsegcount__(self, Py_ssize_t *lenp):
        if lenp != NULL:
            lenp[0] = self.getbuffer(0, NULL)
        return 1

    def __getreadbuffer__(self, Py_ssize_t idx, void **p):
        return self.getbuffer(idx, p)

# --------------------------------------------------------------------
