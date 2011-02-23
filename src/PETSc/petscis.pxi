cdef extern from * nogil:

    ctypedef char* PetscISType "const char*"
    PetscISType ISGENERAL
    PetscISType ISSTRIDE
    PetscISType ISBLOCK

    int ISView(PetscIS,PetscViewer)
    int ISDestroy(PetscIS)
    int ISCreate(MPI_Comm,PetscIS*)
    int ISSetType(PetscIS,PetscISType)
    int ISGetType(PetscIS,PetscISType*)

    int ISCreateGeneral(MPI_Comm,PetscInt,PetscInt[],PetscCopyMode,PetscIS*)
    int ISCreateBlock(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscCopyMode,PetscIS*)
    int ISCreateStride(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscIS*)

    int ISDuplicate(PetscIS,PetscIS*)
    int ISCopy(PetscIS,PetscIS)
    int ISAllGather(PetscIS,PetscIS*)
    int ISInvertPermutation(PetscIS,PetscInt,PetscIS*)

    int ISGetSize(PetscIS,PetscInt*)
    int ISGetLocalSize(PetscIS,PetscInt*)
    int ISGetBlockSize(PetscIS,PetscInt*)
    int ISGetIndices(PetscIS,const_PetscInt*[])
    int ISRestoreIndices(PetscIS,const_PetscInt*[])

    int ISEqual(PetscIS,PetscIS,PetscBool*)

    int ISSetPermutation(PetscIS)
    int ISPermutation(PetscIS,PetscBool*)
    int ISSetIdentity(PetscIS)
    int ISIdentity(PetscIS,PetscBool*)

    int ISSort(PetscIS)
    int ISSorted(PetscIS,PetscBool*)

    int ISSum(PetscIS,PetscIS,PetscIS*)
    int ISExpand(PetscIS,PetscIS,PetscIS*)
    int ISDifference(PetscIS,PetscIS,PetscIS*)
    int ISComplement(PetscIS,PetscInt,PetscInt,PetscIS*)

    int ISGeneralSetIndices(PetscIS,PetscInt,PetscInt[],PetscCopyMode)

    int ISBlockSetIndices(PetscIS,PetscInt,PetscInt,PetscInt[],PetscCopyMode)
    int ISBlockGetIndices(PetscIS,const_PetscInt*[])
    int ISBlockRestoreIndices(PetscIS,const_PetscInt*[])

    int ISStrideSetStride(PetscIS,PetscInt,PetscInt,PetscInt)
    int ISStrideGetInfo(PetscIS,PetscInt*,PetscInt*)

    int ISToGeneral(PetscIS)


cdef extern from * nogil:

    ctypedef enum PetscGLMapType "ISGlobalToLocalMappingType":
        IS_GTOLM_MASK
        IS_GTOLM_DROP

    int ISLocalToGlobalMappingCreate(MPI_Comm,PetscInt,PetscInt[],PetscCopyMode,PetscLGMap*)
    int ISLocalToGlobalMappingCreateIS(PetscIS,PetscLGMap*)
    int ISLocalToGlobalMappingBlock(PetscLGMap,PetscInt,PetscLGMap*)
    int ISLocalToGlobalMappingView(PetscLGMap,PetscViewer)
    int ISLocalToGlobalMappingDestroy(PetscLGMap)
    int ISLocalToGlobalMappingApplyIS(PetscLGMap,PetscIS,PetscIS*)
    int ISLocalToGlobalMappingGetSize(PetscLGMap,PetscInt*)
    int ISLocalToGlobalMappingGetIndices(PetscLGMap,const_PetscInt*[])
    int ISLocalToGlobalMappingRestoreIndices(PetscLGMap,const_PetscInt*[])
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
        cdef PetscIS i = iset.iset
        CHKERR( PetscIncref(<PetscObject>i) )
        self.iset = i
        self.size = 0
        self.data = NULL

    def __dealloc__(self):
        if self.iset != NULL:
            if self.data != NULL:
                CHKERR( ISRestoreIndices(self.iset, &self.data) )
            CHKERR( ISDestroy(self.iset) )

    #

    cdef int acquire(self) except -1:
        if self.iset != NULL and self.data == NULL:
            CHKERR( ISGetLocalSize(self.iset, &self.size) )
            CHKERR( ISGetIndices(self.iset, &self.data) )
        return 0

    cdef int release(self) except -1:
        if self.iset != NULL and self.data != NULL:
            CHKERR( ISRestoreIndices(self.iset, &self.data) )
            self.size = 0
            self.data = NULL
        return 0

    # buffer interface (PEP 3118)

    cdef int acquirebuffer(self, Py_buffer *view, int flags) except -1:
        self.acquire()
        PyPetscBuffer_FillInfo(view, <void*>self.data,
                               self.size, c'i', 0, flags)
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

    # buffer interface (legacy)

    cdef Py_ssize_t getbuffer(self, Py_ssize_t idx, void **p) except -1:
        if idx != 0: raise SystemError(
            "accessing non-existent buffer segment")
        if self.iset != NULL:
            CHKERR( ISGetLocalSize(self.iset, &self.size) )
        if p != NULL:
            if self.iset != NULL and self.data == NULL:
                CHKERR( ISGetIndices(self.iset, &self.data) )
            p[0] = <void*>self.data
        return <Py_ssize_t> (self.size*sizeof(PetscInt))

    def __getsegcount__(self, Py_ssize_t *lenp):
        if lenp != NULL:
            lenp[0] = self.getbuffer(0, NULL)
        return 1

    def __getreadbuffer__(self, Py_ssize_t idx, void **p):
        return self.getbuffer(idx, p)

    # NumPy array interface (legacy)

    property __array_interface__:
        def __get__(self):
            if self.iset != NULL:
                CHKERR( ISGetLocalSize(self.iset, &self.size) )
            cdef object size = toInt(self.size)
            cdef dtype descr = PyArray_DescrFromType(NPY_PETSC_INT)
            cdef str typestr = "=%c%d" % (descr.kind, descr.itemsize)
            return dict(version=3,
                        data=self,
                        shape=(size,),
                        typestr=typestr)

# --------------------------------------------------------------------
