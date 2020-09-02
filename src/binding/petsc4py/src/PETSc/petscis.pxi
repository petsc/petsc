cdef extern from * nogil:

    ctypedef const char* PetscISType "ISType"
    PetscISType ISGENERAL
    PetscISType ISSTRIDE
    PetscISType ISBLOCK

    int ISView(PetscIS,PetscViewer)
    int ISDestroy(PetscIS*)
    int ISCreate(MPI_Comm,PetscIS*)
    int ISSetType(PetscIS,PetscISType)
    int ISGetType(PetscIS,PetscISType*)

    int ISCreateGeneral(MPI_Comm,PetscInt,PetscInt[],PetscCopyMode,PetscIS*)
    int ISCreateBlock(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscCopyMode,PetscIS*)
    int ISCreateStride(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscIS*)

    int ISLoad(PetscIS,PetscViewer)
    int ISDuplicate(PetscIS,PetscIS*)
    int ISCopy(PetscIS,PetscIS)
    int ISAllGather(PetscIS,PetscIS*)
    int ISInvertPermutation(PetscIS,PetscInt,PetscIS*)

    int ISGetSize(PetscIS,PetscInt*)
    int ISGetLocalSize(PetscIS,PetscInt*)
    int ISGetBlockSize(PetscIS,PetscInt*)
    int ISSetBlockSize(PetscIS,PetscInt)
    int ISGetIndices(PetscIS,const PetscInt*[])
    int ISRestoreIndices(PetscIS,const PetscInt*[])

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
    int ISEmbed(PetscIS,PetscIS,PetscBool,PetscIS*)
    int ISRenumber(PetscIS,PetscIS,PetscInt*,PetscIS*)

    int ISGeneralSetIndices(PetscIS,PetscInt,PetscInt[],PetscCopyMode)

    int ISBlockSetIndices(PetscIS,PetscInt,PetscInt,PetscInt[],PetscCopyMode)
    int ISBlockGetIndices(PetscIS,const PetscInt*[])
    int ISBlockRestoreIndices(PetscIS,const PetscInt*[])

    int ISStrideSetStride(PetscIS,PetscInt,PetscInt,PetscInt)
    int ISStrideGetInfo(PetscIS,PetscInt*,PetscInt*)

    int ISToGeneral(PetscIS)


cdef extern from * nogil:

    ctypedef const char* PetscISLocalToGlobalMappingType "ISLocalToGlobalMappingType"
    PetscISLocalToGlobalMappingType ISLOCALTOGLOBALMAPPINGBASIC
    PetscISLocalToGlobalMappingType ISLOCALTOGLOBALMAPPINGHASH

    ctypedef enum PetscGLMapMode "ISGlobalToLocalMappingMode":
        PETSC_IS_GTOLM_MASK "IS_GTOLM_MASK"
        PETSC_IS_GTOLM_DROP "IS_GTOLM_DROP"

    int ISLocalToGlobalMappingCreate(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscCopyMode,PetscLGMap*)
    int ISLocalToGlobalMappingCreateIS(PetscIS,PetscLGMap*)
    int ISLocalToGlobalMappingCreateSF(PetscSF,PetscInt,PetscLGMap*)
    int ISLocalToGlobalMappingSetType(PetscLGMap,PetscISLocalToGlobalMappingType)
    int ISLocalToGlobalMappingSetFromOptions(PetscLGMap)
    int ISLocalToGlobalMappingView(PetscLGMap,PetscViewer)
    int ISLocalToGlobalMappingDestroy(PetscLGMap*)
    int ISLocalToGlobalMappingGetSize(PetscLGMap,PetscInt*)
    int ISLocalToGlobalMappingGetBlockSize(PetscLGMap,PetscInt*)
    int ISLocalToGlobalMappingGetIndices(PetscLGMap,const PetscInt*[])
    int ISLocalToGlobalMappingRestoreIndices(PetscLGMap,const PetscInt*[])
    int ISLocalToGlobalMappingGetBlockIndices(PetscLGMap,const PetscInt*[])
    int ISLocalToGlobalMappingRestoreBlockIndices(PetscLGMap,const PetscInt*[])
    int ISLocalToGlobalMappingGetInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    int ISLocalToGlobalMappingRestoreInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    int ISLocalToGlobalMappingGetBlockInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    int ISLocalToGlobalMappingRestoreBlockInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    int ISLocalToGlobalMappingApply(PetscLGMap,PetscInt,PetscInt[],PetscInt[])
    int ISLocalToGlobalMappingApplyBlock(PetscLGMap,PetscInt,PetscInt[],PetscInt[])
    int ISLocalToGlobalMappingApplyIS(PetscLGMap,PetscIS,PetscIS*)
    int ISGlobalToLocalMappingApply(PetscLGMap,PetscGLMapMode,PetscInt,PetscInt[],PetscInt*,PetscInt[])
    int ISGlobalToLocalMappingApplyBlock(PetscLGMap,PetscGLMapMode,PetscInt,PetscInt[],PetscInt*,PetscInt[])


# --------------------------------------------------------------------

cdef inline IS ref_IS(PetscIS iset):
    cdef IS ob = <IS> IS()
    ob.iset = iset
    PetscINCREF(ob.obj)
    return ob

cdef inline LGMap ref_LGMap(PetscLGMap lgm):
    cdef LGMap ob = <LGMap> LGMap()
    ob.lgm = lgm
    PetscINCREF(ob.obj)
    return ob

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
    cdef const PetscInt *data
    cdef bint hasarray

    def __cinit__(self, IS iset):
        cdef PetscIS i = iset.iset
        CHKERR( PetscINCREF(<PetscObject*>&i) )
        self.iset = i
        self.size = 0
        self.data = NULL
        self.hasarray = 0

    def __dealloc__(self):
        if self.hasarray and self.iset != NULL:
            CHKERR( ISRestoreIndices(self.iset, &self.data) )
        CHKERR( ISDestroy(&self.iset) )

    #

    cdef int acquire(self) except -1:
        if not self.hasarray and self.iset != NULL:
            CHKERR( ISGetLocalSize(self.iset, &self.size) )
            CHKERR( ISGetIndices(self.iset, &self.data) )
            self.hasarray = 1
        return 0

    cdef int release(self) except -1:
        if self.hasarray and self.iset != NULL:
            self.size = 0
            CHKERR( ISRestoreIndices(self.iset, &self.data) )
            self.hasarray = 0
            self.data = NULL
        return 0

    # buffer interface (PEP 3118)

    cdef int acquirebuffer(self, Py_buffer *view, int flags) except -1:
        self.acquire()
        PyPetscBuffer_FillInfo(view, <void*>self.data, self.size,
                               c'i', 1, flags)
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
        elif self.iset != NULL:
            CHKERR( ISGetLocalSize(self.iset, &n) )
        return <Py_ssize_t>(<size_t>n*sizeof(PetscInt))

    def __getsegcount__(self, Py_ssize_t *lenp):
        if lenp != NULL:
            lenp[0] = self.getbuffer(NULL)
        return 1

    def __getreadbuffer__(self, Py_ssize_t idx, void **p):
        if idx != 0: raise SystemError(
            "accessing non-existent buffer segment")
        return self.getbuffer(p)

    # NumPy array interface (legacy)

    property __array_interface__:
        def __get__(self):
            cdef PetscInt n = 0
            if self.iset != NULL:
                CHKERR( ISGetLocalSize(self.iset, &n) )
            cdef object size = toInt(n)
            cdef dtype descr = PyArray_DescrFromType(NPY_PETSC_INT)
            cdef str typestr = "=%c%d" % (descr.kind, descr.itemsize)
            return dict(version=3,
                        data=self,
                        shape=(size,),
                        typestr=typestr)

# --------------------------------------------------------------------
