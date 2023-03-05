cdef extern from * nogil:

    ctypedef const char* PetscISType "ISType"
    PetscISType ISGENERAL
    PetscISType ISSTRIDE
    PetscISType ISBLOCK

    PetscErrorCode ISView(PetscIS,PetscViewer)
    PetscErrorCode ISDestroy(PetscIS*)
    PetscErrorCode ISCreate(MPI_Comm,PetscIS*)
    PetscErrorCode ISSetType(PetscIS,PetscISType)
    PetscErrorCode ISGetType(PetscIS,PetscISType*)

    PetscErrorCode ISCreateGeneral(MPI_Comm,PetscInt,PetscInt[],PetscCopyMode,PetscIS*)
    PetscErrorCode ISCreateBlock(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscCopyMode,PetscIS*)
    PetscErrorCode ISCreateStride(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscIS*)

    PetscErrorCode ISLoad(PetscIS,PetscViewer)
    PetscErrorCode ISDuplicate(PetscIS,PetscIS*)
    PetscErrorCode ISCopy(PetscIS,PetscIS)
    PetscErrorCode ISAllGather(PetscIS,PetscIS*)
    PetscErrorCode ISInvertPermutation(PetscIS,PetscInt,PetscIS*)

    PetscErrorCode ISGetSize(PetscIS,PetscInt*)
    PetscErrorCode ISGetLocalSize(PetscIS,PetscInt*)
    PetscErrorCode ISGetBlockSize(PetscIS,PetscInt*)
    PetscErrorCode ISSetBlockSize(PetscIS,PetscInt)
    PetscErrorCode ISGetIndices(PetscIS,const PetscInt*[])
    PetscErrorCode ISRestoreIndices(PetscIS,const PetscInt*[])

    PetscErrorCode ISEqual(PetscIS,PetscIS,PetscBool*)

    PetscErrorCode ISSetPermutation(PetscIS)
    PetscErrorCode ISPermutation(PetscIS,PetscBool*)
    PetscErrorCode ISSetIdentity(PetscIS)
    PetscErrorCode ISIdentity(PetscIS,PetscBool*)

    PetscErrorCode ISSort(PetscIS)
    PetscErrorCode ISSorted(PetscIS,PetscBool*)

    PetscErrorCode ISSum(PetscIS,PetscIS,PetscIS*)
    PetscErrorCode ISExpand(PetscIS,PetscIS,PetscIS*)
    PetscErrorCode ISDifference(PetscIS,PetscIS,PetscIS*)
    PetscErrorCode ISComplement(PetscIS,PetscInt,PetscInt,PetscIS*)
    PetscErrorCode ISEmbed(PetscIS,PetscIS,PetscBool,PetscIS*)
    PetscErrorCode ISRenumber(PetscIS,PetscIS,PetscInt*,PetscIS*)

    PetscErrorCode ISGeneralSetIndices(PetscIS,PetscInt,PetscInt[],PetscCopyMode)

    PetscErrorCode ISBlockSetIndices(PetscIS,PetscInt,PetscInt,PetscInt[],PetscCopyMode)
    PetscErrorCode ISBlockGetIndices(PetscIS,const PetscInt*[])
    PetscErrorCode ISBlockRestoreIndices(PetscIS,const PetscInt*[])

    PetscErrorCode ISStrideSetStride(PetscIS,PetscInt,PetscInt,PetscInt)
    PetscErrorCode ISStrideGetInfo(PetscIS,PetscInt*,PetscInt*)

    PetscErrorCode ISToGeneral(PetscIS)
    PetscErrorCode ISBuildTwoSided(PetscIS,PetscIS,PetscIS*)


cdef extern from * nogil:

    ctypedef const char* PetscISLocalToGlobalMappingType "ISLocalToGlobalMappingType"
    PetscISLocalToGlobalMappingType ISLOCALTOGLOBALMAPPINGBASIC
    PetscISLocalToGlobalMappingType ISLOCALTOGLOBALMAPPINGHASH

    ctypedef enum PetscGLMapMode "ISGlobalToLocalMappingMode":
        PETSC_IS_GTOLM_MASK "IS_GTOLM_MASK"
        PETSC_IS_GTOLM_DROP "IS_GTOLM_DROP"

    PetscErrorCode ISLocalToGlobalMappingCreate(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscCopyMode,PetscLGMap*)
    PetscErrorCode ISLocalToGlobalMappingCreateIS(PetscIS,PetscLGMap*)
    PetscErrorCode ISLocalToGlobalMappingCreateSF(PetscSF,PetscInt,PetscLGMap*)
    PetscErrorCode ISLocalToGlobalMappingSetType(PetscLGMap,PetscISLocalToGlobalMappingType)
    PetscErrorCode ISLocalToGlobalMappingSetFromOptions(PetscLGMap)
    PetscErrorCode ISLocalToGlobalMappingView(PetscLGMap,PetscViewer)
    PetscErrorCode ISLocalToGlobalMappingDestroy(PetscLGMap*)
    PetscErrorCode ISLocalToGlobalMappingGetSize(PetscLGMap,PetscInt*)
    PetscErrorCode ISLocalToGlobalMappingGetBlockSize(PetscLGMap,PetscInt*)
    PetscErrorCode ISLocalToGlobalMappingGetIndices(PetscLGMap,const PetscInt*[])
    PetscErrorCode ISLocalToGlobalMappingRestoreIndices(PetscLGMap,const PetscInt*[])
    PetscErrorCode ISLocalToGlobalMappingGetBlockIndices(PetscLGMap,const PetscInt*[])
    PetscErrorCode ISLocalToGlobalMappingRestoreBlockIndices(PetscLGMap,const PetscInt*[])
    PetscErrorCode ISLocalToGlobalMappingGetInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    PetscErrorCode ISLocalToGlobalMappingRestoreInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    PetscErrorCode ISLocalToGlobalMappingGetBlockInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    PetscErrorCode ISLocalToGlobalMappingRestoreBlockInfo(PetscLGMap,PetscInt*,PetscInt*[],PetscInt*[],PetscInt**[])
    PetscErrorCode ISLocalToGlobalMappingApply(PetscLGMap,PetscInt,PetscInt[],PetscInt[])
    PetscErrorCode ISLocalToGlobalMappingApplyBlock(PetscLGMap,PetscInt,PetscInt[],PetscInt[])
    PetscErrorCode ISLocalToGlobalMappingApplyIS(PetscLGMap,PetscIS,PetscIS*)
    PetscErrorCode ISGlobalToLocalMappingApply(PetscLGMap,PetscGLMapMode,PetscInt,PetscInt[],PetscInt*,PetscInt[])
    PetscErrorCode ISGlobalToLocalMappingApplyBlock(PetscLGMap,PetscGLMapMode,PetscInt,PetscInt[],PetscInt*,PetscInt[])


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
