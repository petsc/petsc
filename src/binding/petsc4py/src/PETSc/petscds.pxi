cdef extern from * nogil:

    ctypedef const char* PetscDSType
    PetscDSType PETSCDSBASIC

    PetscErrorCode PetscDSCreate(MPI_Comm,PetscDS*)
    PetscErrorCode PetscDSDestroy(PetscDS*)
    PetscErrorCode PetscDSView(PetscDS,PetscViewer)
    PetscErrorCode PetscDSSetType(PetscDS,PetscDSType)
    PetscErrorCode PetscDSGetType(PetscDS,PetscDSType*)
    PetscErrorCode PetscDSSetFromOptions(PetscDS)
    PetscErrorCode PetscDSSetUp(PetscDS)

    PetscErrorCode PetscDSGetHeightSubspace(PetscDS,PetscInt,PetscDS*)
    PetscErrorCode PetscDSGetSpatialDimension(PetscDS,PetscInt*)
    PetscErrorCode PetscDSGetCoordinateDimension(PetscDS,PetscInt*)
    PetscErrorCode PetscDSSetCoordinateDimension(PetscDS,PetscInt)
    PetscErrorCode PetscDSGetNumFields(PetscDS,PetscInt*)
    PetscErrorCode PetscDSGetTotalDimension(PetscDS,PetscInt*)
    PetscErrorCode PetscDSGetTotalComponents(PetscDS,PetscInt*)
    PetscErrorCode PetscDSGetFieldIndex(PetscDS,PetscObject,PetscInt*)
    PetscErrorCode PetscDSGetFieldSize(PetscDS,PetscInt,PetscInt*)
    PetscErrorCode PetscDSGetFieldOffset(PetscDS,PetscInt,PetscInt*)
    PetscErrorCode PetscDSGetDimensions(PetscDS,PetscInt*[])
    PetscErrorCode PetscDSGetComponents(PetscDS,PetscInt*[])
    PetscErrorCode PetscDSGetComponentOffset(PetscDS,PetscInt,PetscInt*)
    PetscErrorCode PetscDSGetComponentOffsets(PetscDS,PetscInt*[])
    PetscErrorCode PetscDSGetComponentDerivativeOffsets(PetscDS,PetscInt*[])

    PetscErrorCode PetscDSGetDiscretization(PetscDS,PetscInt,PetscObject*)
    PetscErrorCode PetscDSSetDiscretization(PetscDS,PetscInt,PetscObject)
    PetscErrorCode PetscDSAddDiscretization(PetscDS,PetscObject)
    PetscErrorCode PetscDSGetImplicit(PetscDS,PetscInt,PetscBool*)
    PetscErrorCode PetscDSSetImplicit(PetscDS,PetscInt,PetscBool)
