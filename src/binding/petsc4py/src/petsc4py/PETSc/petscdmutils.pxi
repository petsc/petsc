
cdef extern from * nogil:

    struct _DMInterpolationInfo
    ctypedef _DMInterpolationInfo* PetscDMInterpolation "DMInterpolationInfo"

    PetscErrorCode DMInterpolationCreate(MPI_Comm, PetscDMInterpolation*)
    PetscErrorCode DMInterpolationDestroy(PetscDMInterpolation*)
    PetscErrorCode DMInterpolationEvaluate(PetscDMInterpolation, PetscDM, PetscVec, PetscVec)
    PetscErrorCode DMInterpolationGetCoordinates(PetscDMInterpolation, PetscVec*)
    PetscErrorCode DMInterpolationGetDim(PetscDMInterpolation, PetscInt*)
    PetscErrorCode DMInterpolationGetDof(PetscDMInterpolation, PetscInt*)
    PetscErrorCode DMInterpolationGetVector(PetscDMInterpolation, PetscVec*)
    PetscErrorCode DMInterpolationRestoreVector(PetscDMInterpolation, PetscVec*)
    PetscErrorCode DMInterpolationSetDim(PetscDMInterpolation, PetscInt)
    PetscErrorCode DMInterpolationSetDof(PetscDMInterpolation, PetscInt)
    PetscErrorCode DMInterpolationSetUp(PetscDMInterpolation, PetscDM, PetscBool, PetscBool)





