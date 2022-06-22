
cdef extern from * nogil:

    struct _DMInterpolationInfo
    ctypedef _DMInterpolationInfo* PetscDMInterpolation "DMInterpolationInfo"

    int DMInterpolationCreate(MPI_Comm, PetscDMInterpolation*)
    int DMInterpolationDestroy(PetscDMInterpolation*)
    int DMInterpolationEvaluate(PetscDMInterpolation, PetscDM, PetscVec, PetscVec)
    int DMInterpolationGetCoordinates(PetscDMInterpolation, PetscVec*)
    int DMInterpolationGetDim(PetscDMInterpolation, PetscInt*)
    int DMInterpolationGetDof(PetscDMInterpolation, PetscInt*)
    int DMInterpolationGetVector(PetscDMInterpolation, PetscVec*)
    int DMInterpolationRestoreVector(PetscDMInterpolation, PetscVec*)
    int DMInterpolationSetDim(PetscDMInterpolation, PetscInt)
    int DMInterpolationSetDof(PetscDMInterpolation, PetscInt)
    int DMInterpolationSetUp(PetscDMInterpolation, PetscDM, PetscBool, PetscBool)





