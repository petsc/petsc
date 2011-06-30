# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef char* PetscDMType "const char*"
    

    int DMCreate(MPI_Comm,PetscDM*)
    int DMDestroy(PetscDM*)
    int DMView(PetscDM,PetscViewer)
    int DMSetType(PetscDM,PetscDMType)
    int DMGetType(PetscDM,PetscDMType*)
    int DMSetOptionsPrefix(PetscDM,char[])
    int DMSetFromOptions(PetscDM)
    int DMSetUp(PetscDM)

    int DMGetBlockSize(PetscDM,PetscInt*)
    int DMCreateLocalVector(PetscDM,PetscVec*)
    int DMCreateGlobalVector(PetscDM,PetscVec*)
    int DMGetMatrix(PetscDM,PetscMatType,PetscMat*)

    int DMGetInterpolation(PetscDM,PetscDM,PetscMat*,PetscVec*)
    int DMGetInjection(PetscDM,PetscDM,PetscScatter*)
    int DMRefine(PetscDM,MPI_Comm,PetscDM*)
    int DMCoarsen(PetscDM,MPI_Comm,PetscDM*)
    int DMGetAggregates(PetscDM,PetscDM,PetscMat*)
    int DMRefineHierarchy(PetscDM,PetscInt,PetscDM[])
    int DMCoarsenHierarchy(PetscDM,PetscInt,PetscDM[])

    int DMGlobalToLocalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMGlobalToLocalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMLocalToGlobalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMLocalToGlobalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)

    int DMGetLocalToGlobalMapping(PetscDM,PetscLGMap*)
    int DMGetLocalToGlobalMappingBlock(PetscDM,PetscLGMap*)

# --------------------------------------------------------------------
