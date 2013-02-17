# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef char* PetscDMType "const char*"
    PetscDMType DMDA_type "DMDA"
    PetscDMType DMADDA_type "DMADDA"
    PetscDMType DMCOMPOSITE
    PetscDMType DMSLICED
    PetscDMType DMMESH
    PetscDMType DMCARTESIAN

    int DMCreate(MPI_Comm,PetscDM*)
    int DMDestroy(PetscDM*)
    int DMView(PetscDM,PetscViewer)
    int DMSetType(PetscDM,PetscDMType)
    int DMGetType(PetscDM,PetscDMType*)
    int DMSetOptionsPrefix(PetscDM,char[])
    int DMSetFromOptions(PetscDM)
    int DMSetUp(PetscDM)

    int DMGetBlockSize(PetscDM,PetscInt*)
    int DMSetVecType(PetscDM,PetscVecType)
    int DMCreateLocalVector(PetscDM,PetscVec*)
    int DMCreateGlobalVector(PetscDM,PetscVec*)
    int DMCreateMatrix(PetscDM,PetscMatType,PetscMat*)

    int DMGetCoordinateDM(PetscDM,PetscDM*)
    int DMSetCoordinates(PetscDM,PetscVec)
    int DMGetCoordinates(PetscDM,PetscVec*)
    int DMSetCoordinatesLocal(PetscDM,PetscVec)
    int DMGetCoordinatesLocal(PetscDM,PetscVec*)

    int DMCreateInterpolation(PetscDM,PetscDM,PetscMat*,PetscVec*)
    int DMCreateInjection(PetscDM,PetscDM,PetscScatter*)
    int DMCreateAggregates(PetscDM,PetscDM,PetscMat*)

    int DMConvert(PetscDM,PetscDMType,PetscDM*)
    int DMRefine(PetscDM,MPI_Comm,PetscDM*)
    int DMCoarsen(PetscDM,MPI_Comm,PetscDM*)
    int DMRefineHierarchy(PetscDM,PetscInt,PetscDM[])
    int DMCoarsenHierarchy(PetscDM,PetscInt,PetscDM[])

    int DMGlobalToLocalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMGlobalToLocalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMLocalToGlobalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMLocalToGlobalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)

    int DMGetLocalToGlobalMapping(PetscDM,PetscLGMap*)
    int DMGetLocalToGlobalMappingBlock(PetscDM,PetscLGMap*)

# --------------------------------------------------------------------
