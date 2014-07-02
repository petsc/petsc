# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef char* PetscDMType "const char*"
    PetscDMType DMDA_type "DMDA"
    PetscDMType DMCOMPOSITE
    PetscDMType DMSLICED
    PetscDMType DMSHELL
    PetscDMType DMMESH
    PetscDMType DMPLEX
    PetscDMType DMCARTESIAN
    PetscDMType DMREDUNDANT
    PetscDMType DMPATCH
    PetscDMType DMMOAB
    PetscDMType DMNETWORK

    ctypedef enum PetscDMBoundaryType"DMBoundaryType":
        DM_BOUNDARY_NONE
        DM_BOUNDARY_GHOSTED
        DM_BOUNDARY_MIRROR
        DM_BOUNDARY_PERIODIC

    int DMCreate(MPI_Comm,PetscDM*)
    int DMClone(PetscDM,PetscDM*)
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
    int DMSetMatType(PetscDM,PetscMatType)
    int DMCreateMatrix(PetscDM,PetscMat*)

    int DMGetCoordinateDM(PetscDM,PetscDM*)
    int DMGetCoordinateSection(PetscDM,PetscSection*)
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
    int DMLocalToLocalBegin(PetscDM,PetscVec,PetscInsertMode,PetscVec)
    int DMLocalToLocalEnd(PetscDM,PetscVec,PetscInsertMode,PetscVec)

    int DMGetLocalToGlobalMapping(PetscDM,PetscLGMap*)

    int DMSetDefaultSection(PetscDM,PetscSection)
    int DMGetDefaultSection(PetscDM,PetscSection*)
    int DMSetDefaultGlobalSection(PetscDM,PetscSection)
    int DMGetDefaultGlobalSection(PetscDM,PetscSection*)
    int DMCreateDefaultSF(PetscDM,PetscSection,PetscSection)
    int DMGetDefaultSF(PetscDM,PetscSF*)
    int DMSetDefaultSF(PetscDM,PetscSF)
    int DMGetPointSF(PetscDM,PetscSF*)

    int DMShellSetGlobalVector(PetscDM,PetscVec)
    int DMShellSetLocalVector(PetscDM,PetscVec)
# --------------------------------------------------------------------
