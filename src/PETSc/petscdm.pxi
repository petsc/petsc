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

cdef inline type subtype_DM(PetscDM dm):
    cdef type klass = DM
    cdef PetscBool match = PETSC_FALSE
    cdef PetscObject obj = <PetscObject> dm
    if obj == NULL: return klass
    # DA
    CHKERR( PetscTypeCompare(obj, b"da", &match) ) # petsc-3.2
    if match == PETSC_FALSE: # petsc-3.1
        CHKERR( PetscTypeCompare(obj, b"da1d", &match) )
        if match == PETSC_FALSE:
            CHKERR( PetscTypeCompare(obj, b"da2d", &match) )
            if match == PETSC_FALSE:
                CHKERR( PetscTypeCompare(obj, b"da3d", &match) )
                if match == PETSC_FALSE: # petsc-3.0
                    CHKERR( PetscTypeCompare(obj, b"DA", &match) )
    if match == PETSC_TRUE: klass = DA
    # --
    return klass

# --------------------------------------------------------------------
