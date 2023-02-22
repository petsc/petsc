# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef enum PetscDMSwarmType "DMSwarmType":
        DMSWARM_BASIC
        DMSWARM_PIC

    ctypedef enum PetscDMSwarmMigrateType "DMSwarmMigrateType":
        DMSWARM_MIGRATE_BASIC
        DMSWARM_MIGRATE_DMCELLNSCATTER
        DMSWARM_MIGRATE_DMCELLEXACT
        DMSWARM_MIGRATE_USER

    ctypedef enum PetscDMSwarmCollectType "DMSwarmCollectType":
        DMSWARM_COLLECT_BASIC
        DMSWARM_COLLECT_DMDABOUNDINGBOX
        DMSWARM_COLLECT_GENERAL
        DMSWARM_COLLECT_USER

    ctypedef enum PetscDMSwarmPICLayoutType 'DMSwarmPICLayoutType':
        DMSWARMPIC_LAYOUT_REGULAR
        DMSWARMPIC_LAYOUT_GAUSS
        DMSWARMPIC_LAYOUT_SUBDIVISION

    PetscErrorCode DMSwarmCreateGlobalVectorFromField(PetscDM,const char[],PetscVec*)
    PetscErrorCode DMSwarmDestroyGlobalVectorFromField(PetscDM,const char[],PetscVec*)
    PetscErrorCode DMSwarmCreateLocalVectorFromField(PetscDM,const char[],PetscVec*)
    PetscErrorCode DMSwarmDestroyLocalVectorFromField(PetscDM,const char[],PetscVec*)

    PetscErrorCode DMSwarmInitializeFieldRegister(PetscDM)
    PetscErrorCode DMSwarmFinalizeFieldRegister(PetscDM)
    PetscErrorCode DMSwarmSetLocalSizes(PetscDM,PetscInt,PetscInt)
    PetscErrorCode DMSwarmRegisterPetscDatatypeField(PetscDM,const char[],PetscInt,PetscDataType)
#    PetscErrorCode DMSwarmRegisterUserStructField(PetscDM,const char[],size_t)
#    PetscErrorCode DMSwarmRegisterUserDatatypeField(PetscDM,const char[],size_t,PetscInt)
    PetscErrorCode DMSwarmGetField(PetscDM,const char[],PetscInt*,PetscDataType*,void**)
    PetscErrorCode DMSwarmRestoreField(PetscDM,const char[],PetscInt*,PetscDataType*,void**)

    PetscErrorCode DMSwarmVectorDefineField(PetscDM,const char[])

    PetscErrorCode DMSwarmAddPoint(PetscDM)
    PetscErrorCode DMSwarmAddNPoints(PetscDM,PetscInt)
    PetscErrorCode DMSwarmRemovePoint(PetscDM)
    PetscErrorCode DMSwarmRemovePointAtIndex(PetscDM,PetscInt)
    PetscErrorCode DMSwarmCopyPoint(PetscDM,PetscInt,PetscInt)

    PetscErrorCode DMSwarmGetLocalSize(PetscDM,PetscInt*)
    PetscErrorCode DMSwarmGetSize(PetscDM,PetscInt*)
    PetscErrorCode DMSwarmMigrate(PetscDM,PetscBool)

    PetscErrorCode DMSwarmCollectViewCreate(PetscDM)
    PetscErrorCode DMSwarmCollectViewDestroy(PetscDM)
    PetscErrorCode DMSwarmSetCellDM(PetscDM,PetscDM)
    PetscErrorCode DMSwarmGetCellDM(PetscDM,PetscDM*)

    PetscErrorCode DMSwarmSetType(PetscDM, PetscDMSwarmType)

    PetscErrorCode DMSwarmSetPointsUniformCoordinates(PetscDM,PetscReal[],PetscReal[],PetscInt[],PetscInsertMode)
    PetscErrorCode DMSwarmSetPointCoordinates(PetscDM,PetscInt,PetscReal*,PetscBool,PetscInsertMode)
    PetscErrorCode DMSwarmInsertPointsUsingCellDM(PetscDM,PetscDMSwarmPICLayoutType,PetscInt)
    PetscErrorCode DMSwarmSetPointCoordinatesCellwise(PetscDM,PetscInt,PetscReal*)
    PetscErrorCode DMSwarmViewFieldsXDMF(PetscDM,const char*,PetscInt,const char**)
    PetscErrorCode DMSwarmViewXDMF(PetscDM,const char*)

    PetscErrorCode DMSwarmSortGetAccess(PetscDM)
    PetscErrorCode DMSwarmSortRestoreAccess(PetscDM)
    PetscErrorCode DMSwarmSortGetPointsPerCell(PetscDM,PetscInt,PetscInt*,PetscInt**)
    PetscErrorCode DMSwarmSortGetNumberOfPointsPerCell(PetscDM,PetscInt,PetscInt*)
    PetscErrorCode DMSwarmSortGetIsValid(PetscDM,PetscBool*)
    PetscErrorCode DMSwarmSortGetSizes(PetscDM,PetscInt*,PetscInt*)
 
    PetscErrorCode DMSwarmProjectFields(PetscDM,PetscInt,const char**,PetscVec**,PetscBool)

