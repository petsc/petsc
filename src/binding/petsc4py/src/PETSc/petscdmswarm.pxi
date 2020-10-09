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

    int DMSwarmCreateGlobalVectorFromField(PetscDM,const char[],PetscVec*)
    int DMSwarmDestroyGlobalVectorFromField(PetscDM,const char[],PetscVec*)
    int DMSwarmCreateLocalVectorFromField(PetscDM,const char[],PetscVec*)
    int DMSwarmDestroyLocalVectorFromField(PetscDM,const char[],PetscVec*)

    int DMSwarmInitializeFieldRegister(PetscDM)
    int DMSwarmFinalizeFieldRegister(PetscDM)
    int DMSwarmSetLocalSizes(PetscDM,PetscInt,PetscInt)
    int DMSwarmRegisterPetscDatatypeField(PetscDM,const char[],PetscInt,PetscDataType)
#    int DMSwarmRegisterUserStructField(PetscDM,const char[],size_t)
#    int DMSwarmRegisterUserDatatypeField(PetscDM,const char[],size_t,PetscInt)
    int DMSwarmGetField(PetscDM,const char[],PetscInt*,PetscDataType*,void**)
    int DMSwarmRestoreField(PetscDM,const char[],PetscInt*,PetscDataType*,void**)

    int DMSwarmVectorDefineField(PetscDM,const char[])

    int DMSwarmAddPoint(PetscDM)
    int DMSwarmAddNPoints(PetscDM,PetscInt)
    int DMSwarmRemovePoint(PetscDM)
    int DMSwarmRemovePointAtIndex(PetscDM,PetscInt)
    int DMSwarmCopyPoint(PetscDM,PetscInt,PetscInt)

    int DMSwarmGetLocalSize(PetscDM,PetscInt*)
    int DMSwarmGetSize(PetscDM,PetscInt*)
    int DMSwarmMigrate(PetscDM,PetscBool)

    int DMSwarmCollectViewCreate(PetscDM)
    int DMSwarmCollectViewDestroy(PetscDM)
    int DMSwarmSetCellDM(PetscDM,PetscDM)
    int DMSwarmGetCellDM(PetscDM,PetscDM*)

    int DMSwarmSetType(PetscDM, PetscDMSwarmType)

    int DMSwarmSetPointsUniformCoordinates(PetscDM,PetscReal[],PetscReal[],PetscInt[],PetscInsertMode)
    int DMSwarmSetPointCoordinates(PetscDM,PetscInt,PetscReal*,PetscBool,PetscInsertMode)
    int DMSwarmInsertPointsUsingCellDM(PetscDM,PetscDMSwarmPICLayoutType,PetscInt)
    int DMSwarmSetPointCoordinatesCellwise(PetscDM,PetscInt,PetscReal*)
    int DMSwarmViewFieldsXDMF(PetscDM,const char*,PetscInt,const char**)
    int DMSwarmViewXDMF(PetscDM,const char*)

    int DMSwarmSortGetAccess(PetscDM)
    int DMSwarmSortRestoreAccess(PetscDM)
    int DMSwarmSortGetPointsPerCell(PetscDM,PetscInt,PetscInt*,PetscInt**)
    int DMSwarmSortGetNumberOfPointsPerCell(PetscDM,PetscInt,PetscInt*)
    int DMSwarmSortGetIsValid(PetscDM,PetscBool*)
    int DMSwarmSortGetSizes(PetscDM,PetscInt*,PetscInt*)
 
    int DMSwarmProjectFields(PetscDM,PetscInt,const char**,PetscVec**,PetscBool)

