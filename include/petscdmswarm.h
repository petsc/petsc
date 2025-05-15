#pragma once

#include <petscdm.h>
#include <petscdt.h>

/* MANSEC = DM */
/* SUBMANSEC = DMSwarm */

typedef struct _p_DMSwarmDataField  *DMSwarmDataField;
typedef struct _p_DMSwarmDataBucket *DMSwarmDataBucket;
typedef struct _p_DMSwarmSort       *DMSwarmSort;

/*E
   DMSwarmType - Defines the type of `DMSWARM`

   Values:
+  `DMSWARM_BASIC` - defines N entries of varied data-types which the user may register.
-  `DMSWARM_PIC`   - suitable for particle-in-cell methods. Configured as `DMSWARM_PIC`, the swarm will be aware of, another `DM` which serves as the background mesh.

   Fields specific to particle-in-cell methods are registered by default. These include spatial coordinates, a unique identifier, a cell index and an index for
   the owning rank. The background mesh will (by default) define the spatial decomposition of the points defined in the swarm. `DMSWARM_PIC` provides support
   for particle-in-cell operations such as defining initial point coordinates, communicating particles between sub-domains, projecting particle data fields on to the mesh.

   Level: beginner

.seealso: [](ch_dmbase), `DMSWARM`, `DMSwarmSetType()`
E*/
typedef enum {
  DMSWARM_BASIC,
  DMSWARM_PIC
} DMSwarmType;
PETSC_EXTERN const char *DMSwarmTypeNames[];

typedef enum {
  DMSWARM_MIGRATE_BASIC,
  DMSWARM_MIGRATE_DMCELLNSCATTER,
  DMSWARM_MIGRATE_DMCELLEXACT,
  DMSWARM_MIGRATE_USER
} DMSwarmMigrateType;
PETSC_EXTERN const char *DMSwarmMigrateTypeNames[];

typedef enum {
  DMSWARM_COLLECT_BASIC,
  DMSWARM_COLLECT_DMDABOUNDINGBOX,
  DMSWARM_COLLECT_GENERAL,
  DMSWARM_COLLECT_USER
} DMSwarmCollectType;
PETSC_EXTERN const char *DMSwarmCollectTypeNames[];

typedef enum {
  DMSWARM_REMAP_NONE    = 0,
  DMSWARM_REMAP_PFAK    = 1,
  DMSWARM_REMAP_COLELLA = 2
} DMSwarmRemapType;
PETSC_EXTERN const char *DMSwarmRemapTypeNames[];

/*E
   DMSwarmPICLayoutType - Defines the method used to define particle coordinates within each cell. The layouts are constructured using the reference cell geometry

   Values:
+  `DMSWARMPIC_LAYOUT_REGULAR`     - defines points on a regular ijk mesh. In this case
                                     the `fill_param` argument of `DMSwarmInsertPointsUsingCellDM()` defines the number of points in each spatial direction.
.  `DMSWARMPIC_LAYOUT_GAUSS`       - defines points using an npoint Gauss-Legendre tensor product quadrature rule. In this case
                                     the `fill_param` argument of `DMSwarmInsertPointsUsingCellDM()` defines the number of quadrature points in each spatial direction.
-  `DMSWARMPIC_LAYOUT_SUBDIVISION` - defines points on the centroid of a sub-divided reference cell. In this case
                                     the `fill_param` argument of `DMSwarmInsertPointsUsingCellDM()` defines the number times the reference cell is sub-divided.

   Level: beginner

.seealso: [](ch_dmbase), `DMSWARM`, `DM`, `DMSwarmInsertPointsUsingCellDM()`
E*/
typedef enum {
  DMSWARMPIC_LAYOUT_REGULAR,
  DMSWARMPIC_LAYOUT_GAUSS,
  DMSWARMPIC_LAYOUT_SUBDIVISION
} DMSwarmPICLayoutType;

/*S
    DMSwarmCellDM - PETSc object for defining a background `DM` for the `DMSWARM` particles

    Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMSwarmAddCellDM()`, `DMSwarmCellDMCreate()`
S*/
typedef struct _p_DMSwarmCellDM *DMSwarmCellDM;
PETSC_EXTERN PetscClassId        DMSWARMCELLDM_CLASSID;

PETSC_EXTERN const char DMSwarmField_pid[];
PETSC_EXTERN const char DMSwarmField_rank[];
PETSC_EXTERN const char DMSwarmPICField_coor[];

PETSC_EXTERN PetscErrorCode DMSwarmCreateGlobalVectorFromField(DM, const char[], Vec *);
PETSC_EXTERN PetscErrorCode DMSwarmDestroyGlobalVectorFromField(DM, const char[], Vec *);
PETSC_EXTERN PetscErrorCode DMSwarmCreateLocalVectorFromField(DM, const char[], Vec *);
PETSC_EXTERN PetscErrorCode DMSwarmDestroyLocalVectorFromField(DM, const char[], Vec *);
PETSC_EXTERN PetscErrorCode DMSwarmCreateGlobalVectorFromFields(DM, PetscInt, const char *[], Vec *);
PETSC_EXTERN PetscErrorCode DMSwarmDestroyGlobalVectorFromFields(DM, PetscInt, const char *[], Vec *);
PETSC_EXTERN PetscErrorCode DMSwarmCreateLocalVectorFromFields(DM, PetscInt, const char *[], Vec *);
PETSC_EXTERN PetscErrorCode DMSwarmDestroyLocalVectorFromFields(DM, PetscInt, const char *[], Vec *);

PETSC_EXTERN PetscErrorCode DMSwarmInitializeFieldRegister(DM);
PETSC_EXTERN PetscErrorCode DMSwarmFinalizeFieldRegister(DM);
PETSC_EXTERN PetscErrorCode DMSwarmSetLocalSizes(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterPetscDatatypeField(DM, const char[], PetscInt, PetscDataType);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserStructField(DM, const char[], size_t);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserDatatypeField(DM, const char[], size_t, PetscInt);
PETSC_EXTERN PetscErrorCode DMSwarmGetField(DM, const char[], PetscInt *, PetscDataType *, void **);
PETSC_EXTERN PetscErrorCode DMSwarmRestoreField(DM, const char[], PetscInt *, PetscDataType *, void **);
PETSC_EXTERN PetscErrorCode DMSwarmGetFieldInfo(DM, const char[], PetscInt *, PetscDataType *);

PETSC_EXTERN PetscErrorCode DMSwarmVectorDefineField(DM, const char[]);
PETSC_EXTERN PetscErrorCode DMSwarmVectorDefineFields(DM, PetscInt, const char *[]);
PETSC_EXTERN PetscErrorCode DMSwarmVectorGetField(DM, PetscInt *, const char **[]);

PETSC_EXTERN PetscErrorCode DMSwarmAddPoint(DM);
PETSC_EXTERN PetscErrorCode DMSwarmAddNPoints(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMSwarmRemovePoint(DM);
PETSC_EXTERN PetscErrorCode DMSwarmRemovePointAtIndex(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMSwarmCopyPoint(DM, PetscInt, PetscInt);

PETSC_EXTERN PetscErrorCode DMSwarmGetLocalSize(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMSwarmGetSize(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMSwarmGetMigrateType(DM, DMSwarmMigrateType *);
PETSC_EXTERN PetscErrorCode DMSwarmSetMigrateType(DM, DMSwarmMigrateType);
PETSC_EXTERN PetscErrorCode DMSwarmMigrate(DM, PetscBool);

PETSC_EXTERN PetscErrorCode DMSwarmCollectViewCreate(DM);
PETSC_EXTERN PetscErrorCode DMSwarmCollectViewDestroy(DM);
PETSC_EXTERN PetscErrorCode DMSwarmSetCellDM(DM, DM);
PETSC_EXTERN PetscErrorCode DMSwarmGetCellDM(DM, DM *);
PETSC_EXTERN PetscErrorCode DMSwarmGetCellDMByName(DM, const char[], DMSwarmCellDM *);
PETSC_EXTERN PetscErrorCode DMSwarmGetCellDMNames(DM, PetscInt *, const char **[]);
PETSC_EXTERN PetscErrorCode DMSwarmSetCellDMActive(DM, const char[]);
PETSC_EXTERN PetscErrorCode DMSwarmGetCellDMActive(DM, DMSwarmCellDM *);
PETSC_EXTERN PetscErrorCode DMSwarmAddCellDM(DM, DMSwarmCellDM);

PETSC_EXTERN PetscErrorCode DMSwarmGetType(DM, DMSwarmType *);
PETSC_EXTERN PetscErrorCode DMSwarmSetType(DM, DMSwarmType);

PETSC_EXTERN PetscErrorCode DMSwarmSetPointsUniformCoordinates(DM, PetscReal *, PetscReal *, PetscInt *, InsertMode);
PETSC_EXTERN PetscErrorCode DMSwarmSetPointCoordinates(DM, PetscInt, PetscReal *, PetscBool, InsertMode);
PETSC_EXTERN PetscErrorCode DMSwarmInsertPointsUsingCellDM(DM, DMSwarmPICLayoutType, PetscInt);
PETSC_EXTERN PetscErrorCode DMSwarmSetPointCoordinatesCellwise(DM, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode DMSwarmSetPointCoordinatesRandom(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMSwarmViewFieldsXDMF(DM, const char *, PetscInt, const char **);
PETSC_EXTERN PetscErrorCode DMSwarmViewXDMF(DM, const char *);

PETSC_EXTERN PetscErrorCode DMSwarmSortDestroy(DMSwarmSort *);
PETSC_EXTERN PetscErrorCode DMSwarmSortGetAccess(DM);
PETSC_EXTERN PetscErrorCode DMSwarmSortRestoreAccess(DM);
PETSC_EXTERN PetscErrorCode DMSwarmSortGetPointsPerCell(DM, PetscInt, PetscInt *, PetscInt **);
PETSC_EXTERN PetscErrorCode DMSwarmSortRestorePointsPerCell(DM, PetscInt, PetscInt *, PetscInt **);
PETSC_EXTERN PetscErrorCode DMSwarmSortGetNumberOfPointsPerCell(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMSwarmSortGetIsValid(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMSwarmSortGetSizes(DM, PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode DMSwarmCreateMassMatrixSquare(DM, DM, Mat *);

PETSC_EXTERN PetscErrorCode DMSwarmGetCellSwarm(DM, PetscInt, DM);
PETSC_EXTERN PetscErrorCode DMSwarmRestoreCellSwarm(DM, PetscInt, DM);
PETSC_EXTERN PetscErrorCode DMSwarmGetNumSpecies(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMSwarmSetNumSpecies(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMSwarmGetCoordinateFunction(DM, PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *));
PETSC_EXTERN PetscErrorCode DMSwarmSetCoordinateFunction(DM, PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *));
PETSC_EXTERN PetscErrorCode DMSwarmGetVelocityFunction(DM, PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *));
PETSC_EXTERN PetscErrorCode DMSwarmSetVelocityFunction(DM, PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *));
PETSC_EXTERN PetscErrorCode DMSwarmComputeLocalSize(DM, PetscInt, PetscProbFn *);
PETSC_EXTERN PetscErrorCode DMSwarmComputeLocalSizeFromOptions(DM);
PETSC_EXTERN PetscErrorCode DMSwarmInitializeCoordinates(DM);
PETSC_EXTERN PetscErrorCode DMSwarmInitializeVelocities(DM, PetscProbFn *, const PetscReal[]);
PETSC_EXTERN PetscErrorCode DMSwarmInitializeVelocitiesFromOptions(DM, const PetscReal[]);

PETSC_EXTERN PetscErrorCode DMSwarmCreatePointPerCellCount(DM, PetscInt *, PetscInt **);

// Interface to internal storage
PETSC_EXTERN PetscErrorCode DMSwarmDataFieldGetEntries(const DMSwarmDataField, void **);
PETSC_EXTERN PetscErrorCode DMSwarmDataFieldRestoreEntries(const DMSwarmDataField, void **);
PETSC_EXTERN PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFieldByName(DMSwarmDataBucket, const char[], DMSwarmDataField *);
PETSC_EXTERN PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFieldIdByName(DMSwarmDataBucket, const char[], PetscInt *);
PETSC_EXTERN PetscErrorCode DMSwarmDataBucketQueryDMSwarmDataFieldByName(DMSwarmDataBucket, const char[], PetscBool *);

PETSC_EXTERN PetscErrorCode DMSwarmDuplicate(DM, DM *);
PETSC_EXTERN PetscErrorCode DMSwarmRemap(DM);
PETSC_EXTERN PetscErrorCode DMSwarmReplace(DM, DM *);

PETSC_EXTERN PetscErrorCode DMSwarmComputeMoments(DM, const char[], const char[], PetscReal[]);

PETSC_EXTERN PetscErrorCode DMSwarmCellDMCreate(DM, PetscInt, const char *[], PetscInt, const char *[], DMSwarmCellDM *);
PETSC_EXTERN PetscErrorCode DMSwarmCellDMDestroy(DMSwarmCellDM *);
PETSC_EXTERN PetscErrorCode DMSwarmCellDMView(DMSwarmCellDM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMSwarmCellDMGetDM(DMSwarmCellDM, DM *);
PETSC_EXTERN PetscErrorCode DMSwarmCellDMGetFields(DMSwarmCellDM, PetscInt *, const char **[]);
PETSC_EXTERN PetscErrorCode DMSwarmCellDMGetCoordinateFields(DMSwarmCellDM, PetscInt *, const char **[]);
PETSC_EXTERN PetscErrorCode DMSwarmCellDMGetCellID(DMSwarmCellDM, const char *[]);
PETSC_EXTERN PetscErrorCode DMSwarmCellDMGetSort(DMSwarmCellDM, DMSwarmSort *);
PETSC_EXTERN PetscErrorCode DMSwarmCellDMSetSort(DMSwarmCellDM, DMSwarmSort);
PETSC_EXTERN PetscErrorCode DMSwarmCellDMGetBlockSize(DMSwarmCellDM, DM, PetscInt *);
