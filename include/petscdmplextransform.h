#pragma once

#include <petscdmplex.h>
#include <petscdmplextransformtypes.h>

/* SUBMANSEC = DM */

PETSC_EXTERN PetscClassId DMPLEXTRANSFORM_CLASSID;

/*J
   DMPlexTransformType - String with the name of a PETSc `DMPlexTransformType`

   Level: beginner

   Note:
   [](plex_transform_table) for a table of available transformation types

.seealso: [](plex_transform_table), [](ch_unstructured), `DMPlexTransformCreate()`, `DMPlexTransform`, `DMPlexTransformRegister()`
J*/
typedef const char *DMPlexTransformType;
#define DMPLEXREFINEREGULAR       "refine_regular"
#define DMPLEXREFINEALFELD        "refine_alfeld"
#define DMPLEXREFINEPOWELLSABIN   "refine_powell_sabin"
#define DMPLEXREFINEBOUNDARYLAYER "refine_boundary_layer"
#define DMPLEXREFINESBR           "refine_sbr"
#define DMPLEXREFINETOBOX         "refine_tobox"
#define DMPLEXREFINETOSIMPLEX     "refine_tosimplex"
#define DMPLEXREFINE1D            "refine_1d"
#define DMPLEXEXTRUDETYPE         "extrude"
#define DMPLEXCOHESIVEEXTRUDE     "cohesive_extrude"
#define DMPLEXTRANSFORMFILTER     "transform_filter"

PETSC_EXTERN PetscFunctionList DMPlexTransformList;
PETSC_EXTERN PetscErrorCode    DMPlexTransformCreate(MPI_Comm, DMPlexTransform *);
PETSC_EXTERN PetscErrorCode    DMPlexTransformSetType(DMPlexTransform, DMPlexTransformType);
PETSC_EXTERN PetscErrorCode    DMPlexTransformGetType(DMPlexTransform, DMPlexTransformType *);
PETSC_EXTERN PetscErrorCode    DMPlexTransformRegister(const char[], PetscErrorCode (*)(DMPlexTransform));
PETSC_EXTERN PetscErrorCode    DMPlexTransformRegisterAll(void);
PETSC_EXTERN PetscErrorCode    DMPlexTransformRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode    DMPlexTransformSetFromOptions(DMPlexTransform);
PETSC_EXTERN PetscErrorCode    DMPlexTransformSetUp(DMPlexTransform);
PETSC_EXTERN PetscErrorCode    DMPlexTransformView(DMPlexTransform, PetscViewer);
PETSC_EXTERN PetscErrorCode    DMPlexTransformDestroy(DMPlexTransform *);

PETSC_EXTERN PetscErrorCode DMPlexGetTransformType(DM, DMPlexTransformType *);
PETSC_EXTERN PetscErrorCode DMPlexSetTransformType(DM, DMPlexTransformType);
PETSC_EXTERN PetscErrorCode DMPlexGetTransform(DM, DMPlexTransform *);
PETSC_EXTERN PetscErrorCode DMPlexSetTransform(DM, DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexGetSaveTransform(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexSetSaveTransform(DM, PetscBool);

PETSC_EXTERN PetscErrorCode DMPlexTransformGetDM(DMPlexTransform, DM *);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetDM(DMPlexTransform, DM);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetDimensions(DMPlexTransform, DM, DM);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetChart(DMPlexTransform, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetCellType(DMPlexTransform, PetscInt, DMPolytopeType *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetCellTypeStratum(DMPlexTransform, DMPolytopeType, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetDepth(DMPlexTransform, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetDepthStratum(DMPlexTransform, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetActive(DMPlexTransform, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetActive(DMPlexTransform, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetTransformTypes(DMPlexTransform, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetTransformTypes(DMPlexTransform, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetMatchStrata(DMPlexTransform, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetMatchStrata(DMPlexTransform, PetscBool);

PETSC_EXTERN PetscErrorCode DMPlexTransformGetTargetPoint(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetSourcePoint(DMPlexTransform, PetscInt, DMPolytopeType *, DMPolytopeType *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformCellTransform(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt *, PetscInt *, DMPolytopeType *[], PetscInt *[], PetscInt *[], PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformCellTransformIdentity(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt *, PetscInt *, DMPolytopeType *[], PetscInt *[], PetscInt *[], PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetSubcellOrientationIdentity(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetSubcellOrientation(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformMapCoordinates(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformCreateDiscLabels(DMPlexTransform, DM);
PETSC_EXTERN PetscErrorCode DMPlexTransformApply(DMPlexTransform, DM, DM *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetConeSize(DMPlexTransform, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetCone(DMPlexTransform, PetscInt, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetConeOriented(DMPlexTransform, PetscInt, PetscInt, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformRestoreCone(DMPlexTransform, PetscInt, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetCellVertices(DMPlexTransform, DMPolytopeType, PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetSubcellVertices(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformAdaptLabel(DM, Vec, DMLabel, DMLabel, DM *);

PETSC_EXTERN PetscErrorCode DMPlexRefineRegularGetAffineTransforms(DMPlexTransform, DMPolytopeType, PetscInt *, PetscReal *[], PetscReal *[], PetscReal *[]);
PETSC_EXTERN PetscErrorCode DMPlexRefineRegularGetAffineFaceTransforms(DMPlexTransform, DMPolytopeType, PetscInt *, PetscReal *[], PetscReal *[], PetscReal *[], PetscReal *[]);

PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeGetLayers(DMPlexTransform, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetLayers(DMPlexTransform, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeGetThickness(DMPlexTransform, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetThickness(DMPlexTransform, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeGetTensor(DMPlexTransform, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetTensor(DMPlexTransform, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeGetSymmetric(DMPlexTransform, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetSymmetric(DMPlexTransform, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeGetPeriodic(DMPlexTransform, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetPeriodic(DMPlexTransform, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeGetNormal(DMPlexTransform, PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetNormal(DMPlexTransform, const PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetNormalFunction(DMPlexTransform, PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *));
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetThicknesses(DMPlexTransform, PetscInt, const PetscReal[]);

PETSC_EXTERN PetscErrorCode DMPlexTransformCohesiveExtrudeGetTensor(DMPlexTransform, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexTransformCohesiveExtrudeSetTensor(DMPlexTransform, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexTransformCohesiveExtrudeGetWidth(DMPlexTransform, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexTransformCohesiveExtrudeSetWidth(DMPlexTransform, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexTransformCohesiveExtrudeGetUnsplit(DMPlexTransform, DMLabel *);

PETSC_EXTERN PetscErrorCode DMPlexCreateEphemeral(DMPlexTransform, const char[], DM *);
