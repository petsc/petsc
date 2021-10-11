#if !defined(PETSCDMPLEXTRANSFORM_H)
#define PETSCDMPLEXTRANSFORM_H

#include <petscdmplex.h>
#include <petscdmplextransformtypes.h>

PETSC_EXTERN PetscClassId DMPLEXTRANSFORM_CLASSID;

typedef const char* DMPlexTransformType;
#define DMPLEXREFINEREGULAR       "refine_regular"
#define DMPLEXREFINEALFELD        "refine_alfeld"
#define DMPLEXREFINEPOWELLSABIN   "refine_powell_sabin"
#define DMPLEXREFINEBOUNDARYLAYER "refine_boundary_layer"
#define DMPLEXREFINESBR           "refine_sbr"
#define DMPLEXREFINETOBOX         "refine_tobox"
#define DMPLEXREFINETOSIMPLEX     "refine_tosimplex"
#define DMPLEXREFINE1D            "refine_1d"
#define DMPLEXEXTRUDE             "extrude"
#define DMPLEXTRANSFORMFILTER     "transform_filter"

PETSC_EXTERN PetscFunctionList DMPlexTransformList;
PETSC_EXTERN PetscErrorCode DMPlexTransformCreate(MPI_Comm, DMPlexTransform *);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetType(DMPlexTransform, DMPlexTransformType);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetType(DMPlexTransform, DMPlexTransformType *);
PETSC_EXTERN PetscErrorCode DMPlexTransformRegister(const char[], PetscErrorCode (*)(DMPlexTransform));
PETSC_EXTERN PetscErrorCode DMPlexTransformRegisterAll(void);
PETSC_EXTERN PetscErrorCode DMPlexTransformRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetFromOptions(DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetUp(DMPlexTransform);
PETSC_EXTERN PetscErrorCode DMPlexTransformView(DMPlexTransform, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexTransformDestroy(DMPlexTransform *);

PETSC_EXTERN PetscErrorCode DMPlexGetTransformType(DM, DMPlexTransformType *);
PETSC_EXTERN PetscErrorCode DMPlexSetTransformType(DM, DMPlexTransformType);

PETSC_EXTERN PetscErrorCode DMPlexTransformGetDM(DMPlexTransform, DM *);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetDM(DMPlexTransform, DM);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetDimensions(DMPlexTransform, DM, DM);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetActive(DMPlexTransform, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexTransformSetActive(DMPlexTransform, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetTargetPoint(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetSourcePoint(DMPlexTransform, PetscInt, DMPolytopeType *, DMPolytopeType *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformCellTransform(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt *, PetscInt *, DMPolytopeType *[], PetscInt *[], PetscInt *[], PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformCellTransformIdentity(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt *, PetscInt *, DMPolytopeType *[], PetscInt *[], PetscInt *[], PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetSubcellOrientationIdentity(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetSubcellOrientation(DMPlexTransform, DMPolytopeType, PetscInt, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexTransformMapCoordinates(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformCreateDiscLabels(DMPlexTransform, DM);
PETSC_EXTERN PetscErrorCode DMPlexTransformApply(DMPlexTransform, DM, DM *);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetCone(DMPlexTransform, PetscInt, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetConeOriented(DMPlexTransform, PetscInt, PetscInt, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformRestoreCone(DMPlexTransform, PetscInt, const PetscInt *[], const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetCellVertices(DMPlexTransform, DMPolytopeType, PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformGetSubcellVertices(DMPlexTransform, DMPolytopeType, DMPolytopeType, PetscInt, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformAdaptLabel(DM, Vec, DMLabel, DM *);

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
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetNormal(DMPlexTransform, const PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexTransformExtrudeSetThicknesses(DMPlexTransform, PetscInt, const PetscReal[]);

#endif
