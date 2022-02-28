/*
      Objects to manage the interactions between the mesh data structures and the algebraic objects
*/
#if !defined(PETSCDM_H)
#define PETSCDM_H
#include <petscmat.h>
#include <petscdmtypes.h>
#include <petscfetypes.h>
#include <petscdstypes.h>
#include <petscdmlabel.h>

PETSC_EXTERN PetscErrorCode DMInitializePackage(void);

PETSC_EXTERN PetscClassId DM_CLASSID;
PETSC_EXTERN PetscClassId DMLABEL_CLASSID;

#define DMLOCATEPOINT_POINT_NOT_FOUND -367

/*J
    DMType - String with the name of a PETSc DM

   Level: beginner

.seealso: DMSetType(), DM
J*/
typedef const char* DMType;
#define DMDA        "da"
#define DMCOMPOSITE "composite"
#define DMSLICED    "sliced"
#define DMSHELL     "shell"
#define DMPLEX      "plex"
#define DMREDUNDANT "redundant"
#define DMPATCH     "patch"
#define DMMOAB      "moab"
#define DMNETWORK   "network"
#define DMFOREST    "forest"
#define DMP4EST     "p4est"
#define DMP8EST     "p8est"
#define DMSWARM     "swarm"
#define DMPRODUCT   "product"
#define DMSTAG      "stag"

PETSC_EXTERN const char *const DMBoundaryTypes[];
PETSC_EXTERN const char *const DMBoundaryConditionTypes[];
PETSC_EXTERN PetscFunctionList DMList;
PETSC_EXTERN DMGeneratorFunctionList DMGenerateList;
PETSC_EXTERN PetscErrorCode DMCreate(MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMClone(DM,DM*);
PETSC_EXTERN PetscErrorCode DMSetType(DM, DMType);
PETSC_EXTERN PetscErrorCode DMGetType(DM, DMType *);
PETSC_EXTERN PetscErrorCode DMRegister(const char[],PetscErrorCode (*)(DM));
PETSC_EXTERN PetscErrorCode DMRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode DMView(DM,PetscViewer);
PETSC_EXTERN PetscErrorCode DMLoad(DM,PetscViewer);
PETSC_EXTERN PetscErrorCode DMDestroy(DM*);
PETSC_EXTERN PetscErrorCode DMCreateGlobalVector(DM,Vec*);
PETSC_EXTERN PetscErrorCode DMCreateLocalVector(DM,Vec*);
PETSC_EXTERN PetscErrorCode DMGetLocalVector(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMRestoreLocalVector(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMGetGlobalVector(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMRestoreGlobalVector(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMClearGlobalVectors(DM);
PETSC_EXTERN PetscErrorCode DMClearLocalVectors(DM);
PETSC_EXTERN PetscErrorCode DMHasNamedGlobalVector(DM,const char*,PetscBool*);
PETSC_EXTERN PetscErrorCode DMGetNamedGlobalVector(DM,const char*,Vec*);
PETSC_EXTERN PetscErrorCode DMRestoreNamedGlobalVector(DM,const char*,Vec*);
PETSC_EXTERN PetscErrorCode DMHasNamedLocalVector(DM,const char*,PetscBool*);
PETSC_EXTERN PetscErrorCode DMGetNamedLocalVector(DM,const char*,Vec*);
PETSC_EXTERN PetscErrorCode DMRestoreNamedLocalVector(DM,const char*,Vec*);
PETSC_EXTERN PetscErrorCode DMGetLocalToGlobalMapping(DM,ISLocalToGlobalMapping*);
PETSC_EXTERN PetscErrorCode DMCreateFieldIS(DM,PetscInt*,char***,IS**);
PETSC_EXTERN PetscErrorCode DMGetBlockSize(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMCreateColoring(DM,ISColoringType,ISColoring*);
PETSC_EXTERN PetscErrorCode DMCreateMatrix(DM,Mat*);
PETSC_EXTERN PetscErrorCode DMSetMatrixPreallocateOnly(DM,PetscBool);
PETSC_EXTERN PetscErrorCode DMSetMatrixStructureOnly(DM,PetscBool);
PETSC_EXTERN PetscErrorCode DMCreateInterpolation(DM,DM,Mat*,Vec*);
PETSC_EXTERN PetscErrorCode DMCreateRestriction(DM,DM,Mat*);
PETSC_EXTERN PetscErrorCode DMCreateInjection(DM,DM,Mat*);
PETSC_EXTERN PetscErrorCode DMCreateMassMatrix(DM,DM,Mat*);
PETSC_EXTERN PetscErrorCode DMCreateMassMatrixLumped(DM,Vec*);
PETSC_EXTERN PetscErrorCode DMGetWorkArray(DM,PetscInt,MPI_Datatype,void*);
PETSC_EXTERN PetscErrorCode DMRestoreWorkArray(DM,PetscInt,MPI_Datatype,void*);
PETSC_EXTERN PetscErrorCode DMRefine(DM,MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMCoarsen(DM,MPI_Comm,DM*);
PETSC_EXTERN PetscErrorCode DMGetCoarseDM(DM,DM*);
PETSC_EXTERN PetscErrorCode DMSetCoarseDM(DM,DM);
PETSC_EXTERN PetscErrorCode DMGetFineDM(DM,DM*);
PETSC_EXTERN PetscErrorCode DMSetFineDM(DM,DM);
PETSC_EXTERN PetscErrorCode DMRefineHierarchy(DM,PetscInt,DM[]);
PETSC_EXTERN PetscErrorCode DMCoarsenHierarchy(DM,PetscInt,DM[]);
PETSC_EXTERN PetscErrorCode DMCoarsenHookAdd(DM,PetscErrorCode (*)(DM,DM,void*),PetscErrorCode (*)(DM,Mat,Vec,Mat,DM,void*),void*);
PETSC_EXTERN PetscErrorCode DMCoarsenHookRemove(DM,PetscErrorCode (*)(DM,DM,void*),PetscErrorCode (*)(DM,Mat,Vec,Mat,DM,void*),void*);
PETSC_EXTERN PetscErrorCode DMRefineHookAdd(DM,PetscErrorCode (*)(DM,DM,void*),PetscErrorCode (*)(DM,Mat,DM,void*),void*);
PETSC_EXTERN PetscErrorCode DMRefineHookRemove(DM,PetscErrorCode (*)(DM,DM,void*),PetscErrorCode (*)(DM,Mat,DM,void*),void*);
PETSC_EXTERN PetscErrorCode DMRestrict(DM,Mat,Vec,Mat,DM);
PETSC_EXTERN PetscErrorCode DMInterpolate(DM,Mat,DM);
PETSC_EXTERN PetscErrorCode DMInterpolateSolution(DM,DM,Mat,Vec,Vec);
PETSC_EXTERN PetscErrorCode DMExtrude(DM,PetscInt,DM*);
PETSC_EXTERN PetscErrorCode DMSetFromOptions(DM);
PETSC_EXTERN PetscErrorCode DMViewFromOptions(DM,PetscObject,const char[]);

PETSC_EXTERN PetscErrorCode DMGenerate(DM, const char [], PetscBool , DM *);
PETSC_EXTERN PetscErrorCode DMGenerateRegister(const char[],PetscErrorCode (*)(DM,PetscBool,DM*),PetscErrorCode (*)(DM,PetscReal*,DM*),PetscErrorCode (*)(DM,Vec,DMLabel,DMLabel,DM*),PetscInt);
PETSC_EXTERN PetscErrorCode DMGenerateRegisterAll(void);
PETSC_EXTERN PetscErrorCode DMGenerateRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode DMAdaptLabel(DM,DMLabel,DM*);
PETSC_EXTERN PetscErrorCode DMAdaptMetric(DM, Vec, DMLabel, DMLabel, DM *);

PETSC_EXTERN PetscErrorCode DMSetUp(DM);
PETSC_EXTERN PetscErrorCode DMCreateInterpolationScale(DM,DM,Mat,Vec*);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use DMDACreateAggregates() or DMCreateRestriction() (since version 3.12)") PetscErrorCode DMCreateAggregates(DM,DM,Mat*);
PETSC_EXTERN PetscErrorCode DMGlobalToLocalHookAdd(DM,PetscErrorCode (*)(DM,Vec,InsertMode,Vec,void*),PetscErrorCode (*)(DM,Vec,InsertMode,Vec,void*),void*);
PETSC_EXTERN PetscErrorCode DMLocalToGlobalHookAdd(DM,PetscErrorCode (*)(DM,Vec,InsertMode,Vec,void*),PetscErrorCode (*)(DM,Vec,InsertMode,Vec,void*),void*);
PETSC_EXTERN PetscErrorCode DMGlobalToLocal(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMGlobalToLocalBegin(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMGlobalToLocalEnd(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToGlobal(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToGlobalBegin(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToGlobalEnd(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToLocalBegin(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToLocalEnd(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMConvert(DM,DMType,DM*);

/* Topology support */
PETSC_EXTERN PetscErrorCode DMGetDimension(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMSetDimension(DM,PetscInt);
PETSC_EXTERN PetscErrorCode DMGetDimPoints(DM,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMGetUseNatural(DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMSetUseNatural(DM,PetscBool);

/* Coordinate support */
PETSC_EXTERN PetscErrorCode DMGetCoordinateDM(DM,DM*);
PETSC_EXTERN PetscErrorCode DMSetCoordinateDM(DM,DM);
PETSC_EXTERN PetscErrorCode DMGetCoordinateDim(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMSetCoordinateDim(DM,PetscInt);
PETSC_EXTERN PetscErrorCode DMGetCoordinateSection(DM,PetscSection*);
PETSC_EXTERN PetscErrorCode DMSetCoordinateSection(DM,PetscInt,PetscSection);
PETSC_EXTERN PetscErrorCode DMGetCoordinates(DM,Vec*);
PETSC_EXTERN PetscErrorCode DMSetCoordinates(DM,Vec);
PETSC_EXTERN PetscErrorCode DMGetCoordinatesLocal(DM,Vec*);
PETSC_EXTERN PetscErrorCode DMGetCoordinatesLocalSetUp(DM);
PETSC_EXTERN PetscErrorCode DMGetCoordinatesLocalNoncollective(DM,Vec*);
PETSC_EXTERN PetscErrorCode DMGetCoordinatesLocalTuple(DM,IS,PetscSection*,Vec*);
PETSC_EXTERN PetscErrorCode DMSetCoordinatesLocal(DM,Vec);
PETSC_EXTERN PetscErrorCode DMLocatePoints(DM,Vec,DMPointLocationType,PetscSF*);
PETSC_EXTERN PetscErrorCode DMGetPeriodicity(DM,PetscBool*,const PetscReal**,const PetscReal**,const DMBoundaryType**);
PETSC_EXTERN PetscErrorCode DMSetPeriodicity(DM,PetscBool,const PetscReal[],const PetscReal[],const DMBoundaryType[]);
PETSC_EXTERN PetscErrorCode DMLocalizeCoordinate(DM, const PetscScalar[], PetscBool, PetscScalar[]);
PETSC_EXTERN PetscErrorCode DMLocalizeCoordinates(DM);
PETSC_EXTERN PetscErrorCode DMGetCoordinatesLocalized(DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMGetCoordinatesLocalizedLocal(DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMGetNeighbors(DM,PetscInt*,const PetscMPIInt**);
PETSC_EXTERN PetscErrorCode DMGetCoordinateField(DM,DMField*);
PETSC_EXTERN PetscErrorCode DMSetCoordinateField(DM,DMField);
PETSC_EXTERN PetscErrorCode DMGetBoundingBox(DM,PetscReal[],PetscReal[]);
PETSC_EXTERN PetscErrorCode DMGetLocalBoundingBox(DM,PetscReal[],PetscReal[]);
PETSC_EXTERN PetscErrorCode DMProjectCoordinates(DM,PetscFE);

/* block hook interface */
PETSC_EXTERN PetscErrorCode DMSubDomainHookAdd(DM,PetscErrorCode (*)(DM,DM,void*),PetscErrorCode (*)(DM,VecScatter,VecScatter,DM,void*),void*);
PETSC_EXTERN PetscErrorCode DMSubDomainHookRemove(DM,PetscErrorCode (*)(DM,DM,void*),PetscErrorCode (*)(DM,VecScatter,VecScatter,DM,void*),void*);
PETSC_EXTERN PetscErrorCode DMSubDomainRestrict(DM,VecScatter,VecScatter,DM);

PETSC_EXTERN PetscErrorCode DMSetOptionsPrefix(DM,const char []);
PETSC_EXTERN PetscErrorCode DMAppendOptionsPrefix(DM,const char []);
PETSC_EXTERN PetscErrorCode DMGetOptionsPrefix(DM,const char*[]);
PETSC_EXTERN PetscErrorCode DMSetVecType(DM,VecType);
PETSC_EXTERN PetscErrorCode DMGetVecType(DM,VecType*);
PETSC_EXTERN PetscErrorCode DMSetMatType(DM,MatType);
PETSC_EXTERN PetscErrorCode DMGetMatType(DM,MatType*);
PETSC_EXTERN PetscErrorCode DMSetISColoringType(DM,ISColoringType);
PETSC_EXTERN PetscErrorCode DMGetISColoringType(DM,ISColoringType*);
PETSC_EXTERN PetscErrorCode DMSetApplicationContext(DM,void*);
PETSC_EXTERN PetscErrorCode DMSetApplicationContextDestroy(DM,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode DMGetApplicationContext(DM,void*);
PETSC_EXTERN PetscErrorCode DMSetVariableBounds(DM,PetscErrorCode (*)(DM,Vec,Vec));
PETSC_EXTERN PetscErrorCode DMHasVariableBounds(DM,PetscBool *);
PETSC_EXTERN PetscErrorCode DMHasColoring(DM,PetscBool *);
PETSC_EXTERN PetscErrorCode DMHasCreateRestriction(DM,PetscBool *);
PETSC_EXTERN PetscErrorCode DMHasCreateInjection(DM,PetscBool *);
PETSC_EXTERN PetscErrorCode DMComputeVariableBounds(DM,Vec,Vec);

PETSC_EXTERN PetscErrorCode DMCreateSubDM(DM, PetscInt, const PetscInt[], IS *, DM *);
PETSC_EXTERN PetscErrorCode DMCreateSuperDM(DM[], PetscInt, IS **, DM *);
PETSC_EXTERN PetscErrorCode DMCreateSectionSubDM(DM,PetscInt,const PetscInt[],IS*,DM*);
PETSC_EXTERN PetscErrorCode DMCreateSectionSuperDM(DM[],PetscInt,IS**,DM*);
PETSC_EXTERN PetscErrorCode DMCreateFieldDecomposition(DM,PetscInt*,char***,IS**,DM**);
PETSC_EXTERN PetscErrorCode DMCreateDomainDecomposition(DM,PetscInt*,char***,IS**,IS**,DM**);
PETSC_EXTERN PetscErrorCode DMCreateDomainDecompositionScatters(DM,PetscInt,DM*,VecScatter**,VecScatter**,VecScatter**);

PETSC_EXTERN PetscErrorCode DMGetRefineLevel(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMSetRefineLevel(DM,PetscInt);
PETSC_EXTERN PetscErrorCode DMGetCoarsenLevel(DM,PetscInt*);
PETSC_EXTERN PetscErrorCode DMSetCoarsenLevel(DM,PetscInt);
PETSC_EXTERN PetscErrorCode DMFinalizePackage(void);

PETSC_EXTERN PetscErrorCode VecGetDM(Vec, DM*);
PETSC_EXTERN PetscErrorCode VecSetDM(Vec, DM);
PETSC_EXTERN PetscErrorCode MatGetDM(Mat, DM*);
PETSC_EXTERN PetscErrorCode MatSetDM(Mat, DM);
PETSC_EXTERN PetscErrorCode MatFDColoringUseDM(Mat,MatFDColoring);

typedef struct NLF_DAAD* NLF;

#define DM_FILE_CLASSID 1211221

/* FEM support */
PETSC_EXTERN PetscErrorCode DMPrintCellVector(PetscInt, const char [], PetscInt, const PetscScalar []);
PETSC_EXTERN PetscErrorCode DMPrintCellMatrix(PetscInt, const char [], PetscInt, PetscInt, const PetscScalar []);
PETSC_EXTERN PetscErrorCode DMPrintLocalVec(DM, const char [], PetscReal, Vec);

PETSC_EXTERN PetscErrorCode DMSetNullSpaceConstructor(DM, PetscInt, PetscErrorCode (*)(DM, PetscInt, PetscInt, MatNullSpace *));
PETSC_EXTERN PetscErrorCode DMGetNullSpaceConstructor(DM, PetscInt, PetscErrorCode (**)(DM, PetscInt, PetscInt, MatNullSpace *));
PETSC_EXTERN PetscErrorCode DMSetNearNullSpaceConstructor(DM, PetscInt, PetscErrorCode (*)(DM, PetscInt, PetscInt, MatNullSpace *));
PETSC_EXTERN PetscErrorCode DMGetNearNullSpaceConstructor(DM, PetscInt, PetscErrorCode (**)(DM, PetscInt, PetscInt, MatNullSpace *));

PETSC_EXTERN PetscErrorCode DMGetSection(DM, PetscSection *); /* Use DMGetLocalSection() in new code (since v3.12) */
PETSC_EXTERN PetscErrorCode DMSetSection(DM, PetscSection);   /* Use DMSetLocalSection() in new code (since v3.12) */
PETSC_EXTERN PetscErrorCode DMGetLocalSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMSetLocalSection(DM, PetscSection);
PETSC_EXTERN PetscErrorCode DMGetGlobalSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMSetGlobalSection(DM, PetscSection);
static inline PETSC_DEPRECATED_FUNCTION("Use DMGetSection() (since v3.9)") PetscErrorCode DMGetDefaultSection(DM dm, PetscSection *s) {return DMGetSection(dm,s);}
static inline PETSC_DEPRECATED_FUNCTION("Use DMSetSection() (since v3.9)") PetscErrorCode DMSetDefaultSection(DM dm, PetscSection s) {return DMSetSection(dm,s);}
static inline PETSC_DEPRECATED_FUNCTION("Use DMGetGlobalSection() (since v3.9)") PetscErrorCode DMGetDefaultGlobalSection(DM dm, PetscSection *s) {return DMGetGlobalSection(dm,s);}
static inline PETSC_DEPRECATED_FUNCTION("Use DMSetGlobalSection() (since v3.9)") PetscErrorCode DMSetDefaultGlobalSection(DM dm, PetscSection s) {return DMSetGlobalSection(dm,s);}

PETSC_EXTERN PetscErrorCode DMGetSectionSF(DM, PetscSF*);
PETSC_EXTERN PetscErrorCode DMSetSectionSF(DM, PetscSF);
PETSC_EXTERN PetscErrorCode DMCreateSectionSF(DM, PetscSection, PetscSection);
static inline PETSC_DEPRECATED_FUNCTION("Use DMGetSectionSF() (since v3.12)") PetscErrorCode DMGetDefaultSF(DM dm, PetscSF *s) {return DMGetSectionSF(dm,s);}
static inline PETSC_DEPRECATED_FUNCTION("Use DMSetSectionSF() (since v3.12)") PetscErrorCode DMSetDefaultSF(DM dm, PetscSF s) {return DMSetSectionSF(dm,s);}
static inline PETSC_DEPRECATED_FUNCTION("Use DMCreateSectionSF() (since v3.12)") PetscErrorCode DMCreateDefaultSF(DM dm, PetscSection l, PetscSection g) {return DMCreateSectionSF(dm,l,g);}
PETSC_EXTERN PetscErrorCode DMGetPointSF(DM, PetscSF *);
PETSC_EXTERN PetscErrorCode DMSetPointSF(DM, PetscSF);
PETSC_EXTERN PetscErrorCode DMGetNaturalSF(DM, PetscSF *);
PETSC_EXTERN PetscErrorCode DMSetNaturalSF(DM, PetscSF);

PETSC_EXTERN PetscErrorCode DMGetDefaultConstraints(DM, PetscSection *, Mat *);
PETSC_EXTERN PetscErrorCode DMSetDefaultConstraints(DM, PetscSection, Mat);

PETSC_EXTERN PetscErrorCode DMGetOutputDM(DM, DM *);
PETSC_EXTERN PetscErrorCode DMGetOutputSequenceNumber(DM, PetscInt *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMSetOutputSequenceNumber(DM, PetscInt, PetscReal);
PETSC_EXTERN PetscErrorCode DMOutputSequenceLoad(DM, PetscViewer, const char *, PetscInt, PetscReal *);

PETSC_EXTERN PetscErrorCode DMGetNumFields(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMSetNumFields(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMGetField(DM, PetscInt, DMLabel *, PetscObject *);
PETSC_EXTERN PetscErrorCode DMSetField(DM, PetscInt, DMLabel, PetscObject);
PETSC_EXTERN PetscErrorCode DMAddField(DM, DMLabel, PetscObject);
PETSC_EXTERN PetscErrorCode DMSetFieldAvoidTensor(DM, PetscInt, PetscBool);
PETSC_EXTERN PetscErrorCode DMGetFieldAvoidTensor(DM, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMClearFields(DM);
PETSC_EXTERN PetscErrorCode DMCopyFields(DM, DM);
PETSC_EXTERN PetscErrorCode DMGetAdjacency(DM, PetscInt, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode DMSetAdjacency(DM, PetscInt, PetscBool, PetscBool);
PETSC_EXTERN PetscErrorCode DMGetBasicAdjacency(DM, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode DMSetBasicAdjacency(DM, PetscBool, PetscBool);

PETSC_EXTERN PetscErrorCode DMGetNumDS(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMGetDS(DM, PetscDS *);
PETSC_EXTERN PetscErrorCode DMGetCellDS(DM, PetscInt, PetscDS *);
PETSC_EXTERN PetscErrorCode DMGetRegionDS(DM, DMLabel, IS *, PetscDS *);
PETSC_EXTERN PetscErrorCode DMSetRegionDS(DM, DMLabel, IS, PetscDS);
PETSC_EXTERN PetscErrorCode DMGetRegionNumDS(DM, PetscInt, DMLabel *, IS *, PetscDS *);
PETSC_EXTERN PetscErrorCode DMSetRegionNumDS(DM, PetscInt, DMLabel, IS, PetscDS);
PETSC_EXTERN PetscErrorCode DMFindRegionNum(DM, PetscDS, PetscInt *);
PETSC_EXTERN PetscErrorCode DMCreateFEDefault(DM, PetscInt, const char[], PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode DMCreateDS(DM);
PETSC_EXTERN PetscErrorCode DMClearDS(DM);
PETSC_EXTERN PetscErrorCode DMCopyDS(DM, DM);
PETSC_EXTERN PetscErrorCode DMCopyDisc(DM, DM);
PETSC_EXTERN PetscErrorCode DMComputeExactSolution(DM, PetscReal, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMGetNumAuxiliaryVec(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMGetAuxiliaryVec(DM, DMLabel, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode DMSetAuxiliaryVec(DM, DMLabel, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode DMGetAuxiliaryLabels(DM, DMLabel[], PetscInt[]);
PETSC_EXTERN PetscErrorCode DMCopyAuxiliaryVec(DM, DM);

/*MC
  DMInterpolationInfo - Structure for holding information about interpolation on a mesh

  Level: intermediate

  Synopsis:
    comm   - The communicator
    dim    - The spatial dimension of points
    nInput - The number of input points
    points - The input point coordinates
    cells  - The cell containing each point
    n      - The number of local points
    coords - The point coordinates
    dof    - The number of components to interpolate

.seealso: DMInterpolationCreate(), DMInterpolationEvaluate(), DMInterpolationAddPoints()
M*/
struct _DMInterpolationInfo {
  MPI_Comm   comm;
  PetscInt   dim;    /*1 The spatial dimension of points */
  PetscInt   nInput; /* The number of input points */
  PetscReal *points; /* The input point coordinates */
  PetscInt  *cells;  /* The cell containing each point */
  PetscInt   n;      /* The number of local points */
  Vec        coords; /* The point coordinates */
  PetscInt   dof;    /* The number of components to interpolate */
};
typedef struct _DMInterpolationInfo *DMInterpolationInfo;

PETSC_EXTERN PetscErrorCode DMInterpolationCreate(MPI_Comm, DMInterpolationInfo *);
PETSC_EXTERN PetscErrorCode DMInterpolationSetDim(DMInterpolationInfo, PetscInt);
PETSC_EXTERN PetscErrorCode DMInterpolationGetDim(DMInterpolationInfo, PetscInt *);
PETSC_EXTERN PetscErrorCode DMInterpolationSetDof(DMInterpolationInfo, PetscInt);
PETSC_EXTERN PetscErrorCode DMInterpolationGetDof(DMInterpolationInfo, PetscInt *);
PETSC_EXTERN PetscErrorCode DMInterpolationAddPoints(DMInterpolationInfo, PetscInt, PetscReal[]);
PETSC_EXTERN PetscErrorCode DMInterpolationSetUp(DMInterpolationInfo, DM, PetscBool, PetscBool);
PETSC_EXTERN PetscErrorCode DMInterpolationGetCoordinates(DMInterpolationInfo, Vec *);
PETSC_EXTERN PetscErrorCode DMInterpolationGetVector(DMInterpolationInfo, Vec *);
PETSC_EXTERN PetscErrorCode DMInterpolationRestoreVector(DMInterpolationInfo, Vec *);
PETSC_EXTERN PetscErrorCode DMInterpolationEvaluate(DMInterpolationInfo, DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMInterpolationDestroy(DMInterpolationInfo *);

PETSC_EXTERN PetscErrorCode DMCreateLabel(DM, const char []);
PETSC_EXTERN PetscErrorCode DMGetLabelValue(DM, const char[], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMSetLabelValue(DM, const char[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMClearLabelValue(DM, const char[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMGetLabelSize(DM, const char[], PetscInt *);
PETSC_EXTERN PetscErrorCode DMGetLabelIdIS(DM, const char[], IS *);
PETSC_EXTERN PetscErrorCode DMGetStratumSize(DM, const char [], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMGetStratumIS(DM, const char [], PetscInt, IS *);
PETSC_EXTERN PetscErrorCode DMSetStratumIS(DM, const char [], PetscInt, IS);
PETSC_EXTERN PetscErrorCode DMClearLabelStratum(DM, const char[], PetscInt);
PETSC_EXTERN PetscErrorCode DMGetLabelOutput(DM, const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode DMSetLabelOutput(DM, const char[], PetscBool);
PETSC_EXTERN PetscErrorCode DMGetFirstLabeledPoint(DM, DM, DMLabel, PetscInt, const PetscInt *, PetscInt, PetscInt *, PetscDS *);

/*E
   DMCopyLabelsMode - Determines how DMCopyLabels() behaves when there is a DMLabel in the source and destination DMs with the same name

   Level: advanced

$ DM_COPY_LABELS_REPLACE  - replace label in destination by label from source
$ DM_COPY_LABELS_KEEP     - keep destination label
$ DM_COPY_LABELS_FAIL     - throw error

E*/
typedef enum {DM_COPY_LABELS_REPLACE, DM_COPY_LABELS_KEEP, DM_COPY_LABELS_FAIL} DMCopyLabelsMode;
PETSC_EXTERN const char *const DMCopyLabelsModes[];

PETSC_EXTERN PetscErrorCode DMGetNumLabels(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMGetLabelName(DM, PetscInt, const char **);
PETSC_EXTERN PetscErrorCode DMHasLabel(DM, const char [], PetscBool *);
PETSC_EXTERN PetscErrorCode DMGetLabel(DM, const char *, DMLabel *);
PETSC_EXTERN PetscErrorCode DMSetLabel(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMGetLabelByNum(DM, PetscInt, DMLabel *);
PETSC_EXTERN PetscErrorCode DMAddLabel(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMRemoveLabel(DM, const char [], DMLabel *);
PETSC_EXTERN PetscErrorCode DMRemoveLabelBySelf(DM, DMLabel *, PetscBool);
PETSC_EXTERN PetscErrorCode DMCopyLabels(DM, DM, PetscCopyMode, PetscBool, DMCopyLabelsMode emode);
PETSC_EXTERN PetscErrorCode DMCompareLabels(DM, DM, PetscBool *, char **);

PETSC_EXTERN PetscErrorCode DMAddBoundary(DM, DMBoundaryConditionType, const char[], DMLabel, PetscInt, const PetscInt[], PetscInt, PetscInt, const PetscInt[], void (*)(void), void (*)(void), void *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMIsBoundaryPoint(DM, PetscInt, PetscBool *);

PETSC_EXTERN PetscErrorCode DMProjectFunction(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void**,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMProjectFunctionLocal(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void**,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMProjectFunctionLabel(DM, PetscReal, DMLabel, PetscInt, const PetscInt[], PetscInt, const PetscInt[], PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **, InsertMode, Vec);
PETSC_EXTERN PetscErrorCode DMProjectFunctionLabelLocal(DM,PetscReal,DMLabel,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void **,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMProjectFieldLocal(DM,PetscReal,Vec,void (**)(PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],PetscReal,const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]),InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMProjectFieldLabelLocal(DM,PetscReal,DMLabel,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Vec,void (**)(PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],PetscReal,const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]),InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMProjectBdFieldLabelLocal(DM,PetscReal,DMLabel,PetscInt,const PetscInt[],PetscInt,const PetscInt[],Vec,void (**)(PetscInt,PetscInt,PetscInt,const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],const PetscInt[],const PetscInt[],const PetscScalar[],const PetscScalar[],const PetscScalar[],PetscReal,const PetscReal[],const PetscReal[],PetscInt,const PetscScalar[],PetscScalar[]),InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMComputeL2Diff(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void **,Vec,PetscReal *);
PETSC_EXTERN PetscErrorCode DMComputeL2GradientDiff(DM, PetscReal, PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal [], const PetscReal [], PetscInt, PetscScalar *, void *), void **, Vec, const PetscReal [], PetscReal *);
PETSC_EXTERN PetscErrorCode DMComputeL2FieldDiff(DM,PetscReal,PetscErrorCode(**)(PetscInt,PetscReal,const PetscReal[],PetscInt,PetscScalar *,void *),void **,Vec,PetscReal *);
PETSC_EXTERN PetscErrorCode DMComputeError(DM, Vec, PetscReal[], Vec *);
PETSC_EXTERN PetscErrorCode DMHasBasisTransform(DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMCopyTransform(DM, DM);

PETSC_EXTERN PetscErrorCode DMGetCompatibility(DM,DM,PetscBool*,PetscBool*);

PETSC_EXTERN PetscErrorCode DMMonitorSet(DM, PetscErrorCode (*)(DM, void *), void *, PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode DMMonitorCancel(DM);
PETSC_EXTERN PetscErrorCode DMMonitorSetFromOptions(DM, const char[], const char[], const char[], PetscErrorCode (*)(DM, void *), PetscErrorCode (*)(DM, PetscViewerAndFormat *), PetscBool *);
PETSC_EXTERN PetscErrorCode DMMonitor(DM);

static inline PetscInt DMPolytopeTypeGetDim(DMPolytopeType ct)
{
  switch (ct) {
    case DM_POLYTOPE_POINT:
      return 0;
    case DM_POLYTOPE_SEGMENT:
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
      return 1;
    case DM_POLYTOPE_TRIANGLE:
    case DM_POLYTOPE_QUADRILATERAL:
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
      return 2;
    case DM_POLYTOPE_TETRAHEDRON:
    case DM_POLYTOPE_HEXAHEDRON:
    case DM_POLYTOPE_TRI_PRISM:
    case DM_POLYTOPE_TRI_PRISM_TENSOR:
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:
    case DM_POLYTOPE_PYRAMID:
      return 3;
    default: return -1;
  }
}

static inline PetscInt DMPolytopeTypeGetConeSize(DMPolytopeType ct)
{
  switch (ct) {
    case DM_POLYTOPE_POINT:              return 0;
    case DM_POLYTOPE_SEGMENT:            return 2;
    case DM_POLYTOPE_POINT_PRISM_TENSOR: return 2;
    case DM_POLYTOPE_TRIANGLE:           return 3;
    case DM_POLYTOPE_QUADRILATERAL:      return 4;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:   return 4;
    case DM_POLYTOPE_TETRAHEDRON:        return 4;
    case DM_POLYTOPE_HEXAHEDRON:         return 6;
    case DM_POLYTOPE_TRI_PRISM:          return 5;
    case DM_POLYTOPE_TRI_PRISM_TENSOR:   return 5;
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:  return 6;
    case DM_POLYTOPE_PYRAMID:            return 5;
    default: return -1;
  }
}

static inline PetscInt DMPolytopeTypeGetNumVertices(DMPolytopeType ct)
{
  switch (ct) {
    case DM_POLYTOPE_POINT:              return 1;
    case DM_POLYTOPE_SEGMENT:            return 2;
    case DM_POLYTOPE_POINT_PRISM_TENSOR: return 2;
    case DM_POLYTOPE_TRIANGLE:           return 3;
    case DM_POLYTOPE_QUADRILATERAL:      return 4;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:   return 4;
    case DM_POLYTOPE_TETRAHEDRON:        return 4;
    case DM_POLYTOPE_HEXAHEDRON:         return 8;
    case DM_POLYTOPE_TRI_PRISM:          return 6;
    case DM_POLYTOPE_TRI_PRISM_TENSOR:   return 6;
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:  return 8;
    case DM_POLYTOPE_PYRAMID:            return 5;
    default: return -1;
  }
}

static inline DMPolytopeType DMPolytopeTypeSimpleShape(PetscInt dim, PetscBool simplex)
{
  return dim == 0 ? DM_POLYTOPE_POINT :
        (dim == 1 ? DM_POLYTOPE_SEGMENT :
        (dim == 2 ? (simplex ? DM_POLYTOPE_TRIANGLE : DM_POLYTOPE_QUADRILATERAL) :
        (dim == 3 ? (simplex ? DM_POLYTOPE_TETRAHEDRON : DM_POLYTOPE_HEXAHEDRON) : DM_POLYTOPE_UNKNOWN)));
}

static inline PetscInt DMPolytopeTypeGetNumArrangments(DMPolytopeType ct)
{
  switch (ct) {
    case DM_POLYTOPE_POINT:              return 1;
    case DM_POLYTOPE_SEGMENT:            return 2;
    case DM_POLYTOPE_POINT_PRISM_TENSOR: return 2;
    case DM_POLYTOPE_TRIANGLE:           return 6;
    case DM_POLYTOPE_QUADRILATERAL:      return 8;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:   return 4;
    case DM_POLYTOPE_TETRAHEDRON:        return 24;
    case DM_POLYTOPE_HEXAHEDRON:         return 48;
    case DM_POLYTOPE_TRI_PRISM:          return 12;
    case DM_POLYTOPE_TRI_PRISM_TENSOR:   return 12;
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:  return 16;
    case DM_POLYTOPE_PYRAMID:            return 8;
    default: return -1;
  }
}

/* An arrangement is a face order combined with an orientation for each face */
static inline const PetscInt *DMPolytopeTypeGetArrangment(DMPolytopeType ct, PetscInt o)
{
  static const PetscInt pntArr[1*2] = {0, 0};
  /* a: swap */
  static const PetscInt segArr[2*2*2] = {
    1, 0,  0, 0, /* -1: a */
    0, 0,  1, 0, /*  0: e */};
  /* a: swap first two
     b: swap last two */
  static const PetscInt triArr[6*3*2] = {
    0, -1,  2, -1,  1, -1, /* -3: b */
    2, -1,  1, -1,  0, -1, /* -2: aba */
    1, -1,  0, -1,  2, -1, /* -1: a */
    0,  0,  1,  0,  2,  0, /*  0: identity */
    1,  0,  2,  0,  0,  0, /*  1: ba */
    2,  0,  0,  0,  1,  0, /*  2: ab */};
  /* a: forward cyclic permutation
     b: swap first and last pairs */
  static const PetscInt quadArr[8*4*2] = {
    1, -1,  0, -1,  3, -1,  2, -1, /* -4: b */
    0, -1,  3, -1,  2, -1,  1, -1, /* -3: b a^3 = a b */
    3, -1,  2, -1,  1, -1,  0, -1, /* -2: b a^2 = a^2 b */
    2, -1,  1, -1,  0, -1,  3, -1, /* -1: b a   = a^3 b */
    0,  0,  1,  0,  2,  0,  3,  0, /*  0: identity */
    1,  0,  2,  0,  3,  0,  0,  0, /*  1: a */
    2,  0,  3,  0,  0,  0,  1,  0, /*  2: a^2 */
    3,  0,  0,  0,  1,  0,  2,  0, /*  3: a^3 */};
  /* r: rotate 180
     b: swap top and bottom segments */
  static const PetscInt tsegArr[4*4*2] = {
    1, -1,  0, -1,  3, -1,  2, -1, /* -2: r b */
    0, -1,  1, -1,  3,  0,  2,  0, /* -1: r */
    0,  0,  1,  0,  2,  0,  3,  0, /*  0: identity */
    1,  0,  0,  0,  2, -1,  3, -1, /*  1: b */};
  /* https://en.wikiversity.org/wiki/Symmetric_group_S4 */
  static const PetscInt tetArr[24*4*2] = {
    3, -2,  2, -3,  0, -1,  1, -1, /* -12: (1324)   p22 */
    3, -1,  1, -3,  2, -1,  0, -1, /* -11: (14)     p21 */
    3, -3,  0, -3,  1, -1,  2, -1, /* -10: (1234)   p18 */
    2, -1,  3, -1,  1, -3,  0, -2, /*  -9: (1423)   p17 */
    2, -3,  0, -1,  3, -2,  1, -3, /*  -8: (1342)   p13 */
    2, -2,  1, -2,  0, -2,  3, -2, /*  -7: (24)     p14 */
    1, -2,  0, -2,  2, -2,  3, -1, /*  -6: (34)     p6  */
    1, -1,  3, -3,  0, -3,  2, -2, /*  -5: (1243)   p10 */
    1, -3,  2, -1,  3, -1,  0, -3, /*  -4: (1432)   p9  */
    0, -3,  1, -1,  3, -3,  2, -3, /*  -3: (12)     p1  */
    0, -2,  2, -2,  1, -2,  3, -3, /*  -2: (23)     p2  */
    0, -1,  3, -2,  2, -3,  1, -2, /*  -1: (13)     p5  */
    0,  0,  1,  0,  2,  0,  3,  0, /*   0: ()       p0  */
    0,  1,  3,  1,  1,  2,  2,  0, /*   1: (123)    p4  */
    0,  2,  2,  1,  3,  0,  1,  2, /*   2: (132)    p3  */
    1,  2,  0,  1,  3,  1,  2,  2, /*   3: (12)(34) p7  */
    1,  0,  2,  0,  0,  0,  3,  1, /*   4: (243)    p8  */
    1,  1,  3,  2,  2,  2,  0,  0, /*   5: (143)    p11 */
    2,  1,  3,  0,  0,  2,  1,  0, /*   6: (13)(24) p16 */
    2,  2,  1,  1,  3,  2,  0,  2, /*   7: (142)    p15 */
    2,  0,  0,  0,  1,  0,  3,  2, /*   8: (234)    p12 */
    3,  2,  2,  2,  1,  1,  0,  1, /*   9: (14)(23) p23 */
    3,  0,  0,  2,  2,  1,  1,  1, /*  10: (134)    p19 */
    3,  1,  1,  2,  0,  1,  2,  1  /*  11: (124)    p20 */};
  /* Each rotation determines a permutation of the four diagonals, and this defines the isomorphism with S_4 */
  static const PetscInt hexArr[48*6*2] = {
    2, -3,  3, -2,  4, -2,  5, -3,  1, -3,  0, -1, /* -24: reflect bottom and use -3 on top */
    4, -2,  5, -2,  0, -1,  1, -4,  3, -2,  2, -3, /* -23: reflect bottom and use -3 on top */
    5, -3,  4, -1,  1, -2,  0, -3,  3, -4,  2, -1, /* -22: reflect bottom and use -3 on top */
    3, -1,  2, -4,  4, -4,  5, -1,  0, -4,  1, -4, /* -21: reflect bottom and use -3 on top */
    3, -3,  2, -2,  5, -1,  4, -4,  1, -1,  0, -3, /* -20: reflect bottom and use -3 on top */
    4, -4,  5, -4,  1, -4,  0, -1,  2, -4,  3, -1, /* -19: reflect bottom and use -3 on top */
    2, -1,  3, -4,  5, -3,  4, -2,  0, -2,  1, -2, /* -18: reflect bottom and use -3 on top */
    5, -1,  4, -3,  0, -3,  1, -2,  2, -2,  3, -3, /* -17: reflect bottom and use -3 on top */
    4, -3,  5, -1,  3, -2,  2, -4,  1, -4,  0, -4, /* -16: reflect bottom and use -3 on top */
    5, -4,  4, -4,  3, -4,  2, -2,  0, -3,  1, -1, /* -15: reflect bottom and use -3 on top */
    3, -4,  2, -1,  1, -1,  0, -4,  4, -4,  5, -4, /* -14: reflect bottom and use -3 on top */
    2, -2,  3, -3,  0, -2,  1, -3,  4, -2,  5, -2, /* -13: reflect bottom and use -3 on top */
    1, -3,  0, -1,  4, -1,  5, -4,  3, -1,  2, -4, /* -12: reflect bottom and use -3 on top */
    1, -1,  0, -3,  5, -4,  4, -1,  2, -1,  3, -4, /* -11: reflect bottom and use -3 on top */
    5, -2,  4, -2,  2, -2,  3, -4,  1, -2,  0, -2, /* -10: reflect bottom and use -3 on top */
    1, -2,  0, -2,  2, -1,  3, -1,  4, -1,  5, -3, /*  -9: reflect bottom and use -3 on top */
    4, -1,  5, -3,  2, -4,  3, -2,  0, -1,  1, -3, /*  -8: reflect bottom and use -3 on top */
    3, -2,  2, -3,  0, -4,  1, -1,  5, -1,  4, -3, /*  -7: reflect bottom and use -3 on top */
    1, -4,  0, -4,  3, -1,  2, -1,  5, -4,  4, -4, /*  -6: reflect bottom and use -3 on top */
    2, -4,  3, -1,  1, -3,  0, -2,  5, -3,  4, -1, /*  -5: reflect bottom and use -3 on top */
    0, -4,  1, -4,  4, -3,  5, -2,  2, -3,  3, -2, /*  -4: reflect bottom and use -3 on top */
    0, -3,  1, -1,  3, -3,  2, -3,  4, -3,  5, -1, /*  -3: reflect bottom and use -3 on top */
    0, -2,  1, -2,  5, -2,  4, -3,  3, -3,  2, -2, /*  -2: reflect bottom and use -3 on top */
    0, -1,  1, -3,  2, -3,  3, -3,  5, -2,  4, -2, /*  -1: reflect bottom and use -3 on top */
    0,  0,  1,  0,  2,  0,  3,  0,  4,  0,  5,  0, /*   0: identity */
    0,  1,  1,  3,  5,  3,  4,  0,  2,  0,  3,  1, /*   1: 90  rotation about z */
    0,  2,  1,  2,  3,  0,  2,  0,  5,  3,  4,  1, /*   2: 180 rotation about z */
    0,  3,  1,  1,  4,  0,  5,  3,  3,  0,  2,  1, /*   3: 270 rotation about z */
    2,  3,  3,  2,  1,  0,  0,  3,  4,  3,  5,  1, /*   4: 90  rotation about x */
    1,  3,  0,  1,  3,  2,  2,  2,  4,  2,  5,  2, /*   5: 180 rotation about x */
    3,  1,  2,  0,  0,  1,  1,  2,  4,  1,  5,  3, /*   6: 270 rotation about x */
    4,  0,  5,  0,  2,  1,  3,  3,  1,  1,  0,  3, /*   7: 90  rotation about y */
    1,  1,  0,  3,  2,  2,  3,  2,  5,  1,  4,  3, /*   8: 180 rotation about y */
    5,  1,  4,  3,  2,  3,  3,  1,  0,  0,  1,  0, /*   9: 270 rotation about y */
    1,  0,  0,  0,  5,  1,  4,  2,  3,  2,  2,  3, /*  10: 180 rotation about x+y */
    1,  2,  0,  2,  4,  2,  5,  1,  2,  2,  3,  3, /*  11: 180 rotation about x-y */
    2,  1,  3,  0,  0,  3,  1,  0,  5,  0,  4,  0, /*  12: 180 rotation about y+z */
    3,  3,  2,  2,  1,  2,  0,  1,  5,  2,  4,  2, /*  13: 180 rotation about y-z */
    5,  3,  4,  1,  3,  1,  2,  3,  1,  3,  0,  1, /*  14: 180 rotation about z+x */
    4,  2,  5,  2,  3,  3,  2,  1,  0,  2,  1,  2, /*  15: 180 rotation about z-x */
    5,  0,  4,  0,  0,  0,  1,  3,  3,  1,  2,  0, /*  16: 120 rotation about x+y+z (v0v6) */
    2,  0,  3,  1,  5,  0,  4,  3,  1,  0,  0,  0, /*  17: 240 rotation about x+y+z (v0v6) */
    4,  3,  5,  1,  1,  1,  0,  2,  3,  3,  2,  2, /*  18: 120 rotation about x+y-z (v4v2) */
    3,  2,  2,  3,  5,  2,  4,  1,  0,  1,  1,  3, /*  19: 240 rotation about x+y-z (v4v2) */
    3,  0,  2,  1,  4,  1,  5,  2,  1,  2,  0,  2, /*  20: 120 rotation about x-y+z (v1v5) */
    5,  2,  4,  2,  1,  3,  0,  0,  2,  3,  3,  2, /*  21: 240 rotation about x-y+z (v1v5) */
    4,  1,  5,  3,  0,  2,  1,  1,  2,  1,  3,  0, /*  22: 120 rotation about x-y-z (v7v3) */
    2,  2,  3,  3,  4,  3,  5,  0,  0,  3,  1,  1, /*  23: 240 rotation about x-y-z (v7v3) */
  };
  static const PetscInt tripArr[12*5*2] = {
    1, -3,  0, -1,  3, -1,  4, -1,  2, -1, /* -6: reflect bottom and top */
    1, -1,  0, -3,  4, -1,  2, -1,  3, -1, /* -5: reflect bottom and top */
    1, -2,  0, -2,  2, -1,  3, -1,  4, -1, /* -4: reflect bottom and top */
    0, -3,  1, -1,  3, -3,  2, -3,  4, -3, /* -3: reflect bottom and top */
    0, -2,  1, -2,  4, -3,  3, -3,  2, -3, /* -2: reflect bottom and top */
    0, -1,  1, -3,  2, -3,  4, -3,  3, -3, /* -1: reflect bottom and top */
    0,  0,  1,  0,  2,  0,  3,  0,  4,  0, /*  0: identity */
    0,  1,  1,  2,  4,  0,  2,  0,  3,  0, /*  1: 120 rotation about z */
    0,  2,  1,  1,  3,  0,  4,  0,  2,  0, /*  2: 240 rotation about z */
    1,  1,  0,  2,  2,  2,  4,  2,  3,  2, /*  3: 180 rotation about y of 0 */
    1,  0,  0,  0,  4,  2,  3,  2,  2,  2, /*  4: 180 rotation about y of 1 */
    1,  2,  0,  1,  3,  2,  2,  2,  4,  2, /*  5: 180 rotation about y of 2 */
  };
  /* a: rotate 120 about z
     b: swap top and bottom segments
     r: reflect */
  static const PetscInt ttriArr[12*5*2] = {
    1, -3,  0, -3,  2, -2,  4, -2,  3, -2, /* -6: r b a^2 */
    1, -2,  0, -2,  4, -2,  3, -2,  2, -2, /* -5: r b a */
    1, -1,  0, -1,  3, -2,  2, -2,  4, -2, /* -4: r b */
    0, -3,  1, -3,  2, -1,  4, -1,  3, -1, /* -3: r a^2 */
    0, -2,  1, -2,  4, -1,  3, -1,  2, -1, /* -2: r a */
    0, -1,  1, -1,  3, -1,  2, -1,  4, -1, /* -1: r */
    0,  0,  1,  0,  2,  0,  3,  0,  4,  0, /*  0: identity */
    0,  1,  1,  1,  3,  0,  4,  0,  2,  0, /*  1: a */
    0,  2,  1,  2,  4,  0,  2,  0,  3,  0, /*  2: a^2 */
    1,  0,  0,  0,  2,  1,  3,  1,  4,  1, /*  3: b */
    1,  1,  0,  1,  3,  1,  4,  1,  2,  1, /*  4: b a */
    1,  2,  0,  2,  4,  1,  2,  1,  3,  1, /*  5: b a^2 */
  };
  /* a: rotate 90 about z
     b: swap top and bottom segments
     r: reflect */
  static const PetscInt tquadArr[16*6*2] = {
    1, -4,  0, -4,  3, -2,  2, -2,  5, -2,  4, -2, /* -8: r b a^3 */
    1, -3,  0, -3,  2, -2,  5, -2,  4, -2,  3, -2, /* -7: r b a^2 */
    1, -2,  0, -2,  5, -2,  4, -2,  3, -2,  2, -2, /* -6: r b a */
    1, -1,  0, -1,  4, -2,  3, -2,  2, -2,  5, -2, /* -5: r b */
    0, -4,  1, -4,  3, -1,  2, -1,  5, -1,  4, -1, /* -4: r a^3 */
    0, -3,  1, -3,  2, -1,  5, -1,  4, -1,  3, -1, /* -3: r a^2 */
    0, -2,  1, -2,  5, -1,  4, -1,  3, -1,  2, -1, /* -2: r a */
    0, -1,  1, -1,  4, -1,  3, -1,  2, -1,  5, -1, /* -1: r */
    0,  0,  1,  0,  2,  0,  3,  0,  4,  0,  5,  0, /*  0: identity */
    0,  1,  1,  1,  3,  0,  4,  0,  5,  0,  2,  0, /*  1: a */
    0,  2,  1,  2,  4,  0,  5,  0,  2,  0,  3,  0, /*  2: a^2 */
    0,  3,  1,  3,  5,  0,  2,  0,  3,  0,  4,  0, /*  3: a^3 */
    1,  0,  0,  0,  2,  1,  3,  1,  4,  1,  5,  1, /*  4: b */
    1,  1,  0,  1,  3,  1,  4,  1,  5,  1,  2,  1, /*  5: b a */
    1,  2,  0,  2,  4,  1,  5,  1,  2,  1,  3,  1, /*  6: b a^2 */
    1,  3,  0,  3,  5,  1,  2,  1,  3,  1,  4,  1, /*  7: b a^3 */
  };
  static const PetscInt pyrArr[8*5*2] = {
    0, -4,  2, -3,  1, -3,  4, -3,  3, -3, /* -4: Reflect bottom face */
    0, -3,  3, -3,  2, -3,  1, -3,  4, -3, /* -3: Reflect bottom face */
    0, -2,  4, -3,  3, -3,  2, -3,  1, -3, /* -2: Reflect bottom face */
    0, -1,  1, -3,  4, -3,  3, -3,  2, -3, /* -1: Reflect bottom face */
    0,  0,  1,  0,  2,  0,  3,  0,  4,  0, /*  0: identity */
    0,  1,  4,  0,  1,  0,  2,  0,  3,  0, /*  1:  90 rotation about z */
    0,  2,  3,  0,  4,  0,  1,  0,  2,  0, /*  2: 180 rotation about z */
    0,  3,  2,  0,  3,  0,  4,  0,  1,  0, /*  3: 270 rotation about z */
  };
  switch (ct) {
    case DM_POLYTOPE_POINT:              return pntArr;
    case DM_POLYTOPE_SEGMENT:            return &segArr[(o+1)*2*2];
    case DM_POLYTOPE_POINT_PRISM_TENSOR: return &segArr[(o+1)*2*2];
    case DM_POLYTOPE_TRIANGLE:           return &triArr[(o+3)*3*2];
    case DM_POLYTOPE_QUADRILATERAL:      return &quadArr[(o+4)*4*2];
    case DM_POLYTOPE_SEG_PRISM_TENSOR:   return &tsegArr[(o+2)*4*2];
    case DM_POLYTOPE_TETRAHEDRON:        return &tetArr[(o+12)*4*2];
    case DM_POLYTOPE_HEXAHEDRON:         return &hexArr[(o+24)*6*2];
    case DM_POLYTOPE_TRI_PRISM:          return &tripArr[(o+6)*5*2];
    case DM_POLYTOPE_TRI_PRISM_TENSOR:   return &ttriArr[(o+6)*5*2];
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:  return &tquadArr[(o+8)*6*2];
    case DM_POLYTOPE_PYRAMID:            return &pyrArr[(o+4)*5*2];
    default: return NULL;
  }
}

/* A vertex arrangment is a vertex order */
static inline const PetscInt *DMPolytopeTypeGetVertexArrangment(DMPolytopeType ct, PetscInt o)
{
  static const PetscInt pntVerts[1]    = {0};
  static const PetscInt segVerts[2*2]  = {
    1, 0,
    0, 1};
  static const PetscInt triVerts[6*3]  = {
    1, 0, 2,
    0, 2, 1,
    2, 1, 0,
    0, 1, 2,
    1, 2, 0,
    2, 0, 1};
  static const PetscInt quadVerts[8*4]  = {
    2, 1, 0, 3,
    1, 0, 3, 2,
    0, 3, 2, 1,
    3, 2, 1, 0,
    0, 1, 2, 3,
    1, 2, 3, 0,
    2, 3, 0, 1,
    3, 0, 1, 2};
  static const PetscInt tsegVerts[4*4]  = {
    3, 2, 1, 0,
    1, 0, 3, 2,
    0, 1, 2, 3,
    2, 3, 0, 1};
  static const PetscInt tetVerts[24*4] = {
    2, 3, 1, 0, /* -12: (1324)   p22 */
    3, 1, 2, 0, /* -11: (14)     p21 */
    1, 2, 3, 0, /* -10: (1234)   p18 */
    3, 2, 0, 1, /*  -9: (1423)   p17 */
    2, 0, 3, 1, /*  -8: (1342)   p13 */
    0, 3, 2, 1, /*  -7: (24)     p14 */
    0, 1, 3, 2, /*  -6: (34)     p6  */
    1, 3, 0, 2, /*  -5: (1243)   p10 */
    3, 0, 1, 2, /*  -4: (1432    p9  */
    1, 0, 2, 3, /*  -3: (12)     p1  */
    0, 2, 1, 3, /*  -2: (23)     p2  */
    2, 1, 0, 3, /*  -1: (13)     p5  */
    0, 1, 2, 3, /*   0: ()       p0  */
    1, 2, 0, 3, /*   1: (123)    p4  */
    2, 0, 1, 3, /*   2: (132)    p3  */
    1, 0, 3, 2, /*   3: (12)(34) p7  */
    0, 3, 1, 2, /*   4: (243)    p8  */
    3, 1, 0, 2, /*   5: (143)    p11 */
    2, 3, 0, 1, /*   6: (13)(24) p16 */
    3, 0, 2, 1, /*   7: (142)    p15 */
    0, 2, 3, 1, /*   8: (234)    p12 */
    3, 2, 1, 0, /*   9: (14)(23) p23 */
    2, 1, 3, 0, /*  10: (134)    p19 */
    1, 3, 2, 0  /*  11: (124)    p20 */};
  static const PetscInt hexVerts[48*8] = {
    3,  0,  4,  5,  2,  6,  7,  1, /* -24: reflected 23 */
    3,  5,  6,  2,  0,  1,  7,  4, /* -23: reflected 22 */
    4,  0,  1,  7,  5,  6,  2,  3, /* -22: reflected 21 */
    6,  7,  1,  2,  5,  3,  0,  4, /* -21: reflected 20 */
    1,  2,  6,  7,  0,  4,  5,  3, /* -20: reflected 19 */
    6,  2,  3,  5,  7,  4,  0,  1, /* -19: reflected 18 */
    4,  5,  3,  0,  7,  1,  2,  6, /* -18: reflected 17 */
    1,  7,  4,  0,  2,  3,  5,  6, /* -17: reflected 16 */
    2,  3,  5,  6,  1,  7,  4,  0, /* -16: reflected 15 */
    7,  4,  0,  1,  6,  2,  3,  5, /* -15: reflected 14 */
    7,  1,  2,  6,  4,  5,  3,  0, /* -14: reflected 13 */
    0,  4,  5,  3,  1,  2,  6,  7, /* -13: reflected 12 */
    5,  4,  7,  6,  3,  2,  1,  0, /* -12: reflected 11 */
    7,  6,  5,  4,  1,  0,  3,  2, /* -11: reflected 10 */
    0,  1,  7,  4,  3,  5,  6,  2, /* -10: reflected  9 */
    4,  7,  6,  5,  0,  3,  2,  1, /*  -9: reflected  8 */
    5,  6,  2,  3,  4,  0,  1,  7, /*  -8: reflected  7 */
    2,  6,  7,  1,  3,  0,  4,  5, /*  -7: reflected  6 */
    6,  5,  4,  7,  2,  1,  0,  3, /*  -6: reflected  5 */
    5,  3,  0,  4,  6,  7,  1,  2, /*  -5: reflected  4 */
    2,  1,  0,  3,  6,  5,  4,  7, /*  -4: reflected  3 */
    1,  0,  3,  2,  7,  6,  5,  4, /*  -3: reflected  2 */
    0,  3,  2,  1,  4,  7,  6,  5, /*  -2: reflected  1 */
    3,  2,  1,  0,  5,  4,  7,  6, /*  -1: reflected  0 */
    0,  1,  2,  3,  4,  5,  6,  7, /*   0: identity */
    1,  2,  3,  0,  7,  4,  5,  6, /*   1: 90  rotation about z */
    2,  3,  0,  1,  6,  7,  4,  5, /*   2: 180 rotation about z */
    3,  0,  1,  2,  5,  6,  7,  4, /*   3: 270 rotation about z */
    4,  0,  3,  5,  7,  6,  2,  1, /*   4: 90  rotation about x */
    7,  4,  5,  6,  1,  2,  3,  0, /*   5: 180 rotation about x */
    1,  7,  6,  2,  0,  3,  5,  4, /*   6: 270 rotation about x */
    3,  2,  6,  5,  0,  4,  7,  1, /*   7: 90  rotation about y */
    5,  6,  7,  4,  3,  0,  1,  2, /*   8: 180 rotation about y */
    4,  7,  1,  0,  5,  3,  2,  6, /*   9: 270 rotation about y */
    4,  5,  6,  7,  0,  1,  2,  3, /*  10: 180 rotation about x+y */
    6,  7,  4,  5,  2,  3,  0,  1, /*  11: 180 rotation about x-y */
    3,  5,  4,  0,  2,  1,  7,  6, /*  12: 180 rotation about y+z */
    6,  2,  1,  7,  5,  4,  0,  3, /*  13: 180 rotation about y-z */
    1,  0,  4,  7,  2,  6,  5,  3, /*  14: 180 rotation about z+x */
    6,  5,  3,  2,  7,  1,  0,  4, /*  15: 180 rotation about z-x */
    0,  4,  7,  1,  3,  2,  6,  5, /*  16: 120 rotation about x+y+z (v0v6) */
    0,  3,  5,  4,  1,  7,  6,  2, /*  17: 240 rotation about x+y+z (v0v6) */
    5,  3,  2,  6,  4,  7,  1,  0, /*  18: 120 rotation about x+y-z (v4v2) */
    7,  6,  2,  1,  4,  0,  3,  5, /*  19: 240 rotation about x+y-z (v4v2) */
    2,  1,  7,  6,  3,  5,  4,  0, /*  20: 120 rotation about x-y+z (v1v5) */
    7,  1,  0,  4,  6,  5,  3,  2, /*  21: 240 rotation about x-y+z (v1v5) */
    2,  6,  5,  3,  1,  0,  4,  7, /*  22: 120 rotation about x-y-z (v7v3) */
    5,  4,  0,  3,  6,  2,  1,  7, /*  23: 240 rotation about x-y-z (v7v3) */
  };
  static const PetscInt tripVerts[12*6] = {
    4,  3,  5,  2,  1,  0, /* -6: reflect bottom and top */
    5,  4,  3,  1,  0,  2, /* -5: reflect bottom and top */
    3,  5,  4,  0,  2,  1, /* -4: reflect bottom and top */
    1,  0,  2,  5,  4,  3, /* -3: reflect bottom and top */
    0,  2,  1,  3,  5,  4, /* -2: reflect bottom and top */
    2,  1,  0,  4,  3,  5, /* -1: reflect bottom and top */
    0,  1,  2,  3,  4,  5, /*  0: identity */
    1,  2,  0,  5,  3,  4, /*  1: 120 rotation about z */
    2,  0,  1,  4,  5,  3, /*  2: 240 rotation about z */
    4,  5,  3,  2,  0,  1, /*  3: 180 rotation about y of 0 */
    3,  4,  5,  0,  1,  2, /*  4: 180 rotation about y of 1 */
    5,  3,  4,  1,  2,  0, /*  5: 180 rotation about y of 2 */
  };
  static const PetscInt ttriVerts[12*6] = {
    4,  3,  5,  1,  0,  2, /* -6: r b a^2 */
    3,  5,  4,  0,  2,  1, /* -5: r b a */
    5,  4,  3,  2,  1,  0, /* -4: r b */
    1,  0,  2,  4,  3,  5, /* -3: r a^2 */
    0,  2,  1,  3,  5,  4, /* -2: r a */
    2,  1,  0,  5,  4,  3, /* -1: r */
    0,  1,  2,  3,  4,  5, /*  0: identity */
    1,  2,  0,  4,  5,  3, /*  1: a */
    2,  0,  1,  5,  3,  4, /*  2: a^2 */
    3,  4,  5,  0,  1,  2, /*  3: b */
    4,  5,  3,  1,  2,  0, /*  4: b a */
    5,  3,  4,  2,  0,  1, /*  5: b a^2 */
  };
  /* a: rotate 90 about z
     b: swap top and bottom segments
     r: reflect */
  static const PetscInt tquadVerts[16*8] = {
    6,  5,  4,  7,  2,  1,  0,  3, /* -8: r b a^3 */
    5,  4,  7,  6,  1,  0,  3,  2, /* -7: r b a^2 */
    4,  7,  6,  5,  0,  3,  2,  1, /* -6: r b a */
    7,  6,  5,  4,  3,  2,  1,  0, /* -5: r b */
    2,  1,  0,  3,  6,  5,  4,  7, /* -4: r a^3 */
    1,  0,  3,  2,  5,  4,  7,  6, /* -3: r a^2 */
    0,  3,  2,  1,  4,  7,  6,  5, /* -2: r a */
    3,  2,  1,  0,  7,  6,  5,  4, /* -1: r */
    0,  1,  2,  3,  4,  5,  6,  7, /*  0: identity */
    1,  2,  3,  0,  5,  6,  7,  4, /*  1: a */
    2,  3,  0,  1,  6,  7,  4,  5, /*  2: a^2 */
    3,  0,  1,  2,  7,  4,  5,  6, /*  3: a^3 */
    4,  5,  6,  7,  0,  1,  2,  3, /*  4: b */
    5,  6,  7,  4,  1,  2,  3,  0, /*  5: b a */
    6,  7,  4,  5,  2,  3,  0,  1, /*  6: b a^2 */
    7,  4,  5,  6,  3,  0,  1,  2, /*  7: b a^3 */
  };
  static const PetscInt pyrVerts[8*5] = {
    2,  1,  0,  3,  4, /* -4: Reflect bottom face */
    1,  0,  3,  2,  4, /* -3: Reflect bottom face */
    0,  3,  2,  1,  4, /* -2: Reflect bottom face */
    3,  2,  1,  0,  4, /* -1: Reflect bottom face */
    0,  1,  2,  3,  4, /*  0: identity */
    1,  2,  3,  0,  4, /*  1:  90 rotation about z */
    2,  3,  0,  1,  4, /*  2: 180 rotation about z */
    3,  0,  1,  2,  4, /*  3: 270 rotation about z */
  };
  switch (ct) {
    case DM_POLYTOPE_POINT:              return pntVerts;
    case DM_POLYTOPE_SEGMENT:            return &segVerts[(o+1)*2];
    case DM_POLYTOPE_POINT_PRISM_TENSOR: return &segVerts[(o+1)*2];
    case DM_POLYTOPE_TRIANGLE:           return &triVerts[(o+3)*3];
    case DM_POLYTOPE_QUADRILATERAL:      return &quadVerts[(o+4)*4];
    case DM_POLYTOPE_SEG_PRISM_TENSOR:   return &tsegVerts[(o+2)*4];
    case DM_POLYTOPE_TETRAHEDRON:        return &tetVerts[(o+12)*4];
    case DM_POLYTOPE_HEXAHEDRON:         return &hexVerts[(o+24)*8];
    case DM_POLYTOPE_TRI_PRISM:          return &tripVerts[(o+6)*6];
    case DM_POLYTOPE_TRI_PRISM_TENSOR:   return &ttriVerts[(o+6)*6];
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:  return &tquadVerts[(o+8)*8];
    case DM_POLYTOPE_PYRAMID:            return &pyrVerts[(o+4)*5];
    default: return NULL;
  }
}

/* This is orientation o1 acting on orientation o2 */
static inline PetscInt DMPolytopeTypeComposeOrientation(DMPolytopeType ct, PetscInt o1, PetscInt o2)
{
  static const PetscInt segMult[2*2] = {
     0, -1,
    -1,  0};
  static const PetscInt triMult[6*6] = {
     0,  2,  1, -3, -1, -2,
     1,  0,  2, -2, -3, -1,
     2,  1,  0, -1, -2, -3,
    -3, -2, -1,  0,  1,  2,
    -2, -1, -3,  1,  2,  0,
    -1, -3, -2,  2,  0,  1};
  static const PetscInt quadMult[8*8] = {
     0,  3,  2,  1, -4, -1, -2, -3,
     1,  0,  3,  2, -3, -4, -1, -2,
     2,  1,  0,  3, -2, -3, -4, -1,
     3,  2,  1,  0, -1, -2, -3, -4,
    -4, -3, -2, -1,  0,  1,  2,  3,
    -3, -2, -1, -4,  1,  2,  3,  0,
    -2, -1, -4, -3,  2,  3,  0,  1,
    -1, -4, -3, -2,  3,  0,  1,  2};
  static const PetscInt tsegMult[4*4] = {
     0,  1, -2, -1,
     1,  0, -1, -2,
    -2, -1,  0,  1,
    -1, -2,  1,  0};
  static const PetscInt tetMult[24*24] = {
    3, 2, 7, 0, 5, 10, 9, 8, 1, 6, 11, 4, -12, -7, -5, -9, -10, -2, -6, -1, -11, -3, -4, -8,
    4, 0, 8, 1, 3, 11, 10, 6, 2, 7, 9, 5, -11, -9, -4, -8, -12, -1, -5, -3, -10, -2, -6, -7,
    5, 1, 6, 2, 4, 9, 11, 7, 0, 8, 10, 3, -10, -8, -6, -7, -11, -3, -4, -2, -12, -1, -5, -9,
    0, 8, 4, 3, 11, 1, 6, 2, 10, 9, 5, 7, -9, -4, -11, -12, -1, -8, -3, -10, -5, -6, -7, -2,
    1, 6, 5, 4, 9, 2, 7, 0, 11, 10, 3, 8, -8, -6, -10, -11, -3, -7, -2, -12, -4, -5, -9, -1,
    2, 7, 3, 5, 10, 0, 8, 1, 9, 11, 4, 6, -7, -5, -12, -10, -2, -9, -1, -11, -6, -4, -8, -3,
    6, 5, 1, 9, 2, 4, 0, 11, 7, 3, 8, 10, -6, -10, -8, -3, -7, -11, -12, -4, -2, -9, -1, -5,
    7, 3, 2, 10, 0, 5, 1, 9, 8, 4, 6, 11, -5, -12, -7, -2, -9, -10, -11, -6, -1, -8, -3, -4,
    8, 4, 0, 11, 1, 3, 2, 10, 6, 5, 7, 9, -4, -11, -9, -1, -8, -12, -10, -5, -3, -7, -2, -6,
    9, 11, 10, 6, 8, 7, 3, 5, 4, 0, 2, 1, -3, -1, -2, -6, -4, -5, -9, -7, -8, -12, -10, -11,
    10, 9, 11, 7, 6, 8, 4, 3, 5, 1, 0, 2, -2, -3, -1, -5, -6, -4, -8, -9, -7, -11, -12, -10,
    11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12,
    -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    -11, -10, -12, -8, -7, -9, -5, -4, -6, -2, -1, -3, 1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9,
    -10, -12, -11, -7, -9, -8, -4, -6, -5, -1, -3, -2, 2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10,
    -9, -5, -1, -12, -2, -4, -3, -11, -7, -6, -8, -10, 3, 10, 8, 0, 7, 11, 9, 4, 2, 6, 1, 5,
    -8, -4, -3, -11, -1, -6, -2, -10, -9, -5, -7, -12, 4, 11, 6, 1, 8, 9, 10, 5, 0, 7, 2, 3,
    -7, -6, -2, -10, -3, -5, -1, -12, -8, -4, -9, -11, 5, 9, 7, 2, 6, 10, 11, 3, 1, 8, 0, 4,
    -3, -8, -4, -6, -11, -1, -9, -2, -10, -12, -5, -7, 6, 4, 11, 9, 1, 8, 0, 10, 5, 3, 7, 2,
    -2, -7, -6, -5, -10, -3, -8, -1, -12, -11, -4, -9, 7, 5, 9, 10, 2, 6, 1, 11, 3, 4, 8, 0,
    -1, -9, -5, -4, -12, -2, -7, -3, -11, -10, -6, -8, 8, 3, 10, 11, 0, 7, 2, 9, 4, 5, 6, 1,
    -6, -2, -7, -3, -5, -10, -12, -8, -1, -9, -11, -4, 9, 7, 5, 6, 10, 2, 3, 1, 11, 0, 4, 8,
    -5, -1, -9, -2, -4, -12, -11, -7, -3, -8, -10, -6, 10, 8, 3, 7, 11, 0, 4, 2, 9, 1, 5, 6,
    -4, -3, -8, -1, -6, -11, -10, -9, -2, -7, -12, -5, 11, 6, 4, 8, 9, 1, 5, 0, 10, 2, 3, 7,
    };
  static const PetscInt hexMult[48*48] = {
    18, 2, 5, 22, 21, 8, 16, 0, 13, 6, 11, 3, 15, 9, 4, 23, 12, 1, 19, 10, 7, 20, 14, 17, -24, -10, -20, -16, -12, -21, -4, -5, -18, -13, -15, -8, -2, -11, -14, -7, -3, -22, -6, -17, -19, -9, -1, -23,
    8, 20, 19, 2, 5, 23, 0, 17, 11, 1, 15, 7, 13, 4, 10, 18, 3, 14, 21, 9, 12, 22, 6, 16, -23, -13, -17, -7, -8, -19, -16, -12, -22, -2, -14, -5, -10, -15, -11, -4, -20, -9, -21, -3, -6, -18, -24, -1,
    2, 17, 23, 8, 0, 19, 5, 20, 1, 11, 9, 14, 12, 6, 3, 16, 10, 7, 22, 15, 13, 21, 4, 18, -22, -14, -19, -5, -15, -17, -10, -2, -23, -12, -13, -7, -16, -8, -4, -11, -24, -3, -18, -9, -1, -21, -20, -6,
    21, 5, 2, 16, 18, 0, 22, 8, 4, 12, 3, 11, 14, 7, 13, 20, 6, 10, 17, 1, 9, 23, 15, 19, -21, -8, -18, -15, -4, -24, -12, -14, -20, -7, -16, -10, -11, -2, -5, -13, -6, -19, -3, -23, -22, -1, -9, -17,
    16, 8, 0, 21, 22, 2, 18, 5, 12, 4, 1, 10, 9, 15, 6, 19, 13, 11, 23, 3, 14, 17, 7, 20, -20, -16, -24, -10, -2, -18, -11, -7, -21, -14, -8, -15, -12, -4, -13, -5, -9, -23, -1, -19, -17, -3, -6, -22,
    5, 19, 20, 0, 8, 17, 2, 23, 10, 3, 7, 15, 6, 12, 11, 22, 1, 9, 16, 14, 4, 18, 13, 21, -19, -5, -22, -14, -16, -23, -8, -11, -17, -4, -7, -13, -15, -10, -12, -2, -21, -6, -20, -1, -9, -24, -18, -3,
    22, 0, 8, 18, 16, 5, 21, 2, 6, 13, 10, 1, 7, 14, 12, 17, 4, 3, 20, 11, 15, 19, 9, 23, -18, -15, -21, -8, -11, -20, -2, -13, -24, -5, -10, -16, -4, -12, -7, -14, -1, -17, -9, -22, -23, -6, -3, -19,
    0, 23, 17, 5, 2, 20, 8, 19, 3, 10, 14, 9, 4, 13, 1, 21, 11, 15, 18, 7, 6, 16, 12, 22, -17, -7, -23, -13, -10, -22, -15, -4, -19, -11, -5, -14, -8, -16, -2, -12, -18, -1, -24, -6, -3, -20, -21, -9,
    10, 13, 6, 1, 11, 12, 3, 4, 8, 0, 22, 18, 19, 23, 5, 15, 2, 21, 9, 16, 17, 7, 20, 14, -16, -24, -10, -20, -23, -8, -19, -6, -15, -3, -21, -18, -22, -17, -9, -1, -14, -12, -7, -4, -11, -13, -5, -2,
    1, 4, 12, 10, 3, 6, 11, 13, 0, 8, 16, 21, 17, 20, 2, 14, 5, 18, 7, 22, 19, 9, 23, 15, -15, -21, -8, -18, -17, -10, -22, -3, -16, -6, -24, -20, -19, -23, -1, -9, -5, -4, -13, -12, -2, -7, -14, -11,
    14, 10, 3, 9, 7, 1, 15, 11, 17, 23, 0, 5, 16, 22, 20, 6, 19, 8, 12, 2, 21, 4, 18, 13, -14, -19, -5, -22, -3, -13, -9, -20, -7, -21, -23, -17, -6, -1, -24, -18, -12, -16, -2, -8, -10, -4, -11, -15,
    7, 3, 10, 15, 14, 11, 9, 1, 20, 19, 5, 0, 18, 21, 17, 4, 23, 2, 13, 8, 22, 6, 16, 12, -13, -17, -7, -23, -9, -14, -3, -24, -5, -18, -22, -19, -1, -6, -20, -21, -2, -10, -12, -15, -16, -11, -4, -8,
    13, 14, 15, 12, 4, 9, 6, 7, 21, 22, 23, 20, 2, 0, 18, 3, 16, 17, 1, 19, 8, 11, 5, 10, -12, -9, -11, -6, -21, -4, -24, -22, -2, -23, -3, -1, -20, -18, -19, -17, -16, -14, -15, -13, -5, -8, -10, -7,
    6, 9, 7, 4, 12, 14, 13, 15, 16, 18, 17, 19, 0, 2, 22, 1, 21, 23, 3, 20, 5, 10, 8, 11, -11, -6, -12, -9, -20, -2, -18, -17, -4, -19, -1, -3, -21, -24, -23, -22, -8, -7, -10, -5, -13, -16, -15, -14,
    3, 12, 4, 11, 1, 13, 10, 6, 2, 5, 21, 16, 23, 19, 0, 9, 8, 22, 15, 18, 20, 14, 17, 7, -10, -20, -16, -24, -22, -15, -17, -1, -8, -9, -18, -21, -23, -19, -3, -6, -13, -2, -5, -11, -4, -14, -7, -12,
    20, 16, 18, 23, 17, 21, 19, 22, 14, 15, 4, 6, 3, 1, 7, 0, 9, 12, 2, 13, 11, 5, 10, 8, -9, -11, -6, -12, -14, -3, -13, -10, -1, -8, -2, -4, -7, -5, -16, -15, -23, -20, -22, -18, -24, -19, -17, -21,
    11, 6, 13, 3, 10, 4, 1, 12, 5, 2, 18, 22, 20, 17, 8, 7, 0, 16, 14, 21, 23, 15, 19, 9, -8, -18, -15, -21, -19, -16, -23, -9, -10, -1, -20, -24, -17, -22, -6, -3, -7, -11, -14, -2, -12, -5, -13, -4,
    9, 11, 1, 14, 15, 3, 7, 10, 23, 17, 2, 8, 21, 18, 19, 13, 20, 5, 4, 0, 16, 12, 22, 6, -7, -23, -13, -17, -1, -5, -6, -21, -14, -20, -19, -22, -9, -3, -18, -24, -11, -8, -4, -16, -15, -2, -12, -10,
    19, 21, 22, 17, 23, 16, 20, 18, 9, 7, 12, 13, 1, 3, 15, 2, 14, 4, 0, 6, 10, 8, 11, 5, -6, -12, -9, -11, -7, -1, -5, -15, -3, -16, -4, -2, -14, -13, -8, -10, -19, -21, -17, -24, -18, -23, -22, -20,
    15, 1, 11, 7, 9, 10, 14, 3, 19, 20, 8, 2, 22, 16, 23, 12, 17, 0, 6, 5, 18, 13, 21, 4, -5, -22, -14, -19, -6, -7, -1, -18, -13, -24, -17, -23, -3, -9, -21, -20, -4, -15, -11, -10, -8, -12, -2, -16,
    4, 15, 14, 6, 13, 7, 12, 9, 18, 16, 20, 23, 5, 8, 21, 11, 22, 19, 10, 17, 0, 3, 2, 1, -4, -1, -2, -3, -24, -12, -21, -19, -11, -17, -6, -9, -18, -20, -22, -23, -15, -5, -16, -7, -14, -10, -8, -13,
    17, 18, 16, 19, 20, 22, 23, 21, 7, 9, 6, 4, 10, 11, 14, 5, 15, 13, 8, 12, 1, 0, 3, 2, -3, -4, -1, -2, -13, -9, -14, -16, -6, -15, -12, -11, -5, -7, -10, -8, -22, -24, -23, -21, -20, -17, -19, -18,
    12, 7, 9, 13, 6, 15, 4, 14, 22, 21, 19, 17, 8, 5, 16, 10, 18, 20, 11, 23, 2, 1, 0, 3, -2, -3, -4, -1, -18, -11, -20, -23, -12, -22, -9, -6, -24, -21, -17, -19, -10, -13, -8, -14, -7, -15, -16, -5,
    23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24,
    -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    -13, -8, -10, -14, -7, -16, -5, -15, -23, -22, -20, -18, -9, -6, -17, -11, -19, -21, -12, -24, -3, -2, -1, -4, 1, 2, 3, 0, 17, 10, 19, 22, 11, 21, 8, 5, 23, 20, 16, 18, 9, 12, 7, 13, 6, 14, 15, 4,
    -18, -19, -17, -20, -21, -23, -24, -22, -8, -10, -7, -5, -11, -12, -15, -6, -16, -14, -9, -13, -2, -1, -4, -3, 2, 3, 0, 1, 12, 8, 13, 15, 5, 14, 11, 10, 4, 6, 9, 7, 21, 23, 22, 20, 19, 16, 18, 17,
    -5, -16, -15, -7, -14, -8, -13, -10, -19, -17, -21, -24, -6, -9, -22, -12, -23, -20, -11, -18, -1, -4, -3, -2, 3, 0, 1, 2, 23, 11, 20, 18, 10, 16, 5, 8, 17, 19, 21, 22, 14, 4, 15, 6, 13, 9, 7, 12,
    -16, -2, -12, -8, -10, -11, -15, -4, -20, -21, -9, -3, -23, -17, -24, -13, -18, -1, -7, -6, -19, -14, -22, -5, 4, 21, 13, 18, 5, 6, 0, 17, 12, 23, 16, 22, 2, 8, 20, 19, 3, 14, 10, 9, 7, 11, 1, 15,
    -20, -22, -23, -18, -24, -17, -21, -19, -10, -8, -13, -14, -2, -4, -16, -3, -15, -5, -1, -7, -11, -9, -12, -6, 5, 11, 8, 10, 6, 0, 4, 14, 2, 15, 3, 1, 13, 12, 7, 9, 18, 20, 16, 23, 17, 22, 21, 19,
    -10, -12, -2, -15, -16, -4, -8, -11, -24, -18, -3, -9, -22, -19, -20, -14, -21, -6, -5, -1, -17, -13, -23, -7, 6, 22, 12, 16, 0, 4, 5, 20, 13, 19, 18, 21, 8, 2, 17, 23, 10, 7, 3, 15, 14, 1, 11, 9,
    -12, -7, -14, -4, -11, -5, -2, -13, -6, -3, -19, -23, -21, -18, -9, -8, -1, -17, -15, -22, -24, -16, -20, -10, 7, 17, 14, 20, 18, 15, 22, 8, 9, 0, 19, 23, 16, 21, 5, 2, 6, 10, 13, 1, 11, 4, 12, 3,
    -21, -17, -19, -24, -18, -22, -20, -23, -15, -16, -5, -7, -4, -2, -8, -1, -10, -13, -3, -14, -12, -6, -11, -9, 8, 10, 5, 11, 13, 2, 12, 9, 0, 7, 1, 3, 6, 4, 15, 14, 22, 19, 21, 17, 23, 18, 16, 20,
    -4, -13, -5, -12, -2, -14, -11, -7, -3, -6, -22, -17, -24, -20, -1, -10, -9, -23, -16, -19, -21, -15, -18, -8, 9, 19, 15, 23, 21, 14, 16, 0, 7, 8, 17, 20, 22, 18, 2, 5, 12, 1, 4, 10, 3, 13, 6, 11,
    -7, -10, -8, -5, -13, -15, -14, -16, -17, -19, -18, -20, -1, -3, -23, -2, -22, -24, -4, -21, -6, -11, -9, -12, 10, 5, 11, 8, 19, 1, 17, 16, 3, 18, 0, 2, 20, 23, 22, 21, 7, 6, 9, 4, 12, 15, 14, 13,
    -14, -15, -16, -13, -5, -10, -7, -8, -22, -23, -24, -21, -3, -1, -19, -4, -17, -18, -2, -20, -9, -12, -6, -11, 11, 8, 10, 5, 20, 3, 23, 21, 1, 22, 2, 0, 19, 17, 18, 16, 15, 13, 14, 12, 4, 7, 9, 6,
    -8, -4, -11, -16, -15, -12, -10, -2, -21, -20, -6, -1, -19, -22, -18, -5, -24, -3, -14, -9, -23, -7, -17, -13, 12, 16, 6, 22, 8, 13, 2, 23, 4, 17, 21, 18, 0, 5, 19, 20, 1, 9, 11, 14, 15, 10, 3, 7,
    -15, -11, -4, -10, -8, -2, -16, -12, -18, -24, -1, -6, -17, -23, -21, -7, -20, -9, -13, -3, -22, -5, -19, -14, 13, 18, 4, 21, 2, 12, 8, 19, 6, 20, 22, 16, 5, 0, 23, 17, 11, 15, 1, 7, 9, 3, 10, 14,
    -2, -5, -13, -11, -4, -7, -12, -14, -1, -9, -17, -22, -18, -21, -3, -15, -6, -19, -8, -23, -20, -10, -24, -16, 14, 20, 7, 17, 16, 9, 21, 2, 15, 5, 23, 19, 18, 22, 0, 8, 4, 3, 12, 11, 1, 6, 13, 10,
    -11, -14, -7, -2, -12, -13, -4, -5, -9, -1, -23, -19, -20, -24, -6, -16, -3, -22, -10, -17, -18, -8, -21, -15, 15, 23, 9, 19, 22, 7, 18, 5, 14, 2, 20, 17, 21, 16, 8, 0, 13, 11, 6, 3, 10, 12, 4, 1,
    -1, -24, -18, -6, -3, -21, -9, -20, -4, -11, -15, -10, -5, -14, -2, -22, -12, -16, -19, -8, -7, -17, -13, -23, 16, 6, 22, 12, 9, 21, 14, 3, 18, 10, 4, 13, 7, 15, 1, 11, 17, 0, 23, 5, 2, 19, 20, 8,
    -23, -1, -9, -19, -17, -6, -22, -3, -7, -14, -11, -2, -8, -15, -13, -18, -5, -4, -21, -12, -16, -20, -10, -24, 17, 14, 20, 7, 10, 19, 1, 12, 23, 4, 9, 15, 3, 11, 6, 13, 0, 16, 8, 21, 22, 5, 2, 18,
    -6, -20, -21, -1, -9, -18, -3, -24, -11, -4, -8, -16, -7, -13, -12, -23, -2, -10, -17, -15, -5, -19, -14, -22, 18, 4, 21, 13, 15, 22, 7, 10, 16, 3, 6, 12, 14, 9, 11, 1, 20, 5, 19, 0, 8, 23, 17, 2,
    -17, -9, -1, -22, -23, -3, -19, -6, -13, -5, -2, -11, -10, -16, -7, -20, -14, -12, -24, -4, -15, -18, -8, -21, 19, 15, 23, 9, 1, 17, 10, 6, 20, 13, 7, 14, 11, 3, 12, 4, 8, 22, 0, 18, 16, 2, 5, 21,
    -22, -6, -3, -17, -19, -1, -23, -9, -5, -13, -4, -12, -15, -8, -14, -21, -7, -11, -18, -2, -10, -24, -16, -20, 20, 7, 17, 14, 3, 23, 11, 13, 19, 6, 15, 9, 10, 1, 4, 12, 5, 18, 2, 22, 21, 0, 8, 16,
    -3, -18, -24, -9, -1, -20, -6, -21, -2, -12, -10, -15, -13, -7, -4, -17, -11, -8, -23, -16, -14, -22, -5, -19, 21, 13, 18, 4, 14, 16, 9, 1, 22, 11, 12, 6, 15, 7, 3, 10, 23, 2, 17, 8, 0, 20, 19, 5,
    -9, -21, -20, -3, -6, -24, -1, -18, -12, -2, -16, -8, -14, -5, -11, -19, -4, -15, -22, -10, -13, -23, -7, -17, 22, 12, 16, 6, 7, 18, 15, 11, 21, 1, 13, 4, 9, 14, 10, 3, 19, 8, 20, 2, 5, 17, 23, 0,
    -19, -3, -6, -23, -22, -9, -17, -1, -14, -7, -12, -4, -16, -10, -5, -24, -13, -2, -20, -11, -8, -21, -15, -18, 23, 9, 19, 15, 11, 20, 3, 4, 17, 12, 14, 7, 1, 10, 13, 6, 2, 21, 5, 16, 18, 8, 0, 22,
    };
  static const PetscInt tripMult[12*12] = {
    1, 0, 2, 3, 5, 4, -6, -4, -5, -2, -3, -1,
    0, 2, 1, 4, 3, 5, -5, -6, -4, -3, -1, -2,
    2, 1, 0, 5, 4, 3, -4, -5, -6, -1, -2, -3,
    4, 3, 5, 0, 2, 1, -3, -1, -2, -5, -6, -4,
    3, 5, 4, 1, 0, 2, -2, -3, -1, -6, -4, -5,
    5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6,
    -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
    -4, -6, -5, -2, -1, -3, 1, 2, 0, 5, 3, 4,
    -5, -4, -6, -1, -3, -2, 2, 0, 1, 4, 5, 3,
    -3, -2, -1, -6, -5, -4, 3, 4, 5, 0, 1, 2,
    -1, -3, -2, -5, -4, -6, 4, 5, 3, 2, 0, 1,
    -2, -1, -3, -4, -6, -5, 5, 3, 4, 1, 2, 0,
  };
  static const PetscInt ttriMult[12*12] = {
    0, 2, 1, 3, 5, 4, -6, -4, -5, -3, -1, -2,
    1, 0, 2, 4, 3, 5, -5, -6, -4, -2, -3, -1,
    2, 1, 0, 5, 4, 3, -4, -5, -6, -1, -2, -3,
    3, 5, 4, 0, 2, 1, -3, -1, -2, -6, -4, -5,
    4, 3, 5, 1, 0, 2, -2, -3, -1, -5, -6, -4,
    5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6,
    -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
    -5, -4, -6, -2, -1, -3, 1, 2, 0, 4, 5, 3,
    -4, -6, -5, -1, -3, -2, 2, 0, 1, 5, 3, 4,
    -3, -2, -1, -6, -5, -4, 3, 4, 5, 0, 1, 2,
    -2, -1, -3, -5, -4, -6, 4, 5, 3, 1, 2, 0,
    -1, -3, -2, -4, -6, -5, 5, 3, 4, 2, 0, 1,
  };
  static const PetscInt tquadMult[16*16] = {
    0, 3, 2, 1, 4, 7, 6, 5, -8, -5, -6, -7, -4, -1, -2, -3,
    1, 0, 3, 2, 5, 4, 7, 6, -7, -8, -5, -6, -3, -4, -1, -2,
    2, 1, 0, 3, 6, 5, 4, 7, -6, -7, -8, -5, -2, -3, -4, -1,
    3, 2, 1, 0, 7, 6, 5, 4, -5, -6, -7, -8, -1, -2, -3, -4,
    4, 7, 6, 5, 0, 3, 2, 1, -4, -1, -2, -3, -8, -5, -6, -7,
    5, 4, 7, 6, 1, 0, 3, 2, -3, -4, -1, -2, -7, -8, -5, -6,
    6, 5, 4, 7, 2, 1, 0, 3, -2, -3, -4, -1, -6, -7, -8, -5,
    7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8,
    -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
    -7, -6, -5, -8, -3, -2, -1, -4, 1, 2, 3, 0, 5, 6, 7, 4,
    -6, -5, -8, -7, -2, -1, -4, -3, 2, 3, 0, 1, 6, 7, 4, 5,
    -5, -8, -7, -6, -1, -4, -3, -2, 3, 0, 1, 2, 7, 4, 5, 6,
    -4, -3, -2, -1, -8, -7, -6, -5, 4, 5, 6, 7, 0, 1, 2, 3,
    -3, -2, -1, -4, -7, -6, -5, -8, 5, 6, 7, 4, 1, 2, 3, 0,
    -2, -1, -4, -3, -6, -5, -8, -7, 6, 7, 4, 5, 2, 3, 0, 1,
    -1, -4, -3, -2, -5, -8, -7, -6, 7, 4, 5, 6, 3, 0, 1, 2,
  };
  static const PetscInt pyrMult[8*8] = {
    0, 3, 2, 1, -4, -1, -2, -3,
    1, 0, 3, 2, -3, -4, -1, -2,
    2, 1, 0, 3, -2, -3, -4, -1,
    3, 2, 1, 0, -1, -2, -3, -4,
    -4, -3, -2, -1, 0, 1, 2, 3,
    -3, -2, -1, -4, 1, 2, 3, 0,
    -2, -1, -4, -3, 2, 3, 0, 1,
    -1, -4, -3, -2, 3, 0, 1, 2,
  };
  switch (ct) {
    case DM_POLYTOPE_POINT:              return 0;
    case DM_POLYTOPE_SEGMENT:
    case DM_POLYTOPE_POINT_PRISM_TENSOR: return segMult[(o1+1)*2+o2+1];
    case DM_POLYTOPE_TRIANGLE:           return triMult[(o1+3)*6+o2+3];
    case DM_POLYTOPE_QUADRILATERAL:      return quadMult[(o1+4)*8+o2+4];
    case DM_POLYTOPE_SEG_PRISM_TENSOR:   return tsegMult[(o1+2)*4+o2+2];
    case DM_POLYTOPE_TETRAHEDRON:        return tetMult[(o1+12)*24+o2+12];
    case DM_POLYTOPE_HEXAHEDRON:         return hexMult[(o1+24)*48+o2+24];
    case DM_POLYTOPE_TRI_PRISM:          return tripMult[(o1+6)*12+o2+6];
    case DM_POLYTOPE_TRI_PRISM_TENSOR:   return ttriMult[(o1+6)*12+o2+6];
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:  return tquadMult[(o1+8)*16+o2+8];
    case DM_POLYTOPE_PYRAMID:            return pyrMult[(o1+4)*8+o2+4];
    default: return 0;
  }
}

/* This is orientation o1 acting on orientation o2^{-1} */
static inline PetscInt DMPolytopeTypeComposeOrientationInv(DMPolytopeType ct, PetscInt o1, PetscInt o2)
{
  static const PetscInt triInv[6]    = {-3, -2, -1, 0, 2, 1};
  static const PetscInt quadInv[8]   = {-4, -3, -2, -1, 0, 3, 2, 1};
  static const PetscInt tetInv[24]   = {-9, -11, -4, -12, -5, -7, -6, -8, -10, -3, -2, -1, 0, 2, 1, 3, 8, 10, 6, 11, 4, 9, 5, 7};
  static const PetscInt hexInv[48]   = {-17, -18, -20, -19, -22, -21, -23, -24, -15, -16, -14, -13, -11, -12, -10, -9, -8, -5, -6, -7, -4, -3, -2, -1,
                                          0,   3,   2,   1,   6,   5,   4,   9,   8,   7,  10,  11,  12,  13,  14, 15, 17, 16, 19, 18, 21, 20, 23, 22};
  static const PetscInt tripInv[12]  = {-5, -6, -4, -3, -2, -1, 0, 2, 1, 3, 4, 5};
  static const PetscInt ttriInv[12]  = {-6, -5, -4, -3, -2, -1, 0, 2, 1, 3, 5, 4};
  static const PetscInt tquadInv[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 3, 2, 1, 4, 7, 6, 5};
  static const PetscInt pyrInv[8]    = {-4, -3, -2, -1, 0, 3, 2, 1};
  switch (ct) {
    case DM_POLYTOPE_POINT:              return 0;
    case DM_POLYTOPE_SEGMENT:
    case DM_POLYTOPE_POINT_PRISM_TENSOR: return DMPolytopeTypeComposeOrientation(ct, o1, o2);
    case DM_POLYTOPE_TRIANGLE:           return DMPolytopeTypeComposeOrientation(ct, o1, triInv[o2+3]);
    case DM_POLYTOPE_QUADRILATERAL:      return DMPolytopeTypeComposeOrientation(ct, o1, quadInv[o2+4]);
    case DM_POLYTOPE_SEG_PRISM_TENSOR:   return DMPolytopeTypeComposeOrientation(ct, o1, o2);
    case DM_POLYTOPE_TETRAHEDRON:        return DMPolytopeTypeComposeOrientation(ct, o1, tetInv[o2+12]);
    case DM_POLYTOPE_HEXAHEDRON:         return DMPolytopeTypeComposeOrientation(ct, o1, hexInv[o2+24]);
    case DM_POLYTOPE_TRI_PRISM:          return DMPolytopeTypeComposeOrientation(ct, o1, tripInv[o2+6]);
    case DM_POLYTOPE_TRI_PRISM_TENSOR:   return DMPolytopeTypeComposeOrientation(ct, o1, ttriInv[o2+6]);
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:  return DMPolytopeTypeComposeOrientation(ct, o1, tquadInv[o2+8]);
    case DM_POLYTOPE_PYRAMID:            return DMPolytopeTypeComposeOrientation(ct, o1, pyrInv[o2+4]);
    default: return 0;
  }
}

PETSC_EXTERN PetscErrorCode DMPolytopeMatchOrientation(DMPolytopeType, const PetscInt[], const PetscInt[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPolytopeMatchVertexOrientation(DMPolytopeType, const PetscInt[], const PetscInt[], PetscInt *, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPolytopeGetOrientation(DMPolytopeType, const PetscInt[], const PetscInt[], PetscInt *);
PETSC_EXTERN PetscErrorCode DMPolytopeGetVertexOrientation(DMPolytopeType, const PetscInt[], const PetscInt[], PetscInt *);
PETSC_EXTERN PetscErrorCode DMPolytopeInCellTest(DMPolytopeType, const PetscReal[], PetscBool *);

#endif
