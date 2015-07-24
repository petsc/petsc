/*
  DMPlex, for parallel unstructured distributed mesh problems.
*/
#if !defined(__PETSCDMPLEX_H)
#define __PETSCDMPLEX_H

#include <petscdm.h>
#include <petscdt.h>
#include <petscfe.h>
#include <petscfv.h>
#include <petscsftypes.h>

PETSC_EXTERN PetscClassId PETSCPARTITIONER_CLASSID;

/*J
  PetscPartitionerType - String with the name of a PETSc graph partitioner

  Level: beginner

.seealso: PetscPartitionerSetType(), PetscPartitioner
J*/
typedef const char *PetscPartitionerType;
#define PETSCPARTITIONERCHACO    "chaco"
#define PETSCPARTITIONERPARMETIS "parmetis"
#define PETSCPARTITIONERSHELL    "shell"
#define PETSCPARTITIONERSIMPLE   "simple"

PETSC_EXTERN PetscFunctionList PetscPartitionerList;
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate(MPI_Comm, PetscPartitioner *);
PETSC_EXTERN PetscErrorCode PetscPartitionerDestroy(PetscPartitioner *);
PETSC_EXTERN PetscErrorCode PetscPartitionerSetType(PetscPartitioner, PetscPartitionerType);
PETSC_EXTERN PetscErrorCode PetscPartitionerGetType(PetscPartitioner, PetscPartitionerType *);
PETSC_EXTERN PetscErrorCode PetscPartitionerSetUp(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerSetFromOptions(PetscPartitioner);
PETSC_STATIC_INLINE PetscErrorCode PetscPartitionerViewFromOptions(PetscPartitioner A,PetscObject B,const char name[]) {return PetscObjectViewFromOptions((PetscObject)A,B,name);}
PETSC_EXTERN PetscErrorCode PetscPartitionerView(PetscPartitioner, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscPartitionerRegister(const char [], PetscErrorCode (*)(PetscPartitioner));
PETSC_EXTERN PetscErrorCode PetscPartitionerRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscPartitionerPartition(PetscPartitioner, DM, PetscSection, IS *);

PETSC_EXTERN PetscErrorCode PetscPartitionerShellSetPartition(PetscPartitioner, PetscInt, const PetscInt[], const PetscInt[]);

PETSC_EXTERN PetscErrorCode DMPlexCreate(MPI_Comm, DM*);
PETSC_EXTERN PetscErrorCode DMPlexCreateCohesiveSubmesh(DM, PetscBool, const char [], PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateFromCellList(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscBool, const int[], PetscInt, const double[], DM*);
PETSC_EXTERN PetscErrorCode DMPlexCreateFromDAG(DM, PetscInt, const PetscInt [], const PetscInt [], const PetscInt [], const PetscInt [], const PetscScalar []);
PETSC_EXTERN PetscErrorCode DMPlexCreateReferenceCell(MPI_Comm, PetscInt, PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMPlexGetChart(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetChart(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetConeSize(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetConeSize(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexAddConeSize(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetCone(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexSetCone(DM, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexInsertCone(DM, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexInsertConeOrientation(DM, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetConeOrientation(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexSetConeOrientation(DM, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexGetSupportSize(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetSupportSize(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetSupport(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexSetSupport(DM, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexInsertSupport(DM, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetConeSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMPlexGetSupportSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMPlexGetCones(DM, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexGetConeOrientations(DM, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexGetMaxSizes(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSymmetrize(DM);
PETSC_EXTERN PetscErrorCode DMPlexStratify(DM);
PETSC_EXTERN PetscErrorCode DMPlexEqual(DM,DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMPlexReverseCell(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexOrient(DM);
PETSC_EXTERN PetscErrorCode DMPlexInterpolate(DM, DM *);
PETSC_EXTERN PetscErrorCode DMPlexUninterpolate(DM, DM *);
PETSC_EXTERN PetscErrorCode DMPlexLoad(PetscViewer, DM);
PETSC_EXTERN PetscErrorCode DMPlexPreallocateOperator(DM, PetscInt, PetscInt[], PetscInt[], PetscInt[], PetscInt[], Mat, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetPointLocal(DM,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalRead(DM,PetscInt,const PetscScalar*,const void*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalRef(DM,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexGetPointLocalField(DM,PetscInt,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalFieldRef(DM,PetscInt,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalFieldRead(DM,PetscInt,PetscInt,const PetscScalar*,const void*);
PETSC_EXTERN PetscErrorCode DMPlexGetPointGlobal(DM,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalRead(DM,PetscInt,const PetscScalar*,const void*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalRef(DM,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexGetPointGlobalField(DM,PetscInt,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalFieldRef(DM,PetscInt,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalFieldRead(DM,PetscInt,PetscInt,const PetscScalar*,const void*);

/*S
  DMLabel - Object which encapsulates a subset of the mesh from this DM

  Level: developer

  Concepts: grids, grid refinement

.seealso:  DM, DMPlexCreate(), DMPlexCreateLabel()
S*/
typedef struct _n_DMLabel *DMLabel;
PETSC_EXTERN PetscErrorCode DMLabelCreate(const char [], DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelView(DMLabel, PetscViewer);
PETSC_EXTERN PetscErrorCode DMLabelDestroy(DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelDuplicate(DMLabel, DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelGetName(DMLabel, const char **);
PETSC_EXTERN PetscErrorCode DMLabelGetValue(DMLabel, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelSetValue(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelClearValue(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelInsertIS(DMLabel, IS, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelGetNumValues(DMLabel, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelGetStratumBounds(DMLabel, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelGetValueIS(DMLabel, IS *);
PETSC_EXTERN PetscErrorCode DMLabelStratumHasPoint(DMLabel, PetscInt, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMLabelGetStratumSize(DMLabel, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelGetStratumIS(DMLabel, PetscInt, IS *);
PETSC_EXTERN PetscErrorCode DMLabelClearStratum(DMLabel, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelCreateIndex(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelDestroyIndex(DMLabel);
PETSC_EXTERN PetscErrorCode DMLabelHasValue(DMLabel, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMLabelHasPoint(DMLabel, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMLabelFilter(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelPermute(DMLabel, IS, DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelDistribute(DMLabel, PetscSF, DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelConvertToSection(DMLabel, PetscSection *, IS *);

PETSC_EXTERN PetscErrorCode DMPlexFilter(DM, DMLabel, PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateLabel(DM, const char []);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelValue(DM, const char[], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetLabelValue(DM, const char[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexClearLabelValue(DM, const char[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelSize(DM, const char[], PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelIdIS(DM, const char[], IS *);
PETSC_EXTERN PetscErrorCode DMPlexGetStratumSize(DM, const char [], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetStratumIS(DM, const char [], PetscInt, IS *);
PETSC_EXTERN PetscErrorCode DMPlexClearLabelStratum(DM, const char[], PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelOutput(DM, const char[], PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexSetLabelOutput(DM, const char[], PetscBool);
PETSC_EXTERN PetscErrorCode PetscSectionCreateGlobalSectionLabel(PetscSection, PetscSF, PetscBool, DMLabel, PetscInt, PetscSection *);

PETSC_EXTERN PetscErrorCode DMPlexGetNumLabels(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelName(DM, PetscInt, const char **);
PETSC_EXTERN PetscErrorCode DMPlexHasLabel(DM, const char [], PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexGetLabel(DM, const char *, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelByNum(DM, PetscInt, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexAddLabel(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexRemoveLabel(DM, const char [], DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexGetCellNumbering(DM, IS *);
PETSC_EXTERN PetscErrorCode DMPlexGetVertexNumbering(DM, IS *);
PETSC_EXTERN PetscErrorCode DMPlexCreatePointNumbering(DM, IS *);

PETSC_EXTERN PetscErrorCode DMPlexGetDepth(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetDepthLabel(DM, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexGetDepthStratum(DM, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetHeightStratum(DM, PetscInt, PetscInt *, PetscInt *);

/* Topological Operations */
PETSC_EXTERN PetscErrorCode DMPlexGetMeet(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetFullMeet(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreMeet(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetJoin(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetFullJoin(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreJoin(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetTransitiveClosure(DM, PetscInt, PetscBool, PetscInt *, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexRestoreTransitiveClosure(DM, PetscInt, PetscBool, PetscInt *, PetscInt *[]);

/* Mesh Generation */
PETSC_EXTERN PetscErrorCode DMPlexGenerate(DM, const char [], PetscBool , DM *);
PETSC_EXTERN PetscErrorCode DMPlexCopyCoordinates(DM, DM);
PETSC_EXTERN PetscErrorCode DMPlexCopyLabels(DM, DM);
PETSC_EXTERN PetscErrorCode DMPlexCreateDoublet(MPI_Comm, PetscInt, PetscBool, PetscBool, PetscBool, PetscReal, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateSquareBoundary(DM, const PetscReal [], const PetscReal [], const PetscInt []);
PETSC_EXTERN PetscErrorCode DMPlexCreateCubeBoundary(DM, const PetscReal [], const PetscReal [], const PetscInt []);
PETSC_EXTERN PetscErrorCode DMPlexCreateSquareMesh(DM, const PetscReal [], const PetscReal [], const PetscInt [], DMBoundaryType, DMBoundaryType);
PETSC_EXTERN PetscErrorCode DMPlexCreateBoxMesh(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateHexBoxMesh(MPI_Comm, PetscInt, const PetscInt[], DMBoundaryType, DMBoundaryType, DMBoundaryType, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateConeSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMPlexInvertCell(PetscInt, PetscInt, int []);
PETSC_EXTERN PetscErrorCode DMPlexLocalizeCoordinate(DM, const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode DMPlexLocalizeCoordinates(DM);
PETSC_EXTERN PetscErrorCode DMPlexCheckSymmetry(DM);
PETSC_EXTERN PetscErrorCode DMPlexCheckSkeleton(DM, PetscBool, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexCheckFaces(DM, PetscBool, PetscInt);

PETSC_EXTERN PetscErrorCode DMPlexTriangleSetOptions(DM, const char *);
PETSC_EXTERN PetscErrorCode DMPlexTetgenSetOptions(DM, const char *);

/* Mesh Partitioning and Distribution */
PETSC_EXTERN PetscErrorCode DMPlexCreateNeighborCSR(DM, PetscInt, PetscInt *, PetscInt **, PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetPartitioner(DM, PetscPartitioner *);
PETSC_EXTERN PetscErrorCode DMPlexSetPartitioner(DM, PetscPartitioner);
PETSC_EXTERN PetscErrorCode DMPlexCreatePartition(DM, const char[], PetscInt, PetscBool, PetscSection *, IS *, PetscSection *, IS *);
PETSC_EXTERN PetscErrorCode DMPlexCreatePartitionerGraph(DM, PetscInt, PetscInt *, PetscInt **, PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexCreatePartitionClosure(DM, PetscSection, IS, PetscSection *, IS *);
PETSC_EXTERN PetscErrorCode DMPlexPartitionLabelInvert(DM, DMLabel, PetscSF, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexPartitionLabelClosure(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexPartitionLabelAdjacency(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexPartitionLabelCreateSF(DM, DMLabel, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexDistribute(DM, PetscInt, PetscSF*, DM*);
PETSC_EXTERN PetscErrorCode DMPlexDistributeOverlap(DM, PetscInt, PetscSF *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexDistributeField(DM,PetscSF,PetscSection,Vec,PetscSection,Vec);
PETSC_EXTERN PetscErrorCode DMPlexDistributeFieldIS(DM, PetscSF, PetscSection, IS, PetscSection, IS *);
PETSC_EXTERN PetscErrorCode DMPlexDistributeData(DM,PetscSF,PetscSection,MPI_Datatype,void*,PetscSection,void**);
PETSC_EXTERN PetscErrorCode DMPlexMigrate(DM, PetscSF, DM);
PETSC_EXTERN PetscErrorCode DMPlexSetAdjacencyUseCone(DM,PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetAdjacencyUseCone(DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMPlexSetAdjacencyUseClosure(DM,PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetAdjacencyUseClosure(DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMPlexSetAdjacencyUseAnchors(DM,PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetAdjacencyUseAnchors(DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMPlexGetAdjacency(DM, PetscInt, PetscInt *, PetscInt *[]);

PETSC_EXTERN PetscErrorCode DMPlexGetOrdering(DM, MatOrderingType, IS *);
PETSC_EXTERN PetscErrorCode DMPlexPermute(DM, IS, DM *);

PETSC_EXTERN PetscErrorCode DMPlexCreateProcessSF(DM, PetscSF, IS *, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexCreateTwoSidedProcessSF(DM, PetscSF, PetscSection, IS, PetscSection, IS, IS *, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexDistributeOwnership(DM, PetscSection, IS *, PetscSection, IS *);
PETSC_EXTERN PetscErrorCode DMPlexCreateOverlap(DM, PetscInt, PetscSection, IS, PetscSection, IS, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexCreateOverlapMigrationSF(DM, PetscSF, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexStratifyMigrationSF(DM, PetscSF, PetscSF *);

/* Submesh Support */
PETSC_EXTERN PetscErrorCode DMPlexCreateSubmesh(DM, DMLabel, PetscInt, DM*);
PETSC_EXTERN PetscErrorCode DMPlexCreateHybridMesh(DM, DMLabel, DMLabel *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexGetSubpointMap(DM, DMLabel*);
PETSC_EXTERN PetscErrorCode DMPlexSetSubpointMap(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexCreateSubpointIS(DM, IS *);

PETSC_EXTERN PetscErrorCode DMPlexMarkBoundaryFaces(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexLabelComplete(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexLabelCohesiveComplete(DM, DMLabel, DMLabel, PetscBool, DM);
PETSC_EXTERN PetscErrorCode DMPlexLabelAddCells(DM, DMLabel);

PETSC_EXTERN PetscErrorCode DMPlexGetRefinementLimit(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexSetRefinementLimit(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexGetRefinementUniform(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexSetRefinementUniform(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetRefinementFunction(DM, PetscErrorCode (**)(const PetscReal [], PetscReal *));
PETSC_EXTERN PetscErrorCode DMPlexSetRefinementFunction(DM, PetscErrorCode (*)(const PetscReal [], PetscReal *));
PETSC_EXTERN PetscErrorCode DMPlexGetCoarseDM(DM, DM *);
PETSC_EXTERN PetscErrorCode DMPlexSetCoarseDM(DM, DM);
PETSC_EXTERN PetscErrorCode DMPlexCreateCoarsePointIS(DM, IS *);
PETSC_EXTERN PetscErrorCode DMPlexGetRegularRefinement(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexSetRegularRefinement(DM, PetscBool);

/* Support for cell-vertex meshes */
PETSC_EXTERN PetscErrorCode DMPlexGetNumFaceVertices(DM, PetscInt, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetOrientedFace(DM, PetscInt, PetscInt, const PetscInt [], PetscInt, PetscInt [], PetscInt [], PetscInt [], PetscBool *);

/* Geometry Support */
PETSC_EXTERN PetscErrorCode DMPlexGetMinRadius(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexSetMinRadius(DM, PetscReal);

/* Point Location */
typedef struct _PetscGridHash *PetscGridHash;
PETSC_EXTERN PetscErrorCode PetscGridHashCreate(MPI_Comm, PetscInt, const PetscScalar[], PetscGridHash *);
PETSC_EXTERN PetscErrorCode PetscGridHashEnlarge(PetscGridHash, const PetscScalar []);
PETSC_EXTERN PetscErrorCode PetscGridHashSetGrid(PetscGridHash, const PetscInt [], const PetscReal []);
PETSC_EXTERN PetscErrorCode PetscGridHashGetEnclosingBox(PetscGridHash, PetscInt, const PetscScalar [], PetscInt [], PetscInt []);
PETSC_EXTERN PetscErrorCode PetscGridHashDestroy(PetscGridHash *);

/* FVM Support */
PETSC_EXTERN PetscErrorCode DMPlexComputeCellGeometryFVM(DM, PetscInt, PetscReal *, PetscReal [], PetscReal []);
PETSC_EXTERN PetscErrorCode DMPlexComputeGeometryFVM(DM, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexComputeGradientFVM(DM, PetscFV, Vec, Vec, DM *);

/* FEM Support */
PETSC_EXTERN PetscErrorCode DMPlexInsertBoundaryValues(DM, Vec, PetscReal, Vec, Vec, Vec);

PETSC_EXTERN PetscErrorCode DMPlexCreateSection(DM, PetscInt, PetscInt, const PetscInt [], const PetscInt [], PetscInt, const PetscInt [], const IS [], const IS [], IS, PetscSection *);

PETSC_EXTERN PetscErrorCode DMPlexComputeCellGeometryAffineFEM(DM, PetscInt, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexComputeCellGeometryFEM(DM, PetscInt, PetscFE, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexComputeGeometryFEM(DM, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexVecGetClosure(DM, PetscSection, Vec, PetscInt, PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexVecRestoreClosure(DM, PetscSection, Vec, PetscInt, PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexVecSetClosure(DM, PetscSection, Vec, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexMatSetClosure(DM, PetscSection, PetscSection, Mat, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexGetClosureIndices(DM, PetscSection, PetscSection, PetscInt, PetscInt *, PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreClosureIndices(DM, PetscSection, PetscSection, PetscInt, PetscInt *, PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexMatSetClosureRefined(DM, PetscSection, PetscSection, DM, PetscSection, PetscSection, Mat, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexMatGetClosureIndicesRefined(DM, PetscSection, PetscSection, DM, PetscSection, PetscSection, PetscInt, PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexCreateClosureIndex(DM, PetscSection);

PETSC_EXTERN PetscErrorCode DMPlexCreateFromFile(MPI_Comm, const char[], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateExodus(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateExodusFromFile(MPI_Comm, const char [], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateCGNS(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateCGNSFromFile(MPI_Comm, const char[], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateGmsh(MPI_Comm, PetscViewer, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateGmshFromFile(MPI_Comm, const char [], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateFluent(MPI_Comm, PetscViewer, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateFluentFromFile(MPI_Comm, const char [], PetscBool, DM *);

PETSC_EXTERN PetscErrorCode DMPlexConstructGhostCells(DM, const char [], PetscInt *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexConstructCohesiveCells(DM, DMLabel, DM *);

PETSC_EXTERN PetscErrorCode DMPlexGetHybridBounds(DM, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetHybridBounds(DM, PetscInt, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetVTKCellHeight(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetVTKCellHeight(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexVTKWriteAll(PetscObject, PetscViewer);

PETSC_EXTERN PetscErrorCode DMPlexGetScale(DM, PetscUnit, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexSetScale(DM, PetscUnit, PetscReal);

typedef struct _n_Boundary *DMBoundary;
PETSC_EXTERN PetscErrorCode DMPlexAddBoundary(DM, PetscBool, const char[], const char[], PetscInt, PetscInt, const PetscInt *, void (*)(), PetscInt, const PetscInt *, void *);
PETSC_EXTERN PetscErrorCode DMPlexGetNumBoundary(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetBoundary(DM, PetscInt, PetscBool *, const char **, const char **, PetscInt *, PetscInt *, const PetscInt **, void (**)(), PetscInt *, const PetscInt **, void **);
PETSC_EXTERN PetscErrorCode DMPlexIsBoundaryPoint(DM, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexCopyBoundary(DM, DM);

typedef struct {
  DM    dm;
  Vec   u; /* The base vector for the Jacbobian action J(u) x */
  Mat   J; /* Preconditioner for testing */
  void *user;
} JacActionCtx;

PETSC_EXTERN PetscErrorCode DMPlexInsertBoundaryValuesFEM(DM, Vec);
PETSC_EXTERN PetscErrorCode DMPlexSetMaxProjectionHeight(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetMaxProjectionHeight(DM, PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexProjectFunction(DM, PetscErrorCode (**)(PetscInt, const PetscReal [], PetscInt, PetscScalar *, void *), void **, InsertMode, Vec);
PETSC_EXTERN PetscErrorCode DMPlexProjectFunctionLocal(DM, PetscErrorCode (**)(PetscInt, const PetscReal [], PetscInt, PetscScalar *, void *), void **, InsertMode, Vec);
PETSC_EXTERN PetscErrorCode DMPlexProjectFieldLocal(DM, Vec, void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal [], PetscScalar []), InsertMode, Vec);
PETSC_EXTERN PetscErrorCode DMPlexComputeL2Diff(DM, PetscErrorCode (**)(PetscInt, const PetscReal [], PetscInt, PetscScalar *, void *), void **, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexComputeL2GradientDiff(DM, PetscErrorCode (**)(PetscInt, const PetscReal [], const PetscReal [], PetscInt, PetscScalar *, void *), void **, Vec, const PetscReal [], PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexComputeL2FieldDiff(DM, PetscErrorCode (**)(PetscInt, const PetscReal [], PetscInt, PetscScalar *, void *), void **, Vec, PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexComputeIntegralFEM(DM, Vec, PetscReal *, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeInterpolatorNested(DM, DM, Mat, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeInterpolatorGeneral(DM, DM, Mat, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeInjectorFEM(DM, DM, VecScatter *, void *);

PETSC_EXTERN PetscErrorCode DMPlexCreateRigidBody(DM, MatNullSpace *);

PETSC_EXTERN PetscErrorCode DMPlexSNESComputeResidualFEM(DM, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexSNESComputeJacobianFEM(DM, Vec, Mat, Mat,void *);

PETSC_EXTERN PetscErrorCode DMPlexTSComputeRHSFunctionFVM(DM, PetscReal, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexTSComputeIFunctionFEM(DM, PetscReal, Vec, Vec, Vec, void *);

PETSC_EXTERN PetscErrorCode DMPlexComputeRHSFunctionFVM(DM, PetscReal, Vec, Vec, void *);

/* anchors */
PETSC_EXTERN PetscErrorCode DMPlexGetAnchors(DM, PetscSection*, IS*);
PETSC_EXTERN PetscErrorCode DMPlexSetAnchors(DM, PetscSection, IS);
/* tree */
PETSC_EXTERN PetscErrorCode DMPlexSetReferenceTree(DM, DM);
PETSC_EXTERN PetscErrorCode DMPlexGetReferenceTree(DM, DM*);
PETSC_EXTERN PetscErrorCode DMPlexReferenceTreeGetChildSymmetry(DM, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexCreateDefaultReferenceTree(MPI_Comm, PetscInt, PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMPlexSetTree(DM, PetscSection, PetscInt [], PetscInt []);
PETSC_EXTERN PetscErrorCode DMPlexGetTree(DM, PetscSection *, PetscInt *[], PetscInt *[], PetscSection *, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexGetTreeParent(DM, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetTreeChildren(DM, PetscInt, PetscInt *, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexTreeRefineCell(DM, PetscInt, DM *);

/* natural order */
PETSC_EXTERN PetscErrorCode DMPlexCreateGlobalToNaturalSF(DM, PetscSection, PetscSF, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexGlobalToNaturalBegin(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexGlobalToNaturalEnd(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexNaturalToGlobalBegin(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexNaturalToGlobalEnd(DM, Vec, Vec);
#endif
