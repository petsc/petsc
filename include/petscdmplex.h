/*
  DMPlex, for parallel unstructured distributed mesh problems.
*/
#if !defined(__PETSCDMPLEX_H)
#define __PETSCDMPLEX_H

#include <petscsftypes.h>
#include <petscdm.h>
#include <petscdt.h>
#include <petscfe.h>

PETSC_EXTERN PetscErrorCode DMPlexCreate(MPI_Comm, DM*);
PETSC_EXTERN PetscErrorCode DMPlexCreateCohesiveSubmesh(DM, PetscBool, const char [], PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateFromCellList(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscBool, const int[], PetscInt, const double[], DM*);
PETSC_EXTERN PetscErrorCode DMPlexCreateFromDAG(DM, PetscInt, const PetscInt [], const PetscInt [], const PetscInt [], const PetscInt [], const PetscScalar []);
PETSC_EXTERN PetscErrorCode DMPlexGetDimension(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetDimension(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetChart(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetChart(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetConeSize(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetConeSize(DM, PetscInt, PetscInt);
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
PETSC_EXTERN PetscErrorCode DMPlexOrient(DM);
PETSC_EXTERN PetscErrorCode DMPlexInterpolate(DM, DM *);
PETSC_EXTERN PetscErrorCode DMPlexUninterpolate(DM, DM *);
PETSC_EXTERN PetscErrorCode DMPlexLoad(PetscViewer, DM);
PETSC_EXTERN PetscErrorCode DMPlexGetCoordinateSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMPlexSetCoordinateSection(DM, PetscSection);
PETSC_EXTERN PetscErrorCode DMPlexSetPreallocationCenterDimension(DM,PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexPreallocateOperator(DM, PetscInt, PetscSection, PetscSection, PetscInt[], PetscInt[], PetscInt[], PetscInt[], Mat, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetPointLocal(DM,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalRef(DM,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalRead(DM,PetscInt,const PetscScalar*,const void*);
PETSC_EXTERN PetscErrorCode DMPlexGetPointGlobal(DM,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalRef(DM,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalRead(DM,PetscInt,const PetscScalar*,const void*);

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
PETSC_EXTERN PetscErrorCode DMLabelGetNumValues(DMLabel, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelGetValueIS(DMLabel, IS *);
PETSC_EXTERN PetscErrorCode DMLabelGetStratumSize(DMLabel, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMLabelGetStratumIS(DMLabel, PetscInt, IS *);
PETSC_EXTERN PetscErrorCode DMLabelClearStratum(DMLabel, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelCreateIndex(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelDestroyIndex(DMLabel);
PETSC_EXTERN PetscErrorCode DMLabelHasPoint(DMLabel, PetscInt, PetscBool *);
PETSC_EXTERN PetscErrorCode DMLabelFilter(DMLabel, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMLabelPermute(DMLabel, IS, DMLabel *);

PETSC_EXTERN PetscErrorCode DMPlexCreateLabel(DM, const char []);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelValue(DM, const char[], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetLabelValue(DM, const char[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexClearLabelValue(DM, const char[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelSize(DM, const char[], PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelIdIS(DM, const char[], IS *);
PETSC_EXTERN PetscErrorCode DMPlexGetStratumSize(DM, const char [], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetStratumIS(DM, const char [], PetscInt, IS *);
PETSC_EXTERN PetscErrorCode DMPlexClearLabelStratum(DM, const char[], PetscInt);
PETSC_EXTERN PetscErrorCode PetscSectionCreateGlobalSectionLabel(PetscSection, PetscSF, PetscBool, DMLabel, PetscInt, PetscSection *);

PETSC_EXTERN PetscErrorCode DMPlexGetNumLabels(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetLabelName(DM, PetscInt, const char **);
PETSC_EXTERN PetscErrorCode DMPlexHasLabel(DM, const char [], PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexGetLabel(DM, const char *, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexAddLabel(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexRemoveLabel(DM, const char [], DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexGetCellNumbering(DM, IS *);
PETSC_EXTERN PetscErrorCode DMPlexGetVertexNumbering(DM, IS *);

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
PETSC_EXTERN PetscErrorCode DMPlexCreateCubeBoundary(DM, const PetscReal [], const PetscReal [], const PetscInt []);
PETSC_EXTERN PetscErrorCode DMPlexCreateBoxMesh(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateHexBoxMesh(MPI_Comm,PetscInt,const PetscInt[],DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateConeSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMPlexInvertCell(PetscInt, PetscInt, int []);
PETSC_EXTERN PetscErrorCode DMPlexCheckSymmetry(DM);
PETSC_EXTERN PetscErrorCode DMPlexCheckSkeleton(DM, PetscBool);

/* Mesh Partitioning and Distribution */
PETSC_EXTERN PetscErrorCode DMPlexCreateNeighborCSR(DM, PetscInt, PetscInt *, PetscInt **, PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexCreatePartition(DM, const char[], PetscInt, PetscBool, PetscSection *, IS *, PetscSection *, IS *);
PETSC_EXTERN PetscErrorCode DMPlexCreatePartitionClosure(DM, PetscSection, IS, PetscSection *, IS *);
PETSC_EXTERN PetscErrorCode DMPlexDistribute(DM, const char[], PetscInt, PetscSF*, DM*);
PETSC_EXTERN PetscErrorCode DMPlexDistributeField(DM,PetscSF,PetscSection,Vec,PetscSection,Vec);
PETSC_EXTERN PetscErrorCode DMPlexDistributeData(DM,PetscSF,PetscSection,MPI_Datatype,void*,PetscSection,void**);

PETSC_EXTERN PetscErrorCode DMPlexGetOrdering(DM, MatOrderingType, IS *);
PETSC_EXTERN PetscErrorCode DMPlexPermute(DM, IS, DM *);

/* Submesh Support */
PETSC_EXTERN PetscErrorCode DMPlexCreateSubmesh(DM, DMLabel, PetscInt, DM*);
PETSC_EXTERN PetscErrorCode DMPlexCreateHybridMesh(DM, DMLabel, DMLabel *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexGetSubpointMap(DM, DMLabel*);
PETSC_EXTERN PetscErrorCode DMPlexSetSubpointMap(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexCreateSubpointIS(DM, IS *);

PETSC_EXTERN PetscErrorCode DMPlexMarkBoundaryFaces(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexLabelComplete(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexLabelCohesiveComplete(DM, DMLabel, PetscBool, DM);

/* Mesh Refinement */
typedef PetscInt CellRefiner;

PETSC_EXTERN PetscErrorCode DMPlexGetRefinementLimit(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexSetRefinementLimit(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexGetRefinementUniform(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexSetRefinementUniform(DM, PetscBool);

/* Support for cell-vertex meshes */
PETSC_EXTERN PetscErrorCode DMPlexGetNumFaceVertices(DM, PetscInt, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetOrientedFace(DM, PetscInt, PetscInt, const PetscInt [], PetscInt, PetscInt [], PetscInt [], PetscInt [], PetscBool *);

/* FVM Support */
PETSC_EXTERN PetscErrorCode DMPlexComputeCellGeometryFVM(DM, PetscInt, PetscReal *, PetscReal [], PetscReal []);

/* FEM Support */
PETSC_EXTERN PetscErrorCode DMPlexCreateSection(DM, PetscInt, PetscInt,const PetscInt [],const PetscInt [], PetscInt,const PetscInt [],const IS [], PetscSection *);

PETSC_EXTERN PetscErrorCode DMPlexComputeCellGeometry(DM, PetscInt, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexVecGetClosure(DM, PetscSection, Vec, PetscInt, PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexVecRestoreClosure(DM, PetscSection, Vec, PetscInt, PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexVecSetClosure(DM, PetscSection, Vec, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexMatSetClosure(DM, PetscSection, PetscSection, Mat, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexCreateClosureIndex(DM, PetscSection);

PETSC_EXTERN PetscErrorCode DMPlexCreateExodus(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateCGNS(MPI_Comm, PetscInt, PetscBool, DM *);

PETSC_EXTERN PetscErrorCode DMPlexConstructGhostCells(DM, const char [], PetscInt *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexConstructCohesiveCells(DM, DMLabel, DM *);

PETSC_EXTERN PetscErrorCode DMPlexGetHybridBounds(DM, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetHybridBounds(DM, PetscInt, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetVTKCellHeight(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetVTKCellHeight(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexVTKWriteAll(PetscObject, PetscViewer);

PETSC_EXTERN PetscErrorCode DMPlexGetScale(DM, PetscUnit, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexSetScale(DM, PetscUnit, PetscReal);

typedef struct {
  DM    dm;
  Vec   u; /* The base vector for the Jacbobian action J(u) x */
  Mat   J; /* Preconditioner for testing */
  void *user;
} JacActionCtx;

PETSC_EXTERN PetscErrorCode DMPlexProjectFunction(DM, PetscFE[], void (**)(const PetscReal [], PetscScalar *), InsertMode, Vec);
PETSC_EXTERN PetscErrorCode DMPlexProjectFunctionLocal(DM, PetscFE[], void (**)(const PetscReal [], PetscScalar *), InsertMode, Vec);
PETSC_EXTERN PetscErrorCode DMPlexComputeL2Diff(DM, PetscFE[], void (**)(const PetscReal [], PetscScalar *), Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexComputeResidualFEM(DM, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeJacobianActionFEM(DM, Mat, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeJacobianFEM(DM, Vec, Mat, Mat, MatStructure*,void *);
#endif
