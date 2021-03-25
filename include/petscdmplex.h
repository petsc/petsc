/*
  DMPlex, for parallel unstructured distributed mesh problems.
*/
#if !defined(PETSCDMPLEX_H)
#define PETSCDMPLEX_H

#include <petscsection.h>
#include <petscpartitioner.h>
#include <petscdm.h>
#include <petscdmplextypes.h>
#include <petscdt.h>
#include <petscfe.h>
#include <petscfv.h>
#include <petscsftypes.h>
#include <petscdmfield.h>
#include <petscviewer.h>

PETSC_EXTERN PetscErrorCode PetscPartitionerDMPlexPartition(PetscPartitioner, DM, PetscSection, PetscSection, IS *);

PETSC_EXTERN PetscErrorCode DMPlexBuildFromCellList(DM, PetscInt, PetscInt, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexBuildFromCellListParallel(DM, PetscInt, PetscInt, PetscInt, PetscInt, const PetscInt[], PetscSF *, PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexBuildCoordinatesFromCellList(DM, PetscInt, const PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexBuildCoordinatesFromCellListParallel(DM, PetscInt, PetscSF, const PetscReal[]);

PETSC_EXTERN PetscErrorCode DMPlexCreate(MPI_Comm, DM*);
PETSC_EXTERN PetscErrorCode DMPlexCreateCohesiveSubmesh(DM, PetscBool, const char [], PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateFromCellListPetsc(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscBool, const PetscInt[], PetscInt, const PetscReal[], DM*);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use DMPlexCreateFromCellListPetsc() (since version 3.14)") PetscErrorCode DMPlexCreateFromCellList(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscBool, const int[], PetscInt, const double[], DM*);
PETSC_EXTERN PetscErrorCode DMPlexCreateFromCellListParallelPetsc(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscInt, PetscBool, const PetscInt[], PetscInt, const PetscReal[], PetscSF *, PetscInt **, DM *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use DMPlexCreateFromCellListParallelPetsc() (since version 3.14)") PetscErrorCode DMPlexCreateFromCellListParallel(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt, PetscBool, const int[], PetscInt, const PetscReal[], PetscSF *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateFromDAG(DM, PetscInt, const PetscInt [], const PetscInt [], const PetscInt [], const PetscInt [], const PetscScalar []);
PETSC_EXTERN PetscErrorCode DMPlexCreateReferenceCell(MPI_Comm, DMPolytopeType, DM *);
PETSC_EXTERN PetscErrorCode DMPlexSetOptionsPrefix(DM, const char []);
PETSC_EXTERN PetscErrorCode DMPlexGetChart(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetChart(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetConeSize(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetConeSize(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexAddConeSize(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetCone(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexGetConeTuple(DM, IS, PetscSection *, IS *);
PETSC_EXTERN PetscErrorCode DMPlexGetConeRecursive(DM, IS, PetscInt *, IS *[], PetscSection *[]);
PETSC_EXTERN PetscErrorCode DMPlexRestoreConeRecursive(DM, IS, PetscInt *, IS *[], PetscSection *[]);
PETSC_EXTERN PetscErrorCode DMPlexGetConeRecursiveVertices(DM, IS, IS *);
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
PETSC_EXTERN PetscErrorCode DMPlexOrientPoint(DM,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexOrient(DM);
PETSC_EXTERN PetscErrorCode DMPlexPreallocateOperator(DM, PetscInt, PetscInt[], PetscInt[], PetscInt[], PetscInt[], Mat, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetPointLocal(DM,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalRead(DM,PetscInt,const PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalRef(DM,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexGetPointLocalField(DM,PetscInt,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalFieldRef(DM,PetscInt,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexPointLocalFieldRead(DM,PetscInt,PetscInt,const PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexGetPointGlobal(DM,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalRead(DM,PetscInt,const PetscScalar*,const void*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalRef(DM,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexGetPointGlobalField(DM,PetscInt,PetscInt,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalFieldRef(DM,PetscInt,PetscInt,PetscScalar*,void*);
PETSC_EXTERN PetscErrorCode DMPlexPointGlobalFieldRead(DM,PetscInt,PetscInt,const PetscScalar*, void*);

/* Topological interpolation */
PETSC_EXTERN const char * const DMPlexInterpolatedFlags[];
/*E
   DMPlexInterpolatedFlag - Describes level of topological interpolatedness.
     It is a local or collective property depending on whether it is returned by DMPlexIsInterpolated() or DMPlexIsInterpolatedCollective().

$  DMPLEX_INTERPOLATED_INVALID - Uninitialized value (internal use only; never returned by DMPlexIsInterpolated() or DMPlexIsInterpolatedCollective())
$  DMPLEX_INTERPOLATED_NONE    - Mesh is not interpolated
$  DMPLEX_INTERPOLATED_PARTIAL - Mesh is partially interpolated. This can e.g. mean DMPlex with cells, faces and vertices but no edges represented, or a mesh with mixed cones (see DMPlexStratify() for an example)
$  DMPLEX_INTERPOLATED_MIXED   - Can be returned only by DMPlexIsInterpolatedCollective(), meaning that DMPlexIsInterpolated() returns different interpolatedness on different ranks
$  DMPLEX_INTERPOLATED_FULL    - Mesh is fully interpolated

   Level: intermediate

   Developer Note:
   Any additions/changes here MUST also be made in include/petsc/finclude/petscdmplex.h and src/dm/f90-mod/petscdmplex.h

.seealso: DMPlexIsInterpolated(), DMPlexIsInterpolatedCollective(), DMPlexInterpolate(), DMPlexUninterpolate()
E*/
typedef enum {
  DMPLEX_INTERPOLATED_INVALID = -1,
  DMPLEX_INTERPOLATED_NONE = 0,
  DMPLEX_INTERPOLATED_PARTIAL,
  DMPLEX_INTERPOLATED_MIXED,
  DMPLEX_INTERPOLATED_FULL
} DMPlexInterpolatedFlag;
PETSC_EXTERN PetscErrorCode DMPlexInterpolate(DM, DM *);
PETSC_EXTERN PetscErrorCode DMPlexUninterpolate(DM, DM *);
PETSC_EXTERN PetscErrorCode DMPlexInterpolatePointSF(DM, PetscSF);
PETSC_EXTERN PetscErrorCode DMPlexIsInterpolated(DM, DMPlexInterpolatedFlag *);
PETSC_EXTERN PetscErrorCode DMPlexIsInterpolatedCollective(DM, DMPlexInterpolatedFlag *);

PETSC_EXTERN PetscErrorCode DMPlexFilter(DM, DMLabel, PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMPlexGetCellNumbering(DM, IS *);
PETSC_EXTERN PetscErrorCode DMPlexGetVertexNumbering(DM, IS *);
PETSC_EXTERN PetscErrorCode DMPlexCreatePointNumbering(DM, IS *);
PETSC_EXTERN PetscErrorCode DMPlexCreateRankField(DM, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexCreateLabelField(DM, DMLabel, Vec *);

PETSC_EXTERN PetscErrorCode DMPlexGetDepth(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetDepthLabel(DM, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexGetDepthStratum(DM, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetHeightStratum(DM, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetPointDepth(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetPointHeight(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetCellTypeLabel(DM, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexGetCellType(DM, PetscInt, DMPolytopeType *);
PETSC_EXTERN PetscErrorCode DMPlexSetCellType(DM, PetscInt, DMPolytopeType);
PETSC_EXTERN PetscErrorCode DMPlexComputeCellTypes(DM);
PETSC_EXTERN PetscErrorCode DMPlexInvertCell(DMPolytopeType, PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexReorderCell(DM, PetscInt, PetscInt[]);

/* Topological Operations */
PETSC_EXTERN PetscErrorCode DMPlexGetMeet(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetFullMeet(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreMeet(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetJoin(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetFullJoin(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreJoin(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetTransitiveClosure(DM, PetscInt, PetscBool, PetscInt *, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexRestoreTransitiveClosure(DM, PetscInt, PetscBool, PetscInt *, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexGetCompressedClosure(DM, PetscSection, PetscInt, PetscInt *, PetscInt **, PetscSection *, IS *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreCompressedClosure(DM, PetscSection, PetscInt, PetscInt *, PetscInt **, PetscSection *, IS *, const PetscInt **);

/* Mesh Generation */
PETSC_EXTERN PetscErrorCode DMPlexGenerate(DM, const char [], PetscBool , DM *);
PETSC_EXTERN PetscErrorCode DMPlexGenerateRegister(const char[],PetscErrorCode (*)(DM,PetscBool,DM*),PetscErrorCode (*)(DM,PetscReal*,DM*),PetscErrorCode (*)(DM,DMLabel,DM*),PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGenerateRegisterAll(void);
PETSC_EXTERN PetscErrorCode DMPlexGenerateRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode DMPlexCopyCoordinates(DM, DM);
PETSC_EXTERN PetscErrorCode DMPlexCreateCoordinateSpace(DM, PetscInt, void (*)(PetscInt, PetscInt, PetscInt,
                                                        const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                        const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                        PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]));
PETSC_EXTERN PetscErrorCode DMPlexCreateDoublet(MPI_Comm, PetscInt, PetscBool, PetscBool, PetscReal, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateBoxMesh(MPI_Comm, PetscInt, PetscBool, const PetscInt[], const PetscReal[], const PetscReal[], const DMBoundaryType[], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateBoxSurfaceMesh(MPI_Comm, PetscInt, const PetscInt[], const PetscReal[], const PetscReal[], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateSphereMesh(MPI_Comm, PetscInt, PetscBool, PetscReal, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateBallMesh(MPI_Comm, PetscInt, PetscReal, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateHexCylinderMesh(MPI_Comm, DMBoundaryType, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateWedgeCylinderMesh(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateWedgeBoxMesh(MPI_Comm, const PetscInt[], const PetscReal[], const PetscReal[], const DMBoundaryType[], PetscBool, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexExtrude(DM, PetscInt, PetscReal, PetscBool, PetscBool, const PetscReal[], const PetscReal[], DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateConeSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMPlexInflateToGeomModel(DM);

PETSC_EXTERN PetscErrorCode DMPlexCheckSymmetry(DM);
PETSC_EXTERN PetscErrorCode DMPlexCheckSkeleton(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexCheckFaces(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexCheckGeometry(DM);
PETSC_EXTERN PetscErrorCode DMPlexCheckPointSF(DM);
PETSC_EXTERN PetscErrorCode DMPlexCheckInterfaceCones(DM);
PETSC_EXTERN PetscErrorCode DMPlexCheckCellShape(DM, PetscBool, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexComputeOrthogonalQuality(DM, PetscFV, PetscReal, Vec *, DMLabel *);

PETSC_EXTERN PetscErrorCode DMPlexTriangleSetOptions(DM, const char *);
PETSC_EXTERN PetscErrorCode DMPlexTetgenSetOptions(DM, const char *);

PETSC_EXTERN PetscErrorCode DMPlexCreateFromFile(MPI_Comm, const char[], const char[], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateExodus(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateExodusFromFile(MPI_Comm, const char [], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateCGNS(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateCGNSFromFile(MPI_Comm, const char[], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateGmsh(MPI_Comm, PetscViewer, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateGmshFromFile(MPI_Comm, const char [], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateFluent(MPI_Comm, PetscViewer, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateFluentFromFile(MPI_Comm, const char [], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateMedFromFile(MPI_Comm, const char [], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreatePLYFromFile(MPI_Comm, const char [], PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateEGADSFromFile(MPI_Comm, const char [], DM *);
PETSC_EXTERN PetscErrorCode DMPlexCreateEGADSLiteFromFile(MPI_Comm, const char [], DM *);

PETSC_EXTERN PetscErrorCode PetscViewerExodusIIOpen(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *exo);
PETSC_EXTERN PetscErrorCode PetscViewerExodusIIGetId(PetscViewer, int *);
PETSC_EXTERN PetscErrorCode PetscViewerExodusIISetOrder(PetscViewer,PetscInt);
PETSC_EXTERN PetscErrorCode PetscViewerExodusIIGetOrder(PetscViewer,PetscInt*);

/* Mesh Partitioning and Distribution */
PETSC_EXTERN PetscErrorCode DMPlexCreateNeighborCSR(DM, PetscInt, PetscInt *, PetscInt **, PetscInt **);
PETSC_EXTERN PetscErrorCode DMPlexGetPartitioner(DM, PetscPartitioner *);
PETSC_EXTERN PetscErrorCode DMPlexSetPartitioner(DM, PetscPartitioner);
PETSC_EXTERN PetscErrorCode DMPlexCreatePartitionerGraph(DM, PetscInt, PetscInt *, PetscInt **, PetscInt **, IS *);
PETSC_EXTERN PetscErrorCode DMPlexPartitionLabelInvert(DM, DMLabel, PetscSF, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexPartitionLabelClosure(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexPartitionLabelAdjacency(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexPartitionLabelPropagate(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexPartitionLabelCreateSF(DM, DMLabel, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexSetPartitionBalance(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetPartitionBalance(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexIsDistributed(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexDistribute(DM, PetscInt, PetscSF*, DM*);
PETSC_EXTERN PetscErrorCode DMPlexDistributeOverlap(DM, PetscInt, PetscSF *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexGetOverlap(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexDistributeField(DM,PetscSF,PetscSection,Vec,PetscSection,Vec);
PETSC_EXTERN PetscErrorCode DMPlexDistributeFieldIS(DM, PetscSF, PetscSection, IS, PetscSection, IS *);
PETSC_EXTERN PetscErrorCode DMPlexDistributeData(DM,PetscSF,PetscSection,MPI_Datatype,void*,PetscSection,void**);
PETSC_EXTERN PetscErrorCode DMPlexRebalanceSharedPoints(DM, PetscInt, PetscBool, PetscBool, PetscBool*);
PETSC_EXTERN PetscErrorCode DMPlexMigrate(DM, PetscSF, DM);
PETSC_EXTERN PetscErrorCode DMPlexGetGatherDM(DM, PetscSF*, DM*);
PETSC_EXTERN PetscErrorCode DMPlexGetRedundantDM(DM, PetscSF*, DM*);
PETSC_EXTERN PetscErrorCode DMPlexSetAdjacencyUser(DM, PetscErrorCode (*)(DM,PetscInt,PetscInt*,PetscInt[],void*),void*);
PETSC_EXTERN PetscErrorCode DMPlexGetAdjacencyUser(DM, PetscErrorCode (**)(DM,PetscInt,PetscInt*,PetscInt[],void*),void**);
PETSC_EXTERN PetscErrorCode DMPlexSetAdjacencyUseAnchors(DM,PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetAdjacencyUseAnchors(DM,PetscBool*);
PETSC_EXTERN PetscErrorCode DMPlexGetAdjacency(DM, PetscInt, PetscInt *, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMPlexSetMigrationSF(DM, PetscSF);
PETSC_EXTERN PetscErrorCode DMPlexGetMigrationSF(DM, PetscSF *);

PETSC_EXTERN PetscErrorCode DMPlexGetOrdering(DM, MatOrderingType, DMLabel, IS *);
PETSC_EXTERN PetscErrorCode DMPlexPermute(DM, IS, DM *);

PETSC_EXTERN PetscErrorCode DMPlexCreateProcessSF(DM, PetscSF, IS *, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexCreateTwoSidedProcessSF(DM, PetscSF, PetscSection, IS, PetscSection, IS, IS *, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexDistributeOwnership(DM, PetscSection, IS *, PetscSection, IS *);
PETSC_EXTERN PetscErrorCode DMPlexCreateOverlapLabel(DM, PetscInt, PetscSection, IS, PetscSection, IS, DMLabel *);
PETSC_EXTERN PetscErrorCode DMPlexCreateOverlapMigrationSF(DM, PetscSF, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexStratifyMigrationSF(DM, PetscSF, PetscSF *);

/* Submesh Support */
PETSC_EXTERN PetscErrorCode DMPlexCreateSubmesh(DM, DMLabel, PetscInt, PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMPlexCreateHybridMesh(DM, DMLabel, DMLabel, DMLabel *, DMLabel *, DM *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexGetSubpointMap(DM, DMLabel*);
PETSC_EXTERN PetscErrorCode DMPlexSetSubpointMap(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexGetSubpointIS(DM, IS *);

PETSC_EXTERN PetscErrorCode DMGetEnclosureRelation(DM, DM, DMEnclosureType *);
PETSC_EXTERN PetscErrorCode DMGetEnclosurePoint(DM, DM, DMEnclosureType, PetscInt, PetscInt *);

PETSC_EXTERN PetscErrorCode DMPlexLabelComplete(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexLabelCohesiveComplete(DM, DMLabel, DMLabel, PetscBool, DM);
PETSC_EXTERN PetscErrorCode DMPlexLabelAddCells(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexLabelAddFaceCells(DM, DMLabel);
PETSC_EXTERN PetscErrorCode DMPlexLabelClearCells(DM, DMLabel);

PETSC_EXTERN PetscErrorCode DMPlexGetRefinementLimit(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexSetRefinementLimit(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexGetRefinementUniform(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexSetRefinementUniform(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexGetRefinementFunction(DM, PetscErrorCode (**)(const PetscReal [], PetscReal *));
PETSC_EXTERN PetscErrorCode DMPlexSetRefinementFunction(DM, PetscErrorCode (*)(const PetscReal [], PetscReal *));
PETSC_EXTERN PetscErrorCode DMPlexCreateCoarsePointIS(DM, IS *);
PETSC_EXTERN PetscErrorCode DMPlexGetRegularRefinement(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexSetRegularRefinement(DM, PetscBool);

/* Support for cell-vertex meshes */
PETSC_EXTERN PetscErrorCode DMPlexGetNumFaceVertices(DM, PetscInt, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetOrientedFace(DM, PetscInt, PetscInt, const PetscInt [], PetscInt, PetscInt [], PetscInt [], PetscInt [], PetscBool *);

/* Geometry Support */
PETSC_EXTERN PetscErrorCode DMPlexGetMinRadius(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexSetMinRadius(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexComputeProjection2Dto1D(PetscScalar[], PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexComputeProjection3Dto1D(PetscScalar[], PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexComputeProjection3Dto2D(PetscInt, PetscScalar[], PetscReal[]);

/* Point Location */
typedef struct _PetscGridHash *PetscGridHash;
PETSC_EXTERN PetscErrorCode PetscGridHashCreate(MPI_Comm, PetscInt, const PetscScalar[], PetscGridHash *);
PETSC_EXTERN PetscErrorCode PetscGridHashEnlarge(PetscGridHash, const PetscScalar []);
PETSC_EXTERN PetscErrorCode PetscGridHashSetGrid(PetscGridHash, const PetscInt [], const PetscReal []);
PETSC_EXTERN PetscErrorCode PetscGridHashGetEnclosingBox(PetscGridHash, PetscInt, const PetscScalar [], PetscInt [], PetscInt []);
PETSC_EXTERN PetscErrorCode PetscGridHashDestroy(PetscGridHash *);
PETSC_EXTERN PetscErrorCode DMPlexFindVertices(DM, PetscInt, const PetscReal [], PetscReal, PetscInt []);

/* FVM Support */
PETSC_EXTERN PetscErrorCode DMPlexComputeCellGeometryFVM(DM, PetscInt, PetscReal *, PetscReal [], PetscReal []);
PETSC_EXTERN PetscErrorCode DMPlexComputeGeometryFVM(DM, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexComputeGradientFVM(DM, PetscFV, Vec, Vec, DM *);
PETSC_EXTERN PetscErrorCode DMPlexGetDataFVM(DM, PetscFV, Vec *, Vec *, DM *);

/* FEM Support */
PETSC_EXTERN PetscErrorCode DMPlexComputeGeometryFEM(DM, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexGetGeometryFVM(DM,Vec*,Vec*,PetscReal*);
PETSC_EXTERN PetscErrorCode DMPlexGetGradientDM(DM,PetscFV,DM*);
PETSC_EXTERN PetscErrorCode DMPlexInsertBoundaryValues(DM, PetscBool, Vec, PetscReal, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexInsertTimeDerivativeBoundaryValues(DM, PetscBool, Vec, PetscReal, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexInsertBoundaryValuesEssential(DM, PetscReal, PetscInt, PetscInt, const PetscInt[], DMLabel, PetscInt, const PetscInt[],
                                                                PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void *, Vec);
PETSC_EXTERN PetscErrorCode DMPlexInsertBoundaryValuesEssentialField(DM, PetscReal, Vec, PetscInt, PetscInt, const PetscInt[], DMLabel, PetscInt, const PetscInt [],
                                                                     void (*)(PetscInt, PetscInt, PetscInt,
                                                                              const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                              const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                              PetscReal, const PetscReal[], PetscInt, const PetscScalar[],
                                                                              PetscScalar[]),
                                                                     void *, Vec);
PETSC_EXTERN PetscErrorCode DMPlexInsertBoundaryValuesEssentialBdField(DM, PetscReal, Vec, PetscInt, PetscInt, const PetscInt[], DMLabel, PetscInt, const PetscInt [],
                                                                       void (*)(PetscInt, PetscInt, PetscInt,
                                                                                const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                                const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                                PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[],
                                                                                PetscScalar[]),
                                                                       void *, Vec);
PETSC_EXTERN PetscErrorCode DMPlexInsertBoundaryValuesRiemann(DM, PetscReal, Vec, Vec, Vec, PetscInt, PetscInt, const PetscInt[], DMLabel, PetscInt, const PetscInt [],
                                                              PetscErrorCode (*)(PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*), void *, Vec);
PETSC_EXTERN PetscErrorCode DMPlexMarkBoundaryFaces(DM, PetscInt, DMLabel);

PETSC_EXTERN PetscErrorCode DMPlexCreateSection(DM, DMLabel [], const PetscInt [], const PetscInt [], PetscInt, const PetscInt [], const IS [], const IS [], IS, PetscSection *);
PETSC_EXTERN PetscErrorCode DMPlexGetSubdomainSection(DM, PetscSection*);

PETSC_EXTERN PetscErrorCode DMPlexComputeCellGeometryAffineFEM(DM, PetscInt, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexComputeCellGeometryFEM(DM, PetscInt, PetscQuadrature, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexCoordinatesToReference(DM, PetscInt, PetscInt, const PetscReal [], PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexReferenceToCoordinates(DM, PetscInt, PetscInt, const PetscReal [], PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexShearGeometry(DM, DMDirection, PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexRemapGeometry(DM, PetscReal,
                                                void (*)(PetscInt, PetscInt, PetscInt,
                                                         const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                         const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                         PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]));

PETSC_EXTERN PetscErrorCode DMPlexVecGetClosure(DM, PetscSection, Vec, PetscInt, PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexVecRestoreClosure(DM, PetscSection, Vec, PetscInt, PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexVecSetClosure(DM, PetscSection, Vec, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexMatSetClosure(DM, PetscSection, PetscSection, Mat, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexMatSetClosureGeneral(DM, PetscSection, PetscSection, DM, PetscSection, PetscSection, Mat, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexGetClosureIndices(DM, PetscSection, PetscSection, PetscInt, PetscBool, PetscInt *, PetscInt *[], PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexRestoreClosureIndices(DM, PetscSection, PetscSection, PetscInt, PetscBool, PetscInt *, PetscInt *[], PetscInt *, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode DMPlexMatSetClosureRefined(DM, PetscSection, PetscSection, DM, PetscSection, PetscSection, Mat, PetscInt, const PetscScalar[], InsertMode);
PETSC_EXTERN PetscErrorCode DMPlexMatGetClosureIndicesRefined(DM, PetscSection, PetscSection, DM, PetscSection, PetscSection, PetscInt, PetscInt[], PetscInt[]);
PETSC_EXTERN PetscErrorCode DMPlexCreateClosureIndex(DM, PetscSection);
PETSC_EXTERN PetscErrorCode DMPlexSetClosurePermutationTensor(DM, PetscInt, PetscSection);

PETSC_EXTERN PetscErrorCode DMPlexConstructGhostCells(DM, const char [], PetscInt *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexConstructCohesiveCells(DM, DMLabel, DMLabel, DM *);

PETSC_EXTERN PetscErrorCode DMPlexGetVTKCellHeight(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexSetVTKCellHeight(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexVTKWriteAll(PetscObject, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexGetGhostCellStratum(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexGetSimplexOrBoxCells(DM, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMPlexIsSimplex(DM, PetscBool *);

PETSC_EXTERN PetscErrorCode DMPlexGetCellFields(DM, IS, Vec, Vec, Vec, PetscScalar **, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreCellFields(DM, IS, Vec, Vec, Vec, PetscScalar **, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode DMPlexGetFaceFields(DM, PetscInt, PetscInt, Vec, Vec, Vec, Vec, Vec, PetscInt *, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreFaceFields(DM, PetscInt, PetscInt, Vec, Vec, Vec, Vec, Vec, PetscInt *, PetscScalar **, PetscScalar **);
PETSC_EXTERN PetscErrorCode DMPlexGetFaceGeometry(DM, PetscInt, PetscInt, Vec, Vec, PetscInt *, PetscFVFaceGeom **, PetscReal **);
PETSC_EXTERN PetscErrorCode DMPlexRestoreFaceGeometry(DM, PetscInt, PetscInt, Vec, Vec, PetscInt *, PetscFVFaceGeom **, PetscReal **);

PETSC_EXTERN PetscErrorCode DMPlexGetScale(DM, PetscUnit, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexSetScale(DM, PetscUnit, PetscReal);

typedef struct {
  DM    dm;
  Vec   u; /* The base vector for the Jacobian action J(u) x */
  Mat   J; /* Preconditioner for testing */
  void *user;
} JacActionCtx;

PETSC_EXTERN PetscErrorCode DMPlexSetMaxProjectionHeight(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexGetMaxProjectionHeight(DM, PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexGetActivePoint(DM, PetscInt*);
PETSC_EXTERN PetscErrorCode DMPlexSetActivePoint(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexProjectFieldLocal(DM, Vec,
                                                    void (**)(PetscInt, PetscInt, PetscInt,
                                                              const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                              const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                              PetscReal, const PetscReal [], PetscScalar []), InsertMode, Vec);
PETSC_EXTERN PetscErrorCode DMPlexComputeL2DiffLocal(DM, PetscReal, PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexComputeL2FieldDiff(DM, PetscReal, PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **, Vec, PetscReal[]);
PETSC_EXTERN PetscErrorCode DMPlexComputeL2DiffVec(DM, PetscReal, PetscErrorCode (**)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexComputeCellwiseIntegralFEM(DM, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeIntegralFEM(DM, Vec, PetscScalar *, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeBdIntegral(DM, Vec, DMLabel, PetscInt, const PetscInt[],
                                                    void (*)(PetscInt, PetscInt, PetscInt,
                                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                             PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),
                                                    PetscScalar *, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeInterpolatorNested(DM, DM, PetscBool, Mat, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeInterpolatorGeneral(DM, DM, Mat, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeGradientClementInterpolant(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexComputeInjectorFEM(DM, DM, VecScatter *, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeMassMatrixNested(DM, DM, Mat, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeMassMatrixGeneral(DM, DM, Mat, void *);

PETSC_EXTERN PetscErrorCode DMPlexCreateRigidBody(DM, PetscInt, MatNullSpace *);
PETSC_EXTERN PetscErrorCode DMPlexCreateRigidBodies(DM, PetscInt, DMLabel, const PetscInt[], const PetscInt[], MatNullSpace *);

PETSC_EXTERN PetscErrorCode DMPlexSetSNESLocalFEM(DM,void *,void *,void *);
PETSC_EXTERN PetscErrorCode DMPlexSNESComputeBoundaryFEM(DM, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexSNESComputeResidualFEM(DM, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexSNESComputeJacobianFEM(DM, Vec, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode DMPlexComputeBdResidualSingle(DM, PetscReal, PetscWeakForm, PetscFormKey, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexComputeBdJacobianSingle(DM, PetscReal, PetscWeakForm, DMLabel, PetscInt, const PetscInt[], PetscInt, Vec, Vec, PetscReal, Mat, Mat);

PETSC_EXTERN PetscErrorCode DMPlexTSComputeBoundary(DM, PetscReal, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexTSComputeRHSFunctionFVM(DM, PetscReal, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexTSComputeIFunctionFEM(DM, PetscReal, Vec, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexTSComputeIJacobianFEM(DM, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);

PETSC_EXTERN PetscErrorCode DMPlexComputeRHSFunctionFVM(DM, PetscReal, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexReconstructGradientsFVM(DM,Vec,Vec);

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
PETSC_EXTERN PetscErrorCode DMPlexComputeInjectorReferenceTree(DM, Mat *);
PETSC_EXTERN PetscErrorCode DMPlexTransferVecTree(DM,Vec,DM,Vec,PetscSF,PetscSF,PetscInt *,PetscInt *,PetscBool,PetscReal);

PETSC_EXTERN PetscErrorCode DMPlexMonitorThroughput(DM, void *);

/* natural order */
PETSC_EXTERN PetscErrorCode DMPlexCreateGlobalToNaturalSF(DM, PetscSection, PetscSF, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexSetGlobalToNaturalSF(DM, PetscSF);
PETSC_EXTERN PetscErrorCode DMPlexGetGlobalToNaturalSF(DM, PetscSF *);
PETSC_EXTERN PetscErrorCode DMPlexGlobalToNaturalBegin(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexGlobalToNaturalEnd(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexNaturalToGlobalBegin(DM, Vec, Vec);
PETSC_EXTERN PetscErrorCode DMPlexNaturalToGlobalEnd(DM, Vec, Vec);

/* mesh adaptation */
PETSC_EXTERN PetscErrorCode DMPlexAdapt(DM, Vec, const char [], DM *);
PETSC_EXTERN PetscErrorCode DMPlexSnapToGeomModel(DM, PetscInt, PetscInt, const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode DMPlexMetricSetIsotropic(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexMetricIsIsotropic(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexMetricSetRestrictAnisotropyFirst(DM, PetscBool);
PETSC_EXTERN PetscErrorCode DMPlexMetricRestrictAnisotropyFirst(DM, PetscBool *);
PETSC_EXTERN PetscErrorCode DMPlexMetricSetMinimumMagnitude(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexMetricGetMinimumMagnitude(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexMetricSetMaximumMagnitude(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexMetricGetMaximumMagnitude(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexMetricSetMaximumAnisotropy(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexMetricGetMaximumAnisotropy(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexMetricSetTargetComplexity(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexMetricGetTargetComplexity(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexMetricSetNormalizationOrder(DM, PetscReal);
PETSC_EXTERN PetscErrorCode DMPlexMetricGetNormalizationOrder(DM, PetscReal *);
PETSC_EXTERN PetscErrorCode DMPlexMetricCreate(DM, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexMetricCreateUniform(DM, PetscInt, PetscReal, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexMetricCreateIsotropic(DM, PetscInt, Vec, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexMetricEnforceSPD(DM, PetscBool, PetscBool, Vec);
PETSC_EXTERN PetscErrorCode DMPlexMetricNormalize(DM, Vec, PetscBool, PetscBool, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexMetricAverage(DM, PetscInt, PetscReal[], Vec[], Vec *);
PETSC_EXTERN PetscErrorCode DMPlexMetricAverage2(DM, Vec, Vec, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexMetricAverage3(DM, Vec, Vec, Vec, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexMetricIntersection(DM, PetscInt, Vec[], Vec *);
PETSC_EXTERN PetscErrorCode DMPlexMetricIntersection2(DM, Vec, Vec, Vec *);
PETSC_EXTERN PetscErrorCode DMPlexMetricIntersection3(DM, Vec, Vec, Vec, Vec *);

PETSC_EXTERN PetscErrorCode DMPlexGlobalToLocalBasis(DM, Vec);
PETSC_EXTERN PetscErrorCode DMPlexLocalToGlobalBasis(DM, Vec);
PETSC_EXTERN PetscErrorCode DMPlexCreateBasisRotation(DM, PetscReal, PetscReal, PetscReal);

/* storage version */
#define DMPLEX_STORAGE_VERSION_STABLE "1.0.0"
#define DMPLEX_STORAGE_VERSION_LATEST "2.0.0"

PETSC_EXTERN PetscErrorCode DMPlexTopologyView(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexCoordinatesView(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexLabelsView(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexSectionView(DM,PetscViewer,DM);
PETSC_EXTERN PetscErrorCode DMPlexGlobalVectorView(DM,PetscViewer,DM,Vec);
PETSC_EXTERN PetscErrorCode DMPlexLocalVectorView(DM,PetscViewer,DM,Vec);

PETSC_EXTERN PetscErrorCode DMPlexTopologyLoad(DM, PetscViewer, PetscSF*);
PETSC_EXTERN PetscErrorCode DMPlexCoordinatesLoad(DM, PetscViewer, PetscSF);
PETSC_EXTERN PetscErrorCode DMPlexLabelsLoad(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMPlexSectionLoad(DM,PetscViewer,DM,PetscSF,PetscSF*,PetscSF*);
PETSC_EXTERN PetscErrorCode DMPlexGlobalVectorLoad(DM,PetscViewer,DM,PetscSF,Vec);
PETSC_EXTERN PetscErrorCode DMPlexLocalVectorLoad(DM,PetscViewer,DM,PetscSF,Vec);

#endif
