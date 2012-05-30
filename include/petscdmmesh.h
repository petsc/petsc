/*
  DMMesh, for easy parallelism of simple unstructured distributed mesh problems.
*/
#if !defined(__PETSCDMMESH_H)
#define __PETSCDMMESH_H

#include <petscsf.h>
#include <petscdm.h>

#if defined(PETSC_HAVE_SIEVE) && defined(__cplusplus)

#include <sieve/Mesh.hh>
#include <sieve/CartesianSieve.hh>
#include <sieve/Distribution.hh>
#include <sieve/Generator.hh>

/*S
  DMMESH - DM object that encapsulates an unstructured mesh. This uses the Sieve package to represent the mesh.

  Level: intermediate

  Concepts: grids, grid refinement

.seealso:  DM, DMMeshCreate()
S*/
/*S
  DMCARTESIAN - DM object that encapsulates a structured, Cartesian mesh in any dimension. This is currently a
  replacement for DMDA in dimensions greater than 3.

  Level: intermediate

  Concepts: grids, grid refinement

.seealso:  DM, DMCartesianCreate(), DMDA
S*/
PETSC_EXTERN PetscLogEvent DMMesh_View, DMMesh_GetGlobalScatter, DMMesh_restrictVector, DMMesh_assembleVector, DMMesh_assembleVectorComplete, DMMesh_assembleMatrix, DMMesh_updateOperator;

PETSC_EXTERN PetscErrorCode DMMeshCreate(MPI_Comm, DM*);
PETSC_EXTERN PetscErrorCode DMMeshCreateMeshFromAdjacency(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt [], PetscInt, PetscInt, const PetscReal [], PetscBool, DM*);
PETSC_EXTERN PetscErrorCode DMMeshGetMesh(DM, ALE::Obj<PETSC_MESH_TYPE>&);
PETSC_EXTERN PetscErrorCode DMMeshSetMesh(DM, const ALE::Obj<PETSC_MESH_TYPE>&);
PETSC_EXTERN PetscErrorCode DMMeshGetGlobalScatter(DM, VecScatter *);
PETSC_EXTERN PetscErrorCode DMMeshFinalize();

/* New Sieve Mesh interface */
PETSC_EXTERN PetscErrorCode DMMeshGetDimension(DM, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshSetDimension(DM, PetscInt);
PETSC_EXTERN PetscErrorCode DMMeshGetChart(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshSetChart(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMMeshGetConeSize(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshSetConeSize(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMMeshGetCone(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMMeshSetCone(DM, PetscInt, const PetscInt[]);
PETSC_EXTERN PetscErrorCode DMMeshGetSupportSize(DM, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshGetSupport(DM, PetscInt, const PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMMeshGetConeSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMMeshGetCones(DM, PetscInt *[]);
PETSC_EXTERN PetscErrorCode DMMeshGetMaxSizes(DM, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshSetUp(DM);
PETSC_EXTERN PetscErrorCode DMMeshSymmetrize(DM);
PETSC_EXTERN PetscErrorCode DMMeshStratify(DM);

PETSC_EXTERN PetscErrorCode DMMeshGetLabelValue(DM, const char[], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshSetLabelValue(DM, const char[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMMeshGetLabelSize(DM, const char[], PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshGetLabelIdIS(DM, const char[], IS *);
PETSC_EXTERN PetscErrorCode DMMeshGetStratumSize(DM, const char [], PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshGetStratumIS(DM, const char [], PetscInt, IS *);

PETSC_EXTERN PetscErrorCode DMMeshJoinPoints(DM, PetscInt, const PetscInt [], PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshMeetPoints(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
PETSC_EXTERN PetscErrorCode DMMeshGetTransitiveClosure(DM, PetscInt, PetscBool, PetscInt *, const PetscInt *[]);

PETSC_EXTERN PetscErrorCode DMMeshCreatePartition(DM, PetscSection *, IS *, PetscInt);
PETSC_EXTERN PetscErrorCode DMMeshCreatePartitionClosure(DM, PetscSection, IS, PetscSection *, IS *);

/* Old Sieve Mesh interface */
PETSC_EXTERN PetscErrorCode DMMeshDistribute(DM, const char[], DM*);
PETSC_EXTERN PetscErrorCode DMMeshDistributeByFace(DM, const char[], DM*);
PETSC_EXTERN PetscErrorCode DMMeshGenerate(DM, PetscBool , DM *);
PETSC_EXTERN PetscErrorCode DMMeshRefine(DM, double, PetscBool , DM*);
PETSC_EXTERN PetscErrorCode DMMeshLoad(PetscViewer, DM);
PETSC_EXTERN PetscErrorCode DMMeshGetMaximumDegree(DM, PetscInt *);

PETSC_EXTERN PetscErrorCode DMCartesianCreate(MPI_Comm, DM *);
PETSC_EXTERN PetscErrorCode DMMeshCartesianGetMesh(DM, ALE::Obj<ALE::CartesianMesh>&);
PETSC_EXTERN PetscErrorCode DMMeshCartesianSetMesh(DM, const ALE::Obj<ALE::CartesianMesh>&);

PETSC_EXTERN PetscErrorCode DMMeshGetCoordinates(DM, PetscBool , PetscInt *, PetscInt *, PetscReal *[]);
PETSC_EXTERN PetscErrorCode DMMeshGetElements(DM, PetscBool , PetscInt *, PetscInt *, PetscInt *[]);

PETSC_EXTERN PetscErrorCode DMMeshCreateBoxMesh(MPI_Comm, PetscInt, PetscBool, DM *);
PETSC_EXTERN PetscErrorCode DMMeshMarkBoundaryCells(DM, const char [], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMMeshGetDepthStratum(DM, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshGetHeightStratum(DM, PetscInt, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshCreateSection(DM, PetscInt, PetscInt, PetscInt [], PetscInt [], PetscInt, PetscInt [], IS [], PetscSection *);
PETSC_EXTERN PetscErrorCode DMMeshConvertSection(const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, PetscSection *);
PETSC_EXTERN PetscErrorCode DMMeshGetSection(DM, const char [], PetscSection *);
PETSC_EXTERN PetscErrorCode DMMeshSetSection(DM, const char [], PetscSection);
PETSC_EXTERN PetscErrorCode DMMeshGetDefaultSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMMeshSetDefaultSection(DM, PetscSection);
PETSC_EXTERN PetscErrorCode DMMeshGetCoordinateSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMMeshSetCoordinateSection(DM, PetscSection);
PETSC_EXTERN PetscErrorCode DMMeshCreateConeSection(DM, PetscSection *);
PETSC_EXTERN PetscErrorCode DMMeshGetCoordinateVec(DM, Vec *);
PETSC_EXTERN PetscErrorCode DMMeshComputeCellGeometry(DM, PetscInt, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMMeshVecSetClosure(DM, Vec, PetscInt, const PetscScalar [], InsertMode);
PETSC_EXTERN PetscErrorCode DMMeshVecGetClosure(DM, Vec, PetscInt, const PetscScalar **);
PETSC_EXTERN PetscErrorCode DMMeshMatSetClosure(DM, Mat, PetscInt, PetscScalar [], InsertMode);

PETSC_EXTERN PetscErrorCode MatSetValuesTopology(Mat, DM, PetscInt, const PetscInt [], DM, PetscInt, const PetscInt [], const PetscScalar [], InsertMode);
PETSC_EXTERN PetscErrorCode DMMeshRestrictVector(Vec, Vec, InsertMode);
PETSC_EXTERN PetscErrorCode DMMeshAssembleVectorComplete(Vec, Vec, InsertMode);
PETSC_EXTERN PetscErrorCode DMMeshAssembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
PETSC_EXTERN PetscErrorCode DMMeshUpdateOperator(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);
PETSC_EXTERN PetscErrorCode DMMeshUpdateOperatorGeneral(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);

/*S
  SectionReal - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionRealCreate(), SectionRealDestroy(), Mesh, DMMeshCreate()
S*/
typedef struct _p_SectionReal* SectionReal;

/* Logging support */
PETSC_EXTERN PetscClassId SECTIONREAL_CLASSID;

PETSC_EXTERN PetscErrorCode SectionRealCreate(MPI_Comm,SectionReal*);
PETSC_EXTERN PetscErrorCode SectionRealDestroy(SectionReal*);
PETSC_EXTERN PetscErrorCode SectionRealView(SectionReal,PetscViewer);
PETSC_EXTERN PetscErrorCode SectionRealDuplicate(SectionReal,SectionReal*);

PETSC_EXTERN PetscErrorCode SectionRealGetSection(SectionReal,ALE::Obj<PETSC_MESH_TYPE::real_section_type>&);
PETSC_EXTERN PetscErrorCode SectionRealSetSection(SectionReal,const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&);
PETSC_EXTERN PetscErrorCode SectionRealGetBundle(SectionReal,ALE::Obj<PETSC_MESH_TYPE>&);
PETSC_EXTERN PetscErrorCode SectionRealSetBundle(SectionReal,const ALE::Obj<PETSC_MESH_TYPE>&);

PETSC_EXTERN PetscErrorCode SectionRealDistribute(SectionReal, DM, SectionReal *);
PETSC_EXTERN PetscErrorCode SectionRealRestrict(SectionReal, PetscInt, PetscScalar *[]);
PETSC_EXTERN PetscErrorCode SectionRealUpdate(SectionReal, PetscInt, const PetscScalar [], InsertMode);
PETSC_EXTERN PetscErrorCode SectionRealZero(SectionReal);
PETSC_EXTERN PetscErrorCode SectionRealCreateLocalVector(SectionReal, Vec*);
PETSC_EXTERN PetscErrorCode SectionRealAddSpace(SectionReal);
PETSC_EXTERN PetscErrorCode SectionRealGetFibration(SectionReal, const PetscInt, SectionReal *);
PETSC_EXTERN PetscErrorCode SectionRealToVec(SectionReal, DM, ScatterMode, Vec);
PETSC_EXTERN PetscErrorCode SectionRealToVec(SectionReal, VecScatter, ScatterMode, Vec);
PETSC_EXTERN PetscErrorCode SectionRealNorm(SectionReal, DM, NormType, PetscReal *);
PETSC_EXTERN PetscErrorCode SectionRealAXPY(SectionReal, DM, PetscScalar, SectionReal);
PETSC_EXTERN PetscErrorCode SectionRealComplete(SectionReal);
PETSC_EXTERN PetscErrorCode SectionRealSet(SectionReal, PetscReal);
PETSC_EXTERN PetscErrorCode SectionRealGetFiberDimension(SectionReal, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode SectionRealSetFiberDimension(SectionReal, PetscInt, const PetscInt);
PETSC_EXTERN PetscErrorCode SectionRealSetFiberDimensionField(SectionReal, PetscInt, const PetscInt, const PetscInt);
PETSC_EXTERN PetscErrorCode SectionRealGetSize(SectionReal, PetscInt *);
PETSC_EXTERN PetscErrorCode SectionRealAllocate(SectionReal);
PETSC_EXTERN PetscErrorCode SectionRealClear(SectionReal);

PETSC_EXTERN PetscErrorCode SectionRealRestrictClosure(SectionReal, DM, PetscInt, PetscInt, PetscScalar []);
PETSC_EXTERN PetscErrorCode SectionRealRestrictClosure(SectionReal, DM, PetscInt, const PetscScalar *[]);
PETSC_EXTERN PetscErrorCode SectionRealUpdateClosure(SectionReal, DM, PetscInt, PetscScalar [], InsertMode);

PETSC_EXTERN PetscErrorCode DMMeshHasSectionReal(DM, const char [], PetscBool  *);
PETSC_EXTERN PetscErrorCode DMMeshGetSectionReal(DM, const char [], SectionReal *);
PETSC_EXTERN PetscErrorCode DMMeshSetSectionReal(DM, const char [], SectionReal);
PETSC_EXTERN PetscErrorCode DMMeshCreateVector(DM, SectionReal, Vec *);
PETSC_EXTERN PetscErrorCode DMMeshCreateMatrix(DM, SectionReal, const MatType, Mat *);
PETSC_EXTERN PetscErrorCode DMMeshCreateGlobalScatter(DM, SectionReal, VecScatter *);
PETSC_EXTERN PetscErrorCode DMMeshAssembleVector(Vec, DM, SectionReal, PetscInt, PetscScalar [], InsertMode);
PETSC_EXTERN PetscErrorCode DMMeshAssembleMatrix(Mat, DM, SectionReal, PetscInt, PetscScalar [], InsertMode);
PETSC_EXTERN PetscErrorCode DMMeshSetupSection(DM, SectionReal);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*DMMeshLocalFunction1)(DM, Vec, Vec, void*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*DMMeshLocalJacobian1)(DM, Vec, Mat, void*);

PETSC_EXTERN PetscErrorCode DMMeshCreateGlobalRealVector(DM, SectionReal, Vec *);
PETSC_EXTERN PetscErrorCode DMMeshGetGlobalScatter(DM,VecScatter *);
PETSC_EXTERN PetscErrorCode DMMeshCreateGlobalScatter(DM,SectionReal,VecScatter *);
PETSC_EXTERN PetscErrorCode DMMeshGetLocalFunction(DM, PetscErrorCode (**)(DM, Vec, Vec, void*));
PETSC_EXTERN PetscErrorCode DMMeshSetLocalFunction(DM, PetscErrorCode (*)(DM, Vec, Vec, void*));
PETSC_EXTERN PetscErrorCode DMMeshGetLocalJacobian(DM, PetscErrorCode (**)(DM, Vec, Mat, void*));
PETSC_EXTERN PetscErrorCode DMMeshSetLocalJacobian(DM, PetscErrorCode (*)(DM, Vec, Mat, void*));

/*S
  SectionInt - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionIntCreate(), SectionIntDestroy(), DM, DMMeshCreate()
S*/
typedef struct _p_SectionInt* SectionInt;

/* Logging support */
PETSC_EXTERN PetscClassId SECTIONINT_CLASSID;

PETSC_EXTERN PetscErrorCode SectionIntCreate(MPI_Comm,SectionInt*);
PETSC_EXTERN PetscErrorCode SectionIntDestroy(SectionInt*);
PETSC_EXTERN PetscErrorCode SectionIntView(SectionInt,PetscViewer);

PETSC_EXTERN PetscErrorCode SectionIntGetSection(SectionInt,ALE::Obj<PETSC_MESH_TYPE::int_section_type>&);
PETSC_EXTERN PetscErrorCode SectionIntSetSection(SectionInt,const ALE::Obj<PETSC_MESH_TYPE::int_section_type>&);
PETSC_EXTERN PetscErrorCode SectionIntGetBundle(SectionInt,ALE::Obj<PETSC_MESH_TYPE>&);
PETSC_EXTERN PetscErrorCode SectionIntSetBundle(SectionInt,const ALE::Obj<PETSC_MESH_TYPE>&);

PETSC_EXTERN PetscErrorCode SectionIntDistribute(SectionInt, DM, SectionInt *);
PETSC_EXTERN PetscErrorCode SectionIntRestrict(SectionInt, PetscInt, PetscInt *[]);
PETSC_EXTERN PetscErrorCode SectionIntUpdate(SectionInt, PetscInt, const PetscInt [], InsertMode);
PETSC_EXTERN PetscErrorCode SectionIntZero(SectionInt);
PETSC_EXTERN PetscErrorCode SectionIntComplete(SectionInt);
PETSC_EXTERN PetscErrorCode SectionIntGetFiberDimension(SectionInt, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode SectionIntSetFiberDimension(SectionInt, PetscInt, const PetscInt);
PETSC_EXTERN PetscErrorCode SectionIntSetFiberDimensionField(SectionInt, PetscInt, const PetscInt, const PetscInt);
PETSC_EXTERN PetscErrorCode SectionIntGetSize(SectionInt, PetscInt *);
PETSC_EXTERN PetscErrorCode SectionIntAllocate(SectionInt);
PETSC_EXTERN PetscErrorCode SectionIntClear(SectionInt);

PETSC_EXTERN PetscErrorCode SectionIntAddSpace(SectionInt);
PETSC_EXTERN PetscErrorCode SectionIntGetFibration(SectionInt, const PetscInt, SectionInt *);
PETSC_EXTERN PetscErrorCode SectionIntSet(SectionInt, PetscInt);

PETSC_EXTERN PetscErrorCode SectionIntRestrictClosure(SectionInt, DM, PetscInt, PetscInt, PetscInt []);
PETSC_EXTERN PetscErrorCode SectionIntUpdateClosure(SectionInt, DM, PetscInt, PetscInt [], InsertMode);

PETSC_EXTERN PetscErrorCode DMMeshHasSectionInt(DM, const char [], PetscBool  *);
PETSC_EXTERN PetscErrorCode DMMeshGetSectionInt(DM, const char [], SectionInt *);
PETSC_EXTERN PetscErrorCode DMMeshSetSectionInt(DM, SectionInt);

/* Misc Mesh functions*/
PETSC_EXTERN PetscErrorCode DMMeshSetMaxDof(DM, PetscInt);
PETSC_EXTERN PetscErrorCode SectionGetArray(DM, const char [], PetscInt *, PetscInt *, PetscScalar *[]);

/* Helper functions for simple distributions */
PETSC_EXTERN PetscErrorCode DMMeshGetVertexSectionReal(DM, const char[], PetscInt, SectionReal *);
PETSC_EXTERN PetscErrorCode DMMeshGetVertexSectionInt(DM, const char[], PetscInt, SectionInt *);
PETSC_EXTERN PetscErrorCode DMMeshGetCellSectionReal(DM, const char[], PetscInt, SectionReal *);
PETSC_EXTERN PetscErrorCode DMMeshGetCellSectionInt(DM, const char[], PetscInt, SectionInt *);
PETSC_EXTERN PetscErrorCode DMMeshCreateSectionRealIS(DM,IS,const char [],PetscInt,SectionReal *);

/* Scatter for simple distributions */
PETSC_EXTERN PetscErrorCode DMMeshCreateScatterToZeroVertex(DM,VecScatter *);
PETSC_EXTERN PetscErrorCode DMMeshCreateScatterToZeroVertexSet(DM,IS,IS,VecScatter *);
PETSC_EXTERN PetscErrorCode DMMeshCreateScatterToZeroCell(DM,VecScatter *);
PETSC_EXTERN PetscErrorCode DMMeshCreateScatterToZeroCellSet(DM,IS,IS,VecScatter *);

/* Support for various mesh formats */
PETSC_EXTERN PetscErrorCode DMMeshCreateExodus(MPI_Comm, const char [], DM *);
PETSC_EXTERN PetscErrorCode DMMeshCreateExodusNG(MPI_Comm, PetscInt, DM *);
PETSC_EXTERN PetscErrorCode DMMeshExodusGetInfo(DM, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode DMMeshViewExodusSplit(DM,PetscInt);

PETSC_EXTERN PetscErrorCode VecViewExodusVertex(DM,Vec,MPI_Comm,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode VecLoadExodusVertex(DM,Vec,MPI_Comm,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode VecViewExodusVertexSet(DM,Vec,PetscInt,MPI_Comm,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode VecLoadExodusVertexSet(DM,Vec,PetscInt,MPI_Comm,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode VecViewExodusCell(DM,Vec,MPI_Comm,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode VecLoadExodusCell(DM,Vec,MPI_Comm,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode VecViewExodusCellSet(DM,Vec,PetscInt,MPI_Comm,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode VecLoadExodusCellSet(DM,Vec,PetscInt,MPI_Comm,PetscInt,PetscInt,PetscInt);

PETSC_EXTERN PetscErrorCode DMMeshCreatePCICE(MPI_Comm, const int, const char [], const char [], PetscBool , const char [], DM *);

PETSC_EXTERN PetscErrorCode DMWriteVTKHeader(PetscViewer);
PETSC_EXTERN PetscErrorCode DMWriteVTKVertices(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMWriteVTKElements(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMWritePCICEVertices(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMWritePCICEElements(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMWritePyLithVertices(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMWritePyLithElements(DM, SectionReal, PetscViewer);
PETSC_EXTERN PetscErrorCode DMWritePyLithVerticesLocal(DM, PetscViewer);
PETSC_EXTERN PetscErrorCode DMWritePyLithElementsLocal(DM, SectionReal, PetscViewer);

struct _DMMeshInterpolationInfo {
  PetscInt   dim;    /*1 The spatial dimension of points */
  PetscInt   nInput; /* The number of input points */
  PetscReal *points; /* The input point coordinates */
  PetscInt  *cells;  /* The cell containing each point */
  PetscInt   n;      /* The number of local points */
  Vec        coords; /* The point coordinates */
  PetscInt   dof;    /* The number of components to interpolate */
};
typedef struct _DMMeshInterpolationInfo *DMMeshInterpolationInfo;

PetscErrorCode DMMeshInterpolationCreate(DM dm, DMMeshInterpolationInfo *ctx);
PetscErrorCode DMMeshInterpolationSetDim(DM dm, PetscInt dim, DMMeshInterpolationInfo ctx);
PetscErrorCode DMMeshInterpolationGetDim(DM dm, PetscInt *dim, DMMeshInterpolationInfo ctx);
PetscErrorCode DMMeshInterpolationSetDof(DM dm, PetscInt dof, DMMeshInterpolationInfo ctx);
PetscErrorCode DMMeshInterpolationGetDof(DM dm, PetscInt *dof, DMMeshInterpolationInfo ctx);
PetscErrorCode DMMeshInterpolationAddPoints(DM dm, PetscInt n, PetscReal points[], DMMeshInterpolationInfo ctx);
PetscErrorCode DMMeshInterpolationSetUp(DM dm, DMMeshInterpolationInfo ctx, PetscBool);
PetscErrorCode DMMeshInterpolationGetCoordinates(DM dm, Vec *points, DMMeshInterpolationInfo ctx);
PetscErrorCode DMMeshInterpolationGetVector(DM dm, Vec *values, DMMeshInterpolationInfo ctx);
PetscErrorCode DMMeshInterpolationRestoreVector(DM dm, Vec *values, DMMeshInterpolationInfo ctx);
PetscErrorCode DMMeshInterpolationEvaluate(DM dm, SectionReal x, Vec v, DMMeshInterpolationInfo ctx);
PetscErrorCode DMMeshInterpolationDestroy(DM dm, DMMeshInterpolationInfo *ctx);

#endif /* Mesh section */
#endif
