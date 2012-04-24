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
extern PetscLogEvent DMMesh_View, DMMesh_GetGlobalScatter, DMMesh_restrictVector, DMMesh_assembleVector, DMMesh_assembleVectorComplete, DMMesh_assembleMatrix, DMMesh_updateOperator;

extern PetscErrorCode DMMeshCreate(MPI_Comm, DM*);
extern PetscErrorCode DMMeshCreateMeshFromAdjacency(MPI_Comm, PetscInt, PetscInt, PetscInt, PetscInt [], PetscInt, PetscInt, const PetscReal [], PetscBool, DM*);
extern PetscErrorCode DMMeshGetMesh(DM, ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode DMMeshSetMesh(DM, const ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode DMMeshGetGlobalScatter(DM, VecScatter *);
extern PetscErrorCode DMMeshFinalize();

/* New Sieve Mesh interface */
extern PetscErrorCode DMMeshGetDimension(DM, PetscInt *);
extern PetscErrorCode DMMeshSetDimension(DM, PetscInt);
extern PetscErrorCode DMMeshGetChart(DM, PetscInt *, PetscInt *);
extern PetscErrorCode DMMeshSetChart(DM, PetscInt, PetscInt);
extern PetscErrorCode DMMeshGetConeSize(DM, PetscInt, PetscInt *);
extern PetscErrorCode DMMeshSetConeSize(DM, PetscInt, PetscInt);
extern PetscErrorCode DMMeshGetCone(DM, PetscInt, const PetscInt *[]);
extern PetscErrorCode DMMeshSetCone(DM, PetscInt, const PetscInt[]);
extern PetscErrorCode DMMeshGetSupportSize(DM, PetscInt, PetscInt *);
extern PetscErrorCode DMMeshGetSupport(DM, PetscInt, const PetscInt *[]);
extern PetscErrorCode DMMeshGetConeSection(DM, PetscSection *);
extern PetscErrorCode DMMeshGetCones(DM, PetscInt *[]);
extern PetscErrorCode DMMeshGetMaxSizes(DM, PetscInt *, PetscInt *);
extern PetscErrorCode DMMeshSetUp(DM);
extern PetscErrorCode DMMeshSymmetrize(DM);
extern PetscErrorCode DMMeshStratify(DM);

extern PetscErrorCode DMMeshGetLabelValue(DM, const char[], PetscInt, PetscInt *);
extern PetscErrorCode DMMeshSetLabelValue(DM, const char[], PetscInt, PetscInt);
extern PetscErrorCode DMMeshGetLabelSize(DM, const char[], PetscInt *);
extern PetscErrorCode DMMeshGetLabelIdIS(DM, const char[], IS *);
extern PetscErrorCode DMMeshGetStratumSize(DM, const char [], PetscInt, PetscInt *);
extern PetscErrorCode DMMeshGetStratumIS(DM, const char [], PetscInt, IS *);

extern PetscErrorCode DMMeshJoinPoints(DM, PetscInt, const PetscInt [], PetscInt *);
extern PetscErrorCode DMMeshMeetPoints(DM, PetscInt, const PetscInt [], PetscInt *, const PetscInt **);
extern PetscErrorCode DMMeshGetTransitiveClosure(DM, PetscInt, PetscBool, PetscInt *, const PetscInt *[]);

extern PetscErrorCode DMMeshCreatePartition(DM, PetscSection *, IS *, PetscInt);
extern PetscErrorCode DMMeshCreatePartitionClosure(DM, PetscSection, IS, PetscSection *, IS *);

/* Old Sieve Mesh interface */
extern PetscErrorCode DMMeshDistribute(DM, const char[], DM*);
extern PetscErrorCode DMMeshDistributeByFace(DM, const char[], DM*);
extern PetscErrorCode DMMeshGenerate(DM, PetscBool , DM *);
extern PetscErrorCode DMMeshRefine(DM, double, PetscBool , DM*);
extern PetscErrorCode DMMeshLoad(PetscViewer, DM);
extern PetscErrorCode DMMeshGetMaximumDegree(DM, PetscInt *);

extern PetscErrorCode DMCartesianCreate(MPI_Comm, DM *);
extern PetscErrorCode DMMeshCartesianGetMesh(DM, ALE::Obj<ALE::CartesianMesh>&);
extern PetscErrorCode DMMeshCartesianSetMesh(DM, const ALE::Obj<ALE::CartesianMesh>&);

extern PetscErrorCode DMMeshGetCoordinates(DM, PetscBool , PetscInt *, PetscInt *, PetscReal *[]);
extern PetscErrorCode DMMeshGetElements(DM, PetscBool , PetscInt *, PetscInt *, PetscInt *[]);

extern PetscErrorCode DMMeshCreateBoxMesh(MPI_Comm, PetscInt, PetscBool, DM *);
extern PetscErrorCode DMMeshMarkBoundaryCells(DM, const char [], PetscInt, PetscInt);
extern PetscErrorCode DMMeshGetDepthStratum(DM, PetscInt, PetscInt *, PetscInt *);
extern PetscErrorCode DMMeshGetHeightStratum(DM, PetscInt, PetscInt *, PetscInt *);
extern PetscErrorCode DMMeshCreateSection(DM, PetscInt, PetscInt, PetscInt [], PetscInt [], PetscInt, PetscInt [], IS [], PetscSection *);
extern PetscErrorCode DMMeshConvertSection(const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, PetscSection *);
extern PetscErrorCode DMMeshGetSection(DM, const char [], PetscSection *);
extern PetscErrorCode DMMeshSetSection(DM, const char [], PetscSection);
extern PetscErrorCode DMMeshGetDefaultSection(DM, PetscSection *);
extern PetscErrorCode DMMeshSetDefaultSection(DM, PetscSection);
extern PetscErrorCode DMMeshGetCoordinateSection(DM, PetscSection *);
extern PetscErrorCode DMMeshSetCoordinateSection(DM, PetscSection);
extern PetscErrorCode DMMeshCreateConeSection(DM, PetscSection *);
extern PetscErrorCode DMMeshGetCoordinateVec(DM, Vec *);
extern PetscErrorCode DMMeshComputeCellGeometry(DM, PetscInt, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
extern PetscErrorCode DMMeshVecSetClosure(DM, Vec, PetscInt, const PetscScalar [], InsertMode);
extern PetscErrorCode DMMeshVecGetClosure(DM, Vec, PetscInt, const PetscScalar **);
extern PetscErrorCode DMMeshMatSetClosure(DM, Mat, PetscInt, PetscScalar [], InsertMode);

extern PetscErrorCode MatSetValuesTopology(Mat, DM, PetscInt, const PetscInt [], DM, PetscInt, const PetscInt [], const PetscScalar [], InsertMode);
extern PetscErrorCode DMMeshRestrictVector(Vec, Vec, InsertMode);
extern PetscErrorCode DMMeshAssembleVectorComplete(Vec, Vec, InsertMode);
extern PetscErrorCode DMMeshAssembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode DMMeshUpdateOperator(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);
extern PetscErrorCode DMMeshUpdateOperatorGeneral(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);

/*S
  SectionReal - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionRealCreate(), SectionRealDestroy(), Mesh, DMMeshCreate()
S*/
typedef struct _p_SectionReal* SectionReal;

/* Logging support */
extern PetscClassId  SECTIONREAL_CLASSID;

extern PetscErrorCode  SectionRealCreate(MPI_Comm,SectionReal*);
extern PetscErrorCode  SectionRealDestroy(SectionReal*);
extern PetscErrorCode  SectionRealView(SectionReal,PetscViewer);
extern PetscErrorCode  SectionRealDuplicate(SectionReal,SectionReal*);

extern PetscErrorCode  SectionRealGetSection(SectionReal,ALE::Obj<PETSC_MESH_TYPE::real_section_type>&);
extern PetscErrorCode  SectionRealSetSection(SectionReal,const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&);
extern PetscErrorCode  SectionRealGetBundle(SectionReal,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode  SectionRealSetBundle(SectionReal,const ALE::Obj<PETSC_MESH_TYPE>&);

extern PetscErrorCode  SectionRealDistribute(SectionReal, DM, SectionReal *);
extern PetscErrorCode  SectionRealRestrict(SectionReal, PetscInt, PetscScalar *[]);
extern PetscErrorCode  SectionRealUpdate(SectionReal, PetscInt, const PetscScalar [], InsertMode);
extern PetscErrorCode  SectionRealZero(SectionReal);
extern PetscErrorCode  SectionRealCreateLocalVector(SectionReal, Vec*);
extern PetscErrorCode  SectionRealAddSpace(SectionReal);
extern PetscErrorCode  SectionRealGetFibration(SectionReal, const PetscInt, SectionReal *);
extern PetscErrorCode  SectionRealToVec(SectionReal, DM, ScatterMode, Vec);
extern PetscErrorCode  SectionRealToVec(SectionReal, VecScatter, ScatterMode, Vec);
extern PetscErrorCode  SectionRealNorm(SectionReal, DM, NormType, PetscReal *);
extern PetscErrorCode  SectionRealAXPY(SectionReal, DM, PetscScalar, SectionReal);
extern PetscErrorCode  SectionRealComplete(SectionReal);
extern PetscErrorCode  SectionRealSet(SectionReal, PetscReal);
extern PetscErrorCode  SectionRealGetFiberDimension(SectionReal, PetscInt, PetscInt*);
extern PetscErrorCode  SectionRealSetFiberDimension(SectionReal, PetscInt, const PetscInt);
extern PetscErrorCode  SectionRealSetFiberDimensionField(SectionReal, PetscInt, const PetscInt, const PetscInt);
extern PetscErrorCode  SectionRealGetSize(SectionReal, PetscInt *);
extern PetscErrorCode  SectionRealAllocate(SectionReal);
extern PetscErrorCode  SectionRealClear(SectionReal);

extern PetscErrorCode  SectionRealRestrictClosure(SectionReal, DM, PetscInt, PetscInt, PetscScalar []);
extern PetscErrorCode  SectionRealRestrictClosure(SectionReal, DM, PetscInt, const PetscScalar *[]);
extern PetscErrorCode  SectionRealUpdateClosure(SectionReal, DM, PetscInt, PetscScalar [], InsertMode);

extern PetscErrorCode DMMeshHasSectionReal(DM, const char [], PetscBool  *);
extern PetscErrorCode DMMeshGetSectionReal(DM, const char [], SectionReal *);
extern PetscErrorCode DMMeshSetSectionReal(DM, const char [], SectionReal);
extern PetscErrorCode DMMeshCreateVector(DM, SectionReal, Vec *);
extern PetscErrorCode DMMeshCreateMatrix(DM, SectionReal, const MatType, Mat *);
extern PetscErrorCode DMMeshCreateGlobalScatter(DM, SectionReal, VecScatter *);
extern PetscErrorCode DMMeshAssembleVector(Vec, DM, SectionReal, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode DMMeshAssembleMatrix(Mat, DM, SectionReal, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode DMMeshSetupSection(DM, SectionReal);

typedef PetscErrorCode (*DMMeshLocalFunction1)(DM, Vec, Vec, void*);
typedef PetscErrorCode (*DMMeshLocalJacobian1)(DM, Vec, Mat, void*);

extern PetscErrorCode DMMeshCreateGlobalRealVector(DM, SectionReal, Vec *);
extern PetscErrorCode DMMeshGetGlobalScatter(DM,VecScatter *);
extern PetscErrorCode DMMeshCreateGlobalScatter(DM,SectionReal,VecScatter *);
extern PetscErrorCode DMMeshGetLocalFunction(DM, PetscErrorCode (**)(DM, Vec, Vec, void*));
extern PetscErrorCode DMMeshSetLocalFunction(DM, PetscErrorCode (*)(DM, Vec, Vec, void*));
extern PetscErrorCode DMMeshGetLocalJacobian(DM, PetscErrorCode (**)(DM, Vec, Mat, void*));
extern PetscErrorCode DMMeshSetLocalJacobian(DM, PetscErrorCode (*)(DM, Vec, Mat, void*));

/*S
  SectionInt - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionIntCreate(), SectionIntDestroy(), DM, DMMeshCreate()
S*/
typedef struct _p_SectionInt* SectionInt;

/* Logging support */
extern PetscClassId  SECTIONINT_CLASSID;

extern PetscErrorCode  SectionIntCreate(MPI_Comm,SectionInt*);
extern PetscErrorCode  SectionIntDestroy(SectionInt*);
extern PetscErrorCode  SectionIntView(SectionInt,PetscViewer);

extern PetscErrorCode  SectionIntGetSection(SectionInt,ALE::Obj<PETSC_MESH_TYPE::int_section_type>&);
extern PetscErrorCode  SectionIntSetSection(SectionInt,const ALE::Obj<PETSC_MESH_TYPE::int_section_type>&);
extern PetscErrorCode  SectionIntGetBundle(SectionInt,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode  SectionIntSetBundle(SectionInt,const ALE::Obj<PETSC_MESH_TYPE>&);

extern PetscErrorCode  SectionIntDistribute(SectionInt, DM, SectionInt *);
extern PetscErrorCode  SectionIntRestrict(SectionInt, PetscInt, PetscInt *[]);
extern PetscErrorCode  SectionIntUpdate(SectionInt, PetscInt, const PetscInt [], InsertMode);
extern PetscErrorCode  SectionIntZero(SectionInt);
extern PetscErrorCode  SectionIntComplete(SectionInt);
extern PetscErrorCode  SectionIntGetFiberDimension(SectionInt, PetscInt, PetscInt*);
extern PetscErrorCode  SectionIntSetFiberDimension(SectionInt, PetscInt, const PetscInt);
extern PetscErrorCode  SectionIntSetFiberDimensionField(SectionInt, PetscInt, const PetscInt, const PetscInt);
extern PetscErrorCode  SectionIntGetSize(SectionInt, PetscInt *);
extern PetscErrorCode  SectionIntAllocate(SectionInt);
extern PetscErrorCode  SectionIntClear(SectionInt);

extern PetscErrorCode  SectionIntAddSpace(SectionInt);
extern PetscErrorCode  SectionIntGetFibration(SectionInt, const PetscInt, SectionInt *);
extern PetscErrorCode  SectionIntSet(SectionInt, PetscInt);

extern PetscErrorCode  SectionIntRestrictClosure(SectionInt, DM, PetscInt, PetscInt, PetscInt []);
extern PetscErrorCode  SectionIntUpdateClosure(SectionInt, DM, PetscInt, PetscInt [], InsertMode);

extern PetscErrorCode  DMMeshHasSectionInt(DM, const char [], PetscBool  *);
extern PetscErrorCode  DMMeshGetSectionInt(DM, const char [], SectionInt *);
extern PetscErrorCode  DMMeshSetSectionInt(DM, SectionInt);

/* Misc Mesh functions*/
extern PetscErrorCode DMMeshSetMaxDof(DM, PetscInt);
extern PetscErrorCode SectionGetArray(DM, const char [], PetscInt *, PetscInt *, PetscScalar *[]);

/* Helper functions for simple distributions */
extern PetscErrorCode DMMeshGetVertexSectionReal(DM, const char[], PetscInt, SectionReal *);
extern PetscErrorCode DMMeshGetVertexSectionInt(DM, const char[], PetscInt, SectionInt *);
extern PetscErrorCode DMMeshGetCellSectionReal(DM, const char[], PetscInt, SectionReal *);
extern PetscErrorCode DMMeshGetCellSectionInt(DM, const char[], PetscInt, SectionInt *);
extern PetscErrorCode DMMeshCreateSectionRealIS(DM,IS,const char [],PetscInt,SectionReal *);

/* Scatter for simple distributions */
extern PetscErrorCode DMMeshCreateScatterToZeroVertex(DM,VecScatter *);
extern PetscErrorCode DMMeshCreateScatterToZeroVertexSet(DM,IS,IS,VecScatter *);
extern PetscErrorCode DMMeshCreateScatterToZeroCell(DM,VecScatter *);
extern PetscErrorCode DMMeshCreateScatterToZeroCellSet(DM,IS,IS,VecScatter *);

/* Support for various mesh formats */
extern PetscErrorCode DMMeshCreateExodus(MPI_Comm, const char [], DM *);
extern PetscErrorCode DMMeshCreateExodusNG(MPI_Comm, PetscInt, DM *);
extern PetscErrorCode DMMeshExodusGetInfo(DM, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
extern PetscErrorCode DMMeshViewExodusSplit(DM,PetscInt);

extern PetscErrorCode VecViewExodusVertex(DM,Vec,MPI_Comm,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode VecLoadExodusVertex(DM,Vec,MPI_Comm,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode VecViewExodusVertexSet(DM,Vec,PetscInt,MPI_Comm,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode VecLoadExodusVertexSet(DM,Vec,PetscInt,MPI_Comm,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode VecViewExodusCell(DM,Vec,MPI_Comm,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode VecLoadExodusCell(DM,Vec,MPI_Comm,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode VecViewExodusCellSet(DM,Vec,PetscInt,MPI_Comm,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode VecLoadExodusCellSet(DM,Vec,PetscInt,MPI_Comm,PetscInt,PetscInt,PetscInt);

extern PetscErrorCode DMMeshCreatePCICE(MPI_Comm, const int, const char [], const char [], PetscBool , const char [], DM *);

extern PetscErrorCode DMWriteVTKHeader(PetscViewer);
extern PetscErrorCode DMWriteVTKVertices(DM, PetscViewer);
extern PetscErrorCode DMWriteVTKElements(DM, PetscViewer);
extern PetscErrorCode DMWritePCICEVertices(DM, PetscViewer);
extern PetscErrorCode DMWritePCICEElements(DM, PetscViewer);
extern PetscErrorCode DMWritePyLithVertices(DM, PetscViewer);
extern PetscErrorCode DMWritePyLithElements(DM, SectionReal, PetscViewer);
extern PetscErrorCode DMWritePyLithVerticesLocal(DM, PetscViewer);
extern PetscErrorCode DMWritePyLithElementsLocal(DM, SectionReal, PetscViewer);

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
