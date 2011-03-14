/*
  DMMesh, for easy parallelism of simple unstructured distributed mesh problems.
*/
#if !defined(__PETSCDMMESH_H)
#define __PETSCDMMESH_H

#include <petscdm.h>

#if defined(PETSC_HAVE_SIEVE) && defined(__cplusplus)

#include <sieve/Mesh.hh>
#include <sieve/CartesianSieve.hh>
#include <sieve/Distribution.hh>
#include <sieve/Generator.hh>

extern PetscLogEvent Mesh_View, Mesh_GetGlobalScatter, Mesh_restrictVector, Mesh_assembleVector, Mesh_assembleVectorComplete, Mesh_assembleMatrix, Mesh_updateOperator;

extern PetscErrorCode DMMeshCreate(MPI_Comm, DM*);
extern PetscErrorCode DMMeshGetMesh(DM, ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode DMMeshSetMesh(DM, const ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode DMMeshGetGlobalScatter(DM, VecScatter *);
extern PetscErrorCode DMMeshFinalize();

extern PetscErrorCode DMMeshDistribute(DM, const char[], DM*);
extern PetscErrorCode DMMeshDistributeByFace(DM, const char[], DM*);
extern PetscErrorCode DMMeshGenerate(DM, PetscBool , DM *);
extern PetscErrorCode DMMeshRefine(DM, double, PetscBool , DM*);
extern PetscErrorCode DMMeshLoad(PetscViewer, DM);
extern PetscErrorCode DMMeshGetMaximumDegree(DM, PetscInt *);

extern PetscErrorCode DMMeshGetLabelSize(DM, const char[], PetscInt *);
extern PetscErrorCode DMMeshGetLabelIds(DM, const char[], PetscInt *);
extern PetscErrorCode DMMeshGetStratumSize(DM, const char [], PetscInt, PetscInt *);
extern PetscErrorCode DMMeshGetStratum(DM, const char [], PetscInt, PetscInt *);

extern PetscErrorCode DMCartesianCreate(MPI_Comm, DM *);
extern PetscErrorCode DMMeshCartesianGetMesh(DM, ALE::Obj<ALE::CartesianMesh>&);
extern PetscErrorCode DMMeshCartesianSetMesh(DM, const ALE::Obj<ALE::CartesianMesh>&);

extern PetscErrorCode DMMeshGetCoordinates(DM, PetscBool , PetscInt *, PetscInt *, PetscReal *[]);
extern PetscErrorCode DMMeshGetElements(DM, PetscBool , PetscInt *, PetscInt *, PetscInt *[]);
extern PetscErrorCode DMMeshGetCone(DM, PetscInt, PetscInt *, PetscInt *[]);

extern PetscErrorCode restrictVector(Vec, Vec, InsertMode);
extern PetscErrorCode assembleVectorComplete(Vec, Vec, InsertMode);
extern PetscErrorCode assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode updateOperator(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);
extern PetscErrorCode updateOperatorGeneral(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);

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
extern PetscErrorCode  SectionRealDestroy(SectionReal);
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
extern PetscErrorCode DMMeshSetSectionReal(DM, SectionReal);
extern PetscErrorCode DMMeshCreateMatrix(DM, SectionReal, MatType, Mat *);
extern PetscErrorCode DMMeshCreateVector(DM, SectionReal, Vec *);
extern PetscErrorCode DMMeshCreateGlobalScatter(DM, SectionReal, VecScatter *);
extern PetscErrorCode assembleVector(Vec, DM, SectionReal, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode assembleMatrix(Mat, DM, SectionReal, PetscInt, PetscScalar [], InsertMode);

extern PetscErrorCode DMMeshCreateGlobalRealVector(DM, SectionReal, Vec *);
extern PetscErrorCode DMMeshGetGlobalScatter(DM,VecScatter *);
extern PetscErrorCode DMMeshCreateGlobalScatter(DM,SectionReal,VecScatter *);
extern PetscErrorCode DMMeshGetLocalFunction(DM, PetscErrorCode (**)(DM, SectionReal, SectionReal, void*));
extern PetscErrorCode DMMeshSetLocalFunction(DM, PetscErrorCode (*)(DM, SectionReal, SectionReal, void*));
extern PetscErrorCode DMMeshGetLocalJacobian(DM, PetscErrorCode (**)(DM, SectionReal, Mat, void*));
extern PetscErrorCode DMMeshSetLocalJacobian(DM, PetscErrorCode (*)(DM, SectionReal, Mat, void*));
extern PetscErrorCode DMMeshFormFunction(DM, SectionReal, SectionReal, void*);
extern PetscErrorCode DMMeshFormJacobian(DM, SectionReal, Mat, void*);
extern PetscErrorCode DMMeshInterpolatePoints(DM, SectionReal, int, double *, double **);

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
extern PetscErrorCode  SectionIntDestroy(SectionInt);
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

typedef PetscErrorCode (*DMMeshLocalFunction1)(DM,SectionReal,SectionReal,void*);
typedef PetscErrorCode (*DMMeshLocalJacobian1)(DM,SectionReal,Mat,void*);

/* Misc Mesh functions*/
extern PetscErrorCode DMMeshSetMaxDof(DM, PetscInt);
extern PetscErrorCode SectionGetArray(DM, const char [], PetscInt *, PetscInt *, PetscScalar *[]);

/* Helper functions for simple distributions */
extern PetscErrorCode DMMeshGetVertexMatrix(DM, MatType, Mat *);
extern PetscErrorCode DMMeshGetVertexSectionReal(DM, const char[], PetscInt, SectionReal *);
PetscPolymorphicSubroutine(DMMeshGetVertexSectionReal,(DM dm, PetscInt fiberDim, SectionReal *section),(dm,"default",fiberDim,section))
extern PetscErrorCode DMMeshGetVertexSectionInt(DM, const char[], PetscInt, SectionInt *);
PetscPolymorphicSubroutine(DMMeshGetVertexSectionInt,(DM dm, PetscInt fiberDim, SectionInt *section),(dm,"default",fiberDim,section))
extern PetscErrorCode DMMeshGetCellMatrix(DM, MatType, Mat *);
extern PetscErrorCode DMMeshGetCellSectionReal(DM, const char[], PetscInt, SectionReal *);
PetscPolymorphicSubroutine(DMMeshGetCellSectionReal,(DM dm, PetscInt fiberDim, SectionReal *section),(dm,"default",fiberDim,section))
extern PetscErrorCode DMMeshGetCellSectionInt(DM, const char[], PetscInt, SectionInt *);
PetscPolymorphicSubroutine(DMMeshGetCellSectionInt,(DM dm, PetscInt fiberDim, SectionInt *section),(dm,"default",fiberDim,section))

/* Support for various mesh formats */
extern PetscErrorCode DMMeshCreateExodus(MPI_Comm, const char [], DM *);
extern PetscErrorCode DMMeshExodusGetInfo(DM, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);

extern PetscErrorCode DMMeshCreatePCICE(MPI_Comm, const int, const char [], const char [], PetscBool , const char [], DM *);

extern PetscErrorCode DMWriteVTKHeader(PetscViewer);
extern PetscErrorCode DMWriteVTKVertices(DM, PetscViewer);
extern PetscErrorCode DMWriteVTKElements(DM, PetscViewer);
extern PetscErrorCode DMWritePCICEVertices(DM, PetscViewer);
extern PetscErrorCode DMWritePCICEElements(DM, PetscViewer);
extern PetscErrorCode DMWritePyLithVertices(DM, PetscViewer);
extern PetscErrorCode DMWritePyLithElements(DM, SectionReal, PetscViewer);
extern PetscErrorCode DMWritePyLithVerticesLocal(DM, PetscViewer);
extern PetscErrorCode WDMritePyLithElementsLocal(DM, SectionReal, PetscViewer);

typedef struct {
  int           numQuadPoints, numBasisFuncs;
  const double *quadPoints, *quadWeights, *basis, *basisDer;
} PetscQuadrature;

#endif /* Mesh section */
#endif
