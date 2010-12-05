/*
  Mesh object, for easy parallelism of simple unstructured distributed mesh problems.
*/
#if !defined(__PETSCMESH_H)
#define __PETSCMESH_H
#include "petscsys.h"

#if defined(PETSC_HAVE_SIEVE) && defined(__cplusplus)

#include <sieve/Mesh.hh>
#include <sieve/CartesianSieve.hh>
#include <sieve/Distribution.hh>
#include <sieve/Generator.hh>
#include "petscdm.h"
PETSC_EXTERN_CXX_BEGIN

/*S
   Mesh - Abstract PETSc object that combines a topology (Sieve) and coordinates (Section).

   Level: beginner

  Concepts: distributed array

.seealso:  MeshCreate(), MeshDestroy(), Section, SectionCreate()
S*/
typedef struct _p_Mesh* Mesh;
/*E
    MeshType - String with the name of a PETSc mesh or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mymatcreate()

   Level: beginner

.seealso: MeshSetType(), Mesh
E*/
#define MeshType char*
#define MESHSIEVE     "sieve"
#define MESHCARTESIAN "cartesian"

/* Logging support */
extern PetscClassId  MESH_CLASSID;
extern PetscLogEvent  Mesh_View, Mesh_GetGlobalScatter, Mesh_restrictVector, Mesh_assembleVector,
  Mesh_assembleVectorComplete, Mesh_assembleMatrix, Mesh_updateOperator;

extern PetscErrorCode  MeshFinalize();

extern PetscErrorCode  MeshView(Mesh, PetscViewer);
extern PetscErrorCode  MeshLoad(PetscViewer, Mesh);
extern PetscErrorCode  MeshCreate(MPI_Comm, Mesh*);
extern PetscErrorCode  MeshDestroy(Mesh);
extern PetscErrorCode  MeshSetType(Mesh, MeshType);
extern PetscErrorCode  MeshCreateGlobalVector(Mesh, Vec*);
extern PetscErrorCode  MeshCreateLocalVector(Mesh, Vec *);
extern PetscErrorCode  MeshGetMatrix(Mesh, const MatType,Mat*);
extern PetscErrorCode  MeshGetVertexMatrix(Mesh, MatType, Mat *);
extern PetscErrorCode  MeshGetGlobalIndices(Mesh,PetscInt*[]);

/*MC
   MeshRegisterDynamic - Adds a type to the Mesh package.

   Synopsis:
   PetscErrorCode MeshRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(Mesh))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create mesh context
-  routine_create - routine to create mesh context

   Notes:
   MeshRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   MeshRegisterDynamic("my_mesh",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyMeshCreate",MyMeshCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MeshSetType(mesh,"my_mesh")
   or at runtime via the option
$     -mesh_type my_mesh

   Level: advanced

   Notes: Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
          and others of the form ${any_environmental_variable} occuring in pathname will be 
          replaced with appropriate values.
         If your function is not being put into a shared library then use MeshRegister() instead

.keywords: Mesh, register

.seealso: MeshRegisterAll(), MeshRegisterDestroy()

M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MeshRegisterDynamic(a,b,c,d) MeshRegister(a,b,c,0)
#else
#define MeshRegisterDynamic(a,b,c,d) MeshRegister(a,b,c,d)
#endif

extern PetscFList MeshList;
extern PetscErrorCode  MeshRegisterAll(const char[]);
extern PetscErrorCode  MeshRegisterDestroy(void);
extern PetscErrorCode  MeshRegister(const char[],const char[],const char[],PetscErrorCode (*)(Mesh));

extern PetscErrorCode  MeshGetMesh(Mesh,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode  MeshSetMesh(Mesh,const ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode  MeshDistribute(Mesh, const char[], Mesh*);
extern PetscErrorCode  MeshDistributeByFace(Mesh, const char[], Mesh*);
extern PetscErrorCode  MeshGenerate(Mesh, PetscBool , Mesh *);
extern PetscErrorCode  MeshRefine(Mesh, double, PetscBool , Mesh*);
extern PetscErrorCode  MeshUnify(Mesh, Mesh*);
extern PetscErrorCode  MeshGetMaximumDegree(Mesh, PetscInt *);

extern PetscErrorCode  MeshSetMaxDof(Mesh, PetscInt);
extern PetscErrorCode  restrictVector(Vec, Vec, InsertMode);
extern PetscErrorCode  assembleVectorComplete(Vec, Vec, InsertMode);
extern PetscErrorCode  assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode  preallocateMatrix(const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type::atlas_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, Mat);
extern PetscErrorCode  updateOperator(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);
extern PetscErrorCode  updateOperator(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, int, int, PetscScalar [], InsertMode);
extern PetscErrorCode  updateOperatorGeneral(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);

extern PetscErrorCode  MeshCreatePCICE(MPI_Comm, const int, const char[], const char[], PetscBool , const char[], Mesh *);
extern PetscErrorCode  MeshGetCoordinates(Mesh, PetscBool , PetscInt *, PetscInt *, PetscReal *[]);
extern PetscErrorCode  MeshGetElements(Mesh, PetscBool , PetscInt *, PetscInt *, PetscInt *[]);
extern PetscErrorCode  MeshGetCone(Mesh, PetscInt, PetscInt *, PetscInt *[]);

extern PetscErrorCode  MeshCreatePyLith(MPI_Comm, const int, const char[], PetscBool , PetscBool , Mesh *);

extern PetscErrorCode  MeshCreateExodus(MPI_Comm, const char[], Mesh *);
extern PetscErrorCode  MeshExodusGetInfo(Mesh, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
extern PetscErrorCode  MeshGetStratumSize(Mesh, const char[], PetscInt, PetscInt *);
extern PetscErrorCode  MeshGetStratum(Mesh, const char[], PetscInt, PetscInt *);

extern PetscErrorCode  SectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscScalar *[]);
extern PetscErrorCode  BCSectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscInt *[]);
extern PetscErrorCode  BCSectionRealCreate(Mesh, const char [], PetscInt);
extern PetscErrorCode  BCSectionRealGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscReal *[]);
extern PetscErrorCode  BCFUNCGetArray(Mesh, PetscInt *, PetscInt *, PetscScalar *[]);
extern PetscErrorCode  WritePCICERestart(Mesh, PetscViewer);
extern PetscErrorCode  PCICERenumberBoundary(Mesh);

extern PetscErrorCode  MeshCartesianGetMesh(Mesh,ALE::Obj<ALE::CartesianMesh>&);
extern PetscErrorCode  MeshCartesianSetMesh(Mesh,const ALE::Obj<ALE::CartesianMesh>&);

/*S
  SectionReal - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionRealCreate(), SectionRealDestroy(), Mesh, MeshCreate()
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

extern PetscErrorCode  SectionRealDistribute(SectionReal, Mesh, SectionReal *);
extern PetscErrorCode  SectionRealRestrict(SectionReal, PetscInt, PetscScalar *[]);
extern PetscErrorCode  SectionRealUpdate(SectionReal, PetscInt, const PetscScalar [], InsertMode);
extern PetscErrorCode  SectionRealZero(SectionReal);
extern PetscErrorCode  SectionRealCreateLocalVector(SectionReal, Vec*);
extern PetscErrorCode  SectionRealAddSpace(SectionReal);
extern PetscErrorCode  SectionRealGetFibration(SectionReal, const PetscInt, SectionReal *);
extern PetscErrorCode  SectionRealToVec(SectionReal, Mesh, ScatterMode, Vec);
extern PetscErrorCode  SectionRealToVec(SectionReal, VecScatter, ScatterMode, Vec);
extern PetscErrorCode  SectionRealNorm(SectionReal, Mesh, NormType, PetscReal *);
extern PetscErrorCode  SectionRealAXPY(SectionReal, Mesh, PetscScalar, SectionReal);
extern PetscErrorCode  SectionRealComplete(SectionReal);
extern PetscErrorCode  SectionRealSet(SectionReal, PetscReal);
extern PetscErrorCode  SectionRealGetFiberDimension(SectionReal, PetscInt, PetscInt*);
extern PetscErrorCode  SectionRealSetFiberDimension(SectionReal, PetscInt, const PetscInt);
extern PetscErrorCode  SectionRealSetFiberDimensionField(SectionReal, PetscInt, const PetscInt, const PetscInt);
extern PetscErrorCode  SectionRealGetSize(SectionReal, PetscInt *);
extern PetscErrorCode  SectionRealAllocate(SectionReal);
extern PetscErrorCode  SectionRealClear(SectionReal);

extern PetscErrorCode  SectionRealRestrictClosure(SectionReal, Mesh, PetscInt, PetscInt, PetscScalar []);
extern PetscErrorCode  SectionRealRestrictClosure(SectionReal, Mesh, PetscInt, const PetscScalar *[]);
extern PetscErrorCode  SectionRealUpdateClosure(SectionReal, Mesh, PetscInt, PetscScalar [], InsertMode);

extern PetscErrorCode  MeshGetVertexSectionReal(Mesh, const char[], PetscInt, SectionReal *);
PetscPolymorphicSubroutine(MeshGetVertexSectionReal,(Mesh mesh, PetscInt fiberDim, SectionReal *section),(mesh,"default",fiberDim,section))
extern PetscErrorCode  MeshGetCellSectionReal(Mesh, const char[], PetscInt, SectionReal *);
extern PetscErrorCode  MeshHasSectionReal(Mesh, const char [], PetscBool  *);
extern PetscErrorCode  MeshGetSectionReal(Mesh, const char [], SectionReal *);
extern PetscErrorCode  MeshSetSectionReal(Mesh, SectionReal);
extern PetscErrorCode  MeshCreateMatrix(Mesh, SectionReal, MatType, Mat *);
extern PetscErrorCode  MeshCreateVector(Mesh, SectionReal, Vec *);
extern PetscErrorCode  assembleVector(Vec, Mesh, SectionReal, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode  assembleMatrix(Mat, Mesh, SectionReal, PetscInt, PetscScalar [], InsertMode);

extern PetscErrorCode  MeshCreateGlobalRealVector(Mesh, SectionReal, Vec *);
extern PetscErrorCode  MeshGetGlobalScatter(Mesh,VecScatter *);
extern PetscErrorCode  MeshCreateGlobalScatter(Mesh,SectionReal,VecScatter *);
extern PetscErrorCode  MeshGetLocalFunction(Mesh, PetscErrorCode (**)(Mesh, SectionReal, SectionReal, void*));
extern PetscErrorCode  MeshSetLocalFunction(Mesh, PetscErrorCode (*)(Mesh, SectionReal, SectionReal, void*));
extern PetscErrorCode  MeshGetLocalJacobian(Mesh, PetscErrorCode (**)(Mesh, SectionReal, Mat, void*));
extern PetscErrorCode  MeshSetLocalJacobian(Mesh, PetscErrorCode (*)(Mesh, SectionReal, Mat, void*));
extern PetscErrorCode  MeshFormFunction(Mesh, SectionReal, SectionReal, void*);
extern PetscErrorCode  MeshFormJacobian(Mesh, SectionReal, Mat, void*);
extern PetscErrorCode  MeshInterpolatePoints(Mesh, SectionReal, int, double *, double **);

/*S
  SectionInt - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionIntCreate(), SectionIntDestroy(), Mesh, MeshCreate()
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

extern PetscErrorCode  SectionIntDistribute(SectionInt, Mesh, SectionInt *);
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

extern PetscErrorCode  SectionIntRestrictClosure(SectionInt, Mesh, PetscInt, PetscInt, PetscInt []);
extern PetscErrorCode  SectionIntUpdateClosure(SectionInt, Mesh, PetscInt, PetscInt [], InsertMode);

extern PetscErrorCode  MeshGetVertexSectionInt(Mesh, const char[], PetscInt, SectionInt *);
extern PetscErrorCode  MeshGetCellSectionInt(Mesh, const char[], PetscInt, SectionInt *);
extern PetscErrorCode  MeshHasSectionInt(Mesh, const char [], PetscBool  *);
extern PetscErrorCode  MeshGetSectionInt(Mesh, const char [], SectionInt *);
extern PetscErrorCode  MeshSetSectionInt(Mesh, SectionInt);

#if 0
/*S
  SectionPair - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionPairCreate(), SectionPairDestroy(), Mesh, MeshCreate()
S*/
typedef struct _p_SectionPair* SectionPair;

/* Logging support */
extern PetscClassId  SECTIONPAIR_CLASSID;

extern PetscErrorCode  SectionPairCreate(MPI_Comm,SectionPair*);
extern PetscErrorCode  SectionPairDestroy(SectionPair);
extern PetscErrorCode  SectionPairView(SectionPair,PetscViewer);

extern PetscErrorCode  SectionPairGetSection(SectionPair,ALE::Obj<PETSC_MESH_TYPE::pair_section_type>&);
extern PetscErrorCode  SectionPairSetSection(SectionPair,const ALE::Obj<PETSC_MESH_TYPE::pair_section_type>&);
extern PetscErrorCode  SectionPairGetBundle(SectionPair,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode  SectionPairSetBundle(SectionPair,const ALE::Obj<PETSC_MESH_TYPE>&);

typedef struct {
  int    i;
  double x, y, z;
} PetscPair;

extern PetscErrorCode  SectionPairDistribute(SectionPair, Mesh, SectionPair *);
extern PetscErrorCode  SectionPairRestrict(SectionPair, PetscInt, PetscPair *[]);
extern PetscErrorCode  SectionPairUpdate(SectionPair, PetscInt, const PetscPair []);

extern PetscErrorCode  MeshHasSectionPair(Mesh, const char [], PetscBool  *);
extern PetscErrorCode  MeshGetSectionPair(Mesh, const char [], SectionPair *);
extern PetscErrorCode  MeshSetSectionPair(Mesh, SectionPair);
#endif

extern PetscErrorCode  WriteVTKHeader(PetscViewer);
extern PetscErrorCode  WriteVTKVertices(Mesh, PetscViewer);
extern PetscErrorCode  WriteVTKElements(Mesh, PetscViewer);
extern PetscErrorCode  WritePCICEVertices(Mesh, PetscViewer);
extern PetscErrorCode  WritePCICEElements(Mesh, PetscViewer);  
extern PetscErrorCode  WritePyLithVertices(Mesh, PetscViewer);
extern PetscErrorCode  WritePyLithElements(Mesh, SectionReal, PetscViewer);
extern PetscErrorCode  WritePyLithVerticesLocal(Mesh, PetscViewer);
extern PetscErrorCode  WritePyLithElementsLocal(Mesh, SectionReal, PetscViewer);

extern PetscErrorCode  MeshGetLabelIds(Mesh, const char[], PetscInt *);
extern PetscErrorCode  MeshGetLabelSize(Mesh, const char[], PetscInt *);

typedef struct {
  int           numQuadPoints, numBasisFuncs;
  const double *quadPoints, *quadWeights, *basis, *basisDer;
} PetscQuadrature;

PETSC_EXTERN_CXX_END

#endif
#endif
