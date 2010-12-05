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
extern PetscClassId PETSCDM_DLLEXPORT MESH_CLASSID;
extern PetscLogEvent  Mesh_View, Mesh_GetGlobalScatter, Mesh_restrictVector, Mesh_assembleVector,
  Mesh_assembleVectorComplete, Mesh_assembleMatrix, Mesh_updateOperator;

extern PetscErrorCode PETSCDM_DLLEXPORT MeshFinalize();

extern PetscErrorCode PETSCDM_DLLEXPORT MeshView(Mesh, PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshLoad(PetscViewer, Mesh);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreate(MPI_Comm, Mesh*);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshDestroy(Mesh);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshSetType(Mesh, MeshType);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalVector(Mesh, Vec*);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreateLocalVector(Mesh, Vec *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh, const MatType,Mat*);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexMatrix(Mesh, MatType, Mat *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalIndices(Mesh,PetscInt*[]);

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
extern PetscErrorCode PETSCDM_DLLEXPORT MeshRegisterAll(const char[]);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshRegisterDestroy(void);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshRegister(const char[],const char[],const char[],PetscErrorCode (*)(Mesh));

extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh,const ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshDistribute(Mesh, const char[], Mesh*);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshDistributeByFace(Mesh, const char[], Mesh*);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGenerate(Mesh, PetscBool , Mesh *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshRefine(Mesh, double, PetscBool , Mesh*);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshUnify(Mesh, Mesh*);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetMaximumDegree(Mesh, PetscInt *);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshSetMaxDof(Mesh, PetscInt);
extern PetscErrorCode PETSCDM_DLLEXPORT restrictVector(Vec, Vec, InsertMode);
extern PetscErrorCode PETSCDM_DLLEXPORT assembleVectorComplete(Vec, Vec, InsertMode);
extern PetscErrorCode PETSCDM_DLLEXPORT assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode PETSCDM_DLLEXPORT preallocateMatrix(const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type::atlas_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, Mat);
extern PetscErrorCode PETSCDM_DLLEXPORT updateOperator(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);
extern PetscErrorCode PETSCDM_DLLEXPORT updateOperator(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, int, int, PetscScalar [], InsertMode);
extern PetscErrorCode PETSCDM_DLLEXPORT updateOperatorGeneral(Mat, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, const ALE::Obj<PETSC_MESH_TYPE>&, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&, const ALE::Obj<PETSC_MESH_TYPE::order_type>&, const PETSC_MESH_TYPE::point_type&, PetscScalar [], InsertMode);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePCICE(MPI_Comm, const int, const char[], const char[], PetscBool , const char[], Mesh *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinates(Mesh, PetscBool , PetscInt *, PetscInt *, PetscReal *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetElements(Mesh, PetscBool , PetscInt *, PetscInt *, PetscInt *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetCone(Mesh, PetscInt, PetscInt *, PetscInt *[]);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePyLith(MPI_Comm, const int, const char[], PetscBool , PetscBool , Mesh *);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreateExodus(MPI_Comm, const char[], Mesh *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshExodusGetInfo(Mesh, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetStratumSize(Mesh, const char[], PetscInt, PetscInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetStratum(Mesh, const char[], PetscInt, PetscInt *);

extern PetscErrorCode PETSCDM_DLLEXPORT SectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscScalar *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT BCSectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscInt *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT BCSectionRealCreate(Mesh, const char [], PetscInt);
extern PetscErrorCode PETSCDM_DLLEXPORT BCSectionRealGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscReal *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT BCFUNCGetArray(Mesh, PetscInt *, PetscInt *, PetscScalar *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT WritePCICERestart(Mesh, PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT PCICERenumberBoundary(Mesh);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshCartesianGetMesh(Mesh,ALE::Obj<ALE::CartesianMesh>&);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshCartesianSetMesh(Mesh,const ALE::Obj<ALE::CartesianMesh>&);

/*S
  SectionReal - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionRealCreate(), SectionRealDestroy(), Mesh, MeshCreate()
S*/
typedef struct _p_SectionReal* SectionReal;

/* Logging support */
extern PetscClassId PETSCDM_DLLEXPORT SECTIONREAL_CLASSID;

extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealCreate(MPI_Comm,SectionReal*);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealDestroy(SectionReal);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealView(SectionReal,PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealDuplicate(SectionReal,SectionReal*);

extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetSection(SectionReal,ALE::Obj<PETSC_MESH_TYPE::real_section_type>&);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetSection(SectionReal,const ALE::Obj<PETSC_MESH_TYPE::real_section_type>&);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetBundle(SectionReal,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetBundle(SectionReal,const ALE::Obj<PETSC_MESH_TYPE>&);

extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealDistribute(SectionReal, Mesh, SectionReal *);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealRestrict(SectionReal, PetscInt, PetscScalar *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealUpdate(SectionReal, PetscInt, const PetscScalar [], InsertMode);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealZero(SectionReal);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealCreateLocalVector(SectionReal, Vec*);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealAddSpace(SectionReal);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetFibration(SectionReal, const PetscInt, SectionReal *);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealToVec(SectionReal, Mesh, ScatterMode, Vec);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealToVec(SectionReal, VecScatter, ScatterMode, Vec);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealNorm(SectionReal, Mesh, NormType, PetscReal *);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealAXPY(SectionReal, Mesh, PetscScalar, SectionReal);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealComplete(SectionReal);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealSet(SectionReal, PetscReal);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetFiberDimension(SectionReal, PetscInt, PetscInt*);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetFiberDimension(SectionReal, PetscInt, const PetscInt);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetFiberDimensionField(SectionReal, PetscInt, const PetscInt, const PetscInt);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetSize(SectionReal, PetscInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealAllocate(SectionReal);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealClear(SectionReal);

extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealRestrictClosure(SectionReal, Mesh, PetscInt, PetscInt, PetscScalar []);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealRestrictClosure(SectionReal, Mesh, PetscInt, const PetscScalar *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionRealUpdateClosure(SectionReal, Mesh, PetscInt, PetscScalar [], InsertMode);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexSectionReal(Mesh, const char[], PetscInt, SectionReal *);
PetscPolymorphicSubroutine(MeshGetVertexSectionReal,(Mesh mesh, PetscInt fiberDim, SectionReal *section),(mesh,"default",fiberDim,section))
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetCellSectionReal(Mesh, const char[], PetscInt, SectionReal *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshHasSectionReal(Mesh, const char [], PetscBool  *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetSectionReal(Mesh, const char [], SectionReal *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshSetSectionReal(Mesh, SectionReal);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreateMatrix(Mesh, SectionReal, MatType, Mat *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreateVector(Mesh, SectionReal, Vec *);
extern PetscErrorCode PETSCDM_DLLEXPORT assembleVector(Vec, Mesh, SectionReal, PetscInt, PetscScalar [], InsertMode);
extern PetscErrorCode PETSCDM_DLLEXPORT assembleMatrix(Mat, Mesh, SectionReal, PetscInt, PetscScalar [], InsertMode);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalRealVector(Mesh, SectionReal, Vec *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalScatter(Mesh,VecScatter *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalScatter(Mesh,SectionReal,VecScatter *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetLocalFunction(Mesh, PetscErrorCode (**)(Mesh, SectionReal, SectionReal, void*));
extern PetscErrorCode PETSCDM_DLLEXPORT MeshSetLocalFunction(Mesh, PetscErrorCode (*)(Mesh, SectionReal, SectionReal, void*));
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetLocalJacobian(Mesh, PetscErrorCode (**)(Mesh, SectionReal, Mat, void*));
extern PetscErrorCode PETSCDM_DLLEXPORT MeshSetLocalJacobian(Mesh, PetscErrorCode (*)(Mesh, SectionReal, Mat, void*));
extern PetscErrorCode PETSCDM_DLLEXPORT MeshFormFunction(Mesh, SectionReal, SectionReal, void*);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshFormJacobian(Mesh, SectionReal, Mat, void*);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshInterpolatePoints(Mesh, SectionReal, int, double *, double **);

/*S
  SectionInt - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionIntCreate(), SectionIntDestroy(), Mesh, MeshCreate()
S*/
typedef struct _p_SectionInt* SectionInt;

/* Logging support */
extern PetscClassId PETSCDM_DLLEXPORT SECTIONINT_CLASSID;

extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntCreate(MPI_Comm,SectionInt*);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntDestroy(SectionInt);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntView(SectionInt,PetscViewer);

extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetSection(SectionInt,ALE::Obj<PETSC_MESH_TYPE::int_section_type>&);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntSetSection(SectionInt,const ALE::Obj<PETSC_MESH_TYPE::int_section_type>&);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetBundle(SectionInt,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntSetBundle(SectionInt,const ALE::Obj<PETSC_MESH_TYPE>&);

extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntDistribute(SectionInt, Mesh, SectionInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntRestrict(SectionInt, PetscInt, PetscInt *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntUpdate(SectionInt, PetscInt, const PetscInt [], InsertMode);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntZero(SectionInt);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntComplete(SectionInt);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetFiberDimension(SectionInt, PetscInt, PetscInt*);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntSetFiberDimension(SectionInt, PetscInt, const PetscInt);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntSetFiberDimensionField(SectionInt, PetscInt, const PetscInt, const PetscInt);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetSize(SectionInt, PetscInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntAllocate(SectionInt);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntClear(SectionInt);

extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntAddSpace(SectionInt);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetFibration(SectionInt, const PetscInt, SectionInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntSet(SectionInt, PetscInt);

extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntRestrictClosure(SectionInt, Mesh, PetscInt, PetscInt, PetscInt []);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionIntUpdateClosure(SectionInt, Mesh, PetscInt, PetscInt [], InsertMode);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexSectionInt(Mesh, const char[], PetscInt, SectionInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetCellSectionInt(Mesh, const char[], PetscInt, SectionInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshHasSectionInt(Mesh, const char [], PetscBool  *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetSectionInt(Mesh, const char [], SectionInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshSetSectionInt(Mesh, SectionInt);

#if 0
/*S
  SectionPair - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionPairCreate(), SectionPairDestroy(), Mesh, MeshCreate()
S*/
typedef struct _p_SectionPair* SectionPair;

/* Logging support */
extern PetscClassId PETSCDM_DLLEXPORT SECTIONPAIR_CLASSID;

extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairCreate(MPI_Comm,SectionPair*);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairDestroy(SectionPair);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairView(SectionPair,PetscViewer);

extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairGetSection(SectionPair,ALE::Obj<PETSC_MESH_TYPE::pair_section_type>&);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairSetSection(SectionPair,const ALE::Obj<PETSC_MESH_TYPE::pair_section_type>&);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairGetBundle(SectionPair,ALE::Obj<PETSC_MESH_TYPE>&);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairSetBundle(SectionPair,const ALE::Obj<PETSC_MESH_TYPE>&);

typedef struct {
  int    i;
  double x, y, z;
} PetscPair;

extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairDistribute(SectionPair, Mesh, SectionPair *);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairRestrict(SectionPair, PetscInt, PetscPair *[]);
extern PetscErrorCode PETSCDM_DLLEXPORT SectionPairUpdate(SectionPair, PetscInt, const PetscPair []);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshHasSectionPair(Mesh, const char [], PetscBool  *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetSectionPair(Mesh, const char [], SectionPair *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshSetSectionPair(Mesh, SectionPair);
#endif

extern PetscErrorCode PETSCDM_DLLEXPORT WriteVTKHeader(PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT WriteVTKVertices(Mesh, PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT WriteVTKElements(Mesh, PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT WritePCICEVertices(Mesh, PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT WritePCICEElements(Mesh, PetscViewer);  
extern PetscErrorCode PETSCDM_DLLEXPORT WritePyLithVertices(Mesh, PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT WritePyLithElements(Mesh, SectionReal, PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT WritePyLithVerticesLocal(Mesh, PetscViewer);
extern PetscErrorCode PETSCDM_DLLEXPORT WritePyLithElementsLocal(Mesh, SectionReal, PetscViewer);

extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetLabelIds(Mesh, const char[], PetscInt *);
extern PetscErrorCode PETSCDM_DLLEXPORT MeshGetLabelSize(Mesh, const char[], PetscInt *);

typedef struct {
  int           numQuadPoints, numBasisFuncs;
  const double *quadPoints, *quadWeights, *basis, *basisDer;
} PetscQuadrature;

PETSC_EXTERN_CXX_END

#endif
#endif
