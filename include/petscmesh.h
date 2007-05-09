/*
  Mesh object, for easy parallelism of simple unstructured distributed mesh problems.
*/
#if !defined(__PETSCMESH_H)
#define __PETSCMESH_H
#include <Mesh.hh>
#include <CartesianSieve.hh>
#include "petscda.h"
PETSC_EXTERN_CXX_BEGIN

/*S
   Mesh - Abstract PETSc object that combines a topology (Sieve) and coordinates (Section).

   Level: beginner

  Concepts: distributed array

.seealso:  MeshCreate(), MeshDestroy(), Section, SectionCreate()
S*/
typedef struct _p_Mesh* Mesh;
#define MeshType const char*
#define MESHSIEVE     "sieve"
#define MESHCARTESIAN "cartesian"

/* Logging support */
extern PetscCookie PETSCDM_DLLEXPORT MESH_COOKIE;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshFinalize();

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView(Mesh,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreate(MPI_Comm,Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDestroy(Mesh);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetType(Mesh, MeshType);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalVector(Mesh,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh, MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexMatrix(Mesh, MatType, Mat *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalIndices(Mesh,PetscInt*[]);

/*MC
   MeshRegisterDynamic - Adds a type to the Mesh package.

   Synopsis:
   PetscErrorCode MeshRegisterDynamic(char *name_solver,char *path,char *name_create,PetscErrorCode (*routine_create)(Mesh))

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
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshRegisterDestroy(void);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshRegister(const char[],const char[],const char[],PetscErrorCode (*)(Mesh));

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh,ALE::Obj<ALE::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh,const ALE::Obj<ALE::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDistribute(Mesh, const char[], Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDistributeByFace(Mesh, const char[], Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGenerate(Mesh, PetscTruth, Mesh *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshRefine(Mesh, double, PetscTruth, Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshUnify(Mesh, Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetMaximumDegree(Mesh, PetscInt *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT restrictVector(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVectorComplete(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleMatrix(Mat, PetscInt, PetscScalar [], InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT preallocateMatrix(const ALE::Obj<ALE::Mesh>&, const ALE::Obj<ALE::Mesh::real_section_type::atlas_type>&, const ALE::Obj<ALE::Mesh::order_type>&, Mat);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT updateOperator(Mat, const ALE::Obj<ALE::Mesh>&, const ALE::Obj<ALE::Mesh::real_section_type>&, const ALE::Obj<ALE::Mesh::order_type>&, const ALE::Mesh::point_type&, PetscScalar [], InsertMode);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePCICE(MPI_Comm, const int, const char[], const char[], PetscTruth, const char[], Mesh *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinates(Mesh, PetscTruth, PetscInt *, PetscInt *, PetscReal *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetElements(Mesh, PetscTruth, PetscInt *, PetscInt *, PetscInt *[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePyLith(MPI_Comm, const int, const char[], PetscTruth, PetscTruth, Mesh *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscScalar *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT BCSectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscInt *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT BCSectionRealCreate(Mesh, const char [], PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT BCSectionRealGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscReal *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT BCFUNCGetArray(Mesh, PetscInt *, PetscInt *, PetscScalar *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePCICERestart(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT PCICERenumberBoundary(Mesh);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCartesianGetMesh(Mesh,ALE::Obj<ALE::CartesianMesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCartesianSetMesh(Mesh,const ALE::Obj<ALE::CartesianMesh>&);

/*S
  SectionReal - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionRealCreate(), SectionRealDestroy(), Mesh, MeshCreate()
S*/
typedef struct _p_SectionReal* SectionReal;

/* Logging support */
extern PetscCookie PETSCDM_DLLEXPORT SECTIONREAL_COOKIE;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealCreate(MPI_Comm,SectionReal*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealDestroy(SectionReal);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealView(SectionReal,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealDuplicate(SectionReal,SectionReal*);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetSection(SectionReal,ALE::Obj<ALE::Mesh::real_section_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetSection(SectionReal,const ALE::Obj<ALE::Mesh::real_section_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetBundle(SectionReal,ALE::Obj<ALE::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetBundle(SectionReal,const ALE::Obj<ALE::Mesh>&);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealDistribute(SectionReal, Mesh, SectionReal *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealRestrict(SectionReal, PetscInt, PetscScalar *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealUpdate(SectionReal, PetscInt, const PetscScalar []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealUpdateAdd(SectionReal, PetscInt, const PetscScalar []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealZero(SectionReal);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealCreateLocalVector(SectionReal, Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealToVec(SectionReal, Mesh, ScatterMode, Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealNorm(SectionReal, Mesh, NormType, PetscReal *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealComplete(SectionReal);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexSectionReal(Mesh, PetscInt, SectionReal *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetCellSectionReal(Mesh, PetscInt, SectionReal *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshHasSectionReal(Mesh, const char [], PetscTruth *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetSectionReal(Mesh, const char [], SectionReal *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetSectionReal(Mesh, SectionReal);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateMatrix(Mesh, SectionReal, MatType, Mat *);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalScatter(Mesh,VecScatter *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalScatter(Mesh,SectionReal,VecScatter *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetLocalFunction(Mesh, PetscErrorCode (*)(Mesh, SectionReal, SectionReal, void*));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetLocalJacobian(Mesh, PetscErrorCode (*)(Mesh, SectionReal, Mat, void*));
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshFormFunction(Mesh, SectionReal, SectionReal, void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshFormJacobian(Mesh, SectionReal, Mat, void*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshInterpolatePoints(Mesh, SectionReal, int, double *, double **);

/*S
  SectionInt - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionIntCreate(), SectionIntDestroy(), Mesh, MeshCreate()
S*/
typedef struct _p_SectionInt* SectionInt;

/* Logging support */
extern PetscCookie PETSCDM_DLLEXPORT SECTIONINT_COOKIE;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntCreate(MPI_Comm,SectionInt*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntDestroy(SectionInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntView(SectionInt,PetscViewer);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetSection(SectionInt,ALE::Obj<ALE::Mesh::int_section_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntSetSection(SectionInt,const ALE::Obj<ALE::Mesh::int_section_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetBundle(SectionInt,ALE::Obj<ALE::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntSetBundle(SectionInt,const ALE::Obj<ALE::Mesh>&);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntDistribute(SectionInt, Mesh, SectionInt *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntRestrict(SectionInt, PetscInt, PetscInt *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntUpdate(SectionInt, PetscInt, const PetscInt []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntUpdateAdd(SectionInt, PetscInt, const PetscInt []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntComplete(SectionInt);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexSectionInt(Mesh, PetscInt, SectionInt *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetCellSectionInt(Mesh, PetscInt, SectionInt *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshHasSectionInt(Mesh, const char [], PetscTruth *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetSectionInt(Mesh, const char [], SectionInt *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetSectionInt(Mesh, SectionInt);

#if 0
/*S
  SectionPair - Abstract PETSc object that manages distributed field data over a topology (Sieve).

  Level: beginner

  Concepts: distributed mesh, field

.seealso:  SectionPairCreate(), SectionPairDestroy(), Mesh, MeshCreate()
S*/
typedef struct _p_SectionPair* SectionPair;

/* Logging support */
extern PetscCookie PETSCDM_DLLEXPORT SECTIONPAIR_COOKIE;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairCreate(MPI_Comm,SectionPair*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairDestroy(SectionPair);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairView(SectionPair,PetscViewer);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairGetSection(SectionPair,ALE::Obj<ALE::Mesh::pair_section_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairSetSection(SectionPair,const ALE::Obj<ALE::Mesh::pair_section_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairGetBundle(SectionPair,ALE::Obj<ALE::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairSetBundle(SectionPair,const ALE::Obj<ALE::Mesh>&);

typedef struct {
  int    i;
  double x, y, z;
} PetscPair;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairDistribute(SectionPair, Mesh, SectionPair *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairRestrict(SectionPair, PetscInt, PetscPair *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairUpdate(SectionPair, PetscInt, const PetscPair []);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshHasSectionPair(Mesh, const char [], PetscTruth *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetSectionPair(Mesh, const char [], SectionPair *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetSectionPair(Mesh, SectionPair);
#endif

EXTERN PetscErrorCode PETSCDM_DLLEXPORT WriteVTKHeader(PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WriteVTKVertices(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WriteVTKElements(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePCICEVertices(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePCICEElements(Mesh, PetscViewer);  
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithVertices(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithElements(Mesh, SectionReal, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithVerticesLocal(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithElementsLocal(Mesh, SectionReal, PetscViewer);

typedef struct {
  int           numQuadPoints, numBasisFuncs;
  const double *quadPoints, *quadWeights, *basis, *basisDer;
} PetscQuadrature;

PETSC_EXTERN_CXX_END

template<typename Section> PetscErrorCode PETSCDM_DLLEXPORT MeshCreateMatrix(const ALE::Obj<ALE::Mesh>&, const ALE::Obj<Section>&, MatType, Mat *);
template<typename Section> PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalScatter(const ALE::Obj<ALE::Mesh>&, const ALE::Obj<Section>&, VecScatter *);

// Compatibility layer for PyLith 0.8
//   This wil definitely go away soon
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCompatGetMesh(Mesh,ALE::Obj<ALECompat::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCompatSetMesh(Mesh,const ALE::Obj<ALECompat::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCompatGetGlobalScatter(Mesh,VecScatter *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT preallocateMatrixCompat(const ALE::Obj<ALECompat::Mesh::topology_type>&, const ALE::Obj<ALECompat::Mesh::real_section_type::atlas_type>&, const ALE::Obj<ALECompat::Mesh::order_type>&, Mat);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCompatCreatePyLith(MPI_Comm, const int, const char[], PetscTruth, PetscTruth, Mesh *);
#endif
