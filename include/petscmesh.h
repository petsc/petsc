/*
  Mesh object, for easy parallelism of simple unstructured distributed mesh problems.
*/
#if !defined(__PETSCMESH_H)
#define __PETSCMESH_H
#include <Mesh.hh>
#include "petscda.h"
PETSC_EXTERN_CXX_BEGIN

/*S
   Mesh - Abstract PETSc object that combines a topology (Sieve) and coordinates (Section).

   Level: beginner

  Concepts: distributed array

.seealso:  MeshCreate(), MeshDestroy(), Section, SectionCreate()
S*/
typedef struct _p_Mesh* Mesh;

/* Logging support */
extern PetscCookie PETSCDM_DLLEXPORT MESH_COOKIE;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshFinalize();

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView(Mesh,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT FieldView(Mesh,const char[],PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreate(MPI_Comm,Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDestroy(Mesh);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalVector(Mesh,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh, MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexMatrix(Mesh, MatType, Mat *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalIndices(Mesh,PetscInt*[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalScatter(ALE::Mesh*,const char [],Vec,VecScatter *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh,ALE::Obj<ALE::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh,const ALE::Obj<ALE::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDistribute(Mesh, Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshUnify(Mesh, Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT restrictVector(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVectorComplete(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleMatrix(Mat, PetscInt, PetscScalar [], InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT preallocateMatrix(ALE::Mesh *, const ALE::Obj<ALE::Mesh::real_section_type>&, const ALE::Obj<ALE::Mesh::order_type>&, Mat);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePCICE(MPI_Comm, const int, const char[], const char[], const char[], const int, const int, Mesh *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinates(Mesh, PetscTruth, PetscInt *, PetscInt *, PetscReal *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetElements(Mesh, PetscTruth, PetscInt *, PetscInt *, PetscInt *[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscScalar *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT BCSectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscInt *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT BCFUNCGetArray(Mesh, PetscInt *, PetscInt *, PetscScalar *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePCICERestart(Mesh, PetscViewer);

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

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetSection(SectionReal,ALE::Obj<ALE::Mesh::real_section_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetSection(SectionReal,const ALE::Obj<ALE::Mesh::real_section_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetTopology(SectionReal,ALE::Obj<ALE::Mesh::topology_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetTopology(SectionReal,const ALE::Obj<ALE::Mesh::topology_type>&);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealDistribute(SectionReal, Mesh, SectionReal *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealRestrict(SectionReal, PetscInt, PetscScalar *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealUpdate(SectionReal, PetscInt, const PetscScalar []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealUpdateAdd(SectionReal, PetscInt, const PetscScalar []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionRealComplete(SectionReal);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexSectionReal(Mesh, PetscInt, SectionReal *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetCellSectionReal(Mesh, PetscInt, SectionReal *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshHasSectionReal(Mesh, const char [], PetscTruth *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetSectionReal(Mesh, const char [], SectionReal *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetSectionReal(Mesh, SectionReal);

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
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetTopology(SectionInt,ALE::Obj<ALE::Mesh::topology_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionIntSetTopology(SectionInt,const ALE::Obj<ALE::Mesh::topology_type>&);

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
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairGetTopology(SectionPair,ALE::Obj<ALE::Mesh::topology_type>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionPairSetTopology(SectionPair,const ALE::Obj<ALE::Mesh::topology_type>&);

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

EXTERN PetscErrorCode PETSCDM_DLLEXPORT WriteVTKHeader(PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WriteVTKVertices(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WriteVTKElements(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePCICEVertices(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePCICEElements(Mesh, PetscViewer);  
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithVertices(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithElements(Mesh, SectionReal, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithVerticesLocal(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithElementsLocal(Mesh, SectionReal, PetscViewer);

PETSC_EXTERN_CXX_END
#endif
