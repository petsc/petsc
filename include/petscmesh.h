/*
  Mesh object, for easy parallelism of simple unstructured distributed mesh problems.
*/
#if !defined(__PETSCMESH_H)
#define __PETSCMESH_H
#include <Mesh.hh>
#include "petscda.h"
PETSC_EXTERN_CXX_BEGIN

/*S
     Mesh - Abstract PETSc object that manages distributed field data for a Sieve.

   Level: beginner

  Concepts: distributed array

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), VecScatter, DACreate(), VecPackCreate(), VecPack
S*/
typedef struct _p_Mesh* Mesh;

/* Logging support */
extern PetscCookie PETSCDM_DLLEXPORT MESH_COOKIE;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView(Mesh,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT FieldView(Mesh,const char[],PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreate(MPI_Comm,Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDestroy(Mesh);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalVector(Mesh,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh, MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalIndices(Mesh,PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetPreallocation(Mesh,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetGhosts(Mesh,PetscInt,PetscInt,PetscInt,const PetscInt[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateVector(ALE::Obj<ALE::Mesh>,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalScatter(ALE::Mesh*,const char [],Vec,VecScatter *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh,ALE::Obj<ALE::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh,const ALE::Obj<ALE::Mesh>&);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDistribute(Mesh, Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshUnify(Mesh, Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT restrictVector(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVectorComplete(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleMatrix(Mat, PetscInt, PetscScalar [], InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT preallocateMatrix(ALE::Mesh *, const ALE::Obj<ALE::Mesh::section_type>&, const ALE::Obj<ALE::Mesh::order_type>&, Mat);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePCICE(MPI_Comm, const int, const char[], const char[], const char[], const int, const int, Mesh *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinates(Mesh, PetscTruth, PetscInt *, PetscInt *, PetscReal *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetElements(Mesh, PetscTruth, PetscInt *, PetscInt *, PetscInt *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT VertexSectionCreate(Mesh, const char [], PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT CellSectionCreate(Mesh, const char [], PetscInt);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT SectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscScalar *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT BCSectionGetArray(Mesh, const char [], PetscInt *, PetscInt *, PetscInt *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT BCFUNCGetArray(Mesh, PetscInt *, PetscInt *, PetscScalar *[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePCICERestart(Mesh, PetscViewer);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT WriteVTKHeader(PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WriteVTKVertices(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WriteVTKElements(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePCICEVertices(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePCICEElements(Mesh, PetscViewer);  
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithVertices(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithElements(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithVerticesLocal(Mesh, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT WritePyLithElementsLocal(Mesh, PetscViewer);

PETSC_EXTERN_CXX_END
#endif
