/*
  Mesh object, for easy parallelism of simple unstructured distributed mesh problems.
*/
#if !defined(__PETSCMESH_H)
#define __PETSCMESH_H
#include <Mesh.hh>
#include "petscda.h"

/*S
     Mesh - Abstract PETSc object that manages distributed field data for a Sieve.

   Level: beginner

  Concepts: distributed array

.seealso:  DACreate1d(), DACreate2d(), DACreate3d(), DADestroy(), VecScatter, DACreate(), VecPackCreate(), VecPack
S*/
typedef struct _p_Mesh* Mesh;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView(Mesh,PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreate(MPI_Comm,Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDestroy(Mesh);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateGlobalVector(Mesh,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetMatrix(Mesh, MatType,Mat*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalIndices(Mesh,PetscInt*[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetPreallocation(Mesh,PetscInt,const PetscInt[],PetscInt,const PetscInt[]);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetGhosts(Mesh,PetscInt,PetscInt,PetscInt,const PetscInt[]);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateVector(ALE::Obj<ALE::Two::Mesh>,ALE::Obj<ALE::Two::Mesh::bundle_type>,Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetGlobalScatter(ALE::Obj<ALE::Two::Mesh>,const char [],Vec,VecScatter *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetMesh(Mesh,ALE::Obj<ALE::Two::Mesh> *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetMesh(Mesh,ALE::Obj<ALE::Two::Mesh>);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDistribute(Mesh);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshUnify(Mesh, Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT restrictVector(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVectorComplete(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleMatrix(Mat, PetscInt, PetscScalar [], InsertMode);
#endif
