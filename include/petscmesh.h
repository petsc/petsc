/*
  Mesh object, for easy parallelism of simple unstructured distributed mesh problems.
*/
#if !defined(__PETSCMESH_H)
#define __PETSCMESH_H
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

#include <IndexBundle.hh>

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetTopology(Mesh, ALE::Obj<ALE::Sieve>*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetTopology(Mesh, ALE::Obj<ALE::Sieve>);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetBoundary(Mesh, ALE::Obj<ALE::Sieve>*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetBoundary(Mesh, ALE::Obj<ALE::Sieve>);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetBoundaryBundle(Mesh, ALE::Obj<ALE::IndexBundle>*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetBoundaryBundle(Mesh, ALE::Obj<ALE::IndexBundle>);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetBundle(Mesh, ALE::Obj<ALE::IndexBundle>*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetBundle(Mesh, ALE::Obj<ALE::IndexBundle>);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetVertexBundle(Mesh, ALE::Obj<ALE::IndexBundle>*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetVertexBundle(Mesh, ALE::Obj<ALE::IndexBundle>);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetElementBundle(Mesh, ALE::Obj<ALE::IndexBundle>*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetElementBundle(Mesh, ALE::Obj<ALE::IndexBundle>);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinateBundle(Mesh, ALE::Obj<ALE::IndexBundle>*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetCoordinateBundle(Mesh, ALE::Obj<ALE::IndexBundle>);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetOrientation(Mesh, ALE::Obj<ALE::PreSieve>*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetOrientation(Mesh, ALE::Obj<ALE::PreSieve>);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetCoordinates(Mesh, Vec*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetCoordinates(Mesh, Vec);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetSpaceFootprint(Mesh, ALE::Obj<ALE::Stack>*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshSetSpaceFootprint(Mesh, ALE::Obj<ALE::Stack>);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshPopulate(Mesh, int, PetscInt, PetscInt, PetscInt *, PetscScalar []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshDistribute(Mesh);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshUnify(Mesh, Mesh*);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetDimension(Mesh, PetscInt *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGetEmbeddingDimension(Mesh, PetscInt *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateBoundary(Mesh, PetscInt, PetscInt, PetscInt [], PetscScalar [], ALE::Obj<ALE::IndexBundle>*, Vec *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreateCoordinates(Mesh, PetscScalar []);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT restrictVector(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVectorComplete(Vec, Vec, InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleVector(Vec, PetscInt, PetscScalar [], InsertMode);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT assembleMatrix(Mat, PetscInt, PetscScalar [], InsertMode);

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePCICE(MPI_Comm, const char [], PetscInt, PetscTruth, Mesh *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCreatePyLith(MPI_Comm, const char [], Mesh *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshGenerate(Mesh, Mesh *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshRefine(Mesh, PetscReal, /*CoSieve*/ Vec, Mesh *);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshCoarsen(Mesh, PetscReal, /*CoSieve*/ Vec, Mesh *);
#endif
