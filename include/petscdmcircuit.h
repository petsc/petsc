/*
  DMCircuit, for parallel unstructured circuit problems.
*/
#if !defined(__PETSCDMCIRCUIT_H)
#define __PETSCDMCIRCUIT_H

#include <petscdm.h>

/*
  DMCircuitComponentGenericDataType - This is the data type that PETSc uses for storing the component data.
            For compatibility with PetscSF, which is used for data distribution, its declared as PetscInt.
	    To get the user-specific data type, one needs to cast it to the appropriate type.
*/
typedef PetscInt DMCircuitComponentGenericDataType;

PETSC_EXTERN PetscErrorCode DMCircuitCreate(MPI_Comm, DM*);
PETSC_EXTERN PetscErrorCode DMCircuitSetSizes(DM, PetscInt, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMCircuitSetEdgeList(DM, int[]);
PETSC_EXTERN PetscErrorCode DMCircuitLayoutSetUp(DM);
PETSC_EXTERN PetscErrorCode DMCircuitRegisterComponent(DM, const char*, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode DMCircuitGetVertexRange(DM, PetscInt*, PetscInt*);
PETSC_EXTERN PetscErrorCode DMCircuitGetEdgeRange(DM, PetscInt*, PetscInt*);
PETSC_EXTERN PetscErrorCode DMCircuitAddComponent(DM, PetscInt, PetscInt, void*);
PETSC_EXTERN PetscErrorCode DMCircuitGetNumComponents(DM, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode DMCircuitGetComponentTypeOffset(DM, PetscInt, PetscInt, PetscInt*, PetscInt*);
PETSC_EXTERN PetscErrorCode DMCircuitGetVariableOffset(DM, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode DMCircuitGetVariableGlobalOffset(DM, PetscInt, PetscInt*);
PETSC_EXTERN PetscErrorCode DMCircuitAddNumVariables(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMCircuitSetNumVariables(DM, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode DMCircuitGetComponentDataArray(DM, DMCircuitComponentGenericDataType**);
PETSC_EXTERN PetscErrorCode DMCircuitDistribute(DM, DM*);
PETSC_EXTERN PetscErrorCode DMCircuitGetSupportingEdges(DM, PetscInt, PetscInt*, const PetscInt*[]);
PETSC_EXTERN PetscErrorCode DMCircuitGetConnectedNodes(DM, PetscInt, const PetscInt*[]);
PETSC_EXTERN PetscErrorCode DMCircuitIsGhostVertex(DM, PetscInt, PetscBool*);


#endif
