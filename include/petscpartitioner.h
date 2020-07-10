#if !defined(PETSCPARTITIONER_H)
#define PETSCPARTITIONER_H

#include <petscsection.h>

/*S
  PetscPartitioner - PETSc object that manages a graph partitioner

  Level: intermediate

.seealso: PetscPartitionerCreate(), PetscPartitionerSetType(), PetscPartitionerType
S*/
typedef struct _p_PetscPartitioner *PetscPartitioner;

PETSC_EXTERN PetscClassId PETSCPARTITIONER_CLASSID;
PETSC_EXTERN PetscErrorCode PetscPartitionerInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscPartitionerFinalizePackage(void);

/*J
  PetscPartitionerType - String with the name of a PETSc graph partitioner

  Level: beginner

.seealso: PetscPartitionerSetType(), PetscPartitioner
J*/
typedef const char *PetscPartitionerType;
#define PETSCPARTITIONERPARMETIS "parmetis"
#define PETSCPARTITIONERPTSCOTCH "ptscotch"
#define PETSCPARTITIONERCHACO    "chaco"
#define PETSCPARTITIONERSIMPLE   "simple"
#define PETSCPARTITIONERSHELL    "shell"
#define PETSCPARTITIONERGATHER   "gather"

PETSC_EXTERN PetscFunctionList PetscPartitionerList;
PETSC_EXTERN PetscErrorCode PetscPartitionerRegister(const char[], PetscErrorCode (*)(PetscPartitioner));

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate(MPI_Comm, PetscPartitioner*);
PETSC_EXTERN PetscErrorCode PetscPartitionerDestroy(PetscPartitioner*);
PETSC_EXTERN PetscErrorCode PetscPartitionerSetType(PetscPartitioner, PetscPartitionerType);
PETSC_EXTERN PetscErrorCode PetscPartitionerGetType(PetscPartitioner, PetscPartitionerType*);
PETSC_EXTERN PetscErrorCode PetscPartitionerSetUp(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerReset(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerSetFromOptions(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerViewFromOptions(PetscPartitioner, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscPartitionerView(PetscPartitioner, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscPartitionerPartition(PetscPartitioner, PetscInt, PetscInt, PetscInt[], PetscInt[], PetscSection, PetscSection, PetscSection, IS*);

PETSC_EXTERN PetscErrorCode PetscPartitionerShellSetPartition(PetscPartitioner, PetscInt, const PetscInt[], const PetscInt[]);
PETSC_EXTERN PetscErrorCode PetscPartitionerShellSetRandom(PetscPartitioner, PetscBool);
PETSC_EXTERN PetscErrorCode PetscPartitionerShellGetRandom(PetscPartitioner, PetscBool*);

/* We should implement MatPartitioning with PetscPartitioner */
#include <petscmat.h>
#define PETSCPARTITIONERMATPARTITIONING "matpartitioning"
PETSC_EXTERN PetscErrorCode PetscPartitionerMatPartitioningGetMatPartitioning(PetscPartitioner, MatPartitioning*);

#endif
