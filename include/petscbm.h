#pragma once

#include <petscis.h>

/* MANSEC = Sys */
/* SUBMANSEC = BM */

/*S
     PetscBench - Abstract PETSc object that manages a benchmark test

   Level: intermediate

.seealso: `PetscBenchCreate()`, `PetscBenchDestroy()`, `PetscBenchSetType()`, `PetscBenchType`
S*/
typedef struct _p_PetscBench *PetscBench;

/*J
    PetscBenchType - String with the name of a PETSc benchmark test

   Level: intermediate

.seealso: `PetscBenchCreate()`, `PetscBenchDestroy()`, `PetscBenchSetType()`, `PetscBench`
J*/
typedef const char *PetscBenchType;

PETSC_EXTERN PetscClassId PetscBench_CLASSID;

PETSC_EXTERN PetscErrorCode PetscBenchInitializePackage(void);

PETSC_EXTERN PetscErrorCode PetscBenchCreate(MPI_Comm, PetscBench *);
PETSC_EXTERN PetscErrorCode PetscBenchSetFromOptions(PetscBench);
PETSC_EXTERN PetscErrorCode PetscBenchSetUp(PetscBench);
PETSC_EXTERN PetscErrorCode PetscBenchRun(PetscBench);
PETSC_EXTERN PetscErrorCode PetscBenchReset(PetscBench);
PETSC_EXTERN PetscErrorCode PetscBenchSetOptionsPrefix(PetscBench, const char[]);
PETSC_EXTERN PetscErrorCode PetscBenchView(PetscBench, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscBenchViewFromOptions(PetscBench, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscBenchDestroy(PetscBench *);
PETSC_EXTERN PetscErrorCode PetscBenchSetType(PetscBench, PetscBenchType);
PETSC_EXTERN PetscErrorCode PetscBenchGetType(PetscBench, PetscBenchType *);
PETSC_EXTERN PetscErrorCode PetscBenchRegister(const char[], PetscErrorCode (*)(PetscBench));
PETSC_EXTERN PetscErrorCode PetscBenchSetSize(PetscBench, PetscInt);
PETSC_EXTERN PetscErrorCode PetscBenchGetSize(PetscBench, PetscInt *);
