#if !defined(GARBAGECOLLECTOR_H)
  #define GARBAGECOLLECTOR_H

  #include <petsc/private/hashmapobj.h>
  #include <petscsys.h>

typedef union _PetscGarbage
{
  PetscHMapObj map;
  void        *ptr;
} PetscGarbage;

PETSC_EXTERN PetscErrorCode PetscObjectDelayedDestroy(PetscObject *);
PETSC_EXTERN void           PetscGarbageKeySortedIntersect(void *, void *, PetscMPIInt *, MPI_Datatype *);
PETSC_EXTERN PetscErrorCode PetscGarbageCleanup(MPI_Comm);
PETSC_EXTERN PetscErrorCode PetscGarbageView(MPI_Comm, PetscViewer);

PETSC_EXTERN PetscErrorCode GarbageKeyAllReduceIntersect_Private(MPI_Comm, PetscInt64 *, PetscInt *);

#endif
