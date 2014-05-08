#if !defined(_PETSCFVTYPES_H)
#define _PETSCFVTYPES_H

/*S
  PetscLimiter - PETSc object that manages a finite volume slope limiter

  Level: intermediate

  Concepts: finite volume, limiter

.seealso: PetscLimiterCreate(), PetscLimiterSetType(), PetscLimiterType
S*/
typedef struct _p_PetscLimiter *PetscLimiter;

/*S
  PetscFV - PETSc object that manages a finite volume discretization

  Level: intermediate

  Concepts: finite volume

.seealso: PetscFVCreate(), PetscFVSetType(), PetscFVType
S*/
typedef struct _p_PetscFV *PetscFV;

#endif
