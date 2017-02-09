#if !defined(VECTAGGERANDOR_H)
#define      VECTAGGERANDOR_H
#include <petsc/private/vecimpl.h>

typedef struct {
  PetscInt      nsubs;
  VecTagger     *subs;
  PetscCopyMode mode;
} VecTagger_AndOr;

PETSC_EXTERN PetscErrorCode VecTaggerGetSubs_AndOr(VecTagger,PetscInt*,VecTagger**);
PETSC_EXTERN PetscErrorCode VecTaggerSetSubs_AndOr(VecTagger,PetscInt,VecTagger*,PetscCopyMode);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_AndOr(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerAndOrIsSubinterval_Private(PetscInt,PetscScalar (*)[2],PetscScalar (*)[2],PetscBool *);
PETSC_EXTERN PetscErrorCode VecTaggerAndOrIntersect_Private(PetscInt,PetscScalar (*)[2],PetscScalar (*)[2],PetscScalar (*)[2],PetscBool *);
#endif

