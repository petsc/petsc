#pragma once
#include <petsc/private/vecimpl.h>

typedef struct {
  PetscInt      nsubs;
  VecTagger    *subs;
  PetscCopyMode mode;
} VecTagger_AndOr;

PETSC_INTERN PetscErrorCode VecTaggerGetSubs_AndOr(VecTagger, PetscInt *, VecTagger **);
PETSC_INTERN PetscErrorCode VecTaggerSetSubs_AndOr(VecTagger, PetscInt, VecTagger *, PetscCopyMode);
PETSC_INTERN PetscErrorCode VecTaggerCreate_AndOr(VecTagger);
PETSC_INTERN PetscErrorCode VecTaggerAndOrIsSubBox_Private(PetscInt, const VecTaggerBox *, const VecTaggerBox *, PetscBool *);
PETSC_INTERN PetscErrorCode VecTaggerAndOrIntersect_Private(PetscInt, const VecTaggerBox *, const VecTaggerBox *, VecTaggerBox *, PetscBool *);
