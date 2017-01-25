#if !defined(VECTAGGERSIMPLE_H)
#define      VECTAGGERSIMPLE_H
#include <petsc/private/vecimpl.h>

typedef struct {
  PetscScalar (*interval)[2];
} VecTagger_Simple;

PETSC_EXTERN PetscErrorCode VecTaggerDestroy_Simple(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerSetFromOptions_Simple(PetscOptionItems *,VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerSetUp_Simple(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerView_Simple(VecTagger,PetscViewer);
PETSC_EXTERN PetscErrorCode VecTaggerSetInterval_Simple(VecTagger,PetscScalar (*interval)[2]);
PETSC_EXTERN PetscErrorCode VecTaggerGetInterval_Simple(VecTagger,const PetscScalar (**interval)[2]);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_Simple(VecTagger);
#endif
