#if !defined(VECTAGGERSIMPLE_H)
#define      VECTAGGERSIMPLE_H
#include <petsc/private/vecimpl.h>

typedef struct {
  VecTaggerBox *box;
} VecTagger_Simple;

PETSC_EXTERN PetscErrorCode VecTaggerDestroy_Simple(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerSetFromOptions_Simple(PetscOptionItems *,VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerSetUp_Simple(VecTagger);
PETSC_EXTERN PetscErrorCode VecTaggerView_Simple(VecTagger,PetscViewer);
PETSC_EXTERN PetscErrorCode VecTaggerSetBox_Simple(VecTagger,VecTaggerBox *);
PETSC_EXTERN PetscErrorCode VecTaggerGetBox_Simple(VecTagger,const VecTaggerBox **);
PETSC_EXTERN PetscErrorCode VecTaggerCreate_Simple(VecTagger);
#endif
