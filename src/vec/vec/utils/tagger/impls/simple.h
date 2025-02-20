#pragma once
#include <petsc/private/vecimpl.h>

typedef struct {
  VecTaggerBox *box;
} VecTagger_Simple;

PETSC_INTERN PetscErrorCode VecTaggerSetFromOptions_Simple(VecTagger, PetscOptionItems);
PETSC_INTERN PetscErrorCode VecTaggerView_Simple(VecTagger, PetscViewer);
PETSC_INTERN PetscErrorCode VecTaggerSetBox_Simple(VecTagger, VecTaggerBox *);
PETSC_INTERN PetscErrorCode VecTaggerGetBox_Simple(VecTagger, const VecTaggerBox **);
PETSC_INTERN PetscErrorCode VecTaggerCreate_Simple(VecTagger);
