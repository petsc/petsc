#pragma once

#include <petscsystypes.h>

PetscErrorCode testExplicitSynopsis(PetscInt, PetscReal, void *);

extern void ExternHeaderFunctionShouldNotGetStatic(void);

// clang-format off
PETSC_EXTERN       void         ExternHeaderBadFormattingShouldNotGetStatic            (    void    )   ;
// clang-format on

PETSC_EXTERN char *PetscExternHeaderPointerShouldNotGetStatic();
PETSC_EXTERN char *PetscInternHeaderPointerShouldNotGetStatic();

// clang-format off
PETSC_EXTERN           char *PetscExternHeaderPointerBadFormattingShouldNotGetStatic          (  ) ;
PETSC_EXTERN char *          PetscInternHeaderPointerBadFormattingShouldNotGetStatic       ( )    ;
// clang-format on
