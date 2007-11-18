/*
  Defines the interface functions for the ASA geometric preconditioner
*/
#ifndef __PETSCDMMG_H
#define __PETSCDMMG_H
#include "petscksp.h"
#include "petscda.h"
PETSC_EXTERN_CXX_BEGIN


EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASASetDM(PC, DM);
EXTERN PetscErrorCode PETSCKSP_DLLEXPORT PCASASetTolerances(PC, PetscReal,PetscReal, PetscReal, PetscInt);
EXTERN PetscErrorCode PCSolve_ASA(PC, Vec, Vec);

PETSC_EXTERN_CXX_END
#endif
