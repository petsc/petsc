/*
  Defines the interface functions for the ASA geometric preconditioner
*/
#ifndef __PETSCPCASA_H
#define __PETSCPCASA_H
#include "petscksp.h"
#include "petscdm.h"


PETSC_EXTERN PetscErrorCode PCASASetTolerances(PC, PetscReal,PetscReal, PetscReal, PetscInt);
PETSC_EXTERN PetscErrorCode PCSolve_ASA(PC, Vec, Vec);

#endif
