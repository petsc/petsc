/*
  Defines the interface functions for the ASA geometric preconditioner
*/
#ifndef __PETSCPCASA_H
#define __PETSCPCASA_H
#include "petscksp.h"
#include "petscdm.h"
PETSC_EXTERN_CXX_BEGIN


extern PetscErrorCode  PCASASetTolerances(PC, PetscReal,PetscReal, PetscReal, PetscInt);
extern PetscErrorCode PCSolve_ASA(PC, Vec, Vec);

PETSC_EXTERN_CXX_END
#endif
