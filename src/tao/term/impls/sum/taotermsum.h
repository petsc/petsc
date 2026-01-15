#pragma once
#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

#define TaoTermSumGetSubVec(params, sub_params, is_dummy, i) ((!(params) || ((is_dummy) && ((is_dummy)[(i)]))) ? NULL : (sub_params)[(i)])

PETSC_INTERN PetscErrorCode TaoTermSumVecNestGetSubVecsRead(Vec params, PetscInt *n, Vec **subparams, PetscBool **is_dummy);
PETSC_INTERN PetscErrorCode TaoTermSumVecNestRestoreSubVecsRead(Vec params, PetscInt *n, Vec **subparams, PetscBool **is_dummy);
PETSC_INTERN PetscErrorCode TaoTermViewSumPrintSubterm(TaoTerm, PetscViewer, Vec, PetscInt, PetscBool, PetscBool, const char[], const char[], const char[], const char[]);
PETSC_INTERN PetscErrorCode TaoTermViewSumPrintMapName(PetscViewer, Mat, PetscInt, const char[], PetscBool);
