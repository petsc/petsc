/*
   Private data structure for ILU/ICC/LU/Cholesky preconditioners.
*/
#ifndef __FACTOR_H
#define __FACTOR_H

#include <petsc/private/pcimpl.h>

typedef struct {
  Mat             fact; /* factored matrix */
  MatFactorInfo   info;
  MatOrderingType ordering; /* matrix reordering */
  char           *solvertype;
  MatFactorType   factortype;
  PetscReal       actualfill;
  PetscBool       inplace;       /* flag indicating in-place factorization */
  PetscBool       reuseordering; /* reuses previous reordering computed */
  PetscBool       reusefill;     /* reuse fill from previous LU */
} PC_Factor;

PETSC_INTERN PetscErrorCode PCFactorInitialize(PC, MatFactorType);
PETSC_INTERN PetscErrorCode PCFactorGetMatrix_Factor(PC, Mat *);

PETSC_INTERN PetscErrorCode PCFactorSetZeroPivot_Factor(PC, PetscReal);
PETSC_INTERN PetscErrorCode PCFactorGetZeroPivot_Factor(PC, PetscReal *);
PETSC_INTERN PetscErrorCode PCFactorSetShiftType_Factor(PC, MatFactorShiftType);
PETSC_INTERN PetscErrorCode PCFactorGetShiftType_Factor(PC, MatFactorShiftType *);
PETSC_INTERN PetscErrorCode PCFactorSetShiftAmount_Factor(PC, PetscReal);
PETSC_INTERN PetscErrorCode PCFactorGetShiftAmount_Factor(PC, PetscReal *);
PETSC_INTERN PetscErrorCode PCFactorSetDropTolerance_Factor(PC, PetscReal, PetscReal, PetscInt);
PETSC_INTERN PetscErrorCode PCFactorSetFill_Factor(PC, PetscReal);
PETSC_INTERN PetscErrorCode PCFactorSetMatOrderingType_Factor(PC, MatOrderingType);
PETSC_INTERN PetscErrorCode PCFactorGetLevels_Factor(PC, PetscInt *);
PETSC_INTERN PetscErrorCode PCFactorSetLevels_Factor(PC, PetscInt);
PETSC_INTERN PetscErrorCode PCFactorSetAllowDiagonalFill_Factor(PC, PetscBool);
PETSC_INTERN PetscErrorCode PCFactorGetAllowDiagonalFill_Factor(PC, PetscBool *);
PETSC_INTERN PetscErrorCode PCFactorSetPivotInBlocks_Factor(PC, PetscBool);
PETSC_INTERN PetscErrorCode PCFactorSetMatSolverType_Factor(PC, MatSolverType);
PETSC_INTERN PetscErrorCode PCFactorSetUpMatSolverType_Factor(PC);
PETSC_INTERN PetscErrorCode PCFactorGetMatSolverType_Factor(PC, MatSolverType *);
PETSC_INTERN PetscErrorCode PCFactorSetColumnPivot_Factor(PC, PetscReal);
PETSC_INTERN PetscErrorCode PCSetFromOptions_Factor(PC, PetscOptionItems *PetscOptionsObject);
PETSC_INTERN PetscErrorCode PCView_Factor(PC, PetscViewer);
PETSC_INTERN PetscErrorCode PCFactorSetDefaultOrdering_Factor(PC);
PETSC_INTERN PetscErrorCode PCFactorClearComposedFunctions(PC);

#endif
