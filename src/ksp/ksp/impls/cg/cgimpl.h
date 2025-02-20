/*
    Private Krylov Context Structure (KSP) for Conjugate Gradient

    This one is very simple. It contains a flag indicating the symmetry
   structure of the matrix and work space for (optionally) computing
   eigenvalues.
*/

#pragma once

/*
        Defines the basic KSP object
*/
#include <petsc/private/kspimpl.h>

PETSC_INTERN PetscErrorCode KSPDestroy_CG(KSP);
PETSC_INTERN PetscErrorCode KSPReset_CG(KSP);
PETSC_INTERN PetscErrorCode KSPView_CG(KSP, PetscViewer);
PETSC_INTERN PetscErrorCode KSPSetFromOptions_CG(KSP, PetscOptionItems);
PETSC_INTERN PetscErrorCode KSPCGSetType_CG(KSP, KSPCGType);

/*
    This struct is shared by several KSP implementations
*/

typedef struct {
  KSPCGType type; /* type of system (symmetric or Hermitian) */

  // The following arrays are of size ksp->maxit
  PetscScalar *e, *d;
  PetscReal   *ee, *dd; /* work space for Lanczos algorithm */

  /* Trust region support */
  PetscReal radius;
  PetscReal obj;
  PetscReal obj_min;

  PetscBool singlereduction; /* use variant of CG that combines both inner products */
} KSP_CG;
