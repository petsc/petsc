
/*
    Private Krylov Context Structure (KSP) for Conjugate Gradient

    This one is very simple. It contains a flag indicating the symmetry
   structure of the matrix and work space for (optionally) computing
   eigenvalues.

*/

#ifndef __CGIMPL_H
#define __CGIMPL_H

/*
        Defines the basic KSP object
*/
#include <petsc/private/kspimpl.h>

PETSC_INTERN PetscErrorCode KSPDestroy_CG(KSP);
PETSC_INTERN PetscErrorCode KSPReset_CG(KSP);
PETSC_INTERN PetscErrorCode KSPView_CG(KSP, PetscViewer);
PETSC_INTERN PetscErrorCode KSPSetFromOptions_CG(KSP, PetscOptionItems *PetscOptionsObject);
PETSC_INTERN PetscErrorCode KSPCGSetType_CG(KSP, KSPCGType);

/*
    The field should remain the same since it is shared by the BiCG code
*/

typedef struct {
  KSPCGType   type;       /* type of system (symmetric or Hermitian) */
  PetscScalar emin, emax; /* eigenvalues */
  // The following arrays are of size ksp->maxit
  PetscScalar *e, *d;
  PetscReal   *ee, *dd; /* work space for Lanczos algorithm */

  PetscBool singlereduction; /* use variant of CG that combines both inner products */
} KSP_CG;

#endif
