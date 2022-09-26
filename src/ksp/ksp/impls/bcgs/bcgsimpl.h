/*
   Private data structure used by the BCGS method. This data structure
  must be identical to the beginning of the KSP_FBCGS and KSP_QMRCGS data structure
  so if you CHANGE anything here you must also change it there.
*/
#ifndef PETSC_BCGSIMPL_H
#define PETSC_BCGSIMPL_H

#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

typedef struct {
  Vec guess; /* if using right preconditioning with nonzero initial guess must keep that around to "fix" solution */
} KSP_BCGS;

PETSC_INTERN PetscErrorCode KSPSetFromOptions_BCGS(KSP, PetscOptionItems *PetscOptionsObject);
PETSC_INTERN PetscErrorCode KSPSetUp_BCGS(KSP);
PETSC_INTERN PetscErrorCode KSPSolve_BCGS(KSP);
PETSC_INTERN PetscErrorCode KSPBuildSolution_BCGS(KSP, Vec, Vec *);
PETSC_INTERN PetscErrorCode KSPReset_BCGS(KSP);
PETSC_INTERN PetscErrorCode KSPDestroy_BCGS(KSP);

#endif // PETSC_BCGSIMPL_H
