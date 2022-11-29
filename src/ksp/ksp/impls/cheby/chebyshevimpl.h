/*
    Private data structure for Chebyshev Iteration
*/

#ifndef PETSC_CHEBYSHEVIMPL_H
#define PETSC_CHEBYSHEVIMPL_H

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal emin, emax;                   /* store user provided estimates of extreme eigenvalues or computed with kspest and transformed with tform[] */
  PetscReal emin_computed, emax_computed; /* eigenvalues as computed by kspest, if computed */
  PetscReal emin_provided, emax_provided; /* provided by PCGAMG; discarded unless preconditioned by Jacobi */
  KSP       kspest;                       /* KSP used to estimate eigenvalues */
  PetscReal tform[4];                     /* transform from Krylov estimates to Chebyshev bounds */
  PetscInt  eststeps;                     /* number of kspest steps in KSP used to estimate eigenvalues */
  PetscBool usenoisy;                     /* use noisy right hand side vector to estimate eigenvalues */
  /* For tracking when to update the eigenvalue estimates */
  PetscObjectId    amatid, pmatid;
  PetscObjectState amatstate, pmatstate;
} KSP_Chebyshev;

#endif // PETSC_CHEBYSHEVIMPL_H
