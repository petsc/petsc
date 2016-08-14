/*
    Private data structure for Chebyshev Iteration
*/

#if !defined(__CHEBY)
#define __CHEBY

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal        emin,emax;    /* store user provided estimates of extreme eigenvalues */
  KSP              kspest;       /* KSP used to estimate eigenvalues */
  PetscReal        tform[4];     /* transform from Krylov estimates to Chebyshev bounds */
  PetscInt         eststeps;     /* number of kspest steps in KSP used to estimate eigenvalues */
  PetscBool        usenoisy;    /* use noisy right hand side vector to estimate eigenvalues */
  /* For tracking when to update the eigenvalue estimates */
  PetscObjectId    amatid,    pmatid;
  PetscObjectState amatstate, pmatstate;
} KSP_Chebyshev;

#endif
