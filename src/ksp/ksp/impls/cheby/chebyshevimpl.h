/*
    Private data structure for Chebyshev Iteration
*/

#if !defined(__CHEBY)
#define __CHEBY

typedef struct {
  PetscReal        emin,emax;    /* store user provided estimates of extreme eigenvalues */
  KSP              kspest;       /* KSP used to estimate eigenvalues */
  PetscReal        tform[4];     /* transform from Krylov estimates to Chebyshev bounds */
  PetscInt         eststeps;     /* number of kspest steps in KSP used to estimate eigenvalues */
  PetscBool        userandom;    /* use random right hand side vector to estimate eigenvalues */
  PetscRandom      random;
  /* For tracking when to update the eigenvalue estimates */
  PetscObjectId    amatid,    pmatid;
  PetscObjectState amatstate, pmatstate;
} KSP_Chebyshev;

#endif
