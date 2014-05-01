/*
    Private data structure for Chebyshev Iteration
*/

#if !defined(__CHEBY)
#define __CHEBY

typedef struct {
  PetscReal emin,emax;    /* store user provided estimates of extreme eigenvalues */
  KSP       kspest;       /* KSP used to estimate eigenvalues */
  PC        pcnone;       /* Dummy PC to drop in so PCSetFromOptions doesn't get called extra times */
  PetscReal tform[4];     /* transform from Krylov estimates to Chebyshev bounds */
  PetscBool estimate_current;
  PetscInt  eststeps;     /* number of est steps in KSP used to estimate eigenvalues */
  PetscRandom random;
} KSP_Chebyshev;

#endif
