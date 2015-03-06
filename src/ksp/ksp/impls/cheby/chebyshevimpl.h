/*
    Private data structure for Chebyshev Iteration
*/

#if !defined(__CHEBY)
#define __CHEBY

typedef struct {
  PetscReal emin,emax;    /* store user provided estimates of extreme eigenvalues */
  KSP       kspest;       /* KSP used to estimate eigenvalues */
  PetscReal tform[4];     /* transform from Krylov estimates to Chebyshev bounds */
  PetscObjectId    amatid,    pmatid;
  PetscObjectState amatstate, pmatstate;
  PetscInt  eststeps;     /* number of est steps in KSP used to estimate eigenvalues */
  PetscRandom random;
} KSP_Chebyshev;

#endif
