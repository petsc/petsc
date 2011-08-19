/*  
    Private data structure for Chebychev Iteration 
*/

#if !defined(__CHEBY)
#define __CHEBY

typedef struct {
  PetscReal emin,emax;   /* store user provided estimates of extreme eigenvalues */
  KSP kspest;            /* KSP used to estimate eigenvalues */
  PC  pcnone;            /* Dummy PC to drop in so PCSetFromOptions doesn't get called extra times */
  PetscReal tform[4];    /* transform from Krylov estimates to Chebychev bounds */
  PetscBool estimate_current;
} KSP_Chebychev;

#endif
