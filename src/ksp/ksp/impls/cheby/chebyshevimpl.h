/*  
    Private data structure for Chebyshev Iteration 
*/

#if !defined(__CHEBY)
#define __CHEBY

typedef struct {
  PetscReal emin,emax;   /* store user provided estimates of extreme eigenvalues */
  KSP       kspest;      /* KSP used to estimate eigenvalues */
  PC        pcnone;      /* Dummy PC to drop in so PCSetFromOptions doesn't get called extra times */
  PetscReal tform[4];    /* transform from Krylov estimates to Chebyshev bounds */
  PetscBool estimate_current;
  PetscBool hybrid;      /* flag for using Hybrid Chebyshev */
  PetscInt  chebysteps;  /* number of Chebyshev steps in Hybrid Chebyshev */
  PetscInt  its;         /* total hybrid iterations, used to determine when to call GMRES step in hybrid impl */          
  PetscInt  purification;/* the hybrid method uses the GMRES steps to imporve the approximate solution 
                            purification <= 0: no purification
                                         = 1: purification only for new matrix (See case cheb->its = 0 in KSPSolve_Chebyshev())
                                         >1 : purification */
} KSP_Chebyshev;

#endif
