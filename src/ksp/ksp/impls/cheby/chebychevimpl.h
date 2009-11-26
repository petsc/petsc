/*  
    Private data structure for Chebychev Iteration 
*/

#if !defined(__CHEBY)
#define __CHEBY

typedef struct {
  PetscReal emin,emax;   /* store user provided estimates of extreme eigenvalues */
} KSP_Chebychev;

#endif
