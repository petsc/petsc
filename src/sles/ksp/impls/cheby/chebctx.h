/*$Id: chebctx.h,v 1.4 1999/11/24 21:54:53 bsmith Exp $*/
/*  
    Private data structure for Chebychev Iteration 
*/

#if !defined(__CHEBY)
#define __CHEBY

typedef struct {
  PetscReal emin,emax;   /* store user provided estimates of extreme eigenvalues */
} KSP_Chebychev;

#endif
