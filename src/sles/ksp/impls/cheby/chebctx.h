/*$Id: rich.c,v 1.85 1999/11/05 14:46:46 bsmith Exp bsmith $*/
/*  
    Private data structure for Chebychev Iteration 
*/

#if !defined(__CHEBY)
#define __CHEBY

typedef struct {
  double emin,emax;   /* store user provided estimates of extreme eigenvalues */
} KSP_Chebychev;

#endif
