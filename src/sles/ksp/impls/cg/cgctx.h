/*  
    Private Context Structure for Conjugate Gradient 
*/

#if !defined(__CG)
#define __CG

typedef struct {
  double emin,emax;
  double *e,*d,*ee,*dd;       /* work space for running Lanczo algorithm*/
} CGCntx;

#endif
