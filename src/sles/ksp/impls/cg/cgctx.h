/*  
    Private Context Structure for Conjugate Gradient 
*/

#if !defined(__CG)
#define __CG

typedef struct {
  Scalar emin,emax;
  Scalar *e,*d,*ee,*dd;       /* work space for running Lanczo algorithm*/
} KSP_CG;

#endif
