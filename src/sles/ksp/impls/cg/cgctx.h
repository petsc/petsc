/* "$Id: cgctx.h,v 1.9 1999/01/31 16:08:45 bsmith Exp bsmith $"; */

/*  
    Private Krylov Context Structure (KSP) for Conjugate Gradient 

    This one is very simple. It contains a flag indicating the symmetry 
   structure of the matrix and work space for (optionally) computing
   eigenvalues.

*/

#if !defined(__CGCTX_H)
#define __CGCTX_H

/*
        Defines the basic KSP object
*/
#include "src/sles/ksp/kspimpl.h"

/*
    The field should remain the same since it is shared by the BiCG code
*/

typedef struct {
  KSPCGType type;                 /* type of system (symmetric or Hermitian) */
  Scalar    emin,emax;           /* eigenvalues */
  Scalar    *e,*d;
  double    *ee,*dd;             /* work space for Lanczos algorithm */
} KSP_CG;

#endif
