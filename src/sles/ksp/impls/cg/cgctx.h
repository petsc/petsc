/*  
    Private Context Structure for Conjugate Gradient 
*/

#if !defined(__CG)
#define __CG

#include "petsc.h"
#include "kspimpl.h"

typedef struct {
  Scalar emin, emax;           /* eigenvalues */
  Scalar *e, *d, *ee, *dd;     /* work space for Lanczos algorithm */
  CGType type;                 /* type of system for complex case (symmetric or Hermitian) */
} KSP_CG;

#endif
