/*  
    Private Context Structure for Conjugate Gradient 
*/

#if !defined(__CG)
#define __CG

#include "petsc.h"
#include "src/ksp/kspimpl.h"

/*
    The field should remain the same since it is shared by the BiCG code
*/

typedef struct {
  KSPCGType type;                 /* type of system (symmetric or Hermitian) */
  Scalar    emin, emax;           /* eigenvalues */
  Scalar    *e, *d;
  double    *ee, *dd;             /* work space for Lanczos algorithm */
} KSP_CG;

#endif
