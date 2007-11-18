
/*  
    Private Krylov Context Structure (KSP) for LCD 

    This one is very simple. It contains a flag indicating the symmetry 
   structure of the matrix and work space for (optionally) computing
   eigenvalues.

*/

#if !defined(__LCDCTX_H)
#define __LCDCTX_H

/*
        Defines the basic KSP object
*/
#include "include/private/kspimpl.h"

/*
    The field should remain the same since it is shared by the BiCG code
*/

typedef struct {

  PetscInt restart;
  PetscInt max_iters;
  PetscReal haptol;
  Vec *P;
  Vec *Q;
}KSP_LCD;

#endif
