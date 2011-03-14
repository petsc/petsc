
/*  
    Private Krylov Context Structure (KSP) for the Anderson mixing method aka nonlinear Krylov applied to a linear equation

*/

#if !defined(__NGMRESIMPL_H)
#define __NGMRESIMPL_H

/*
        Defines the basic KSP object
*/
#include <private/kspimpl.h>

typedef struct {
  Vec       *v,*w;
  PetscReal *f2;     /* 2-norms of function (residual) at each stage */
  PetscInt  msize;   /* maximum size of space */
  PetscInt  csize;   /* current size of space */
} KSP_NGMRES;

#endif
