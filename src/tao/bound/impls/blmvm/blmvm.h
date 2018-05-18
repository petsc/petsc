/*
  Private data structure used for blmvm method.
*/

#if !defined(__TAO_BLMVM_H)
#define __TAO_BLMVM_H

#include <petsc/private/taoimpl.h>

/*
 Context for limited memory variable metric method for bound constrained
 optimization.
*/
typedef struct {
  Mat M, Mscale;

  Vec unprojected_gradient;
  Vec red_grad;
  Vec Xold, Gold;
  Vec W, work;
  
  PetscInt  as_type;
  PetscReal as_step, as_tol;
  IS active_lower, active_upper, active_fixed;
  IS inactive_idx, active_idx;

  PetscInt n_free;
  PetscInt n_bind;

  PetscInt bfgs;
  PetscInt grad;
  Mat      H0;
  
  PetscBool recycle, no_scale;
} TAO_BLMVM;

#endif  /* if !defined(__TAO_BLMVM_H) */
