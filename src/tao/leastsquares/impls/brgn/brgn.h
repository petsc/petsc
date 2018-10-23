/*
Context for Bounded Regularized Gauss-Newton algorithm
*/

#if !defined(__TAO_BRGN_H)
#define __TAO_BRGN_H

#include <../src/tao/bound/impls/bnk/bnk.h>

typedef struct {
  Mat J, H;
  Vec x_old, x_work, r_work, diag;
  Tao subsolver, parent;
  PetscReal lambda, epsilon;
} TAO_BRGN;

#endif /* if !defined(__TAO_BRGN_H) */
