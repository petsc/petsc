/*
Context for Bounded Regularized Gauss-Newton algorithm
*/

#if !defined(__TAO_BRGN_H)
#define __TAO_BRGN_H

#include <../src/tao/bound/impls/bnk/bnk.h>

typedef struct {
  Mat J, H, D;  /* XH: gn->J is not used?, D matrix added for ||D*x||_1 */
  Vec x_old, x_work, r_work, diag, y, y_work;  /* XH: Dx_work added, Dx_work=D*x whose dimension maybe different from x and r_work*/
  Tao subsolver, parent;
  PetscReal lambda, epsilon; /* lambda is regularizer weight for both L2-norm Gaussian-Newton and L1-norm, ||x||_1 is approximated with sum(sqrt(x.^2+epsilon^2)-epsilon)*/
} TAO_BRGN;

#endif /* if !defined(__TAO_BRGN_H) */
