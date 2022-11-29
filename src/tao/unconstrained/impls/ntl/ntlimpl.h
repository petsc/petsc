/*
  Context for a Newton trust-region, line-search method for unconstrained
  minimization
*/

#ifndef __TAO_NTL_H
#define __TAO_NTL_H
#include <petsc/private/taoimpl.h>

typedef struct {
  Mat M;
  PC  bfgs_pre;

  Vec W;
  Vec Xold;
  Vec Gold;

  /* Parameters when updating the trust-region radius based on steplength

     if   step < nu1            (very bad step)
       radius = omega1 * min(norm(d), radius)
     elif step < nu2            (bad step)
       radius = omega2 * min(norm(d), radius)
     elif step < nu3            (okay step)
       radius = omega3 * radius;
     elif step < nu4            (good step)
       radius = max(omega4 * norm(d), radius)
     else                       (very good step)
       radius = max(omega5 * norm(d), radius)
     fi
  */

  PetscReal nu1; /* used to compute trust-region radius */
  PetscReal nu2; /* used to compute trust-region radius */
  PetscReal nu3; /* used to compute trust-region radius */
  PetscReal nu4; /* used to compute trust-region radius */

  PetscReal omega1; /* factor used for trust-region update */
  PetscReal omega2; /* factor used for trust-region update */
  PetscReal omega3; /* factor used for trust-region update */
  PetscReal omega4; /* factor used for trust-region update */
  PetscReal omega5; /* factor used for trust-region update */

  /* Parameters when updating the trust-region radius based on reduction

     kappa = ared / pred
     if   kappa < eta1          (very bad step)
       radius = alpha1 * min(norm(d), radius)
     elif kappa < eta2          (bad step)
       radius = alpha2 * min(norm(d), radius)
     elif kappa < eta3          (okay step)
       radius = alpha3 * radius;
     elif kappa < eta4          (good step)
       radius = max(alpha4 * norm(d), radius)
     else                       (very good step)
       radius = max(alpha5 * norm(d), radius)
     fi
  */

  PetscReal eta1; /* used to compute trust-region radius */
  PetscReal eta2; /* used to compute trust-region radius */
  PetscReal eta3; /* used to compute trust-region radius */
  PetscReal eta4; /* used to compute trust-region radius */

  PetscReal alpha1; /* factor used for trust-region update */
  PetscReal alpha2; /* factor used for trust-region update */
  PetscReal alpha3; /* factor used for trust-region update */
  PetscReal alpha4; /* factor used for trust-region update */
  PetscReal alpha5; /* factor used for trust-region update */

  /* Parameters when updating the trust-region radius based on interpolation
     kappa = ared / pred
     if   kappa >= 1.0 - mu1    (very good step)
       choose tau in [gamma3, gamma4]
       radius = max(tau * norm(d), radius)
     elif kappa >= 1.0 - mu2    (good step)
       choose tau in [gamma2, gamma3]
       if (tau >= 1.0)
         radius = max(tau * norm(d), radius)
       else
         radius = tau * min(norm(d), radius)
       fi
     else                       (bad step)
       choose tau in [gamma1, 1.0]
       radius = tau * min(norm(d), radius)
     fi
  */

  PetscReal mu1; /* used for model agreement in interpolation */
  PetscReal mu2; /* used for model agreement in interpolation */

  PetscReal gamma1; /* factor used for interpolation */
  PetscReal gamma2; /* factor used for interpolation */
  PetscReal gamma3; /* factor used for interpolation */
  PetscReal gamma4; /* factor used for interpolation */

  PetscReal theta; /* factor used for interpolation */

  /* Parameters when initializing trust-region radius based on interpolation */
  PetscReal mu1_i; /* used for model agreement in interpolation */
  PetscReal mu2_i; /* used for model agreement in interpolation */

  PetscReal gamma1_i; /* factor used for interpolation */
  PetscReal gamma2_i; /* factor used for interpolation */
  PetscReal gamma3_i; /* factor used for interpolation */
  PetscReal gamma4_i; /* factor used for interpolation */

  PetscReal theta_i; /* factor used for interpolation */

  /* Other parameters */
  PetscReal min_radius; /* lower bound on initial radius value */
  PetscReal max_radius; /* upper bound on trust region radius */
  PetscReal epsilon;    /* tolerance used when computing ared/pred */

  PetscInt ntrust; /* Trust-region steps accepted */
  PetscInt newt;   /* Newton directions attempted */
  PetscInt bfgs;   /* BFGS directions attempted */
  PetscInt grad;   /* Gradient directions attempted */

  PetscInt init_type;   /* Trust-region initialization method */
  PetscInt update_type; /* Trust-region update method */
} TAO_NTL;

#endif /* if !defined(__TAO_NTL_H) */
