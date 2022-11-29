/*
Context for a Newton line search method (unconstrained minimization)
*/

#ifndef __TAO_NLS_H
#define __TAO_NLS_H
#include <petsc/private/taoimpl.h>

typedef struct {
  Mat M;
  PC  bfgs_pre;

  Vec D;
  Vec W;

  Vec Xold;
  Vec Gold;

  /* Parameters when updating the perturbation added to the Hessian matrix
     according to the following scheme:

     pert = sval;

     do until convergence
       shift Hessian by pert
       solve Newton system

       if (linear solver failed or did not compute a descent direction)
         use steepest descent direction and increase perturbation

         if (0 == pert)
           initialize perturbation
           pert = min(imax, max(imin, imfac * norm(G)))
         else
           increase perturbation
           pert = min(pmax, max(pgfac * pert, pmgfac * norm(G)))
         fi
       else
         use linear solver direction and decrease perturbation

         pert = min(psfac * pert, pmsfac * norm(G))
         if (pert < pmin)
           pert = 0
         fi
       fi

       perform line search
       function and gradient evaluation
       check convergence
     od
  */
  PetscReal sval; /*  Starting perturbation value, default zero */

  PetscReal imin;  /*  Minimum perturbation added during initialization  */
  PetscReal imax;  /*  Maximum perturbation added during initialization */
  PetscReal imfac; /*  Merit function factor during initialization */

  PetscReal pmin;   /*  Minimim perturbation value */
  PetscReal pmax;   /*  Maximum perturbation value */
  PetscReal pgfac;  /*  Perturbation growth factor */
  PetscReal psfac;  /*  Perturbation shrink factor */
  PetscReal pmgfac; /*  Merit function growth factor */
  PetscReal pmsfac; /*  Merit function shrink factor */

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
  PetscReal nu1; /*  used to compute trust-region radius */
  PetscReal nu2; /*  used to compute trust-region radius */
  PetscReal nu3; /*  used to compute trust-region radius */
  PetscReal nu4; /*  used to compute trust-region radius */

  PetscReal omega1; /*  factor used for trust-region update */
  PetscReal omega2; /*  factor used for trust-region update */
  PetscReal omega3; /*  factor used for trust-region update */
  PetscReal omega4; /*  factor used for trust-region update */
  PetscReal omega5; /*  factor used for trust-region update */

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
  PetscReal eta1; /*  used to compute trust-region radius */
  PetscReal eta2; /*  used to compute trust-region radius */
  PetscReal eta3; /*  used to compute trust-region radius */
  PetscReal eta4; /*  used to compute trust-region radius */

  PetscReal alpha1; /*  factor used for trust-region update */
  PetscReal alpha2; /*  factor used for trust-region update */
  PetscReal alpha3; /*  factor used for trust-region update */
  PetscReal alpha4; /*  factor used for trust-region update */
  PetscReal alpha5; /*  factor used for trust-region update */

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
  PetscReal mu1; /*  used for model agreement in interpolation */
  PetscReal mu2; /*  used for model agreement in interpolation */

  PetscReal gamma1; /*  factor used for interpolation */
  PetscReal gamma2; /*  factor used for interpolation */
  PetscReal gamma3; /*  factor used for interpolation */
  PetscReal gamma4; /*  factor used for interpolation */

  PetscReal theta; /*  factor used for interpolation */

  /*  Parameters when initializing trust-region radius based on interpolation */
  PetscReal mu1_i; /*  used for model agreement in interpolation */
  PetscReal mu2_i; /*  used for model agreement in interpolation */

  PetscReal gamma1_i; /*  factor used for interpolation */
  PetscReal gamma2_i; /*  factor used for interpolation */
  PetscReal gamma3_i; /*  factor used for interpolation */
  PetscReal gamma4_i; /*  factor used for interpolation */

  PetscReal theta_i; /*  factor used for interpolation */

  /*  Other parameters */
  PetscReal min_radius; /*  lower bound on initial radius value */
  PetscReal max_radius; /*  upper bound on trust region radius */
  PetscReal epsilon;    /*  tolerance used when computing ared/pred */

  PetscInt newt; /*  Newton directions attempted */
  PetscInt bfgs; /*  BFGS directions attempted */
  PetscInt grad; /*  Gradient directions attempted */

  PetscInt init_type;   /*  Trust-region initialization method */
  PetscInt update_type; /*  Trust-region update method */

  PetscInt ksp_atol;
  PetscInt ksp_rtol;
  PetscInt ksp_ctol;
  PetscInt ksp_negc;
  PetscInt ksp_dtol;
  PetscInt ksp_iter;
  PetscInt ksp_othr;
} TAO_NLS;

#endif /* if !defined(__TAO_NLS_H) */
