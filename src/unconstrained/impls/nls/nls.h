/* 
Context for a Newton line search method (unconstrained minimization)
*/

#ifndef __TAO_NLS_H
#define __TAO_NLS_H
#include "include/private/taosolver_impl.h"

typedef struct {
  Mat M;

  Vec G;
  Vec D;
  Vec W;

  Vec Xold;
  Vec Gold;
  Vec Diag;

  // Parameters when updating the perturbation added to the Hessian matrix
  PetscScalar sval;          // Starting perturbation value, default zero
                        
  PetscScalar imin;          // Minimum perturbation added during initialization 
  PetscScalar imax;          // Maximum perturbation added during initialization
  PetscScalar imfac;         // Merit function factor during initialization

  PetscScalar pmin;          // Minimim perturbation value
  PetscScalar pmax;          // Maximum perturbation value
  PetscScalar pgfac;         // Perturbation growth factor
  PetscScalar psfac;         // Perturbation shrink factor
  PetscScalar pmgfac;        // Merit function growth factor
  PetscScalar pmsfac;        // Merit function shrink factor

  // The perturbation to the Hessian matrix is initialized and updated
  // according to the following scheme:
  //
  //   pert = sval;
  //
  //   do until convergence
  //     shift Hessian by pert
  //     solve Newton system
  //
  //     if (linear solver failed or did not compute a descent direction)
  //       use steepest descent direction and increase perturbation
  //
  //       if (0 == pert)
  //         initialize perturbation
  //         pert = min(imax, max(imin, imfac * norm(G)))
  //       else
  //         increase perturbation
  //         pert = min(pmax, max(pgfac * pert, pmgfac * norm(G)))
  //       fi
  //     else
  //       use linear solver direction and decrease perturbation
  //
  //       pert = min(psfac * pert, pmsfac * norm(G))
  //       if (pert < pmin)
  //         pert = 0
  //       fi
  //     fi
  //
  //     perform line search
  //     function and gradient evaluation
  //     check convergence
  //   od

  // Parameters when updating the trust-region radius based on steplength
  PetscScalar nu1;		// used to compute trust-region radius
  PetscScalar nu2;		// used to compute trust-region radius
  PetscScalar nu3;		// used to compute trust-region radius
  PetscScalar nu4;		// used to compute trust-region radius

  PetscScalar omega1;        // factor used for trust-region update
  PetscScalar omega2;        // factor used for trust-region update
  PetscScalar omega3;        // factor used for trust-region update
  PetscScalar omega4;        // factor used for trust-region update
  PetscScalar omega5;        // factor used for trust-region update

  // if   step < nu1  		(very bad step)
  //   radius = omega1 * min(norm(d), radius)
  // elif step < nu2		(bad step)
  //   radius = omega2 * min(norm(d), radius)
  // elif step < nu3		(okay step)
  //   radius = omega3 * radius;
  // elif step < nu4		(good step)
  //   radius = max(omega4 * norm(d), radius)
  // else 			(very good step)
  //   radius = max(omega5 * norm(d), radius)
  // fi
 
  // Parameters when updating the trust-region radius based on reduction
  PetscScalar eta1;		// used to compute trust-region radius
  PetscScalar eta2;		// used to compute trust-region radius
  PetscScalar eta3;		// used to compute trust-region radius
  PetscScalar eta4;		// used to compute trust-region radius

  PetscScalar alpha1;        // factor used for trust-region update
  PetscScalar alpha2;        // factor used for trust-region update
  PetscScalar alpha3;        // factor used for trust-region update
  PetscScalar alpha4;        // factor used for trust-region update
  PetscScalar alpha5;        // factor used for trust-region update

  // kappa = ared / pred
  // if   kappa < eta1 		(very bad step)
  //   radius = alpha1 * min(norm(d), radius)
  // elif kappa < eta2		(bad step)
  //   radius = alpha2 * min(norm(d), radius)
  // elif kappa < eta3		(okay step)
  //   radius = alpha3 * radius;
  // elif kappa < eta4		(good step)
  //   radius = max(alpha4 * norm(d), radius)
  // else 			(very good step)
  //   radius = max(alpha5 * norm(d), radius)
  // fi
 
  // Parameters when updating the trust-region radius based on interpolation
  PetscScalar mu1;		// used for model agreement in interpolation
  PetscScalar mu2;		// used for model agreement in interpolation

  PetscScalar gamma1;	// factor used for interpolation
  PetscScalar gamma2;	// factor used for interpolation
  PetscScalar gamma3;	// factor used for interpolation
  PetscScalar gamma4;	// factor used for interpolation

  PetscScalar theta;		// factor used for interpolation

  // kappa = ared / pred
  // if   kappa >= 1.0 - mu1	(very good step)
  //   choose tau in [gamma3, gamma4]
  //   radius = max(tau * norm(d), radius)
  // elif kappa >= 1.0 - mu2    (good step)
  //   choose tau in [gamma2, gamma3]
  //   if (tau >= 1.0)
  //     radius = max(tau * norm(d), radius)
  //   else
  //     radius = tau * min(norm(d), radius)
  //   fi
  // else 			(bad step)
  //   choose tau in [gamma1, 1.0]
  //   radius = tau * min(norm(d), radius)
  // fi
 
  // Parameters when initializing trust-region radius based on interpolation
  PetscScalar mu1_i;		// used for model agreement in interpolation
  PetscScalar mu2_i;		// used for model agreement in interpolation

  PetscScalar gamma1_i;	// factor used for interpolation
  PetscScalar gamma2_i;	// factor used for interpolation
  PetscScalar gamma3_i;	// factor used for interpolation
  PetscScalar gamma4_i;	// factor used for interpolation

  PetscScalar theta_i;	// factor used for interpolation

  // Other parameters
  PetscScalar min_radius;    // lower bound on initial radius value
  PetscScalar max_radius;    // upper bound on trust region radius
  PetscScalar epsilon;       // tolerance used when computing ared/pred

  PetscInt newt;		// Newton directions attempted
  PetscInt bfgs;		// BFGS directions attempted
  PetscInt sgrad;		// Scaled gradient directions attempted
  PetscInt grad;		// Gradient directions attempted


  PetscInt ksp_type;		// KSP method for the code
  PetscInt pc_type;		// Preconditioner for the code
  PetscInt bfgs_scale_type;	// Scaling matrix to used for the bfgs preconditioner
  PetscInt init_type;	// Trust-region initialization method
  PetscInt update_type;      // Trust-region update method

  PetscInt ksp_atol;
  PetscInt ksp_rtol;
  PetscInt ksp_ctol;
  PetscInt ksp_negc;
  PetscInt ksp_dtol;
  PetscInt ksp_iter;
  PetscInt ksp_othr;
} TAO_NLS;

#endif /* ifndef __TAO_NLS_H */
