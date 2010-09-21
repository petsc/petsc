/*
  Context for a Newton trust region method (unconstrained minimization)
*/

#ifndef __TAO_NTR_H
#define __TAO_NTR_H
#include "include/private/taosolver_impl.h"

typedef struct {
  Mat M;

  Vec D;
  Vec W;

  Vec Diag;
  PetscScalar radius,trust0;
  // Parameters when updating the trust-region radius based on reduction
  PetscScalar eta1;          // used to compute trust-region radius
  PetscScalar eta2;          // used to compute trust-region radius
  PetscScalar eta3;          // used to compute trust-region radius
  PetscScalar eta4;          // used to compute trust-region radius

  PetscScalar alpha1;	// factor used for trust-region update
  PetscScalar alpha2;	// factor used for trust-region update
  PetscScalar alpha3;	// factor used for trust-region update
  PetscScalar alpha4;	// factor used for trust-region update
  PetscScalar alpha5;	// factor used for trust-region update

  // kappa = ared / pred
  // if   kappa < eta1          (very bad step)
  //   radius = alpha1 * min(norm(d), radius)
  // elif kappa < eta2          (bad step)
  //   radius = alpha2 * min(norm(d), radius)
  // elif kappa < eta3          (okay step)
  //   radius = alpha3 * radius;
  // elif kappa < eta4          (good step)
  //   radius = max(alpha4 * norm(d), radius)
  // else                       (very good step)
  //   radius = max(alpha5 * norm(d), radius)
  // fi

  // Parameters when updating the trust-region radius based on interpolation
  PetscScalar mu1;		// used for model agreement in radius update
  PetscScalar mu2;		// used for model agreement in radius update

  PetscScalar gamma1;	// factor used for radius update
  PetscScalar gamma2;	// factor used for radius update
  PetscScalar gamma3;	// factor used for radius update
  PetscScalar gamma4;	// factor used for radius update

  PetscScalar theta;		// factor used for radius update

  // kappa = ared / pred
  // if   kappa >= 1.0 - mu1    (very good step)
  //   choose tau in [gamma3, gamma4]
  //   radius = max(tau * norm(d), radius)
  // elif kappa >= 1.0 - mu2    (good step)
  //   choose tau in [gamma2, gamma3]
  //   if (tau >= 1.0)
  //     radius = max(tau * norm(d), radius)
  //   else
  //     radius = tau * min(norm(d), radius)
  //   fi
  // else                       (bad step)
  //   choose tau in [gamma1, 1.0]
  //   radius = tau * min(norm(d), radius)
  // fi

  // Parameters when initializing trust-region radius based on interpolation
  PetscScalar mu1_i;         // used for model agreement in interpolation
  PetscScalar mu2_i;         // used for model agreement in interpolation

  PetscScalar gamma1_i;      // factor used for interpolation
  PetscScalar gamma2_i;      // factor used for interpolation
  PetscScalar gamma3_i;      // factor used for interpolation
  PetscScalar gamma4_i;      // factor used for interpolation

  PetscScalar theta_i;       // factor used for interpolation

  PetscScalar min_radius;	// lower bound on initial radius value
  PetscScalar max_radius;	// upper bound on trust region radius
  PetscScalar epsilon;	// tolerance used when computing actred/prered

  PetscInt ksp_type;         // KSP method for the code
  PetscInt pc_type;          // Preconditioner for the code
  PetscInt bfgs_scale_type;  // Scaling matrix for the bfgs preconditioner
  PetscInt init_type;	// Trust-region initialization method
  PetscInt update_type;	// Trust-region update method
} TAO_NTR;

#endif /* ifndef __TAO_NTR_H */
