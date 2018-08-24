/*
    Context for bound-constrained nonlinear conjugate gradient method
 */


#ifndef __TAO_BNCG_H
#define __TAO_BNCG_H

#include <petsc/private/taoimpl.h>

typedef struct {
  Mat B;
  Mat pc;
  Vec G_old, X_old, W, work;
  Vec g_work, y_work, d_work;
  Vec sk, yk;
  Vec unprojected_gradient, unprojected_gradient_old;
  Vec inactive_grad, inactive_step;

  IS  active_lower, active_upper, active_fixed, active_idx, inactive_idx, inactive_old, new_inactives;

  PetscReal alpha;                   /* convex ratio in the scalar scaling */
  PetscReal hz_theta;
  PetscReal xi;                      /* Parameter for KD, DK, and HZ methods. */
  PetscReal theta;                   /* The convex combination parameter in the SSML_Broyden method. */
  PetscReal zeta;                    /* Another parameter, exclusive to Kou-Dai, modifying the y_k scalar contribution */
  PetscReal hz_eta, dk_eta;
  PetscReal bfgs_scale, dfp_scale;   /* Scaling of the bfgs/dfp tau parameter in the bfgs and broyden methods. Default 1. */
  PetscReal tau_bfgs, tau_dfp;
  PetscReal as_step, as_tol, yts, yty, sts;
  PetscReal f, delta_min, delta_max;
  PetscReal epsilon;                 /* Machine precision unless changed by user */
  PetscReal eps_23;                  /*  Two-thirds power of machine precision */

  PetscInt cg_type;                  /*  Formula to use */
  PetscInt  min_restart_num;         /* Restarts every x*n iterations, where n is the dimension */
  PetscInt ls_fails, resets, descent_error, skipped_updates, pure_gd_steps;
  PetscInt iter_quad, min_quad;      /* Dynamic restart variables in Dai-Kou, SIAM J. Optim. Vol 23, pp. 296-320, Algorithm 4.1 */
  PetscInt  as_type;

  PetscBool recycle;
  PetscBool inv_sig;
  PetscReal tol_quad;                /* tolerance for Dai-Kou dynamic restart */
  PetscBool dynamic_restart;         /* Keeps track of whether or not to do a dynamic (KD) restart */
  PetscBool spaced_restart;          /* If true, restarts the CG method every x*n iterations */
  PetscBool use_dynamic_restart;
  PetscBool neg_xi;
  PetscBool unscaled_restart;        /* Gradient descent restarts are done without rescaling*/
  PetscBool diag_scaling;
  PetscBool no_scaling;

} TAO_BNCG;

#endif /* ifndef __TAO_BNCG_H */

PETSC_INTERN PetscErrorCode TaoBNCGEstimateActiveSet(Tao, PetscInt);
PETSC_INTERN PetscErrorCode TaoBNCGBoundStep(Tao, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode TaoBNCGSetRecycleFlag(Tao, PetscBool);
PETSC_INTERN PetscErrorCode TaoBNCGComputeScalarScaling(PetscReal, PetscReal, PetscReal, PetscReal*, PetscReal);
PETSC_INTERN PetscErrorCode TaoBNCGConductIteration(Tao, PetscReal);
PETSC_INTERN PetscErrorCode TaoBNCGStepDirectionUpdate(Tao, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscBool);
PETSC_INTERN PetscErrorCode TaoBNCGComputeDiagScaling(Tao, PetscReal, PetscReal);
PETSC_INTERN PetscErrorCode TaoBNCGResetUpdate(Tao, PetscReal);
PETSC_INTERN PetscErrorCode TaoBNCGCheckDynamicRestart(Tao, PetscReal, PetscReal, PetscReal, PetscBool*, PetscReal);
