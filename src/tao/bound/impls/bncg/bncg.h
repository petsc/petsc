/*
    Context for bound-constrained nonlinear conjugate gradient method
 */


#ifndef __TAO_BNCG_H
#define __TAO_BNCG_H

#include <petsc/private/taoimpl.h>

typedef struct {
  Vec G_old, X_old, U, V, W, work;
  Vec g_work, y_work, d_work;
  Vec sk, yk;
  Vec steffnm1, steffn, steffnp1;
  Vec steffva, steffvatmp, steffm1va, steffm2va;
  Vec unprojected_gradient, unprojected_gradient_old;
  IS  active_lower, active_upper, active_fixed, active_idx, inactive_idx, inactive_old, new_inactives;
  Vec inactive_grad, inactive_step;
  Vec diag; /* might be unused. Might have to use if Alp's approach with inverses isn't right */
  Vec invD;
  Vec invDnew;
  PetscInt  as_type;
  PetscReal as_step, as_tol, yts, yty, sts;

  PetscReal f;
  PetscReal rho, pow;
  PetscReal eta;         /*  Restart tolerance */
  PetscReal epsilon;     /*  Machine precision  */
  PetscReal eps_23;      /*  Two-thirds power of machine precision */
  PetscBool recycle;
  PetscBool inv_sig;
  PetscInt cg_type;           /*  Formula to use */
  PetscReal theta;        /* The convex combination parameter in the SSML_Broyden method. */
  PetscReal gamma;          /* Stabilization parameter for multiple BNCG methods */
  PetscReal zeta;         /* Another parameter, exclusive to Kou-Dai, modifying the y_k scalar contribution */
  PetscReal bfgs_scale;   /* Scaling of the bfgs tau parameter in the bfgs and broyden methods. Default 1. */
  PetscReal tau_bfgs, tau_dfp;

  Vec bfgs_work;
  PetscReal dfp_scale;    /* Scaling of the dfp tau parameter in the dfp and broyden methods. Default 1. */
  Vec dfp_work;

  PetscInt ls_fails, resets, broken_ortho, descent_error;
  PetscInt iter_quad, min_quad;   /* Dynamic restart variables in Dai-Kou, SIAM J. Optim. Vol 23, pp. 296-320, Algorithm 4.1 */
  PetscReal tol_quad;           /* tolerance for Dai-Kou dynamic restart */
  PetscBool dynamic_restart;    /* Keeps track of whether or not to do a dynamic (KD) restart */
  PetscBool use_steffenson;     /* In development - attempting to use vector-based steffenson acceleration to the fixed point */
  PetscBool spaced_restart;     /* If true, restarts the CG method every x*n iterations */
  PetscBool use_dynamic_restart;
  PetscInt  min_restart_num;    /* Restarts every x*n iterations, where n is the dimension */
  PetscBool neg_xi;
  PetscBool unscaled_restart;
  PetscBool diag_scaling;
  PetscReal alpha; /* convex ratio in the scaling */
  PetscReal beta; /* Exponential factor in the diagonal scaling */
  PetscReal hz_theta;
  PetscReal xi; /* Parameter for KD, DK, and HZ methods. */
} TAO_BNCG;

#endif /* ifndef __TAO_BNCG_H */

PETSC_INTERN PetscErrorCode TaoBNCGEstimateActiveSet(Tao, PetscInt);
PETSC_INTERN PetscErrorCode TaoBNCGBoundStep(Tao, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode TaoBNCGSetRecycleFlag(Tao, PetscBool);
PETSC_INTERN PetscErrorCode TaoBNCGComputeScalarScaling(PetscReal, PetscReal, PetscReal, PetscReal*, PetscReal);
PETSC_INTERN PetscErrorCode TaoBNCGConductIteration(Tao, PetscReal); 
PETSC_INTERN PetscErrorCode TaoBNCGStepDirectionUpdate(Tao, PetscReal, PetscReal, PetscReal, PetscReal, PetscBool, PetscReal, PetscReal);
PETSC_INTERN PetscErrorCode TaoBNCGComputeDiagScaling(Tao, PetscReal, PetscReal);
PETSC_INTERN PetscErrorCode TaoBNCGResetUpdate(Tao, PetscReal);
PETSC_INTERN PetscErrorCode TaoBNCGCheckDynamicRestart(Tao, PetscReal, PetscReal, PetscReal, PetscBool*, PetscReal);
PETSC_INTERN PetscErrorCode TaoBNCGSteffensonAcceleration(Tao);
