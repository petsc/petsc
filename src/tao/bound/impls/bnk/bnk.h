/*
Context for bounded Newton-Krylov type optimization algorithms
*/

#if !defined(__TAO_BNK_H)
#define __TAO_BNK_H
#include <petsc/private/taoimpl.h>
#include <../src/tao/matrix/lmvmmat.h>

typedef struct {
  Mat H_inactive, Hpre_inactive, M;
  Vec W, Xwork, Gwork, inactive_work;
  Vec Xold, Gold, Diag;
  Vec unprojected_gradient, unprojected_gradient_old;
  Vec inactive_gradient, inactive_step;
  IS  inactive_idx, inactive_old;
  
  /* Scalar values for the solution and step */
  PetscReal fold, f, gnorm, dnorm;

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
  PetscReal sval;               /*  Starting perturbation value, default zero */

  PetscReal imin;               /*  Minimum perturbation added during initialization  */
  PetscReal imax;               /*  Maximum perturbation added during initialization */
  PetscReal imfac;              /*  Merit function factor during initialization */

  PetscReal pert;               /*  Current perturbation value */
  PetscReal pmin;               /*  Minimim perturbation value */
  PetscReal pmax;               /*  Maximum perturbation value */
  PetscReal pgfac;              /*  Perturbation growth factor */
  PetscReal psfac;              /*  Perturbation shrink factor */
  PetscReal pmgfac;             /*  Merit function growth factor */
  PetscReal pmsfac;             /*  Merit function shrink factor */

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
  PetscReal nu1;                /*  used to compute trust-region radius */
  PetscReal nu2;                /*  used to compute trust-region radius */
  PetscReal nu3;                /*  used to compute trust-region radius */
  PetscReal nu4;                /*  used to compute trust-region radius */

  PetscReal omega1;             /*  factor used for trust-region update */
  PetscReal omega2;             /*  factor used for trust-region update */
  PetscReal omega3;             /*  factor used for trust-region update */
  PetscReal omega4;             /*  factor used for trust-region update */
  PetscReal omega5;             /*  factor used for trust-region update */

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
  PetscReal eta1;               /*  used to compute trust-region radius */
  PetscReal eta2;               /*  used to compute trust-region radius */
  PetscReal eta3;               /*  used to compute trust-region radius */
  PetscReal eta4;               /*  used to compute trust-region radius */

  PetscReal alpha1;             /*  factor used for trust-region update */
  PetscReal alpha2;             /*  factor used for trust-region update */
  PetscReal alpha3;             /*  factor used for trust-region update */
  PetscReal alpha4;             /*  factor used for trust-region update */
  PetscReal alpha5;             /*  factor used for trust-region update */

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
  PetscReal mu1;                /*  used for model agreement in interpolation */
  PetscReal mu2;                /*  used for model agreement in interpolation */

  PetscReal gamma1;             /*  factor used for interpolation */
  PetscReal gamma2;             /*  factor used for interpolation */
  PetscReal gamma3;             /*  factor used for interpolation */
  PetscReal gamma4;             /*  factor used for interpolation */

  PetscReal theta;              /*  factor used for interpolation */

  /*  Parameters when initializing trust-region radius based on interpolation */
  PetscReal mu1_i;              /*  used for model agreement in interpolation */
  PetscReal mu2_i;              /*  used for model agreement in interpolation */

  PetscReal gamma1_i;           /*  factor used for interpolation */
  PetscReal gamma2_i;           /*  factor used for interpolation */
  PetscReal gamma3_i;           /*  factor used for interpolation */
  PetscReal gamma4_i;           /*  factor used for interpolation */

  PetscReal theta_i;            /*  factor used for interpolation */

  /*  Other parameters */
  PetscReal min_radius;         /*  lower bound on initial radius value */
  PetscReal max_radius;         /*  upper bound on trust region radius */
  PetscReal epsilon;            /*  tolerance used when computing ared/pred */

  PetscInt newt;                /*  Newton directions attempted */
  PetscInt bfgs;                /*  BFGS directions attempted */
  PetscInt sgrad;               /*  Scaled gradient directions attempted */
  PetscInt grad;                /*  Gradient directions attempted */

  PetscInt pc_type;             /*  Preconditioner for the code */
  PetscInt bfgs_scale_type;     /*  Scaling matrix to used for the bfgs preconditioner */
  PetscInt init_type;           /*  Trust-region initialization method */
  PetscInt update_type;         /*  Trust-region update method */

  PetscInt ksp_atol;
  PetscInt ksp_rtol;
  PetscInt ksp_ctol;
  PetscInt ksp_negc;
  PetscInt ksp_dtol;
  PetscInt ksp_iter;
  PetscInt ksp_othr;
  PetscBool is_nash, is_stcg, is_gltr;
} TAO_BNK;

#endif /* if !defined(__TAO_BNK_H) */

#define BNK_NEWTON              0
#define BNK_BFGS                1
#define BNK_SCALED_GRADIENT     2
#define BNK_GRADIENT            3

#define BNK_PC_NONE     0
#define BNK_PC_AHESS    1
#define BNK_PC_BFGS     2
#define BNK_PC_PETSC    3
#define BNK_PC_TYPES    4

#define BFGS_SCALE_AHESS        0
#define BFGS_SCALE_PHESS        1
#define BFGS_SCALE_BFGS         2
#define BFGS_SCALE_TYPES        3

#define BNK_INIT_CONSTANT         0
#define BNK_INIT_DIRECTION        1
#define BNK_INIT_INTERPOLATION    2
#define BNK_INIT_TYPES            3

#define BNK_UPDATE_STEP           0
#define BNK_UPDATE_REDUCTION      1
#define BNK_UPDATE_INTERPOLATION  2
#define BNK_UPDATE_TYPES          3

static const char *BNK_PC[64] = {"none", "ahess", "bfgs", "petsc"};

static const char *BFGS_SCALE[64] = {"ahess", "phess", "bfgs"};

static const char *BNK_INIT[64] = {"constant", "direction", "interpolation"};

static const char *BNK_UPDATE[64] = {"step", "reduction", "interpolation"};

PETSC_INTERN PetscErrorCode TaoCreate_BNK(Tao);

PETSC_INTERN PetscErrorCode MatLMVMSolveShell(PC, Vec, Vec);
PETSC_INTERN PetscErrorCode TaoBNKInitialize(Tao);
PETSC_INTERN PetscErrorCode TaoBNKComputeStep(Tao, PetscBool, PetscInt*);
PETSC_INTERN PetscErrorCode TaoBNKPerformLineSearch(Tao, PetscInt, PetscReal*, TaoLineSearchConvergedReason*);
PETSC_INTERN PetscErrorCode TaoBNKUpdateTrustRadius(Tao, PetscReal, PetscReal, PetscInt, PetscBool*);
