#include "src/matrix/lmvmmat.h"
#include "ntr.h"

#include "petscksp.h"
#include "petscpc.h"
#include "private/kspimpl.h"
#include "private/pcimpl.h"

#define NTR_KSP_NASH    0
#define NTR_KSP_STCG    1
#define NTR_KSP_GLTR    2
#define NTR_KSP_TYPES   3

#define NTR_PC_NONE	0
#define NTR_PC_AHESS    1
#define NTR_PC_BFGS     2
#define NTR_PC_PETSC    3
#define NTR_PC_TYPES    4

#define BFGS_SCALE_AHESS   0
#define BFGS_SCALE_BFGS    1
#define BFGS_SCALE_TYPES   2

#define NTR_INIT_CONSTANT	  0
#define NTR_INIT_DIRECTION	  1
#define NTR_INIT_INTERPOLATION	  2
#define NTR_INIT_TYPES		  3

#define NTR_UPDATE_REDUCTION      0
#define NTR_UPDATE_INTERPOLATION  1
#define NTR_UPDATE_TYPES          2

static const char *NTR_KSP[64] = {
  "nash", "stcg", "gltr"
};

static const char *NTR_PC[64] = {
  "none", "ahess", "bfgs", "petsc"
};

static const char *BFGS_SCALE[64] = {
  "ahess", "bfgs"
};

static const char *NTR_INIT[64] = {
  "constant", "direction", "interpolation"
};

static const char *NTR_UPDATE[64] = {
  "reduction", "interpolation"
};

// Routine for BFGS preconditioner
static PetscErrorCode MatLMVMSolveShell(PC pc, Vec xin, Vec xout);


// TaoSolve_NTR - Implements Newton's Method with a trust region approach 
// for solving unconstrained minimization problems.  
//
// The basic algorithm is taken from MINPACK-2 (dstrn).
//
// TaoSolve_NTR computes a local minimizer of a twice differentiable function
// f by applying a trust region variant of Newton's method.  At each stage 
// of the algorithm, we use the prconditioned conjugate gradient method to
// determine an approximate minimizer of the quadratic equation
//
//      q(s) = <s, Hs + g>
//
// subject to the trust region constraint
//
//      || s ||_M <= radius,
//
// where radius is the trust region radius and M is a symmetric positive
// definite matrix (the preconditioner).  Here g is the gradient and H 
// is the Hessian matrix.
//
// Note:  TaoSolve_NTR MUST use the iterative solver KSPNASH, KSPSTCG,
//        or KSPGLTR.  Thus, we set KSPNASH, KSPSTCG, or KSPGLTR in this 
//        routine regardless of what the user may have previously specified.

#undef __FUNCT__  
#define __FUNCT__ "TaoSolve_NTR"
static PetscErrorCode TaoSolve_NTR(TaoSolver tao)
{
  TAO_NTR *tr = (TAO_NTR *)tao->data;
  Vec Diag = tr->Diag;
  Mat H;

  KSP ksp;
  PC pc;

  KSPConvergedReason ksp_reason;
  TaoLineSearchTerminationReason ls_reason;
  TaoSolverConvergedReason reason;
  
  PetscReal fmin, ftrial, prered, actred, kappa, sigma, beta;
  PetscReal tau, tau_1, tau_2, tau_max, tau_min, max_radius;
  PetscReal f, gnorm;

  PetscReal delta;
  PetscReal radius, norm_d;
  PetscErrorCode ierr;

  PetscInt iter = 0;
  PetscInt bfgsUpdates = 0;
  PetscInt needH;

  PetscInt i_max = 5;
  PetscInt j_max = 1;
  PetscInt i, j;

  PetscFunctionBegin;

  /*// Get the initial trust-region radius
  ierr = TaoGetInitialTrustRegionRadius(tao, &radius); CHKERRQ(ierr);
  if (radius < 0.0) {
    SETERRQ(1, "Initial radius negative");
    }*/
  radius = tr->radius;
  // Modify the radius if it is too large or small
  radius = PetscMax(radius, tr->min_radius);
  radius = PetscMin(radius, tr->max_radius);

  ierr = TaoGetHessian(tao, &H); CHKERRQ(ierr);

  if (NTR_PC_BFGS == tr->pc_type && !tr->M) {
    ierr = VecGetLocalSize(tao->solution,&n); CHKERRQ(ierr);
    ierr = VecGetSize(tao->solution,&N); CHKERRQ(ierr);
    ierr = MatCreateLMVM(((PetscObject)tao)->comm,n,N,&tr->M); CHKERRQ(ierr);
    ierr = MatLMVMAllocateVectors(tr->M,tao->solution); CHKERRQ(ierr);
  }

  // Check convergence criteria
  ierr = TaoSolverComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient); CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
  if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
    SETERRQ(1, "User provided compute function generated Inf or NaN");
  }
  needH = 1;

  ierr = TaoSolverMonitor(tao, iter, f, gnorm, 0.0, 1.0, &reason); CHKERRQ(ierr);
  if (reason != TAO_CONTINUE_ITERATING) {
    PetscFunctionReturn(0);
  }

  // Create vectors for the limited memory preconditioner
  if ((NTR_PC_BFGS == tr->pc_type) && 
      (BFGS_SCALE_BFGS != tr->bfgs_scale_type)) {
    if (!tr->Diag) {
	ierr = VecDuplicate(tao->solution, &tr->Diag); CHKERRQ(ierr);
    }
  }
  
  switch(tr->ksp_type) {
  case NTR_KSP_NASH:
    ierr = KSPSetType(tao->ksp, KSPNASH); CHKERRQ(ierr);
    if (tao->ksp->ops->setfromoptions) {
      (*tao->ksp->ops->setfromoptions)(tao->ksp);
    }
    break;

  case NTR_KSP_STCG:
    ierr = KSPSetType(tao->ksp, KSPSTCG); CHKERRQ(ierr);
    if (tao->ksp->ops->setfromoptions) {
      (*tao->ksp->ops->setfromoptions)(tao->ksp);
    }
    break;

  default:
    ierr = KSPSetType(tao->ksp, KSPGLTR); CHKERRQ(ierr);
    if (tao->ksp->ops->setfromoptions) {
      (*tao->ksp->ops->setfromoptions)(tao->ksp);
    }
    break;
  }

  // Modify the preconditioner to use the bfgs approximation
  ierr = KSPGetPC(tao->ksp, &pc); CHKERRQ(ierr);
  switch(tr->pc_type) {
  case NTR_PC_NONE:
    ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
    if (pc->ops->setfromoptions) {
      (*pc->ops->setfromoptions)(pc);
    }
    break;
 
  case NTR_PC_AHESS:
    ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
    if (pc->ops->setfromoptions) {
      (*pc->ops->setfromoptions)(pc);
    }
    ierr = PCJacobiSetUseAbs(pc); CHKERRQ(ierr);
    break;

  case NTR_PC_BFGS:
    ierr = PCSetType(pc, PCSHELL); CHKERRQ(ierr);
    if (pc->ops->setfromoptions) {
      (*pc->ops->setfromoptions)(pc);
    }
    ierr = PCShellSetName(pc, "bfgs"); CHKERRQ(ierr);
    ierr = PCShellSetContext(pc, M); CHKERRQ(ierr);
    ierr = PCShellSetApply(pc, MatLMVMSolveShell); CHKERRQ(ierr);
    break;

  default:
    // Use the pc method set by pc_type
    break;
  }

  // Initialize trust-region radius
  switch(tr->init_type) {
  case NTR_INIT_CONSTANT:
    // Use the initial radius specified
    break;

  case NTR_INIT_INTERPOLATION:
    // Use the initial radius specified
    max_radius = 0.0;

    for (j = 0; j < j_max; ++j) {
      fmin = f;
      sigma = 0.0;

      if (needH) {
        ierr = TaoComputeHessian(tao, X, H); CHKERRQ(ierr);
        needH = 0;
      }

      for (i = 0; i < i_max; ++i) {
	ierr = W->Waxpby(1.0, X, -radius / gnorm, G); CHKERRQ(ierr);

        ierr = TaoComputeFunction(tao, W, &ftrial); CHKERRQ(ierr);
        if (TaoIerrrNaN(ftrial)) {
	  tau = tr->gamma1_i;
        }
        else {
	  if (ftrial < fmin) {
            fmin = ftrial;
            sigma = -radius / gnorm;
          }

          ierr = H->Multiply(G, D); CHKERRQ(ierr);
          ierr = D->Dot(G, &prered); CHKERRQ(ierr);

          prered = radius * (gnorm - 0.5 * radius * prered / (gnorm * gnorm));
          actred = f - ftrial;
	  if ((fabs(actred) <= tr->epsilon) && 
              (fabs(prered) <= tr->epsilon)) {
	    kappa = 1.0;
	  }
	  else {
	    kappa = actred / prered;
	  }

	  tau_1 = tr->theta_i * gnorm * radius / (tr->theta_i * gnorm * radius + (1.0 - tr->theta_i) * prered - actred);
          tau_2 = tr->theta_i * gnorm * radius / (tr->theta_i * gnorm * radius - (1.0 + tr->theta_i) * prered + actred);
	  tau_min = TaoMin(tau_1, tau_2);
	  tau_max = TaoMax(tau_1, tau_2);

	  if (fabs(kappa - 1.0) <= tr->mu1_i) {
	    // Great agreement
            max_radius = TaoMax(max_radius, radius);

	    if (tau_max < 1.0) {
              tau = tr->gamma3_i;
            }
            else if (tau_max > tr->gamma4_i) {
              tau = tr->gamma4_i;
            }
            else {
              tau = tau_max;
            }
          }
          else if (fabs(kappa - 1.0) <= tr->mu2_i) {
	    // Good agreement
            max_radius = TaoMax(max_radius, radius);

	    if (tau_max < tr->gamma2_i) {
              tau = tr->gamma2_i;
            }
            else if (tau_max > tr->gamma3_i) {
              tau = tr->gamma3_i;
            }
            else {
              tau = tau_max;
            }
          }
          else {
	    // Not good agreement
	    if (tau_min > 1.0) {
	      tau = tr->gamma2_i;
            }
            else if (tau_max < tr->gamma1_i) {
              tau = tr->gamma1_i;
            }
	    else if ((tau_min < tr->gamma1_i) && (tau_max >= 1.0)) {
	      tau = tr->gamma1_i;
            }
	    else if ((tau_1 >= tr->gamma1_i) && (tau_1 < 1.0) && 
                     ((tau_2 < tr->gamma1_i) || (tau_2 >= 1.0))) {
              tau = tau_1;
            }
	    else if ((tau_2 >= tr->gamma1_i) && (tau_2 < 1.0) && 
                     ((tau_1 < tr->gamma1_i) || (tau_2 >= 1.0))) {
              tau = tau_2;
            }
            else {
              tau = tau_max;
            }
          }
        }
        radius = tau * radius;
      }
  
      if (fmin < f) {
        f = fmin;
        ierr = X->Axpy(sigma, G); CHKERRQ(ierr);
        ierr = TaoComputeGradient(tao, X, G); CHKERRQ(ierr);

        ierr = G->Norm2(&gnorm); CHKERRQ(ierr);
        if (TaoIerrrNaN(f) || TaoIerrrNaN(gnorm)) {
          SETERRQ(1, "User provided compute function generated Inf or NaN");
        }
        needH = 1;

        ierr = TaoMonitor(tao, iter, f, gnorm, 0.0, 1.0, &reason); CHKERRQ(ierr);
        if (reason != TAO_CONTINUE_ITERATING) {
          PetscFunctionReturn(0);
        }
      }
    }
    radius = TaoMax(radius, max_radius);

    // Modify the radius if it is too large or small
    radius = TaoMax(radius, tr->min_radius);
    radius = TaoMin(radius, tr->max_radius);
    break;

  default:
    // Norm of the first direction will initialize radius
    radius = 0.0;
    break;
  }

  // Set initial scaling for the BFGS preconditioner 
  // This step is done after computing the initial trust-region radius
  // since the function value may have decreased
  if (NTR_PC_BFGS == tr->pc_type) {
    if (f != 0.0) {
      delta = 2.0 * TaoAbsDouble(f) / (gnorm*gnorm);
    }
    else {
      delta = 2.0 / (gnorm*gnorm);
    }
    ierr = M->SetDelta(delta); CHKERRQ(ierr);
  }

  // Have not converged; continue with Newton method
  while (reason == TAO_CONTINUE_ITERATING) {
    ++iter;

    // Compute the Hessian
    if (needH) {
      ierr = TaoComputeHessian(tao, X, H); CHKERRQ(ierr);
      needH = 0;
    }

    if (NTR_PC_BFGS == tr->pc_type) {
      if (BFGS_SCALE_AHESS == tr->bfgs_scale_type) {
        // Obtain diagonal for the bfgs preconditioner
        ierr = H->GetDiagonal(Diag); CHKERRQ(ierr);
        ierr = Diag->AbsoluteValue(); CHKERRQ(ierr);
        ierr = Diag->Reciprocal(); CHKERRQ(ierr);
        ierr = M->SetScale(Diag); CHKERRQ(ierr);
      }

      // Update the limited memory preconditioner
      ierr = M->Update(X, G); CHKERRQ(ierr);
      ++bfgsUpdates;
    }

    while (reason == TAO_CONTINUE_ITERATING) {
      // Solve the trust region subproblem
      ierr = TaoPreLinearSolve(tao, H); CHKERRQ(ierr);
      ierr = TaoLinearSolveTrustRegion(tao, H, G, D, radius, &success); CHKERRQ(ierr);
      ierr = pls->GetNormDirection(&norm_d); CHKERRQ(ierr);
      if (0.0 == radius) {
        // Radius was uninitialized; use the norm of the direction
        if (norm_d > 0.0) {
          radius = norm_d;

          // Modify the radius if it is too large or small
          radius = TaoMax(radius, tr->min_radius);
          radius = TaoMin(radius, tr->max_radius);
        }
        else {
          // The direction was bad; set radius to default value and re-solve 
          // the trust-region subproblem to get a direction
          ierr = TaoGetInitialTrustRegionRadius(tao, &radius); CHKERRQ(ierr);

          // Modify the radius if it is too large or small
          radius = TaoMax(radius, tr->min_radius);
          radius = TaoMin(radius, tr->max_radius);

          ierr = TaoLinearSolveTrustRegion(tao, H, G, D, radius, &success); CHKERRQ(ierr);
          ierr = pls->GetNormDirection(&norm_d); CHKERRQ(ierr);
	  if (norm_d == 0.0) {
            SETERRQ(1, "Initial direction zero");
          }
        }
      }
      ierr = D->Negate(); CHKERRQ(ierr);

      ierr = KSPGetConvergedReason(tao->ksp, &ksp_reason); CHKERRQ(ierr);
      if ((KSP_DIVERGED_INDEFINITE_PC == ksp_reason) &&
          (NTR_PC_BFGS == tr->pc_type) && (bfgsUpdates > 1)) {
        // Preconditioner is numerically indefinite; reset the
        // approximate if using BFGS preconditioning.
  
        if (f != 0.0) {
          delta = 2.0 * TaoAbsDouble(f) / (gnorm*gnorm);
        }
        else {
          delta = 2.0 / (gnorm*gnorm);
        }
        ierr = M->SetDelta(delta); CHKERRQ(ierr);
        ierr = M->Reset(); CHKERRQ(ierr);
        ierr = M->Update(X, G); CHKERRQ(ierr);
        bfgsUpdates = 1;
      }

      if (NTR_UPDATE_REDUCTION == tr->update_type) {
	// Get predicted reduction
	ierr = pls->GetObjFcn(&prered); CHKERRQ(ierr);
	
	if (prered >= 0.0) {
	  // The predicted reduction has the wrong sign.  This cannot
	  // happen in infinite precision arithmetic.  Step should
	  // be rejected!
	  radius = tr->alpha1 * TaoMin(radius, norm_d);
	}
	else {
	  // Compute trial step and function value
	  ierr = W->Waxpby(1.0, X, 1.0, D); CHKERRQ(ierr);
	  ierr = TaoComputeFunction(tao, W, &ftrial); CHKERRQ(ierr);
	  if (TaoIerrrNaN(ftrial)) {
	    radius = tr->alpha1 * TaoMin(radius, norm_d);
	  }
	  else {
	    // Compute and actual reduction
	    actred = f - ftrial;
	    prered = -prered;
	    if ((fabs(actred) <= tr->epsilon) && 
                (fabs(prered) <= tr->epsilon)) {
	      kappa = 1.0;
	    }
	    else {
	      kappa = actred / prered;
	    }
	    
	    // Accept or reject the step and update radius
	    if (kappa < tr->eta1) {
	      // Reject the step
	      radius = tr->alpha1 * TaoMin(radius, norm_d);
	    } 
	    else {
	      // Accept the step
	      if (kappa < tr->eta2) { 
		// Marginal bad step
		radius = tr->alpha2 * TaoMin(radius, norm_d);
	      }
	      else if (kappa < tr->eta3) {
		// Reasonable step
		radius = tr->alpha3 * radius;
	      }
	      else if (kappa < tr->eta4) { 
		// Good step
		radius = TaoMax(tr->alpha4 * norm_d, radius);
	      }
	      else {
		// Very good step
		radius = TaoMax(tr->alpha5 * norm_d, radius);
	      }
	      break;
	    }
	  }
	} 
      }
      else {
	// Get predicted reduction
	ierr = pls->GetObjFcn(&prered); CHKERRQ(ierr);

	if (prered >= 0.0) {
	  // The predicted reduction has the wrong sign.  This cannot
	  // happen in infinite precision arithmetic.  Step should
	  // be rejected!
	  radius = tr->gamma1 * TaoMin(radius, norm_d);
	}
	else {
	  ierr = W->Waxpby(1.0, X, 1.0, D); CHKERRQ(ierr);
	  ierr = TaoComputeFunction(tao, W, &ftrial); CHKERRQ(ierr);
	  if (TaoIerrrNaN(ftrial)) {
	    radius = tr->gamma1 * TaoMin(radius, norm_d);
	  }
	  else {
	    ierr = D->Dot(G, &beta); CHKERRQ(ierr);

	    actred = f - ftrial;
	    prered = -prered;
	    if ((fabs(actred) <= tr->epsilon) && 
                (fabs(prered) <= tr->epsilon)) {
	      kappa = 1.0;
	    }
	    else {
	      kappa = actred / prered;
	    }

	    tau_1 = tr->theta * beta / (tr->theta * beta - (1.0 - tr->theta) * prered + actred);
	    tau_2 = tr->theta * beta / (tr->theta * beta + (1.0 + tr->theta) * prered - actred);
	    tau_min = TaoMin(tau_1, tau_2);
	    tau_max = TaoMax(tau_1, tau_2);

	    if (kappa >= 1.0 - tr->mu1) {
	      // Great agreement; accept step and update radius
	      if (tau_max < 1.0) {
		radius = TaoMax(radius, tr->gamma3 * norm_d);
	      }
	      else if (tau_max > tr->gamma4) {
		radius = TaoMax(radius, tr->gamma4 * norm_d);
	      }
	      else {
		radius = TaoMax(radius, tau_max * norm_d);
	      }
	      break;
	    }
	    else if (kappa >= 1.0 - tr->mu2) {
	      // Good agreement

	      if (tau_max < tr->gamma2) {
		radius = tr->gamma2 * TaoMin(radius, norm_d);
	      }
	      else if (tau_max > tr->gamma3) {
		radius = TaoMax(radius, tr->gamma3 * norm_d);
	      }
	      else if (tau_max < 1.0) {
		radius = tau_max * TaoMin(radius, norm_d);
	      }
	      else {
		radius = TaoMax(radius, tau_max * norm_d);
	      }
	      break;
	    }
	    else {
	      // Not good agreement
	      if (tau_min > 1.0) {
		radius = tr->gamma2 * TaoMin(radius, norm_d);
	      }
	      else if (tau_max < tr->gamma1) {
		radius = tr->gamma1 * TaoMin(radius, norm_d);
	      }
	      else if ((tau_min < tr->gamma1) && (tau_max >= 1.0)) {
		radius = tr->gamma1 * TaoMin(radius, norm_d);
	      }
	      else if ((tau_1 >= tr->gamma1) && (tau_1 < 1.0) && 
		       ((tau_2 < tr->gamma1) || (tau_2 >= 1.0))) {
		radius = tau_1 * TaoMin(radius, norm_d);
	      }
	      else if ((tau_2 >= tr->gamma1) && (tau_2 < 1.0) && 
		       ((tau_1 < tr->gamma1) || (tau_2 >= 1.0))) {
		radius = tau_2 * TaoMin(radius, norm_d);
	      }
	      else {
		radius = tau_max * TaoMin(radius, norm_d);
	      }
	    }
	  }
	}
      }

      // The step computed was not good and the radius was decreased.
      // Monitor the radius to terminate.
      ierr = TaoMonitor(tao, iter, f, gnorm, 0.0, radius, &reason); CHKERRQ(ierr);
    }

    // The radius may have been increased; modify if it is too large
    radius = TaoMin(radius, tr->max_radius);

    if (reason == TAO_CONTINUE_ITERATING) {
      ierr = X->CopyFrom(W); CHKERRQ(ierr);
      f = ftrial;
      ierr = TaoComputeGradient(tao, X, G); CHKERRQ(ierr);
      ierr = G->Norm2(&gnorm); CHKERRQ(ierr);
      if (TaoIerrrNaN(f) || TaoIerrrNaN(gnorm)) {
	SETERRQ(1, "User provided compute function generated Inf or NaN");
      }
      needH = 1;

      ierr = TaoMonitor(tao, iter, f, gnorm, 0.0, radius, &reason); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSetUp_NTR"
static PetscErrorCode TaoSetUp_NTR(TAO_SOLVER tao, void *solver)
{
  TAO_NTR *tr = (TAO_NTR *)solver;
  TaoVec *X;
  TaoMat *H;
  int ierr;

  PetscFunctionBegin;

  ierr = TaoGetSolution(tao, &X); CHKERRQ(ierr);
  ierr = X->Clone(&tr->G); CHKERRQ(ierr);
  ierr = X->Clone(&tr->D); CHKERRQ(ierr);
  ierr = X->Clone(&tr->W); CHKERRQ(ierr);

  tr->Diag = 0;
  tr->M = 0;

  ierr = TaoSetLagrangianGradientVector(tao, tr->G); CHKERRQ(ierr);
  ierr = TaoSetStepDirectionVector(tao, tr->D); CHKERRQ(ierr);

  // Set linear solver to default for trust region
  ierr = TaoGetHessian(tao, &H); CHKERRQ(ierr);
  ierr = TaoCreateLinearSolver(tao, H, 200, 0); CHKERRQ(ierr); 

  // Check sizes for compatability
  ierr = TaoCheckFGH(tao); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoDestroy_NTR"
static PetscErrorCode TaoDestroy_NTR(TAO_SOLVER tao, void *solver)
{
  TAO_NTR *tr = (TAO_NTR *)solver;
  int ierr;

  PetscFunctionBegin;
  ierr = TaoVecDestroy(tr->G); CHKERRQ(ierr);
  ierr = TaoVecDestroy(tr->D); CHKERRQ(ierr);
  ierr = TaoVecDestroy(tr->W); CHKERRQ(ierr);

  ierr = TaoSetLagrangianGradientVector(tao, 0); CHKERRQ(ierr);
  ierr = TaoSetStepDirectionVector(tao, 0); CHKERRQ(ierr);

  if (tr->Diag) {
    ierr = TaoVecDestroy(tr->Diag); CHKERRQ(ierr);
    tr->Diag = 0;
  }

  if (tr->M) {
    ierr = TaoMatDestroy(tr->M); CHKERRQ(ierr);
    tr->M = 0;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSetOptions_NTR"
static PetscErrorCode TaoSetOptions_NTR(TAO_SOLVER tao, void*solver)
{
  TAO_NTR *tr = (TAO_NTR *)solver;
  int ierr;

  PetscFunctionBegin;
  ierr = TaoOptionsHead("Newton trust region method for unconstrained optimization"); CHKERRQ(ierr);
  ierr = TaoOptionList("-tao_ntr_ksp_type", "ksp type", "", NTR_KSP, NTR_KSP_TYPES, NTR_KSP[tr->ksp_type], &tr->ksp_type, 0); CHKERRQ(ierr);
  ierr = TaoOptionList("-tao_ntr_pc_type", "pc type", "", NTR_PC, NTR_PC_TYPES, NTR_PC[tr->pc_type], &tr->pc_type, 0); CHKERRQ(ierr);
  ierr = TaoOptionList("-tao_ntr_bfgs_scale_type", "bfgs scale type", "", BFGS_SCALE, BFGS_SCALE_TYPES, BFGS_SCALE[tr->bfgs_scale_type], &tr->bfgs_scale_type, 0); CHKERRQ(ierr);
  ierr = TaoOptionList("-tao_ntr_init_type", "radius initialization type", "", NTR_INIT, NTR_INIT_TYPES, NTR_INIT[tr->init_type], &tr->init_type, 0); CHKERRQ(ierr);
  ierr = TaoOptionList("-tao_ntr_update_type", "radius update type", "", NTR_UPDATE, NTR_UPDATE_TYPES, NTR_UPDATE[tr->update_type], &tr->update_type, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_eta1", "step is unsuccessful if actual reduction < eta1 * predicted reduction", "", tr->eta1, &tr->eta1, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_eta2", "", "", tr->eta2, &tr->eta2, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_eta3", "", "", tr->eta3, &tr->eta3, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_eta4", "", "", tr->eta4, &tr->eta4, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_alpha1", "", "", tr->alpha1, &tr->alpha1, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_alpha2", "", "", tr->alpha2, &tr->alpha2, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_alpha3", "", "", tr->alpha3, &tr->alpha3, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_alpha4", "", "", tr->alpha4, &tr->alpha4, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_alpha5", "", "", tr->alpha5, &tr->alpha5, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_mu1", "", "", tr->mu1, &tr->mu1, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_mu2", "", "", tr->mu2, &tr->mu2, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_gamma1", "", "", tr->gamma1, &tr->gamma1, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_gamma2", "", "", tr->gamma2, &tr->gamma2, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_gamma3", "", "", tr->gamma3, &tr->gamma3, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_gamma4", "", "", tr->gamma4, &tr->gamma4, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_theta", "", "", tr->theta, &tr->theta, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_mu1_i", "", "", tr->mu1_i, &tr->mu1_i, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_mu2_i", "", "", tr->mu2_i, &tr->mu2_i, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_gamma1_i", "", "", tr->gamma1_i, &tr->gamma1_i, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_gamma2_i", "", "", tr->gamma2_i, &tr->gamma2_i, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_gamma3_i", "", "", tr->gamma3_i, &tr->gamma3_i, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_gamma4_i", "", "", tr->gamma4_i, &tr->gamma4_i, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_theta_i", "", "", tr->theta_i, &tr->theta_i, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_min_radius", "lower bound on initial trust-region radius", "", tr->min_radius, &tr->min_radius, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_max_radius", "upper bound on trust-region radius", "", tr->max_radius, &tr->max_radius, 0); CHKERRQ(ierr);
  ierr = TaoOptionDouble("-tao_ntr_epsilon", "tolerance used when computing actual and predicted reduction", "", tr->epsilon, &tr->epsilon, 0); CHKERRQ(ierr);
  ierr = TaoOptionsTail(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoView_NTR"
static PetscErrorCode TaoView_NTR(TAO_SOLVER tao,void*solver)
{
  TAO_NTR *tr = (TAO_NTR *)solver;
  int ierr;

  PetscFunctionBegin;
  if (NTR_PC_BFGS == tr->pc_type && tr->M) {
    ierr = TaoPrintInt(tao, "  Rejected matrix updates: %d\n", tr->M->GetRejects()); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoCreate_NTR"
PetscErrorCode TaoCreate_NTR(TAO_SOLVER tao)
{
  TAO_NTR *tr;
  int ierr;

  PetscFunctionBegin;

  ierr = TaoNew(TAO_NTR, &tr); CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(tao, sizeof(TAO_NTR)); CHKERRQ(ierr);

  ierr = TaoSetTaoSolveRoutine(tao, TaoSolve_NTR, (void *)tr); CHKERRQ(ierr);
  ierr = TaoSetTaoSetUpDownRoutines(tao, TaoSetUp_NTR, TaoDestroy_NTR); CHKERRQ(ierr);
  ierr = TaoSetTaoOptionsRoutine(tao, TaoSetOptions_NTR); CHKERRQ(ierr);
  ierr = TaoSetTaoViewRoutine(tao, TaoView_NTR); CHKERRQ(ierr);

  ierr = TaoSetMaximumIterates(tao, 50); CHKERRQ(ierr);
  ierr = TaoSetTolerances(tao, 1e-10, 1e-10, 0, 0); CHKERRQ(ierr);

  ierr = TaoSetTrustRegionRadius(tao, 100.0); CHKERRQ(ierr);
  ierr = TaoSetTrustRegionTolerance(tao, 1.0e-12); CHKERRQ(ierr);

  // Standard trust region update parameters
  tr->eta1 = 1.0e-4;
  tr->eta2 = 0.25;
  tr->eta3 = 0.50;
  tr->eta4 = 0.90;

  tr->alpha1 = 0.25;
  tr->alpha2 = 0.50;
  tr->alpha3 = 1.00;
  tr->alpha4 = 2.00;
  tr->alpha5 = 4.00;

  // Interpolation parameters
  tr->mu1_i = 0.35;
  tr->mu2_i = 0.50;

  tr->gamma1_i = 0.0625;
  tr->gamma2_i = 0.50;
  tr->gamma3_i = 2.00;
  tr->gamma4_i = 5.00;

  tr->theta_i = 0.25;

  // Interpolation trust region update parameters
  tr->mu1 = 0.10;
  tr->mu2 = 0.50;

  tr->gamma1 = 0.25;
  tr->gamma2 = 0.50;
  tr->gamma3 = 2.00;
  tr->gamma4 = 4.00;

  tr->theta = 0.05;

  tr->min_radius = 1.0e-10;
  tr->max_radius = 1.0e10;
  tr->epsilon = 1.0e-6;

  tr->ksp_type        = NTR_KSP_STCG;
  tr->pc_type         = NTR_PC_BFGS;
  tr->bfgs_scale_type = BFGS_SCALE_AHESS;
  tr->init_type	      = NTR_INIT_INTERPOLATION;
  tr->update_type     = NTR_UPDATE_REDUCTION;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif

#undef __FUNCT__
#define __FUNCT__ "MatLMVMSolveShell"
static PetscErrorCode MatLMVMSolveShell(PC pc, Vec b, Vec x) 
{
    PetscErrorCode ierr;
    Mat M;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(pc,PC_COOKIE,1);
    PetscValidHeaderSpecific(b,VEC_COOKIE,2);
    PetscValidHeaderSpecific(x,VEC_COOKIE,3);
    ierr = PCShellGetContext(pc,(void**)&M); CHKERRQ(ierr);
    ierr = MatLMVMSolve(M, b, x); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
