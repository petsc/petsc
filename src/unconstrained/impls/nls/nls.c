#include "taolinesearch.h"
#include "src/matrix/lmvmmat.h"
#include "nls.h"

#include "petscksp.h"
#include "petscpc.h"
#include "private/kspimpl.h"
#include "private/pcimpl.h"

#define NLS_KSP_CG	0
#define NLS_KSP_NASH	1
#define NLS_KSP_STCG	2
#define NLS_KSP_GLTR	3
#define NLS_KSP_PETSC	4
#define NLS_KSP_TYPES	5

#define NLS_PC_NONE	0
#define NLS_PC_AHESS	1
#define NLS_PC_BFGS	2
#define NLS_PC_PETSC	3
#define NLS_PC_TYPES	4

#define BFGS_SCALE_AHESS	0
#define BFGS_SCALE_PHESS	1
#define BFGS_SCALE_BFGS		2
#define BFGS_SCALE_TYPES	3

#define NLS_INIT_CONSTANT         0
#define NLS_INIT_DIRECTION        1
#define NLS_INIT_INTERPOLATION    2
#define NLS_INIT_TYPES            3

#define NLS_UPDATE_STEP           0
#define NLS_UPDATE_REDUCTION      1
#define NLS_UPDATE_INTERPOLATION  2
#define NLS_UPDATE_TYPES          3

static const char *NLS_KSP[64] = {
  "cg", "nash", "stcg", "gltr", "petsc"
};

static const char *NLS_PC[64] = {
  "none", "ahess", "bfgs", "petsc"
};

static const char *BFGS_SCALE[64] = {
  "ahess", "phess", "bfgs"
};

static const char *NLS_INIT[64] = {
  "constant", "direction", "interpolation"
};

static const char *NLS_UPDATE[64] = {
  "step", "reduction", "interpolation"
};

static PetscErrorCode MatLMVMSolveShell(PC pc, Vec b, Vec x) ;
// Routine for BFGS preconditioner


// Implements Newton's Method with a line search approach for solving
// unconstrained minimization problems.  A More'-Thuente line search 
// is used to guarantee that the bfgs preconditioner remains positive
// definite.
//
// The method can shift the Hessian matrix.  The shifting procedure is
// adapted from the PATH algorithm for solving complementarity
// problems.
//
// The linear system solve should be done with a conjugate gradient
// method, although any method can be used.

#define NLS_NEWTON 		0
#define NLS_BFGS 		1
#define NLS_SCALED_GRADIENT 	2
#define NLS_GRADIENT 		3

#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSolve_NLS"
static PetscErrorCode TaoSolverSolve_NLS(TaoSolver tao)
{
  PetscErrorCode ierr;
  TAO_NLS *nlsP = (TAO_NLS *)tao->data;

  PC pc;

  KSPConvergedReason ksp_reason;
  TaoLineSearchTerminationReason ls_reason;
  TaoSolverTerminationReason reason;
  
  PetscReal fmin, ftrial, f_full, prered, actred, kappa, sigma;
  PetscReal tau, tau_1, tau_2, tau_max, tau_min, max_radius;
  PetscReal f, fold, gdx, gnorm, pert;
  PetscReal step = 1.0;

  PetscReal delta;
  PetscReal norm_d = 0.0, e_min;

  MatStructure matflag;

  PetscInt stepType;
  PetscInt iter = 0;
  PetscInt bfgsUpdates = 0;
  PetscInt n,N,kspits;
  PetscInt needH;
  
  PetscInt i_max = 5;
  PetscInt j_max = 1;
  PetscInt i, j;

  PetscFunctionBegin;

  if (tao->XL || tao->XU || tao->ops->computebounds) {
    ierr = PetscPrintf(((PetscObject)tao)->comm,"WARNING: Variable bounds have been set but will be ignored by nls algorithm\n"); CHKERRQ(ierr);
  }

  // Initialized variables
  pert = nlsP->sval;

  nlsP->ksp_atol = 0;
  nlsP->ksp_rtol = 0;
  nlsP->ksp_dtol = 0;
  nlsP->ksp_ctol = 0;
  nlsP->ksp_negc = 0;
  nlsP->ksp_iter = 0;
  nlsP->ksp_othr = 0;



  // Modify the linear solver to a trust region method if desired

  switch(nlsP->ksp_type) {
  case NLS_KSP_CG:
    ierr = KSPSetType(tao->ksp, KSPCG); CHKERRQ(ierr);
    if (tao->ksp->ops->setfromoptions) {
	(*tao->ksp->ops->setfromoptions)(tao->ksp);
    }
    break;

  case NLS_KSP_NASH:
    ierr = KSPSetType(tao->ksp, KSPNASH); CHKERRQ(ierr);
    if (tao->ksp->ops->setfromoptions) {
      (*tao->ksp->ops->setfromoptions)(tao->ksp);
    }
    break;

  case NLS_KSP_STCG:
    ierr = KSPSetType(tao->ksp, KSPSTCG); CHKERRQ(ierr);
    if (tao->ksp->ops->setfromoptions) {
      (*tao->ksp->ops->setfromoptions)(tao->ksp);
    }
    break;

  case NLS_KSP_GLTR:
    ierr = KSPSetType(tao->ksp, KSPGLTR); CHKERRQ(ierr);
    if (tao->ksp->ops->setfromoptions) {
      (*tao->ksp->ops->setfromoptions)(tao->ksp);
    }
    break;

  default:
    // Use the method set by the ksp_type
    break;
  }

  // Initialize trust-region radius when using nash, stcg, or gltr
  // Will be reset during the first iteration
  if (NLS_KSP_NASH == nlsP->ksp_type) {
      ierr = KSPNASHSetRadius(tao->ksp,nlsP->max_radius); CHKERRQ(ierr);
  } else if (NLS_KSP_STCG == nlsP->ksp_type) {
      ierr = KSPSTCGSetRadius(tao->ksp,nlsP->max_radius); CHKERRQ(ierr);
  } else if (NLS_KSP_GLTR == nlsP->ksp_type) {
      ierr = KSPGLTRSetRadius(tao->ksp,nlsP->max_radius); CHKERRQ(ierr);
  }
  
  
  if (NLS_KSP_NASH == nlsP->ksp_type ||
      NLS_KSP_STCG == nlsP->ksp_type || 
      NLS_KSP_GLTR == nlsP->ksp_type) {
    tao->trust = tao->trust0;

    if (tao->trust < 0.0) {
      SETERRQ(PETSC_COMM_SELF,1, "Initial radius negative");
    }

    // Modify the radius if it is too large or small
    tao->trust = PetscMax(tao->trust, nlsP->min_radius);
    tao->trust = PetscMin(tao->trust, nlsP->max_radius);
  }

  // Get vectors we will need

  if (NLS_PC_BFGS == nlsP->pc_type && !nlsP->M) {
    ierr = VecGetLocalSize(tao->solution,&n); CHKERRQ(ierr);
    ierr = VecGetSize(tao->solution,&N); CHKERRQ(ierr);
    ierr = MatCreateLMVM(((PetscObject)tao)->comm,n,N,&nlsP->M); CHKERRQ(ierr);
    ierr = MatLMVMAllocateVectors(nlsP->M,tao->solution); CHKERRQ(ierr);
  }

  // Check convergence criteria
  ierr = TaoSolverComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient); CHKERRQ(ierr);
  ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
  if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
    SETERRQ(PETSC_COMM_SELF,1, "User provided compute function generated Inf or NaN");
  }
  needH = 1;

  ierr = TaoSolverMonitor(tao, iter, f, gnorm, 0.0, 1.0, &reason); CHKERRQ(ierr);
  if (reason != TAO_CONTINUE_ITERATING) {
    PetscFunctionReturn(0);
  }

  // Create vectors for the limited memory preconditioner
  if ((NLS_PC_BFGS == nlsP->pc_type) && 
      (BFGS_SCALE_BFGS != nlsP->bfgs_scale_type)) {
    if (!nlsP->Diag) {
	ierr = VecDuplicate(tao->solution,&nlsP->Diag); CHKERRQ(ierr);
    }
  }


  // Modify the preconditioner to use the bfgs approximation
  ierr = KSPGetPC(tao->ksp, &pc); CHKERRQ(ierr);
  switch(nlsP->pc_type) {
  case NLS_PC_NONE:
    ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
    if (pc->ops->setfromoptions) {
      (*pc->ops->setfromoptions)(pc);
    }
    break;

  case NLS_PC_AHESS:
    ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
    if (pc->ops->setfromoptions) {
      (*pc->ops->setfromoptions)(pc);
    }
    ierr = PCJacobiSetUseAbs(pc); CHKERRQ(ierr);
    break;

  case NLS_PC_BFGS:
    ierr = PCSetType(pc, PCSHELL); CHKERRQ(ierr);
    if (pc->ops->setfromoptions) {
      (*pc->ops->setfromoptions)(pc);
    }
    ierr = PCShellSetName(pc, "bfgs"); CHKERRQ(ierr);
    ierr = PCShellSetContext(pc, nlsP->M); CHKERRQ(ierr);
    ierr = PCShellSetApply(pc, MatLMVMSolveShell); CHKERRQ(ierr);
    break;

  default:
    // Use the pc method set by pc_type
    break;
  }

  // Initialize trust-region radius.  The initialization is only performed 
  // when we are using Nash, Steihaug-Toint or the Generalized Lanczos method.
  if (NLS_KSP_NASH == nlsP->ksp_type ||
      NLS_KSP_STCG == nlsP->ksp_type || 
      NLS_KSP_GLTR == nlsP->ksp_type) {
    switch(nlsP->init_type) {
    case NLS_INIT_CONSTANT:
      // Use the initial radius specified
      break;

    case NLS_INIT_INTERPOLATION:
      // Use the initial radius specified
      max_radius = 0.0;
  
      for (j = 0; j < j_max; ++j) {
        fmin = f;
        sigma = 0.0;
  
        if (needH) {
	    ierr = TaoSolverComputeHessian(tao, tao->solution, &tao->hessian, &tao->hessian_pre, &matflag); CHKERRQ(ierr);
	    needH = 0;
        }
  
        for (i = 0; i < i_max; ++i) {
          ierr = VecCopy(tao->solution,nlsP->W); CHKERRQ(ierr);
	  ierr = VecAXPY(nlsP->W,-tao->trust/gnorm,tao->gradient); CHKERRQ(ierr);
	  ierr = TaoSolverComputeObjective(tao, nlsP->W, &ftrial); CHKERRQ(ierr);
          if (TaoInfOrNaN(ftrial)) {
            tau = nlsP->gamma1_i;
          }
          else {
            if (ftrial < fmin) {
              fmin = ftrial;
              sigma = -tao->trust / gnorm;
            }
	    
	    ierr = MatMult(tao->hessian, tao->gradient, nlsP->D); CHKERRQ(ierr);
	    ierr = VecDot(tao->gradient, nlsP->D, &prered); CHKERRQ(ierr);
  
            prered = tao->trust * (gnorm - 0.5 * tao->trust * prered / (gnorm * gnorm));
            actred = f - ftrial;
            if ((PetscAbsScalar(actred) <= nlsP->epsilon) && 
                (PetscAbsScalar(prered) <= nlsP->epsilon)) {
              kappa = 1.0;
            }
            else {
              kappa = actred / prered;
            }
  
            tau_1 = nlsP->theta_i * gnorm * tao->trust / (nlsP->theta_i * gnorm * tao->trust + (1.0 - nlsP->theta_i) * prered - actred);
            tau_2 = nlsP->theta_i * gnorm * tao->trust / (nlsP->theta_i * gnorm * tao->trust - (1.0 + nlsP->theta_i) * prered + actred);
            tau_min = PetscMin(tau_1, tau_2);
            tau_max = PetscMax(tau_1, tau_2);
  
            if (PetscAbsScalar(kappa - 1.0) <= nlsP->mu1_i) {
              // Great agreement
              max_radius = PetscMax(max_radius, tao->trust);
  
              if (tau_max < 1.0) {
                tau = nlsP->gamma3_i;
              }
              else if (tau_max > nlsP->gamma4_i) {
                tau = nlsP->gamma4_i;
              }
              else if (tau_1 >= 1.0 && tau_1 <= nlsP->gamma4_i && tau_2 < 1.0) {
                tau = tau_1;
              }
              else if (tau_2 >= 1.0 && tau_2 <= nlsP->gamma4_i && tau_1 < 1.0) {
                tau = tau_2;
              }
              else {
                tau = tau_max;
              }
            }
            else if (PetscAbsScalar(kappa - 1.0) <= nlsP->mu2_i) {
              // Good agreement
              max_radius = PetscMax(max_radius, tao->trust);
  
              if (tau_max < nlsP->gamma2_i) {
                tau = nlsP->gamma2_i;
              }
              else if (tau_max > nlsP->gamma3_i) {
                tau = nlsP->gamma3_i;
              }
              else {
                tau = tau_max;
              }
            }
            else {
              // Not good agreement
              if (tau_min > 1.0) {
                tau = nlsP->gamma2_i;
              }
              else if (tau_max < nlsP->gamma1_i) {
                tau = nlsP->gamma1_i;
              }
              else if ((tau_min < nlsP->gamma1_i) && (tau_max >= 1.0)) {
                tau = nlsP->gamma1_i;
              }
              else if ((tau_1 >= nlsP->gamma1_i) && (tau_1 < 1.0) &&
                       ((tau_2 < nlsP->gamma1_i) || (tau_2 >= 1.0))) {
                tau = tau_1;
              }
              else if ((tau_2 >= nlsP->gamma1_i) && (tau_2 < 1.0) &&
                       ((tau_1 < nlsP->gamma1_i) || (tau_2 >= 1.0))) {
                tau = tau_2;
              }
              else {
                tau = tau_max;
              }
            }
          }
          tao->trust = tau * tao->trust;
        }
  
        if (fmin < f) {
          f = fmin;
	  ierr = VecAXPY(tao->solution,sigma,tao->gradient); CHKERRQ(ierr);
	  ierr = TaoSolverComputeGradient(tao,tao->solution,tao->gradient); CHKERRQ(ierr);

	  ierr = VecNorm(tao->gradient,NORM_2,&gnorm); CHKERRQ(ierr);
          if (TaoInfOrNaN(gnorm)) {
            SETERRQ(PETSC_COMM_SELF,1, "User provided compute gradient generated Inf or NaN");
          }
          needH = 1;
  
          ierr = TaoSolverMonitor(tao, iter, f, gnorm, 0.0, 1.0, &reason); CHKERRQ(ierr);
          if (reason != TAO_CONTINUE_ITERATING) {
            PetscFunctionReturn(0);
          }
        }
      }
      tao->trust = PetscMax(tao->trust, max_radius);

      // Modify the radius if it is too large or small
      tao->trust = PetscMax(tao->trust, nlsP->min_radius);
      tao->trust = PetscMin(tao->trust, nlsP->max_radius);
      break;

    default:
      // Norm of the first direction will initialize radius
      tao->trust = 0.0;
      break;
    }
  } 

  // Set initial scaling for the BFGS preconditioner
  // This step is done after computing the initial trust-region radius
  // since the function value may have decreased
  if (NLS_PC_BFGS == nlsP->pc_type) {
    if (f != 0.0) {
      delta = 2.0 * PetscAbsScalar(f) / (gnorm*gnorm);
    }
    else {
      delta = 2.0 / (gnorm*gnorm);
    }
    ierr = MatLMVMSetDelta(nlsP->M,delta); CHKERRQ(ierr);
  }

  // Set counter for gradient/reset steps
  nlsP->newt = 0;
  nlsP->bfgs = 0;
  nlsP->sgrad = 0;
  nlsP->grad = 0;

  // Have not converged; continue with Newton method
  while (reason == TAO_CONTINUE_ITERATING) {
    ++iter;

    // Compute the Hessian
    if (needH) {
	ierr = TaoSolverComputeHessian(tao, tao->solution, &tao->hessian, &tao->hessian_pre, &matflag); CHKERRQ(ierr);
      needH = 0;
    }

    if ((NLS_PC_BFGS == nlsP->pc_type) && 
        (BFGS_SCALE_AHESS == nlsP->bfgs_scale_type)) {
      // Obtain diagonal for the bfgs preconditioner 
      ierr = MatGetDiagonal(tao->hessian, nlsP->Diag); CHKERRQ(ierr);
      ierr = VecAbs(nlsP->Diag); CHKERRQ(ierr);
      ierr = VecReciprocal(nlsP->Diag); CHKERRQ(ierr);
      ierr = MatLMVMSetScale(nlsP->M,nlsP->Diag); CHKERRQ(ierr);
    }
 
    // Shift the Hessian matrix
    if (pert > 0) {
      ierr = MatShift(tao->hessian, pert);
      if (tao->hessian != tao->hessian_pre) {
	ierr = MatShift(tao->hessian_pre, pert); CHKERRQ(ierr);
      }
    }

    
    if (NLS_PC_BFGS == nlsP->pc_type) {
      if (BFGS_SCALE_PHESS == nlsP->bfgs_scale_type) {
	// Obtain diagonal for the bfgs preconditioner 
	  ierr = MatGetDiagonal(tao->hessian, nlsP->Diag); CHKERRQ(ierr);
	  ierr = VecAbs(nlsP->Diag); CHKERRQ(ierr);
	  ierr = VecReciprocal(nlsP->Diag); CHKERRQ(ierr);
	  ierr = MatLMVMSetScale(nlsP->M,nlsP->Diag); CHKERRQ(ierr);
      }
      // Update the limited memory preconditioner
      ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
      ++bfgsUpdates;
    }

    // Solve the Newton system of equations
    ierr = KSPSetOperators(tao->ksp,tao->hessian,tao->hessian_pre,matflag); CHKERRQ(ierr);
    if (NLS_KSP_NASH == nlsP->ksp_type ||
        NLS_KSP_STCG == nlsP->ksp_type || 
        NLS_KSP_GLTR == nlsP->ksp_type) {

      if (NLS_KSP_NASH == nlsP->ksp_type) {
	ierr = KSPNASHSetRadius(tao->ksp,nlsP->max_radius); CHKERRQ(ierr);
      } else if (NLS_KSP_STCG == nlsP->ksp_type) {
	 ierr = KSPSTCGSetRadius(tao->ksp,nlsP->max_radius); CHKERRQ(ierr);
      } else if (NLS_KSP_GLTR == nlsP->ksp_type) {
	ierr = KSPGLTRSetRadius(tao->ksp,nlsP->max_radius); CHKERRQ(ierr);
      }
	
      ierr = KSPSolve(tao->ksp, tao->gradient, nlsP->D); CHKERRQ(ierr);

      if (NLS_KSP_NASH == nlsP->ksp_type) {
	ierr = KSPNASHGetNormD(tao->ksp,&norm_d); CHKERRQ(ierr);
      } else if (NLS_KSP_STCG == nlsP->ksp_type) {
	 ierr = KSPSTCGGetNormD(tao->ksp,&norm_d); CHKERRQ(ierr);
      } else if (NLS_KSP_GLTR == nlsP->ksp_type) {
	ierr = KSPGLTRGetNormD(tao->ksp,&norm_d); CHKERRQ(ierr);
      }

      if (0.0 == tao->trust) {
        // Radius was uninitialized; use the norm of the direction
        if (norm_d > 0.0) {
          tao->trust = norm_d;

          // Modify the radius if it is too large or small
          tao->trust = PetscMax(tao->trust, nlsP->min_radius);
          tao->trust = PetscMin(tao->trust, nlsP->max_radius);
        }
        else {
          // The direction was bad; set radius to default value and re-solve 
	  // the trust-region subproblem to get a direction
	  tao->trust = tao->trust0;

          // Modify the radius if it is too large or small
          tao->trust = PetscMax(tao->trust, nlsP->min_radius);
          tao->trust = PetscMin(tao->trust, nlsP->max_radius);

	  if (NLS_KSP_NASH == nlsP->ksp_type) {
	    ierr = KSPNASHSetRadius(tao->ksp,nlsP->max_radius); CHKERRQ(ierr);
	  } else if (NLS_KSP_STCG == nlsP->ksp_type) {
	    ierr = KSPSTCGSetRadius(tao->ksp,nlsP->max_radius); CHKERRQ(ierr);
	  } else if (NLS_KSP_GLTR == nlsP->ksp_type) {
	    ierr = KSPGLTRSetRadius(tao->ksp,nlsP->max_radius); CHKERRQ(ierr);
	  }
	
	  ierr = KSPSolve(tao->ksp, tao->gradient, nlsP->D); CHKERRQ(ierr);
	  if (NLS_KSP_NASH == nlsP->ksp_type) {
	    ierr = KSPNASHGetNormD(tao->ksp,&norm_d); CHKERRQ(ierr);
	  } else if (NLS_KSP_STCG == nlsP->ksp_type) {
	    ierr = KSPSTCGGetNormD(tao->ksp,&norm_d); CHKERRQ(ierr);
	  } else if (NLS_KSP_GLTR == nlsP->ksp_type) {
	    ierr = KSPGLTRGetNormD(tao->ksp,&norm_d); CHKERRQ(ierr);
	  }

          if (norm_d == 0.0) {
            SETERRQ(PETSC_COMM_SELF,1, "Initial direction zero");
          }
        }
      }
    }
    else {
      ierr = KSPSolve(tao->ksp, tao->gradient, nlsP->D); CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(tao->ksp, &kspits); CHKERRQ(ierr);
    }
    ierr = VecScale(nlsP->D, -1.0); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(tao->ksp, &ksp_reason); CHKERRQ(ierr);
    if ((KSP_DIVERGED_INDEFINITE_PC == ksp_reason) &&
        (NLS_PC_BFGS == nlsP->pc_type) && (bfgsUpdates > 1)) {
      // Preconditioner is numerically indefinite; reset the
      // approximate if using BFGS preconditioning.

      if (f != 0.0) {
        delta = 2.0 * PetscAbsScalar(f) / (gnorm*gnorm);
      }
      else {
        delta = 2.0 / (gnorm*gnorm);
      }
      ierr = MatLMVMSetDelta(nlsP->M,delta); CHKERRQ(ierr);
      ierr = MatLMVMReset(nlsP->M); CHKERRQ(ierr);
      ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
      bfgsUpdates = 1;
    }

    if (KSP_CONVERGED_ATOL == ksp_reason) {
      ++nlsP->ksp_atol;
    }
    else if (KSP_CONVERGED_RTOL == ksp_reason) {
      ++nlsP->ksp_rtol;
    }
    else if (KSP_CONVERGED_CG_CONSTRAINED == ksp_reason) {
      ++nlsP->ksp_ctol;
    }
    else if (KSP_CONVERGED_CG_NEG_CURVE == ksp_reason) {
      ++nlsP->ksp_negc;
    }
    else if (KSP_DIVERGED_DTOL == ksp_reason) {
      ++nlsP->ksp_dtol;
    }
    else if (KSP_DIVERGED_ITS == ksp_reason) {
      ++nlsP->ksp_iter;
    }
    else {
      ++nlsP->ksp_othr;
    } 

    // Check for success (descent direction)
    ierr = VecDot(nlsP->D, tao->gradient, &gdx); CHKERRQ(ierr);
    if ((gdx >= 0.0) || TaoInfOrNaN(gdx)) {
      // Newton step is not descent or direction produced Inf or NaN
      // Update the perturbation for next time
      if (pert <= 0.0) {
	// Initialize the perturbation
	pert = PetscMin(nlsP->imax, PetscMax(nlsP->imin, nlsP->imfac * gnorm));
        if (NLS_KSP_GLTR == nlsP->ksp_type) {
	  ierr = KSPGLTRGetMinEig(tao->ksp,&e_min); CHKERRQ(ierr);
	  pert = PetscMax(pert, -e_min);
        }
      }
      else {
	// Increase the perturbation
	pert = PetscMin(nlsP->pmax, PetscMax(nlsP->pgfac * pert, nlsP->pmgfac * gnorm));
      }

      if (NLS_PC_BFGS != nlsP->pc_type) {
	// We don't have the bfgs matrix around and updated
        // Must use gradient direction in this case
	ierr = VecCopy(tao->gradient, nlsP->D); CHKERRQ(ierr);
	ierr = VecScale(nlsP->D, -1.0); CHKERRQ(ierr);
	++nlsP->grad;
        stepType = NLS_GRADIENT;
      }
      else {
        // Attempt to use the BFGS direction
	ierr = MatLMVMSolve(nlsP->M, tao->gradient, nlsP->D); CHKERRQ(ierr);
	ierr = VecScale(nlsP->D, -1.0); CHKERRQ(ierr);

        // Check for success (descent direction)
	ierr = VecDot(tao->gradient, nlsP->D, &gdx); CHKERRQ(ierr);
        if ((gdx >= 0) || TaoInfOrNaN(gdx)) {
          // BFGS direction is not descent or direction produced not a number
          // We can assert bfgsUpdates > 1 in this case because
          // the first solve produces the scaled gradient direction,
          // which is guaranteed to be descent
	  //
          // Use steepest descent direction (scaled)

          if (f != 0.0) {
            delta = 2.0 * PetscAbsScalar(f) / (gnorm*gnorm);
          }
          else {
            delta = 2.0 / (gnorm*gnorm);
          }
	  ierr = MatLMVMSetDelta(nlsP->M, delta); CHKERRQ(ierr);
	  ierr = MatLMVMReset(nlsP->M); CHKERRQ(ierr);
	  ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
	  ierr = MatLMVMSolve(nlsP->M, tao->gradient, nlsP->D); CHKERRQ(ierr);
	  ierr = VecScale(nlsP->D, -1.0); CHKERRQ(ierr);
  
          bfgsUpdates = 1;
          ++nlsP->sgrad;
          stepType = NLS_SCALED_GRADIENT;
        }
        else {
          if (1 == bfgsUpdates) {
	    // The first BFGS direction is always the scaled gradient
            ++nlsP->sgrad;
            stepType = NLS_SCALED_GRADIENT;
          }
          else {
            ++nlsP->bfgs;
            stepType = NLS_BFGS;
          }
        }
      }
    }
    else {
      // Computed Newton step is descent
      switch (ksp_reason) {
      case KSP_DIVERGED_NAN:
      case KSP_DIVERGED_BREAKDOWN:
      case KSP_DIVERGED_INDEFINITE_MAT:
      case KSP_DIVERGED_INDEFINITE_PC:
      case KSP_CONVERGED_CG_NEG_CURVE:
        // Matrix or preconditioner is indefinite; increase perturbation
        if (pert <= 0.0) {
	  // Initialize the perturbation
          pert = PetscMin(nlsP->imax, PetscMax(nlsP->imin, nlsP->imfac * gnorm));
          if (NLS_KSP_GLTR == nlsP->ksp_type) {
	    ierr = KSPGLTRGetMinEig(tao->ksp, &e_min); CHKERRQ(ierr);
	    pert = PetscMax(pert, -e_min);
          }
        }
        else {
	  // Increase the perturbation
	  pert = PetscMin(nlsP->pmax, PetscMax(nlsP->pgfac * pert, nlsP->pmgfac * gnorm));
        }
        break;

      default:
        // Newton step computation is good; decrease perturbation
        pert = PetscMin(nlsP->psfac * pert, nlsP->pmsfac * gnorm);
        if (pert < nlsP->pmin) {
	  pert = 0.0;
        }
        break; 
      }

      ++nlsP->newt;
      stepType = NLS_NEWTON;
    }

    // Perform the linesearch
    fold = f;
    ierr = VecCopy(tao->solution, nlsP->Xold); CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, nlsP->Gold); CHKERRQ(ierr);

    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, nlsP->D, &step, &ls_reason); CHKERRQ(ierr);

    while (ls_reason < 0  && stepType != NLS_GRADIENT) {
      // Linesearch failed
      f = fold;
      ierr = VecCopy(nlsP->Xold, tao->solution); CHKERRQ(ierr);
      ierr = VecCopy(nlsP->Gold, tao->gradient); CHKERRQ(ierr);

      switch(stepType) {
      case NLS_NEWTON:
        // Failed to obtain acceptable iterate with Newton 1step
        // Update the perturbation for next time
        if (pert <= 0.0) {
          // Initialize the perturbation
          pert = PetscMin(nlsP->imax, PetscMax(nlsP->imin, nlsP->imfac * gnorm));
          if (NLS_KSP_GLTR == nlsP->ksp_type) {
	    ierr = KSPGLTRGetMinEig(tao->ksp,&e_min); CHKERRQ(ierr);
	    pert = PetscMax(pert, -e_min);
          }
        }
        else {
          // Increase the perturbation
          pert = PetscMin(nlsP->pmax, PetscMax(nlsP->pgfac * pert, nlsP->pmgfac * gnorm));
        }

        if (NLS_PC_BFGS != nlsP->pc_type) {
	  // We don't have the bfgs matrix around and being updated
          // Must use gradient direction in this case
	  ierr = VecCopy(tao->gradient, nlsP->D); CHKERRQ(ierr);
	  ++nlsP->grad;
          stepType = NLS_GRADIENT;
        }
        else {
          // Attempt to use the BFGS direction
	  ierr = MatLMVMSolve(nlsP->M, tao->gradient, nlsP->D); CHKERRQ(ierr);
          // Check for success (descent direction)
	  ierr = VecDot(tao->solution, nlsP->D, &gdx); CHKERRQ(ierr);
          if ((gdx <= 0) || TaoInfOrNaN(gdx)) {
            // BFGS direction is not descent or direction produced not a number
            // We can assert bfgsUpdates > 1 in this case
            // Use steepest descent direction (scaled)
    
            if (f != 0.0) {
              delta = 2.0 * PetscAbsScalar(f) / (gnorm*gnorm);
            }
            else {
              delta = 2.0 / (gnorm*gnorm);
            }
	    ierr = MatLMVMSetDelta(nlsP->M, delta); CHKERRQ(ierr);
	    ierr = MatLMVMReset(nlsP->M); CHKERRQ(ierr);
	    ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
	    ierr = MatLMVMSolve(nlsP->M, tao->gradient, nlsP->D); CHKERRQ(ierr);
  
            bfgsUpdates = 1;
            ++nlsP->sgrad;
            stepType = NLS_SCALED_GRADIENT;
          }
          else {
            if (1 == bfgsUpdates) {
	      // The first BFGS direction is always the scaled gradient
              ++nlsP->sgrad;
              stepType = NLS_SCALED_GRADIENT;
            }
            else {
              ++nlsP->bfgs;
              stepType = NLS_BFGS;
            }
          }
        }
	break;

      case NLS_BFGS:
        // Can only enter if pc_type == NLS_PC_BFGS
        // Failed to obtain acceptable iterate with BFGS step
        // Attempt to use the scaled gradient direction

        if (f != 0.0) {
          delta = 2.0 * PetscAbsScalar(f) / (gnorm*gnorm);
        }
        else {
          delta = 2.0 / (gnorm*gnorm);
        }
	ierr = MatLMVMSetDelta(nlsP->M, delta); CHKERRQ(ierr);
	ierr = MatLMVMReset(nlsP->M); CHKERRQ(ierr);
	ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
	ierr = MatLMVMSolve(nlsP->M, tao->gradient, nlsP->D); CHKERRQ(ierr);

        bfgsUpdates = 1;
        ++nlsP->sgrad;
        stepType = NLS_SCALED_GRADIENT;
        break;

      case NLS_SCALED_GRADIENT:
        // Can only enter if pc_type == NLS_PC_BFGS
        // The scaled gradient step did not produce a new iterate;
        // attemp to use the gradient direction.
        // Need to make sure we are not using a different diagonal scaling
	
	ierr = MatLMVMSetScale(nlsP->M,0); CHKERRQ(ierr);
	ierr = MatLMVMSetDelta(nlsP->M,1.0); CHKERRQ(ierr);
	ierr = MatLMVMReset(nlsP->M); CHKERRQ(ierr);
	ierr = MatLMVMUpdate(nlsP->M, tao->solution, tao->gradient); CHKERRQ(ierr);
	ierr = MatLMVMSolve(nlsP->M, tao->gradient, nlsP->D); CHKERRQ(ierr);
	
        bfgsUpdates = 1;
	++nlsP->grad;
        stepType = NLS_GRADIENT;
        break;
      }
      ierr = VecScale(nlsP->D, -1.0); CHKERRQ(ierr);

      ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, nlsP->D, &step, &ls_reason); CHKERRQ(ierr);
      ierr = TaoLineSearchGetFullStepObjective(tao->linesearch, &f_full); CHKERRQ(ierr);
    }

    if (ls_reason < 0) {
      // Failed to find an improving point
      f = fold;
      ierr = VecCopy(nlsP->Xold, tao->solution); CHKERRQ(ierr);
      ierr = VecCopy(nlsP->Gold, tao->gradient); CHKERRQ(ierr);
      step = 0.0;
      reason = TAO_DIVERGED_LS_FAILURE;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
      break;
    }

    // Update trust region radius
    if (NLS_KSP_NASH == nlsP->ksp_type ||
        NLS_KSP_STCG == nlsP->ksp_type || 
        NLS_KSP_GLTR == nlsP->ksp_type) {
      switch(nlsP->update_type) {
      case NLS_UPDATE_STEP:
        if (stepType == NLS_NEWTON) {
          if (step < nlsP->nu1) {
            // Very bad step taken; reduce radius
            tao->trust = nlsP->omega1 * PetscMin(norm_d, tao->trust);
          }
          else if (step < nlsP->nu2) {
            // Reasonably bad step taken; reduce radius
            tao->trust = nlsP->omega2 * PetscMin(norm_d, tao->trust);
          }
          else if (step < nlsP->nu3) {
            // Reasonable step was taken; leave radius alone
            if (nlsP->omega3 < 1.0) {
              tao->trust = nlsP->omega3 * PetscMin(norm_d, tao->trust);
            }
            else if (nlsP->omega3 > 1.0) {
              tao->trust = PetscMax(nlsP->omega3 * norm_d, tao->trust);  
            }
          }
          else if (step < nlsP->nu4) {
            // Full step taken; increase the radius
            tao->trust = PetscMax(nlsP->omega4 * norm_d, tao->trust);  
          }
          else {
            // More than full step taken; increase the radius
            tao->trust = PetscMax(nlsP->omega5 * norm_d, tao->trust);  
          }
        }
        else {
          // Newton step was not good; reduce the radius
          tao->trust = nlsP->omega1 * PetscMin(norm_d, tao->trust);
        }
        break;

      case NLS_UPDATE_REDUCTION:
        if (stepType == NLS_NEWTON) {
	  // Get predicted reduction

	  if (NLS_KSP_STCG == nlsP->ksp_type) {
	      ierr = KSPSTCGGetObjFcn(tao->ksp,&prered);
	  } else if (NLS_KSP_NASH == nlsP->ksp_type)  {
	      ierr = KSPNASHGetObjFcn(tao->ksp,&prered);
	  } else {
	      ierr = KSPGLTRGetObjFcn(tao->ksp,&prered); CHKERRQ(ierr);
	  }
	  
	      
	  

          if (prered >= 0.0) {
            // The predicted reduction has the wrong sign.  This cannot
            // happen in infinite precision arithmetic.  Step should
            // be rejected!
            tao->trust = nlsP->alpha1 * PetscMin(tao->trust, norm_d);
          }
          else {
            if (TaoInfOrNaN(f_full)) {
              tao->trust = nlsP->alpha1 * PetscMin(tao->trust, norm_d);
            }
            else {
              // Compute and actual reduction
              actred = fold - f_full;
              prered = -prered;
              if ((PetscAbsScalar(actred) <= nlsP->epsilon) && 
                  (PetscAbsScalar(prered) <= nlsP->epsilon)) {
                kappa = 1.0;
              }
              else {
                kappa = actred / prered;
              }
  
              // Accept of reject the step and update radius
              if (kappa < nlsP->eta1) {
                // Very bad step
                tao->trust = nlsP->alpha1 * PetscMin(tao->trust, norm_d);
              }
              else if (kappa < nlsP->eta2) {
                // Marginal bad step
                tao->trust = nlsP->alpha2 * PetscMin(tao->trust, norm_d);
              }
              else if (kappa < nlsP->eta3) {
                // Reasonable step
                if (nlsP->alpha3 < 1.0) {
                  tao->trust = nlsP->alpha3 * PetscMin(norm_d, tao->trust);
                }
                else if (nlsP->alpha3 > 1.0) {
                  tao->trust = PetscMax(nlsP->alpha3 * norm_d, tao->trust);  
                }
              }
              else if (kappa < nlsP->eta4) {
                // Good step
                tao->trust = PetscMax(nlsP->alpha4 * norm_d, tao->trust);
              }
              else {
                // Very good step
                tao->trust = PetscMax(nlsP->alpha5 * norm_d, tao->trust);
              }
            }
          }
        }
        else {
          // Newton step was not good; reduce the radius
          tao->trust = nlsP->alpha1 * PetscMin(norm_d, tao->trust);
        }
        break;

      default:
        if (stepType == NLS_NEWTON) {

	  if (NLS_KSP_STCG == nlsP->ksp_type) {
	      ierr = KSPSTCGGetObjFcn(tao->ksp,&prered);
	  } else if (NLS_KSP_NASH == nlsP->ksp_type)  {
	      ierr = KSPNASHGetObjFcn(tao->ksp,&prered);
	  } else {
	      ierr = KSPGLTRGetObjFcn(tao->ksp,&prered); CHKERRQ(ierr);
	  }
          if (prered >= 0.0) {
            // The predicted reduction has the wrong sign.  This cannot
            // happen in infinite precision arithmetic.  Step should
            // be rejected!
            tao->trust = nlsP->gamma1 * PetscMin(tao->trust, norm_d);
          }
          else {
            if (TaoInfOrNaN(f_full)) {
              tao->trust = nlsP->gamma1 * PetscMin(tao->trust, norm_d);
            }
            else {
              actred = fold - f_full;
              prered = -prered;
              if ((PetscAbsScalar(actred) <= nlsP->epsilon) && 
                  (PetscAbsScalar(prered) <= nlsP->epsilon)) {
                kappa = 1.0;
              }
              else {
                kappa = actred / prered;
              }

              tau_1 = nlsP->theta * gdx / (nlsP->theta * gdx - (1.0 - nlsP->theta) * prered + actred);
              tau_2 = nlsP->theta * gdx / (nlsP->theta * gdx + (1.0 + nlsP->theta) * prered - actred);
              tau_min = PetscMin(tau_1, tau_2);
              tau_max = PetscMax(tau_1, tau_2);

              if (kappa >= 1.0 - nlsP->mu1) {
                // Great agreement
                if (tau_max < 1.0) {
                  tao->trust = PetscMax(tao->trust, nlsP->gamma3 * norm_d);
                }
                else if (tau_max > nlsP->gamma4) {
                  tao->trust = PetscMax(tao->trust, nlsP->gamma4 * norm_d);
                }
                else {
                  tao->trust = PetscMax(tao->trust, tau_max * norm_d);
                }
              }
              else if (kappa >= 1.0 - nlsP->mu2) {
                // Good agreement

                if (tau_max < nlsP->gamma2) {
                  tao->trust = nlsP->gamma2 * PetscMin(tao->trust, norm_d);
                }
                else if (tau_max > nlsP->gamma3) {
                  tao->trust = PetscMax(tao->trust, nlsP->gamma3 * norm_d);
                }
                else if (tau_max < 1.0) {
                  tao->trust = tau_max * PetscMin(tao->trust, norm_d);
                }
                else {
                  tao->trust = PetscMax(tao->trust, tau_max * norm_d);
                }
              }
              else {
                // Not good agreement
                if (tau_min > 1.0) {
                  tao->trust = nlsP->gamma2 * PetscMin(tao->trust, norm_d);
                }
                else if (tau_max < nlsP->gamma1) {
                  tao->trust = nlsP->gamma1 * PetscMin(tao->trust, norm_d);
                }
                else if ((tau_min < nlsP->gamma1) && (tau_max >= 1.0)) {
                  tao->trust = nlsP->gamma1 * PetscMin(tao->trust, norm_d);
                }
                else if ((tau_1 >= nlsP->gamma1) && (tau_1 < 1.0) &&
                         ((tau_2 < nlsP->gamma1) || (tau_2 >= 1.0))) {
                  tao->trust = tau_1 * PetscMin(tao->trust, norm_d);
                }
                else if ((tau_2 >= nlsP->gamma1) && (tau_2 < 1.0) &&
                         ((tau_1 < nlsP->gamma1) || (tau_2 >= 1.0))) {
                  tao->trust = tau_2 * PetscMin(tao->trust, norm_d);
                }
                else {
                  tao->trust = tau_max * PetscMin(tao->trust, norm_d);
                }
              }
            } 
          }
        }
        else {
          // Newton step was not good; reduce the radius
          tao->trust = nlsP->gamma1 * PetscMin(norm_d, tao->trust);
        }
        break;
      }

      // The radius may have been increased; modify if it is too large
      tao->trust = PetscMin(tao->trust, nlsP->max_radius);
    }

    // Check for termination
    ierr = VecNorm(tao->gradient, NORM_2, &gnorm); CHKERRQ(ierr);
    if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
      SETERRQ(PETSC_COMM_SELF,1,"User provided compute function generated Not-a-Number");
    }
    needH = 1;

    ierr = TaoSolverMonitor(tao, iter, f, gnorm, 0.0, step, &reason); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetUp_NLS"
static PetscErrorCode TaoSolverSetUp_NLS(TaoSolver tao)
{
  TAO_NLS *nlsP = (TAO_NLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (!tao->gradient) {ierr = VecDuplicate(tao->solution,&tao->gradient); CHKERRQ(ierr);  }
  if (!tao->stepdirection) {ierr = VecDuplicate(tao->solution,&tao->stepdirection); CHKERRQ(ierr);  }
  if (!nlsP->W) {ierr = VecDuplicate(tao->solution,&nlsP->W); CHKERRQ(ierr);  }
  if (!nlsP->D) {ierr = VecDuplicate(tao->solution,&nlsP->D); CHKERRQ(ierr);  }
  if (!nlsP->Xold) {ierr = VecDuplicate(tao->solution,&nlsP->Xold); CHKERRQ(ierr);  }
  if (!nlsP->Gold) {ierr = VecDuplicate(tao->solution,&nlsP->Gold); CHKERRQ(ierr);  }

  nlsP->Diag = 0;
  nlsP->M = 0;

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverDestroy_NLS"
static PetscErrorCode TaoSolverDestroy_NLS(TaoSolver tao)
{
  TAO_NLS *nlsP = (TAO_NLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&nlsP->D); CHKERRQ(ierr);
    ierr = VecDestroy(&nlsP->W); CHKERRQ(ierr);
    ierr = VecDestroy(&nlsP->Xold); CHKERRQ(ierr);
    ierr = VecDestroy(&nlsP->Gold); CHKERRQ(ierr);
  }
  if (nlsP->Diag) {
    ierr = VecDestroy(&nlsP->Diag); CHKERRQ(ierr);
  }
  if (nlsP->M) {
    ierr = MatDestroy(&nlsP->M); CHKERRQ(ierr);
  }

  ierr = PetscFree(tao->data); CHKERRQ(ierr);
  tao->data = PETSC_NULL;

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverSetFromOptions_NLS"
static PetscErrorCode TaoSolverSetFromOptions_NLS(TaoSolver tao)
{
  TAO_NLS *nlsP = (TAO_NLS *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Newton line search method for unconstrained optimization"); CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_nls_ksp_type", "ksp type", "", NLS_KSP, NLS_KSP_TYPES, NLS_KSP[nlsP->ksp_type], &nlsP->ksp_type, 0); CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_nls_pc_type", "pc type", "", NLS_PC, NLS_PC_TYPES, NLS_PC[nlsP->pc_type], &nlsP->pc_type, 0); CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_nls_bfgs_scale_type", "bfgs scale type", "", BFGS_SCALE, BFGS_SCALE_TYPES, BFGS_SCALE[nlsP->bfgs_scale_type], &nlsP->bfgs_scale_type, 0); CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_nls_init_type", "radius initialization type", "", NLS_INIT, NLS_INIT_TYPES, NLS_INIT[nlsP->init_type], &nlsP->init_type, 0); CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_nls_update_type", "radius update type", "", NLS_UPDATE, NLS_UPDATE_TYPES, NLS_UPDATE[nlsP->update_type], &nlsP->update_type, 0); CHKERRQ(ierr);
 ierr = PetscOptionsReal("-tao_nls_sval", "perturbation starting value", "", nlsP->sval, &nlsP->sval, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_imin", "minimum initial perturbation", "", nlsP->imin, &nlsP->imin, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_imax", "maximum initial perturbation", "", nlsP->imax, &nlsP->imax, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_imfac", "initial merit factor", "", nlsP->imfac, &nlsP->imfac, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pmin", "minimum perturbation", "", nlsP->pmin, &nlsP->pmin, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pmax", "maximum perturbation", "", nlsP->pmax, &nlsP->pmax, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pgfac", "growth factor", "", nlsP->pgfac, &nlsP->pgfac, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_psfac", "shrink factor", "", nlsP->psfac, &nlsP->psfac, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pmgfac", "merit growth factor", "", nlsP->pmgfac, &nlsP->pmgfac, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_pmsfac", "merit shrink factor", "", nlsP->pmsfac, &nlsP->pmsfac, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_eta1", "poor steplength; reduce radius", "", nlsP->eta1, &nlsP->eta1, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_eta2", "reasonable steplength; leave radius alone", "", nlsP->eta2, &nlsP->eta2, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_eta3", "good steplength; increase radius", "", nlsP->eta3, &nlsP->eta3, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_eta4", "excellent steplength; greatly increase radius", "", nlsP->eta4, &nlsP->eta4, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha1", "", "", nlsP->alpha1, &nlsP->alpha1, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha2", "", "", nlsP->alpha2, &nlsP->alpha2, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha3", "", "", nlsP->alpha3, &nlsP->alpha3, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha4", "", "", nlsP->alpha4, &nlsP->alpha4, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_alpha5", "", "", nlsP->alpha5, &nlsP->alpha5, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_nu1", "poor steplength; reduce radius", "", nlsP->nu1, &nlsP->nu1, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_nu2", "reasonable steplength; leave radius alone", "", nlsP->nu2, &nlsP->nu2, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_nu3", "good steplength; increase radius", "", nlsP->nu3, &nlsP->nu3, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_nu4", "excellent steplength; greatly increase radius", "", nlsP->nu4, &nlsP->nu4, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega1", "", "", nlsP->omega1, &nlsP->omega1, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega2", "", "", nlsP->omega2, &nlsP->omega2, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega3", "", "", nlsP->omega3, &nlsP->omega3, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega4", "", "", nlsP->omega4, &nlsP->omega4, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_omega5", "", "", nlsP->omega5, &nlsP->omega5, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_mu1_i", "", "", nlsP->mu1_i, &nlsP->mu1_i, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_mu2_i", "", "", nlsP->mu2_i, &nlsP->mu2_i, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma1_i", "", "", nlsP->gamma1_i, &nlsP->gamma1_i, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma2_i", "", "", nlsP->gamma2_i, &nlsP->gamma2_i, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma3_i", "", "", nlsP->gamma3_i, &nlsP->gamma3_i, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma4_i", "", "", nlsP->gamma4_i, &nlsP->gamma4_i, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_theta_i", "", "", nlsP->theta_i, &nlsP->theta_i, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_mu1", "", "", nlsP->mu1, &nlsP->mu1, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_mu2", "", "", nlsP->mu2, &nlsP->mu2, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma1", "", "", nlsP->gamma1, &nlsP->gamma1, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma2", "", "", nlsP->gamma2, &nlsP->gamma2, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma3", "", "", nlsP->gamma3, &nlsP->gamma3, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_gamma4", "", "", nlsP->gamma4, &nlsP->gamma4, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_theta", "", "", nlsP->theta, &nlsP->theta, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_min_radius", "lower bound on initial radius", "", nlsP->min_radius, &nlsP->min_radius, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_max_radius", "upper bound on radius", "", nlsP->max_radius, &nlsP->max_radius, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nls_epsilon", "tolerance used when computing actual and predicted reduction", "", nlsP->epsilon, &nlsP->epsilon, 0); CHKERRQ(ierr);
  ierr = PetscOptionsTail(); CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverView_NLS"
static PetscErrorCode TaoSolverView_NLS(TaoSolver tao, PetscViewer viewer)
{
  TAO_NLS *nlsP = (TAO_NLS *)tao->data;
  PetscInt nrejects;
  PetscBool isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer); CHKERRQ(ierr);
    if (NLS_PC_BFGS == nlsP->pc_type && nlsP->M) {
      ierr = MatLMVMGetRejects(nlsP->M,&nrejects); CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "Rejected matrix updates: %d\n",nrejects); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer, "Newton steps: %d\n", nlsP->newt); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "BFGS steps: %d\n", nlsP->bfgs); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Scaled gradient steps: %d\n", nlsP->sgrad); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Gradient steps: %d\n", nlsP->grad); CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp atol: %d\n", nlsP->ksp_atol); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp rtol: %d\n", nlsP->ksp_rtol); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp ctol: %d\n", nlsP->ksp_ctol); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp negc: %d\n", nlsP->ksp_negc); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp dtol: %d\n", nlsP->ksp_dtol); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp iter: %d\n", nlsP->ksp_iter); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "nls ksp othr: %d\n", nlsP->ksp_othr); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer); CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)tao)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for TAO NLS",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoSolverCreate_NLS"
PetscErrorCode TaoSolverCreate_NLS(TaoSolver tao)
{
  TAO_NLS *nlsP;
  const char *morethuente_type = TAOLINESEARCH_MT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,TAO_NLS,&nlsP); CHKERRQ(ierr);

  tao->ops->setup = TaoSolverSetUp_NLS;
  tao->ops->solve = TaoSolverSolve_NLS;
  tao->ops->view = TaoSolverView_NLS;
  tao->ops->setfromoptions = TaoSolverSetFromOptions_NLS;
  tao->ops->destroy = TaoSolverDestroy_NLS;

  tao->max_its = 50;
  tao->fatol = 1e-10;
  tao->frtol = 1e-10;
  tao->data = (void*)nlsP;
  tao->trust0 = 100.0;

  //  ierr = TaoSetTrustRegionTolerance(tao, 1.0e-12); CHKERRQ(ierr);

  nlsP->sval   = 0.0;
  nlsP->imin   = 1.0e-4;
  nlsP->imax   = 1.0e+2;
  nlsP->imfac  = 1.0e-1;

  nlsP->pmin   = 1.0e-12;
  nlsP->pmax   = 1.0e+2;
  nlsP->pgfac  = 1.0e+1;
  nlsP->psfac  = 4.0e-1;
  nlsP->pmgfac = 1.0e-1;
  nlsP->pmsfac = 1.0e-1;

  // Default values for trust-region radius update based on steplength
  nlsP->nu1 = 0.25;
  nlsP->nu2 = 0.50;
  nlsP->nu3 = 1.00;
  nlsP->nu4 = 1.25;

  nlsP->omega1 = 0.25;
  nlsP->omega2 = 0.50;
  nlsP->omega3 = 1.00;
  nlsP->omega4 = 2.00;
  nlsP->omega5 = 4.00;

  // Default values for trust-region radius update based on reduction
  nlsP->eta1 = 1.0e-4;
  nlsP->eta2 = 0.25;
  nlsP->eta3 = 0.50;
  nlsP->eta4 = 0.90;

  nlsP->alpha1 = 0.25;
  nlsP->alpha2 = 0.50;
  nlsP->alpha3 = 1.00;
  nlsP->alpha4 = 2.00;
  nlsP->alpha5 = 4.00;

  // Default values for trust-region radius update based on interpolation
  nlsP->mu1 = 0.10;
  nlsP->mu2 = 0.50;

  nlsP->gamma1 = 0.25;
  nlsP->gamma2 = 0.50;
  nlsP->gamma3 = 2.00;
  nlsP->gamma4 = 4.00;

  nlsP->theta = 0.05;

  // Default values for trust region initialization based on interpolation
  nlsP->mu1_i = 0.35;
  nlsP->mu2_i = 0.50;

  nlsP->gamma1_i = 0.0625;
  nlsP->gamma2_i = 0.5;
  nlsP->gamma3_i = 2.0;
  nlsP->gamma4_i = 5.0;
  
  nlsP->theta_i = 0.25;

  // Remaining parameters
  nlsP->min_radius = 1.0e-10;
  nlsP->max_radius = 1.0e10;
  nlsP->epsilon = 1.0e-6;

  nlsP->ksp_type        = NLS_KSP_STCG;
  nlsP->pc_type         = NLS_PC_BFGS;
  nlsP->bfgs_scale_type = BFGS_SCALE_PHESS;
  nlsP->init_type       = NLS_INIT_INTERPOLATION;
  nlsP->update_type     = NLS_UPDATE_STEP;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch); CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch,morethuente_type); CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoSolverRoutines(tao->linesearch,tao); CHKERRQ(ierr);

  // Set linear solver to default for symmetric matrices
  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_END




#undef __FUNCT__
#define __FUNCT__ "MatLMVMSolveShell"
static PetscErrorCode MatLMVMSolveShell(PC pc, Vec b, Vec x) 
{
    PetscErrorCode ierr;
    Mat M;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(pc,PC_CLASSID,1);
    PetscValidHeaderSpecific(b,VEC_CLASSID,2);
    PetscValidHeaderSpecific(x,VEC_CLASSID,3);
    ierr = PCShellGetContext(pc,(void**)&M); CHKERRQ(ierr);
    ierr = MatLMVMSolve(M, b, x); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
