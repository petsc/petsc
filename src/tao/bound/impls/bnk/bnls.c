#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

/*
 Implements Newton's Method with a line search approach for 
 solving bound constrained minimization problems.
 
 ------------------------------------------------------------
 
 x_0 = VecMedian(x_0)
 f_0, g_0 = TaoComputeObjectiveAndGradient(x_0)
 pg_0 = project(g_0)
 check convergence at pg_0
 needH = TaoBNKInitialize(default:BNK_INIT_DIRECTION)
 niter = 0
 step_accepted = true
 
 while niter < max_it
    niter += 1
    
    if needH
      If max_cg_steps > 0
        x_k, g_k, pg_k = TaoSolve(BNCG)
      end

      H_k = TaoComputeHessian(x_k)
      if pc_type == BNK_PC_BFGS
        add correction to BFGS approx
        if scale_type == BNK_SCALE_AHESS
          D = VecMedian(1e-6, abs(diag(H_k)), 1e6)
          scale BFGS with VecReciprocal(D)
        end
      end
      needH = False
    end
    
    if pc_type = BNK_PC_BFGS
      B_k = BFGS
    else
      B_k = VecMedian(1e-6, abs(diag(H_k)), 1e6)
      B_k = VecReciprocal(B_k)
    end
    w = x_k - VecMedian(x_k - 0.001*B_k*g_k)
    eps = min(eps, norm2(w))
    determine the active and inactive index sets such that
      L = {i : (x_k)_i <= l_i + eps && (g_k)_i > 0}
      U = {i : (x_k)_i >= u_i - eps && (g_k)_i < 0}
      F = {i : l_i = (x_k)_i = u_i}
      A = {L + U + F}
      IA = {i : i not in A}
    
    generate the reduced system Hr_k dr_k = -gr_k for variables in IA
    if p > 0
      Hr_k += p*
    end
    if pc_type == BNK_PC_BFGS && scale_type == BNK_SCALE_PHESS
      D = VecMedian(1e-6, abs(diag(Hr_k)), 1e6)
      scale BFGS with VecReciprocal(D)
    end
    solve Hr_k dr_k = -gr_k
    set d_k to (l - x) for variables in L, (u - x) for variables in U, and 0 for variables in F
    
    if dot(d_k, pg_k)) >= 0 || norm(d_k) == NaN || norm(d_k) == Inf
      dr_k = -BFGS*gr_k for variables in I
      if dot(d_k, pg_k)) >= 0 || norm(d_k) == NaN || norm(d_k) == Inf
        reset the BFGS preconditioner
        calculate scale delta and apply it to BFGS
        dr_k = -BFGS*gr_k for variables in I
        if dot(d_k, pg_k)) >= 0 || norm(d_k) == NaN || norm(d_k) == Inf
          dr_k = -gr_k for variables in I
        end
      end
    end
    
    x_{k+1}, f_{k+1}, g_{k+1}, ls_failed = TaoBNKPerformLineSearch()
    if ls_failed
      f_{k+1} = f_k
      x_{k+1} = x_k
      g_{k+1} = g_k
      pg_{k+1} = pg_k
      terminate
    else
      pg_{k+1} = project(g_{k+1})
      count the accepted step type (Newton, BFGS, scaled grad or grad)
    end 
    
    check convergence at pg_{k+1}
 end
*/

PetscErrorCode TaoSolve_BNLS(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  KSPConvergedReason           ksp_reason;
  TaoLineSearchConvergedReason ls_reason;

  PetscReal                    steplen = 1.0, resnorm;
  PetscBool                    cgTerminate, needH = PETSC_TRUE, stepAccepted, shift = PETSC_TRUE;
  PetscInt                     stepType;
  
  PetscFunctionBegin;
  /* Initialize the preconditioner, KSP solver and trust radius/line search */
  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoBNKInitialize(tao, bnk->init_type, &needH);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }
    ++tao->niter;
    
    if (needH && bnk->inactive_idx) {
      /* Take BNCG steps (if enabled) to trade-off Hessian evaluations for more gradient evaluations */
      ierr = TaoBNKTakeCGSteps(tao, &cgTerminate);CHKERRQ(ierr);
      if (cgTerminate) {
        tao->reason = bnk->bncg->reason;
        PetscFunctionReturn(0);
      }
      /* Compute the hessian and update the BFGS preconditioner at the new iterate */
      ierr = (*bnk->computehessian)(tao);CHKERRQ(ierr);
      needH = PETSC_FALSE;
    }
    
    /* Use the common BNK kernel to compute the safeguarded Newton step (for inactive variables only) */
    ierr = (*bnk->computestep)(tao, shift, &ksp_reason, &stepType);CHKERRQ(ierr);
    ierr = TaoBNKSafeguardStep(tao, ksp_reason, &stepType);CHKERRQ(ierr);

    /* Store current solution before it changes */
    bnk->fold = bnk->f;
    ierr = VecCopy(tao->solution, bnk->Xold);CHKERRQ(ierr);
    ierr = VecCopy(tao->gradient, bnk->Gold);CHKERRQ(ierr);
    ierr = VecCopy(bnk->unprojected_gradient, bnk->unprojected_gradient_old);CHKERRQ(ierr);
    
    /* Trigger the line search */
    ierr = TaoBNKPerformLineSearch(tao, &stepType, &steplen, &ls_reason);CHKERRQ(ierr);

    if (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER) {
      /* Failed to find an improving point */
      needH = PETSC_FALSE;
      bnk->f = bnk->fold;
      ierr = VecCopy(bnk->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(bnk->Gold, tao->gradient);CHKERRQ(ierr);
      ierr = VecCopy(bnk->unprojected_gradient_old, bnk->unprojected_gradient);CHKERRQ(ierr);
      steplen = 0.0;
      tao->reason = TAO_DIVERGED_LS_FAILURE;
    } else {
      /* new iterate so we need to recompute the Hessian */
      needH = PETSC_TRUE;
      /* compute the projected gradient */
      ierr = TaoBNKEstimateActiveSet(tao, bnk->as_type);CHKERRQ(ierr);
      ierr = VecCopy(bnk->unprojected_gradient, tao->gradient);CHKERRQ(ierr);
      ierr = VecISSet(tao->gradient, bnk->active_idx, 0.0);CHKERRQ(ierr);
      ierr = TaoGradientNorm(tao, tao->gradient, NORM_2, &bnk->gnorm);CHKERRQ(ierr);
      /* update the trust radius based on the step length */
      ierr = TaoBNKUpdateTrustRadius(tao, 0.0, 0.0, BNK_UPDATE_STEP, stepType, &stepAccepted);CHKERRQ(ierr);
      /* count the accepted step type */
      ierr = TaoBNKAddStepCounts(tao, stepType);CHKERRQ(ierr);
      /* active BNCG recycling for next iteration */
      ierr = TaoBNCGSetRecycleFlag(bnk->bncg, PETSC_TRUE);CHKERRQ(ierr);
    }

    /*  Check for termination */
    ierr = VecFischer(tao->solution, bnk->unprojected_gradient, tao->XL, tao->XU, bnk->W);CHKERRQ(ierr);
    ierr = VecNorm(bnk->W, NORM_2, &resnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(resnorm)) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
    ierr = TaoLogConvergenceHistory(tao, bnk->f, resnorm, 0.0, tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao, tao->niter, bnk->f, resnorm, 0.0, steplen);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao, tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
/*MC
  TAOBNLS - Bounded Newton Line Search for nonlinear minimization with bound constraints.

  Options Database Keys:
+ -tao_bnk_max_cg_its - maximum number of bounded conjugate-gradient iterations taken in each Newton loop
. -tao_bnk_init_type - trust radius initialization method ("constant", "direction", "interpolation")
. -tao_bnk_update_type - trust radius update method ("step", "direction", "interpolation")
- -tao_bnk_as_type - active-set estimation method ("none", "bertsekas")

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BNLS(Tao tao)
{
  TAO_BNK        *bnk;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoCreate_BNK(tao);CHKERRQ(ierr);
  tao->ops->solve = TaoSolve_BNLS;
  
  bnk = (TAO_BNK *)tao->data;
  bnk->init_type = BNK_INIT_DIRECTION;
  bnk->update_type = BNK_UPDATE_STEP; /* trust region updates based on line search step length */
  PetscFunctionReturn(0);
}
