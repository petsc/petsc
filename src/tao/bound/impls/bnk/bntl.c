#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

/*
 Implements Newton's Method with a trust region approach for solving
 bound constrained minimization problems.

 In this variant, the trust region failures trigger a line search with
 the existing Newton step instead of re-solving the step with a
 different radius.

 ------------------------------------------------------------

 x_0 = VecMedian(x_0)
 f_0, g_0 = TaoComputeObjectiveAndGradient(x_0)
 pg_0 = project(g_0)
 check convergence at pg_0
 needH = TaoBNKInitialize(default:BNK_INIT_INTERPOLATION)
 niter = 0
 step_accepted = true

 while niter <= max_it
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
    if pc_type == BNK_PC_BFGS && scale_type == BNK_SCALE_PHESS
      D = VecMedian(1e-6, abs(diag(Hr_k)), 1e6)
      scale BFGS with VecReciprocal(D)
    end
    solve Hr_k dr_k = -gr_k
    set d_k to (l - x) for variables in L, (u - x) for variables in U, and 0 for variables in F

    x_{k+1} = VecMedian(x_k + d_k)
    s = x_{k+1} - x_k
    prered = dot(s, 0.5*gr_k - Hr_k*s)
    f_{k+1} = TaoComputeObjective(x_{k+1})
    actred = f_k - f_{k+1}

    oldTrust = trust
    step_accepted, trust = TaoBNKUpdateTrustRadius(default: BNK_UPDATE_REDUCTION)
    if step_accepted
      g_{k+1} = TaoComputeGradient(x_{k+1})
      pg_{k+1} = project(g_{k+1})
      count the accepted Newton step
    else
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
        trust = oldTrust
        trust = TaoBNKUpdateTrustRadius(BNK_UPDATE_STEP)
        count the accepted step type (Newton, BFGS, scaled grad or grad)
      end
    end

    check convergence at pg_{k+1}
 end
*/

PetscErrorCode TaoSolve_BNTL(Tao tao)
{
  TAO_BNK                      *bnk = (TAO_BNK *)tao->data;
  KSPConvergedReason           ksp_reason;
  TaoLineSearchConvergedReason ls_reason;

  PetscReal                    oldTrust, prered, actred, steplen, resnorm;
  PetscBool                    cgTerminate, needH = PETSC_TRUE, stepAccepted, shift = PETSC_FALSE;
  PetscInt                     stepType, nDiff;

  PetscFunctionBegin;
  /* Initialize the preconditioner, KSP solver and trust radius/line search */
  tao->reason = TAO_CONTINUE_ITERATING;
  CHKERRQ(TaoBNKInitialize(tao, bnk->init_type, &needH));
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    ++tao->niter;

    if (needH && bnk->inactive_idx) {
      /* Take BNCG steps (if enabled) to trade-off Hessian evaluations for more gradient evaluations */
      CHKERRQ(TaoBNKTakeCGSteps(tao, &cgTerminate));
      if (cgTerminate) {
        tao->reason = bnk->bncg->reason;
        PetscFunctionReturn(0);
      }
      /* Compute the hessian and update the BFGS preconditioner at the new iterate */
      CHKERRQ((*bnk->computehessian)(tao));
      needH = PETSC_FALSE;
    }

    /* Use the common BNK kernel to compute the Newton step (for inactive variables only) */
    CHKERRQ((*bnk->computestep)(tao, shift, &ksp_reason, &stepType));

    /* Store current solution before it changes */
    oldTrust = tao->trust;
    bnk->fold = bnk->f;
    CHKERRQ(VecCopy(tao->solution, bnk->Xold));
    CHKERRQ(VecCopy(tao->gradient, bnk->Gold));
    CHKERRQ(VecCopy(bnk->unprojected_gradient, bnk->unprojected_gradient_old));

    /* Temporarily accept the step and project it into the bounds */
    CHKERRQ(VecAXPY(tao->solution, 1.0, tao->stepdirection));
    CHKERRQ(TaoBoundSolution(tao->solution, tao->XL,tao->XU, 0.0, &nDiff, tao->solution));

    /* Check if the projection changed the step direction */
    if (nDiff > 0) {
      /* Projection changed the step, so we have to recompute the step and
         the predicted reduction. Leave the trust radius unchanged. */
      CHKERRQ(VecCopy(tao->solution, tao->stepdirection));
      CHKERRQ(VecAXPY(tao->stepdirection, -1.0, bnk->Xold));
      CHKERRQ(TaoBNKRecomputePred(tao, tao->stepdirection, &prered));
    } else {
      /* Step did not change, so we can just recover the pre-computed prediction */
      CHKERRQ(KSPCGGetObjFcn(tao->ksp, &prered));
    }
    prered = -prered;

    /* Compute the actual reduction and update the trust radius */
    CHKERRQ(TaoComputeObjective(tao, tao->solution, &bnk->f));
    PetscCheck(!PetscIsInfOrNanReal(bnk->f),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
    actred = bnk->fold - bnk->f;
    CHKERRQ(TaoBNKUpdateTrustRadius(tao, prered, actred, bnk->update_type, stepType, &stepAccepted));

    if (stepAccepted) {
      /* Step is good, evaluate the gradient and the hessian */
      steplen = 1.0;
      needH = PETSC_TRUE;
      ++bnk->newt;
      CHKERRQ(TaoComputeGradient(tao, tao->solution, bnk->unprojected_gradient));
      CHKERRQ(TaoBNKEstimateActiveSet(tao, bnk->as_type));
      CHKERRQ(VecCopy(bnk->unprojected_gradient, tao->gradient));
      CHKERRQ(VecISSet(tao->gradient, bnk->active_idx, 0.0));
      CHKERRQ(TaoGradientNorm(tao, tao->gradient, NORM_2, &bnk->gnorm));
    } else {
      /* Trust-region rejected the step. Revert the solution. */
      bnk->f = bnk->fold;
      CHKERRQ(VecCopy(bnk->Xold, tao->solution));
      /* Trigger the line search */
      CHKERRQ(TaoBNKSafeguardStep(tao, ksp_reason, &stepType));
      CHKERRQ(TaoBNKPerformLineSearch(tao, &stepType, &steplen, &ls_reason));
      if (ls_reason != TAOLINESEARCH_SUCCESS && ls_reason != TAOLINESEARCH_SUCCESS_USER) {
        /* Line search failed, revert solution and terminate */
        stepAccepted = PETSC_FALSE;
        needH = PETSC_FALSE;
        bnk->f = bnk->fold;
        CHKERRQ(VecCopy(bnk->Xold, tao->solution));
        CHKERRQ(VecCopy(bnk->Gold, tao->gradient));
        CHKERRQ(VecCopy(bnk->unprojected_gradient_old, bnk->unprojected_gradient));
        tao->trust = 0.0;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
      } else {
        /* new iterate so we need to recompute the Hessian */
        needH = PETSC_TRUE;
        /* compute the projected gradient */
        CHKERRQ(TaoBNKEstimateActiveSet(tao, bnk->as_type));
        CHKERRQ(VecCopy(bnk->unprojected_gradient, tao->gradient));
        CHKERRQ(VecISSet(tao->gradient, bnk->active_idx, 0.0));
        CHKERRQ(TaoGradientNorm(tao, tao->gradient, NORM_2, &bnk->gnorm));
        /* Line search succeeded so we should update the trust radius based on the LS step length */
        tao->trust = oldTrust;
        CHKERRQ(TaoBNKUpdateTrustRadius(tao, prered, actred, BNK_UPDATE_STEP, stepType, &stepAccepted));
        /* count the accepted step type */
        CHKERRQ(TaoBNKAddStepCounts(tao, stepType));
      }
    }

    /*  Check for termination */
    CHKERRQ(VecFischer(tao->solution, bnk->unprojected_gradient, tao->XL, tao->XU, bnk->W));
    CHKERRQ(VecNorm(bnk->W, NORM_2, &resnorm));
    PetscCheck(!PetscIsInfOrNanReal(resnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
    CHKERRQ(TaoLogConvergenceHistory(tao, bnk->f, resnorm, 0.0, tao->ksp_its));
    CHKERRQ(TaoMonitor(tao, tao->niter, bnk->f, resnorm, 0.0, steplen));
    CHKERRQ((*tao->ops->convergencetest)(tao, tao->cnvP));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetUp_BNTL(Tao tao)
{
  KSP               ksp;
  PetscVoidFunction valid;

  PetscFunctionBegin;
  CHKERRQ(TaoSetUp_BNK(tao));
  CHKERRQ(TaoGetKSP(tao,&ksp));
  CHKERRQ(PetscObjectQueryFunction((PetscObject)ksp,"KSPCGSetRadius_C",&valid));
  PetscCheck(valid,PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Not for KSP type %s. Must use a trust-region CG method for KSP (e.g. KSPNASH, KSPSTCG, KSPGLTR)",((PetscObject)ksp)->type_name);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_BNTL(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;

  PetscFunctionBegin;
  CHKERRQ(TaoSetFromOptions_BNK(PetscOptionsObject, tao));
  if (bnk->update_type == BNK_UPDATE_STEP) bnk->update_type = BNK_UPDATE_REDUCTION;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
/*MC
  TAOBNTL - Bounded Newton Trust Region method with line-search fall-back for nonlinear
            minimization with bound constraints.

  Options Database Keys:
  + -tao_bnk_max_cg_its - maximum number of bounded conjugate-gradient iterations taken in each Newton loop
  . -tao_bnk_init_type - trust radius initialization method ("constant", "direction", "interpolation")
  . -tao_bnk_update_type - trust radius update method ("step", "direction", "interpolation")
  - -tao_bnk_as_type - active-set estimation method ("none", "bertsekas")

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BNTL(Tao tao)
{
  TAO_BNK        *bnk;

  PetscFunctionBegin;
  CHKERRQ(TaoCreate_BNK(tao));
  tao->ops->solve=TaoSolve_BNTL;
  tao->ops->setup=TaoSetUp_BNTL;
  tao->ops->setfromoptions=TaoSetFromOptions_BNTL;

  bnk = (TAO_BNK *)tao->data;
  bnk->update_type = BNK_UPDATE_REDUCTION; /* trust region updates based on predicted/actual reduction */
  PetscFunctionReturn(0);
}
