#include <../src/tao/bound/impls/bnk/bnk.h>
#include <petscksp.h>

/*
 Implements Newton's Method with a trust region approach for solving
 bound constrained minimization problems.

 ------------------------------------------------------------

 x_0 = VecMedian(x_0)
 f_0, g_0= TaoComputeObjectiveAndGradient(x_0)
 pg_0 = project(g_0)
 check convergence at pg_0
 needH = TaoBNKInitialize(default:BNK_INIT_INTERPOLATION)
 niter = 0
 step_accepted = false

 while niter <= max_it

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

    while !stepAccepted
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
        needH = True
      else
        f_{k+1} = f_k
        x_{k+1} = x_k
        g_{k+1} = g_k
        pg_{k+1} = pg_k
        if trust == oldTrust
          terminate because we cannot shrink the radius any further
        end
      end

    end
    check convergence at pg_{k+1}
    niter += 1

 end
*/

PetscErrorCode TaoSolve_BNTR(Tao tao)
{
  TAO_BNK           *bnk = (TAO_BNK *)tao->data;
  KSPConvergedReason ksp_reason;

  PetscReal oldTrust, prered, actred, steplen = 0.0, resnorm;
  PetscBool cgTerminate, needH = PETSC_TRUE, stepAccepted, shift = PETSC_FALSE;
  PetscInt  stepType, nDiff;

  PetscFunctionBegin;
  /* Initialize the preconditioner, KSP solver and trust radius/line search */
  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoBNKInitialize(tao, bnk->init_type, &needH));
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      PetscUseTypeMethod(tao, update, tao->niter, tao->user_update);
      PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &bnk->f, bnk->unprojected_gradient));
    }

    if (needH && bnk->inactive_idx) {
      /* Take BNCG steps (if enabled) to trade-off Hessian evaluations for more gradient evaluations */
      PetscCall(TaoBNKTakeCGSteps(tao, &cgTerminate));
      if (cgTerminate) {
        tao->reason = bnk->bncg->reason;
        PetscFunctionReturn(0);
      }
      /* Compute the hessian and update the BFGS preconditioner at the new iterate */
      PetscCall((*bnk->computehessian)(tao));
      needH = PETSC_FALSE;
    }

    /* Store current solution before it changes */
    bnk->fold = bnk->f;
    PetscCall(VecCopy(tao->solution, bnk->Xold));
    PetscCall(VecCopy(tao->gradient, bnk->Gold));
    PetscCall(VecCopy(bnk->unprojected_gradient, bnk->unprojected_gradient_old));

    /* Enter into trust region loops */
    stepAccepted = PETSC_FALSE;
    while (!stepAccepted && tao->reason == TAO_CONTINUE_ITERATING) {
      tao->ksp_its = 0;

      /* Use the common BNK kernel to compute the Newton step (for inactive variables only) */
      PetscCall((*bnk->computestep)(tao, shift, &ksp_reason, &stepType));

      /* Temporarily accept the step and project it into the bounds */
      PetscCall(VecAXPY(tao->solution, 1.0, tao->stepdirection));
      PetscCall(TaoBoundSolution(tao->solution, tao->XL, tao->XU, 0.0, &nDiff, tao->solution));

      /* Check if the projection changed the step direction */
      if (nDiff > 0) {
        /* Projection changed the step, so we have to recompute the step and
           the predicted reduction. Leave the trust radius unchanged. */
        PetscCall(VecCopy(tao->solution, tao->stepdirection));
        PetscCall(VecAXPY(tao->stepdirection, -1.0, bnk->Xold));
        PetscCall(TaoBNKRecomputePred(tao, tao->stepdirection, &prered));
      } else {
        /* Step did not change, so we can just recover the pre-computed prediction */
        PetscCall(KSPCGGetObjFcn(tao->ksp, &prered));
      }
      prered = -prered;

      /* Compute the actual reduction and update the trust radius */
      PetscCall(TaoComputeObjective(tao, tao->solution, &bnk->f));
      PetscCheck(!PetscIsInfOrNanReal(bnk->f), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
      actred   = bnk->fold - bnk->f;
      oldTrust = tao->trust;
      PetscCall(TaoBNKUpdateTrustRadius(tao, prered, actred, bnk->update_type, stepType, &stepAccepted));

      if (stepAccepted) {
        /* Step is good, evaluate the gradient and flip the need-Hessian switch */
        steplen = 1.0;
        needH   = PETSC_TRUE;
        ++bnk->newt;
        PetscCall(TaoComputeGradient(tao, tao->solution, bnk->unprojected_gradient));
        PetscCall(TaoBNKEstimateActiveSet(tao, bnk->as_type));
        PetscCall(VecCopy(bnk->unprojected_gradient, tao->gradient));
        PetscCall(VecISSet(tao->gradient, bnk->active_idx, 0.0));
        PetscCall(TaoGradientNorm(tao, tao->gradient, NORM_2, &bnk->gnorm));
      } else {
        /* Step is bad, revert old solution and re-solve with new radius*/
        steplen = 0.0;
        needH   = PETSC_FALSE;
        bnk->f  = bnk->fold;
        PetscCall(VecCopy(bnk->Xold, tao->solution));
        PetscCall(VecCopy(bnk->Gold, tao->gradient));
        PetscCall(VecCopy(bnk->unprojected_gradient_old, bnk->unprojected_gradient));
        if (oldTrust == tao->trust) {
          /* Can't change the radius anymore so just terminate */
          tao->reason = TAO_DIVERGED_TR_REDUCTION;
        }
      }
    }
    /*  Check for termination */
    PetscCall(VecFischer(tao->solution, bnk->unprojected_gradient, tao->XL, tao->XU, bnk->W));
    PetscCall(VecNorm(bnk->W, NORM_2, &resnorm));
    PetscCheck(!PetscIsInfOrNanReal(resnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
    ++tao->niter;
    PetscCall(TaoLogConvergenceHistory(tao, bnk->f, resnorm, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, bnk->f, resnorm, 0.0, steplen));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetUp_BNTR(Tao tao)
{
  KSP               ksp;
  PetscVoidFunction valid;

  PetscFunctionBegin;
  PetscCall(TaoSetUp_BNK(tao));
  PetscCall(TaoGetKSP(tao, &ksp));
  PetscCall(PetscObjectQueryFunction((PetscObject)ksp, "KSPCGSetRadius_C", &valid));
  PetscCheck(valid, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "Not for KSP type %s. Must use a trust-region CG method for KSP (e.g. KSPNASH, KSPSTCG, KSPGLTR)", ((PetscObject)ksp)->type_name);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TaoSetFromOptions_BNTR(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_BNK *bnk = (TAO_BNK *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoSetFromOptions_BNK(tao, PetscOptionsObject));
  if (bnk->update_type == BNK_UPDATE_STEP) bnk->update_type = BNK_UPDATE_REDUCTION;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
/*MC
  TAOBNTR - Bounded Newton Trust Region for nonlinear minimization with bound constraints.

  Options Database Keys:
+ -tao_bnk_max_cg_its - maximum number of bounded conjugate-gradient iterations taken in each Newton loop
. -tao_bnk_init_type - trust radius initialization method ("constant", "direction", "interpolation")
. -tao_bnk_update_type - trust radius update method ("step", "direction", "interpolation")
- -tao_bnk_as_type - active-set estimation method ("none", "bertsekas")

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BNTR(Tao tao)
{
  TAO_BNK *bnk;

  PetscFunctionBegin;
  PetscCall(TaoCreate_BNK(tao));
  tao->ops->solve          = TaoSolve_BNTR;
  tao->ops->setup          = TaoSetUp_BNTR;
  tao->ops->setfromoptions = TaoSetFromOptions_BNTR;

  bnk              = (TAO_BNK *)tao->data;
  bnk->update_type = BNK_UPDATE_REDUCTION; /* trust region updates based on predicted/actual reduction */
  PetscFunctionReturn(0);
}
