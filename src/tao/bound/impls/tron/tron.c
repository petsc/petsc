#include <../src/tao/bound/impls/tron/tron.h>
#include <../src/tao/matrix/submatfree.h>

/* TRON Routines */
static PetscErrorCode TronGradientProjections(Tao,TAO_TRON*);
/*------------------------------------------------------------*/
static PetscErrorCode TaoDestroy_TRON(Tao tao)
{
  TAO_TRON       *tron = (TAO_TRON *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&tron->X_New));
  PetscCall(VecDestroy(&tron->G_New));
  PetscCall(VecDestroy(&tron->Work));
  PetscCall(VecDestroy(&tron->DXFree));
  PetscCall(VecDestroy(&tron->R));
  PetscCall(VecDestroy(&tron->diag));
  PetscCall(VecScatterDestroy(&tron->scatter));
  PetscCall(ISDestroy(&tron->Free_Local));
  PetscCall(MatDestroy(&tron->H_sub));
  PetscCall(MatDestroy(&tron->Hpre_sub));
  PetscCall(KSPDestroy(&tao->ksp));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_TRON(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_TRON       *tron = (TAO_TRON *)tao->data;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Newton Trust Region Method for bound constrained optimization");
  PetscCall(PetscOptionsInt("-tao_tron_maxgpits","maximum number of gradient projections per TRON iterate","TaoSetMaxGPIts",tron->maxgpits,&tron->maxgpits,&flg));
  PetscOptionsHeadEnd();
  PetscCall(KSPSetFromOptions(tao->ksp));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_TRON(Tao tao, PetscViewer viewer)
{
  TAO_TRON         *tron = (TAO_TRON *)tao->data;
  PetscBool        isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Total PG its: %" PetscInt_FMT ",",tron->total_gp_its));
    PetscCall(PetscViewerASCIIPrintf(viewer,"PG tolerance: %g \n",(double)tron->pg_ftol));
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoSetup_TRON(Tao tao)
{
  TAO_TRON       *tron = (TAO_TRON *)tao->data;

  PetscFunctionBegin;
  /* Allocate some arrays */
  PetscCall(VecDuplicate(tao->solution, &tron->diag));
  PetscCall(VecDuplicate(tao->solution, &tron->X_New));
  PetscCall(VecDuplicate(tao->solution, &tron->G_New));
  PetscCall(VecDuplicate(tao->solution, &tron->Work));
  PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_TRON(Tao tao)
{
  TAO_TRON                     *tron = (TAO_TRON *)tao->data;
  PetscInt                     its;
  TaoLineSearchConvergedReason ls_reason = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    prered,actred,delta,f,f_new,rhok,gdx,xdiff,stepsize;

  PetscFunctionBegin;
  tron->pgstepsize = 1.0;
  tao->trust = tao->trust0;
  /*   Project the current point onto the feasible set */
  PetscCall(TaoComputeVariableBounds(tao));
  PetscCall(TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU));

  /* Project the initial point onto the feasible region */
  PetscCall(VecMedian(tao->XL,tao->solution,tao->XU,tao->solution));

  /* Compute the objective function and gradient */
  PetscCall(TaoComputeObjectiveAndGradient(tao,tao->solution,&tron->f,tao->gradient));
  PetscCall(VecNorm(tao->gradient,NORM_2,&tron->gnorm));
  PetscCheck(!PetscIsInfOrNanReal(tron->f) && !PetscIsInfOrNanReal(tron->gnorm),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  /* Project the gradient and calculate the norm */
  PetscCall(VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU,tao->gradient));
  PetscCall(VecNorm(tao->gradient,NORM_2,&tron->gnorm));

  /* Initialize trust region radius */
  tao->trust=tao->trust0;
  if (tao->trust <= 0) {
    tao->trust=PetscMax(tron->gnorm*tron->gnorm,1.0);
  }

  /* Initialize step sizes for the line searches */
  tron->pgstepsize=1.0;
  tron->stepsize=tao->trust;

  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao,tron->f,tron->gnorm,0.0,tao->ksp_its));
  PetscCall(TaoMonitor(tao,tao->niter,tron->f,tron->gnorm,0.0,tron->stepsize));
  PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
  while (tao->reason==TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      PetscCall((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }

    /* Perform projected gradient iterations */
    PetscCall(TronGradientProjections(tao,tron));

    PetscCall(VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU,tao->gradient));
    PetscCall(VecNorm(tao->gradient,NORM_2,&tron->gnorm));

    tao->ksp_its=0;
    f=tron->f; delta=tao->trust;
    tron->n_free_last = tron->n_free;
    PetscCall(TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre));

    /* Generate index set (IS) of which bound constraints are active */
    PetscCall(ISDestroy(&tron->Free_Local));
    PetscCall(VecWhichInactive(tao->XL,tao->solution,tao->gradient,tao->XU,PETSC_TRUE,&tron->Free_Local));
    PetscCall(ISGetSize(tron->Free_Local, &tron->n_free));

    /* If no free variables */
    if (tron->n_free == 0) {
      PetscCall(VecNorm(tao->gradient,NORM_2,&tron->gnorm));
      PetscCall(TaoLogConvergenceHistory(tao,tron->f,tron->gnorm,0.0,tao->ksp_its));
      PetscCall(TaoMonitor(tao,tao->niter,tron->f,tron->gnorm,0.0,delta));
      PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
      if (!tao->reason) {
        tao->reason = TAO_CONVERGED_STEPTOL;
      }
      break;
    }
    /* use free_local to mask/submat gradient, hessian, stepdirection */
    PetscCall(TaoVecGetSubVec(tao->gradient,tron->Free_Local,tao->subset_type,0.0,&tron->R));
    PetscCall(TaoVecGetSubVec(tao->gradient,tron->Free_Local,tao->subset_type,0.0,&tron->DXFree));
    PetscCall(VecSet(tron->DXFree,0.0));
    PetscCall(VecScale(tron->R, -1.0));
    PetscCall(TaoMatGetSubMat(tao->hessian, tron->Free_Local, tron->diag, tao->subset_type, &tron->H_sub));
    if (tao->hessian == tao->hessian_pre) {
      PetscCall(MatDestroy(&tron->Hpre_sub));
      PetscCall(PetscObjectReference((PetscObject)(tron->H_sub)));
      tron->Hpre_sub = tron->H_sub;
    } else {
      PetscCall(TaoMatGetSubMat(tao->hessian_pre, tron->Free_Local, tron->diag, tao->subset_type,&tron->Hpre_sub));
    }
    PetscCall(KSPReset(tao->ksp));
    PetscCall(KSPSetOperators(tao->ksp, tron->H_sub, tron->Hpre_sub));
    while (1) {

      /* Approximately solve the reduced linear system */
      PetscCall(KSPCGSetRadius(tao->ksp,delta));

      PetscCall(KSPSolve(tao->ksp, tron->R, tron->DXFree));
      PetscCall(KSPGetIterationNumber(tao->ksp,&its));
      tao->ksp_its+=its;
      tao->ksp_tot_its+=its;
      PetscCall(VecSet(tao->stepdirection,0.0));

      /* Add dxfree matrix to compute step direction vector */
      PetscCall(VecISAXPY(tao->stepdirection,tron->Free_Local,1.0,tron->DXFree));

      PetscCall(VecDot(tao->gradient, tao->stepdirection, &gdx));
      PetscCall(VecCopy(tao->solution, tron->X_New));
      PetscCall(VecCopy(tao->gradient, tron->G_New));

      stepsize=1.0;f_new=f;

      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
      PetscCall(TaoLineSearchApply(tao->linesearch, tron->X_New, &f_new, tron->G_New, tao->stepdirection,&stepsize,&ls_reason));
      PetscCall(TaoAddLineSearchCounts(tao));

      PetscCall(MatMult(tao->hessian, tao->stepdirection, tron->Work));
      PetscCall(VecAYPX(tron->Work, 0.5, tao->gradient));
      PetscCall(VecDot(tao->stepdirection, tron->Work, &prered));
      actred = f_new - f;
      if ((PetscAbsScalar(actred) <= 1e-6) && (PetscAbsScalar(prered) <= 1e-6)) {
        rhok = 1.0;
      } else if (actred<0) {
        rhok=PetscAbs(-actred/prered);
      } else {
        rhok=0.0;
      }

      /* Compare actual improvement to the quadratic model */
      if (rhok > tron->eta1) { /* Accept the point */
        /* d = x_new - x */
        PetscCall(VecCopy(tron->X_New, tao->stepdirection));
        PetscCall(VecAXPY(tao->stepdirection, -1.0, tao->solution));

        PetscCall(VecNorm(tao->stepdirection, NORM_2, &xdiff));
        xdiff *= stepsize;

        /* Adjust trust region size */
        if (rhok < tron->eta2) {
          delta = PetscMin(xdiff,delta)*tron->sigma1;
        } else if (rhok > tron->eta4) {
          delta= PetscMin(xdiff,delta)*tron->sigma3;
        } else if (rhok > tron->eta3) {
          delta=PetscMin(xdiff,delta)*tron->sigma2;
        }
        PetscCall(VecBoundGradientProjection(tron->G_New,tron->X_New, tao->XL, tao->XU, tao->gradient));
        PetscCall(ISDestroy(&tron->Free_Local));
        PetscCall(VecWhichInactive(tao->XL,tron->X_New,tao->gradient,tao->XU,PETSC_TRUE,&tron->Free_Local));
        f=f_new;
        PetscCall(VecNorm(tao->gradient,NORM_2,&tron->gnorm));
        PetscCall(VecCopy(tron->X_New, tao->solution));
        PetscCall(VecCopy(tron->G_New, tao->gradient));
        break;
      }
      else if (delta <= 1e-30) {
        break;
      }
      else {
        delta /= 4.0;
      }
    } /* end linear solve loop */

    tron->f=f; tron->actred=actred; tao->trust=delta;
    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao,tron->f,tron->gnorm,0.0,tao->ksp_its));
    PetscCall(TaoMonitor(tao,tao->niter,tron->f,tron->gnorm,0.0,stepsize));
    PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
  }  /* END MAIN LOOP  */
  PetscFunctionReturn(0);
}

static PetscErrorCode TronGradientProjections(Tao tao,TAO_TRON *tron)
{
  PetscInt                     i;
  TaoLineSearchConvergedReason ls_reason;
  PetscReal                    actred=-1.0,actred_max=0.0;
  PetscReal                    f_new;
  /*
     The gradient and function value passed into and out of this
     routine should be current and correct.

     The free, active, and binding variables should be already identified
  */
  PetscFunctionBegin;

  for (i=0;i<tron->maxgpits;++i) {

    if (-actred <= (tron->pg_ftol)*actred_max) break;

    ++tron->gp_iterates;
    ++tron->total_gp_its;
    f_new=tron->f;

    PetscCall(VecCopy(tao->gradient,tao->stepdirection));
    PetscCall(VecScale(tao->stepdirection,-1.0));
    PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch,tron->pgstepsize));
    PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f_new, tao->gradient, tao->stepdirection,&tron->pgstepsize, &ls_reason));
    PetscCall(TaoAddLineSearchCounts(tao));

    PetscCall(VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU,tao->gradient));
    PetscCall(VecNorm(tao->gradient,NORM_2,&tron->gnorm));

    /* Update the iterate */
    actred = f_new - tron->f;
    actred_max = PetscMax(actred_max,-(f_new - tron->f));
    tron->f = f_new;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoComputeDual_TRON(Tao tao, Vec DXL, Vec DXU)
{

  TAO_TRON       *tron = (TAO_TRON *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(DXL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(DXU,VEC_CLASSID,3);
  PetscCheck(tron->Work && tao->gradient,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Dual variables don't exist yet or no longer exist.");

  PetscCall(VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU,tron->Work));
  PetscCall(VecCopy(tron->Work,DXL));
  PetscCall(VecAXPY(DXL,-1.0,tao->gradient));
  PetscCall(VecSet(DXU,0.0));
  PetscCall(VecPointwiseMax(DXL,DXL,DXU));

  PetscCall(VecCopy(tao->gradient,DXU));
  PetscCall(VecAXPY(DXU,-1.0,tron->Work));
  PetscCall(VecSet(tron->Work,0.0));
  PetscCall(VecPointwiseMin(DXU,tron->Work,DXU));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
/*MC
  TAOTRON - The TRON algorithm is an active-set Newton trust region method
  for bound-constrained minimization.

  Options Database Keys:
+ -tao_tron_maxgpits - maximum number of gradient projections per TRON iterate
- -tao_subset_type - "subvec","mask","matrix-free", strategies for handling active-sets

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_TRON(Tao tao)
{
  TAO_TRON       *tron;
  const char     *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  tao->ops->setup          = TaoSetup_TRON;
  tao->ops->solve          = TaoSolve_TRON;
  tao->ops->view           = TaoView_TRON;
  tao->ops->setfromoptions = TaoSetFromOptions_TRON;
  tao->ops->destroy        = TaoDestroy_TRON;
  tao->ops->computedual    = TaoComputeDual_TRON;

  PetscCall(PetscNewLog(tao,&tron));
  tao->data = (void*)tron;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 50;
  if (!tao->trust0_changed) tao->trust0 = 1.0;
  if (!tao->steptol_changed) tao->steptol = 0.0;

  /* Initialize pointers and variables */
  tron->n            = 0;
  tron->maxgpits     = 3;
  tron->pg_ftol      = 0.001;

  tron->eta1         = 1.0e-4;
  tron->eta2         = 0.25;
  tron->eta3         = 0.50;
  tron->eta4         = 0.90;

  tron->sigma1       = 0.5;
  tron->sigma2       = 2.0;
  tron->sigma3       = 4.0;

  tron->gp_iterates  = 0; /* Cumulative number */
  tron->total_gp_its = 0;
  tron->n_free       = 0;

  tron->DXFree=NULL;
  tron->R=NULL;
  tron->X_New=NULL;
  tron->G_New=NULL;
  tron->Work=NULL;
  tron->Free_Local=NULL;
  tron->H_sub=NULL;
  tron->Hpre_sub=NULL;
  tao->subset_type = TAO_SUBSET_SUBVEC;

  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch,morethuente_type));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch,tao));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));

  PetscCall(KSPCreate(((PetscObject)tao)->comm, &tao->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  PetscCall(KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix));
  PetscCall(KSPSetType(tao->ksp,KSPSTCG));
  PetscFunctionReturn(0);
}
