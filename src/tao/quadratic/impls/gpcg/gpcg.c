#include <petscksp.h>
#include <../src/tao/quadratic/impls/gpcg/gpcg.h>        /*I "gpcg.h" I*/

static PetscErrorCode GPCGGradProjections(Tao tao);
static PetscErrorCode GPCGObjectiveAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);

/*------------------------------------------------------------*/
static PetscErrorCode TaoDestroy_GPCG(Tao tao)
{
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;

  /* Free allocated memory in GPCG structure */
  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&gpcg->B));
  CHKERRQ(VecDestroy(&gpcg->Work));
  CHKERRQ(VecDestroy(&gpcg->X_New));
  CHKERRQ(VecDestroy(&gpcg->G_New));
  CHKERRQ(VecDestroy(&gpcg->DXFree));
  CHKERRQ(VecDestroy(&gpcg->R));
  CHKERRQ(VecDestroy(&gpcg->PG));
  CHKERRQ(MatDestroy(&gpcg->Hsub));
  CHKERRQ(MatDestroy(&gpcg->Hsub_pre));
  CHKERRQ(ISDestroy(&gpcg->Free_Local));
  CHKERRQ(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_GPCG(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Gradient Projection, Conjugate Gradient method for bound constrained optimization"));
  CHKERRQ(PetscOptionsInt("-tao_gpcg_maxpgits","maximum number of gradient projections per GPCG iterate",NULL,gpcg->maxgpits,&gpcg->maxgpits,&flg));
  CHKERRQ(PetscOptionsTail());
  CHKERRQ(KSPSetFromOptions(tao->ksp));
  CHKERRQ(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_GPCG(Tao tao, PetscViewer viewer)
{
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Total PG its: %D,",gpcg->total_gp_its));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"PG tolerance: %g \n",(double)gpcg->pg_ftol));
  }
  CHKERRQ(TaoLineSearchView(tao->linesearch,viewer));
  PetscFunctionReturn(0);
}

/* GPCGObjectiveAndGradient()
   Compute f=0.5 * x'Hx + b'x + c
           g=Hx + b
*/
static PetscErrorCode GPCGObjectiveAndGradient(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void*tptr)
{
  Tao            tao = (Tao)tptr;
  TAO_GPCG       *gpcg = (TAO_GPCG*)tao->data;
  PetscReal      f1,f2;

  PetscFunctionBegin;
  CHKERRQ(MatMult(tao->hessian,X,G));
  CHKERRQ(VecDot(G,X,&f1));
  CHKERRQ(VecDot(gpcg->B,X,&f2));
  CHKERRQ(VecAXPY(G,1.0,gpcg->B));
  *f=f1/2.0 + f2 + gpcg->c;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoSetup_GPCG(Tao tao)
{
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;

  PetscFunctionBegin;
  /* Allocate some arrays */
  if (!tao->gradient) {
      CHKERRQ(VecDuplicate(tao->solution, &tao->gradient));
  }
  if (!tao->stepdirection) {
      CHKERRQ(VecDuplicate(tao->solution, &tao->stepdirection));
  }
  if (!tao->XL) {
      CHKERRQ(VecDuplicate(tao->solution,&tao->XL));
      CHKERRQ(VecSet(tao->XL,PETSC_NINFINITY));
  }
  if (!tao->XU) {
      CHKERRQ(VecDuplicate(tao->solution,&tao->XU));
      CHKERRQ(VecSet(tao->XU,PETSC_INFINITY));
  }

  CHKERRQ(VecDuplicate(tao->solution,&gpcg->B));
  CHKERRQ(VecDuplicate(tao->solution,&gpcg->Work));
  CHKERRQ(VecDuplicate(tao->solution,&gpcg->X_New));
  CHKERRQ(VecDuplicate(tao->solution,&gpcg->G_New));
  CHKERRQ(VecDuplicate(tao->solution,&gpcg->DXFree));
  CHKERRQ(VecDuplicate(tao->solution,&gpcg->R));
  CHKERRQ(VecDuplicate(tao->solution,&gpcg->PG));
  /*
    if (gpcg->ksp_type == GPCG_KSP_NASH) {
        CHKERRQ(KSPSetType(tao->ksp,KSPNASH));
      } else if (gpcg->ksp_type == GPCG_KSP_STCG) {
        CHKERRQ(KSPSetType(tao->ksp,KSPSTCG));
      } else {
        CHKERRQ(KSPSetType(tao->ksp,KSPGLTR));
      }
      if (tao->ksp->ops->setfromoptions) {
        (*tao->ksp->ops->setfromoptions)(tao->ksp);
      }

    }
  */
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_GPCG(Tao tao)
{
  TAO_GPCG                     *gpcg = (TAO_GPCG *)tao->data;
  PetscInt                     its;
  PetscReal                    actred,f,f_new,gnorm,gdx,stepsize,xtb;
  PetscReal                    xtHx;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;

  CHKERRQ(TaoComputeVariableBounds(tao));
  CHKERRQ(VecMedian(tao->XL,tao->solution,tao->XU,tao->solution));
  CHKERRQ(TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU));

  /* Using f = .5*x'Hx + x'b + c and g=Hx + b,  compute b,c */
  CHKERRQ(TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre));
  CHKERRQ(TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient));
  CHKERRQ(VecCopy(tao->gradient, gpcg->B));
  CHKERRQ(MatMult(tao->hessian,tao->solution,gpcg->Work));
  CHKERRQ(VecDot(gpcg->Work, tao->solution, &xtHx));
  CHKERRQ(VecAXPY(gpcg->B,-1.0,gpcg->Work));
  CHKERRQ(VecDot(gpcg->B,tao->solution,&xtb));
  gpcg->c=f-xtHx/2.0-xtb;
  if (gpcg->Free_Local) {
      CHKERRQ(ISDestroy(&gpcg->Free_Local));
  }
  CHKERRQ(VecWhichInactive(tao->XL,tao->solution,tao->gradient,tao->XU,PETSC_TRUE,&gpcg->Free_Local));

  /* Project the gradient and calculate the norm */
  CHKERRQ(VecCopy(tao->gradient,gpcg->G_New));
  CHKERRQ(VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU,gpcg->PG));
  CHKERRQ(VecNorm(gpcg->PG,NORM_2,&gpcg->gnorm));
  tao->step=1.0;
  gpcg->f = f;

    /* Check Stopping Condition      */
  tao->reason = TAO_CONTINUE_ITERATING;
  CHKERRQ(TaoLogConvergenceHistory(tao,f,gpcg->gnorm,0.0,tao->ksp_its));
  CHKERRQ(TaoMonitor(tao,tao->niter,f,gpcg->gnorm,0.0,tao->step));
  CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    tao->ksp_its=0;

    CHKERRQ(GPCGGradProjections(tao));
    CHKERRQ(ISGetSize(gpcg->Free_Local,&gpcg->n_free));

    f=gpcg->f; gnorm=gpcg->gnorm;

    CHKERRQ(KSPReset(tao->ksp));

    if (gpcg->n_free > 0) {
      /* Create a reduced linear system */
      CHKERRQ(VecDestroy(&gpcg->R));
      CHKERRQ(VecDestroy(&gpcg->DXFree));
      CHKERRQ(TaoVecGetSubVec(tao->gradient,gpcg->Free_Local, tao->subset_type, 0.0, &gpcg->R));
      CHKERRQ(VecScale(gpcg->R, -1.0));
      CHKERRQ(TaoVecGetSubVec(tao->stepdirection,gpcg->Free_Local,tao->subset_type, 0.0, &gpcg->DXFree));
      CHKERRQ(VecSet(gpcg->DXFree,0.0));

      CHKERRQ(TaoMatGetSubMat(tao->hessian, gpcg->Free_Local, gpcg->Work, tao->subset_type, &gpcg->Hsub));

      if (tao->hessian_pre == tao->hessian) {
        CHKERRQ(MatDestroy(&gpcg->Hsub_pre));
        CHKERRQ(PetscObjectReference((PetscObject)gpcg->Hsub));
        gpcg->Hsub_pre = gpcg->Hsub;
      }  else {
        CHKERRQ(TaoMatGetSubMat(tao->hessian, gpcg->Free_Local, gpcg->Work, tao->subset_type, &gpcg->Hsub_pre));
      }

      CHKERRQ(KSPReset(tao->ksp));
      CHKERRQ(KSPSetOperators(tao->ksp,gpcg->Hsub,gpcg->Hsub_pre));

      CHKERRQ(KSPSolve(tao->ksp,gpcg->R,gpcg->DXFree));
      CHKERRQ(KSPGetIterationNumber(tao->ksp,&its));
      tao->ksp_its+=its;
      tao->ksp_tot_its+=its;
      CHKERRQ(VecSet(tao->stepdirection,0.0));
      CHKERRQ(VecISAXPY(tao->stepdirection,gpcg->Free_Local,1.0,gpcg->DXFree));

      CHKERRQ(VecDot(tao->stepdirection,tao->gradient,&gdx));
      CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,1.0));
      f_new=f;
      CHKERRQ(TaoLineSearchApply(tao->linesearch,tao->solution,&f_new,tao->gradient,tao->stepdirection,&stepsize,&ls_status));

      actred = f_new - f;

      /* Evaluate the function and gradient at the new point */
      CHKERRQ(VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU, gpcg->PG));
      CHKERRQ(VecNorm(gpcg->PG, NORM_2, &gnorm));
      f=f_new;
      CHKERRQ(ISDestroy(&gpcg->Free_Local));
      CHKERRQ(VecWhichInactive(tao->XL,tao->solution,tao->gradient,tao->XU,PETSC_TRUE,&gpcg->Free_Local));
    } else {
      actred = 0; gpcg->step=1.0;
      /* if there were no free variables, no cg method */
    }

    tao->niter++;
    gpcg->f=f;gpcg->gnorm=gnorm; gpcg->actred=actred;
    CHKERRQ(TaoLogConvergenceHistory(tao,f,gpcg->gnorm,0.0,tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter,f,gpcg->gnorm,0.0,tao->step));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
    if (tao->reason != TAO_CONTINUE_ITERATING) break;
  }  /* END MAIN LOOP  */

  PetscFunctionReturn(0);
}

static PetscErrorCode GPCGGradProjections(Tao tao)
{
  TAO_GPCG                       *gpcg = (TAO_GPCG *)tao->data;
  PetscInt                       i;
  PetscReal                      actred=-1.0,actred_max=0.0, gAg,gtg=gpcg->gnorm,alpha;
  PetscReal                      f_new,gdx,stepsize;
  Vec                            DX=tao->stepdirection,XL=tao->XL,XU=tao->XU,Work=gpcg->Work;
  Vec                            X=tao->solution,G=tao->gradient;
  TaoLineSearchConvergedReason lsflag=TAOLINESEARCH_CONTINUE_ITERATING;

  /*
     The free, active, and binding variables should be already identified
  */
  PetscFunctionBegin;
  for (i=0;i<gpcg->maxgpits;i++) {
    if (-actred <= (gpcg->pg_ftol)*actred_max) break;
    CHKERRQ(VecBoundGradientProjection(G,X,XL,XU,DX));
    CHKERRQ(VecScale(DX,-1.0));
    CHKERRQ(VecDot(DX,G,&gdx));

    CHKERRQ(MatMult(tao->hessian,DX,Work));
    CHKERRQ(VecDot(DX,Work,&gAg));

    gpcg->gp_iterates++;
    gpcg->total_gp_its++;

    gtg=-gdx;
    if (PetscAbsReal(gAg) == 0.0) {
      alpha = 1.0;
    } else {
      alpha = PetscAbsReal(gtg/gAg);
    }
    CHKERRQ(TaoLineSearchSetInitialStepLength(tao->linesearch,alpha));
    f_new=gpcg->f;
    CHKERRQ(TaoLineSearchApply(tao->linesearch,X,&f_new,G,DX,&stepsize,&lsflag));

    /* Update the iterate */
    actred = f_new - gpcg->f;
    actred_max = PetscMax(actred_max,-(f_new - gpcg->f));
    gpcg->f = f_new;
    CHKERRQ(ISDestroy(&gpcg->Free_Local));
    CHKERRQ(VecWhichInactive(XL,X,tao->gradient,XU,PETSC_TRUE,&gpcg->Free_Local));
  }

  gpcg->gnorm=gtg;
  PetscFunctionReturn(0);
} /* End gradient projections */

static PetscErrorCode TaoComputeDual_GPCG(Tao tao, Vec DXL, Vec DXU)
{
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;

  PetscFunctionBegin;
  CHKERRQ(VecBoundGradientProjection(tao->gradient, tao->solution, tao->XL, tao->XU, gpcg->Work));
  CHKERRQ(VecCopy(gpcg->Work, DXL));
  CHKERRQ(VecAXPY(DXL,-1.0,tao->gradient));
  CHKERRQ(VecSet(DXU,0.0));
  CHKERRQ(VecPointwiseMax(DXL,DXL,DXU));

  CHKERRQ(VecCopy(tao->gradient,DXU));
  CHKERRQ(VecAXPY(DXU,-1.0,gpcg->Work));
  CHKERRQ(VecSet(gpcg->Work,0.0));
  CHKERRQ(VecPointwiseMin(DXU,gpcg->Work,DXU));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
/*MC
  TAOGPCG - gradient projected conjugate gradient algorithm is an active-set
        conjugate-gradient based method for bound-constrained minimization

  Options Database Keys:
+ -tao_gpcg_maxpgits - maximum number of gradient projections for GPCG iterate
- -tao_subset_type - "subvec","mask","matrix-free", strategies for handling active-sets

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_GPCG(Tao tao)
{
  TAO_GPCG       *gpcg;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_GPCG;
  tao->ops->solve = TaoSolve_GPCG;
  tao->ops->view  = TaoView_GPCG;
  tao->ops->setfromoptions = TaoSetFromOptions_GPCG;
  tao->ops->destroy = TaoDestroy_GPCG;
  tao->ops->computedual = TaoComputeDual_GPCG;

  CHKERRQ(PetscNewLog(tao,&gpcg));
  tao->data = (void*)gpcg;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it=500;
  if (!tao->max_funcs_changed) tao->max_funcs = 100000;
#if defined(PETSC_USE_REAL_SINGLE)
  if (!tao->gatol_changed) tao->gatol=1e-6;
  if (!tao->grtol_changed) tao->grtol=1e-6;
#else
  if (!tao->gatol_changed) tao->gatol=1e-12;
  if (!tao->grtol_changed) tao->grtol=1e-12;
#endif

  /* Initialize pointers and variables */
  gpcg->n=0;
  gpcg->maxgpits = 8;
  gpcg->pg_ftol = 0.1;

  gpcg->gp_iterates=0; /* Cumulative number */
  gpcg->total_gp_its = 0;

  /* Initialize pointers and variables */
  gpcg->n_bind=0;
  gpcg->n_free = 0;
  gpcg->n_upper=0;
  gpcg->n_lower=0;
  gpcg->subset_type = TAO_SUBSET_MASK;
  gpcg->Hsub=NULL;
  gpcg->Hsub_pre=NULL;

  CHKERRQ(KSPCreate(((PetscObject)tao)->comm, &tao->ksp));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  CHKERRQ(KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix));
  CHKERRQ(KSPSetType(tao->ksp,KSPNASH));

  CHKERRQ(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  CHKERRQ(TaoLineSearchSetType(tao->linesearch, TAOLINESEARCHGPCG));
  CHKERRQ(TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch, GPCGObjectiveAndGradient, tao));
  CHKERRQ(TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix));
  PetscFunctionReturn(0);
}
