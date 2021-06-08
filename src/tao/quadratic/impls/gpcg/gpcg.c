#include <petscksp.h>
#include <../src/tao/quadratic/impls/gpcg/gpcg.h>        /*I "gpcg.h" I*/

static PetscErrorCode GPCGGradProjections(Tao tao);
static PetscErrorCode GPCGObjectiveAndGradient(TaoLineSearch,Vec,PetscReal*,Vec,void*);

/*------------------------------------------------------------*/
static PetscErrorCode TaoDestroy_GPCG(Tao tao)
{
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;
  PetscErrorCode ierr;

  /* Free allocated memory in GPCG structure */
  PetscFunctionBegin;
  ierr = VecDestroy(&gpcg->B);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->Work);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->X_New);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->G_New);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->DXFree);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->R);CHKERRQ(ierr);
  ierr = VecDestroy(&gpcg->PG);CHKERRQ(ierr);
  ierr = MatDestroy(&gpcg->Hsub);CHKERRQ(ierr);
  ierr = MatDestroy(&gpcg->Hsub_pre);CHKERRQ(ierr);
  ierr = ISDestroy(&gpcg->Free_Local);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_GPCG(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Gradient Projection, Conjugate Gradient method for bound constrained optimization");CHKERRQ(ierr);
  ierr=PetscOptionsInt("-tao_gpcg_maxpgits","maximum number of gradient projections per GPCG iterate",NULL,gpcg->maxgpits,&gpcg->maxgpits,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_GPCG(Tao tao, PetscViewer viewer)
{
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Total PG its: %D,",gpcg->total_gp_its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"PG tolerance: %g \n",(double)gpcg->pg_ftol);CHKERRQ(ierr);
  }
  ierr = TaoLineSearchView(tao->linesearch,viewer);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscReal      f1,f2;

  PetscFunctionBegin;
  ierr = MatMult(tao->hessian,X,G);CHKERRQ(ierr);
  ierr = VecDot(G,X,&f1);CHKERRQ(ierr);
  ierr = VecDot(gpcg->B,X,&f2);CHKERRQ(ierr);
  ierr = VecAXPY(G,1.0,gpcg->B);CHKERRQ(ierr);
  *f=f1/2.0 + f2 + gpcg->c;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoSetup_GPCG(Tao tao)
{
  PetscErrorCode ierr;
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;

  PetscFunctionBegin;
  /* Allocate some arrays */
  if (!tao->gradient) {
      ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
  }
  if (!tao->stepdirection) {
      ierr = VecDuplicate(tao->solution, &tao->stepdirection);CHKERRQ(ierr);
  }
  if (!tao->XL) {
      ierr = VecDuplicate(tao->solution,&tao->XL);CHKERRQ(ierr);
      ierr = VecSet(tao->XL,PETSC_NINFINITY);CHKERRQ(ierr);
  }
  if (!tao->XU) {
      ierr = VecDuplicate(tao->solution,&tao->XU);CHKERRQ(ierr);
      ierr = VecSet(tao->XU,PETSC_INFINITY);CHKERRQ(ierr);
  }

  ierr=VecDuplicate(tao->solution,&gpcg->B);CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->Work);CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->X_New);CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->G_New);CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->DXFree);CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->R);CHKERRQ(ierr);
  ierr=VecDuplicate(tao->solution,&gpcg->PG);CHKERRQ(ierr);
  /*
    if (gpcg->ksp_type == GPCG_KSP_NASH) {
        ierr = KSPSetType(tao->ksp,KSPNASH);CHKERRQ(ierr);
      } else if (gpcg->ksp_type == GPCG_KSP_STCG) {
        ierr = KSPSetType(tao->ksp,KSPSTCG);CHKERRQ(ierr);
      } else {
        ierr = KSPSetType(tao->ksp,KSPGLTR);CHKERRQ(ierr);
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
  PetscErrorCode               ierr;
  PetscInt                     its;
  PetscReal                    actred,f,f_new,gnorm,gdx,stepsize,xtb;
  PetscReal                    xtHx;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;

  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = VecMedian(tao->XL,tao->solution,tao->XU,tao->solution);CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);

  /* Using f = .5*x'Hx + x'b + c and g=Hx + b,  compute b,c */
  ierr = TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
  ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&f,tao->gradient);CHKERRQ(ierr);
  ierr = VecCopy(tao->gradient, gpcg->B);CHKERRQ(ierr);
  ierr = MatMult(tao->hessian,tao->solution,gpcg->Work);CHKERRQ(ierr);
  ierr = VecDot(gpcg->Work, tao->solution, &xtHx);CHKERRQ(ierr);
  ierr = VecAXPY(gpcg->B,-1.0,gpcg->Work);CHKERRQ(ierr);
  ierr = VecDot(gpcg->B,tao->solution,&xtb);CHKERRQ(ierr);
  gpcg->c=f-xtHx/2.0-xtb;
  if (gpcg->Free_Local) {
      ierr = ISDestroy(&gpcg->Free_Local);CHKERRQ(ierr);
  }
  ierr = VecWhichInactive(tao->XL,tao->solution,tao->gradient,tao->XU,PETSC_TRUE,&gpcg->Free_Local);CHKERRQ(ierr);

  /* Project the gradient and calculate the norm */
  ierr = VecCopy(tao->gradient,gpcg->G_New);CHKERRQ(ierr);
  ierr = VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU,gpcg->PG);CHKERRQ(ierr);
  ierr = VecNorm(gpcg->PG,NORM_2,&gpcg->gnorm);CHKERRQ(ierr);
  tao->step=1.0;
  gpcg->f = f;

    /* Check Stopping Condition      */
  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao,f,gpcg->gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,f,gpcg->gnorm,0.0,tao->step);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }
    tao->ksp_its=0;

    ierr = GPCGGradProjections(tao);CHKERRQ(ierr);
    ierr = ISGetSize(gpcg->Free_Local,&gpcg->n_free);CHKERRQ(ierr);

    f=gpcg->f; gnorm=gpcg->gnorm;

    ierr = KSPReset(tao->ksp);CHKERRQ(ierr);

    if (gpcg->n_free > 0) {
      /* Create a reduced linear system */
      ierr = VecDestroy(&gpcg->R);CHKERRQ(ierr);
      ierr = VecDestroy(&gpcg->DXFree);CHKERRQ(ierr);
      ierr = TaoVecGetSubVec(tao->gradient,gpcg->Free_Local, tao->subset_type, 0.0, &gpcg->R);CHKERRQ(ierr);
      ierr = VecScale(gpcg->R, -1.0);CHKERRQ(ierr);
      ierr = TaoVecGetSubVec(tao->stepdirection,gpcg->Free_Local,tao->subset_type, 0.0, &gpcg->DXFree);CHKERRQ(ierr);
      ierr = VecSet(gpcg->DXFree,0.0);CHKERRQ(ierr);

      ierr = TaoMatGetSubMat(tao->hessian, gpcg->Free_Local, gpcg->Work, tao->subset_type, &gpcg->Hsub);CHKERRQ(ierr);

      if (tao->hessian_pre == tao->hessian) {
        ierr = MatDestroy(&gpcg->Hsub_pre);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)gpcg->Hsub);CHKERRQ(ierr);
        gpcg->Hsub_pre = gpcg->Hsub;
      }  else {
        ierr = TaoMatGetSubMat(tao->hessian, gpcg->Free_Local, gpcg->Work, tao->subset_type, &gpcg->Hsub_pre);CHKERRQ(ierr);
      }

      ierr = KSPReset(tao->ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(tao->ksp,gpcg->Hsub,gpcg->Hsub_pre);CHKERRQ(ierr);

      ierr = KSPSolve(tao->ksp,gpcg->R,gpcg->DXFree);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
      tao->ksp_its+=its;
      tao->ksp_tot_its+=its;
      ierr = VecSet(tao->stepdirection,0.0);CHKERRQ(ierr);
      ierr = VecISAXPY(tao->stepdirection,gpcg->Free_Local,1.0,gpcg->DXFree);CHKERRQ(ierr);

      ierr = VecDot(tao->stepdirection,tao->gradient,&gdx);CHKERRQ(ierr);
      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);CHKERRQ(ierr);
      f_new=f;
      ierr = TaoLineSearchApply(tao->linesearch,tao->solution,&f_new,tao->gradient,tao->stepdirection,&stepsize,&ls_status);CHKERRQ(ierr);

      actred = f_new - f;

      /* Evaluate the function and gradient at the new point */
      ierr = VecBoundGradientProjection(tao->gradient,tao->solution,tao->XL,tao->XU, gpcg->PG);CHKERRQ(ierr);
      ierr = VecNorm(gpcg->PG, NORM_2, &gnorm);CHKERRQ(ierr);
      f=f_new;
      ierr = ISDestroy(&gpcg->Free_Local);CHKERRQ(ierr);
      ierr = VecWhichInactive(tao->XL,tao->solution,tao->gradient,tao->XU,PETSC_TRUE,&gpcg->Free_Local);CHKERRQ(ierr);
    } else {
      actred = 0; gpcg->step=1.0;
      /* if there were no free variables, no cg method */
    }

    tao->niter++;
    gpcg->f=f;gpcg->gnorm=gnorm; gpcg->actred=actred;
    ierr = TaoLogConvergenceHistory(tao,f,gpcg->gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,f,gpcg->gnorm,0.0,tao->step);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
    if (tao->reason != TAO_CONTINUE_ITERATING) break;
  }  /* END MAIN LOOP  */

  PetscFunctionReturn(0);
}

static PetscErrorCode GPCGGradProjections(Tao tao)
{
  PetscErrorCode                 ierr;
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
    ierr = VecBoundGradientProjection(G,X,XL,XU,DX);CHKERRQ(ierr);
    ierr = VecScale(DX,-1.0);CHKERRQ(ierr);
    ierr = VecDot(DX,G,&gdx);CHKERRQ(ierr);

    ierr = MatMult(tao->hessian,DX,Work);CHKERRQ(ierr);
    ierr = VecDot(DX,Work,&gAg);CHKERRQ(ierr);

    gpcg->gp_iterates++;
    gpcg->total_gp_its++;

    gtg=-gdx;
    if (PetscAbsReal(gAg) == 0.0) {
      alpha = 1.0;
    } else {
      alpha = PetscAbsReal(gtg/gAg);
    }
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,alpha);CHKERRQ(ierr);
    f_new=gpcg->f;
    ierr = TaoLineSearchApply(tao->linesearch,X,&f_new,G,DX,&stepsize,&lsflag);CHKERRQ(ierr);

    /* Update the iterate */
    actred = f_new - gpcg->f;
    actred_max = PetscMax(actred_max,-(f_new - gpcg->f));
    gpcg->f = f_new;
    ierr = ISDestroy(&gpcg->Free_Local);CHKERRQ(ierr);
    ierr = VecWhichInactive(XL,X,tao->gradient,XU,PETSC_TRUE,&gpcg->Free_Local);CHKERRQ(ierr);
  }

  gpcg->gnorm=gtg;
  PetscFunctionReturn(0);
} /* End gradient projections */

static PetscErrorCode TaoComputeDual_GPCG(Tao tao, Vec DXL, Vec DXU)
{
  TAO_GPCG       *gpcg = (TAO_GPCG *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecBoundGradientProjection(tao->gradient, tao->solution, tao->XL, tao->XU, gpcg->Work);CHKERRQ(ierr);
  ierr = VecCopy(gpcg->Work, DXL);CHKERRQ(ierr);
  ierr = VecAXPY(DXL,-1.0,tao->gradient);CHKERRQ(ierr);
  ierr = VecSet(DXU,0.0);CHKERRQ(ierr);
  ierr = VecPointwiseMax(DXL,DXL,DXU);CHKERRQ(ierr);

  ierr = VecCopy(tao->gradient,DXU);CHKERRQ(ierr);
  ierr = VecAXPY(DXU,-1.0,gpcg->Work);CHKERRQ(ierr);
  ierr = VecSet(gpcg->Work,0.0);CHKERRQ(ierr);
  ierr = VecPointwiseMin(DXU,gpcg->Work,DXU);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_GPCG;
  tao->ops->solve = TaoSolve_GPCG;
  tao->ops->view  = TaoView_GPCG;
  tao->ops->setfromoptions = TaoSetFromOptions_GPCG;
  tao->ops->destroy = TaoDestroy_GPCG;
  tao->ops->computedual = TaoComputeDual_GPCG;

  ierr = PetscNewLog(tao,&gpcg);CHKERRQ(ierr);
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

  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix);CHKERRQ(ierr);
  ierr = KSPSetType(tao->ksp,KSPNASH);CHKERRQ(ierr);

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, TAOLINESEARCHGPCG);CHKERRQ(ierr);
  ierr = TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch, GPCGObjectiveAndGradient, tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
