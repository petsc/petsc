#include <../src/snes/impls/ngmres/snesngmres.h> /*I "petscsnes.h" I*/

static PetscErrorCode SNESSetFromOptions_Anderson(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscBool      monitor = PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"SNES NGMRES options");
  PetscCall(PetscOptionsInt("-snes_anderson_m",            "Number of directions","SNES",ngmres->msize,&ngmres->msize,NULL));
  PetscCall(PetscOptionsReal("-snes_anderson_beta",        "Mixing parameter","SNES",ngmres->andersonBeta,&ngmres->andersonBeta,NULL));
  PetscCall(PetscOptionsInt("-snes_anderson_restart",      "Iterations before forced restart", "SNES",ngmres->restart_periodic,&ngmres->restart_periodic,NULL));
  PetscCall(PetscOptionsInt("-snes_anderson_restart_it",   "Tolerance iterations before restart","SNES",ngmres->restart_it,&ngmres->restart_it,NULL));
  PetscCall(PetscOptionsEnum("-snes_anderson_restart_type","Restart type","SNESNGMRESSetRestartType",SNESNGMRESRestartTypes,(PetscEnum)ngmres->restart_type,(PetscEnum*)&ngmres->restart_type,NULL));
  PetscCall(PetscOptionsBool("-snes_anderson_monitor",     "Monitor steps of Anderson Mixing","SNES",ngmres->monitor ? PETSC_TRUE : PETSC_FALSE,&monitor,NULL));
  if (monitor) {
    ngmres->monitor = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_Anderson(SNES snes)
{
  SNES_NGMRES         *ngmres = (SNES_NGMRES*) snes->data;
  /* present solution, residual, and preconditioned residual */
  Vec                 X,F,B,D;
  /* candidate linear combination answers */
  Vec                 XA,FA,XM,FM;

  /* coefficients and RHS to the minimization problem */
  PetscReal           fnorm,fMnorm,fAnorm;
  PetscReal           xnorm,ynorm;
  PetscReal           dnorm,dminnorm=0.0,fminnorm;
  PetscInt            restart_count=0;
  PetscInt            k,k_restart,l,ivec;
  PetscBool           selectRestart;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  PetscCheck(!snes->xl && !snes->xu && !snes->ops->computevariablebounds,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  PetscCall(PetscCitationsRegister(SNESCitation,&SNEScite));
  /* variable initialization */
  snes->reason = SNES_CONVERGED_ITERATING;
  X            = snes->vec_sol;
  F            = snes->vec_func;
  B            = snes->vec_rhs;
  XA           = snes->vec_sol_update;
  FA           = snes->work[0];
  D            = snes->work[1];

  /* work for the line search */
  XM = snes->work[3];
  FM = snes->work[4];

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));

  /* initialization */

  /* r = F(x) */

  if (snes->npc && snes->npcside== PC_LEFT) {
    PetscCall(SNESApplyNPC(snes,X,NULL,F));
    PetscCall(SNESGetConvergedReason(snes->npc,&reason));
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    PetscCall(VecNorm(F,NORM_2,&fnorm));
  } else {
    if (!snes->vec_func_init_set) {
      PetscCall(SNESComputeFunction(snes,X,F));
    } else snes->vec_func_init_set = PETSC_FALSE;

    PetscCall(VecNorm(F,NORM_2,&fnorm));
    SNESCheckFunctionNorm(snes,fnorm);
  }
  fminnorm = fnorm;

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESLogConvergenceHistory(snes,fnorm,0));
  PetscCall(SNESMonitor(snes,0,fnorm));
  PetscCall((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
  if (snes->reason) PetscFunctionReturn(0);

  k_restart = 0;
  l         = 0;
  ivec      = 0;
  for (k=1; k < snes->max_its+1; k++) {
    /* select which vector of the stored subspace will be updated */
    if (snes->npc && snes->npcside== PC_RIGHT) {
      PetscCall(VecCopy(X,XM));
      PetscCall(SNESSetInitialFunction(snes->npc,F));

      PetscCall(PetscLogEventBegin(SNES_NPCSolve,snes->npc,XM,B,0));
      PetscCall(SNESSolve(snes->npc,B,XM));
      PetscCall(PetscLogEventEnd(SNES_NPCSolve,snes->npc,XM,B,0));

      PetscCall(SNESGetConvergedReason(snes->npc,&reason));
      if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      PetscCall(SNESGetNPCFunction(snes,FM,&fMnorm));
      if (ngmres->andersonBeta != 1.0) {
        PetscCall(VecAXPBY(XM,(1.0 - ngmres->andersonBeta),ngmres->andersonBeta,X));
      }
    } else {
      PetscCall(VecCopy(F,FM));
      PetscCall(VecCopy(X,XM));
      PetscCall(VecAXPY(XM,-ngmres->andersonBeta,FM));
      fMnorm = fnorm;
    }

    PetscCall(SNESNGMRESFormCombinedSolution_Private(snes,ivec,l,XM,FM,fMnorm,X,XA,FA));
    ivec = k_restart % ngmres->msize;
    if (ngmres->restart_type == SNES_NGMRES_RESTART_DIFFERENCE) {
      PetscCall(SNESNGMRESNorms_Private(snes,l,X,F,XM,FM,XA,FA,D,&dnorm,&dminnorm,NULL,NULL,NULL,&xnorm,&fAnorm,&ynorm));
      PetscCall(SNESNGMRESSelectRestart_Private(snes,l,fMnorm,fnorm,dnorm,fminnorm,dminnorm,&selectRestart));
      /* if the restart conditions persist for more than restart_it iterations, restart. */
      if (selectRestart) restart_count++;
      else restart_count = 0;
    } else if (ngmres->restart_type == SNES_NGMRES_RESTART_PERIODIC) {
      PetscCall(SNESNGMRESNorms_Private(snes,l,X,F,XM,FM,XA,FA,D,NULL,NULL,NULL,NULL,NULL,&xnorm,&fAnorm,&ynorm));
      if (k_restart > ngmres->restart_periodic) {
        if (ngmres->monitor) PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"periodic restart after %" PetscInt_FMT " iterations\n",k_restart));
        restart_count = ngmres->restart_it;
      }
    } else {
      PetscCall(SNESNGMRESNorms_Private(snes,l,X,F,XM,FM,XA,FA,D,NULL,NULL,NULL,NULL,NULL,&xnorm,&fAnorm,&ynorm));
    }
    /* restart after restart conditions have persisted for a fixed number of iterations */
    if (restart_count >= ngmres->restart_it) {
      if (ngmres->monitor) {
        PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"Restarted at iteration %" PetscInt_FMT "\n",k_restart));
      }
      restart_count = 0;
      k_restart     = 0;
      l             = 0;
      ivec          = 0;
    } else {
      if (l < ngmres->msize) l++;
      k_restart++;
      PetscCall(SNESNGMRESUpdateSubspace_Private(snes,ivec,l,FM,fnorm,XM));
    }

    fnorm = fAnorm;
    if (fminnorm > fnorm) fminnorm = fnorm;

    PetscCall(VecCopy(XA,X));
    PetscCall(VecCopy(FA,F));

    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = k;
    snes->norm = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = ynorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,snes->iter));
    PetscCall(SNESMonitor(snes,snes->iter,snes->norm));
    PetscCall((*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) PetscFunctionReturn(0);
  }
  snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(0);
}

/*MC
  SNESANDERSON - Anderson Mixing method.

   Level: beginner

   Options Database:
+  -snes_anderson_m                - Number of stored previous solutions and residuals
.  -snes_anderson_beta             - Anderson mixing parameter
.  -snes_anderson_restart_type     - Type of restart (see SNESNGMRES)
.  -snes_anderson_restart_it       - Number of iterations of restart conditions before restart
.  -snes_anderson_restart          - Number of iterations before periodic restart
-  -snes_anderson_monitor          - Prints relevant information about the ngmres iteration

   Notes:

   The Anderson Mixing method combines m previous solutions into a minimum-residual solution by solving a small linearized
   optimization problem at each iteration.

   Very similar to the SNESNGMRES algorithm.

   References:
+  * -  D. G. Anderson. Iterative procedures for nonlinear integral equations.
    J. Assoc. Comput. Mach., 12, 1965."
-  * - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu,"Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

.seealso: `SNESNGMRES`, `SNESCreate()`, `SNES`, `SNESSetType()`, `SNESType`
M*/

PETSC_EXTERN PetscErrorCode SNESCreate_Anderson(SNES snes)
{
  SNES_NGMRES    *ngmres;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_NGMRES;
  snes->ops->setup          = SNESSetUp_NGMRES;
  snes->ops->setfromoptions = SNESSetFromOptions_Anderson;
  snes->ops->view           = SNESView_NGMRES;
  snes->ops->solve          = SNESSolve_Anderson;
  snes->ops->reset          = SNESReset_NGMRES;

  snes->usesnpc = PETSC_TRUE;
  snes->usesksp = PETSC_FALSE;
  snes->npcside = PC_RIGHT;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  PetscCall(PetscNewLog(snes,&ngmres));
  snes->data    = (void*) ngmres;
  ngmres->msize = 30;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
  }

  PetscCall(SNESGetLineSearch(snes,&linesearch));
  if (!((PetscObject)linesearch)->type_name) {
    PetscCall(SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC));
  }

  ngmres->additive_linesearch = NULL;
  ngmres->approxfunc       = PETSC_FALSE;
  ngmres->restart_type     = SNES_NGMRES_RESTART_NONE;
  ngmres->restart_it       = 2;
  ngmres->restart_periodic = 30;
  ngmres->gammaA           = 2.0;
  ngmres->gammaC           = 2.0;
  ngmres->deltaB           = 0.9;
  ngmres->epsilonB         = 0.1;

  ngmres->andersonBeta = 1.0;
  PetscFunctionReturn(0);
}
