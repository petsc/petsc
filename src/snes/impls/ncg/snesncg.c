#include <../src/snes/impls/ncg/snesncgimpl.h> /*I "petscsnes.h" I*/
const char *const SNESNCGTypes[] = {"FR","PRP","HS","DY","CD","SNESNCGType","SNES_NCG_",NULL};

static PetscErrorCode SNESReset_NCG(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*
  SNESDestroy_NCG - Destroys the private SNES_NCG context that was created with SNESCreate_NCG().

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESDestroy()
*/
static PetscErrorCode SNESDestroy_NCG(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(0);
}

/*
   SNESSetUp_NCG - Sets up the internal data structures for the later use
   of the SNESNCG nonlinear solver.

   Input Parameters:
+  snes - the SNES context
-  x - the solution vector

   Application Interface Routine: SNESSetUp()
 */

static PetscErrorCode SNESSetUp_NCG(SNES snes)
{
  PetscFunctionBegin;
  PetscCall(SNESSetWorkVecs(snes,2));
  PetscCheck(snes->npcside!= PC_RIGHT,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNESNCG only supports left preconditioning");
  if (snes->functype == SNES_FUNCTION_DEFAULT) snes->functype = SNES_FUNCTION_UNPRECONDITIONED;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESLineSearchApply_NCGLinear(SNESLineSearch linesearch)
{
  PetscScalar    alpha, ptAp;
  Vec            X, Y, F, W;
  SNES           snes;
  PetscReal      *fnorm, *xnorm, *ynorm;

  PetscFunctionBegin;
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  X     = linesearch->vec_sol;
  W     = linesearch->vec_sol_new;
  F     = linesearch->vec_func;
  Y     = linesearch->vec_update;
  fnorm = &linesearch->fnorm;
  xnorm = &linesearch->xnorm;
  ynorm = &linesearch->ynorm;

  if (!snes->jacobian) {
    PetscCall(SNESSetUpMatrices(snes));
  }

  /*

   The exact step size for unpreconditioned linear CG is just:
   alpha = (r, r) / (p, Ap) = (f, f) / (y, Jy)
   */
  PetscCall(SNESComputeJacobian(snes, X, snes->jacobian, snes->jacobian_pre));
  PetscCall(VecDot(F, F, &alpha));
  PetscCall(MatMult(snes->jacobian, Y, W));
  PetscCall(VecDot(Y, W, &ptAp));
  alpha = alpha / ptAp;
  PetscCall(VecAXPY(X,-alpha,Y));
  PetscCall(SNESComputeFunction(snes,X,F));

  PetscCall(VecNorm(F,NORM_2,fnorm));
  PetscCall(VecNorm(X,NORM_2,xnorm));
  PetscCall(VecNorm(Y,NORM_2,ynorm));
  PetscFunctionReturn(0);
}

/*MC
   SNESLINESEARCHNCGLINEAR - Special line search only for SNESNCG

   This line search uses the length "as if" the problem is linear (that is what is computed by the linear CG method) using the Jacobian of the function.
   alpha = (r, r) / (p, Ap) = (f, f) / (y, Jy) where r (f) is the current residual (function value), p (y) is the current search direction.

   Notes: This requires a Jacobian-vector product but does not require the solution of a linear system with the Jacobian

   This is a "odd-ball" line search, we don't know if it is in the literature or used in practice by anyone.

   Level: advanced

.seealso: SNESLineSearchCreate(), SNESLineSearchSetType()
M*/

PETSC_EXTERN PetscErrorCode SNESLineSearchCreate_NCGLinear(SNESLineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = SNESLineSearchApply_NCGLinear;
  linesearch->ops->destroy        = NULL;
  linesearch->ops->setfromoptions = NULL;
  linesearch->ops->reset          = NULL;
  linesearch->ops->view           = NULL;
  linesearch->ops->setup          = NULL;
  PetscFunctionReturn(0);
}

/*
  SNESSetFromOptions_NCG - Sets various parameters for the SNESNCG method.

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESSetFromOptions()
*/
static PetscErrorCode SNESSetFromOptions_NCG(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_NCG       *ncg = (SNES_NCG*)snes->data;
  PetscBool      debug = PETSC_FALSE;
  SNESNCGType    ncgtype=ncg->type;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"SNES NCG options");
  PetscCall(PetscOptionsBool("-snes_ncg_monitor","Monitor the beta values used in the NCG iterations","SNES",ncg->monitor ? PETSC_TRUE : PETSC_FALSE, &debug, NULL));
  if (debug) {
    ncg->monitor = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));
  }
  PetscCall(PetscOptionsEnum("-snes_ncg_type","NCG Beta type used","SNESNCGSetType",SNESNCGTypes,(PetscEnum)ncg->type,(PetscEnum*)&ncgtype,NULL));
  PetscCall(SNESNCGSetType(snes, ncgtype));
  PetscOptionsHeadEnd();
  if (!snes->linesearch) {
    PetscCall(SNESGetLineSearch(snes, &linesearch));
    if (!((PetscObject)linesearch)->type_name) {
      if (!snes->npc) {
        PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHCP));
      } else {
        PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHL2));
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  SNESView_NCG - Prints info from the SNESNCG data structure.

  Input Parameters:
+ SNES - the SNES context
- viewer - visualization context

  Application Interface Routine: SNESView()
*/
static PetscErrorCode SNESView_NCG(SNES snes, PetscViewer viewer)
{
  SNES_NCG      *ncg = (SNES_NCG *) snes->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  type: %s\n", SNESNCGTypes[ncg->type]));
  }
  PetscFunctionReturn(0);
}

/*

 Assuming F = SNESComputeFunction(X) compute Y^tJ^tF using a simple secant approximation of the jacobian.

 This routine is not currently used. I don't know what its intended purpose is.

 Note it has a hardwired differencing parameter of 1e-5

 */
PetscErrorCode SNESNCGComputeYtJtF_Private(SNES snes, Vec X, Vec F, Vec Y, Vec W, Vec G, PetscScalar * ytJtf)
{
  PetscScalar    ftf, ftg, fty, h;

  PetscFunctionBegin;
  PetscCall(VecDot(F, F, &ftf));
  PetscCall(VecDot(F, Y, &fty));
  h      = 1e-5*fty / fty;
  PetscCall(VecCopy(X, W));
  PetscCall(VecAXPY(W, -h, Y));          /* this is arbitrary */
  PetscCall(SNESComputeFunction(snes, W, G));
  PetscCall(VecDot(G, F, &ftg));
  *ytJtf = (ftg - ftf) / h;
  PetscFunctionReturn(0);
}

/*@
    SNESNCGSetType - Sets the conjugate update type for SNESNCG.

    Logically Collective on SNES

    Input Parameters:
+   snes - the iterative context
-   btype - update type

    Options Database:
.   -snes_ncg_type <prp,fr,hs,dy,cd> -strategy for selecting algorithm for computing beta

    Level: intermediate

    SNESNCGSelectTypes:
+   SNES_NCG_FR - Fletcher-Reeves update
.   SNES_NCG_PRP - Polak-Ribiere-Polyak update
.   SNES_NCG_HS - Hestenes-Steifel update
.   SNES_NCG_DY - Dai-Yuan update
-   SNES_NCG_CD - Conjugate Descent update

   Notes:
   SNES_NCG_PRP is the default, and the only one that tolerates generalized search directions.

   It is not clear what "generalized search directions" means, does it mean use with a nonlinear preconditioner,
   that is using -npc_snes_type <type>, SNESSetNPC(), or SNESGetNPC()?

@*/
PetscErrorCode SNESNCGSetType(SNES snes, SNESNCGType btype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscTryMethod(snes,"SNESNCGSetType_C",(SNES,SNESNCGType),(snes,btype));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESNCGSetType_NCG(SNES snes, SNESNCGType btype)
{
  SNES_NCG *ncg = (SNES_NCG*)snes->data;

  PetscFunctionBegin;
  ncg->type = btype;
  PetscFunctionReturn(0);
}

/*
  SNESSolve_NCG - Solves a nonlinear system with the Nonlinear Conjugate Gradient method.

  Input Parameters:
. snes - the SNES context

  Output Parameter:
. outits - number of iterations until termination

  Application Interface Routine: SNESSolve()
*/
static PetscErrorCode SNESSolve_NCG(SNES snes)
{
  SNES_NCG             *ncg = (SNES_NCG*)snes->data;
  Vec                  X,dX,lX,F,dXold;
  PetscReal            fnorm, ynorm, xnorm, beta = 0.0;
  PetscScalar          dXdotdX, dXolddotdXold, dXdotdXold, lXdotdX, lXdotdXold;
  PetscInt             maxits, i;
  SNESLineSearchReason lsresult = SNES_LINESEARCH_SUCCEEDED;
  SNESLineSearch       linesearch;
  SNESConvergedReason  reason;

  PetscFunctionBegin;
  PetscCheckFalse(snes->xl || snes->xu || snes->ops->computevariablebounds,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  PetscCall(PetscCitationsRegister(SNESCitation,&SNEScite));
  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;            /* maximum number of iterations */
  X      = snes->vec_sol;            /* X^n */
  dXold  = snes->work[0];            /* The previous iterate of X */
  dX     = snes->work[1];            /* the preconditioned direction */
  lX     = snes->vec_sol_update;     /* the conjugate direction */
  F      = snes->vec_func;           /* residual vector */

  PetscCall(SNESGetLineSearch(snes, &linesearch));

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter = 0;
  snes->norm = 0.;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));

  /* compute the initial function and preconditioned update dX */

  if (snes->npc && snes->functype == SNES_FUNCTION_PRECONDITIONED) {
    PetscCall(SNESApplyNPC(snes,X,NULL,dX));
    PetscCall(SNESGetConvergedReason(snes->npc,&reason));
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    PetscCall(VecCopy(dX,F));
    PetscCall(VecNorm(F,NORM_2,&fnorm));
  } else {
    if (!snes->vec_func_init_set) {
      PetscCall(SNESComputeFunction(snes,X,F));
    } else snes->vec_func_init_set = PETSC_FALSE;

    /* convergence test */
    PetscCall(VecNorm(F,NORM_2,&fnorm));
    SNESCheckFunctionNorm(snes,fnorm);
    PetscCall(VecCopy(F,dX));
  }
  if (snes->npc) {
    if (snes->functype == SNES_FUNCTION_UNPRECONDITIONED) {
      PetscCall(SNESApplyNPC(snes,X,F,dX));
      PetscCall(SNESGetConvergedReason(snes->npc,&reason));
      if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
    }
  }
  PetscCall(VecCopy(dX,lX));
  PetscCall(VecDot(dX, dX, &dXdotdX));

  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->norm = fnorm;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
  PetscCall(SNESLogConvergenceHistory(snes,fnorm,0));
  PetscCall(SNESMonitor(snes,0,fnorm));

  /* test convergence */
  PetscCall((*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
  if (snes->reason) PetscFunctionReturn(0);

  /* Call general purpose update function */
  if (snes->ops->update) {
    PetscCall((*snes->ops->update)(snes, snes->iter));
  }

  /* first update -- just use the (preconditioned) residual direction for the initial conjugate direction */

  for (i = 1; i < maxits + 1; i++) {
    /* some update types require the old update direction or conjugate direction */
    if (ncg->type != SNES_NCG_FR) {
      PetscCall(VecCopy(dX, dXold));
    }
    PetscCall(SNESLineSearchApply(linesearch,X,F,&fnorm,lX));
    PetscCall(SNESLineSearchGetReason(linesearch, &lsresult));
    PetscCall(SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &ynorm));
    if (lsresult) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscFunctionReturn(0);
      }
    }
    if (snes->nfuncs >= snes->max_funcs && snes->max_funcs >= 0) {
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      PetscFunctionReturn(0);
    }
    /* Monitor convergence */
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = i;
    snes->norm = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = ynorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes,snes->norm,0));
    PetscCall(SNESMonitor(snes,snes->iter,snes->norm));

    /* Test for convergence */
    PetscCall((*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP));
    if (snes->reason) PetscFunctionReturn(0);

    /* Call general purpose update function */
    if (snes->ops->update) {
      PetscCall((*snes->ops->update)(snes, snes->iter));
    }
    if (snes->npc) {
      if (snes->functype == SNES_FUNCTION_PRECONDITIONED) {
        PetscCall(SNESApplyNPC(snes,X,NULL,dX));
        PetscCall(SNESGetConvergedReason(snes->npc,&reason));
        if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
          snes->reason = SNES_DIVERGED_INNER;
          PetscFunctionReturn(0);
        }
        PetscCall(VecCopy(dX,F));
      } else {
        PetscCall(SNESApplyNPC(snes,X,F,dX));
        PetscCall(SNESGetConvergedReason(snes->npc,&reason));
        if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
          snes->reason = SNES_DIVERGED_INNER;
          PetscFunctionReturn(0);
        }
      }
    } else {
      PetscCall(VecCopy(F,dX));
    }

    /* compute the conjugate direction lX = dX + beta*lX with beta = ((dX, dX) / (dX_old, dX_old) (Fletcher-Reeves update)*/
    switch (ncg->type) {
    case SNES_NCG_FR: /* Fletcher-Reeves */
      dXolddotdXold= dXdotdX;
      PetscCall(VecDot(dX, dX, &dXdotdX));
      beta         = PetscRealPart(dXdotdX / dXolddotdXold);
      break;
    case SNES_NCG_PRP: /* Polak-Ribiere-Poylak */
      dXolddotdXold= dXdotdX;
      PetscCall(VecDotBegin(dX, dX, &dXdotdXold));
      PetscCall(VecDotBegin(dXold, dX, &dXdotdXold));
      PetscCall(VecDotEnd(dX, dX, &dXdotdX));
      PetscCall(VecDotEnd(dXold, dX, &dXdotdXold));
      beta         = PetscRealPart(((dXdotdX - dXdotdXold) / dXolddotdXold));
      if (beta < 0.0) beta = 0.0; /* restart */
      break;
    case SNES_NCG_HS: /* Hestenes-Stiefel */
      PetscCall(VecDotBegin(dX, dX, &dXdotdX));
      PetscCall(VecDotBegin(dX, dXold, &dXdotdXold));
      PetscCall(VecDotBegin(lX, dX, &lXdotdX));
      PetscCall(VecDotBegin(lX, dXold, &lXdotdXold));
      PetscCall(VecDotEnd(dX, dX, &dXdotdX));
      PetscCall(VecDotEnd(dX, dXold, &dXdotdXold));
      PetscCall(VecDotEnd(lX, dX, &lXdotdX));
      PetscCall(VecDotEnd(lX, dXold, &lXdotdXold));
      beta = PetscRealPart((dXdotdX - dXdotdXold) / (lXdotdX - lXdotdXold));
      break;
    case SNES_NCG_DY: /* Dai-Yuan */
      PetscCall(VecDotBegin(dX, dX, &dXdotdX));
      PetscCall(VecDotBegin(lX, dX, &lXdotdX));
      PetscCall(VecDotBegin(lX, dXold, &lXdotdXold));
      PetscCall(VecDotEnd(dX, dX, &dXdotdX));
      PetscCall(VecDotEnd(lX, dX, &lXdotdX));
      PetscCall(VecDotEnd(lX, dXold, &lXdotdXold));
      beta = PetscRealPart(dXdotdX / (lXdotdXold - lXdotdX));
      break;
    case SNES_NCG_CD: /* Conjugate Descent */
      PetscCall(VecDotBegin(dX, dX, &dXdotdX));
      PetscCall(VecDotBegin(lX, dXold, &lXdotdXold));
      PetscCall(VecDotEnd(dX, dX, &dXdotdX));
      PetscCall(VecDotEnd(lX, dXold, &lXdotdXold));
      beta = PetscRealPart(dXdotdX / lXdotdXold);
      break;
    }
    if (ncg->monitor) {
      PetscCall(PetscViewerASCIIPrintf(ncg->monitor, "beta = %e\n", (double)beta));
    }
    PetscCall(VecAYPX(lX, beta, dX));
  }
  PetscCall(PetscInfo(snes, "Maximum number of iterations has been reached: %D\n", maxits));
  if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(0);
}

/*MC
  SNESNCG - Nonlinear Conjugate-Gradient method for the solution of general nonlinear systems.

  Level: beginner

  Options Database:
+   -snes_ncg_type <fr, prp, dy, hs, cd> - Choice of conjugate-gradient update parameter, default is prp.
.   -snes_linesearch_type <cp,l2,basic> - Line search type.
-   -snes_ncg_monitor - Print the beta values used in the ncg iteration, .

   Notes:
    This solves the nonlinear system of equations F(x) = 0 using the nonlinear generalization of the conjugate
          gradient method.  This may be used with a nonlinear preconditioner used to pick the new search directions, but otherwise
          chooses the initial search direction as F(x) for the initial guess x.

          Only supports left non-linear preconditioning.

    Default line search is SNESLINESEARCHCP, unless a nonlinear preconditioner is used with -npc_snes_type <type>, SNESSetNPC(), or SNESGetNPC() then
    SNESLINESEARCHL2 is used. Also supports the special purpose line search SNESLINESEARCHNCGLINEAR

   References:
.  * -  Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu,"Composing Scalable Nonlinear Algebraic Solvers",
   SIAM Review, 57(4), 2015

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESNEWTONTR, SNESNGMRES, SNESQN, SNESLINESEARCHNCGLINEAR, SNESNCGSetType(), SNESNCGGetType(), SNESLineSearchSetType()
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_NCG(SNES snes)
{
  SNES_NCG       * neP;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_NCG;
  snes->ops->setup          = SNESSetUp_NCG;
  snes->ops->setfromoptions = SNESSetFromOptions_NCG;
  snes->ops->view           = SNESView_NCG;
  snes->ops->solve          = SNESSolve_NCG;
  snes->ops->reset          = SNESReset_NCG;

  snes->usesksp = PETSC_FALSE;
  snes->usesnpc = PETSC_TRUE;
  snes->npcside = PC_LEFT;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
    snes->stol      = 1e-20;
  }

  PetscCall(PetscNewLog(snes,&neP));
  snes->data   = (void*) neP;
  neP->monitor = NULL;
  neP->type    = SNES_NCG_PRP;
  PetscCall(PetscObjectComposeFunction((PetscObject)snes,"SNESNCGSetType_C", SNESNCGSetType_NCG));
  PetscFunctionReturn(0);
}
