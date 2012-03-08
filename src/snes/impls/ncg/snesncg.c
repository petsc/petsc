#include <../src/snes/impls/ncg/snesncgimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SNESReset_NCG"
PetscErrorCode SNESReset_NCG(SNES snes)
{
  PetscErrorCode ierr;
  SNES_NCG       *ncg = (SNES_NCG *)snes->data;

  PetscFunctionBegin;
  ierr = LineSearchDestroy(&ncg->linesearch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define LINESEARCHNCGLINEAR "linear"

/*
  SNESDestroy_NCG - Destroys the private SNES_NCG context that was created with SNESCreate_NCG().

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NCG"
PetscErrorCode SNESDestroy_NCG(SNES snes)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
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

EXTERN_C_BEGIN;
extern PetscErrorCode LineSearchCreate_NCGLinear(LineSearch);
EXTERN_C_END;

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NCG"
PetscErrorCode SNESSetUp_NCG(SNES snes)
{
  PetscErrorCode ierr;
  SNES_NCG       *ncg = (SNES_NCG *)snes->data;
  const char     *optionsprefix;

  PetscFunctionBegin;
  ierr = SNESDefaultGetWork(snes,2);CHKERRQ(ierr);
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  ierr = LineSearchRegisterDynamic(LINESEARCHNCGLINEAR, PETSC_NULL,"LineSearchCreate_NCGLinear", LineSearchCreate_NCGLinear);CHKERRQ(ierr);
  ierr = SNESGetOptionsPrefix(snes, &optionsprefix);CHKERRQ(ierr);
  ierr = LineSearchCreate(((PetscObject)snes)->comm, &ncg->linesearch);CHKERRQ(ierr);
  ierr = LineSearchSetSNES(ncg->linesearch, snes);CHKERRQ(ierr);
  if (!snes->pc) {
    ierr = LineSearchSetType(ncg->linesearch, LINESEARCHCP);CHKERRQ(ierr);
  } else {
    ierr = LineSearchSetType(ncg->linesearch, LINESEARCHL2);CHKERRQ(ierr);
  }
  ierr = LineSearchAppendOptionsPrefix(ncg->linesearch, optionsprefix);CHKERRQ(ierr);
  ierr = LineSearchSetFromOptions(ncg->linesearch);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/*
  SNESSetFromOptions_NCG - Sets various parameters for the SNESNCG method.

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NCG"
static PetscErrorCode SNESSetFromOptions_NCG(SNES snes)
{
  SNES_NCG           *ncg     = (SNES_NCG *)snes->data;
  const char         *betas[] = {"FR","PRP","HS", "DY", "CD"};
  PetscErrorCode     ierr;
  PetscBool          debug, flg;
  PetscInt           indx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES NCG options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ncg_monitor", "Monitor NCG iterations", "SNES", ncg->monitor ? PETSC_TRUE: PETSC_FALSE, &debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-snes_ncg_type","Nonlinear CG update used","", betas, 5, "FR",&indx, &flg);CHKERRQ(ierr);
  if (flg) {
    ncg->betatype = indx;
  }
  if (debug) {
    ncg->monitor = PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  SNESView_NCG - Prints info from the SNESNCG data structure.

  Input Parameters:
+ SNES - the SNES context
- viewer - visualization context

  Application Interface Routine: SNESView()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESView_NCG"
static PetscErrorCode SNESView_NCG(SNES snes, PetscViewer viewer)
{
  PetscBool        iascii;
  PetscErrorCode   ierr;
  SNES_NCG         *ncg = (SNES_NCG *)snes->data;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  line search type variant: %s\n", ((PetscObject)ncg->linesearch)->type_name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LineSearchApply_NCGLinear"
PetscErrorCode LineSearchApply_NCGLinear(LineSearch linesearch)
{
  PetscScalar      alpha, ptAp;
  Vec              X, Y, F, W;
  SNES             snes;
  PetscErrorCode   ierr;
  PetscReal        *fnorm, *xnorm, *ynorm;
  MatStructure     flg = DIFFERENT_NONZERO_PATTERN;

  PetscFunctionBegin;

  ierr = LineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  X = linesearch->vec_sol;
  W = linesearch->vec_sol_new;
  F = linesearch->vec_func;
  Y = linesearch->vec_update;
  fnorm = &linesearch->fnorm;
  xnorm = &linesearch->xnorm;
  ynorm = &linesearch->ynorm;

  /*

   The exact step size for unpreconditioned linear CG is just:
   alpha = (r, r) / (p, Ap) = (f, f) / (y, Jy)
   */
  ierr = SNESComputeJacobian(snes, X, &snes->jacobian, &snes->jacobian_pre, &flg);CHKERRQ(ierr);
  ierr = VecDot(F, F, &alpha);CHKERRQ(ierr);
  ierr = MatMult(snes->jacobian, Y, W);CHKERRQ(ierr);
  ierr = VecDot(Y, W, &ptAp);CHKERRQ(ierr);
  alpha = alpha / ptAp;
  ierr = PetscPrintf(((PetscObject)snes)->comm, "alpha: %G\n", PetscRealPart(alpha));CHKERRQ(ierr);
  ierr = VecAXPY(X, alpha, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);

  ierr = VecNorm(F, NORM_2, fnorm);CHKERRQ(ierr);
  ierr = VecNorm(X, NORM_2, xnorm);CHKERRQ(ierr);
  ierr = VecNorm(Y, NORM_2, ynorm);CHKERRQ(ierr);

 PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "LineSearchCreate_NCGLinear"

PetscErrorCode LineSearchCreate_NCGLinear(LineSearch linesearch)
{
  PetscFunctionBegin;
  linesearch->ops->apply          = LineSearchApply_NCGLinear;
  linesearch->ops->destroy        = PETSC_NULL;
  linesearch->ops->setfromoptions = PETSC_NULL;
  linesearch->ops->reset          = PETSC_NULL;
  linesearch->ops->view           = PETSC_NULL;
  linesearch->ops->setup          = PETSC_NULL;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "ComputeYtJtF_Private"
/*

 Assuming F = SNESComputeFunction(X) compute Y^tJ^tF using a simple secant approximation of the jacobian.

 */
PetscErrorCode ComputeYtJtF_Private(SNES snes, Vec X, Vec F, Vec Y, Vec W, Vec G, PetscScalar * ytJtf) {
  PetscErrorCode ierr;
  PetscScalar    ftf, ftg, fty, h;
  PetscFunctionBegin;
  ierr = VecDot(F, F, &ftf);CHKERRQ(ierr);
  ierr = VecDot(F, Y, &fty);CHKERRQ(ierr);
  h = 1e-5*fty / fty;
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, -h, Y);CHKERRQ(ierr);            /* this is arbitrary */
  ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
  ierr = VecDot(G, F, &ftg);CHKERRQ(ierr);
  *ytJtf = (ftg - ftf) / h;
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
#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NCG"
PetscErrorCode SNESSolve_NCG(SNES snes)
{
  SNES_NCG            *ncg = (SNES_NCG *)snes->data;
  Vec                 X, dX, lX, F, B, Fold;
  PetscReal           fnorm, ynorm, xnorm, beta = 0.0;
  PetscScalar         dXdotF, dXolddotFold, dXdotFold, lXdotF, lXdotFold;
  PetscInt            maxits, i;
  PetscErrorCode      ierr;
  SNESConvergedReason reason;
  PetscBool           lsSuccess = PETSC_TRUE;

  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;            /* maximum number of iterations */
  X      = snes->vec_sol;            /* X^n */
  Fold   = snes->work[0];            /* The previous iterate of X */
  dX     = snes->work[1];            /* the preconditioned direction */
  lX     = snes->vec_sol_update;     /* the conjugate direction */
  F      = snes->vec_func;           /* residual vector */
  B      = snes->vec_rhs;            /* the right hand side */

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  /* compute the initial function and preconditioned update dX */
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  /* convergence test */
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
 }

  /* first update -- just use the (preconditioned) residual direction for the initial conjugate direction */

  if (snes->pc) {
    ierr = VecCopy(X, dX);CHKERRQ(ierr);
    ierr = SNESSolve(snes->pc, B, dX);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
    if (reason < 0 && (reason != SNES_DIVERGED_MAX_IT)) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    ierr = VecAYPX(dX,-1.0,X);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(F, dX);CHKERRQ(ierr);
  }
  ierr = VecCopy(dX, lX);CHKERRQ(ierr);
  ierr = VecDot(F, dX, &dXdotF);CHKERRQ(ierr);
  /*
  } else {
    ierr = ComputeYtJtF_Private(snes, X, F, dX, W, G, &dXdotF);CHKERRQ(ierr);
  }
   */
  for(i = 1; i < maxits + 1; i++) {
    lsSuccess = PETSC_TRUE;
    /* some update types require the old update direction or conjugate direction */
    if (ncg->betatype == 1 || ncg->betatype == 2 || ncg->betatype == 3 || ncg->betatype == 4) { /* prp, hs, dy, cd need dXold*/
      ierr = VecCopy(F, Fold);CHKERRQ(ierr);
    }
    ierr = LineSearchApply(ncg->linesearch, X, F, &fnorm, lX);CHKERRQ(ierr);
    ierr = LineSearchGetSuccess(ncg->linesearch, &lsSuccess);CHKERRQ(ierr);
    if (!lsSuccess) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscFunctionReturn(0);
      }
    }
    if (snes->nfuncs >= snes->max_funcs) {
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      PetscFunctionReturn(0);
    }
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
    ierr = LineSearchGetNorms(ncg->linesearch, &xnorm, &fnorm, &ynorm);CHKERRQ(ierr);
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);

    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    if (snes->pc) {
      ierr = VecCopy(X,dX);CHKERRQ(ierr);
      ierr = SNESSolve(snes->pc, B, dX);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
      if (reason < 0 && (reason != SNES_DIVERGED_MAX_IT)) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = VecAYPX(dX,-1.0,X);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(F, dX);CHKERRQ(ierr);
    }

    /* compute the conjugate direction lX = dX + beta*lX with beta = ((dX, dX) / (dX_old, dX_old) (Fletcher-Reeves update)*/
    switch(ncg->betatype) {
    case 0: /* Fletcher-Reeves */
      dXolddotFold = dXdotF;
      ierr = VecDot(dX, dX, &dXdotF);CHKERRQ(ierr);
      beta = PetscRealPart(dXdotF / dXolddotFold);
      break;
    case 1: /* Polak-Ribiere-Poylak */
      dXolddotFold = dXdotF;
      ierr = VecDot(F, dX, &dXdotF);CHKERRQ(ierr);
      ierr = VecDot(Fold, dX, &dXdotFold);CHKERRQ(ierr);
      beta = PetscRealPart(((dXdotF - dXdotFold) / dXolddotFold));
      if (beta < 0.0) beta = 0.0; /* restart */
      break;
    case 2: /* Hestenes-Stiefel */
      ierr = VecDot(dX, F, &dXdotF);CHKERRQ(ierr);
      ierr = VecDot(dX, Fold, &dXdotFold);CHKERRQ(ierr);
      ierr = VecDot(lX, F, &lXdotF);CHKERRQ(ierr);
      ierr = VecDot(lX, Fold, &lXdotFold);CHKERRQ(ierr);
      beta = PetscRealPart((dXdotF - dXdotFold) / (lXdotF - lXdotFold));
      break;
    case 3: /* Dai-Yuan */
      ierr = VecDot(dX, F, &dXdotF);CHKERRQ(ierr);
      ierr = VecDot(lX, F, &lXdotF);CHKERRQ(ierr);
      ierr = VecDot(lX, Fold, &lXdotFold);CHKERRQ(ierr);
      beta = PetscRealPart(dXdotF / (lXdotFold - lXdotF));CHKERRQ(ierr);
      break;
    case 4: /* Conjugate Descent */
      ierr = VecDot(dX, F, &dXdotF);CHKERRQ(ierr);
      ierr = VecDot(lX, Fold, &lXdotFold);CHKERRQ(ierr);
      beta = PetscRealPart(dXdotF / lXdotFold);CHKERRQ(ierr);
      break;
    }
    if (ncg->monitor) {
      ierr = PetscViewerASCIIPrintf(ncg->monitor, "beta = %e\n", beta);CHKERRQ(ierr);
    }
    ierr = VecAYPX(lX, beta, dX);CHKERRQ(ierr);
  }
  ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", maxits);CHKERRQ(ierr);
  if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(0);
}

/*MC
  SNESNCG - Nonlinear Conjugate-Gradient method for the solution of general nonlinear systems.

  Level: beginner

  Options Database:
+   -snes_ncg_type <fr, prp, dy, hs, cd> - Choice of conjugate-gradient update parameter.
.   -snes_ls <basic,basicnormnorms,quadratic,critical,test> - Line search type.
-   -snes_ncg_monitor - Print relevant information about the ncg iteration.

Notes: This solves the nonlinear system of equations F(x) = 0 using the nonlinear generalization of the conjugate
gradient method.  This may be used with a nonlinear preconditioner used to pick the new search directions, but otherwise
chooses the initial search direction as F(x) for the initial guess x.


.seealso:  SNESCreate(), SNES, SNESSetType(), SNESLS, SNESTR, SNESNGMRES, SNESNQN
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NCG"
PetscErrorCode  SNESCreate_NCG(SNES snes)
{
  PetscErrorCode   ierr;
  SNES_NCG * neP;

  PetscFunctionBegin;
  snes->ops->destroy         = SNESDestroy_NCG;
  snes->ops->setup           = SNESSetUp_NCG;
  snes->ops->setfromoptions  = SNESSetFromOptions_NCG;
  snes->ops->view            = SNESView_NCG;
  snes->ops->solve           = SNESSolve_NCG;
  snes->ops->reset           = SNESReset_NCG;

  snes->usesksp              = PETSC_FALSE;
  snes->usespc               = PETSC_TRUE;

  snes->max_funcs = 30000;
  snes->max_its   = 10000;

  ierr = PetscNewLog(snes, SNES_NCG, &neP);CHKERRQ(ierr);
  snes->data = (void*) neP;
  neP->monitor = PETSC_NULL;
  neP->betatype = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END
