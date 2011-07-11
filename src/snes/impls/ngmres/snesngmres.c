/* Defines the basic SNES object */
#include <private/snesimpl.h>

/* Private structure for the Anderson mixing method aka nonlinear Krylov */
typedef struct {
  Vec       *v, *w;
  PetscReal *f2;    /* 2-norms of function (residual) at each stage */
  PetscInt   msize; /* maximum size of space */
  PetscInt   csize; /* current size of space */
} SNES_NGMRES;

#undef __FUNCT__
#define __FUNCT__ "SNESReset_NGMRES"
PetscErrorCode SNESReset_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES*) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(ngmres->msize, &ngmres->v);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ngmres->msize, &ngmres->w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NGMRES"
PetscErrorCode SNESDestroy_NGMRES(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_NGMRES(snes);CHKERRQ(ierr);
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork, &snes->work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NGMRES"
PetscErrorCode SNESSetUp_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if 0
  if (snes->pc_side != PC_LEFT) {SETERRQ(((PetscObject) snes)->comm, PETSC_ERR_SUP, "Only left preconditioning allowed for SNESNGMRES");}
#endif
  ierr = VecDuplicateVecs(snes->vec_sol, ngmres->msize, &ngmres->v);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(snes->vec_sol, ngmres->msize, &ngmres->w);CHKERRQ(ierr);
  ierr = SNESDefaultGetWork(snes, 1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NGMRES"
PetscErrorCode SNESSetFromOptions_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES NGMRES options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-snes_gmres_restart", "Number of directions", "SNES", ngmres->msize, &ngmres->msize, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_NGMRES"
PetscErrorCode SNESView_NGMRES(SNES snes, PetscViewer viewer)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "  Size of space %d\n", ngmres->msize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NGMRES"
PetscErrorCode SNESSolve_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  SNES           pc;
  Vec            X, F, r, rOld, *V = ngmres->v, *W = ngmres->w;
  PetscScalar    wdot;
  PetscReal      fnorm;
  PetscInt       i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->reason  = SNES_CONVERGED_ITERATING;
  X             = snes->vec_sol;
  F             = snes->vec_func;
  rOld          = snes->work[0];

  ierr = SNESGetPC(snes, &pc);CHKERRQ(ierr);
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FP, "Infinite or not-a-number generated in norm");
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes, fnorm, 0);
  ierr = SNESMonitor(snes, 0, fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

#ifndef OLD_BARRY_CODE
  ierr = VecCopy(F, rOld);CHKERRQ(ierr);
#else
  /* Barry: What the heck is this part doing? */
  /* determine optimal scale factor -- slow code */
  ierr = VecDuplicate(P, &y);CHKERRQ(ierr);
  ierr = VecDuplicate(P, &w);CHKERRQ(ierr);
  ierr = MatMult(Amat, Pold, y);CHKERRQ(ierr);
  /*ierr = KSP_PCApplyBAorAB(ksp,Pold,y,w);CHKERRQ(ierr);  */    /* y = BAp */
  ierr  = VecDotNorm2(Pold, y, &rdot, &abr);CHKERRQ(ierr);   /*   rdot = (p)^T(BAp); abr = (BAp)^T (BAp) */
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  A0   = rdot/abr;
  ierr = VecAXPY(X,A0,Pold);CHKERRQ(ierr);             /*   x  <- x + scale p */
#endif

  /* Loop over batches of directions */
  /* Code from Barry I do not understand to solve the least-squares problem, Time to try again */
  for(k = 0; k < snes->max_its; k += ngmres->msize) {
    /* Loop over updates for this batch */
    /*   TODO: Incorporate the variant which use the analytic Jacobian */
    /*   TODO: Incorporate selection criteria for u^A from paper (need to store residual and update norms) */
    /*   TODO: Incorporate criteria for restarting from paper */
    for(i = 0; i < ngmres->msize && k+i < snes->max_its; ++i) {
      /* Get a new approxiate solution u^k */
      ierr = SNESSolve(pc, PETSC_NULL, X);CHKERRQ(ierr);
      /* Solve least squares problem using last msize iterates */
      ierr = SNESGetFunction(pc, &r, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr); /* r = F(u^M)                    */
      for(j = 0; j < i; ++j) {                                              /* r = \prod_j (I + v_j w^T_j) r */
        ierr = VecDot(W[j], r, &wdot);CHKERRQ(ierr);
        ierr = VecAXPY(r, wdot, V[j]);CHKERRQ(ierr);
      }
      ierr = VecCopy(rOld, W[i]);CHKERRQ(ierr);                             /* w_i = r_{old}                 */
      ierr = VecAXPY(rOld, -1.0, r);CHKERRQ(ierr);                          /* v_i =         r               */
      ierr = VecDot(W[i], rOld, &wdot);CHKERRQ(ierr);                       /*       -------------------     */
      ierr = VecCopy(r, V[i]);CHKERRQ(ierr);                                /*       w^T_i (r_{old} - r)     */
      ierr = VecScale(V[i], 1.0/wdot);CHKERRQ(ierr);
      ierr = VecDot(W[i], r, &wdot);CHKERRQ(ierr);                          /* r = (I + v_i w^T_i) r         */
      ierr = VecAXPY(r, wdot, V[i]);CHKERRQ(ierr);
      ierr = VecCopy(r, rOld);CHKERRQ(ierr);                                /* r_{old} = r                   */

      ierr = VecAXPY(X, 1.0, r);CHKERRQ(ierr);                              /* u^A = u^M + r                 */
      /* Monitor and log history */
      ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);                 /* F(u^k) */
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr);
      SNESLogConvHistory(snes, fnorm, 0);
      ierr = SNESMonitor(snes, i+k+1, fnorm);CHKERRQ(ierr);
      /* Check convergence */
      ierr = (*snes->ops->converged)(snes, i+k+1, 0.0, 0.0, fnorm, &snes->reason, snes->cnvP);CHKERRQ(ierr);
      if (snes->reason) PetscFunctionReturn(0);
      /* Select update between u^M and u^A */
      /* Decide whether to restart */
    }
  }
  snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(0);
}

/*MC
  SNESNGMRES - The Nonlinear Generalized Minimum Residual (NGMRES) method of Oosterlee and Washio.

   Level: beginner

   Notes: Supports only left preconditioning

   "Krylov Subspace Acceleration of Nonlinear Multigrid with Application to Recirculating Flows", C. W. Oosterlee and T. Washio,
   SIAM Journal on Scientific Computing, 21(5), 2000.

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NGMRES"
PetscErrorCode SNESCreate_NGMRES(SNES snes)
{
  SNES_NGMRES   *ngmres;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_NGMRES;
  snes->ops->setup          = SNESSetUp_NGMRES;
  snes->ops->setfromoptions = SNESSetFromOptions_NGMRES;
  snes->ops->view           = SNESView_NGMRES;
  snes->ops->solve          = SNESSolve_NGMRES;
  snes->ops->reset          = SNESReset_NGMRES;

  ierr = PetscNewLog(snes, SNES_NGMRES, &ngmres);CHKERRQ(ierr);
  snes->data = (void*) ngmres;
  ngmres->msize = 30;
  ngmres->csize = 0;

  ierr = SNESGetPC(snes, &snes->pc);CHKERRQ(ierr);
#if 0
  if (ksp->pc_side != PC_LEFT) {ierr = PetscInfo(ksp,"WARNING! Setting PC_SIDE for NGMRES to left!\n");CHKERRQ(ierr);}
  snes->pc_side = PC_LEFT;
#endif
  PetscFunctionReturn(0);
}
EXTERN_C_END
