
/*
    Defines the multigrid preconditioner interface.
*/
#include <petsc/private/pcmgimpl.h>                    /*I "petscksp.h" I*/
#include <petsc/private/kspimpl.h>
#include <petscdm.h>
PETSC_INTERN PetscErrorCode PCPreSolveChangeRHS(PC,PetscBool*);

/*
   Contains the list of registered coarse space construction routines
*/
PetscFunctionList PCMGCoarseList = NULL;

PetscErrorCode PCMGMCycle_Private(PC pc,PC_MG_Levels **mglevelsin,PetscBool transpose,PetscBool matapp,PCRichardsonConvergedReason *reason)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_MG_Levels   *mgc,*mglevels = *mglevelsin;
  PetscErrorCode ierr;
  PetscInt       cycles = (mglevels->level == 1) ? 1 : (PetscInt) mglevels->cycles;

  PetscFunctionBegin;
  if (mglevels->eventsmoothsolve) {ierr = PetscLogEventBegin(mglevels->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  if (!transpose) {
    if (matapp) {
      ierr = KSPMatSolve(mglevels->smoothd,mglevels->B,mglevels->X);CHKERRQ(ierr);  /* pre-smooth */
      ierr = KSPCheckSolve(mglevels->smoothd,pc,NULL);CHKERRQ(ierr);
    } else {
      ierr = KSPSolve(mglevels->smoothd,mglevels->b,mglevels->x);CHKERRQ(ierr);  /* pre-smooth */
      ierr = KSPCheckSolve(mglevels->smoothd,pc,mglevels->x);CHKERRQ(ierr);
    }
  } else {
    PetscCheckFalse(matapp,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not supported");
    ierr = KSPSolveTranspose(mglevels->smoothu,mglevels->b,mglevels->x);CHKERRQ(ierr); /* transpose of post-smooth */
    ierr = KSPCheckSolve(mglevels->smoothu,pc,mglevels->x);CHKERRQ(ierr);
  }
  if (mglevels->eventsmoothsolve) {ierr = PetscLogEventEnd(mglevels->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  if (mglevels->level) {  /* not the coarsest grid */
    if (mglevels->eventresidual) {ierr = PetscLogEventBegin(mglevels->eventresidual,0,0,0,0);CHKERRQ(ierr);}
    if (matapp && !mglevels->R) {
      ierr = MatDuplicate(mglevels->B,MAT_DO_NOT_COPY_VALUES,&mglevels->R);CHKERRQ(ierr);
    }
    if (!transpose) {
      if (matapp) { ierr = (*mglevels->matresidual)(mglevels->A,mglevels->B,mglevels->X,mglevels->R);CHKERRQ(ierr); }
      else { ierr = (*mglevels->residual)(mglevels->A,mglevels->b,mglevels->x,mglevels->r);CHKERRQ(ierr); }
    } else {
      if (matapp) { ierr = (*mglevels->matresidualtranspose)(mglevels->A,mglevels->B,mglevels->X,mglevels->R);CHKERRQ(ierr); }
      else { ierr = (*mglevels->residualtranspose)(mglevels->A,mglevels->b,mglevels->x,mglevels->r);CHKERRQ(ierr); }
    }
    if (mglevels->eventresidual) {ierr = PetscLogEventEnd(mglevels->eventresidual,0,0,0,0);CHKERRQ(ierr);}

    /* if on finest level and have convergence criteria set */
    if (mglevels->level == mglevels->levels-1 && mg->ttol && reason) {
      PetscReal rnorm;
      ierr = VecNorm(mglevels->r,NORM_2,&rnorm);CHKERRQ(ierr);
      if (rnorm <= mg->ttol) {
        if (rnorm < mg->abstol) {
          *reason = PCRICHARDSON_CONVERGED_ATOL;
          ierr    = PetscInfo(pc,"Linear solver has converged. Residual norm %g is less than absolute tolerance %g\n",(double)rnorm,(double)mg->abstol);CHKERRQ(ierr);
        } else {
          *reason = PCRICHARDSON_CONVERGED_RTOL;
          ierr    = PetscInfo(pc,"Linear solver has converged. Residual norm %g is less than relative tolerance times initial residual norm %g\n",(double)rnorm,(double)mg->ttol);CHKERRQ(ierr);
        }
        PetscFunctionReturn(0);
      }
    }

    mgc = *(mglevelsin - 1);
    if (mglevels->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (!transpose) {
      if (matapp) { ierr = MatMatRestrict(mglevels->restrct,mglevels->R,&mgc->B);CHKERRQ(ierr); }
      else { ierr = MatRestrict(mglevels->restrct,mglevels->r,mgc->b);CHKERRQ(ierr); }
    } else {
      if (matapp) { ierr = MatMatRestrict(mglevels->interpolate,mglevels->R,&mgc->B);CHKERRQ(ierr); }
      else { ierr = MatRestrict(mglevels->interpolate,mglevels->r,mgc->b);CHKERRQ(ierr); }
    }
    if (mglevels->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (matapp) {
      if (!mgc->X) {
        ierr = MatDuplicate(mgc->B,MAT_DO_NOT_COPY_VALUES,&mgc->X);CHKERRQ(ierr);
      } else {
        ierr = MatZeroEntries(mgc->X);CHKERRQ(ierr);
      }
    } else {
      ierr = VecZeroEntries(mgc->x);CHKERRQ(ierr);
    }
    while (cycles--) {
      ierr = PCMGMCycle_Private(pc,mglevelsin-1,transpose,matapp,reason);CHKERRQ(ierr);
    }
    if (mglevels->eventinterprestrict) {ierr = PetscLogEventBegin(mglevels->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (!transpose) {
      if (matapp) { ierr = MatMatInterpolateAdd(mglevels->interpolate,mgc->X,mglevels->X,&mglevels->X);CHKERRQ(ierr); }
      else { ierr = MatInterpolateAdd(mglevels->interpolate,mgc->x,mglevels->x,mglevels->x);CHKERRQ(ierr); }
    } else {
      ierr = MatInterpolateAdd(mglevels->restrct,mgc->x,mglevels->x,mglevels->x);CHKERRQ(ierr);
    }
    if (mglevels->eventinterprestrict) {ierr = PetscLogEventEnd(mglevels->eventinterprestrict,0,0,0,0);CHKERRQ(ierr);}
    if (mglevels->eventsmoothsolve) {ierr = PetscLogEventBegin(mglevels->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
    if (!transpose) {
      if (matapp) {
        ierr = KSPMatSolve(mglevels->smoothu,mglevels->B,mglevels->X);CHKERRQ(ierr);    /* post smooth */
        ierr = KSPCheckSolve(mglevels->smoothu,pc,NULL);CHKERRQ(ierr);
      } else {
        ierr = KSPSolve(mglevels->smoothu,mglevels->b,mglevels->x);CHKERRQ(ierr);    /* post smooth */
        ierr = KSPCheckSolve(mglevels->smoothu,pc,mglevels->x);CHKERRQ(ierr);
      }
    } else {
      PetscCheckFalse(matapp,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not supported");
      ierr = KSPSolveTranspose(mglevels->smoothd,mglevels->b,mglevels->x);CHKERRQ(ierr);    /* post smooth */
      ierr = KSPCheckSolve(mglevels->smoothd,pc,mglevels->x);CHKERRQ(ierr);
    }
    if (mglevels->cr) {
      PetscCheckFalse(matapp,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Not supported");
      /* TODO Turn on copy and turn off noisy if we have an exact solution
      ierr = VecCopy(mglevels->x, mglevels->crx);CHKERRQ(ierr);
      ierr = VecCopy(mglevels->b, mglevels->crb);CHKERRQ(ierr); */
      ierr = KSPSetNoisy_Private(mglevels->crx);CHKERRQ(ierr);
      ierr = KSPSolve(mglevels->cr,mglevels->crb,mglevels->crx);CHKERRQ(ierr);    /* compatible relaxation */
      ierr = KSPCheckSolve(mglevels->cr,pc,mglevels->crx);CHKERRQ(ierr);
    }
    if (mglevels->eventsmoothsolve) {ierr = PetscLogEventEnd(mglevels->eventsmoothsolve,0,0,0,0);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyRichardson_MG(PC pc,Vec b,Vec x,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool zeroguess,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PC             tpc;
  PetscBool      changeu,changed;
  PetscInt       levels = mglevels[0]->levels,i;

  PetscFunctionBegin;
  /* When the DM is supplying the matrix then it will not exist until here */
  for (i=0; i<levels; i++) {
    if (!mglevels[i]->A) {
      ierr = KSPGetOperators(mglevels[i]->smoothu,&mglevels[i]->A,NULL);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)mglevels[i]->A);CHKERRQ(ierr);
    }
  }

  ierr = KSPGetPC(mglevels[levels-1]->smoothd,&tpc);CHKERRQ(ierr);
  ierr = PCPreSolveChangeRHS(tpc,&changed);CHKERRQ(ierr);
  ierr = KSPGetPC(mglevels[levels-1]->smoothu,&tpc);CHKERRQ(ierr);
  ierr = PCPreSolveChangeRHS(tpc,&changeu);CHKERRQ(ierr);
  if (!changed && !changeu) {
    ierr = VecDestroy(&mglevels[levels-1]->b);CHKERRQ(ierr);
    mglevels[levels-1]->b = b;
  } else { /* if the smoother changes the rhs during PreSolve, we cannot use the input vector */
    if (!mglevels[levels-1]->b) {
      Vec *vec;

      ierr = KSPCreateVecs(mglevels[levels-1]->smoothd,1,&vec,0,NULL);CHKERRQ(ierr);
      mglevels[levels-1]->b = *vec;
      ierr = PetscFree(vec);CHKERRQ(ierr);
    }
    ierr = VecCopy(b,mglevels[levels-1]->b);CHKERRQ(ierr);
  }
  mglevels[levels-1]->x = x;

  mg->rtol   = rtol;
  mg->abstol = abstol;
  mg->dtol   = dtol;
  if (rtol) {
    /* compute initial residual norm for relative convergence test */
    PetscReal rnorm;
    if (zeroguess) {
      ierr = VecNorm(b,NORM_2,&rnorm);CHKERRQ(ierr);
    } else {
      ierr = (*mglevels[levels-1]->residual)(mglevels[levels-1]->A,b,x,w);CHKERRQ(ierr);
      ierr = VecNorm(w,NORM_2,&rnorm);CHKERRQ(ierr);
    }
    mg->ttol = PetscMax(rtol*rnorm,abstol);
  } else if (abstol) mg->ttol = abstol;
  else mg->ttol = 0.0;

  /* since smoother is applied to full system, not just residual we need to make sure that smoothers don't
     stop prematurely due to small residual */
  for (i=1; i<levels; i++) {
    ierr = KSPSetTolerances(mglevels[i]->smoothu,0,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    if (mglevels[i]->smoothu != mglevels[i]->smoothd) {
      /* For Richardson the initial guess is nonzero since it is solving in each cycle the original system not just applying as a preconditioner */
      ierr = KSPSetInitialGuessNonzero(mglevels[i]->smoothd,PETSC_TRUE);CHKERRQ(ierr);
      ierr = KSPSetTolerances(mglevels[i]->smoothd,0,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    }
  }

  *reason = (PCRichardsonConvergedReason)0;
  for (i=0; i<its; i++) {
    ierr = PCMGMCycle_Private(pc,mglevels+levels-1,PETSC_FALSE,PETSC_FALSE,reason);CHKERRQ(ierr);
    if (*reason) break;
  }
  if (!*reason) *reason = PCRICHARDSON_CONVERGED_ITS;
  *outits = i;
  if (!changed && !changeu) mglevels[levels-1]->b = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PCReset_MG(PC pc)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,c,n;

  PetscFunctionBegin;
  if (mglevels) {
    n = mglevels[0]->levels;
    for (i=0; i<n-1; i++) {
      ierr = VecDestroy(&mglevels[i+1]->r);CHKERRQ(ierr);
      ierr = VecDestroy(&mglevels[i]->b);CHKERRQ(ierr);
      ierr = VecDestroy(&mglevels[i]->x);CHKERRQ(ierr);
      ierr = MatDestroy(&mglevels[i+1]->R);CHKERRQ(ierr);
      ierr = MatDestroy(&mglevels[i]->B);CHKERRQ(ierr);
      ierr = MatDestroy(&mglevels[i]->X);CHKERRQ(ierr);
      ierr = VecDestroy(&mglevels[i]->crx);CHKERRQ(ierr);
      ierr = VecDestroy(&mglevels[i]->crb);CHKERRQ(ierr);
      ierr = MatDestroy(&mglevels[i+1]->restrct);CHKERRQ(ierr);
      ierr = MatDestroy(&mglevels[i+1]->interpolate);CHKERRQ(ierr);
      ierr = MatDestroy(&mglevels[i+1]->inject);CHKERRQ(ierr);
      ierr = VecDestroy(&mglevels[i+1]->rscale);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&mglevels[n-1]->crx);CHKERRQ(ierr);
    ierr = VecDestroy(&mglevels[n-1]->crb);CHKERRQ(ierr);
    /* this is not null only if the smoother on the finest level
       changes the rhs during PreSolve */
    ierr = VecDestroy(&mglevels[n-1]->b);CHKERRQ(ierr);
    ierr = MatDestroy(&mglevels[n-1]->B);CHKERRQ(ierr);

    for (i=0; i<n; i++) {
      if (mglevels[i]->coarseSpace) for (c = 0; c < mg->Nc; ++c) {ierr = VecDestroy(&mglevels[i]->coarseSpace[c]);CHKERRQ(ierr);}
      ierr = PetscFree(mglevels[i]->coarseSpace);CHKERRQ(ierr);
      mglevels[i]->coarseSpace = NULL;
      ierr = MatDestroy(&mglevels[i]->A);CHKERRQ(ierr);
      if (mglevels[i]->smoothd != mglevels[i]->smoothu) {
        ierr = KSPReset(mglevels[i]->smoothd);CHKERRQ(ierr);
      }
      ierr = KSPReset(mglevels[i]->smoothu);CHKERRQ(ierr);
      if (mglevels[i]->cr) {ierr = KSPReset(mglevels[i]->cr);CHKERRQ(ierr);}
    }
    mg->Nc = 0;
  }
  PetscFunctionReturn(0);
}

/* Implementing CR

We only want to make corrections that ``do not change'' the coarse solution. What we mean by not changing is that if I prolong my coarse solution to the fine grid and then inject that fine solution back to the coarse grid, I get the same answer. Injection is what Brannick calls R. We want the complementary projector to Inj, which we will call S, after Brannick, so that Inj S = 0. Now the orthogonal projector onto the range of Inj^T is

  Inj^T (Inj Inj^T)^{-1} Inj

and if Inj is a VecScatter, as it is now in PETSc, we have

  Inj^T Inj

and

  S = I - Inj^T Inj

since

  Inj S = Inj - (Inj Inj^T) Inj = 0.

Brannick suggests

  A \to S^T A S  \qquad\mathrm{and}\qquad M \to S^T M S

but I do not think his :math:`S^T S = I` is correct. Our S is an orthogonal projector, so :math:`S^T S = S^2 = S`. We will use

  M^{-1} A \to S M^{-1} A S

In fact, since it is somewhat hard in PETSc to do the symmetric application, we will just apply S on the left.

  Check: || Inj P - I ||_F < tol
  Check: In general, Inj Inj^T = I
*/

typedef struct {
  PC       mg;  /* The PCMG object */
  PetscInt l;   /* The multigrid level for this solver */
  Mat      Inj; /* The injection matrix */
  Mat      S;   /* I - Inj^T Inj */
} CRContext;

static PetscErrorCode CRSetup_Private(PC pc)
{
  CRContext     *ctx;
  Mat            It;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PCShellGetContext(pc, &ctx);CHKERRQ(ierr);
  ierr = PCMGGetInjection(ctx->mg, ctx->l, &It);CHKERRQ(ierr);
  PetscCheckFalse(!It,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONGSTATE, "CR requires that injection be defined for this PCMG");
  ierr = MatCreateTranspose(It, &ctx->Inj);CHKERRQ(ierr);
  ierr = MatCreateNormal(ctx->Inj, &ctx->S);CHKERRQ(ierr);
  ierr = MatScale(ctx->S, -1.0);CHKERRQ(ierr);
  ierr = MatShift(ctx->S,  1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CRApply_Private(PC pc, Vec x, Vec y)
{
  CRContext     *ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PCShellGetContext(pc, &ctx);CHKERRQ(ierr);
  ierr = MatMult(ctx->S, x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CRDestroy_Private(PC pc)
{
  CRContext     *ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PCShellGetContext(pc, &ctx);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->Inj);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->S);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = PCShellSetContext(pc, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateCR_Private(PC pc, PetscInt l, PC *cr)
{
  CRContext     *ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PCCreate(PetscObjectComm((PetscObject) pc), cr);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *cr, "S (complementary projector to injection)");CHKERRQ(ierr);
  ierr = PetscCalloc1(1, &ctx);CHKERRQ(ierr);
  ctx->mg = pc;
  ctx->l  = l;
  ierr = PCSetType(*cr, PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetContext(*cr, ctx);CHKERRQ(ierr);
  ierr = PCShellSetApply(*cr, CRApply_Private);CHKERRQ(ierr);
  ierr = PCShellSetSetUp(*cr, CRSetup_Private);CHKERRQ(ierr);
  ierr = PCShellSetDestroy(*cr, CRDestroy_Private);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGSetLevels_MG(PC pc,PetscInt levels,MPI_Comm *comms)
{
  PetscErrorCode ierr;
  PC_MG          *mg        = (PC_MG*)pc->data;
  MPI_Comm       comm;
  PC_MG_Levels   **mglevels = mg->levels;
  PCMGType       mgtype     = mg->am;
  PetscInt       mgctype    = (PetscInt) PC_MG_CYCLE_V;
  PetscInt       i;
  PetscMPIInt    size;
  const char     *prefix;
  PC             ipc;
  PetscInt       n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,levels,2);
  if (mg->nlevels == levels) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  if (mglevels) {
    mgctype = mglevels[0]->cycles;
    /* changing the number of levels so free up the previous stuff */
    ierr = PCReset_MG(pc);CHKERRQ(ierr);
    n    = mglevels[0]->levels;
    for (i=0; i<n; i++) {
      if (mglevels[i]->smoothd != mglevels[i]->smoothu) {
        ierr = KSPDestroy(&mglevels[i]->smoothd);CHKERRQ(ierr);
      }
      ierr = KSPDestroy(&mglevels[i]->smoothu);CHKERRQ(ierr);
      ierr = KSPDestroy(&mglevels[i]->cr);CHKERRQ(ierr);
      ierr = PetscFree(mglevels[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(mg->levels);CHKERRQ(ierr);
  }

  mg->nlevels = levels;

  ierr = PetscMalloc1(levels,&mglevels);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)pc,levels*(sizeof(PC_MG*)));CHKERRQ(ierr);

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);

  mg->stageApply = 0;
  for (i=0; i<levels; i++) {
    ierr = PetscNewLog(pc,&mglevels[i]);CHKERRQ(ierr);

    mglevels[i]->level               = i;
    mglevels[i]->levels              = levels;
    mglevels[i]->cycles              = mgctype;
    mg->default_smoothu              = 2;
    mg->default_smoothd              = 2;
    mglevels[i]->eventsmoothsetup    = 0;
    mglevels[i]->eventsmoothsolve    = 0;
    mglevels[i]->eventresidual       = 0;
    mglevels[i]->eventinterprestrict = 0;

    if (comms) comm = comms[i];
    if (comm != MPI_COMM_NULL) {
      ierr = KSPCreate(comm,&mglevels[i]->smoothd);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(mglevels[i]->smoothd,pc->erroriffailure);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)mglevels[i]->smoothd,(PetscObject)pc,levels-i);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(mglevels[i]->smoothd,prefix);CHKERRQ(ierr);
      ierr = PetscObjectComposedDataSetInt((PetscObject) mglevels[i]->smoothd, PetscMGLevelId, mglevels[i]->level);CHKERRQ(ierr);
      if (i || levels == 1) {
        char tprefix[128];

        ierr = KSPSetType(mglevels[i]->smoothd,KSPCHEBYSHEV);CHKERRQ(ierr);
        ierr = KSPSetConvergenceTest(mglevels[i]->smoothd,KSPConvergedSkip,NULL,NULL);CHKERRQ(ierr);
        ierr = KSPSetNormType(mglevels[i]->smoothd,KSP_NORM_NONE);CHKERRQ(ierr);
        ierr = KSPGetPC(mglevels[i]->smoothd,&ipc);CHKERRQ(ierr);
        ierr = PCSetType(ipc,PCSOR);CHKERRQ(ierr);
        ierr = KSPSetTolerances(mglevels[i]->smoothd,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT, mg->default_smoothd);CHKERRQ(ierr);

        ierr = PetscSNPrintf(tprefix,128,"mg_levels_%d_",(int)i);CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(mglevels[i]->smoothd,tprefix);CHKERRQ(ierr);
      } else {
        ierr = KSPAppendOptionsPrefix(mglevels[0]->smoothd,"mg_coarse_");CHKERRQ(ierr);

        /* coarse solve is (redundant) LU by default; set shifttype NONZERO to avoid annoying zero-pivot in LU preconditioner */
        ierr = KSPSetType(mglevels[0]->smoothd,KSPPREONLY);CHKERRQ(ierr);
        ierr = KSPGetPC(mglevels[0]->smoothd,&ipc);CHKERRQ(ierr);
        ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
        if (size > 1) {
          ierr = PCSetType(ipc,PCREDUNDANT);CHKERRQ(ierr);
        } else {
          ierr = PCSetType(ipc,PCLU);CHKERRQ(ierr);
        }
        ierr = PCFactorSetShiftType(ipc,MAT_SHIFT_INBLOCKS);CHKERRQ(ierr);
      }
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)mglevels[i]->smoothd);CHKERRQ(ierr);
    }
    mglevels[i]->smoothu = mglevels[i]->smoothd;
    mg->rtol             = 0.0;
    mg->abstol           = 0.0;
    mg->dtol             = 0.0;
    mg->ttol             = 0.0;
    mg->cyclesperpcapply = 1;
  }
  mg->levels = mglevels;
  ierr = PCMGSetType(pc,mgtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PCMGSetLevels - Sets the number of levels to use with MG.
   Must be called before any other MG routine.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  levels - the number of levels
-  comms - optional communicators for each level; this is to allow solving the coarser problems
           on smaller sets of processes. For processes that are not included in the computation
           you must pass MPI_COMM_NULL. Use comms = NULL to specify that all processes
           should participate in each level of problem.

   Level: intermediate

   Notes:
     If the number of levels is one then the multigrid uses the -mg_levels prefix
     for setting the level options rather than the -mg_coarse prefix.

     You can free the information in comms after this routine is called.

     The array of MPI communicators must contain MPI_COMM_NULL for those ranks that at each level
     are not participating in the coarser solve. For example, with 2 levels and 1 and 2 ranks on
     the two levels, rank 0 in the original communicator will pass in an array of 2 communicators
     of size 2 and 1, while rank 1 in the original communicator will pass in array of 2 communicators
     the first of size 2 and the second of value MPI_COMM_NULL since the rank 1 does not participate
     in the coarse grid solve.

     Since each coarser level may have a new MPI_Comm with fewer ranks than the previous, one
     must take special care in providing the restriction and interpolation operation. We recommend
     providing these as two step operations; first perform a standard restriction or interpolation on
     the full number of ranks for that level and then use an MPI call to copy the resulting vector
     array entries (after calls to VecGetArray()) to the smaller or larger number of ranks, note in both
     cases the MPI calls must be made on the larger of the two communicators. Traditional MPI send and
     recieves or MPI_AlltoAllv() could be used to do the reshuffling of the vector entries.

   Fortran Notes:
     Use comms = PETSC_NULL_MPI_COMM as the equivalent of NULL in the C interface. Note PETSC_NULL_MPI_COMM
     is not MPI_COMM_NULL. It is more like PETSC_NULL_INTEGER, PETSC_NULL_REAL etc.

.seealso: PCMGSetType(), PCMGGetLevels()
@*/
PetscErrorCode PCMGSetLevels(PC pc,PetscInt levels,MPI_Comm *comms)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (comms) PetscValidPointer(comms,3);
  ierr = PetscTryMethod(pc,"PCMGSetLevels_C",(PC,PetscInt,MPI_Comm*),(pc,levels,comms));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_MG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscInt       i,n;

  PetscFunctionBegin;
  ierr = PCReset_MG(pc);CHKERRQ(ierr);
  if (mglevels) {
    n = mglevels[0]->levels;
    for (i=0; i<n; i++) {
      if (mglevels[i]->smoothd != mglevels[i]->smoothu) {
        ierr = KSPDestroy(&mglevels[i]->smoothd);CHKERRQ(ierr);
      }
      ierr = KSPDestroy(&mglevels[i]->smoothu);CHKERRQ(ierr);
      ierr = KSPDestroy(&mglevels[i]->cr);CHKERRQ(ierr);
      ierr = PetscFree(mglevels[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(mg->levels);CHKERRQ(ierr);
  }
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGetInterpolations_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGetCoarseOperators_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   PCApply_MG - Runs either an additive, multiplicative, Kaskadic
             or full cycle of multigrid.

  Note:
  A simple wrapper which calls PCMGMCycle(),PCMGACycle(), or PCMGFCycle().
*/
static PetscErrorCode PCApply_MG_Internal(PC pc,Vec b,Vec x,Mat B,Mat X,PetscBool transpose)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PC             tpc;
  PetscInt       levels = mglevels[0]->levels,i;
  PetscBool      changeu,changed,matapp;

  PetscFunctionBegin;
  matapp = (PetscBool)(B && X);
  if (mg->stageApply) {ierr = PetscLogStagePush(mg->stageApply);CHKERRQ(ierr);}
  /* When the DM is supplying the matrix then it will not exist until here */
  for (i=0; i<levels; i++) {
    if (!mglevels[i]->A) {
      ierr = KSPGetOperators(mglevels[i]->smoothu,&mglevels[i]->A,NULL);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)mglevels[i]->A);CHKERRQ(ierr);
    }
  }

  ierr = KSPGetPC(mglevels[levels-1]->smoothd,&tpc);CHKERRQ(ierr);
  ierr = PCPreSolveChangeRHS(tpc,&changed);CHKERRQ(ierr);
  ierr = KSPGetPC(mglevels[levels-1]->smoothu,&tpc);CHKERRQ(ierr);
  ierr = PCPreSolveChangeRHS(tpc,&changeu);CHKERRQ(ierr);
  if (!changeu && !changed) {
    if (matapp) {
      ierr = MatDestroy(&mglevels[levels-1]->B);CHKERRQ(ierr);
      mglevels[levels-1]->B = B;
    } else {
      ierr = VecDestroy(&mglevels[levels-1]->b);CHKERRQ(ierr);
      mglevels[levels-1]->b = b;
    }
  } else { /* if the smoother changes the rhs during PreSolve, we cannot use the input vector */
    if (matapp) {
      if (mglevels[levels-1]->B) {
        PetscInt  N1,N2;
        PetscBool flg;

        ierr = MatGetSize(mglevels[levels-1]->B,NULL,&N1);CHKERRQ(ierr);
        ierr = MatGetSize(B,NULL,&N2);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)mglevels[levels-1]->B,((PetscObject)B)->type_name,&flg);CHKERRQ(ierr);
        if (N1 != N2 || !flg) {
          ierr = MatDestroy(&mglevels[levels-1]->B);CHKERRQ(ierr);
        }
      }
      if (!mglevels[levels-1]->B) {
        ierr = MatDuplicate(B,MAT_COPY_VALUES,&mglevels[levels-1]->B);CHKERRQ(ierr);
      } else {
        ierr = MatCopy(B,mglevels[levels-1]->B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      }
    } else {
      if (!mglevels[levels-1]->b) {
        Vec *vec;

        ierr = KSPCreateVecs(mglevels[levels-1]->smoothd,1,&vec,0,NULL);CHKERRQ(ierr);
        mglevels[levels-1]->b = *vec;
        ierr = PetscFree(vec);CHKERRQ(ierr);
      }
      ierr = VecCopy(b,mglevels[levels-1]->b);CHKERRQ(ierr);
    }
  }
  if (matapp) { mglevels[levels-1]->X = X; }
  else { mglevels[levels-1]->x = x; }

  /* If coarser Xs are present, it means we have already block applied the PC at least once
     Reset operators if sizes/type do no match */
  if (matapp && levels > 1 && mglevels[levels-2]->X) {
    PetscInt  Xc,Bc;
    PetscBool flg;

    ierr = MatGetSize(mglevels[levels-2]->X,NULL,&Xc);CHKERRQ(ierr);
    ierr = MatGetSize(mglevels[levels-1]->B,NULL,&Bc);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)mglevels[levels-2]->X,((PetscObject)mglevels[levels-1]->X)->type_name,&flg);CHKERRQ(ierr);
    if (Xc != Bc || !flg) {
      ierr = MatDestroy(&mglevels[levels-1]->R);CHKERRQ(ierr);
      for (i=0;i<levels-1;i++) {
        ierr = MatDestroy(&mglevels[i]->R);CHKERRQ(ierr);
        ierr = MatDestroy(&mglevels[i]->B);CHKERRQ(ierr);
        ierr = MatDestroy(&mglevels[i]->X);CHKERRQ(ierr);
      }
    }
  }

  if (mg->am == PC_MG_MULTIPLICATIVE) {
    if (matapp) { ierr = MatZeroEntries(X);CHKERRQ(ierr); }
    else { ierr = VecZeroEntries(x);CHKERRQ(ierr); }
    for (i=0; i<mg->cyclesperpcapply; i++) {
      ierr = PCMGMCycle_Private(pc,mglevels+levels-1,transpose,matapp,NULL);CHKERRQ(ierr);
    }
  } else if (mg->am == PC_MG_ADDITIVE) {
    ierr = PCMGACycle_Private(pc,mglevels,transpose,matapp);CHKERRQ(ierr);
  } else if (mg->am == PC_MG_KASKADE) {
    ierr = PCMGKCycle_Private(pc,mglevels,transpose,matapp);CHKERRQ(ierr);
  } else {
    ierr = PCMGFCycle_Private(pc,mglevels,transpose,matapp);CHKERRQ(ierr);
  }
  if (mg->stageApply) {ierr = PetscLogStagePop();CHKERRQ(ierr);}
  if (!changeu && !changed) {
    if (matapp) { mglevels[levels-1]->B = NULL; }
    else { mglevels[levels-1]->b = NULL; }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_MG(PC pc,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApply_MG_Internal(pc,b,x,NULL,NULL,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_MG(PC pc,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApply_MG_Internal(pc,b,x,NULL,NULL,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_MG(PC pc,Mat b,Mat x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCApply_MG_Internal(pc,NULL,NULL,b,x,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_MG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode   ierr;
  PetscInt         levels,cycles;
  PetscBool        flg, flg2;
  PC_MG            *mg = (PC_MG*)pc->data;
  PC_MG_Levels     **mglevels;
  PCMGType         mgtype;
  PCMGCycleType    mgctype;
  PCMGGalerkinType gtype;

  PetscFunctionBegin;
  levels = PetscMax(mg->nlevels,1);
  ierr = PetscOptionsHead(PetscOptionsObject,"Multigrid options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_mg_levels","Number of Levels","PCMGSetLevels",levels,&levels,&flg);CHKERRQ(ierr);
  if (!flg && !mg->levels && pc->dm) {
    ierr = DMGetRefineLevel(pc->dm,&levels);CHKERRQ(ierr);
    levels++;
    mg->usedmfornumberoflevels = PETSC_TRUE;
  }
  ierr = PCMGSetLevels(pc,levels,NULL);CHKERRQ(ierr);
  mglevels = mg->levels;

  mgctype = (PCMGCycleType) mglevels[0]->cycles;
  ierr    = PetscOptionsEnum("-pc_mg_cycle_type","V cycle or for W-cycle","PCMGSetCycleType",PCMGCycleTypes,(PetscEnum)mgctype,(PetscEnum*)&mgctype,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCMGSetCycleType(pc,mgctype);CHKERRQ(ierr);
  }
  gtype = mg->galerkin;
  ierr = PetscOptionsEnum("-pc_mg_galerkin","Use Galerkin process to compute coarser operators","PCMGSetGalerkin",PCMGGalerkinTypes,(PetscEnum)gtype,(PetscEnum*)&gtype,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCMGSetGalerkin(pc,gtype);CHKERRQ(ierr);
  }
  flg2 = PETSC_FALSE;
  ierr = PetscOptionsBool("-pc_mg_adapt_interp","Adapt interpolation using some coarse space","PCMGSetAdaptInterpolation",PETSC_FALSE,&flg2,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCMGSetAdaptInterpolation(pc, flg2);CHKERRQ(ierr);}
  ierr = PetscOptionsInt("-pc_mg_adapt_interp_n","Size of the coarse space for adaptive interpolation","PCMGSetCoarseSpace",mg->Nc,&mg->Nc,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_mg_adapt_interp_coarse_space","Type of coarse space: polynomial, harmonic, eigenvector, generalized_eigenvector","PCMGSetAdaptCoarseSpaceType",PCMGCoarseSpaceTypes,(PetscEnum)mg->coarseSpaceType,(PetscEnum*)&mg->coarseSpaceType,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_mg_mesp_monitor","Monitor the multilevel eigensolver","PCMGSetAdaptInterpolation",PETSC_FALSE,&mg->mespMonitor,&flg);CHKERRQ(ierr);
  flg2 = PETSC_FALSE;
  ierr = PetscOptionsBool("-pc_mg_adapt_cr","Monitor coarse space quality using Compatible Relaxation (CR)","PCMGSetAdaptCR",PETSC_FALSE,&flg2,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PCMGSetAdaptCR(pc, flg2);CHKERRQ(ierr);}
  flg = PETSC_FALSE;
  ierr = PetscOptionsBool("-pc_mg_distinct_smoothup","Create separate smoothup KSP and append the prefix _up","PCMGSetDistinctSmoothUp",PETSC_FALSE,&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PCMGSetDistinctSmoothUp(pc);CHKERRQ(ierr);
  }
  mgtype = mg->am;
  ierr   = PetscOptionsEnum("-pc_mg_type","Multigrid type","PCMGSetType",PCMGTypes,(PetscEnum)mgtype,(PetscEnum*)&mgtype,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCMGSetType(pc,mgtype);CHKERRQ(ierr);
  }
  if (mg->am == PC_MG_MULTIPLICATIVE) {
    ierr = PetscOptionsInt("-pc_mg_multiplicative_cycles","Number of cycles for each preconditioner step","PCMGMultiplicativeSetCycles",mg->cyclesperpcapply,&cycles,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCMGMultiplicativeSetCycles(pc,cycles);CHKERRQ(ierr);
    }
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-pc_mg_log","Log times for each multigrid level","None",flg,&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    PetscInt i;
    char     eventname[128];

    levels = mglevels[0]->levels;
    for (i=0; i<levels; i++) {
      sprintf(eventname,"MGSetup Level %d",(int)i);
      ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->classid,&mglevels[i]->eventsmoothsetup);CHKERRQ(ierr);
      sprintf(eventname,"MGSmooth Level %d",(int)i);
      ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->classid,&mglevels[i]->eventsmoothsolve);CHKERRQ(ierr);
      if (i) {
        sprintf(eventname,"MGResid Level %d",(int)i);
        ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->classid,&mglevels[i]->eventresidual);CHKERRQ(ierr);
        sprintf(eventname,"MGInterp Level %d",(int)i);
        ierr = PetscLogEventRegister(eventname,((PetscObject)pc)->classid,&mglevels[i]->eventinterprestrict);CHKERRQ(ierr);
      }
    }

#if defined(PETSC_USE_LOG)
    {
      const char    *sname = "MG Apply";
      PetscStageLog stageLog;
      PetscInt      st;

      ierr = PetscLogGetStageLog(&stageLog);CHKERRQ(ierr);
      for (st = 0; st < stageLog->numStages; ++st) {
        PetscBool same;

        ierr = PetscStrcmp(stageLog->stageInfo[st].name, sname, &same);CHKERRQ(ierr);
        if (same) mg->stageApply = st;
      }
      if (!mg->stageApply) {
        ierr = PetscLogStageRegister(sname, &mg->stageApply);CHKERRQ(ierr);
      }
    }
#endif
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  /* Check option consistency */
  ierr = PCMGGetGalerkin(pc, &gtype);CHKERRQ(ierr);
  ierr = PCMGGetAdaptInterpolation(pc, &flg);CHKERRQ(ierr);
  PetscCheckFalse(flg && (gtype >= PC_MG_GALERKIN_NONE),PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_INCOMP, "Must use Galerkin coarse operators when adapting the interpolator");
  PetscFunctionReturn(0);
}

const char *const PCMGTypes[] = {"MULTIPLICATIVE","ADDITIVE","FULL","KASKADE","PCMGType","PC_MG",NULL};
const char *const PCMGCycleTypes[] = {"invalid","v","w","PCMGCycleType","PC_MG_CYCLE",NULL};
const char *const PCMGGalerkinTypes[] = {"both","pmat","mat","none","external","PCMGGalerkinType","PC_MG_GALERKIN",NULL};
const char *const PCMGCoarseSpaceTypes[] = {"polynomial","harmonic","eigenvector","generalized_eigenvector","PCMGCoarseSpaceType","PCMG_POLYNOMIAL",NULL};

#include <petscdraw.h>
PetscErrorCode PCView_MG(PC pc,PetscViewer viewer)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       levels = mglevels ? mglevels[0]->levels : 0,i;
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    const char *cyclename = levels ? (mglevels[0]->cycles == PC_MG_CYCLE_V ? "v" : "w") : "unknown";
    ierr = PetscViewerASCIIPrintf(viewer,"  type is %s, levels=%D cycles=%s\n", PCMGTypes[mg->am],levels,cyclename);CHKERRQ(ierr);
    if (mg->am == PC_MG_MULTIPLICATIVE) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Cycles per PCApply=%d\n",mg->cyclesperpcapply);CHKERRQ(ierr);
    }
    if (mg->galerkin == PC_MG_GALERKIN_BOTH) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Using Galerkin computed coarse grid matrices\n");CHKERRQ(ierr);
    } else if (mg->galerkin == PC_MG_GALERKIN_PMAT) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Using Galerkin computed coarse grid matrices for pmat\n");CHKERRQ(ierr);
    } else if (mg->galerkin == PC_MG_GALERKIN_MAT) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Using Galerkin computed coarse grid matrices for mat\n");CHKERRQ(ierr);
    } else if (mg->galerkin == PC_MG_GALERKIN_EXTERNAL) {
      ierr = PetscViewerASCIIPrintf(viewer,"    Using externally compute Galerkin coarse grid matrices\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"    Not using Galerkin computed coarse grid matrices\n");CHKERRQ(ierr);
    }
    if (mg->view) {
      ierr = (*mg->view)(pc,viewer);CHKERRQ(ierr);
    }
    for (i=0; i<levels; i++) {
      if (!i) {
        ierr = PetscViewerASCIIPrintf(viewer,"Coarse grid solver -- level -------------------------------\n",i);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Down solver (pre-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = KSPView(mglevels[i]->smoothd,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      if (i && mglevels[i]->smoothd == mglevels[i]->smoothu) {
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) same as down solver (pre-smoother)\n");CHKERRQ(ierr);
      } else if (i) {
        ierr = PetscViewerASCIIPrintf(viewer,"Up solver (post-smoother) on level %D -------------------------------\n",i);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = KSPView(mglevels[i]->smoothu,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
      if (i && mglevels[i]->cr) {
        ierr = PetscViewerASCIIPrintf(viewer,"CR solver on level %D -------------------------------\n",i);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = KSPView(mglevels[i]->cr,viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      }
    }
  } else if (isbinary) {
    for (i=levels-1; i>=0; i--) {
      ierr = KSPView(mglevels[i]->smoothd,viewer);CHKERRQ(ierr);
      if (i && mglevels[i]->smoothd != mglevels[i]->smoothu) {
        ierr = KSPView(mglevels[i]->smoothu,viewer);CHKERRQ(ierr);
      }
    }
  } else if (isdraw) {
    PetscDraw draw;
    PetscReal x,w,y,bottom,th;
    ierr   = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr   = PetscDrawGetCurrentPoint(draw,&x,&y);CHKERRQ(ierr);
    ierr   = PetscDrawStringGetSize(draw,NULL,&th);CHKERRQ(ierr);
    bottom = y - th;
    for (i=levels-1; i>=0; i--) {
      if (!mglevels[i]->smoothu || (mglevels[i]->smoothu == mglevels[i]->smoothd)) {
        ierr = PetscDrawPushCurrentPoint(draw,x,bottom);CHKERRQ(ierr);
        ierr = KSPView(mglevels[i]->smoothd,viewer);CHKERRQ(ierr);
        ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
      } else {
        w    = 0.5*PetscMin(1.0-x,x);
        ierr = PetscDrawPushCurrentPoint(draw,x+w,bottom);CHKERRQ(ierr);
        ierr = KSPView(mglevels[i]->smoothd,viewer);CHKERRQ(ierr);
        ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
        ierr = PetscDrawPushCurrentPoint(draw,x-w,bottom);CHKERRQ(ierr);
        ierr = KSPView(mglevels[i]->smoothu,viewer);CHKERRQ(ierr);
        ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
      }
      ierr    = PetscDrawGetBoundingBox(draw,NULL,&bottom,NULL,NULL);CHKERRQ(ierr);
      bottom -= th;
    }
  }
  PetscFunctionReturn(0);
}

#include <petsc/private/kspimpl.h>

/*
    Calls setup for the KSP on each level
*/
PetscErrorCode PCSetUp_MG(PC pc)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,n;
  PC             cpc;
  PetscBool      dump = PETSC_FALSE,opsset,use_amat,missinginterpolate = PETSC_FALSE;
  Mat            dA,dB;
  Vec            tvec;
  DM             *dms;
  PetscViewer    viewer = NULL;
  PetscBool      dAeqdB = PETSC_FALSE, needRestricts = PETSC_FALSE, doCR = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCheckFalse(!mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels with PCMGSetLevels() before setting up");
  n = mglevels[0]->levels;
  /* FIX: Move this to PCSetFromOptions_MG? */
  if (mg->usedmfornumberoflevels) {
    PetscInt levels;
    ierr = DMGetRefineLevel(pc->dm,&levels);CHKERRQ(ierr);
    levels++;
    if (levels > n) { /* the problem is now being solved on a finer grid */
      ierr     = PCMGSetLevels(pc,levels,NULL);CHKERRQ(ierr);
      n        = levels;
      ierr     = PCSetFromOptions(pc);CHKERRQ(ierr); /* it is bad to call this here, but otherwise will never be called for the new hierarchy */
      mglevels = mg->levels;
    }
  }
  ierr = KSPGetPC(mglevels[0]->smoothd,&cpc);CHKERRQ(ierr);

  /* If user did not provide fine grid operators OR operator was not updated since last global KSPSetOperators() */
  /* so use those from global PC */
  /* Is this what we always want? What if user wants to keep old one? */
  ierr = KSPGetOperatorsSet(mglevels[n-1]->smoothd,NULL,&opsset);CHKERRQ(ierr);
  if (opsset) {
    Mat mmat;
    ierr = KSPGetOperators(mglevels[n-1]->smoothd,NULL,&mmat);CHKERRQ(ierr);
    if (mmat == pc->pmat) opsset = PETSC_FALSE;
  }

  /* Create CR solvers */
  ierr = PCMGGetAdaptCR(pc, &doCR);CHKERRQ(ierr);
  if (doCR) {
    const char *prefix;

    ierr = PCGetOptionsPrefix(pc, &prefix);CHKERRQ(ierr);
    for (i = 1; i < n; ++i) {
      PC   ipc, cr;
      char crprefix[128];

      ierr = KSPCreate(PetscObjectComm((PetscObject) pc), &mglevels[i]->cr);CHKERRQ(ierr);
      ierr = KSPSetErrorIfNotConverged(mglevels[i]->cr, PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject) mglevels[i]->cr, (PetscObject) pc, n-i);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(mglevels[i]->cr, prefix);CHKERRQ(ierr);
      ierr = PetscObjectComposedDataSetInt((PetscObject) mglevels[i]->cr, PetscMGLevelId, mglevels[i]->level);CHKERRQ(ierr);
      ierr = KSPSetType(mglevels[i]->cr, KSPCHEBYSHEV);CHKERRQ(ierr);
      ierr = KSPSetConvergenceTest(mglevels[i]->cr, KSPConvergedSkip, NULL, NULL);CHKERRQ(ierr);
      ierr = KSPSetNormType(mglevels[i]->cr, KSP_NORM_PRECONDITIONED);CHKERRQ(ierr);
      ierr = KSPGetPC(mglevels[i]->cr, &ipc);CHKERRQ(ierr);

      ierr = PCSetType(ipc, PCCOMPOSITE);CHKERRQ(ierr);
      ierr = PCCompositeSetType(ipc, PC_COMPOSITE_MULTIPLICATIVE);CHKERRQ(ierr);
      ierr = PCCompositeAddPCType(ipc, PCSOR);CHKERRQ(ierr);
      ierr = CreateCR_Private(pc, i, &cr);CHKERRQ(ierr);
      ierr = PCCompositeAddPC(ipc, cr);CHKERRQ(ierr);
      ierr = PCDestroy(&cr);CHKERRQ(ierr);

      ierr = KSPSetTolerances(mglevels[i]->cr, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, mg->default_smoothd);CHKERRQ(ierr);
      ierr = KSPSetInitialGuessNonzero(mglevels[i]->cr, PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscSNPrintf(crprefix, 128, "mg_levels_%d_cr_", (int) i);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(mglevels[i]->cr, crprefix);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject) pc, (PetscObject) mglevels[i]->cr);CHKERRQ(ierr);
    }
  }

  if (!opsset) {
    ierr = PCGetUseAmat(pc,&use_amat);CHKERRQ(ierr);
    if (use_amat) {
      ierr = PetscInfo(pc,"Using outer operators to define finest grid operator \n  because PCMGGetSmoother(pc,nlevels-1,&ksp);KSPSetOperators(ksp,...); was not called.\n");CHKERRQ(ierr);
      ierr = KSPSetOperators(mglevels[n-1]->smoothd,pc->mat,pc->pmat);CHKERRQ(ierr);
    } else {
      ierr = PetscInfo(pc,"Using matrix (pmat) operators to define finest grid operator \n  because PCMGGetSmoother(pc,nlevels-1,&ksp);KSPSetOperators(ksp,...); was not called.\n");CHKERRQ(ierr);
      ierr = KSPSetOperators(mglevels[n-1]->smoothd,pc->pmat,pc->pmat);CHKERRQ(ierr);
    }
  }

  for (i=n-1; i>0; i--) {
    if (!(mglevels[i]->interpolate || mglevels[i]->restrct)) {
      missinginterpolate = PETSC_TRUE;
      continue;
    }
  }

  ierr = KSPGetOperators(mglevels[n-1]->smoothd,&dA,&dB);CHKERRQ(ierr);
  if (dA == dB) dAeqdB = PETSC_TRUE;
  if ((mg->galerkin == PC_MG_GALERKIN_NONE) || (((mg->galerkin == PC_MG_GALERKIN_PMAT) || (mg->galerkin == PC_MG_GALERKIN_MAT)) && !dAeqdB)) {
    needRestricts = PETSC_TRUE;  /* user must compute either mat, pmat, or both so must restrict x to coarser levels */
  }

  /*
   Skipping if user has provided all interpolation/restriction needed (since DM might not be able to produce them (when coming from SNES/TS)
   Skipping for galerkin==2 (externally managed hierarchy such as ML and GAMG). Cleaner logic here would be great. Wrap ML/GAMG as DMs?
  */
  if (missinginterpolate && pc->dm && mg->galerkin != PC_MG_GALERKIN_EXTERNAL && !pc->setupcalled) {
        /* construct the interpolation from the DMs */
    Mat p;
    Vec rscale;
    ierr     = PetscMalloc1(n,&dms);CHKERRQ(ierr);
    dms[n-1] = pc->dm;
    /* Separately create them so we do not get DMKSP interference between levels */
    for (i=n-2; i>-1; i--) {ierr = DMCoarsen(dms[i+1],MPI_COMM_NULL,&dms[i]);CHKERRQ(ierr);}
        /*
           Force the mat type of coarse level operator to be AIJ because usually we want to use LU for coarse level.
           Notice that it can be overwritten by -mat_type because KSPSetUp() reads command line options.
           But it is safe to use -dm_mat_type.

           The mat type should not be hardcoded like this, we need to find a better way.
    ierr = DMSetMatType(dms[0],MATAIJ);CHKERRQ(ierr);
    */
    for (i=n-2; i>-1; i--) {
      DMKSP     kdm;
      PetscBool dmhasrestrict, dmhasinject;
      ierr = KSPSetDM(mglevels[i]->smoothd,dms[i]);CHKERRQ(ierr);
      if (!needRestricts) {ierr = KSPSetDMActive(mglevels[i]->smoothd,PETSC_FALSE);CHKERRQ(ierr);}
      if (mglevels[i]->smoothd != mglevels[i]->smoothu) {
        ierr = KSPSetDM(mglevels[i]->smoothu,dms[i]);CHKERRQ(ierr);
        if (!needRestricts) {ierr = KSPSetDMActive(mglevels[i]->smoothu,PETSC_FALSE);CHKERRQ(ierr);}
      }
      if (mglevels[i]->cr) {
        ierr = KSPSetDM(mglevels[i]->cr,dms[i]);CHKERRQ(ierr);
        if (!needRestricts) {ierr = KSPSetDMActive(mglevels[i]->cr,PETSC_FALSE);CHKERRQ(ierr);}
      }
      ierr = DMGetDMKSPWrite(dms[i],&kdm);CHKERRQ(ierr);
      /* Ugly hack so that the next KSPSetUp() will use the RHS that we set. A better fix is to change dmActive to take
       * a bitwise OR of computing the matrix, RHS, and initial iterate. */
      kdm->ops->computerhs = NULL;
      kdm->rhsctx          = NULL;
      if (!mglevels[i+1]->interpolate) {
        ierr = DMCreateInterpolation(dms[i],dms[i+1],&p,&rscale);CHKERRQ(ierr);
        ierr = PCMGSetInterpolation(pc,i+1,p);CHKERRQ(ierr);
        if (rscale) {ierr = PCMGSetRScale(pc,i+1,rscale);CHKERRQ(ierr);}
        ierr = VecDestroy(&rscale);CHKERRQ(ierr);
        ierr = MatDestroy(&p);CHKERRQ(ierr);
      }
      ierr = DMHasCreateRestriction(dms[i],&dmhasrestrict);CHKERRQ(ierr);
      if (dmhasrestrict && !mglevels[i+1]->restrct) {
        ierr = DMCreateRestriction(dms[i],dms[i+1],&p);CHKERRQ(ierr);
        ierr = PCMGSetRestriction(pc,i+1,p);CHKERRQ(ierr);
        ierr = MatDestroy(&p);CHKERRQ(ierr);
      }
      ierr = DMHasCreateInjection(dms[i],&dmhasinject);CHKERRQ(ierr);
      if (dmhasinject && !mglevels[i+1]->inject) {
        ierr = DMCreateInjection(dms[i],dms[i+1],&p);CHKERRQ(ierr);
        ierr = PCMGSetInjection(pc,i+1,p);CHKERRQ(ierr);
        ierr = MatDestroy(&p);CHKERRQ(ierr);
      }
    }

    for (i=n-2; i>-1; i--) {ierr = DMDestroy(&dms[i]);CHKERRQ(ierr);}
    ierr = PetscFree(dms);CHKERRQ(ierr);
  }

  if (pc->dm && !pc->setupcalled) {
    /* finest smoother also gets DM but it is not active, independent of whether galerkin==PC_MG_GALERKIN_EXTERNAL */
    ierr = KSPSetDM(mglevels[n-1]->smoothd,pc->dm);CHKERRQ(ierr);
    ierr = KSPSetDMActive(mglevels[n-1]->smoothd,PETSC_FALSE);CHKERRQ(ierr);
    if (mglevels[n-1]->smoothd != mglevels[n-1]->smoothu) {
      ierr = KSPSetDM(mglevels[n-1]->smoothu,pc->dm);CHKERRQ(ierr);
      ierr = KSPSetDMActive(mglevels[n-1]->smoothu,PETSC_FALSE);CHKERRQ(ierr);
    }
    if (mglevels[n-1]->cr) {
      ierr = KSPSetDM(mglevels[n-1]->cr,pc->dm);CHKERRQ(ierr);
      ierr = KSPSetDMActive(mglevels[n-1]->cr,PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  if (mg->galerkin < PC_MG_GALERKIN_NONE) {
    Mat       A,B;
    PetscBool doA = PETSC_FALSE,doB = PETSC_FALSE;
    MatReuse  reuse = MAT_INITIAL_MATRIX;

    if ((mg->galerkin == PC_MG_GALERKIN_PMAT) || (mg->galerkin == PC_MG_GALERKIN_BOTH)) doB = PETSC_TRUE;
    if ((mg->galerkin == PC_MG_GALERKIN_MAT) || ((mg->galerkin == PC_MG_GALERKIN_BOTH) && (dA != dB))) doA = PETSC_TRUE;
    if (pc->setupcalled) reuse = MAT_REUSE_MATRIX;
    for (i=n-2; i>-1; i--) {
      PetscCheckFalse(!mglevels[i+1]->restrct && !mglevels[i+1]->interpolate,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must provide interpolation or restriction for each MG level except level 0");
      if (!mglevels[i+1]->interpolate) {
        ierr = PCMGSetInterpolation(pc,i+1,mglevels[i+1]->restrct);CHKERRQ(ierr);
      }
      if (!mglevels[i+1]->restrct) {
        ierr = PCMGSetRestriction(pc,i+1,mglevels[i+1]->interpolate);CHKERRQ(ierr);
      }
      if (reuse == MAT_REUSE_MATRIX) {
        ierr = KSPGetOperators(mglevels[i]->smoothd,&A,&B);CHKERRQ(ierr);
      }
      if (doA) {
        ierr = MatGalerkin(mglevels[i+1]->restrct,dA,mglevels[i+1]->interpolate,reuse,1.0,&A);CHKERRQ(ierr);
      }
      if (doB) {
        ierr = MatGalerkin(mglevels[i+1]->restrct,dB,mglevels[i+1]->interpolate,reuse,1.0,&B);CHKERRQ(ierr);
      }
      /* the management of the PetscObjectReference() and PetscObjecDereference() below is rather delicate */
      if (!doA && dAeqdB) {
        if (reuse == MAT_INITIAL_MATRIX) {ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);}
        A = B;
      } else if (!doA && reuse == MAT_INITIAL_MATRIX) {
        ierr = KSPGetOperators(mglevels[i]->smoothd,&A,NULL);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
      }
      if (!doB && dAeqdB) {
        if (reuse == MAT_INITIAL_MATRIX) {ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);}
        B = A;
      } else if (!doB && reuse == MAT_INITIAL_MATRIX) {
        ierr = KSPGetOperators(mglevels[i]->smoothd,NULL,&B);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
      }
      if (reuse == MAT_INITIAL_MATRIX) {
        ierr = KSPSetOperators(mglevels[i]->smoothd,A,B);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)A);CHKERRQ(ierr);
        ierr = PetscObjectDereference((PetscObject)B);CHKERRQ(ierr);
      }
      dA = A;
      dB = B;
    }
  }

  /* Adapt interpolation matrices */
  if (mg->adaptInterpolation) {
    mg->Nc = mg->Nc < 0 ? 6 : mg->Nc; /* Default to 6 modes */
    for (i = 0; i < n; ++i) {
      ierr = PCMGComputeCoarseSpace_Internal(pc, i, mg->coarseSpaceType, mg->Nc, !i ? NULL : mglevels[i-1]->coarseSpace, &mglevels[i]->coarseSpace);CHKERRQ(ierr);
      if (i) {ierr = PCMGAdaptInterpolator_Internal(pc, i, mglevels[i-1]->smoothu, mglevels[i]->smoothu, mg->Nc, mglevels[i-1]->coarseSpace, mglevels[i]->coarseSpace);CHKERRQ(ierr);}
    }
    for (i = n-2; i > -1; --i) {
      ierr = PCMGRecomputeLevelOperators_Internal(pc, i);CHKERRQ(ierr);
    }
  }

  if (needRestricts && pc->dm) {
    for (i=n-2; i>=0; i--) {
      DM  dmfine,dmcoarse;
      Mat Restrict,Inject;
      Vec rscale;
      ierr = KSPGetDM(mglevels[i+1]->smoothd,&dmfine);CHKERRQ(ierr);
      ierr = KSPGetDM(mglevels[i]->smoothd,&dmcoarse);CHKERRQ(ierr);
      ierr = PCMGGetRestriction(pc,i+1,&Restrict);CHKERRQ(ierr);
      ierr = PCMGGetRScale(pc,i+1,&rscale);CHKERRQ(ierr);
      ierr = PCMGGetInjection(pc,i+1,&Inject);CHKERRQ(ierr);
      ierr = DMRestrict(dmfine,Restrict,rscale,Inject,dmcoarse);CHKERRQ(ierr);
    }
  }

  if (!pc->setupcalled) {
    for (i=0; i<n; i++) {
      ierr = KSPSetFromOptions(mglevels[i]->smoothd);CHKERRQ(ierr);
    }
    for (i=1; i<n; i++) {
      if (mglevels[i]->smoothu && (mglevels[i]->smoothu != mglevels[i]->smoothd)) {
        ierr = KSPSetFromOptions(mglevels[i]->smoothu);CHKERRQ(ierr);
      }
      if (mglevels[i]->cr) {
        ierr = KSPSetFromOptions(mglevels[i]->cr);CHKERRQ(ierr);
      }
    }
    /* insure that if either interpolation or restriction is set the other other one is set */
    for (i=1; i<n; i++) {
      ierr = PCMGGetInterpolation(pc,i,NULL);CHKERRQ(ierr);
      ierr = PCMGGetRestriction(pc,i,NULL);CHKERRQ(ierr);
    }
    for (i=0; i<n-1; i++) {
      if (!mglevels[i]->b) {
        Vec *vec;
        ierr = KSPCreateVecs(mglevels[i]->smoothd,1,&vec,0,NULL);CHKERRQ(ierr);
        ierr = PCMGSetRhs(pc,i,*vec);CHKERRQ(ierr);
        ierr = VecDestroy(vec);CHKERRQ(ierr);
        ierr = PetscFree(vec);CHKERRQ(ierr);
      }
      if (!mglevels[i]->r && i) {
        ierr = VecDuplicate(mglevels[i]->b,&tvec);CHKERRQ(ierr);
        ierr = PCMGSetR(pc,i,tvec);CHKERRQ(ierr);
        ierr = VecDestroy(&tvec);CHKERRQ(ierr);
      }
      if (!mglevels[i]->x) {
        ierr = VecDuplicate(mglevels[i]->b,&tvec);CHKERRQ(ierr);
        ierr = PCMGSetX(pc,i,tvec);CHKERRQ(ierr);
        ierr = VecDestroy(&tvec);CHKERRQ(ierr);
      }
      if (doCR) {
        ierr = VecDuplicate(mglevels[i]->b,&mglevels[i]->crx);CHKERRQ(ierr);
        ierr = VecDuplicate(mglevels[i]->b,&mglevels[i]->crb);CHKERRQ(ierr);
        ierr = VecZeroEntries(mglevels[i]->crb);CHKERRQ(ierr);
      }
    }
    if (n != 1 && !mglevels[n-1]->r) {
      /* PCMGSetR() on the finest level if user did not supply it */
      Vec *vec;
      ierr = KSPCreateVecs(mglevels[n-1]->smoothd,1,&vec,0,NULL);CHKERRQ(ierr);
      ierr = PCMGSetR(pc,n-1,*vec);CHKERRQ(ierr);
      ierr = VecDestroy(vec);CHKERRQ(ierr);
      ierr = PetscFree(vec);CHKERRQ(ierr);
    }
    if (doCR) {
      ierr = VecDuplicate(mglevels[n-1]->r, &mglevels[n-1]->crx);CHKERRQ(ierr);
      ierr = VecDuplicate(mglevels[n-1]->r, &mglevels[n-1]->crb);CHKERRQ(ierr);
      ierr = VecZeroEntries(mglevels[n-1]->crb);CHKERRQ(ierr);
    }
  }

  if (pc->dm) {
    /* need to tell all the coarser levels to rebuild the matrix using the DM for that level */
    for (i=0; i<n-1; i++) {
      if (mglevels[i]->smoothd->setupstage != KSP_SETUP_NEW) mglevels[i]->smoothd->setupstage = KSP_SETUP_NEWMATRIX;
    }
  }

  for (i=1; i<n; i++) {
    if (mglevels[i]->smoothu == mglevels[i]->smoothd || mg->am == PC_MG_FULL || mg->am == PC_MG_KASKADE || mg->cyclesperpcapply > 1) {
      /* if doing only down then initial guess is zero */
      ierr = KSPSetInitialGuessNonzero(mglevels[i]->smoothd,PETSC_TRUE);CHKERRQ(ierr);
    }
    if (mglevels[i]->cr) {ierr = KSPSetInitialGuessNonzero(mglevels[i]->cr,PETSC_TRUE);CHKERRQ(ierr);}
    if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventBegin(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
    ierr = KSPSetUp(mglevels[i]->smoothd);CHKERRQ(ierr);
    if (mglevels[i]->smoothd->reason == KSP_DIVERGED_PC_FAILED) {
      pc->failedreason = PC_SUBPC_ERROR;
    }
    if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventEnd(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
    if (!mglevels[i]->residual) {
      Mat mat;
      ierr = KSPGetOperators(mglevels[i]->smoothd,&mat,NULL);CHKERRQ(ierr);
      ierr = PCMGSetResidual(pc,i,PCMGResidualDefault,mat);CHKERRQ(ierr);
    }
    if (!mglevels[i]->residualtranspose) {
      Mat mat;
      ierr = KSPGetOperators(mglevels[i]->smoothd,&mat,NULL);CHKERRQ(ierr);
      ierr = PCMGSetResidualTranspose(pc,i,PCMGResidualTransposeDefault,mat);CHKERRQ(ierr);
    }
  }
  for (i=1; i<n; i++) {
    if (mglevels[i]->smoothu && mglevels[i]->smoothu != mglevels[i]->smoothd) {
      Mat downmat,downpmat;

      /* check if operators have been set for up, if not use down operators to set them */
      ierr = KSPGetOperatorsSet(mglevels[i]->smoothu,&opsset,NULL);CHKERRQ(ierr);
      if (!opsset) {
        ierr = KSPGetOperators(mglevels[i]->smoothd,&downmat,&downpmat);CHKERRQ(ierr);
        ierr = KSPSetOperators(mglevels[i]->smoothu,downmat,downpmat);CHKERRQ(ierr);
      }

      ierr = KSPSetInitialGuessNonzero(mglevels[i]->smoothu,PETSC_TRUE);CHKERRQ(ierr);
      if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventBegin(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
      ierr = KSPSetUp(mglevels[i]->smoothu);CHKERRQ(ierr);
      if (mglevels[i]->smoothu->reason == KSP_DIVERGED_PC_FAILED) {
        pc->failedreason = PC_SUBPC_ERROR;
      }
      if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventEnd(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
    }
    if (mglevels[i]->cr) {
      Mat downmat,downpmat;

      /* check if operators have been set for up, if not use down operators to set them */
      ierr = KSPGetOperatorsSet(mglevels[i]->cr,&opsset,NULL);CHKERRQ(ierr);
      if (!opsset) {
        ierr = KSPGetOperators(mglevels[i]->smoothd,&downmat,&downpmat);CHKERRQ(ierr);
        ierr = KSPSetOperators(mglevels[i]->cr,downmat,downpmat);CHKERRQ(ierr);
      }

      ierr = KSPSetInitialGuessNonzero(mglevels[i]->cr,PETSC_TRUE);CHKERRQ(ierr);
      if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventBegin(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
      ierr = KSPSetUp(mglevels[i]->cr);CHKERRQ(ierr);
      if (mglevels[i]->cr->reason == KSP_DIVERGED_PC_FAILED) {
        pc->failedreason = PC_SUBPC_ERROR;
      }
      if (mglevels[i]->eventsmoothsetup) {ierr = PetscLogEventEnd(mglevels[i]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
    }
  }

  if (mglevels[0]->eventsmoothsetup) {ierr = PetscLogEventBegin(mglevels[0]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}
  ierr = KSPSetUp(mglevels[0]->smoothd);CHKERRQ(ierr);
  if (mglevels[0]->smoothd->reason == KSP_DIVERGED_PC_FAILED) {
    pc->failedreason = PC_SUBPC_ERROR;
  }
  if (mglevels[0]->eventsmoothsetup) {ierr = PetscLogEventEnd(mglevels[0]->eventsmoothsetup,0,0,0,0);CHKERRQ(ierr);}

  /*
     Dump the interpolation/restriction matrices plus the
   Jacobian/stiffness on each level. This allows MATLAB users to
   easily check if the Galerkin condition A_c = R A_f R^T is satisfied.

   Only support one or the other at the same time.
  */
#if defined(PETSC_USE_SOCKET_VIEWER)
  ierr = PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_mg_dump_matlab",&dump,NULL);CHKERRQ(ierr);
  if (dump) viewer = PETSC_VIEWER_SOCKET_(PetscObjectComm((PetscObject)pc));
  dump = PETSC_FALSE;
#endif
  ierr = PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_mg_dump_binary",&dump,NULL);CHKERRQ(ierr);
  if (dump) viewer = PETSC_VIEWER_BINARY_(PetscObjectComm((PetscObject)pc));

  if (viewer) {
    for (i=1; i<n; i++) {
      ierr = MatView(mglevels[i]->restrct,viewer);CHKERRQ(ierr);
    }
    for (i=0; i<n; i++) {
      ierr = KSPGetPC(mglevels[i]->smoothd,&pc);CHKERRQ(ierr);
      ierr = MatView(pc->mat,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/

PetscErrorCode PCMGGetLevels_MG(PC pc, PetscInt *levels)
{
  PC_MG *mg = (PC_MG *) pc->data;

  PetscFunctionBegin;
  *levels = mg->nlevels;
  PetscFunctionReturn(0);
}

/*@
   PCMGGetLevels - Gets the number of levels to use with MG.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output parameter:
.  levels - the number of levels

   Level: advanced

.seealso: PCMGSetLevels()
@*/
PetscErrorCode PCMGGetLevels(PC pc,PetscInt *levels)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(levels,2);
  *levels = 0;
  ierr = PetscTryMethod(pc,"PCMGGetLevels_C",(PC,PetscInt*),(pc,levels));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCMGGetGridComplexity - compute operator and grid complexity of MG hierarchy

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  gc - grid complexity = sum_i(n_i) / n_0
-  oc - operator complexity = sum_i(nnz_i) / nnz_0

   Level: advanced

.seealso: PCMGGetLevels()
@*/
PetscErrorCode PCMGGetGridComplexity(PC pc, PetscReal *gc, PetscReal *oc)
{
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscInt       lev,N;
  PetscLogDouble nnz0 = 0, sgc = 0, soc = 0, n0 = 0;
  MatInfo        info;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (gc) PetscValidRealPointer(gc,2);
  if (oc) PetscValidRealPointer(oc,3);
  if (!pc->setupcalled) {
    if (gc) *gc = 0;
    if (oc) *oc = 0;
    PetscFunctionReturn(0);
  }
  PetscCheck(mg->nlevels > 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MG has no levels");
  for (lev=0; lev<mg->nlevels; lev++) {
    Mat dB;
    ierr = KSPGetOperators(mglevels[lev]->smoothd,NULL,&dB);CHKERRQ(ierr);
    ierr = MatGetInfo(dB,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr); /* global reduction */
    ierr = MatGetSize(dB,&N,NULL);CHKERRQ(ierr);
    sgc += N;
    soc += info.nz_used;
    if (lev==mg->nlevels-1) {nnz0 = info.nz_used; n0 = N;}
  }
  if (n0 > 0 && gc) *gc = (PetscReal)(sgc/n0);
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number for grid points on finest level is not available");
  if (nnz0 > 0 && oc) *oc = (PetscReal)(soc/nnz0);
  PetscFunctionReturn(0);
}

/*@
   PCMGSetType - Determines the form of multigrid to use:
   multiplicative, additive, full, or the Kaskade algorithm.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  form - multigrid form, one of PC_MG_MULTIPLICATIVE, PC_MG_ADDITIVE,
   PC_MG_FULL, PC_MG_KASKADE

   Options Database Key:
.  -pc_mg_type <form> - Sets <form>, one of multiplicative,
   additive, full, kaskade

   Level: advanced

.seealso: PCMGSetLevels()
@*/
PetscErrorCode  PCMGSetType(PC pc,PCMGType form)
{
  PC_MG *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,form,2);
  mg->am = form;
  if (form == PC_MG_MULTIPLICATIVE) pc->ops->applyrichardson = PCApplyRichardson_MG;
  else pc->ops->applyrichardson = NULL;
  PetscFunctionReturn(0);
}

/*@
   PCMGGetType - Determines the form of multigrid to use:
   multiplicative, additive, full, or the Kaskade algorithm.

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  type - one of PC_MG_MULTIPLICATIVE, PC_MG_ADDITIVE,PC_MG_FULL, PC_MG_KASKADE

   Level: advanced

.seealso: PCMGSetLevels()
@*/
PetscErrorCode  PCMGGetType(PC pc,PCMGType *type)
{
  PC_MG *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  *type = mg->am;
  PetscFunctionReturn(0);
}

/*@
   PCMGSetCycleType - Sets the type cycles to use.  Use PCMGSetCycleTypeOnLevel() for more
   complicated cycling.

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
-  n - either PC_MG_CYCLE_V or PC_MG_CYCLE_W

   Options Database Key:
.  -pc_mg_cycle_type <v,w> - provide the cycle desired

   Level: advanced

.seealso: PCMGSetCycleTypeOnLevel()
@*/
PetscErrorCode  PCMGSetCycleType(PC pc,PCMGCycleType n)
{
  PC_MG        *mg        = (PC_MG*)pc->data;
  PC_MG_Levels **mglevels = mg->levels;
  PetscInt     i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,n,2);
  PetscCheckFalse(!mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Must set MG levels with PCMGSetLevels() before calling");
  levels = mglevels[0]->levels;
  for (i=0; i<levels; i++) mglevels[i]->cycles = n;
  PetscFunctionReturn(0);
}

/*@
   PCMGMultiplicativeSetCycles - Sets the number of cycles to use for each preconditioner step
         of multigrid when PCMGType of PC_MG_MULTIPLICATIVE is used

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
-  n - number of cycles (default is 1)

   Options Database Key:
.  -pc_mg_multiplicative_cycles n - set the number of cycles

   Level: advanced

   Notes:
    This is not associated with setting a v or w cycle, that is set with PCMGSetCycleType()

.seealso: PCMGSetCycleTypeOnLevel(), PCMGSetCycleType()
@*/
PetscErrorCode  PCMGMultiplicativeSetCycles(PC pc,PetscInt n)
{
  PC_MG        *mg        = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,n,2);
  mg->cyclesperpcapply = n;
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGSetGalerkin_MG(PC pc,PCMGGalerkinType use)
{
  PC_MG *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  mg->galerkin = use;
  PetscFunctionReturn(0);
}

/*@
   PCMGSetGalerkin - Causes the coarser grid matrices to be computed from the
      finest grid via the Galerkin process: A_i-1 = r_i * A_i * p_i

   Logically Collective on PC

   Input Parameters:
+  pc - the multigrid context
-  use - one of PC_MG_GALERKIN_BOTH,PC_MG_GALERKIN_PMAT,PC_MG_GALERKIN_MAT, or PC_MG_GALERKIN_NONE

   Options Database Key:
.  -pc_mg_galerkin <both,pmat,mat,none> - set the matrices to form via the Galerkin process

   Level: intermediate

   Notes:
    Some codes that use PCMG such as PCGAMG use Galerkin internally while constructing the hierarchy and thus do not
     use the PCMG construction of the coarser grids.

.seealso: PCMGGetGalerkin(), PCMGGalerkinType

@*/
PetscErrorCode PCMGSetGalerkin(PC pc,PCMGGalerkinType use)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCMGSetGalerkin_C",(PC,PCMGGalerkinType),(pc,use));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCMGGetGalerkin - Checks if Galerkin multigrid is being used, i.e.
      A_i-1 = r_i * A_i * p_i

   Not Collective

   Input Parameter:
.  pc - the multigrid context

   Output Parameter:
.  galerkin - one of PC_MG_GALERKIN_BOTH,PC_MG_GALERKIN_PMAT,PC_MG_GALERKIN_MAT, PC_MG_GALERKIN_NONE, or PC_MG_GALERKIN_EXTERNAL

   Level: intermediate

.seealso: PCMGSetGalerkin(), PCMGGalerkinType

@*/
PetscErrorCode  PCMGGetGalerkin(PC pc,PCMGGalerkinType  *galerkin)
{
  PC_MG *mg = (PC_MG*)pc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  *galerkin = mg->galerkin;
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGSetAdaptInterpolation_MG(PC pc, PetscBool adapt)
{
  PC_MG *mg = (PC_MG *) pc->data;

  PetscFunctionBegin;
  mg->adaptInterpolation = adapt;
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGGetAdaptInterpolation_MG(PC pc, PetscBool *adapt)
{
  PC_MG *mg = (PC_MG *) pc->data;

  PetscFunctionBegin;
  *adapt = mg->adaptInterpolation;
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGSetAdaptCR_MG(PC pc, PetscBool cr)
{
  PC_MG *mg = (PC_MG *) pc->data;

  PetscFunctionBegin;
  mg->compatibleRelaxation = cr;
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGGetAdaptCR_MG(PC pc, PetscBool *cr)
{
  PC_MG *mg = (PC_MG *) pc->data;

  PetscFunctionBegin;
  *cr = mg->compatibleRelaxation;
  PetscFunctionReturn(0);
}

/*@
  PCMGSetAdaptInterpolation - Adapt the interpolator based upon a vector space which should be accurately captured by the next coarser mesh, and thus accurately interpolated.

  Logically Collective on PC

  Input Parameters:
+ pc    - the multigrid context
- adapt - flag for adaptation of the interpolator

  Options Database Keys:
+ -pc_mg_adapt_interp                     - Turn on adaptation
. -pc_mg_adapt_interp_n <int>             - The number of modes to use, should be divisible by dimension
- -pc_mg_adapt_interp_coarse_space <type> - The type of coarse space: polynomial, harmonic, eigenvector, generalized_eigenvector

  Level: intermediate

.keywords: MG, set, Galerkin
.seealso: PCMGGetAdaptInterpolation(), PCMGSetGalerkin()
@*/
PetscErrorCode PCMGSetAdaptInterpolation(PC pc, PetscBool adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscTryMethod(pc,"PCMGSetAdaptInterpolation_C",(PC,PetscBool),(pc,adapt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PCMGGetAdaptInterpolation - Get the flag to adapt the interpolator based upon a vector space which should be accurately captured by the next coarser mesh, and thus accurately interpolated.

  Not collective

  Input Parameter:
. pc    - the multigrid context

  Output Parameter:
. adapt - flag for adaptation of the interpolator

  Level: intermediate

.keywords: MG, set, Galerkin
.seealso: PCMGSetAdaptInterpolation(), PCMGSetGalerkin()
@*/
PetscErrorCode PCMGGetAdaptInterpolation(PC pc, PetscBool *adapt)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidPointer(adapt, 2);
  ierr = PetscUseMethod(pc,"PCMGGetAdaptInterpolation_C",(PC,PetscBool*),(pc,adapt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PCMGSetAdaptCR - Monitor the coarse space quality using an auxiliary solve with compatible relaxation.

  Logically Collective on PC

  Input Parameters:
+ pc - the multigrid context
- cr - flag for compatible relaxation

  Options Database Keys:
. -pc_mg_adapt_cr - Turn on compatible relaxation

  Level: intermediate

.keywords: MG, set, Galerkin
.seealso: PCMGGetAdaptCR(), PCMGSetAdaptInterpolation(), PCMGSetGalerkin()
@*/
PetscErrorCode PCMGSetAdaptCR(PC pc, PetscBool cr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscTryMethod(pc,"PCMGSetAdaptCR_C",(PC,PetscBool),(pc,cr));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PCMGGetAdaptCR - Get the flag to monitor coarse space quality using an auxiliary solve with compatible relaxation.

  Not collective

  Input Parameter:
. pc    - the multigrid context

  Output Parameter:
. cr - flag for compatible relaxaion

  Level: intermediate

.keywords: MG, set, Galerkin
.seealso: PCMGSetAdaptCR(), PCMGGetAdaptInterpolation(), PCMGSetGalerkin()
@*/
PetscErrorCode PCMGGetAdaptCR(PC pc, PetscBool *cr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidPointer(cr, 2);
  ierr = PetscUseMethod(pc,"PCMGGetAdaptCR_C",(PC,PetscBool*),(pc,cr));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCMGSetNumberSmooth - Sets the number of pre and post-smoothing steps to use
   on all levels.  Use PCMGDistinctSmoothUp() to create separate up and down smoothers if you want different numbers of
   pre- and post-smoothing steps.

   Logically Collective on PC

   Input Parameters:
+  mg - the multigrid context
-  n - the number of smoothing steps

   Options Database Key:
.  -mg_levels_ksp_max_it <n> - Sets number of pre and post-smoothing steps

   Level: advanced

   Notes:
    this does not set a value on the coarsest grid, since we assume that
    there is no separate smooth up on the coarsest grid.

.seealso: PCMGSetDistinctSmoothUp()
@*/
PetscErrorCode  PCMGSetNumberSmooth(PC pc,PetscInt n)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,levels;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,n,2);
  PetscCheckFalse(!mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Must set MG levels with PCMGSetLevels() before calling");
  levels = mglevels[0]->levels;

  for (i=1; i<levels; i++) {
    ierr = KSPSetTolerances(mglevels[i]->smoothu,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);CHKERRQ(ierr);
    ierr = KSPSetTolerances(mglevels[i]->smoothd,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,n);CHKERRQ(ierr);
    mg->default_smoothu = n;
    mg->default_smoothd = n;
  }
  PetscFunctionReturn(0);
}

/*@
   PCMGSetDistinctSmoothUp - sets the up (post) smoother to be a separate KSP from the down (pre) smoother on all levels
       and adds the suffix _up to the options name

   Logically Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_mg_distinct_smoothup <bool> - use distinct smoothing objects

   Level: advanced

   Notes:
    this does not set a value on the coarsest grid, since we assume that
    there is no separate smooth up on the coarsest grid.

.seealso: PCMGSetNumberSmooth()
@*/
PetscErrorCode  PCMGSetDistinctSmoothUp(PC pc)
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscErrorCode ierr;
  PetscInt       i,levels;
  KSP            subksp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscCheckFalse(!mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Must set MG levels with PCMGSetLevels() before calling");
  levels = mglevels[0]->levels;

  for (i=1; i<levels; i++) {
    const char *prefix = NULL;
    /* make sure smoother up and down are different */
    ierr = PCMGGetSmootherUp(pc,i,&subksp);CHKERRQ(ierr);
    ierr = KSPGetOptionsPrefix(mglevels[i]->smoothd,&prefix);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(subksp,prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(subksp,"up_");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* No new matrices are created, and the coarse operator matrices are the references to the original ones */
PetscErrorCode  PCGetInterpolations_MG(PC pc,PetscInt *num_levels,Mat *interpolations[])
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  Mat            *mat;
  PetscInt       l;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  ierr = PetscMalloc1(mg->nlevels,&mat);CHKERRQ(ierr);
  for (l=1; l< mg->nlevels; l++) {
    mat[l-1] = mglevels[l]->interpolate;
    ierr = PetscObjectReference((PetscObject)mat[l-1]);CHKERRQ(ierr);
  }
  *num_levels = mg->nlevels;
  *interpolations = mat;
  PetscFunctionReturn(0);
}

/* No new matrices are created, and the coarse operator matrices are the references to the original ones */
PetscErrorCode  PCGetCoarseOperators_MG(PC pc,PetscInt *num_levels,Mat *coarseOperators[])
{
  PC_MG          *mg        = (PC_MG*)pc->data;
  PC_MG_Levels   **mglevels = mg->levels;
  PetscInt       l;
  Mat            *mat;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!mglevels,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Must set MG levels before calling");
  ierr = PetscMalloc1(mg->nlevels,&mat);CHKERRQ(ierr);
  for (l=0; l<mg->nlevels-1; l++) {
    ierr = KSPGetOperators(mglevels[l]->smoothd,NULL,&(mat[l]));CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)mat[l]);CHKERRQ(ierr);
  }
  *num_levels = mg->nlevels;
  *coarseOperators = mat;
  PetscFunctionReturn(0);
}

/*@C
  PCMGRegisterCoarseSpaceConstructor -  Adds a method to the PCMG package for coarse space construction.

  Not collective

  Input Parameters:
+ name     - name of the constructor
- function - constructor routine

  Notes:
  Calling sequence for the routine:
$ my_csp(PC pc, PetscInt l, DM dm, KSP smooth, PetscInt Nc, const Vec initGuess[], Vec **coarseSp)
$   pc        - The PC object
$   l         - The multigrid level, 0 is the coarse level
$   dm        - The DM for this level
$   smooth    - The level smoother
$   Nc        - The size of the coarse space
$   initGuess - Basis for an initial guess for the space
$   coarseSp  - A basis for the computed coarse space

  Level: advanced

.seealso: PCMGGetCoarseSpaceConstructor(), PCRegister()
@*/
PetscErrorCode PCMGRegisterCoarseSpaceConstructor(const char name[], PetscErrorCode (*function)(PC, PetscInt, DM, KSP, PetscInt, const Vec[], Vec **))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PCMGCoarseList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PCMGGetCoarseSpaceConstructor -  Returns the given coarse space construction method.

  Not collective

  Input Parameter:
. name     - name of the constructor

  Output Parameter:
. function - constructor routine

  Notes:
  Calling sequence for the routine:
$ my_csp(PC pc, PetscInt l, DM dm, KSP smooth, PetscInt Nc, const Vec initGuess[], Vec **coarseSp)
$   pc        - The PC object
$   l         - The multigrid level, 0 is the coarse level
$   dm        - The DM for this level
$   smooth    - The level smoother
$   Nc        - The size of the coarse space
$   initGuess - Basis for an initial guess for the space
$   coarseSp  - A basis for the computed coarse space

  Level: advanced

.seealso: PCMGRegisterCoarseSpaceConstructor(), PCRegister()
@*/
PetscErrorCode PCMGGetCoarseSpaceConstructor(const char name[], PetscErrorCode (**function)(PC, PetscInt, DM, KSP, PetscInt, const Vec[], Vec **))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListFind(PCMGCoarseList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/

/*MC
   PCMG - Use multigrid preconditioning. This preconditioner requires you provide additional
    information about the coarser grid matrices and restriction/interpolation operators.

   Options Database Keys:
+  -pc_mg_levels <nlevels> - number of levels including finest
.  -pc_mg_cycle_type <v,w> - provide the cycle desired
.  -pc_mg_type <additive,multiplicative,full,kaskade> - multiplicative is the default
.  -pc_mg_log - log information about time spent on each level of the solver
.  -pc_mg_distinct_smoothup - configure up (after interpolation) and down (before restriction) smoothers separately (with different options prefixes)
.  -pc_mg_galerkin <both,pmat,mat,none> - use Galerkin process to compute coarser operators, i.e. Acoarse = R A R'
.  -pc_mg_multiplicative_cycles - number of cycles to use as the preconditioner (defaults to 1)
.  -pc_mg_dump_matlab - dumps the matrices for each level and the restriction/interpolation matrices
                        to the Socket viewer for reading from MATLAB.
-  -pc_mg_dump_binary - dumps the matrices for each level and the restriction/interpolation matrices
                        to the binary output file called binaryoutput

   Notes:
    If one uses a Krylov method such GMRES or CG as the smoother then one must use KSPFGMRES, KSPGCR, or KSPRICHARDSON as the outer Krylov method

       When run with a single level the smoother options are used on that level NOT the coarse grid solver options

       When run with KSPRICHARDSON the convergence test changes slightly if monitor is turned on. The iteration count may change slightly. This
       is because without monitoring the residual norm is computed WITHIN each multigrid cycle on the finest level after the pre-smoothing
       (because the residual has just been computed for the multigrid algorithm and is hence available for free) while with monitoring the
       residual is computed at the end of each cycle.

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMGType, PCEXOTIC, PCGAMG, PCML, PCHYPRE
           PCMGSetLevels(), PCMGGetLevels(), PCMGSetType(), PCMGSetCycleType(),
           PCMGSetDistinctSmoothUp(), PCMGGetCoarseSolve(), PCMGSetResidual(), PCMGSetInterpolation(),
           PCMGSetRestriction(), PCMGGetSmoother(), PCMGGetSmootherUp(), PCMGGetSmootherDown(),
           PCMGSetCycleTypeOnLevel(), PCMGSetRhs(), PCMGSetX(), PCMGSetR()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_MG(PC pc)
{
  PC_MG          *mg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr         = PetscNewLog(pc,&mg);CHKERRQ(ierr);
  pc->data     = mg;
  mg->nlevels  = -1;
  mg->am       = PC_MG_MULTIPLICATIVE;
  mg->galerkin = PC_MG_GALERKIN_NONE;
  mg->adaptInterpolation = PETSC_FALSE;
  mg->Nc                 = -1;
  mg->eigenvalue         = -1;

  pc->useAmat = PETSC_TRUE;

  pc->ops->apply          = PCApply_MG;
  pc->ops->applytranspose = PCApplyTranspose_MG;
  pc->ops->matapply       = PCMatApply_MG;
  pc->ops->setup          = PCSetUp_MG;
  pc->ops->reset          = PCReset_MG;
  pc->ops->destroy        = PCDestroy_MG;
  pc->ops->setfromoptions = PCSetFromOptions_MG;
  pc->ops->view           = PCView_MG;

  ierr = PetscObjectComposedDataRegister(&mg->eigenvalue);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMGSetGalerkin_C",PCMGSetGalerkin_MG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMGGetLevels_C",PCMGGetLevels_MG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMGSetLevels_C",PCMGSetLevels_MG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGetInterpolations_C",PCGetInterpolations_MG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCGetCoarseOperators_C",PCGetCoarseOperators_MG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMGSetAdaptInterpolation_C",PCMGSetAdaptInterpolation_MG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMGGetAdaptInterpolation_C",PCMGGetAdaptInterpolation_MG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMGSetAdaptCR_C",PCMGSetAdaptCR_MG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCMGGetAdaptCR_C",PCMGGetAdaptCR_MG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
