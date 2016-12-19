#include <../src/tao/matrix/lmvmmat.h>   /*I "lmvmmat.h" */
#include <petsctao.h>  /*I "petsctao.h" */
#include <petscksp.h>
#include <petsc/private/petscimpl.h>

/* This is a vile hack */
#if defined(PETSC_USE_COMPLEX)
#define VecDot VecDotRealPart
#endif

#define TaoMid(a,b,c)    (((a) < (b)) ?                    \
                           (((b) < (c)) ? (b) :            \
                             (((a) < (c)) ? (c) : (a))) :  \
                           (((a) < (c)) ? (a) :            \
                             (((b) < (c)) ? (c) : (b))))

/* These lists are used for setting options */
static const char *Scale_Table[64] = {"none","scalar","broyden"};

static const char *Rescale_Table[64] = {"none","scalar","gl"};

static const char *Limit_Table[64] = {"none","average","relative","absolute"};

/*@C
  MatCreateLMVM - Creates a limited memory matrix for lmvm algorithms.

  Collective on A

  Input Parameters:
+ comm - MPI Communicator
. n - local size of vectors
- N - global size of vectors

  Output Parameters:
. A - New LMVM matrix

  Level: developer

@*/
extern PetscErrorCode MatCreateLMVM(MPI_Comm comm, PetscInt n, PetscInt N, Mat *A)
{
  MatLMVMCtx     *ctx;
  PetscErrorCode ierr;
  PetscInt       nhistory;

  PetscFunctionBegin;
  /*  create data structure and populate with default values */
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->lm=5;
  ctx->eps=0.0;
  ctx->limitType=MatLMVM_Limit_None;
  ctx->scaleType=MatLMVM_Scale_Broyden;
  ctx->rScaleType = MatLMVM_Rescale_Scalar;
  ctx->s_alpha = 1.0;
  ctx->r_alpha = 1.0;
  ctx->r_beta = 0.5;
  ctx->mu = 1.0;
  ctx->nu = 100.0;
  
  ctx->phi = 0.125;
  
  ctx->scalar_history = 1;
  ctx->rescale_history = 1;
  
  ctx->delta_min = 1e-7;
  ctx->delta_max = 100.0;

  /*  Begin configuration */
  ierr = PetscOptionsBegin(comm,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_lmm_vectors", "vectors to use for approximation", "", ctx->lm, &ctx->lm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lmm_limit_mu", "mu limiting factor", "", ctx->mu, &ctx->mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lmm_limit_nu", "nu limiting factor", "", ctx->nu, &ctx->nu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lmm_broyden_phi", "phi factor for Broyden scaling", "", ctx->phi, &ctx->phi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lmm_scalar_alpha", "alpha factor for scalar scaling", "",ctx->s_alpha, &ctx->s_alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lmm_rescale_alpha", "alpha factor for rescaling diagonal", "", ctx->r_alpha, &ctx->r_alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lmm_rescale_beta", "beta factor for rescaling diagonal", "", ctx->r_beta, &ctx->r_beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_lmm_scalar_history", "amount of history for scalar scaling", "", ctx->scalar_history, &ctx->scalar_history,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_lmm_rescale_history", "amount of history for rescaling diagonal", "", ctx->rescale_history, &ctx->rescale_history,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lmm_eps", "rejection tolerance", "", ctx->eps, &ctx->eps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_lmm_scale_type", "scale type", "", Scale_Table, MatLMVM_Scale_Types, Scale_Table[ctx->scaleType], &ctx->scaleType,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_lmm_rescale_type", "rescale type", "", Rescale_Table, MatLMVM_Rescale_Types, Rescale_Table[ctx->rScaleType], &ctx->rScaleType,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_lmm_limit_type", "limit type", "", Limit_Table, MatLMVM_Limit_Types, Limit_Table[ctx->limitType], &ctx->limitType,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lmm_delta_min", "minimum delta value", "", ctx->delta_min, &ctx->delta_min,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_lmm_delta_max", "maximum delta value", "", ctx->delta_max, &ctx->delta_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /*  Complete configuration */
  ctx->rescale_history = PetscMin(ctx->rescale_history, ctx->lm);
  ierr = PetscMalloc1(ctx->lm+1,&ctx->rho);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->lm+1,&ctx->beta);CHKERRQ(ierr);

  nhistory = PetscMax(ctx->scalar_history,1);
  ierr = PetscMalloc1(nhistory,&ctx->yy_history);CHKERRQ(ierr);
  ierr = PetscMalloc1(nhistory,&ctx->ys_history);CHKERRQ(ierr);
  ierr = PetscMalloc1(nhistory,&ctx->ss_history);CHKERRQ(ierr);

  nhistory = PetscMax(ctx->rescale_history,1);
  ierr = PetscMalloc1(nhistory,&ctx->yy_rhistory);CHKERRQ(ierr);
  ierr = PetscMalloc1(nhistory,&ctx->ys_rhistory);CHKERRQ(ierr);
  ierr = PetscMalloc1(nhistory,&ctx->ss_rhistory);CHKERRQ(ierr);

  /*  Finish initializations */
  ctx->lmnow = 0;
  ctx->iter = 0;
  ctx->nupdates = 0;
  ctx->nrejects = 0;
  ctx->delta = 1.0;

  ctx->Gprev = 0;
  ctx->Xprev = 0;

  ctx->scale = 0;
  ctx->useScale = PETSC_FALSE;

  ctx->H0_mat = 0;
  ctx->H0_ksp = 0;
  ctx->H0_norm = 0;
  ctx->useDefaultH0 = PETSC_TRUE;

  ierr = MatCreateShell(comm, n, n, N, N, ctx, A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_DESTROY,(void(*)(void))MatDestroy_LMVM);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*A,MATOP_VIEW,(void(*)(void))MatView_LMVM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMSolve(Mat A, Vec b, Vec x)
{
  PetscReal      sq, yq, dd;
  PetscInt       ll;
  PetscBool      scaled;
  MatLMVMCtx     *shell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  ierr = MatShellGetContext(A,(void**)&shell);CHKERRQ(ierr);
  if (shell->lmnow < 1) {
    shell->rho[0] = 1.0;
    shell->theta = 1.0;
  }

  ierr = VecCopy(b,x);CHKERRQ(ierr);
  for (ll = 0; ll < shell->lmnow; ++ll) {
    ierr = VecDot(x,shell->S[ll],&sq);CHKERRQ(ierr);
    shell->beta[ll] = sq * shell->rho[ll];
    ierr = VecAXPY(x,-shell->beta[ll],shell->Y[ll]);CHKERRQ(ierr);
  }

  scaled = PETSC_FALSE;
  if (!scaled && !shell->useDefaultH0 && shell->H0_mat) {
    ierr = KSPSolve(shell->H0_ksp,x,shell->U);CHKERRQ(ierr);
    ierr = VecScale(shell->U, shell->theta);CHKERRQ(ierr);
    ierr = VecDot(x,shell->U,&dd);CHKERRQ(ierr);
    if ((dd > 0.0) && !PetscIsInfOrNanReal(dd)) {
      /*  Accept Hessian solve */
      ierr = VecCopy(shell->U,x);CHKERRQ(ierr);
      scaled = PETSC_TRUE;
    }
  }

  if (!scaled && shell->useScale) {
    ierr = VecPointwiseMult(shell->U,x,shell->scale);CHKERRQ(ierr);
    ierr = VecDot(x,shell->U,&dd);CHKERRQ(ierr);
    if ((dd > 0.0) && !PetscIsInfOrNanReal(dd)) {
      /*  Accept scaling */
      ierr = VecCopy(shell->U,x);CHKERRQ(ierr);
      scaled = PETSC_TRUE;
    }
  }

  if (!scaled) {
    switch(shell->scaleType) {
    case MatLMVM_Scale_None:
      break;

    case MatLMVM_Scale_Scalar:
      ierr = VecScale(x,shell->sigma);CHKERRQ(ierr);
      break;

    case MatLMVM_Scale_Broyden:
      ierr = VecPointwiseMult(x,x,shell->D);CHKERRQ(ierr);
      break;
    }
  }
  for (ll = shell->lmnow-1; ll >= 0; --ll) {
    ierr = VecDot(x,shell->Y[ll],&yq);CHKERRQ(ierr);
    ierr = VecAXPY(x,shell->beta[ll]-yq*shell->rho[ll],shell->S[ll]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatView_LMVM(Mat A, PetscViewer pv)
{
  PetscBool      isascii;
  PetscErrorCode ierr;
  MatLMVMCtx     *lmP;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&lmP);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)pv,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(pv,"LMVM Matrix\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv," Number of vectors: %D\n",lmP->lm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv," scale type: %s\n",Scale_Table[lmP->scaleType]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv," rescale type: %s\n",Rescale_Table[lmP->rScaleType]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv," limit type: %s\n",Limit_Table[lmP->limitType]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv," updates: %D\n",lmP->nupdates);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv," rejects: %D\n",lmP->nrejects);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatDestroy_LMVM(Mat M)
{
  MatLMVMCtx     *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  if (ctx->allocated) {
    if (ctx->Xprev) {
      ierr = PetscObjectDereference((PetscObject)ctx->Xprev);CHKERRQ(ierr);
    }
    if (ctx->Gprev) {
      ierr = PetscObjectDereference((PetscObject)ctx->Gprev);CHKERRQ(ierr);
    }

    ierr = VecDestroyVecs(ctx->lm+1,&ctx->S);CHKERRQ(ierr);
    ierr = VecDestroyVecs(ctx->lm+1,&ctx->Y);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->D);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->U);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->V);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->W);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->P);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->Q);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->H0_norm);CHKERRQ(ierr);
    if (ctx->scale) {
      ierr = VecDestroy(&ctx->scale);CHKERRQ(ierr);
    }
  }
  if (ctx->H0_mat) {
    ierr = PetscObjectDereference((PetscObject)ctx->H0_mat);CHKERRQ(ierr);
    ierr = KSPDestroy(&ctx->H0_ksp);CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx->rho);CHKERRQ(ierr);
  ierr = PetscFree(ctx->beta);CHKERRQ(ierr);
  ierr = PetscFree(ctx->yy_history);CHKERRQ(ierr);
  ierr = PetscFree(ctx->ys_history);CHKERRQ(ierr);
  ierr = PetscFree(ctx->ss_history);CHKERRQ(ierr);
  ierr = PetscFree(ctx->yy_rhistory);CHKERRQ(ierr);
  ierr = PetscFree(ctx->ys_rhistory);CHKERRQ(ierr);
  ierr = PetscFree(ctx->ss_rhistory);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMReset(Mat M)
{
  PetscErrorCode ierr;
  MatLMVMCtx     *ctx;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  if (ctx->Gprev) {
    ierr = PetscObjectDereference((PetscObject)ctx->Gprev);CHKERRQ(ierr);
  }
  if (ctx->Xprev) {
    ierr = PetscObjectDereference((PetscObject)ctx->Xprev);CHKERRQ(ierr);
  }
  ctx->Gprev = ctx->Y[ctx->lm];
  ctx->Xprev = ctx->S[ctx->lm];
  ierr = PetscObjectReference((PetscObject)ctx->Gprev);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)ctx->Xprev);CHKERRQ(ierr);
  for (i=0; i<ctx->lm; ++i) {
    ctx->rho[i] = 0.0;
  }
  ctx->rho[0] = 1.0;

  /*  Set the scaling and diagonal scaling matrix */
  switch(ctx->scaleType) {
  case MatLMVM_Scale_None:
    ctx->sigma = 1.0;
    break;
  case MatLMVM_Scale_Scalar:
    ctx->sigma = ctx->delta;
    break;
  case MatLMVM_Scale_Broyden:
    ierr = VecSet(ctx->D,ctx->delta);CHKERRQ(ierr);
    break;
  }

  ctx->iter=0;
  ctx->nupdates=0;
  ctx->lmnow=0;
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMUpdate(Mat M, Vec x, Vec g)
{
  MatLMVMCtx     *ctx;
  PetscReal      rhotemp, rhotol;
  PetscReal      y0temp, s0temp;
  PetscReal      yDy, yDs, sDs;
  PetscReal      sigmanew, denom;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      same;
  PetscReal      yy_sum=0.0, ys_sum=0.0, ss_sum=0.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  ierr = PetscObjectTypeCompare((PetscObject)M,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix M is not type MatLMVM");
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  if (!ctx->allocated) {
    ierr = MatLMVMAllocateVectors(M, x); CHKERRQ(ierr);
  }

  if (0 == ctx->iter) {
    ierr = MatLMVMReset(M);CHKERRQ(ierr);
  }  else {
    ierr = VecAYPX(ctx->Gprev,-1.0,g);CHKERRQ(ierr);
    ierr = VecAYPX(ctx->Xprev,-1.0,x);CHKERRQ(ierr);

    ierr = VecDot(ctx->Gprev,ctx->Xprev,&rhotemp);CHKERRQ(ierr);
    ierr = VecDot(ctx->Gprev,ctx->Gprev,&y0temp);CHKERRQ(ierr);

    rhotol = ctx->eps * y0temp;
    if (rhotemp > rhotol) {
      ++ctx->nupdates;

      ctx->lmnow = PetscMin(ctx->lmnow+1, ctx->lm);
      ierr=PetscObjectDereference((PetscObject)ctx->S[ctx->lm]);CHKERRQ(ierr);
      ierr=PetscObjectDereference((PetscObject)ctx->Y[ctx->lm]);CHKERRQ(ierr);
      for (i = ctx->lm-1; i >= 0; --i) {
        ctx->S[i+1] = ctx->S[i];
        ctx->Y[i+1] = ctx->Y[i];
        ctx->rho[i+1] = ctx->rho[i];
      }
      ctx->S[0] = ctx->Xprev;
      ctx->Y[0] = ctx->Gprev;
      PetscObjectReference((PetscObject)ctx->S[0]);
      PetscObjectReference((PetscObject)ctx->Y[0]);
      ctx->rho[0] = 1.0 / rhotemp;

      /*  Compute the scaling */
      switch(ctx->scaleType) {
      case MatLMVM_Scale_None:
        break;

      case MatLMVM_Scale_Scalar:
        /*  Compute s^T s  */
          ierr = VecDot(ctx->Xprev,ctx->Xprev,&s0temp);CHKERRQ(ierr);

        /*  Scalar is positive; safeguards are not required. */

        /*  Save information for scalar scaling */
        ctx->yy_history[(ctx->nupdates - 1) % ctx->scalar_history] = y0temp;
        ctx->ys_history[(ctx->nupdates - 1) % ctx->scalar_history] = rhotemp;
        ctx->ss_history[(ctx->nupdates - 1) % ctx->scalar_history] = s0temp;

        /*  Compute summations for scalar scaling */
        yy_sum = 0;     /*  No safeguard required; y^T y > 0 */
        ys_sum = 0;     /*  No safeguard required; y^T s > 0 */
        ss_sum = 0;     /*  No safeguard required; s^T s > 0 */
        for (i = 0; i < PetscMin(ctx->nupdates, ctx->scalar_history); ++i) {
          yy_sum += ctx->yy_history[i];
          ys_sum += ctx->ys_history[i];
          ss_sum += ctx->ss_history[i];
        }

        if (0.0 == ctx->s_alpha) {
          /*  Safeguard ys_sum  */
          if (0.0 == ys_sum) {
            ys_sum = TAO_ZERO_SAFEGUARD;
          }

          sigmanew = ss_sum / ys_sum;
        } else if (1.0 == ctx->s_alpha) {
          /*  Safeguard yy_sum  */
          if (0.0 == yy_sum) {
            yy_sum = TAO_ZERO_SAFEGUARD;
          }

          sigmanew = ys_sum / yy_sum;
        } else {
          denom = 2*ctx->s_alpha*yy_sum;

          /*  Safeguard denom */
          if (0.0 == denom) {
            denom = TAO_ZERO_SAFEGUARD;
          }

          sigmanew = ((2*ctx->s_alpha-1)*ys_sum +  PetscSqrtReal((2*ctx->s_alpha-1)*(2*ctx->s_alpha-1)*ys_sum*ys_sum - 4*(ctx->s_alpha)*(ctx->s_alpha-1)*yy_sum*ss_sum)) / denom;
        }

        switch(ctx->limitType) {
        case MatLMVM_Limit_Average:
          if (1.0 == ctx->mu) {
            ctx->sigma = sigmanew;
          } else if (ctx->mu) {
            ctx->sigma = ctx->mu * sigmanew + (1.0 - ctx->mu) * ctx->sigma;
          }
          break;

        case MatLMVM_Limit_Relative:
          if (ctx->mu) {
            ctx->sigma = TaoMid((1.0 - ctx->mu) * ctx->sigma, sigmanew, (1.0 + ctx->mu) * ctx->sigma);
          }
          break;

        case MatLMVM_Limit_Absolute:
          if (ctx->nu) {
            ctx->sigma = TaoMid(ctx->sigma - ctx->nu, sigmanew, ctx->sigma + ctx->nu);
          }
          break;

        default:
          ctx->sigma = sigmanew;
          break;
        }
        break;

      case MatLMVM_Scale_Broyden:
        /*  Original version */
        /*  Combine DFP and BFGS */

        /*  This code appears to be numerically unstable.  We use the */
        /*  original version because this was used to generate all of */
        /*  the data and because it may be the least unstable of the */
        /*  bunch. */

        /*  P = Q = inv(D); */
        ierr = VecCopy(ctx->D,ctx->P);CHKERRQ(ierr);
        ierr = VecReciprocal(ctx->P);CHKERRQ(ierr);
        ierr = VecCopy(ctx->P,ctx->Q);CHKERRQ(ierr);

        /*  V = y*y */
        ierr = VecPointwiseMult(ctx->V,ctx->Gprev,ctx->Gprev);CHKERRQ(ierr);

        /*  W = inv(D)*s */
        ierr = VecPointwiseMult(ctx->W,ctx->Xprev,ctx->P);CHKERRQ(ierr);
        ierr = VecDot(ctx->W,ctx->Xprev,&sDs);CHKERRQ(ierr);

        /*  Safeguard rhotemp and sDs */
        if (0.0 == rhotemp) {
          rhotemp = TAO_ZERO_SAFEGUARD;
        }

        if (0.0 == sDs) {
          sDs = TAO_ZERO_SAFEGUARD;
        }

        if (1.0 != ctx->phi) {
          /*  BFGS portion of the update */
          /*  U = (inv(D)*s)*(inv(D)*s) */
          ierr = VecPointwiseMult(ctx->U,ctx->W,ctx->W);CHKERRQ(ierr);

          /*  Assemble */
          ierr = VecAXPY(ctx->P,1.0/rhotemp,ctx->V);CHKERRQ(ierr);
          ierr = VecAXPY(ctx->P,-1.0/sDs,ctx->U);CHKERRQ(ierr);
        }

        if (0.0 != ctx->phi) {
          /*  DFP portion of the update */
          /*  U = inv(D)*s*y */
          ierr = VecPointwiseMult(ctx->U, ctx->W, ctx->Gprev);CHKERRQ(ierr);

          /*  Assemble */
          ierr = VecAXPY(ctx->Q,1.0/rhotemp + sDs/(rhotemp*rhotemp), ctx->V);CHKERRQ(ierr);
          ierr = VecAXPY(ctx->Q,-2.0/rhotemp,ctx->U);CHKERRQ(ierr);
        }

        if (0.0 == ctx->phi) {
            ierr = VecCopy(ctx->P,ctx->U);CHKERRQ(ierr);
        } else if (1.0 == ctx->phi) {
            ierr = VecCopy(ctx->Q,ctx->U);CHKERRQ(ierr);
        } else {
          /*  Broyden update U=(1-phi)*P + phi*Q */
            ierr = VecCopy(ctx->Q,ctx->U);CHKERRQ(ierr);
            ierr = VecAXPBY(ctx->U,1.0-ctx->phi, ctx->phi, ctx->P);CHKERRQ(ierr);
        }

        /*  Obtain inverse and ensure positive definite */
        ierr = VecReciprocal(ctx->U);CHKERRQ(ierr);
        ierr = VecAbs(ctx->U);CHKERRQ(ierr);

        switch(ctx->rScaleType) {
        case MatLMVM_Rescale_None:
            break;

        case MatLMVM_Rescale_Scalar:
        case MatLMVM_Rescale_GL:
          if (ctx->rScaleType == MatLMVM_Rescale_GL) {
            /*  Gilbert and Lemarachal use the old diagonal */
            ierr = VecCopy(ctx->D,ctx->P);CHKERRQ(ierr);
          } else {
            /*  The default version uses the current diagonal */
              ierr = VecCopy(ctx->U,ctx->P);CHKERRQ(ierr);
          }

          /*  Compute s^T s  */
          ierr = VecDot(ctx->Xprev,ctx->Xprev,&s0temp);CHKERRQ(ierr);

          /*  Save information for special cases of scalar rescaling */
          ctx->yy_rhistory[(ctx->nupdates - 1) % ctx->rescale_history] = y0temp;
          ctx->ys_rhistory[(ctx->nupdates - 1) % ctx->rescale_history] = rhotemp;
          ctx->ss_rhistory[(ctx->nupdates - 1) % ctx->rescale_history] = s0temp;

          if (0.5 == ctx->r_beta) {
            if (1 == PetscMin(ctx->nupdates, ctx->rescale_history)) {
              ierr = VecPointwiseMult(ctx->V,ctx->Y[0],ctx->P);CHKERRQ(ierr);
              ierr = VecDot(ctx->V,ctx->Y[0],&yy_sum);CHKERRQ(ierr);

              ierr = VecPointwiseDivide(ctx->W,ctx->S[0],ctx->P);CHKERRQ(ierr);
              ierr = VecDot(ctx->W,ctx->S[0],&ss_sum);CHKERRQ(ierr);

              ys_sum = ctx->ys_rhistory[0];
            } else {
              ierr = VecCopy(ctx->P,ctx->Q);CHKERRQ(ierr);
              ierr = VecReciprocal(ctx->Q);CHKERRQ(ierr);

              /*  Compute summations for scalar scaling */
              yy_sum = 0;       /*  No safeguard required */
              ys_sum = 0;       /*  No safeguard required */
              ss_sum = 0;       /*  No safeguard required */
              for (i = 0; i < PetscMin(ctx->nupdates, ctx->rescale_history); ++i) {
                ierr = VecPointwiseMult(ctx->V,ctx->Y[i],ctx->P);CHKERRQ(ierr);
                ierr = VecDot(ctx->V,ctx->Y[i],&yDy);CHKERRQ(ierr);
                yy_sum += yDy;

                ierr = VecPointwiseMult(ctx->W,ctx->S[i],ctx->Q);CHKERRQ(ierr);
                ierr = VecDot(ctx->W,ctx->S[i],&sDs);CHKERRQ(ierr);
                ss_sum += sDs;
                ys_sum += ctx->ys_rhistory[i];
              }
            }
          } else if (0.0 == ctx->r_beta) {
            if (1 == PetscMin(ctx->nupdates, ctx->rescale_history)) {
              /*  Compute summations for scalar scaling */
              ierr = VecPointwiseDivide(ctx->W,ctx->S[0],ctx->P);CHKERRQ(ierr);

              ierr = VecDot(ctx->W, ctx->Y[0], &ys_sum);CHKERRQ(ierr);
              ierr = VecDot(ctx->W, ctx->W, &ss_sum);CHKERRQ(ierr);
              yy_sum += ctx->yy_rhistory[0];
            } else {
              ierr = VecCopy(ctx->Q, ctx->P);CHKERRQ(ierr);
              ierr = VecReciprocal(ctx->Q);CHKERRQ(ierr);

              /*  Compute summations for scalar scaling */
              yy_sum = 0;       /*  No safeguard required */
              ys_sum = 0;       /*  No safeguard required */
              ss_sum = 0;       /*  No safeguard required */
              for (i = 0; i < PetscMin(ctx->nupdates, ctx->rescale_history); ++i) {
                ierr = VecPointwiseMult(ctx->W, ctx->S[i], ctx->Q);CHKERRQ(ierr);
                ierr = VecDot(ctx->W, ctx->Y[i], &yDs);CHKERRQ(ierr);
                ys_sum += yDs;

                ierr = VecDot(ctx->W, ctx->W, &sDs);CHKERRQ(ierr);
                ss_sum += sDs;

                yy_sum += ctx->yy_rhistory[i];
              }
            }
          } else if (1.0 == ctx->r_beta) {
            /*  Compute summations for scalar scaling */
            yy_sum = 0; /*  No safeguard required */
            ys_sum = 0; /*  No safeguard required */
            ss_sum = 0; /*  No safeguard required */
            for (i = 0; i < PetscMin(ctx->nupdates, ctx->rescale_history); ++i) {
              ierr = VecPointwiseMult(ctx->V, ctx->Y[i], ctx->P);CHKERRQ(ierr);
              ierr = VecDot(ctx->V, ctx->S[i], &yDs);CHKERRQ(ierr);
              ys_sum += yDs;

              ierr = VecDot(ctx->V, ctx->V, &yDy);CHKERRQ(ierr);
              yy_sum += yDy;

              ss_sum += ctx->ss_rhistory[i];
            }
          } else {
            ierr = VecCopy(ctx->Q, ctx->P);CHKERRQ(ierr);

            ierr = VecPow(ctx->P, ctx->r_beta);CHKERRQ(ierr);
            ierr = VecPointwiseDivide(ctx->Q, ctx->P, ctx->Q);CHKERRQ(ierr);

            /*  Compute summations for scalar scaling */
            yy_sum = 0; /*  No safeguard required */
            ys_sum = 0; /*  No safeguard required */
            ss_sum = 0; /*  No safeguard required */
            for (i = 0; i < PetscMin(ctx->nupdates, ctx->rescale_history); ++i) {
              ierr = VecPointwiseMult(ctx->V, ctx->P, ctx->Y[i]);CHKERRQ(ierr);
              ierr = VecPointwiseMult(ctx->W, ctx->Q, ctx->S[i]);CHKERRQ(ierr);

              ierr = VecDot(ctx->V, ctx->V, &yDy);CHKERRQ(ierr);
              ierr = VecDot(ctx->V, ctx->W, &yDs);CHKERRQ(ierr);
              ierr = VecDot(ctx->W, ctx->W, &sDs);CHKERRQ(ierr);

              yy_sum += yDy;
              ys_sum += yDs;
              ss_sum += sDs;
            }
          }

          if (0.0 == ctx->r_alpha) {
            /*  Safeguard ys_sum  */
            if (0.0 == ys_sum) {
              ys_sum = TAO_ZERO_SAFEGUARD;
            }

            sigmanew = ss_sum / ys_sum;
          } else if (1.0 == ctx->r_alpha) {
            /*  Safeguard yy_sum  */
            if (0.0 == yy_sum) {
              ys_sum = TAO_ZERO_SAFEGUARD;
            }

            sigmanew = ys_sum / yy_sum;
          } else {
            denom = 2*ctx->r_alpha*yy_sum;

            /*  Safeguard denom */
            if (0.0 == denom) {
              denom = TAO_ZERO_SAFEGUARD;
            }

            sigmanew = ((2*ctx->r_alpha-1)*ys_sum + PetscSqrtReal((2*ctx->r_alpha-1)*(2*ctx->r_alpha-1)*ys_sum*ys_sum - 4*ctx->r_alpha*(ctx->r_alpha-1)*yy_sum*ss_sum)) / denom;
          }

          /*  If Q has small values, then Q^(r_beta - 1) */
          /*  can have very large values.  Hence, ys_sum */
          /*  and ss_sum can be infinity.  In this case, */
          /*  sigmanew can either be not-a-number or infinity. */

          if (PetscIsInfOrNanReal(sigmanew)) {
            /*  sigmanew is not-a-number; skip rescaling */
          } else if (!sigmanew) {
            /*  sigmanew is zero; this is a bad case; skip rescaling */
          } else {
            /*  sigmanew is positive */
            ierr = VecScale(ctx->U, sigmanew);CHKERRQ(ierr);
          }
          break;
        }

        /*  Modify for previous information */
        switch(ctx->limitType) {
        case MatLMVM_Limit_Average:
          if (1.0 == ctx->mu) {
            ierr = VecCopy(ctx->D, ctx->U);CHKERRQ(ierr);
          } else if (ctx->mu) {
            ierr = VecAXPBY(ctx->D,ctx->mu, 1.0-ctx->mu,ctx->U);CHKERRQ(ierr);
          }
          break;

        case MatLMVM_Limit_Relative:
          if (ctx->mu) {
            /*  P = (1-mu) * D */
            ierr = VecAXPBY(ctx->P, 1.0-ctx->mu, 0.0, ctx->D);CHKERRQ(ierr);
            /*  Q = (1+mu) * D */
            ierr = VecAXPBY(ctx->Q, 1.0+ctx->mu, 0.0, ctx->D);CHKERRQ(ierr);
            ierr = VecMedian(ctx->P, ctx->U, ctx->Q, ctx->D);CHKERRQ(ierr);
          }
          break;

        case MatLMVM_Limit_Absolute:
          if (ctx->nu) {
            ierr = VecCopy(ctx->P, ctx->D);CHKERRQ(ierr);
            ierr = VecShift(ctx->P, -ctx->nu);CHKERRQ(ierr);
            ierr = VecCopy(ctx->D, ctx->Q);CHKERRQ(ierr);
            ierr = VecShift(ctx->Q, ctx->nu);CHKERRQ(ierr);
            ierr = VecMedian(ctx->P, ctx->U, ctx->Q, ctx->P);CHKERRQ(ierr);
          }
          break;

        default:
            ierr = VecCopy(ctx->U, ctx->D);CHKERRQ(ierr);
          break;
        }
        break;
      }
      ierr = PetscObjectDereference((PetscObject)ctx->Xprev);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)ctx->Gprev);CHKERRQ(ierr);
      ctx->Xprev = ctx->S[ctx->lm];
      ctx->Gprev = ctx->Y[ctx->lm];
      ierr = PetscObjectReference((PetscObject)ctx->S[ctx->lm]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)ctx->Y[ctx->lm]);CHKERRQ(ierr);

    } else {
      ++ctx->nrejects;
    }
  }

  ++ctx->iter;
  ierr = VecCopy(x, ctx->Xprev);CHKERRQ(ierr);
  ierr = VecCopy(g, ctx->Gprev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMSetDelta(Mat m, PetscReal d)
{
  MatLMVMCtx     *ctx;
  PetscErrorCode ierr;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)m,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatLMVM");
  ierr = MatShellGetContext(m,(void**)&ctx);CHKERRQ(ierr);
  ctx->delta = PetscAbsReal(d);
  ctx->delta = PetscMax(ctx->delta_min, ctx->delta);
  ctx->delta = PetscMin(ctx->delta_max, ctx->delta);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMSetScale(Mat m, Vec s)
{
  MatLMVMCtx     *ctx;
  PetscErrorCode ierr;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)m,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatLMVM");
  ierr = MatShellGetContext(m,(void**)&ctx);CHKERRQ(ierr);

  ierr = VecDestroy(&ctx->scale);CHKERRQ(ierr);
  if (s) {
    ierr = VecDuplicate(s,&ctx->scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMGetRejects(Mat m, PetscInt *nrejects)
{
  MatLMVMCtx     *ctx;
  PetscErrorCode ierr;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)m,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatLMVM");
  ierr = MatShellGetContext(m,(void**)&ctx);CHKERRQ(ierr);
  *nrejects = ctx->nrejects;
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMSetH0(Mat m, Mat H0)
{
  MatLMVMCtx     *ctx;
  PetscErrorCode ierr;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)m,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatLMVM");
  ierr = MatShellGetContext(m,(void**)&ctx);CHKERRQ(ierr);

  ctx->H0_mat = H0;
  ierr = PetscObjectReference((PetscObject)ctx->H0_mat);CHKERRQ(ierr);

  ctx->useDefaultH0 = PETSC_FALSE;

  ierr = KSPCreate(PetscObjectComm((PetscObject)H0), &ctx->H0_ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ctx->H0_ksp, H0, H0);CHKERRQ(ierr);
  /* its options prefix and setup is handled in TaoSolve_LMVM/TaoSolve_BLMVM */
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMGetH0(Mat m, Mat *H0)
{
  MatLMVMCtx     *ctx;
  PetscErrorCode ierr;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)m,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatLMVM");

  ierr = MatShellGetContext(m,(void**)&ctx);CHKERRQ(ierr);
  *H0  = ctx->H0_mat;
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMGetH0KSP(Mat m, KSP *H0ksp)
{
  MatLMVMCtx     *ctx;
  PetscErrorCode ierr;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)m,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatLMVM");

  ierr = MatShellGetContext(m,(void**)&ctx);CHKERRQ(ierr);
  *H0ksp  = ctx->H0_ksp;
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMGetX0(Mat m, Vec x)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMSetPrev(Mat M, Vec x, Vec g)
{
  MatLMVMCtx     *ctx;
  PetscErrorCode ierr;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,3);
  ierr = PetscObjectTypeCompare((PetscObject)M,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix M is not type MatLMVM");
  ierr = MatShellGetContext(M,(void**)&ctx);CHKERRQ(ierr);
  if (ctx->nupdates == 0) {
    ierr = MatLMVMUpdate(M,x,g);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,ctx->Xprev);CHKERRQ(ierr);
    ierr = VecCopy(g,ctx->Gprev);CHKERRQ(ierr);
    /*  TODO scaling specific terms */
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMRefine(Mat coarse, Mat op, Vec fineX, Vec fineG)
{
  PetscErrorCode ierr;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(coarse,MAT_CLASSID,1);
  PetscValidHeaderSpecific(op,MAT_CLASSID,2);
  PetscValidHeaderSpecific(fineX,VEC_CLASSID,3);
  PetscValidHeaderSpecific(fineG,VEC_CLASSID,4);
  ierr = PetscObjectTypeCompare((PetscObject)coarse,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatLMVM");
  ierr = PetscObjectTypeCompare((PetscObject)op,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatLMVM");
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatLMVMAllocateVectors(Mat m, Vec v)
{
  PetscErrorCode ierr;
  MatLMVMCtx     *ctx;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAT_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)m,MATSHELL,&same);CHKERRQ(ierr);
  if (!same) SETERRQ(PETSC_COMM_SELF,1,"Matrix m is not type MatLMVM");
  ierr = MatShellGetContext(m,(void**)&ctx);CHKERRQ(ierr);

  /*  Perform allocations */
  ierr = VecDuplicateVecs(v,ctx->lm+1,&ctx->S);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(v,ctx->lm+1,&ctx->Y);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&ctx->D);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&ctx->U);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&ctx->V);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&ctx->W);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&ctx->P);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&ctx->Q);CHKERRQ(ierr);
  ierr = VecDuplicate(v,&ctx->H0_norm);CHKERRQ(ierr);
  ctx->allocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

