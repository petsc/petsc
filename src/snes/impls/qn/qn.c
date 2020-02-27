#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/
#include <petscdm.h>

#define H(i,j)  qn->dXdFmat[i*qn->m + j]

const char *const SNESQNScaleTypes[] =        {"DEFAULT","NONE","SHANNO","LINESEARCH","JACOBIAN","SNESQNScaleType","SNES_QN_SCALING_",0};
const char *const SNESQNRestartTypes[] =      {"DEFAULT","NONE","POWELL","PERIODIC","SNESQNRestartType","SNES_QN_RESTART_",0};
const char *const SNESQNTypes[] =             {"LBFGS","BROYDEN","BADBROYDEN","SNESQNType","SNES_QN_",0};

typedef struct {
  Vec               *U;                   /* Stored past states (vary from method to method) */
  Vec               *V;                   /* Stored past states (vary from method to method) */
  PetscInt          m;                    /* The number of kept previous steps */
  PetscReal         *lambda;              /* The line search history of the method */
  PetscReal         *norm;                /* norms of the steps */
  PetscScalar       *alpha, *beta;
  PetscScalar       *dXtdF, *dFtdX, *YtdX;
  PetscBool         singlereduction;      /* Aggregated reduction implementation */
  PetscScalar       *dXdFmat;             /* A matrix of values for dX_i dot dF_j */
  PetscViewer       monitor;
  PetscReal         powell_gamma;         /* Powell angle restart condition */
  PetscReal         scaling;              /* scaling of H0 */
  SNESQNType        type;                 /* the type of quasi-newton method used */
  SNESQNScaleType   scale_type;           /* the type of scaling used */
  SNESQNRestartType restart_type;         /* determine the frequency and type of restart conditions */
} SNES_QN;

PetscErrorCode SNESQNApply_Broyden(SNES snes,PetscInt it,Vec Y,Vec X,Vec Xold,Vec D)
{
  PetscErrorCode     ierr;
  SNES_QN            *qn = (SNES_QN*)snes->data;
  Vec                W   = snes->work[3];
  Vec                *U  = qn->U;
  PetscInt           m = qn->m;
  PetscInt           k,i,j,l = m;
  PetscReal          unorm,a,b;
  PetscReal          *lambda=qn->lambda;
  PetscScalar        gdot;
  PetscReal          udot;

  PetscFunctionBegin;
  if (it < m) l = it;
  if (it > 0) {
    k = (it-1)%l;
    ierr = SNESLineSearchGetLambda(snes->linesearch,&lambda[k]);CHKERRQ(ierr);
    ierr = VecCopy(Xold,U[k]);CHKERRQ(ierr);
    ierr = VecAXPY(U[k],-1.0,X);CHKERRQ(ierr);
    if (qn->monitor) {
      ierr = PetscViewerASCIIAddTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(qn->monitor, "scaling vector %D of %D by lambda: %14.12e \n",k,l,(double)lambda[k]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
    }
  }
  if (qn->scale_type == SNES_QN_SCALE_JACOBIAN) {
    ierr = KSPSolve(snes->ksp,D,W);CHKERRQ(ierr);
    SNESCheckKSPSolve(snes);
    ierr              = VecCopy(W,Y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(D,Y);CHKERRQ(ierr);
    ierr = VecScale(Y,qn->scaling);CHKERRQ(ierr);
  }

  /* inward recursion starting at the first update and working forward */
  for (i = 0; i < l-1; i++) {
    j = (it+i-l)%l;
    k = (it+i-l+1)%l;
    ierr = VecNorm(U[j],NORM_2,&unorm);CHKERRQ(ierr);
    ierr = VecDot(U[j],Y,&gdot);CHKERRQ(ierr);
    unorm *= unorm;
    udot = PetscRealPart(gdot);
    a = (lambda[j]/lambda[k]);
    b = -(1.-lambda[j]);
    a *= udot/unorm;
    b *= udot/unorm;
    ierr = VecAXPBYPCZ(Y,a,b,1.,U[k],U[j]);CHKERRQ(ierr);

    if (qn->monitor) {
      ierr = PetscViewerASCIIAddTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(qn->monitor, "using vector %D and %D, gdot: %14.12e\n",k,j,(double)PetscRealPart(gdot));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
    }
  }
  if (l > 0) {
    k = (it-1)%l;
    ierr = VecDot(U[k],Y,&gdot);CHKERRQ(ierr);
    ierr = VecNorm(U[k],NORM_2,&unorm);CHKERRQ(ierr);
    unorm *= unorm;
    udot = PetscRealPart(gdot);
    a = unorm/(unorm-lambda[k]*udot);
    b = -(1.-lambda[k])*udot/(unorm-lambda[k]*udot);
    if (qn->monitor) {
      ierr = PetscViewerASCIIAddTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(qn->monitor, "using vector %D: a: %14.12e b: %14.12e \n",k,(double)a,(double)b);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
    }
    ierr = VecAXPBY(Y,b,a,U[k]);CHKERRQ(ierr);
  }
  l = m;
  if (it+1<m)l=it+1;
  k = it%l;
  if (qn->monitor) {
    ierr = PetscViewerASCIIAddTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(qn->monitor, "setting vector %D of %D\n",k,l);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESQNApply_BadBroyden(SNES snes,PetscInt it,Vec Y,Vec X,Vec Xold,Vec D,Vec Dold)
{
  PetscErrorCode ierr;
  SNES_QN        *qn = (SNES_QN*)snes->data;
  Vec            W   = snes->work[3];
  Vec            *U  = qn->U;
  Vec            *T  = qn->V;

  /* ksp thing for Jacobian scaling */
  PetscInt           h,k,j,i;
  PetscInt           m = qn->m;
  PetscScalar        gdot,udot;
  PetscInt           l = m;

  PetscFunctionBegin;
  if (it < m) l = it;
  ierr = VecCopy(D,Y);CHKERRQ(ierr);
  if (l > 0) {
    k    = (it-1)%l;
    ierr = SNESLineSearchGetLambda(snes->linesearch,&qn->lambda[k]);CHKERRQ(ierr);
    ierr = VecCopy(Dold,U[k]);CHKERRQ(ierr);
    ierr = VecAXPY(U[k],-1.0,D);CHKERRQ(ierr);
    ierr = VecCopy(Xold,T[k]);CHKERRQ(ierr);
    ierr = VecAXPY(T[k],-1.0,X);CHKERRQ(ierr);
  }

  if (qn->scale_type == SNES_QN_SCALE_JACOBIAN) {
    ierr = KSPSolve(snes->ksp,Y,W);CHKERRQ(ierr);
    SNESCheckKSPSolve(snes);
    ierr = VecCopy(W,Y);CHKERRQ(ierr);
  } else {
    ierr = VecScale(Y,qn->scaling);CHKERRQ(ierr);
  }

  /* inward recursion starting at the first update and working forward */
  if (l > 0) {
    for (i = 0; i < l-1; i++) {
      j    = (it+i-l)%l;
      k    = (it+i-l+1)%l;
      h    = (it-1)%l;
      ierr = VecDotBegin(U[j],U[h],&gdot);CHKERRQ(ierr);
      ierr = VecDotBegin(U[j],U[j],&udot);CHKERRQ(ierr);
      ierr = VecDotEnd(U[j],U[h],&gdot);CHKERRQ(ierr);
      ierr = VecDotEnd(U[j],U[j],&udot);CHKERRQ(ierr);
      ierr = VecAXPY(Y,PetscRealPart(gdot)/PetscRealPart(udot),T[k]);CHKERRQ(ierr);
      ierr = VecAXPY(Y,-(1.-qn->lambda[k])*PetscRealPart(gdot)/PetscRealPart(udot),T[j]);CHKERRQ(ierr);
      if (qn->monitor) {
        ierr = PetscViewerASCIIAddTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(qn->monitor, "it: %D k: %D gdot: %14.12e\n", it, k, (double)PetscRealPart(gdot));CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESQNApply_LBFGS(SNES snes,PetscInt it,Vec Y,Vec X,Vec Xold,Vec D,Vec Dold)
{
  PetscErrorCode ierr;
  SNES_QN        *qn    = (SNES_QN*)snes->data;
  Vec            W      = snes->work[3];
  Vec            *dX    = qn->U;
  Vec            *dF    = qn->V;
  PetscScalar    *alpha = qn->alpha;
  PetscScalar    *beta  = qn->beta;
  PetscScalar    *dXtdF = qn->dXtdF;
  PetscScalar    *dFtdX = qn->dFtdX;
  PetscScalar    *YtdX  = qn->YtdX;

  /* ksp thing for Jacobian scaling */
  PetscInt           k,i,j,g;
  PetscInt           m = qn->m;
  PetscScalar        t;
  PetscInt           l = m;

  PetscFunctionBegin;
  if (it < m) l = it;
  ierr = VecCopy(D,Y);CHKERRQ(ierr);
  if (it > 0) {
    k    = (it - 1) % l;
    ierr = VecCopy(D,dF[k]);CHKERRQ(ierr);
    ierr = VecAXPY(dF[k], -1.0, Dold);CHKERRQ(ierr);
    ierr = VecCopy(X, dX[k]);CHKERRQ(ierr);
    ierr = VecAXPY(dX[k], -1.0, Xold);CHKERRQ(ierr);
    if (qn->singlereduction) {
      ierr = VecMDotBegin(dF[k],l,dX,dXtdF);CHKERRQ(ierr);
      ierr = VecMDotBegin(dX[k],l,dF,dFtdX);CHKERRQ(ierr);
      ierr = VecMDotBegin(Y,l,dX,YtdX);CHKERRQ(ierr);
      ierr = VecMDotEnd(dF[k],l,dX,dXtdF);CHKERRQ(ierr);
      ierr = VecMDotEnd(dX[k],l,dF,dFtdX);CHKERRQ(ierr);
      ierr = VecMDotEnd(Y,l,dX,YtdX);CHKERRQ(ierr);
      for (j = 0; j < l; j++) {
        H(k, j) = dFtdX[j];
        H(j, k) = dXtdF[j];
      }
      /* copy back over to make the computation of alpha and beta easier */
      for (j = 0; j < l; j++) dXtdF[j] = H(j, j);
    } else {
      ierr = VecDot(dX[k], dF[k], &dXtdF[k]);CHKERRQ(ierr);
    }
    if (qn->scale_type == SNES_QN_SCALE_LINESEARCH) {
      ierr = SNESLineSearchGetLambda(snes->linesearch,&qn->scaling);CHKERRQ(ierr);
    }
  }

  /* outward recursion starting at iteration k's update and working back */
  for (i=0; i<l; i++) {
    k = (it-i-1)%l;
    if (qn->singlereduction) {
      /* construct t = dX[k] dot Y as Y_0 dot dX[k] + sum(-alpha[j]dX[k]dF[j]) */
      t = YtdX[k];
      for (j=0; j<i; j++) {
        g  = (it-j-1)%l;
        t -= alpha[g]*H(k, g);
      }
      alpha[k] = t/H(k,k);
    } else {
      ierr     = VecDot(dX[k],Y,&t);CHKERRQ(ierr);
      alpha[k] = t/dXtdF[k];
    }
    if (qn->monitor) {
      ierr = PetscViewerASCIIAddTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(qn->monitor, "it: %D k: %D alpha:        %14.12e\n", it, k, (double)PetscRealPart(alpha[k]));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
    }
    ierr = VecAXPY(Y,-alpha[k],dF[k]);CHKERRQ(ierr);
  }

  if (qn->scale_type == SNES_QN_SCALE_JACOBIAN) {
    ierr = KSPSolve(snes->ksp,Y,W);CHKERRQ(ierr);
    SNESCheckKSPSolve(snes);
    ierr              = VecCopy(W, Y);CHKERRQ(ierr);
  } else {
    ierr = VecScale(Y, qn->scaling);CHKERRQ(ierr);
  }
  if (qn->singlereduction) {
    ierr = VecMDot(Y,l,dF,YtdX);CHKERRQ(ierr);
  }
  /* inward recursion starting at the first update and working forward */
  for (i = 0; i < l; i++) {
    k = (it + i - l) % l;
    if (qn->singlereduction) {
      t = YtdX[k];
      for (j = 0; j < i; j++) {
        g  = (it + j - l) % l;
        t += (alpha[g] - beta[g])*H(g, k);
      }
      beta[k] = t / H(k, k);
    } else {
      ierr    = VecDot(dF[k], Y, &t);CHKERRQ(ierr);
      beta[k] = t / dXtdF[k];
    }
    ierr = VecAXPY(Y, (alpha[k] - beta[k]), dX[k]);CHKERRQ(ierr);
    if (qn->monitor) {
      ierr = PetscViewerASCIIAddTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(qn->monitor, "it: %D k: %D alpha - beta: %14.12e\n", it, k, (double)PetscRealPart(alpha[k] - beta[k]));CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_QN(SNES snes)
{
  PetscErrorCode       ierr;
  SNES_QN              *qn = (SNES_QN*) snes->data;
  Vec                  X,Xold;
  Vec                  F,W;
  Vec                  Y,D,Dold;
  PetscInt             i, i_r;
  PetscReal            fnorm,xnorm,ynorm,gnorm;
  SNESLineSearchReason lssucceed;
  PetscBool            powell,periodic;
  PetscScalar          DolddotD,DolddotDold;
  SNESConvergedReason  reason;

  /* basically just a regular newton's method except for the application of the Jacobian */

  PetscFunctionBegin;
  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);

  ierr = PetscCitationsRegister(SNESCitation,&SNEScite);CHKERRQ(ierr);
  F    = snes->vec_func;                /* residual vector */
  Y    = snes->vec_sol_update;          /* search direction generated by J^-1D*/
  W    = snes->work[3];
  X    = snes->vec_sol;                 /* solution vector */
  Xold = snes->work[0];

  /* directions generated by the preconditioned problem with F_pre = F or x - M(x, b) */
  D    = snes->work[1];
  Dold = snes->work[2];

  snes->reason = SNES_CONVERGED_ITERATING;

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  if (snes->npc && snes->npcside== PC_LEFT && snes->functype == SNES_FUNCTION_PRECONDITIONED) {
    ierr = SNESApplyNPC(snes,X,NULL,F);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->npc,&reason);CHKERRQ(ierr);
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
  } else {
    if (!snes->vec_func_init_set) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    } else snes->vec_func_init_set = PETSC_FALSE;

    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
    SNESCheckFunctionNorm(snes,fnorm);
  }
  if (snes->npc && snes->npcside== PC_LEFT && snes->functype == SNES_FUNCTION_UNPRECONDITIONED) {
    ierr = SNESApplyNPC(snes,X,F,D);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->npc,&reason);CHKERRQ(ierr);
    if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  } else {
    ierr = VecCopy(F,D);CHKERRQ(ierr);
  }

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  if (snes->npc && snes->npcside== PC_RIGHT) {
    ierr = PetscLogEventBegin(SNES_NPCSolve,snes->npc,X,0,0);CHKERRQ(ierr);
    ierr = SNESSolve(snes->npc,snes->vec_rhs,X);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(SNES_NPCSolve,snes->npc,X,0,0);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->npc,&reason);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
    ierr = SNESGetNPCFunction(snes,F,&fnorm);CHKERRQ(ierr);
    ierr = VecCopy(F,D);CHKERRQ(ierr);
  }

  /* general purpose update */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
  }

  /* scale the initial update */
  if (qn->scale_type == SNES_QN_SCALE_JACOBIAN) {
    ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    SNESCheckJacobianDomainerror(snes);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
  }

  for (i = 0, i_r = 0; i < snes->max_its; i++, i_r++) {
    if (qn->scale_type == SNES_QN_SCALE_SHANNO && i_r > 0) {
      PetscScalar ff,xf;
      ierr = VecCopy(Dold,Y);CHKERRQ(ierr);
      ierr = VecCopy(Xold,W);CHKERRQ(ierr);
      ierr = VecAXPY(Y,-1.0,D);CHKERRQ(ierr);
      ierr = VecAXPY(W,-1.0,X);CHKERRQ(ierr);
      ierr = VecDotBegin(Y,Y,&ff);CHKERRQ(ierr);
      ierr = VecDotBegin(W,Y,&xf);CHKERRQ(ierr);
      ierr = VecDotEnd(Y,Y,&ff);CHKERRQ(ierr);
      ierr = VecDotEnd(W,Y,&xf);CHKERRQ(ierr);
      qn->scaling = PetscRealPart(xf)/PetscRealPart(ff);
      if (qn->monitor) {
        ierr = PetscViewerASCIIAddTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(qn->monitor, "Shanno scaling %D %g\n", i,(double)qn->scaling);CHKERRQ(ierr);
        ierr = PetscViewerASCIISubtractTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
      }
    }
    switch (qn->type) {
    case SNES_QN_BADBROYDEN:
      ierr = SNESQNApply_BadBroyden(snes,i_r,Y,X,Xold,D,Dold);CHKERRQ(ierr);
      break;
    case SNES_QN_BROYDEN:
      ierr = SNESQNApply_Broyden(snes,i_r,Y,X,Xold,D);CHKERRQ(ierr);
      break;
    case SNES_QN_LBFGS:
      ierr = SNESQNApply_LBFGS(snes,i_r,Y,X,Xold,D,Dold);CHKERRQ(ierr);
      break;
    }
    /* line search for lambda */
    ynorm = 1; gnorm = fnorm;
    ierr  = VecCopy(D, Dold);CHKERRQ(ierr);
    ierr  = VecCopy(X, Xold);CHKERRQ(ierr);
    ierr  = SNESLineSearchApply(snes->linesearch, X, F, &fnorm, Y);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    ierr = SNESLineSearchGetReason(snes->linesearch, &lssucceed);CHKERRQ(ierr);
    ierr = SNESLineSearchGetNorms(snes->linesearch, &xnorm, &fnorm, &ynorm);CHKERRQ(ierr);
    if (lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        break;
      }
    }
    if (qn->scale_type == SNES_QN_SCALE_LINESEARCH) {
      ierr = SNESLineSearchGetLambda(snes->linesearch, &qn->scaling);CHKERRQ(ierr);
    }

    /* convergence monitoring */
    ierr = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);

    if (snes->npc && snes->npcside== PC_RIGHT) {
      ierr = PetscLogEventBegin(SNES_NPCSolve,snes->npc,X,0,0);CHKERRQ(ierr);
      ierr = SNESSolve(snes->npc,snes->vec_rhs,X);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(SNES_NPCSolve,snes->npc,X,0,0);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes->npc,&reason);CHKERRQ(ierr);
      if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = SNESGetNPCFunction(snes,F,&fnorm);CHKERRQ(ierr);
    }

    ierr = SNESSetIterationNumber(snes, i+1);CHKERRQ(ierr);
    snes->norm = fnorm;
    snes->xnorm = xnorm;
    snes->ynorm = ynorm;

    ierr = SNESLogConvergenceHistory(snes,snes->norm,snes->iter);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* set parameter for default relative tolerance convergence test */
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
    if (snes->npc && snes->npcside== PC_LEFT && snes->functype == SNES_FUNCTION_UNPRECONDITIONED) {
      ierr = SNESApplyNPC(snes,X,F,D);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes->npc,&reason);CHKERRQ(ierr);
      if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
    } else {
      ierr = VecCopy(F, D);CHKERRQ(ierr);
    }

    /* general purpose update */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }

    powell = PETSC_FALSE;
    if (qn->restart_type == SNES_QN_RESTART_POWELL && i_r > 1) {
      /* check restart by Powell's Criterion: |F^T H_0 Fold| > powell_gamma * |Fold^T H_0 Fold| */
      if (qn->scale_type == SNES_QN_SCALE_JACOBIAN) {
        ierr = MatMult(snes->jacobian_pre,Dold,W);CHKERRQ(ierr);
      } else {
        ierr = VecCopy(Dold,W);CHKERRQ(ierr);
      }
      ierr = VecDotBegin(W, Dold, &DolddotDold);CHKERRQ(ierr);
      ierr = VecDotBegin(W, D, &DolddotD);CHKERRQ(ierr);
      ierr = VecDotEnd(W, Dold, &DolddotDold);CHKERRQ(ierr);
      ierr = VecDotEnd(W, D, &DolddotD);CHKERRQ(ierr);
      if (PetscAbs(PetscRealPart(DolddotD)) > qn->powell_gamma*PetscAbs(PetscRealPart(DolddotDold))) powell = PETSC_TRUE;
    }
    periodic = PETSC_FALSE;
    if (qn->restart_type == SNES_QN_RESTART_PERIODIC) {
      if (i_r>qn->m-1) periodic = PETSC_TRUE;
    }
    /* restart if either powell or periodic restart is satisfied. */
    if (powell || periodic) {
      if (qn->monitor) {
        ierr = PetscViewerASCIIAddTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
        if (powell) {
          ierr = PetscViewerASCIIPrintf(qn->monitor, "Powell restart! |%14.12e| > %6.4f*|%14.12e| i_r = %D\n", (double)PetscRealPart(DolddotD), (double)qn->powell_gamma, (double)PetscRealPart(DolddotDold),i_r);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(qn->monitor, "Periodic restart! i_r = %D\n", i_r);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISubtractTab(qn->monitor,((PetscObject)snes)->tablevel+2);CHKERRQ(ierr);
      }
      i_r = -1;
      if (qn->scale_type == SNES_QN_SCALE_JACOBIAN) {
        ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
        SNESCheckJacobianDomainerror(snes);
      }
    }
  }
  if (i == snes->max_its) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", snes->max_its);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_QN(SNES snes)
{
  SNES_QN        *qn = (SNES_QN*)snes->data;
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;

  if (!snes->vec_sol) {
    ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm,&snes->vec_sol);CHKERRQ(ierr);
  }

  ierr = VecDuplicateVecs(snes->vec_sol, qn->m, &qn->U);CHKERRQ(ierr);
  if (qn->type != SNES_QN_BROYDEN) ierr = VecDuplicateVecs(snes->vec_sol, qn->m, &qn->V);CHKERRQ(ierr);
  ierr = PetscMalloc4(qn->m,&qn->alpha,qn->m,&qn->beta,qn->m,&qn->dXtdF,qn->m,&qn->lambda);CHKERRQ(ierr);

  if (qn->singlereduction) {
    ierr = PetscMalloc3(qn->m*qn->m,&qn->dXdFmat,qn->m,&qn->dFtdX,qn->m,&qn->YtdX);CHKERRQ(ierr);
  }
  ierr = SNESSetWorkVecs(snes,4);CHKERRQ(ierr);
  /* set method defaults */
  if (qn->scale_type == SNES_QN_SCALE_DEFAULT) {
    if (qn->type == SNES_QN_BADBROYDEN) {
      qn->scale_type = SNES_QN_SCALE_NONE;
    } else {
      qn->scale_type = SNES_QN_SCALE_SHANNO;
    }
  }
  if (qn->restart_type == SNES_QN_RESTART_DEFAULT) {
    if (qn->type == SNES_QN_LBFGS) {
      qn->restart_type = SNES_QN_RESTART_POWELL;
    } else {
      qn->restart_type = SNES_QN_RESTART_PERIODIC;
    }
  }

  if (qn->scale_type == SNES_QN_SCALE_JACOBIAN) {
    ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  }
  if (snes->npcside== PC_LEFT && snes->functype == SNES_FUNCTION_DEFAULT) {snes->functype = SNES_FUNCTION_UNPRECONDITIONED;}
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESReset_QN(SNES snes)
{
  PetscErrorCode ierr;
  SNES_QN        *qn;

  PetscFunctionBegin;
  if (snes->data) {
    qn = (SNES_QN*)snes->data;
    if (qn->U) {
      ierr = VecDestroyVecs(qn->m, &qn->U);CHKERRQ(ierr);
    }
    if (qn->V) {
      ierr = VecDestroyVecs(qn->m, &qn->V);CHKERRQ(ierr);
    }
    if (qn->singlereduction) {
      ierr = PetscFree3(qn->dXdFmat, qn->dFtdX, qn->YtdX);CHKERRQ(ierr);
    }
    ierr = PetscFree4(qn->alpha,qn->beta,qn->dXtdF,qn->lambda);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_QN(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_QN(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_QN(PetscOptionItems *PetscOptionsObject,SNES snes)
{

  PetscErrorCode    ierr;
  SNES_QN           *qn    = (SNES_QN*)snes->data;
  PetscBool         monflg = PETSC_FALSE,flg;
  SNESLineSearch    linesearch;
  SNESQNRestartType rtype = qn->restart_type;
  SNESQNScaleType   stype = qn->scale_type;
  SNESQNType        qtype = qn->type;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SNES QN options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_qn_m","Number of past states saved for L-BFGS methods","SNESQN",qn->m,&qn->m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_qn_powell_gamma","Powell angle tolerance",          "SNESQN", qn->powell_gamma, &qn->powell_gamma, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_qn_monitor",         "Monitor for the QN methods",      "SNESQN", monflg, &monflg, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_qn_single_reduction", "Aggregate reductions",           "SNESQN", qn->singlereduction, &qn->singlereduction, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-snes_qn_scale_type","Scaling type","SNESQNSetScaleType",SNESQNScaleTypes,(PetscEnum)stype,(PetscEnum*)&stype,&flg);CHKERRQ(ierr);
  if (flg) ierr = SNESQNSetScaleType(snes,stype);CHKERRQ(ierr);

  ierr = PetscOptionsEnum("-snes_qn_restart_type","Restart type","SNESQNSetRestartType",SNESQNRestartTypes,(PetscEnum)rtype,(PetscEnum*)&rtype,&flg);CHKERRQ(ierr);
  if (flg) ierr = SNESQNSetRestartType(snes,rtype);CHKERRQ(ierr);

  ierr = PetscOptionsEnum("-snes_qn_type","Quasi-Newton update type","",SNESQNTypes,(PetscEnum)qtype,(PetscEnum*)&qtype,&flg);CHKERRQ(ierr);
  if (flg) {ierr = SNESQNSetType(snes,qtype);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (!snes->linesearch) {
    ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
    if (qn->type == SNES_QN_LBFGS) {
      ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHCP);CHKERRQ(ierr);
    } else if (qn->type == SNES_QN_BROYDEN) {
      ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC);CHKERRQ(ierr);
    } else {
      ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHL2);CHKERRQ(ierr);
    }
  }
  if (monflg) {
    qn->monitor = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)snes));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_QN(SNES snes, PetscViewer viewer)
{
  SNES_QN        *qn    = (SNES_QN*)snes->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  type is %s, restart type is %s, scale type is %s\n",SNESQNTypes[qn->type],SNESQNRestartTypes[qn->restart_type],SNESQNScaleTypes[qn->scale_type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Stored subspace size: %D\n", qn->m);CHKERRQ(ierr);
    if (qn->singlereduction) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using the single reduction variant.\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
    SNESQNSetRestartType - Sets the restart type for SNESQN.

    Logically Collective on SNES

    Input Parameters:
+   snes - the iterative context
-   rtype - restart type

    Options Database:
+   -snes_qn_restart_type <powell,periodic,none> - set the restart type
-   -snes_qn_m <m> - sets the number of stored updates and the restart period for periodic

    Level: intermediate

    SNESQNRestartTypes:
+   SNES_QN_RESTART_NONE - never restart
.   SNES_QN_RESTART_POWELL - restart based upon descent criteria
-   SNES_QN_RESTART_PERIODIC - restart after a fixed number of iterations

@*/
PetscErrorCode SNESQNSetRestartType(SNES snes, SNESQNRestartType rtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESQNSetRestartType_C",(SNES,SNESQNRestartType),(snes,rtype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    SNESQNSetScaleType - Sets the scaling type for the inner inverse Jacobian in SNESQN.

    Logically Collective on SNES

    Input Parameters:
+   snes - the iterative context
-   stype - scale type

    Options Database:
.   -snes_qn_scale_type <shanno,none,linesearch,jacobian>

    Level: intermediate

    SNESQNScaleTypes:
+   SNES_QN_SCALE_NONE - don't scale the problem
.   SNES_QN_SCALE_SHANNO - use shanno scaling
.   SNES_QN_SCALE_LINESEARCH - scale based upon line search lambda
-   SNES_QN_SCALE_JACOBIAN - scale by solving a linear system coming from the Jacobian you provided with SNESSetJacobian() computed at the first iteration
                             of QN and at ever restart.

.seealso: SNES, SNESQN, SNESLineSearch, SNESQNScaleType, SNESetJacobian()
@*/

PetscErrorCode SNESQNSetScaleType(SNES snes, SNESQNScaleType stype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESQNSetScaleType_C",(SNES,SNESQNScaleType),(snes,stype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESQNSetScaleType_QN(SNES snes, SNESQNScaleType stype)
{
  SNES_QN *qn = (SNES_QN*)snes->data;

  PetscFunctionBegin;
  qn->scale_type = stype;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESQNSetRestartType_QN(SNES snes, SNESQNRestartType rtype)
{
  SNES_QN *qn = (SNES_QN*)snes->data;

  PetscFunctionBegin;
  qn->restart_type = rtype;
  PetscFunctionReturn(0);
}

/*@
    SNESQNSetType - Sets the quasi-Newton variant to be used in SNESQN.

    Logically Collective on SNES

    Input Parameters:
+   snes - the iterative context
-   qtype - variant type

    Options Database:
.   -snes_qn_type <lbfgs,broyden,badbroyden>

    Level: beginner

    SNESQNTypes:
+   SNES_QN_LBFGS - LBFGS variant
.   SNES_QN_BROYDEN - Broyden variant
-   SNES_QN_BADBROYDEN - Bad Broyden variant

@*/

PetscErrorCode SNESQNSetType(SNES snes, SNESQNType qtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESQNSetType_C",(SNES,SNESQNType),(snes,qtype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESQNSetType_QN(SNES snes, SNESQNType qtype)
{
  SNES_QN *qn = (SNES_QN*)snes->data;

  PetscFunctionBegin;
  qn->type = qtype;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESQN - Limited-Memory Quasi-Newton methods for the solution of nonlinear systems.

      Options Database:

+     -snes_qn_m <m> - Number of past states saved for the L-Broyden methods.
+     -snes_qn_restart_type <powell,periodic,none> - set the restart type
.     -snes_qn_powell_gamma - Angle condition for restart.
.     -snes_qn_powell_descent - Descent condition for restart.
.     -snes_qn_type <lbfgs,broyden,badbroyden> - QN type
.     -snes_qn_scale_type <shanno,none,linesearch,jacobian> - scaling performed on inner Jacobian
.     -snes_linesearch_type <cp, l2, basic> - Type of line search.
-     -snes_qn_monitor - Monitors the quasi-newton Jacobian.

      Notes:
    This implements the L-BFGS, Broyden, and "Bad" Broyden algorithms for the solution of F(x) = b using
      previous change in F(x) and x to form the approximate inverse Jacobian using a series of multiplicative rank-one
      updates.

      When using a nonlinear preconditioner, one has two options as to how the preconditioner is applied.  The first of
      these options, sequential, uses the preconditioner to generate a new solution and function and uses those at this
      iteration as the current iteration's values when constructing the approximate Jacobian.  The second, composed,
      perturbs the problem the Jacobian represents to be P(x, b) - x = 0, where P(x, b) is the preconditioner.

      Uses left nonlinear preconditioning by default.

      References:
+   1. -   Kelley, C.T., Iterative Methods for Linear and Nonlinear Equations, Chapter 8, SIAM, 1995.
.   2. -   R. Byrd, J. Nocedal, R. Schnabel, Representations of Quasi Newton Matrices and their use in Limited Memory Methods,
      Technical Report, Northwestern University, June 1992.
.   3. -   Peter N. Brown, Alan C. Hindmarsh, Homer F. Walker, Experiments with Quasi-Newton Methods in Solving Stiff ODE
      Systems, SIAM J. Sci. Stat. Comput. Vol 6(2), April 1985.
-   4. -   Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers",
       SIAM Review, 57(4), 2015

      Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESNEWTONLS, SNESNEWTONTR

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_QN(SNES snes)
{
  PetscErrorCode ierr;
  SNES_QN        *qn;

  PetscFunctionBegin;
  snes->ops->setup          = SNESSetUp_QN;
  snes->ops->solve          = SNESSolve_QN;
  snes->ops->destroy        = SNESDestroy_QN;
  snes->ops->setfromoptions = SNESSetFromOptions_QN;
  snes->ops->view           = SNESView_QN;
  snes->ops->reset          = SNESReset_QN;

  snes->npcside= PC_LEFT;

  snes->usesnpc = PETSC_TRUE;
  snes->usesksp = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_TRUE;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
  }

  ierr                = PetscNewLog(snes,&qn);CHKERRQ(ierr);
  snes->data          = (void*) qn;
  qn->m               = 10;
  qn->scaling         = 1.0;
  qn->U               = NULL;
  qn->V               = NULL;
  qn->dXtdF           = NULL;
  qn->dFtdX           = NULL;
  qn->dXdFmat         = NULL;
  qn->monitor         = NULL;
  qn->singlereduction = PETSC_TRUE;
  qn->powell_gamma    = 0.9999;
  qn->scale_type      = SNES_QN_SCALE_DEFAULT;
  qn->restart_type    = SNES_QN_RESTART_DEFAULT;
  qn->type            = SNES_QN_LBFGS;

  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESQNSetScaleType_C",SNESQNSetScaleType_QN);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESQNSetRestartType_C",SNESQNSetRestartType_QN);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESQNSetType_C",SNESQNSetType_QN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
