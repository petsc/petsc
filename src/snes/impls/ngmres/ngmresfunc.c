#include <../src/snes/impls/ngmres/snesngmres.h> /*I "petscsnes.h" I*/
#include <petscblaslapack.h>

PetscErrorCode SNESNGMRESUpdateSubspace_Private(SNES snes,PetscInt ivec,PetscInt l,Vec F,PetscReal fnorm,Vec X)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  Vec            *Fdot   = ngmres->Fdot;
  Vec            *Xdot   = ngmres->Xdot;

  PetscFunctionBegin;
  PetscCheckFalse(ivec > l,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Cannot update vector %D with space size %D!",ivec,l);
  PetscCall(VecCopy(F,Fdot[ivec]));
  PetscCall(VecCopy(X,Xdot[ivec]));

  ngmres->fnorms[ivec] = fnorm;
  PetscFunctionReturn(0);
}

PetscErrorCode SNESNGMRESFormCombinedSolution_Private(SNES snes,PetscInt ivec,PetscInt l,Vec XM,Vec FM,PetscReal fMnorm,Vec X,Vec XA,Vec FA)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscInt       i,j;
  Vec            *Fdot      = ngmres->Fdot;
  Vec            *Xdot      = ngmres->Xdot;
  PetscScalar    *beta      = ngmres->beta;
  PetscScalar    *xi        = ngmres->xi;
  PetscScalar    alph_total = 0.;
  PetscReal      nu;
  Vec            Y = snes->work[2];
  PetscBool      changed_y,changed_w;

  PetscFunctionBegin;
  nu = fMnorm*fMnorm;

  /* construct the right hand side and xi factors */
  if (l > 0) {
    PetscCall(VecMDotBegin(FM,l,Fdot,xi));
    PetscCall(VecMDotBegin(Fdot[ivec],l,Fdot,beta));
    PetscCall(VecMDotEnd(FM,l,Fdot,xi));
    PetscCall(VecMDotEnd(Fdot[ivec],l,Fdot,beta));
    for (i = 0; i < l; i++) {
      Q(i,ivec) = beta[i];
      Q(ivec,i) = beta[i];
    }
  } else {
    Q(0,0) = ngmres->fnorms[ivec]*ngmres->fnorms[ivec];
  }

  for (i = 0; i < l; i++) beta[i] = nu - xi[i];

  /* construct h */
  for (j = 0; j < l; j++) {
    for (i = 0; i < l; i++) {
      H(i,j) = Q(i,j)-xi[i]-xi[j]+nu;
    }
  }
  if (l == 1) {
    /* simply set alpha[0] = beta[0] / H[0, 0] */
    if (H(0,0) != 0.) beta[0] = beta[0]/H(0,0);
    else beta[0] = 0.;
  } else {
    PetscCall(PetscBLASIntCast(l,&ngmres->m));
    PetscCall(PetscBLASIntCast(l,&ngmres->n));
    ngmres->info  = 0;
    ngmres->rcond = -1.;
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&ngmres->m,&ngmres->n,&ngmres->nrhs,ngmres->h,&ngmres->lda,ngmres->beta,&ngmres->ldb,ngmres->s,&ngmres->rcond,&ngmres->rank,ngmres->work,&ngmres->lwork,ngmres->rwork,&ngmres->info));
#else
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&ngmres->m,&ngmres->n,&ngmres->nrhs,ngmres->h,&ngmres->lda,ngmres->beta,&ngmres->ldb,ngmres->s,&ngmres->rcond,&ngmres->rank,ngmres->work,&ngmres->lwork,&ngmres->info));
#endif
    PetscCall(PetscFPTrapPop());
    PetscCheckFalse(ngmres->info < 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"Bad argument to GELSS");
    PetscCheckFalse(ngmres->info > 0,PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD failed to converge");
  }
  for (i=0; i<l; i++) {
    PetscCheckFalse(PetscIsInfOrNanScalar(beta[i]),PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD generated inconsistent output");
  }
  alph_total = 0.;
  for (i = 0; i < l; i++) alph_total += beta[i];

  PetscCall(VecCopy(XM,XA));
  PetscCall(VecScale(XA,1.-alph_total));
  PetscCall(VecMAXPY(XA,l,beta,Xdot));
  /* check the validity of the step */
  PetscCall(VecCopy(XA,Y));
  PetscCall(VecAXPY(Y,-1.0,X));
  PetscCall(SNESLineSearchPostCheck(snes->linesearch,X,Y,XA,&changed_y,&changed_w));
  if (!ngmres->approxfunc) {
    if (snes->npc && snes->npcside== PC_LEFT) {
      PetscCall(SNESApplyNPC(snes,XA,NULL,FA));
    } else {
      PetscCall(SNESComputeFunction(snes,XA,FA));
    }
  } else {
    PetscCall(VecCopy(FM,FA));
    PetscCall(VecScale(FA,1.-alph_total));
    PetscCall(VecMAXPY(FA,l,beta,Fdot));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESNGMRESNorms_Private(SNES snes,PetscInt l,Vec X,Vec F,Vec XM,Vec FM,Vec XA,Vec FA,Vec D,PetscReal *dnorm,PetscReal *dminnorm,PetscReal *xMnorm,PetscReal *fMnorm,PetscReal *yMnorm, PetscReal *xAnorm,PetscReal *fAnorm,PetscReal *yAnorm)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscReal      dcurnorm,dmin = -1.0;
  Vec            *Xdot = ngmres->Xdot;
  PetscInt       i;

  PetscFunctionBegin;
  if (xMnorm) {
    PetscCall(VecNormBegin(XM,NORM_2,xMnorm));
  }
  if (fMnorm) {
    PetscCall(VecNormBegin(FM,NORM_2,fMnorm));
  }
  if (yMnorm) {
    PetscCall(VecCopy(X,D));
    PetscCall(VecAXPY(D,-1.0,XM));
    PetscCall(VecNormBegin(D,NORM_2,yMnorm));
  }
  if (xAnorm) {
    PetscCall(VecNormBegin(XA,NORM_2,xAnorm));
  }
  if (fAnorm) {
    PetscCall(VecNormBegin(FA,NORM_2,fAnorm));
  }
  if (yAnorm) {
    PetscCall(VecCopy(X,D));
    PetscCall(VecAXPY(D,-1.0,XA));
    PetscCall(VecNormBegin(D,NORM_2,yAnorm));
  }
  if (dnorm) {
    PetscCall(VecCopy(XA,D));
    PetscCall(VecAXPY(D,-1.0,XM));
    PetscCall(VecNormBegin(D,NORM_2,dnorm));
  }
  if (dminnorm) {
    for (i=0; i<l; i++) {
      PetscCall(VecCopy(Xdot[i],D));
      PetscCall(VecAXPY(D,-1.0,XA));
      PetscCall(VecNormBegin(D,NORM_2,&ngmres->xnorms[i]));
    }
  }
  if (xMnorm) PetscCall(VecNormEnd(XM,NORM_2,xMnorm));
  if (fMnorm) PetscCall(VecNormEnd(FM,NORM_2,fMnorm));
  if (yMnorm) PetscCall(VecNormEnd(D,NORM_2,yMnorm));
  if (xAnorm) PetscCall(VecNormEnd(XA,NORM_2,xAnorm));
  if (fAnorm) PetscCall(VecNormEnd(FA,NORM_2,fAnorm));
  if (yAnorm) PetscCall(VecNormEnd(D,NORM_2,yAnorm));
  if (dnorm) PetscCall(VecNormEnd(D,NORM_2,dnorm));
  if (dminnorm) {
    for (i=0; i<l; i++) {
      PetscCall(VecNormEnd(D,NORM_2,&ngmres->xnorms[i]));
      dcurnorm = ngmres->xnorms[i];
      if ((dcurnorm < dmin) || (dmin < 0.0)) dmin = dcurnorm;
    }
    *dminnorm = dmin;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESNGMRESSelect_Private(SNES snes,PetscInt k_restart,Vec XM,Vec FM,PetscReal xMnorm,PetscReal fMnorm,PetscReal yMnorm,Vec XA,Vec FA,PetscReal xAnorm,PetscReal fAnorm,PetscReal yAnorm,PetscReal dnorm,PetscReal fminnorm,PetscReal dminnorm,Vec X,Vec F,Vec Y,PetscReal *xnorm,PetscReal *fnorm,PetscReal *ynorm)
{
  SNES_NGMRES          *ngmres = (SNES_NGMRES*) snes->data;
  SNESLineSearchReason lssucceed;
  PetscBool            selectA;

  PetscFunctionBegin;
  if (ngmres->select_type == SNES_NGMRES_SELECT_LINESEARCH) {
    /* X = X + \lambda(XA - X) */
    if (ngmres->monitor) {
      PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"||F_A||_2 = %e, ||F_M||_2 = %e\n",fAnorm,fMnorm));
    }
    PetscCall(VecCopy(FM,F));
    PetscCall(VecCopy(XM,X));
    PetscCall(VecCopy(XA,Y));
    PetscCall(VecAYPX(Y,-1.0,X));
    *fnorm = fMnorm;
    PetscCall(SNESLineSearchApply(ngmres->additive_linesearch,X,F,fnorm,Y));
    PetscCall(SNESLineSearchGetReason(ngmres->additive_linesearch,&lssucceed));
    PetscCall(SNESLineSearchGetNorms(ngmres->additive_linesearch,xnorm,fnorm,ynorm));
    if (lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscFunctionReturn(0);
      }
    }
    if (ngmres->monitor) {
      PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"Additive solution: ||F||_2 = %e\n",*fnorm));
    }
  } else if (ngmres->select_type == SNES_NGMRES_SELECT_DIFFERENCE) {
    selectA = PETSC_TRUE;
    /* Conditions for choosing the accelerated answer */
    /* Criterion A -- the norm of the function isn't increased above the minimum by too much */
    if (fAnorm >= ngmres->gammaA*fminnorm) selectA = PETSC_FALSE;

    /* Criterion B -- the choice of x^A isn't too close to some other choice */
    if (ngmres->epsilonB*dnorm<dminnorm || PetscSqrtReal(*fnorm)<ngmres->deltaB*PetscSqrtReal(fminnorm)) {
    } else selectA=PETSC_FALSE;

    if (selectA) {
      if (ngmres->monitor) {
        PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"picked X_A, ||F_A||_2 = %e, ||F_M||_2 = %e\n",fAnorm,fMnorm));
      }
      /* copy it over */
      *xnorm = xAnorm;
      *fnorm = fAnorm;
      *ynorm = yAnorm;
      PetscCall(VecCopy(FA,F));
      PetscCall(VecCopy(XA,X));
    } else {
      if (ngmres->monitor) {
        PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"picked X_M, ||F_A||_2 = %e, ||F_M||_2 = %e\n",fAnorm,fMnorm));
      }
      *xnorm = xMnorm;
      *fnorm = fMnorm;
      *ynorm = yMnorm;
      PetscCall(VecCopy(XM,Y));
      PetscCall(VecAXPY(Y,-1.0,X));
      PetscCall(VecCopy(FM,F));
      PetscCall(VecCopy(XM,X));
    }
  } else { /* none */
    *xnorm = xAnorm;
    *fnorm = fAnorm;
    *ynorm = yAnorm;
    PetscCall(VecCopy(FA,F));
    PetscCall(VecCopy(XA,X));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SNESNGMRESSelectRestart_Private(SNES snes,PetscInt l,PetscReal fMnorm, PetscReal fAnorm,PetscReal dnorm,PetscReal fminnorm,PetscReal dminnorm,PetscBool *selectRestart)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*)snes->data;

  PetscFunctionBegin;
  *selectRestart = PETSC_FALSE;
  /* difference stagnation restart */
  if ((ngmres->epsilonB*dnorm > dminnorm) && (PetscSqrtReal(fAnorm) > ngmres->deltaB*PetscSqrtReal(fminnorm)) && l > 0) {
    if (ngmres->monitor) {
      PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"difference restart: %e > %e\n",ngmres->epsilonB*dnorm,dminnorm));
    }
    *selectRestart = PETSC_TRUE;
  }
  /* residual stagnation restart */
  if (PetscSqrtReal(fAnorm) > ngmres->gammaC*PetscSqrtReal(fminnorm)) {
    if (ngmres->monitor) {
      PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"residual restart: %e > %e\n",PetscSqrtReal(fAnorm),ngmres->gammaC*PetscSqrtReal(fminnorm)));
    }
    *selectRestart = PETSC_TRUE;
  }

  /* F_M stagnation restart */
  if (ngmres->restart_fm_rise && fMnorm > snes->norm) {
    if (ngmres->monitor) {
      PetscCall(PetscViewerASCIIPrintf(ngmres->monitor,"F_M rise restart: %e > %e\n",fMnorm,snes->norm));
    }
    *selectRestart = PETSC_TRUE;
  }

  PetscFunctionReturn(0);
}
