#include <../src/snes/impls/ngmres/snesngmres.h> /*I "petscsnes.h" I*/
#include <petscblaslapack.h>

#undef __FUNCT__
#define __FUNCT__ "SNESNGMRESUpdateSubspace_Private"
PetscErrorCode SNESNGMRESUpdateSubspace_Private(SNES snes,PetscInt ivec,PetscInt l,Vec F,PetscReal fnorm,Vec X)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  Vec            *Fdot   = ngmres->Fdot;
  Vec            *Xdot   = ngmres->Xdot;
  PetscScalar    *xi     = ngmres->xi;
  PetscInt       i;
  PetscReal      nu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ivec > l) SETERRQ2(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE,"Cannot update vector %d with space size %d!",ivec,l);
  ierr = VecCopy(F,Fdot[ivec]);CHKERRQ(ierr);
  ierr = VecCopy(X,Xdot[ivec]);CHKERRQ(ierr);

  ngmres->fnorms[ivec] = fnorm;
  if (l > 0) {
    ierr = VecMDot(F,l,Fdot,xi);CHKERRQ(ierr);
    for (i = 0; i < l; i++) {
      Q(i,ivec) = xi[i];
      Q(ivec,i) = xi[i];
    }
  } else {
    nu     = fnorm*fnorm;
    Q(0,0) = nu;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNGMRESFormCombinedSolution_Private"
PetscErrorCode SNESNGMRESFormCombinedSolution_Private(SNES snes,PetscInt l,Vec XM,Vec FM,PetscReal fMnorm,Vec X,Vec XA,Vec FA)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscInt       i,j;
  Vec            *Fdot      = ngmres->Fdot;
  Vec            *Xdot      = ngmres->Xdot;
  PetscScalar    *beta      = ngmres->beta;
  PetscScalar    *xi        = ngmres->xi;
  PetscScalar    alph_total = 0.;
  PetscErrorCode ierr;
  PetscReal      nu;
  Vec            Y = snes->work[2];
  PetscBool      changed_y,changed_w;

  PetscFunctionBegin;
  nu = fMnorm*fMnorm;

  /* construct the right hand side and xi factors */
  ierr = VecMDot(FM,l,Fdot,xi);CHKERRQ(ierr);
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
#if defined(PETSC_MISSING_LAPACK_GELSS)
    SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_SUP,"NGMRES with LS requires the LAPACK GELSS routine.");
#else
    ierr          = PetscBLASIntCast(l,&ngmres->m);CHKERRQ(ierr);
    ierr          = PetscBLASIntCast(l,&ngmres->n);CHKERRQ(ierr);
    ngmres->info  = 0;
    ngmres->rcond = -1.;
    ierr          = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&ngmres->m,&ngmres->n,&ngmres->nrhs,ngmres->h,&ngmres->lda,ngmres->beta,&ngmres->ldb,ngmres->s,&ngmres->rcond,&ngmres->rank,ngmres->work,&ngmres->lwork,ngmres->rwork,&ngmres->info));
#else
    PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&ngmres->m,&ngmres->n,&ngmres->nrhs,ngmres->h,&ngmres->lda,ngmres->beta,&ngmres->ldb,ngmres->s,&ngmres->rcond,&ngmres->rank,ngmres->work,&ngmres->lwork,&ngmres->info));
#endif
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (ngmres->info < 0) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"Bad argument to GELSS");
    if (ngmres->info > 0) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD failed to converge");
#endif
  }
  for (i=0; i<l; i++) {
    if (PetscIsInfOrNanScalar(beta[i])) SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_LIB,"SVD generated inconsistent output");
  }
  alph_total = 0.;
  for (i = 0; i < l; i++) alph_total += beta[i];

  ierr = VecCopy(XM,XA);CHKERRQ(ierr);
  ierr = VecScale(XA,1.-alph_total);CHKERRQ(ierr);
  ierr = VecMAXPY(XA,l,beta,Xdot);CHKERRQ(ierr);
  /* check the validity of the step */
  ierr = VecCopy(XA,Y);CHKERRQ(ierr);
  ierr = VecAXPY(Y,-1.0,X);CHKERRQ(ierr);
  ierr = SNESLineSearchPostCheck(snes->linesearch,X,Y,XA,&changed_y,&changed_w);CHKERRQ(ierr);
  if (!ngmres->approxfunc) {
    if (snes->pc && snes->pcside == PC_LEFT) {
      ierr = SNESApplyPC(snes,XA,NULL,NULL,FA);CHKERRQ(ierr);
    } else {
      ierr = SNESComputeFunction(snes,XA,FA);CHKERRQ(ierr);
    }
  }
  else {
    ierr = VecCopy(FM,FA);CHKERRQ(ierr);
    ierr = VecScale(FA,1.-alph_total);CHKERRQ(ierr);
    ierr = VecMAXPY(FA,l,beta,Fdot);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNGMRESCalculateDifferences_Private"
PetscErrorCode SNESNGMRESCalculateDifferences_Private(SNES snes,PetscInt l,Vec X,Vec F,Vec XM,Vec FM,Vec XA,Vec FA,Vec D,PetscReal *dnorm,PetscReal *dminnorm,PetscReal *fAnorm)
{
  PetscErrorCode ierr;
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscReal      dcurnorm;
  Vec            *Xdot = ngmres->Xdot;
  PetscInt       i;

  PetscFunctionBegin;
  if (ngmres->singlereduction) {
    *dminnorm = -1.0;
    if (fAnorm) {
      ierr = VecNormBegin(FA,NORM_2,fAnorm);CHKERRQ(ierr);
    }
    if (dnorm) {
      ierr = VecCopy(XA,D);CHKERRQ(ierr);
      ierr = VecAXPY(D,-1.0,XM);CHKERRQ(ierr);
      ierr = VecNormBegin(D,NORM_2,dnorm);CHKERRQ(ierr);
    }
    if (dminnorm) {
      for (i=0; i<l; i++) {
        ierr=VecAXPY(Xdot[i],-1.0,XA);CHKERRQ(ierr);
      }
      for (i=0; i<l; i++) {
        ierr = VecNormBegin(Xdot[i],NORM_2,&ngmres->xnorms[i]);CHKERRQ(ierr);
      }
    }
    if (fAnorm) {ierr = VecNormEnd(FA,NORM_2,fAnorm);CHKERRQ(ierr);}
    if (dnorm) {ierr = VecNormEnd(D,NORM_2,dnorm);CHKERRQ(ierr);}
    if (dminnorm) {
      for (i=0; i<l; i++) {
        ierr = VecNormEnd(Xdot[i],NORM_2,&ngmres->xnorms[i]);CHKERRQ(ierr);
      }
      for (i=0; i<l; i++) {
        dcurnorm = ngmres->xnorms[i];
        if ((dcurnorm < *dminnorm) || (*dminnorm < 0.0)) *dminnorm = dcurnorm;
        ierr=VecAXPY(Xdot[i],1.0,XA);CHKERRQ(ierr);
      }
    }
  } else {
    if (dnorm) {
      ierr=VecCopy(XA,D);CHKERRQ(ierr);
      ierr=VecAXPY(D,-1.0,XM);CHKERRQ(ierr);
      ierr=VecNormBegin(D,NORM_2,dnorm);CHKERRQ(ierr);
    }
    if (fAnorm) {
      ierr=VecNormBegin(FA,NORM_2,fAnorm);CHKERRQ(ierr);
    }
    if (dnorm) {
      ierr=VecNormEnd(D,NORM_2,dnorm);CHKERRQ(ierr);
    }
    if (fAnorm) {
      ierr=VecNormEnd(FA,NORM_2,fAnorm);CHKERRQ(ierr);
    }
    if (dminnorm) {
      *dminnorm = -1.0;
      for (i=0; i<l; i++) {
        ierr=VecCopy(XA,D);CHKERRQ(ierr);
        ierr=VecAXPY(D,-1.0,Xdot[i]);CHKERRQ(ierr);
        ierr=VecNorm(D,NORM_2,&dcurnorm);CHKERRQ(ierr);
        if ((dcurnorm < *dminnorm) || (*dminnorm < 0.0)) *dminnorm = dcurnorm;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNGMRESSelect_Private"
PetscErrorCode SNESNGMRESSelect_Private(SNES snes,PetscInt k_restart,Vec XM,Vec FM,PetscReal fMnorm,Vec XA,Vec FA,PetscReal fAnorm,PetscReal dnorm,PetscReal fminnorm,PetscReal dminnorm,Vec X,Vec F,Vec Y,PetscReal *fnorm)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*) snes->data;
  PetscErrorCode ierr;
  PetscBool      lssucceed,selectA;

  PetscFunctionBegin;
  if (ngmres->select_type == SNES_NGMRES_SELECT_LINESEARCH) {
    /* X = X + \lambda(XA - X) */
    if (ngmres->monitor) {
      ierr = PetscViewerASCIIPrintf(ngmres->monitor,"||F_A||_2 = %e, ||F_M||_2 = %e\n",fAnorm,fMnorm);CHKERRQ(ierr);
    }
    ierr   = VecCopy(FM,F);CHKERRQ(ierr);
    ierr   = VecCopy(XM,X);CHKERRQ(ierr);
    ierr   = VecCopy(XA,Y);CHKERRQ(ierr);
    ierr   = VecAYPX(Y,-1.0,X);CHKERRQ(ierr);
    *fnorm = fMnorm;
    ierr   = SNESLineSearchApply(ngmres->additive_linesearch,X,F,fnorm,Y);CHKERRQ(ierr);
    ierr   = SNESLineSearchGetSuccess(ngmres->additive_linesearch,&lssucceed);CHKERRQ(ierr);
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        PetscFunctionReturn(0);
      }
    }
    if (ngmres->monitor) {
      ierr = PetscViewerASCIIPrintf(ngmres->monitor,"Additive solution: ||F||_2 = %e\n",*fnorm);CHKERRQ(ierr);
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
        ierr = PetscViewerASCIIPrintf(ngmres->monitor,"picked X_A, ||F_A||_2 = %e, ||F_M||_2 = %e\n",fAnorm,fMnorm);CHKERRQ(ierr);
      }
      /* copy it over */
      *fnorm = fAnorm;
      ierr   = VecCopy(FA,F);CHKERRQ(ierr);
      ierr   = VecCopy(XA,X);CHKERRQ(ierr);
    } else {
      if (ngmres->monitor) {
        ierr = PetscViewerASCIIPrintf(ngmres->monitor,"picked X_M, ||F_A||_2 = %e, ||F_M||_2 = %e\n",fAnorm,fMnorm);CHKERRQ(ierr);
      }
      *fnorm = fMnorm;
      ierr   = VecCopy(XM,Y);CHKERRQ(ierr);
      ierr   = VecAXPY(Y,-1.0,X);CHKERRQ(ierr);
      ierr   = VecCopy(FM,F);CHKERRQ(ierr);
      ierr   = VecCopy(XM,X);CHKERRQ(ierr);
    }
  } else { /* none */
    *fnorm = fAnorm;
    ierr   = VecCopy(FA,F);CHKERRQ(ierr);
    ierr   = VecCopy(XA,X);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESNGMRESSelectRestart_Private"
PetscErrorCode SNESNGMRESSelectRestart_Private(SNES snes,PetscInt l,PetscReal fAnorm,PetscReal dnorm,PetscReal fminnorm,PetscReal dminnorm,PetscBool *selectRestart)
{
  SNES_NGMRES    *ngmres = (SNES_NGMRES*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *selectRestart = PETSC_FALSE;
  /* difference stagnation restart */
  if ((ngmres->epsilonB*dnorm > dminnorm) && (PetscSqrtReal(fAnorm) > ngmres->deltaB*PetscSqrtReal(fminnorm)) && l > 0) {
    if (ngmres->monitor) {
      ierr = PetscViewerASCIIPrintf(ngmres->monitor,"difference restart: %e > %e\n",ngmres->epsilonB*dnorm,dminnorm);CHKERRQ(ierr);
    }
    *selectRestart = PETSC_TRUE;
  }
  /* residual stagnation restart */
  if (PetscSqrtReal(fAnorm) > ngmres->gammaC*PetscSqrtReal(fminnorm)) {
    if (ngmres->monitor) {
      ierr = PetscViewerASCIIPrintf(ngmres->monitor,"residual restart: %e > %e\n",PetscSqrtReal(fAnorm),ngmres->gammaC*PetscSqrtReal(fminnorm));CHKERRQ(ierr);
    }
    *selectRestart = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}
