#include <../src/tao/constrained/impls/admm/admm.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsc/private/petscimpl.h>

/* Updates terminating criteria
 *
 * 1  ||r_k|| = ||Ax+Bz-c|| =< catol_admm* max{||Ax||,||Bz||,||c||}
 *
 * 2. Updates dual residual, d_k
 *
 * 3. ||d_k|| = ||mu*A^T*B(z_k-z_{k-1})|| =< gatol_admm * ||A^Ty||   */

static PetscBool cited = PETSC_FALSE;
static const char citation[] =
  "@misc{xu2017adaptive,\n"
  "   title={Adaptive Relaxed ADMM: Convergence Theory and Practical Implementation},\n"
  "   author={Zheng Xu and Mario A. T. Figueiredo and Xiaoming Yuan and Christoph Studer and Tom Goldstein},\n"
  "   year={2017},\n"
  "   eprint={1704.02712},\n"
  "   archivePrefix={arXiv},\n"
  "   primaryClass={cs.CV}\n"
  "}  \n";

const char *const TaoADMMRegularizerTypes[] = {"REGULARIZER_USER","REGULARIZER_SOFT_THRESH","TaoADMMRegularizerType","TAO_ADMM_",NULL};
const char *const TaoADMMUpdateTypes[]      = {"UPDATE_BASIC","UPDATE_ADAPTIVE","UPDATE_ADAPTIVE_RELAXED","TaoADMMUpdateType","TAO_ADMM_",NULL};
const char *const TaoALMMTypes[]            = {"CLASSIC","PHR","TaoALMMType","TAO_ALMM_",NULL};

static PetscErrorCode TaoADMMToleranceUpdate(Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;
  PetscReal      Axnorm,Bznorm,ATynorm,temp;
  Vec            tempJR,tempL;
  Tao            mis;

  PetscFunctionBegin;
  mis    = am->subsolverX;
  tempJR = am->workJacobianRight;
  tempL  = am->workLeft;
  /* ATy */
  ierr = TaoComputeJacobianEquality(mis, am->y, mis->jacobian_equality, mis->jacobian_equality_pre);CHKERRQ(ierr);
  ierr = MatMultTranspose(mis->jacobian_equality,am->y,tempJR);CHKERRQ(ierr);
  ierr = VecNorm(tempJR,NORM_2,&ATynorm);CHKERRQ(ierr);
  /* dualres = mu * ||AT(Bz-Bzold)||_2 */
  ierr = VecWAXPY(tempJR,-1.,am->Bzold,am->Bz);CHKERRQ(ierr);
  ierr = MatMultTranspose(mis->jacobian_equality,tempJR,tempL);CHKERRQ(ierr);
  ierr = VecNorm(tempL,NORM_2,&am->dualres);CHKERRQ(ierr);
  am->dualres *= am->mu;

  /* ||Ax||_2, ||Bz||_2 */
  ierr = VecNorm(am->Ax,NORM_2,&Axnorm);CHKERRQ(ierr);
  ierr = VecNorm(am->Bz,NORM_2,&Bznorm);CHKERRQ(ierr);

  /* Set catol to be catol_admm *  max{||Ax||,||Bz||,||c||} *
   * Set gatol to be gatol_admm *  ||A^Ty|| *
   * while cnorm is ||r_k||_2, and gnorm is ||d_k||_2 */
  temp = am->catol_admm * PetscMax(Axnorm, (!am->const_norm) ? Bznorm : PetscMax(Bznorm,am->const_norm));
  ierr = TaoSetConstraintTolerances(tao,temp,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = TaoSetTolerances(tao, am->gatol_admm*ATynorm, PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Penaly Update for Adaptive ADMM. */
static PetscErrorCode AdaptiveADMMPenaltyUpdate(Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;
  PetscReal      ydiff_norm, yhatdiff_norm, Axdiff_norm, Bzdiff_norm, Axyhat, Bzy, a_sd, a_mg, a_k, b_sd, b_mg, b_k;
  PetscBool      hflag, gflag;
  Vec            tempJR,tempJR2;

  PetscFunctionBegin;
  tempJR  = am->workJacobianRight;
  tempJR2 = am->workJacobianRight2;
  hflag   = PETSC_FALSE;
  gflag   = PETSC_FALSE;
  a_k     = -1;
  b_k     = -1;

  ierr = VecWAXPY(tempJR,-1.,am->Axold,am->Ax);CHKERRQ(ierr);
  ierr = VecWAXPY(tempJR2,-1.,am->yhatold,am->yhat);CHKERRQ(ierr);
  ierr = VecNorm(tempJR,NORM_2,&Axdiff_norm);CHKERRQ(ierr);
  ierr = VecNorm(tempJR2,NORM_2,&yhatdiff_norm);CHKERRQ(ierr);
  ierr = VecDot(tempJR,tempJR2,&Axyhat);CHKERRQ(ierr);

  ierr = VecWAXPY(tempJR,-1.,am->Bz0,am->Bz);CHKERRQ(ierr);
  ierr = VecWAXPY(tempJR2,-1.,am->y,am->y0);CHKERRQ(ierr);
  ierr = VecNorm(tempJR,NORM_2,&Bzdiff_norm);CHKERRQ(ierr);
  ierr = VecNorm(tempJR2,NORM_2,&ydiff_norm);CHKERRQ(ierr);
  ierr = VecDot(tempJR,tempJR2,&Bzy);CHKERRQ(ierr);

  if (Axyhat > am->orthval*Axdiff_norm*yhatdiff_norm + am->mueps) {
    hflag = PETSC_TRUE;
    a_sd  = PetscSqr(yhatdiff_norm)/Axyhat; /* alphaSD */
    a_mg  = Axyhat/PetscSqr(Axdiff_norm);   /* alphaMG */
    a_k   = (a_mg/a_sd) > 0.5 ? a_mg : a_sd - 0.5*a_mg;
  }
  if (Bzy > am->orthval*Bzdiff_norm*ydiff_norm + am->mueps) {
    gflag = PETSC_TRUE;
    b_sd  = PetscSqr(ydiff_norm)/Bzy;  /* betaSD */
    b_mg  = Bzy/PetscSqr(Bzdiff_norm); /* betaMG */
    b_k   = (b_mg/b_sd) > 0.5 ? b_mg : b_sd - 0.5*b_mg;
  }
  am->muold = am->mu;
  if (gflag && hflag) {
    am->mu = PetscSqrtReal(a_k*b_k);
  } else if (hflag) {
    am->mu = a_k;
  } else if (gflag) {
    am->mu = b_k;
  }
  if (am->mu > am->muold) {
    am->mu = am->muold;
  }
  if (am->mu < am->mumin) {
    am->mu = am->mumin;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  TaoADMMSetRegularizerType_ADMM(Tao tao, TaoADMMRegularizerType type)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  am->regswitch = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode  TaoADMMGetRegularizerType_ADMM(Tao tao, TaoADMMRegularizerType *type)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  *type = am->regswitch;
  PetscFunctionReturn(0);
}

static PetscErrorCode  TaoADMMSetUpdateType_ADMM(Tao tao, TaoADMMUpdateType type)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  am->update = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode  TaoADMMGetUpdateType_ADMM(Tao tao, TaoADMMUpdateType *type)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  *type = am->update;
  PetscFunctionReturn(0);
}

/* This routine updates Jacobians with new x,z vectors,
 * and then updates Ax and Bz vectors, then computes updated residual vector*/
static PetscErrorCode ADMMUpdateConstraintResidualVector(Tao tao, Vec x, Vec z, Vec Ax, Vec Bz, Vec residual)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  Tao            mis,reg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mis  = am->subsolverX;
  reg  = am->subsolverZ;
  ierr = TaoComputeJacobianEquality(mis, x, mis->jacobian_equality, mis->jacobian_equality_pre);CHKERRQ(ierr);
  ierr = MatMult(mis->jacobian_equality,x,Ax);CHKERRQ(ierr);
  ierr = TaoComputeJacobianEquality(reg, z, reg->jacobian_equality, reg->jacobian_equality_pre);CHKERRQ(ierr);
  ierr = MatMult(reg->jacobian_equality,z,Bz);CHKERRQ(ierr);

  ierr = VecWAXPY(residual,1.,Bz,Ax);CHKERRQ(ierr);
  if (am->constraint != NULL) {
    ierr = VecAXPY(residual,-1.,am->constraint);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Updates Augmented Lagrangians to given routines *
 * For subsolverX, routine needs to be ComputeObjectiveAndGraidnet
 * Separate Objective and Gradient routines are not supported.  */
static PetscErrorCode SubObjGradUpdate(Tao tao, Vec x, PetscReal *f, Vec g, void *ptr)
{
  Tao            parent = (Tao)ptr;
  TAO_ADMM       *am    = (TAO_ADMM*)parent->data;
  PetscErrorCode ierr;
  PetscReal      temp,temp2;
  Vec            tempJR;

  PetscFunctionBegin;
  tempJR = am->workJacobianRight;
  ierr   = ADMMUpdateConstraintResidualVector(parent, x, am->subsolverZ->solution, am->Ax, am->Bz, am->residual);CHKERRQ(ierr);
  ierr   = (*am->ops->misfitobjgrad)(am->subsolverX,x,f,g,am->misfitobjgradP);CHKERRQ(ierr);

  am->last_misfit_val = *f;
  /* Objective  Add + yT(Ax+Bz-c) + mu/2*||Ax+Bz-c||_2^2 */
  ierr = VecTDot(am->residual,am->y,&temp);CHKERRQ(ierr);
  ierr = VecTDot(am->residual,am->residual,&temp2);CHKERRQ(ierr);
  *f   += temp + (am->mu/2)*temp2;

  /* Gradient. Add + mu*AT(Ax+Bz-c) + yTA*/
  ierr = MatMultTranspose(tao->jacobian_equality,am->residual,tempJR);CHKERRQ(ierr);
  ierr = VecAXPY(g,am->mu,tempJR);CHKERRQ(ierr);
  ierr = MatMultTranspose(tao->jacobian_equality,am->y,tempJR);CHKERRQ(ierr);
  ierr = VecAXPY(g,1.,tempJR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Updates Augmented Lagrangians to given routines
 * For subsolverZ, routine needs to be ComputeObjectiveAndGraidnet
 * Separate Objective and Gradient routines are not supported.  */
static PetscErrorCode RegObjGradUpdate(Tao tao, Vec z, PetscReal *f, Vec g, void *ptr)
{
  Tao            parent = (Tao)ptr;
  TAO_ADMM       *am    = (TAO_ADMM*)parent->data;
  PetscErrorCode ierr;
  PetscReal      temp,temp2;
  Vec            tempJR;

  PetscFunctionBegin;
  tempJR = am->workJacobianRight;
  ierr   = ADMMUpdateConstraintResidualVector(parent, am->subsolverX->solution, z, am->Ax, am->Bz, am->residual);CHKERRQ(ierr);
  ierr   = (*am->ops->regobjgrad)(am->subsolverZ,z,f,g,am->regobjgradP);CHKERRQ(ierr);
  am->last_reg_val= *f;
  /* Objective  Add  + yT(Ax+Bz-c) + mu/2*||Ax+Bz-c||_2^2 */
  ierr = VecTDot(am->residual,am->y,&temp);CHKERRQ(ierr);
  ierr = VecTDot(am->residual,am->residual,&temp2);CHKERRQ(ierr);
  *f   += temp + (am->mu/2)*temp2;

  /* Gradient. Add + mu*BT(Ax+Bz-c) + yTB*/
  ierr = MatMultTranspose(am->subsolverZ->jacobian_equality,am->residual,tempJR);CHKERRQ(ierr);
  ierr = VecAXPY(g,am->mu,tempJR);CHKERRQ(ierr);
  ierr = MatMultTranspose(am->subsolverZ->jacobian_equality,am->y,tempJR);CHKERRQ(ierr);
  ierr = VecAXPY(g,1.,tempJR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Computes epsilon padded L1 norm lambda*sum(sqrt(x^2+eps^2)-eps */
static PetscErrorCode ADMML1EpsilonNorm(Tao tao, Vec x, PetscReal eps, PetscReal *norm)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscInt       N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = VecGetSize(am->workLeft,&N);CHKERRQ(ierr);
  ierr   = VecPointwiseMult(am->workLeft,x,x);CHKERRQ(ierr);
  ierr   = VecShift(am->workLeft,am->l1epsilon*am->l1epsilon);CHKERRQ(ierr);
  ierr   = VecSqrtAbs(am->workLeft);CHKERRQ(ierr);
  ierr   = VecSum(am->workLeft,norm);CHKERRQ(ierr);
  *norm += N*am->l1epsilon;
  *norm *= am->lambda;
  PetscFunctionReturn(0);
}

static PetscErrorCode ADMMInternalHessianUpdate(Mat H, Mat Constraint, PetscBool Identity, void *ptr)
{
  TAO_ADMM       *am = (TAO_ADMM*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (am->update) {
  case (TAO_ADMM_UPDATE_BASIC):
    break;
  case (TAO_ADMM_UPDATE_ADAPTIVE):
  case (TAO_ADMM_UPDATE_ADAPTIVE_RELAXED):
    if (H && (am->muold != am->mu)) {
      if (!Identity) {
        ierr = MatAXPY(H,am->mu-am->muold,Constraint,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatShift(H,am->mu-am->muold);CHKERRQ(ierr);
      }
    }
    break;
  }
  PetscFunctionReturn(0);
}

/* Updates Hessian - adds second derivative of augmented Lagrangian
 * H \gets H + \rho*ATA
 * Here, \rho does not change in TAO_ADMM_UPDATE_BASIC - thus no-op
 * For ADAPTAIVE,ADAPTIVE_RELAXED,
 * H \gets H + (\rho-\rhoold)*ATA
 * Here, we assume that A is linear constraint i.e., doesnt change.
 * Thus, for both ADAPTIVE, and RELAXED, ATA matrix is pre-set (except for A=I (null case)) see TaoSetUp_ADMM */
static PetscErrorCode SubHessianUpdate(Tao tao, Vec x, Mat H, Mat Hpre, void *ptr)
{
  Tao            parent = (Tao)ptr;
  TAO_ADMM       *am    = (TAO_ADMM*)parent->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (am->Hxchange) {
    /* Case where Hessian gets updated with respect to x vector input. */
    ierr = (*am->ops->misfithess)(am->subsolverX,x,H,Hpre,am->misfithessP);CHKERRQ(ierr);
    ierr = ADMMInternalHessianUpdate(am->subsolverX->hessian,am->ATA,am->xJI,am);CHKERRQ(ierr);
  } else if (am->Hxbool) {
    /* Hessian doesn't get updated. H(x) = c */
    /* Update Lagrangian only only per TAO call */
    ierr       = ADMMInternalHessianUpdate(am->subsolverX->hessian,am->ATA,am->xJI,am);CHKERRQ(ierr);
    am->Hxbool = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/* Same as SubHessianUpdate, except for B matrix instead of A matrix */
static PetscErrorCode RegHessianUpdate(Tao tao, Vec z, Mat H, Mat Hpre, void *ptr)
{
  Tao            parent = (Tao)ptr;
  TAO_ADMM       *am    = (TAO_ADMM*)parent->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (am->Hzchange) {
    /* Case where Hessian gets updated with respect to x vector input. */
    ierr = (*am->ops->reghess)(am->subsolverZ,z,H,Hpre,am->reghessP);CHKERRQ(ierr);
    ierr = ADMMInternalHessianUpdate(am->subsolverZ->hessian,am->BTB,am->zJI,am);CHKERRQ(ierr);
  } else if (am->Hzbool) {
    /* Hessian doesn't get updated. H(x) = c */
    /* Update Lagrangian only only per TAO call */
    ierr = ADMMInternalHessianUpdate(am->subsolverZ->hessian,am->BTB,am->zJI,am);CHKERRQ(ierr);
    am->Hzbool = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/* Shell Matrix routine for A matrix.
 * This gets used when user puts NULL for
 * TaoSetJacobianEqualityRoutine(tao, NULL,NULL, ...)
 * Essentially sets A=I*/
static PetscErrorCode JacobianIdentity(Mat mat,Vec in,Vec out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(in,out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Shell Matrix routine for B matrix.
 * This gets used when user puts NULL for
 * TaoADMMSetRegularizerConstraintJacobian(tao, NULL,NULL, ...)
 * Sets B=-I */
static PetscErrorCode JacobianIdentityB(Mat mat,Vec in,Vec out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(in,out);CHKERRQ(ierr);
  ierr = VecScale(out,-1.);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Solve f(x) + g(z) s.t. Ax + Bz = c */
static PetscErrorCode TaoSolve_ADMM(Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;
  PetscInt       N;
  PetscReal      reg_func;
  PetscBool      is_reg_shell;
  Vec            tempL;

  PetscFunctionBegin;
  if (am->regswitch != TAO_ADMM_REGULARIZER_SOFT_THRESH) {
    PetscCheckFalse(!am->subsolverX->ops->computejacobianequality,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"Must call TaoADMMSetMisfitConstraintJacobian() first");
    PetscCheckFalse(!am->subsolverZ->ops->computejacobianequality,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"Must call TaoADMMSetRegularizerConstraintJacobian() first");
    if (am->constraint != NULL) {
      ierr = VecNorm(am->constraint,NORM_2,&am->const_norm);CHKERRQ(ierr);
    }
  }
  tempL = am->workLeft;
  ierr  = VecGetSize(tempL,&N);CHKERRQ(ierr);

  if (am->Hx && am->ops->misfithess) {
    ierr = TaoSetHessian(am->subsolverX, am->Hx, am->Hx, SubHessianUpdate, tao);CHKERRQ(ierr);
  }

  if (!am->zJI) {
    /* Currently, B is assumed to be a linear system, i.e., not getting updated*/
    ierr = MatTransposeMatMult(am->JB,am->JB,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&(am->BTB));CHKERRQ(ierr);
  }
  if (!am->xJI) {
    /* Currently, A is assumed to be a linear system, i.e., not getting updated*/
    ierr = MatTransposeMatMult(am->subsolverX->jacobian_equality,am->subsolverX->jacobian_equality,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&(am->ATA));CHKERRQ(ierr);
  }

  is_reg_shell = PETSC_FALSE;

  ierr = PetscObjectTypeCompare((PetscObject)am->subsolverZ, TAOSHELL, &is_reg_shell);CHKERRQ(ierr);

  if (!is_reg_shell) {
    switch (am->regswitch) {
    case (TAO_ADMM_REGULARIZER_USER):
      PetscCheckFalse(!am->ops->regobjgrad,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"Must call TaoADMMSetRegularizerObjectiveAndGradientRoutine() first if one wishes to use TAO_ADMM_REGULARIZER_USER with non-TAOSHELL type");
      break;
    case (TAO_ADMM_REGULARIZER_SOFT_THRESH):
      /* Soft Threshold. */
      break;
    }
    if (am->ops->regobjgrad) {
      ierr = TaoSetObjectiveAndGradient(am->subsolverZ, NULL, RegObjGradUpdate, tao);CHKERRQ(ierr);
    }
    if (am->Hz && am->ops->reghess) {
      ierr = TaoSetHessian(am->subsolverZ, am->Hz, am->Hzpre, RegHessianUpdate, tao);CHKERRQ(ierr);
    }
  }

  switch (am->update) {
  case TAO_ADMM_UPDATE_BASIC:
    if (am->subsolverX->hessian) {
      /* In basic case, Hessian does not get updated w.r.t. to spectral penalty
       * Here, when A is set, i.e., am->xJI, add mu*ATA to Hessian*/
      if (!am->xJI) {
        ierr = MatAXPY(am->subsolverX->hessian,am->mu,am->ATA,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatShift(am->subsolverX->hessian,am->mu);CHKERRQ(ierr);
      }
    }
    if (am->subsolverZ->hessian && am->regswitch == TAO_ADMM_REGULARIZER_USER) {
      if (am->regswitch == TAO_ADMM_REGULARIZER_USER && !am->zJI) {
        ierr = MatAXPY(am->subsolverZ->hessian,am->mu,am->BTB,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatShift(am->subsolverZ->hessian,am->mu);CHKERRQ(ierr);
      }
    }
    break;
  case TAO_ADMM_UPDATE_ADAPTIVE:
  case TAO_ADMM_UPDATE_ADAPTIVE_RELAXED:
    break;
  }

  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  tao->reason = TAO_CONTINUE_ITERATING;

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }
    ierr = VecCopy(am->Bz, am->Bzold);CHKERRQ(ierr);

    /* x update */
    ierr = TaoSolve(am->subsolverX);CHKERRQ(ierr);
    ierr = TaoComputeJacobianEquality(am->subsolverX, am->subsolverX->solution, am->subsolverX->jacobian_equality, am->subsolverX->jacobian_equality_pre);CHKERRQ(ierr);
    ierr = MatMult(am->subsolverX->jacobian_equality, am->subsolverX->solution,am->Ax);CHKERRQ(ierr);

    am->Hxbool = PETSC_TRUE;

    /* z update */
    switch (am->regswitch) {
    case TAO_ADMM_REGULARIZER_USER:
      ierr = TaoSolve(am->subsolverZ);CHKERRQ(ierr);
      break;
    case TAO_ADMM_REGULARIZER_SOFT_THRESH:
      /* L1 assumes A,B jacobians are identity nxn matrix */
      ierr = VecWAXPY(am->workJacobianRight,1/am->mu,am->y,am->Ax);CHKERRQ(ierr);
      ierr = TaoSoftThreshold(am->workJacobianRight,-am->lambda/am->mu,am->lambda/am->mu,am->subsolverZ->solution);CHKERRQ(ierr);
      break;
    }
    am->Hzbool = PETSC_TRUE;
    /* Returns Ax + Bz - c with updated Ax,Bz vectors */
    ierr = ADMMUpdateConstraintResidualVector(tao, am->subsolverX->solution, am->subsolverZ->solution, am->Ax, am->Bz, am->residual);CHKERRQ(ierr);
    /* Dual variable, y += y + mu*(Ax+Bz-c) */
    ierr = VecWAXPY(am->y, am->mu, am->residual, am->yold);CHKERRQ(ierr);

    /* stopping tolerance update */
    ierr = TaoADMMToleranceUpdate(tao);CHKERRQ(ierr);

    /* Updating Spectral Penalty */
    switch (am->update) {
    case TAO_ADMM_UPDATE_BASIC:
      am->muold = am->mu;
      break;
    case TAO_ADMM_UPDATE_ADAPTIVE:
    case TAO_ADMM_UPDATE_ADAPTIVE_RELAXED:
      if (tao->niter == 0) {
        ierr = VecCopy(am->y, am->y0);CHKERRQ(ierr);
        ierr = VecWAXPY(am->residual, 1., am->Ax, am->Bzold);CHKERRQ(ierr);
        if (am->constraint) {
          ierr = VecAXPY(am->residual, -1., am->constraint);CHKERRQ(ierr);
        }
        ierr      = VecWAXPY(am->yhatold, -am->mu, am->residual, am->yold);CHKERRQ(ierr);
        ierr      = VecCopy(am->Ax, am->Axold);CHKERRQ(ierr);
        ierr      = VecCopy(am->Bz, am->Bz0);CHKERRQ(ierr);
        am->muold = am->mu;
      } else if (tao->niter % am->T == 1) {
        /* we have compute Bzold in a previous iteration, and we computed Ax above */
        ierr = VecWAXPY(am->residual, 1., am->Ax, am->Bzold);CHKERRQ(ierr);
        if (am->constraint) {
          ierr = VecAXPY(am->residual, -1., am->constraint);CHKERRQ(ierr);
        }
        ierr = VecWAXPY(am->yhat, -am->mu, am->residual, am->yold);CHKERRQ(ierr);
        ierr = AdaptiveADMMPenaltyUpdate(tao);CHKERRQ(ierr);
        ierr = VecCopy(am->Ax, am->Axold);CHKERRQ(ierr);
        ierr = VecCopy(am->Bz, am->Bz0);CHKERRQ(ierr);
        ierr = VecCopy(am->yhat, am->yhatold);CHKERRQ(ierr);
        ierr = VecCopy(am->y, am->y0);CHKERRQ(ierr);
      } else {
        am->muold = am->mu;
      }
      break;
    default:
      break;
    }
    tao->niter++;

    /* Calculate original function values. misfit part was done in TaoADMMToleranceUpdate*/
    switch (am->regswitch) {
    case TAO_ADMM_REGULARIZER_USER:
      if (is_reg_shell) {
        ierr = ADMML1EpsilonNorm(tao,am->subsolverZ->solution,am->l1epsilon,&reg_func);CHKERRQ(ierr);
      } else {
        (*am->ops->regobjgrad)(am->subsolverZ,am->subsolverX->solution,&reg_func,tempL,am->regobjgradP);
      }
      break;
    case TAO_ADMM_REGULARIZER_SOFT_THRESH:
      ierr = ADMML1EpsilonNorm(tao,am->subsolverZ->solution,am->l1epsilon,&reg_func);CHKERRQ(ierr);
      break;
    }
    ierr = VecCopy(am->y,am->yold);CHKERRQ(ierr);
    ierr = ADMMUpdateConstraintResidualVector(tao, am->subsolverX->solution, am->subsolverZ->solution, am->Ax, am->Bz, am->residual);CHKERRQ(ierr);
    ierr = VecNorm(am->residual,NORM_2,&am->resnorm);CHKERRQ(ierr);
    ierr = TaoLogConvergenceHistory(tao,am->last_misfit_val + reg_func,am->dualres,am->resnorm,tao->ksp_its);CHKERRQ(ierr);

    ierr = TaoMonitor(tao,tao->niter,am->last_misfit_val + reg_func,am->dualres,am->resnorm,1.0);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  /* Update vectors */
  ierr = VecCopy(am->subsolverX->solution,tao->solution);CHKERRQ(ierr);
  ierr = VecCopy(am->subsolverX->gradient,tao->gradient);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)am->subsolverX,"TaoGetADMMParentTao_ADMM", NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)am->subsolverZ,"TaoGetADMMParentTao_ADMM", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoADMMSetRegularizerType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoADMMGetRegularizerType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoADMMSetUpdateType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoADMMGetUpdateType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_ADMM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"ADMM problem that solves f(x) in a form of f(x) + g(z) subject to x - z = 0. Norm 1 and 2 are supported. Different subsolver routines can be selected. ");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_admm_regularizer_coefficient","regularizer constant","",am->lambda,&am->lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_admm_spectral_penalty","Constant for Augmented Lagrangian term.","",am->mu,&am->mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_admm_relaxation_parameter","x relaxation parameter for Z update.","",am->gamma,&am->gamma,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_admm_tolerance_update_factor","ADMM dynamic tolerance update factor.","",am->tol,&am->tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_admm_spectral_penalty_update_factor","ADMM spectral penalty update curvature safeguard value.","",am->orthval,&am->orthval,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_admm_minimum_spectral_penalty","Set ADMM minimum spectral penalty.","",am->mumin,&am->mumin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tao_admm_dual_update","Lagrangian dual update policy","TaoADMMUpdateType",
                          TaoADMMUpdateTypes,(PetscEnum)am->update,(PetscEnum*)&am->update,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tao_admm_regularizer_type","ADMM regularizer update rule","TaoADMMRegularizerType",
                          TaoADMMRegularizerTypes,(PetscEnum)am->regswitch,(PetscEnum*)&am->regswitch,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoSetFromOptions(am->subsolverX);CHKERRQ(ierr);
  if (am->regswitch != TAO_ADMM_REGULARIZER_SOFT_THRESH) {
    ierr = TaoSetFromOptions(am->subsolverZ);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_ADMM(Tao tao,PetscViewer viewer)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = TaoView(am->subsolverX,viewer);CHKERRQ(ierr);
  ierr = TaoView(am->subsolverZ,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_ADMM(Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;
  PetscInt       n,N,M;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
  ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);
  /* If Jacobian is given as NULL, it means Jacobian is identity matrix with size of solution vector */
  if (!am->JB) {
    am->zJI   = PETSC_TRUE;
    ierr      = MatCreateShell(PetscObjectComm((PetscObject)tao),n,n,PETSC_DETERMINE,PETSC_DETERMINE,NULL,&am->JB);CHKERRQ(ierr);
    ierr      = MatShellSetOperation(am->JB,MATOP_MULT,(void (*)(void))JacobianIdentityB);CHKERRQ(ierr);
    ierr      = MatShellSetOperation(am->JB,MATOP_MULT_TRANSPOSE,(void (*)(void))JacobianIdentityB);CHKERRQ(ierr);
    ierr      = MatShellSetOperation(am->JB,MATOP_TRANSPOSE_MAT_MULT,(void (*)(void))JacobianIdentityB);CHKERRQ(ierr);
    am->JBpre = am->JB;
  }
  if (!am->JA) {
    am->xJI   = PETSC_TRUE;
    ierr      = MatCreateShell(PetscObjectComm((PetscObject)tao),n,n,PETSC_DETERMINE,PETSC_DETERMINE,NULL,&am->JA);CHKERRQ(ierr);
    ierr      = MatShellSetOperation(am->JA,MATOP_MULT,(void (*)(void))JacobianIdentity);CHKERRQ(ierr);
    ierr      = MatShellSetOperation(am->JA,MATOP_MULT_TRANSPOSE,(void (*)(void))JacobianIdentity);CHKERRQ(ierr);
    ierr      = MatShellSetOperation(am->JA,MATOP_TRANSPOSE_MAT_MULT,(void (*)(void))JacobianIdentity);CHKERRQ(ierr);
    am->JApre = am->JA;
  }
  ierr = MatCreateVecs(am->JA,NULL,&am->Ax);CHKERRQ(ierr);
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  }
  ierr = TaoSetSolution(am->subsolverX, tao->solution);CHKERRQ(ierr);
  if (!am->z) {
    ierr = VecDuplicate(tao->solution,&am->z);CHKERRQ(ierr);
    ierr = VecSet(am->z,0.0);CHKERRQ(ierr);
  }
  ierr = TaoSetSolution(am->subsolverZ, am->z);CHKERRQ(ierr);
  if (!am->workLeft) {
    ierr = VecDuplicate(tao->solution,&am->workLeft);CHKERRQ(ierr);
  }
  if (!am->Axold) {
    ierr = VecDuplicate(am->Ax,&am->Axold);CHKERRQ(ierr);
  }
  if (!am->workJacobianRight) {
    ierr = VecDuplicate(am->Ax,&am->workJacobianRight);CHKERRQ(ierr);
  }
  if (!am->workJacobianRight2) {
    ierr = VecDuplicate(am->Ax,&am->workJacobianRight2);CHKERRQ(ierr);
  }
  if (!am->Bz) {
    ierr = VecDuplicate(am->Ax,&am->Bz);CHKERRQ(ierr);
  }
  if (!am->Bzold) {
    ierr = VecDuplicate(am->Ax,&am->Bzold);CHKERRQ(ierr);
  }
  if (!am->Bz0) {
    ierr = VecDuplicate(am->Ax,&am->Bz0);CHKERRQ(ierr);
  }
  if (!am->y) {
    ierr = VecDuplicate(am->Ax,&am->y);CHKERRQ(ierr);
    ierr = VecSet(am->y,0.0);CHKERRQ(ierr);
  }
  if (!am->yold) {
    ierr = VecDuplicate(am->Ax,&am->yold);CHKERRQ(ierr);
    ierr = VecSet(am->yold,0.0);CHKERRQ(ierr);
  }
  if (!am->y0) {
    ierr = VecDuplicate(am->Ax,&am->y0);CHKERRQ(ierr);
    ierr = VecSet(am->y0,0.0);CHKERRQ(ierr);
  }
  if (!am->yhat) {
    ierr = VecDuplicate(am->Ax,&am->yhat);CHKERRQ(ierr);
    ierr = VecSet(am->yhat,0.0);CHKERRQ(ierr);
  }
  if (!am->yhatold) {
    ierr = VecDuplicate(am->Ax,&am->yhatold);CHKERRQ(ierr);
    ierr = VecSet(am->yhatold,0.0);CHKERRQ(ierr);
  }
  if (!am->residual) {
    ierr = VecDuplicate(am->Ax,&am->residual);CHKERRQ(ierr);
    ierr = VecSet(am->residual,0.0);CHKERRQ(ierr);
  }
  if (!am->constraint) {
    am->constraint = NULL;
  } else {
    ierr = VecGetSize(am->constraint,&M);CHKERRQ(ierr);
    PetscCheckFalse(M != N,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"Solution vector and constraint vector must be of same size!");
  }

  /* Save changed tao tolerance for adaptive tolerance */
  if (tao->gatol_changed) {
    am->gatol_admm = tao->gatol;
  }
  if (tao->catol_changed) {
    am->catol_admm = tao->catol;
  }

  /*Update spectral and dual elements to X subsolver */
  ierr = TaoSetObjectiveAndGradient(am->subsolverX, NULL, SubObjGradUpdate, tao);CHKERRQ(ierr);
  ierr = TaoSetJacobianEqualityRoutine(am->subsolverX,am->JA,am->JApre, am->ops->misfitjac, am->misfitjacobianP);CHKERRQ(ierr);
  ierr = TaoSetJacobianEqualityRoutine(am->subsolverZ,am->JB,am->JBpre, am->ops->regjac, am->regjacobianP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_ADMM(Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&am->z);CHKERRQ(ierr);
  ierr = VecDestroy(&am->Ax);CHKERRQ(ierr);
  ierr = VecDestroy(&am->Axold);CHKERRQ(ierr);
  ierr = VecDestroy(&am->Bz);CHKERRQ(ierr);
  ierr = VecDestroy(&am->Bzold);CHKERRQ(ierr);
  ierr = VecDestroy(&am->Bz0);CHKERRQ(ierr);
  ierr = VecDestroy(&am->residual);CHKERRQ(ierr);
  ierr = VecDestroy(&am->y);CHKERRQ(ierr);
  ierr = VecDestroy(&am->yold);CHKERRQ(ierr);
  ierr = VecDestroy(&am->y0);CHKERRQ(ierr);
  ierr = VecDestroy(&am->yhat);CHKERRQ(ierr);
  ierr = VecDestroy(&am->yhatold);CHKERRQ(ierr);
  ierr = VecDestroy(&am->workLeft);CHKERRQ(ierr);
  ierr = VecDestroy(&am->workJacobianRight);CHKERRQ(ierr);
  ierr = VecDestroy(&am->workJacobianRight2);CHKERRQ(ierr);

  ierr = MatDestroy(&am->JA);CHKERRQ(ierr);
  ierr = MatDestroy(&am->JB);CHKERRQ(ierr);
  if (!am->xJI) {
    ierr = MatDestroy(&am->JApre);CHKERRQ(ierr);
  }
  if (!am->zJI) {
    ierr = MatDestroy(&am->JBpre);CHKERRQ(ierr);
  }
  if (am->Hx) {
    ierr = MatDestroy(&am->Hx);CHKERRQ(ierr);
    ierr = MatDestroy(&am->Hxpre);CHKERRQ(ierr);
  }
  if (am->Hz) {
    ierr = MatDestroy(&am->Hz);CHKERRQ(ierr);
    ierr = MatDestroy(&am->Hzpre);CHKERRQ(ierr);
  }
  ierr       = MatDestroy(&am->ATA);CHKERRQ(ierr);
  ierr       = MatDestroy(&am->BTB);CHKERRQ(ierr);
  ierr       = TaoDestroy(&am->subsolverX);CHKERRQ(ierr);
  ierr       = TaoDestroy(&am->subsolverZ);CHKERRQ(ierr);
  am->parent = NULL;
  ierr       = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC

  TAOADMM - Alternating direction method of multipliers method fo solving linear problems with
            constraints. in a min_x f(x) + g(z)  s.t. Ax+Bz=c.
            This algorithm employs two sub Tao solvers, of which type can be specified
            by the user. User need to provide ObjectiveAndGradient routine, and/or HessianRoutine for both subsolvers.
            Hessians can be given boolean flag determining whether they change with respect to a input vector. This can be set via
            TaoADMMSet{Misfit,Regularizer}HessianChangeStatus.
            Second subsolver does support TAOSHELL. It should be noted that L1-norm is used for objective value for TAOSHELL type.
            There is option to set regularizer option, and currently soft-threshold is implemented. For spectral penalty update,
            currently there are basic option and adaptive option.
            Constraint is set at Ax+Bz=c, and A and B can be set with TaoADMMSet{Misfit,Regularizer}ConstraintJacobian.
            c can be set with TaoADMMSetConstraintVectorRHS.
            The user can also provide regularizer weight for second subsolver.

  References:
. * - Xu, Zheng and Figueiredo, Mario A. T. and Yuan, Xiaoming and Studer, Christoph and Goldstein, Tom
          "Adaptive Relaxed ADMM: Convergence Theory and Practical Implementation"
          The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July, 2017.

  Options Database Keys:
+ -tao_admm_regularizer_coefficient        - regularizer constant (default 1.e-6)
. -tao_admm_spectral_penalty               - Constant for Augmented Lagrangian term (default 1.)
. -tao_admm_relaxation_parameter           - relaxation parameter for Z update (default 1.)
. -tao_admm_tolerance_update_factor        - ADMM dynamic tolerance update factor (default 1.e-12)
. -tao_admm_spectral_penalty_update_factor - ADMM spectral penalty update curvature safeguard value (default 0.2)
. -tao_admm_minimum_spectral_penalty       - Set ADMM minimum spectral penalty (default 0)
. -tao_admm_dual_update                    - Lagrangian dual update policy ("basic","adaptive","adaptive-relaxed") (default "basic")
- -tao_admm_regularizer_type               - ADMM regularizer update rule ("user","soft-threshold") (default "soft-threshold")

  Level: beginner

.seealso: TaoADMMSetMisfitHessianChangeStatus(), TaoADMMSetRegHessianChangeStatus(), TaoADMMGetSpectralPenalty(),
          TaoADMMGetMisfitSubsolver(), TaoADMMGetRegularizationSubsolver(), TaoADMMSetConstraintVectorRHS(),
          TaoADMMSetMinimumSpectralPenalty(), TaoADMMSetRegularizerCoefficient(),
          TaoADMMSetRegularizerConstraintJacobian(), TaoADMMSetMisfitConstraintJacobian(),
          TaoADMMSetMisfitObjectiveAndGradientRoutine(), TaoADMMSetMisfitHessianRoutine(),
          TaoADMMSetRegularizerObjectiveAndGradientRoutine(), TaoADMMSetRegularizerHessianRoutine(),
          TaoGetADMMParentTao(), TaoADMMGetDualVector(), TaoADMMSetRegularizerType(),
          TaoADMMGetRegularizerType(), TaoADMMSetUpdateType(), TaoADMMGetUpdateType()
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_ADMM(Tao tao)
{
  TAO_ADMM       *am;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,&am);CHKERRQ(ierr);

  tao->ops->destroy        = TaoDestroy_ADMM;
  tao->ops->setup          = TaoSetUp_ADMM;
  tao->ops->setfromoptions = TaoSetFromOptions_ADMM;
  tao->ops->view           = TaoView_ADMM;
  tao->ops->solve          = TaoSolve_ADMM;

  tao->data           = (void*)am;
  am->l1epsilon       = 1e-6;
  am->lambda          = 1e-4;
  am->mu              = 1.;
  am->muold           = 0.;
  am->mueps           = PETSC_MACHINE_EPSILON;
  am->mumin           = 0.;
  am->orthval         = 0.2;
  am->T               = 2;
  am->parent          = tao;
  am->update          = TAO_ADMM_UPDATE_BASIC;
  am->regswitch       = TAO_ADMM_REGULARIZER_SOFT_THRESH;
  am->tol             = PETSC_SMALL;
  am->const_norm      = 0;
  am->resnorm         = 0;
  am->dualres         = 0;
  am->ops->regobjgrad = NULL;
  am->ops->reghess    = NULL;
  am->gamma           = 1;
  am->regobjgradP     = NULL;
  am->reghessP        = NULL;
  am->gatol_admm      = 1e-8;
  am->catol_admm      = 0;
  am->Hxchange        = PETSC_TRUE;
  am->Hzchange        = PETSC_TRUE;
  am->Hzbool          = PETSC_TRUE;
  am->Hxbool          = PETSC_TRUE;

  ierr = TaoCreate(PetscObjectComm((PetscObject)tao),&am->subsolverX);CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(am->subsolverX,"misfit_");CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)am->subsolverX,(PetscObject)tao,1);CHKERRQ(ierr);
  ierr = TaoCreate(PetscObjectComm((PetscObject)tao),&am->subsolverZ);CHKERRQ(ierr);
  ierr = TaoSetOptionsPrefix(am->subsolverZ,"reg_");CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)am->subsolverZ,(PetscObject)tao,1);CHKERRQ(ierr);

  ierr = TaoSetType(am->subsolverX,TAONLS);CHKERRQ(ierr);
  ierr = TaoSetType(am->subsolverZ,TAONLS);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)am->subsolverX,"TaoGetADMMParentTao_ADMM", (PetscObject) tao);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)am->subsolverZ,"TaoGetADMMParentTao_ADMM", (PetscObject) tao);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoADMMSetRegularizerType_C",TaoADMMSetRegularizerType_ADMM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoADMMGetRegularizerType_C",TaoADMMGetRegularizerType_ADMM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoADMMSetUpdateType_C",TaoADMMSetUpdateType_ADMM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)tao,"TaoADMMGetUpdateType_C",TaoADMMGetUpdateType_ADMM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TaoADMMSetMisfitHessianChangeStatus - Set boolean that determines  whether Hessian matrix of misfit subsolver changes with respect to input vector.

  Collective on Tao

  Input Parameters:
+  tao - the Tao solver context.
-  b - the Hessian matrix change status boolean, PETSC_FALSE  when the Hessian matrix does not change, TRUE otherwise.

  Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMSetMisfitHessianChangeStatus(Tao tao, PetscBool b)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  am->Hxchange = b;
  PetscFunctionReturn(0);
}

/*@
  TaoADMMSetRegHessianChangeStatus - Set boolean that determines whether Hessian matrix of regularization subsolver changes with respect to input vector.

  Collective on Tao

  Input Parameters:
+  tao - the Tao solver context
-  b - the Hessian matrix change status boolean, PETSC_FALSE when the Hessian matrix does not change, TRUE otherwise.

  Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMSetRegHessianChangeStatus(Tao tao, PetscBool b)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  am->Hzchange = b;
  PetscFunctionReturn(0);
}

/*@
  TaoADMMSetSpectralPenalty - Set the spectral penalty (mu) value

  Collective on Tao

  Input Parameters:
+  tao - the Tao solver context
-  mu - spectral penalty

  Level: advanced

.seealso: TaoADMMSetMinimumSpectralPenalty(), TAOADMM
@*/
PetscErrorCode TaoADMMSetSpectralPenalty(Tao tao, PetscReal mu)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  am->mu = mu;
  PetscFunctionReturn(0);
}

/*@
  TaoADMMGetSpectralPenalty - Get the spectral penalty (mu) value

  Collective on Tao

  Input Parameter:
.  tao - the Tao solver context

  Output Parameter:
.  mu - spectral penalty

  Level: advanced

.seealso: TaoADMMSetMinimumSpectralPenalty(), TaoADMMSetSpectralPenalty(), TAOADMM
@*/
PetscErrorCode TaoADMMGetSpectralPenalty(Tao tao, PetscReal *mu)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidRealPointer(mu,2);
  *mu = am->mu;
  PetscFunctionReturn(0);
}

/*@
  TaoADMMGetMisfitSubsolver - Get the pointer to the misfit subsolver inside ADMM

  Collective on Tao

  Input Parameter:
.  tao - the Tao solver context

   Output Parameter:
.  misfit - the Tao subsolver context

  Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMGetMisfitSubsolver(Tao tao, Tao *misfit)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  *misfit = am->subsolverX;
  PetscFunctionReturn(0);
}

/*@
  TaoADMMGetRegularizationSubsolver - Get the pointer to the regularization subsolver inside ADMM

  Collective on Tao

  Input Parameter:
.  tao - the Tao solver context

  Output Parameter:
.  reg - the Tao subsolver context

  Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMGetRegularizationSubsolver(Tao tao, Tao *reg)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  *reg = am->subsolverZ;
  PetscFunctionReturn(0);
}

/*@
  TaoADMMSetConstraintVectorRHS - Set the RHS constraint vector for ADMM

  Collective on Tao

  Input Parameters:
+ tao - the Tao solver context
- c - RHS vector

  Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMSetConstraintVectorRHS(Tao tao, Vec c)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  am->constraint = c;
  PetscFunctionReturn(0);
}

/*@
  TaoADMMSetMinimumSpectralPenalty - Set the minimum value for the spectral penalty

  Collective on Tao

  Input Parameters:
+  tao - the Tao solver context
-  mu  - minimum spectral penalty value

  Level: advanced

.seealso: TaoADMMGetSpectralPenalty(), TAOADMM
@*/
PetscErrorCode TaoADMMSetMinimumSpectralPenalty(Tao tao, PetscReal mu)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  am->mumin= mu;
  PetscFunctionReturn(0);
}

/*@
  TaoADMMSetRegularizerCoefficient - Set the regularization coefficient lambda for L1 norm regularization case

  Collective on Tao

  Input Parameters:
+  tao - the Tao solver context
-  lambda - L1-norm regularizer coefficient

  Level: advanced

.seealso: TaoADMMSetMisfitConstraintJacobian(), TaoADMMSetRegularizerConstraintJacobian(), TAOADMM

@*/
PetscErrorCode TaoADMMSetRegularizerCoefficient(Tao tao, PetscReal lambda)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  am->lambda = lambda;
  PetscFunctionReturn(0);
}

/*@C
  TaoADMMSetMisfitConstraintJacobian - Set the constraint matrix B for the ADMM algorithm. Matrix B constrains the z variable.

  Collective on Tao

  Input Parameters:
+ tao - the Tao solver context
. J - user-created regularizer constraint Jacobian matrix
. Jpre - user-created regularizer Jacobian constraint preconditioner matrix
. func - function pointer for the regularizer constraint Jacobian update function
- ctx - user context for the regularizer Hessian

  Level: advanced

.seealso: TaoADMMSetRegularizerCoefficient(), TaoADMMSetRegularizerConstraintJacobian(), TAOADMM

@*/
PetscErrorCode TaoADMMSetMisfitConstraintJacobian(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (ctx)  am->misfitjacobianP = ctx;
  if (func) am->ops->misfitjac  = func;

  if (J) {
    ierr   = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr   = MatDestroy(&am->JA);CHKERRQ(ierr);
    am->JA = J;
  }
  if (Jpre) {
    ierr      = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr      = MatDestroy(&am->JApre);CHKERRQ(ierr);
    am->JApre = Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoADMMSetRegularizerConstraintJacobian - Set the constraint matrix B for ADMM algorithm. Matrix B constraints z variable.

  Collective on Tao

  Input Parameters:
+ tao - the Tao solver context
. J - user-created regularizer constraint Jacobian matrix
. Jpre - user-created regularizer Jacobian constraint preconditioner matrix
. func - function pointer for the regularizer constraint Jacobian update function
- ctx - user context for the regularizer Hessian

  Level: advanced

.seealso: TaoADMMSetRegularizerCoefficient(), TaoADMMSetMisfitConstraintJacobian(), TAOADMM

@*/
PetscErrorCode TaoADMMSetRegularizerConstraintJacobian(Tao tao, Mat J, Mat Jpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (J) {
    PetscValidHeaderSpecific(J,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,J,2);
  }
  if (Jpre) {
    PetscValidHeaderSpecific(Jpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Jpre,3);
  }
  if (ctx)  am->regjacobianP = ctx;
  if (func) am->ops->regjac  = func;

  if (J) {
    ierr   = PetscObjectReference((PetscObject)J);CHKERRQ(ierr);
    ierr   = MatDestroy(&am->JB);CHKERRQ(ierr);
    am->JB = J;
  }
  if (Jpre) {
    ierr      = PetscObjectReference((PetscObject)Jpre);CHKERRQ(ierr);
    ierr      = MatDestroy(&am->JBpre);CHKERRQ(ierr);
    am->JBpre = Jpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoADMMSetMisfitObjectiveAndGradientRoutine - Sets the user-defined misfit call-back function

   Collective on tao

   Input Parameters:
+    tao - the Tao context
.    func - function pointer for the misfit value and gradient evaluation
-    ctx - user context for the misfit

   Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMSetMisfitObjectiveAndGradientRoutine(Tao tao, PetscErrorCode (*func)(Tao, Vec, PetscReal*, Vec, void*), void *ctx)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  am->misfitobjgradP     = ctx;
  am->ops->misfitobjgrad = func;
  PetscFunctionReturn(0);
}

/*@C
   TaoADMMSetMisfitHessianRoutine - Sets the user-defined misfit Hessian call-back
   function into the algorithm, to be used for subsolverX.

   Collective on tao

   Input Parameters:
   + tao - the Tao context
   . H - user-created matrix for the Hessian of the misfit term
   . Hpre - user-created matrix for the preconditioner of Hessian of the misfit term
   . func - function pointer for the misfit Hessian evaluation
   - ctx - user context for the misfit Hessian

   Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMSetMisfitHessianRoutine(Tao tao, Mat H, Mat Hpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (H) {
    PetscValidHeaderSpecific(H,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,H,2);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Hpre,3);
  }
  if (ctx) {
    am->misfithessP = ctx;
  }
  if (func) {
    am->ops->misfithess = func;
  }
  if (H) {
    ierr   = PetscObjectReference((PetscObject)H);CHKERRQ(ierr);
    ierr   = MatDestroy(&am->Hx);CHKERRQ(ierr);
    am->Hx = H;
  }
  if (Hpre) {
    ierr      = PetscObjectReference((PetscObject)Hpre);CHKERRQ(ierr);
    ierr      = MatDestroy(&am->Hxpre);CHKERRQ(ierr);
    am->Hxpre = Hpre;
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoADMMSetRegularizerObjectiveAndGradientRoutine - Sets the user-defined regularizer call-back function

   Collective on tao

   Input Parameters:
   + tao - the Tao context
   . func - function pointer for the regularizer value and gradient evaluation
   - ctx - user context for the regularizer

   Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMSetRegularizerObjectiveAndGradientRoutine(Tao tao, PetscErrorCode (*func)(Tao, Vec, PetscReal*, Vec, void*), void *ctx)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  am->regobjgradP     = ctx;
  am->ops->regobjgrad = func;
  PetscFunctionReturn(0);
}

/*@C
   TaoADMMSetRegularizerHessianRoutine - Sets the user-defined regularizer Hessian call-back
   function, to be used for subsolverZ.

   Collective on tao

   Input Parameters:
   + tao - the Tao context
   . H - user-created matrix for the Hessian of the regularization term
   . Hpre - user-created matrix for the preconditioner of Hessian of the regularization term
   . func - function pointer for the regularizer Hessian evaluation
   - ctx - user context for the regularizer Hessian

   Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMSetRegularizerHessianRoutine(Tao tao, Mat H, Mat Hpre, PetscErrorCode (*func)(Tao, Vec, Mat, Mat, void*), void *ctx)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (H) {
    PetscValidHeaderSpecific(H,MAT_CLASSID,2);
    PetscCheckSameComm(tao,1,H,2);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre,MAT_CLASSID,3);
    PetscCheckSameComm(tao,1,Hpre,3);
  }
  if (ctx) {
    am->reghessP = ctx;
  }
  if (func) {
    am->ops->reghess = func;
  }
  if (H) {
    ierr   = PetscObjectReference((PetscObject)H);CHKERRQ(ierr);
    ierr   = MatDestroy(&am->Hz);CHKERRQ(ierr);
    am->Hz = H;
  }
  if (Hpre) {
    ierr      = PetscObjectReference((PetscObject)Hpre);CHKERRQ(ierr);
    ierr      = MatDestroy(&am->Hzpre);CHKERRQ(ierr);
    am->Hzpre = Hpre;
  }
  PetscFunctionReturn(0);
}

/*@
   TaoGetADMMParentTao - Gets pointer to parent ADMM tao, used by inner subsolver.

   Collective on tao

   Input Parameter:
   . tao - the Tao context

   Output Parameter:
   . admm_tao - the parent Tao context

   Level: advanced

.seealso: TAOADMM

@*/
PetscErrorCode TaoGetADMMParentTao(Tao tao, Tao *admm_tao)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  ierr = PetscObjectQuery((PetscObject)tao,"TaoGetADMMParentTao_ADMM", (PetscObject*) admm_tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TaoADMMGetDualVector - Returns the dual vector associated with the current TAOADMM state

  Not Collective

  Input Parameter:
  . tao - the Tao context

  Output Parameter:
  . Y - the current solution

  Level: intermediate

.seealso: TAOADMM

@*/
PetscErrorCode TaoADMMGetDualVector(Tao tao, Vec *Y)
{
  TAO_ADMM *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  *Y = am->y;
  PetscFunctionReturn(0);
}

/*@
  TaoADMMSetRegularizerType - Set regularizer type for ADMM routine

  Not Collective

  Input Parameters:
+ tao  - the Tao context
- type - regularizer type

  Options Database:
.  -tao_admm_regularizer_type <admm_regularizer_user,admm_regularizer_soft_thresh>

  Level: intermediate

.seealso: TaoADMMGetRegularizerType(), TaoADMMRegularizerType, TAOADMM
@*/
PetscErrorCode TaoADMMSetRegularizerType(Tao tao, TaoADMMRegularizerType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveEnum(tao,type,2);
  ierr = PetscTryMethod(tao,"TaoADMMSetRegularizerType_C",(Tao,TaoADMMRegularizerType),(tao,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TaoADMMGetRegularizerType - Gets the type of regularizer routine for ADMM

   Not Collective

   Input Parameter:
.  tao - the Tao context

   Output Parameter:
.  type - the type of regularizer

  Options Database:
.  -tao_admm_regularizer_type <admm_regularizer_user,admm_regularizer_soft_thresh>

   Level: intermediate

.seealso: TaoADMMSetRegularizerType(), TaoADMMRegularizerType, TAOADMM
@*/
PetscErrorCode TaoADMMGetRegularizerType(Tao tao, TaoADMMRegularizerType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  ierr = PetscUseMethod(tao,"TaoADMMGetRegularizerType_C",(Tao,TaoADMMRegularizerType*),(tao,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TaoADMMSetUpdateType - Set update routine for ADMM routine

  Not Collective

  Input Parameters:
+ tao  - the Tao context
- type - spectral parameter update type

  Level: intermediate

.seealso: TaoADMMGetUpdateType(), TaoADMMUpdateType, TAOADMM
@*/
PetscErrorCode TaoADMMSetUpdateType(Tao tao, TaoADMMUpdateType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveEnum(tao,type,2);
  ierr = PetscTryMethod(tao,"TaoADMMSetUpdateType_C",(Tao,TaoADMMUpdateType),(tao,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   TaoADMMGetUpdateType - Gets the type of spectral penalty update routine for ADMM

   Not Collective

   Input Parameter:
.  tao - the Tao context

   Output Parameter:
.  type - the type of spectral penalty update routine

   Level: intermediate

.seealso: TaoADMMSetUpdateType(), TaoADMMUpdateType, TAOADMM
@*/
PetscErrorCode TaoADMMGetUpdateType(Tao tao, TaoADMMUpdateType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  ierr = PetscUseMethod(tao,"TaoADMMGetUpdateType_C",(Tao,TaoADMMUpdateType*),(tao,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
