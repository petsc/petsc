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
  PetscReal      Axnorm,Bznorm,ATynorm,temp;
  Vec            tempJR,tempL;
  Tao            mis;

  PetscFunctionBegin;
  mis    = am->subsolverX;
  tempJR = am->workJacobianRight;
  tempL  = am->workLeft;
  /* ATy */
  PetscCall(TaoComputeJacobianEquality(mis, am->y, mis->jacobian_equality, mis->jacobian_equality_pre));
  PetscCall(MatMultTranspose(mis->jacobian_equality,am->y,tempJR));
  PetscCall(VecNorm(tempJR,NORM_2,&ATynorm));
  /* dualres = mu * ||AT(Bz-Bzold)||_2 */
  PetscCall(VecWAXPY(tempJR,-1.,am->Bzold,am->Bz));
  PetscCall(MatMultTranspose(mis->jacobian_equality,tempJR,tempL));
  PetscCall(VecNorm(tempL,NORM_2,&am->dualres));
  am->dualres *= am->mu;

  /* ||Ax||_2, ||Bz||_2 */
  PetscCall(VecNorm(am->Ax,NORM_2,&Axnorm));
  PetscCall(VecNorm(am->Bz,NORM_2,&Bznorm));

  /* Set catol to be catol_admm *  max{||Ax||,||Bz||,||c||} *
   * Set gatol to be gatol_admm *  ||A^Ty|| *
   * while cnorm is ||r_k||_2, and gnorm is ||d_k||_2 */
  temp = am->catol_admm * PetscMax(Axnorm, (!am->const_norm) ? Bznorm : PetscMax(Bznorm,am->const_norm));
  PetscCall(TaoSetConstraintTolerances(tao,temp,PETSC_DEFAULT));
  PetscCall(TaoSetTolerances(tao, am->gatol_admm*ATynorm, PETSC_DEFAULT,PETSC_DEFAULT));
  PetscFunctionReturn(0);
}

/* Penaly Update for Adaptive ADMM. */
static PetscErrorCode AdaptiveADMMPenaltyUpdate(Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
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

  PetscCall(VecWAXPY(tempJR,-1.,am->Axold,am->Ax));
  PetscCall(VecWAXPY(tempJR2,-1.,am->yhatold,am->yhat));
  PetscCall(VecNorm(tempJR,NORM_2,&Axdiff_norm));
  PetscCall(VecNorm(tempJR2,NORM_2,&yhatdiff_norm));
  PetscCall(VecDot(tempJR,tempJR2,&Axyhat));

  PetscCall(VecWAXPY(tempJR,-1.,am->Bz0,am->Bz));
  PetscCall(VecWAXPY(tempJR2,-1.,am->y,am->y0));
  PetscCall(VecNorm(tempJR,NORM_2,&Bzdiff_norm));
  PetscCall(VecNorm(tempJR2,NORM_2,&ydiff_norm));
  PetscCall(VecDot(tempJR,tempJR2,&Bzy));

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

  PetscFunctionBegin;
  mis  = am->subsolverX;
  reg  = am->subsolverZ;
  PetscCall(TaoComputeJacobianEquality(mis, x, mis->jacobian_equality, mis->jacobian_equality_pre));
  PetscCall(MatMult(mis->jacobian_equality,x,Ax));
  PetscCall(TaoComputeJacobianEquality(reg, z, reg->jacobian_equality, reg->jacobian_equality_pre));
  PetscCall(MatMult(reg->jacobian_equality,z,Bz));

  PetscCall(VecWAXPY(residual,1.,Bz,Ax));
  if (am->constraint != NULL) {
    PetscCall(VecAXPY(residual,-1.,am->constraint));
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
  PetscReal      temp,temp2;
  Vec            tempJR;

  PetscFunctionBegin;
  tempJR = am->workJacobianRight;
  PetscCall(ADMMUpdateConstraintResidualVector(parent, x, am->subsolverZ->solution, am->Ax, am->Bz, am->residual));
  PetscCall((*am->ops->misfitobjgrad)(am->subsolverX,x,f,g,am->misfitobjgradP));

  am->last_misfit_val = *f;
  /* Objective  Add + yT(Ax+Bz-c) + mu/2*||Ax+Bz-c||_2^2 */
  PetscCall(VecTDot(am->residual,am->y,&temp));
  PetscCall(VecTDot(am->residual,am->residual,&temp2));
  *f   += temp + (am->mu/2)*temp2;

  /* Gradient. Add + mu*AT(Ax+Bz-c) + yTA*/
  PetscCall(MatMultTranspose(tao->jacobian_equality,am->residual,tempJR));
  PetscCall(VecAXPY(g,am->mu,tempJR));
  PetscCall(MatMultTranspose(tao->jacobian_equality,am->y,tempJR));
  PetscCall(VecAXPY(g,1.,tempJR));
  PetscFunctionReturn(0);
}

/* Updates Augmented Lagrangians to given routines
 * For subsolverZ, routine needs to be ComputeObjectiveAndGraidnet
 * Separate Objective and Gradient routines are not supported.  */
static PetscErrorCode RegObjGradUpdate(Tao tao, Vec z, PetscReal *f, Vec g, void *ptr)
{
  Tao            parent = (Tao)ptr;
  TAO_ADMM       *am    = (TAO_ADMM*)parent->data;
  PetscReal      temp,temp2;
  Vec            tempJR;

  PetscFunctionBegin;
  tempJR = am->workJacobianRight;
  PetscCall(ADMMUpdateConstraintResidualVector(parent, am->subsolverX->solution, z, am->Ax, am->Bz, am->residual));
  PetscCall((*am->ops->regobjgrad)(am->subsolverZ,z,f,g,am->regobjgradP));
  am->last_reg_val= *f;
  /* Objective  Add  + yT(Ax+Bz-c) + mu/2*||Ax+Bz-c||_2^2 */
  PetscCall(VecTDot(am->residual,am->y,&temp));
  PetscCall(VecTDot(am->residual,am->residual,&temp2));
  *f   += temp + (am->mu/2)*temp2;

  /* Gradient. Add + mu*BT(Ax+Bz-c) + yTB*/
  PetscCall(MatMultTranspose(am->subsolverZ->jacobian_equality,am->residual,tempJR));
  PetscCall(VecAXPY(g,am->mu,tempJR));
  PetscCall(MatMultTranspose(am->subsolverZ->jacobian_equality,am->y,tempJR));
  PetscCall(VecAXPY(g,1.,tempJR));
  PetscFunctionReturn(0);
}

/* Computes epsilon padded L1 norm lambda*sum(sqrt(x^2+eps^2)-eps */
static PetscErrorCode ADMML1EpsilonNorm(Tao tao, Vec x, PetscReal eps, PetscReal *norm)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscInt       N;

  PetscFunctionBegin;
  PetscCall(VecGetSize(am->workLeft,&N));
  PetscCall(VecPointwiseMult(am->workLeft,x,x));
  PetscCall(VecShift(am->workLeft,am->l1epsilon*am->l1epsilon));
  PetscCall(VecSqrtAbs(am->workLeft));
  PetscCall(VecSum(am->workLeft,norm));
  *norm += N*am->l1epsilon;
  *norm *= am->lambda;
  PetscFunctionReturn(0);
}

static PetscErrorCode ADMMInternalHessianUpdate(Mat H, Mat Constraint, PetscBool Identity, void *ptr)
{
  TAO_ADMM       *am = (TAO_ADMM*)ptr;

  PetscFunctionBegin;
  switch (am->update) {
  case (TAO_ADMM_UPDATE_BASIC):
    break;
  case (TAO_ADMM_UPDATE_ADAPTIVE):
  case (TAO_ADMM_UPDATE_ADAPTIVE_RELAXED):
    if (H && (am->muold != am->mu)) {
      if (!Identity) {
        PetscCall(MatAXPY(H,am->mu-am->muold,Constraint,DIFFERENT_NONZERO_PATTERN));
      } else {
        PetscCall(MatShift(H,am->mu-am->muold));
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

  PetscFunctionBegin;
  if (am->Hxchange) {
    /* Case where Hessian gets updated with respect to x vector input. */
    PetscCall((*am->ops->misfithess)(am->subsolverX,x,H,Hpre,am->misfithessP));
    PetscCall(ADMMInternalHessianUpdate(am->subsolverX->hessian,am->ATA,am->xJI,am));
  } else if (am->Hxbool) {
    /* Hessian doesn't get updated. H(x) = c */
    /* Update Lagrangian only only per TAO call */
    PetscCall(ADMMInternalHessianUpdate(am->subsolverX->hessian,am->ATA,am->xJI,am));
    am->Hxbool = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/* Same as SubHessianUpdate, except for B matrix instead of A matrix */
static PetscErrorCode RegHessianUpdate(Tao tao, Vec z, Mat H, Mat Hpre, void *ptr)
{
  Tao            parent = (Tao)ptr;
  TAO_ADMM       *am    = (TAO_ADMM*)parent->data;

  PetscFunctionBegin;

  if (am->Hzchange) {
    /* Case where Hessian gets updated with respect to x vector input. */
    PetscCall((*am->ops->reghess)(am->subsolverZ,z,H,Hpre,am->reghessP));
    PetscCall(ADMMInternalHessianUpdate(am->subsolverZ->hessian,am->BTB,am->zJI,am));
  } else if (am->Hzbool) {
    /* Hessian doesn't get updated. H(x) = c */
    /* Update Lagrangian only only per TAO call */
    PetscCall(ADMMInternalHessianUpdate(am->subsolverZ->hessian,am->BTB,am->zJI,am));
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
  PetscFunctionBegin;
  PetscCall(VecCopy(in,out));
  PetscFunctionReturn(0);
}

/* Shell Matrix routine for B matrix.
 * This gets used when user puts NULL for
 * TaoADMMSetRegularizerConstraintJacobian(tao, NULL,NULL, ...)
 * Sets B=-I */
static PetscErrorCode JacobianIdentityB(Mat mat,Vec in,Vec out)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(in,out));
  PetscCall(VecScale(out,-1.));
  PetscFunctionReturn(0);
}

/* Solve f(x) + g(z) s.t. Ax + Bz = c */
static PetscErrorCode TaoSolve_ADMM(Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscInt       N;
  PetscReal      reg_func;
  PetscBool      is_reg_shell;
  Vec            tempL;

  PetscFunctionBegin;
  if (am->regswitch != TAO_ADMM_REGULARIZER_SOFT_THRESH) {
    PetscCheck(am->subsolverX->ops->computejacobianequality,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"Must call TaoADMMSetMisfitConstraintJacobian() first");
    PetscCheck(am->subsolverZ->ops->computejacobianequality,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"Must call TaoADMMSetRegularizerConstraintJacobian() first");
    if (am->constraint != NULL) {
      PetscCall(VecNorm(am->constraint,NORM_2,&am->const_norm));
    }
  }
  tempL = am->workLeft;
  PetscCall(VecGetSize(tempL,&N));

  if (am->Hx && am->ops->misfithess) {
    PetscCall(TaoSetHessian(am->subsolverX, am->Hx, am->Hx, SubHessianUpdate, tao));
  }

  if (!am->zJI) {
    /* Currently, B is assumed to be a linear system, i.e., not getting updated*/
    PetscCall(MatTransposeMatMult(am->JB,am->JB,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&(am->BTB)));
  }
  if (!am->xJI) {
    /* Currently, A is assumed to be a linear system, i.e., not getting updated*/
    PetscCall(MatTransposeMatMult(am->subsolverX->jacobian_equality,am->subsolverX->jacobian_equality,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&(am->ATA)));
  }

  is_reg_shell = PETSC_FALSE;

  PetscCall(PetscObjectTypeCompare((PetscObject)am->subsolverZ, TAOSHELL, &is_reg_shell));

  if (!is_reg_shell) {
    switch (am->regswitch) {
    case (TAO_ADMM_REGULARIZER_USER):
      PetscCheck(am->ops->regobjgrad,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"Must call TaoADMMSetRegularizerObjectiveAndGradientRoutine() first if one wishes to use TAO_ADMM_REGULARIZER_USER with non-TAOSHELL type");
      break;
    case (TAO_ADMM_REGULARIZER_SOFT_THRESH):
      /* Soft Threshold. */
      break;
    }
    if (am->ops->regobjgrad) {
      PetscCall(TaoSetObjectiveAndGradient(am->subsolverZ, NULL, RegObjGradUpdate, tao));
    }
    if (am->Hz && am->ops->reghess) {
      PetscCall(TaoSetHessian(am->subsolverZ, am->Hz, am->Hzpre, RegHessianUpdate, tao));
    }
  }

  switch (am->update) {
  case TAO_ADMM_UPDATE_BASIC:
    if (am->subsolverX->hessian) {
      /* In basic case, Hessian does not get updated w.r.t. to spectral penalty
       * Here, when A is set, i.e., am->xJI, add mu*ATA to Hessian*/
      if (!am->xJI) {
        PetscCall(MatAXPY(am->subsolverX->hessian,am->mu,am->ATA,DIFFERENT_NONZERO_PATTERN));
      } else {
        PetscCall(MatShift(am->subsolverX->hessian,am->mu));
      }
    }
    if (am->subsolverZ->hessian && am->regswitch == TAO_ADMM_REGULARIZER_USER) {
      if (am->regswitch == TAO_ADMM_REGULARIZER_USER && !am->zJI) {
        PetscCall(MatAXPY(am->subsolverZ->hessian,am->mu,am->BTB,DIFFERENT_NONZERO_PATTERN));
      } else {
        PetscCall(MatShift(am->subsolverZ->hessian,am->mu));
      }
    }
    break;
  case TAO_ADMM_UPDATE_ADAPTIVE:
  case TAO_ADMM_UPDATE_ADAPTIVE_RELAXED:
    break;
  }

  PetscCall(PetscCitationsRegister(citation,&cited));
  tao->reason = TAO_CONTINUE_ITERATING;

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    if (tao->ops->update) {
      PetscCall((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    PetscCall(VecCopy(am->Bz, am->Bzold));

    /* x update */
    PetscCall(TaoSolve(am->subsolverX));
    PetscCall(TaoComputeJacobianEquality(am->subsolverX, am->subsolverX->solution, am->subsolverX->jacobian_equality, am->subsolverX->jacobian_equality_pre));
    PetscCall(MatMult(am->subsolverX->jacobian_equality, am->subsolverX->solution,am->Ax));

    am->Hxbool = PETSC_TRUE;

    /* z update */
    switch (am->regswitch) {
    case TAO_ADMM_REGULARIZER_USER:
      PetscCall(TaoSolve(am->subsolverZ));
      break;
    case TAO_ADMM_REGULARIZER_SOFT_THRESH:
      /* L1 assumes A,B jacobians are identity nxn matrix */
      PetscCall(VecWAXPY(am->workJacobianRight,1/am->mu,am->y,am->Ax));
      PetscCall(TaoSoftThreshold(am->workJacobianRight,-am->lambda/am->mu,am->lambda/am->mu,am->subsolverZ->solution));
      break;
    }
    am->Hzbool = PETSC_TRUE;
    /* Returns Ax + Bz - c with updated Ax,Bz vectors */
    PetscCall(ADMMUpdateConstraintResidualVector(tao, am->subsolverX->solution, am->subsolverZ->solution, am->Ax, am->Bz, am->residual));
    /* Dual variable, y += y + mu*(Ax+Bz-c) */
    PetscCall(VecWAXPY(am->y, am->mu, am->residual, am->yold));

    /* stopping tolerance update */
    PetscCall(TaoADMMToleranceUpdate(tao));

    /* Updating Spectral Penalty */
    switch (am->update) {
    case TAO_ADMM_UPDATE_BASIC:
      am->muold = am->mu;
      break;
    case TAO_ADMM_UPDATE_ADAPTIVE:
    case TAO_ADMM_UPDATE_ADAPTIVE_RELAXED:
      if (tao->niter == 0) {
        PetscCall(VecCopy(am->y, am->y0));
        PetscCall(VecWAXPY(am->residual, 1., am->Ax, am->Bzold));
        if (am->constraint) {
          PetscCall(VecAXPY(am->residual, -1., am->constraint));
        }
        PetscCall(VecWAXPY(am->yhatold, -am->mu, am->residual, am->yold));
        PetscCall(VecCopy(am->Ax, am->Axold));
        PetscCall(VecCopy(am->Bz, am->Bz0));
        am->muold = am->mu;
      } else if (tao->niter % am->T == 1) {
        /* we have compute Bzold in a previous iteration, and we computed Ax above */
        PetscCall(VecWAXPY(am->residual, 1., am->Ax, am->Bzold));
        if (am->constraint) {
          PetscCall(VecAXPY(am->residual, -1., am->constraint));
        }
        PetscCall(VecWAXPY(am->yhat, -am->mu, am->residual, am->yold));
        PetscCall(AdaptiveADMMPenaltyUpdate(tao));
        PetscCall(VecCopy(am->Ax, am->Axold));
        PetscCall(VecCopy(am->Bz, am->Bz0));
        PetscCall(VecCopy(am->yhat, am->yhatold));
        PetscCall(VecCopy(am->y, am->y0));
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
        PetscCall(ADMML1EpsilonNorm(tao,am->subsolverZ->solution,am->l1epsilon,&reg_func));
      } else {
        (*am->ops->regobjgrad)(am->subsolverZ,am->subsolverX->solution,&reg_func,tempL,am->regobjgradP);
      }
      break;
    case TAO_ADMM_REGULARIZER_SOFT_THRESH:
      PetscCall(ADMML1EpsilonNorm(tao,am->subsolverZ->solution,am->l1epsilon,&reg_func));
      break;
    }
    PetscCall(VecCopy(am->y,am->yold));
    PetscCall(ADMMUpdateConstraintResidualVector(tao, am->subsolverX->solution, am->subsolverZ->solution, am->Ax, am->Bz, am->residual));
    PetscCall(VecNorm(am->residual,NORM_2,&am->resnorm));
    PetscCall(TaoLogConvergenceHistory(tao,am->last_misfit_val + reg_func,am->dualres,am->resnorm,tao->ksp_its));

    PetscCall(TaoMonitor(tao,tao->niter,am->last_misfit_val + reg_func,am->dualres,am->resnorm,1.0));
    PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
  }
  /* Update vectors */
  PetscCall(VecCopy(am->subsolverX->solution,tao->solution));
  PetscCall(VecCopy(am->subsolverX->gradient,tao->gradient));
  PetscCall(PetscObjectCompose((PetscObject)am->subsolverX,"TaoGetADMMParentTao_ADMM", NULL));
  PetscCall(PetscObjectCompose((PetscObject)am->subsolverZ,"TaoGetADMMParentTao_ADMM", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao,"TaoADMMSetRegularizerType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao,"TaoADMMGetRegularizerType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao,"TaoADMMSetUpdateType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao,"TaoADMMGetUpdateType_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_ADMM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"ADMM problem that solves f(x) in a form of f(x) + g(z) subject to x - z = 0. Norm 1 and 2 are supported. Different subsolver routines can be selected. "));
  PetscCall(PetscOptionsReal("-tao_admm_regularizer_coefficient","regularizer constant","",am->lambda,&am->lambda,NULL));
  PetscCall(PetscOptionsReal("-tao_admm_spectral_penalty","Constant for Augmented Lagrangian term.","",am->mu,&am->mu,NULL));
  PetscCall(PetscOptionsReal("-tao_admm_relaxation_parameter","x relaxation parameter for Z update.","",am->gamma,&am->gamma,NULL));
  PetscCall(PetscOptionsReal("-tao_admm_tolerance_update_factor","ADMM dynamic tolerance update factor.","",am->tol,&am->tol,NULL));
  PetscCall(PetscOptionsReal("-tao_admm_spectral_penalty_update_factor","ADMM spectral penalty update curvature safeguard value.","",am->orthval,&am->orthval,NULL));
  PetscCall(PetscOptionsReal("-tao_admm_minimum_spectral_penalty","Set ADMM minimum spectral penalty.","",am->mumin,&am->mumin,NULL));
  ierr = PetscOptionsEnum("-tao_admm_dual_update","Lagrangian dual update policy","TaoADMMUpdateType",
                          TaoADMMUpdateTypes,(PetscEnum)am->update,(PetscEnum*)&am->update,NULL);PetscCall(ierr);
  ierr = PetscOptionsEnum("-tao_admm_regularizer_type","ADMM regularizer update rule","TaoADMMRegularizerType",
                          TaoADMMRegularizerTypes,(PetscEnum)am->regswitch,(PetscEnum*)&am->regswitch,NULL);PetscCall(ierr);
  PetscCall(PetscOptionsTail());
  PetscCall(TaoSetFromOptions(am->subsolverX));
  if (am->regswitch != TAO_ADMM_REGULARIZER_SOFT_THRESH) {
    PetscCall(TaoSetFromOptions(am->subsolverZ));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_ADMM(Tao tao,PetscViewer viewer)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(TaoView(am->subsolverX,viewer));
  PetscCall(TaoView(am->subsolverZ,viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_ADMM(Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;
  PetscInt       n,N,M;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(tao->solution,&n));
  PetscCall(VecGetSize(tao->solution,&N));
  /* If Jacobian is given as NULL, it means Jacobian is identity matrix with size of solution vector */
  if (!am->JB) {
    am->zJI   = PETSC_TRUE;
    PetscCall(MatCreateShell(PetscObjectComm((PetscObject)tao),n,n,PETSC_DETERMINE,PETSC_DETERMINE,NULL,&am->JB));
    PetscCall(MatShellSetOperation(am->JB,MATOP_MULT,(void (*)(void))JacobianIdentityB));
    PetscCall(MatShellSetOperation(am->JB,MATOP_MULT_TRANSPOSE,(void (*)(void))JacobianIdentityB));
    PetscCall(MatShellSetOperation(am->JB,MATOP_TRANSPOSE_MAT_MULT,(void (*)(void))JacobianIdentityB));
    am->JBpre = am->JB;
  }
  if (!am->JA) {
    am->xJI   = PETSC_TRUE;
    PetscCall(MatCreateShell(PetscObjectComm((PetscObject)tao),n,n,PETSC_DETERMINE,PETSC_DETERMINE,NULL,&am->JA));
    PetscCall(MatShellSetOperation(am->JA,MATOP_MULT,(void (*)(void))JacobianIdentity));
    PetscCall(MatShellSetOperation(am->JA,MATOP_MULT_TRANSPOSE,(void (*)(void))JacobianIdentity));
    PetscCall(MatShellSetOperation(am->JA,MATOP_TRANSPOSE_MAT_MULT,(void (*)(void))JacobianIdentity));
    am->JApre = am->JA;
  }
  PetscCall(MatCreateVecs(am->JA,NULL,&am->Ax));
  if (!tao->gradient) {
    PetscCall(VecDuplicate(tao->solution,&tao->gradient));
  }
  PetscCall(TaoSetSolution(am->subsolverX, tao->solution));
  if (!am->z) {
    PetscCall(VecDuplicate(tao->solution,&am->z));
    PetscCall(VecSet(am->z,0.0));
  }
  PetscCall(TaoSetSolution(am->subsolverZ, am->z));
  if (!am->workLeft) {
    PetscCall(VecDuplicate(tao->solution,&am->workLeft));
  }
  if (!am->Axold) {
    PetscCall(VecDuplicate(am->Ax,&am->Axold));
  }
  if (!am->workJacobianRight) {
    PetscCall(VecDuplicate(am->Ax,&am->workJacobianRight));
  }
  if (!am->workJacobianRight2) {
    PetscCall(VecDuplicate(am->Ax,&am->workJacobianRight2));
  }
  if (!am->Bz) {
    PetscCall(VecDuplicate(am->Ax,&am->Bz));
  }
  if (!am->Bzold) {
    PetscCall(VecDuplicate(am->Ax,&am->Bzold));
  }
  if (!am->Bz0) {
    PetscCall(VecDuplicate(am->Ax,&am->Bz0));
  }
  if (!am->y) {
    PetscCall(VecDuplicate(am->Ax,&am->y));
    PetscCall(VecSet(am->y,0.0));
  }
  if (!am->yold) {
    PetscCall(VecDuplicate(am->Ax,&am->yold));
    PetscCall(VecSet(am->yold,0.0));
  }
  if (!am->y0) {
    PetscCall(VecDuplicate(am->Ax,&am->y0));
    PetscCall(VecSet(am->y0,0.0));
  }
  if (!am->yhat) {
    PetscCall(VecDuplicate(am->Ax,&am->yhat));
    PetscCall(VecSet(am->yhat,0.0));
  }
  if (!am->yhatold) {
    PetscCall(VecDuplicate(am->Ax,&am->yhatold));
    PetscCall(VecSet(am->yhatold,0.0));
  }
  if (!am->residual) {
    PetscCall(VecDuplicate(am->Ax,&am->residual));
    PetscCall(VecSet(am->residual,0.0));
  }
  if (!am->constraint) {
    am->constraint = NULL;
  } else {
    PetscCall(VecGetSize(am->constraint,&M));
    PetscCheck(M == N,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"Solution vector and constraint vector must be of same size!");
  }

  /* Save changed tao tolerance for adaptive tolerance */
  if (tao->gatol_changed) {
    am->gatol_admm = tao->gatol;
  }
  if (tao->catol_changed) {
    am->catol_admm = tao->catol;
  }

  /*Update spectral and dual elements to X subsolver */
  PetscCall(TaoSetObjectiveAndGradient(am->subsolverX, NULL, SubObjGradUpdate, tao));
  PetscCall(TaoSetJacobianEqualityRoutine(am->subsolverX,am->JA,am->JApre, am->ops->misfitjac, am->misfitjacobianP));
  PetscCall(TaoSetJacobianEqualityRoutine(am->subsolverZ,am->JB,am->JBpre, am->ops->regjac, am->regjacobianP));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_ADMM(Tao tao)
{
  TAO_ADMM       *am = (TAO_ADMM*)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&am->z));
  PetscCall(VecDestroy(&am->Ax));
  PetscCall(VecDestroy(&am->Axold));
  PetscCall(VecDestroy(&am->Bz));
  PetscCall(VecDestroy(&am->Bzold));
  PetscCall(VecDestroy(&am->Bz0));
  PetscCall(VecDestroy(&am->residual));
  PetscCall(VecDestroy(&am->y));
  PetscCall(VecDestroy(&am->yold));
  PetscCall(VecDestroy(&am->y0));
  PetscCall(VecDestroy(&am->yhat));
  PetscCall(VecDestroy(&am->yhatold));
  PetscCall(VecDestroy(&am->workLeft));
  PetscCall(VecDestroy(&am->workJacobianRight));
  PetscCall(VecDestroy(&am->workJacobianRight2));

  PetscCall(MatDestroy(&am->JA));
  PetscCall(MatDestroy(&am->JB));
  if (!am->xJI) {
    PetscCall(MatDestroy(&am->JApre));
  }
  if (!am->zJI) {
    PetscCall(MatDestroy(&am->JBpre));
  }
  if (am->Hx) {
    PetscCall(MatDestroy(&am->Hx));
    PetscCall(MatDestroy(&am->Hxpre));
  }
  if (am->Hz) {
    PetscCall(MatDestroy(&am->Hz));
    PetscCall(MatDestroy(&am->Hzpre));
  }
  PetscCall(MatDestroy(&am->ATA));
  PetscCall(MatDestroy(&am->BTB));
  PetscCall(TaoDestroy(&am->subsolverX));
  PetscCall(TaoDestroy(&am->subsolverZ));
  am->parent = NULL;
  PetscCall(PetscFree(tao->data));
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

  PetscFunctionBegin;
  PetscCall(PetscNewLog(tao,&am));

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

  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao),&am->subsolverX));
  PetscCall(TaoSetOptionsPrefix(am->subsolverX,"misfit_"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)am->subsolverX,(PetscObject)tao,1));
  PetscCall(TaoCreate(PetscObjectComm((PetscObject)tao),&am->subsolverZ));
  PetscCall(TaoSetOptionsPrefix(am->subsolverZ,"reg_"));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)am->subsolverZ,(PetscObject)tao,1));

  PetscCall(TaoSetType(am->subsolverX,TAONLS));
  PetscCall(TaoSetType(am->subsolverZ,TAONLS));
  PetscCall(PetscObjectCompose((PetscObject)am->subsolverX,"TaoGetADMMParentTao_ADMM", (PetscObject) tao));
  PetscCall(PetscObjectCompose((PetscObject)am->subsolverZ,"TaoGetADMMParentTao_ADMM", (PetscObject) tao));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao,"TaoADMMSetRegularizerType_C",TaoADMMSetRegularizerType_ADMM));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao,"TaoADMMGetRegularizerType_C",TaoADMMGetRegularizerType_ADMM));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao,"TaoADMMSetUpdateType_C",TaoADMMSetUpdateType_ADMM));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao,"TaoADMMGetUpdateType_C",TaoADMMGetUpdateType_ADMM));
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
    PetscCall(PetscObjectReference((PetscObject)J));
    PetscCall(MatDestroy(&am->JA));
    am->JA = J;
  }
  if (Jpre) {
    PetscCall(PetscObjectReference((PetscObject)Jpre));
    PetscCall(MatDestroy(&am->JApre));
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
    PetscCall(PetscObjectReference((PetscObject)J));
    PetscCall(MatDestroy(&am->JB));
    am->JB = J;
  }
  if (Jpre) {
    PetscCall(PetscObjectReference((PetscObject)Jpre));
    PetscCall(MatDestroy(&am->JBpre));
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
    PetscCall(PetscObjectReference((PetscObject)H));
    PetscCall(MatDestroy(&am->Hx));
    am->Hx = H;
  }
  if (Hpre) {
    PetscCall(PetscObjectReference((PetscObject)Hpre));
    PetscCall(MatDestroy(&am->Hxpre));
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
    PetscCall(PetscObjectReference((PetscObject)H));
    PetscCall(MatDestroy(&am->Hz));
    am->Hz = H;
  }
  if (Hpre) {
    PetscCall(PetscObjectReference((PetscObject)Hpre));
    PetscCall(MatDestroy(&am->Hzpre));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(PetscObjectQuery((PetscObject)tao,"TaoGetADMMParentTao_ADMM", (PetscObject*) admm_tao));
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
.  -tao_admm_regularizer_type <admm_regularizer_user,admm_regularizer_soft_thresh> - select the regularizer

  Level: intermediate

.seealso: TaoADMMGetRegularizerType(), TaoADMMRegularizerType, TAOADMM
@*/
PetscErrorCode TaoADMMSetRegularizerType(Tao tao, TaoADMMRegularizerType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveEnum(tao,type,2);
  PetscCall(PetscTryMethod(tao,"TaoADMMSetRegularizerType_C",(Tao,TaoADMMRegularizerType),(tao,type)));
  PetscFunctionReturn(0);
}

/*@
   TaoADMMGetRegularizerType - Gets the type of regularizer routine for ADMM

   Not Collective

   Input Parameter:
.  tao - the Tao context

   Output Parameter:
.  type - the type of regularizer

   Level: intermediate

.seealso: TaoADMMSetRegularizerType(), TaoADMMRegularizerType, TAOADMM
@*/
PetscErrorCode TaoADMMGetRegularizerType(Tao tao, TaoADMMRegularizerType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(PetscUseMethod(tao,"TaoADMMGetRegularizerType_C",(Tao,TaoADMMRegularizerType*),(tao,type)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveEnum(tao,type,2);
  PetscCall(PetscTryMethod(tao,"TaoADMMSetUpdateType_C",(Tao,TaoADMMUpdateType),(tao,type)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(PetscUseMethod(tao,"TaoADMMGetUpdateType_C",(Tao,TaoADMMUpdateType*),(tao,type)));
  PetscFunctionReturn(0);
}
