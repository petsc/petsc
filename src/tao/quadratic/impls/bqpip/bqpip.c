/*
    This file implements a Mehrotra predictor-corrector method for
    bound-constrained quadratic programs.

 */

#include <../src/tao/quadratic/impls/bqpip/bqpipimpl.h>
#include <petscksp.h>

static PetscErrorCode QPIPComputeResidual(TAO_BQPIP *qp,Tao tao)
{
  PetscReal      dtmp = 1.0 - qp->psteplength;

  PetscFunctionBegin;
  /* Compute R3 and R5 */

  CHKERRQ(VecScale(qp->R3,dtmp));
  CHKERRQ(VecScale(qp->R5,dtmp));
  qp->pinfeas=dtmp*qp->pinfeas;

  CHKERRQ(VecCopy(qp->S,tao->gradient));
  CHKERRQ(VecAXPY(tao->gradient,-1.0,qp->Z));

  CHKERRQ(MatMult(tao->hessian,tao->solution,qp->RHS));
  CHKERRQ(VecScale(qp->RHS,-1.0));
  CHKERRQ(VecAXPY(qp->RHS,-1.0,qp->C));
  CHKERRQ(VecAXPY(tao->gradient,-1.0,qp->RHS));

  CHKERRQ(VecNorm(tao->gradient,NORM_1,&qp->dinfeas));
  qp->rnorm=(qp->dinfeas+qp->pinfeas)/(qp->m+qp->n);
  PetscFunctionReturn(0);
}

static PetscErrorCode  QPIPSetInitialPoint(TAO_BQPIP *qp, Tao tao)
{
  PetscReal      two=2.0,p01=1;
  PetscReal      gap1,gap2,fff,mu;

  PetscFunctionBegin;
  /* Compute function, Gradient R=Hx+b, and Hessian */
  CHKERRQ(MatMult(tao->hessian,tao->solution,tao->gradient));
  CHKERRQ(VecCopy(qp->C,qp->Work));
  CHKERRQ(VecAXPY(qp->Work,0.5,tao->gradient));
  CHKERRQ(VecAXPY(tao->gradient,1.0,qp->C));
  CHKERRQ(VecDot(tao->solution,qp->Work,&fff));
  qp->pobj = fff + qp->d;

  PetscCheck(!PetscIsInfOrNanReal(qp->pobj),PETSC_COMM_SELF,PETSC_ERR_USER, "User provided data contains Inf or NaN");

  /* Initialize slack vectors */
  /* T = XU - X; G = X - XL */
  CHKERRQ(VecCopy(qp->XU,qp->T));
  CHKERRQ(VecAXPY(qp->T,-1.0,tao->solution));
  CHKERRQ(VecCopy(tao->solution,qp->G));
  CHKERRQ(VecAXPY(qp->G,-1.0,qp->XL));

  CHKERRQ(VecSet(qp->GZwork,p01));
  CHKERRQ(VecSet(qp->TSwork,p01));

  CHKERRQ(VecPointwiseMax(qp->G,qp->G,qp->GZwork));
  CHKERRQ(VecPointwiseMax(qp->T,qp->T,qp->TSwork));

  /* Initialize Dual Variable Vectors */
  CHKERRQ(VecCopy(qp->G,qp->Z));
  CHKERRQ(VecReciprocal(qp->Z));

  CHKERRQ(VecCopy(qp->T,qp->S));
  CHKERRQ(VecReciprocal(qp->S));

  CHKERRQ(MatMult(tao->hessian,qp->Work,qp->RHS));
  CHKERRQ(VecAbs(qp->RHS));
  CHKERRQ(VecSet(qp->Work,p01));
  CHKERRQ(VecPointwiseMax(qp->RHS,qp->RHS,qp->Work));

  CHKERRQ(VecPointwiseDivide(qp->RHS,tao->gradient,qp->RHS));
  CHKERRQ(VecNorm(qp->RHS,NORM_1,&gap1));
  mu = PetscMin(10.0,(gap1+10.0)/qp->m);

  CHKERRQ(VecScale(qp->S,mu));
  CHKERRQ(VecScale(qp->Z,mu));

  CHKERRQ(VecSet(qp->TSwork,p01));
  CHKERRQ(VecSet(qp->GZwork,p01));
  CHKERRQ(VecPointwiseMax(qp->S,qp->S,qp->TSwork));
  CHKERRQ(VecPointwiseMax(qp->Z,qp->Z,qp->GZwork));

  qp->mu=0;qp->dinfeas=1.0;qp->pinfeas=1.0;
  while ((qp->dinfeas+qp->pinfeas)/(qp->m+qp->n) >= qp->mu) {
    CHKERRQ(VecScale(qp->G,two));
    CHKERRQ(VecScale(qp->Z,two));
    CHKERRQ(VecScale(qp->S,two));
    CHKERRQ(VecScale(qp->T,two));

    CHKERRQ(QPIPComputeResidual(qp,tao));

    CHKERRQ(VecCopy(tao->solution,qp->R3));
    CHKERRQ(VecAXPY(qp->R3,-1.0,qp->G));
    CHKERRQ(VecAXPY(qp->R3,-1.0,qp->XL));

    CHKERRQ(VecCopy(tao->solution,qp->R5));
    CHKERRQ(VecAXPY(qp->R5,1.0,qp->T));
    CHKERRQ(VecAXPY(qp->R5,-1.0,qp->XU));

    CHKERRQ(VecNorm(qp->R3,NORM_INFINITY,&gap1));
    CHKERRQ(VecNorm(qp->R5,NORM_INFINITY,&gap2));
    qp->pinfeas=PetscMax(gap1,gap2);

    /* Compute the duality gap */
    CHKERRQ(VecDot(qp->G,qp->Z,&gap1));
    CHKERRQ(VecDot(qp->T,qp->S,&gap2));

    qp->gap  = gap1+gap2;
    qp->dobj = qp->pobj - qp->gap;
    if (qp->m>0) {
      qp->mu=qp->gap/(qp->m);
    } else {
      qp->mu=0.0;
    }
    qp->rgap=qp->gap/(PetscAbsReal(qp->dobj) + PetscAbsReal(qp->pobj) + 1.0);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode QPIPStepLength(TAO_BQPIP *qp)
{
  PetscReal      tstep1,tstep2,tstep3,tstep4,tstep;

  PetscFunctionBegin;
  /* Compute stepsize to the boundary */
  CHKERRQ(VecStepMax(qp->G,qp->DG,&tstep1));
  CHKERRQ(VecStepMax(qp->T,qp->DT,&tstep2));
  CHKERRQ(VecStepMax(qp->S,qp->DS,&tstep3));
  CHKERRQ(VecStepMax(qp->Z,qp->DZ,&tstep4));

  tstep = PetscMin(tstep1,tstep2);
  qp->psteplength = PetscMin(0.95*tstep,1.0);

  tstep = PetscMin(tstep3,tstep4);
  qp->dsteplength = PetscMin(0.95*tstep,1.0);

  qp->psteplength = PetscMin(qp->psteplength,qp->dsteplength);
  qp->dsteplength = qp->psteplength;
  PetscFunctionReturn(0);
}

static PetscErrorCode QPIPComputeNormFromCentralPath(TAO_BQPIP *qp,PetscReal *norm)
{
  PetscReal      gap[2],mu[2],nmu;

  PetscFunctionBegin;
  CHKERRQ(VecPointwiseMult(qp->GZwork,qp->G,qp->Z));
  CHKERRQ(VecPointwiseMult(qp->TSwork,qp->T,qp->S));
  CHKERRQ(VecNorm(qp->TSwork,NORM_1,&mu[0]));
  CHKERRQ(VecNorm(qp->GZwork,NORM_1,&mu[1]));

  nmu=-(mu[0]+mu[1])/qp->m;

  CHKERRQ(VecShift(qp->GZwork,nmu));
  CHKERRQ(VecShift(qp->TSwork,nmu));

  CHKERRQ(VecNorm(qp->GZwork,NORM_2,&gap[0]));
  CHKERRQ(VecNorm(qp->TSwork,NORM_2,&gap[1]));
  gap[0]*=gap[0];
  gap[1]*=gap[1];

  qp->pathnorm=PetscSqrtScalar(gap[0]+gap[1]);
  *norm=qp->pathnorm;
  PetscFunctionReturn(0);
}

static PetscErrorCode QPIPComputeStepDirection(TAO_BQPIP *qp,Tao tao)
{
  PetscFunctionBegin;
  /* Calculate DG */
  CHKERRQ(VecCopy(tao->stepdirection,qp->DG));
  CHKERRQ(VecAXPY(qp->DG,1.0,qp->R3));

  /* Calculate DT */
  CHKERRQ(VecCopy(tao->stepdirection,qp->DT));
  CHKERRQ(VecScale(qp->DT,-1.0));
  CHKERRQ(VecAXPY(qp->DT,-1.0,qp->R5));

  /* Calculate DZ */
  CHKERRQ(VecAXPY(qp->DZ,-1.0,qp->Z));
  CHKERRQ(VecPointwiseDivide(qp->GZwork,qp->DG,qp->G));
  CHKERRQ(VecPointwiseMult(qp->GZwork,qp->GZwork,qp->Z));
  CHKERRQ(VecAXPY(qp->DZ,-1.0,qp->GZwork));

  /* Calculate DS */
  CHKERRQ(VecAXPY(qp->DS,-1.0,qp->S));
  CHKERRQ(VecPointwiseDivide(qp->TSwork,qp->DT,qp->T));
  CHKERRQ(VecPointwiseMult(qp->TSwork,qp->TSwork,qp->S));
  CHKERRQ(VecAXPY(qp->DS,-1.0,qp->TSwork));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetup_BQPIP(Tao tao)
{
  TAO_BQPIP      *qp =(TAO_BQPIP*)tao->data;

  PetscFunctionBegin;
  /* Set pointers to Data */
  CHKERRQ(VecGetSize(tao->solution,&qp->n));

  /* Allocate some arrays */
  if (!tao->gradient) {
    CHKERRQ(VecDuplicate(tao->solution,&tao->gradient));
  }
  if (!tao->stepdirection) {
    CHKERRQ(VecDuplicate(tao->solution,&tao->stepdirection));
  }
  if (!tao->XL) {
    CHKERRQ(VecDuplicate(tao->solution,&tao->XL));
    CHKERRQ(VecSet(tao->XL,PETSC_NINFINITY));
  }
  if (!tao->XU) {
    CHKERRQ(VecDuplicate(tao->solution,&tao->XU));
    CHKERRQ(VecSet(tao->XU,PETSC_INFINITY));
  }

  CHKERRQ(VecDuplicate(tao->solution,&qp->Work));
  CHKERRQ(VecDuplicate(tao->solution,&qp->XU));
  CHKERRQ(VecDuplicate(tao->solution,&qp->XL));
  CHKERRQ(VecDuplicate(tao->solution,&qp->HDiag));
  CHKERRQ(VecDuplicate(tao->solution,&qp->DiagAxpy));
  CHKERRQ(VecDuplicate(tao->solution,&qp->RHS));
  CHKERRQ(VecDuplicate(tao->solution,&qp->RHS2));
  CHKERRQ(VecDuplicate(tao->solution,&qp->C));

  CHKERRQ(VecDuplicate(tao->solution,&qp->G));
  CHKERRQ(VecDuplicate(tao->solution,&qp->DG));
  CHKERRQ(VecDuplicate(tao->solution,&qp->S));
  CHKERRQ(VecDuplicate(tao->solution,&qp->Z));
  CHKERRQ(VecDuplicate(tao->solution,&qp->DZ));
  CHKERRQ(VecDuplicate(tao->solution,&qp->GZwork));
  CHKERRQ(VecDuplicate(tao->solution,&qp->R3));

  CHKERRQ(VecDuplicate(tao->solution,&qp->T));
  CHKERRQ(VecDuplicate(tao->solution,&qp->DT));
  CHKERRQ(VecDuplicate(tao->solution,&qp->DS));
  CHKERRQ(VecDuplicate(tao->solution,&qp->TSwork));
  CHKERRQ(VecDuplicate(tao->solution,&qp->R5));
  qp->m=2*qp->n;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_BQPIP(Tao tao)
{
  TAO_BQPIP          *qp = (TAO_BQPIP*)tao->data;
  PetscInt           its;
  PetscReal          d1,d2,ksptol,sigmamu;
  PetscReal          gnorm,dstep,pstep,step=0;
  PetscReal          gap[4];
  PetscBool          getdiagop;

  PetscFunctionBegin;
  qp->dobj        = 0.0;
  qp->pobj        = 1.0;
  qp->gap         = 10.0;
  qp->rgap        = 1.0;
  qp->mu          = 1.0;
  qp->dinfeas     = 1.0;
  qp->psteplength = 0.0;
  qp->dsteplength = 0.0;

  /* TODO
     - Remove fixed variables and treat them correctly
     - Use index sets for the infinite versus finite bounds
     - Update remaining code for fixed and free variables
     - Fix inexact solves for predictor and corrector
  */

  /* Tighten infinite bounds, things break when we don't do this
    -- see test_bqpip.c
  */
  CHKERRQ(TaoComputeVariableBounds(tao));
  CHKERRQ(VecSet(qp->XU,1.0e20));
  CHKERRQ(VecSet(qp->XL,-1.0e20));
  CHKERRQ(VecPointwiseMax(qp->XL,qp->XL,tao->XL));
  CHKERRQ(VecPointwiseMin(qp->XU,qp->XU,tao->XU));
  CHKERRQ(VecMedian(qp->XL,tao->solution,qp->XU,tao->solution));

  /* Evaluate gradient and Hessian at zero to get the correct values
     without contaminating them with numerical artifacts.
  */
  CHKERRQ(VecSet(qp->Work,0));
  CHKERRQ(TaoComputeObjectiveAndGradient(tao,qp->Work,&qp->d,qp->C));
  CHKERRQ(TaoComputeHessian(tao,qp->Work,tao->hessian,tao->hessian_pre));
  CHKERRQ(MatHasOperation(tao->hessian,MATOP_GET_DIAGONAL,&getdiagop));
  if (getdiagop) {
    CHKERRQ(MatGetDiagonal(tao->hessian,qp->HDiag));
  }

  /* Initialize starting point and residuals */
  CHKERRQ(QPIPSetInitialPoint(qp,tao));
  CHKERRQ(QPIPComputeResidual(qp,tao));

  /* Enter main loop */
  tao->reason = TAO_CONTINUE_ITERATING;
  while (1) {

    /* Check Stopping Condition      */
    gnorm = PetscSqrtScalar(qp->gap + qp->dinfeas);
    CHKERRQ(TaoLogConvergenceHistory(tao,qp->pobj,gnorm,qp->pinfeas,tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter,qp->pobj,gnorm,qp->pinfeas,step));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
    if (tao->reason != TAO_CONTINUE_ITERATING) break;
    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    tao->niter++;
    tao->ksp_its = 0;

    /*
       Dual Infeasibility Direction should already be in the right
       hand side from computing the residuals
    */

    CHKERRQ(QPIPComputeNormFromCentralPath(qp,&d1));

    CHKERRQ(VecSet(qp->DZ,0.0));
    CHKERRQ(VecSet(qp->DS,0.0));

    /*
       Compute the Primal Infeasiblitiy RHS and the
       Diagonal Matrix to be added to H and store in Work
    */
    CHKERRQ(VecPointwiseDivide(qp->DiagAxpy,qp->Z,qp->G));
    CHKERRQ(VecPointwiseMult(qp->GZwork,qp->DiagAxpy,qp->R3));
    CHKERRQ(VecAXPY(qp->RHS,-1.0,qp->GZwork));

    CHKERRQ(VecPointwiseDivide(qp->TSwork,qp->S,qp->T));
    CHKERRQ(VecAXPY(qp->DiagAxpy,1.0,qp->TSwork));
    CHKERRQ(VecPointwiseMult(qp->TSwork,qp->TSwork,qp->R5));
    CHKERRQ(VecAXPY(qp->RHS,-1.0,qp->TSwork));

    /*  Determine the solving tolerance */
    ksptol = qp->mu/10.0;
    ksptol = PetscMin(ksptol,0.001);
    CHKERRQ(KSPSetTolerances(tao->ksp,ksptol,1e-30,1e30,PetscMax(10,qp->n)));

    /* Shift the diagonals of the Hessian matrix */
    CHKERRQ(MatDiagonalSet(tao->hessian,qp->DiagAxpy,ADD_VALUES));
    if (!getdiagop) {
      CHKERRQ(VecCopy(qp->DiagAxpy,qp->HDiag));
      CHKERRQ(VecScale(qp->HDiag,-1.0));
    }
    CHKERRQ(MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY));

    CHKERRQ(KSPSetOperators(tao->ksp,tao->hessian,tao->hessian_pre));
    CHKERRQ(KSPSolve(tao->ksp,qp->RHS,tao->stepdirection));
    CHKERRQ(KSPGetIterationNumber(tao->ksp,&its));
    tao->ksp_its += its;
    tao->ksp_tot_its += its;

    /* Restore the true diagonal of the Hessian matrix */
    if (getdiagop) {
      CHKERRQ(MatDiagonalSet(tao->hessian,qp->HDiag,INSERT_VALUES));
    } else {
      CHKERRQ(MatDiagonalSet(tao->hessian,qp->HDiag,ADD_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY));
    CHKERRQ(QPIPComputeStepDirection(qp,tao));
    CHKERRQ(QPIPStepLength(qp));

    /* Calculate New Residual R1 in Work vector */
    CHKERRQ(MatMult(tao->hessian,tao->stepdirection,qp->RHS2));
    CHKERRQ(VecAXPY(qp->RHS2,1.0,qp->DS));
    CHKERRQ(VecAXPY(qp->RHS2,-1.0,qp->DZ));
    CHKERRQ(VecAYPX(qp->RHS2,qp->dsteplength,tao->gradient));

    CHKERRQ(VecNorm(qp->RHS2,NORM_2,&qp->dinfeas));
    CHKERRQ(VecDot(qp->DZ,qp->DG,gap));
    CHKERRQ(VecDot(qp->DS,qp->DT,gap+1));

    qp->rnorm = (qp->dinfeas+qp->psteplength*qp->pinfeas)/(qp->m+qp->n);
    pstep     = qp->psteplength;
    step      = PetscMin(qp->psteplength,qp->dsteplength);
    sigmamu   = (pstep*pstep*(gap[0]+gap[1]) + (1 - pstep)*qp->gap)/qp->m;

    if (qp->predcorr && step < 0.9) {
      if (sigmamu < qp->mu) {
        sigmamu = sigmamu/qp->mu;
        sigmamu = sigmamu*sigmamu*sigmamu;
      } else {
        sigmamu = 1.0;
      }
      sigmamu = sigmamu*qp->mu;

      /* Compute Corrector Step */
      CHKERRQ(VecPointwiseMult(qp->DZ,qp->DG,qp->DZ));
      CHKERRQ(VecScale(qp->DZ,-1.0));
      CHKERRQ(VecShift(qp->DZ,sigmamu));
      CHKERRQ(VecPointwiseDivide(qp->DZ,qp->DZ,qp->G));

      CHKERRQ(VecPointwiseMult(qp->DS,qp->DS,qp->DT));
      CHKERRQ(VecScale(qp->DS,-1.0));
      CHKERRQ(VecShift(qp->DS,sigmamu));
      CHKERRQ(VecPointwiseDivide(qp->DS,qp->DS,qp->T));

      CHKERRQ(VecCopy(qp->DZ,qp->RHS2));
      CHKERRQ(VecAXPY(qp->RHS2,-1.0,qp->DS));
      CHKERRQ(VecAXPY(qp->RHS2,1.0,qp->RHS));

      /* Approximately solve the linear system */
      CHKERRQ(MatDiagonalSet(tao->hessian,qp->DiagAxpy,ADD_VALUES));
      if (!getdiagop) {
        CHKERRQ(VecCopy(qp->DiagAxpy,qp->HDiag));
        CHKERRQ(VecScale(qp->HDiag,-1.0));
      }
      CHKERRQ(MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY));

      /* Solve using the previous tolerances that were set */
      CHKERRQ(KSPSolve(tao->ksp,qp->RHS2,tao->stepdirection));
      CHKERRQ(KSPGetIterationNumber(tao->ksp,&its));
      tao->ksp_its += its;
      tao->ksp_tot_its += its;

      if (getdiagop) {
        CHKERRQ(MatDiagonalSet(tao->hessian,qp->HDiag,INSERT_VALUES));
      } else {
        CHKERRQ(MatDiagonalSet(tao->hessian,qp->HDiag,ADD_VALUES));
      }
      CHKERRQ(MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY));
      CHKERRQ(QPIPComputeStepDirection(qp,tao));
      CHKERRQ(QPIPStepLength(qp));
    }  /* End Corrector step */

    /* Take the step */
    dstep = qp->dsteplength;

    CHKERRQ(VecAXPY(qp->Z,dstep,qp->DZ));
    CHKERRQ(VecAXPY(qp->S,dstep,qp->DS));
    CHKERRQ(VecAXPY(tao->solution,dstep,tao->stepdirection));
    CHKERRQ(VecAXPY(qp->G,dstep,qp->DG));
    CHKERRQ(VecAXPY(qp->T,dstep,qp->DT));

    /* Compute Residuals */
    CHKERRQ(QPIPComputeResidual(qp,tao));

    /* Evaluate quadratic function */
    CHKERRQ(MatMult(tao->hessian,tao->solution,qp->Work));

    CHKERRQ(VecDot(tao->solution,qp->Work,&d1));
    CHKERRQ(VecDot(tao->solution,qp->C,&d2));
    CHKERRQ(VecDot(qp->G,qp->Z,gap));
    CHKERRQ(VecDot(qp->T,qp->S,gap+1));

    /* Compute the duality gap */
    qp->pobj = d1/2.0 + d2+qp->d;
    qp->gap  = gap[0]+gap[1];
    qp->dobj = qp->pobj - qp->gap;
    if (qp->m > 0) {
      qp->mu = qp->gap/(qp->m);
    }
    qp->rgap = qp->gap/(PetscAbsReal(qp->dobj) + PetscAbsReal(qp->pobj) + 1.0);
  }  /* END MAIN LOOP  */
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BQPIP(Tao tao,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BQPIP(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BQPIP      *qp = (TAO_BQPIP*)tao->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Interior point method for bound constrained quadratic optimization"));
  CHKERRQ(PetscOptionsInt("-tao_bqpip_predcorr","Use a predictor-corrector method","",qp->predcorr,&qp->predcorr,NULL));
  CHKERRQ(PetscOptionsTail());
  CHKERRQ(KSPSetFromOptions(tao->ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BQPIP(Tao tao)
{
  TAO_BQPIP      *qp = (TAO_BQPIP*)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    CHKERRQ(VecDestroy(&qp->G));
    CHKERRQ(VecDestroy(&qp->DG));
    CHKERRQ(VecDestroy(&qp->Z));
    CHKERRQ(VecDestroy(&qp->DZ));
    CHKERRQ(VecDestroy(&qp->GZwork));
    CHKERRQ(VecDestroy(&qp->R3));
    CHKERRQ(VecDestroy(&qp->S));
    CHKERRQ(VecDestroy(&qp->DS));
    CHKERRQ(VecDestroy(&qp->T));

    CHKERRQ(VecDestroy(&qp->DT));
    CHKERRQ(VecDestroy(&qp->TSwork));
    CHKERRQ(VecDestroy(&qp->R5));
    CHKERRQ(VecDestroy(&qp->HDiag));
    CHKERRQ(VecDestroy(&qp->Work));
    CHKERRQ(VecDestroy(&qp->XL));
    CHKERRQ(VecDestroy(&qp->XU));
    CHKERRQ(VecDestroy(&qp->DiagAxpy));
    CHKERRQ(VecDestroy(&qp->RHS));
    CHKERRQ(VecDestroy(&qp->RHS2));
    CHKERRQ(VecDestroy(&qp->C));
  }
  CHKERRQ(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoComputeDual_BQPIP(Tao tao,Vec DXL,Vec DXU)
{
  TAO_BQPIP       *qp = (TAO_BQPIP*)tao->data;

  PetscFunctionBegin;
  CHKERRQ(VecCopy(qp->Z,DXL));
  CHKERRQ(VecCopy(qp->S,DXU));
  CHKERRQ(VecScale(DXU,-1.0));
  PetscFunctionReturn(0);
}

/*MC
 TAOBQPIP - interior-point method for quadratic programs with
    box constraints.

 Notes:
    This algorithms solves quadratic problems only, the Hessian will
        only be computed once.

 Options Database Keys:
. -tao_bqpip_predcorr - use a predictor/corrector method

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_BQPIP(Tao tao)
{
  TAO_BQPIP      *qp;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(tao,&qp));

  tao->ops->setup = TaoSetup_BQPIP;
  tao->ops->solve = TaoSolve_BQPIP;
  tao->ops->view = TaoView_BQPIP;
  tao->ops->setfromoptions = TaoSetFromOptions_BQPIP;
  tao->ops->destroy = TaoDestroy_BQPIP;
  tao->ops->computedual = TaoComputeDual_BQPIP;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it=100;
  if (!tao->max_funcs_changed) tao->max_funcs = 500;
#if defined(PETSC_USE_REAL_SINGLE)
  if (!tao->catol_changed) tao->catol=1e-6;
#else
  if (!tao->catol_changed) tao->catol=1e-12;
#endif

  /* Initialize pointers and variables */
  qp->n = 0;
  qp->m = 0;

  qp->predcorr = 1;
  tao->data    = (void*)qp;

  CHKERRQ(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  CHKERRQ(KSPSetOptionsPrefix(tao->ksp,tao->hdr.prefix));
  CHKERRQ(KSPSetType(tao->ksp,KSPCG));
  CHKERRQ(KSPSetTolerances(tao->ksp,1e-14,1e-30,1e30,PetscMax(10,qp->n)));
  PetscFunctionReturn(0);
}
