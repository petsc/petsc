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

  PetscCall(VecScale(qp->R3,dtmp));
  PetscCall(VecScale(qp->R5,dtmp));
  qp->pinfeas=dtmp*qp->pinfeas;

  PetscCall(VecCopy(qp->S,tao->gradient));
  PetscCall(VecAXPY(tao->gradient,-1.0,qp->Z));

  PetscCall(MatMult(tao->hessian,tao->solution,qp->RHS));
  PetscCall(VecScale(qp->RHS,-1.0));
  PetscCall(VecAXPY(qp->RHS,-1.0,qp->C));
  PetscCall(VecAXPY(tao->gradient,-1.0,qp->RHS));

  PetscCall(VecNorm(tao->gradient,NORM_1,&qp->dinfeas));
  qp->rnorm=(qp->dinfeas+qp->pinfeas)/(qp->m+qp->n);
  PetscFunctionReturn(0);
}

static PetscErrorCode  QPIPSetInitialPoint(TAO_BQPIP *qp, Tao tao)
{
  PetscReal      two=2.0,p01=1;
  PetscReal      gap1,gap2,fff,mu;

  PetscFunctionBegin;
  /* Compute function, Gradient R=Hx+b, and Hessian */
  PetscCall(MatMult(tao->hessian,tao->solution,tao->gradient));
  PetscCall(VecCopy(qp->C,qp->Work));
  PetscCall(VecAXPY(qp->Work,0.5,tao->gradient));
  PetscCall(VecAXPY(tao->gradient,1.0,qp->C));
  PetscCall(VecDot(tao->solution,qp->Work,&fff));
  qp->pobj = fff + qp->d;

  PetscCheck(!PetscIsInfOrNanReal(qp->pobj),PETSC_COMM_SELF,PETSC_ERR_USER, "User provided data contains Inf or NaN");

  /* Initialize slack vectors */
  /* T = XU - X; G = X - XL */
  PetscCall(VecCopy(qp->XU,qp->T));
  PetscCall(VecAXPY(qp->T,-1.0,tao->solution));
  PetscCall(VecCopy(tao->solution,qp->G));
  PetscCall(VecAXPY(qp->G,-1.0,qp->XL));

  PetscCall(VecSet(qp->GZwork,p01));
  PetscCall(VecSet(qp->TSwork,p01));

  PetscCall(VecPointwiseMax(qp->G,qp->G,qp->GZwork));
  PetscCall(VecPointwiseMax(qp->T,qp->T,qp->TSwork));

  /* Initialize Dual Variable Vectors */
  PetscCall(VecCopy(qp->G,qp->Z));
  PetscCall(VecReciprocal(qp->Z));

  PetscCall(VecCopy(qp->T,qp->S));
  PetscCall(VecReciprocal(qp->S));

  PetscCall(MatMult(tao->hessian,qp->Work,qp->RHS));
  PetscCall(VecAbs(qp->RHS));
  PetscCall(VecSet(qp->Work,p01));
  PetscCall(VecPointwiseMax(qp->RHS,qp->RHS,qp->Work));

  PetscCall(VecPointwiseDivide(qp->RHS,tao->gradient,qp->RHS));
  PetscCall(VecNorm(qp->RHS,NORM_1,&gap1));
  mu = PetscMin(10.0,(gap1+10.0)/qp->m);

  PetscCall(VecScale(qp->S,mu));
  PetscCall(VecScale(qp->Z,mu));

  PetscCall(VecSet(qp->TSwork,p01));
  PetscCall(VecSet(qp->GZwork,p01));
  PetscCall(VecPointwiseMax(qp->S,qp->S,qp->TSwork));
  PetscCall(VecPointwiseMax(qp->Z,qp->Z,qp->GZwork));

  qp->mu=0;qp->dinfeas=1.0;qp->pinfeas=1.0;
  while ((qp->dinfeas+qp->pinfeas)/(qp->m+qp->n) >= qp->mu) {
    PetscCall(VecScale(qp->G,two));
    PetscCall(VecScale(qp->Z,two));
    PetscCall(VecScale(qp->S,two));
    PetscCall(VecScale(qp->T,two));

    PetscCall(QPIPComputeResidual(qp,tao));

    PetscCall(VecCopy(tao->solution,qp->R3));
    PetscCall(VecAXPY(qp->R3,-1.0,qp->G));
    PetscCall(VecAXPY(qp->R3,-1.0,qp->XL));

    PetscCall(VecCopy(tao->solution,qp->R5));
    PetscCall(VecAXPY(qp->R5,1.0,qp->T));
    PetscCall(VecAXPY(qp->R5,-1.0,qp->XU));

    PetscCall(VecNorm(qp->R3,NORM_INFINITY,&gap1));
    PetscCall(VecNorm(qp->R5,NORM_INFINITY,&gap2));
    qp->pinfeas=PetscMax(gap1,gap2);

    /* Compute the duality gap */
    PetscCall(VecDot(qp->G,qp->Z,&gap1));
    PetscCall(VecDot(qp->T,qp->S,&gap2));

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
  PetscCall(VecStepMax(qp->G,qp->DG,&tstep1));
  PetscCall(VecStepMax(qp->T,qp->DT,&tstep2));
  PetscCall(VecStepMax(qp->S,qp->DS,&tstep3));
  PetscCall(VecStepMax(qp->Z,qp->DZ,&tstep4));

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
  PetscCall(VecPointwiseMult(qp->GZwork,qp->G,qp->Z));
  PetscCall(VecPointwiseMult(qp->TSwork,qp->T,qp->S));
  PetscCall(VecNorm(qp->TSwork,NORM_1,&mu[0]));
  PetscCall(VecNorm(qp->GZwork,NORM_1,&mu[1]));

  nmu=-(mu[0]+mu[1])/qp->m;

  PetscCall(VecShift(qp->GZwork,nmu));
  PetscCall(VecShift(qp->TSwork,nmu));

  PetscCall(VecNorm(qp->GZwork,NORM_2,&gap[0]));
  PetscCall(VecNorm(qp->TSwork,NORM_2,&gap[1]));
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
  PetscCall(VecCopy(tao->stepdirection,qp->DG));
  PetscCall(VecAXPY(qp->DG,1.0,qp->R3));

  /* Calculate DT */
  PetscCall(VecCopy(tao->stepdirection,qp->DT));
  PetscCall(VecScale(qp->DT,-1.0));
  PetscCall(VecAXPY(qp->DT,-1.0,qp->R5));

  /* Calculate DZ */
  PetscCall(VecAXPY(qp->DZ,-1.0,qp->Z));
  PetscCall(VecPointwiseDivide(qp->GZwork,qp->DG,qp->G));
  PetscCall(VecPointwiseMult(qp->GZwork,qp->GZwork,qp->Z));
  PetscCall(VecAXPY(qp->DZ,-1.0,qp->GZwork));

  /* Calculate DS */
  PetscCall(VecAXPY(qp->DS,-1.0,qp->S));
  PetscCall(VecPointwiseDivide(qp->TSwork,qp->DT,qp->T));
  PetscCall(VecPointwiseMult(qp->TSwork,qp->TSwork,qp->S));
  PetscCall(VecAXPY(qp->DS,-1.0,qp->TSwork));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetup_BQPIP(Tao tao)
{
  TAO_BQPIP      *qp =(TAO_BQPIP*)tao->data;

  PetscFunctionBegin;
  /* Set pointers to Data */
  PetscCall(VecGetSize(tao->solution,&qp->n));

  /* Allocate some arrays */
  if (!tao->gradient) {
    PetscCall(VecDuplicate(tao->solution,&tao->gradient));
  }
  if (!tao->stepdirection) {
    PetscCall(VecDuplicate(tao->solution,&tao->stepdirection));
  }
  if (!tao->XL) {
    PetscCall(VecDuplicate(tao->solution,&tao->XL));
    PetscCall(VecSet(tao->XL,PETSC_NINFINITY));
  }
  if (!tao->XU) {
    PetscCall(VecDuplicate(tao->solution,&tao->XU));
    PetscCall(VecSet(tao->XU,PETSC_INFINITY));
  }

  PetscCall(VecDuplicate(tao->solution,&qp->Work));
  PetscCall(VecDuplicate(tao->solution,&qp->XU));
  PetscCall(VecDuplicate(tao->solution,&qp->XL));
  PetscCall(VecDuplicate(tao->solution,&qp->HDiag));
  PetscCall(VecDuplicate(tao->solution,&qp->DiagAxpy));
  PetscCall(VecDuplicate(tao->solution,&qp->RHS));
  PetscCall(VecDuplicate(tao->solution,&qp->RHS2));
  PetscCall(VecDuplicate(tao->solution,&qp->C));

  PetscCall(VecDuplicate(tao->solution,&qp->G));
  PetscCall(VecDuplicate(tao->solution,&qp->DG));
  PetscCall(VecDuplicate(tao->solution,&qp->S));
  PetscCall(VecDuplicate(tao->solution,&qp->Z));
  PetscCall(VecDuplicate(tao->solution,&qp->DZ));
  PetscCall(VecDuplicate(tao->solution,&qp->GZwork));
  PetscCall(VecDuplicate(tao->solution,&qp->R3));

  PetscCall(VecDuplicate(tao->solution,&qp->T));
  PetscCall(VecDuplicate(tao->solution,&qp->DT));
  PetscCall(VecDuplicate(tao->solution,&qp->DS));
  PetscCall(VecDuplicate(tao->solution,&qp->TSwork));
  PetscCall(VecDuplicate(tao->solution,&qp->R5));
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
  PetscCall(TaoComputeVariableBounds(tao));
  PetscCall(VecSet(qp->XU,1.0e20));
  PetscCall(VecSet(qp->XL,-1.0e20));
  PetscCall(VecPointwiseMax(qp->XL,qp->XL,tao->XL));
  PetscCall(VecPointwiseMin(qp->XU,qp->XU,tao->XU));
  PetscCall(VecMedian(qp->XL,tao->solution,qp->XU,tao->solution));

  /* Evaluate gradient and Hessian at zero to get the correct values
     without contaminating them with numerical artifacts.
  */
  PetscCall(VecSet(qp->Work,0));
  PetscCall(TaoComputeObjectiveAndGradient(tao,qp->Work,&qp->d,qp->C));
  PetscCall(TaoComputeHessian(tao,qp->Work,tao->hessian,tao->hessian_pre));
  PetscCall(MatHasOperation(tao->hessian,MATOP_GET_DIAGONAL,&getdiagop));
  if (getdiagop) {
    PetscCall(MatGetDiagonal(tao->hessian,qp->HDiag));
  }

  /* Initialize starting point and residuals */
  PetscCall(QPIPSetInitialPoint(qp,tao));
  PetscCall(QPIPComputeResidual(qp,tao));

  /* Enter main loop */
  tao->reason = TAO_CONTINUE_ITERATING;
  while (1) {

    /* Check Stopping Condition      */
    gnorm = PetscSqrtScalar(qp->gap + qp->dinfeas);
    PetscCall(TaoLogConvergenceHistory(tao,qp->pobj,gnorm,qp->pinfeas,tao->ksp_its));
    PetscCall(TaoMonitor(tao,tao->niter,qp->pobj,gnorm,qp->pinfeas,step));
    PetscCall((*tao->ops->convergencetest)(tao,tao->cnvP));
    if (tao->reason != TAO_CONTINUE_ITERATING) break;
    /* Call general purpose update function */
    if (tao->ops->update) {
      PetscCall((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    tao->niter++;
    tao->ksp_its = 0;

    /*
       Dual Infeasibility Direction should already be in the right
       hand side from computing the residuals
    */

    PetscCall(QPIPComputeNormFromCentralPath(qp,&d1));

    PetscCall(VecSet(qp->DZ,0.0));
    PetscCall(VecSet(qp->DS,0.0));

    /*
       Compute the Primal Infeasiblitiy RHS and the
       Diagonal Matrix to be added to H and store in Work
    */
    PetscCall(VecPointwiseDivide(qp->DiagAxpy,qp->Z,qp->G));
    PetscCall(VecPointwiseMult(qp->GZwork,qp->DiagAxpy,qp->R3));
    PetscCall(VecAXPY(qp->RHS,-1.0,qp->GZwork));

    PetscCall(VecPointwiseDivide(qp->TSwork,qp->S,qp->T));
    PetscCall(VecAXPY(qp->DiagAxpy,1.0,qp->TSwork));
    PetscCall(VecPointwiseMult(qp->TSwork,qp->TSwork,qp->R5));
    PetscCall(VecAXPY(qp->RHS,-1.0,qp->TSwork));

    /*  Determine the solving tolerance */
    ksptol = qp->mu/10.0;
    ksptol = PetscMin(ksptol,0.001);
    PetscCall(KSPSetTolerances(tao->ksp,ksptol,1e-30,1e30,PetscMax(10,qp->n)));

    /* Shift the diagonals of the Hessian matrix */
    PetscCall(MatDiagonalSet(tao->hessian,qp->DiagAxpy,ADD_VALUES));
    if (!getdiagop) {
      PetscCall(VecCopy(qp->DiagAxpy,qp->HDiag));
      PetscCall(VecScale(qp->HDiag,-1.0));
    }
    PetscCall(MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY));

    PetscCall(KSPSetOperators(tao->ksp,tao->hessian,tao->hessian_pre));
    PetscCall(KSPSolve(tao->ksp,qp->RHS,tao->stepdirection));
    PetscCall(KSPGetIterationNumber(tao->ksp,&its));
    tao->ksp_its += its;
    tao->ksp_tot_its += its;

    /* Restore the true diagonal of the Hessian matrix */
    if (getdiagop) {
      PetscCall(MatDiagonalSet(tao->hessian,qp->HDiag,INSERT_VALUES));
    } else {
      PetscCall(MatDiagonalSet(tao->hessian,qp->HDiag,ADD_VALUES));
    }
    PetscCall(MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY));
    PetscCall(QPIPComputeStepDirection(qp,tao));
    PetscCall(QPIPStepLength(qp));

    /* Calculate New Residual R1 in Work vector */
    PetscCall(MatMult(tao->hessian,tao->stepdirection,qp->RHS2));
    PetscCall(VecAXPY(qp->RHS2,1.0,qp->DS));
    PetscCall(VecAXPY(qp->RHS2,-1.0,qp->DZ));
    PetscCall(VecAYPX(qp->RHS2,qp->dsteplength,tao->gradient));

    PetscCall(VecNorm(qp->RHS2,NORM_2,&qp->dinfeas));
    PetscCall(VecDot(qp->DZ,qp->DG,gap));
    PetscCall(VecDot(qp->DS,qp->DT,gap+1));

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
      PetscCall(VecPointwiseMult(qp->DZ,qp->DG,qp->DZ));
      PetscCall(VecScale(qp->DZ,-1.0));
      PetscCall(VecShift(qp->DZ,sigmamu));
      PetscCall(VecPointwiseDivide(qp->DZ,qp->DZ,qp->G));

      PetscCall(VecPointwiseMult(qp->DS,qp->DS,qp->DT));
      PetscCall(VecScale(qp->DS,-1.0));
      PetscCall(VecShift(qp->DS,sigmamu));
      PetscCall(VecPointwiseDivide(qp->DS,qp->DS,qp->T));

      PetscCall(VecCopy(qp->DZ,qp->RHS2));
      PetscCall(VecAXPY(qp->RHS2,-1.0,qp->DS));
      PetscCall(VecAXPY(qp->RHS2,1.0,qp->RHS));

      /* Approximately solve the linear system */
      PetscCall(MatDiagonalSet(tao->hessian,qp->DiagAxpy,ADD_VALUES));
      if (!getdiagop) {
        PetscCall(VecCopy(qp->DiagAxpy,qp->HDiag));
        PetscCall(VecScale(qp->HDiag,-1.0));
      }
      PetscCall(MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY));

      /* Solve using the previous tolerances that were set */
      PetscCall(KSPSolve(tao->ksp,qp->RHS2,tao->stepdirection));
      PetscCall(KSPGetIterationNumber(tao->ksp,&its));
      tao->ksp_its += its;
      tao->ksp_tot_its += its;

      if (getdiagop) {
        PetscCall(MatDiagonalSet(tao->hessian,qp->HDiag,INSERT_VALUES));
      } else {
        PetscCall(MatDiagonalSet(tao->hessian,qp->HDiag,ADD_VALUES));
      }
      PetscCall(MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY));
      PetscCall(QPIPComputeStepDirection(qp,tao));
      PetscCall(QPIPStepLength(qp));
    }  /* End Corrector step */

    /* Take the step */
    dstep = qp->dsteplength;

    PetscCall(VecAXPY(qp->Z,dstep,qp->DZ));
    PetscCall(VecAXPY(qp->S,dstep,qp->DS));
    PetscCall(VecAXPY(tao->solution,dstep,tao->stepdirection));
    PetscCall(VecAXPY(qp->G,dstep,qp->DG));
    PetscCall(VecAXPY(qp->T,dstep,qp->DT));

    /* Compute Residuals */
    PetscCall(QPIPComputeResidual(qp,tao));

    /* Evaluate quadratic function */
    PetscCall(MatMult(tao->hessian,tao->solution,qp->Work));

    PetscCall(VecDot(tao->solution,qp->Work,&d1));
    PetscCall(VecDot(tao->solution,qp->C,&d2));
    PetscCall(VecDot(qp->G,qp->Z,gap));
    PetscCall(VecDot(qp->T,qp->S,gap+1));

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
  PetscCall(PetscOptionsHead(PetscOptionsObject,"Interior point method for bound constrained quadratic optimization"));
  PetscCall(PetscOptionsInt("-tao_bqpip_predcorr","Use a predictor-corrector method","",qp->predcorr,&qp->predcorr,NULL));
  PetscCall(PetscOptionsTail());
  PetscCall(KSPSetFromOptions(tao->ksp));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BQPIP(Tao tao)
{
  TAO_BQPIP      *qp = (TAO_BQPIP*)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&qp->G));
    PetscCall(VecDestroy(&qp->DG));
    PetscCall(VecDestroy(&qp->Z));
    PetscCall(VecDestroy(&qp->DZ));
    PetscCall(VecDestroy(&qp->GZwork));
    PetscCall(VecDestroy(&qp->R3));
    PetscCall(VecDestroy(&qp->S));
    PetscCall(VecDestroy(&qp->DS));
    PetscCall(VecDestroy(&qp->T));

    PetscCall(VecDestroy(&qp->DT));
    PetscCall(VecDestroy(&qp->TSwork));
    PetscCall(VecDestroy(&qp->R5));
    PetscCall(VecDestroy(&qp->HDiag));
    PetscCall(VecDestroy(&qp->Work));
    PetscCall(VecDestroy(&qp->XL));
    PetscCall(VecDestroy(&qp->XU));
    PetscCall(VecDestroy(&qp->DiagAxpy));
    PetscCall(VecDestroy(&qp->RHS));
    PetscCall(VecDestroy(&qp->RHS2));
    PetscCall(VecDestroy(&qp->C));
  }
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoComputeDual_BQPIP(Tao tao,Vec DXL,Vec DXU)
{
  TAO_BQPIP       *qp = (TAO_BQPIP*)tao->data;

  PetscFunctionBegin;
  PetscCall(VecCopy(qp->Z,DXL));
  PetscCall(VecCopy(qp->S,DXU));
  PetscCall(VecScale(DXU,-1.0));
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
  PetscCall(PetscNewLog(tao,&qp));

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

  PetscCall(KSPCreate(((PetscObject)tao)->comm,&tao->ksp));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1));
  PetscCall(KSPSetOptionsPrefix(tao->ksp,tao->hdr.prefix));
  PetscCall(KSPSetType(tao->ksp,KSPCG));
  PetscCall(KSPSetTolerances(tao->ksp,1e-14,1e-30,1e30,PetscMax(10,qp->n)));
  PetscFunctionReturn(0);
}
