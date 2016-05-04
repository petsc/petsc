#include <../src/tao/bound/impls/bqpip/bqpip.h>
#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "TaoSetUp_BQPIP"
static PetscErrorCode TaoSetUp_BQPIP(Tao tao)
{
  TAO_BQPIP      *qp =(TAO_BQPIP*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Set pointers to Data */
  ierr = VecGetSize(tao->solution,&qp->n);CHKERRQ(ierr);

  /* Allocate some arrays */
  if (!tao->gradient) {
      ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
  }
  if (!tao->stepdirection) {
      ierr = VecDuplicate(tao->solution, &tao->stepdirection);CHKERRQ(ierr);
  }
  if (!tao->XL) {
      ierr = VecDuplicate(tao->solution, &tao->XL);CHKERRQ(ierr);
      ierr = VecSet(tao->XL, -1.0e-20);CHKERRQ(ierr);
  }
  if (!tao->XU) {
      ierr = VecDuplicate(tao->solution, &tao->XU);CHKERRQ(ierr);
      ierr = VecSet(tao->XU, 1.0e20);CHKERRQ(ierr);
  }

  ierr = VecDuplicate(tao->solution, &qp->Work);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->XU);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->XL);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->HDiag);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->DiagAxpy);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->RHS);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->RHS2);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->C0);CHKERRQ(ierr);

  ierr = VecDuplicate(tao->solution, &qp->G);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->DG);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->S);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->Z);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->DZ);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->GZwork);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->R3);CHKERRQ(ierr);

  ierr = VecDuplicate(tao->solution, &qp->T);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->DT);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->DS);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->TSwork);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &qp->R5);CHKERRQ(ierr);
  qp->m=2*qp->n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPIPSetInitialPoint"
static PetscErrorCode  QPIPSetInitialPoint(TAO_BQPIP *qp, Tao tao)
{
  PetscErrorCode ierr;
  PetscReal      two=2.0,p01=1;
  PetscReal      gap1,gap2,fff,mu;

  PetscFunctionBegin;
  /* Compute function, Gradient R=Hx+b, and Hessian */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = VecMedian(qp->XL, tao->solution, qp->XU, tao->solution);CHKERRQ(ierr);
  ierr = MatMult(tao->hessian, tao->solution, tao->gradient);CHKERRQ(ierr);
  ierr = VecCopy(qp->C0, qp->Work);CHKERRQ(ierr);
  ierr = VecAXPY(qp->Work, 0.5, tao->gradient);CHKERRQ(ierr);
  ierr = VecAXPY(tao->gradient, 1.0, qp->C0);CHKERRQ(ierr);
  ierr = VecDot(tao->solution, qp->Work, &fff);CHKERRQ(ierr);
  qp->pobj = fff + qp->c;

  /* Initialize Primal Vectors */
  /* T = XU - X; G = X - XL */
  ierr = VecCopy(qp->XU, qp->T);CHKERRQ(ierr);
  ierr = VecAXPY(qp->T, -1.0, tao->solution);CHKERRQ(ierr);
  ierr = VecCopy(tao->solution, qp->G);CHKERRQ(ierr);
  ierr = VecAXPY(qp->G, -1.0, qp->XL);CHKERRQ(ierr);

  ierr = VecSet(qp->GZwork, p01);CHKERRQ(ierr);
  ierr = VecSet(qp->TSwork, p01);CHKERRQ(ierr);

  ierr = VecPointwiseMax(qp->G, qp->G, qp->GZwork);CHKERRQ(ierr);
  ierr = VecPointwiseMax(qp->T, qp->T, qp->TSwork);CHKERRQ(ierr);

  /* Initialize Dual Variable Vectors */
  ierr = VecCopy(qp->G, qp->Z);CHKERRQ(ierr);
  ierr = VecReciprocal(qp->Z);CHKERRQ(ierr);

  ierr = VecCopy(qp->T, qp->S);CHKERRQ(ierr);
  ierr = VecReciprocal(qp->S);CHKERRQ(ierr);

  ierr = MatMult(tao->hessian, qp->Work, qp->RHS);CHKERRQ(ierr);
  ierr = VecAbs(qp->RHS);CHKERRQ(ierr);
  ierr = VecSet(qp->Work, p01);CHKERRQ(ierr);
  ierr = VecPointwiseMax(qp->RHS, qp->RHS, qp->Work);CHKERRQ(ierr);

  ierr = VecPointwiseDivide(qp->RHS, tao->gradient, qp->RHS);CHKERRQ(ierr);
  ierr = VecNorm(qp->RHS, NORM_1, &gap1);CHKERRQ(ierr);
  mu = PetscMin(10.0,(gap1+10.0)/qp->m);

  ierr = VecScale(qp->S, mu);CHKERRQ(ierr);
  ierr = VecScale(qp->Z, mu);CHKERRQ(ierr);

  ierr = VecSet(qp->TSwork, p01);CHKERRQ(ierr);
  ierr = VecSet(qp->GZwork, p01);CHKERRQ(ierr);
  ierr = VecPointwiseMax(qp->S, qp->S, qp->TSwork);CHKERRQ(ierr);
  ierr = VecPointwiseMax(qp->Z, qp->Z, qp->GZwork);CHKERRQ(ierr);

  qp->mu=0;qp->dinfeas=1.0;qp->pinfeas=1.0;
  while ( (qp->dinfeas+qp->pinfeas)/(qp->m+qp->n) >= qp->mu ){

    ierr = VecScale(qp->G, two);CHKERRQ(ierr);
    ierr = VecScale(qp->Z, two);CHKERRQ(ierr);
    ierr = VecScale(qp->S, two);CHKERRQ(ierr);
    ierr = VecScale(qp->T, two);CHKERRQ(ierr);

    ierr = QPIPComputeResidual(qp,tao);CHKERRQ(ierr);

    ierr = VecCopy(tao->solution, qp->R3);CHKERRQ(ierr);
    ierr = VecAXPY(qp->R3, -1.0, qp->G);CHKERRQ(ierr);
    ierr = VecAXPY(qp->R3, -1.0, qp->XL);CHKERRQ(ierr);

    ierr = VecCopy(tao->solution, qp->R5);CHKERRQ(ierr);
    ierr = VecAXPY(qp->R5, 1.0, qp->T);CHKERRQ(ierr);
    ierr = VecAXPY(qp->R5, -1.0, qp->XU);CHKERRQ(ierr);

    ierr = VecNorm(qp->R3, NORM_INFINITY, &gap1);CHKERRQ(ierr);
    ierr = VecNorm(qp->R5, NORM_INFINITY, &gap2);CHKERRQ(ierr);
    qp->pinfeas=PetscMax(gap1,gap2);

    /* Compute the duality gap */
    ierr = VecDot(qp->G, qp->Z, &gap1);CHKERRQ(ierr);
    ierr = VecDot(qp->T, qp->S, &gap2);CHKERRQ(ierr);

    qp->gap = (gap1+gap2);
    qp->dobj = qp->pobj - qp->gap;
    if (qp->m>0) qp->mu=qp->gap/(qp->m); else qp->mu=0.0;
    qp->rgap=qp->gap/( PetscAbsReal(qp->dobj) + PetscAbsReal(qp->pobj) + 1.0 );
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoDestroy_BQPIP"
static PetscErrorCode TaoDestroy_BQPIP(Tao tao)
{
  TAO_BQPIP      *qp = (TAO_BQPIP*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&qp->G);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->DG);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->Z);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->DZ);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->GZwork);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->R3);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->S);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->DS);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->T);CHKERRQ(ierr);

    ierr = VecDestroy(&qp->DT);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->TSwork);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->R5);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->HDiag);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->Work);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->XL);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->XU);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->DiagAxpy);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->RHS);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->RHS2);CHKERRQ(ierr);
    ierr = VecDestroy(&qp->C0);CHKERRQ(ierr);
  }
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSolve_BQPIP"
static PetscErrorCode TaoSolve_BQPIP(Tao tao)
{
  TAO_BQPIP          *qp = (TAO_BQPIP*)tao->data;
  PetscErrorCode     ierr;
  PetscInt           its;
  PetscReal          d1,d2,ksptol,sigma;
  PetscReal          sigmamu;
  PetscReal          dstep,pstep,step=0;
  PetscReal          gap[4];
  TaoConvergedReason reason;

  PetscFunctionBegin;
  qp->dobj           = 0.0;
  qp->pobj           = 1.0;
  qp->gap            = 10.0;
  qp->rgap           = 1.0;
  qp->mu             = 1.0;
  qp->sigma          = 1.0;
  qp->dinfeas        = 1.0;
  qp->psteplength    = 0.0;
  qp->dsteplength    = 0.0;

  /* Tighten infinite bounds, things break when we don't do this
    -- see test_bqpip.c
  */
  ierr = VecSet(qp->XU,1.0e20);CHKERRQ(ierr);
  ierr = VecSet(qp->XL,-1.0e20);CHKERRQ(ierr);
  ierr = VecPointwiseMax(qp->XL,qp->XL,tao->XL);CHKERRQ(ierr);
  ierr = VecPointwiseMin(qp->XU,qp->XU,tao->XU);CHKERRQ(ierr);

  ierr = TaoComputeObjectiveAndGradient(tao,tao->solution,&qp->c,qp->C0);CHKERRQ(ierr);
  ierr = TaoComputeHessian(tao,tao->solution,tao->hessian,tao->hessian_pre);CHKERRQ(ierr);
  ierr = MatMult(tao->hessian, tao->solution, qp->Work);CHKERRQ(ierr);
  ierr = VecDot(tao->solution, qp->Work, &d1);CHKERRQ(ierr);
  ierr = VecAXPY(qp->C0, -1.0, qp->Work);CHKERRQ(ierr);
  ierr = VecDot(qp->C0, tao->solution, &d2);CHKERRQ(ierr);
  qp->c -= (d1/2.0+d2);
  ierr = MatGetDiagonal(tao->hessian, qp->HDiag);CHKERRQ(ierr);

  ierr = QPIPSetInitialPoint(qp,tao);CHKERRQ(ierr);
  ierr = QPIPComputeResidual(qp,tao);CHKERRQ(ierr);

  /* Enter main loop */
  while (PETSC_TRUE){

    /* Check Stopping Condition      */
    ierr = TaoMonitor(tao,tao->niter,qp->pobj,PetscSqrtScalar(qp->gap + qp->dinfeas),qp->pinfeas, step, &reason);CHKERRQ(ierr);
    if (reason != TAO_CONTINUE_ITERATING) break;
    tao->niter++;
    tao->ksp_its=0;

    /*
       Dual Infeasibility Direction should already be in the right
       hand side from computing the residuals
    */

    ierr = QPIPComputeNormFromCentralPath(qp,&d1);CHKERRQ(ierr);

    if (tao->niter > 0 && (qp->rnorm>5*qp->mu || d1*d1>qp->m*qp->mu*qp->mu) ) {
      sigma=1.0;sigmamu=qp->mu;
      sigma=0.0;sigmamu=0;
    } else {
      sigma=0.0;sigmamu=0;
    }
    ierr = VecSet(qp->DZ, sigmamu);CHKERRQ(ierr);
    ierr = VecSet(qp->DS, sigmamu);CHKERRQ(ierr);

    if (sigmamu !=0){
      ierr = VecPointwiseDivide(qp->DZ, qp->DZ, qp->G);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(qp->DS, qp->DS, qp->T);CHKERRQ(ierr);
      ierr = VecCopy(qp->DZ,qp->RHS2);CHKERRQ(ierr);
      ierr = VecAXPY(qp->RHS2, 1.0, qp->DS);CHKERRQ(ierr);
    } else {
      ierr = VecZeroEntries(qp->RHS2);CHKERRQ(ierr);
    }


    /*
       Compute the Primal Infeasiblitiy RHS and the
       Diagonal Matrix to be added to H and store in Work
    */
    ierr = VecPointwiseDivide(qp->DiagAxpy, qp->Z, qp->G);CHKERRQ(ierr);
    ierr = VecPointwiseMult(qp->GZwork, qp->DiagAxpy, qp->R3);CHKERRQ(ierr);
    ierr = VecAXPY(qp->RHS, -1.0, qp->GZwork);CHKERRQ(ierr);

    ierr = VecPointwiseDivide(qp->TSwork, qp->S, qp->T);CHKERRQ(ierr);
    ierr = VecAXPY(qp->DiagAxpy, 1.0, qp->TSwork);CHKERRQ(ierr);
    ierr = VecPointwiseMult(qp->TSwork, qp->TSwork, qp->R5);CHKERRQ(ierr);
    ierr = VecAXPY(qp->RHS, -1.0, qp->TSwork);CHKERRQ(ierr);
    ierr = VecAXPY(qp->RHS2, 1.0, qp->RHS);CHKERRQ(ierr);

    /*  Determine the solving tolerance */
    ksptol = qp->mu/10.0;
    ksptol = PetscMin(ksptol,0.001);

    ierr = MatDiagonalSet(tao->hessian, qp->DiagAxpy, ADD_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = KSPSetOperators(tao->ksp, tao->hessian, tao->hessian_pre);CHKERRQ(ierr);
    ierr = KSPSolve(tao->ksp, qp->RHS, tao->stepdirection);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
    tao->ksp_its+=its;
    tao->ksp_tot_its+=its;

    ierr = VecScale(qp->DiagAxpy, -1.0);CHKERRQ(ierr);
    ierr = MatDiagonalSet(tao->hessian, qp->DiagAxpy, ADD_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecScale(qp->DiagAxpy, -1.0);CHKERRQ(ierr);
    ierr = QPComputeStepDirection(qp,tao);CHKERRQ(ierr);
    ierr = QPStepLength(qp); CHKERRQ(ierr);

    /* Calculate New Residual R1 in Work vector */
    ierr = MatMult(tao->hessian, tao->stepdirection, qp->RHS2);CHKERRQ(ierr);
    ierr = VecAXPY(qp->RHS2, 1.0, qp->DS);CHKERRQ(ierr);
    ierr = VecAXPY(qp->RHS2, -1.0, qp->DZ);CHKERRQ(ierr);
    ierr = VecAYPX(qp->RHS2, qp->dsteplength, tao->gradient);CHKERRQ(ierr);

    ierr = VecNorm(qp->RHS2, NORM_2, &qp->dinfeas);CHKERRQ(ierr);
    ierr = VecDot(qp->DZ, qp->DG, gap);CHKERRQ(ierr);
    ierr = VecDot(qp->DS, qp->DT, gap+1);CHKERRQ(ierr);

    qp->rnorm=(qp->dinfeas+qp->psteplength*qp->pinfeas)/(qp->m+qp->n);
    pstep = qp->psteplength;
    step = PetscMin(qp->psteplength,qp->dsteplength);
    sigmamu= ( pstep*pstep*(gap[0]+gap[1]) +  (1 - pstep + pstep*sigma)*qp->gap  )/qp->m;

    if (qp->predcorr && step < 0.9){
      if (sigmamu < qp->mu){
        sigmamu=sigmamu/qp->mu;
        sigmamu=sigmamu*sigmamu*sigmamu;
      } else {sigmamu = 1.0;}
      sigmamu = sigmamu*qp->mu;

      /* Compute Corrector Step */
      ierr = VecPointwiseMult(qp->DZ, qp->DG, qp->DZ);CHKERRQ(ierr);
      ierr = VecScale(qp->DZ, -1.0);CHKERRQ(ierr);
      ierr = VecShift(qp->DZ, sigmamu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(qp->DZ, qp->DZ, qp->G);CHKERRQ(ierr);

      ierr = VecPointwiseMult(qp->DS, qp->DS, qp->DT);CHKERRQ(ierr);
      ierr = VecScale(qp->DS, -1.0);CHKERRQ(ierr);
      ierr = VecShift(qp->DS, sigmamu);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(qp->DS, qp->DS, qp->T);CHKERRQ(ierr);

      ierr = VecCopy(qp->DZ, qp->RHS2);CHKERRQ(ierr);
      ierr = VecAXPY(qp->RHS2, -1.0, qp->DS);CHKERRQ(ierr);
      ierr = VecAXPY(qp->RHS2, 1.0, qp->RHS);CHKERRQ(ierr);

      /* Approximately solve the linear system */
      ierr = MatDiagonalSet(tao->hessian, qp->DiagAxpy, ADD_VALUES);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = KSPSolve(tao->ksp, qp->RHS2, tao->stepdirection);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(tao->ksp,&its);CHKERRQ(ierr);
      tao->ksp_its+=its;
      tao->ksp_tot_its+=its;

      ierr = MatDiagonalSet(tao->hessian, qp->HDiag, INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(tao->hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(tao->hessian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = QPComputeStepDirection(qp,tao);CHKERRQ(ierr);
      ierr = QPStepLength(qp);CHKERRQ(ierr);

    }  /* End Corrector step */


    /* Take the step */
    dstep = qp->dsteplength;

    ierr = VecAXPY(qp->Z, dstep, qp->DZ);CHKERRQ(ierr);
    ierr = VecAXPY(qp->S, dstep, qp->DS);CHKERRQ(ierr);
    ierr = VecAXPY(tao->solution, dstep, tao->stepdirection);CHKERRQ(ierr);
    ierr = VecAXPY(qp->G, dstep, qp->DG);CHKERRQ(ierr);
    ierr = VecAXPY(qp->T, dstep, qp->DT);CHKERRQ(ierr);

    /* Compute Residuals */
    ierr = QPIPComputeResidual(qp,tao);CHKERRQ(ierr);

    /* Evaluate quadratic function */
    ierr = MatMult(tao->hessian, tao->solution, qp->Work);CHKERRQ(ierr);

    ierr = VecDot(tao->solution, qp->Work, &d1);CHKERRQ(ierr);
    ierr = VecDot(tao->solution, qp->C0, &d2);CHKERRQ(ierr);
    ierr = VecDot(qp->G, qp->Z, gap);CHKERRQ(ierr);
    ierr = VecDot(qp->T, qp->S, gap+1);CHKERRQ(ierr);

    qp->pobj=d1/2.0 + d2+qp->c;
    /* Compute the duality gap */
    qp->gap = (gap[0]+gap[1]);
    qp->dobj = qp->pobj - qp->gap;
    if (qp->m>0) qp->mu=qp->gap/(qp->m);
    qp->rgap=qp->gap/( PetscAbsReal(qp->dobj) + PetscAbsReal(qp->pobj) + 1.0 );
  }  /* END MAIN LOOP  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPComputeStepDirection"
static PetscErrorCode QPComputeStepDirection(TAO_BQPIP *qp, Tao tao)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Calculate DG */
  ierr = VecCopy(tao->stepdirection, qp->DG);CHKERRQ(ierr);
  ierr = VecAXPY(qp->DG, 1.0, qp->R3);CHKERRQ(ierr);

  /* Calculate DT */
  ierr = VecCopy(tao->stepdirection, qp->DT);CHKERRQ(ierr);
  ierr = VecScale(qp->DT, -1.0);CHKERRQ(ierr);
  ierr = VecAXPY(qp->DT, -1.0, qp->R5);CHKERRQ(ierr);

  /* Calculate DZ */
  ierr = VecAXPY(qp->DZ, -1.0, qp->Z);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(qp->GZwork, qp->DG, qp->G);CHKERRQ(ierr);
  ierr = VecPointwiseMult(qp->GZwork, qp->GZwork, qp->Z);CHKERRQ(ierr);
  ierr = VecAXPY(qp->DZ, -1.0, qp->GZwork);CHKERRQ(ierr);

  /* Calculate DS */
  ierr = VecAXPY(qp->DS, -1.0, qp->S);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(qp->TSwork, qp->DT, qp->T);CHKERRQ(ierr);
  ierr = VecPointwiseMult(qp->TSwork, qp->TSwork, qp->S);CHKERRQ(ierr);
  ierr = VecAXPY(qp->DS, -1.0, qp->TSwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPIPComputeResidual"
static PetscErrorCode QPIPComputeResidual(TAO_BQPIP *qp, Tao tao)
{
  PetscErrorCode ierr;
  PetscReal      dtmp = 1.0 - qp->psteplength;

  PetscFunctionBegin;
  /* Compute R3 and R5 */

  ierr = VecScale(qp->R3, dtmp);CHKERRQ(ierr);
  ierr = VecScale(qp->R5, dtmp);CHKERRQ(ierr);
  qp->pinfeas=dtmp*qp->pinfeas;

  ierr = VecCopy(qp->S, tao->gradient);CHKERRQ(ierr);
  ierr = VecAXPY(tao->gradient, -1.0, qp->Z);CHKERRQ(ierr);

  ierr = MatMult(tao->hessian, tao->solution, qp->RHS);CHKERRQ(ierr);
  ierr = VecScale(qp->RHS, -1.0);CHKERRQ(ierr);
  ierr = VecAXPY(qp->RHS, -1.0, qp->C0);CHKERRQ(ierr);
  ierr = VecAXPY(tao->gradient, -1.0, qp->RHS);CHKERRQ(ierr);

  ierr = VecNorm(tao->gradient, NORM_1, &qp->dinfeas);CHKERRQ(ierr);
  qp->rnorm=(qp->dinfeas+qp->pinfeas)/(qp->m+qp->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPStepLength"
static PetscErrorCode QPStepLength(TAO_BQPIP *qp)
{
  PetscReal      tstep1,tstep2,tstep3,tstep4,tstep;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Compute stepsize to the boundary */
  ierr = VecStepMax(qp->G, qp->DG, &tstep1);CHKERRQ(ierr);
  ierr = VecStepMax(qp->T, qp->DT, &tstep2);CHKERRQ(ierr);
  ierr = VecStepMax(qp->S, qp->DS, &tstep3);CHKERRQ(ierr);
  ierr = VecStepMax(qp->Z, qp->DZ, &tstep4);CHKERRQ(ierr);


  tstep = PetscMin(tstep1,tstep2);
  qp->psteplength = PetscMin(0.95*tstep,1.0);

  tstep = PetscMin(tstep3,tstep4);
  qp->dsteplength = PetscMin(0.95*tstep,1.0);

  qp->psteplength = PetscMin(qp->psteplength,qp->dsteplength);
  qp->dsteplength = qp->psteplength;

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TaoComputeDual_BQPIP"
PetscErrorCode TaoComputeDual_BQPIP(Tao tao,Vec DXL, Vec DXU)
{
  TAO_BQPIP       *qp = (TAO_BQPIP*)tao->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecCopy(qp->Z, DXL);CHKERRQ(ierr);
  ierr = VecCopy(qp->S, DXU);CHKERRQ(ierr);
  ierr = VecScale(DXU, -1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QPIPComputeNormFromCentralPath"
PetscErrorCode QPIPComputeNormFromCentralPath(TAO_BQPIP *qp, PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscReal      gap[2],mu[2], nmu;

  PetscFunctionBegin;
  ierr = VecPointwiseMult(qp->GZwork, qp->G, qp->Z);CHKERRQ(ierr);
  ierr = VecPointwiseMult(qp->TSwork, qp->T, qp->S);CHKERRQ(ierr);
  ierr = VecNorm(qp->TSwork, NORM_1, &mu[0]);CHKERRQ(ierr);
  ierr = VecNorm(qp->GZwork, NORM_1, &mu[1]);CHKERRQ(ierr);

  nmu=-(mu[0]+mu[1])/qp->m;

  ierr = VecShift(qp->GZwork,nmu);CHKERRQ(ierr);
  ierr = VecShift(qp->TSwork,nmu);CHKERRQ(ierr);

  ierr = VecNorm(qp->GZwork,NORM_2,&gap[0]);CHKERRQ(ierr);
  ierr = VecNorm(qp->TSwork,NORM_2,&gap[1]);CHKERRQ(ierr);
  gap[0]*=gap[0];
  gap[1]*=gap[1];


  qp->pathnorm=PetscSqrtScalar( (gap[0]+gap[1]) );
  *norm=qp->pathnorm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoSetFromOptions_BQPIP"
static PetscErrorCode TaoSetFromOptions_BQPIP(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BQPIP      *qp = (TAO_BQPIP*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Interior point method for bound constrained quadratic optimization");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_bqpip_predcorr","Use a predictor-corrector method","",qp->predcorr,&qp->predcorr,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaoView_BQPIP"
static PetscErrorCode TaoView_BQPIP(Tao tao, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------- */
/*MC
 TAOBQPIP - bounded quadratic interior point algorithm for quadratic 
    optimization with box constraints.

 Notes: This algorithms solves quadratic problems only, the linear Hessian will
        only be computed once.

 Options Database Keys:
. -tao_bqpip_predcorr - use a predictor/corrector method

  Level: beginner
M*/

#undef __FUNCT__
#define __FUNCT__ "TaoCreate_BQPIP"
PETSC_EXTERN PetscErrorCode TaoCreate_BQPIP(Tao tao)
{
  TAO_BQPIP      *qp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,&qp);CHKERRQ(ierr);
  tao->ops->setup = TaoSetUp_BQPIP;
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
  qp->n              = 0;
  qp->m              = 0;
  qp->ksp_tol       = 0.1;

  qp->predcorr       = 1;
  tao->data = (void*)qp;

  ierr = KSPCreate(((PetscObject)tao)->comm, &tao->ksp);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix);CHKERRQ(ierr);
  ierr = KSPSetType(tao->ksp, KSPCG);CHKERRQ(ierr);
  ierr = KSPSetTolerances(tao->ksp, 1e-14, 1e-30, 1e30, PetscMax(10,qp->n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

