/*$Id: umtr.c,v 1.109 2001/07/10 18:08:00 buschelm Exp bsmith $*/

#include "src/snes/impls/umtr/umtr.h"                /*I "petscsnes.h" I*/

/*
    SNESSolve_UM_TR - Implements Newton's Method with a trust region approach 
    for solving unconstrained minimization problems.  

    The basic algorithm is taken from MINPACK-2 (dstrn).

    SNESSolve_UM_TR computes a local minimizer of a twice differentiable function
    f  by applying a trust region variant of Newton's method.  At each stage 
    of the algorithm, we us the prconditioned conjugate gradient method to
    determine an approximate minimizer of the quadratic equation

	     q(s) = g^T * s + .5 * s^T * H * s

    subject to the Euclidean norm trust region constraint

	     || D * s || <= delta,

    where delta is the trust region radius and D is a scaling matrix.
    Here g is the gradient and H is the Hessian matrix.

    Note:  SNESSolve_UM_TR MUST use the iterative solver KSPQCG; thus, we
           set KSPQCG in this routine regardless of what the user may have
           previously specified.
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSolve_UM_TR"
static int SNESSolve_UM_TR(SNES snes,int *outits)
{
  SNES_UM_TR          *neP = (SNES_UM_TR*)snes->data;
  int                 maxits,i,nlconv,ierr,qits;
  PetscTruth          newton;
  double              xnorm,max_val,ftrial,delta,ltsnrm,quadratic;
  double              zero = 0.0,two = 2.0,four = 4.0;
  Scalar              one = 1.0;
  Vec                 X,G,S,Xtrial;
  MatStructure        flg = DIFFERENT_NONZERO_PATTERN;
  SLES                sles;
  KSP                 ksp;
  SNESConvergedReason reason;
  KSPConvergedReason  kreason;

  PetscFunctionBegin;

  snes->reason  = SNES_CONVERGED_ITERATING;

  nlconv        = 0;
  maxits	= snes->max_its;         /* maximum number of iterations */
  X		= snes->vec_sol; 	 /* solution vector */
  G		= snes->vec_func;	 /* gradient vector */
  S		= snes->work[0]; 	 /* work vectors */
  Xtrial	= snes->work[1]; 
  delta	        = neP->delta;           /* trust region radius */

  ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);              /* xnorm = || X || */
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeMinimizationFunction(snes,X,&snes->fc);CHKERRQ(ierr); /* f(X) */
  ierr = SNESComputeGradient(snes,X,G);CHKERRQ(ierr);  /* G(X) <- gradient */
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  ierr = VecNorm(G,NORM_2,&snes->norm);CHKERRQ(ierr);               /* &snes->norm = || G || */
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,snes->norm,0);
  SNESMonitor(snes,0,snes->norm);

  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetKSP(sles,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPQCG);CHKERRQ(ierr);
  PetscLogInfo(snes,"SNESSolve_UM_TR: setting KSPType = KSPQCG\n");

  for (i=0; i<maxits && !nlconv; i++) {
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter      = i+1;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    newton          = PETSC_FALSE;
    neP->success    = 0;
    snes->nfailures = 0;
    ierr = SNESComputeHessian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr);

    if (!i) {			/* Initialize delta */
      if (delta <= 0) {
        if (xnorm > zero) delta = neP->factor1*xnorm;
        else delta = neP->delta0;
        ierr = MatNorm(snes->jacobian,NORM_1,&max_val);CHKERRQ(ierr);
        if (ierr == PETSC_ERR_SUP) {
          PetscLogInfo(snes,"SNESSolve_UM_TR: Initial delta computed without matrix norm info\n");
        } else {
          if (PetscAbsDouble(max_val)<1.e-14)SETERRQ(PETSC_ERR_PLIB,"Hessian norm is too small");
          delta = PetscMax(delta,snes->norm/max_val);
        }
      } else { 
        delta = neP->delta0;
      }
    }
    do {
      /* Minimize the quadratic to compute the step s */
      ierr = KSPQCGSetTrustRegionRadius(ksp,delta);CHKERRQ(ierr);

      ierr = SLESSolve(snes->sles,G,S,&qits);CHKERRQ(ierr);
      snes->linear_its += qits;
      ierr = KSPQCGGetTrialStepNorm(ksp,&ltsnrm);CHKERRQ(ierr);
      ierr = KSPQCGGetQuadratic(ksp,&quadratic);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(ksp,&kreason);CHKERRQ(ierr);
      if ((int)kreason < 0) SETERRQ(PETSC_ERR_PLIB,"Failure in SLESSolve");
      if (kreason != KSP_CONVERGED_QCG_NEG_CURVE && kreason != KSP_CONVERGED_QCG_CONSTRAINED) {
        newton = PETSC_TRUE;
      }
      PetscLogInfo(snes,"SNESSolve_UM_TR: %d: ltsnrm=%g, delta=%g, q=%g, qits=%d\n", 
               i,ltsnrm,delta,quadratic,qits);

      ierr = VecWAXPY(&one,X,S,Xtrial);CHKERRQ(ierr); /* Xtrial <- X + S */
      ierr = VecNorm(Xtrial,NORM_2,&xnorm);CHKERRQ(ierr);

                           		               /* ftrial = f(Xtrial) */
      ierr = SNESComputeMinimizationFunction(snes,Xtrial,&ftrial);CHKERRQ(ierr);

      /* Compute the function reduction and the step size */
      neP->prered = -quadratic;
      neP->actred = snes->fc - ftrial;

      /* Adjust delta for the first Newton step */
      if (!i && (newton)) delta = PetscMin(delta,ltsnrm);

      if (neP->actred < neP->eta1 * neP->prered) {  /* Unsuccessful step */

         PetscLogInfo(snes,"SNESSolve_UM_TR: Rejecting step\n");
         snes->nfailures += 1;

         /* If iterate is Newton step, reduce delta to current step length */
         if (newton) {
           delta  = ltsnrm;
           newton = PETSC_FALSE;
         }
         delta /= four; 

      } else {          /* Successful iteration; adjust trust radius */

        neP->success = 1;
        PetscLogInfo(snes,"SNESSolve_UM_TR: Accepting step\n");
        if (newton) {
           delta = sqrt(ltsnrm*delta);
           if (neP->actred < neP->eta2 * neP->prered) delta /= two;
        } else if (neP->actred < neP->eta2 * neP->prered)
           delta /= delta;
        else if ((neP->actred >= neP->eta3 * neP->prered) && 
           (neP->actred < neP->eta4 * neP->prered))
           delta *= two;
        else if (neP->actred >= neP->eta4 * neP->prered)
           delta *= four;
        else neP->sflag = 1;
      }

      neP->delta = delta;
      ierr = (*snes->converged)(snes,xnorm,snes->norm,ftrial,&reason,snes->cnvP);CHKERRQ(ierr);
      if (reason) nlconv = 1;
    } while (!neP->success && !nlconv);

    /* Question:  If (!neP->success && break), then last step was rejected, 
       but convergence was detected.  Should this update really happen? */
    snes->fc = ftrial;
    ierr = VecCopy(Xtrial,X);CHKERRQ(ierr);
    snes->vec_sol_always = X;
    /* Note:  At last iteration, the gradient evaluation is unnecessary */
    ierr = SNESComputeGradient(snes,X,G);CHKERRQ(ierr);
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    ierr = VecNorm(G,NORM_2,&snes->norm);CHKERRQ(ierr);
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    snes->vec_func_always = G;

    SNESLogConvHistory(snes,snes->norm,qits);
    SNESMonitor(snes,i+1,snes->norm);
  }
  /* Verify solution is in corect location */
  if (X != snes->vec_sol) {
    ierr = VecCopy(X,snes->vec_sol);CHKERRQ(ierr);
    snes->vec_sol_always  = snes->vec_sol;
    snes->vec_func_always = snes->vec_func; 
  }
  if (i == maxits) {
    PetscLogInfo(snes,"SNESSolve_UM_TR: Maximum number of iterations reached: %d\n",maxits);
    i--;
    reason = SNES_DIVERGED_MAX_IT;
  }
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->reason = reason;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  *outits = i;  /* not i+1, since update for i happens in loop above */
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp_UM_TR"
static int SNESSetUp_UM_TR(SNES snes)
{
  int        ierr;
  PetscTruth ilu,bjacobi;
  SLES       sles;
  PC         pc;

  PetscFunctionBegin;
  snes->nwork = 4;
  ierr = VecDuplicateVecs(snes->vec_sol,snes->nwork,&snes->work);CHKERRQ(ierr);
  PetscLogObjectParents(snes,snes->nwork,snes->work);
  snes->vec_sol_update_always = snes->work[3];

  /* 
       If PC was set by default to ILU then change it to Jacobi
  */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCILU,&ilu);CHKERRQ(ierr);
  if (ilu) {
    ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  } else {
    ierr = PetscTypeCompare((PetscObject)pc,PCBJACOBI,&bjacobi);CHKERRQ(ierr);
    if (bjacobi) {
       /* cannot do this; since PC may not have been setup yet */
       /* ierr = PCBJacobiGetSubSLES(pc,0,0,&subsles);CHKERRQ(ierr);
       ierr = SLESGetPC(*subsles,&subpc);CHKERRQ(ierr);
       ierr = PetscTypeCompare((PetscObject)subpc,PCILU,&ilu);CHKERRQ(ierr);
       if (ilu) {
         ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
       } */
       /* don't really want to do this, since user may have selected BJacobi plus something
         that is symmetric on each processor; really only want to catch the default ILU */
      ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy_UM_TR"
static int SNESDestroy_UM_TR(SNES snes)
{
  int  ierr;

  PetscFunctionBegin;
  if (snes->nwork) {
    ierr = VecDestroyVecs(snes->work,snes->nwork);CHKERRQ(ierr);
  }
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "SNESConverged_UM_TR"
/*@C
   SNESConverged_UM_TR - Monitors the convergence of the SNESSolve_UM_TR()
   routine (default). 

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  gnorm - 2-norm of current gradient
.  f - objective function value
-  dummy - unused dummy context

   Output Parameter:
.  reason - one of 
$   SNES_CONVERGED_FNORM_ABS         (f < fmin),
$   SNES_CONVERGED_TR_REDUCTION      (abs(ared) <= rtol*abs(f) && pred <= rtol*abs(f)),
$   SNES_CONVERGED_TR_DELTA          (delta <= deltatol*xnorm),
$   SNES_DIVERGED_TR_REDUCTION       (abs(ared) <= epsmch && pred <= epsmch),
$   SNES_DIVERGED_FUNCTION_COUNT     (nfunc > max_func),
$   SNES_DIVERGED_FNORM_NAN          (f = NaN),
$   SNES_CONVERGED_ITERATING         (otherwise).

   where
+    ared     - actual reduction
.    delta    - trust region paramenter
.    deltatol - trust region size tolerance,
                set with SNESSetTrustRegionTolerance()
.    epsmch   - machine epsilon
.    fmin     - lower bound on function value,
                set with SNESSetMinimizationFunctionTolerance()
.    nfunc    - number of function evaluations
.    maxfunc  - maximum number of function evaluations, 
                set with SNESSetTolerances()
.    pred     - predicted reduction
-    rtol     - relative function tolerance, 
                set with SNESSetTolerances()

   Level: intermediate

@*/
int SNESConverged_UM_TR(SNES snes,double xnorm,double gnorm,double f,SNESConvergedReason *reason,void *dummy)
{
  SNES_UM_TR *neP = (SNES_UM_TR*)snes->data;
  double     rtol = snes->rtol,delta = neP->delta,ared = neP->actred,pred = neP->prered;
  double     epsmch = 1.0e-14;   /* This must be fixed */

  PetscFunctionBegin;
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"For SNES_UNCONSTRAINED_MINIMIZATION only");
  } else if (f != f) {
    PetscLogInfo(snes,"SNESConverged_UM_TR:Failed to converged, function is NaN\n");
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if ((!neP->success || neP->sflag) && (delta <= snes->deltatol * xnorm)) {
    neP->sflag = 0;
    PetscLogInfo(snes,"SNESConverged_UM_TR: Trust region param satisfies tolerance: %g<=%g*%g\n",
             delta,snes->deltatol,xnorm);  
    *reason = SNES_CONVERGED_TR_DELTA;
  } else if ((PetscAbsDouble(ared) <= PetscAbsDouble(f) * rtol) && (pred) <= rtol*PetscAbsDouble(f)) {
    PetscLogInfo(snes,"SNESConverged_UM_TR:Actual (%g) and predicted (%g) reductions<%g*%g\n",
             PetscAbsDouble(ared),pred,rtol,PetscAbsDouble(f));
    *reason = SNES_CONVERGED_TR_REDUCTION;
  } else if (f < snes->fmin) {
    PetscLogInfo(snes,"SNESConverged_UM_TR:Function value (%g)<f_{minimum} (%g)\n",f,snes->fmin);
    *reason = SNES_CONVERGED_FNORM_ABS ;
  } else if ((PetscAbsDouble(ared) <= epsmch) && pred <= epsmch) {
    PetscLogInfo(snes,"SNESConverged_UM_TR:Actual (%g) and predicted (%g) reductions<epsmch (%g)\n",
             PetscAbsDouble(ared),pred,epsmch);
    *reason = SNES_DIVERGED_TR_REDUCTION;
  } else if (snes->nfuncs > snes->max_funcs) {
    PetscLogInfo(snes,"SNESConverged_UM_TR:Exceeded maximum number of function evaluations:%d>%d\n",
             snes->nfuncs,snes->max_funcs); 
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  } else {
    *reason = SNES_CONVERGED_ITERATING;
  }
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_UM_TR"
static int SNESSetFromOptions_UM_TR(SNES snes)
{
  SNES_UM_TR *ctx = (SNES_UM_TR *)snes->data;
  int        ierr;
  SLES       sles;
  PC         pc;
  PetscTruth ismatshell,nopcset;
  Mat        pmat;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES trust region options for minimization");CHKERRQ(ierr);
    ierr = PetscOptionsDouble("-snes_trtol","Trust region tolerance","SNESSetTrustRegionTolerance",snes->deltatol,&snes->deltatol,0);CHKERRQ(ierr);
    ierr = PetscOptionsDouble("-snes_um_eta1","eta1","None",ctx->eta1,&ctx->eta1,0);CHKERRQ(ierr);
    ierr = PetscOptionsDouble("-snes_um_eta2","step unsuccessful if reduction < eta1 * predicted reduction","None",ctx->eta2,&ctx->eta2,0);CHKERRQ(ierr);
    ierr = PetscOptionsDouble("-snes_um_eta3","eta3","None",ctx->eta3,&ctx->eta3,0);CHKERRQ(ierr);
    ierr = PetscOptionsDouble("-snes_um_eta4","eta4","None",ctx->eta4,&ctx->eta4,0);CHKERRQ(ierr);
    ierr = PetscOptionsDouble("-snes_um_delta0","delta0","None",ctx->delta,&ctx->delta,0);CHKERRQ(ierr);
    ierr = PetscOptionsDouble("-snes_um_factor1","factor1","None",ctx->factor1,&ctx->factor1,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  /* if preconditioner has not been set yet, and not using a matrix shell then
     set preconditioner to Jacobi. This is to prevent PCSetFromOptions() from 
     setting a default of ILU or block Jacobi-ILU which won't work since TR 
     requires a symmetric preconditioner
  */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,0,&nopcset);CHKERRQ(ierr);
  if (nopcset) {
    ierr = PCGetOperators(pc,PETSC_NULL,&pmat,PETSC_NULL);CHKERRQ(ierr);
    if (pmat) {
      ierr = PetscTypeCompare((PetscObject)pmat,MATSHELL,&ismatshell);CHKERRQ(ierr);
      if (!ismatshell) {
        ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "SNESView_UM_TR"
static int SNESView_UM_TR(SNES snes,PetscViewer viewer)
{
  SNES_UM_TR *tr = (SNES_UM_TR *)snes->data;
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  eta1=%g, eta1=%g, eta3=%g, eta4=%g\n",tr->eta1,tr->eta2,tr->eta3,tr->eta4);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  delta0=%g, factor1=%g\n",tr->delta0,tr->factor1);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for SNES UM TR",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_UM_TR"
int SNESCreate_UM_TR(SNES snes)
{
  SNES_UM_TR *neP;
  int        ierr;

  PetscFunctionBegin;
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"For SNES_UNCONSTRAINED_MINIMIZATION only");
  }
  snes->setup		= SNESSetUp_UM_TR;
  snes->solve		= SNESSolve_UM_TR;
  snes->destroy		= SNESDestroy_UM_TR;
  snes->converged	= SNESConverged_UM_TR;
  snes->setfromoptions  = SNESSetFromOptions_UM_TR;
  snes->view            = SNESView_UM_TR;

  snes->nwork           = 0;

  ierr			= PetscNew(SNES_UM_TR,&neP);CHKERRQ(ierr);
  PetscLogObjectMemory(snes,sizeof(SNES_UM_TR));
  snes->data	        = (void*)neP;
  neP->delta0		= 1.0e-6;
  neP->delta 		= 0.0;
  neP->eta1		= 1.0e-4;
  neP->eta2		= 0.25;
  neP->eta3		= 0.50;
  neP->eta4		= 0.90;
  neP->factor1		= 1.0e-8;
  neP->actred		= 0.0;
  neP->prered		= 0.0;
  neP->success		= 0;
  neP->sflag		= 0;
 
  PetscFunctionReturn(0);
}
EXTERN_C_END
