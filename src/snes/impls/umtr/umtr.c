#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: umtr.c,v 1.85 1999/09/02 14:54:05 bsmith Exp bsmith $";
#endif

#include "src/snes/impls/umtr/umtr.h"                /*I "snes.h" I*/
#include "src/sles/ksp/kspimpl.h"
#include "src/sles/ksp/impls/qcg/qcg.h"

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
#undef __FUNC__  
#define __FUNC__ "SNESSolve_UM_TR"
static int SNESSolve_UM_TR(SNES snes,int *outits)
{
  SNES_UMTR           *neP = (SNES_UMTR *) snes->data;
  int                 maxits, i, nlconv, ierr, qits, newton;
  double              *gnorm, xnorm, max_val, ftrial, delta;
  double              zero = 0.0, *f, two = 2.0, four = 4.0;
  Scalar              one = 1.0;
  Vec                 X, G,  S, Xtrial;
  MatStructure        flg = DIFFERENT_NONZERO_PATTERN;
  SLES                sles;
  KSP                 ksp;
  KSP_QCG             *qcgP;
  SNESConvergedReason reason;

  PetscFunctionBegin;

  snes->reason  = SNES_CONVERGED_ITERATING;

  nlconv        = 0;
  maxits	= snes->max_its;         /* maximum number of iterations */
  X		= snes->vec_sol; 	 /* solution vector */
  G		= snes->vec_func;	 /* gradient vector */
  S		= snes->work[0]; 	 /* work vectors */
  Xtrial	= snes->work[1]; 
  delta	        = neP->delta;           /* trust region radius */
  f		= &(snes->fc);		/* function to minimize */
  gnorm		= &(snes->norm);	/* gradient norm */

  ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);              /* xnorm = || X || */
  ierr = PetscAMSTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  ierr = PetscAMSGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeMinimizationFunction(snes,X,f);CHKERRQ(ierr); /* f(X) */
  ierr = SNESComputeGradient(snes,X,G);CHKERRQ(ierr);  /* G(X) <- gradient */
  ierr = PetscAMSTakeAccess(snes);CHKERRQ(ierr);
  ierr = VecNorm(G,NORM_2,gnorm);CHKERRQ(ierr);               /* gnorm = || G || */
  ierr = PetscAMSGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,*gnorm,0);
  SNESMonitor(snes,0,*gnorm);

  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetKSP(sles,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPQCG);CHKERRQ(ierr);
  PLogInfo(snes,"SNESSolve_UM_TR: setting KSPType = KSPQCG\n");
  qcgP = (KSP_QCG *) ksp->data;

  for ( i=0; i<maxits && !nlconv; i++ ) {
    ierr = PetscAMSTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    ierr = PetscAMSGrantAccess(snes);CHKERRQ(ierr);
    newton = 0;
    neP->success = 0;
    snes->nfailures = 0;
    ierr = SNESComputeHessian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr);

    if (i == 0) {			/* Initialize delta */
      if (delta <= 0) {
        if (xnorm > zero) delta = neP->factor1*xnorm;
        else delta = neP->delta0;
        ierr = MatNorm(snes->jacobian,NORM_1,&max_val);CHKERRQ(ierr);
        if (ierr == PETSC_ERR_SUP) {
          PLogInfo(snes,"SNESSolve_UM_TR: Initial delta computed without matrix norm info\n");
        } else {
          if (PetscAbsDouble(max_val)<1.e-14)SETERRQ(PETSC_ERR_PLIB,0,"Hessian norm is too small");
          delta = PetscMax(delta,*gnorm/max_val);
        }
      } else { 
        delta = neP->delta0;
      }
    }
    do {
      /* Minimize the quadratic to compute the step s */
      qcgP->delta = delta;
      ierr = SLESSolve(snes->sles,G,S,&qits);CHKERRQ(ierr);
      snes->linear_its += PetscAbsInt(qits);
      if (qits < 0) SETERRQ(PETSC_ERR_PLIB,0,"Failure in SLESSolve");
      if (qcgP->info == 3) newton = 1;	            /* truncated Newton step */
      PLogInfo(snes,"SNESSolve_UM_TR: %d: ltsnrm=%g, delta=%g, q=%g, qits=%d\n", 
               i, qcgP->ltsnrm, delta, qcgP->quadratic, qits );

      ierr = VecWAXPY(&one,X,S,Xtrial);CHKERRQ(ierr); /* Xtrial <- X + S */
      ierr = VecNorm(Xtrial,NORM_2,&xnorm);CHKERRQ(ierr);
                           		               /* ftrial = f(Xtrial) */
      ierr = SNESComputeMinimizationFunction(snes,Xtrial,&ftrial);CHKERRQ(ierr);

      /* Compute the function reduction and the step size */
      neP->actred = *f - ftrial;
      neP->prered = -qcgP->quadratic;

      /* Adjust delta for the first Newton step */
      if ((i == 0) && (newton)) delta = PetscMin(delta,qcgP->ltsnrm);

      if (neP->actred < neP->eta1 * neP->prered) {  /* Unsuccessful step */

         PLogInfo(snes,"SNESSolve_UM_TR: Rejecting step\n");
         snes->nfailures += 1;

         /* If iterate is Newton step, reduce delta to current step length */
         if (newton) {
           delta = qcgP->ltsnrm;
           newton = 0;
         }
         delta /= four; 

      } else {          /* Successful iteration; adjust trust radius */

        neP->success = 1;
        PLogInfo(snes,"SNESSolve_UM_TR: Accepting step\n");
        if (newton) {
           delta = sqrt(qcgP->ltsnrm*delta);
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
      ierr = (*snes->converged)(snes,xnorm,*gnorm,ftrial,&reason,snes->cnvP);CHKERRQ(ierr);
      if (reason) nlconv = 1;
    } while (!neP->success && !nlconv);

    /* Question:  If (!neP->success && break), then last step was rejected, 
       but convergence was detected.  Should this update really happen? */
    *f = ftrial;
    ierr = VecCopy(Xtrial,X);CHKERRQ(ierr);
    snes->vec_sol_always = X;
    /* Note:  At last iteration, the gradient evaluation is unnecessary */
    ierr = SNESComputeGradient(snes,X,G);CHKERRQ(ierr);
    ierr = PetscAMSTakeAccess(snes);CHKERRQ(ierr);
    ierr = VecNorm(G,NORM_2,gnorm);CHKERRQ(ierr);
    ierr = PetscAMSGrantAccess(snes);CHKERRQ(ierr);
    snes->vec_func_always = G;

    SNESLogConvHistory(snes,*gnorm,qits);
    SNESMonitor(snes,i+1,*gnorm);
  }
  /* Verify solution is in corect location */
  if (X != snes->vec_sol) {
    ierr = VecCopy(X,snes->vec_sol);CHKERRQ(ierr);
    snes->vec_sol_always = snes->vec_sol;
    snes->vec_func_always = snes->vec_func; 
  }
  if (i == maxits) {
    PLogInfo(snes,"SNESSolve_UM_TR: Maximum number of iterations reached: %d\n",maxits);
    i--;
    reason = SNES_DIVERGED_MAX_IT;
  }
  snes->reason = SNES_DIVERGED_MAX_IT;
  *outits = i;  /* not i+1, since update for i happens in loop above */
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESSetUp_UM_TR"
static int SNESSetUp_UM_TR(SNES snes)
{
  int ierr;

  PetscFunctionBegin;
  snes->nwork = 4;
  ierr = VecDuplicateVecs(snes->vec_sol,snes->nwork,&snes->work);CHKERRQ(ierr);
  PLogObjectParents(snes,snes->nwork,snes->work);
  snes->vec_sol_update_always = snes->work[3];
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESDestroy_UM_TR"
static int SNESDestroy_UM_TR(SNES snes )
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
#undef __FUNC__  
#define __FUNC__ "SNESConverged_UM_TR"
/*@ 
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
$   SNES_CONVERGED_FNORM_ABS         ( f < fmin ),
$   SNES_CONVERGED_TR_REDUCTION      ( abs(ared) <= rtol*abs(f) && pred <= rtol*abs(f) ),
$   SNES_CONVERGED_TR_DELTA          ( delta <= deltatol*xnorm ),
$   SNES_DIVERGED_TR_REDUCTION       ( abs(ared) <= epsmch && pred <= epsmch ),
$   SNES_DIVERGED_FUNCTION_COUNT     ( nfunc > max_func ),
$   SNES_DIVERGED_FNORM_NAN          ( f = NaN ),
$   SNES_CONVERGED_ITERATING         ( otherwise ).

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
  SNES_UMTR *neP = (SNES_UMTR *) snes->data;
  double    rtol = snes->rtol, delta = neP->delta,ared = neP->actred, pred = neP->prered;
  double    epsmch = 1.0e-14;   /* This must be fixed */

  PetscFunctionBegin;
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"For SNES_UNCONSTRAINED_MINIMIZATION only");
  } else if (f != f) {
    PLogInfo(snes,"SNESConverged_UM_TR:Failed to converged, function is NaN\n");
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if ((!neP->success || neP->sflag) && (delta <= snes->deltatol * xnorm)) {
    neP->sflag = 0;
    PLogInfo(snes,"SNESConverged_UM_TR: Trust region param satisfies tolerance: %g<=%g*%g\n",
             delta,snes->deltatol,xnorm);  
    *reason = SNES_CONVERGED_TR_DELTA;
  } else if ((PetscAbsDouble(ared) <= PetscAbsDouble(f) * rtol) && (pred) <= rtol*PetscAbsDouble(f)) {
    PLogInfo(snes,"SNESConverged_UM_TR:Actual (%g) and predicted (%g) reductions<%g*%g\n",
             PetscAbsDouble(ared),pred,rtol,PetscAbsDouble(f));
    *reason = SNES_CONVERGED_TR_REDUCTION;
  } else if (f < snes->fmin) {
    PLogInfo(snes,"SNESConverged_UM_TR:Function value (%g)<f_{minimum} (%g)\n",f,snes->fmin);
    *reason = SNES_CONVERGED_FNORM_ABS ;
  } else if ( (PetscAbsDouble(ared) <= epsmch) && pred <= epsmch ) {
    PLogInfo(snes,"SNESConverged_UM_TR:Actual (%g) and predicted (%g) reductions<epsmch (%g)\n",
             PetscAbsDouble(ared),pred,epsmch);
    *reason = SNES_DIVERGED_TR_REDUCTION;
  } else if (snes->nfuncs > snes->max_funcs) {
    PLogInfo(snes,"SNESConverged_UM_TR:Exceeded maximum number of function evaluations:%d>%d\n",
             snes->nfuncs, snes->max_funcs ); 
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  } else {
    *reason = SNES_CONVERGED_ITERATING;
  }
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESSetFromOptions_UM_TR"
static int SNESSetFromOptions_UM_TR(SNES snes)
{
  SNES_UMTR *ctx = (SNES_UMTR *)snes->data;
  double    tmp;
  int       ierr, flg;

  PetscFunctionBegin;
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_eta1",&tmp,&flg);CHKERRQ(ierr);
  if (flg) {ctx->eta1 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_eta2",&tmp,&flg);CHKERRQ(ierr);
  if (flg) {ctx->eta2 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_eta3",&tmp,&flg);CHKERRQ(ierr);
  if (flg) {ctx->eta3 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_eta4",&tmp,&flg);CHKERRQ(ierr);
  if (flg) {ctx->eta4 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_delta0",&tmp,&flg);CHKERRQ(ierr);
  if (flg) {ctx->delta0 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_um_factor1",&tmp,&flg);CHKERRQ(ierr);
  if (flg) {ctx->factor1 = tmp;}

  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESPrintHelp_UM_TR"
static int SNESPrintHelp_UM_TR(SNES snes,char *p)
{
  SNES_UMTR *ctx = (SNES_UMTR *)snes->data;
  int       ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(snes->comm," method SNES_UM_TR (umtr) for unconstrained minimization:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(snes->comm,"   %ssnes_um_tr_eta1 <eta1> (default %g)\n",p,ctx->eta1);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(snes->comm,"   %ssnes_um_tr_eta2 <eta2> (default %g)\n",p,ctx->eta2);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(snes->comm,"   %ssnes_um_tr_eta3 <eta3> (default %g)\n",p,ctx->eta3);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(snes->comm,"   %ssnes_um_tr_eta4 <eta4> (default %g)\n",p,ctx->eta4);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(snes->comm,"   %ssnes_um_tr_delta0 <delta0> (default %g)\n",p,ctx->delta0);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(snes->comm,"   %ssnes_um_tr_factor1 <factor1> (default %g)\n",p,ctx->factor1);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(snes->comm,"   delta0, factor1: used to initialize trust region parameter\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(snes->comm,"   eta2, eta3, eta4: used to compute trust region parameter\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(snes->comm,"   eta1: step is unsuccessful if actred < eta1 * prered, where\n");CHKERRQ(ierr); 
  ierr = (*PetscHelpPrintf)(snes->comm,"         pred = predicted reduction, actred = actual reduction\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESView_UM_TR"
static int SNESView_UM_TR(SNES snes,Viewer viewer)
{
  SNES_UMTR  *tr = (SNES_UMTR *)snes->data;
  int        ierr;
  ViewerType vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype);CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    ierr = ViewerASCIIPrintf(viewer,"  eta1=%g, eta1=%g, eta3=%g, eta4=%g\n",tr->eta1,tr->eta2,tr->eta3,tr->eta4);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"  delta0=%g, factor1=%g\n",tr->delta0,tr->factor1);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "SNESCreate_UM_TR"
int SNESCreate_UM_TR(SNES snes)
{
  SNES_UMTR *neP;
  SLES      sles;
  PC        pc;
  int       ierr;

  PetscFunctionBegin;
  if (snes->method_class != SNES_UNCONSTRAINED_MINIMIZATION) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"For SNES_UNCONSTRAINED_MINIMIZATION only");
  }
  snes->setup		= SNESSetUp_UM_TR;
  snes->solve		= SNESSolve_UM_TR;
  snes->destroy		= SNESDestroy_UM_TR;
  snes->converged	= SNESConverged_UM_TR;
  snes->printhelp       = SNESPrintHelp_UM_TR;
  snes->setfromoptions  = SNESSetFromOptions_UM_TR;
  snes->view            = SNESView_UM_TR;

  snes->nwork           = 0;

  neP			= PetscNew(SNES_UMTR);CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_UM_TR));
  snes->data	        = (void *) neP;
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
 
  /* Set default preconditioner to be Jacobi, to override SLES default. */
  /* This implementation currently requires a symmetric preconditioner. */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
