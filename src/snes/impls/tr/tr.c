/*$Id: tr.c,v 1.107 1999/11/10 03:21:21 bsmith Exp bsmith $*/

#include "src/snes/impls/tr/tr.h"                /*I   "snes.h"   I*/

/*
   This convergence test determines if the two norm of the 
   solution lies outside the trust region, if so it halts.
*/
#undef __FUNC__  
#define __FUNC__ "SNES_EQ_TR_KSPConverged_Private"
int SNES_EQ_TR_KSPConverged_Private(KSP ksp,int n, double rnorm, void *ctx)
{
  SNES                snes = (SNES) ctx;
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;
  SNES_EQ_TR          *neP = (SNES_EQ_TR*)snes->data;
  Vec                 x;
  double              norm;
  int                 ierr, convinfo;

  PetscFunctionBegin;
  if (snes->ksp_ewconv) {
    if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Eisenstat-Walker onvergence context not created");
    if (!n) {ierr = SNES_KSP_EW_ComputeRelativeTolerance_Private(snes,ksp);CHKERRQ(ierr);}
    kctx->lresid_last = rnorm;
  }
  convinfo = KSPDefaultConverged(ksp,n,rnorm,ctx);
  if (convinfo) {
    PLogInfo(snes,"SNES_EQ_TR_KSPConverged_Private: KSP iterations=%d, rnorm=%g\n",n,rnorm);
    PetscFunctionReturn(convinfo);
  }

  /* Determine norm of solution */
  ierr = KSPBuildSolution(ksp,0,&x);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  if (norm >= neP->delta) {
    PLogInfo(snes,"SNES_EQ_TR_KSPConverged_Private: KSP iterations=%d, rnorm=%g\n",n,rnorm);
    PLogInfo(snes,"SNES_EQ_TR_KSPConverged_Private: Ending linear iteration early, delta=%g, length=%g\n",neP->delta,norm);
    PetscFunctionReturn(1);
  }
  PetscFunctionReturn(0);
}

/*
   SNESSolve_EQ_TR - Implements Newton's Method with a very simple trust 
   region approach for solving systems of nonlinear equations. 

   The basic algorithm is taken from "The Minpack Project", by More', 
   Sorensen, Garbow, Hillstrom, pages 88-111 of "Sources and Development 
   of Mathematical Software", Wayne Cowell, editor.

   This is intended as a model implementation, since it does not 
   necessarily have many of the bells and whistles of other 
   implementations.  
*/
#undef __FUNC__  
#define __FUNC__ "SNESSolve_EQ_TR"
static int SNESSolve_EQ_TR(SNES snes,int *its)
{
  SNES_EQ_TR          *neP = (SNES_EQ_TR *) snes->data;
  Vec                 X, F, Y, G, TMP, Ytmp;
  int                 maxits, i, ierr, lits, breakout = 0;
  MatStructure        flg = DIFFERENT_NONZERO_PATTERN;
  double              rho, fnorm, gnorm, gpnorm, xnorm, delta,norm,ynorm,norm1;
  Scalar              mone = -1.0,cnorm;
  KSP                 ksp;
  SLES                sles;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  Ytmp          = snes->work[2];

  ierr       = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);         /* xnorm = || X || */  

  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);          /* F(X) */
  ierr = VecNorm(F, NORM_2,&fnorm );CHKERRQ(ierr);             /* fnorm <- || F || */
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  snes->iter = 0;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  delta = neP->delta0*fnorm;         
  neP->delta = delta;
  SNESLogConvHistory(snes,fnorm,0);
  SNESMonitor(snes,0,fnorm);

 if (fnorm < snes->atol) {*its = 0; snes->reason = SNES_CONVERGED_FNORM_ABS; PetscFunctionReturn(0);}

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;

  /* Set the stopping criteria to use the More' trick. */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetKSP(sles,&ksp);CHKERRQ(ierr);
  ierr = KSPSetConvergenceTest(ksp,SNES_EQ_TR_KSPConverged_Private,(void *)snes);CHKERRQ(ierr);
 
  for ( i=0; i<maxits; i++ ) {
    ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,&flg);CHKERRQ(ierr);
    ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,flg);CHKERRQ(ierr);

    /* Solve J Y = F, where J is Jacobian matrix */
    ierr = SLESSolve(snes->sles,F,Ytmp,&lits);CHKERRQ(ierr);
    snes->linear_its += PetscAbsInt(lits);
    PLogInfo(snes,"SNESSolve_EQ_TR: iter=%d, linear solve iterations=%d\n",snes->iter,lits);
    ierr = VecNorm(Ytmp,NORM_2,&norm);CHKERRQ(ierr);
    norm1 = norm;
    while(1) {
      ierr = VecCopy(Ytmp,Y);CHKERRQ(ierr);
      norm = norm1;

      /* Scale Y if need be and predict new value of F norm */
      if (norm >= delta) {
        norm = delta/norm;
        gpnorm = (1.0 - norm)*fnorm;
        cnorm = norm;
        PLogInfo(snes,"SNESSolve_EQ_TR: Scaling direction by %g\n",norm );
        ierr = VecScale(&cnorm,Y);CHKERRQ(ierr);
        norm = gpnorm;
        ynorm = delta;
      } else {
        gpnorm = 0.0;
        PLogInfo(snes,"SNESSolve_EQ_TR: Direction is in Trust Region\n" );
        ynorm = norm;
      }
      ierr = VecAYPX(&mone,X,Y);CHKERRQ(ierr);            /* Y <- X - Y */
      ierr = VecCopy(X,snes->vec_sol_update_always);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes,Y,G);CHKERRQ(ierr); /*  F(X) */
      ierr = VecNorm(G,NORM_2,&gnorm);CHKERRQ(ierr);      /* gnorm <- || g || */
      if (fnorm == gpnorm) rho = 0.0;
      else rho = (fnorm*fnorm - gnorm*gnorm)/(fnorm*fnorm - gpnorm*gpnorm); 

      /* Update size of trust region */
      if      (rho < neP->mu)  delta *= neP->delta1;
      else if (rho < neP->eta) delta *= neP->delta2;
      else                     delta *= neP->delta3;
      PLogInfo(snes,"SNESSolve_EQ_TR: fnorm=%g, gnorm=%g, ynorm=%g\n",fnorm,gnorm,ynorm);
      PLogInfo(snes,"SNESSolve_EQ_TR: gpred=%g, rho=%g, delta=%g\n",gpnorm,rho,delta);
      neP->delta = delta;
      if (rho > neP->sigma) break;
      PLogInfo(snes,"SNESSolve_EQ_TR: Trying again in smaller region\n");
      /* check to see if progress is hopeless */
      neP->itflag = 0;
      ierr = (*snes->converged)(snes,xnorm,ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      if (reason) {
        /* We're not progressing, so return with the current iterate */
        breakout = 1;
        break;
      }
      snes->nfailures++;
    }
    if (!breakout) {
      fnorm = gnorm;
      ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
      snes->iter = i+1;
      snes->norm = fnorm;
      ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
      TMP = F; F = G; snes->vec_func_always = F; G = TMP;
      TMP = X; X = Y; snes->vec_sol_always = X; Y = TMP;
      VecNorm(X, NORM_2,&xnorm );		/* xnorm = || X || */
      SNESLogConvHistory(snes,fnorm,lits);
      SNESMonitor(snes,i+1,fnorm);

      /* Test for convergence */
      neP->itflag = 1;
      ierr = (*snes->converged)(snes,xnorm,ynorm,fnorm,&reason,snes->cnvP);CHKERRQ(ierr);
      if (reason) {
        break;
      } 
    } else {
      break;
    }
  }
  if (X != snes->vec_sol) {
    /* Verify solution is in corect location */
    ierr = VecCopy(X,snes->vec_sol);CHKERRQ(ierr);
    snes->vec_sol_always  = snes->vec_sol;
    snes->vec_func_always = snes->vec_func; 
  }
  if (i == maxits) {
    PLogInfo(snes,"SNESSolve_EQ_TR: Maximum number of iterations has been reached: %d\n",maxits);
    i--;
    reason = SNES_DIVERGED_MAX_IT;
  }
  *its = i+1;
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->reason = reason;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESSetUp_EQ_TR"
static int SNESSetUp_EQ_TR(SNES snes)
{
  int ierr;

  PetscFunctionBegin;
  snes->nwork = 4;
  ierr = VecDuplicateVecs(snes->vec_sol,snes->nwork,&snes->work );CHKERRQ(ierr);
  PLogObjectParents(snes,snes->nwork,snes->work);
  snes->vec_sol_update_always = snes->work[3];
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "SNESDestroy_EQ_TR"
static int SNESDestroy_EQ_TR(SNES snes )
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
#define __FUNC__ "SNESSetFromOptions_EQ_TR"
static int SNESSetFromOptions_EQ_TR(SNES snes)
{
  SNES_EQ_TR *ctx = (SNES_EQ_TR *)snes->data;
  double     tmp;
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_tr_mu",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {ctx->mu = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_tr_eta",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {ctx->eta = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_tr_sigma",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {ctx->sigma = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_tr_delta0",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {ctx->delta0 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_tr_delta1",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {ctx->delta1 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_tr_delta2",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {ctx->delta2 = tmp;}
  ierr = OptionsGetDouble(snes->prefix,"-snes_eq_tr_delta3",&tmp, &flg);CHKERRQ(ierr);
  if (flg) {ctx->delta3 = tmp;}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESPrintHelp_EQ_TR"
static int SNESPrintHelp_EQ_TR(SNES snes,char *p)
{
  SNES_EQ_TR *ctx = (SNES_EQ_TR *)snes->data;
  int        ierr;
  MPI_Comm   comm = snes->comm;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(comm," method SNESEQTR (tr) for systems of nonlinear equations:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"   %ssnes_eq_tr_mu <mu> (default %g)\n",p,ctx->mu);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"   %ssnes_eq_tr_eta <eta> (default %g)\n",p,ctx->eta);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"   %ssnes_eq_tr_sigma <sigma> (default %g)\n",p,ctx->sigma);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"   %ssnes_eq_tr_delta0 <delta0> (default %g)\n",p,ctx->delta0);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"   %ssnes_eq_tr_delta1 <delta1> (default %g)\n",p,ctx->delta1);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"   %ssnes_eq_tr_delta2 <delta2> (default %g)\n",p,ctx->delta2);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"   %ssnes_eq_tr_delta3 <delta3> (default %g)\n",p,ctx->delta3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESView_EQ_TR"
static int SNESView_EQ_TR(SNES snes,Viewer viewer)
{
  SNES_EQ_TR *tr = (SNES_EQ_TR *)snes->data;
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"  mu=%g, eta=%g, sigma=%g\n",tr->mu,tr->eta,tr->sigma);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"  delta0=%g, delta1=%g, delta2=%g, delta3=%g\n",tr->delta0,tr->delta1,tr->delta2,tr->delta3);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for SNES EQ TR",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SNESConverged_EQ_TR"
/*@
   SNESConverged_EQ_TR - Monitors the convergence of the trust region
   method SNESEQTR for solving systems of nonlinear equations (default).

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  pnorm - 2-norm of current step 
.  fnorm - 2-norm of function
-  dummy - unused context

   Output Parameter:
.   reason - one of
$  SNES_CONVERGED_FNORM_ABS       - ( fnorm < atol ),
$  SNES_CONVERGED_PNORM_RELATIVE  - ( pnorm < xtol*xnorm ),
$  SNES_CONVERGED_FNORM_RELATIVE  - ( fnorm < rtol*fnorm0 ),
$  SNES_DIVERGED_FUNCTION_COUNT   - ( nfct > maxf ),
$  SNES_DIVERGED_FNORM_NAN        - ( fnorm == NaN ),
$  SNES_CONVERGED_TR_DELTA        - ( delta < xnorm*deltatol ),
$  SNES_CONVERGED_ITERATING       - ( otherwise )

   where
+    delta    - trust region paramenter
.    deltatol - trust region size tolerance,
                set with SNESSetTrustRegionTolerance()
.    maxf - maximum number of function evaluations,
            set with SNESSetTolerances()
.    nfct - number of function evaluations,
.    atol - absolute function norm tolerance,
            set with SNESSetTolerances()
-    xtol - relative function norm tolerance,
            set with SNESSetTolerances()

   Level: intermediate

.keywords: SNES, nonlinear, default, converged, convergence

.seealso: SNESSetConvergenceTest(), SNESEisenstatWalkerConverged()
@*/
int SNESConverged_EQ_TR(SNES snes,double xnorm,double pnorm,double fnorm,SNESConvergedReason *reason,void *dummy)
{
  SNES_EQ_TR *neP = (SNES_EQ_TR *)snes->data;
  int        ierr;

  PetscFunctionBegin;
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"For SNES_NONLINEAR_EQUATIONS only");
  }

  if (fnorm != fnorm) {
    PLogInfo(snes,"SNESConverged_EQ_TR:Failed to converged, function norm is NaN\n");
    *reason = SNES_DIVERGED_FNORM_NAN;
  } else if (neP->delta < xnorm * snes->deltatol) {
    PLogInfo(snes,"SNESConverged_EQ_TR: Converged due to trust region param %g<%g*%g\n",neP->delta,xnorm,snes->deltatol);
    *reason = SNES_CONVERGED_TR_DELTA;
  } else if (neP->itflag) {
    ierr = SNESConverged_EQ_LS(snes,xnorm,pnorm,fnorm,reason,dummy);CHKERRQ(ierr);
  } else if (snes->nfuncs > snes->max_funcs) {
    PLogInfo(snes,"SNESConverged_EQ_TR: Exceeded maximum number of function evaluations: %d > %d\n",snes->nfuncs, snes->max_funcs );
    *reason = SNES_DIVERGED_FUNCTION_COUNT;
  } else {
    *reason = SNES_CONVERGED_ITERATING;
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "SNESCreate_EQ_TR"
int SNESCreate_EQ_TR(SNES snes )
{
  SNES_EQ_TR *neP;

  PetscFunctionBegin;
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"For SNES_NONLINEAR_EQUATIONS only");
  }
  snes->setup		= SNESSetUp_EQ_TR;
  snes->solve		= SNESSolve_EQ_TR;
  snes->destroy		= SNESDestroy_EQ_TR;
  snes->converged	= SNESConverged_EQ_TR;
  snes->printhelp       = SNESPrintHelp_EQ_TR;
  snes->setfromoptions  = SNESSetFromOptions_EQ_TR;
  snes->view            = SNESView_EQ_TR;
  snes->nwork           = 0;
  
  neP			= PetscNew(SNES_EQ_TR);CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_EQ_TR));
  snes->data	        = (void *) neP;
  neP->mu		= 0.25;
  neP->eta		= 0.75;
  neP->delta		= 0.0;
  neP->delta0		= 0.2;
  neP->delta1		= 0.3;
  neP->delta2		= 0.75;
  neP->delta3		= 2.0;
  neP->sigma		= 0.0001;
  neP->itflag		= 0;
  neP->rnorm0		= 0;
  neP->ttol		= 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

