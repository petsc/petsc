#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: snesut.c,v 1.46 1999/04/19 22:15:28 bsmith Exp balay $";
#endif

#include "src/snes/snesimpl.h"       /*I   "snes.h"   I*/

#undef __FUNC__  
#define __FUNC__ "SNESVecViewMonitor"
/*@C
   SNESVecViewMonitor - Monitors progress of the SNES solvers by calling 
   VecView() for the approximate solution at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: SNES, nonlinear, vector, monitor, view

.seealso: SNESSetMonitor(), SNESDefaultMonitor(), VecView()
@*/
int SNESVecViewMonitor(SNES snes,int its,double fgnorm,void *dummy)
{
  int    ierr;
  Vec    x;
  Viewer viewer = (Viewer) dummy;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes,&x);CHKERRQ(ierr);
  if (!viewer) {
    MPI_Comm comm;
    ierr   = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
    viewer = VIEWER_DRAW_(comm);
  }
  ierr = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESVecViewMonitorUpdate"
/*@C
   SNESVecViewMonitorUpdate - Monitors progress of the SNES solvers by calling 
   VecView() for the UPDATE to the solution at each iteration.

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy - either a viewer or PETSC_NULL

   Level: intermediate

.keywords: SNES, nonlinear, vector, monitor, view

.seealso: SNESSetMonitor(), SNESDefaultMonitor(), VecView()
@*/
int SNESVecViewMonitorUpdate(SNES snes,int its,double fgnorm,void *dummy)
{
  int    ierr;
  Vec    x;
  Viewer viewer = (Viewer) dummy;

  PetscFunctionBegin;
  ierr = SNESGetSolutionUpdate(snes,&x);CHKERRQ(ierr);
  if (!viewer) {
    MPI_Comm comm;
    ierr   = PetscObjectGetComm((PetscObject)snes,&comm);CHKERRQ(ierr);
    viewer = VIEWER_DRAW_(comm);
  }
  ierr = VecView(x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNESDefaultMonitor"
/*@C
   SNESDefaultMonitor - Monitoring progress of the SNES solvers (default).

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
-  dummy - unused context

   Notes:
   For SNES_NONLINEAR_EQUATIONS methods the routine prints the 
   residual norm at each iteration.

   For SNES_UNCONSTRAINED_MINIMIZATION methods the routine prints the
   function value and gradient norm at each iteration.

   Level: intermediate

.keywords: SNES, nonlinear, default, monitor, norm

.seealso: SNESSetMonitor(), SNESVecViewMonitor()
@*/
int SNESDefaultMonitor(SNES snes,int its,double fgnorm,void *dummy)
{
  int ierr;

  PetscFunctionBegin;
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    ierr = PetscPrintf(snes->comm, "iter = %d, SNES Function norm %g \n",its,fgnorm);CHKERRQ(ierr);
  } else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
    ierr = PetscPrintf(snes->comm,"iter = %d, SNES Function value %g, Gradient norm %g \n",its,snes->fc,fgnorm);CHKERRQ(ierr);
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown method class");
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SNESDefaultSMonitor"
/*
     Default (short) SNES Monitor, same as SNESDefaultMonitor() except
  it prints fewer digits of the residual as the residual gets smaller.
  This is because the later digits are meaningless and are often 
  different on different machines; by using this routine different 
  machines will usually generate the same output.
*/
int SNESDefaultSMonitor(SNES snes,int its, double fgnorm,void *dummy)
{
  int ierr;

  PetscFunctionBegin;
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    if (fgnorm > 1.e-9) {
      ierr = PetscPrintf(snes->comm, "iter = %d, SNES Function norm %g \n",its,fgnorm);CHKERRQ(ierr);
    } else if (fgnorm > 1.e-11){
      ierr = PetscPrintf(snes->comm, "iter = %d, SNES Function norm %5.3e \n",its,fgnorm);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(snes->comm, "iter = %d, SNES Function norm < 1.e-11\n",its);CHKERRQ(ierr);
    }
  } else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
    if (fgnorm > 1.e-9) {
      ierr = PetscPrintf(snes->comm,"iter = %d, SNES Function value %g, Gradient norm %g \n",its,snes->fc,fgnorm);CHKERRQ(ierr);
    } else if (fgnorm > 1.e-11) {
      ierr = PetscPrintf(snes->comm,"iter = %d, SNES Function value %g, Gradient norm %5.3e \n",its,snes->fc,fgnorm);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(snes->comm,"iter = %d, SNES Function value %g, Gradient norm < 1.e-11\n",its,snes->fc);CHKERRQ(ierr);
    }
  } else SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Unknown method class");
  PetscFunctionReturn(0);
}
/* ---------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SNESConverged_EQ_LS"
/*@C 
   SNESConverged_EQ_LS - Monitors the convergence of the solvers for
   systems of nonlinear equations (default).

   Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  pnorm - 2-norm of current step 
.  fnorm - 2-norm of function
-  dummy - unused context

   Returns:
+  2  - if  ( fnorm < atol ),
.  3  - if  ( pnorm < xtol*xnorm ),
.  4  - if  ( fnorm < rtol*fnorm0 ),
. -2  - if  ( nfct > maxf ),
-  0  - otherwise,

   where
+    maxf - maximum number of function evaluations,
            set with SNESSetTolerances()
.    nfct - number of function evaluations,
.    atol - absolute function norm tolerance,
            set with SNESSetTolerances()
-    rtol - relative function norm tolerance, set with SNESSetTolerances()

   Level: intermediate

.keywords: SNES, nonlinear, default, converged, convergence

.seealso: SNESSetConvergenceTest(), SNESEisenstatWalkerConverged()
@*/
int SNESConverged_EQ_LS(SNES snes,double xnorm,double pnorm,double fnorm,void *dummy)
{
  PetscFunctionBegin;
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) {
     SETERRQ(PETSC_ERR_ARG_WRONG,0,"For SNES_NONLINEAR_EQUATIONS only");
  }
  /* Note:  Reserve return code 1, -1 for compatibility with SNESConverged_EQ_TR */
  if (fnorm != fnorm) {
    PLogInfo(snes,"SNESConverged_EQ_LS:Failed to converged, function norm is NaN\n");
    PetscFunctionReturn(-3);
  }
  if (fnorm <= snes->ttol) {
    PLogInfo(snes,"SNESConverged_EQ_LS:Converged due to function norm %g < %g (relative tolerance)\n",fnorm,snes->ttol);
    PetscFunctionReturn(4);
  }

  if (fnorm < snes->atol) {
    PLogInfo(snes,"SNESConverged_EQ_LS:Converged due to function norm %g < %g\n",fnorm,snes->atol);
    PetscFunctionReturn(2);
  }
  if (pnorm < snes->xtol*(xnorm)) {
    PLogInfo(snes,"SNESConverged_EQ_LS:Converged due to small update length: %g < %g * %g\n",pnorm,snes->xtol,xnorm);
    PetscFunctionReturn(3);
  }
  if (snes->nfuncs > snes->max_funcs) {
    PLogInfo(snes,"SNESConverged_EQ_LS:Exceeded maximum number of function evaluations: %d > %d\n",snes->nfuncs, snes->max_funcs);
    PetscFunctionReturn(-2);
  }  
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */
#undef __FUNC__  
#define __FUNC__ "SNES_KSP_SetConvergenceTestEW"
/*@
   SNES_KSP_SetConvergenceTestEW - Sets alternative convergence test
   for the linear solvers within an inexact Newton method.  

   Collective on SNES

   Input Parameter:
.  snes - SNES context

   Notes:
   Currently, the default is to use a constant relative tolerance for 
   the inner linear solvers.  Alternatively, one can use the 
   Eisenstat-Walker method, where the relative convergence tolerance 
   is reset at each Newton iteration according progress of the nonlinear 
   solver. 

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", SISC 17 (1), pp.16-32, 1996.

.keywords: SNES, KSP, Eisenstat, Walker, convergence, test, inexact, Newton
@*/
int SNES_KSP_SetConvergenceTestEW(SNES snes)
{
  PetscFunctionBegin;
  snes->ksp_ewconv = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNES_KSP_SetParametersEW"
/*@
   SNES_KSP_SetParametersEW - Sets parameters for Eisenstat-Walker
   convergence criteria for the linear solvers within an inexact
   Newton method.

   Collective on SNES
 
   Input Parameters:
+    snes - SNES context
.    version - version 1 or 2 (default is 2)
.    rtol_0 - initial relative tolerance (0 <= rtol_0 < 1)
.    rtol_max - maximum relative tolerance (0 <= rtol_max < 1)
.    alpha - power for version 2 rtol computation (1 < alpha <= 2)
.    alpha2 - power for safeguard
.    gamma2 - multiplicative factor for version 2 rtol computation
              (0 <= gamma2 <= 1)
-    threshold - threshold for imposing safeguard (0 < threshold < 1)

   Note:
   Use PETSC_DEFAULT to retain the default for any of the parameters.

   Level: advanced

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", Utah State University Math. Stat. Dept. Res. 
   Report 6/94/75, June, 1994, to appear in SIAM J. Sci. Comput. 

.keywords: SNES, KSP, Eisenstat, Walker, set, parameters

.seealso: SNES_KSP_SetConvergenceTestEW()
@*/
int SNES_KSP_SetParametersEW(SNES snes,int version,double rtol_0,
                             double rtol_max,double gamma2,double alpha,
                             double alpha2,double threshold)
{
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;

  PetscFunctionBegin;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"No Eisenstat-Walker context existing");
  if (version != PETSC_DEFAULT)   kctx->version   = version;
  if (rtol_0 != PETSC_DEFAULT)    kctx->rtol_0    = rtol_0;
  if (rtol_max != PETSC_DEFAULT)  kctx->rtol_max  = rtol_max;
  if (gamma2 != PETSC_DEFAULT)    kctx->gamma     = gamma2;
  if (alpha != PETSC_DEFAULT)     kctx->alpha     = alpha;
  if (alpha2 != PETSC_DEFAULT)    kctx->alpha2    = alpha2;
  if (threshold != PETSC_DEFAULT) kctx->threshold = threshold;
  if (kctx->rtol_0 < 0.0 || kctx->rtol_0 >= 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"0.0 <= rtol_0 < 1.0: %g",kctx->rtol_0);
  }
  if (kctx->rtol_max < 0.0 || kctx->rtol_max >= 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"0.0 <= rtol_max < 1.0\n",kctx->rtol_max);
  }
  if (kctx->threshold <= 0.0 || kctx->threshold >= 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"0.0 < threshold < 1.0\n",kctx->threshold);
  }
  if (kctx->gamma < 0.0 || kctx->gamma > 1.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"0.0 <= alpha <= 1.0\n",kctx->gamma);
  }
  if (kctx->alpha <= 1.0 || kctx->alpha > 2.0) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"1.0 < alpha <= 2.0\n",kctx->alpha);
  }
  if (kctx->version != 1 && kctx->version !=2) {
    SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Only versions 1 and 2 are supported: %d",kctx->version);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNES_KSP_EW_ComputeRelativeTolerance_Private"
int SNES_KSP_EW_ComputeRelativeTolerance_Private(SNES snes,KSP ksp)
{
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;
  double              rtol = 0.0, stol;
  int                 ierr;

  PetscFunctionBegin;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"No Eisenstat-Walker context exists");
  if (snes->iter == 1) {
    rtol = kctx->rtol_0;
  } else {
    if (kctx->version == 1) {
      rtol = (snes->norm - kctx->lresid_last)/kctx->norm_last; 
      if (rtol < 0.0) rtol = -rtol;
      stol = pow(kctx->rtol_last,kctx->alpha2);
      if (stol > kctx->threshold) rtol = PetscMax(rtol,stol);
    } else if (kctx->version == 2) {
      rtol = kctx->gamma * pow(snes->norm/kctx->norm_last,kctx->alpha);
      stol = kctx->gamma * pow(kctx->rtol_last,kctx->alpha);
      if (stol > kctx->threshold) rtol = PetscMax(rtol,stol);
    } else SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"Only versions 1 or 2 are supported: %d",kctx->version);
  }
  rtol = PetscMin(rtol,kctx->rtol_max);
  kctx->rtol_last = rtol;
  PLogInfo(snes,"SNESConverged_EQ_LS: iter %d, Eisenstat-Walker (version %d) KSP rtol = %g\n",snes->iter,kctx->version,rtol);
  ierr = KSPSetTolerances(ksp,rtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  kctx->norm_last = snes->norm;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "SNES_KSP_EW_Converged_Private"
int SNES_KSP_EW_Converged_Private(KSP ksp,int n,double rnorm,void *ctx)
{
  SNES                snes = (SNES)ctx;
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;
  int                 convinfo;

  PetscFunctionBegin;
  if (!kctx) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"No Eisenstat-Walker context set");
  if (n == 0) SNES_KSP_EW_ComputeRelativeTolerance_Private(snes,ksp);
  convinfo = KSPDefaultConverged(ksp,n,rnorm,ctx);
  kctx->lresid_last = rnorm;
  if (convinfo) {
    PLogInfo(snes,"SNES_KSP_EW_Converged_Private: KSP iterations=%d, rnorm=%g\n",n,rnorm);
  }
  PetscFunctionReturn(convinfo);
}


