#ifndef lint
static char vcid[] = "$Id: snesut.c,v 1.7 1996/01/23 00:19:51 bsmith Exp bsmith $";
#endif

#include <math.h>
#include "snesimpl.h"       /*I   "snes.h"   I*/

/*@C
   SNESDefaultMonitor - Default SNES monitoring routine.

   Input Parameters:
.  snes - the SNES context
.  its - iteration number
.  fgnorm - 2-norm of residual (or gradient)
.  dummy - unused context

   Notes:
   For SNES_NONLINEAR_EQUATIONS methods the routine prints the 
   residual norm at each iteration.

   For SNES_UNCONSTRAINED_MINIMIZATION methods the routine prints the
   function value and gradient norm at each iteration.

.keywords: SNES, nonlinear, default, monitor, norm

.seealso: SNESSetMonitor()
@*/
int SNESDefaultMonitor(SNES snes,int its,double fgnorm,void *dummy)
{
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS)
    MPIU_printf(snes->comm, "iter = %d, SNES Function norm %g \n",its,fgnorm);
  else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION)
    MPIU_printf(snes->comm,
     "iter = %d, Function value %g, Gradient norm %g \n",its,snes->fc,fgnorm);
  else SETERRQ(1,"SNESDefaultMonitor:Unknown method class");
  return 0;
}
/* ---------------------------------------------------------------- */
int SNESDefaultSMonitor(SNES snes,int its, double fgnorm,void *dummy)
{
  if (snes->method_class == SNES_NONLINEAR_EQUATIONS) {
    if (fgnorm > 1.e-9 || fgnorm == 0.0) {
      MPIU_printf(snes->comm, "iter = %d, Function norm %g \n",its,fgnorm);
    }
    else if (fgnorm > 1.e-11){
      MPIU_printf(snes->comm, "iter = %d, Function norm %5.3e \n",its,fgnorm);
    }
    else {
      MPIU_printf(snes->comm, "iter = %d, Function norm < 1.e-11\n",its);
    }
  } else if (snes->method_class == SNES_UNCONSTRAINED_MINIMIZATION) {
    if (fgnorm > 1.e-9 || fgnorm == 0.0) {
      MPIU_printf(snes->comm,
       "iter = %d, Function value %g, Gradient norm %g \n",
       its,snes->fc,fgnorm);
    }
    else if (fgnorm > 1.e-11) {
      MPIU_printf(snes->comm,
        "iter = %d, Function value %g, Gradient norm %5.3e \n",
        its,snes->fc,fgnorm);
    }
    else {
      MPIU_printf(snes->comm,
        "iter = %d, Function value %g, Gradient norm < 1.e-11\n",
        its,snes->fc);
    }
  } else SETERRQ(1,"SNESDefaultSMonitor:Unknown method class");
  return 0;
}
/* ---------------------------------------------------------------- */
/*@C 
   SNESDefaultConverged - Default test for monitoring the convergence 
   of the solvers for systems of nonlinear equations.

   Input Parameters:
.  snes - the SNES context
.  xnorm - 2-norm of current iterate
.  pnorm - 2-norm of current step 
.  fnorm - 2-norm of function
.  dummy - unused context

   Returns:
$  2  if  ( fnorm < atol ),
$  3  if  ( pnorm < xtol*xnorm ),
$  4  if  ( fnorm < rtol*fnorm0 ),
$ -2  if  ( nfct > maxf ),
$  0  otherwise,

   where
$    maxf - maximum number of function evaluations,
$           set with SNESSetMaxFunctionEvaluations()
$    nfct - number of function evaluations,
$    atol - absolute function norm tolerance,
$           set with SNESSetAbsoluteTolerance()
$    xtol - relative function norm tolerance,
$           set with SNESSetRelativeTolerance()

.keywords: SNES, nonlinear, default, converged, convergence

.seealso: SNESSetConvergenceTest(), SNESEisenstatWalkerConverged()
@*/
int SNESDefaultConverged(SNES snes,double xnorm,double pnorm,double fnorm,void *dummy)
{
  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESDefaultConverged:For SNES_NONLINEAR_EQUATIONS only");
  /* Note:  Reserve return code 1, -1 for compatibility with 
  SNESTrustRegionDefaultConverged */
  if (snes->iter == 1) { /* first iteration so set ttol */
    snes->ttol = fnorm*snes->rtol;
  }
  else {
    if (fnorm <= snes->ttol) {
      PLogInfo((PetscObject)snes,
      "SNES:Converged due to function norm %g < %g (relative tolerance)\n",fnorm,snes->ttol);
      return 4;
    }
  }

  if (fnorm < snes->atol) {
    PLogInfo((PetscObject)snes,
      "SNES: Converged due to function norm %g < %g\n",fnorm,snes->atol);
    return 2;
  }
  if (pnorm < snes->xtol*(xnorm)) {
    PLogInfo((PetscObject)snes,
      "SNES: Converged due to small update length: %g < %g * %g\n",
       pnorm,snes->xtol,xnorm);
    return 3;
  }
  if (snes->nfuncs > snes->max_funcs) {
    PLogInfo((PetscObject)snes,
      "SNES: Exceeded maximum number of function evaluations: %d > %d\n",
      snes->nfuncs, snes->max_funcs );
    return -2;
  }  
  return 0;
}
/* ------------------------------------------------------------ */
/*@
   SNES_KSP_SetConvergenceTestEW - Sets alternative convergence test for
   for the linear solvers within an inexact Newton method.  

   Input Parameter:
.  snes - SNES context

   Notes:
   Currently, the default is to use a constant relative tolerance for 
   the inner linear solvers.  Alternatively, one can use the 
   Eisenstat-Walker method, where the relative convergence tolerance 
   is reset at each Newton iteration according progress of the nonlinear 
   solver. 

   Reference:
   S. C. Eisenstat and H. F. Walker, "Choosing the forcing terms in an 
   inexact Newton method", Utah State University Math. Stat. Dept. Res. 
   Report 6/94/75, June, 1994, to appear in SIAM J. Sci. Comput. 

.keywords: SNES, KSP, Eisenstat, Walker, convergence, test, inexact, Newton
@*/
int SNES_KSP_SetConvergenceTestEW(SNES snes)
{
  snes->ksp_ewconv = 1;
  return 0;
}

/*@
   SNES_KSP_SetParametersEW - Sets parameters for Eisenstat-Walker
   convergence criteria for the linear solvers within an inexact
   Newton method.

   Input Parameters:
.  snes - SNES context
.  version - version 1 or 2 (default is 2)
.  rtol_0 - initial relative tolerance 
$    (0 <= rtol_0 < 1)
.  rtol_max - maximum relative tolerance
$    (0 <= rtol_max < 1)
.  alpha - power for version 2 rtol computation
$    (1 < alpha <= 2)
.  alpha2 - power for safeguard
.  gamma2 - multiplicative factor for version 2 rtol computation
$    (0 <= gamma2 <= 1)
.  threshold - threshold for imposing safeguard
$    (0 < threshold < 1)

   Note:
   Use PETSC_DEFAULT to retain the default for any of the parameters.

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
  if (!kctx) SETERRQ(1,"SNES_KSP_SetParametersEW:No context");
  if (version != PETSC_DEFAULT)   kctx->version = version;
  if (rtol_0 != PETSC_DEFAULT)    kctx->rtol_0 = rtol_0;
  if (rtol_max != PETSC_DEFAULT)  kctx->rtol_max = rtol_max;
  if (gamma2 != PETSC_DEFAULT)    kctx->gamma = gamma2;
  if (alpha != PETSC_DEFAULT)     kctx->alpha = alpha;
  if (alpha2 != PETSC_DEFAULT)    kctx->alpha2 = alpha2;
  if (threshold != PETSC_DEFAULT) kctx->threshold = threshold;
  if (kctx->rtol_0 < 0.0 || kctx->rtol_0 >= 1.0) SETERRQ(1,
    "SNES_KSP_SetParametersEW: 0.0 <= rtol_0 < 1.0\n");
  if (kctx->rtol_max < 0.0 || kctx->rtol_max >= 1.0) SETERRQ(1,
    "SNES_KSP_SetParametersEW: 0.0 <= rtol_max < 1.0\n");
  if (kctx->threshold <= 0.0 || kctx->threshold >= 1.0) SETERRQ(1,
    "SNES_KSP_SetParametersEW: 0.0 < threshold < 1.0\n");
  if (kctx->gamma < 0.0 || kctx->gamma > 1.0) SETERRQ(1,
    "SNES_KSP_SetParametersEW: 0.0 <= alpha <= 1.0\n");
  if (kctx->alpha <= 1.0 || kctx->alpha > 2.0) SETERRQ(1,
    "SNES_KSP_SetParametersEW: 1.0 < alpha <= 2.0\n");
  if (kctx->version != 1 && kctx->version !=2) SETERRQ(1,
     "SNES_KSP_SetParametersEW: Only versions 1 and 2 are supported");
  return 0;
}

int SNES_KSP_EW_ComputeRelativeTolerance_Private(SNES snes,KSP ksp)
{
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;
  double rtol, stol;
  int    ierr;
  if (!kctx) 
    SETERRQ(1,"SNES_KSP_EW_ComputeRelativeTolerance_Private:No context");
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
    } else SETERRQ(1,
     "SNES_KSP_EW_Converged_Private:Only versions 1 or 2 are supported");
  }
  rtol = PetscMin(rtol,kctx->rtol_max);
  kctx->rtol_last = rtol;
  PLogInfo((PetscObject)snes,
    "SNES: iter %d, Eisenstat-Walker (version %d) KSP rtol = %g\n",
     snes->iter,kctx->version,rtol);
  ierr = KSPSetTolerances(ksp,rtol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
  CHKERRQ(ierr);
  kctx->norm_last = snes->norm;
  return 0;
}

int SNES_KSP_EW_Converged_Private(KSP ksp,int n,double rnorm,void *ctx)
{
  SNES                snes = (SNES)ctx;
  SNES_KSP_EW_ConvCtx *kctx = (SNES_KSP_EW_ConvCtx*)snes->kspconvctx;
  int                 convinfo;

  if (!kctx) SETERRQ(1,"SNES_KSP_EW_Converged_Private:No convergence context");
  if (n == 0) SNES_KSP_EW_ComputeRelativeTolerance_Private(snes,ksp);
  convinfo = KSPDefaultConverged(ksp,n,rnorm,ctx);
  kctx->lresid_last = rnorm;
  if (convinfo) 
    PLogInfo((PetscObject)snes,"SNES: KSP iterations=%d, rnorm=%g\n",n,rnorm);
  return convinfo;
}


