#ifndef lint
static char vcid[] = "$Id: rich.c,v 1.9 1995/03/06 03:49:35 bsmith Exp bsmith $";
#endif
/*          
            This implements Richardson Iteration.       
*/
#include <stdio.h>
#include <math.h>
#include "petsc.h"
#include "kspimpl.h"         /*I "ksp.h" I*/
#include "richctx.h"

int KSPSetUp_Richardson(KSP itP)
{
  int ierr;
  if (itP->method != KSPRICHARDSON) {
    SETERR(1,"Attempt to use Richardson Setup on wrong context"); 
  }
  /* check user parameters and functions */
  if (itP->right_pre) {
    SETERR(2,"Right-inverse preconditioning not supported for Richardson"); 
  }
  if ((ierr = KSPCheckDef( itP ))) return ierr;
  /* get work vectors from user code */
  return KSPiDefaultGetWork( itP,  2 );
}

/*@
    KSPRichardsonSetScale - Called after KSPCreate(KSPRICHARDSON), sets
    the "damping" factor; if this routine is not called, the  
    factor defaults to 1.0.

    Input Parameters:
.   itP - the iterative context
.   scale - the relaxation factor
@*/
int KSPRichardsonSetScale(KSP itP,double scale)
{
  KSP_Richardson *richardsonP;
  VALIDHEADER(itP,KSP_COOKIE);
  if (itP->method != KSPRICHARDSON) return 0;
  richardsonP = (KSP_Richardson *) itP->MethodPrivate;
  richardsonP->scale = scale;
  return 0;
}

int  KSPSolve_Richardson(KSP itP,int *its)
{
  int                i = 0,maxit,pres, brokeout = 0, hist_len, cerr;
  double             rnorm,*history;
  Scalar             scale, mone = -1.0;
  Vec                x,b,r,z;
  KSP_Richardson  *richardsonP;
  richardsonP = (KSP_Richardson *) itP->MethodPrivate;

  x       = itP->vec_sol;
  b       = itP->vec_rhs;
  r       = itP->work[0];
  maxit   = itP->max_it;

  /* if user has provided fast Richardson code use that */
  if (PCApplyRichardsonExists(itP->B)) {
    *its = maxit;
    return PCApplyRichardson(itP->B,b,x,r,maxit);
  }

  z       = itP->work[1];
  history = itP->residual_history;
  hist_len= itP->res_hist_size;
  scale   = richardsonP->scale;
  pres    = itP->use_pres;

  if (!itP->guess_zero) {                       /*   r <- b - A x     */
    MatMult(PCGetMat(itP->B),x,r);
    VecAYPX(&mone,b,r);
  }
  else VecCopy(b,r);

  for ( i=0; i<maxit; i++ ) {
     PCApply(itP->B,r,z);                       /*   z <- B r         */
     if (itP->calc_res) {
	if (!pres) VecNorm(r,&rnorm);         /*   rnorm <- r'*r    */
	else       VecNorm(z,&rnorm);         /*   rnorm <- z'*z    */
        if (history && hist_len > i) history[i] = rnorm;
        MONITOR(itP,rnorm,i);
        if (CONVERGED(itP,rnorm,i)) {brokeout = 1; break;}
     }
   
     VecAXPY(&scale,z,x);                     /*   x  <- x + scale z */
     MatMult(PCGetMat(itP->B),x,r);           /*   r  <- b - Ax      */
     VecAYPX(&mone,b,r);
  }
  if (itP->calc_res && !brokeout) {
    if (!pres) VecNorm(r,&rnorm);             /*   rnorm <- r'*r    */
    else {
      PCApply(itP->B,r,z);                    /*   z <- B r         */
      VecNorm(z,&rnorm);                      /*   rnorm <- z'*z    */
    }
    if (history && hist_len > i) history[i] = rnorm;
    MONITOR(itP,rnorm,i);
  }
  if (history) itP->res_act_size = (hist_len < i) ? hist_len : i;

  itP->namult += (i+1);
  itP->nbinv  += (i+1);
  itP->nvectors += 4*(i+1);

  *its = RCONV(itP,i+1);
  return 0;
}

int KSPCreate_Richardson(KSP itP)
{
  KSP_Richardson *richardsonP;
  richardsonP = NEW(KSP_Richardson); CHKPTR(richardsonP);
  itP->MethodPrivate = (void *) richardsonP;
  itP->method               = KSPRICHARDSON;
  richardsonP->scale        = 1.0;
  itP->setup      = KSPSetUp_Richardson;
  itP->solver     = KSPSolve_Richardson;
  itP->adjustwork = KSPiDefaultAdjustWork;
  itP->destroy    = KSPiDefaultDestroy;
  itP->calc_res   = 1;
  itP->converged            = KSPDefaultConverged;
  itP->BuildSolution        = KSPDefaultBuildSolution;
  itP->BuildResidual        = KSPDefaultBuildResidual;
  return 0;
}
