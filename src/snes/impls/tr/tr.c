#ifndef lint
static char vcid[] = "$Id: newtr1.c,v 1.8 1995/02/22 00:52:41 curfman Exp $";
#endif

#include <math.h>
#include "nonlin/nlall.h"
#include "nonlin/snes/nlepriv.h"

/*D
    NLE_NTR1 - Implements Newton's Method with a trust region 
    approach for solving systems of nonlinear equations. 

    Input parameters:
.   nlP - nonlinear context obtained from NLCreate()

    Returns:
    Number of global iterations until termination.  The precise type of 
    termination can be examined by calling NLGetTerminationType() after 
    NLSolve().
    
    Calling sequence:
$   nlP = NLCreate(NLE_NTR1,0)
$   NLCreateDVectors()
$   NLSet***()
$   NLSetUp()
$   NLSolve()
$   NLDestroy()

    Notes:
    See NLCreate() and NLSetUp() for information on the definition and
    initialization of the nonlinear solver context.  

    The basic algorithm is taken from "The Minpack Project", by More', 
    Sorensen, Garbow, Hillstrom, pages 88-111 of "Sources and Development 
    of Mathematical Software", Wayne Cowell, editor.  See the examples 
    in nonlin/examples.
D*/
/*
   This is intended as a model implementation, since it does not 
   necessarily have many of the bells and whistles of other 
   implementations.  

   The code is DATA-STRUCTURE NEUTRAL and can be called RECURSIVELY.  
   The following context variable is used:
     NLCtx *nlP - The nonlinear solver context, which is created by 
                  calling NLCreate(NLE_NTR1).

   The step_compute routine must return two values: 
     1) ynorm - the norm of the step 
     2) gpnorm - the predicted value for the residual norm at the new 
        point, assuming a local linearization.  The value is 0 if the 
        step lies within the trust region and is > 0 otherwise.
*/
int NLENewtonTR1Solve( nlP )
NLCtx *nlP;
{
   NLENewtonTR1Ctx *neP = (NLENewtonTR1Ctx *) nlP->MethodPrivate;
   void            *X, *F, *Y, *G, *TMP;
   int             maxits, i, iters, history_len, nlconv;
   double          rho, fnorm, gnorm, gpnorm, xnorm, delta;
   double          *history, ynorm;
   FILE            *fp = nlP->fp;
   NLMonCore       *mc = &nlP->mon.core;
   VECntx          *vc = NLGetVectorCtx(nlP);

   CHKCOOKIEN(nlP,NL_COOKIE);
   nlconv	= 0;			/* convergence monitor */
   history	= nlP->conv_hist;	/* convergence history */
   history_len	= nlP->conv_hist_len;	/* convergence history length */
   maxits	= nlP->max_its;		/* maximum number of iterations */
   X		= nlP->vec_sol;		/* solution vector */
   F		= nlP->vec_rg;		/* residual vector */
   Y		= nlP->work[0];		/* work vectors */
   G		= nlP->work[1];

   INITIAL_GUESS( X );			/* X <- X_0 */
   VNORM( vc, X, &xnorm ); 		/* xnorm = || X || */
   
   RESIDUAL( X, F );			/* Evaluate (+/-) F(X) */
   VNORM( vc, F, &fnorm );		/* fnorm <- || F || */ 
   nlP->norm = fnorm;
   if (history && history_len > 0) history[0] = fnorm;
   delta = neP->delta0*fnorm;         
   neP->delta = delta;
   mc->nvectors += 4; mc->nscalars += 3;
   MONITOR( X, F, &fnorm );		/* Monitor progress */
 
   for ( i=0; i<maxits; i++ ) {
       nlP->iter = i+1;

       STEP_SETUP( X );			/* Step set-up phase */
       while(1) {
           iters = STEP_COMPUTE( X, F, Y, &fnorm, &delta, 
		   &(nlP->trunctol), &gpnorm, &ynorm, (void *)0 );
		   CHKERRV(1,-(NL));	/* Step compute phase */
           VAXPY( vc, 1.0, X, Y );	/* Y <- X + Y */
           RESIDUAL( Y, G );		/* Evaluate (+/-) G(Y) */
           VNORM( vc, G, &gnorm );	/* gnorm <- || g || */ 
           if (fnorm == gpnorm) rho = 0.0;
           else rho = (fnorm*fnorm - gnorm*gnorm)/
                      (fnorm*fnorm - gpnorm*gpnorm); 

           /* Update size of trust region */
           if      (rho < neP->mu)  delta *= neP->delta1;
           else if (rho < neP->eta) delta *= neP->delta2;
           else                     delta *= neP->delta3;

           if (fp) fprintf(fp,"%d:  f=%g, g=%g, ynorm=%g\n",
                   i, fnorm, gnorm, ynorm );
           if (fp) fprintf(fp,"     gpred=%g, rho=%g, delta=%g, iters=%d\n", 
                   gpnorm, rho, delta, iters);

           neP->delta = delta;
           mc->nvectors += 4; mc->nscalars += 8;
           if (rho > neP->sigma) break;
           neP->itflag = 0;
           if (CONVERGED( &xnorm, &ynorm, &fnorm )) {
              /* We're not progressing, so return with the current iterate */
              if (X != nlP->vec_sol) VCOPY( vc, X, nlP->vec_sol );
              return i;
              }
           nlP->mon.nunsuc++;
           }
       STEP_DESTROY();			/* Step destroy phase */
       fnorm = gnorm;
       nlP->norm = fnorm;
       if (history && history_len > i+1) history[i+1] = fnorm;
       TMP = F; F = G; G = TMP;
       TMP = X; X = Y; Y = TMP;
       VNORM( vc, X, &xnorm );		/* xnorm = || X || */
       mc->nvectors += 2;
       mc->nscalars++;
       MONITOR( X, F, &fnorm );		/* Monitor progress */

       /* Test for convergence */
       neP->itflag = 1;
       if (CONVERGED( &xnorm, &ynorm, &fnorm )) {
           /* Verify solution is in corect location */
           if (X != nlP->vec_sol) VCOPY( vc, X, nlP->vec_sol );
           break;
           } 
       }
   if (i == maxits) i--;
   return i+1;
}
/* -------------------------------------------------------------*/
void NLENewtonTR1Create( nlP )
NLCtx *nlP;
{
  NLENewtonTR1Ctx *neP;
 
  CHKCOOKIE(nlP,NL_COOKIE);
  nlP->method		= NLE_NTR1;
  nlP->method_type	= NLE;
  nlP->setup		= NLENewtonTR1SetUp;
  nlP->solver		= NLENewtonTR1Solve;
  nlP->destroy		= NLENewtonTR1Destroy;
  nlP->set_param	= NLENewtonTR1SetParameter;
  nlP->get_param	= NLENewtonTR1GetParameter;
  nlP->usr_monitor	= NLENewtonDefaultMonitor;
  nlP->converged	= NLENewtonTR1DefaultConverged;
  nlP->term_type	= NLENewtonTR1DefaultConvergedType;

  neP			= NEW(NLENewtonTR1Ctx); CHKPTR(neP);
  nlP->MethodPrivate	= (void *) neP;
  neP->mu		= 0.25;
  neP->eta		= 0.75;
  neP->delta		= 0.0;
  neP->delta0		= 0.2;
  neP->delta1		= 0.3;
  neP->delta2		= 0.75;
  neP->delta3		= 2.0;
  neP->sigma		= 0.0001;
  neP->itflag		= 0;
}
/*------------------------------------------------------------*/
/*ARGSUSED*/
void  NLENewtonTR1SetUp( nlP )
NLCtx *nlP;
{
  CHKCOOKIE(nlP,NL_COOKIE);
  nlP->nwork = 2;
  nlP->work = VGETVECS( nlP->vc, nlP->nwork );	CHKPTR(nlP->work);
  NLiBasicSetUp( nlP, "NLENewtonTR1SetUp" );	CHKERR(1);
}
/*------------------------------------------------------------*/
/*ARGSUSED*/
void NLENewtonTR1Destroy( nlP )
NLCtx *nlP;
{
  CHKCOOKIE(nlP,NL_COOKIE);
  VFREEVECS( nlP->vc, nlP->work, nlP->nwork );
  NLiBasicDestroy( nlP );	CHKERR(1);
}
/*------------------------------------------------------------*/
/*ARGSUSED*/
/*@
   NLENewtonTR1DefaultConverged - Default test for monitoring the 
   convergence of the method NLENewtonTR1Solve.

   Input Parameters:
.  nlP - nonlinear context obtained from NLCreate()
.  xnorm - 2-norm of current iterate
.  pnorm - 2-norm of current step 
.  fnorm - 2-norm of residual

   Returns:
$  1  if  ( delta < xnorm*deltatol ),
$  2  if  ( fnorm < atol ),
$  3  if  ( pnorm < xtol*xnorm ),
$ -2  if  ( nres > max_res ),
$ -1  if  ( delta < xnorm*epsmch ),
$  0  otherwise,

   where
$    atol     - absolute residual norm tolerance,
$               set with NLSetAbsConvergenceTol()
$    delta    - trust region paramenter
$    deltatol - trust region size tolerance,
$               set with NLSetTrustRegionTol()
$    epsmch   - machine epsilon
$    max_res  - maximum number of residual evaluations,
$               set with NLSetMaxResidualEvaluations()
$    nres     - number of residual evaluations
$    xtol     - relative residual norm tolerance,
$               set with NLSetRelConvergenceTol()

   Note:  
   Call NLGetConvergenceType() after calling NLSolve() to obtain
   information about the type of termination which occurred for the
   nonlinear solver.
@*/
int NLENewtonTR1DefaultConverged( nlP, xnorm, pnorm, fnorm )
NLCtx  *nlP;
double *xnorm;
double *pnorm;
double *fnorm;
{
  NLENewtonTR1Ctx *neP = (NLENewtonTR1Ctx *)nlP->MethodPrivate;
  double          epsmch = 1.0e-14;   /* This must be fixed */

  CHKCOOKIEN(nlP,NL_COOKIE);
  if (nlP->method_type != NLE) {
      SETERRC(1,"Compatible with NLE component only");
      return 0;
      }
  nlP->conv_info = 0;
  if (neP->delta < *xnorm * nlP->deltatol) 	nlP->conv_info = 1;
  if (neP->itflag) {
      if (nlP->conv_info) return nlP->conv_info;
      nlP->conv_info = 
          NLENewtonDefaultConverged( nlP, xnorm, pnorm, fnorm );
      }
  if (neP->delta < *xnorm * epsmch)		nlP->conv_info = -1;
  return nlP->conv_info;
}
/*------------------------------------------------------------*/
/*
   NLENewtonTR1DefaultConvergedType - Returns information regarding 
   the type of termination which occurred within the 
   NLENewtonTR1DefaultConverged() test.

   Input Parameter:
.  nlP - nonlinear context obtained from NLCreate()

   Returns:
   Character string - message detailing the type of termination which
   occurred.
*/
char *NLENewtonTR1DefaultConvergedType( nlP )
NLCtx *nlP;
{
  char *mesg;

  CHKCOOKIEN(nlP,NL_COOKIE);
  if ((int)nlP->converged != (int)NLENewtonTR1DefaultConverged) {
     mesg = "Compatible only with NLENewtonTR1DefaultConverged.\n";
     SETERRC(1,"Compatible only with NLENewtonTR1DefaultConverged.");
  } else { 
  switch (nlP->conv_info) {
   case 1:
     mesg = "Trust region parameter satisfies the trust region tolerance.\n";
     break;
   case -1:
     mesg = "Machine epsilon tolerance exceeds the trust region parameter.\n";
     break;
   default:
     mesg = NLENewtonDefaultConvergedType( nlP );
  } }
  return mesg;
}
/*------------------------------------------------------------*/
/*
   NLENewtonTR1SetParameter - Sets a chosen parameter used by the
   NLE_NTR1 method to the desired value.

   Note:
   Possible parameters for the NLE_NTR1 method are
$       param = "mu" - used to compute trust region parameter
$       param = "eta" - used to compute trust region parameter
$       param = "sigma" - used to determine termination
$       param = "delta0" - used to initialize trust region parameter
$       param = "delta1" - used to compute trust region parameter
$       param = "delta2" - used to compute trust region parameter
$       param = "delta3" - used to compute trust region parameter
*/
void NLENewtonTR1SetParameter( nlP, param, value )
NLCtx  *nlP;
char   *param;
double *value;
{
  NLENewtonTR1Ctx *ctx = (NLENewtonTR1Ctx *)nlP->MethodPrivate;
 
  CHKCOOKIE(nlP,NL_COOKIE);
  if (nlP->method != NLE_NTR1) {
      SETERRC(1,"Compatible only with NLE_NTR1 method");
      return;
      }
  if (!strcmp(param,"mu"))		ctx->mu     = *value;
  else if (!strcmp(param,"eta"))	ctx->eta    = *value;
  else if (!strcmp(param,"sigma"))	ctx->sigma  = *value;
  else if (!strcmp(param,"delta0"))	ctx->delta0 = *value;
  else if (!strcmp(param,"delta1"))	ctx->delta1 = *value;
  else if (!strcmp(param,"delta2"))	ctx->delta2 = *value;
  else if (!strcmp(param,"delta3"))	ctx->delta3 = *value;
  else SETERRC(1,"Invalid parameter name for NLE_NTR1");
}
/*------------------------------------------------------------*/
/*
   NLENewtonTR1GetParameter - Returns the value of a chosen parameter
   used by the NLE_NTR1 method.

   Note:
   Possible parameters for the NLE_NTR1 method are
$       param = "mu" - used to compute trust region parameter
$       param = "eta" - used to compute trust region parameter
$       param = "sigma" - used to determine termination
$       param = "delta0" - used to initialize trust region parameter
$       param = "delta1" - used to compute trust region parameter
$       param = "delta2" - used to compute trust region parameter
$       param = "delta3" - used to compute trust region parameter
*/
double NLENewtonTR1GetParameter( nlP, param )
NLCtx *nlP;
char  *param;
{
  NLENewtonTR1Ctx *ctx = (NLENewtonTR1Ctx *)nlP->MethodPrivate;
  double          value = 0.0;

  CHKCOOKIEN(nlP,NL_COOKIE);
  if (nlP->method != NLE_NTR1) {
      SETERRC(1,"Compatible only with NLE_NTR1 method");
      return value;
      }
  if (!strcmp(param,"mu"))		value = ctx->mu;
  else if (!strcmp(param,"eta"))	value = ctx->eta;
  else if (!strcmp(param,"sigma"))	value = ctx->sigma;
  else if (!strcmp(param,"delta0"))	value = ctx->delta0;
  else if (!strcmp(param,"delta1"))	value = ctx->delta1;
  else if (!strcmp(param,"delta2"))	value = ctx->delta2;
  else if (!strcmp(param,"delta3"))	value = ctx->delta3;
  else SETERRC(1,"Invalid parameter name for NLE_NTR1");
  return value;
}
