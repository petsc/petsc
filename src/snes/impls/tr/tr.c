#ifndef lint
static char vcid[] = "$Id: tr.c,v 1.1 1995/04/12 20:36:40 bsmith Exp bsmith $";
#endif

#include <math.h>
#include "tr.h"

/*
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
*/
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
static int SNESSolve_TR(SNES snes, int *its )
{
  SNES_TR  *neP = (SNES_TR *) snes->data;
  Vec      X, F, Y, G, TMP;
  int      maxits, i, iters, history_len, nlconv,ierr,lits;
  double   rho, fnorm, gnorm, gpnorm, xnorm, delta,norm;
  double   *history, ynorm;
  Scalar   one = 1.0;

  nlconv	= 0;			/* convergence monitor */
  history	= snes->conv_hist;	/* convergence history */
  history_len	= snes->conv_hist_len;	/* convergence history length */
  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;		/* solution vector */
  F		= snes->vec_res;		/* residual vector */
  Y		= snes->work[0];		/* work vectors */
  G		= snes->work[1];

  ierr = SNESComputeInitialGuess(snes,X); CHKERR(ierr);  /* X <- X_0 */
  VecNorm(X, &xnorm ); 		/* xnorm = || X || */
   
  ierr = SNESComputeResidual(snes,X,F); CHKERR(ierr); /* (+/-) F(X) */
  VecNorm(F, &fnorm );		/* fnorm <- || F || */ 
  snes->norm = fnorm;
  if (history && history_len > 0) history[0] = fnorm;
  delta = neP->delta0*fnorm;         
  neP->delta = delta;
  if (snes->Monitor)(*snes->Monitor)(snes,0,X,F,fnorm,snes->monP);
 
   for ( i=0; i<maxits; i++ ) {
     snes->iter = i+1;

     (*snes->ComputeJacobian)(X,&snes->jacobian,snes->jacP);
     while(1) {
       ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian,0);
       ierr = SLESSolve(snes->sles,F,Y,&lits); CHKERR(ierr);
       /* Scale Y if need be and predict new value of F norm */
       VecNorm( Y, &norm );
       if (norm > delta) {
         norm = delta/norm;
         gpnorm = (1.0 - norm)*fnorm;
         VecScale( &norm, Y );
         PLogInfo((PetscObject)snes, "Scaling direction by %g \n",norm );
         ynorm = delta;
       } else {
         gpnorm = 0.0;
         PLogInfo((PetscObject)snes,"Direction is in Trust Region \n" );
         ynorm = norm;
       }
       VecAXPY(&one, X, Y );	/* Y <- X + Y */
       ierr = SNESComputeResidual(snes,Y,G); CHKERR(ierr); /* (+/-) F(X) */
       VecNorm( G, &gnorm );	/* gnorm <- || g || */ 
       if (fnorm == gpnorm) rho = 0.0;
       else rho = (fnorm*fnorm - gnorm*gnorm)/(fnorm*fnorm - gpnorm*gpnorm); 

       /* Update size of trust region */
       if      (rho < neP->mu)  delta *= neP->delta1;
       else if (rho < neP->eta) delta *= neP->delta2;
       else                     delta *= neP->delta3;

       PLogInfo((PetscObject)snes,"%d:  f_norm=%g, g_norm=%g, ynorm=%g\n",
                                             i, fnorm, gnorm, ynorm );
       PLogInfo((PetscObject)snes,"gpred=%g, rho=%g, delta=%g,iters=%d\n", 
                                               gpnorm, rho, delta, lits);

       neP->delta = delta;
       if (rho > neP->sigma) break;
       neP->itflag = 0;
       if ((*snes->Converged)( snes, xnorm, ynorm, fnorm,snes->cnvP)) {
         /* We're not progressing, so return with the current iterate */
         if (X != snes->vec_sol) VecCopy( X, snes->vec_sol );
         return i;
       }
     }
     fnorm = gnorm;
     snes->norm = fnorm;
     if (history && history_len > i+1) history[i+1] = fnorm;
     TMP = F; F = G; G = TMP;
     TMP = X; X = Y; Y = TMP;
     VecNorm(X, &xnorm );		/* xnorm = || X || */
     if (snes->Monitor) (*snes->Monitor)(snes,0,X,F,fnorm,snes->monP);

     /* Test for convergence */
     neP->itflag = 1;
     if ((*snes->Converged)( snes, xnorm, ynorm, fnorm,snes->cnvP )) {
       /* Verify solution is in corect location */
       if (X != snes->vec_sol) VecCopy(X, snes->vec_sol );
       break;
     } 
   }
   if (i == maxits) *its = i-1; else *its = i;
   return 0;
}
/* -------------------------------------------------------------*/

/*------------------------------------------------------------*/
static int SNESSetUp_TR( SNES snes )
{
  int ierr;
  snes->nwork = 2;
  ierr = VecGetVecs(snes->vec_sol,snes->nwork,&snes->work ); CHKERR(ierr);
  return 0;
}
/*------------------------------------------------------------*/
static int SNESDestroy_TR(PetscObject obj )
{
  SNES snes = (SNES) obj;
  VecFreeVecs(snes->work, snes->nwork );
  return 0;
}
/*------------------------------------------------------------*/
/*@
   SNESTRDefaultConverged - Default test for monitoring the 
   convergence of the method NLENewtonTR1Solve.

   Input Parameters:
.  snes - nonlinear context obtained from NLCreate()
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
int SNESTRDefaultConverged(SNES snes, double xnorm, double pnorm, 
                           double fnorm, void *ctx )
{
  SNES_TR *neP = (SNES_TR *)snes->data;
  double  epsmch = 1.0e-14;   /* This must be fixed */

  if (neP->delta < xnorm * neP->deltatol) 	return  1;
  if (neP->itflag) {
          return SNESDefaultConverged( snes, xnorm, pnorm, fnorm,ctx );
      }
  if (neP->delta < xnorm * epsmch)		return -1;
  return 0;
}

static int SNESSetFromOptions_TR(SNES snes)
{
  SNES_TR *ctx = (SNES_TR *)snes->data;
  double  tmp;

  if (OptionsGetDouble(0,snes->prefix,"-mu",&tmp)) {ctx->mu = tmp;}
  if (OptionsGetDouble(0,snes->prefix,"-eta",&tmp)) {ctx->eta = tmp;}
  if (OptionsGetDouble(0,snes->prefix,"-sigma",&tmp)) {ctx->sigma = tmp;}
  if (OptionsGetDouble(0,snes->prefix,"-delta0",&tmp)) {ctx->delta0 = tmp;}
  if (OptionsGetDouble(0,snes->prefix,"-delta1",&tmp)) {ctx->delta1 = tmp;}
  if (OptionsGetDouble(0,snes->prefix,"-delta2",&tmp)) {ctx->delta2 = tmp;}
  if (OptionsGetDouble(0,snes->prefix,"-delta3",&tmp)) {ctx->delta3 = tmp;}
  return 0;
}

static int SNESPrintHelp_TR(SNES snes)
{
  SNES_TR *ctx = (SNES_TR *)snes->data;
  char    *prefix = "-";
  if (snes->prefix) prefix = snes->prefix;
  fprintf(stderr,"%smu mu (default %g)\n",prefix,ctx->mu);
  fprintf(stderr,"%seta eta (default %g)\n",prefix,ctx->eta);
  fprintf(stderr,"%ssigma sigma (default %g)\n",prefix,ctx->sigma);
  fprintf(stderr,"%sdelta0 delta0 (default %g)\n",prefix,ctx->delta0);
  fprintf(stderr,"%sdelta1 delta1 (default %g)\n",prefix,ctx->delta1);
  fprintf(stderr,"%sdelta2 delta2 (default %g)\n",prefix,ctx->delta2);
  fprintf(stderr,"%sdelta3 delta3 (default %g)\n",prefix,ctx->delta3);
  return 0;
}

int SNESCreate_TR(SNES snes )
{
  SNES_TR *neP;

  snes->type 		= SNES_NTR;
  snes->Setup		= SNESSetUp_TR;
  snes->Solver		= SNESSolve_TR;
  snes->destroy		= SNESDestroy_TR;
  snes->Monitor  	= 0;
  snes->Converged	= SNESTRDefaultConverged;
  snes->PrintHelp       = SNESPrintHelp_TR;
  snes->SetFromOptions  = SNESSetFromOptions_TR;

  neP			= NEW(SNES_TR); CHKPTR(neP);
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
}
