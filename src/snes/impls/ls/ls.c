#ifndef lint
static char vcid[] = "$Id: newls1.c,v 1.9 1995/02/22 00:52:02 curfman Exp $";
#endif

#include <math.h>
#include "nonlin/nlall.h"
#include "nonlin/snes/nlepriv.h"
#include "system/system.h"
#include "system/time/usec.h"

/*D
    NLE_NLS1 - Implements a line search variant of Newton's Method 
    for solving systems of nonlinear equations.  

    Input parameters:
.   nlP - nonlinear context obtained from NLCreate()

    Returns:
    Number of global iterations until termination.  The precise type of
    termination can be examined by calling NLGetTerminationType() after 
    NLSolve().

    Calling sequence:
$   nlP = NLCreate(NE_NLS1,0);
$   NLCreateDVectors()
$   NLSetXXX()
$   NLSetUp()
$   NLSolve()
$   NLDestroy()

    Notes:
    See NLCreate() and NLSetUp() for information on the definition and
    initialization of the nonlinear solver context.  

    This implements essentially a truncated Newton method with a
    line search.  By default a cubic backtracking line search 
    is employed, as described in the text "Numerical Methods for
    Unconstrained Optimization and Nonlinear Equations" by Dennis 
    and Schnabel.  See the examples in nonlin/snes/examples.
D*/
/*
   This is intended as a model implementation, since it does not 
   necessarily have many of the bells and whistles of other 
   implementations. 

   The code is DATA-STRUCTURE NEUTRAL and can be called RECURSIVELY.  
   The following context variable is used:
      NLCtx *nlP - The nonlinear solver context, which is created by
                   calling NLCreate(NLE_NLS1).
 */

int NLENewtonLS1Solve( nlP )
NLCtx *nlP;
{
  NLENewtonLS1Ctx *neP = (NLENewtonLS1Ctx *) nlP->MethodPrivate;
  int             maxits, i, iters, line, nlconv, history_len;
  double          fnorm, gnorm, gpnorm, xnorm, ynorm, *history;
  double          t_elp, t_cpu;
  void            *Y, *X, *F, *G, *W, *TMP;
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
  W		= nlP->work[2];

  INITIAL_GUESS( X );			/* X <- X_0 */
  VNORM( vc, X, &xnorm );		/* xnorm = || X || */
  RESIDUAL( X, F );			/* Evaluate (+/-) F(X) */
  VNORM( vc, F, &fnorm );		/* fnorm <- || F || */  
  nlP->norm = fnorm;
  if (history && history_len > 0) history[0] = fnorm;
  mc->nvectors += 4; mc->nscalars += 2;
  MONITOR( X, F, &fnorm );		/* Monitor progress */
        
  for ( i=0; i<maxits; i++ ) {
       nlP->iter = i+1;

       /* Solve J Y = -F, where J is Jacobian matrix */
       STEP_SETUP( X );			/* Step set-up phase */
       iters = STEP_COMPUTE( X, F, Y, &fnorm, &(neP->maxstep), 
	       &(nlP->trunctol), &gpnorm, &ynorm, (void *)0 ); 
	       CHKERRV(1,-(NL));	/* Step compute phase,
					   excluding line search */

       /* Line search should really be part of step compute phase */
       line = (*neP->line_search)( nlP, X, F, G, Y, W, fnorm, 
              &ynorm, &gnorm );		CHKERRV(1,-(NL));

       if (!line) nlP->mon.nunsuc++;
       if (fp) fprintf(fp,"%d:  fnorm=%g, gnorm=%g, ynorm=%g, iters=%d\n", 
               nlP->iter, fnorm, gnorm, ynorm, iters );
       TMP = F; F = G; G = TMP;
       TMP = X; X = Y; Y = TMP;
       fnorm = gnorm;

       STEP_DESTROY();			/* Step destroy phase */
       nlP->norm = fnorm;
       if (history && history_len > i+1) history[i+1] = fnorm;
       VNORM( vc, X, &xnorm );		/* xnorm = || X || */
       mc->nvectors += 2;
       mc->nscalars++;
       MONITOR( X, F, &fnorm );		/* Monitor progress */

       /* Test for convergence */
       if (CONVERGED( &xnorm, &ynorm, &fnorm )) {
           /* Verify that solution is in corect location */
           if (X != nlP->vec_sol) VCOPY( vc, X, nlP->vec_sol );
           break;
           }
       }
  if (i == maxits) i--;
  return i+1;
}
/* ------------------------------------------------------------ */
/*ARGSUSED*/
void NLENewtonLS1SetUp( nlP )
NLCtx *nlP;
{
  NLENewtonLS1Ctx *ctx = (NLENewtonLS1Ctx *)nlP->MethodPrivate;

  CHKCOOKIE(nlP,NL_COOKIE);
  nlP->nwork = 3;
  nlP->work = VGETVECS( nlP->vc, nlP->nwork );	CHKPTR(nlP->work);
  if (!ctx->line_search) {
      SETERRC(1,"NLENewtonLS1SetUp needs line search routine!\n");
      return;
      }
  NLiBasicSetUp( nlP, "NLENewtonLS1SetUp" );	CHKERR(1);
}
/* ------------------------------------------------------------ */
void NLENewtonLS1Create( nlP )
NLCtx *nlP;
{
  NLENewtonLS1Ctx *neP;

  CHKCOOKIE(nlP,NL_COOKIE);
  nlP->method		= NLE_NLS1;
  nlP->method_type	= NLE;
  nlP->setup		= NLENewtonLS1SetUp;
  nlP->solver		= NLENewtonLS1Solve;
  nlP->destroy		= NLENewtonLS1Destroy;
  nlP->set_param	= NLENewtonLS1SetParameter;
  nlP->get_param	= NLENewtonLS1GetParameter;
  nlP->usr_monitor	= NLENewtonDefaultMonitor;
  nlP->converged	= NLENewtonDefaultConverged;
  nlP->term_type	= NLENewtonDefaultConvergedType;

  neP			= NEW(NLENewtonLS1Ctx);   CHKPTR(neP);
  nlP->MethodPrivate	= (void *) neP;
  neP->line_search	= NLStepDefaultLineSearch;
  neP->alpha		= 1.e-4;
  neP->maxstep		= 1.e8;
  neP->steptol		= 1.e-12;
}
/* ------------------------------------------------------------ */
/*ARGSUSED*/
void NLENewtonLS1Destroy( nlP )
NLCtx *nlP;
{
   VFREEVECS( nlP->vc, nlP->work, nlP->nwork );
   NLiBasicDestroy( nlP );	CHKERR(1);
}
/* ------------------------------------------------------------ */
/*ARGSUSED*/
/*@
   NLStepSimpleLineSearch - This routine is not a line search at all; 
   it simply uses the full Newton step.  Thus, this routine is intended 
   to serve as a template and is not recommended for general use.  

   Input Parameters:
.  nlP - nonlinear context
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction (contains new iterate on output)
.  w - work vector
.  fnorm - 2-norm of f

   Output parameters:
.  g - residual evaluated at new iterate y
.  y - new iterate (contains search direction on input)
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length

   Returns:
   1, indicating success of the step.

   Notes:
   Use either NLSetStepLineSearchRoutines() or NLSetLineSearchRoutine()
   to set this routine within the NLE_NLS1 method.  
@*/
int NLStepSimpleLineSearch( nlP, x, f, g, y, w, fnorm, 
                            ynorm, gnorm )
NLCtx  *nlP;
void   *x; 
void   *f; 
void   *g; 
void   *y; 
void   *w; 
double fnorm;
double *ynorm;
double *gnorm;
{
  NLMonCore *mc = &nlP->mon.core;
  VECntx    *vc = NLGetVectorCtx(nlP);

  CHKCOOKIEN(nlP,NL_COOKIE);
  VNORM( vc, y, ynorm );	/* ynorm = || y ||    */
  VAXPY( vc, 1.0, x, y );	/* y <- x + y         */
  RESIDUAL( y, g );		/* Evaluate (+/-) g(y) */
  VNORM( vc, g, gnorm ); 	/* gnorm = || g ||    */
  mc->nvectors += 6;
  mc->nscalars += 2;
  return 1;
}
/* ------------------------------------------------------------------ */
/*@
   NLStepDefaultLineSearch - This routine performs a cubic line search.

   Input Parameters:
.  nlP - nonlinear context
.  x - current iterate
.  f - residual evaluated at x
.  y - search direction (contains new iterate on output)
.  w - work vector
.  fnorm - 2-norm of f

   Output parameters:
.  g - residual evaluated at new iterate y
.  y - new iterate (contains search direction on input)
.  gnorm - 2-norm of g
.  ynorm - 2-norm of search length

   Returns:
   1 if the line search succeeds; 0 if the line search fails.

   Notes:
   Use either NLSetStepLineSearchRoutines() or NLSetLineSearchRoutine()
   to set this routine within the NLE_NLS1 method.  

   This line search is taken from "Numerical Methods for Unconstrained 
   Optimization and Nonlinear Equations" by Dennis and Schnabel, page 325.

@*/
int NLStepDefaultLineSearch( nlP, x, f, g, y, w, fnorm, ynorm, gnorm )
NLCtx  *nlP;
void   *x; 
void   *f; 
void   *g; 
void   *y; 
void   *w; 
double fnorm;
double *ynorm;
double *gnorm;
{
  double          alpha, maxstep, steptol, initslope;
  double          minlambda, lambda, lambdaprev, gnormprev, lambdatemp;
  double          a, b, d, t1, t2;
  int             count;
  FILE            *fp = nlP->fp;
  NLENewtonLS1Ctx *neP = (NLENewtonLS1Ctx *) nlP->MethodPrivate;
  NLMonCore       *mc = &nlP->mon.core;
  VECntx          *vc = NLGetVectorCtx(nlP);
  
  CHKCOOKIEN(nlP,NL_COOKIE);
  alpha   = neP->alpha;
  maxstep = neP->maxstep;
  steptol = neP->steptol;

  VNORM( vc, y, ynorm );
  if (*ynorm > maxstep) {	/* Step too big, so scale back */
      VSCALE( vc, maxstep/(*ynorm), y ); 
      *ynorm = maxstep;
      mc->nvectors++;
      }
  minlambda = steptol/(*ynorm);
  VDOT( vc, f, y, &initslope ); 
  if (initslope > 0.0) initslope = -initslope;
  if (initslope == 0.0) initslope = -1.0;

  VCOPY( vc, y, w );
  VAXPY( vc, 1.0, x, w );
  RESIDUAL( w, g );		/* Evaluate (+/-) g(w) */
  VNORM( vc, g, gnorm ); 
  if (*gnorm <= fnorm + alpha*initslope) {	/* Sufficient reduction */
      if (fp) fprintf(fp,"Taking full Newton step\n");
      VCOPY( vc, w, y );
      mc->nvectors += 8; mc->nscalars += 3;
      return 1;
      }

  /* Fit points with quadratic */
  lambda = 1.0; count = 0;
  lambdatemp = -initslope/(2.0*(*gnorm - fnorm - initslope));
  lambdaprev = lambda;
  gnormprev = *gnorm;
  if (lambdatemp <= .1*lambda) { 
      lambda = .1*lambda; 
      mc->nscalars++; 
  } else lambda = lambdatemp;
  VCOPY( vc, x, w );
  VAXPY( vc, lambda, y, w );
  RESIDUAL( w, g );		/* Evaluate (+/-) g(w) */
  VNORM( vc, g, gnorm ); 
  if (*gnorm <= fnorm + alpha*initslope) {      /* sufficient reduction */
      if (fp) fprintf(fp,"Taking Newton step from quadratic \n");
      VCOPY( vc, w, y );
      mc->nvectors += 12; mc->nscalars += 8;
      return 1;
      }

  /* Fit points with cubic */
  count = 1;
  while (1) {
       if (lambda <= minlambda) { /* bad luck; use full step */
           fprintf(stderr,"Unable to find good step length! %d \n",count);
           fprintf(stderr, "f %g fnew %g ynorm %g lambda %g \n",
                   fnorm,*gnorm, *ynorm,lambda);
           VCOPY( vc, w, y );
           mc->nvectors += 12 + 4*(count-1);
           mc->nscalars += 8 + 28*(count-1);
           return 0;
           }
      t1 = *gnorm - fnorm - lambda*initslope;
      t2 = gnormprev  - fnorm - lambdaprev*initslope;
      a = (t1/(lambda*lambda) - 
                t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
      b = (-lambdaprev*t1/(lambda*lambda) + 
                lambda*t2/(lambdaprev*lambdaprev))/(lambda-lambdaprev);
      d = b*b - 3*a*initslope;
      if (d < 0.0) d = 0.0;
      if (a == 0.0) {
         lambdatemp = -initslope/(2.0*b);
         mc->nscalars += 2;
      } else {
         lambdatemp = (-b + sqrt(d))/(3.0*a);
         mc->nscalars += 4;
         }
      if (lambdatemp > .5*lambda) {
         lambdatemp = .5*lambda;
         mc->nscalars++;
         }
      lambdaprev = lambda;
      gnormprev = *gnorm;
      if (lambdatemp <= .1*lambda) {
         lambda = .1*lambda;
         mc->nscalars++;
         }
      else lambda = lambdatemp;
      VCOPY( vc, x, w );
      VAXPY( vc, lambda, y, w );
      RESIDUAL( w, g );		/* Evaluate (+/-) g(w) */
      VNORM( vc, g, gnorm ); 
      if (*gnorm <= fnorm + alpha*initslope) {      /* is reduction enough */
         if (fp) fprintf(fp,"Taking Newton step from cubic %d\n",count);
         VCOPY( vc, w, y );
         mc->nvectors += 12 + 4*count;
         mc->nscalars += 8 + 28*count;
         return 1;
         }
      count++;
      }
}
/* ------------------------------------------------------------ */
/*
   NLENewtonLS1SetParameter - Sets a chosen parameter used by the 
   NLE_NLS1 method to the specified value.

   Note:
   Possible parameters for the NLE_NLS1 method are
$    param = "alpha" - used to determine sufficient reduction
$    param = "maxstep" - maximum step size
$    param = "steptol" - step comvergence tolerance
*/
void NLENewtonLS1SetParameter( nlP, param, value )
NLCtx  *nlP;
char   *param;
double *value;
{
  NLENewtonLS1Ctx *ctx = (NLENewtonLS1Ctx *)nlP->MethodPrivate;

  CHKCOOKIE(nlP,NL_COOKIE);
  if (nlP->method != NLE_NLS1) {
      SETERRC(1,"Compatible only with NLE_NLS1 method");
      return;
      }  
  if (!strcmp(param,"alpha"))		ctx->alpha   = *value;
  else if (!strcmp(param,"maxstep"))	ctx->maxstep = *value;
  else if (!strcmp(param,"steptol"))	ctx->steptol = *value;
  else SETERRC(1,"Invalid parameter name for NLE_NLS1 method");
}
/* ------------------------------------------------------------ */
/*
   NLENewtonLS1GetParameter - Returns the value of a chosen parameter
   used by the NLE_NLS1 method.

   Note:
   Possible parameters for the NLE_NLS1 method are
$    param = "alpha" - used to determine sufficient reduction
$    param = "maxstep" - maximum step size
$    param = "steptol" - step comvergence tolerance
*/
double NLENewtonLS1GetParameter( nlP, param )
NLCtx  *nlP;
char   *param;
{
  NLENewtonLS1Ctx *ctx = (NLENewtonLS1Ctx *)nlP->MethodPrivate;
  double          value = 0.0;

  CHKCOOKIEN(nlP,NL_COOKIE);
  if (nlP->method != NLE_NLS1) {
      SETERRC(1,"Compatible only with NLE_NLS1 method");
      return value;
      }  
  if (!strcmp(param,"alpha"))		value = ctx->alpha;
  else if (!strcmp(param,"maxstep"))	value = ctx->maxstep;
  else if (!strcmp(param,"steptol"))	value = ctx->steptol;
  else SETERRC(1,"Invalid parameter name for NLE_NLS1 method");
  return value;
}
/* ------------------------------------------------------------ */
/*@C
   NLSetLineSearchRoutine - Sets the line search routine to be used
   by the method NLE_NLS1.

   Input Parameters:
.  nlP - nonlinear context obtained from NLCreate()
.  func - pointer to int function

   Possible routines:
   NLStepDefaultLineSearch() - default line search
   NLStepSimpleLineSearch() - the full Newton step (actually not a
   line search)

   Calling sequence of func:
.  func (NLCtx *nlP, void *x, void *f, void *g, void *y,
         void *w, double fnorm, double *ynorm, double *gnorm )

    Input parameters for func:
.   nlP - nonlinear context
.   x - current iterate
.   f - residual evaluated at x
.   y - search direction (contains new iterate on output)
.   w - work vector
.   fnorm - 2-norm of f

    Output parameters for func:
.   g - residual evaluated at new iterate y
.   y - new iterate (contains search direction on input)
.   gnorm - 2-norm of g
.   ynorm - 2-norm of search length

    Returned by func:
    1 if the line search succeeds; 0 if the line search fails.
@*/
void NLSetLineSearchRoutine( nlP, func )
NLCtx *nlP;
int   (*func)();
{
  CHKCOOKIE(nlP,NL_COOKIE);
  if ((nlP)->method == NLE_NLS1)
     ((NLENewtonLS1Ctx *)(nlP->MethodPrivate))->line_search = func;
}
