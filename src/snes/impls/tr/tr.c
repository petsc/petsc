#ifndef lint
static char vcid[] = "$Id: tr.c,v 1.17 1995/07/20 15:35:04 curfman Exp curfman $";
#endif

#include <math.h>
#include "tr.h"
#include "pviewer.h"

/*
      This convergence test determines if the two norm of the 
   solution lies outside the trust region, if so it halts.
*/
int TRConverged_Private(KSP ksp,int n, double rnorm, void *ctx)
{
  SNES    snes = (SNES) ctx;
  SNES_TR *neP = (SNES_TR*)snes->data;
  double  rtol,atol,dtol,norm;
  Vec     x;
  int     ierr, mkit;

  KSPGetTolerances(ksp,&rtol,&atol,&dtol,&mkit);

  if ( n == 0 ) {
    neP->ttol   = MAX(rtol*rnorm,atol);
    neP->rnorm0 = rnorm;
  }
  if ( rnorm <= neP->ttol )      return 1;
  if ( rnorm >= dtol*neP->rnorm0 || rnorm != rnorm) return -1;

  /* Determine norm of solution */
  ierr = KSPBuildSolution(ksp,0,&x); CHKERRQ(ierr);
  ierr = VecNorm(x,&norm); CHKERRQ(ierr);
  if (norm >= neP->delta) {
    PLogInfo((PetscObject)snes,
      "Ending linear iteration early, delta %g length %g\n",neP->delta,norm);
    return 1; 
  }
  return(0);
}
/*
   SNESSolve_TR - Implements Newton's Method with a very simple trust 
   region approach for solving systems of nonlinear equations. 

   The basic algorithm is taken from "The Minpack Project", by More', 
   Sorensen, Garbow, Hillstrom, pages 88-111 of "Sources and Development 
   of Mathematical Software", Wayne Cowell, editor.

   This is intended as a model implementation, since it does not 
   necessarily have many of the bells and whistles of other 
   implementations.  
*/
static int SNESSolve_TR(SNES snes,int *its)
{
  SNES_TR      *neP = (SNES_TR *) snes->data;
  Vec          X, F, Y, G, TMP, Ytmp;
  int          maxits, i, history_len, ierr, lits;
  MatStructure flg = ALLMAT_DIFFERENT_NONZERO_PATTERN;
  double       rho, fnorm, gnorm, gpnorm, xnorm, delta,norm;
  double       *history, ynorm;
  Scalar       one = 1.0,cnorm;
  double       epsmch = 1.0e-14;   /* This must be fixed */
  KSP          ksp;
  SLES         sles;

  history	= snes->conv_hist;	/* convergence history */
  history_len	= snes->conv_hist_len;	/* convergence history length */
  maxits	= snes->max_its;	/* maximum number of iterations */
  X		= snes->vec_sol;	/* solution vector */
  F		= snes->vec_func;	/* residual vector */
  Y		= snes->work[0];	/* work vectors */
  G		= snes->work[1];
  Ytmp          = snes->work[2];

  ierr = SNESComputeInitialGuess(snes,X); CHKERRQ(ierr);  /* X <- X_0 */
  VecNorm(X, &xnorm ); 		/* xnorm = || X || */
   
  ierr = SNESComputeFunction(snes,X,F); CHKERRQ(ierr); /* (+/-) F(X) */
  VecNorm(F, &fnorm );		/* fnorm <- || F || */ 
  snes->norm = fnorm;
  if (history && history_len > 0) history[0] = fnorm;
  delta = neP->delta0*fnorm;         
  neP->delta = delta;
  if (snes->monitor)
    {ierr = (*snes->monitor)(snes,0,fnorm,snes->monP); CHKERRQ(ierr);}

  /* Set the stopping criteria to use the More' trick. */
  ierr = SNESGetSLES(snes,&sles); CHKERRQ(ierr);
  ierr = SLESGetKSP(sles,&ksp); CHKERRQ(ierr);
  ierr = KSPSetConvergenceTest(ksp,TRConverged_Private,(void *) snes);
  CHKERRQ(ierr);
 
  for ( i=0; i<maxits; i++ ) {
     snes->iter = i+1;

     ierr = SNESComputeJacobian(snes,X,&snes->jacobian,&snes->jacobian_pre,
                                             &flg); CHKERRQ(ierr);
     ierr = SLESSetOperators(snes->sles,snes->jacobian,snes->jacobian_pre,
                            flg); CHKERRQ(ierr);
     ierr = SLESSolve(snes->sles,F,Ytmp,&lits); CHKERRQ(ierr);
     VecNorm( Ytmp, &norm );
     while(1) {
       VecCopy(Ytmp,Y);
       /* Scale Y if need be and predict new value of F norm */

       if (norm >= delta) {
         norm = delta/norm;
         gpnorm = (1.0 - norm)*fnorm;
         cnorm = norm;
         PLogInfo((PetscObject)snes, "Scaling direction by %g \n",norm );
         VecScale( &cnorm, Y );
         norm = gpnorm;
         ynorm = delta;
       } else {
         gpnorm = 0.0;
         PLogInfo((PetscObject)snes,"Direction is in Trust Region \n" );
         ynorm = norm;
       }
       VecAXPY(&one, X, Y );	/* Y <- X + Y */
       ierr = VecCopy(X,snes->vec_sol_update_always); CHKERRQ(ierr);
       ierr = SNESComputeFunction(snes,Y,G); CHKERRQ(ierr); /* (+/-) F(X) */
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
       PLogInfo((PetscObject)snes,"Trying again in smaller region\n");
       /* check to see if progress is hopeless */
       if (neP->delta < xnorm * epsmch)	return -1;
     }
     fnorm = gnorm;
     snes->norm = fnorm;
     if (history && history_len > i+1) history[i+1] = fnorm;
     TMP = F; F = G; snes->vec_func_always = F; G = TMP;
     TMP = X; X = Y; snes->vec_sol_always = X; Y = TMP;
     VecNorm(X, &xnorm );		/* xnorm = || X || */
     if (snes->monitor) 
       {(*snes->monitor)(snes,i+1,fnorm,snes->monP); CHKERRQ(ierr);}

     /* Test for convergence */
     if ((*snes->converged)( snes, xnorm, ynorm, fnorm,snes->cnvP )) {
       /* Verify solution is in corect location */
       if (X != snes->vec_sol) {
         VecCopy(X, snes->vec_sol );
         snes->vec_sol_always = snes->vec_sol;
         snes->vec_func_always = snes->vec_func; 
       }
       break;
     } 
   }
   if (i == maxits) *its = i-1; else *its = i + 1;
   return 0;
}
/*------------------------------------------------------------*/
static int SNESSetUp_TR( SNES snes )
{
  int ierr;
  snes->nwork = 4;
  ierr = VecGetVecs(snes->vec_sol,snes->nwork,&snes->work ); CHKERRQ(ierr);
  snes->vec_sol_update_always = snes->work[3];
  return 0;
}
/*------------------------------------------------------------*/
static int SNESDestroy_TR(PetscObject obj )
{
  SNES snes = (SNES) obj;
  VecFreeVecs(snes->work, snes->nwork );
  PETSCFREE(snes->data);
  return 0;
}
/*------------------------------------------------------------*/

static int SNESSetFromOptions_TR(SNES snes)
{
  SNES_TR *ctx = (SNES_TR *)snes->data;
  double  tmp;

  if (OptionsGetDouble(snes->prefix,"-mu",&tmp)) {ctx->mu = tmp;}
  if (OptionsGetDouble(snes->prefix,"-eta",&tmp)) {ctx->eta = tmp;}
  if (OptionsGetDouble(snes->prefix,"-sigma",&tmp)) {ctx->sigma = tmp;}
  if (OptionsGetDouble(snes->prefix,"-delta0",&tmp)) {ctx->delta0 = tmp;}
  if (OptionsGetDouble(snes->prefix,"-delta1",&tmp)) {ctx->delta1 = tmp;}
  if (OptionsGetDouble(snes->prefix,"-delta2",&tmp)) {ctx->delta2 = tmp;}
  if (OptionsGetDouble(snes->prefix,"-delta3",&tmp)) {ctx->delta3 = tmp;}
  return 0;
}

static int SNESPrintHelp_TR(SNES snes)
{
  SNES_TR *ctx = (SNES_TR *)snes->data;
  char    *prefix = "-";
  if (snes->prefix) prefix = snes->prefix;
  MPIU_fprintf(snes->comm,stdout," method tr:\n");
  MPIU_fprintf(snes->comm,stdout,"   %smu mu (default %g)\n",prefix,ctx->mu);
  MPIU_fprintf(snes->comm,stdout,"   %seta eta (default %g)\n",prefix,ctx->eta);
  MPIU_fprintf(snes->comm,stdout,"   %ssigma sigma (default %g)\n",prefix,ctx->sigma);
  MPIU_fprintf(snes->comm,stdout,"   %sdelta0 delta0 (default %g)\n",prefix,ctx->delta0);
  MPIU_fprintf(snes->comm,stdout,"   %sdelta1 delta1 (default %g)\n",prefix,ctx->delta1);
  MPIU_fprintf(snes->comm,stdout,"   %sdelta2 delta2 (default %g)\n",prefix,ctx->delta2);
  MPIU_fprintf(snes->comm,stdout,"   %sdelta3 delta3 (default %g)\n",prefix,ctx->delta3);
  return 0;
}

static int SNESView_TR(PetscObject obj,Viewer viewer)
{
  SNES    snes = (SNES)obj;
  SNES_TR *tr = (SNES_TR *)snes->data;
  FILE    *fd = ViewerFileGetPointer_Private(viewer);
  
  MPIU_fprintf(snes->comm,fd,"    mu=%g, eta=%g, sigma=%g\n",
    tr->mu,tr->eta,tr->sigma);
  MPIU_fprintf(snes->comm,fd,
    "    delta0=%g, delta1=%g, delta2=%g, delta3=%g\n",
    tr->delta0,tr->delta1,tr->delta2,tr->delta3);
  return 0;
}

int SNESCreate_TR(SNES snes )
{
  SNES_TR *neP;

  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,
    "SNESCreate_TR: Valid for SNES_NONLINEAR_EQUATIONS problems only");
  snes->type 		= SNES_EQ_NTR;
  snes->setup		= SNESSetUp_TR;
  snes->solve		= SNESSolve_TR;
  snes->destroy		= SNESDestroy_TR;
  snes->converged	= SNESDefaultConverged;
  snes->printhelp       = SNESPrintHelp_TR;
  snes->setfromoptions  = SNESSetFromOptions_TR;
  snes->view            = SNESView_TR;

  neP			= PETSCNEW(SNES_TR); CHKPTRQ(neP);
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
  return 0;
}
