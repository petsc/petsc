#ifndef lint
static char vcid[] = "$Id: itcl.c,v 1.26 1995/05/19 00:45:01 bsmith Exp bsmith $";
#endif
/*
    Command line interface for KSP
*/

#include "petsc.h"
#include "draw.h"
#include "kspimpl.h"
#include "sys.h"

extern int KSPGetMethodFromOptions_Private(KSP,KSPMethod *);

/*@
   KSPSetFromOptions - Sets KSP options from the options database.
   This routine must be called before KSPSetUp() if the user is to be 
   allowed to set the Krylov method. 

   Input Parameters:
.  ctx - the Krylov space context

.keywords: KSP, set, from, options, database

.seealso: KSPPrintHelp()
@*/
int KSPSetFromOptions(KSP ctx)
{
  KSPMethod method;
  int       restart;
  VALIDHEADER(ctx,KSP_COOKIE);

  if (OptionsHasName(0,"-help")) {
    KSPPrintHelp(ctx);
  }
  if (KSPGetMethodFromOptions_Private(ctx,&method)) {
    KSPSetMethod(ctx,method);
  }
  OptionsGetInt(ctx->prefix,"-ksp_max_it",&ctx->max_it);
  OptionsGetDouble(ctx->prefix,"-ksp_rtol",&ctx->rtol);  
  OptionsGetDouble(ctx->prefix,"-ksp_atol",&ctx->atol);
  OptionsGetDouble(ctx->prefix,"-ksp_divtol",&ctx->divtol);
  if (OptionsHasName(ctx->prefix,"-ksp_monitor")){
    int mytid = 0;
    MPI_Initialized(&mytid);
    if (mytid) MPI_Comm_rank(ctx->comm,&mytid);
    if (!mytid) {
      KSPSetMonitor(ctx,KSPDefaultMonitor,(void *)0);
    }
  }
  /* this is not good!
       1) there is no way to free lg at end of KSP
  */
  {
  int loc[4] = {0,0,300,300},nmax = 4;
  if (OptionsGetIntArray(ctx->prefix,"-ksp_xmonitor",loc,&nmax)){
    int       ierr,mytid = 0;
    DrawLGCtx lg;
    MPI_Initialized(&mytid);
    if (mytid) MPI_Comm_rank(ctx->comm,&mytid);
    if (!mytid) {
      ierr = KSPLGMonitorCreate(0,0,loc[0],loc[1],loc[2],loc[3],&lg); 
      CHKERRQ(ierr);
      KSPSetMonitor(ctx,KSPLGMonitor,(void *)lg);
    }
  }
  }
  if (OptionsHasName(ctx->prefix,"-ksp_preres")) {
    KSPSetUsePreconditionedResidual(ctx);
  }
  if (OptionsGetInt(ctx->prefix,"-ksp_gmres_restart",&restart)) {
    KSPGMRESSetRestart(ctx,restart);
  }
  if (OptionsHasName(ctx->prefix,"-ksp_gmres_unmodifiedgrammschmidt")) {
    KSPGMRESSetUseUnmodifiedGrammSchmidt(ctx);
  }
  if (OptionsHasName(ctx->prefix,"-ksp_eigen")) {
    KSPSetCalculateEigenvalues(ctx);
  }
  return 0;
}
  
extern int KSPPrintMethods_Private(char *,char *);

/*@ 
   KSPPrintHelp - Prints all options for the KSP component.

   Input Parameters:
.  ctx - the KSP context

.keywords: KSP, help

.seealso: KSPSetFromOptions()
@*/
int KSPPrintHelp(KSP ctx)
{
  char *p;
  int  mytid = 0;
  MPI_Initialized(&mytid);
  if (mytid) MPI_Comm_rank(ctx->comm,&mytid);
    
  if (!mytid) {
    if (ctx->prefix) p = ctx->prefix;
    else             p = "-";
    VALIDHEADER(ctx,KSP_COOKIE);
    fprintf(stderr,"KSP Options -------------------------------------\n");
    KSPPrintMethods_Private(p,"ksp_method");
    fprintf(stderr," %sksp_rtol tol: relative tolerance, defaults to %g\n",
                     p,ctx->rtol);
    fprintf(stderr," %sksp_atol tol: absolute tolerance, defaults to %g\n",
                     p,ctx->atol);
    fprintf(stderr," %sksp_divtol tol: divergence tolerance, defaults to %g\n",
                     p,ctx->divtol);
    fprintf(stderr," %sksp_max_it maxit: maximum iterations, defaults to %d\n",
                     p,ctx->max_it);
    fprintf(stderr," %sksp_preres: use precond. resid. in converg. test\n",p);
    fprintf(stderr," %sksp_monitor: use residual convergence monitor)\n",p);
    fprintf(stderr," %sksp_xmonitor [x,y,w,h]: use X graphics residual convergence monitor\n",p);
    fprintf(stderr," %sksp_gmres_restart num: gmres restart, defaults to 10)\n",p);
    fprintf(stderr," %sksp_eigen: calculate eigenvalues during linear solve\n",p);
    fprintf(stderr," %sksp_gmres_unmodifiedgrammschmidt\n",p);
  }
  return 1;
}

/*@
   KSPSetOptionsPrefix - Sets the prefix used for searching for all 
   KSP options in the database.

   Input Parameters:
.  ksp - the Krylov context
.  prefix - the prefix string to prepend to all KSP option requests

.keywords: KSP, set, options, prefix, database
@*/
int KSPSetOptionsPrefix(KSP ksp,char *prefix)
{
  ksp->prefix = prefix;
  return 0;
}
