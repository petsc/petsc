#ifndef lint
static char vcid[] = "$Id: itcl.c,v 1.13 1995/04/12 18:40:18 curfman Exp curfman $";
#endif
/*
    Command line interface for KSP
*/

#include "petsc.h"
#include "draw.h"
#include "kspimpl.h"
#include "sys.h"
#include "options.h"

/*@
   KSPSetFromOptions - Sets KSP options from the options database.
   This must be called before KSPSetUp() if the 
   user is to be allowed to set the Krylov method. 

   Input Parameters:
.  ctx - the Krylov space context
   
   Note:
   See KSPPrintHelp() for a list of avaliable KSP options.
@*/
int KSPSetFromOptions(KSP ctx)
{
  KSPMETHOD method;
  int       restart;
  VALIDHEADER(ctx,KSP_COOKIE);

  if (OptionsHasName(0,0,"-help")) {
    KSPPrintHelp(ctx);
  }
  if (KSPGetMethodFromOptions(ctx,&method)) {
    KSPSetMethod(ctx,method);
  }
  OptionsGetInt(0,ctx->prefix,"-kspmax_it",&ctx->max_it);
  OptionsGetDouble(0,ctx->prefix,"-ksprtol",&ctx->rtol);  
  OptionsGetDouble(0,ctx->prefix,"-kspatol",&ctx->atol);
  OptionsGetDouble(0,ctx->prefix,"-kspdivtol",&ctx->divtol);
  if (OptionsHasName(0,ctx->prefix,"-kspmonitor")){
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
  if (OptionsHasName(0,ctx->prefix,"-kspxmonitor")){
    int       ierr,mytid = 0;
    DrawLGCtx lg;
    MPI_Initialized(&mytid);
    if (mytid) MPI_Comm_rank(ctx->comm,&mytid);
    if (!mytid) {
      ierr = KSPLGMonitorCreate(0,0,0,0,300,300,&lg); CHKERR(ierr);
      KSPSetMonitor(ctx,KSPLGMonitor,(void *)lg);
    }
  }
  if (OptionsHasName(0,ctx->prefix,"-ksppreres")) {
    KSPSetUsePreconditionedResidual(ctx);
  }
  if (OptionsGetInt(0,ctx->prefix,"-kspgmres_restart",&restart)) {
    KSPGMRESSetRestart(ctx,restart);
  }
  if (OptionsHasName(0,0,"-kspeigen")) {
    KSPSetCalculateEigenvalues(ctx);
  }
  return 0;
}
  
/*@ 
   KSPPrintHelp - Prints all options for the KSP component.

   Input Parameters:
.  ctx - the KSP context
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
    KSPPrintMethods(p,"kspmethod");
    fprintf(stderr," %sksprtol tol: relative tolerance, defaults to %g\n",
                     p,ctx->rtol);
    fprintf(stderr," %skspatol tol: absolute tolerance, defaults to %g\n",
                     p,ctx->atol);
    fprintf(stderr," %skspdivtol tol: divergence tolerance, defaults to %g\n",
                     p,ctx->divtol);
    fprintf(stderr," %skspmax_it maxit: maximum iterations, defaults to %d\n",
                     p,ctx->max_it);
    fprintf(stderr," %sksppreres: use precond. resid. in converg. test\n",p);
    fprintf(stderr," %skspmonitor: use residual convergence monitor)\n",p);
    fprintf(stderr," %skspxmonitor [x,y,w,h]: use X graphics residual convergence monitor\n",p);
    fprintf(stderr," %skspgmres_restart num: gmres restart, defaults to 10)\n",p);
    fprintf(stderr," %skspeigen: calculate eigenvalues during linear solve\n",p);
  }
  return 1;
}

/*@
   KSPSetOptionsPrefix - Sets the prefix used for searching for all 
   KSP options in the database.

   Input Parameters:
.  ksp - the Krylov context
.  prefix - the prefix string to prepend to all KSP option requests
@*/
int KSPSetOptionsPrefix(KSP ksp,char *prefix)
{
  ksp->prefix = prefix;
  return 0;
}
