#ifndef lint
static char vcid[] = "$Id: itcl.c,v 1.45 1995/12/21 18:29:54 bsmith Exp bsmith $";
#endif
/*
    Code for setting KSP options from the options database.
*/

#include "draw.h"     /*I "draw.h" I*/
#include "kspimpl.h"  /*I "ksp.h" I*/
#include "sys.h"

extern int KSPGetTypeFromOptions_Private(KSP,KSPType *);

/*@
   KSPSetFromOptions - Sets KSP options from the options database.
   This routine must be called before KSPSetUp() if the user is to be 
   allowed to set the Krylov type. 

   Input Parameters:
.  ctx - the Krylov space context

.keywords: KSP, set, from, options, database

.seealso: KSPPrintHelp()
@*/
int KSPSetFromOptions(KSP ctx)
{
  KSPType   method;
  int       restart;
  PETSCVALIDHEADERSPECIFIC(ctx,KSP_COOKIE);

  if (OptionsHasName(PETSC_NULL,"-help")) {
    KSPPrintHelp(ctx);
  }
  if (KSPGetTypeFromOptions_Private(ctx,&method)) {
    KSPSetType(ctx,method);
  }
  OptionsGetInt(ctx->prefix,"-ksp_max_it",&ctx->max_it);
  OptionsGetDouble(ctx->prefix,"-ksp_rtol",&ctx->rtol);  
  OptionsGetDouble(ctx->prefix,"-ksp_atol",&ctx->atol);
  OptionsGetDouble(ctx->prefix,"-ksp_divtol",&ctx->divtol);
  if (OptionsHasName(ctx->prefix,"-ksp_gmres_preallocate")){
    KSPGMRESSetPreAllocateVectors(ctx);
  }

  if (OptionsHasName(ctx->prefix,"-ksp_monitor")){
    int rank = 0;
    MPI_Comm_rank(ctx->comm,&rank);
    if (!rank) {
      KSPSetMonitor(ctx,KSPDefaultMonitor,(void *)0);
    }
  }
  if (OptionsHasName(ctx->prefix,"-ksp_smonitor")){
    int rank = 0;
    MPI_Comm_rank(ctx->comm,&rank);
    if (!rank) {
      KSPSetMonitor(ctx,KSPDefaultSMonitor,(void *)0);
    }
  }
  /* this is not good!
       1) there is no way to free lg at end of KSP
  */
  {
  int loc[4] = {0,0,300,300},nmax = 4;
  if (OptionsGetIntArray(ctx->prefix,"-ksp_xmonitor",loc,&nmax)){
    int       ierr,rank = 0;
    DrawLG lg;
    MPI_Initialized(&rank);
    if (rank) MPI_Comm_rank(ctx->comm,&rank);
    if (!rank) {
      ierr = KSPLGMonitorCreate(0,0,loc[0],loc[1],loc[2],loc[3],&lg); 
      PLogObjectParent(ctx,(PetscObject) lg);
      CHKERRQ(ierr);
      KSPSetMonitor(ctx,KSPLGMonitor,(void *)lg);
    }
  }
  }
  if (OptionsHasName(ctx->prefix,"-ksp_preres")) {
    KSPSetUsePreconditionedResidual(ctx);
  }
  if (OptionsHasName(ctx->prefix,"-ksp_right_pc")) {
    KSPSetRightPreconditioner(ctx);
  }
  if (OptionsGetInt(ctx->prefix,"-ksp_gmres_restart",&restart)) {
    KSPGMRESSetRestart(ctx,restart);
  }
  if (OptionsHasName(ctx->prefix,"-ksp_gmres_unmodifiedgramschmidt")) {
    KSPGMRESSetOrthogRoutine(ctx, KSPGMRESUnmodifiedOrthog );
  }
  if (OptionsHasName(ctx->prefix,"-ksp_gmres_irorthog")) {
    KSPGMRESSetOrthogRoutine(ctx, KSPGMRESIROrthog );
  }
  if (OptionsHasName(ctx->prefix,"-ksp_eigen")) {
    KSPSetCalculateEigenvalues(ctx);
  }
  return 0;
}
  
extern int KSPPrintTypes_Private(char *,char *);

/*@ 
   KSPPrintHelp - Prints all options for the KSP component.

   Input Parameter:
.  ctx - the KSP context

   Options Database Keys:
$  -help, -h

.keywords: KSP, help

.seealso: KSPSetFromOptions()
@*/
int KSPPrintHelp(KSP ctx)
{
  char *p;
  int  rank = 0;

  MPI_Comm_rank(ctx->comm,&rank);
    
  if (!rank) {
    if (ctx->prefix) p = ctx->prefix;
    else             p = "-";
    PETSCVALIDHEADERSPECIFIC(ctx,KSP_COOKIE);
    MPIU_printf(ctx->comm,"KSP Options -------------------------------------\n");
    KSPPrintTypes_Private(p,"ksp_type");
    MPIU_printf(ctx->comm," %sksp_rtol tol: relative tolerance, defaults to %g\n",
                     p,ctx->rtol);
    MPIU_printf(ctx->comm," %sksp_atol tol: absolute tolerance, defaults to %g\n",
                     p,ctx->atol);
    MPIU_printf(ctx->comm," %sksp_divtol tol: divergence tolerance, defaults to %g\n",
                     p,ctx->divtol);
    MPIU_printf(ctx->comm," %sksp_max_it maxit: maximum iterations, defaults to %d\n",
                     p,ctx->max_it);
    MPIU_printf(ctx->comm," %sksp_preres: use precond. resid. in converg. test\n",p);
    MPIU_printf(ctx->comm," %sksp_right_pc: use right preconditioner instead of left",p);
    MPIU_printf(ctx->comm," %sksp_monitor: use residual convergence monitor)\n",p);
    MPIU_printf(ctx->comm," %sksp_xmonitor [x,y,w,h]: use X graphics residual convergence monitor\n",p);
    MPIU_printf(ctx->comm," %sksp_gmres_restart num: gmres restart, defaults to 10)\n",p);
    MPIU_printf(ctx->comm," %sksp_eigen: calculate eigenvalues during linear solve\n",p);
    MPIU_printf(ctx->comm," %sksp_gmres_unmodifiedgramschmidt\n",p);
  }
  return 1;
}

/*@C
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

