#ifndef lint
static char vcid[] = "$Id: itcl.c,v 1.57 1996/03/05 00:26:07 curfman Exp balay $";
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
  int       restart, flg, ierr;
  PETSCVALIDHEADERSPECIFIC(ctx,KSP_COOKIE);

  ierr = OptionsHasName(PETSC_NULL,"-help", &flg);  CHKERRQ(ierr);
  if (flg) { KSPPrintHelp(ctx);  }
  if (KSPGetTypeFromOptions_Private(ctx,&method)) {
    KSPSetType(ctx,method);
  }
  ierr = OptionsGetInt(ctx->prefix,"-ksp_max_it",&ctx->max_it, &flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(ctx->prefix,"-ksp_rtol",&ctx->rtol, &flg);  CHKERRQ(ierr);
  ierr = OptionsGetDouble(ctx->prefix,"-ksp_atol",&ctx->atol, &flg);  CHKERRQ(ierr);
  ierr = OptionsGetDouble(ctx->prefix,"-ksp_divtol",&ctx->divtol, &flg); CHKERRQ(ierr);
  ierr = OptionsHasName(ctx->prefix,"-ksp_gmres_preallocate", &flg); CHKERRQ(ierr);
  if(flg){
    KSPGMRESSetPreAllocateVectors(ctx);
  }
  ierr = OptionsHasName(ctx->prefix,"-ksp_monitor", &flg);  CHKERRQ(ierr);
  if (flg) {
    int rank = 0;
    MPI_Comm_rank(ctx->comm,&rank);
    if (!rank) {
      KSPSetMonitor(ctx,KSPDefaultMonitor,(void *)0);
    }
  }
  ierr = OptionsHasName(ctx->prefix,"-ksp_smonitor", &flg); CHKERRQ(ierr); 
  if (flg){
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
  int loc[4], nmax = 4;
  loc[0] = 0; loc[1] = 0; loc[2] = 300; loc[3] = 300;
  ierr = OptionsGetIntArray(ctx->prefix,"-ksp_xmonitor",loc,&nmax, &flg); CHKERRQ(ierr);
  if (flg){
    int    rank = 0;
    DrawLG lg;
    MPI_Initialized(&rank);
    if (rank) MPI_Comm_rank(ctx->comm,&rank);
    if (!rank) {
      ierr = KSPLGMonitorCreate(0,0,loc[0],loc[1],loc[2],loc[3],&lg); CHKERRQ(ierr);
      PLogObjectParent(ctx,(PetscObject) lg);
      KSPSetMonitor(ctx,KSPLGMonitor,(void *)lg);
    }
  }
  }
  ierr = OptionsHasName(ctx->prefix,"-ksp_preres", &flg); CHKERRQ(ierr);
  if (flg) { KSPSetUsePreconditionedResidual(ctx); }
  ierr = OptionsHasName(ctx->prefix,"-ksp_left_pc", &flg); CHKERRQ(ierr);
  if (flg) { KSPSetPreconditionerSide(ctx,PC_LEFT); }
  ierr = OptionsHasName(ctx->prefix,"-ksp_right_pc", &flg); CHKERRQ(ierr);
  if (flg) { KSPSetPreconditionerSide(ctx,PC_RIGHT); }
  ierr = OptionsHasName(ctx->prefix,"-ksp_symmetric_pc", &flg); CHKERRQ(ierr);
  if (flg) {  KSPSetPreconditionerSide(ctx,PC_SYMMETRIC); }
  ierr = OptionsGetInt(ctx->prefix,"-ksp_gmres_restart",&restart, &flg); CHKERRQ(ierr);
  if (flg) { KSPGMRESSetRestart(ctx,restart); }
  ierr = OptionsHasName(ctx->prefix,"-ksp_gmres_unmodifiedgramschmidt",&flg);CHKERRQ(ierr);
  if (flg) { KSPGMRESSetOrthogRoutine(ctx, KSPGMRESUnmodifiedOrthog ); }
  ierr = OptionsHasName(ctx->prefix,"-ksp_gmres_irorthog", &flg); CHKERRQ(ierr);
  if(flg) {  KSPGMRESSetOrthogRoutine(ctx, KSPGMRESIROrthog ); }
  ierr = OptionsHasName(ctx->prefix,"-ksp_eigen", &flg); CHKERRQ(ierr);
  if (flg) { KSPSetCalculateEigenvalues(ctx); }
  return 0;
}
  
extern int KSPPrintTypes_Private(MPI_Comm,char *,char *);

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
  char p[64];
  int  rank = 0;

  MPI_Comm_rank(ctx->comm,&rank);
    
  if (!rank) {
    PetscStrcpy(p,"-");
    if (ctx->prefix)  PetscStrcat(p,ctx->prefix);
    PETSCVALIDHEADERSPECIFIC(ctx,KSP_COOKIE);
    MPIU_printf(ctx->comm,"KSP Options -------------------------------------\n");
    KSPPrintTypes_Private(ctx->comm,p,"ksp_type");
    MPIU_printf(ctx->comm," %sksp_rtol tol: relative tolerance, defaults to %g\n",
                     p,ctx->rtol);
    MPIU_printf(ctx->comm," %sksp_atol tol: absolute tolerance, defaults to %g\n",
                     p,ctx->atol);
    MPIU_printf(ctx->comm," %sksp_divtol tol: divergence tolerance, defaults to %g\n",
                     p,ctx->divtol);
    MPIU_printf(ctx->comm," %sksp_max_it maxit: maximum iterations, defaults to %d\n",
                     p,ctx->max_it);
    MPIU_printf(ctx->comm," %sksp_preres: use precond. resid. in converg. test\n",p);
    MPIU_printf(ctx->comm," %sksp_right_pc: use right preconditioner instead of left\n",p);
    MPIU_printf(ctx->comm," %sksp_monitor: at each iteration print residual norm to stdout\n",p);
    MPIU_printf(ctx->comm," %sksp_xmonitor [x,y,w,h]: use X graphics residual convergence monitor\n",p);
    MPIU_printf(ctx->comm," %sksp_gmres_restart num: gmres restart, defaults to 30\n",p);
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
  PETSCVALIDHEADERSPECIFIC(ksp,KSP_COOKIE);
  return PetscObjectSetPrefix((PetscObject)ksp, prefix);
}

 
/*@C
   KSPAppendOptionsPrefix - Appends to the prefix used for searching for all 
   KSP options in the database.

   Input Parameters:
.  ksp - the Krylov context
.  prefix - the prefix string to prepend to all KSP option requests

.keywords: KSP, append, options, prefix, database
@*/
int KSPAppendOptionsPrefix(KSP ksp,char *prefix)
{
  PETSCVALIDHEADERSPECIFIC(ksp,KSP_COOKIE);
  return PetscObjectAppendPrefix((PetscObject)ksp, prefix);
}

 

/*@
   KSPGetOptionsPrefix - Gets the prefix used for searching for all 
   KSP options in the database.

   Input Parameters:
.  ksp - the Krylov context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

.keywords: KSP, set, options, prefix, database
@*/
int KSPGetOptionsPrefix(KSP ksp,char **prefix)
{
  PETSCVALIDHEADERSPECIFIC(ksp,KSP_COOKIE);
  return PetscObjectGetPrefix((PetscObject)ksp, prefix);
}

 



