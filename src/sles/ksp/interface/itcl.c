#ifndef lint
static char vcid[] = "$Id: itcl.c,v 1.81 1996/12/12 03:40:46 bsmith Exp balay $";
#endif
/*
    Code for setting KSP options from the options database.
*/

#include "draw.h"             /*I "draw.h" I*/
#include "src/ksp/kspimpl.h"  /*I "ksp.h" I*/
#include "sys.h"

extern int KSPGetTypeFromOptions_Private(KSP,KSPType *);
extern int KSPMonitor_MPIRowbs(KSP,int,double,void *);

/*
       We retain a list of functions that also take KSP command 
    line options. These are called at the end KSPSetFromOptions()
*/
#define MAXSETFROMOPTIONS 5
static int numberofsetfromoptions;
static int (*othersetfromoptions[MAXSETFROMOPTIONS])(KSP);

#undef __FUNCTION__  
#define __FUNCTION__ "KSPAddOptionsChecker"
/*@
    KSPAddOptionsChecker - Adds an additional function to check for KSP options.

  Input Parameter:
.   kspcheck - function that checks for options

.seealso: KSPSetFromOptions()
@*/
int KSPAddOptionsChecker(int (*kspcheck)(KSP) )
{
  if (numberofsetfromoptions >= MAXSETFROMOPTIONS) {
    SETERRQ(1,"KSPAddOptionsChecker:Too many options checkers, only 5 allowed");
  }

  othersetfromoptions[numberofsetfromoptions++] = kspcheck;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "KSPSetFromOptions"
/*@
   KSPSetFromOptions - Sets KSP options from the options database.
   This routine must be called before KSPSetUp() if the user is to be 
   allowed to set the Krylov type. 

   Input Parameters:
.  ksp - the Krylov space context

.keywords: KSP, set, from, options, database

.seealso: KSPPrintHelp()
@*/
int KSPSetFromOptions(KSP ksp)
{
  KSPType   method;
  int       restart, flg, ierr,loc[4], nmax = 4,i;
  double    tmp;

  loc[0] = PETSC_DECIDE; loc[1] = PETSC_DECIDE; loc[2] = 300; loc[3] = 300;

  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  ierr = OptionsHasName(PETSC_NULL,"-help", &flg); CHKERRQ(ierr);
  if (flg) { KSPPrintHelp(ksp);  }
  if (KSPGetTypeFromOptions_Private(ksp,&method)) {
    ierr = KSPSetType(ksp,method); CHKERRQ(ierr);
  }
  ierr = OptionsGetInt(ksp->prefix,"-ksp_max_it",&ksp->max_it, &flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(ksp->prefix,"-ksp_rtol",&ksp->rtol, &flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(ksp->prefix,"-ksp_atol",&ksp->atol, &flg); CHKERRQ(ierr);
  ierr = OptionsGetDouble(ksp->prefix,"-ksp_divtol",&ksp->divtol, &flg); CHKERRQ(ierr);
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_preallocate", &flg); CHKERRQ(ierr);
  if (flg) {
    ierr = KSPGMRESSetPreAllocateVectors(ksp); CHKERRQ(ierr);
  }
  /* -----------------------------------------------------------------------*/
  /*
     Cancels all monitors hardwired into code before call to KSPSetFromOptions()
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_cancelmonitors",&flg); CHKERRQ(ierr);
  if (flg) {
    KSPSetMonitor(ksp,0,(void *)0);
  }
  /*
     Prints preconditioned residual norm at each iteration
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_monitor",&flg); CHKERRQ(ierr);
  if (flg) {
    int rank = 0;
    MPI_Comm_rank(ksp->comm,&rank);
    if (!rank) {
      KSPSetMonitor(ksp,KSPDefaultMonitor,(void *)0);
    }
  }
  /*
     Prints preconditioned and true residual norm at each iteration
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_truemonitor",&flg); CHKERRQ(ierr);
  if (flg) {
    KSPSetMonitor(ksp,KSPTrueMonitor,(void *)0); 
  }
  /*
     Prints extreme eigenvalue estimates at each iteration
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_singmonitor",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetComputeSingularValues(ksp); CHKERRQ(ierr);
    KSPSetMonitor(ksp,KSPSingularValueMonitor,(void *)0); 
  }
  /*
     Prints true residual for BlockSolve95 preconditioners
  */
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
  ierr = OptionsHasName(ksp->prefix,"-ksp_bsmonitor",&flg); CHKERRQ(ierr);
  if (flg) {
    KSPSetMonitor(ksp,KSPMonitor_MPIRowbs,(void *)0);
  }
#endif
  /*
     Prints preconditioned residual norm with fewer digits
  */
  ierr = OptionsHasName(ksp->prefix,"-ksp_smonitor",&flg); CHKERRQ(ierr); 
  if (flg) {
    int rank = 0;
    MPI_Comm_rank(ksp->comm,&rank);
    if (!rank) {
      KSPSetMonitor(ksp,KSPDefaultSMonitor,(void *)0);
    }
  }
  /*
     Graphically plots preconditioned residual norm
  */
  nmax = 4;
  ierr = OptionsGetIntArray(ksp->prefix,"-ksp_xmonitor",loc,&nmax,&flg); CHKERRQ(ierr);
  if (flg) {
    int    rank = 0;
    DrawLG lg;
    MPI_Comm_rank(ksp->comm,&rank);
    if (!rank) {
      ierr = KSPLGMonitorCreate(0,0,loc[0],loc[1],loc[2],loc[3],&lg); CHKERRQ(ierr);
      PLogObjectParent(ksp,(PetscObject) lg);
      KSPSetMonitor(ksp,KSPLGMonitor,(void *)lg);
      ksp->xmonitor = lg; 
    }
  }
  /*
     Graphically plots preconditioned and true residual norm
  */
  nmax = 4;
  ierr = OptionsGetIntArray(ksp->prefix,"-ksp_xtruemonitor",loc,&nmax,&flg);CHKERRQ(ierr);
  if (flg){
    int    rank = 0;
    DrawLG lg;
    MPI_Comm_rank(ksp->comm,&rank);
    if (!rank) {
      ierr = KSPLGTrueMonitorCreate(ksp->comm,0,0,loc[0],loc[1],loc[2],loc[3],&lg);CHKERRQ(ierr);
      PLogObjectParent(ksp,(PetscObject) lg);
      KSPSetMonitor(ksp,KSPLGTrueMonitor,(void *)lg);
      ksp->xmonitor = lg; 
    } 
  }
  /* -----------------------------------------------------------------------*/
  ierr = OptionsHasName(ksp->prefix,"-ksp_preres",&flg); CHKERRQ(ierr);
  if (flg) { KSPSetUsePreconditionedResidual(ksp); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_left_pc",&flg); CHKERRQ(ierr);
  if (flg) { KSPSetPreconditionerSide(ksp,PC_LEFT); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_right_pc",&flg); CHKERRQ(ierr);
  if (flg) { KSPSetPreconditionerSide(ksp,PC_RIGHT); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_symmetric_pc",&flg); CHKERRQ(ierr);
  if (flg) { KSPSetPreconditionerSide(ksp,PC_SYMMETRIC); }
  ierr = OptionsGetInt(ksp->prefix,"-ksp_gmres_restart",&restart,&flg); CHKERRQ(ierr);
  if (flg) { KSPGMRESSetRestart(ksp,restart); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_unmodifiedgramschmidt",&flg);CHKERRQ(ierr);
  if (flg) { KSPGMRESSetOrthogonalization(ksp, 
             KSPGMRESUnmodifiedGramSchmidtOrthogonalization ); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_gmres_irorthog",&flg);CHKERRQ(ierr);
  if (flg) { KSPGMRESSetOrthogonalization(ksp, KSPGMRESIROrthogonalization);}
  ierr = OptionsHasName(ksp->prefix,"-ksp_compute_singularvalues",&flg); CHKERRQ(ierr);
  if (flg) { KSPSetComputeSingularValues(ksp); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_compute_eigenvalues",&flg);CHKERRQ(ierr);
  if (flg) { KSPSetComputeSingularValues(ksp); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_plot_eigenvalues",&flg);CHKERRQ(ierr);
  if (flg) { KSPSetComputeSingularValues(ksp); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_cg_Hermitian",&flg);CHKERRQ(ierr);
  if (flg) { KSPCGSetType(ksp,KSP_CG_HERMITIAN); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_cg_symmetric",&flg);CHKERRQ(ierr);
  if (flg) { KSPCGSetType(ksp,KSP_CG_SYMMETRIC); }
  ierr = OptionsGetDouble(ksp->prefix,"-ksp_richardson_scale",&tmp,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPRichardsonSetScale(ksp,tmp); CHKERRQ(ierr); }

  for ( i=0; i<numberofsetfromoptions; i++ ) {
    ierr = (*othersetfromoptions[i])(ksp); CHKERRQ(ierr);
  }
  return 0;
}
  
extern int KSPPrintTypes_Private(MPI_Comm,char *,char *);

#undef __FUNCTION__  
#define __FUNCTION__ "KSPPrintHelp"
/*@ 
   KSPPrintHelp - Prints all options for the KSP component.

   Input Parameter:
.  ksp - the KSP context

   Options Database Keys:
$  -help, -h

.keywords: KSP, help

.seealso: KSPSetFromOptions()
@*/
int KSPPrintHelp(KSP ksp)
{
  char p[64];
  int  rank = 0;

  MPI_Comm_rank(ksp->comm,&rank);
    
  if (!rank) {
    PetscStrcpy(p,"-");
    if (ksp->prefix)  PetscStrcat(p,ksp->prefix);
    PetscValidHeaderSpecific(ksp,KSP_COOKIE);
    PetscPrintf(ksp->comm,"KSP options -------------------------------------------------\n");
    KSPPrintTypes_Private(ksp->comm,p,"ksp_type");
    PetscPrintf(ksp->comm," %sksp_rtol <tol>: relative tolerance, defaults to %g\n",
                     p,ksp->rtol);
    PetscPrintf(ksp->comm," %sksp_atol <tol>: absolute tolerance, defaults to %g\n",
                     p,ksp->atol);
    PetscPrintf(ksp->comm," %sksp_divtol <tol>: divergence tolerance, defaults to %g\n",
                     p,ksp->divtol);
    PetscPrintf(ksp->comm," %sksp_max_it <maxit>: maximum iterations, defaults to %d\n",
                     p,ksp->max_it);
    PetscPrintf(ksp->comm," %sksp_preres: use preconditioned residual norm in convergence test\n",p);
    PetscPrintf(ksp->comm," %sksp_right_pc: use right preconditioner instead of left\n",p);
    PetscPrintf(ksp->comm," KSP Monitoring Options: Choose any of the following\n");
    PetscPrintf(ksp->comm,"   %sksp_cancelmonitors: cancels all monitors hardwired in code\n",p);
    PetscPrintf(ksp->comm,"   %sksp_monitor: at each iteration print (usually preconditioned) \n\
    residual norm to stdout\n",p);
    PetscPrintf(ksp->comm,"   %sksp_xmonitor [x,y,w,h]: use X graphics monitor of (usually \n\
    preconditioned) residual norm\n",p);
    PetscPrintf(ksp->comm,"   %sksp_truemonitor: at each iteration print true and preconditioned\n",p);
    PetscPrintf(ksp->comm,"                      residual norms to stdout\n");
    PetscPrintf(ksp->comm,"   %sksp_xtruemonitor [x,y,w,h]: use X graphics monitor of true\n",p);
    PetscPrintf(ksp->comm,"                                 residual norm\n");
    PetscPrintf(ksp->comm,"   %sksp_singmonitor: calculate singular values during linear solve\n",p);
    PetscPrintf(ksp->comm,"       (only for CG and GMRES)\n");
    PetscPrintf(ksp->comm,"   %sksp_bsmonitor: at each iteration print the unscaled and \n",p);
    PetscPrintf(ksp->comm,"       (only for ICC and ILU in BlockSolve95)\n");
    PetscPrintf(ksp->comm,"   %sksp_plot_eigenvalues_explicitly\n",p);
    PetscPrintf(ksp->comm,"   %sksp_plot_eigenvalues\n",p);
    PetscPrintf(ksp->comm," GMRES Options:\n");
    PetscPrintf(ksp->comm,"   %sksp_gmres_restart <num>: GMRES restart, defaults to 30\n",p);
    PetscPrintf(ksp->comm,"   %sksp_gmres_unmodifiedgramschmidt: use alternative orthogonalization\n",p);
    PetscPrintf(ksp->comm,"   %sksp_gmres_irorthog: use iterative refinement in orthogonalization\n",p);
    PetscPrintf(ksp->comm,"   %sksp_gmres_preallocate: preallocate GMRES work vectors\n",p);
#if defined(PETSC_COMPLEX)
    PetscPrintf(ksp->comm," CG Options:\n");
    PetscPrintf(ksp->comm,"   %sksp_cg_Hermitian: use CG for complex, Hermitian matrix (default)\n",p);
    PetscPrintf(ksp->comm,"   %sksp_cg_symmetric: use CG for complex, symmetric matrix\n",p);
#endif
  }
  return 1;
}

#undef __FUNCTION__  
#define __FUNCTION__ "KSPSetOptionsPrefix"
/*@C
   KSPSetOptionsPrefix - Sets the prefix used for searching for all 
   KSP options in the database. You must not include the - at the beginning of 
   the prefix name.

   Input Parameters:
.  ksp - the Krylov context
.  prefix - the prefix string to prepend to all KSP option requests

.keywords: KSP, set, options, prefix, database
@*/
int KSPSetOptionsPrefix(KSP ksp,char *prefix)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  return PetscObjectSetOptionsPrefix((PetscObject)ksp, prefix);
}
 
#undef __FUNCTION__  
#define __FUNCTION__ "KSPAppendOptionsPrefix"
/*@C
   KSPAppendOptionsPrefix - Appends to the prefix used for searching for all 
   KSP options in the database. You must NOT include the - at the beginning of 
   the prefix name. 

   Input Parameters:
.  ksp - the Krylov context
.  prefix - the prefix string to prepend to all KSP option requests

.keywords: KSP, append, options, prefix, database
@*/
int KSPAppendOptionsPrefix(KSP ksp,char *prefix)
{
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  return PetscObjectAppendOptionsPrefix((PetscObject)ksp, prefix);
}

#undef __FUNCTION__  
#define __FUNCTION__ "KSPGetOptionsPrefix"
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
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  return PetscObjectGetOptionsPrefix((PetscObject)ksp, prefix);
}

 



