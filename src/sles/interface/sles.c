#ifndef lint
static char vcid[] = "$Id: sles.c,v 1.21 1995/05/16 00:35:26 curfman Exp curfman $";
#endif

#include "slesimpl.h"

/*@
   SLESPrintHelp - Prints SLES options.

   Input Parameter:
.  sles - the SLES context

.keywords: SLES, help

.seealso: SLESSetFromOptions()
@*/
int SLESPrintHelp(SLES sles)
{
  VALIDHEADER(sles,SLES_COOKIE);
  fprintf(stderr,"SLES options:\n");
  KSPPrintHelp(sles->ksp);
  PCPrintHelp(sles->pc);
  return 0;
}

/*@
   SLESSetOptionsPrefix - Sets the prefix used for searching for all 
   SLES options in the database.

   Input Parameter:
.  sles - the SLES context
.  prefix - the prefix to prepend to all option names

   Notes:
   This prefix is particularly useful for nested use of SLES.  For
   example, the block Jacobi and block diagonal preconditioners use
   the prefix "-sub" for options relating to the individual blocks.  

.keywords: SLES, set, options, prefix, database
@*/
int SLESSetOptionsPrefix(SLES sles,char *prefix)
{
  VALIDHEADER(sles,SLES_COOKIE);
  KSPSetOptionsPrefix(sles->ksp,prefix);
  PCSetOptionsPrefix(sles->pc,prefix);
  return 0;
}

/*@
   SLESSetFromOptions - Sets various SLES parameters from user options.
   Also takes all KSP and PC options.

   Input Parameter:
.  sles - the SLES context

.keywords: SLES, set, options, database

.seealso: SLESPrintHelp()
@*/
int SLESSetFromOptions(SLES sles)
{
  VALIDHEADER(sles,SLES_COOKIE);
  KSPSetFromOptions(sles->ksp);
  PCSetFromOptions(sles->pc);
  return 0;
}
/*@
   SLESCreate - Creates a linear equation solver context.

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  sles - the newly created SLES context

.keywords: SLES, create, context

.seealso: SLESSolve(), SLESDestroy()
@*/
int SLESCreate(MPI_Comm comm,SLES *outsles)
{
  int ierr;
  SLES sles;
  *outsles = 0;
  PETSCHEADERCREATE(sles,_SLES,SLES_COOKIE,0,comm);
  PLogObjectCreate(sles);
  if ((ierr = KSPCreate(comm,&sles->ksp))) SETERR(ierr,0);
  if ((ierr = PCCreate(comm,&sles->pc))) SETERR(ierr,0);
  PLogObjectParent(sles,sles->ksp);
  PLogObjectParent(sles,sles->pc);
  sles->setupcalled = 0;
  *outsles = sles;
  return 0;
}

/*@
   SLESDestroy - Destroys the SLES context.

   Input Parameters:
.  sles - the SLES context

.keywords: SLES, destroy, context

.seealso: SLESCreate(), SLESSolve()
@*/
int SLESDestroy(SLES sles)
{
  int ierr;
  VALIDHEADER(sles,SLES_COOKIE);
  ierr = KSPDestroy(sles->ksp); CHKERR(ierr);
  ierr = PCDestroy(sles->pc); CHKERR(ierr);
  PLogObjectDestroy(sles);
  PETSCHEADERDESTROY(sles);
  return 0;
}
extern int PCPreSolve(PC,KSP),PCPostSolve(PC,KSP);
/*@
   SLESSolve - Solves a linear system.

   Input Parameters:
.  sles - the SLES context
.  b - the right hand side

   Output Parameters:
.  x - the approximate solution
.  its - the number of iterations used

.keywords: SLES, solve, linear system

.seealso: SLESCreate(), SLESDestroy()
@*/
int SLESSolve(SLES sles,Vec b,Vec x,int *its)
{
  int ierr;
  KSP ksp;
  PC  pc;
  VALIDHEADER(sles,SLES_COOKIE);
  PLogEventBegin(SLES_Solve,sles,b,x,0);
  ksp = sles->ksp; pc = sles->pc;
  KSPSetRhs(ksp,b);
  KSPSetSolution(ksp,x);
  KSPSetBinv(ksp,pc);
  if (!sles->setupcalled) {
    if ((ierr = PCSetVector(pc,b))) SETERR(ierr,0);
    if ((ierr = KSPSetUp(sles->ksp))) SETERR(ierr,0);
    if ((ierr = PCSetUp(sles->pc))) SETERR(ierr,0);
    sles->setupcalled = 1;
  }
  ierr = PCPreSolve(pc,ksp); CHKERR(ierr);
  ierr = KSPSolve(ksp,its); CHKERR(ierr);
  ierr = PCPostSolve(pc,ksp); CHKERR(ierr);
  PLogEventEnd(SLES_Solve,sles,b,x,0);
  return 0;
}

/*@
   SLESGetKSP - Returns the KSP context for a SLES solver.

   Input Parameter:
.  sles - the SLES context

   Output Parameter:
.  ksp - the Krylov space context

   Notes:  
   The user can then directly manipulate the KSP context to set various 
   options, etc.
   
.keywords: SLES, get, KSP, context

.seealso: SLESGetPC()
@*/
int SLESGetKSP(SLES sles,KSP *ksp)
{
  VALIDHEADER(sles,SLES_COOKIE);
  *ksp = sles->ksp;
  return 0;
}
/*@
   SLESGetPC - Returns the preconditioner (PC) context for a SLES solver.

   Input Parameter:
.  sles - the SLES context

   Output Parameter:
.  pc - the preconditioner context

   Notes:  
   The user can then directly manipulate the PC context to set various 
   options, etc.

.keywords: SLES, get, PC, context

.seealso: SLESGetKSP()
@*/
int SLESGetPC(SLES sles,PC *pc)
{
  VALIDHEADER(sles,SLES_COOKIE);
  *pc = sles->pc;
  return 0;
}

#include "mat/matimpl.h"
/*@
   SLESSetOperators - Sets the matrix associated with the linear system
   and a (possibly) different one associated with the preconditioner. 

   Input Parameters:
.  sles - the sles context
.  Amat - the matrix associated with the linear system
.  Pmat - matrix to be used in constructing preconditioner, usually the same
          as Amat.  If Pmat is 0 for repeated linear solves, the old 
          preconditioner is used.
.  flag - flag indicating information about matrix structure.  When solving
   just one linear system, this flag is NOT used and can thus be set to 0.

   Notes: 
   The flag can be used to eliminate unnecessary work in the repeated
   solution of linear systems of the same size.  The available options are
$    MAT_SAME_NONZERO_PATTERN - 
$       Amat has the same nonzero structure 
$       during successive linear solves
$    PMAT_SAME_NONZERO_PATTERN -
$       Pmat has the same nonzero structure 
$       during successive linear solves
$    ALLMAT_SAME_NONZERO_PATTERN -
$       Both Amat and Pmat have the same nonzero
$       structure during successive linear solves
$    ALLMAT_DIFFERENT_NONZERO_PATTERN -
$       Neither Amat nor Pmat has same nonzero structure

.keywords: SLES, set, operators, matrix, preconditioner, linear system

.seealso: SLESSolve()
@*/
int SLESSetOperators(SLES sles,Mat Amat,Mat Pmat,MatStructure flag)
{
  VALIDHEADER(sles,SLES_COOKIE);
  VALIDHEADER(Amat,MAT_COOKIE);
  if (Pmat) {VALIDHEADER(Pmat,MAT_COOKIE);}
  PCSetOperators(sles->pc,Amat,Pmat,flag);
  sles->setupcalled = 0;  /* so that next solve call will call setup */
  return 0;
}
