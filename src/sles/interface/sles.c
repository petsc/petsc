#ifndef lint
static char vcid[] = "$Id: sles.c,v 1.29 1995/07/11 13:29:37 curfman Exp bsmith $";
#endif

#include "slesimpl.h"     /*I  "sles.h"    I*/
#include "pviewer.h"

/*@ 
   SLESView - Prints the SLES data structure.

   Input Parameters:
.  SLES - the SLES context
.  viewer - the location to display context (usually 0)

   Options Database Key:
$  -sles_view : calls SLESView() at end of SLESSolve()

.keywords: SLES, view
@*/
int SLESView(SLES sles,Viewer viewer)
{
  PetscObject vobj = (PetscObject) viewer;
  FILE        *fd;
  KSP         ksp;
  PC          pc;
  PCMethod    pcmethod;
  int         ierr;
  if (vobj->cookie == VIEWER_COOKIE && (vobj->type == FILE_VIEWER ||
                                        vobj->type == FILES_VIEWER)){
    fd = ViewerFileGetPointer_Private(viewer);
    SLESGetPC(sles,&pc);
    SLESGetKSP(sles,&ksp);
    PCGetMethodFromContext(pc,&pcmethod);
    if (pcmethod != PCLU) {
      ierr = KSPView(ksp,viewer); CHKERRQ(ierr);
    }
    ierr = PCView(pc,viewer); CHKERRQ(ierr);
  }
  return 0;
}

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
  MPIU_printf(sles->comm,"SLES options:\n");
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
  ierr = KSPCreate(comm,&sles->ksp); CHKERRQ(ierr);
  ierr = PCCreate(comm,&sles->pc); CHKERRQ(ierr);
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
  ierr = KSPDestroy(sles->ksp); CHKERRQ(ierr);
  ierr = PCDestroy(sles->pc); CHKERRQ(ierr);
  PLogObjectDestroy(sles);
  PETSCHEADERDESTROY(sles);
  return 0;
}
extern int PCPreSolve(PC,KSP),PCPostSolve(PC,KSP);
/*@
   SLESSetUp - Set up to solve a linear system.

   Input Parameters:
.  sles - the SLES context
.  b - the right hand side

   Output Parameters:
.  x - the approximate solution
.  its - the number of iterations used

.keywords: SLES, solve, linear system

.seealso: SLESCreate(), SLESDestroy(), SLESDestroy()
@*/
int SLESSetUp(SLES sles,Vec b,Vec x)
{
  int ierr;
  KSP ksp;
  PC  pc;
  VALIDHEADER(sles,SLES_COOKIE);
  PLogEventBegin(SLES_SetUp,sles,b,x,0);
  ksp = sles->ksp; pc = sles->pc;
  KSPSetRhs(ksp,b);
  KSPSetSolution(ksp,x);
  KSPSetBinv(ksp,pc);
  if (!sles->setupcalled) {
    ierr = PCSetVector(pc,b); CHKERRQ(ierr);
    ierr = KSPSetUp(sles->ksp); CHKERRQ(ierr);
    ierr = PCSetUp(sles->pc); CHKERRQ(ierr);
    sles->setupcalled = 1;
  }
  PLogEventEnd(SLES_SetUp,sles,b,x,0);
  return 0;
}
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
    ierr = SLESSetUp(sles,b,x); CHKERRQ(ierr);
  }
  ierr = PCPreSolve(pc,ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,its); CHKERRQ(ierr);
  ierr = PCPostSolve(pc,ksp); CHKERRQ(ierr);
  PLogEventEnd(SLES_Solve,sles,b,x,0);
  if (OptionsHasName(0,"-sles_view")) SLESView(sles,SYNC_STDOUT_VIEWER);
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
