#ifndef lint
static char vcid[] = "$Id: sles.c,v 1.49 1996/01/15 21:54:31 balay Exp bsmith $";
#endif

#include "slesimpl.h"     /*I  "sles.h"    I*/
#include "pinclude/pviewer.h"

/*@ 
   SLESView - Prints the SLES data structure.

   Input Parameters:
.  SLES - the SLES context
.  viewer - optional visualization context

   Options Database Key:
$  -sles_view : calls SLESView() at end of SLESSolve()

   Note:
   The available visualization contexts include
$     STDOUT_VIEWER_SELF - standard output (default)
$     STDOUT_VIEWER_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file

.keywords: SLES, view

.seealso: ViewerFileOpenASCII()
@*/
int SLESView(SLES sles,Viewer viewer)
{
  PetscObject vobj = (PetscObject) viewer;
  FILE        *fd;
  KSP         ksp;
  PC          pc;
  PCType      pcmethod;
  int         ierr;
  if (vobj->cookie == VIEWER_COOKIE && (vobj->type == ASCII_FILE_VIEWER ||
                                        vobj->type == ASCII_FILES_VIEWER)){
    ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
    SLESGetPC(sles,&pc);
    SLESGetKSP(sles,&ksp);
    PCGetType(pc,&pcmethod,PETSC_NULL);
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
  char    *prefix = "-";
  if (sles->prefix) prefix = sles->prefix;
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
  MPIU_printf(sles->comm,"SLES options:\n");
  MPIU_printf(sles->comm," %ssles_view: view SLES info after each linear solve\n",prefix);
  KSPPrintHelp(sles->ksp);
  PCPrintHelp(sles->pc);
  return 0;
}

/*@C
   SLESSetOptionsPrefix - Sets the prefix used for searching for all 
   SLES options in the database.

   Input Parameter:
.  sles - the SLES context
.  prefix - the prefix to prepend to all option names

   Notes:
   This prefix is particularly useful for nested use of SLES.  For
   example, the block Jacobi and block diagonal preconditioners use
   the prefix "sub" for options relating to the individual blocks.  

.keywords: SLES, set, options, prefix, database
@*/
int SLESSetOptionsPrefix(SLES sles,char *prefix)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
  ierr = PetscObjectSetPrefix((PetscObject)sles, prefix); CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(sles->ksp,prefix);CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(sles->pc,prefix); CHKERRQ(ierr);
  return 0;
}
/*@C
   SLESAppendOptionsPrefix - Appends to the prefix used for searching for all 
   SLES options in the database.

   Input Parameter:
.  sles - the SLES context
.  prefix - the prefix to prepend to all option names

   Notes:
   This prefix is particularly useful for nested use of SLES.  For
   example, the block Jacobi and block diagonal preconditioners use
   the prefix "sub" for options relating to the individual blocks.  

.keywords: SLES, append, options, prefix, database
@*/
int SLESAppendOptionsPrefix(SLES sles,char *prefix)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
  ierr = PetscObjectAppendPrefix((PetscObject)sles, prefix); CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(sles->ksp,prefix);CHKERRQ(ierr);
  ierr = PCAppendOptionsPrefix(sles->pc,prefix); CHKERRQ(ierr);
  return 0;
}
/*@
   SLESGetOptionsPrefix - Gets the prefix used for searching for all 
   SLES options in the database.

   Input Parameter:
.  sles - the SLES context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes:
   This prefix is particularly useful for nested use of SLES.  For
   example, the block Jacobi and block diagonal preconditioners use
   the prefix "sub" for options relating to the individual blocks.  

.keywords: SLES, get, options, prefix, database
@*/
int SLESGetOptionsPrefix(SLES sles,char **prefix)
{
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
  return PetscObjectGetPrefix((PetscObject)sles, prefix); 
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
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
  KSPSetFromOptions(sles->ksp);
  PCSetFromOptions(sles->pc);
  return 0;
}
/*@C
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
  PetscHeaderCreate(sles,_SLES,SLES_COOKIE,0,comm);
  PLogObjectCreate(sles);
  ierr = KSPCreate(comm,&sles->ksp); CHKERRQ(ierr);
  ierr = PCCreate(comm,&sles->pc); CHKERRQ(ierr);
  PLogObjectParent(sles,sles->ksp);
  PLogObjectParent(sles,sles->pc);
  sles->setupcalled = 0;
  *outsles = sles;
  return 0;
}

/*@C
   SLESDestroy - Destroys the SLES context.

   Input Parameters:
.  sles - the SLES context

.keywords: SLES, destroy, context

.seealso: SLESCreate(), SLESSolve()
@*/
int SLESDestroy(SLES sles)
{
  int ierr;
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
  ierr = KSPDestroy(sles->ksp); CHKERRQ(ierr);
  ierr = PCDestroy(sles->pc); CHKERRQ(ierr);
  PLogObjectDestroy(sles);
  PetscHeaderDestroy(sles);
  return 0;
}
extern int PCPreSolve(PC,KSP),PCPostSolve(PC,KSP);
/*@
   SLESSetUp - Performs set up required for solving a linear system.

   Input Parameters:
.  sles - the SLES context
.  b - the right hand side

   Output Parameters:
.  x - the approximate solution

   Note:
   For basic use of the SLES solvers the user need not explicitly call
   SLESSetUp(), since these actions will automatically occur during
   the call to SLESSolve().  However, if one wishes to generate
   performance data for this computational phase (for example, for
   incomplete factorization using the ILU preconditioner) using the 
   PETSc log facilities, calling SLESSetUp() is required.

.keywords: SLES, solve, linear system

.seealso: SLESCreate(), SLESDestroy(), SLESDestroy()
@*/
int SLESSetUp(SLES sles,Vec b,Vec x)
{
  int ierr;
  KSP ksp;
  PC  pc;
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
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
  int ierr, flg;
  KSP ksp;
  PC  pc;

  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
  if (b == x) SETERRQ(1,"SLESSolve:b and x must be different vectors");
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
  ierr = OptionsHasName(sles->prefix,"-sles_view", &flg); CHKERRQ(ierr); 
  if (flg) { ierr = SLESView(sles,STDOUT_VIEWER_WORLD); CHKERRQ(ierr); }
  return 0;
}

/*@C
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
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
  *ksp = sles->ksp;
  return 0;
}
/*@C
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
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
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
  PETSCVALIDHEADERSPECIFIC(sles,SLES_COOKIE);
  PETSCVALIDHEADERSPECIFIC(Amat,MAT_COOKIE);
  if (Pmat) {PETSCVALIDHEADERSPECIFIC(Pmat,MAT_COOKIE);}
  PCSetOperators(sles->pc,Amat,Pmat,flag);
  sles->setupcalled = 0;  /* so that next solve call will call setup */
  return 0;
}
