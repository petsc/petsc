
#include "slesimpl.h"
#include "options.h"

/*@
   SLESPrintHelp - Prints SLES options.

  Input Parameter:
.  sles - the solver context
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
    SLESSetOptionsPrefix - Sets the prefix to use on all options setable from 
                           SLES.

  Input Parameter:
.   sles - the linear equation solver context
.   prefix - the prefix to prepend to all option names

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

  Input Parameter:
.   sles - the linear equation solver context

  Also takes all KSP and PC options.
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

  Output Parameter:
.   sles - the create context
@*/
int SLESCreate(SLES *outsles)
{
  int ierr;
  SLES sles;
  *outsles = 0;
  CREATEHEADER(sles,_SLES);
  if (ierr = KSPCreate(&sles->ksp)) SETERR(ierr,0);
  if (ierr = PCCreate(&sles->pc)) SETERR(ierr,0);
  sles->cookie      = SLES_COOKIE;
  sles->type        = 0;
  sles->setupcalled = 0;
  *outsles = sles;
  return 0;
}

/*@
   SLESDestroy - Destroys the SLES context.

  Input Parameters:
.   sles - the SLES context

  Keywords: sles, destroy
@*/
int SLESDestroy(SLES sles)
{
  int ierr;
  VALIDHEADER(sles,SLES_COOKIE);
  ierr = KSPDestroy(sles->ksp); CHKERR(ierr);
  ierr = PCDestroy(sles->pc); CHKERR(ierr);
  FREE(sles);
  return 0;
}

/*@
   SLESSolve - Solves a linear system.

  Input Parameters:
.   sles - the solver context
.   b - the right hand side

  Output Parameters:
.   x - the approximate solution
.   its - the number of iterations used.
@*/
int SLESSolve(SLES sles,Vec b,Vec x,int *its)
{
  int ierr;
  KSP ksp;
  PC  pc;
  Mat mat;
  VALIDHEADER(sles,SLES_COOKIE);
  ksp = sles->ksp; pc = sles->pc;
  KSPSetRhs(ksp,b);
  KSPSetSolution(ksp,x);
  KSPSetBinv(ksp,pc);
  if (!sles->setupcalled) {
    if (ierr = PCSetVector(pc,b)) SETERR(ierr,0);
    if (ierr = KSPSetUp(sles->ksp)) SETERR(ierr,0);
    if (ierr = PCSetUp(sles->pc)) SETERR(ierr,0);
    sles->setupcalled = 1;
  }
  ierr = PCPreSolve(pc,ksp); CHKERR(ierr);
  ierr = KSPSolve(ksp,its); CHKERR(ierr);
  ierr = PCPostSolve(pc,ksp); CHKERR(ierr);
  return 0;
}

/*@
    SLESGetKSP - Returns the Krylov Space context for a sles solver.

  Input Parameter:
.   sles - the solver context

  Output Paramter:
.   ksp - the Krylov space context
@*/
int SLESGetKSP(SLES sles,KSP *ksp)
{
  VALIDHEADER(sles,SLES_COOKIE);
  *ksp = sles->ksp;
  return 0;
}
/*@
    SLESGetPC - Returns the preconditioner context for a sles solver.

  Input Parameter:
.   sles - the solver context

  Output Paramter:
.  pc - the preconditioner context
@*/
int SLESGetPC(SLES sles,PC *pc)
{
  VALIDHEADER(sles,SLES_COOKIE);
  *pc = sles->pc;
  return 0;
}

#include "mat/matimpl.h"
/*@
    SLESSetMat - Sets the matrix to use for the sles solver.

  Input Parameters:
.   sles - the sles context
.   mat - the matrix to use
@*/
int SLESSetMat(SLES sles,Mat mat)
{
  VALIDHEADER(sles,SLES_COOKIE);
  VALIDHEADER(mat,MAT_COOKIE);
  PCSetMat(sles->pc,mat);
  return 0;
}




