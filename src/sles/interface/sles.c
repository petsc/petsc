
#include "slesimpl.h"
#include "options.h"

/*@
   SLESPrintHelp - Prints SLES options.

  Input Parameter:
.  sles - the solver context
@*/
int SLESPrintHelp(SLES sles)
{
  fprintf(stderr,"SLES options:\n");
  KSPPrintHelp(sles->ksp);
  PCPrintHelp(sles->pc);
  return 0;
}

/*@
    SLESSetFromOptions - Sets various SLES parameters from user options.

  Input Parameter:
.   sles - the linear equation solver context

  Options:
.  -slesiterative * use an iterative method
.  -slesdirect    * use a direct solver

  Also takes all KSP and PC options.
@*/
int SLESSetFromOptions(SLES sles)
{
  if (OptionsHasName(0,"-slesiterative")) {
    SLESSetSolverType(sles,SLES_ITERATIVE);
  }
  else if (OptionsHasName(0,"-slesdirect")) {
    SLESSetSolverType(sles,SLES_DIRECT);
  }
  KSPSetFromOptions(sles->ksp);
  PCSetFromOptions(sles->pc);
  return 0;
}
/*@
    SLESCreate - Creates a linear equation solver context

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
  sles->type        = SLES_DIRECT;
  sles->setupcalled = 0;
  *outsles = sles;
  return 0;
}


/*@
   SLESSolve - Solves a linear system.

  Input Parameters:
.   sles - the solver context
.   b,x - the right hand side and result
@*/
int SLESSolve(SLES sles,Vec b,Vec x)
{
  int ierr;
  if (sles->type == SLES_DIRECT) {
    if (!sles->setupcalled) {
      if (ierr = MatLUFactor(sles->mat)) SETERR(ierr,0);
    }
    if (ierr = MatSolve(sles->mat,b,x)) SETERR(ierr,0);
  }
  if (sles->type == SLES_ITERATIVE) {
    KSP ksp; PC pc;Mat mat; int its;
    SLESGetKSP(sles,&ksp);
    if (ierr = SLESGetPC(sles,&pc)) SETERR(ierr,0);
    KSPSetRhs(ksp,b);
    KSPSetSolution(ksp,x);
    KSPSetAmult(ksp,(int (*)(void *,Vec,Vec))MatMult,(void *)sles->mat);
    KSPSetBinv(ksp,(int (*)(void*,Vec,Vec))PCApply,(void*)pc);
    if (!sles->setupcalled) {

      if (ierr = PCSetVector(pc,b)) SETERR(ierr,0);
      if (ierr = PCGetMatrix(pc,&mat)) SETERR(ierr,0);
      if (!mat) {if (ierr = PCSetMatrix(pc,sles->mat)) SETERR(ierr,0);}
      if (ierr = KSPSetUp(sles->ksp)) SETERR(ierr,0);
      if (ierr = PCSetUp(sles->pc)) SETERR(ierr,0);
    }
    KSPSolve(ksp,&its);
printf("number of its %d\n",its);
  }
  sles->setupcalled = 1;
  return 0;
}

/*@
   SLESSetSolverType - Sets basic type of solver to use. Either a 
       direct solver, a preconditioned Krylov based solver or 
       a simple iterative solver like SOR. For iterative methods
       you will want SLES_ITERATIVE.

  Input Paramters:
.  sles - the solver context
.  type - either SLES_DIRECT, SLES_ITERATIVE.
@*/
int SLESSetSolverType(SLES sles,int type)
{
  VALIDHEADER(sles,SLES_COOKIE);
  sles->type = type;
  return 0;
}

/*@
    SLESGetKSP - returns the Krylov Space context for a sles solver.

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
    SLESGetPC - returns the preconditioner context for a sles solver.

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
    SLESSetMat - sets the matrix to use for the sles solver.

  Input Parameters:
.   sles - the sles context
.   mat - the matrix to use
@*/
int SLESSetMat(SLES sles,Mat mat)
{
  VALIDHEADER(sles,SLES_COOKIE);
  VALIDHEADER(mat,MAT_COOKIE);
  sles->mat = mat;
  return 0;
}




