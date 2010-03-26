#define PETSCKSP_DLL

/*
    Code for setting KSP options from the options database.
*/

#include "private/kspimpl.h"  /*I "petscksp.h" I*/

/*
       We retain a list of functions that also take KSP command 
    line options. These are called at the end KSPSetFromOptions()
*/
#define MAXSETFROMOPTIONS 5
PetscInt numberofsetfromoptions = 0;
PetscErrorCode (*othersetfromoptions[MAXSETFROMOPTIONS])(KSP) = {0};

extern PetscTruth KSPRegisterAllCalled;

#undef __FUNCT__  
#define __FUNCT__ "KSPAddOptionsChecker"
/*@C
    KSPAddOptionsChecker - Adds an additional function to check for KSP options.

    Not Collective

    Input Parameter:
.   kspcheck - function that checks for options

    Level: developer

.keywords: KSP, add, options, checker

.seealso: KSPSetFromOptions()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPAddOptionsChecker(PetscErrorCode (*kspcheck)(KSP))
{
  PetscFunctionBegin;
  if (numberofsetfromoptions >= MAXSETFROMOPTIONS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many options checkers, only 5 allowed");
  }

  othersetfromoptions[numberofsetfromoptions++] = kspcheck;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetOptionsPrefix"
/*@C
   KSPSetOptionsPrefix - Sets the prefix used for searching for all 
   KSP options in the database.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
-  prefix - the prefix string to prepend to all KSP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   For example, to distinguish between the runtime options for two
   different KSP contexts, one could call
.vb
      KSPSetOptionsPrefix(ksp1,"sys1_")
      KSPSetOptionsPrefix(ksp2,"sys2_")
.ve

   This would enable use of different options for each system, such as
.vb
      -sys1_ksp_type gmres -sys1_ksp_rtol 1.e-3
      -sys2_ksp_type bcgs  -sys2_ksp_rtol 1.e-4
.ve

   Level: advanced

.keywords: KSP, set, options, prefix, database

.seealso: KSPAppendOptionsPrefix(), KSPGetOptionsPrefix()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetOptionsPrefix(KSP ksp,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCSetOptionsPrefix(ksp->pc,prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}
 
#undef __FUNCT__  
#define __FUNCT__ "KSPAppendOptionsPrefix"
/*@C
   KSPAppendOptionsPrefix - Appends to the prefix used for searching for all 
   KSP options in the database.

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
-  prefix - the prefix string to prepend to all KSP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.keywords: KSP, append, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPGetOptionsPrefix()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPAppendOptionsPrefix(KSP ksp,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCAppendOptionsPrefix(ksp->pc,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetUseFischerGuess"
/*@C
   KSPSetUseFischerGuess - Use the Paul Fischer algorithm, see KSPFischerGuessCreate()

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
.  model - use model 1, model 2 or 0 to turn it off
-  size - size of subspace used to generate initial guess

    Options Database:
.   -ksp_fischer_guess <model,size> - uses the Fischer initial guess generator for repeated linear solves

   Level: advanced

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPSetFischerGuess(), KSPGetFischerInitialGuess()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetUseFischerGuess(KSP ksp,PetscInt model,PetscInt size)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (ksp->guess) {
    ierr = KSPFischerGuessDestroy(ksp->guess);CHKERRQ(ierr);
    ksp->guess = PETSC_NULL;
  }
  if (model == 1 || model == 2) {
    ierr = KSPFischerGuessCreate(ksp,model,size,&ksp->guess);CHKERRQ(ierr);
    ierr = KSPFischerGuessSetFromOptions(ksp->guess);CHKERRQ(ierr);
  } else if (model != 0) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Model must be 1 or 2 (or 0 to turn off guess generation)");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFischerGuess"
/*@C
   KSPSetFischerGuess - Use the Paul Fischer algorithm created by KSPFischerGuessCreate()

   Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
-  guess - the object created with KSPFischerGuessCreate()

   Level: advanced

   Notes: this allows a single KSP to be used with several different initial guess generators (likely for different linear
          solvers, see KSPSetPC()).

          This increases the reference count of the guess object, you must destroy the object with KSPFischerGuessDestroy()
          before the end of the program.

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPSetFischerGuess(), KSPGetFischerGuess()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetFischerGuess(KSP ksp,KSPFischerGuess guess)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (ksp->guess) {
    ierr = KSPFischerGuessDestroy(ksp->guess);CHKERRQ(ierr);
  }
  ksp->guess = guess;
  if (guess) guess->refcnt++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetFischerGuess"
/*@C
   KSPGetFischerGuess - Gets the initial guess generator set with either KSPSetFischerGuess() or KSPCreateFischerGuess()/KSPSetFischerGuess()

   Collective on KSP

   Input Parameters:
.  ksp - the Krylov context

   Output Parameters:
.   guess - the object

   Level: developer

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPSetFischerGuess()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGetFischerGuess(KSP ksp,KSPFischerGuess *guess)
{
  PetscFunctionBegin;
  *guess = ksp->guess;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPGetOptionsPrefix"
/*@C
   KSPGetOptionsPrefix - Gets the prefix used for searching for all 
   KSP options in the database.

   Not Collective

   Input Parameters:
.  ksp - the Krylov context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prifix' of
   sufficient length to hold the prefix.

   Level: advanced

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGetOptionsPrefix(KSP ksp,const char *prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions"
/*@
   KSPSetFromOptions - Sets KSP options from the options database.
   This routine must be called before KSPSetUp() if the user is to be 
   allowed to set the Krylov type. 

   Collective on KSP

   Input Parameters:
.  ksp - the Krylov space context

   Options Database Keys:
+   -ksp_max_it - maximum number of linear iterations
.   -ksp_rtol rtol - relative tolerance used in default determination of convergence, i.e.
                if residual norm decreases by this factor than convergence is declared
.   -ksp_atol abstol - absolute tolerance used in default convergence test, i.e. if residual 
                norm is less than this then convergence is declared
.   -ksp_divtol tol - if residual norm increases by this factor than divergence is declared
.   -ksp_converged_use_initial_residual_norm - see KSPDefaultConvergedSetUIRNorm()
.   -ksp_converged_use_min_initial_residual_norm - see KSPDefaultConvergedSetUMIRNorm()
.   -ksp_norm_type - none - skip norms used in convergence tests (useful only when not using 
$                       convergence test (say you always want to run with 5 iterations) to 
$                       save on communication overhead
$                    preconditioned - default for left preconditioning 
$                    unpreconditioned - see KSPSetNormType()
$                    natural - see KSPSetNormType()
.   -ksp_check_norm_iteration it - do not compute residual norm until iteration number it (does compute at 0th iteration)
$       works only for PCBCGS, PCIBCGS and and PCCG
.   -ksp_fischer_guess <model,size> - uses the Fischer initial guess generator for repeated linear solves
.   -ksp_constant_null_space - assume the operator (matrix) has the constant vector in its null space
.   -ksp_test_null_space - tests the null space set with KSPSetNullSpace() to see if it truly is a null space
.   -ksp_knoll - compute initial guess by applying the preconditioner to the right hand side
.   -ksp_monitor_cancel - cancel all previous convergene monitor routines set
.   -ksp_monitor <optional filename> - print residual norm at each iteration
.   -ksp_monitor_draw - plot residual norm at each iteration
.   -ksp_monitor_solution - plot solution at each iteration
-   -ksp_monitor_singular_value - monitor extremem singular values at each iteration

   Notes:  
   To see all options, run your program with the -help option
   or consult the users manual.

   Level: beginner

.keywords: KSP, set, from, options, database

.seealso: KSPSetUseFischerInitialGuess()

@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPSetFromOptions(KSP ksp)
{
  PetscErrorCode          ierr;
  PetscInt                indx;
  const char             *convtests[] = {"default","skip"};
  char                    type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscViewerASCIIMonitor monviewer;
  PetscTruth              flg,flag;
  PetscInt                i,model[2],nmax = 2;
  void                    *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCSetFromOptions(ksp->pc);CHKERRQ(ierr);

  if (!KSPRegisterAllCalled) {ierr = KSPRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsBegin(((PetscObject)ksp)->comm,((PetscObject)ksp)->prefix,"Krylov Method (KSP) Options","KSP");CHKERRQ(ierr);
    ierr = PetscOptionsList("-ksp_type","Krylov method","KSPSetType",KSPList,(char*)(((PetscObject)ksp)->type_name?((PetscObject)ksp)->type_name:KSPGMRES),type,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetType(ksp,type);CHKERRQ(ierr);
    }
    /*
      Set the type if it was never set.
    */
    if (!((PetscObject)ksp)->type_name) {
      ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
    }

    ierr = PetscOptionsInt("-ksp_max_it","Maximum number of iterations","KSPSetTolerances",ksp->max_it,&ksp->max_it,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ksp_rtol","Relative decrease in residual norm","KSPSetTolerances",ksp->rtol,&ksp->rtol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ksp_atol","Absolute value of residual norm","KSPSetTolerances",ksp->abstol,&ksp->abstol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-ksp_divtol","Residual norm increase cause divergence","KSPSetTolerances",ksp->divtol,&ksp->divtol,PETSC_NULL);CHKERRQ(ierr);

    flag = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_converged_use_initial_residual_norm","Use initial residual residual norm for computing relative convergence","KSPDefaultConvergedSetUIRNorm",flag,&flag,PETSC_NULL);CHKERRQ(ierr);
    if (flag) {ierr = KSPDefaultConvergedSetUIRNorm(ksp);CHKERRQ(ierr);}
    flag = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_converged_use_min_initial_residual_norm","Use minimum of initial residual norm and b for computing relative convergence","KSPDefaultConvergedSetUMIRNorm",flag,&flag,PETSC_NULL);CHKERRQ(ierr);
    if (flag) {ierr = KSPDefaultConvergedSetUMIRNorm(ksp);CHKERRQ(ierr);}
    ierr = KSPGetInitialGuessNonzero(ksp,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ksp_initial_guess_nonzero","Use the contents of the solution vector for initial guess","KSPSetInitialNonzero",flag,&flag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetInitialGuessNonzero(ksp,flag);CHKERRQ(ierr);
    }

    ierr = PetscOptionsTruth("-ksp_knoll","Use preconditioner applied to b for initial guess","KSPSetInitialGuessKnoll",ksp->guess_knoll,&ksp->guess_knoll,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-ksp_fischer_guess","Use Paul Fischer's algorihtm for initial guess","KSPSetUseFischerGuess",model,&nmax,&flag);CHKERRQ(ierr);
    if (flag) {
      if (nmax != 2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Must pass in model,size as arguments");
      ierr = KSPSetUseFischerGuess(ksp,model[0],model[1]);CHKERRQ(ierr);
    }

    ierr = PetscOptionsEList("-ksp_convergence_test","Convergence test","KSPSetConvergenceTest",convtests,2,"default",&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      switch (indx) {
      case 0: 
        ierr = KSPDefaultConvergedCreate(&ctx);CHKERRQ(ierr);
        ierr = KSPSetConvergenceTest(ksp,KSPDefaultConverged,ctx,KSPDefaultConvergedDestroy);CHKERRQ(ierr); 
        break;
      case 1: ierr = KSPSetConvergenceTest(ksp,KSPSkipConverged,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);    break;
      }
    }

    ierr = PetscOptionsEList("-ksp_norm_type","KSP Norm type","KSPSetNormType",KSPNormTypes,4,"preconditioned",&indx,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetNormType(ksp,(KSPNormType)indx);CHKERRQ(ierr); }

    ierr = PetscOptionsInt("-ksp_check_norm_iteration","First iteration to compute residual norm","KSPSetCheckNormIteration",ksp->chknorm,&ksp->chknorm,PETSC_NULL);CHKERRQ(ierr);

    flag  = ksp->lagnorm;
    ierr = PetscOptionsTruth("-ksp_lag_norm","Lag the calculation of the residual norm","KSPSetLagNorm",flag,&flag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetLagNorm(ksp,flag);CHKERRQ(ierr);
    }

    ierr = KSPGetDiagonalScale(ksp,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ksp_diagonal_scale","Diagonal scale matrix before building preconditioner","KSPSetDiagonalScale",flag,&flag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetDiagonalScale(ksp,flag);CHKERRQ(ierr);
    }
    ierr = KSPGetDiagonalScaleFix(ksp,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-ksp_diagonal_scale_fix","Fix diagonally scaled matrix after solve","KSPSetDiagonalScaleFix",flag,&flag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetDiagonalScaleFix(ksp,flag);CHKERRQ(ierr);
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_constant_null_space","Add constant null space to Krylov solver","KSPSetNullSpace",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      MatNullSpace nsp;

      ierr = MatNullSpaceCreate(((PetscObject)ksp)->comm,PETSC_TRUE,0,0,&nsp);CHKERRQ(ierr);
      ierr = KSPSetNullSpace(ksp,nsp);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(nsp);CHKERRQ(ierr);
    }

    /* option is actually checked in KSPSetUp(), just here so goes into help message */
    if (ksp->nullsp) {
      ierr = PetscOptionsName("-ksp_test_null_space","Is provided null space correct","None",&flg);CHKERRQ(ierr);
    }

    /*
      Prints reason for convergence or divergence of each linear solve
    */
    flg = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_converged_reason","Print reason for converged or diverged","KSPSolve",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ksp->printreason = PETSC_TRUE;
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_monitor_cancel","Remove any hardwired monitor routines","KSPMonitorCancel",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    /* -----------------------------------------------------------------------*/
    /*
      Cancels all monitors hardwired into code before call to KSPSetFromOptions()
    */
    if (flg) {
      ierr = KSPMonitorCancel(ksp);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned residual norm at each iteration
    */
    ierr = PetscOptionsString("-ksp_monitor","Monitor preconditioned residual norm","KSPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)ksp)->comm,monfilename,((PetscObject)ksp)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorDefault,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned residual norm at each iteration
    */
    ierr = PetscOptionsString("-ksp_monitor_range","Monitor percent of residual entries more than 10 percent of max","KSPMonitorRange","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)ksp)->comm,monfilename,((PetscObject)ksp)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorRange,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    /*
      Plots the vector solution 
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_monitor_solution","Monitor solution graphically","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPMonitorSet(ksp,KSPMonitorSolution,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned and true residual norm at each iteration
    */
    ierr = PetscOptionsString("-ksp_monitor_true_residual","Monitor true residual norm","KSPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)ksp)->comm,monfilename,((PetscObject)ksp)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorTrueResidualNorm,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    /*
      Prints extreme eigenvalue estimates at each iteration
    */
    ierr = PetscOptionsString("-ksp_monitor_singular_value","Monitor singular values","KSPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)ksp)->comm,monfilename,((PetscObject)ksp)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorSingularValue,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned residual norm with fewer digits
    */
    ierr = PetscOptionsString("-ksp_monitor_short","Monitor preconditioned residual norm with fewer digits","KSPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIMonitorCreate(((PetscObject)ksp)->comm,monfilename,((PetscObject)ksp)->tablevel,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorDefaultShort,monviewer,(PetscErrorCode (*)(void*))PetscViewerASCIIMonitorDestroy);CHKERRQ(ierr);
    }
    /*
      Graphically plots preconditioned residual norm
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_monitor_draw","Monitor graphically preconditioned residual norm","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPMonitorSet(ksp,KSPMonitorLG,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    /*
      Graphically plots preconditioned and true residual norm
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_monitor_draw_true_residual","Monitor graphically true residual norm","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg){
      ierr = KSPMonitorSet(ksp,KSPMonitorLGTrueResidualNorm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    /*
      Graphically plots preconditioned residual norm and range of residual element values
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_monitor_range_draw","Monitor graphically preconditioned residual norm","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPMonitorSet(ksp,KSPMonitorLGRange,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }

    /* -----------------------------------------------------------------------*/

    ierr = PetscOptionsTruthGroupBegin("-ksp_left_pc","Use left preconditioning","KSPSetPreconditionerSide",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_LEFT);CHKERRQ(ierr); }
    ierr = PetscOptionsTruthGroup("-ksp_right_pc","Use right preconditioning","KSPSetPreconditionerSide",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_RIGHT);CHKERRQ(ierr);}
    ierr = PetscOptionsTruthGroupEnd("-ksp_symmetric_pc","Use symmetric (factorized) preconditioning","KSPSetPreconditionerSide",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetPreconditionerSide(ksp,PC_SYMMETRIC);CHKERRQ(ierr);}

    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_compute_singularvalues","Compute singular values of preconditioned operator","KSPSetComputeSingularValues",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_compute_eigenvalues","Compute eigenvalues of preconditioned operator","KSPSetComputeSingularValues",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-ksp_plot_eigenvalues","Scatter plot extreme eigenvalues","KSPSetComputeSingularValues",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }

    for(i = 0; i < numberofsetfromoptions; i++) {
      ierr = (*othersetfromoptions[i])(ksp);CHKERRQ(ierr);
    }

    if (ksp->ops->setfromoptions) {
      ierr = (*ksp->ops->setfromoptions)(ksp);CHKERRQ(ierr);
    }
    /* actually check in setup this is just here so goes into help message */
    ierr = PetscOptionsName("-ksp_view","View linear solver parameters","KSPView",&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
