
/*
    Code for setting KSP options from the options database.
*/

#include <petsc-private/kspimpl.h>  /*I "petscksp.h" I*/

extern PetscBool  KSPRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "KSPSetOptionsPrefix"
/*@C
   KSPSetOptionsPrefix - Sets the prefix used for searching for all
   KSP options in the database.

   Logically Collective on KSP

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
PetscErrorCode  KSPSetOptionsPrefix(KSP ksp,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
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

   Logically Collective on KSP

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
PetscErrorCode  KSPAppendOptionsPrefix(KSP ksp,const char prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCAppendOptionsPrefix(ksp->pc,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)ksp,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGetTabLevel"
/*@
   KSPGetTabLevel - Gets the number of tabs that ASCII output used by ksp.

   Not Collective

   Input Parameter:
.  ksp - a KSP object.

   Output Parameter:
.   tab - the number of tabs

   Level: developer

    Notes: this is used in conjunction with KSPSetTabLevel() to manage the output from the KSP and its PC coherently.


.seealso:  KSPSetTabLevel()

@*/
PetscErrorCode  KSPGetTabLevel(KSP ksp,PetscInt *tab)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscObjectGetTabLevel((PetscObject)ksp, tab); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetTabLevel"
/*@
   KSPSetTabLevel - Sets the number of tabs that ASCII output for the ksp andn its pc will use.

   Not Collective

   Input Parameters:
+  ksp - a KSP object
-  tab - the number of tabs

   Level: developer

    Notes: this is used to manage the output from KSP and PC objects that are imbedded in other objects,
           for example, the KSP object inside a SNES object. By indenting each lower level further the heirarchy
           of objects is very clear.  By setting the KSP object's tab level with KSPSetTabLevel() its PC object
           automatically receives the same tab level, so that whatever objects the pc might create are tabbed
           appropriately, too.

.seealso:  KSPGetTabLevel()
@*/
PetscErrorCode  KSPSetTabLevel(KSP ksp, PetscInt tab)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = PetscObjectSetTabLevel((PetscObject)ksp, tab);              CHKERRQ(ierr);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);                      CHKERRQ(ierr);}
  /* Do we need a PCSetTabLevel()? */
  ierr = PetscObjectSetTabLevel((PetscObject)ksp->pc, tab);          CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetUseFischerGuess"
/*@C
   KSPSetUseFischerGuess - Use the Paul Fischer algorithm, see KSPFischerGuessCreate()

   Logically Collective on KSP

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
PetscErrorCode  KSPSetUseFischerGuess(KSP ksp,PetscInt model,PetscInt size)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,model,2);
  PetscValidLogicalCollectiveInt(ksp,model,3);
  ierr = KSPFischerGuessDestroy(&ksp->guess);CHKERRQ(ierr);
  if (model == 1 || model == 2) {
    ierr = KSPFischerGuessCreate(ksp,model,size,&ksp->guess);CHKERRQ(ierr);
    ierr = KSPFischerGuessSetFromOptions(ksp->guess);CHKERRQ(ierr);
  } else if (model != 0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Model must be 1 or 2 (or 0 to turn off guess generation)");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFischerGuess"
/*@C
   KSPSetFischerGuess - Use the Paul Fischer algorithm created by KSPFischerGuessCreate()

   Logically Collective on KSP

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
PetscErrorCode  KSPSetFischerGuess(KSP ksp,KSPFischerGuess guess)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  ierr = KSPFischerGuessDestroy(&ksp->guess);CHKERRQ(ierr);
  ksp->guess = guess;
  if (guess) guess->refcnt++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPGetFischerGuess"
/*@C
   KSPGetFischerGuess - Gets the initial guess generator set with either KSPSetFischerGuess() or KSPCreateFischerGuess()/KSPSetFischerGuess()

   Not Collective

   Input Parameters:
.  ksp - the Krylov context

   Output Parameters:
.   guess - the object

   Level: developer

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPSetFischerGuess()
@*/
PetscErrorCode  KSPGetFischerGuess(KSP ksp,KSPFischerGuess *guess)
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
PetscErrorCode  KSPGetOptionsPrefix(KSP ksp,const char *prefix[])
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
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
    -ksp_lag_norm - compute the norm of the residual for the ith iteration on the i+1 iteration; this means that one can use
$       the norm of the residual for convergence test WITHOUT an extra MPI_Allreduce() limiting global synchronizations.
$       This will require 1 more iteration of the solver than usual.
.   -ksp_fischer_guess <model,size> - uses the Fischer initial guess generator for repeated linear solves
.   -ksp_constant_null_space - assume the operator (matrix) has the constant vector in its null space
.   -ksp_test_null_space - tests the null space set with KSPSetNullSpace() to see if it truly is a null space
.   -ksp_knoll - compute initial guess by applying the preconditioner to the right hand side
.   -ksp_monitor_cancel - cancel all previous convergene monitor routines set
.   -ksp_monitor <optional filename> - print residual norm at each iteration
.   -ksp_monitor_lg_residualnorm - plot residual norm at each iteration
.   -ksp_monitor_solution - plot solution at each iteration
-   -ksp_monitor_singular_value - monitor extremem singular values at each iteration

   Notes:
   To see all options, run your program with the -help option
   or consult <A href="../../docs/manual.pdf#nameddest=ch_ksp">KSP chapter of the users manual</A>.

   Level: beginner

.keywords: KSP, set, from, options, database

.seealso: KSPSetUseFischerInitialGuess()

@*/
PetscErrorCode  KSPSetFromOptions(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       indx;
  const char     *convtests[] = {"default","skip"};
  char           type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer    monviewer;
  PetscBool      flg,flag;
  PetscInt       model[2]={0,0},nmax;
  KSPNormType    normtype;
  PCSide         pcside;
  void           *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  ierr = PCSetFromOptions(ksp->pc);CHKERRQ(ierr);

  if (!KSPRegisterAllCalled) {ierr = KSPRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscObjectOptionsBegin((PetscObject)ksp);CHKERRQ(ierr);
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
    ierr = PetscOptionsBool("-ksp_converged_use_initial_residual_norm","Use initial residual residual norm for computing relative convergence","KSPDefaultConvergedSetUIRNorm",flag,&flag,PETSC_NULL);CHKERRQ(ierr);
    if (flag) {ierr = KSPDefaultConvergedSetUIRNorm(ksp);CHKERRQ(ierr);}
    flag = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_converged_use_min_initial_residual_norm","Use minimum of initial residual norm and b for computing relative convergence","KSPDefaultConvergedSetUMIRNorm",flag,&flag,PETSC_NULL);CHKERRQ(ierr);
    if (flag) {ierr = KSPDefaultConvergedSetUMIRNorm(ksp);CHKERRQ(ierr);}
    ierr = KSPGetInitialGuessNonzero(ksp,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ksp_initial_guess_nonzero","Use the contents of the solution vector for initial guess","KSPSetInitialNonzero",flag,&flag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetInitialGuessNonzero(ksp,flag);CHKERRQ(ierr);
    }

    ierr = PetscOptionsBool("-ksp_knoll","Use preconditioner applied to b for initial guess","KSPSetInitialGuessKnoll",ksp->guess_knoll,&ksp->guess_knoll,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ksp_error_if_not_converged","Generate error if solver does not converge","KSPSetErrorIfNotConverged",ksp->errorifnotconverged,&ksp->errorifnotconverged,PETSC_NULL);CHKERRQ(ierr);
    nmax = 2;
    ierr = PetscOptionsIntArray("-ksp_fischer_guess","Use Paul Fischer's algorithm for initial guess","KSPSetUseFischerGuess",model,&nmax,&flag);CHKERRQ(ierr);
    if (flag) {
      if (nmax != 2) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Must pass in model,size as arguments");
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

    ierr = KSPSetUpNorms_Private(ksp,&normtype,&pcside);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-ksp_norm_type","KSP Norm type","KSPSetNormType",KSPNormTypes,(PetscEnum)normtype,(PetscEnum*)&normtype,&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetNormType(ksp,normtype);CHKERRQ(ierr); }

    ierr = PetscOptionsInt("-ksp_check_norm_iteration","First iteration to compute residual norm","KSPSetCheckNormIteration",ksp->chknorm,&ksp->chknorm,PETSC_NULL);CHKERRQ(ierr);

    flag  = ksp->lagnorm;
    ierr = PetscOptionsBool("-ksp_lag_norm","Lag the calculation of the residual norm","KSPSetLagNorm",flag,&flag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetLagNorm(ksp,flag);CHKERRQ(ierr);
    }

    ierr = KSPGetDiagonalScale(ksp,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ksp_diagonal_scale","Diagonal scale matrix before building preconditioner","KSPSetDiagonalScale",flag,&flag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetDiagonalScale(ksp,flag);CHKERRQ(ierr);
    }
    ierr = KSPGetDiagonalScaleFix(ksp,&flag);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ksp_diagonal_scale_fix","Fix diagonally scaled matrix after solve","KSPSetDiagonalScaleFix",flag,&flag,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetDiagonalScaleFix(ksp,flag);CHKERRQ(ierr);
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_constant_null_space","Add constant null space to Krylov solver","KSPSetNullSpace",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      MatNullSpace nsp;

      ierr = MatNullSpaceCreate(((PetscObject)ksp)->comm,PETSC_TRUE,0,0,&nsp);CHKERRQ(ierr);
      ierr = KSPSetNullSpace(ksp,nsp);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
    }

    /* option is actually checked in KSPSetUp(), just here so goes into help message */
    if (ksp->nullsp) {
      ierr = PetscOptionsName("-ksp_test_null_space","Is provided null space correct","None",&flg);CHKERRQ(ierr);
    }

    /*
      Prints reason for convergence or divergence of each linear solve
    */
    flg = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_converged_reason","Print reason for converged or diverged","KSPSolve",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ksp->printreason = PETSC_TRUE;
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_monitor_cancel","Remove any hardwired monitor routines","KSPMonitorCancel",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
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
      ierr = PetscViewerASCIIOpen(((PetscObject)ksp)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorDefault,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned residual norm at each iteration
    */
    ierr = PetscOptionsString("-ksp_monitor_range","Monitor percent of residual entries more than 10 percent of max","KSPMonitorRange","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(((PetscObject)ksp)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorRange,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
      A hack to using dynamic tolerance in preconditioner
    */
    ierr = PetscOptionsString("-sub_ksp_dynamic_tolerance","Use dynamic tolerance for PC if PC is a KSP","KSPMonitorDynamicTolerance","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      KSPDynTolCtx *scale = NULL;
      PetscReal    defaultv = 1.0;
      ierr = PetscMalloc(1*sizeof(KSPDynTolCtx),&scale);
      scale->bnrm = -1.0;
      scale->coef = defaultv;
      ierr = PetscOptionsReal("-sub_ksp_dynamic_tolerance_param","Parameter of dynamic tolerance for PC if PC is a KSP","KSPMonitorDynamicToleranceParam",defaultv,&(scale->coef),&flg);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorDynamicTolerance,scale,KSPMonitorDynamicToleranceDestroy);CHKERRQ(ierr);
    }
    /*
      Plots the vector solution
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_monitor_solution","Monitor solution graphically","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPMonitorSet(ksp,KSPMonitorSolution,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned and true residual norm at each iteration
    */
    ierr = PetscOptionsString("-ksp_monitor_true_residual","Monitor true residual norm","KSPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(((PetscObject)ksp)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorTrueResidualNorm,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
      Prints with max norm at each iteration
    */
    ierr = PetscOptionsString("-ksp_monitor_max","Monitor true residual max norm","KSPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(((PetscObject)ksp)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorTrueResidualMaxNorm,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
      Prints extreme eigenvalue estimates at each iteration
    */
    ierr = PetscOptionsString("-ksp_monitor_singular_value","Monitor singular values","KSPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(((PetscObject)ksp)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorSingularValue,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
      Prints preconditioned residual norm with fewer digits
    */
    ierr = PetscOptionsString("-ksp_monitor_short","Monitor preconditioned residual norm with fewer digits","KSPMonitorSet","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscViewerASCIIOpen(((PetscObject)ksp)->comm,monfilename,&monviewer);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorDefaultShort,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }
    /*
     Calls Python function
    */
    ierr = PetscOptionsString("-ksp_monitor_python","Use Python function","KSPMonitorSet",0,monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PetscPythonMonitorSet((PetscObject)ksp,monfilename);CHKERRQ(ierr);}
    /*
      Graphically plots preconditioned residual norm
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_monitor_lg_residualnorm","Monitor graphically preconditioned residual norm","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscDrawLG ctx;

      ierr = KSPMonitorLGResidualNormCreate(0,0,PETSC_DECIDE,PETSC_DECIDE,300,300,&ctx);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorLGResidualNorm,ctx,(PetscErrorCode (*)(void**))KSPMonitorLGResidualNormDestroy);CHKERRQ(ierr);
    }
    /*
      Graphically plots preconditioned and true residual norm
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_monitor_lg_true_residualnorm","Monitor graphically true residual norm","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg){
      PetscDrawLG ctx;

      ierr = KSPMonitorLGTrueResidualNormCreate(((PetscObject)ksp)->comm,0,0,PETSC_DECIDE,PETSC_DECIDE,300,300,&ctx);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorLGTrueResidualNorm,ctx,(PetscErrorCode (*)(void**))KSPMonitorLGTrueResidualNormDestroy);CHKERRQ(ierr);
    }
    /*
      Graphically plots preconditioned residual norm and range of residual element values
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_monitor_lg_range","Monitor graphically range of preconditioned residual norm","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscViewer ctx;

      ierr = PetscViewerDrawOpen(((PetscObject)ksp)->comm,0,0,PETSC_DECIDE,PETSC_DECIDE,300,300,&ctx);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorLGRange,ctx,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
    }

    /*
      Publishes convergence information using AMS
    */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_monitor_ams","Publish KSP progress using AMS","KSPMonitorSet",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      char amscommname[256];
      void *ctx;
      ierr = PetscSNPrintf(amscommname,sizeof(amscommname),"%sksp_monitor_ams",((PetscObject)ksp)->prefix?((PetscObject)ksp)->prefix:"");CHKERRQ(ierr);
      ierr = KSPMonitorAMSCreate(ksp,amscommname,&ctx);CHKERRQ(ierr);
      ierr = KSPMonitorSet(ksp,KSPMonitorAMS,ctx,KSPMonitorAMSDestroy);CHKERRQ(ierr);
      ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr);
    }

    /* -----------------------------------------------------------------------*/
    ierr = KSPSetUpNorms_Private(ksp,&normtype,&pcside);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-ksp_pc_side","KSP preconditioner side","KSPSetPCSide",PCSides,(PetscEnum)pcside,(PetscEnum*)&pcside,&flg);CHKERRQ(ierr);
    if (flg) {ierr = KSPSetPCSide(ksp,pcside);CHKERRQ(ierr);}

    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_compute_singularvalues","Compute singular values of preconditioned operator","KSPSetComputeSingularValues",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_compute_eigenvalues","Compute eigenvalues of preconditioned operator","KSPSetComputeSingularValues",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsBool("-ksp_plot_eigenvalues","Scatter plot extreme eigenvalues","KSPSetComputeSingularValues",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr); }


    if (ksp->ops->setfromoptions) {
      ierr = (*ksp->ops->setfromoptions)(ksp);CHKERRQ(ierr);
    }
    /* actually check in setup this is just here so goes into help message */
    ierr = PetscOptionsName("-ksp_view","View linear solver parameters","KSPView",&flg);CHKERRQ(ierr);

    /* process any options handlers added with PetscObjectAddOptionsHandler() */
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)ksp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
