
/*
    Code for setting KSP options from the options database.
*/

#include <petsc/private/kspimpl.h>  /*I "petscksp.h" I*/
#include <petscdraw.h>

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
  ierr = PetscObjectGetTabLevel((PetscObject)ksp, tab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  ierr = PetscObjectSetTabLevel((PetscObject)ksp, tab);CHKERRQ(ierr);
  if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
  /* Do we need a PCSetTabLevel()? */
  ierr = PetscObjectSetTabLevel((PetscObject)ksp->pc, tab);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPSetUseFischerGuess - Use the Paul Fischer algorithm

   Logically Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
.  model - use model 1, model 2 or any other number to turn it off
-  size - size of subspace used to generate initial guess

    Options Database:
.   -ksp_fischer_guess <model,size> - uses the Fischer initial guess generator for repeated linear solves

   Level: advanced

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPSetGuess(), KSPGetGuess()
@*/
PetscErrorCode  KSPSetUseFischerGuess(KSP ksp,PetscInt model,PetscInt size)
{
  KSPGuess       guess;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,model,2);
  PetscValidLogicalCollectiveInt(ksp,size,3);
  ierr = KSPGetGuess(ksp,&guess);CHKERRQ(ierr);
  ierr = KSPGuessSetType(guess,KSPGUESSFISCHER);CHKERRQ(ierr);
  ierr = KSPGuessFischerSetModel(guess,model,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   KSPSetGuess - Set the initial guess object

   Logically Collective on KSP

   Input Parameters:
+  ksp - the Krylov context
-  guess - the object created with KSPGuessCreate()

   Level: advanced

   Notes: this allows a single KSP to be used with several different initial guess generators (likely for different linear
          solvers, see KSPSetPC()).

          This increases the reference count of the guess object, you must destroy the object with KSPGuessDestroy()
          before the end of the program.

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPGetGuess()
@*/
PetscErrorCode  KSPSetGuess(KSP ksp,KSPGuess guess)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)guess);CHKERRQ(ierr);
  ierr = KSPGuessDestroy(&ksp->guess);CHKERRQ(ierr);
  ksp->guess = guess;
  ksp->guess->ksp = ksp;
  PetscFunctionReturn(0);
}

/*@
   KSPGetGuess - Gets the initial guess generator for the KSP.

   Not Collective

   Input Parameters:
.  ksp - the Krylov context

   Output Parameters:
.   guess - the object

   Level: developer

.keywords: KSP, set, options, prefix, database

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPSetGuess()
@*/
PetscErrorCode  KSPGetGuess(KSP ksp,KSPGuess *guess)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(guess,2);
  if (!ksp->guess) {
    const char* prefix;

    ierr = KSPGuessCreate(PetscObjectComm((PetscObject)ksp),&ksp->guess);CHKERRQ(ierr);
    ierr = PetscObjectGetOptionsPrefix((PetscObject)ksp,&prefix);CHKERRQ(ierr);
    if (prefix) {
      ierr = PetscObjectSetOptionsPrefix((PetscObject)ksp->guess,prefix);CHKERRQ(ierr);
    }
    ksp->guess->ksp = ksp;
  }
  *guess = ksp->guess;
  PetscFunctionReturn(0);
}

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

/*@C
   KSPMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

   Collective on KSP

   Input Parameters:
+  ksp - KSP object you wish to monitor
.  name - the monitor type one is seeking
.  help - message indicating what monitoring is done
.  manual - manual page for the monitor
-  monitor - the monitor function, the context for this object is a PetscViewerAndFormat

   Level: developer

.seealso: PetscOptionsGetViewer(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode  KSPMonitorSetFromOptions(KSP ksp,const char name[],const char help[], const char manual[],PetscErrorCode (*monitor)(KSP,PetscInt,PetscReal,PetscViewerAndFormat*))
{
  PetscErrorCode       ierr;
  PetscBool            flg;
  PetscViewer          viewer;
  PetscViewerFormat    format;

  PetscFunctionBegin;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)ksp),((PetscObject)ksp)->prefix,name,&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscViewerAndFormat *vf;
    ierr = PetscViewerAndFormatCreate(viewer,format,&vf);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)viewer);CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp,(PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))monitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
.   -ksp_converged_use_initial_residual_norm - see KSPConvergedDefaultSetUIRNorm()
.   -ksp_converged_use_min_initial_residual_norm - see KSPConvergedDefaultSetUMIRNorm()
.   -ksp_norm_type - none - skip norms used in convergence tests (useful only when not using
                       convergence test (say you always want to run with 5 iterations) to
                       save on communication overhead
                    preconditioned - default for left preconditioning
                    unpreconditioned - see KSPSetNormType()
                    natural - see KSPSetNormType()
.   -ksp_check_norm_iteration it - do not compute residual norm until iteration number it (does compute at 0th iteration)
       works only for PCBCGS, PCIBCGS and and PCCG
.   -ksp_lag_norm - compute the norm of the residual for the ith iteration on the i+1 iteration; this means that one can use
       the norm of the residual for convergence test WITHOUT an extra MPI_Allreduce() limiting global synchronizations.
       This will require 1 more iteration of the solver than usual.
.   -ksp_guess_type - Type of initial guess generator for repeated linear solves
.   -ksp_fischer_guess <model,size> - uses the Fischer initial guess generator for repeated linear solves
.   -ksp_constant_null_space - assume the operator (matrix) has the constant vector in its null space
.   -ksp_test_null_space - tests the null space set with MatSetNullSpace() to see if it truly is a null space
.   -ksp_knoll - compute initial guess by applying the preconditioner to the right hand side
.   -ksp_monitor_cancel - cancel all previous convergene monitor routines set
.   -ksp_monitor <optional filename> - print residual norm at each iteration
.   -ksp_monitor_lg_residualnorm - plot residual norm at each iteration
.   -ksp_monitor_solution [ascii binary or draw][:filename][:format option] - plot solution at each iteration
-   -ksp_monitor_singular_value - monitor extreme singular values at each iteration

   Notes:
   To see all options, run your program with the -help option
   or consult Users-Manual: ch_ksp

   Level: beginner

.keywords: KSP, set, from, options, database

.seealso: KSPSetUseFischerGuess()

@*/
PetscErrorCode  KSPSetFromOptions(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       indx;
  const char     *convtests[] = {"default","skip"};
  char           type[256], guesstype[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscBool      flg,flag,reuse,set;
  PetscInt       model[2]={0,0},nmax;
  KSPNormType    normtype;
  PCSide         pcside;
  void           *ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->skippcsetfromoptions) {
    if (!ksp->pc) {ierr = KSPGetPC(ksp,&ksp->pc);CHKERRQ(ierr);}
    ierr = PCSetFromOptions(ksp->pc);CHKERRQ(ierr);
  }

  ierr = KSPRegisterAll();CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)ksp);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-ksp_type","Krylov method","KSPSetType",KSPList,(char*)(((PetscObject)ksp)->type_name ? ((PetscObject)ksp)->type_name : KSPGMRES),type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetType(ksp,type);CHKERRQ(ierr);
  }
  /*
    Set the type if it was never set.
  */
  if (!((PetscObject)ksp)->type_name) {
    ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);
  }

  ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPPREONLY,&flg);CHKERRQ(ierr);
  if (flg) goto skipoptions;

  ierr = PetscOptionsInt("-ksp_max_it","Maximum number of iterations","KSPSetTolerances",ksp->max_it,&ksp->max_it,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_rtol","Relative decrease in residual norm","KSPSetTolerances",ksp->rtol,&ksp->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_atol","Absolute value of residual norm","KSPSetTolerances",ksp->abstol,&ksp->abstol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ksp_divtol","Residual norm increase cause divergence","KSPSetTolerances",ksp->divtol,&ksp->divtol,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-ksp_converged_use_initial_residual_norm","Use initial residual residual norm for computing relative convergence","KSPConvergedDefaultSetUIRNorm",PETSC_FALSE,&flag,&set);CHKERRQ(ierr);
  if (set && flag) {ierr = KSPConvergedDefaultSetUIRNorm(ksp);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-ksp_converged_use_min_initial_residual_norm","Use minimum of initial residual norm and b for computing relative convergence","KSPConvergedDefaultSetUMIRNorm",PETSC_FALSE,&flag,&set);CHKERRQ(ierr);
  if (set && flag) {ierr = KSPConvergedDefaultSetUMIRNorm(ksp);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-ksp_initial_guess_nonzero","Use the contents of the solution vector for initial guess","KSPSetInitialNonzero",ksp->guess_zero ? PETSC_FALSE : PETSC_TRUE,&flag,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetInitialGuessNonzero(ksp,flag);CHKERRQ(ierr);
  }
  ierr = PCGetReusePreconditioner(ksp->pc,&reuse);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_reuse_preconditioner","Use initial preconditioner and don't ever compute a new one ","KSPReusePreconditioner",reuse,&reuse,NULL);CHKERRQ(ierr);
  ierr = KSPSetReusePreconditioner(ksp,reuse);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-ksp_knoll","Use preconditioner applied to b for initial guess","KSPSetInitialGuessKnoll",ksp->guess_knoll,&ksp->guess_knoll,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ksp_error_if_not_converged","Generate error if solver does not converge","KSPSetErrorIfNotConverged",ksp->errorifnotconverged,&ksp->errorifnotconverged,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-ksp_guess_type","Initial guess in Krylov method",NULL,KSPGuessList,NULL,guesstype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPGetGuess(ksp,&ksp->guess);CHKERRQ(ierr);
    ierr = KSPGuessSetType(ksp->guess,guesstype);CHKERRQ(ierr);
    ierr = KSPGuessSetFromOptions(ksp->guess);CHKERRQ(ierr);
  } else { /* old option for KSP */
    nmax = 2;
    ierr = PetscOptionsIntArray("-ksp_fischer_guess","Use Paul Fischer's algorithm for initial guess","KSPSetUseFischerGuess",model,&nmax,&flag);CHKERRQ(ierr);
    if (flag) {
      if (nmax != 2) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Must pass in model,size as arguments");
      ierr = KSPSetUseFischerGuess(ksp,model[0],model[1]);CHKERRQ(ierr);
    }
  }

  ierr = PetscOptionsEList("-ksp_convergence_test","Convergence test","KSPSetConvergenceTest",convtests,2,"default",&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (indx) {
    case 0:
      ierr = KSPConvergedDefaultCreate(&ctx);CHKERRQ(ierr);
      ierr = KSPSetConvergenceTest(ksp,KSPConvergedDefault,ctx,KSPConvergedDefaultDestroy);CHKERRQ(ierr);
      break;
    case 1: ierr = KSPSetConvergenceTest(ksp,KSPConvergedSkip,NULL,NULL);CHKERRQ(ierr);    break;
    }
  }

  ierr = KSPSetUpNorms_Private(ksp,PETSC_FALSE,&normtype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-ksp_norm_type","KSP Norm type","KSPSetNormType",KSPNormTypes,(PetscEnum)normtype,(PetscEnum*)&normtype,&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPSetNormType(ksp,normtype);CHKERRQ(ierr); }

  ierr = PetscOptionsInt("-ksp_check_norm_iteration","First iteration to compute residual norm","KSPSetCheckNormIteration",ksp->chknorm,&ksp->chknorm,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsBool("-ksp_lag_norm","Lag the calculation of the residual norm","KSPSetLagNorm",ksp->lagnorm,&flag,&flg);CHKERRQ(ierr);
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

  ierr = PetscOptionsBool("-ksp_constant_null_space","Add constant null space to Krylov solver matrix","MatSetNullSpace",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set && flg) {
    MatNullSpace nsp;
    Mat          Amat;

    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)ksp),PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
    ierr = PCGetOperators(ksp->pc,&Amat,NULL);CHKERRQ(ierr);
    if (Amat) {
      ierr = MatSetNullSpace(Amat,nsp);CHKERRQ(ierr);
      ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_WRONGSTATE,"Cannot set nullspace, matrix has not yet been provided");
  }

  ierr = PetscOptionsBool("-ksp_monitor_cancel","Remove any hardwired monitor routines","KSPMonitorCancel",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  /* -----------------------------------------------------------------------*/
  /*
    Cancels all monitors hardwired into code before call to KSPSetFromOptions()
  */
  if (set && flg) {
    ierr = KSPMonitorCancel(ksp);CHKERRQ(ierr);
  }
  ierr = KSPMonitorSetFromOptions(ksp,"-ksp_monitor","Monitor the (preconditioned) residual norm","KSPMonitorDefault",KSPMonitorDefault);CHKERRQ(ierr);
  ierr = KSPMonitorSetFromOptions(ksp,"-ksp_monitor_range","Monitor the percentage of large entries in the residual","KSPMonitorRange",KSPMonitorRange);CHKERRQ(ierr);
  ierr = KSPMonitorSetFromOptions(ksp,"-ksp_monitor_true_residual","Monitor the unprecondiitoned residual norm","KSPMOnitorTrueResidual",KSPMonitorTrueResidualNorm);CHKERRQ(ierr);
  ierr = KSPMonitorSetFromOptions(ksp,"-ksp_monitor_max","Monitor the maximum norm of the residual","KSPMonitorTrueResidualMaxNorm",KSPMonitorTrueResidualMaxNorm);CHKERRQ(ierr);
  ierr = KSPMonitorSetFromOptions(ksp,"-ksp_monitor_short","Monitor preconditioned residual norm with fewer digits","KSPMonitorDefaultShort",KSPMonitorDefaultShort);CHKERRQ(ierr);
  ierr = KSPMonitorSetFromOptions(ksp,"-ksp_monitor_solution","Monitor the solution","KSPMonitorSolution",KSPMonitorSolution);CHKERRQ(ierr);
  ierr = KSPMonitorSetFromOptions(ksp,"-ksp_monitor_singular_value","Monitor singular values","KSPMonitorSingularValue",KSPMonitorSingularValue);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,((PetscObject)ksp)->prefix,"-ksp_monitor_singular_value",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscObjectTypeCompare((PetscObject)ksp->pc,PCKSP,&flg);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ksp->pc,PCBJACOBI,&flag);CHKERRQ(ierr);

  if (flg || flag) {
    /* A hack for using dynamic tolerance in preconditioner */
    ierr = PetscOptionsString("-sub_ksp_dynamic_tolerance","Use dynamic tolerance for PC if PC is a KSP","KSPMonitorDynamicTolerance","stdout",monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (flg) {
      KSPDynTolCtx *scale;
      ierr        = PetscMalloc1(1,&scale);CHKERRQ(ierr);
      scale->bnrm = -1.0;
      scale->coef = 1.0;
      ierr        = PetscOptionsReal("-sub_ksp_dynamic_tolerance_param","Parameter of dynamic tolerance for inner PCKSP","KSPMonitorDynamicToleranceParam",scale->coef,&scale->coef,&flg);CHKERRQ(ierr);
      ierr        = KSPMonitorSet(ksp,KSPMonitorDynamicTolerance,scale,KSPMonitorDynamicToleranceDestroy);CHKERRQ(ierr);
    }
  }
 

  /*
   Calls Python function
  */
  ierr = PetscOptionsString("-ksp_monitor_python","Use Python function","KSPMonitorSet",0,monfilename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscPythonMonitorSet((PetscObject)ksp,monfilename);CHKERRQ(ierr);}
  /*
    Graphically plots preconditioned residual norm
  */
  ierr = PetscOptionsBool("-ksp_monitor_lg_residualnorm","Monitor graphically preconditioned residual norm","KSPMonitorSet",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set && flg) {
    PetscDrawLG ctx;

    ierr = KSPMonitorLGResidualNormCreate(PetscObjectComm((PetscObject)ksp),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&ctx);CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp,KSPMonitorLGResidualNorm,ctx,(PetscErrorCode (*)(void**))PetscDrawLGDestroy);CHKERRQ(ierr);
  }
  /*
    Graphically plots preconditioned and true residual norm
  */
  ierr = PetscOptionsBool("-ksp_monitor_lg_true_residualnorm","Monitor graphically true residual norm","KSPMonitorSet",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set && flg) {
    PetscDrawLG ctx;

    ierr = KSPMonitorLGTrueResidualNormCreate(PetscObjectComm((PetscObject)ksp),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&ctx);CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp,KSPMonitorLGTrueResidualNorm,ctx,(PetscErrorCode (*)(void**))PetscDrawLGDestroy);CHKERRQ(ierr);
  }
  /*
    Graphically plots preconditioned residual norm and range of residual element values
  */
  ierr = PetscOptionsBool("-ksp_monitor_lg_range","Monitor graphically range of preconditioned residual norm","KSPMonitorSet",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set && flg) {
    PetscViewer ctx;

    ierr = PetscViewerDrawOpen(PetscObjectComm((PetscObject)ksp),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&ctx);CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp,KSPMonitorLGRange,ctx,(PetscErrorCode (*)(void**))PetscViewerDestroy);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_SAWS)
  /*
    Publish convergence information using AMS
  */
  ierr = PetscOptionsBool("-ksp_monitor_saws","Publish KSP progress using SAWs","KSPMonitorSet",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set && flg) {
    void *ctx;
    ierr = KSPMonitorSAWsCreate(ksp,&ctx);CHKERRQ(ierr);
    ierr = KSPMonitorSet(ksp,KSPMonitorSAWs,ctx,KSPMonitorSAWsDestroy);CHKERRQ(ierr);
    ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr);
  }
#endif

  /* -----------------------------------------------------------------------*/
  ierr = KSPSetUpNorms_Private(ksp,PETSC_FALSE,NULL,&pcside);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-ksp_pc_side","KSP preconditioner side","KSPSetPCSide",PCSides,(PetscEnum)pcside,(PetscEnum*)&pcside,&flg);CHKERRQ(ierr);
  if (flg) {ierr = KSPSetPCSide(ksp,pcside);CHKERRQ(ierr);}

  ierr = PetscOptionsBool("-ksp_compute_singularvalues","Compute singular values of preconditioned operator","KSPSetComputeSingularValues",ksp->calc_sings,&flg,&set);CHKERRQ(ierr);
  if (set) { ierr = KSPSetComputeSingularValues(ksp,flg);CHKERRQ(ierr); }
  ierr = PetscOptionsBool("-ksp_compute_eigenvalues","Compute eigenvalues of preconditioned operator","KSPSetComputeSingularValues",ksp->calc_sings,&flg,&set);CHKERRQ(ierr);
  if (set) { ierr = KSPSetComputeSingularValues(ksp,flg);CHKERRQ(ierr); }
  ierr = PetscOptionsBool("-ksp_plot_eigenvalues","Scatter plot extreme eigenvalues","KSPSetComputeSingularValues",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) { ierr = KSPSetComputeSingularValues(ksp,flg);CHKERRQ(ierr); }

#if defined(PETSC_HAVE_SAWS)
  {
  PetscBool set;
  flg  = PETSC_FALSE;
  ierr = PetscOptionsBool("-ksp_saws_block","Block for SAWs at end of KSPSolve","PetscObjectSAWsBlock",((PetscObject)ksp)->amspublishblock,&flg,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PetscObjectSAWsSetBlock((PetscObject)ksp,flg);CHKERRQ(ierr);
  }
  }
#endif

  if (ksp->ops->setfromoptions) {
    ierr = (*ksp->ops->setfromoptions)(PetscOptionsObject,ksp);CHKERRQ(ierr);
  }
  skipoptions:
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)ksp);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
