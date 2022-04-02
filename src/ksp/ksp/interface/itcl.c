
/*
    Code for setting KSP options from the options database.
*/

#include <petsc/private/kspimpl.h>  /*I "petscksp.h" I*/
#include <petscdraw.h>

/*@C
   KSPSetOptionsPrefix - Sets the prefix used for searching for all
   KSP options in the database.

   Logically Collective on ksp

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

.seealso: KSPAppendOptionsPrefix(), KSPGetOptionsPrefix()
@*/
PetscErrorCode  KSPSetOptionsPrefix(KSP ksp,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp,&ksp->pc));
  PetscCall(PCSetOptionsPrefix(ksp->pc,prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ksp,prefix));
  PetscFunctionReturn(0);
}

/*@C
   KSPAppendOptionsPrefix - Appends to the prefix used for searching for all
   KSP options in the database.

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov context
-  prefix - the prefix string to prepend to all KSP option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: KSPSetOptionsPrefix(), KSPGetOptionsPrefix()
@*/
PetscErrorCode  KSPAppendOptionsPrefix(KSP ksp,const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp,&ksp->pc));
  PetscCall(PCAppendOptionsPrefix(ksp->pc,prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)ksp,prefix));
  PetscFunctionReturn(0);
}

/*@
   KSPSetUseFischerGuess - Use the Paul Fischer algorithm or its variants

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov context
.  model - use model 1, model 2, model 3, or any other number to turn it off
-  size - size of subspace used to generate initial guess

    Options Database:
.   -ksp_fischer_guess <model,size> - uses the Fischer initial guess generator for repeated linear solves

   Level: advanced

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPSetGuess(), KSPGetGuess()
@*/
PetscErrorCode  KSPSetUseFischerGuess(KSP ksp,PetscInt model,PetscInt size)
{
  KSPGuess       guess;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidLogicalCollectiveInt(ksp,model,2);
  PetscValidLogicalCollectiveInt(ksp,size,3);
  PetscCall(KSPGetGuess(ksp,&guess));
  PetscCall(KSPGuessSetType(guess,KSPGUESSFISCHER));
  PetscCall(KSPGuessFischerSetModel(guess,model,size));
  PetscFunctionReturn(0);
}

/*@
   KSPSetGuess - Set the initial guess object

   Logically Collective on ksp

   Input Parameters:
+  ksp - the Krylov context
-  guess - the object created with KSPGuessCreate()

   Level: advanced

   Notes:
    this allows a single KSP to be used with several different initial guess generators (likely for different linear
          solvers, see KSPSetPC()).

          This increases the reference count of the guess object, you must destroy the object with KSPGuessDestroy()
          before the end of the program.

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPGetGuess()
@*/
PetscErrorCode  KSPSetGuess(KSP ksp,KSPGuess guess)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidHeaderSpecific(guess,KSPGUESS_CLASSID,2);
  PetscCall(PetscObjectReference((PetscObject)guess));
  PetscCall(KSPGuessDestroy(&ksp->guess));
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

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix(), KSPSetUseFischerGuess(), KSPSetGuess()
@*/
PetscErrorCode  KSPGetGuess(KSP ksp,KSPGuess *guess)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscValidPointer(guess,2);
  if (!ksp->guess) {
    const char* prefix;

    PetscCall(KSPGuessCreate(PetscObjectComm((PetscObject)ksp),&ksp->guess));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ksp,&prefix));
    if (prefix) {
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ksp->guess,prefix));
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

   Notes:
    On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: KSPSetOptionsPrefix(), KSPAppendOptionsPrefix()
@*/
PetscErrorCode  KSPGetOptionsPrefix(KSP ksp,const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ksp,prefix));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerAndFormatCreate_Internal(PetscViewer viewer, PetscViewerFormat format, void *ctx, PetscViewerAndFormat **vf)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscFunctionReturn(0);
}

/*@C
   KSPMonitorSetFromOptions - Sets a monitor function and viewer appropriate for the type indicated by the user

   Collective on ksp

   Input Parameters:
+  ksp  - KSP object you wish to monitor
.  opt  - the command line option for this monitor
.  name - the monitor type one is seeking
-  ctx  - An optional user context for the monitor, or NULL

   Level: developer

.seealso: PetscOptionsGetViewer(), PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHeadBegin(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList()
@*/
PetscErrorCode KSPMonitorSetFromOptions(KSP ksp, const char opt[], const char name[], void *ctx)
{
  PetscErrorCode      (*mfunc)(KSP, PetscInt, PetscReal, void *);
  PetscErrorCode      (*cfunc)(PetscViewer, PetscViewerFormat, void *, PetscViewerAndFormat **);
  PetscErrorCode      (*dfunc)(PetscViewerAndFormat **);
  PetscViewerAndFormat *vf;
  PetscViewer           viewer;
  PetscViewerFormat     format;
  PetscViewerType       vtype;
  char                  key[PETSC_MAX_PATH_LEN];
  PetscBool             all, flg;
  const char           *prefix = NULL;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(opt, "-all_ksp_monitor", &all));
  if (!all) PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject) ksp), ((PetscObject) ksp)->options, prefix, opt, &viewer, &format, &flg));
  if (!flg) PetscFunctionReturn(0);

  PetscCall(PetscViewerGetType(viewer, &vtype));
  PetscCall(KSPMonitorMakeKey_Internal(name, vtype, format, key));
  PetscCall(PetscFunctionListFind(KSPMonitorList, key, &mfunc));
  PetscCall(PetscFunctionListFind(KSPMonitorCreateList, key, &cfunc));
  PetscCall(PetscFunctionListFind(KSPMonitorDestroyList, key, &dfunc));
  if (!cfunc) cfunc = PetscViewerAndFormatCreate_Internal;
  if (!dfunc) dfunc = PetscViewerAndFormatDestroy;

  PetscCall((*cfunc)(viewer, format, ctx, &vf));
  PetscCall(PetscObjectDereference((PetscObject) viewer));
  PetscCall(KSPMonitorSet(ksp, mfunc, vf, (PetscErrorCode (*)(void **)) dfunc));
  PetscFunctionReturn(0);
}

/*@
   KSPSetFromOptions - Sets KSP options from the options database.
   This routine must be called before KSPSetUp() if the user is to be
   allowed to set the Krylov type.

   Collective on ksp

   Input Parameter:
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
.   -ksp_converged_maxits - see KSPConvergedDefaultSetConvergedMaxits()
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
.   -ksp_monitor - print residual norm at each iteration
.   -ksp_monitor draw::draw_lg - plot residual norm at each iteration
.   -ksp_monitor_true_residual - print true residual norm at each iteration
.   -all_ksp_monitor <optional filename> - print residual norm at each iteration for ALL KSP solves, regardless of their prefix. This is
                                           useful for PCFIELDSPLIT, PCMG, etc that have inner solvers and you wish to track the convergence of all the solvers
.   -ksp_monitor_solution [ascii binary or draw][:filename][:format option] - plot solution at each iteration
.   -ksp_monitor_singular_value - monitor extreme singular values at each iteration
.   -ksp_converged_reason - view the convergence state at the end of the solve
.   -ksp_use_explicittranspose - transpose the system explicitly in KSPSolveTranspose
.   -ksp_error_if_not_converged - stop the program as soon as an error is detected in a KSPSolve(), KSP_DIVERGED_ITS is not treated as an error on inner KSPSolves
-   -ksp_converged_rate - view the convergence rate at the end of the solve

   Notes:
   To see all options, run your program with the -help option
   or consult Users-Manual: ch_ksp

   Level: beginner

.seealso: KSPSetOptionsPrefix(), KSPResetFromOptions(), KSPSetUseFischerGuess()

@*/
PetscErrorCode  KSPSetFromOptions(KSP ksp)
{
  const char     *convtests[]={"default","skip","lsqr"},*prefix;
  char           type[256],guesstype[256],monfilename[PETSC_MAX_PATH_LEN];
  PetscBool      flg,flag,reuse,set;
  PetscInt       indx,model[2]={0,0},nmax;
  KSPNormType    normtype;
  PCSide         pcside;
  void           *ctx;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  PetscCall(PetscObjectGetComm((PetscObject) ksp, &comm));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) ksp, &prefix));
  if (!ksp->skippcsetfromoptions) {
    if (!ksp->pc) PetscCall(KSPGetPC(ksp,&ksp->pc));
    PetscCall(PCSetFromOptions(ksp->pc));
  }

  PetscCall(KSPRegisterAll());
  PetscObjectOptionsBegin((PetscObject)ksp);
  PetscCall(PetscOptionsFList("-ksp_type","Krylov method","KSPSetType",KSPList,(char*)(((PetscObject)ksp)->type_name ? ((PetscObject)ksp)->type_name : KSPGMRES),type,256,&flg));
  if (flg) {
    PetscCall(KSPSetType(ksp,type));
  }
  /*
    Set the type if it was never set.
  */
  if (!((PetscObject)ksp)->type_name) {
    PetscCall(KSPSetType(ksp,KSPGMRES));
  }

  PetscCall(KSPResetViewers(ksp));

  /* Cancels all monitors hardwired into code before call to KSPSetFromOptions() */
  PetscCall(PetscOptionsBool("-ksp_monitor_cancel","Remove any hardwired monitor routines","KSPMonitorCancel",PETSC_FALSE,&flg,&set));
  if (set && flg) PetscCall(KSPMonitorCancel(ksp));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor", "preconditioned_residual", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_short", "preconditioned_residual_short", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-all_ksp_monitor", "preconditioned_residual", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_range", "preconditioned_residual_range", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_true_residual", "true_residual", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_max", "true_residual_max", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_solution", "solution", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_singular_value", "singular_value", ksp));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_error", "error", ksp));
  PetscCall(PetscOptionsBool("-ksp_monitor_pause_final", "Pauses all draw monitors at the final iterate", "KSPMonitorPauseFinal_Internal", PETSC_FALSE, &ksp->pauseFinal, NULL));
  PetscCall(PetscOptionsBool("-ksp_initial_guess_nonzero","Use the contents of the solution vector for initial guess","KSPSetInitialNonzero",ksp->guess_zero ? PETSC_FALSE : PETSC_TRUE,&flag,&flg));
  if (flg) {
    PetscCall(KSPSetInitialGuessNonzero(ksp,flag));
  }

  PetscCall(PetscObjectTypeCompare((PetscObject)ksp,KSPPREONLY,&flg));
  if (flg) {
    PetscCall(KSPGetReusePreconditioner(ksp,&reuse));
    PetscCall(PetscOptionsBool("-ksp_reuse_preconditioner","Use initial preconditioner and don't ever compute a new one","KSPReusePreconditioner",reuse,&reuse,NULL));
    PetscCall(KSPSetReusePreconditioner(ksp,reuse));
    PetscCall(PetscOptionsBool("-ksp_error_if_not_converged","Generate error if solver does not converge","KSPSetErrorIfNotConverged",ksp->errorifnotconverged,&ksp->errorifnotconverged,NULL));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view",&ksp->viewer, &ksp->format,&ksp->view));
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-ksp_converged_reason_view_cancel","Cancel all the converged reason view functions set using KSPConvergedReasonViewSet","KSPConvergedReasonViewCancel",PETSC_FALSE,&flg,&set));
    if (set && flg) {
      PetscCall(KSPConvergedReasonViewCancel(ksp));
    }
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_mat",&ksp->viewerMat,&ksp->formatMat,&ksp->viewMat));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_pmat",&ksp->viewerPMat,&ksp->formatPMat,&ksp->viewPMat));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_rhs",&ksp->viewerRhs,&ksp->formatRhs,&ksp->viewRhs));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_solution",&ksp->viewerSol,&ksp->formatSol,&ksp->viewSol));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_mat_explicit",&ksp->viewerMatExp,&ksp->formatMatExp,&ksp->viewMatExp));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_final_residual",&ksp->viewerFinalRes,&ksp->formatFinalRes,&ksp->viewFinalRes));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_preconditioned_operator_explicit",&ksp->viewerPOpExp,&ksp->formatPOpExp,&ksp->viewPOpExp));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_diagonal_scale",&ksp->viewerDScale,&ksp->formatDScale,&ksp->viewDScale));

    PetscCall(KSPGetDiagonalScale(ksp,&flag));
    PetscCall(PetscOptionsBool("-ksp_diagonal_scale","Diagonal scale matrix before building preconditioner","KSPSetDiagonalScale",flag,&flag,&flg));
    if (flg) {
      PetscCall(KSPSetDiagonalScale(ksp,flag));
    }
    PetscCall(KSPGetDiagonalScaleFix(ksp,&flag));
    PetscCall(PetscOptionsBool("-ksp_diagonal_scale_fix","Fix diagonally scaled matrix after solve","KSPSetDiagonalScaleFix",flag,&flag,&flg));
    if (flg) {
      PetscCall(KSPSetDiagonalScaleFix(ksp,flag));
    }
    nmax = ksp->nmax;
    PetscCall(PetscOptionsDeprecated("-ksp_matsolve_block_size","-ksp_matsolve_batch_size","3.15",NULL));
    PetscCall(PetscOptionsInt("-ksp_matsolve_batch_size", "Maximum number of columns treated simultaneously", "KSPSetMatSolveBatchSize", nmax, &nmax, &flg));
    if (flg) {
      PetscCall(KSPSetMatSolveBatchSize(ksp, nmax));
    }
    goto skipoptions;
  }

  PetscCall(PetscOptionsInt("-ksp_max_it","Maximum number of iterations","KSPSetTolerances",ksp->max_it,&ksp->max_it,NULL));
  PetscCall(PetscOptionsReal("-ksp_rtol","Relative decrease in residual norm","KSPSetTolerances",ksp->rtol,&ksp->rtol,NULL));
  PetscCall(PetscOptionsReal("-ksp_atol","Absolute value of residual norm","KSPSetTolerances",ksp->abstol,&ksp->abstol,NULL));
  PetscCall(PetscOptionsReal("-ksp_divtol","Residual norm increase cause divergence","KSPSetTolerances",ksp->divtol,&ksp->divtol,NULL));

  PetscCall(PetscOptionsBool("-ksp_converged_use_initial_residual_norm","Use initial residual norm for computing relative convergence","KSPConvergedDefaultSetUIRNorm",PETSC_FALSE,&flag,&set));
  if (set && flag) PetscCall(KSPConvergedDefaultSetUIRNorm(ksp));
  PetscCall(PetscOptionsBool("-ksp_converged_use_min_initial_residual_norm","Use minimum of initial residual norm and b for computing relative convergence","KSPConvergedDefaultSetUMIRNorm",PETSC_FALSE,&flag,&set));
  if (set && flag) PetscCall(KSPConvergedDefaultSetUMIRNorm(ksp));
  PetscCall(PetscOptionsBool("-ksp_converged_maxits","Declare convergence if the maximum number of iterations is reached","KSPConvergedDefaultSetConvergedMaxits",PETSC_FALSE,&flag,&set));
  if (set) PetscCall(KSPConvergedDefaultSetConvergedMaxits(ksp,flag));
  PetscCall(KSPGetReusePreconditioner(ksp,&reuse));
  PetscCall(PetscOptionsBool("-ksp_reuse_preconditioner","Use initial preconditioner and don't ever compute a new one","KSPReusePreconditioner",reuse,&reuse,NULL));
  PetscCall(KSPSetReusePreconditioner(ksp,reuse));

  PetscCall(PetscOptionsBool("-ksp_knoll","Use preconditioner applied to b for initial guess","KSPSetInitialGuessKnoll",ksp->guess_knoll,&ksp->guess_knoll,NULL));
  PetscCall(PetscOptionsBool("-ksp_error_if_not_converged","Generate error if solver does not converge","KSPSetErrorIfNotConverged",ksp->errorifnotconverged,&ksp->errorifnotconverged,NULL));
  PetscCall(PetscOptionsFList("-ksp_guess_type","Initial guess in Krylov method",NULL,KSPGuessList,NULL,guesstype,256,&flg));
  if (flg) {
    PetscCall(KSPGetGuess(ksp,&ksp->guess));
    PetscCall(KSPGuessSetType(ksp->guess,guesstype));
    PetscCall(KSPGuessSetFromOptions(ksp->guess));
  } else { /* old option for KSP */
    nmax = 2;
    PetscCall(PetscOptionsIntArray("-ksp_fischer_guess","Use Paul Fischer's algorithm or its variants for initial guess","KSPSetUseFischerGuess",model,&nmax,&flag));
    if (flag) {
      PetscCheck(nmax == 2,comm,PETSC_ERR_ARG_OUTOFRANGE,"Must pass in model,size as arguments");
      PetscCall(KSPSetUseFischerGuess(ksp,model[0],model[1]));
    }
  }

  PetscCall(PetscOptionsEList("-ksp_convergence_test","Convergence test","KSPSetConvergenceTest",convtests,3,"default",&indx,&flg));
  if (flg) {
    switch (indx) {
    case 0:
      PetscCall(KSPConvergedDefaultCreate(&ctx));
      PetscCall(KSPSetConvergenceTest(ksp,KSPConvergedDefault,ctx,KSPConvergedDefaultDestroy));
      break;
    case 1:
      PetscCall(KSPSetConvergenceTest(ksp,KSPConvergedSkip,NULL,NULL));
      break;
    case 2:
      PetscCall(KSPConvergedDefaultCreate(&ctx));
      PetscCall(KSPSetConvergenceTest(ksp,KSPLSQRConvergedDefault,ctx,KSPConvergedDefaultDestroy));
      break;
    }
  }

  PetscCall(KSPSetUpNorms_Private(ksp,PETSC_FALSE,&normtype,NULL));
  PetscCall(PetscOptionsEnum("-ksp_norm_type","KSP Norm type","KSPSetNormType",KSPNormTypes,(PetscEnum)normtype,(PetscEnum*)&normtype,&flg));
  if (flg) PetscCall(KSPSetNormType(ksp,normtype));

  PetscCall(PetscOptionsInt("-ksp_check_norm_iteration","First iteration to compute residual norm","KSPSetCheckNormIteration",ksp->chknorm,&ksp->chknorm,NULL));

  PetscCall(PetscOptionsBool("-ksp_lag_norm","Lag the calculation of the residual norm","KSPSetLagNorm",ksp->lagnorm,&flag,&flg));
  if (flg) {
    PetscCall(KSPSetLagNorm(ksp,flag));
  }

  PetscCall(KSPGetDiagonalScale(ksp,&flag));
  PetscCall(PetscOptionsBool("-ksp_diagonal_scale","Diagonal scale matrix before building preconditioner","KSPSetDiagonalScale",flag,&flag,&flg));
  if (flg) {
    PetscCall(KSPSetDiagonalScale(ksp,flag));
  }
  PetscCall(KSPGetDiagonalScaleFix(ksp,&flag));
  PetscCall(PetscOptionsBool("-ksp_diagonal_scale_fix","Fix diagonally scaled matrix after solve","KSPSetDiagonalScaleFix",flag,&flag,&flg));
  if (flg) {
    PetscCall(KSPSetDiagonalScaleFix(ksp,flag));
  }

  PetscCall(PetscOptionsBool("-ksp_constant_null_space","Add constant null space to Krylov solver matrix","MatSetNullSpace",PETSC_FALSE,&flg,&set));
  if (set && flg) {
    MatNullSpace nsp;
    Mat          Amat = NULL;

    PetscCall(MatNullSpaceCreate(comm,PETSC_TRUE,0,NULL,&nsp));
    if (ksp->pc) PetscCall(PCGetOperators(ksp->pc,&Amat,NULL));
    if (Amat) {
      PetscCall(MatSetNullSpace(Amat,nsp));
      PetscCall(MatNullSpaceDestroy(&nsp));
    } else SETERRQ(comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot set nullspace, matrix has not yet been provided");
  }

  flg = PETSC_FALSE;
  if (ksp->pc) {
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc,PCKSP,&flg));
    if (!flg) PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc,PCBJACOBI,&flg));
    if (!flg) PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc,PCDEFLATION,&flg));
  }

  if (flg) {
    /* A hack for using dynamic tolerance in preconditioner */
    PetscCall(PetscOptionsString("-sub_ksp_dynamic_tolerance","Use dynamic tolerance for PC if PC is a KSP","KSPMonitorDynamicTolerance","stdout",monfilename,sizeof(monfilename),&flg));
    if (flg) {
      KSPDynTolCtx *scale;
      PetscCall(PetscMalloc1(1,&scale));
      scale->bnrm = -1.0;
      scale->coef = 1.0;
      PetscCall(PetscOptionsReal("-sub_ksp_dynamic_tolerance_param","Parameter of dynamic tolerance for inner PCKSP","KSPMonitorDynamicToleranceParam",scale->coef,&scale->coef,&flg));
      PetscCall(KSPMonitorSet(ksp,KSPMonitorDynamicTolerance,scale,KSPMonitorDynamicToleranceDestroy));
    }
  }

  /*
   Calls Python function
  */
  PetscCall(PetscOptionsString("-ksp_monitor_python","Use Python function","KSPMonitorSet",NULL,monfilename,sizeof(monfilename),&flg));
  if (flg) PetscCall(PetscPythonMonitorSet((PetscObject)ksp,monfilename));
  /*
    Graphically plots preconditioned residual norm and range of residual element values
  */
  PetscCall(PetscOptionsBool("-ksp_monitor_lg_range","Monitor graphically range of preconditioned residual norm","KSPMonitorSet",PETSC_FALSE,&flg,&set));
  if (set && flg) {
    PetscViewer ctx;

    PetscCall(PetscViewerDrawOpen(comm,NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&ctx));
    PetscCall(KSPMonitorSet(ksp,KSPMonitorLGRange,ctx,(PetscErrorCode (*)(void**))PetscViewerDestroy));
  }
  /* TODO Do these show up in help? */
  PetscCall(PetscOptionsHasName(((PetscObject) ksp)->options, prefix, "-ksp_converged_rate", &flg));
  if (flg) {
    const char *RateTypes[] = {"default", "residual", "error", "PetscRateType", "RATE_", NULL};
    PetscEnum rtype = (PetscEnum) 1;

    PetscCall(PetscOptionsGetEnum(((PetscObject) ksp)->options, prefix, "-ksp_converged_rate_type", RateTypes, &rtype, &flg));
    if (rtype == (PetscEnum) 0 || rtype == (PetscEnum) 1) PetscCall(KSPSetResidualHistory(ksp, NULL, PETSC_DETERMINE, PETSC_TRUE));
    if (rtype == (PetscEnum) 0 || rtype == (PetscEnum) 2) PetscCall(KSPSetErrorHistory(ksp, NULL, PETSC_DETERMINE, PETSC_TRUE));
  }
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view",&ksp->viewer,&ksp->format,&ksp->view));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_pre",&ksp->viewerPre,&ksp->formatPre,&ksp->viewPre));

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-ksp_converged_reason_view_cancel","Cancel all the converged reason view functions set using KSPConvergedReasonViewSet","KSPConvergedReasonViewCancel",PETSC_FALSE,&flg,&set));
  if (set && flg) {
    PetscCall(KSPConvergedReasonViewCancel(ksp));
  }
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_converged_rate",&ksp->viewerRate,&ksp->formatRate,&ksp->viewRate));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_mat",&ksp->viewerMat,&ksp->formatMat,&ksp->viewMat));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_pmat",&ksp->viewerPMat,&ksp->formatPMat,&ksp->viewPMat));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_rhs",&ksp->viewerRhs,&ksp->formatRhs,&ksp->viewRhs));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_solution",&ksp->viewerSol,&ksp->formatSol,&ksp->viewSol));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_mat_explicit",&ksp->viewerMatExp,&ksp->formatMatExp,&ksp->viewMatExp));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_eigenvalues",&ksp->viewerEV,&ksp->formatEV,&ksp->viewEV));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_singularvalues",&ksp->viewerSV,&ksp->formatSV,&ksp->viewSV));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_eigenvalues_explicit",&ksp->viewerEVExp,&ksp->formatEVExp,&ksp->viewEVExp));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_final_residual",&ksp->viewerFinalRes,&ksp->formatFinalRes,&ksp->viewFinalRes));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_preconditioned_operator_explicit",&ksp->viewerPOpExp,&ksp->formatPOpExp,&ksp->viewPOpExp));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_view_diagonal_scale",&ksp->viewerDScale,&ksp->formatDScale,&ksp->viewDScale));

  /* Deprecated options */
  if (!ksp->viewEV) {
    PetscCall(PetscOptionsDeprecated("-ksp_compute_eigenvalues",NULL,"3.9","Use -ksp_view_eigenvalues"));
    PetscCall(PetscOptionsGetViewer(comm, ((PetscObject) ksp)->options,prefix, "-ksp_compute_eigenvalues",&ksp->viewerEV,&ksp->formatEV,&ksp->viewEV));
  }
  if (!ksp->viewEV) {
    PetscCall(PetscOptionsDeprecated("-ksp_plot_eigenvalues",NULL,"3.9","Use -ksp_view_eigenvalues draw"));
    PetscCall(PetscOptionsName("-ksp_plot_eigenvalues", "[deprecated since PETSc 3.9; use -ksp_view_eigenvalues draw]", "KSPView", &ksp->viewEV));
    if (ksp->viewEV) {
      ksp->formatEV = PETSC_VIEWER_DEFAULT;
      ksp->viewerEV = PETSC_VIEWER_DRAW_(comm);
      PetscCall(PetscObjectReference((PetscObject) ksp->viewerEV));
    }
  }
  if (!ksp->viewEV) {
    PetscCall(PetscOptionsDeprecated("-ksp_plot_eigencontours",NULL,"3.9","Use -ksp_view_eigenvalues draw::draw_contour"));
    PetscCall(PetscOptionsName("-ksp_plot_eigencontours", "[deprecated since PETSc 3.9; use -ksp_view_eigenvalues draw::draw_contour]", "KSPView", &ksp->viewEV));
    if (ksp->viewEV) {
      ksp->formatEV = PETSC_VIEWER_DRAW_CONTOUR;
      ksp->viewerEV = PETSC_VIEWER_DRAW_(comm);
      PetscCall(PetscObjectReference((PetscObject) ksp->viewerEV));
    }
  }
  if (!ksp->viewEVExp) {
    PetscCall(PetscOptionsDeprecated("-ksp_compute_eigenvalues_explicitly",NULL,"3.9","Use -ksp_view_eigenvalues_explicit"));
    PetscCall(PetscOptionsGetViewer(comm, ((PetscObject) ksp)->options,prefix, "-ksp_compute_eigenvalues_explicitly",&ksp->viewerEVExp,&ksp->formatEVExp,&ksp->viewEVExp));
  }
  if (!ksp->viewEVExp) {
    PetscCall(PetscOptionsDeprecated("-ksp_plot_eigenvalues_explicitly",NULL,"3.9","Use -ksp_view_eigenvalues_explicit draw"));
    PetscCall(PetscOptionsName("-ksp_plot_eigenvalues_explicitly","[deprecated since PETSc 3.9; use -ksp_view_eigenvalues_explicit draw]","KSPView",&ksp->viewEVExp));
    if (ksp->viewEVExp) {
      ksp->formatEVExp = PETSC_VIEWER_DEFAULT;
      ksp->viewerEVExp = PETSC_VIEWER_DRAW_(comm);
      PetscCall(PetscObjectReference((PetscObject) ksp->viewerEVExp));
    }
  }
  if (!ksp->viewSV) {
    PetscCall(PetscOptionsDeprecated("-ksp_compute_singularvalues",NULL,"3.9","Use -ksp_view_singularvalues"));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_compute_singularvalues",&ksp->viewerSV,&ksp->formatSV,&ksp->viewSV));
  }
  if (!ksp->viewFinalRes) {
    PetscCall(PetscOptionsDeprecated("-ksp_final_residual",NULL,"3.9","Use -ksp_view_final_residual"));
    PetscCall(PetscOptionsGetViewer(comm,((PetscObject) ksp)->options,prefix,"-ksp_final_residual",&ksp->viewerFinalRes,&ksp->formatFinalRes,&ksp->viewFinalRes));
  }

#if defined(PETSC_HAVE_SAWS)
  /*
    Publish convergence information using AMS
  */
  PetscCall(PetscOptionsBool("-ksp_monitor_saws","Publish KSP progress using SAWs","KSPMonitorSet",PETSC_FALSE,&flg,&set));
  if (set && flg) {
    void *ctx;
    PetscCall(KSPMonitorSAWsCreate(ksp,&ctx));
    PetscCall(KSPMonitorSet(ksp,KSPMonitorSAWs,ctx,KSPMonitorSAWsDestroy));
    PetscCall(KSPSetComputeSingularValues(ksp,PETSC_TRUE));
  }
#endif

  /* -----------------------------------------------------------------------*/
  PetscCall(KSPSetUpNorms_Private(ksp,PETSC_FALSE,NULL,&pcside));
  PetscCall(PetscOptionsEnum("-ksp_pc_side","KSP preconditioner side","KSPSetPCSide",PCSides,(PetscEnum)pcside,(PetscEnum*)&pcside,&flg));
  if (flg) PetscCall(KSPSetPCSide(ksp,pcside));

  if (ksp->viewSV || ksp->viewEV) {
    PetscCall(KSPSetComputeSingularValues(ksp,PETSC_TRUE));
  }

#if defined(PETSC_HAVE_SAWS)
  {
    PetscBool set;
    flg  = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-ksp_saws_block","Block for SAWs at end of KSPSolve","PetscObjectSAWsBlock",((PetscObject)ksp)->amspublishblock,&flg,&set));
    if (set) {
      PetscCall(PetscObjectSAWsSetBlock((PetscObject)ksp,flg));
    }
  }
#endif

  nmax = ksp->nmax;
  PetscCall(PetscOptionsDeprecated("-ksp_matsolve_block_size","-ksp_matsolve_batch_size","3.15",NULL));
  PetscCall(PetscOptionsInt("-ksp_matsolve_batch_size", "Maximum number of columns treated simultaneously", "KSPSetMatSolveBatchSize", nmax, &nmax, &flg));
  if (flg) {
    PetscCall(KSPSetMatSolveBatchSize(ksp, nmax));
  }

  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-ksp_use_explicittranspose","Explicitly transpose the system in KSPSolveTranspose","KSPSetUseExplicitTranspose",ksp->transpose.use_explicittranspose,&flg,&set));
  if (set) {
    PetscCall(KSPSetUseExplicitTranspose(ksp,flg));
  }

  if (ksp->ops->setfromoptions) {
    PetscCall((*ksp->ops->setfromoptions)(PetscOptionsObject,ksp));
  }
  skipoptions:
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)ksp));
  PetscOptionsEnd();
  ksp->setfromoptionscalled++;
  PetscFunctionReturn(0);
}

/*@
   KSPResetFromOptions - Sets various KSP parameters from user options ONLY if the KSP was previously set from options

   Collective on ksp

   Input Parameter:
.  ksp - the KSP context

   Level: beginner

.seealso: KSPSetFromOptions(), KSPSetOptionsPrefix()
@*/
PetscErrorCode KSPResetFromOptions(KSP ksp)
{
  PetscFunctionBegin;
  if (ksp->setfromoptionscalled) PetscCall(KSPSetFromOptions(ksp));
  PetscFunctionReturn(0);
}
