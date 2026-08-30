/*
    Code for setting KSP options from the options database.
*/

#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <petscdraw.h>

/*@
  KSPSetOptionsPrefix - Sets the prefix used for searching for all
  `KSP` options in the database.

  Logically Collective

  Input Parameters:
+ ksp    - the Krylov context
- prefix - the prefix string to prepend to all `KSP` option requests

  Level: intermediate

  Notes:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the
  hyphen.

  For example, to distinguish between the runtime options for two
  different `KSP` contexts, one could call
.vb
      KSPSetOptionsPrefix(ksp1,"sys1_")
      KSPSetOptionsPrefix(ksp2,"sys2_")
.ve

  This would enable use of different options for each system, such as
.vb
      -sys1_ksp_type gmres -sys1_ksp_rtol 1.e-3
      -sys2_ksp_type bcgs  -sys2_ksp_rtol 1.e-4
.ve

.seealso: [](ch_ksp), `KSP`, `KSPAppendOptionsPrefix()`, `KSPGetOptionsPrefix()`, `KSPSetFromOptions()`
@*/
PetscErrorCode KSPSetOptionsPrefix(KSP ksp, const char prefix[])
{
  PetscBool ispcmpi;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp, &ksp->pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc, PCMPI, &ispcmpi));
  if (ispcmpi) {
    size_t     len;
    const char suffix[] = "mpi_linear_solver_server_";
    char      *newprefix;

    PetscCall(PetscStrlen(prefix, &len));
    PetscCall(PetscMalloc1(len + sizeof(suffix) + 1, &newprefix));
    PetscCall(PetscStrncpy(newprefix, prefix, len + sizeof(suffix)));
    PetscCall(PetscStrlcat(newprefix, suffix, len + sizeof(suffix)));
    PetscCall(PCSetOptionsPrefix(ksp->pc, newprefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ksp, newprefix));
    PetscCall(PetscFree(newprefix));
  } else {
    PetscCall(PCSetOptionsPrefix(ksp->pc, prefix));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ksp, prefix));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPAppendOptionsPrefix - Appends to the prefix used for searching for all
  `KSP` options in the database.

  Logically Collective

  Input Parameters:
+ ksp    - the Krylov context
- prefix - the prefix string to prepend to all `KSP` option requests

  Level: intermediate

  Note:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

.seealso: [](ch_ksp), `KSP`, `KSPSetOptionsPrefix()`, `KSPGetOptionsPrefix()`, `KSPSetFromOptions()`
@*/
PetscErrorCode KSPAppendOptionsPrefix(KSP ksp, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (!ksp->pc) PetscCall(KSPGetPC(ksp, &ksp->pc));
  PetscCall(PCAppendOptionsPrefix(ksp->pc, prefix));
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)ksp, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPSetUseFischerGuess - Use the Paul Fischer algorithm or its variants to compute initial guesses for a set of solves with related right-hand sides

  Logically Collective

  Input Parameters:
+ ksp   - the Krylov context
. model - use model 1, model 2, model 3, or any other number to turn it off
- size  - size of subspace used to generate initial guess

  Options Database Key:
. -ksp_fischer_guess model,size - uses the Fischer initial guess generator for repeated linear solves

  Level: advanced

.seealso: [](ch_ksp), `KSP`, `KSPSetOptionsPrefix()`, `KSPAppendOptionsPrefix()`, `KSPSetGuess()`, `KSPGetGuess()`, `KSPGuess`
@*/
PetscErrorCode KSPSetUseFischerGuess(KSP ksp, PetscInt model, PetscInt size)
{
  KSPGuess guess;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveInt(ksp, model, 2);
  PetscValidLogicalCollectiveInt(ksp, size, 3);
  PetscCall(KSPGetGuess(ksp, &guess));
  PetscCall(KSPGuessSetType(guess, KSPGUESSFISCHER));
  PetscCall(KSPGuessFischerSetModel(guess, model, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPSetGuess - Set the initial guess object `KSPGuess` to be used by the `KSP` object to generate initial guesses

  Logically Collective

  Input Parameters:
+ ksp   - the Krylov context
- guess - the object created with `KSPGuessCreate()`

  Level: advanced

  Notes:
  this allows a single `KSP` to be used with several different initial guess generators (likely for different linear
  solvers, see `KSPSetPC()`).

  This increases the reference count of the guess object, you must destroy the object with `KSPGuessDestroy()`
  before the end of the program.

.seealso: [](ch_ksp), `KSP`, `KSPGuess`, `KSPSetOptionsPrefix()`, `KSPAppendOptionsPrefix()`, `KSPSetUseFischerGuess()`, `KSPGetGuess()`
@*/
PetscErrorCode KSPSetGuess(KSP ksp, KSPGuess guess)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidHeaderSpecific(guess, KSPGUESS_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)guess));
  PetscCall(KSPGuessDestroy(&ksp->guess));
  ksp->guess      = guess;
  ksp->guess->ksp = ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGetGuess - Gets the initial guess generator for the `KSP`.

  Not Collective

  Input Parameter:
. ksp - the Krylov context

  Output Parameter:
. guess - the object

  Level: developer

.seealso: [](ch_ksp), `KSPGuess`, `KSP`, `KSPSetOptionsPrefix()`, `KSPAppendOptionsPrefix()`, `KSPSetUseFischerGuess()`, `KSPSetGuess()`
@*/
PetscErrorCode KSPGetGuess(KSP ksp, KSPGuess *guess)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscAssertPointer(guess, 2);
  if (!ksp->guess) {
    const char *prefix;

    PetscCall(KSPGuessCreate(PetscObjectComm((PetscObject)ksp), &ksp->guess));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ksp, &prefix));
    if (prefix) PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ksp->guess, prefix));
    ksp->guess->ksp = ksp;
  }
  *guess = ksp->guess;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGetOptionsPrefix - Gets the prefix used for searching for all
  `KSP` options in the database.

  Not Collective

  Input Parameter:
. ksp - the Krylov context

  Output Parameter:
. prefix - pointer to the prefix string used is returned

  Level: advanced

.seealso: [](ch_ksp), `KSP`, `KSPSetFromOptions()`, `KSPSetOptionsPrefix()`, `KSPAppendOptionsPrefix()`
@*/
PetscErrorCode KSPGetOptionsPrefix(KSP ksp, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ksp, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerAndFormatCreate_Internal(PetscViewer viewer, PetscViewerFormat format, PetscCtx ctx, PetscViewerAndFormat **vf)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerAndFormatCreate(viewer, format, vf));
  (*vf)->data = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPMonitorSetFromOptions - Sets a monitor function, viewer, and viewer format based on the viewer specification in the options database

  Collective

  Input Parameters:
+ ksp  - `KSP` object you wish to monitor
. opt  - the command line option for this monitor, for example `-ksp_monitor`
. name - the monitor type one is seeking, for example, `"preconditioned_residual"`
- ctx  - An optional application context for the monitor, or `NULL`

  Level: developer

  Note:
  See `PetscOptionsCreateViewer()` for details on the viewer specification, for example, `-ksp_monitor ascii:myfile::append`

.seealso: [](ch_ksp), `KSPMonitorRegister()`, `KSPMonitorSet()`, `PetscOptionsCreateViewer()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`
@*/
PetscErrorCode KSPMonitorSetFromOptions(KSP ksp, const char opt[], const char name[], PetscCtx ctx)
{
  PetscErrorCode (*mfunc)(KSP, PetscInt, PetscReal, void *);
  PetscErrorCode (*cfunc)(PetscViewer, PetscViewerFormat, void *, PetscViewerAndFormat **);
  PetscErrorCode (*dfunc)(PetscViewerAndFormat **);
  PetscViewerAndFormat *vf;
  PetscViewer           viewer;
  PetscViewerFormat     format;
  PetscViewerType       vtype;
  char                  key[PETSC_MAX_PATH_LEN];
  PetscBool             all, flg;
  const char           *prefix = NULL;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(opt, "-all_ksp_monitor", &all));
  if (!all) PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ksp, &prefix));
  PetscCall(PetscOptionsCreateViewer(PetscObjectComm((PetscObject)ksp), ((PetscObject)ksp)->options, prefix, opt, &viewer, &format, &flg));
  if (!flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscViewerGetType(viewer, &vtype));
  PetscCall(KSPMonitorMakeKey_Internal(name, vtype, format, key));
  PetscCall(PetscFunctionListFind(KSPMonitorList, key, &mfunc));
  PetscCall(PetscFunctionListFind(KSPMonitorCreateList, key, &cfunc));
  PetscCall(PetscFunctionListFind(KSPMonitorDestroyList, key, &dfunc));
  if (!cfunc) cfunc = PetscViewerAndFormatCreate_Internal;
  if (!dfunc) dfunc = PetscViewerAndFormatDestroy;

  PetscCall((*cfunc)(viewer, format, ctx, &vf));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(KSPMonitorSet(ksp, mfunc, vf, (PetscCtxDestroyFn *)dfunc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode KSPCheckPCMPI(KSP);

/*@
  KSPSetFromOptions - Sets `KSP` options from the options database.
  This routine must be called before `KSPSetUp()` if the user is to be
  allowed to set the Krylov type.

  Collective

  Input Parameter:
. ksp - the Krylov space context

  Options Database Keys:
+ -ksp_rtol rtol                                                                          - relative tolerance used in default determination of convergence, i.e.
                                                                                            if residual norm decreases by this factor than convergence is declared
. -ksp_atol abstol                                                                        - absolute tolerance used in default convergence test, i.e. if residual
                                                                                            norm is less than this then convergence is declared
. -ksp_divtol tol                                                                         - if residual norm increases by this factor than divergence is declared
. -ksp_max_it maxits                                                                      - maximum number of linear iterations
. -ksp_min_it minits                                                                      - minimum number of linear iterations to use, defaults to zero
. -ksp_reuse_preconditioner (true|false)                                                  - reuse the previously computed preconditioner
. -ksp_converged_use_initial_residual_norm (true|false)                                   - see `KSPConvergedDefaultSetUIRNorm()`
. -ksp_converged_use_min_initial_residual_norm (true|false)                               - see `KSPConvergedDefaultSetUMIRNorm()`
. -ksp_converged_maxits (true|false)                                                      - see `KSPConvergedDefaultSetConvergedMaxits()`
. -ksp_norm_type (none|preconditioned|unpreconditioned|natural)                           - see `KSPSetNormType()`
. -ksp_check_norm_iteration it                                                            - do not compute residual norm until iteration number it (does compute at 0th iteration)
                                                                                            works only for `KSPBCGS`, `KSPIBCGS`, and `KSPCG`
. -ksp_lag_norm (true|false)                                                              - compute the norm of the residual for the ith iteration on the i+1 iteration;
                                                                                            this means that one can use the norm of the residual for convergence test WITHOUT
                                                                                            an extra `MPI_Allreduce()` limiting global synchronizations.
                                                                                            This will require 1 more iteration of the solver than usual.
. -ksp_guess_type (fischer|pod)                                                           - type of initial guess generator for repeated linear solves,
                                                                                            see `KSPGuessSetFromOptions()` for additional options to control initial guess generation
. -ksp_fischer_guess model,size                                                           - uses the Fischer initial guess generator for repeated linear solves
. -ksp_constant_null_space (true|false)                                                   - assume the operator (matrix) has the constant vector in its null space
. -ksp_test_null_space (true|false)                                                       - tests if the null space associated with the linear system operator of the `KSP` is actually a null space of the operator
. -ksp_knoll (true|false)                                                                 - compute initial guess by applying the preconditioner to the right-hand side
. -ksp_orthogonalization (cgs|mgs)                                                        - use either classical (default) or modified Gram-Schmidt to orthogonalize
                                                                                            the Krylov space basis vectors
. -ksp_orthogonalization_cgs_refinement_type (refine_never|refine_ifneeded|refine_always) - determine if iterative refinement is used to increase the stability of the
                                                                                            classical Gram-Schmidt orthogonalization
. -ksp_use_explicittranspose (true|false)                                                 - transpose the system explicitly in `KSPSolveTranspose()`
. -ksp_error_if_not_converged (true|false)                                                - stop the program as soon as an error is detected in `KSPSolve()`,
                                                                                            `KSP_DIVERGED_ITS` is not treated as an error on inner solves
. -ksp_monitor_cancel (true|false)                                                        - cancel all previous convergence monitor routines set
. -ksp_monitor viewer_specification                                                       - monitor the (preconditioned) residual norm
. -ksp_monitor_true_residual viewer_specification                                         - monitor the true l2 residual norm, see `KSPMonitorTrueResidual()`
. -ksp_monitor_solution viewer_specification                                              - monitor the solution
. -ksp_monitor_singular_value viewer_specification                                        - monitor the extreme singular values
. -ksp_monitor_range viewer_specification                                                 - monitor the range of values in the (preconditioned) residual
. -ksp_monitor_max viewer_specification                                                   - monitor the maximum value in the true residual
. -ksp_monitor_error viewer_specification                                                 - monitor the error or its l2 norm, see `KSPMonitorError()`, `KSPMonitorErrorDraw()`, and `KSPMonitorErrorDrawLG()`
. -ksp_monitor_pause_final (true|false)                                                   - pauses all draw monitors at the final iterate
. -all_ksp_monitor viewer_specification                                                   - monitor the (preconditioned) residual norm for all `KSP` solves, regardless of their prefix. This is
                                                                                            useful for `PCFIELDSPLIT`, `PCMG`, etc that have inner solvers and
                                                                                            you wish to track the convergence of all the solvers.
. -ksp_view_pre viewer_specification                                                      - view the `KSP` object before each `KSPSolve()`
. -ksp_view viewer_specification                                                          - view the `KSP` object after each `KSPSolve()`
. -ksp_converged_reason viewer_specification                                              - view the convergence state at the end of the solve
. -ksp_converged_rate viewer_specification                                                - view the computed convergence rate of the iterative solver
. -ksp_view_mat viewer_specification                                                      - view the matrix defining the linear system
. -ksp_view_pmat viewer_specification                                                     - view the matrix from which the preconditioner is constructed
. -ksp_view_rhs viewer_specification                                                      - view the right hand side of the linear system
. -ksp_view_solution viewer_specification                                                 - view the computed solution
. -ksp_view_mat_explicit viewer_specification                                             - view the matrix computed explicitly via `MatComputeOperator()`, useful when the operator is provided matrix-free
. -ksp_view_eigenvalues viewer_specification                                              - view the approximate eigenvalues of the preconditioned operator computed via `KSPComputeEigenvalues()`
. -ksp_view_singularvalues viewer_specification                                           - view the approximate singular values of the preconditioned operator computed via `KSPComputeExtremeSingularValues()`
. -ksp_view_eigenvalues_explicit viewer_specification                                     - view the approximate eigenvalues of the preconditioned operator computed with LAPACK
. -ksp_view_preconditioned_operator_explicit viewer_specification                         - view the preconditioned operator computed via `KSPComputeOperator()`
. -ksp_view_final_residual viewer_specification                                           - view the final true residual norm
- -ksp_view_final_residual_vec viewer_specification                                       - view the final true residual vector, must also use `-ksp_view_final_residual ascii:`

  Level: beginner

  Notes:
  See `PetscOptionsCreateViewer()` for the values of `viewer_specification`.

  The monitors are called at every iteration.

  Except for `-ksp_view_pre` all the `-ksp_view` viewers are called at the end of `KSPSolve()`.

  To see all options, run your program with the `-help` option or consult [](ch_ksp).

.seealso: [](ch_ksp), `KSP`, `KSPSetOptionsPrefix()`, `KSPResetFromOptions()`, `KSPSetUseFischerGuess()`
@*/
PetscErrorCode KSPSetFromOptions(KSP ksp)
{
  const char *convtests[] = {"default", "skip", "lsqr"}, *orthogs[] = {"cgs", "mgs"}, *prefix;
  char        type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscBool   flg, flag, reuse, set;
  PetscInt    indx, model[2] = {0, 0}, nmax, max_it;
  KSPNormType normtype;
  PCSide      pcside;
  void       *ctx;
  MPI_Comm    comm;
  PetscReal   rtol, abstol, divtol;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);

  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)ksp, &prefix));

  PetscCall(KSPRegisterAll());
  PetscObjectOptionsBegin((PetscObject)ksp);
  PetscCall(PetscOptionsFList("-ksp_type", "Krylov method", "KSPSetType", KSPList, (char *)(((PetscObject)ksp)->type_name ? ((PetscObject)ksp)->type_name : KSPGMRES), type, sizeof(type), &flg));
  if (flg) PetscCall(KSPSetType(ksp, type));
  /*
    Set the type if it was never set.
  */
  if (!((PetscObject)ksp)->type_name) PetscCall(KSPSetType(ksp, KSPGMRES));

  PetscCall(KSPResetViewers(ksp));

  /* Cancels all monitors hardwired into code before call to KSPSetFromOptions() */
  PetscCall(PetscOptionsBool("-ksp_monitor_cancel", "Remove any hardwired monitor routines", "KSPMonitorCancel", PETSC_FALSE, &flg, &set));
  if (set && flg) PetscCall(KSPMonitorCancel(ksp));
  PetscCall(PetscOptionsDeprecated("-ksp_monitor_short", "-ksp_monitor", "3.26", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor", "preconditioned_residual", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-all_ksp_monitor", "preconditioned_residual", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_range", "preconditioned_residual_range", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_true_residual", "true_residual", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_max", "true_residual_max", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_solution", "solution", NULL));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_singular_value", "singular_value", ksp));
  PetscCall(KSPMonitorSetFromOptions(ksp, "-ksp_monitor_error", "error", ksp));
  PetscCall(PetscOptionsBool("-ksp_monitor_pause_final", "Pauses all draw monitors at the final iterate", "KSPMonitorPauseFinal_Internal", PETSC_FALSE, &ksp->pauseFinal, NULL));
  PetscCall(PetscOptionsBool("-ksp_initial_guess_nonzero", "Use the contents of the solution vector for initial guess", "KSPSetInitialNonzero", ksp->guess_zero ? PETSC_FALSE : PETSC_TRUE, &flag, &flg));
  if (flg) PetscCall(KSPSetInitialGuessNonzero(ksp, flag));

  PetscCall(KSPGetReusePreconditioner(ksp, &reuse));
  PetscCall(PetscOptionsBool("-ksp_reuse_preconditioner", "Use initial preconditioner and don't ever compute a new one", "KSPReusePreconditioner", reuse, &reuse, NULL));
  PetscCall(KSPSetReusePreconditioner(ksp, reuse));
  PetscCall(PetscOptionsBool("-ksp_error_if_not_converged", "Generate error if solver does not converge", "KSPSetErrorIfNotConverged", ksp->errorifnotconverged, &ksp->errorifnotconverged, &set));
  if (set) PetscCall(KSPSetErrorIfNotConverged(ksp, ksp->errorifnotconverged));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view", &ksp->viewer, &ksp->format, &ksp->view));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_pre", &ksp->viewerPre, &ksp->formatPre, &ksp->viewPre));
  PetscCall(PetscViewerDestroy(&ksp->convergedreasonviewer));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, ((PetscObject)ksp)->prefix, "-ksp_converged_reason", &ksp->convergedreasonviewer, &ksp->convergedreasonformat, NULL));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-ksp_converged_reason_view_cancel", "Cancel all the converged reason view functions set using KSPConvergedReasonViewSet", "KSPConvergedReasonViewCancel", PETSC_FALSE, &flg, &set));
  if (set && flg) PetscCall(KSPConvergedReasonViewCancel(ksp));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_mat", &ksp->viewerMat, &ksp->formatMat, &ksp->viewMat));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_pmat", &ksp->viewerPMat, &ksp->formatPMat, &ksp->viewPMat));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_rhs", &ksp->viewerRhs, &ksp->formatRhs, &ksp->viewRhs));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_solution", &ksp->viewerSol, &ksp->formatSol, &ksp->viewSol));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_mat_explicit", &ksp->viewerMatExp, &ksp->formatMatExp, &ksp->viewMatExp));
  PetscCall(PetscOptionsDeprecated("-ksp_final_residual", "-ksp_view_final_residual", "3.9", NULL));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_final_residual", &ksp->viewerFinalRes, &ksp->formatFinalRes, &ksp->viewFinalRes));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_preconditioned_operator_explicit", &ksp->viewerPOpExp, &ksp->formatPOpExp, &ksp->viewPOpExp));
  nmax = ksp->nmax;
  PetscCall(PetscOptionsDeprecated("-ksp_matsolve_block_size", "-ksp_matsolve_batch_size", "3.15", NULL));
  PetscCall(PetscOptionsInt("-ksp_matsolve_batch_size", "Maximum number of columns treated simultaneously", "KSPSetMatSolveBatchSize", nmax, &nmax, &flg));
  if (flg) PetscCall(KSPSetMatSolveBatchSize(ksp, nmax));
  PetscCall(PetscObjectTypeCompare((PetscObject)ksp, KSPPREONLY, &flg));
  if (flg) goto skipoptions;

  rtol   = ksp->rtol;
  abstol = ksp->abstol;
  divtol = ksp->divtol;
  max_it = ksp->max_it;
  PetscCall(PetscOptionsReal("-ksp_rtol", "Relative decrease in residual norm", "KSPSetTolerances", ksp->rtol, &rtol, NULL));
  PetscCall(PetscOptionsReal("-ksp_atol", "Absolute value of residual norm", "KSPSetTolerances", ksp->abstol, &abstol, NULL));
  PetscCall(PetscOptionsReal("-ksp_divtol", "Residual norm increase cause divergence", "KSPSetTolerances", ksp->divtol, &divtol, NULL));
  PetscCall(PetscOptionsInt("-ksp_max_it", "Maximum number of iterations", "KSPSetTolerances", ksp->max_it, &max_it, &flg));
  PetscCall(KSPSetTolerances(ksp, rtol, abstol, divtol, max_it));
  PetscCall(PetscOptionsRangeInt("-ksp_min_it", "Minimum number of iterations", "KSPSetMinimumIterations", ksp->min_it, &ksp->min_it, NULL, 0, ksp->max_it));

  PetscCall(PetscOptionsBool("-ksp_converged_use_initial_residual_norm", "Use initial residual norm for computing relative convergence", "KSPConvergedDefaultSetUIRNorm", PETSC_FALSE, &flag, &set));
  if (set && flag) PetscCall(KSPConvergedDefaultSetUIRNorm(ksp));
  PetscCall(PetscOptionsBool("-ksp_converged_use_min_initial_residual_norm", "Use minimum of initial residual norm and b for computing relative convergence", "KSPConvergedDefaultSetUMIRNorm", PETSC_FALSE, &flag, &set));
  if (set && flag) PetscCall(KSPConvergedDefaultSetUMIRNorm(ksp));
  PetscCall(PetscOptionsBool("-ksp_converged_maxits", "Declare convergence if the maximum number of iterations is reached", "KSPConvergedDefaultSetConvergedMaxits", PETSC_FALSE, &flag, &set));
  if (set) PetscCall(KSPConvergedDefaultSetConvergedMaxits(ksp, flag));
  PetscCall(KSPGetConvergedNegativeCurvature(ksp, &flag));
  PetscCall(PetscOptionsBool("-ksp_converged_neg_curve", "Declare convergence if negative curvature is detected", "KSPConvergedNegativeCurvature", flag, &flag, &set));
  if (set) PetscCall(KSPSetConvergedNegativeCurvature(ksp, flag));

  PetscCall(PetscOptionsBool("-ksp_knoll", "Use preconditioner applied to b for initial guess", "KSPSetInitialGuessKnoll", ksp->guess_knoll, &ksp->guess_knoll, NULL));
  PetscCall(PetscOptionsFList("-ksp_guess_type", "Initial guess in Krylov method", NULL, KSPGuessList, NULL, type, sizeof(type), &flg));
  if (flg) {
    PetscCall(KSPGetGuess(ksp, &ksp->guess));
    PetscCall(KSPGuessSetType(ksp->guess, type));
    PetscCall(KSPGuessSetFromOptions(ksp->guess));
  } else { /* old option for KSP */
    nmax = 2;
    PetscCall(PetscOptionsIntArray("-ksp_fischer_guess", "Use Paul Fischer's algorithm or its variants for initial guess", "KSPSetUseFischerGuess", model, &nmax, &flag));
    if (flag) {
      PetscCheck(nmax == 2, comm, PETSC_ERR_ARG_OUTOFRANGE, "Must pass in model,size as arguments");
      PetscCall(KSPSetUseFischerGuess(ksp, model[0], model[1]));
    }
  }

  PetscCall(PetscOptionsEList("-ksp_convergence_test", "Convergence test", "KSPSetConvergenceTest", convtests, 3, "default", &indx, &flg));
  if (flg) {
    switch (indx) {
    case 0:
      PetscCall(KSPConvergedDefaultCreate(&ctx));
      PetscCall(KSPSetConvergenceTest(ksp, KSPConvergedDefault, ctx, KSPConvergedDefaultDestroy));
      break;
    case 1:
      PetscCall(KSPSetConvergenceTest(ksp, KSPConvergedSkip, NULL, NULL));
      break;
    case 2:
      PetscCall(KSPConvergedDefaultCreate(&ctx));
      PetscCall(KSPSetConvergenceTest(ksp, KSPLSQRConvergedDefault, ctx, KSPConvergedDefaultDestroy));
      break;
    }
  }

  PetscCall(KSPSetUpNorms_Private(ksp, PETSC_FALSE, &normtype, NULL));
  PetscCall(PetscOptionsEnum("-ksp_norm_type", "KSP Norm type", "KSPSetNormType", KSPNormTypes, (PetscEnum)normtype, (PetscEnum *)&normtype, &flg));
  if (flg) PetscCall(KSPSetNormType(ksp, normtype));

  PetscCall(PetscOptionsInt("-ksp_check_norm_iteration", "First iteration to compute residual norm", "KSPSetCheckNormIteration", ksp->chknorm, &ksp->chknorm, NULL));

  PetscCall(PetscOptionsBool("-ksp_lag_norm", "Lag the calculation of the residual norm", "KSPSetLagNorm", ksp->lagnorm, &flag, &flg));
  if (flg) PetscCall(KSPSetLagNorm(ksp, flag));

  PetscCall(PetscOptionsBool("-ksp_constant_null_space", "Add constant null space to Krylov solver matrix", "MatSetNullSpace", PETSC_FALSE, &flg, &set));
  if (set && flg) {
    MatNullSpace nsp;
    Mat          Amat = NULL;

    PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nsp));
    if (ksp->pc) PetscCall(PCGetOperators(ksp->pc, &Amat, NULL));
    PetscCheck(Amat, comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot set nullspace, matrix has not yet been provided");
    PetscCall(MatSetNullSpace(Amat, nsp));
    PetscCall(MatNullSpaceDestroy(&nsp));
  }

  PetscCall(PetscOptionsDeprecated("-ksp_gmres_classicalgramschmidt", NULL, "3.26", "Use -ksp_orthogonalization cgs"));
  PetscCall(PetscOptionsDeprecated("-ksp_gmres_modifiedgramschmidt", NULL, "3.26", "Use -ksp_orthogonalization mgs"));
  PetscCall(PetscOptionsGetBool(((PetscObject)ksp)->options, prefix, "-ksp_gmres_classicalgramschmidt", &flg, &set));
  if (set && flg) PetscCall(KSPOrthogonalizationSet(ksp, KSPOrthogonalizationClassicalGramSchmidt));
  PetscCall(PetscOptionsGetBool(((PetscObject)ksp)->options, prefix, "-ksp_gmres_modifiedgramschmidt", &flg, &set));
  if (set && flg) PetscCall(KSPOrthogonalizationSet(ksp, KSPOrthogonalizationModifiedGramSchmidt));
  PetscCall(PetscOptionsEList("-ksp_orthogonalization", "Orthogonalization method", "KSPOrthogonalizationSet", orthogs, 2, "cgs", &indx, &flg));
  if (flg) {
    switch (indx) {
    case 0:
      PetscCall(KSPOrthogonalizationSet(ksp, KSPOrthogonalizationClassicalGramSchmidt));
      break;
    case 1:
      PetscCall(KSPOrthogonalizationSet(ksp, KSPOrthogonalizationModifiedGramSchmidt));
      break;
    }
  }
  PetscCall(PetscOptionsDeprecated("-ksp_gmres_cgs_refinement_type", "-ksp_orthogonalization_cgs_refinement_type", "3.26", NULL));
  PetscCall(PetscOptionsEnum("-ksp_orthogonalization_cgs_refinement_type", "Type of iterative refinement for classical (unmodified) Gram-Schmidt", "KSPOrthogonalizationSetCGSRefinementType", KSPOrthogonalizationCGSRefinementTypes, (PetscEnum)ksp->cgstype,
                             (PetscEnum *)&ksp->cgstype, &flg));

  flg = PETSC_FALSE;
  if (ksp->pc) {
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc, PCKSP, &flg));
    if (!flg) PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc, PCBJACOBI, &flg));
    if (!flg) PetscCall(PetscObjectTypeCompare((PetscObject)ksp->pc, PCDEFLATION, &flg));
  }

  if (flg) {
    /* Using dynamic tolerance in preconditioner */
    PetscCall(PetscOptionsString("-sub_ksp_dynamic_tolerance", "Use dynamic tolerance for inner PC", "KSPMonitorDynamicTolerance", "stdout", monfilename, sizeof(monfilename), &flg));
    if (flg) {
      void     *scale;
      PetscReal coeff = 1.0;

      PetscCall(KSPMonitorDynamicToleranceCreate(&scale));
      PetscCall(PetscOptionsReal("-sub_ksp_dynamic_tolerance", "Coefficient of dynamic tolerance for inner PC", "KSPMonitorDynamicTolerance", coeff, &coeff, &flg));
      if (flg) PetscCall(KSPMonitorDynamicToleranceSetCoefficient(scale, coeff));
      PetscCall(KSPMonitorSet(ksp, KSPMonitorDynamicTolerance, scale, KSPMonitorDynamicToleranceDestroy));
    }
  }

  /*
   Calls Python function
  */
  PetscCall(PetscOptionsString("-ksp_monitor_python", "Use Python function", "KSPMonitorSet", NULL, monfilename, sizeof(monfilename), &flg));
  if (flg) PetscCall(PetscPythonMonitorSet((PetscObject)ksp, monfilename));
  /*
    Graphically plots preconditioned residual norm and range of residual element values
  */
  PetscCall(PetscOptionsBool("-ksp_monitor_lg_range", "Monitor graphically range of preconditioned residual norm", "KSPMonitorSet", PETSC_FALSE, &flg, &set));
  if (set && flg) {
    PetscViewer ctx;

    PetscCall(PetscViewerDrawOpen(comm, NULL, NULL, PETSC_DECIDE, PETSC_DECIDE, 400, 300, &ctx));
    PetscCall(KSPMonitorSet(ksp, KSPMonitorLGRange, ctx, (PetscCtxDestroyFn *)PetscViewerDestroy));
  }
  /* TODO Do these show up in help? */
  PetscCall(PetscOptionsHasName(((PetscObject)ksp)->options, prefix, "-ksp_converged_rate", &flg));
  if (flg) {
    const char *RateTypes[] = {"default", "residual", "error", "PetscRateType", "RATE_", NULL};
    PetscEnum   rtype       = (PetscEnum)1;

    PetscCall(PetscOptionsGetEnum(((PetscObject)ksp)->options, prefix, "-ksp_converged_rate_type", RateTypes, &rtype, &flg));
    if (rtype == (PetscEnum)0 || rtype == (PetscEnum)1) PetscCall(KSPSetResidualHistory(ksp, NULL, PETSC_DETERMINE, PETSC_TRUE));
    if (rtype == (PetscEnum)0 || rtype == (PetscEnum)2) PetscCall(KSPSetErrorHistory(ksp, NULL, PETSC_DETERMINE, PETSC_TRUE));
  }

  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_converged_rate", &ksp->viewerRate, &ksp->formatRate, &ksp->viewRate));
  PetscCall(PetscOptionsDeprecated("-ksp_compute_eigenvalues", "-ksp_view_eigenvalues", "3.9", NULL));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_eigenvalues", &ksp->viewerEV, &ksp->formatEV, &ksp->viewEV));
  PetscCall(PetscOptionsDeprecated("-ksp_compute_singularvalues", "-ksp_view_singularvalues", "3.9", NULL));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_singularvalues", &ksp->viewerSV, &ksp->formatSV, &ksp->viewSV));
  PetscCall(PetscOptionsDeprecated("-ksp_compute_eigenvalues_explicitly", "-ksp_view_eigenvalues_explicit", "3.9", NULL));
  PetscCall(PetscOptionsCreateViewer(comm, ((PetscObject)ksp)->options, prefix, "-ksp_view_eigenvalues_explicit", &ksp->viewerEVExp, &ksp->formatEVExp, &ksp->viewEVExp));

#if PetscDefined(HAVE_SAWS)
  /*
    Publish convergence information using AMS
  */
  PetscCall(PetscOptionsBool("-ksp_monitor_saws", "Publish KSP progress using SAWs", "KSPMonitorSet", PETSC_FALSE, &flg, &set));
  if (set && flg) {
    PetscCtx ctx;
    PetscCall(KSPMonitorSAWsCreate(ksp, &ctx));
    PetscCall(KSPMonitorSet(ksp, KSPMonitorSAWs, ctx, KSPMonitorSAWsDestroy));
    PetscCall(KSPSetComputeSingularValues(ksp, PETSC_TRUE));
  }
#endif

  PetscCall(KSPSetUpNorms_Private(ksp, PETSC_FALSE, NULL, &pcside));
  PetscCall(PetscOptionsEnum("-ksp_pc_side", "KSP preconditioner side", "KSPSetPCSide", PCSides, (PetscEnum)pcside, (PetscEnum *)&pcside, &flg));
  if (flg) PetscCall(KSPSetPCSide(ksp, pcside));

  if (ksp->viewSV || ksp->viewEV) PetscCall(KSPSetComputeSingularValues(ksp, PETSC_TRUE));

#if PetscDefined(HAVE_SAWS)
  {
    PetscBool set;
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-ksp_saws_block", "Block for SAWs at end of KSPSolve", "PetscObjectSAWsBlock", ((PetscObject)ksp)->amspublishblock, &flg, &set));
    if (set) PetscCall(PetscObjectSAWsSetBlock((PetscObject)ksp, flg));
  }
#endif

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-ksp_use_explicittranspose", "Explicitly transpose the system in KSPSolveTranspose() and KSPMatSolveTranspose()", "KSPSetUseExplicitTranspose", ksp->transpose.use_explicittranspose, &flg, &set));
  if (set) PetscCall(KSPSetUseExplicitTranspose(ksp, flg));

  PetscTryTypeMethod(ksp, setfromoptions, PetscOptionsObject);
skipoptions:
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)ksp, PetscOptionsObject));
  PetscOptionsEnd();
  ksp->setfromoptionscalled++;

  if (!ksp->pc) PetscCall(KSPGetPC(ksp, &ksp->pc));
  if (!ksp->skippcsetfromoptions) PetscCall(PCSetFromOptions(ksp->pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPResetFromOptions - Sets `KSP` parameters from user options ONLY if the `KSP` was previously set from options

  Collective

  Input Parameter:
. ksp - the `KSP` context

  Level: advanced

.seealso: [](ch_ksp), `KSPSetFromOptions()`, `KSPSetOptionsPrefix()`
@*/
PetscErrorCode KSPResetFromOptions(KSP ksp)
{
  PetscFunctionBegin;
  if (ksp->setfromoptionscalled) PetscCall(KSPSetFromOptions(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
