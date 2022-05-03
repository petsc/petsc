#define TAO_DLL

#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

PetscBool TaoRegisterAllCalled = PETSC_FALSE;
PetscFunctionList TaoList = NULL;

PetscClassId TAO_CLASSID;

PetscLogEvent TAO_Solve;
PetscLogEvent TAO_ObjectiveEval;
PetscLogEvent TAO_GradientEval;
PetscLogEvent TAO_ObjGradEval;
PetscLogEvent TAO_HessianEval;
PetscLogEvent TAO_JacobianEval;
PetscLogEvent TAO_ConstraintsEval;

const char *TaoSubSetTypes[] = {"subvec","mask","matrixfree","TaoSubSetType","TAO_SUBSET_",NULL};

struct _n_TaoMonitorDrawCtx {
  PetscViewer viewer;
  PetscInt    howoften;  /* when > 0 uses iteration % howoften, when negative only final solution plotted */
};

/*@
  TaoCreate - Creates a TAO solver

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. newtao - the new Tao context

  Available methods include:
+    nls - Newton's method with line search for unconstrained minimization
.    ntr - Newton's method with trust region for unconstrained minimization
.    ntl - Newton's method with trust region, line search for unconstrained minimization
.    lmvm - Limited memory variable metric method for unconstrained minimization
.    cg - Nonlinear conjugate gradient method for unconstrained minimization
.    nm - Nelder-Mead algorithm for derivate-free unconstrained minimization
.    tron - Newton Trust Region method for bound constrained minimization
.    gpcg - Newton Trust Region method for quadratic bound constrained minimization
.    blmvm - Limited memory variable metric method for bound constrained minimization
.    lcl - Linearly constrained Lagrangian method for pde-constrained minimization
-    pounders - Model-based algorithm for nonlinear least squares

   Options Database Keys:
.   -tao_type - select which method TAO should use

   Level: beginner

.seealso: `TaoSolve()`, `TaoDestroy()`
@*/
PetscErrorCode TaoCreate(MPI_Comm comm, Tao *newtao)
{
  Tao            tao;

  PetscFunctionBegin;
  PetscValidPointer(newtao,2);
  PetscCall(TaoInitializePackage());
  PetscCall(TaoLineSearchInitializePackage());
  PetscCall(PetscHeaderCreate(tao,TAO_CLASSID,"Tao","Optimization solver","Tao",comm,TaoDestroy,TaoView));

  /* Set non-NULL defaults */
  tao->ops->convergencetest = TaoDefaultConvergenceTest;

  tao->max_it    = 10000;
  tao->max_funcs = -1;
#if defined(PETSC_USE_REAL_SINGLE)
  tao->gatol     = 1e-5;
  tao->grtol     = 1e-5;
  tao->crtol     = 1e-5;
  tao->catol     = 1e-5;
#else
  tao->gatol     = 1e-8;
  tao->grtol     = 1e-8;
  tao->crtol     = 1e-8;
  tao->catol     = 1e-8;
#endif
  tao->gttol     = 0.0;
  tao->steptol   = 0.0;
  tao->trust0    = PETSC_INFINITY;
  tao->fmin      = PETSC_NINFINITY;

  tao->hist_reset  = PETSC_TRUE;

  PetscCall(TaoResetStatistics(tao));
  *newtao = tao;
  PetscFunctionReturn(0);
}

/*@
  TaoSolve - Solves an optimization problem min F(x) s.t. l <= x <= u

  Collective on Tao

  Input Parameters:
. tao - the Tao context

  Notes:
  The user must set up the Tao with calls to TaoSetSolution(),
  TaoSetObjective(),
  TaoSetGradient(), and (if using 2nd order method) TaoSetHessian().

  You should call TaoGetConvergedReason() or run with -tao_converged_reason to determine if the optimization algorithm actually succeeded or
  why it failed.

  Level: beginner

.seealso: `TaoCreate()`, `TaoSetObjective()`, `TaoSetGradient()`, `TaoSetHessian()`, `TaoGetConvergedReason()`
 @*/
PetscErrorCode TaoSolve(Tao tao)
{
  static PetscBool set = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(PetscCitationsRegister("@TechReport{tao-user-ref,\n"
                                   "title   = {Toolkit for Advanced Optimization (TAO) Users Manual},\n"
                                   "author  = {Todd Munson and Jason Sarich and Stefan Wild and Steve Benson and Lois Curfman McInnes},\n"
                                   "Institution = {Argonne National Laboratory},\n"
                                   "Year   = 2014,\n"
                                   "Number = {ANL/MCS-TM-322 - Revision 3.5},\n"
                                   "url    = {https://www.mcs.anl.gov/research/projects/tao/}\n}\n",&set));
  tao->header_printed = PETSC_FALSE;
  PetscCall(TaoSetUp(tao));
  PetscCall(TaoResetStatistics(tao));
  if (tao->linesearch) {
    PetscCall(TaoLineSearchReset(tao->linesearch));
  }

  PetscCall(PetscLogEventBegin(TAO_Solve,tao,0,0,0));
  if (tao->ops->solve) PetscCall((*tao->ops->solve)(tao));
  PetscCall(PetscLogEventEnd(TAO_Solve,tao,0,0,0));

  PetscCall(VecViewFromOptions(tao->solution,(PetscObject)tao,"-tao_view_solution"));

  tao->ntotalits += tao->niter;
  PetscCall(TaoViewFromOptions(tao,NULL,"-tao_view"));

  if (tao->printreason) {
    if (tao->reason > 0) {
      PetscCall(PetscPrintf(((PetscObject)tao)->comm,"TAO solve converged due to %s iterations %" PetscInt_FMT "\n",TaoConvergedReasons[tao->reason],tao->niter));
    } else {
      PetscCall(PetscPrintf(((PetscObject)tao)->comm,"TAO solve did not converge due to %s iteration %" PetscInt_FMT "\n",TaoConvergedReasons[tao->reason],tao->niter));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  TaoSetUp - Sets up the internal data structures for the later use
  of a Tao solver

  Collective on tao

  Input Parameters:
. tao - the TAO context

  Notes:
  The user will not need to explicitly call TaoSetUp(), as it will
  automatically be called in TaoSolve().  However, if the user
  desires to call it explicitly, it should come after TaoCreate()
  and any TaoSetSomething() routines, but before TaoSolve().

  Level: advanced

.seealso: `TaoCreate()`, `TaoSolve()`
@*/
PetscErrorCode TaoSetUp(Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID,1);
  if (tao->setupcalled) PetscFunctionReturn(0);
  PetscCheck(tao->solution,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_WRONGSTATE,"Must call TaoSetSolution");
  if (tao->ops->setup) {
    PetscCall((*tao->ops->setup)(tao));
  }
  tao->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  TaoDestroy - Destroys the TAO context that was created with
  TaoCreate()

  Collective on Tao

  Input Parameter:
. tao - the Tao context

  Level: beginner

.seealso: `TaoCreate()`, `TaoSolve()`
@*/
PetscErrorCode TaoDestroy(Tao *tao)
{
  PetscFunctionBegin;
  if (!*tao) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*tao,TAO_CLASSID,1);
  if (--((PetscObject)*tao)->refct > 0) {*tao = NULL;PetscFunctionReturn(0);}

  if ((*tao)->ops->destroy) {
    PetscCall((*((*tao))->ops->destroy)(*tao));
  }
  PetscCall(KSPDestroy(&(*tao)->ksp));
  PetscCall(TaoLineSearchDestroy(&(*tao)->linesearch));

  if ((*tao)->ops->convergencedestroy) {
    PetscCall((*(*tao)->ops->convergencedestroy)((*tao)->cnvP));
    if ((*tao)->jacobian_state_inv) {
      PetscCall(MatDestroy(&(*tao)->jacobian_state_inv));
    }
  }
  PetscCall(VecDestroy(&(*tao)->solution));
  PetscCall(VecDestroy(&(*tao)->gradient));
  PetscCall(VecDestroy(&(*tao)->ls_res));

  if ((*tao)->gradient_norm) {
    PetscCall(PetscObjectDereference((PetscObject)(*tao)->gradient_norm));
    PetscCall(VecDestroy(&(*tao)->gradient_norm_tmp));
  }

  PetscCall(VecDestroy(&(*tao)->XL));
  PetscCall(VecDestroy(&(*tao)->XU));
  PetscCall(VecDestroy(&(*tao)->IL));
  PetscCall(VecDestroy(&(*tao)->IU));
  PetscCall(VecDestroy(&(*tao)->DE));
  PetscCall(VecDestroy(&(*tao)->DI));
  PetscCall(VecDestroy(&(*tao)->constraints));
  PetscCall(VecDestroy(&(*tao)->constraints_equality));
  PetscCall(VecDestroy(&(*tao)->constraints_inequality));
  PetscCall(VecDestroy(&(*tao)->stepdirection));
  PetscCall(MatDestroy(&(*tao)->hessian_pre));
  PetscCall(MatDestroy(&(*tao)->hessian));
  PetscCall(MatDestroy(&(*tao)->ls_jac));
  PetscCall(MatDestroy(&(*tao)->ls_jac_pre));
  PetscCall(MatDestroy(&(*tao)->jacobian_pre));
  PetscCall(MatDestroy(&(*tao)->jacobian));
  PetscCall(MatDestroy(&(*tao)->jacobian_state_pre));
  PetscCall(MatDestroy(&(*tao)->jacobian_state));
  PetscCall(MatDestroy(&(*tao)->jacobian_state_inv));
  PetscCall(MatDestroy(&(*tao)->jacobian_design));
  PetscCall(MatDestroy(&(*tao)->jacobian_equality));
  PetscCall(MatDestroy(&(*tao)->jacobian_equality_pre));
  PetscCall(MatDestroy(&(*tao)->jacobian_inequality));
  PetscCall(MatDestroy(&(*tao)->jacobian_inequality_pre));
  PetscCall(ISDestroy(&(*tao)->state_is));
  PetscCall(ISDestroy(&(*tao)->design_is));
  PetscCall(VecDestroy(&(*tao)->res_weights_v));
  PetscCall(TaoCancelMonitors(*tao));
  if ((*tao)->hist_malloc) {
    PetscCall(PetscFree4((*tao)->hist_obj,(*tao)->hist_resid,(*tao)->hist_cnorm,(*tao)->hist_lits));
  }
  if ((*tao)->res_weights_n) {
    PetscCall(PetscFree((*tao)->res_weights_rows));
    PetscCall(PetscFree((*tao)->res_weights_cols));
    PetscCall(PetscFree((*tao)->res_weights_w));
  }
  PetscCall(PetscHeaderDestroy(tao));
  PetscFunctionReturn(0);
}

/*@
  TaoSetFromOptions - Sets various Tao parameters from user
  options.

  Collective on Tao

  Input Parameter:
. tao - the Tao solver context

  options Database Keys:
+ -tao_type <type> - The algorithm that TAO uses (lmvm, nls, etc.)
. -tao_gatol <gatol> - absolute error tolerance for ||gradient||
. -tao_grtol <grtol> - relative error tolerance for ||gradient||
. -tao_gttol <gttol> - reduction of ||gradient|| relative to initial gradient
. -tao_max_it <max> - sets maximum number of iterations
. -tao_max_funcs <max> - sets maximum number of function evaluations
. -tao_fmin <fmin> - stop if function value reaches fmin
. -tao_steptol <tol> - stop if trust region radius less than <tol>
. -tao_trust0 <t> - initial trust region radius
. -tao_monitor - prints function value and residual at each iteration
. -tao_smonitor - same as tao_monitor, but truncates very small values
. -tao_cmonitor - prints function value, residual, and constraint norm at each iteration
. -tao_view_solution - prints solution vector at each iteration
. -tao_view_ls_residual - prints least-squares residual vector at each iteration
. -tao_view_stepdirection - prints step direction vector at each iteration
. -tao_view_gradient - prints gradient vector at each iteration
. -tao_draw_solution - graphically view solution vector at each iteration
. -tao_draw_step - graphically view step vector at each iteration
. -tao_draw_gradient - graphically view gradient at each iteration
. -tao_fd_gradient - use gradient computed with finite differences
. -tao_fd_hessian - use hessian computed with finite differences
. -tao_mf_hessian - use matrix-free hessian computed with finite differences
. -tao_cancelmonitors - cancels all monitors (except those set with command line)
. -tao_view - prints information about the Tao after solving
- -tao_converged_reason - prints the reason TAO stopped iterating

  Notes:
  To see all options, run your program with the -help option or consult the
  user's manual. Should be called after TaoCreate() but before TaoSolve()

  Level: beginner
@*/
PetscErrorCode TaoSetFromOptions(Tao tao)
{
  TaoType        default_type = TAOLMVM;
  char           type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer    monviewer;
  PetscBool      flg;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(PetscObjectGetComm((PetscObject)tao,&comm));

  /* So no warnings are given about unused options */
  PetscCall(PetscOptionsHasName(((PetscObject)tao)->options,((PetscObject)tao)->prefix,"-tao_ls_type",&flg));

  PetscObjectOptionsBegin((PetscObject)tao);
  {
    if (((PetscObject)tao)->type_name) default_type = ((PetscObject)tao)->type_name;
    /* Check for type from options */
    PetscCall(PetscOptionsFList("-tao_type","Tao Solver type","TaoSetType",TaoList,default_type,type,256,&flg));
    if (flg) {
      PetscCall(TaoSetType(tao,type));
    } else if (!((PetscObject)tao)->type_name) {
      PetscCall(TaoSetType(tao,default_type));
    }

    PetscCall(PetscOptionsReal("-tao_catol","Stop if constraints violations within","TaoSetConstraintTolerances",tao->catol,&tao->catol,&flg));
    if (flg) tao->catol_changed = PETSC_TRUE;
    PetscCall(PetscOptionsReal("-tao_crtol","Stop if relative contraint violations within","TaoSetConstraintTolerances",tao->crtol,&tao->crtol,&flg));
    if (flg) tao->crtol_changed = PETSC_TRUE;
    PetscCall(PetscOptionsReal("-tao_gatol","Stop if norm of gradient less than","TaoSetTolerances",tao->gatol,&tao->gatol,&flg));
    if (flg) tao->gatol_changed = PETSC_TRUE;
    PetscCall(PetscOptionsReal("-tao_grtol","Stop if norm of gradient divided by the function value is less than","TaoSetTolerances",tao->grtol,&tao->grtol,&flg));
    if (flg) tao->grtol_changed = PETSC_TRUE;
    PetscCall(PetscOptionsReal("-tao_gttol","Stop if the norm of the gradient is less than the norm of the initial gradient times tol","TaoSetTolerances",tao->gttol,&tao->gttol,&flg));
    if (flg) tao->gttol_changed = PETSC_TRUE;
    PetscCall(PetscOptionsInt("-tao_max_it","Stop if iteration number exceeds","TaoSetMaximumIterations",tao->max_it,&tao->max_it,&flg));
    if (flg) tao->max_it_changed = PETSC_TRUE;
    PetscCall(PetscOptionsInt("-tao_max_funcs","Stop if number of function evaluations exceeds","TaoSetMaximumFunctionEvaluations",tao->max_funcs,&tao->max_funcs,&flg));
    if (flg) tao->max_funcs_changed = PETSC_TRUE;
    PetscCall(PetscOptionsReal("-tao_fmin","Stop if function less than","TaoSetFunctionLowerBound",tao->fmin,&tao->fmin,&flg));
    if (flg) tao->fmin_changed = PETSC_TRUE;
    PetscCall(PetscOptionsReal("-tao_steptol","Stop if step size or trust region radius less than","",tao->steptol,&tao->steptol,&flg));
    if (flg) tao->steptol_changed = PETSC_TRUE;
    PetscCall(PetscOptionsReal("-tao_trust0","Initial trust region radius","TaoSetTrustRegionRadius",tao->trust0,&tao->trust0,&flg));
    if (flg) tao->trust0_changed = PETSC_TRUE;
    PetscCall(PetscOptionsString("-tao_view_solution","view solution vector after each evaluation","TaoSetMonitor","stdout",monfilename,sizeof(monfilename),&flg));
    if (flg) {
      PetscCall(PetscViewerASCIIOpen(comm,monfilename,&monviewer));
      PetscCall(TaoSetMonitor(tao,TaoSolutionMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy));
    }

    PetscCall(PetscOptionsBool("-tao_converged_reason","Print reason for TAO converged","TaoSolve",tao->printreason,&tao->printreason,NULL));
    PetscCall(PetscOptionsString("-tao_view_gradient","view gradient vector after each evaluation","TaoSetMonitor","stdout",monfilename,sizeof(monfilename),&flg));
    if (flg) {
      PetscCall(PetscViewerASCIIOpen(comm,monfilename,&monviewer));
      PetscCall(TaoSetMonitor(tao,TaoGradientMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy));
    }

    PetscCall(PetscOptionsString("-tao_view_stepdirection","view step direction vector after each iteration","TaoSetMonitor","stdout",monfilename,sizeof(monfilename),&flg));
    if (flg) {
      PetscCall(PetscViewerASCIIOpen(comm,monfilename,&monviewer));
      PetscCall(TaoSetMonitor(tao,TaoStepDirectionMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy));
    }

    PetscCall(PetscOptionsString("-tao_view_residual","view least-squares residual vector after each evaluation","TaoSetMonitor","stdout",monfilename,sizeof(monfilename),&flg));
    if (flg) {
      PetscCall(PetscViewerASCIIOpen(comm,monfilename,&monviewer));
      PetscCall(TaoSetMonitor(tao,TaoResidualMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy));
    }

    PetscCall(PetscOptionsString("-tao_monitor","Use the default convergence monitor","TaoSetMonitor","stdout",monfilename,sizeof(monfilename),&flg));
    if (flg) {
      PetscCall(PetscViewerASCIIOpen(comm,monfilename,&monviewer));
      PetscCall(TaoSetMonitor(tao,TaoMonitorDefault,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy));
    }

    PetscCall(PetscOptionsString("-tao_gmonitor","Use the convergence monitor with extra globalization info","TaoSetMonitor","stdout",monfilename,sizeof(monfilename),&flg));
    if (flg) {
      PetscCall(PetscViewerASCIIOpen(comm,monfilename,&monviewer));
      PetscCall(TaoSetMonitor(tao,TaoDefaultGMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy));
    }

    PetscCall(PetscOptionsString("-tao_smonitor","Use the short convergence monitor","TaoSetMonitor","stdout",monfilename,sizeof(monfilename),&flg));
    if (flg) {
      PetscCall(PetscViewerASCIIOpen(comm,monfilename,&monviewer));
      PetscCall(TaoSetMonitor(tao,TaoDefaultSMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy));
    }

    PetscCall(PetscOptionsString("-tao_cmonitor","Use the default convergence monitor with constraint norm","TaoSetMonitor","stdout",monfilename,sizeof(monfilename),&flg));
    if (flg) {
      PetscCall(PetscViewerASCIIOpen(comm,monfilename,&monviewer));
      PetscCall(TaoSetMonitor(tao,TaoDefaultCMonitor,monviewer,(PetscErrorCode (*)(void**))PetscViewerDestroy));
    }

    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-tao_cancelmonitors","cancel all monitors and call any registered destroy routines","TaoCancelMonitors",flg,&flg,NULL));
    if (flg) PetscCall(TaoCancelMonitors(tao));

    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-tao_draw_solution","Plot solution vector at each iteration","TaoSetMonitor",flg,&flg,NULL));
    if (flg) {
      TaoMonitorDrawCtx drawctx;
      PetscInt          howoften = 1;
      PetscCall(TaoMonitorDrawCtxCreate(PetscObjectComm((PetscObject)tao),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,300,300,howoften,&drawctx));
      PetscCall(TaoSetMonitor(tao,TaoDrawSolutionMonitor,drawctx,(PetscErrorCode (*)(void**))TaoMonitorDrawCtxDestroy));
    }

    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-tao_draw_step","plots step direction at each iteration","TaoSetMonitor",flg,&flg,NULL));
    if (flg) {
      PetscCall(TaoSetMonitor(tao,TaoDrawStepMonitor,NULL,NULL));
    }

    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-tao_draw_gradient","plots gradient at each iteration","TaoSetMonitor",flg,&flg,NULL));
    if (flg) {
      TaoMonitorDrawCtx drawctx;
      PetscInt          howoften = 1;
      PetscCall(TaoMonitorDrawCtxCreate(PetscObjectComm((PetscObject)tao),NULL,NULL,PETSC_DECIDE,PETSC_DECIDE,300,300,howoften,&drawctx));
      PetscCall(TaoSetMonitor(tao,TaoDrawGradientMonitor,drawctx,(PetscErrorCode (*)(void**))TaoMonitorDrawCtxDestroy));
    }
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-tao_fd_gradient","compute gradient using finite differences","TaoDefaultComputeGradient",flg,&flg,NULL));
    if (flg) {
      PetscCall(TaoSetGradient(tao,NULL,TaoDefaultComputeGradient,NULL));
    }
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-tao_fd_hessian","compute hessian using finite differences","TaoDefaultComputeHessian",flg,&flg,NULL));
    if (flg) {
      Mat H;

      PetscCall(MatCreate(PetscObjectComm((PetscObject)tao),&H));
      PetscCall(MatSetType(H,MATAIJ));
      PetscCall(TaoSetHessian(tao,H,H,TaoDefaultComputeHessian,NULL));
      PetscCall(MatDestroy(&H));
    }
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-tao_mf_hessian","compute matrix-free hessian using finite differences","TaoDefaultComputeHessianMFFD",flg,&flg,NULL));
    if (flg) {
      Mat H;

      PetscCall(MatCreate(PetscObjectComm((PetscObject)tao),&H));
      PetscCall(TaoSetHessian(tao,H,H,TaoDefaultComputeHessianMFFD,NULL));
      PetscCall(MatDestroy(&H));
    }
    flg = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-tao_recycle_history","enable recycling/re-using information from the previous TaoSolve() call for some algorithms","TaoSetRecycleHistory",flg,&flg,NULL));
    if (flg) {
      PetscCall(TaoSetRecycleHistory(tao,PETSC_TRUE));
    }
    PetscCall(PetscOptionsEnum("-tao_subset_type","subset type","",TaoSubSetTypes,(PetscEnum)tao->subset_type,(PetscEnum*)&tao->subset_type,NULL));

    if (tao->linesearch) {
      PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
    }

    if (tao->ops->setfromoptions) {
      PetscCall((*tao->ops->setfromoptions)(PetscOptionsObject,tao));
    }
  }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@C
   TaoViewFromOptions - View from Options

   Collective on Tao

   Input Parameters:
+  A - the  Tao context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso: `Tao`, `TaoView`, `PetscObjectViewFromOptions()`, `TaoCreate()`
@*/
PetscErrorCode  TaoViewFromOptions(Tao A,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,TAO_CLASSID,1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A,obj,name));
  PetscFunctionReturn(0);
}

/*@C
  TaoView - Prints information about the Tao

  Collective on Tao

  InputParameters:
+ tao - the Tao context
- viewer - visualization context

  Options Database Key:
. -tao_view - Calls TaoView() at the end of TaoSolve()

  Notes:
  The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

  Level: beginner

.seealso: `PetscViewerASCIIOpen()`
@*/
PetscErrorCode TaoView(Tao tao, PetscViewer viewer)
{
  PetscBool           isascii,isstring;
  TaoType             type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (!viewer) {
    PetscCall(PetscViewerASCIIGetStdout(((PetscObject)tao)->comm,&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(tao,1,viewer,2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)tao,viewer));

    if (tao->ops->view) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall((*tao->ops->view)(tao,viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    if (tao->linesearch) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(TaoLineSearchView(tao->linesearch,viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    if (tao->ksp) {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(KSPView(tao->ksp,viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer,"total KSP iterations: %" PetscInt_FMT "\n",tao->ksp_tot_its));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }

    PetscCall(PetscViewerASCIIPushTab(viewer));

    if (tao->XL || tao->XU) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Active Set subset type: %s\n",TaoSubSetTypes[tao->subset_type]));
    }

    PetscCall(PetscViewerASCIIPrintf(viewer,"convergence tolerances: gatol=%g,",(double)tao->gatol));
    PetscCall(PetscViewerASCIIPrintf(viewer," steptol=%g,",(double)tao->steptol));
    PetscCall(PetscViewerASCIIPrintf(viewer," gttol=%g\n",(double)tao->gttol));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Residual in Function/Gradient:=%g\n",(double)tao->residual));

    if (tao->constrained) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"convergence tolerances:"));
      PetscCall(PetscViewerASCIIPrintf(viewer," catol=%g,",(double)tao->catol));
      PetscCall(PetscViewerASCIIPrintf(viewer," crtol=%g\n",(double)tao->crtol));
      PetscCall(PetscViewerASCIIPrintf(viewer,"Residual in Constraints:=%g\n",(double)tao->cnorm));
    }

    if (tao->trust < tao->steptol) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"convergence tolerances: steptol=%g\n",(double)tao->steptol));
      PetscCall(PetscViewerASCIIPrintf(viewer,"Final trust region radius:=%g\n",(double)tao->trust));
    }

    if (tao->fmin>-1.e25) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"convergence tolerances: function minimum=%g\n",(double)tao->fmin));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"Objective value=%g\n",(double)tao->fc));

    PetscCall(PetscViewerASCIIPrintf(viewer,"total number of iterations=%" PetscInt_FMT ",          ",tao->niter));
    PetscCall(PetscViewerASCIIPrintf(viewer,"              (max: %" PetscInt_FMT ")\n",tao->max_it));

    if (tao->nfuncs>0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"total number of function evaluations=%" PetscInt_FMT ",",tao->nfuncs));
      PetscCall(PetscViewerASCIIPrintf(viewer,"                max: %" PetscInt_FMT "\n",tao->max_funcs));
    }
    if (tao->ngrads>0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"total number of gradient evaluations=%" PetscInt_FMT ",",tao->ngrads));
      PetscCall(PetscViewerASCIIPrintf(viewer,"                max: %" PetscInt_FMT "\n",tao->max_funcs));
    }
    if (tao->nfuncgrads>0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"total number of function/gradient evaluations=%" PetscInt_FMT ",",tao->nfuncgrads));
      PetscCall(PetscViewerASCIIPrintf(viewer,"    (max: %" PetscInt_FMT ")\n",tao->max_funcs));
    }
    if (tao->nhess>0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"total number of Hessian evaluations=%" PetscInt_FMT "\n",tao->nhess));
    }
    if (tao->nconstraints>0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"total number of constraint function evaluations=%" PetscInt_FMT "\n",tao->nconstraints));
    }
    if (tao->njac>0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"total number of Jacobian evaluations=%" PetscInt_FMT "\n",tao->njac));
    }

    if (tao->reason>0) {
      PetscCall(PetscViewerASCIIPrintf(viewer,    "Solution converged: "));
      switch (tao->reason) {
      case TAO_CONVERGED_GATOL:
        PetscCall(PetscViewerASCIIPrintf(viewer," ||g(X)|| <= gatol\n"));
        break;
      case TAO_CONVERGED_GRTOL:
        PetscCall(PetscViewerASCIIPrintf(viewer," ||g(X)||/|f(X)| <= grtol\n"));
        break;
      case TAO_CONVERGED_GTTOL:
        PetscCall(PetscViewerASCIIPrintf(viewer," ||g(X)||/||g(X0)|| <= gttol\n"));
        break;
      case TAO_CONVERGED_STEPTOL:
        PetscCall(PetscViewerASCIIPrintf(viewer," Steptol -- step size small\n"));
        break;
      case TAO_CONVERGED_MINF:
        PetscCall(PetscViewerASCIIPrintf(viewer," Minf --  f < fmin\n"));
        break;
      case TAO_CONVERGED_USER:
        PetscCall(PetscViewerASCIIPrintf(viewer," User Terminated\n"));
        break;
      default:
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
        break;
      }
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Solver terminated: %d",tao->reason));
      switch (tao->reason) {
      case TAO_DIVERGED_MAXITS:
        PetscCall(PetscViewerASCIIPrintf(viewer," Maximum Iterations\n"));
        break;
      case TAO_DIVERGED_NAN:
        PetscCall(PetscViewerASCIIPrintf(viewer," NAN or Inf encountered\n"));
        break;
      case TAO_DIVERGED_MAXFCN:
        PetscCall(PetscViewerASCIIPrintf(viewer," Maximum Function Evaluations\n"));
        break;
      case TAO_DIVERGED_LS_FAILURE:
        PetscCall(PetscViewerASCIIPrintf(viewer," Line Search Failure\n"));
        break;
      case TAO_DIVERGED_TR_REDUCTION:
        PetscCall(PetscViewerASCIIPrintf(viewer," Trust Region too small\n"));
        break;
      case TAO_DIVERGED_USER:
        PetscCall(PetscViewerASCIIPrintf(viewer," User Terminated\n"));
        break;
      default:
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
        break;
      }
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isstring) {
    PetscCall(TaoGetType(tao,&type));
    PetscCall(PetscViewerStringSPrintf(viewer," %-3.3s",type));
  }
  PetscFunctionReturn(0);
}

/*@
  TaoSetRecycleHistory - Sets the boolean flag to enable/disable re-using
  iterate information from the previous TaoSolve(). This feature is disabled by
  default.

  For conjugate gradient methods (BNCG), this re-uses the latest search direction
  from the previous TaoSolve() call when computing the first search direction in a
  new solution. By default, CG methods set the first search direction to the
  negative gradient.

  For quasi-Newton family of methods (BQNLS, BQNKLS, BQNKTR, BQNKTL), this re-uses
  the accumulated quasi-Newton Hessian approximation from the previous TaoSolve()
  call. By default, QN family of methods reset the initial Hessian approximation to
  the identity matrix.

  For any other algorithm, this setting has no effect.

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
- recycle - boolean flag

  Options Database Keys:
. -tao_recycle_history <true,false> - reuse the history

  Level: intermediate

.seealso: `TaoSetRecycleHistory()`, `TAOBNCG`, `TAOBQNLS`, `TAOBQNKLS`, `TAOBQNKTR`, `TAOBQNKTL`

@*/
PetscErrorCode TaoSetRecycleHistory(Tao tao, PetscBool recycle)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveBool(tao,recycle,2);
  tao->recycle = recycle;
  PetscFunctionReturn(0);
}

/*@
  TaoGetRecycleHistory - Retrieve the boolean flag for re-using iterate information
  from the previous TaoSolve(). This feature is disabled by default.

  Logically collective on Tao

  Input Parameters:
. tao - the Tao context

  Output Parameters:
. recycle - boolean flag

  Level: intermediate

.seealso: `TaoGetRecycleHistory()`, `TAOBNCG`, `TAOBQNLS`, `TAOBQNKLS`, `TAOBQNKTR`, `TAOBQNKTL`

@*/
PetscErrorCode TaoGetRecycleHistory(Tao tao, PetscBool *recycle)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidBoolPointer(recycle,2);
  *recycle = tao->recycle;
  PetscFunctionReturn(0);
}

/*@
  TaoSetTolerances - Sets parameters used in TAO convergence tests

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
. gatol - stop if norm of gradient is less than this
. grtol - stop if relative norm of gradient is less than this
- gttol - stop if norm of gradient is reduced by this factor

  Options Database Keys:
+ -tao_gatol <gatol> - Sets gatol
. -tao_grtol <grtol> - Sets grtol
- -tao_gttol <gttol> - Sets gttol

  Stopping Criteria:
$ ||g(X)||                            <= gatol
$ ||g(X)|| / |f(X)|                   <= grtol
$ ||g(X)|| / ||g(X0)||                <= gttol

  Notes:
  Use PETSC_DEFAULT to leave one or more tolerances unchanged.

  Level: beginner

.seealso: `TaoGetTolerances()`

@*/
PetscErrorCode TaoSetTolerances(Tao tao, PetscReal gatol, PetscReal grtol, PetscReal gttol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveReal(tao,gatol,2);
  PetscValidLogicalCollectiveReal(tao,grtol,3);
  PetscValidLogicalCollectiveReal(tao,gttol,4);

  if (gatol != PETSC_DEFAULT) {
    if (gatol<0) {
      PetscCall(PetscInfo(tao,"Tried to set negative gatol -- ignored.\n"));
    } else {
      tao->gatol = PetscMax(0,gatol);
      tao->gatol_changed = PETSC_TRUE;
    }
  }

  if (grtol != PETSC_DEFAULT) {
    if (grtol<0) {
      PetscCall(PetscInfo(tao,"Tried to set negative grtol -- ignored.\n"));
    } else {
      tao->grtol = PetscMax(0,grtol);
      tao->grtol_changed = PETSC_TRUE;
    }
  }

  if (gttol != PETSC_DEFAULT) {
    if (gttol<0) {
      PetscCall(PetscInfo(tao,"Tried to set negative gttol -- ignored.\n"));
    } else {
      tao->gttol = PetscMax(0,gttol);
      tao->gttol_changed = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

/*@
  TaoSetConstraintTolerances - Sets constraint tolerance parameters used in TAO  convergence tests

  Logically collective on Tao

  Input Parameters:
+ tao - the Tao context
. catol - absolute constraint tolerance, constraint norm must be less than catol for used for gatol convergence criteria
- crtol - relative contraint tolerance, constraint norm must be less than crtol for used for gatol, gttol convergence criteria

  Options Database Keys:
+ -tao_catol <catol> - Sets catol
- -tao_crtol <crtol> - Sets crtol

  Notes:
  Use PETSC_DEFAULT to leave any tolerance unchanged.

  Level: intermediate

.seealso: `TaoGetTolerances()`, `TaoGetConstraintTolerances()`, `TaoSetTolerances()`

@*/
PetscErrorCode TaoSetConstraintTolerances(Tao tao, PetscReal catol, PetscReal crtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveReal(tao,catol,2);
  PetscValidLogicalCollectiveReal(tao,crtol,3);

  if (catol != PETSC_DEFAULT) {
    if (catol<0) {
      PetscCall(PetscInfo(tao,"Tried to set negative catol -- ignored.\n"));
    } else {
      tao->catol = PetscMax(0,catol);
      tao->catol_changed = PETSC_TRUE;
    }
  }

  if (crtol != PETSC_DEFAULT) {
    if (crtol<0) {
      PetscCall(PetscInfo(tao,"Tried to set negative crtol -- ignored.\n"));
    } else {
      tao->crtol = PetscMax(0,crtol);
      tao->crtol_changed = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

/*@
  TaoGetConstraintTolerances - Gets constraint tolerance parameters used in TAO  convergence tests

  Not ollective

  Input Parameter:
. tao - the Tao context

  Output Parameters:
+ catol - absolute constraint tolerance, constraint norm must be less than catol for used for gatol convergence criteria
- crtol - relative contraint tolerance, constraint norm must be less than crtol for used for gatol, gttol convergence criteria

  Level: intermediate

.seealso: `TaoGetTolerances()`, `TaoSetTolerances()`, `TaoSetConstraintTolerances()`

@*/
PetscErrorCode TaoGetConstraintTolerances(Tao tao, PetscReal *catol, PetscReal *crtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (catol) *catol = tao->catol;
  if (crtol) *crtol = tao->crtol;
  PetscFunctionReturn(0);
}

/*@
   TaoSetFunctionLowerBound - Sets a bound on the solution objective value.
   When an approximate solution with an objective value below this number
   has been found, the solver will terminate.

   Logically Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
-  fmin - the tolerance

   Options Database Keys:
.    -tao_fmin <fmin> - sets the minimum function value

   Level: intermediate

.seealso: `TaoSetTolerances()`
@*/
PetscErrorCode TaoSetFunctionLowerBound(Tao tao,PetscReal fmin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveReal(tao,fmin,2);
  tao->fmin = fmin;
  tao->fmin_changed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   TaoGetFunctionLowerBound - Gets the bound on the solution objective value.
   When an approximate solution with an objective value below this number
   has been found, the solver will terminate.

   Not collective on Tao

   Input Parameters:
.  tao - the Tao solver context

   OutputParameters:
.  fmin - the minimum function value

   Level: intermediate

.seealso: `TaoSetFunctionLowerBound()`
@*/
PetscErrorCode TaoGetFunctionLowerBound(Tao tao,PetscReal *fmin)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidRealPointer(fmin,2);
  *fmin = tao->fmin;
  PetscFunctionReturn(0);
}

/*@
   TaoSetMaximumFunctionEvaluations - Sets a maximum number of
   function evaluations.

   Logically Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
-  nfcn - the maximum number of function evaluations (>=0)

   Options Database Keys:
.    -tao_max_funcs <nfcn> - sets the maximum number of function evaluations

   Level: intermediate

.seealso: `TaoSetTolerances()`, `TaoSetMaximumIterations()`
@*/

PetscErrorCode TaoSetMaximumFunctionEvaluations(Tao tao,PetscInt nfcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveInt(tao,nfcn,2);
  if (nfcn >= 0) { tao->max_funcs = PetscMax(0,nfcn); }
  else { tao->max_funcs = -1; }
  tao->max_funcs_changed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   TaoGetMaximumFunctionEvaluations - Sets a maximum number of
   function evaluations.

   Not Collective

   Input Parameters:
.  tao - the Tao solver context

   Output Parameters:
.  nfcn - the maximum number of function evaluations

   Level: intermediate

.seealso: `TaoSetMaximumFunctionEvaluations()`, `TaoGetMaximumIterations()`
@*/

PetscErrorCode TaoGetMaximumFunctionEvaluations(Tao tao,PetscInt *nfcn)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidIntPointer(nfcn,2);
  *nfcn = tao->max_funcs;
  PetscFunctionReturn(0);
}

/*@
   TaoGetCurrentFunctionEvaluations - Get current number of
   function evaluations.

   Not Collective

   Input Parameters:
.  tao - the Tao solver context

   Output Parameters:
.  nfuncs - the current number of function evaluations (maximum between gradient and function evaluations)

   Level: intermediate

.seealso: `TaoSetMaximumFunctionEvaluations()`, `TaoGetMaximumFunctionEvaluations()`, `TaoGetMaximumIterations()`
@*/

PetscErrorCode TaoGetCurrentFunctionEvaluations(Tao tao,PetscInt *nfuncs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidIntPointer(nfuncs,2);
  *nfuncs = PetscMax(tao->nfuncs,tao->nfuncgrads);
  PetscFunctionReturn(0);
}

/*@
   TaoSetMaximumIterations - Sets a maximum number of iterates.

   Logically Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
-  maxits - the maximum number of iterates (>=0)

   Options Database Keys:
.    -tao_max_it <its> - sets the maximum number of iterations

   Level: intermediate

.seealso: `TaoSetTolerances()`, `TaoSetMaximumFunctionEvaluations()`
@*/
PetscErrorCode TaoSetMaximumIterations(Tao tao,PetscInt maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveInt(tao,maxits,2);
  tao->max_it = PetscMax(0,maxits);
  tao->max_it_changed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   TaoGetMaximumIterations - Sets a maximum number of iterates.

   Not Collective

   Input Parameters:
.  tao - the Tao solver context

   Output Parameters:
.  maxits - the maximum number of iterates

   Level: intermediate

.seealso: `TaoSetMaximumIterations()`, `TaoGetMaximumFunctionEvaluations()`
@*/
PetscErrorCode TaoGetMaximumIterations(Tao tao,PetscInt *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidIntPointer(maxits,2);
  *maxits = tao->max_it;
  PetscFunctionReturn(0);
}

/*@
   TaoSetInitialTrustRegionRadius - Sets the initial trust region radius.

   Logically collective on Tao

   Input Parameters:
+  tao - a TAO optimization solver
-  radius - the trust region radius

   Level: intermediate

   Options Database Key:
.  -tao_trust0 <t0> - sets initial trust region radius

.seealso: `TaoGetTrustRegionRadius()`, `TaoSetTrustRegionTolerance()`
@*/
PetscErrorCode TaoSetInitialTrustRegionRadius(Tao tao, PetscReal radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveReal(tao,radius,2);
  tao->trust0 = PetscMax(0.0,radius);
  tao->trust0_changed = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   TaoGetInitialTrustRegionRadius - Sets the initial trust region radius.

   Not Collective

   Input Parameter:
.  tao - a TAO optimization solver

   Output Parameter:
.  radius - the trust region radius

   Level: intermediate

.seealso: `TaoSetInitialTrustRegionRadius()`, `TaoGetCurrentTrustRegionRadius()`
@*/
PetscErrorCode TaoGetInitialTrustRegionRadius(Tao tao, PetscReal *radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidRealPointer(radius,2);
  *radius = tao->trust0;
  PetscFunctionReturn(0);
}

/*@
   TaoGetCurrentTrustRegionRadius - Gets the current trust region radius.

   Not Collective

   Input Parameter:
.  tao - a TAO optimization solver

   Output Parameter:
.  radius - the trust region radius

   Level: intermediate

.seealso: `TaoSetInitialTrustRegionRadius()`, `TaoGetInitialTrustRegionRadius()`
@*/
PetscErrorCode TaoGetCurrentTrustRegionRadius(Tao tao, PetscReal *radius)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidRealPointer(radius,2);
  *radius = tao->trust;
  PetscFunctionReturn(0);
}

/*@
  TaoGetTolerances - gets the current values of tolerances

  Not Collective

  Input Parameter:
. tao - the Tao context

  Output Parameters:
+ gatol - stop if norm of gradient is less than this
. grtol - stop if relative norm of gradient is less than this
- gttol - stop if norm of gradient is reduced by a this factor

  Note: NULL can be used as an argument if not all tolerances values are needed

.seealso `TaoSetTolerances()`

  Level: intermediate
@*/
PetscErrorCode TaoGetTolerances(Tao tao, PetscReal *gatol, PetscReal *grtol, PetscReal *gttol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (gatol) *gatol = tao->gatol;
  if (grtol) *grtol = tao->grtol;
  if (gttol) *gttol = tao->gttol;
  PetscFunctionReturn(0);
}

/*@
  TaoGetKSP - Gets the linear solver used by the optimization solver.
  Application writers should use TaoGetKSP if they need direct access
  to the PETSc KSP object.

  Not Collective

   Input Parameters:
.  tao - the TAO solver

   Output Parameters:
.  ksp - the KSP linear solver used in the optimization solver

   Level: intermediate

@*/
PetscErrorCode TaoGetKSP(Tao tao, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidPointer(ksp,2);
  *ksp = tao->ksp;
  PetscFunctionReturn(0);
}

/*@
   TaoGetLinearSolveIterations - Gets the total number of linear iterations
   used by the TAO solver

   Not Collective

   Input Parameter:
.  tao - TAO context

   Output Parameter:
.  lits - number of linear iterations

   Notes:
   This counter is reset to zero for each successive call to TaoSolve()

   Level: intermediate

.seealso: `TaoGetKSP()`
@*/
PetscErrorCode TaoGetLinearSolveIterations(Tao tao, PetscInt *lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidIntPointer(lits,2);
  *lits = tao->ksp_tot_its;
  PetscFunctionReturn(0);
}

/*@
  TaoGetLineSearch - Gets the line search used by the optimization solver.
  Application writers should use TaoGetLineSearch if they need direct access
  to the TaoLineSearch object.

  Not Collective

   Input Parameters:
.  tao - the TAO solver

   Output Parameters:
.  ls - the line search used in the optimization solver

   Level: intermediate

@*/
PetscErrorCode TaoGetLineSearch(Tao tao, TaoLineSearch *ls)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidPointer(ls,2);
  *ls = tao->linesearch;
  PetscFunctionReturn(0);
}

/*@
  TaoAddLineSearchCounts - Adds the number of function evaluations spent
  in the line search to the running total.

   Input Parameters:
+  tao - the TAO solver
-  ls - the line search used in the optimization solver

   Level: developer

.seealso: `TaoLineSearchApply()`
@*/
PetscErrorCode TaoAddLineSearchCounts(Tao tao)
{
  PetscBool      flg;
  PetscInt       nfeval,ngeval,nfgeval;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (tao->linesearch) {
    PetscCall(TaoLineSearchIsUsingTaoRoutines(tao->linesearch,&flg));
    if (!flg) {
      PetscCall(TaoLineSearchGetNumberFunctionEvaluations(tao->linesearch,&nfeval,&ngeval,&nfgeval));
      tao->nfuncs += nfeval;
      tao->ngrads += ngeval;
      tao->nfuncgrads += nfgeval;
    }
  }
  PetscFunctionReturn(0);
}

/*@
  TaoGetSolution - Returns the vector with the current TAO solution

  Not Collective

  Input Parameter:
. tao - the Tao context

  Output Parameter:
. X - the current solution

  Level: intermediate

  Note:  The returned vector will be the same object that was passed into TaoSetSolution()
@*/
PetscErrorCode TaoGetSolution(Tao tao, Vec *X)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidPointer(X,2);
  *X = tao->solution;
  PetscFunctionReturn(0);
}

/*@
   TaoResetStatistics - Initialize the statistics used by TAO for all of the solvers.
   These statistics include the iteration number, residual norms, and convergence status.
   This routine gets called before solving each optimization problem.

   Collective on Tao

   Input Parameters:
.  solver - the Tao context

   Level: developer

.seealso: `TaoCreate()`, `TaoSolve()`
@*/
PetscErrorCode TaoResetStatistics(Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  tao->niter        = 0;
  tao->nfuncs       = 0;
  tao->nfuncgrads   = 0;
  tao->ngrads       = 0;
  tao->nhess        = 0;
  tao->njac         = 0;
  tao->nconstraints = 0;
  tao->ksp_its      = 0;
  tao->ksp_tot_its  = 0;
  tao->reason       = TAO_CONTINUE_ITERATING;
  tao->residual     = 0.0;
  tao->cnorm        = 0.0;
  tao->step         = 0.0;
  tao->lsflag       = PETSC_FALSE;
  if (tao->hist_reset) tao->hist_len = 0;
  PetscFunctionReturn(0);
}

/*@C
  TaoSetUpdate - Sets the general-purpose update function called
  at the beginning of every iteration of the nonlinear solve. Specifically
  it is called at the top of every iteration, after the new solution and the gradient
  is determined, but before the Hessian is computed (if applicable).

  Logically Collective on Tao

  Input Parameters:
+ tao - The tao solver context
- func - The function

  Calling sequence of func:
$ func (Tao tao, PetscInt step);

. step - The current step of the iteration

  Level: advanced

.seealso `TaoSolve()`
@*/
PetscErrorCode TaoSetUpdate(Tao tao, PetscErrorCode (*func)(Tao, PetscInt, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID,1);
  tao->ops->update = func;
  tao->user_update = ctx;
  PetscFunctionReturn(0);
}

/*@C
  TaoSetConvergenceTest - Sets the function that is to be used to test
  for convergence o fthe iterative minimization solution.  The new convergence
  testing routine will replace TAO's default convergence test.

  Logically Collective on Tao

  Input Parameters:
+ tao - the Tao object
. conv - the routine to test for convergence
- ctx - [optional] context for private data for the convergence routine
        (may be NULL)

  Calling sequence of conv:
$   PetscErrorCode conv(Tao tao, void *ctx)

+ tao - the Tao object
- ctx - [optional] convergence context

  Note: The new convergence testing routine should call TaoSetConvergedReason().

  Level: advanced

.seealso: `TaoSetConvergedReason()`, `TaoGetSolutionStatus()`, `TaoGetTolerances()`, `TaoSetMonitor`

@*/
PetscErrorCode TaoSetConvergenceTest(Tao tao, PetscErrorCode (*conv)(Tao, void*), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  tao->ops->convergencetest = conv;
  tao->cnvP = ctx;
  PetscFunctionReturn(0);
}

/*@C
   TaoSetMonitor - Sets an ADDITIONAL function that is to be used at every
   iteration of the solver to display the iteration's
   progress.

   Logically Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
.  mymonitor - monitoring routine
-  mctx - [optional] user-defined context for private data for the
          monitor routine (may be NULL)

   Calling sequence of mymonitor:
.vb
     PetscErrorCode mymonitor(Tao tao,void *mctx)
.ve

+    tao - the Tao solver context
-    mctx - [optional] monitoring context

   Options Database Keys:
+    -tao_monitor        - sets TaoMonitorDefault()
.    -tao_smonitor       - sets short monitor
.    -tao_cmonitor       - same as smonitor plus constraint norm
.    -tao_view_solution   - view solution at each iteration
.    -tao_view_gradient   - view gradient at each iteration
.    -tao_view_ls_residual - view least-squares residual vector at each iteration
-    -tao_cancelmonitors - cancels all monitors that have been hardwired into a code by calls to TaoSetMonitor(), but does not cancel those set via the options database.

   Notes:
   Several different monitoring routines may be set by calling
   TaoSetMonitor() multiple times; all will be called in the
   order in which they were set.

   Fortran Notes:
    Only one monitor function may be set

   Level: intermediate

.seealso: `TaoMonitorDefault()`, `TaoCancelMonitors()`, `TaoSetDestroyRoutine()`
@*/
PetscErrorCode TaoSetMonitor(Tao tao, PetscErrorCode (*func)(Tao, void*), void *ctx,PetscErrorCode (*dest)(void**))
{
  PetscInt       i;
  PetscBool      identical;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCheck(tao->numbermonitors < MAXTAOMONITORS,PetscObjectComm((PetscObject)tao),PETSC_ERR_SUP,"Cannot attach another monitor -- max=%d",MAXTAOMONITORS);

  for (i=0; i<tao->numbermonitors;i++) {
    PetscCall(PetscMonitorCompare((PetscErrorCode (*)(void))func,ctx,dest,(PetscErrorCode (*)(void))tao->monitor[i],tao->monitorcontext[i],tao->monitordestroy[i],&identical));
    if (identical) PetscFunctionReturn(0);
  }
  tao->monitor[tao->numbermonitors] = func;
  tao->monitorcontext[tao->numbermonitors] = (void*)ctx;
  tao->monitordestroy[tao->numbermonitors] = dest;
  ++tao->numbermonitors;
  PetscFunctionReturn(0);
}

/*@
   TaoCancelMonitors - Clears all the monitor functions for a Tao object.

   Logically Collective on Tao

   Input Parameters:
.  tao - the Tao solver context

   Options Database:
.  -tao_cancelmonitors - cancels all monitors that have been hardwired
    into a code by calls to TaoSetMonitor(), but does not cancel those
    set via the options database

   Notes:
   There is no way to clear one specific monitor from a Tao object.

   Level: advanced

.seealso: `TaoMonitorDefault()`, `TaoSetMonitor()`
@*/
PetscErrorCode TaoCancelMonitors(Tao tao)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  for (i=0;i<tao->numbermonitors;i++) {
    if (tao->monitordestroy[i]) {
      PetscCall((*tao->monitordestroy[i])(&tao->monitorcontext[i]));
    }
  }
  tao->numbermonitors = 0;
  PetscFunctionReturn(0);
}

/*@
   TaoMonitorDefault - Default routine for monitoring progress of the
   Tao solvers (default).  This monitor prints the function value and gradient
   norm at each iteration.  It can be turned on from the command line using the
   -tao_monitor option

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context or NULL

   Options Database Keys:
.  -tao_monitor - turn on default monitoring

   Level: advanced

.seealso: `TaoDefaultSMonitor()`, `TaoSetMonitor()`
@*/
PetscErrorCode TaoMonitorDefault(Tao tao, void *ctx)
{
  PetscInt       its, tabs;
  PetscReal      fct,gnorm;
  PetscViewer    viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  its = tao->niter;
  fct = tao->fc;
  gnorm = tao->residual;
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  if (its == 0 && ((PetscObject)tao)->prefix && !tao->header_printed) {
     PetscCall(PetscViewerASCIIPrintf(viewer,"  Iteration information for %s solve.\n",((PetscObject)tao)->prefix));
     tao->header_printed = PETSC_TRUE;
   }
  PetscCall(PetscViewerASCIIPrintf(viewer,"%3" PetscInt_FMT " TAO,",its));
  PetscCall(PetscViewerASCIIPrintf(viewer,"  Function value: %g,",(double)fct));
  if (gnorm >= PETSC_INFINITY) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Residual: Inf \n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Residual: %g \n",(double)gnorm));
  }
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(0);
}

/*@
   TaoDefaultGMonitor - Default routine for monitoring progress of the
   Tao solvers (default) with extra detail on the globalization method.
   This monitor prints the function value and gradient norm at each
   iteration, as well as the step size and trust radius. Note that the
   step size and trust radius may be the same for some algorithms.
   It can be turned on from the command line using the
   -tao_gmonitor option

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context or NULL

   Options Database Keys:
.  -tao_gmonitor - turn on monitoring with globalization information

   Level: advanced

.seealso: `TaoDefaultSMonitor()`, `TaoSetMonitor()`
@*/
PetscErrorCode TaoDefaultGMonitor(Tao tao, void *ctx)
{
  PetscInt       its, tabs;
  PetscReal      fct,gnorm,stp,tr;
  PetscViewer    viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  its = tao->niter;
  fct = tao->fc;
  gnorm = tao->residual;
  stp = tao->step;
  tr = tao->trust;
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  if (its == 0 && ((PetscObject)tao)->prefix && !tao->header_printed) {
     PetscCall(PetscViewerASCIIPrintf(viewer,"  Iteration information for %s solve.\n",((PetscObject)tao)->prefix));
     tao->header_printed = PETSC_TRUE;
   }
  PetscCall(PetscViewerASCIIPrintf(viewer,"%3" PetscInt_FMT " TAO,",its));
  PetscCall(PetscViewerASCIIPrintf(viewer,"  Function value: %g,",(double)fct));
  if (gnorm >= PETSC_INFINITY) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Residual: Inf,"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Residual: %g,",(double)gnorm));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer,"  Step: %g,  Trust: %g\n",(double)stp,(double)tr));
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(0);
}

/*@
   TaoDefaultSMonitor - Default routine for monitoring progress of the
   solver. Same as TaoMonitorDefault() except
   it prints fewer digits of the residual as the residual gets smaller.
   This is because the later digits are meaningless and are often
   different on different machines; by using this routine different
   machines will usually generate the same output. It can be turned on
   by using the -tao_smonitor option

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context of type ASCII

   Options Database Keys:
.  -tao_smonitor - turn on default short monitoring

   Level: advanced

.seealso: `TaoMonitorDefault()`, `TaoSetMonitor()`
@*/
PetscErrorCode TaoDefaultSMonitor(Tao tao, void *ctx)
{
  PetscInt       its, tabs;
  PetscReal      fct,gnorm;
  PetscViewer    viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  its = tao->niter;
  fct = tao->fc;
  gnorm = tao->residual;
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer,"iter = %3" PetscInt_FMT ",",its));
  PetscCall(PetscViewerASCIIPrintf(viewer," Function value %g,",(double)fct));
  if (gnorm >= PETSC_INFINITY) {
    PetscCall(PetscViewerASCIIPrintf(viewer," Residual: Inf \n"));
  } else if (gnorm > 1.e-6) {
    PetscCall(PetscViewerASCIIPrintf(viewer," Residual: %g \n",(double)gnorm));
  } else if (gnorm > 1.e-11) {
    PetscCall(PetscViewerASCIIPrintf(viewer," Residual: < 1.0e-6 \n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer," Residual: < 1.0e-11 \n"));
  }
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(0);
}

/*@
   TaoDefaultCMonitor - same as TaoMonitorDefault() except
   it prints the norm of the constraints function. It can be turned on
   from the command line using the -tao_cmonitor option

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context or NULL

   Options Database Keys:
.  -tao_cmonitor - monitor the constraints

   Level: advanced

.seealso: `TaoMonitorDefault()`, `TaoSetMonitor()`
@*/
PetscErrorCode TaoDefaultCMonitor(Tao tao, void *ctx)
{
  PetscInt       its, tabs;
  PetscReal      fct,gnorm;
  PetscViewer    viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  its = tao->niter;
  fct = tao->fc;
  gnorm = tao->residual;
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIISetTab(viewer, ((PetscObject)tao)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer,"iter = %" PetscInt_FMT ",",its));
  PetscCall(PetscViewerASCIIPrintf(viewer," Function value: %g,",(double)fct));
  PetscCall(PetscViewerASCIIPrintf(viewer,"  Residual: %g ",(double)gnorm));
  PetscCall(PetscViewerASCIIPrintf(viewer,"  Constraint: %g \n",(double)tao->cnorm));
  PetscCall(PetscViewerASCIISetTab(viewer, tabs));
  PetscFunctionReturn(0);
}

/*@C
   TaoSolutionMonitor - Views the solution at each iteration
   It can be turned on from the command line using the
   -tao_view_solution option

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context or NULL

   Options Database Keys:
.  -tao_view_solution - view the solution

   Level: advanced

.seealso: `TaoDefaultSMonitor()`, `TaoSetMonitor()`
@*/
PetscErrorCode TaoSolutionMonitor(Tao tao, void *ctx)
{
  PetscViewer    viewer  = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall(VecView(tao->solution,viewer));
  PetscFunctionReturn(0);
}

/*@C
   TaoGradientMonitor - Views the gradient at each iteration
   It can be turned on from the command line using the
   -tao_view_gradient option

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context or NULL

   Options Database Keys:
.  -tao_view_gradient - view the gradient at each iteration

   Level: advanced

.seealso: `TaoDefaultSMonitor()`, `TaoSetMonitor()`
@*/
PetscErrorCode TaoGradientMonitor(Tao tao, void *ctx)
{
  PetscViewer    viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall(VecView(tao->gradient,viewer));
  PetscFunctionReturn(0);
}

/*@C
   TaoStepDirectionMonitor - Views the step-direction at each iteration

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context or NULL

   Options Database Keys:
.  -tao_view_gradient - view the gradient at each iteration

   Level: advanced

.seealso: `TaoDefaultSMonitor()`, `TaoSetMonitor()`
@*/
PetscErrorCode TaoStepDirectionMonitor(Tao tao, void *ctx)
{
  PetscViewer    viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall(VecView(tao->stepdirection,viewer));
  PetscFunctionReturn(0);
}

/*@C
   TaoDrawSolutionMonitor - Plots the solution at each iteration
   It can be turned on from the command line using the
   -tao_draw_solution option

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - TaoMonitorDraw context

   Options Database Keys:
.  -tao_draw_solution - draw the solution at each iteration

   Level: advanced

.seealso: `TaoSolutionMonitor()`, `TaoSetMonitor()`, `TaoDrawGradientMonitor`
@*/
PetscErrorCode TaoDrawSolutionMonitor(Tao tao, void *ctx)
{
  TaoMonitorDrawCtx ictx = (TaoMonitorDrawCtx)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (!(((ictx->howoften > 0) && (!(tao->niter % ictx->howoften))) || ((ictx->howoften == -1) && tao->reason))) PetscFunctionReturn(0);
  PetscCall(VecView(tao->solution,ictx->viewer));
  PetscFunctionReturn(0);
}

/*@C
   TaoDrawGradientMonitor - Plots the gradient at each iteration
   It can be turned on from the command line using the
   -tao_draw_gradient option

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context

   Options Database Keys:
.  -tao_draw_gradient - draw the gradient at each iteration

   Level: advanced

.seealso: `TaoGradientMonitor()`, `TaoSetMonitor()`, `TaoDrawSolutionMonitor`
@*/
PetscErrorCode TaoDrawGradientMonitor(Tao tao, void *ctx)
{
  TaoMonitorDrawCtx ictx = (TaoMonitorDrawCtx)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (!(((ictx->howoften > 0) && (!(tao->niter % ictx->howoften))) || ((ictx->howoften == -1) && tao->reason))) PetscFunctionReturn(0);
  PetscCall(VecView(tao->gradient,ictx->viewer));
  PetscFunctionReturn(0);
}

/*@C
   TaoDrawStepMonitor - Plots the step direction at each iteration

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context

   Options Database Keys:
.  -tao_draw_step - draw the step direction at each iteration

   Level: advanced

.seealso: `TaoSetMonitor()`, `TaoDrawSolutionMonitor`
@*/
PetscErrorCode TaoDrawStepMonitor(Tao tao, void *ctx)
{
  PetscViewer    viewer = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall(VecView(tao->stepdirection,viewer));
  PetscFunctionReturn(0);
}

/*@C
   TaoResidualMonitor - Views the least-squares residual at each iteration

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  ctx - PetscViewer context or NULL

   Options Database Keys:
.  -tao_view_ls_residual - view the least-squares residual at each iteration

   Level: advanced

.seealso: `TaoDefaultSMonitor()`, `TaoSetMonitor()`
@*/
PetscErrorCode TaoResidualMonitor(Tao tao, void *ctx)
{
  PetscViewer    viewer  = (PetscViewer)ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCall(VecView(tao->ls_res,viewer));
  PetscFunctionReturn(0);
}

/*@
   TaoDefaultConvergenceTest - Determines whether the solver should continue iterating
   or terminate.

   Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  dummy - unused dummy context

   Output Parameter:
.  reason - for terminating

   Notes:
   This routine checks the residual in the optimality conditions, the
   relative residual in the optimity conditions, the number of function
   evaluations, and the function value to test convergence.  Some
   solvers may use different convergence routines.

   Level: developer

.seealso: `TaoSetTolerances()`, `TaoGetConvergedReason()`, `TaoSetConvergedReason()`
@*/

PetscErrorCode TaoDefaultConvergenceTest(Tao tao,void *dummy)
{
  PetscInt           niter=tao->niter, nfuncs=PetscMax(tao->nfuncs,tao->nfuncgrads);
  PetscInt           max_funcs=tao->max_funcs;
  PetscReal          gnorm=tao->residual, gnorm0=tao->gnorm0;
  PetscReal          f=tao->fc, steptol=tao->steptol,trradius=tao->step;
  PetscReal          gatol=tao->gatol,grtol=tao->grtol,gttol=tao->gttol;
  PetscReal          catol=tao->catol,crtol=tao->crtol;
  PetscReal          fmin=tao->fmin, cnorm=tao->cnorm;
  TaoConvergedReason reason=tao->reason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  if (PetscIsInfOrNanReal(f)) {
    PetscCall(PetscInfo(tao,"Failed to converged, function value is Inf or NaN\n"));
    reason = TAO_DIVERGED_NAN;
  } else if (f <= fmin && cnorm <=catol) {
    PetscCall(PetscInfo(tao,"Converged due to function value %g < minimum function value %g\n", (double)f,(double)fmin));
    reason = TAO_CONVERGED_MINF;
  } else if (gnorm<= gatol && cnorm <=catol) {
    PetscCall(PetscInfo(tao,"Converged due to residual norm ||g(X)||=%g < %g\n",(double)gnorm,(double)gatol));
    reason = TAO_CONVERGED_GATOL;
  } else if (f!=0 && PetscAbsReal(gnorm/f) <= grtol && cnorm <= crtol) {
    PetscCall(PetscInfo(tao,"Converged due to residual ||g(X)||/|f(X)| =%g < %g\n",(double)(gnorm/f),(double)grtol));
    reason = TAO_CONVERGED_GRTOL;
  } else if (gnorm0 != 0 && ((gttol == 0 && gnorm == 0) || gnorm/gnorm0 < gttol) && cnorm <= crtol) {
    PetscCall(PetscInfo(tao,"Converged due to relative residual norm ||g(X)||/||g(X0)|| = %g < %g\n",(double)(gnorm/gnorm0),(double)gttol));
    reason = TAO_CONVERGED_GTTOL;
  } else if (max_funcs >=0 && nfuncs > max_funcs) {
    PetscCall(PetscInfo(tao,"Exceeded maximum number of function evaluations: %" PetscInt_FMT " > %" PetscInt_FMT "\n", nfuncs,max_funcs));
    reason = TAO_DIVERGED_MAXFCN;
  } else if (tao->lsflag != 0) {
    PetscCall(PetscInfo(tao,"Tao Line Search failure.\n"));
    reason = TAO_DIVERGED_LS_FAILURE;
  } else if (trradius < steptol && niter > 0) {
    PetscCall(PetscInfo(tao,"Trust region/step size too small: %g < %g\n", (double)trradius,(double)steptol));
    reason = TAO_CONVERGED_STEPTOL;
  } else if (niter >= tao->max_it) {
    PetscCall(PetscInfo(tao,"Exceeded maximum number of iterations: %" PetscInt_FMT " > %" PetscInt_FMT "\n",niter,tao->max_it));
    reason = TAO_DIVERGED_MAXITS;
  } else {
    reason = TAO_CONTINUE_ITERATING;
  }
  tao->reason = reason;
  PetscFunctionReturn(0);
}

/*@C
   TaoSetOptionsPrefix - Sets the prefix used for searching for all
   TAO options in the database.

   Logically Collective on Tao

   Input Parameters:
+  tao - the Tao context
-  prefix - the prefix string to prepend to all TAO option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   For example, to distinguish between the runtime options for two
   different TAO solvers, one could call
.vb
      TaoSetOptionsPrefix(tao1,"sys1_")
      TaoSetOptionsPrefix(tao2,"sys2_")
.ve

   This would enable use of different options for each system, such as
.vb
      -sys1_tao_method blmvm -sys1_tao_grtol 1.e-3
      -sys2_tao_method lmvm  -sys2_tao_grtol 1.e-4
.ve

   Level: advanced

.seealso: `TaoAppendOptionsPrefix()`, `TaoGetOptionsPrefix()`
@*/

PetscErrorCode TaoSetOptionsPrefix(Tao tao, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)tao,p));
  if (tao->linesearch) {
    PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch,p));
  }
  if (tao->ksp) {
    PetscCall(KSPSetOptionsPrefix(tao->ksp,p));
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoAppendOptionsPrefix - Appends to the prefix used for searching for all
   TAO options in the database.

   Logically Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
-  prefix - the prefix string to prepend to all TAO option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: `TaoSetOptionsPrefix()`, `TaoGetOptionsPrefix()`
@*/
PetscErrorCode TaoAppendOptionsPrefix(Tao tao, const char p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)tao,p));
  if (tao->linesearch) {
    PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)tao->linesearch,p));
  }
  if (tao->ksp) {
    PetscCall(KSPAppendOptionsPrefix(tao->ksp,p));
  }
  PetscFunctionReturn(0);
}

/*@C
  TaoGetOptionsPrefix - Gets the prefix used for searching for all
  TAO options in the database

  Not Collective

  Input Parameters:
. tao - the Tao context

  Output Parameters:
. prefix - pointer to the prefix string used is returned

  Notes:
    On the fortran side, the user should pass in a string 'prefix' of
  sufficient length to hold the prefix.

  Level: advanced

.seealso: `TaoSetOptionsPrefix()`, `TaoAppendOptionsPrefix()`
@*/
PetscErrorCode TaoGetOptionsPrefix(Tao tao, const char *p[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)tao,p));
  PetscFunctionReturn(0);
}

/*@C
   TaoSetType - Sets the method for the unconstrained minimization solver.

   Collective on Tao

   Input Parameters:
+  solver - the Tao solver context
-  type - a known method

   Options Database Key:
.  -tao_type <type> - Sets the method; use -help for a list
   of available methods (for instance, "-tao_type lmvm" or "-tao_type tron")

   Available methods include:
+    nls - Newton's method with line search for unconstrained minimization
.    ntr - Newton's method with trust region for unconstrained minimization
.    ntl - Newton's method with trust region, line search for unconstrained minimization
.    lmvm - Limited memory variable metric method for unconstrained minimization
.    cg - Nonlinear conjugate gradient method for unconstrained minimization
.    nm - Nelder-Mead algorithm for derivate-free unconstrained minimization
.    tron - Newton Trust Region method for bound constrained minimization
.    gpcg - Newton Trust Region method for quadratic bound constrained minimization
.    blmvm - Limited memory variable metric method for bound constrained minimization
-    pounders - Model-based algorithm pounder extended for nonlinear least squares

  Level: intermediate

.seealso: `TaoCreate()`, `TaoGetType()`, `TaoType`

@*/
PetscErrorCode TaoSetType(Tao tao, TaoType type)
{
  PetscErrorCode (*create_xxx)(Tao);
  PetscBool      issame;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);

  PetscCall(PetscObjectTypeCompare((PetscObject)tao,type,&issame));
  if (issame) PetscFunctionReturn(0);

  PetscCall(PetscFunctionListFind(TaoList, type, (void(**)(void))&create_xxx));
  PetscCheck(create_xxx,PetscObjectComm((PetscObject)tao),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested Tao type %s",type);

  /* Destroy the existing solver information */
  if (tao->ops->destroy) {
    PetscCall((*tao->ops->destroy)(tao));
  }
  PetscCall(KSPDestroy(&tao->ksp));
  PetscCall(TaoLineSearchDestroy(&tao->linesearch));
  PetscCall(VecDestroy(&tao->gradient));
  PetscCall(VecDestroy(&tao->stepdirection));

  tao->ops->setup = NULL;
  tao->ops->solve = NULL;
  tao->ops->view  = NULL;
  tao->ops->setfromoptions = NULL;
  tao->ops->destroy = NULL;

  tao->setupcalled = PETSC_FALSE;

  PetscCall((*create_xxx)(tao));
  PetscCall(PetscObjectChangeTypeName((PetscObject)tao,type));
  PetscFunctionReturn(0);
}

/*MC
   TaoRegister - Adds a method to the TAO package for unconstrained minimization.

   Synopsis:
   TaoRegister(char *name_solver,char *path,char *name_Create,PetscErrorCode (*routine_Create)(Tao))

   Not collective

   Input Parameters:
+  sname - name of a new user-defined solver
-  func - routine to Create method context

   Notes:
   TaoRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   TaoRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     TaoSetType(tao,"my_solver")
   or at runtime via the option
$     -tao_type my_solver

   Level: advanced

.seealso: `TaoRegisterAll()`, `TaoRegisterDestroy()`
M*/
PetscErrorCode TaoRegister(const char sname[], PetscErrorCode (*func)(Tao))
{
  PetscFunctionBegin;
  PetscCall(TaoInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoList,sname,(void (*)(void))func));
  PetscFunctionReturn(0);
}

/*@C
   TaoRegisterDestroy - Frees the list of minimization solvers that were
   registered by TaoRegisterDynamic().

   Not Collective

   Level: advanced

.seealso: `TaoRegisterAll()`, `TaoRegister()`
@*/
PetscErrorCode TaoRegisterDestroy(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&TaoList));
  TaoRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   TaoGetIterationNumber - Gets the number of Tao iterations completed
   at this time.

   Not Collective

   Input Parameter:
.  tao - Tao context

   Output Parameter:
.  iter - iteration number

   Notes:
   For example, during the computation of iteration 2 this would return 1.

   Level: intermediate

.seealso: `TaoGetLinearSolveIterations()`, `TaoGetResidualNorm()`, `TaoGetObjective()`
@*/
PetscErrorCode  TaoGetIterationNumber(Tao tao,PetscInt *iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidIntPointer(iter,2);
  *iter = tao->niter;
  PetscFunctionReturn(0);
}

/*@
   TaoGetResidualNorm - Gets the current value of the norm of the residual
   at this time.

   Not Collective

   Input Parameter:
.  tao - Tao context

   Output Parameter:
.  value - the current value

   Level: intermediate

   Developer Note: This is the 2-norm of the residual, we cannot use TaoGetGradientNorm() because that has
                   a different meaning. For some reason Tao sometimes calls the gradient the residual.

.seealso: `TaoGetLinearSolveIterations()`, `TaoGetIterationNumber()`, `TaoGetObjective()`
@*/
PetscErrorCode TaoGetResidualNorm(Tao tao,PetscReal *value)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidRealPointer(value,2);
  *value = tao->residual;
  PetscFunctionReturn(0);
}

/*@
   TaoSetIterationNumber - Sets the current iteration number.

   Logically Collective on Tao

   Input Parameters:
+  tao - Tao context
-  iter - iteration number

   Level: developer

.seealso: `TaoGetLinearSolveIterations()`
@*/
PetscErrorCode  TaoSetIterationNumber(Tao tao,PetscInt iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveInt(tao,iter,2);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)tao));
  tao->niter = iter;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)tao));
  PetscFunctionReturn(0);
}

/*@
   TaoGetTotalIterationNumber - Gets the total number of Tao iterations
   completed. This number keeps accumulating if multiple solves
   are called with the Tao object.

   Not Collective

   Input Parameter:
.  tao - Tao context

   Output Parameter:
.  iter - iteration number

   Notes:
   The total iteration count is updated after each solve, if there is a current
   TaoSolve() in progress then those iterations are not yet counted.

   Level: intermediate

.seealso: `TaoGetLinearSolveIterations()`
@*/
PetscErrorCode  TaoGetTotalIterationNumber(Tao tao,PetscInt *iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidIntPointer(iter,2);
  *iter = tao->ntotalits;
  PetscFunctionReturn(0);
}

/*@
   TaoSetTotalIterationNumber - Sets the current total iteration number.

   Logically Collective on Tao

   Input Parameters:
+  tao - Tao context
-  iter - iteration number

   Level: developer

.seealso: `TaoGetLinearSolveIterations()`
@*/
PetscErrorCode  TaoSetTotalIterationNumber(Tao tao,PetscInt iter)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveInt(tao,iter,2);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)tao));
  tao->ntotalits = iter;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)tao));
  PetscFunctionReturn(0);
}

/*@
  TaoSetConvergedReason - Sets the termination flag on a Tao object

  Logically Collective on Tao

  Input Parameters:
+ tao - the Tao context
- reason - one of
$     TAO_CONVERGED_ATOL (2),
$     TAO_CONVERGED_RTOL (3),
$     TAO_CONVERGED_STEPTOL (4),
$     TAO_CONVERGED_MINF (5),
$     TAO_CONVERGED_USER (6),
$     TAO_DIVERGED_MAXITS (-2),
$     TAO_DIVERGED_NAN (-4),
$     TAO_DIVERGED_MAXFCN (-5),
$     TAO_DIVERGED_LS_FAILURE (-6),
$     TAO_DIVERGED_TR_REDUCTION (-7),
$     TAO_DIVERGED_USER (-8),
$     TAO_CONTINUE_ITERATING (0)

   Level: intermediate

@*/
PetscErrorCode TaoSetConvergedReason(Tao tao, TaoConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidLogicalCollectiveEnum(tao,reason,2);
  tao->reason = reason;
  PetscFunctionReturn(0);
}

/*@
   TaoGetConvergedReason - Gets the reason the Tao iteration was stopped.

   Not Collective

   Input Parameter:
.  tao - the Tao solver context

   Output Parameter:
.  reason - one of
$  TAO_CONVERGED_GATOL (3)           ||g(X)|| < gatol
$  TAO_CONVERGED_GRTOL (4)           ||g(X)|| / f(X)  < grtol
$  TAO_CONVERGED_GTTOL (5)           ||g(X)|| / ||g(X0)|| < gttol
$  TAO_CONVERGED_STEPTOL (6)         step size small
$  TAO_CONVERGED_MINF (7)            F < F_min
$  TAO_CONVERGED_USER (8)            User defined
$  TAO_DIVERGED_MAXITS (-2)          its > maxits
$  TAO_DIVERGED_NAN (-4)             Numerical problems
$  TAO_DIVERGED_MAXFCN (-5)          fevals > max_funcsals
$  TAO_DIVERGED_LS_FAILURE (-6)      line search failure
$  TAO_DIVERGED_TR_REDUCTION (-7)    trust region failure
$  TAO_DIVERGED_USER(-8)             (user defined)
$  TAO_CONTINUE_ITERATING (0)

   where
+  X - current solution
.  X0 - initial guess
.  f(X) - current function value
.  f(X*) - true solution (estimated)
.  g(X) - current gradient
.  its - current iterate number
.  maxits - maximum number of iterates
.  fevals - number of function evaluations
-  max_funcsals - maximum number of function evaluations

   Level: intermediate

.seealso: `TaoSetConvergenceTest()`, `TaoSetTolerances()`

@*/
PetscErrorCode TaoGetConvergedReason(Tao tao, TaoConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidPointer(reason,2);
  *reason = tao->reason;
  PetscFunctionReturn(0);
}

/*@
   TaoGetSolutionStatus - Get the current iterate, objective value,
   residual, infeasibility, and termination

   Not Collective

   Input Parameter:
.  tao - the Tao context

   Output Parameters:
+  iterate - the current iterate number (>=0)
.  f - the current function value
.  gnorm - the square of the gradient norm, duality gap, or other measure indicating distance from optimality.
.  cnorm - the infeasibility of the current solution with regard to the constraints.
.  xdiff - the step length or trust region radius of the most recent iterate.
-  reason - The termination reason, which can equal TAO_CONTINUE_ITERATING

   Level: intermediate

   Note:
   TAO returns the values set by the solvers in the routine TaoMonitor().

   Note:
   If any of the output arguments are set to NULL, no corresponding value will be returned.

.seealso: `TaoMonitor()`, `TaoGetConvergedReason()`
@*/
PetscErrorCode TaoGetSolutionStatus(Tao tao, PetscInt *its, PetscReal *f, PetscReal *gnorm, PetscReal *cnorm, PetscReal *xdiff, TaoConvergedReason *reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (its) *its = tao->niter;
  if (f) *f = tao->fc;
  if (gnorm) *gnorm = tao->residual;
  if (cnorm) *cnorm = tao->cnorm;
  if (reason) *reason = tao->reason;
  if (xdiff) *xdiff = tao->step;
  PetscFunctionReturn(0);
}

/*@C
   TaoGetType - Gets the current Tao algorithm.

   Not Collective

   Input Parameter:
.  tao - the Tao solver context

   Output Parameter:
.  type - Tao method

   Level: intermediate

@*/
PetscErrorCode TaoGetType(Tao tao,TaoType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)tao)->type_name;
  PetscFunctionReturn(0);
}

/*@C
  TaoMonitor - Monitor the solver and the current solution.  This
  routine will record the iteration number and residual statistics,
  call any monitors specified by the user, and calls the convergence-check routine.

   Input Parameters:
+  tao - the Tao context
.  its - the current iterate number (>=0)
.  f - the current objective function value
.  res - the gradient norm, square root of the duality gap, or other measure indicating distince from optimality.  This measure will be recorded and
          used for some termination tests.
.  cnorm - the infeasibility of the current solution with regard to the constraints.
-  steplength - multiple of the step direction added to the previous iterate.

   Output Parameters:
.  reason - The termination reason, which can equal TAO_CONTINUE_ITERATING

   Options Database Key:
.  -tao_monitor - Use the default monitor, which prints statistics to standard output

.seealso `TaoGetConvergedReason()`, `TaoMonitorDefault()`, `TaoSetMonitor()`

   Level: developer

@*/
PetscErrorCode TaoMonitor(Tao tao, PetscInt its, PetscReal f, PetscReal res, PetscReal cnorm, PetscReal steplength)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  tao->fc = f;
  tao->residual = res;
  tao->cnorm = cnorm;
  tao->step = steplength;
  if (!its) {
    tao->cnorm0 = cnorm;
    tao->gnorm0 = res;
  }
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(res),PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");
  for (i=0;i<tao->numbermonitors;i++) {
    PetscCall((*tao->monitor[i])(tao,tao->monitorcontext[i]));
  }
  PetscFunctionReturn(0);
}

/*@
   TaoSetConvergenceHistory - Sets the array used to hold the convergence history.

   Logically Collective on Tao

   Input Parameters:
+  tao - the Tao solver context
.  obj   - array to hold objective value history
.  resid - array to hold residual history
.  cnorm - array to hold constraint violation history
.  lits - integer array holds the number of linear iterations for each Tao iteration
.  na  - size of obj, resid, and cnorm
-  reset - PetscTrue indicates each new minimization resets the history counter to zero,
           else it continues storing new values for new minimizations after the old ones

   Notes:
   If set, TAO will fill the given arrays with the indicated
   information at each iteration.  If 'obj','resid','cnorm','lits' are
   *all* NULL then space (using size na, or 1000 if na is PETSC_DECIDE or
   PETSC_DEFAULT) is allocated for the history.
   If not all are NULL, then only the non-NULL information categories
   will be stored, the others will be ignored.

   Any convergence information after iteration number 'na' will not be stored.

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: intermediate

.seealso: `TaoGetConvergenceHistory()`

@*/
PetscErrorCode TaoSetConvergenceHistory(Tao tao, PetscReal obj[], PetscReal resid[], PetscReal cnorm[], PetscInt lits[], PetscInt na,PetscBool reset)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (obj) PetscValidRealPointer(obj,2);
  if (resid) PetscValidRealPointer(resid,3);
  if (cnorm) PetscValidRealPointer(cnorm,4);
  if (lits) PetscValidIntPointer(lits,5);

  if (na == PETSC_DECIDE || na == PETSC_DEFAULT) na = 1000;
  if (!obj && !resid && !cnorm && !lits) {
    PetscCall(PetscCalloc4(na,&obj,na,&resid,na,&cnorm,na,&lits));
    tao->hist_malloc = PETSC_TRUE;
  }

  tao->hist_obj = obj;
  tao->hist_resid = resid;
  tao->hist_cnorm = cnorm;
  tao->hist_lits = lits;
  tao->hist_max   = na;
  tao->hist_reset = reset;
  tao->hist_len = 0;
  PetscFunctionReturn(0);
}

/*@C
   TaoGetConvergenceHistory - Gets the arrays used to hold the convergence history.

   Collective on Tao

   Input Parameter:
.  tao - the Tao context

   Output Parameters:
+  obj   - array used to hold objective value history
.  resid - array used to hold residual history
.  cnorm - array used to hold constraint violation history
.  lits  - integer array used to hold linear solver iteration count
-  nhist  - size of obj, resid, cnorm, and lits

   Notes:
    This routine must be preceded by calls to TaoSetConvergenceHistory()
    and TaoSolve(), otherwise it returns useless information.

    The calling sequence for this routine in Fortran is
$   call TaoGetConvergenceHistory(Tao tao, PetscInt nhist, PetscErrorCode ierr)

   This routine is useful, e.g., when running a code for purposes
   of accurate performance monitoring, when no I/O should be done
   during the section of code that is being timed.

   Level: advanced

.seealso: `TaoSetConvergenceHistory()`

@*/
PetscErrorCode TaoGetConvergenceHistory(Tao tao, PetscReal **obj, PetscReal **resid, PetscReal **cnorm, PetscInt **lits, PetscInt *nhist)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  if (obj)   *obj   = tao->hist_obj;
  if (cnorm) *cnorm = tao->hist_cnorm;
  if (resid) *resid = tao->hist_resid;
  if (nhist) *nhist = tao->hist_len;
  PetscFunctionReturn(0);
}

/*@
   TaoSetApplicationContext - Sets the optional user-defined context for
   a solver.

   Logically Collective on Tao

   Input Parameters:
+  tao  - the Tao context
-  usrP - optional user context

   Level: intermediate

.seealso: `TaoGetApplicationContext()`, `TaoSetApplicationContext()`
@*/
PetscErrorCode  TaoSetApplicationContext(Tao tao,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  tao->user = usrP;
  PetscFunctionReturn(0);
}

/*@
   TaoGetApplicationContext - Gets the user-defined context for a
   TAO solvers.

   Not Collective

   Input Parameter:
.  tao  - Tao context

   Output Parameter:
.  usrP - user context

   Level: intermediate

.seealso: `TaoSetApplicationContext()`
@*/
PetscErrorCode  TaoGetApplicationContext(Tao tao,void *usrP)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidPointer(usrP,2);
  *(void**)usrP = tao->user;
  PetscFunctionReturn(0);
}

/*@
   TaoSetGradientNorm - Sets the matrix used to define the inner product that measures the size of the gradient.

   Collective on tao

   Input Parameters:
+  tao  - the Tao context
-  M    - gradient norm

   Level: beginner

.seealso: `TaoGetGradientNorm()`, `TaoGradientNorm()`
@*/
PetscErrorCode  TaoSetGradientNorm(Tao tao, Mat M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(M,MAT_CLASSID,2);
  PetscCall(PetscObjectReference((PetscObject)M));
  PetscCall(MatDestroy(&tao->gradient_norm));
  PetscCall(VecDestroy(&tao->gradient_norm_tmp));
  tao->gradient_norm = M;
  PetscCall(MatCreateVecs(M, NULL, &tao->gradient_norm_tmp));
  PetscFunctionReturn(0);
}

/*@
   TaoGetGradientNorm - Returns the matrix used to define the inner product for measuring the size of the gradient.

   Not Collective

   Input Parameter:
.  tao  - Tao context

   Output Parameter:
.  M - gradient norm

   Level: beginner

.seealso: `TaoSetGradientNorm()`, `TaoGradientNorm()`
@*/
PetscErrorCode  TaoGetGradientNorm(Tao tao, Mat *M)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidPointer(M,2);
  *M = tao->gradient_norm;
  PetscFunctionReturn(0);
}

/*@C
   TaoGradientNorm - Compute the norm with respect to the inner product the user has set.

   Collective on tao

   Input Parameters:
+  tao      - the Tao context
.  gradient - the gradient to be computed
-  norm     - the norm type

   Output Parameter:
.  gnorm    - the gradient norm

   Level: developer

.seealso: `TaoSetGradientNorm()`, `TaoGetGradientNorm()`
@*/
PetscErrorCode  TaoGradientNorm(Tao tao, Vec gradient, NormType type, PetscReal *gnorm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(gradient,VEC_CLASSID,2);
  PetscValidLogicalCollectiveEnum(tao,type,3);
  PetscValidRealPointer(gnorm,4);
  if (tao->gradient_norm) {
    PetscScalar gnorms;

    PetscCheck(type == NORM_2,PetscObjectComm((PetscObject)gradient), PETSC_ERR_ARG_WRONG, "Norm type must be NORM_2 if an inner product for the gradient norm is set.");
    PetscCall(MatMult(tao->gradient_norm, gradient, tao->gradient_norm_tmp));
    PetscCall(VecDot(gradient, tao->gradient_norm_tmp, &gnorms));
    *gnorm = PetscRealPart(PetscSqrtScalar(gnorms));
  } else {
    PetscCall(VecNorm(gradient, type, gnorm));
  }
  PetscFunctionReturn(0);
}

/*@C
   TaoMonitorDrawCtxCreate - Creates the monitor context for TaoMonitorDrawCtx

   Collective on Tao

   Output Patameter:
.    ctx - the monitor context

   Options Database:
.   -tao_draw_solution_initial - show initial guess as well as current solution

   Level: intermediate

.seealso: `TaoMonitorSet()`, `TaoMonitorDefault()`, `VecView()`, `TaoMonitorDrawCtx()`
@*/
PetscErrorCode  TaoMonitorDrawCtxCreate(MPI_Comm comm,const char host[],const char label[],int x,int y,int m,int n,PetscInt howoften,TaoMonitorDrawCtx *ctx)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(ctx));
  PetscCall(PetscViewerDrawOpen(comm,host,label,x,y,m,n,&(*ctx)->viewer));
  PetscCall(PetscViewerSetFromOptions((*ctx)->viewer));
  (*ctx)->howoften = howoften;
  PetscFunctionReturn(0);
}

/*@C
   TaoMonitorDrawCtxDestroy - Destroys the monitor context for TaoMonitorDrawSolution()

   Collective on Tao

   Input Parameters:
.    ctx - the monitor context

   Level: intermediate

.seealso: `TaoMonitorSet()`, `TaoMonitorDefault()`, `VecView()`, `TaoMonitorDrawSolution()`
@*/
PetscErrorCode  TaoMonitorDrawCtxDestroy(TaoMonitorDrawCtx *ictx)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerDestroy(&(*ictx)->viewer));
  PetscCall(PetscFree(*ictx));
  PetscFunctionReturn(0);
}
