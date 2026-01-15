#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_Shell TaoTerm_Shell;

struct _n_TaoTerm_Shell {
  PetscContainer ctxcontainer;
  PetscBool3     iscomputehessianfdpossible;
};

/*@C
  TaoTermShellSetContextDestroy - Set a method to destroy the context resources when a `TAOTERMSHELL` is destroyed

  Logically collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSHELL`
- destroy - the context destroy function

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellSetContext()`, `TaoTermShellGetContext()`
@*/
PetscErrorCode TaoTermShellSetContextDestroy(TaoTerm term, PetscCtxDestroyFn *destroy)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetContextDestroy_C", (TaoTerm, PetscCtxDestroyFn *), (term, destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetContextDestroy_Shell(TaoTerm term, PetscCtxDestroyFn *f)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;

  PetscFunctionBegin;
  if (shell->ctxcontainer) PetscCall(PetscContainerSetCtxDestroy(shell->ctxcontainer, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermShellGetContext - Get the context for a `TAOTERMSHELL`

  Not collective

  Input Parameter:
. term - a `TaoTerm` of type `TAOTERMSHELL`

  Output Parameter:
. ctx - a context

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellSetContext()`, `TaoTermShellSetContextDestroy()`
@*/
PetscErrorCode TaoTermShellGetContext(TaoTerm term, PetscCtxRt ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(ctx, 2);
  PetscUseMethod(term, "TaoTermShellGetContext_C", (TaoTerm, PetscCtxRt), (term, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellGetContext_Shell(TaoTerm term, PetscCtxRt ctx)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;

  PetscFunctionBegin;
  if (shell->ctxcontainer) PetscCall(PetscContainerGetPointer(shell->ctxcontainer, (void **)ctx));
  else *(void **)ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermShellSetContext - Set a context for a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term - a `TaoTerm` of type `TAOTERMSHELL`
- ctx  - a context

  Level: intermediate

  Note:
  The context can be accessed in callbacks using `TaoTermShellGetContext()`

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`
@*/
PetscErrorCode TaoTermShellSetContext(TaoTerm term, PetscCtx ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetContext_C", (TaoTerm, PetscCtx), (term, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetContext_Shell(TaoTerm term, PetscCtx ctx)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;

  PetscFunctionBegin;
  if (ctx) {
    PetscContainer ctxcontainer;

    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)term), &ctxcontainer));
    PetscCall(PetscContainerSetPointer(ctxcontainer, ctx));
    PetscCall(PetscObjectCompose((PetscObject)term, "TaoTermShell ctx", (PetscObject)ctxcontainer));
    shell->ctxcontainer = ctxcontainer;
    PetscCall(PetscContainerDestroy(&ctxcontainer));
  } else {
    PetscCall(PetscObjectCompose((PetscObject)term, "TaoTermShell ctx", NULL));
    shell->ctxcontainer = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Shell(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscBool any;
    if (shell->ctxcontainer) PetscCall(PetscViewerASCIIPrintf(viewer, "User context has been set\n"));
    else PetscCall(PetscViewerASCIIPrintf(viewer, "No user context has been set\n"));

    PetscCall(PetscViewerASCIIPrintf(viewer, "The following methods have been set:"));
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
    any = PETSC_FALSE;
    if (term->ops->objective) {
      any = PETSC_TRUE;
      PetscCall(PetscViewerASCIIPrintf(viewer, " objective,"));
    }
    if (term->ops->gradient) {
      any = PETSC_TRUE;
      PetscCall(PetscViewerASCIIPrintf(viewer, " gradient,"));
    }
    if (term->ops->objectiveandgradient) {
      any = PETSC_TRUE;
      PetscCall(PetscViewerASCIIPrintf(viewer, " objectiveandgradient,"));
    }
    if (term->ops->hessian) {
      any = PETSC_TRUE;
      PetscCall(PetscViewerASCIIPrintf(viewer, " hessian,"));
    }
    if (any == PETSC_FALSE) PetscCall(PetscViewerASCIIPrintf(viewer, " (none)"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermDestroy_Shell(TaoTerm term)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;

  PetscFunctionBegin;
  PetscCall(TaoTermShellSetContext_Shell(term, NULL));
  PetscCall(PetscFree(shell));
  term->data                      = NULL;
  term->ops->objective            = NULL;
  term->ops->gradient             = NULL;
  term->ops->objectiveandgradient = NULL;
  term->ops->hessian              = NULL;
  term->ops->view                 = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetContextDestroy_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetContext_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellGetContext_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetObjective_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetObjectiveAndGradient_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetHessian_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetIsComputeHessianFDPossible_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetView_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetCreateSolutionVec_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetCreateParametersVec_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetCreateHessianMatrices_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetObjective - Set the objective function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term      - a `TaoTerm` of type `TAOTERMSHELL`
- objective - a `TaoTermObjectiveFn` function pointer

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetView()`,
          `TaoTermObjectiveFn`
@*/
PetscErrorCode TaoTermShellSetObjective(TaoTerm term, TaoTermObjectiveFn *objective)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetObjective_C", (TaoTerm, TaoTermObjectiveFn *), (term, objective));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetObjective_Shell(TaoTerm term, TaoTermObjectiveFn *objective)
{
  PetscFunctionBegin;
  term->ops->objective = objective;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetGradient - Set the gradient function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term     - a `TaoTerm` of type `TAOTERMSHELL`
- gradient - a `TaoTermGradientFn` function pointer

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetView()`,
          `TaoTermGradientFn`
@*/
PetscErrorCode TaoTermShellSetGradient(TaoTerm term, TaoTermGradientFn *gradient)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetGradient_C", (TaoTerm, TaoTermGradientFn *), (term, gradient));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetGradient_Shell(TaoTerm term, TaoTermGradientFn *gradient)
{
  PetscFunctionBegin;
  term->ops->gradient = gradient;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetObjectiveAndGradient - Set the objective and gradient function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term       - a `TaoTerm` of type `TAOTERMSHELL`
- objandgrad - a `TaoTermObjectiveAndGradientFn` function pointer

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermShellSetView()`,
          `TaoTermObjectiveAndGradientFn`
@*/
PetscErrorCode TaoTermShellSetObjectiveAndGradient(TaoTerm term, TaoTermObjectiveAndGradientFn *objandgrad)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetObjectiveAndGradient_C", (TaoTerm, TaoTermObjectiveAndGradientFn *), (term, objandgrad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetObjectiveAndGradient_Shell(TaoTerm term, TaoTermObjectiveAndGradientFn *objandgrad)
{
  PetscFunctionBegin;
  term->ops->objectiveandgradient = objandgrad;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetHessian - Set the Hessian function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERMSHELL`
- hessian - a `TaoTermHessianFn` function pointer

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetView()`,
          `TaoTermHessianFn`
@*/
PetscErrorCode TaoTermShellSetHessian(TaoTerm term, TaoTermHessianFn *hessian)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetHessian_C", (TaoTerm, TaoTermHessianFn *), (term, hessian));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetHessian_Shell(TaoTerm term, TaoTermHessianFn *hessian)
{
  PetscFunctionBegin;
  term->ops->hessian = hessian;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsComputeHessianFDPossible_Shell(TaoTerm term, PetscBool3 *ispossible)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;

  PetscFunctionBegin;
  *ispossible = shell->iscomputehessianfdpossible;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetIsComputeHessianFDPossible - Set whether this term can compute Hessian with finite differences for a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term       - a `TaoTerm` of type `TAOTERMSHELL`
- ispossible - whether Hessian computation with finite differences is possible

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`,
          `TaoTermIsComputeHessianFDPossible()`
@*/
PetscErrorCode TaoTermShellSetIsComputeHessianFDPossible(TaoTerm term, PetscBool3 ispossible)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetIsComputeHessianFDPossible_C", (TaoTerm, PetscBool3), (term, ispossible));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetIsComputeHessianFDPossible_Shell(TaoTerm term, PetscBool3 ispossible)
{
  TaoTerm_Shell *shell = (TaoTerm_Shell *)term->data;

  PetscFunctionBegin;
  shell->iscomputehessianfdpossible     = ispossible;
  term->ops->iscomputehessianfdpossible = TaoTermIsComputeHessianFDPossible_Shell;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetView - Set the view function of a `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term - a `TaoTerm` of type `TAOTERMSHELL`
- view - a function with the same signature as `TaoTermView()`

  Calling sequence of `view`:
+ term   - the `TaoTerm`
- viewer - a `PetscViewer`

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`
@*/
PetscErrorCode TaoTermShellSetView(TaoTerm term, PetscErrorCode (*view)(TaoTerm term, PetscViewer viewer))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetView_C", (TaoTerm, PetscErrorCode (*)(TaoTerm, PetscViewer)), (term, view));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetView_Shell(TaoTerm term, PetscErrorCode (*view)(TaoTerm, PetscViewer))
{
  PetscFunctionBegin;
  term->ops->view = view;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetCreateSolutionVec - Set the routine that creates solution vector for a `TaoTerm` of type `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term              - a `TaoTerm` of type `TAOTERMSHELL`
- createsolutionvec - a function with the same signature as `TaoTermCreateSolutionVec()`

  Calling sequence of `createsolutionvec`:
+ term     - the `TaoTerm`
- solution - a solution vector for `term`

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetCreateHessianMatrices()`
@*/
PetscErrorCode TaoTermShellSetCreateSolutionVec(TaoTerm term, PetscErrorCode (*createsolutionvec)(TaoTerm term, Vec *solution))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetCreateSolutionVec_C", (TaoTerm, PetscErrorCode (*)(TaoTerm, Vec *)), (term, createsolutionvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetCreateParametersVec - Set the routine that creates parameters vector for a `TaoTerm` of type `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term                - a `TaoTerm` of type `TAOTERMSHELL`
- createparametersvec - a function with the same signature as `TaoTermCreateParametersVec()`

  Calling sequence of `createparametersvec`:
+ term       - the `TaoTerm`
- parameters - a parameters vector for `term`

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetCreateHessianMatrices()`
@*/
PetscErrorCode TaoTermShellSetCreateParametersVec(TaoTerm term, PetscErrorCode (*createparametersvec)(TaoTerm term, Vec *parameters))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetCreateParametersVec_C", (TaoTerm, PetscErrorCode (*)(TaoTerm, Vec *)), (term, createparametersvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetCreateSolutionVec_Shell(TaoTerm term, PetscErrorCode (*createsolutionvec)(TaoTerm, Vec *))
{
  PetscFunctionBegin;
  term->ops->createsolutionvec = createsolutionvec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetCreateParametersVec_Shell(TaoTerm term, PetscErrorCode (*createparametersvec)(TaoTerm, Vec *))
{
  PetscFunctionBegin;
  term->ops->createparametersvec = createparametersvec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermShellSetCreateHessianMatrices - Set the routine that creates Hessian matrices for a `TaoTerm` of type `TAOTERMSHELL`

  Logically collective

  Input Parameters:
+ term       - a `TaoTerm` of type `TAOTERMSHELL`
- createmats - a function with the same signature as `TaoTermCreateHessianMatrices()`

  Calling sequence of `createmats`:
+ f    - the `TaoTerm`
. H    - (optional) a matrix of the appropriate type and size for the Hessian of `term`
- Hpre - (optional) a matrix of the appropriate type and size for constructing a preconditioner for the Hessian of `term`

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`,
          `TaoTermShellSetCreateSolutionVec()`, `TaoTermShellSetCreateParametersVec()`
@*/
PetscErrorCode TaoTermShellSetCreateHessianMatrices(TaoTerm term, PetscErrorCode (*createmats)(TaoTerm f, Mat *H, Mat *Hpre))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscTryMethod(term, "TaoTermShellSetCreateHessianMatrices_C", (TaoTerm, PetscErrorCode (*)(TaoTerm, Mat *, Mat *)), (term, createmats));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermShellSetCreateHessianMatrices_Shell(TaoTerm term, PetscErrorCode (*createhessianmatrices)(TaoTerm, Mat *, Mat *))
{
  PetscFunctionBegin;
  term->ops->createhessianmatrices = createhessianmatrices;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMSHELL - A `TaoTerm` that uses user-provided function callbacks for its operations

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermShellGetContext()`, `TaoTermShellSetContextDestroy()`, `TaoTermCreateShell()`,
          `TaoTermShellSetObjective()`,
          `TaoTermShellSetGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermShellSetHessian()`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Shell(TaoTerm term)
{
  TaoTerm_Shell *shell;

  PetscFunctionBegin;
  PetscCall(PetscNew(&shell));
  term->data = (void *)shell;

  shell->iscomputehessianfdpossible     = PETSC_BOOL3_UNKNOWN;
  term->ops->iscomputehessianfdpossible = NULL;
  term->ops->destroy                    = TaoTermDestroy_Shell;
  term->ops->view                       = TaoTermView_Shell;

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetContextDestroy_C", TaoTermShellSetContextDestroy_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetContext_C", TaoTermShellSetContext_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellGetContext_C", TaoTermShellGetContext_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetObjective_C", TaoTermShellSetObjective_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetGradient_C", TaoTermShellSetGradient_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetObjectiveAndGradient_C", TaoTermShellSetObjectiveAndGradient_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetHessian_C", TaoTermShellSetHessian_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetIsComputeHessianFDPossible_C", TaoTermShellSetIsComputeHessianFDPossible_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetView_C", TaoTermShellSetView_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetCreateSolutionVec_C", TaoTermShellSetCreateSolutionVec_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetCreateParametersVec_C", TaoTermShellSetCreateParametersVec_Shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermShellSetCreateHessianMatrices_C", TaoTermShellSetCreateHessianMatrices_Shell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermCreateShell - Create a `TaoTerm` of type `TAOTERMSHELL` that is ready to accept user-provided callback operations.

  Collective

  Input Parameters:
+ comm    - the MPI communicator for computing the term
. ctx     - (optional) a context to be used by routines
- destroy - (optional) a routine to destroy the context when `term` is destroyed

  Output Parameter:
. term - a `TaoTerm` of type `TAOTERMSHELL`

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TAOTERMSHELL`
@*/
PetscErrorCode TaoTermCreateShell(MPI_Comm comm, PetscCtx ctx, PetscCtxDestroyFn *destroy, TaoTerm *term)
{
  PetscFunctionBegin;
  PetscAssertPointer(term, 4);
  PetscCall(TaoTermCreate(comm, term));
  PetscCall(TaoTermSetType(*term, TAOTERMSHELL));
  if (ctx) PetscCall(TaoTermShellSetContext(*term, ctx));
  if (destroy) PetscCall(TaoTermShellSetContextDestroy(*term, destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}
