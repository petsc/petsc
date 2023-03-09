#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoShell Tao_Shell;

struct _n_TaoShell {
  PetscErrorCode (*solve)(Tao);
  void *ctx;
};

/*@C
   TaoShellSetSolve - Sets routine to apply as solver

   Logically Collective

   Input Parameters:
+  tao - the nonlinear solver context
-  solve - the application-provided solver routine

   Calling sequence of `solve`:
$   PetscErrorCode solve(Tao tao)
.  tao - the optimizer, get the application context with `TaoShellGetContext()`

   Level: advanced

.seealso: `Tao`, `TAOSHELL`, `TaoShellSetContext()`, `TaoShellGetContext()`
@*/
PetscErrorCode TaoShellSetSolve(Tao tao, PetscErrorCode (*solve)(Tao))
{
  Tao_Shell *shell = (Tao_Shell *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  shell->solve = solve;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    TaoShellGetContext - Returns the user-provided context associated with a `TAOSHELL`

    Not Collective

    Input Parameter:
.   tao - should have been created with `TaoSetType`(tao,`TAOSHELL`);

    Output Parameter:
.   ctx - the user provided context

    Level: advanced

    Note:
    This routine is intended for use within various shell routines

.seealso: `Tao`, `TAOSHELL`, `TaoCreateShell()`, `TaoShellSetContext()`
@*/
PetscErrorCode TaoShellGetContext(Tao tao, void *ctx)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidPointer(ctx, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOSHELL, &flg));
  if (!flg) *(void **)ctx = NULL;
  else *(void **)ctx = ((Tao_Shell *)(tao->data))->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
    TaoShellSetContext - sets the context for a `TAOSHELL`

   Logically Collective

    Input Parameters:
+   tao - the shell Tao
-   ctx - the context

   Level: advanced

   Fortran Note:
  The context can only be an integer or a `PetscObject`

.seealso: `Tao`, `TAOSHELL`, `TaoCreateShell()`, `TaoShellGetContext()`
@*/
PetscErrorCode TaoShellSetContext(Tao tao, void *ctx)
{
  Tao_Shell *shell = (Tao_Shell *)tao->data;
  PetscBool  flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOSHELL, &flg));
  if (flg) shell->ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_Shell(Tao tao)
{
  Tao_Shell *shell = (Tao_Shell *)tao->data;

  PetscFunctionBegin;
  PetscCheck(shell->solve, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Must call TaoShellSetSolve() first");
  tao->reason = TAO_CONVERGED_USER;
  PetscCall((*(shell->solve))(tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoDestroy_Shell(Tao tao)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoSetUp_Shell(Tao tao)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoSetFromOptions_Shell(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoView_Shell(Tao tao, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOSHELL - a user provided optimizer

   Level: advanced

.seealso: `TaoCreate()`, `Tao`, `TaoSetType()`, `TaoType`
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_Shell(Tao tao)
{
  Tao_Shell *shell;

  PetscFunctionBegin;
  tao->ops->destroy        = TaoDestroy_Shell;
  tao->ops->setup          = TaoSetUp_Shell;
  tao->ops->setfromoptions = TaoSetFromOptions_Shell;
  tao->ops->view           = TaoView_Shell;
  tao->ops->solve          = TaoSolve_Shell;

  PetscCall(PetscNew(&shell));
  tao->data = (void *)shell;
  PetscFunctionReturn(PETSC_SUCCESS);
}
