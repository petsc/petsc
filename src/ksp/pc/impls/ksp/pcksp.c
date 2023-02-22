#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <petscksp.h> /*I "petscksp.h" I*/

typedef struct {
  KSP      ksp;
  PetscInt its; /* total number of iterations KSP uses */
} PC_KSP;

static PetscErrorCode PCKSPCreateKSP_KSP(PC pc)
{
  const char *prefix;
  PC_KSP     *jac = (PC_KSP *)pc->data;
  DM          dm;

  PetscFunctionBegin;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &jac->ksp));
  PetscCall(KSPSetErrorIfNotConverged(jac->ksp, pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->ksp, (PetscObject)pc, 1));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(KSPSetOptionsPrefix(jac->ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(jac->ksp, "ksp_"));
  PetscCall(PCGetDM(pc, &dm));
  if (dm) {
    PetscCall(KSPSetDM(jac->ksp, dm));
    PetscCall(KSPSetDMActive(jac->ksp, PETSC_FALSE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_KSP(PC pc, Vec x, Vec y)
{
  PetscInt its;
  PC_KSP  *jac = (PC_KSP *)pc->data;

  PetscFunctionBegin;
  if (jac->ksp->presolve) {
    PetscCall(VecCopy(x, y));
    PetscCall(KSPSolve(jac->ksp, y, y));
  } else {
    PetscCall(KSPSolve(jac->ksp, x, y));
  }
  PetscCall(KSPCheckSolve(jac->ksp, pc, y));
  PetscCall(KSPGetIterationNumber(jac->ksp, &its));
  jac->its += its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatApply_KSP(PC pc, Mat X, Mat Y)
{
  PetscInt its;
  PC_KSP  *jac = (PC_KSP *)pc->data;

  PetscFunctionBegin;
  if (jac->ksp->presolve) {
    PetscCall(MatCopy(X, Y, SAME_NONZERO_PATTERN));
    PetscCall(KSPMatSolve(jac->ksp, Y, Y)); /* TODO FIXME: this will fail since KSPMatSolve does not allow inplace solve yet */
  } else {
    PetscCall(KSPMatSolve(jac->ksp, X, Y));
  }
  PetscCall(KSPCheckSolve(jac->ksp, pc, NULL));
  PetscCall(KSPGetIterationNumber(jac->ksp, &its));
  jac->its += its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_KSP(PC pc, Vec x, Vec y)
{
  PetscInt its;
  PC_KSP  *jac = (PC_KSP *)pc->data;

  PetscFunctionBegin;
  if (jac->ksp->presolve) {
    PetscCall(VecCopy(x, y));
    PetscCall(KSPSolve(jac->ksp, y, y));
  } else {
    PetscCall(KSPSolveTranspose(jac->ksp, x, y));
  }
  PetscCall(KSPCheckSolve(jac->ksp, pc, y));
  PetscCall(KSPGetIterationNumber(jac->ksp, &its));
  jac->its += its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_KSP(PC pc)
{
  PC_KSP *jac = (PC_KSP *)pc->data;
  Mat     mat;

  PetscFunctionBegin;
  if (!jac->ksp) {
    PetscCall(PCKSPCreateKSP_KSP(pc));
    PetscCall(KSPSetFromOptions(jac->ksp));
  }
  if (pc->useAmat) mat = pc->mat;
  else mat = pc->pmat;
  PetscCall(KSPSetOperators(jac->ksp, mat, pc->pmat));
  PetscCall(KSPSetUp(jac->ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Default destroy, if it has never been setup */
static PetscErrorCode PCReset_KSP(PC pc)
{
  PC_KSP *jac = (PC_KSP *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&jac->ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_KSP(PC pc)
{
  PC_KSP *jac = (PC_KSP *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&jac->ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCKSPGetKSP_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCKSPSetKSP_C", NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_KSP(PC pc, PetscViewer viewer)
{
  PC_KSP   *jac = (PC_KSP *)pc->data;
  PetscBool iascii;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCKSPCreateKSP_KSP(pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    if (pc->useAmat) PetscCall(PetscViewerASCIIPrintf(viewer, "  Using Amat (not Pmat) as operator on inner solve\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  KSP and PC on KSP preconditioner follow\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  ---------------------------------\n"));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(KSPView(jac->ksp, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "  ---------------------------------\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCKSPSetKSP_KSP(PC pc, KSP ksp)
{
  PC_KSP *jac = (PC_KSP *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)ksp));
  PetscCall(KSPDestroy(&jac->ksp));
  jac->ksp = ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCKSPSetKSP - Sets the `KSP` context for a `PCKSP`.

   Collective

   Input Parameters:
+  pc - the preconditioner context
-  ksp - the `KSP` solver

   Level: advanced

   Notes:
   The `PC` and the `KSP` must have the same communicator

   This would rarely be used, the standard usage is to call `PCKSPGetKSP()` and then change options on that `KSP`

.seealso: `PCKSP`, `PCKSPGetKSP()`
@*/
PetscErrorCode PCKSPSetKSP(PC pc, KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 2);
  PetscCheckSameComm(pc, 1, ksp, 2);
  PetscTryMethod(pc, "PCKSPSetKSP_C", (PC, KSP), (pc, ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCKSPGetKSP_KSP(PC pc, KSP *ksp)
{
  PC_KSP *jac = (PC_KSP *)pc->data;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCKSPCreateKSP_KSP(pc));
  *ksp = jac->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCKSPGetKSP - Gets the `KSP` context for a `PCKSP`.

   Not Collective but ksp returned is parallel if pc was parallel

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  ksp - the `KSP` solver

   Note:
   If the `PC` is not a `PCKSP` object it raises an error

   Level: advanced

.seealso: `PCKSP`, `PCKSPSetKSP()`
@*/
PetscErrorCode PCKSPGetKSP(PC pc, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidPointer(ksp, 2);
  PetscUseMethod(pc, "PCKSPGetKSP_C", (PC, KSP *), (pc, ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_KSP(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_KSP *jac = (PC_KSP *)pc->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PC KSP options");
  if (jac->ksp) PetscCall(KSPSetFromOptions(jac->ksp));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCKSP -    Defines a preconditioner as any `KSP` solver.
                 This allows, for example, embedding a Krylov method inside a preconditioner.

   Options Database Key:
.   -pc_use_amat - use the matrix that defines the linear system, Amat as the matrix for the
                    inner solver, otherwise by default it uses the matrix used to construct
                    the preconditioner, Pmat (see `PCSetOperators()`)

   Level: intermediate

   Note:
    The application of an inexact Krylov solve is a nonlinear operation. Thus, performing a solve with `KSP` is,
    in general, a nonlinear operation, so `PCKSP` is in general a nonlinear preconditioner.
    Thus, one can see divergence or an incorrect answer unless using a flexible Krylov method (e.g. `KSPFGMRES`, `KSPGCR`, or `KSPFCG`) for the outer Krylov solve.

   Developer Note:
    If the outer Krylov method has a nonzero initial guess it will compute a new residual based on that initial guess
    and pass that as the right hand side into this `KSP` (and hence this `KSP` will always have a zero initial guess). For all outer Krylov methods
    except Richardson this is necessary since Krylov methods, even the flexible ones, need to "see" the result of the action of the preconditioner on the
    input (current residual) vector, the action of the preconditioner cannot depend also on some other vector (the "initial guess"). For
    `KSPRICHARDSON` it is possible to provide a `PCApplyRichardson_PCKSP()` that short circuits returning to the `KSP` object at each iteration to compute the
    residual, see for example `PCApplyRichardson_SOR()`. We do not implement a `PCApplyRichardson_PCKSP()`  because (1) using a `KSP` directly inside a Richardson
    is not an efficient algorithm anyways and (2) implementing it for its > 1 would essentially require that we implement Richardson (reimplementing the
    Richardson code) inside the `PCApplyRichardson_PCKSP()` leading to duplicate code.

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `PCSHELL`, `PCCOMPOSITE`, `PCSetUseAmat()`, `PCKSPGetKSP()`, `KSPFGMRES`, `KSPGCR`, `KSPFCG`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_KSP(PC pc)
{
  PC_KSP *jac;

  PetscFunctionBegin;
  PetscCall(PetscNew(&jac));
  pc->data = (void *)jac;

  PetscCall(PetscMemzero(pc->ops, sizeof(struct _PCOps)));
  pc->ops->apply          = PCApply_KSP;
  pc->ops->matapply       = PCMatApply_KSP;
  pc->ops->applytranspose = PCApplyTranspose_KSP;
  pc->ops->setup          = PCSetUp_KSP;
  pc->ops->reset          = PCReset_KSP;
  pc->ops->destroy        = PCDestroy_KSP;
  pc->ops->setfromoptions = PCSetFromOptions_KSP;
  pc->ops->view           = PCView_KSP;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCKSPGetKSP_C", PCKSPGetKSP_KSP));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCKSPSetKSP_C", PCKSPSetKSP_KSP));
  PetscFunctionReturn(PETSC_SUCCESS);
}
