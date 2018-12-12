/*
      Defines a SNES that can consist of a collection of SNESes on patches of the domain
*/
#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/
#include <petsc/private/pcpatchimpl.h> /* We need internal access to PCPatch right now, until that part is moved to Plex */
#include <petscsf.h>

typedef struct {
  PC pc; /* The linear patch preconditioner */
  SNESCompositeType type;
} SNES_Patch;

static PetscErrorCode SNESPatchComputeResidual_Private(SNES snes, Vec x, Vec F, void *ctx)
{
  PC             pc      = (PC) ctx;
  PC_PATCH      *pcpatch = (PC_PATCH *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCPatchComputeFunction_Internal(pc, x, F, pcpatch->currentPatch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESPatchComputeJacobian_Private(SNES snes, Vec x, Mat J, Mat M, void *ctx)
{
  PC             pc      = (PC) ctx;
  PC_PATCH      *pcpatch = (PC_PATCH *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCPatchComputeOperator_Internal(pc, x, M, pcpatch->currentPatch, PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_PATCH_Nonlinear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  const char    *prefix;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    ierr = PetscMalloc1(patch->npatch, &patch->solver);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc, &prefix);CHKERRQ(ierr);
    for (i = 0; i < patch->npatch; ++i) {
      SNES snes;
      KSP  subksp;

      ierr = SNESCreate(PETSC_COMM_SELF, &snes);CHKERRQ(ierr);
      ierr = SNESSetOptionsPrefix(snes, prefix);CHKERRQ(ierr);
      ierr = SNESAppendOptionsPrefix(snes, "sub_");CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject) snes, (PetscObject) pc, 1);CHKERRQ(ierr);
      ierr = SNESGetKSP(snes, &subksp);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject) subksp, (PetscObject) pc, 1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject) pc, (PetscObject) snes);CHKERRQ(ierr);
      patch->solver[i] = (PetscObject) snes;
    }

    ierr = PetscMalloc1(patch->npatch, &patch->patchResidual);CHKERRQ(ierr);
    ierr = PetscMalloc1(patch->npatch, &patch->patchState);CHKERRQ(ierr);
    for (i = 0; i < patch->npatch; ++i) {
      ierr = VecDuplicate(patch->patchRHS[i], &patch->patchResidual[i]);CHKERRQ(ierr);
      ierr = VecDuplicate(patch->patchUpdate[i], &patch->patchState[i]);CHKERRQ(ierr);
    }
    ierr = VecDuplicate(patch->localUpdate, &patch->localState);CHKERRQ(ierr);
  }
  for (i = 0; i < patch->npatch; ++i) {
    SNES snes = (SNES) patch->solver[i];

    ierr = SNESSetFunction(snes, patch->patchResidual[i], SNESPatchComputeResidual_Private, pc);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes, patch->mat[i], patch->mat[i], SNESPatchComputeJacobian_Private, pc);CHKERRQ(ierr);
  }
  if (!pc->setupcalled && patch->optionsSet) for (i = 0; i < patch->npatch; ++i) {ierr = SNESSetFromOptions((SNES) patch->solver[i]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PATCH_Nonlinear(PC pc, PetscInt i, Vec patchRHS, Vec patchUpdate)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       pStart;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  patch->currentPatch = i;
  ierr = PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0);CHKERRQ(ierr);

  /* Scatter the overlapped global state to our patch state vector */
  ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, NULL);CHKERRQ(ierr);
  ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localState, patch->patchState[i], INSERT_VALUES, SCATTER_FORWARD, PETSC_FALSE);CHKERRQ(ierr);

  /* Set initial guess to be current state*/
  ierr = VecCopy(patch->patchState[i], patchUpdate);CHKERRQ(ierr);
  /* Solve for new state */
  ierr = SNESSolve((SNES) patch->solver[i], patchRHS, patchUpdate);CHKERRQ(ierr);
  /* To compute update, subtract off previous state */
  ierr = VecAXPY(patchUpdate, -1.0, patch->patchState[i]);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH_Nonlinear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) {ierr = SNESReset((SNES) patch->solver[i]);CHKERRQ(ierr);}
  }

  if (patch->patchResidual) {
    for (i = 0; i < patch->npatch; ++i) {ierr = VecDestroy(&patch->patchResidual[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->patchResidual);CHKERRQ(ierr);
  }

  if (patch->patchState) {
    for (i = 0; i < patch->npatch; ++i) {ierr = VecDestroy(&patch->patchState[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->patchState);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&patch->localState);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH_Nonlinear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) {ierr = SNESDestroy((SNES *) &patch->solver[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->solver);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_Patch(SNES snes)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  DM             dm;
  Mat            dummy;
  Vec            F;
  PetscInt       n, N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  ierr = PCSetDM(patch->pc, dm);CHKERRQ(ierr);
  ierr = SNESGetFunction(snes, &F, NULL, NULL);CHKERRQ(ierr);
  ierr = VecGetLocalSize(F, &n);CHKERRQ(ierr);
  ierr = VecGetSize(F, &N);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject) snes), n, n, N, N, (void *) snes, &dummy);CHKERRQ(ierr);
  ierr = PCSetOperators(patch->pc, dummy, dummy);CHKERRQ(ierr);
  ierr = MatDestroy(&dummy);CHKERRQ(ierr);
  ierr = PCSetUp(patch->pc);CHKERRQ(ierr);
  /* allocate workspace */
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESReset_Patch(SNES snes)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset(patch->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_Patch(SNES snes)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_Patch(snes);CHKERRQ(ierr);
  ierr = PCDestroy(&patch->pc);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_Patch(PetscOptionItems *PetscOptionsObject, SNES snes)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscBool      flg;
  const char    *prefix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "Patch nonlinear preconditioner options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-snes_patch_type", "Type of composition", "SNESPatchSetType", SNESCompositeTypes, (PetscEnum) patch->type, (PetscEnum *) &patch->type, &flg);CHKERRQ(ierr);
  if (flg) {ierr = SNESPatchSetType(snes, patch->type);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  ierr = PetscObjectGetOptionsPrefix((PetscObject)snes, &prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)patch->pc, prefix);CHKERRQ(ierr);
  ierr = PCSetFromOptions(patch->pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_Patch(SNES snes,PetscViewer viewer)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  type - %s\n",SNESCompositeTypes[patch->type]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PCView(patch->pc, viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_Patch(SNES snes)
{
  SNES_Patch *patch = (SNES_Patch *) snes->data;
  PC_PATCH   *pcpatch = (PC_PATCH *) patch->pc->data;
  Vec rhs, update, state;
  const PetscScalar *globalState  = NULL;
  PetscScalar       *localState   = NULL;

  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = SNESGetSolution(snes, &state);CHKERRQ(ierr);
  ierr = SNESGetSolutionUpdate(snes, &update);CHKERRQ(ierr);
  ierr = SNESGetRhs(snes, &rhs);CHKERRQ(ierr);

  /* These happen in a loop */
  {
  /* Scatter state vector to overlapped vector on all patches.
     The vector pcpatch->localState is scattered to each patch
     in PCApply_PATCH_Nonlinear. */
  ierr = VecGetArrayRead(state, &globalState);CHKERRQ(ierr);
  ierr = VecGetArray(pcpatch->localState, &localState);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(pcpatch->defaultSF, MPIU_SCALAR, globalState, localState);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(pcpatch->defaultSF, MPIU_SCALAR, globalState, localState);CHKERRQ(ierr);
  ierr = VecRestoreArray(pcpatch->localState, &localState);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(state, &globalState);CHKERRQ(ierr);

  ierr = PCApply(patch->pc, rhs, update);
  ierr = VecAXPY(state, 1.0, update);CHKERRQ(ierr); /* FIXME: replace with line search */
  }

  ierr = SNESSetConvergedReason(snes, SNES_CONVERGED_ITS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESPatchSetType_Patch(SNES snes, SNESCompositeType type)
{
  SNES_Patch *patch = (SNES_Patch *) snes->data;

  PetscFunctionBegin;
  patch->type = type;
  PetscFunctionReturn(0);
}

/*MC
  SNESPATCH - Solve a nonlinear problem by composing together many nonlinear solvers on patches

  Options Database Keys:
. -snes_patch_type <type: one of additive, multiplicative, symmetric_multiplicative> - Sets composition type

  Level: intermediate

  Concepts: composing solvers

.seealso:  SNESCreate(), SNESSetType(), SNESType (for list of available types), SNES,
           PCPATCH

   References:
.  1. - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers", SIAM Review, 57(4), 2015

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_Patch(SNES snes)
{
  PetscErrorCode ierr;
  SNES_Patch    *patch;

  PetscFunctionBegin;
  ierr = PetscNewLog(snes, &patch);CHKERRQ(ierr);

  snes->ops->solve          = SNESSolve_Patch;
  snes->ops->setup          = SNESSetUp_Patch;
  snes->ops->reset          = SNESReset_Patch;
  snes->ops->destroy        = SNESDestroy_Patch;
  snes->ops->setfromoptions = SNESSetFromOptions_Patch;
  snes->ops->view           = SNESView_Patch;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  snes->data = (void *) patch;
  patch->type = SNES_COMPOSITE_ADDITIVE;
  ierr = PCCreate(PetscObjectComm((PetscObject) snes), &patch->pc);CHKERRQ(ierr);
  ierr = PCSetType(patch->pc, PCPATCH);CHKERRQ(ierr);

  ((PC_PATCH *) patch->pc->data)->setupsolver   = PCSetUp_PATCH_Nonlinear;
  ((PC_PATCH *) patch->pc->data)->applysolver   = PCApply_PATCH_Nonlinear;
  ((PC_PATCH *) patch->pc->data)->resetsolver   = PCReset_PATCH_Nonlinear;
  ((PC_PATCH *) patch->pc->data)->destroysolver = PCDestroy_PATCH_Nonlinear;

  ierr = PetscObjectComposeFunction((PetscObject) snes, "SNESPatchSetType_C", SNESPatchSetType_Patch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  SNESPatchSetType - Sets the type of composition.

  Logically Collective on SNES

  Input Parameter:
+ snes - the preconditioner context
- type - SNES_COMPOSITE_ADDITIVE (default), SNES_COMPOSITE_MULTIPLICATIVE, etc.

  Options Database Key:
. -snes_composite_type <type: one of additive, multiplicative, etc> - Sets composite preconditioner type

  Level: Developer

.keywords: SNES, set, type, composite preconditioner, additive, multiplicative
@*/
PetscErrorCode SNESPatchSetType(SNES snes, SNESCompositeType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(snes,type, 2);
  ierr = PetscTryMethod(snes, "SNESPatchSetType_C", (SNES,SNESCompositeType), (snes,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetDiscretisationInfo(SNES snes, PetscInt nsubspaces, DM *dms, PetscInt *bs, PetscInt *nodesPerCell, const PetscInt **cellNodeMap,
                                            const PetscInt *subspaceOffsets, PetscInt numGhostBcs, const PetscInt *ghostBcNodes, PetscInt numGlobalBcs, const PetscInt *globalBcNodes)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscErrorCode ierr;
  DM dm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch SNES\n");
  ierr = PCSetDM(patch->pc, dm);CHKERRQ(ierr);
  ierr = PCPatchSetDiscretisationInfo(patch->pc, nsubspaces, dms, bs, nodesPerCell, cellNodeMap, subspaceOffsets, numGhostBcs, ghostBcNodes, numGlobalBcs, globalBcNodes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetComputeOperator(SNES snes, PetscErrorCode (*func)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, void *), void *ctx)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCPatchSetComputeOperator(patch->pc, func, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetComputeFunction(SNES snes, PetscErrorCode (*func)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, void *), void *ctx)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCPatchSetComputeFunction(patch->pc, func, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetConstructType(SNES snes, PCPatchConstructType ctype, PetscErrorCode (*func)(PC, PetscInt *, IS **, IS *, void *), void *ctx)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCPatchSetConstructType(patch->pc, ctype, func, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetCellNumbering(SNES snes, PetscSection cellNumbering)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCPatchSetCellNumbering(patch->pc, cellNumbering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
