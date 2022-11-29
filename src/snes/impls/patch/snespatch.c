/*
      Defines a SNES that can consist of a collection of SNESes on patches of the domain
*/
#include <petsc/private/vecimpl.h>     /* For vec->map */
#include <petsc/private/snesimpl.h>    /*I "petscsnes.h" I*/
#include <petsc/private/pcpatchimpl.h> /* We need internal access to PCPatch right now, until that part is moved to Plex */
#include <petscsf.h>
#include <petscsection.h>

typedef struct {
  PC pc; /* The linear patch preconditioner */
} SNES_Patch;

static PetscErrorCode SNESPatchComputeResidual_Private(SNES snes, Vec x, Vec F, void *ctx)
{
  PC                 pc      = (PC)ctx;
  PC_PATCH          *pcpatch = (PC_PATCH *)pc->data;
  PetscInt           pt, size, i;
  const PetscInt    *indices;
  const PetscScalar *X;
  PetscScalar       *XWithAll;

  PetscFunctionBegin;

  /* scatter from x to patch->patchStateWithAll[pt] */
  pt = pcpatch->currentPatch;
  PetscCall(ISGetSize(pcpatch->dofMappingWithoutToWithAll[pt], &size));

  PetscCall(ISGetIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices));
  PetscCall(VecGetArrayRead(x, &X));
  PetscCall(VecGetArray(pcpatch->patchStateWithAll, &XWithAll));

  for (i = 0; i < size; ++i) XWithAll[indices[i]] = X[i];

  PetscCall(VecRestoreArray(pcpatch->patchStateWithAll, &XWithAll));
  PetscCall(VecRestoreArrayRead(x, &X));
  PetscCall(ISRestoreIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices));

  PetscCall(PCPatchComputeFunction_Internal(pc, pcpatch->patchStateWithAll, F, pt));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESPatchComputeJacobian_Private(SNES snes, Vec x, Mat J, Mat M, void *ctx)
{
  PC                 pc      = (PC)ctx;
  PC_PATCH          *pcpatch = (PC_PATCH *)pc->data;
  PetscInt           pt, size, i;
  const PetscInt    *indices;
  const PetscScalar *X;
  PetscScalar       *XWithAll;

  PetscFunctionBegin;
  /* scatter from x to patch->patchStateWithAll[pt] */
  pt = pcpatch->currentPatch;
  PetscCall(ISGetSize(pcpatch->dofMappingWithoutToWithAll[pt], &size));

  PetscCall(ISGetIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices));
  PetscCall(VecGetArrayRead(x, &X));
  PetscCall(VecGetArray(pcpatch->patchStateWithAll, &XWithAll));

  for (i = 0; i < size; ++i) XWithAll[indices[i]] = X[i];

  PetscCall(VecRestoreArray(pcpatch->patchStateWithAll, &XWithAll));
  PetscCall(VecRestoreArrayRead(x, &X));
  PetscCall(ISRestoreIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices));

  PetscCall(PCPatchComputeOperator_Internal(pc, pcpatch->patchStateWithAll, M, pcpatch->currentPatch, PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_PATCH_Nonlinear(PC pc)
{
  PC_PATCH   *patch = (PC_PATCH *)pc->data;
  const char *prefix;
  PetscInt    i, pStart, dof, maxDof = -1;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscCall(PetscMalloc1(patch->npatch, &patch->solver));
    PetscCall(PCGetOptionsPrefix(pc, &prefix));
    PetscCall(PetscSectionGetChart(patch->gtolCounts, &pStart, NULL));
    for (i = 0; i < patch->npatch; ++i) {
      SNES snes;

      PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
      PetscCall(SNESSetOptionsPrefix(snes, prefix));
      PetscCall(SNESAppendOptionsPrefix(snes, "sub_"));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)snes, (PetscObject)pc, 2));
      patch->solver[i] = (PetscObject)snes;

      PetscCall(PetscSectionGetDof(patch->gtolCountsWithAll, i + pStart, &dof));
      maxDof = PetscMax(maxDof, dof);
    }
    PetscCall(VecDuplicate(patch->localUpdate, &patch->localState));
    PetscCall(VecDuplicate(patch->patchRHS, &patch->patchResidual));
    PetscCall(VecDuplicate(patch->patchUpdate, &patch->patchState));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, maxDof, &patch->patchStateWithAll));
    PetscCall(VecSetUp(patch->patchStateWithAll));
  }
  for (i = 0; i < patch->npatch; ++i) {
    SNES snes = (SNES)patch->solver[i];

    PetscCall(SNESSetFunction(snes, patch->patchResidual, SNESPatchComputeResidual_Private, pc));
    PetscCall(SNESSetJacobian(snes, patch->mat[i], patch->mat[i], SNESPatchComputeJacobian_Private, pc));
  }
  if (!pc->setupcalled && patch->optionsSet)
    for (i = 0; i < patch->npatch; ++i) PetscCall(SNESSetFromOptions((SNES)patch->solver[i]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PATCH_Nonlinear(PC pc, PetscInt i, Vec patchRHS, Vec patchUpdate)
{
  PC_PATCH *patch = (PC_PATCH *)pc->data;
  PetscInt  pStart, n;

  PetscFunctionBegin;
  patch->currentPatch = i;
  PetscCall(PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0));

  /* Scatter the overlapped global state to our patch state vector */
  PetscCall(PetscSectionGetChart(patch->gtolCounts, &pStart, NULL));
  PetscCall(PCPatch_ScatterLocal_Private(pc, i + pStart, patch->localState, patch->patchState, INSERT_VALUES, SCATTER_FORWARD, SCATTER_INTERIOR));
  PetscCall(PCPatch_ScatterLocal_Private(pc, i + pStart, patch->localState, patch->patchStateWithAll, INSERT_VALUES, SCATTER_FORWARD, SCATTER_WITHALL));

  PetscCall(MatGetLocalSize(patch->mat[i], NULL, &n));
  patch->patchState->map->n = n;
  patch->patchState->map->N = n;
  patchUpdate->map->n       = n;
  patchUpdate->map->N       = n;
  patchRHS->map->n          = n;
  patchRHS->map->N          = n;
  /* Set initial guess to be current state*/
  PetscCall(VecCopy(patch->patchState, patchUpdate));
  /* Solve for new state */
  PetscCall(SNESSolve((SNES)patch->solver[i], patchRHS, patchUpdate));
  /* To compute update, subtract off previous state */
  PetscCall(VecAXPY(patchUpdate, -1.0, patch->patchState));

  PetscCall(PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH_Nonlinear(PC pc)
{
  PC_PATCH *patch = (PC_PATCH *)pc->data;
  PetscInt  i;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(SNESReset((SNES)patch->solver[i]));
  }

  PetscCall(VecDestroy(&patch->patchResidual));
  PetscCall(VecDestroy(&patch->patchState));
  PetscCall(VecDestroy(&patch->patchStateWithAll));

  PetscCall(VecDestroy(&patch->localState));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH_Nonlinear(PC pc)
{
  PC_PATCH *patch = (PC_PATCH *)pc->data;
  PetscInt  i;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(SNESDestroy((SNES *)&patch->solver[i]));
    PetscCall(PetscFree(patch->solver));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCUpdateMultiplicative_PATCH_Nonlinear(PC pc, PetscInt i, PetscInt pStart)
{
  PC_PATCH *patch = (PC_PATCH *)pc->data;

  PetscFunctionBegin;
  PetscCall(PCPatch_ScatterLocal_Private(pc, i + pStart, patch->patchUpdate, patch->localState, ADD_VALUES, SCATTER_REVERSE, SCATTER_INTERIOR));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_Patch(SNES snes)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;
  DM          dm;
  Mat         dummy;
  Vec         F;
  PetscInt    n, N;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(PCSetDM(patch->pc, dm));
  PetscCall(SNESGetFunction(snes, &F, NULL, NULL));
  PetscCall(VecGetLocalSize(F, &n));
  PetscCall(VecGetSize(F, &N));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)snes), n, n, N, N, (void *)snes, &dummy));
  PetscCall(PCSetOperators(patch->pc, dummy, dummy));
  PetscCall(MatDestroy(&dummy));
  PetscCall(PCSetUp(patch->pc));
  /* allocate workspace */
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESReset_Patch(SNES snes)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;

  PetscFunctionBegin;
  PetscCall(PCReset(patch->pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_Patch(SNES snes)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;

  PetscFunctionBegin;
  PetscCall(SNESReset_Patch(snes));
  PetscCall(PCDestroy(&patch->pc));
  PetscCall(PetscFree(snes->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_Patch(SNES snes, PetscOptionItems *PetscOptionsObject)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)snes, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)patch->pc, prefix));
  PetscCall(PCSetFromOptions(patch->pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_Patch(SNES snes, PetscViewer viewer)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;
  PetscBool   iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "SNESPATCH\n"));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PCView(patch->pc, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_Patch(SNES snes)
{
  SNES_Patch        *patch   = (SNES_Patch *)snes->data;
  PC_PATCH          *pcpatch = (PC_PATCH *)patch->pc->data;
  SNESLineSearch     ls;
  Vec                rhs, update, state, residual;
  const PetscScalar *globalState = NULL;
  PetscScalar       *localState  = NULL;
  PetscInt           its         = 0;
  PetscReal          xnorm = 0.0, ynorm = 0.0, fnorm = 0.0;

  PetscFunctionBegin;
  PetscCall(SNESGetSolution(snes, &state));
  PetscCall(SNESGetSolutionUpdate(snes, &update));
  PetscCall(SNESGetRhs(snes, &rhs));

  PetscCall(SNESGetFunction(snes, &residual, NULL, NULL));
  PetscCall(SNESGetLineSearch(snes, &ls));

  PetscCall(SNESSetConvergedReason(snes, SNES_CONVERGED_ITERATING));
  PetscCall(VecSet(update, 0.0));
  PetscCall(SNESComputeFunction(snes, state, residual));

  PetscCall(VecNorm(state, NORM_2, &xnorm));
  PetscCall(VecNorm(residual, NORM_2, &fnorm));
  snes->ttol = fnorm * snes->rtol;

  if (snes->ops->converged) {
    PetscUseTypeMethod(snes, converged, its, xnorm, ynorm, fnorm, &snes->reason, snes->cnvP);
  } else {
    PetscCall(SNESConvergedSkip(snes, its, xnorm, ynorm, fnorm, &snes->reason, NULL));
  }
  PetscCall(SNESLogConvergenceHistory(snes, fnorm, 0)); /* should we count lits from the patches? */
  PetscCall(SNESMonitor(snes, its, fnorm));

  /* The main solver loop */
  for (its = 0; its < snes->max_its; its++) {
    PetscCall(SNESSetIterationNumber(snes, its));

    /* Scatter state vector to overlapped vector on all patches.
       The vector pcpatch->localState is scattered to each patch
       in PCApply_PATCH_Nonlinear. */
    PetscCall(VecGetArrayRead(state, &globalState));
    PetscCall(VecGetArray(pcpatch->localState, &localState));
    PetscCall(PetscSFBcastBegin(pcpatch->sectionSF, MPIU_SCALAR, globalState, localState, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(pcpatch->sectionSF, MPIU_SCALAR, globalState, localState, MPI_REPLACE));
    PetscCall(VecRestoreArray(pcpatch->localState, &localState));
    PetscCall(VecRestoreArrayRead(state, &globalState));

    /* The looping over patches happens here */
    PetscCall(PCApply(patch->pc, rhs, update));

    /* Apply a line search. This will often be basic with
       damping = 1/(max number of patches a dof can be in),
       but not always */
    PetscCall(VecScale(update, -1.0));
    PetscCall(SNESLineSearchApply(ls, state, residual, &fnorm, update));

    PetscCall(VecNorm(state, NORM_2, &xnorm));
    PetscCall(VecNorm(update, NORM_2, &ynorm));

    if (snes->ops->converged) {
      PetscUseTypeMethod(snes, converged, its, xnorm, ynorm, fnorm, &snes->reason, snes->cnvP);
    } else {
      PetscCall(SNESConvergedSkip(snes, its, xnorm, ynorm, fnorm, &snes->reason, NULL));
    }
    PetscCall(SNESLogConvergenceHistory(snes, fnorm, 0)); /* FIXME: should we count lits? */
    PetscCall(SNESMonitor(snes, its, fnorm));
  }

  if (its == snes->max_its) PetscCall(SNESSetConvergedReason(snes, SNES_DIVERGED_MAX_IT));
  PetscFunctionReturn(0);
}

/*MC
  SNESPATCH - Solve a nonlinear problem or apply a nonlinear smoother by composing together many nonlinear solvers on (often overlapping) patches

  Level: intermediate

   References:
.  * - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers", SIAM Review, 57(4), 2015

.seealso: `SNESFAS`, `SNESCreate()`, `SNESSetType()`, `SNESType`, `SNES`, `PCPATCH`
M*/
PETSC_EXTERN PetscErrorCode SNESCreate_Patch(SNES snes)
{
  SNES_Patch    *patch;
  PC_PATCH      *patchpc;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  PetscCall(PetscNew(&patch));

  snes->ops->solve          = SNESSolve_Patch;
  snes->ops->setup          = SNESSetUp_Patch;
  snes->ops->reset          = SNESReset_Patch;
  snes->ops->destroy        = SNESDestroy_Patch;
  snes->ops->setfromoptions = SNESSetFromOptions_Patch;
  snes->ops->view           = SNESView_Patch;

  PetscCall(SNESGetLineSearch(snes, &linesearch));
  if (!((PetscObject)linesearch)->type_name) PetscCall(SNESLineSearchSetType(linesearch, SNESLINESEARCHBASIC));
  snes->usesksp = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  snes->data = (void *)patch;
  PetscCall(PCCreate(PetscObjectComm((PetscObject)snes), &patch->pc));
  PetscCall(PCSetType(patch->pc, PCPATCH));

  patchpc              = (PC_PATCH *)patch->pc->data;
  patchpc->classname   = "snes";
  patchpc->isNonlinear = PETSC_TRUE;

  patchpc->setupsolver          = PCSetUp_PATCH_Nonlinear;
  patchpc->applysolver          = PCApply_PATCH_Nonlinear;
  patchpc->resetsolver          = PCReset_PATCH_Nonlinear;
  patchpc->destroysolver        = PCDestroy_PATCH_Nonlinear;
  patchpc->updatemultiplicative = PCUpdateMultiplicative_PATCH_Nonlinear;

  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetDiscretisationInfo(SNES snes, PetscInt nsubspaces, DM *dms, PetscInt *bs, PetscInt *nodesPerCell, const PetscInt **cellNodeMap, const PetscInt *subspaceOffsets, PetscInt numGhostBcs, const PetscInt *ghostBcNodes, PetscInt numGlobalBcs, const PetscInt *globalBcNodes)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;
  DM          dm;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCheck(dm, PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch SNES");
  PetscCall(PCSetDM(patch->pc, dm));
  PetscCall(PCPatchSetDiscretisationInfo(patch->pc, nsubspaces, dms, bs, nodesPerCell, cellNodeMap, subspaceOffsets, numGhostBcs, ghostBcNodes, numGlobalBcs, globalBcNodes));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetComputeOperator(SNES snes, PetscErrorCode (*func)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;

  PetscFunctionBegin;
  PetscCall(PCPatchSetComputeOperator(patch->pc, func, ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetComputeFunction(SNES snes, PetscErrorCode (*func)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;

  PetscFunctionBegin;
  PetscCall(PCPatchSetComputeFunction(patch->pc, func, ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetConstructType(SNES snes, PCPatchConstructType ctype, PetscErrorCode (*func)(PC, PetscInt *, IS **, IS *, void *), void *ctx)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;

  PetscFunctionBegin;
  PetscCall(PCPatchSetConstructType(patch->pc, ctype, func, ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetCellNumbering(SNES snes, PetscSection cellNumbering)
{
  SNES_Patch *patch = (SNES_Patch *)snes->data;

  PetscFunctionBegin;
  PetscCall(PCPatchSetCellNumbering(patch->pc, cellNumbering));
  PetscFunctionReturn(0);
}
