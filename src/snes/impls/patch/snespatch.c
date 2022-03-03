/*
      Defines a SNES that can consist of a collection of SNESes on patches of the domain
*/
#include <petsc/private/vecimpl.h>         /* For vec->map */
#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/
#include <petsc/private/pcpatchimpl.h> /* We need internal access to PCPatch right now, until that part is moved to Plex */
#include <petscsf.h>
#include <petscsection.h>

typedef struct {
  PC pc; /* The linear patch preconditioner */
} SNES_Patch;

static PetscErrorCode SNESPatchComputeResidual_Private(SNES snes, Vec x, Vec F, void *ctx)
{
  PC                pc      = (PC) ctx;
  PC_PATCH          *pcpatch = (PC_PATCH *) pc->data;
  PetscInt          pt, size, i;
  const PetscInt    *indices;
  const PetscScalar *X;
  PetscScalar       *XWithAll;

  PetscFunctionBegin;

  /* scatter from x to patch->patchStateWithAll[pt] */
  pt = pcpatch->currentPatch;
  CHKERRQ(ISGetSize(pcpatch->dofMappingWithoutToWithAll[pt], &size));

  CHKERRQ(ISGetIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices));
  CHKERRQ(VecGetArrayRead(x, &X));
  CHKERRQ(VecGetArray(pcpatch->patchStateWithAll, &XWithAll));

  for (i = 0; i < size; ++i) {
    XWithAll[indices[i]] = X[i];
  }

  CHKERRQ(VecRestoreArray(pcpatch->patchStateWithAll, &XWithAll));
  CHKERRQ(VecRestoreArrayRead(x, &X));
  CHKERRQ(ISRestoreIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices));

  CHKERRQ(PCPatchComputeFunction_Internal(pc, pcpatch->patchStateWithAll, F, pt));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESPatchComputeJacobian_Private(SNES snes, Vec x, Mat J, Mat M, void *ctx)
{
  PC                pc      = (PC) ctx;
  PC_PATCH          *pcpatch = (PC_PATCH *) pc->data;
  PetscInt          pt, size, i;
  const PetscInt    *indices;
  const PetscScalar *X;
  PetscScalar       *XWithAll;

  PetscFunctionBegin;
  /* scatter from x to patch->patchStateWithAll[pt] */
  pt = pcpatch->currentPatch;
  CHKERRQ(ISGetSize(pcpatch->dofMappingWithoutToWithAll[pt], &size));

  CHKERRQ(ISGetIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices));
  CHKERRQ(VecGetArrayRead(x, &X));
  CHKERRQ(VecGetArray(pcpatch->patchStateWithAll, &XWithAll));

  for (i = 0; i < size; ++i) {
    XWithAll[indices[i]] = X[i];
  }

  CHKERRQ(VecRestoreArray(pcpatch->patchStateWithAll, &XWithAll));
  CHKERRQ(VecRestoreArrayRead(x, &X));
  CHKERRQ(ISRestoreIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices));

  CHKERRQ(PCPatchComputeOperator_Internal(pc, pcpatch->patchStateWithAll, M, pcpatch->currentPatch, PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_PATCH_Nonlinear(PC pc)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  const char     *prefix;
  PetscInt       i, pStart, dof, maxDof = -1;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    CHKERRQ(PetscMalloc1(patch->npatch, &patch->solver));
    CHKERRQ(PCGetOptionsPrefix(pc, &prefix));
    CHKERRQ(PetscSectionGetChart(patch->gtolCounts, &pStart, NULL));
    for (i = 0; i < patch->npatch; ++i) {
      SNES snes;

      CHKERRQ(SNESCreate(PETSC_COMM_SELF, &snes));
      CHKERRQ(SNESSetOptionsPrefix(snes, prefix));
      CHKERRQ(SNESAppendOptionsPrefix(snes, "sub_"));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject) snes, (PetscObject) pc, 2));
      CHKERRQ(PetscLogObjectParent((PetscObject) pc, (PetscObject) snes));
      patch->solver[i] = (PetscObject) snes;

      CHKERRQ(PetscSectionGetDof(patch->gtolCountsWithAll, i+pStart, &dof));
      maxDof = PetscMax(maxDof, dof);
    }
    CHKERRQ(VecDuplicate(patch->localUpdate, &patch->localState));
    CHKERRQ(VecDuplicate(patch->patchRHS, &patch->patchResidual));
    CHKERRQ(VecDuplicate(patch->patchUpdate, &patch->patchState));

    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, maxDof, &patch->patchStateWithAll));
    CHKERRQ(VecSetUp(patch->patchStateWithAll));
  }
  for (i = 0; i < patch->npatch; ++i) {
    SNES snes = (SNES) patch->solver[i];

    CHKERRQ(SNESSetFunction(snes, patch->patchResidual, SNESPatchComputeResidual_Private, pc));
    CHKERRQ(SNESSetJacobian(snes, patch->mat[i], patch->mat[i], SNESPatchComputeJacobian_Private, pc));
  }
  if (!pc->setupcalled && patch->optionsSet) for (i = 0; i < patch->npatch; ++i) CHKERRQ(SNESSetFromOptions((SNES) patch->solver[i]));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PATCH_Nonlinear(PC pc, PetscInt i, Vec patchRHS, Vec patchUpdate)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       pStart, n;

  PetscFunctionBegin;
  patch->currentPatch = i;
  CHKERRQ(PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0));

  /* Scatter the overlapped global state to our patch state vector */
  CHKERRQ(PetscSectionGetChart(patch->gtolCounts, &pStart, NULL));
  CHKERRQ(PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localState, patch->patchState, INSERT_VALUES, SCATTER_FORWARD, SCATTER_INTERIOR));
  CHKERRQ(PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localState, patch->patchStateWithAll, INSERT_VALUES, SCATTER_FORWARD, SCATTER_WITHALL));

  CHKERRQ(MatGetLocalSize(patch->mat[i], NULL, &n));
  patch->patchState->map->n = n;
  patch->patchState->map->N = n;
  patchUpdate->map->n = n;
  patchUpdate->map->N = n;
  patchRHS->map->n = n;
  patchRHS->map->N = n;
  /* Set initial guess to be current state*/
  CHKERRQ(VecCopy(patch->patchState, patchUpdate));
  /* Solve for new state */
  CHKERRQ(SNESSolve((SNES) patch->solver[i], patchRHS, patchUpdate));
  /* To compute update, subtract off previous state */
  CHKERRQ(VecAXPY(patchUpdate, -1.0, patch->patchState));

  CHKERRQ(PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH_Nonlinear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(SNESReset((SNES) patch->solver[i]));
  }

  CHKERRQ(VecDestroy(&patch->patchResidual));
  CHKERRQ(VecDestroy(&patch->patchState));
  CHKERRQ(VecDestroy(&patch->patchStateWithAll));

  CHKERRQ(VecDestroy(&patch->localState));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH_Nonlinear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(SNESDestroy((SNES *) &patch->solver[i]));
    CHKERRQ(PetscFree(patch->solver));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCUpdateMultiplicative_PATCH_Nonlinear(PC pc, PetscInt i, PetscInt pStart)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  CHKERRQ(PCPatch_ScatterLocal_Private(pc, i + pStart, patch->patchUpdate, patch->localState, ADD_VALUES, SCATTER_REVERSE, SCATTER_INTERIOR));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_Patch(SNES snes)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  DM             dm;
  Mat            dummy;
  Vec            F;
  PetscInt       n, N;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes, &dm));
  CHKERRQ(PCSetDM(patch->pc, dm));
  CHKERRQ(SNESGetFunction(snes, &F, NULL, NULL));
  CHKERRQ(VecGetLocalSize(F, &n));
  CHKERRQ(VecGetSize(F, &N));
  CHKERRQ(MatCreateShell(PetscObjectComm((PetscObject) snes), n, n, N, N, (void *) snes, &dummy));
  CHKERRQ(PCSetOperators(patch->pc, dummy, dummy));
  CHKERRQ(MatDestroy(&dummy));
  CHKERRQ(PCSetUp(patch->pc));
  /* allocate workspace */
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESReset_Patch(SNES snes)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;

  PetscFunctionBegin;
  CHKERRQ(PCReset(patch->pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_Patch(SNES snes)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;

  PetscFunctionBegin;
  CHKERRQ(SNESReset_Patch(snes));
  CHKERRQ(PCDestroy(&patch->pc));
  CHKERRQ(PetscFree(snes->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_Patch(PetscOptionItems *PetscOptionsObject, SNES snes)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  const char    *prefix;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)snes, &prefix));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)patch->pc, prefix));
  CHKERRQ(PCSetFromOptions(patch->pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_Patch(SNES snes,PetscViewer viewer)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"SNESPATCH\n"));
  }
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PCView(patch->pc, viewer));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_Patch(SNES snes)
{
  SNES_Patch        *patch = (SNES_Patch *) snes->data;
  PC_PATCH          *pcpatch = (PC_PATCH *) patch->pc->data;
  SNESLineSearch    ls;
  Vec               rhs, update, state, residual;
  const PetscScalar *globalState  = NULL;
  PetscScalar       *localState   = NULL;
  PetscInt          its = 0;
  PetscReal         xnorm = 0.0, ynorm = 0.0, fnorm = 0.0;

  PetscFunctionBegin;
  CHKERRQ(SNESGetSolution(snes, &state));
  CHKERRQ(SNESGetSolutionUpdate(snes, &update));
  CHKERRQ(SNESGetRhs(snes, &rhs));

  CHKERRQ(SNESGetFunction(snes, &residual, NULL, NULL));
  CHKERRQ(SNESGetLineSearch(snes, &ls));

  CHKERRQ(SNESSetConvergedReason(snes, SNES_CONVERGED_ITERATING));
  CHKERRQ(VecSet(update, 0.0));
  CHKERRQ(SNESComputeFunction(snes, state, residual));

  CHKERRQ(VecNorm(state, NORM_2, &xnorm));
  CHKERRQ(VecNorm(residual, NORM_2, &fnorm));
  snes->ttol = fnorm*snes->rtol;

  if (snes->ops->converged) {
    CHKERRQ((*snes->ops->converged)(snes,its,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP));
  } else {
    CHKERRQ(SNESConvergedSkip(snes,its,xnorm,ynorm,fnorm,&snes->reason,NULL));
  }
  CHKERRQ(SNESLogConvergenceHistory(snes, fnorm, 0)); /* should we count lits from the patches? */
  CHKERRQ(SNESMonitor(snes, its, fnorm));

  /* The main solver loop */
  for (its = 0; its < snes->max_its; its++) {

    CHKERRQ(SNESSetIterationNumber(snes, its));

    /* Scatter state vector to overlapped vector on all patches.
       The vector pcpatch->localState is scattered to each patch
       in PCApply_PATCH_Nonlinear. */
    CHKERRQ(VecGetArrayRead(state, &globalState));
    CHKERRQ(VecGetArray(pcpatch->localState, &localState));
    CHKERRQ(PetscSFBcastBegin(pcpatch->sectionSF, MPIU_SCALAR, globalState, localState,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(pcpatch->sectionSF, MPIU_SCALAR, globalState, localState,MPI_REPLACE));
    CHKERRQ(VecRestoreArray(pcpatch->localState, &localState));
    CHKERRQ(VecRestoreArrayRead(state, &globalState));

    /* The looping over patches happens here */
    CHKERRQ(PCApply(patch->pc, rhs, update));

    /* Apply a line search. This will often be basic with
       damping = 1/(max number of patches a dof can be in),
       but not always */
    CHKERRQ(VecScale(update, -1.0));
    CHKERRQ(SNESLineSearchApply(ls, state, residual, &fnorm, update));

    CHKERRQ(VecNorm(state, NORM_2, &xnorm));
    CHKERRQ(VecNorm(update, NORM_2, &ynorm));

    if (snes->ops->converged) {
      CHKERRQ((*snes->ops->converged)(snes,its,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP));
    } else {
      CHKERRQ(SNESConvergedSkip(snes,its,xnorm,ynorm,fnorm,&snes->reason,NULL));
    }
    CHKERRQ(SNESLogConvergenceHistory(snes, fnorm, 0)); /* FIXME: should we count lits? */
    CHKERRQ(SNESMonitor(snes, its, fnorm));
  }

  if (its == snes->max_its) CHKERRQ(SNESSetConvergedReason(snes, SNES_DIVERGED_MAX_IT));
  PetscFunctionReturn(0);
}

/*MC
  SNESPATCH - Solve a nonlinear problem by composing together many nonlinear solvers on patches

  Level: intermediate

.seealso:  SNESCreate(), SNESSetType(), SNESType (for list of available types), SNES,
           PCPATCH

   References:
.  * - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers", SIAM Review, 57(4), 2015

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_Patch(SNES snes)
{
  SNES_Patch     *patch;
  PC_PATCH       *patchpc;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(snes, &patch));

  snes->ops->solve          = SNESSolve_Patch;
  snes->ops->setup          = SNESSetUp_Patch;
  snes->ops->reset          = SNESReset_Patch;
  snes->ops->destroy        = SNESDestroy_Patch;
  snes->ops->setfromoptions = SNESSetFromOptions_Patch;
  snes->ops->view           = SNESView_Patch;

  CHKERRQ(SNESGetLineSearch(snes,&linesearch));
  if (!((PetscObject)linesearch)->type_name) {
    CHKERRQ(SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC));
  }
  snes->usesksp        = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  snes->data = (void *) patch;
  CHKERRQ(PCCreate(PetscObjectComm((PetscObject) snes), &patch->pc));
  CHKERRQ(PCSetType(patch->pc, PCPATCH));

  patchpc = (PC_PATCH*) patch->pc->data;
  patchpc->classname = "snes";
  patchpc->isNonlinear = PETSC_TRUE;

  patchpc->setupsolver   = PCSetUp_PATCH_Nonlinear;
  patchpc->applysolver   = PCApply_PATCH_Nonlinear;
  patchpc->resetsolver   = PCReset_PATCH_Nonlinear;
  patchpc->destroysolver = PCDestroy_PATCH_Nonlinear;
  patchpc->updatemultiplicative = PCUpdateMultiplicative_PATCH_Nonlinear;

  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetDiscretisationInfo(SNES snes, PetscInt nsubspaces, DM *dms, PetscInt *bs, PetscInt *nodesPerCell, const PetscInt **cellNodeMap,
                                            const PetscInt *subspaceOffsets, PetscInt numGhostBcs, const PetscInt *ghostBcNodes, PetscInt numGlobalBcs, const PetscInt *globalBcNodes)
{
  SNES_Patch     *patch = (SNES_Patch *) snes->data;
  DM             dm;

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch SNES");
  CHKERRQ(PCSetDM(patch->pc, dm));
  CHKERRQ(PCPatchSetDiscretisationInfo(patch->pc, nsubspaces, dms, bs, nodesPerCell, cellNodeMap, subspaceOffsets, numGhostBcs, ghostBcNodes, numGlobalBcs, globalBcNodes));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetComputeOperator(SNES snes, PetscErrorCode (*func)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;

  PetscFunctionBegin;
  CHKERRQ(PCPatchSetComputeOperator(patch->pc, func, ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetComputeFunction(SNES snes, PetscErrorCode (*func)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;

  PetscFunctionBegin;
  CHKERRQ(PCPatchSetComputeFunction(patch->pc, func, ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetConstructType(SNES snes, PCPatchConstructType ctype, PetscErrorCode (*func)(PC, PetscInt *, IS **, IS *, void *), void *ctx)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;

  PetscFunctionBegin;
  CHKERRQ(PCPatchSetConstructType(patch->pc, ctype, func, ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetCellNumbering(SNES snes, PetscSection cellNumbering)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;

  PetscFunctionBegin;
  CHKERRQ(PCPatchSetCellNumbering(patch->pc, cellNumbering));
  PetscFunctionReturn(0);
}
