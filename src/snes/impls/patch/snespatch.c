/*
      Defines a SNES that can consist of a collection of SNESes on patches of the domain
*/
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;

  /* scatter from x to patch->patchStateWithAll[pt] */
  pt = pcpatch->currentPatch;
  ierr = ISGetSize(pcpatch->dofMappingWithoutToWithAll[pt], &size);CHKERRQ(ierr);

  ierr = ISGetIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &X);CHKERRQ(ierr);
  ierr = VecGetArray(pcpatch->patchStateWithAll[pt], &XWithAll);CHKERRQ(ierr);

  for (i = 0; i < size; ++i) {
    XWithAll[indices[i]] = X[i];
  }

  ierr = VecRestoreArray(pcpatch->patchStateWithAll[pt], &XWithAll);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &X);CHKERRQ(ierr);
  ierr = ISRestoreIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices);CHKERRQ(ierr);

  ierr = PCPatchComputeFunction_Internal(pc, pcpatch->patchStateWithAll[pt], F, pt);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* scatter from x to patch->patchStateWithAll[pt] */
  pt = pcpatch->currentPatch;
  ierr = ISGetSize(pcpatch->dofMappingWithoutToWithAll[pt], &size);CHKERRQ(ierr);

  ierr = ISGetIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &X);CHKERRQ(ierr);
  ierr = VecGetArray(pcpatch->patchStateWithAll[pt], &XWithAll);CHKERRQ(ierr);

  for (i = 0; i < size; ++i) {
    XWithAll[indices[i]] = X[i];
  }

  ierr = VecRestoreArray(pcpatch->patchStateWithAll[pt], &XWithAll);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &X);CHKERRQ(ierr);
  ierr = ISRestoreIndices(pcpatch->dofMappingWithoutToWithAll[pt], &indices);CHKERRQ(ierr);

  ierr = PCPatchComputeOperator_Internal(pc, pcpatch->patchStateWithAll[pt], M, pcpatch->currentPatch, PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_PATCH_Nonlinear(PC pc)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  const char     *prefix;
  PetscInt       i, pStart, dof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    ierr = PetscMalloc1(patch->npatch, &patch->solver);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc, &prefix);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, NULL);CHKERRQ(ierr);
    for (i = 0; i < patch->npatch; ++i) {
      SNES snes;

      ierr = SNESCreate(PETSC_COMM_SELF, &snes);CHKERRQ(ierr);
      ierr = SNESSetOptionsPrefix(snes, prefix);CHKERRQ(ierr);
      ierr = SNESAppendOptionsPrefix(snes, "sub_");CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject) snes, (PetscObject) pc, 2);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject) pc, (PetscObject) snes);CHKERRQ(ierr);
      patch->solver[i] = (PetscObject) snes;
    }

    ierr = PetscMalloc1(patch->npatch, &patch->patchResidual);CHKERRQ(ierr);
    ierr = PetscMalloc1(patch->npatch, &patch->patchState);CHKERRQ(ierr);
    ierr = PetscMalloc1(patch->npatch, &patch->patchStateWithAll);CHKERRQ(ierr);
    for (i = 0; i < patch->npatch; ++i) {
      ierr = VecDuplicate(patch->patchRHS[i], &patch->patchResidual[i]);CHKERRQ(ierr);
      ierr = VecDuplicate(patch->patchUpdate[i], &patch->patchState[i]);CHKERRQ(ierr);

      ierr = PetscSectionGetDof(patch->gtolCountsWithAll, i+pStart, &dof);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF, dof, &patch->patchStateWithAll[i]);CHKERRQ(ierr);
      ierr = VecSetUp(patch->patchStateWithAll[i]);CHKERRQ(ierr);
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
  ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localState, patch->patchState[i], INSERT_VALUES, SCATTER_FORWARD, SCATTER_INTERIOR);CHKERRQ(ierr);
  ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localState, patch->patchStateWithAll[i], INSERT_VALUES, SCATTER_FORWARD, SCATTER_WITHALL);CHKERRQ(ierr);

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

  if (patch->patchStateWithAll) {
    for (i = 0; i < patch->npatch; ++i) {ierr = VecDestroy(&patch->patchStateWithAll[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->patchStateWithAll);CHKERRQ(ierr);
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

static PetscErrorCode PCUpdateMultiplicative_PATCH_Nonlinear(PC pc, PetscInt i, PetscInt pStart)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCPatch_ScatterLocal_Private(pc, i + pStart, patch->patchUpdate[i], patch->localState, ADD_VALUES, SCATTER_REVERSE, SCATTER_INTERIOR);CHKERRQ(ierr);
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
  const char    *prefix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
    ierr = PetscViewerASCIIPrintf(viewer,"SNESPATCH\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PCView(patch->pc, viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes, &state);CHKERRQ(ierr);
  ierr = SNESGetSolutionUpdate(snes, &update);CHKERRQ(ierr);
  ierr = SNESGetRhs(snes, &rhs);CHKERRQ(ierr);

  ierr = SNESGetFunction(snes, &residual, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESGetLineSearch(snes, &ls);CHKERRQ(ierr);

  ierr = SNESSetConvergedReason(snes, SNES_CONVERGED_ITERATING);CHKERRQ(ierr);
  ierr = VecSet(update, 0.0);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, state, residual);CHKERRQ(ierr);

  ierr = VecNorm(state, NORM_2, &xnorm);CHKERRQ(ierr);
  ierr = VecNorm(residual, NORM_2, &fnorm);CHKERRQ(ierr);
  snes->ttol = fnorm*snes->rtol;

  if (snes->ops->converged) {
    ierr = (*snes->ops->converged)(snes,its,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  } else {
    ierr = SNESConvergedSkip(snes,its,xnorm,ynorm,fnorm,&snes->reason,0);CHKERRQ(ierr);
  }
  ierr = SNESLogConvergenceHistory(snes, fnorm, 0);CHKERRQ(ierr); /* should we count lits from the patches? */
  ierr = SNESMonitor(snes, its, fnorm);CHKERRQ(ierr);

  /* The main solver loop */
  for (its = 0; its < snes->max_its; its++) {

    ierr = SNESSetIterationNumber(snes, its);CHKERRQ(ierr);

    /* Scatter state vector to overlapped vector on all patches.
       The vector pcpatch->localState is scattered to each patch
       in PCApply_PATCH_Nonlinear. */
    ierr = VecGetArrayRead(state, &globalState);CHKERRQ(ierr);
    ierr = VecGetArray(pcpatch->localState, &localState);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(pcpatch->sectionSF, MPIU_SCALAR, globalState, localState);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pcpatch->sectionSF, MPIU_SCALAR, globalState, localState);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcpatch->localState, &localState);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(state, &globalState);CHKERRQ(ierr);

    /* The looping over patches happens here */
    ierr = PCApply(patch->pc, rhs, update);CHKERRQ(ierr);

    /* Apply a line search. This will often be basic with
       damping = 1/(max number of patches a dof can be in),
       but not always */
    ierr = VecScale(update, -1.0);CHKERRQ(ierr);
    ierr = SNESLineSearchApply(ls, state, residual, &fnorm, update);CHKERRQ(ierr);

    ierr = VecNorm(state, NORM_2, &xnorm);CHKERRQ(ierr);
    ierr = VecNorm(update, NORM_2, &ynorm);CHKERRQ(ierr);

    if (snes->ops->converged) {
      ierr = (*snes->ops->converged)(snes,its,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    } else {
      ierr = SNESConvergedSkip(snes,its,xnorm,ynorm,fnorm,&snes->reason,0);CHKERRQ(ierr);
    }
    ierr = SNESLogConvergenceHistory(snes, fnorm, 0);CHKERRQ(ierr); /* FIXME: should we count lits? */
    ierr = SNESMonitor(snes, its, fnorm);CHKERRQ(ierr);
  }

  if (its == snes->max_its) { ierr = SNESSetConvergedReason(snes, SNES_DIVERGED_MAX_IT);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/*MC
  SNESPATCH - Solve a nonlinear problem by composing together many nonlinear solvers on patches

  Level: intermediate

.seealso:  SNESCreate(), SNESSetType(), SNESType (for list of available types), SNES,
           PCPATCH

   References:
.  1. - Peter R. Brune, Matthew G. Knepley, Barry F. Smith, and Xuemin Tu, "Composing Scalable Nonlinear Algebraic Solvers", SIAM Review, 57(4), 2015

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_Patch(SNES snes)
{
  PetscErrorCode ierr;
  SNES_Patch     *patch;
  PC_PATCH       *patchpc;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  ierr = PetscNewLog(snes, &patch);CHKERRQ(ierr);

  snes->ops->solve          = SNESSolve_Patch;
  snes->ops->setup          = SNESSetUp_Patch;
  snes->ops->reset          = SNESReset_Patch;
  snes->ops->destroy        = SNESDestroy_Patch;
  snes->ops->setfromoptions = SNESSetFromOptions_Patch;
  snes->ops->view           = SNESView_Patch;

  ierr = SNESGetLineSearch(snes,&linesearch);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);CHKERRQ(ierr);
  snes->usesksp        = PETSC_FALSE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  snes->data = (void *) patch;
  ierr = PCCreate(PetscObjectComm((PetscObject) snes), &patch->pc);CHKERRQ(ierr);
  ierr = PCSetType(patch->pc, PCPATCH);CHKERRQ(ierr);

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
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject)snes), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch SNES\n");
  ierr = PCSetDM(patch->pc, dm);CHKERRQ(ierr);
  ierr = PCPatchSetDiscretisationInfo(patch->pc, nsubspaces, dms, bs, nodesPerCell, cellNodeMap, subspaceOffsets, numGhostBcs, ghostBcNodes, numGlobalBcs, globalBcNodes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetComputeOperator(SNES snes, PetscErrorCode (*func)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
{
  SNES_Patch    *patch = (SNES_Patch *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCPatchSetComputeOperator(patch->pc, func, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SNESPatchSetComputeFunction(SNES snes, PetscErrorCode (*func)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
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
