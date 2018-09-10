#include <petsc/private/pcpatchimpl.h>     /*I "petscpc.h" I*/
#include <petsc/private/dmpleximpl.h> /* For DMPlexComputeJacobian_Patch_Internal() */
#include <petscsf.h>
#include <petscbt.h>
#include <petscds.h>

PetscLogEvent PC_Patch_CreatePatches, PC_Patch_ComputeOp, PC_Patch_Solve, PC_Patch_Scatter, PC_Patch_Apply, PC_Patch_Prealloc;

PETSC_STATIC_INLINE PetscErrorCode ObjectView(PetscObject obj, PetscViewer viewer, PetscViewerFormat format)
{
  PetscErrorCode ierr;

  ierr = PetscViewerPushFormat(viewer, format);CHKERRQ(ierr);
  ierr = PetscObjectView(obj, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  return(0);
}

static PetscErrorCode PCPatchConstruct_Star(void *vpatch, DM dm, PetscInt point, PetscHSetI ht)
{
  PetscInt       starSize;
  PetscInt      *star = NULL, si;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscHSetIClear(ht);
  /* To start with, add the point we care about */
  ierr = PetscHSetIAdd(ht, point);CHKERRQ(ierr);
  /* Loop over all the points that this point connects to */
  ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  for (si = 0; si < starSize*2; si += 2) {ierr = PetscHSetIAdd(ht, star[si]);CHKERRQ(ierr);}
  ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchConstruct_Vanka(void *vpatch, DM dm, PetscInt point, PetscHSetI ht)
{
  PC_PATCH      *patch = (PC_PATCH *) vpatch;
  PetscInt       starSize;
  PetscInt      *star = NULL;
  PetscBool      shouldIgnore = PETSC_FALSE;
  PetscInt       cStart, cEnd, iStart, iEnd, si;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHSetIClear(ht);CHKERRQ(ierr);
  /* To start with, add the point we care about */
  ierr = PetscHSetIAdd(ht, point);CHKERRQ(ierr);
  /* Should we ignore any points of a certain dimension? */
  if (patch->vankadim >= 0) {
    shouldIgnore = PETSC_TRUE;
    ierr = DMPlexGetDepthStratum(dm, patch->vankadim, &iStart, &iEnd);CHKERRQ(ierr);
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  /* Loop over all the cells that this point connects to */
  ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  for (si = 0; si < starSize*2; si += 2) {
    const PetscInt cell = star[si];
    PetscInt       closureSize;
    PetscInt      *closure = NULL, ci;

    if (cell < cStart || cell >= cEnd) continue;
    /* now loop over all entities in the closure of that cell */
    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (ci = 0; ci < closureSize*2; ci += 2) {
      const PetscInt newpoint = closure[ci];

      /* We've been told to ignore entities of this type.*/
      if (shouldIgnore && newpoint >= iStart && newpoint < iEnd) continue;
      ierr = PetscHSetIAdd(ht, newpoint);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The user's already set the patches in patch->userIS. Build the hash tables */
static PetscErrorCode PCPatchConstruct_User(void *vpatch, DM dm, PetscInt point, PetscHSetI ht)
{
  PC_PATCH       *patch   = (PC_PATCH *) vpatch;
  IS              patchis = patch->userIS[point];
  PetscInt        n;
  const PetscInt *patchdata;
  PetscInt        pStart, pEnd, i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscHSetIClear(ht);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = ISGetLocalSize(patchis, &n);CHKERRQ(ierr);
  ierr = ISGetIndices(patchis, &patchdata);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) {
    const PetscInt ownedpoint = patchdata[i];

    if (ownedpoint < pStart || ownedpoint >= pEnd) {
      SETERRQ3(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D was not in [%D, %D)", ownedpoint, pStart, pEnd);
    }
    ierr = PetscHSetIAdd(ht, ownedpoint);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(patchis, &patchdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchCreateDefaultSF_Private(PC pc, PetscInt n, const PetscSF *sf, const PetscInt *bs)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n == 1 && bs[0] == 1) {
    patch->defaultSF = sf[0];
    ierr = PetscObjectReference((PetscObject) patch->defaultSF);CHKERRQ(ierr);
  } else {
    PetscInt     allRoots = 0, allLeaves = 0;
    PetscInt     leafOffset = 0;
    PetscInt    *ilocal = NULL;
    PetscSFNode *iremote = NULL;
    PetscInt    *remoteOffsets = NULL;
    PetscInt     index = 0;
    PetscHMapI   rankToIndex;
    PetscInt     numRanks = 0;
    PetscSFNode *remote = NULL;
    PetscSF      rankSF;
    PetscInt    *ranks = NULL;
    PetscInt    *offsets = NULL;
    MPI_Datatype contig;
    PetscHSetI   ranksUniq;

    /* First figure out how many dofs there are in the concatenated numbering.
     * allRoots: number of owned global dofs;
     * allLeaves: number of visible dofs (global + ghosted).
     */
    for (i = 0; i < n; ++i) {
      PetscInt nroots, nleaves;

      ierr = PetscSFGetGraph(sf[i], &nroots, &nleaves, NULL, NULL);CHKERRQ(ierr);
      allRoots  += nroots * bs[i];
      allLeaves += nleaves * bs[i];
    }
    ierr = PetscMalloc1(allLeaves, &ilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(allLeaves, &iremote);CHKERRQ(ierr);
    /* Now build an SF that just contains process connectivity. */
    ierr = PetscHSetICreate(&ranksUniq);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      const PetscMPIInt *ranks = NULL;
      PetscInt           nranks, j;

      ierr = PetscSFSetUp(sf[i]);CHKERRQ(ierr);
      ierr = PetscSFGetRanks(sf[i], &nranks, &ranks, NULL, NULL, NULL);CHKERRQ(ierr);
      /* These are all the ranks who communicate with me. */
      for (j = 0; j < nranks; ++j) {
        ierr = PetscHSetIAdd(ranksUniq, (PetscInt) ranks[j]);CHKERRQ(ierr);
      }
    }
    ierr = PetscHSetIGetSize(ranksUniq, &numRanks);CHKERRQ(ierr);
    ierr = PetscMalloc1(numRanks, &remote);CHKERRQ(ierr);
    ierr = PetscMalloc1(numRanks, &ranks);CHKERRQ(ierr);
    ierr = PetscHSetIGetElems(ranksUniq, &index, ranks);CHKERRQ(ierr);

    ierr = PetscHMapICreate(&rankToIndex);CHKERRQ(ierr);
    for (i = 0; i < numRanks; ++i) {
      remote[i].rank  = ranks[i];
      remote[i].index = 0;
      ierr = PetscHMapISet(rankToIndex, ranks[i], i);CHKERRQ(ierr);
    }
    ierr = PetscFree(ranks);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&ranksUniq);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject) pc), &rankSF);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(rankSF, 1, numRanks, NULL, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetUp(rankSF);CHKERRQ(ierr);
    /* OK, use it to communicate the root offset on the remote
     * processes for each subspace. */
    ierr = PetscMalloc1(n, &offsets);CHKERRQ(ierr);
    ierr = PetscMalloc1(n*numRanks, &remoteOffsets);CHKERRQ(ierr);

    offsets[0] = 0;
    for (i = 1; i < n; ++i) {
      PetscInt nroots;

      ierr = PetscSFGetGraph(sf[i-1], &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
      offsets[i] = offsets[i-1] + nroots*bs[i-1];
    }
    /* Offsets are the offsets on the current process of the
     * global dof numbering for the subspaces. */
    ierr = MPI_Type_contiguous(n, MPIU_INT, &contig);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&contig);CHKERRQ(ierr);

    ierr = PetscSFBcastBegin(rankSF, contig, offsets, remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(rankSF, contig, offsets, remoteOffsets);CHKERRQ(ierr);
    ierr = MPI_Type_free(&contig);CHKERRQ(ierr);
    ierr = PetscFree(offsets);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&rankSF);CHKERRQ(ierr);
    /* Now remoteOffsets contains the offsets on the remote
     * processes who communicate with me.  So now we can
     * concatenate the list of SFs into a single one. */
    index = 0;
    for (i = 0; i < n; ++i) {
      const PetscSFNode *remote = NULL;
      const PetscInt    *local  = NULL;
      PetscInt           nroots, nleaves, j;

      ierr = PetscSFGetGraph(sf[i], &nroots, &nleaves, &local, &remote);CHKERRQ(ierr);
      for (j = 0; j < nleaves; ++j) {
        PetscInt rank = remote[j].rank;
        PetscInt idx, rootOffset, k;

        ierr = PetscHMapIGet(rankToIndex, rank, &idx);CHKERRQ(ierr);
        if (idx == -1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Didn't find rank, huh?");
        /* Offset on given rank for ith subspace */
        rootOffset = remoteOffsets[n*idx + i];
        for (k = 0; k < bs[i]; ++k) {
          ilocal[index]        = (local ? local[j] : j)*bs[i] + k + leafOffset;
          iremote[index].rank  = remote[j].rank;
          iremote[index].index = remote[j].index*bs[i] + k + rootOffset;
          ++index;
        }
      }
      leafOffset += nleaves * bs[i];
    }
    ierr = PetscHMapIDestroy(&rankToIndex);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)pc), &patch->defaultSF);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(patch->defaultSF, allRoots, allLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetIgnoreDim(PC pc, PetscInt dim)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  patch->ignoredim = dim;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchGetIgnoreDim(PC pc, PetscInt *dim)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  *dim = patch->ignoredim;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetSaveOperators(PC pc, PetscBool flg)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  patch->save_operators = flg;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchGetSaveOperators(PC pc, PetscBool *flg)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  *flg = patch->save_operators;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetPartitionOfUnity(PC pc, PetscBool flg)
{
    PC_PATCH *patch = (PC_PATCH *) pc->data;
    PetscFunctionBegin;
    patch->partition_of_unity = flg;
    PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchGetPartitionOfUnity(PC pc, PetscBool *flg)
{
    PC_PATCH *patch = (PC_PATCH *) pc->data;
    PetscFunctionBegin;
    *flg = patch->partition_of_unity;
    PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetSubMatType(PC pc, MatType sub_mat_type)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (patch->sub_mat_type) {ierr = PetscFree(patch->sub_mat_type);CHKERRQ(ierr);}
  ierr = PetscStrallocpy(sub_mat_type, (char **) &patch->sub_mat_type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchGetSubMatType(PC pc, MatType *sub_mat_type)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  *sub_mat_type = patch->sub_mat_type;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetCellNumbering(PC pc, PetscSection cellNumbering)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  patch->cellNumbering = cellNumbering;
  ierr = PetscObjectReference((PetscObject) cellNumbering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchGetCellNumbering(PC pc, PetscSection *cellNumbering)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  *cellNumbering = patch->cellNumbering;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetConstructType(PC pc, PCPatchConstructType ctype, PetscErrorCode (*func)(PC, PetscInt *, IS **, IS *, void *), void *ctx)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  patch->ctype = ctype;
  switch (ctype) {
  case PC_PATCH_STAR:
    patch->user_patches     = PETSC_FALSE;
    patch->patchconstructop = PCPatchConstruct_Star;
    break;
  case PC_PATCH_VANKA:
    patch->user_patches     = PETSC_FALSE;
    patch->patchconstructop = PCPatchConstruct_Vanka;
    break;
  case PC_PATCH_USER:
  case PC_PATCH_PYTHON:
    patch->user_patches     = PETSC_TRUE;
    patch->patchconstructop = PCPatchConstruct_User;
    if (func) {
      patch->userpatchconstructionop = func;
      patch->userpatchconstructctx   = ctx;
    }
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) pc), PETSC_ERR_USER, "Unknown patch construction type %D", (PetscInt) patch->ctype);
  }
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchGetConstructType(PC pc, PCPatchConstructType *ctype, PetscErrorCode (**func)(PC, PetscInt *, IS **, IS *, void *), void **ctx)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  *ctype = patch->ctype;
  switch (patch->ctype) {
  case PC_PATCH_STAR:
  case PC_PATCH_VANKA:
    break;
  case PC_PATCH_USER:
  case PC_PATCH_PYTHON:
    *func = patch->userpatchconstructionop;
    *ctx  = patch->userpatchconstructctx;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) pc), PETSC_ERR_USER, "Unknown patch construction type %D", (PetscInt) patch->ctype);
  }
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetDiscretisationInfo(PC pc, PetscInt nsubspaces, DM *dms, PetscInt *bs, PetscInt *nodesPerCell, const PetscInt **cellNodeMap,
                                            const PetscInt *subspaceOffsets, PetscInt numGhostBcs, const PetscInt *ghostBcNodes, PetscInt numGlobalBcs, const PetscInt *globalBcNodes)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  DM             dm;
  PetscSF       *sfs;
  PetscInt       cStart, cEnd, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsubspaces, &sfs);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsubspaces, &patch->dofSection);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsubspaces, &patch->bs);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsubspaces, &patch->nodesPerCell);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsubspaces, &patch->cellNodeMap);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsubspaces+1, &patch->subspaceOffsets);CHKERRQ(ierr);

  patch->nsubspaces       = nsubspaces;
  patch->totalDofsPerCell = 0;
  for (i = 0; i < nsubspaces; ++i) {
    ierr = DMGetDefaultSection(dms[i], &patch->dofSection[i]);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) patch->dofSection[i]);CHKERRQ(ierr);
    ierr = DMGetDefaultSF(dms[i], &sfs[i]);CHKERRQ(ierr);
    patch->bs[i]              = bs[i];
    patch->nodesPerCell[i]    = nodesPerCell[i];
    patch->totalDofsPerCell  += nodesPerCell[i]*bs[i];
    ierr = PetscMalloc1((cEnd-cStart)*nodesPerCell[i]*bs[i], &patch->cellNodeMap[i]);CHKERRQ(ierr);
    for (j = 0; j < (cEnd-cStart)*nodesPerCell[i]*bs[i]; ++j) patch->cellNodeMap[i][j] = cellNodeMap[i][j];
    patch->subspaceOffsets[i] = subspaceOffsets[i];
  }
  ierr = PCPatchCreateDefaultSF_Private(pc, nsubspaces, sfs, patch->bs);CHKERRQ(ierr);
  ierr = PetscFree(sfs);CHKERRQ(ierr);

  patch->subspaceOffsets[nsubspaces] = subspaceOffsets[nsubspaces];
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGhostBcs, ghostBcNodes, PETSC_COPY_VALUES, &patch->ghostBcNodes);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalBcs, globalBcNodes, PETSC_COPY_VALUES, &patch->globalBcNodes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetDiscretisationInfoCombined(PC pc, DM dm, PetscInt *nodesPerCell, const PetscInt **cellNodeMap, PetscInt numGhostBcs, const PetscInt *ghostBcNodes, PetscInt numGlobalBcs, const PetscInt *globalBcNodes)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       cStart, cEnd, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  patch->combined = PETSC_TRUE;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &patch->nsubspaces);CHKERRQ(ierr);
  ierr = PetscCalloc1(patch->nsubspaces, &patch->dofSection);CHKERRQ(ierr);
  ierr = PetscMalloc1(patch->nsubspaces, &patch->bs);CHKERRQ(ierr);
  ierr = PetscMalloc1(patch->nsubspaces, &patch->nodesPerCell);CHKERRQ(ierr);
  ierr = PetscMalloc1(patch->nsubspaces, &patch->cellNodeMap);CHKERRQ(ierr);
  ierr = PetscCalloc1(patch->nsubspaces+1, &patch->subspaceOffsets);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &patch->dofSection[0]);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) patch->dofSection[0]);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(patch->dofSection[0], &patch->subspaceOffsets[patch->nsubspaces]);CHKERRQ(ierr);
  patch->totalDofsPerCell = 0;
  for (i = 0; i < patch->nsubspaces; ++i) {
    patch->bs[i]             = 1;
    patch->nodesPerCell[i]   = nodesPerCell[i];
    patch->totalDofsPerCell += nodesPerCell[i];
    ierr = PetscMalloc1((cEnd-cStart)*nodesPerCell[i], &patch->cellNodeMap[i]);CHKERRQ(ierr);
    for (j = 0; j < (cEnd-cStart)*nodesPerCell[i]; ++j) patch->cellNodeMap[i][j] = cellNodeMap[i][j];
  }
  ierr = DMGetDefaultSF(dm, &patch->defaultSF);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) patch->defaultSF);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGhostBcs, ghostBcNodes, PETSC_COPY_VALUES, &patch->ghostBcNodes);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalBcs, globalBcNodes, PETSC_COPY_VALUES, &patch->globalBcNodes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C

  PCPatchSetComputeOperator - Set the callback used to compute patch matrices

  Input Parameters:
+ pc   - The PC
. func - The callback
- ctx  - The user context

  Level: advanced

  Note:
  The callback has signature:
+  usercomputeop(pc, point, mat, cellIS, n, u, ctx)
+  pc     - The PC
+  point  - The point
+  mat    - The patch matrix
+  cellIS - An array of the cell numbers
+  n      - The size of g2l
+  g2l    - The global to local dof translation table
+  ctx    - The user context
  and can assume that the matrix entries have been set to zero before the call.

.seealso: PCPatchGetComputeOperator(), PCPatchSetDiscretisationInfo()
@*/
PetscErrorCode PCPatchSetComputeOperator(PC pc, PetscErrorCode (*func)(PC, PetscInt, Mat, IS, PetscInt, const PetscInt *, void *), void *ctx)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  patch->usercomputeop  = func;
  patch->usercomputectx = ctx;
  PetscFunctionReturn(0);
}

/* On entry, ht contains the topological entities whose dofs we are responsible for solving for;
   on exit, cht contains all the topological entities we need to compute their residuals.
   In full generality this should incorporate knowledge of the sparsity pattern of the matrix;
   here we assume a standard FE sparsity pattern.*/
/* TODO: Use DMPlexGetAdjacency() */
static PetscErrorCode PCPatchCompleteCellPatch(PC pc, PetscHSetI ht, PetscHSetI cht)
{
  DM             dm;
  PetscHashIter  hi;
  PetscInt       point;
  PetscInt      *star = NULL, *closure = NULL;
  PetscInt       ignoredim, iStart = 0, iEnd = -1, starSize, closureSize, si, ci;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  ierr = PCPatchGetIgnoreDim(pc, &ignoredim);CHKERRQ(ierr);
  if (ignoredim >= 0) {ierr = DMPlexGetDepthStratum(dm, ignoredim, &iStart, &iEnd);CHKERRQ(ierr);}
  ierr = PetscHSetIClear(cht);CHKERRQ(ierr);
  PetscHashIterBegin(ht, hi);
  while (!PetscHashIterAtEnd(ht, hi)) {

    PetscHashIterGetKey(ht, hi, point);
    PetscHashIterNext(ht, hi);

    /* Loop over all the cells that this point connects to */
    ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for (si = 0; si < starSize*2; si += 2) {
      const PetscInt ownedpoint = star[si];
      /* TODO Check for point in cht before running through closure again */
      /* now loop over all entities in the closure of that cell */
      ierr = DMPlexGetTransitiveClosure(dm, ownedpoint, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (ci = 0; ci < closureSize*2; ci += 2) {
        const PetscInt seenpoint = closure[ci];
        if (ignoredim >= 0 && seenpoint >= iStart && seenpoint < iEnd) continue;
        ierr = PetscHSetIAdd(cht, seenpoint);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, NULL, &closure);CHKERRQ(ierr);
  ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_FALSE, NULL, &star);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchGetGlobalDofs(PC pc, PetscSection dofSection[], PetscInt f, PetscBool combined, PetscInt p, PetscInt *dof, PetscInt *off)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (combined) {
    if (f < 0) {
      if (dof) {ierr = PetscSectionGetDof(dofSection[0], p, dof);CHKERRQ(ierr);}
      if (off) {ierr = PetscSectionGetOffset(dofSection[0], p, off);CHKERRQ(ierr);}
    } else {
      if (dof) {ierr = PetscSectionGetFieldDof(dofSection[0], p, f, dof);CHKERRQ(ierr);}
      if (off) {ierr = PetscSectionGetFieldOffset(dofSection[0], p, f, off);CHKERRQ(ierr);}
    }
  } else {
    if (f < 0) {
      PC_PATCH *patch = (PC_PATCH *) pc->data;
      PetscInt  fdof, g;

      if (dof) {
        *dof = 0;
        for (g = 0; g < patch->nsubspaces; ++g) {
          ierr = PetscSectionGetDof(dofSection[g], p, &fdof);CHKERRQ(ierr);
          *dof += fdof;
        }
      }
      if (off) {ierr = PetscSectionGetOffset(dofSection[0], p, off);CHKERRQ(ierr);}
    } else {
      if (dof) {ierr = PetscSectionGetDof(dofSection[f], p, dof);CHKERRQ(ierr);}
      if (off) {ierr = PetscSectionGetOffset(dofSection[f], p, off);CHKERRQ(ierr);}
    }
  }
  PetscFunctionReturn(0);
}

/* Given a hash table with a set of topological entities (pts), compute the degrees of
   freedom in global concatenated numbering on those entities.
   For Vanka smoothing, this needs to do something special: ignore dofs of the
   constraint subspace on entities that aren't the base entity we're building the patch
   around. */
static PetscErrorCode PCPatchGetPointDofs(PC pc, PetscHSetI pts, PetscHSetI dofs, PetscInt base, PetscInt exclude_subspace)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscHashIter  hi;
  PetscInt       ldof, loff;
  PetscInt       k, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHSetIClear(dofs);CHKERRQ(ierr);
  for (k = 0; k < patch->nsubspaces; ++k) {
    PetscInt subspaceOffset = patch->subspaceOffsets[k];
    PetscInt bs             = patch->bs[k];
    PetscInt j, l;

    if (k == exclude_subspace) {
      /* only get this subspace dofs at the base entity, not any others */
      ierr = PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, base, &ldof, &loff);CHKERRQ(ierr);
      if (0 == ldof) continue;
      for (j = loff; j < ldof + loff; ++j) {
        for (l = 0; l < bs; ++l) {
          PetscInt dof = bs*j + l + subspaceOffset;
          ierr = PetscHSetIAdd(dofs, dof);CHKERRQ(ierr);
        }
      }
      continue; /* skip the other dofs of this subspace */
    }

    PetscHashIterBegin(pts, hi);
    while (!PetscHashIterAtEnd(pts, hi)) {
      PetscHashIterGetKey(pts, hi, p);
      PetscHashIterNext(pts, hi);
      ierr = PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, p, &ldof, &loff);CHKERRQ(ierr);
      if (0 == ldof) continue;
      for (j = loff; j < ldof + loff; ++j) {
        for (l = 0; l < bs; ++l) {
          PetscInt dof = bs*j + l + subspaceOffset;
          ierr = PetscHSetIAdd(dofs, dof);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Given two hash tables A and B, compute the keys in B that are not in A, and put them in C */
static PetscErrorCode PCPatchComputeSetDifference_Private(PetscHSetI A, PetscHSetI B, PetscHSetI C)
{
  PetscHashIter  hi;
  PetscInt       key;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHSetIClear(C);CHKERRQ(ierr);
  PetscHashIterBegin(B, hi);
  while (!PetscHashIterAtEnd(B, hi)) {
    PetscHashIterGetKey(B, hi, key);
    PetscHashIterNext(B, hi);
    ierr = PetscHSetIHas(A, key, &flg);CHKERRQ(ierr);
    if (!flg) {ierr = PetscHSetIAdd(C, key);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/*
 * PCPatchCreateCellPatches - create patches.
 *
 * Input Parameters:
 * + dm - The DMPlex object defining the mesh
 *
 * Output Parameters:
 * + cellCounts  - Section with counts of cells around each vertex
 * . cells       - IS of the cell point indices of cells in each patch
 * . pointCounts - Section with counts of cells around each vertex
 * - point       - IS of the cell point indices of cells in each patch
 */
static PetscErrorCode PCPatchCreateCellPatches(PC pc)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  DMLabel         ghost = NULL;
  DM              dm, plex;
  PetscHSetI      ht, cht;
  PetscSection    cellCounts,  pointCounts;
  PetscInt       *cellsArray, *pointsArray;
  PetscInt        numCells,    numPoints;
  const PetscInt *leaves;
  PetscInt        nleaves, pStart, pEnd, cStart, cEnd, vStart, vEnd, v;
  PetscBool       isFiredrake;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Used to keep track of the cells in the patch. */
  ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&cht);CHKERRQ(ierr);

  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch PC\n");
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexGetChart(plex, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);

  if (patch->user_patches) {
    ierr = patch->userpatchconstructionop(pc, &patch->npatch, &patch->userIS, &patch->iterationSet, patch->userpatchconstructctx);CHKERRQ(ierr);
    vStart = 0; vEnd = patch->npatch;
  } else if (patch->codim < 0) {
    if (patch->dim < 0) {ierr = DMPlexGetDepthStratum(plex,  0,            &vStart, &vEnd);CHKERRQ(ierr);}
    else                {ierr = DMPlexGetDepthStratum(plex,  patch->dim,   &vStart, &vEnd);CHKERRQ(ierr);}
  } else                {ierr = DMPlexGetHeightStratum(plex, patch->codim, &vStart, &vEnd);CHKERRQ(ierr);}
  patch->npatch = vEnd - vStart;

  /* These labels mark the owned points.  We only create patches around points that this process owns. */
  ierr = DMHasLabel(dm, "pyop2_ghost", &isFiredrake);CHKERRQ(ierr);
  if (isFiredrake) {
    ierr = DMGetLabel(dm, "pyop2_ghost", &ghost);CHKERRQ(ierr);
    ierr = DMLabelCreateIndex(ghost, pStart, pEnd);CHKERRQ(ierr);
  } else {
    PetscSF sf;

    ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL);CHKERRQ(ierr);
    nleaves = PetscMax(nleaves, 0);
  }

  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->cellCounts);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->cellCounts, "Patch Cell Layout");CHKERRQ(ierr);
  cellCounts = patch->cellCounts;
  ierr = PetscSectionSetChart(cellCounts, vStart, vEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->pointCounts);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->pointCounts, "Patch Point Layout");CHKERRQ(ierr);
  pointCounts = patch->pointCounts;
  ierr = PetscSectionSetChart(pointCounts, vStart, vEnd);CHKERRQ(ierr);
  /* Count cells and points in the patch surrounding each entity */
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIter hi;
    PetscInt       chtSize, loc = -1;
    PetscBool      flg;

    if (!patch->user_patches) {
      if (ghost) {ierr = DMLabelHasPoint(ghost, v, &flg);CHKERRQ(ierr);}
      else       {ierr = PetscFindInt(v, nleaves, leaves, &loc); flg = loc >=0 ? PETSC_TRUE : PETSC_FALSE;}
      /* Not an owned entity, don't make a cell patch. */
      if (flg) continue;
    }

    ierr = patch->patchconstructop((void *) patch, dm, v, ht);CHKERRQ(ierr);
    ierr = PCPatchCompleteCellPatch(pc, ht, cht);CHKERRQ(ierr);
    ierr = PetscHSetIGetSize(cht, &chtSize);CHKERRQ(ierr);
    /* empty patch, continue */
    if (chtSize == 0) continue;

    /* safe because size(cht) > 0 from above */
    PetscHashIterBegin(cht, hi);
    while (!PetscHashIterAtEnd(cht, hi)) {
      PetscInt point, pdof;

      PetscHashIterGetKey(cht, hi, point);
      ierr = PCPatchGetGlobalDofs(pc, patch->dofSection, -1, patch->combined, point, &pdof, NULL);CHKERRQ(ierr);
      if (pdof)                            {ierr = PetscSectionAddDof(pointCounts, v, 1);CHKERRQ(ierr);}
      if (point >= cStart && point < cEnd) {ierr = PetscSectionAddDof(cellCounts, v, 1);CHKERRQ(ierr);}
      PetscHashIterNext(cht, hi);
    }
  }
  if (isFiredrake) {ierr = DMLabelDestroyIndex(ghost);CHKERRQ(ierr);}

  ierr = PetscSectionSetUp(cellCounts);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(cellCounts, &numCells);CHKERRQ(ierr);
  ierr = PetscMalloc1(numCells, &cellsArray);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(pointCounts);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(pointCounts, &numPoints);CHKERRQ(ierr);
  ierr = PetscMalloc1(numPoints, &pointsArray);CHKERRQ(ierr);

  /* Now that we know how much space we need, run through again and actually remember the cells. */
  for (v = vStart; v < vEnd; v++ ) {
    PetscHashIter hi;
    PetscInt       dof, off, cdof, coff, pdof, n = 0, cn = 0;

    ierr = PetscSectionGetDof(pointCounts, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pointCounts, v, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(cellCounts, v, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cellCounts, v, &coff);CHKERRQ(ierr);
    if (dof <= 0) continue;
    ierr = patch->patchconstructop((void *) patch, dm, v, ht);CHKERRQ(ierr);
    ierr = PCPatchCompleteCellPatch(pc, ht, cht);CHKERRQ(ierr);
    PetscHashIterBegin(cht, hi);
    while (!PetscHashIterAtEnd(cht, hi)) {
      PetscInt point;

      PetscHashIterGetKey(cht, hi, point);
      ierr = PCPatchGetGlobalDofs(pc, patch->dofSection, -1, patch->combined, point, &pdof, NULL);CHKERRQ(ierr);
      if (pdof)                            {pointsArray[off + n++] = point;}
      if (point >= cStart && point < cEnd) {cellsArray[coff + cn++] = point;}
      PetscHashIterNext(cht, hi);
    }
    if (cn != cdof) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of cells in patch %D is %D, but should be %D", v, cn, cdof);
    if (n  != dof)  SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of points in patch %D is %D, but should be %D", v, n, dof);
  }
  ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);
  ierr = PetscHSetIDestroy(&cht);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF, numCells,  cellsArray,  PETSC_OWN_POINTER, &patch->cells);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->cells,  "Patch Cells");CHKERRQ(ierr);
  if (patch->viewCells) {
    ierr = ObjectView((PetscObject) patch->cellCounts, patch->viewerCells, patch->formatCells);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) patch->cells,      patch->viewerCells, patch->formatCells);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numPoints, pointsArray, PETSC_OWN_POINTER, &patch->points);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->points, "Patch Points");CHKERRQ(ierr);
  if (patch->viewPoints) {
    ierr = ObjectView((PetscObject) patch->pointCounts, patch->viewerPoints, patch->formatPoints);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) patch->points,      patch->viewerPoints, patch->formatPoints);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
 * PCPatchCreateCellPatchDiscretisationInfo - Build the dof maps for cell patches
 *
 * Input Parameters:
 * + dm - The DMPlex object defining the mesh
 * . cellCounts - Section with counts of cells around each vertex
 * . cells - IS of the cell point indices of cells in each patch
 * . cellNumbering - Section mapping plex cell points to Firedrake cell indices.
 * . nodesPerCell - number of nodes per cell.
 * - cellNodeMap - map from cells to node indices (nodesPerCell * numCells)
 *
 * Output Parameters:
 * + dofs - IS of local dof numbers of each cell in the patch, where local is a patch local numbering
 * . gtolCounts - Section with counts of dofs per cell patch
 * - gtol - IS mapping from global dofs to local dofs for each patch.
 */
static PetscErrorCode PCPatchCreateCellPatchDiscretisationInfo(PC pc)
{
  PC_PATCH       *patch           = (PC_PATCH *) pc->data;
  PetscSection    cellCounts      = patch->cellCounts;
  PetscSection    pointCounts     = patch->pointCounts;
  PetscSection    gtolCounts;
  IS              cells           = patch->cells;
  IS              points          = patch->points;
  PetscSection    cellNumbering   = patch->cellNumbering;
  PetscInt        Nf              = patch->nsubspaces;
  PetscInt        numCells, numPoints;
  PetscInt        numDofs;
  PetscInt        numGlobalDofs;
  PetscInt        totalDofsPerCell = patch->totalDofsPerCell;
  PetscInt        vStart, vEnd, v;
  const PetscInt *cellsArray, *pointsArray;
  PetscInt       *newCellsArray   = NULL;
  PetscInt       *dofsArray       = NULL;
  PetscInt       *offsArray       = NULL;
  PetscInt       *asmArray        = NULL;
  PetscInt       *globalDofsArray = NULL;
  PetscInt        globalIndex     = 0;
  PetscInt        key             = 0;
  PetscInt        asmKey          = 0;
  DM              dm              = NULL;
  const PetscInt *bcNodes         = NULL;
  PetscHMapI      ht;
  PetscHSetI      globalBcs;
  PetscInt        numBcs;
  PetscHSetI      ownedpts, seenpts, owneddofs, seendofs, artificialbcs;
  PetscInt        pStart, pEnd, p, i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = PCGetDM(pc, &dm); CHKERRQ(ierr);
  /* dofcounts section is cellcounts section * dofPerCell */
  ierr = PetscSectionGetStorageSize(cellCounts, &numCells);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(patch->pointCounts, &numPoints);CHKERRQ(ierr);
  numDofs = numCells * totalDofsPerCell;
  ierr = PetscMalloc1(numDofs, &dofsArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(numPoints*Nf, &offsArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(numDofs, &asmArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(numCells, &newCellsArray);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(cellCounts, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCounts);CHKERRQ(ierr);
  gtolCounts = patch->gtolCounts;
  ierr = PetscSectionSetChart(gtolCounts, vStart, vEnd);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->gtolCounts, "Patch Global Index Section");CHKERRQ(ierr);

  /* Outside the patch loop, get the dofs that are globally-enforced Dirichlet
   conditions */
  ierr = PetscHSetICreate(&globalBcs);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->ghostBcNodes, &bcNodes); CHKERRQ(ierr);
  ierr = ISGetSize(patch->ghostBcNodes, &numBcs); CHKERRQ(ierr);
  for (i = 0; i < numBcs; ++i) {
    ierr = PetscHSetIAdd(globalBcs, bcNodes[i]);CHKERRQ(ierr); /* these are already in concatenated numbering */
  }
  ierr = ISRestoreIndices(patch->ghostBcNodes, &bcNodes); CHKERRQ(ierr);
  ierr = ISDestroy(&patch->ghostBcNodes); CHKERRQ(ierr); /* memory optimisation */

  /* Hash tables for artificial BC construction */
  ierr = PetscHSetICreate(&ownedpts);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&seenpts);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&owneddofs);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&seendofs);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&artificialbcs);CHKERRQ(ierr);

  ierr = ISGetIndices(cells, &cellsArray);CHKERRQ(ierr);
  ierr = ISGetIndices(points, &pointsArray);CHKERRQ(ierr);
  ierr = PetscHMapICreate(&ht);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt localIndex = 0;
    PetscInt dof, off, i, j, k, l;

    ierr = PetscHMapIClear(ht);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(cellCounts, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cellCounts, v, &off);CHKERRQ(ierr);
    if (dof <= 0) continue;

    /* Calculate the global numbers of the artificial BC dofs here first */
    ierr = patch->patchconstructop((void*)patch, dm, v, ownedpts); CHKERRQ(ierr);
    ierr = PCPatchCompleteCellPatch(pc, ownedpts, seenpts); CHKERRQ(ierr);
    ierr = PCPatchGetPointDofs(pc, ownedpts, owneddofs, v, patch->exclude_subspace); CHKERRQ(ierr);
    ierr = PCPatchGetPointDofs(pc, seenpts, seendofs, v, -1); CHKERRQ(ierr);
    ierr = PCPatchComputeSetDifference_Private(owneddofs, seendofs, artificialbcs); CHKERRQ(ierr);
    if (patch->viewPatches) {
      PetscHSetI globalbcdofs;
      PetscHashIter hi;
      MPI_Comm comm = PetscObjectComm((PetscObject)pc);

      ierr = PetscHSetICreate(&globalbcdofs);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: owned dofs:\n", v); CHKERRQ(ierr);
      PetscHashIterBegin(owneddofs, hi);
      while (!PetscHashIterAtEnd(owneddofs, hi)) {
        PetscInt globalDof;

        PetscHashIterGetKey(owneddofs, hi, globalDof);
        PetscHashIterNext(owneddofs, hi);
        ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof); CHKERRQ(ierr);
      }
      ierr = PetscSynchronizedPrintf(comm, "\n"); CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: seen dofs:\n", v); CHKERRQ(ierr);
      PetscHashIterBegin(seendofs, hi);
      while (!PetscHashIterAtEnd(seendofs, hi)) {
        PetscInt globalDof;
        PetscBool flg;

        PetscHashIterGetKey(seendofs, hi, globalDof);
        PetscHashIterNext(seendofs, hi);
        ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof); CHKERRQ(ierr);

        ierr = PetscHSetIHas(globalBcs, globalDof, &flg);CHKERRQ(ierr);
        if (flg) {ierr = PetscHSetIAdd(globalbcdofs, globalDof);CHKERRQ(ierr);}
      }
      ierr = PetscSynchronizedPrintf(comm, "\n"); CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: global BCs:\n", v); CHKERRQ(ierr);
      ierr = PetscHSetIGetSize(globalbcdofs, &numBcs);CHKERRQ(ierr);
      if (numBcs > 0) {
        PetscHashIterBegin(globalbcdofs, hi);
        while (!PetscHashIterAtEnd(globalbcdofs, hi)) {
          PetscInt globalDof;
          PetscHashIterGetKey(globalbcdofs, hi, globalDof);
          PetscHashIterNext(globalbcdofs, hi);
          ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof);CHKERRQ(ierr);
        }
      }
      ierr = PetscSynchronizedPrintf(comm, "\n");CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: artificial BCs:\n", v);CHKERRQ(ierr);
      ierr = PetscHSetIGetSize(artificialbcs, &numBcs);CHKERRQ(ierr);
      if (numBcs > 0) {
        PetscHashIterBegin(artificialbcs, hi);
        while (!PetscHashIterAtEnd(artificialbcs, hi)) {
          PetscInt globalDof;
          PetscHashIterGetKey(artificialbcs, hi, globalDof);
          PetscHashIterNext(artificialbcs, hi);
          ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof); CHKERRQ(ierr);
        }
      }
      ierr = PetscSynchronizedPrintf(comm, "\n\n"); CHKERRQ(ierr);
      ierr = PetscHSetIDestroy(&globalbcdofs);CHKERRQ(ierr);
    }
   for (k = 0; k < patch->nsubspaces; ++k) {
      const PetscInt *cellNodeMap    = patch->cellNodeMap[k];
      PetscInt        nodesPerCell   = patch->nodesPerCell[k];
      PetscInt        subspaceOffset = patch->subspaceOffsets[k];
      PetscInt        bs             = patch->bs[k];

      for (i = off; i < off + dof; ++i) {
        /* Walk over the cells in this patch. */
        const PetscInt c    = cellsArray[i];
        PetscInt       cell = c;

        /* TODO Change this to an IS */
        if (cellNumbering) {
          ierr = PetscSectionGetDof(cellNumbering, c, &cell);CHKERRQ(ierr);
          if (cell <= 0) SETERRQ1(PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_OUTOFRANGE, "Cell %D doesn't appear in cell numbering map", c);
          ierr = PetscSectionGetOffset(cellNumbering, c, &cell);CHKERRQ(ierr);
        }
        newCellsArray[i] = cell;
        for (j = 0; j < nodesPerCell; ++j) {
          /* For each global dof, map it into contiguous local storage. */
          const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + subspaceOffset;
          /* finally, loop over block size */
          for (l = 0; l < bs; ++l) {
            PetscInt  localDof;
            PetscBool isGlobalBcDof, isArtificialBcDof;

            /* first, check if this is either a globally enforced or locally enforced BC dof */
            ierr = PetscHSetIHas(globalBcs, globalDof + l, &isGlobalBcDof);CHKERRQ(ierr);
            ierr = PetscHSetIHas(artificialbcs, globalDof + l, &isArtificialBcDof);CHKERRQ(ierr);

            /* if it's either, don't ever give it a local dof number */
            if (isGlobalBcDof || isArtificialBcDof) {
              dofsArray[globalIndex++] = -1; /* don't use this in assembly in this patch */
            } else {
              ierr = PetscHMapIGet(ht, globalDof + l, &localDof);CHKERRQ(ierr);
              if (localDof == -1) {
                localDof = localIndex++;
                ierr = PetscHMapISet(ht, globalDof + l, localDof);CHKERRQ(ierr);
              }
              if ( globalIndex >= numDofs ) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
              /* And store. */
              dofsArray[globalIndex++] = localDof;
            }
          }
        }
      }
    }
    /* How many local dofs in this patch? */
    ierr = PetscHMapIGetSize(ht, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(gtolCounts, v, dof);CHKERRQ(ierr);
  }
  if (globalIndex != numDofs) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Expected number of dofs (%d) doesn't match found number (%d)", numDofs, globalIndex);
  ierr = PetscSectionSetUp(gtolCounts);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(gtolCounts, &numGlobalDofs);CHKERRQ(ierr);
  ierr = PetscMalloc1(numGlobalDofs, &globalDofsArray);CHKERRQ(ierr);

  /* Now populate the global to local map.  This could be merged into the above loop if we were willing to deal with reallocs. */
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIter hi;
    PetscInt      dof, off, Np, ooff, i, j, k, l;

    ierr = PetscHMapIClear(ht);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(cellCounts, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cellCounts, v, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(pointCounts, v, &Np);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pointCounts, v, &ooff);CHKERRQ(ierr);
    if (dof <= 0) continue;

    for (k = 0; k < patch->nsubspaces; ++k) {
      const PetscInt *cellNodeMap    = patch->cellNodeMap[k];
      PetscInt        nodesPerCell   = patch->nodesPerCell[k];
      PetscInt        subspaceOffset = patch->subspaceOffsets[k];
      PetscInt        bs             = patch->bs[k];
      PetscInt        goff;

      for (i = off; i < off + dof; ++i) {
        /* Reconstruct mapping of global-to-local on this patch. */
        const PetscInt c    = cellsArray[i];
        PetscInt       cell = c;

        if (cellNumbering) {ierr = PetscSectionGetOffset(cellNumbering, c, &cell);CHKERRQ(ierr);}
        for (j = 0; j < nodesPerCell; ++j) {
          for (l = 0; l < bs; ++l) {
            const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + l + subspaceOffset;
            const PetscInt localDof  = dofsArray[key++];

            if (localDof >= 0) {ierr = PetscHMapISet(ht, globalDof, localDof);CHKERRQ(ierr);}
          }
        }
      }

      /* Shove it in the output data structure. */
      ierr = PetscSectionGetOffset(gtolCounts, v, &goff);CHKERRQ(ierr);
      PetscHashIterBegin(ht, hi);
      while (!PetscHashIterAtEnd(ht, hi)) {
        PetscInt globalDof, localDof;

        PetscHashIterGetKey(ht, hi, globalDof);
        PetscHashIterGetVal(ht, hi, localDof);
        if (globalDof >= 0) globalDofsArray[goff + localDof] = globalDof;
        PetscHashIterNext(ht, hi);
      }

      for (p = 0; p < Np; ++p) {
        const PetscInt point = pointsArray[ooff + p];
        PetscInt       globalDof, localDof;

        ierr = PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, point, NULL, &globalDof);CHKERRQ(ierr);
        ierr = PetscHMapIGet(ht, globalDof, &localDof);CHKERRQ(ierr);
        offsArray[(ooff + p)*Nf + k] = localDof;
      }
    }

    ierr = PetscHSetIDestroy(&globalBcs);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&ownedpts);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&seenpts);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&owneddofs);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&seendofs);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&artificialbcs);CHKERRQ(ierr);

      /* At this point, we have a hash table ht built that maps globalDof -> localDof.
     We need to create the dof table laid out cellwise first, then by subspace,
     as the assembler assembles cell-wise and we need to stuff the different
     contributions of the different function spaces to the right places. So we loop
     over cells, then over subspaces. */
    if (patch->nsubspaces > 1) { /* for nsubspaces = 1, data we need is already in dofsArray */
      for (i = off; i < off + dof; ++i) {
        const PetscInt c    = cellsArray[i];
        PetscInt       cell = c;

        if (cellNumbering) {ierr = PetscSectionGetOffset(cellNumbering, c, &cell);CHKERRQ(ierr);}
        for (k = 0; k < patch->nsubspaces; ++k) {
          const PetscInt *cellNodeMap    = patch->cellNodeMap[k];
          PetscInt        nodesPerCell   = patch->nodesPerCell[k];
          PetscInt        subspaceOffset = patch->subspaceOffsets[k];
          PetscInt        bs             = patch->bs[k];

          for (j = 0; j < nodesPerCell; ++j) {
            for (l = 0; l < bs; ++l) {
              const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + l + subspaceOffset;
              PetscInt       localDof;

              ierr = PetscHMapIGet(ht, globalDof, &localDof);CHKERRQ(ierr);
              /* If it's not in the hash table, i.e. is a BC dof,
               then the PetscHSetIMap above gives -1, which matches
               exactly the convention for PETSc's matrix assembly to
               ignore the dof. So we don't need to do anything here */
              asmArray[asmKey++] = localDof;
            }
          }
        }
      }
    }
  }
  if (1 == patch->nsubspaces) {ierr = PetscMemcpy(asmArray, dofsArray, numDofs * sizeof(PetscInt));CHKERRQ(ierr);}

  ierr = PetscHMapIDestroy(&ht);CHKERRQ(ierr);
  ierr = ISRestoreIndices(cells, &cellsArray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(points, &pointsArray);CHKERRQ(ierr);
  ierr = PetscFree(dofsArray);CHKERRQ(ierr);
  /* Create placeholder section for map from points to patch dofs */
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->patchSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(patch->patchSection, patch->nsubspaces);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(patch->dofSection[0], &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(patch->patchSection, pStart, pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, fdof, f;

    ierr = PetscSectionGetDof(patch->dofSection[0], p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(patch->patchSection, p, dof);CHKERRQ(ierr);
    for (f = 0; f < patch->nsubspaces; ++f) {
      ierr = PetscSectionGetFieldDof(patch->dofSection[0], p, f, &fdof);
      if (ierr == 0) {
        ierr = PetscSectionSetFieldDof(patch->patchSection, p, f, fdof);CHKERRQ(ierr);
      }
      else {
        /* assume only one field */
        ierr = PetscSectionSetFieldDof(patch->patchSection, p, f, dof);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(patch->patchSection);CHKERRQ(ierr);
  ierr = PetscSectionSetUseFieldOffsets(patch->patchSection, PETSC_TRUE);CHKERRQ(ierr);
  /* Replace cell indices with firedrake-numbered ones. */
  ierr = ISGeneralSetIndices(cells, numCells, (const PetscInt *) newCellsArray, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofs, globalDofsArray, PETSC_OWN_POINTER, &patch->gtol);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->gtol, "Global Indices");CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(patch->gtolCounts, (PetscObject) pc, "-pc_patch_g2l_view");CHKERRQ(ierr);
  ierr = ISViewFromOptions(patch->gtol, (PetscObject) pc, "-pc_patch_g2l_view");CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArray, PETSC_OWN_POINTER, &patch->dofs);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArray, PETSC_OWN_POINTER, &patch->offs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchZeroFillMatrix_Private(Mat mat, const PetscInt ncell, const PetscInt ndof, const PetscInt *dof)
{
  PetscScalar    *values = NULL;
  PetscInt        rows, c, i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc1(ndof*ndof, &values);CHKERRQ(ierr);
  for (c = 0; c < ncell; ++c) {
    const PetscInt *idx = &dof[ndof*c];
    ierr = MatSetValues(mat, ndof, idx, ndof, idx, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatGetLocalSize(mat, &rows, NULL);CHKERRQ(ierr);
  for (i = 0; i < rows; ++i) {
    ierr = MatSetValues(mat, 1, &i, 1, &i, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchCreateMatrix_Private(PC pc, PetscInt point, Mat *mat)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  Vec            x, y;
  PetscBool      flg;
  PetscInt       csize, rsize;
  const char    *prefix = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  x = patch->patchX[point];
  y = patch->patchY[point];
  ierr = VecGetSize(x, &csize);CHKERRQ(ierr);
  ierr = VecGetSize(y, &rsize);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, mat);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc, &prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(*mat, prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(*mat, "pc_patch_sub_");CHKERRQ(ierr);
  if (patch->sub_mat_type)       {ierr = MatSetType(*mat, patch->sub_mat_type);CHKERRQ(ierr);}
  else if (!patch->sub_mat_type) {ierr = MatSetType(*mat, MATDENSE);CHKERRQ(ierr);}
  ierr = MatSetSizes(*mat, rsize, csize, rsize, csize);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) *mat, MATDENSE, &flg);CHKERRQ(ierr);
  if (!flg) {ierr = PetscObjectTypeCompare((PetscObject)*mat, MATSEQDENSE, &flg);CHKERRQ(ierr);}
  /* Sparse patch matrices */
  if (!flg) {
    PetscBT         bt;
    PetscInt       *dnnz      = NULL;
    const PetscInt *dofsArray = NULL;
    PetscInt        pStart, pEnd, ncell, offset, c, i, j;

    ierr = ISGetIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd);CHKERRQ(ierr);
    point += pStart;
    if (point >= pEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Operator point %D not in [%D, %D)\n", point, pStart, pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(patch->cellCounts, point, &ncell);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(patch->cellCounts, point, &offset);CHKERRQ(ierr);
    ierr = PetscCalloc1(rsize, &dnnz);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(PC_Patch_Prealloc, pc, 0, 0, 0);CHKERRQ(ierr);
    /* XXX: This uses N^2 bits to store the sparsity pattern on a
     * patch.  This is probably OK if the patches are not too big,
     * but could use quite a bit of memory for planes in 3D.
     * Should we switch based on the value of rsize to a
     * hash-table (slower, but more memory efficient) approach? */
    ierr = PetscBTCreate(rsize*rsize, &bt);CHKERRQ(ierr);
    for (c = 0; c < ncell; ++c) {
      const PetscInt *idx = dofsArray + (offset + c)*patch->totalDofsPerCell;
      for (i = 0; i < patch->totalDofsPerCell; ++i) {
        const PetscInt row = idx[i];
        if (row < 0) continue;
        for (j = 0; j < patch->totalDofsPerCell; ++j) {
          const PetscInt col = idx[j];
          const PetscInt key = row*rsize + col;
          if (col < 0) continue;
          if (!PetscBTLookupSet(bt, key)) ++dnnz[row];
        }
      }
    }
    ierr = PetscBTDestroy(&bt);CHKERRQ(ierr);
    ierr = MatXAIJSetPreallocation(*mat, 1, dnnz, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = PetscFree(dnnz);CHKERRQ(ierr);
    ierr = PCPatchZeroFillMatrix_Private(*mat, ncell, patch->totalDofsPerCell, &dofsArray[offset*patch->totalDofsPerCell]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_Patch_Prealloc, pc, 0, 0, 0);CHKERRQ(ierr);
    ierr = ISRestoreIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
  }
  ierr = MatSetUp(*mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchComputeOperator_DMPlex_Private(PC pc, PetscInt patchNum, Mat J, IS cellIS, PetscInt n, const PetscInt *l2p, void *ctx)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  DM              dm;
  PetscSection    s;
  const PetscInt *parray, *oarray;
  PetscInt        Nf = patch->nsubspaces, Np, poff, p, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &s);CHKERRQ(ierr);
  /* Set offset into patch */
  ierr = PetscSectionGetDof(patch->pointCounts, patchNum, &Np);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(patch->pointCounts, patchNum, &poff);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->points, &parray);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->offs,   &oarray);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    for (p = 0; p < Np; ++p) {
      const PetscInt point = parray[poff+p];
      PetscInt       dof;

      ierr = PetscSectionGetFieldDof(patch->patchSection, point, f, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldOffset(patch->patchSection, point, f, oarray[(poff+p)*Nf+f]);CHKERRQ(ierr);
      if (patch->nsubspaces == 1) {ierr = PetscSectionSetOffset(patch->patchSection, point, oarray[(poff+p)*Nf+f]);CHKERRQ(ierr);}
      else                        {ierr = PetscSectionSetOffset(patch->patchSection, point, -1);CHKERRQ(ierr);}
    }
  }
  ierr = ISRestoreIndices(patch->points, &parray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(patch->offs,   &oarray);CHKERRQ(ierr);
  if (patch->viewSection) {ierr = ObjectView((PetscObject) patch->patchSection, patch->viewerSection, patch->formatSection);CHKERRQ(ierr);}
  /* TODO Shut off MatViewFromOptions() in MatAssemblyEnd() here */
  ierr = DMPlexComputeJacobian_Patch_Internal(pc->dm, patch->patchSection, patch->patchSection, cellIS, 0.0, 0.0, NULL, NULL, J, J, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchComputeOperator_Private(PC pc, Mat mat, PetscInt point)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  const PetscInt *dofsArray;
  const PetscInt *cellsArray;
  PetscInt        ncell, offset, pStart, pEnd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0);CHKERRQ(ierr);
  if (!patch->usercomputeop) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PCPatchSetComputeOperator() to set user callback\n");
  ierr = ISGetIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->cells, &cellsArray);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd);CHKERRQ(ierr);

  point += pStart;
  if (point >= pEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Operator point %D not in [%D, %D)\n", point, pStart, pEnd);CHKERRQ(ierr);

  ierr = PetscSectionGetDof(patch->cellCounts, point, &ncell);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(patch->cellCounts, point, &offset);CHKERRQ(ierr);
  if (ncell <= 0) {
    ierr = PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  PetscStackPush("PCPatch user callback");
  /* Cannot reuse the same IS because the geometry info is being cached in it */
  ierr = ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray + offset, PETSC_USE_POINTER, &patch->cellIS);CHKERRQ(ierr);
  ierr = patch->usercomputeop(pc, point, mat, patch->cellIS, ncell*patch->totalDofsPerCell, dofsArray + offset*patch->totalDofsPerCell, patch->usercomputectx);CHKERRQ(ierr);
  PetscStackPop;
  ierr = ISDestroy(&patch->cellIS);CHKERRQ(ierr);
  ierr = ISRestoreIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(patch->cells, &cellsArray);CHKERRQ(ierr);
  if (patch->viewMatrix) {
    char name[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "Patch matrix for Point %D", point);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) mat, name);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) mat, patch->viewerMatrix, patch->formatMatrix);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatch_ScatterLocal_Private(PC pc, PetscInt p, Vec x, Vec y, InsertMode mode, ScatterMode scat)
{
  PC_PATCH          *patch     = (PC_PATCH *) pc->data;
  const PetscScalar *xArray    = NULL;
  PetscScalar       *yArray    = NULL;
  const PetscInt    *gtolArray = NULL;
  PetscInt           dof, offset, lidx;
  PetscErrorCode     ierr;

  PetscFunctionBeginHot;
  ierr = PetscLogEventBegin(PC_Patch_Scatter, pc, 0, 0, 0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &xArray);CHKERRQ(ierr);
  ierr = VecGetArray(y, &yArray);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(patch->gtolCounts, p, &dof);CHKERRQ(ierr);
  ierr = PetscSectionGetOffset(patch->gtolCounts, p, &offset);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->gtol, &gtolArray);CHKERRQ(ierr);
  if (mode == INSERT_VALUES && scat != SCATTER_FORWARD) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't insert if not scattering forward\n");
  if (mode == ADD_VALUES    && scat != SCATTER_REVERSE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't add if not scattering reverse\n");
  for (lidx = 0; lidx < dof; ++lidx) {
    const PetscInt gidx = gtolArray[offset+lidx];

    if (mode == INSERT_VALUES) yArray[lidx]  = xArray[gidx]; /* Forward */
    else                       yArray[gidx] += xArray[lidx]; /* Reverse */
  }
  ierr = ISRestoreIndices(patch->gtol, &gtolArray);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xArray);CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &yArray);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_Patch_Scatter, pc, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_PATCH(PC pc)
{
  PC_PATCH       *patch   = (PC_PATCH *) pc->data;
  PetscInt        i;
  const char     *prefix;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscInt pStart, pEnd, p;
    PetscInt localSize;

    ierr = PetscLogEventBegin(PC_Patch_CreatePatches, pc, 0, 0, 0);CHKERRQ(ierr);

    if (!patch->nsubspaces) {
      DM           dm;
      PetscDS      prob;
      PetscSection s;
      PetscInt     cStart, cEnd, c, Nf, f, numGlobalBcs = 0, *globalBcs, *Nb, totNb = 0, **cellDofs;

      ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
      if (!dm) SETERRQ(PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONG, "Must set DM for PCPATCH or call PCPatchSetDiscretisationInfo()");
      ierr = DMGetDefaultSection(dm, &s);CHKERRQ(ierr);
      ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
      ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        PetscInt cdof;
        ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
        numGlobalBcs += cdof;
      }
      ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
      ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
      ierr = PetscMalloc3(Nf, &Nb, Nf, &cellDofs, numGlobalBcs, &globalBcs);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscFE        fe;
        PetscDualSpace sp;
        PetscInt       cdoff = 0;

        ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
        /* ierr = PetscFEGetNumComponents(fe, &Nc[f]);CHKERRQ(ierr); */
        ierr = PetscFEGetDualSpace(fe, &sp);CHKERRQ(ierr);
        ierr = PetscDualSpaceGetDimension(sp, &Nb[f]);CHKERRQ(ierr);
        totNb += Nb[f];

        ierr = PetscMalloc1((cEnd-cStart)*Nb[f], &cellDofs[f]);CHKERRQ(ierr);
        for (c = cStart; c < cEnd; ++c) {
          PetscInt *closure = NULL;
          PetscInt  clSize  = 0, cl;

          ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
          for (cl = 0; cl < clSize*2; cl += 2) {
            const PetscInt p = closure[cl];
            PetscInt       fdof, d, foff;

            ierr = PetscSectionGetFieldDof(s, p, f, &fdof);CHKERRQ(ierr);
            ierr = PetscSectionGetFieldOffset(s, p, f, &foff);CHKERRQ(ierr);
            for (d = 0; d < fdof; ++d, ++cdoff) cellDofs[f][cdoff] = foff + d;
          }
          ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure);CHKERRQ(ierr);
        }
        if (cdoff != (cEnd-cStart)*Nb[f]) SETERRQ4(PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_SIZ, "Total number of cellDofs %D for field %D should be Nc (%D) * cellDof (%D)", cdoff, f, cEnd-cStart, Nb[f]);
      }
      numGlobalBcs = 0;
      for (p = pStart; p < pEnd; ++p) {
        const PetscInt *ind;
        PetscInt        off, cdof, d;

        ierr = PetscSectionGetOffset(s, p, &off);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintIndices(s, p, &ind);CHKERRQ(ierr);
        for (d = 0; d < cdof; ++d) globalBcs[numGlobalBcs++] = off + ind[d];
      }

      ierr = PCPatchSetDiscretisationInfoCombined(pc, dm, Nb, (const PetscInt **) cellDofs, numGlobalBcs, globalBcs, numGlobalBcs, globalBcs);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        ierr = PetscFree(cellDofs[f]);CHKERRQ(ierr);
      }
      ierr = PetscFree3(Nb, cellDofs, globalBcs);CHKERRQ(ierr);
      ierr = PCPatchSetComputeOperator(pc, PCPatchComputeOperator_DMPlex_Private, NULL);CHKERRQ(ierr);
    }

    localSize = patch->subspaceOffsets[patch->nsubspaces];
    ierr = VecCreateSeq(PETSC_COMM_SELF, localSize, &patch->localX);CHKERRQ(ierr);
    ierr = VecSetUp(patch->localX);CHKERRQ(ierr);
    ierr = VecDuplicate(patch->localX, &patch->localY);CHKERRQ(ierr);
    ierr = PCPatchCreateCellPatches(pc);CHKERRQ(ierr);
    ierr = PCPatchCreateCellPatchDiscretisationInfo(pc);CHKERRQ(ierr);

    /* OK, now build the work vectors */
    ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(patch->npatch, &patch->patchX);CHKERRQ(ierr);
    ierr = PetscMalloc1(patch->npatch, &patch->patchY);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof;

      ierr = PetscSectionGetDof(patch->gtolCounts, p, &dof);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF, dof, &patch->patchX[p-pStart]);CHKERRQ(ierr);
      ierr = VecSetUp(patch->patchX[p-pStart]);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF, dof, &patch->patchY[p-pStart]);CHKERRQ(ierr);
      ierr = VecSetUp(patch->patchY[p-pStart]);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(patch->npatch, &patch->ksp);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc, &prefix);CHKERRQ(ierr);
    for (i = 0; i < patch->npatch; ++i) {
      PC subpc;

      ierr = KSPCreate(PETSC_COMM_SELF, &patch->ksp[i]);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(patch->ksp[i], prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(patch->ksp[i], "sub_");CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject) patch->ksp[i], (PetscObject) pc, 1);CHKERRQ(ierr);
      ierr = KSPGetPC(patch->ksp[i], &subpc);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject) subpc, (PetscObject) pc, 1);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject) pc, (PetscObject) patch->ksp[i]);CHKERRQ(ierr);
    }
    if (patch->save_operators) {
      ierr = PetscMalloc1(patch->npatch, &patch->mat);CHKERRQ(ierr);
      for (i = 0; i < patch->npatch; ++i) {
        ierr = PCPatchCreateMatrix_Private(pc, i, &patch->mat[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscLogEventEnd(PC_Patch_CreatePatches, pc, 0, 0, 0);CHKERRQ(ierr);

    /* If desired, calculate weights for dof multiplicity */
    if (patch->partition_of_unity) {
      ierr = VecDuplicate(patch->localX, &patch->dof_weights);CHKERRQ(ierr);
      for (i = 0; i < patch->npatch; ++i) {
        PetscInt dof;

        ierr = PetscSectionGetDof(patch->gtolCounts, i+pStart, &dof);CHKERRQ(ierr);
        if (dof <= 0) continue;
        ierr = VecSet(patch->patchX[i], 1.0);CHKERRQ(ierr);
        ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchX[i], patch->dof_weights, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
      }
      ierr = VecReciprocal(patch->dof_weights);CHKERRQ(ierr);
    }
  }
  if (patch->save_operators) {
    for (i = 0; i < patch->npatch; ++i) {
      ierr = MatZeroEntries(patch->mat[i]);CHKERRQ(ierr);
      ierr = PCPatchComputeOperator_Private(pc, patch->mat[i], i);CHKERRQ(ierr);
      ierr = KSPSetOperators(patch->ksp[i], patch->mat[i], patch->mat[i]);CHKERRQ(ierr);
    }
  }
  if (!pc->setupcalled && patch->optionsSet) for (i = 0; i < patch->npatch; ++i) {ierr = KSPSetFromOptions(patch->ksp[i]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PATCH(PC pc, Vec x, Vec y)
{
  PC_PATCH          *patch    = (PC_PATCH *) pc->data;
  const PetscScalar *globalX  = NULL;
  PetscScalar       *localX   = NULL;
  PetscScalar       *globalY  = NULL;
  const PetscInt    *bcNodes  = NULL;
  PetscInt           nsweep   = patch->symmetrise_sweep ? 2 : 1;
  PetscInt           start[2] = {0, 0};
  PetscInt           end[2]   = {-1, -1};
  const PetscInt     inc[2]   = {1, -1};
  const PetscScalar *localY;
  const PetscInt    *iterationSet;
  PetscInt           pStart, numBcs, n, sweep, bc, j;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_Patch_Apply, pc, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscOptionsPushGetViewerOff(PETSC_TRUE);CHKERRQ(ierr);
  end[0]   = patch->npatch;
  start[1] = patch->npatch-1;
  if (patch->user_patches) {
    ierr = ISGetLocalSize(patch->iterationSet, &end[0]);CHKERRQ(ierr);
    start[1] = end[0] - 1;
    ierr = ISGetIndices(patch->iterationSet, &iterationSet);CHKERRQ(ierr);
  }
  /* Scatter from global space into overlapped local spaces */
  ierr = VecGetArrayRead(x, &globalX);CHKERRQ(ierr);
  ierr = VecGetArray(patch->localX, &localX);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(patch->defaultSF, MPIU_SCALAR, globalX, localX);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(patch->defaultSF, MPIU_SCALAR, globalX, localX);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &globalX);CHKERRQ(ierr);
  ierr = VecRestoreArray(patch->localX, &localX);CHKERRQ(ierr);

  ierr = VecSet(patch->localY, 0.0);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, NULL);CHKERRQ(ierr);
  for (sweep = 0; sweep < nsweep; sweep++) {
    for (j = start[sweep]; j*inc[sweep] < end[sweep]*inc[sweep]; j += inc[sweep]) {
      PetscInt i       = patch->user_patches ? iterationSet[j] : j;
      PetscInt start, len;

      ierr = PetscSectionGetDof(patch->gtolCounts, i+pStart, &len);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(patch->gtolCounts, i+pStart, &start);CHKERRQ(ierr);
      /* TODO: Squash out these guys in the setup as well. */
      if (len <= 0) continue;
      /* TODO: Do we need different scatters for X and Y? */
      ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localX, patch->patchX[i], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
      if (!patch->save_operators) {
        Mat mat;

        ierr = PCPatchCreateMatrix_Private(pc, i, &mat);CHKERRQ(ierr);
        /* Populate operator here. */
        ierr = PCPatchComputeOperator_Private(pc, mat, i);CHKERRQ(ierr);
        ierr = KSPSetOperators(patch->ksp[i], mat, mat);
        /* Drop reference so the KSPSetOperators below will blow it away. */
        ierr = MatDestroy(&mat);CHKERRQ(ierr);
      }
      ierr = PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0);CHKERRQ(ierr);
      ierr = KSPSolve(patch->ksp[i], patch->patchX[i], patch->patchY[i]);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0);CHKERRQ(ierr);

      if (!patch->save_operators) {
        PC pc;
        ierr = KSPSetOperators(patch->ksp[i], NULL, NULL);CHKERRQ(ierr);
        ierr = KSPGetPC(patch->ksp[i], &pc);CHKERRQ(ierr);
        /* Destroy PC context too, otherwise the factored matrix hangs around. */
        ierr = PCReset(pc);CHKERRQ(ierr);
      }

      ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchY[i], patch->localY, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
    }
  }
  if (patch->user_patches) {ierr = ISRestoreIndices(patch->iterationSet, &iterationSet);CHKERRQ(ierr);}
  /* XXX: should we do this on the global vector? */
  if (patch->partition_of_unity) {
    ierr = VecPointwiseMult(patch->localY, patch->localY, patch->dof_weights);CHKERRQ(ierr);
  }
  /* Now patch->localY contains the solution of the patch solves, so we need to combine them all. */
  ierr = VecSet(y, 0.0);CHKERRQ(ierr);
  ierr = VecGetArray(y, &globalY);CHKERRQ(ierr);
  ierr = VecGetArrayRead(patch->localY, &localY);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(patch->defaultSF, MPIU_SCALAR, localY, globalY, MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(patch->defaultSF, MPIU_SCALAR, localY, globalY, MPI_SUM);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(patch->localY, &localY);CHKERRQ(ierr);

  /* Now we need to send the global BC values through */
  ierr = VecGetArrayRead(x, &globalX);CHKERRQ(ierr);
  ierr = ISGetSize(patch->globalBcNodes, &numBcs);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->globalBcNodes, &bcNodes);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x, &n);CHKERRQ(ierr);
  for (bc = 0; bc < numBcs; ++bc) {
    const PetscInt idx = bcNodes[bc];
    if (idx < n) globalY[idx] = globalX[idx];
  }

  ierr = ISRestoreIndices(patch->globalBcNodes, &bcNodes);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &globalX);CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &globalY);CHKERRQ(ierr);

  ierr = PetscOptionsPopGetViewerOff();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_Patch_Apply, pc, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* TODO: Get rid of all these ifs */
  ierr = PetscSFDestroy(&patch->defaultSF);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->cellCounts);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->pointCounts);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->cellNumbering);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->gtolCounts);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->gtol);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->cells);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->points);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->dofs);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->offs);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->patchSection);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->ghostBcNodes);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->globalBcNodes);CHKERRQ(ierr);

  if (patch->dofSection) for (i = 0; i < patch->nsubspaces; i++) {ierr = PetscSectionDestroy(&patch->dofSection[i]);CHKERRQ(ierr);}
  ierr = PetscFree(patch->dofSection);CHKERRQ(ierr);
  ierr = PetscFree(patch->bs);CHKERRQ(ierr);
  ierr = PetscFree(patch->nodesPerCell);CHKERRQ(ierr);
  if (patch->cellNodeMap) for (i = 0; i < patch->nsubspaces; i++) {ierr = PetscFree(patch->cellNodeMap[i]);CHKERRQ(ierr);}
  ierr = PetscFree(patch->cellNodeMap);CHKERRQ(ierr);
  ierr = PetscFree(patch->subspaceOffsets);CHKERRQ(ierr);

  if (patch->ksp) {
    for (i = 0; i < patch->npatch; ++i) {ierr = KSPReset(patch->ksp[i]);CHKERRQ(ierr);}
  }

  ierr = VecDestroy(&patch->localX);CHKERRQ(ierr);
  ierr = VecDestroy(&patch->localY);CHKERRQ(ierr);
  if (patch->patchX) {
    for (i = 0; i < patch->npatch; ++i) {ierr = VecDestroy(&patch->patchX[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->patchX);CHKERRQ(ierr);
  }
  if (patch->patchY) {
    for (i = 0; i < patch->npatch; ++i) {ierr = VecDestroy(&patch->patchY[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->patchY);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&patch->dof_weights);CHKERRQ(ierr);
  if (patch->patch_dof_weights) {
    for (i = 0; i < patch->npatch; ++i) {ierr = VecDestroy(&patch->patch_dof_weights[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->patch_dof_weights);CHKERRQ(ierr);
  }
  if (patch->mat) {
    for (i = 0; i < patch->npatch; ++i) {ierr = MatDestroy(&patch->mat[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->mat);CHKERRQ(ierr);
  }
  ierr = PetscFree(patch->sub_mat_type);CHKERRQ(ierr);
  if (patch->userIS) {
    for (i = 0; i < patch->npatch; ++i) {ierr = ISDestroy(&patch->userIS[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->userIS);CHKERRQ(ierr);
  }
  patch->bs          = 0;
  patch->cellNodeMap = NULL;
  patch->nsubspaces  = 0;
  ierr = ISDestroy(&patch->iterationSet);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&patch->viewerSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_PATCH(pc);CHKERRQ(ierr);
  if (patch->ksp) {
    for (i = 0; i < patch->npatch; ++i) {ierr = KSPDestroy(&patch->ksp[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->ksp);CHKERRQ(ierr);
  }
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_PATCH(PetscOptionItems *PetscOptionsObject, PC pc)
{
  PC_PATCH            *patch = (PC_PATCH *) pc->data;
  PCPatchConstructType patchConstructionType = PC_PATCH_STAR;
  char                 sub_mat_type[PETSC_MAX_PATH_LEN];
  const char          *prefix;
  PetscBool            flg, dimflg, codimflg;
  MPI_Comm             comm;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) pc, &comm);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) pc, &prefix);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject, "Vertex-patch Additive Schwarz options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_patch_save_operators",  "Store all patch operators for lifetime of PC?", "PCPatchSetSaveOperators", patch->save_operators, &patch->save_operators, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_patch_partition_of_unity", "Weight contributions by dof multiplicity?", "PCPatchSetPartitionOfUnity", patch->partition_of_unity, &patch->partition_of_unity, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_patch_construct_dim", "What dimension of mesh point to construct patches by? (0 = vertices)", "PCPATCH", patch->dim, &patch->dim, &dimflg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_patch_construct_codim", "What co-dimension of mesh point to construct patches by? (0 = cells)", "PCPATCH", patch->codim, &patch->codim, &codimflg);CHKERRQ(ierr);
  if (dimflg && codimflg) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Can only set one of dimension or co-dimension");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_patch_construct_type", "How should the patches be constructed?", "PCPatchSetConstructType", PCPatchConstructTypes, (PetscEnum) patchConstructionType, (PetscEnum *) &patchConstructionType, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PCPatchSetConstructType(pc, patchConstructionType, NULL, NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsInt("-pc_patch_vanka_dim", "Topological dimension of entities for Vanka to ignore", "PCPATCH", patch->vankadim, &patch->vankadim, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_patch_ignore_dim", "Topological dimension of entities for completion to ignore", "PCPATCH", patch->ignoredim, &patch->ignoredim, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-pc_patch_sub_mat_type", "Matrix type for patch solves", "PCPatchSetSubMatType", MatList, NULL, sub_mat_type, PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PCPatchSetSubMatType(pc, sub_mat_type);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-pc_patch_symmetrise_sweep", "Go start->end, end->start?", "PCPATCH", patch->symmetrise_sweep, &patch->symmetrise_sweep, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_patch_exclude_subspace", "What subspace (if any) to exclude in construction?", "PCPATCH", patch->exclude_subspace, &patch->exclude_subspace, &flg);CHKERRQ(ierr);
  if (patch->exclude_subspace && (patchConstructionType == PC_PATCH_USER)) SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "We cannot support excluding a subspace with user patches because we do not index patches with a mesh point");

  ierr = PetscOptionsBool("-pc_patch_patches_view", "Print out information during patch construction", "PCPATCH", patch->viewPatches, &patch->viewPatches, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm, prefix, "-pc_patch_cells_view",   &patch->viewerCells,   &patch->formatCells,   &patch->viewCells);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm, prefix, "-pc_patch_points_view",  &patch->viewerPoints,  &patch->formatPoints,  &patch->viewPoints);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm, prefix, "-pc_patch_section_view", &patch->viewerSection, &patch->formatSection, &patch->viewSection);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm, prefix, "-pc_patch_mat_view",     &patch->viewerMatrix,  &patch->formatMatrix,  &patch->viewMatrix);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  patch->optionsSet = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_PATCH(PC pc)
{
  PC_PATCH          *patch = (PC_PATCH*) pc->data;
  KSPConvergedReason reason;
  PetscInt           i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  for (i = 0; i < patch->npatch; ++i) {
    ierr = KSPSetUp(patch->ksp[i]);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(patch->ksp[i], &reason);CHKERRQ(ierr);
    if (reason == KSP_DIVERGED_PCSETUP_FAILED) pc->failedreason = PC_SUBPC_ERROR;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_PATCH(PC pc, PetscViewer viewer)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscViewer    sviewer;
  PetscBool      isascii;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* TODO Redo tabbing with set tbas in new style */
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) pc), &rank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Subspace Correction preconditioner with %d patches\n", patch->npatch);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Schwarz type: additive\n");CHKERRQ(ierr);
  if (patch->partition_of_unity) {ierr = PetscViewerASCIIPrintf(viewer, "Weighting by partition of unity\n");CHKERRQ(ierr);}
  else                           {ierr = PetscViewerASCIIPrintf(viewer, "Not weighting by partition of unity\n");CHKERRQ(ierr);}
  if (patch->symmetrise_sweep) {ierr = PetscViewerASCIIPrintf(viewer, "Symmetrising sweep (start->end, then end->start)\n");CHKERRQ(ierr);}
  else                         {ierr = PetscViewerASCIIPrintf(viewer, "Not symmetrising sweep\n");CHKERRQ(ierr);}
  if (!patch->save_operators) {ierr = PetscViewerASCIIPrintf(viewer, "Not saving patch operators (rebuilt every PCApply)\n");CHKERRQ(ierr);}
  else                        {ierr = PetscViewerASCIIPrintf(viewer, "Saving patch operators (rebuilt every PCSetUp)\n");CHKERRQ(ierr);}
  if (patch->patchconstructop == PCPatchConstruct_Star)       {ierr = PetscViewerASCIIPrintf(viewer, "Patch construction operator: star\n");CHKERRQ(ierr);}
  else if (patch->patchconstructop == PCPatchConstruct_Vanka) {ierr = PetscViewerASCIIPrintf(viewer, "Patch construction operator: Vanka\n");CHKERRQ(ierr);}
  else if (patch->patchconstructop == PCPatchConstruct_User)  {ierr = PetscViewerASCIIPrintf(viewer, "Patch construction operator: user-specified\n");CHKERRQ(ierr);}
  else                                                        {ierr = PetscViewerASCIIPrintf(viewer, "Patch construction operator: unknown\n");CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPrintf(viewer, "KSP on patches (all same):\n");CHKERRQ(ierr);
  if (patch->ksp) {
    ierr = PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sviewer);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscViewerASCIIPushTab(sviewer);CHKERRQ(ierr);
      ierr = KSPView(patch->ksp[0], sviewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(sviewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sviewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "KSP not yet set.\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  PCPATCH - A PC object that encapsulates flexible definition of blocks for overlapping and non-overlapping
            small block additive preconditioners. Block definition is based on topology from
            a DM and equation numbering from a PetscSection.

  Options Database Keys:
+ -pc_patch_cells_view   - Views the process local cell numbers for each patch
. -pc_patch_points_view  - Views the process local mesh point numbers for each patch
. -pc_patch_g2l_view     - Views the map between global dofs and patch local dofs for each patch
. -pc_patch_patches_view - Views the global dofs associated with each patch and its boundary
- -pc_patch_sub_mat_view - Views the matrix associated with each patch

  Level: intermediate

.seealso: PCType, PCCreate(), PCSetType()
M*/
PETSC_EXTERN PetscErrorCode PCCreate_Patch(PC pc)
{
  PC_PATCH      *patch;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc, &patch);CHKERRQ(ierr);

  /* Set some defaults */
  patch->combined           = PETSC_FALSE;
  patch->save_operators     = PETSC_TRUE;
  patch->partition_of_unity = PETSC_FALSE;
  patch->codim              = -1;
  patch->dim                = -1;
  patch->exclude_subspace   = -1;
  patch->vankadim           = -1;
  patch->ignoredim          = -1;
  patch->patchconstructop   = PCPatchConstruct_Star;
  patch->symmetrise_sweep   = PETSC_FALSE;
  patch->npatch             = 0;
  patch->userIS             = NULL;
  patch->optionsSet         = PETSC_FALSE;
  patch->iterationSet       = NULL;
  patch->user_patches       = PETSC_FALSE;
  ierr = PetscStrallocpy(MATDENSE, (char **) &patch->sub_mat_type);CHKERRQ(ierr);
  patch->viewPatches        = PETSC_FALSE;
  patch->viewCells          = PETSC_FALSE;
  patch->viewPoints         = PETSC_FALSE;
  patch->viewSection        = PETSC_FALSE;
  patch->viewMatrix         = PETSC_FALSE;

  pc->data                 = (void *) patch;
  pc->ops->apply           = PCApply_PATCH;
  pc->ops->applytranspose  = 0; /* PCApplyTranspose_PATCH; */
  pc->ops->setup           = PCSetUp_PATCH;
  pc->ops->reset           = PCReset_PATCH;
  pc->ops->destroy         = PCDestroy_PATCH;
  pc->ops->setfromoptions  = PCSetFromOptions_PATCH;
  pc->ops->setuponblocks   = PCSetUpOnBlocks_PATCH;
  pc->ops->view            = PCView_PATCH;
  pc->ops->applyrichardson = 0;

  PetscFunctionReturn(0);
}
