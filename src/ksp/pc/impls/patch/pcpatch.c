#include <petsc/private/pcpatchimpl.h>     /*I "petscpc.h" I*/
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscbt.h>

PetscLogEvent PC_Patch_CreatePatches, PC_Patch_ComputeOp, PC_Patch_Solve, PC_Patch_Scatter, PC_Patch_Apply, PC_Patch_Prealloc;

static PetscErrorCode PCPatchConstruct_Star(void *vpatch, DM dm, PetscInt point, PetscHashI ht)
{
  PetscInt       starSize;
  PetscInt      *star = NULL, si;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscHashIClear(ht);
  /* To start with, add the point we care about */
  PetscHashIAdd(ht, point, 0);
  /* Loop over all the points that this point connects to */
  ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  for (si = 0; si < starSize; si += 2) {PetscHashIAdd(ht, star[si], 0);}
  ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchConstruct_Vanka(void *vpatch, DM dm, PetscInt point, PetscHashI ht)
{
  PC_PATCH      *patch = (PC_PATCH *) vpatch;
  PetscInt       starSize;
  PetscInt      *star = NULL;
  PetscBool      shouldIgnore = PETSC_FALSE;
  PetscInt       cStart, cEnd, iStart, iEnd, si;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscHashIClear(ht);
  /* To start with, add the point we care about */
  PetscHashIAdd(ht, point, 0);
  /* Should we ignore any points of a certain dimension? */
  if (patch->vankadim >= 0) {
    shouldIgnore = PETSC_TRUE;
    ierr = DMPlexGetDepthStratum(dm, patch->vankadim, &iStart, &iEnd);CHKERRQ(ierr);
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  /* Loop over all the cells that this point connects to */
  ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  for (si = 0; si < starSize; si += 2) {
    const PetscInt cell = star[si];
    PetscInt       closureSize;
    PetscInt      *closure = NULL, ci;

    if (cell < cStart || cell >= cEnd) continue;
    /* now loop over all entities in the closure of that cell */
    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (ci = 0; ci < closureSize; ci += 2) {
      const PetscInt newpoint = closure[ci];

      /* We've been told to ignore entities of this type.*/
      if (shouldIgnore && newpoint >= iStart && newpoint < iEnd) continue;
      PetscHashIAdd(ht, newpoint, 0);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The user's already set the patches in patch->userIS. Build the hash tables */
static PetscErrorCode PCPatchConstruct_User(void *vpatch, DM dm, PetscInt point, PetscHashI ht)
{
  PC_PATCH       *patch   = (PC_PATCH *) vpatch;
  IS              patchis = patch->userIS[point];
  PetscInt        n;
  const PetscInt *patchdata;
  PetscInt        pStart, pEnd, i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscHashIClear(ht);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);
  ierr = ISGetLocalSize(patchis, &n);CHKERRQ(ierr);
  ierr = ISGetIndices(patchis, &patchdata);CHKERRQ(ierr);
  for (i = 0; i < n; ++i) {
    const PetscInt ownedpoint = patchdata[i];

    if (ownedpoint < pStart || ownedpoint >= pEnd) {
      SETERRQ3(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D was not in [%D, %D)", ownedpoint, pStart, pEnd);
    }
    PetscHashIAdd(ht, ownedpoint, 0);
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
    PetscHashI   rankToIndex;
    PetscInt     numRanks = 0;
    PetscSFNode *remote = NULL;
    PetscSF      rankSF;
    PetscInt    *ranks = NULL;
    PetscInt    *offsets = NULL;
    MPI_Datatype contig;
    PetscHashI   ranksUniq;

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
    PetscHashICreate(ranksUniq);
    for (i = 0; i < n; ++i) {
      const PetscMPIInt *ranks = NULL;
      PetscInt           nranks, j;

      ierr = PetscSFSetUp(sf[i]);CHKERRQ(ierr);
      ierr = PetscSFGetRanks(sf[i], &nranks, &ranks, NULL, NULL, NULL);CHKERRQ(ierr);
      /* These are all the ranks who communicate with me. */
      for (j = 0; j < nranks; ++j) {
        PetscHashIAdd(ranksUniq, (PetscInt) ranks[j], 0);
      }
    }
    PetscHashISize(ranksUniq, numRanks);
    ierr = PetscMalloc1(numRanks, &remote);CHKERRQ(ierr);
    ierr = PetscMalloc1(numRanks, &ranks);CHKERRQ(ierr);
    PetscHashIGetKeys(ranksUniq, &index, ranks);

    PetscHashICreate(rankToIndex);
    for (i = 0; i < numRanks; ++i) {
      remote[i].rank  = ranks[i];
      remote[i].index = 0;
      PetscHashIAdd(rankToIndex, ranks[i], i);
    }
    ierr = PetscFree(ranks);CHKERRQ(ierr);
    PetscHashIDestroy(ranksUniq);
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

        PetscHashIMap(rankToIndex, rank, idx);
        if (idx == -1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Didn't find rank, huh?");
        /* Offset on given rank for ith subspace */
        rootOffset = remoteOffsets[n*idx + i];
        for (k = 0; k < bs[i]; ++k) {
          ilocal[index]        = local[j]*bs[i] + k + leafOffset;
          iremote[index].rank  = remote[j].rank;
          iremote[index].index = remote[j].index*bs[i] + k + rootOffset;
          ++index;
        }
      }
      leafOffset += nleaves * bs[i];
    }
    PetscHashIDestroy(rankToIndex);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)pc), &patch->defaultSF);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(patch->defaultSF, allRoots, allLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
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
PetscErrorCode PCPatchSetMultiplicative(PC pc, PetscBool flg)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  patch->multiplicative = flg;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchGetMultiplicative(PC pc, PetscBool *flg)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  *flg = patch->multiplicative;
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
    patch->patchconstructop = PCPatchConstruct_Star;
    break;
  case PC_PATCH_VANKA:
    patch->patchconstructop = PCPatchConstruct_Vanka;
    break;
  case PC_PATCH_USER:
  case PC_PATCH_PYTHON:
    patch->user_patches            = PETSC_TRUE;
    patch->patchconstructop        = PCPatchConstruct_User;
    patch->userpatchconstructionop = func;
    patch->userpatchconstructctx   = ctx;
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
  PetscSF       *sfs;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
    patch->cellNodeMap[i]     = cellNodeMap[i];
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
PetscErrorCode PCPatchSetComputeOperator(PC pc, PetscErrorCode (*func)(PC, Mat, PetscInt, const PetscInt *, PetscInt, const PetscInt *, void *), void *ctx)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  /* User op can assume matrix is zeroed */
  patch->usercomputeop  = func;
  patch->usercomputectx = ctx;
  PetscFunctionReturn(0);
}

/* On entry, ht contains the topological entities whose dofs we are responsible for solving for;
   on exit, cht contains all the topological entities we need to compute their residuals.
   In full generality this should incorporate knowledge of the sparsity pattern of the matrix;
   here we assume a standard FE sparsity pattern.*/
/* TODO: Use DMPlexGetAdjacency() */
/* TODO: Look at temp buffer management for GetClosure() */
static PetscErrorCode PCPatchCompleteCellPatch(DM dm, PetscHashI ht, PetscHashI cht)
{
  PetscHashIIter hi;
  PetscInt       point;
  PetscInt      *star = NULL, *closure = NULL;
  PetscInt       starSize, closureSize, si, ci;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscHashIClear(cht);
  PetscHashIIterBegin(ht, hi);
  while (!PetscHashIIterAtEnd(ht, hi)) {
    PetscHashIIterGetKey(ht, hi, point);
    PetscHashIIterNext(ht, hi);

    /* Loop over all the cells that this point connects to */
    ierr = DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for (si = 0; si < starSize; si += 2) {
      PetscInt ownedpoint = star[si];
      /* now loop over all entities in the closure of that cell */
      ierr = DMPlexGetTransitiveClosure(dm, ownedpoint, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (ci = 0; ci < closureSize; ci += 2) {
        PetscInt seenpoint = closure[ci];
        PetscHashIAdd(cht, seenpoint, 0);
      }
    }
  }
  /* Only restore work arrays at very end. */
  if (closure) {ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, NULL, &closure);CHKERRQ(ierr);}
  if (star)    {ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_FALSE, NULL, &star);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* Given a hash table with a set of topological entities (pts), compute the degrees of
   freedom in global concatenated numbering on those entities.
   For Vanka smoothing, this needs to do something special: ignore dofs of the
   constraint subspace on entities that aren't the base entity we're building the patch
   around. */
static PetscErrorCode PCPatchGetPointDofs(PC_PATCH *patch, PetscHashI pts, PetscHashI dofs, PetscInt base, PetscInt exclude_subspace)
{
  PetscHashIIter hi;
  PetscInt       ldof, loff;
  PetscInt       k, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscHashIClear(dofs);
  for (k = 0; k < patch->nsubspaces; ++k) {
    PetscSection dofSection     = patch->dofSection[k];
    PetscInt     subspaceOffset = patch->subspaceOffsets[k];
    PetscInt     bs             = patch->bs[k];
    PetscInt     j, l;

    if (k == exclude_subspace) {
      /* only get this subspace dofs at the base entity, not any others */
      ierr = PetscSectionGetDof(dofSection, base, &ldof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(dofSection, base, &loff);CHKERRQ(ierr);
      if (0 == ldof) continue;
      for (j = loff; j < ldof + loff; ++j) {
        for (l = 0; l < bs; ++l) {
          PetscInt dof = bs*j + l + subspaceOffset;
          PetscHashIAdd(dofs, dof, 0);
        }
      }
      continue; /* skip the other dofs of this subspace */
    }

    PetscHashIIterBegin(pts, hi);
    while (!PetscHashIIterAtEnd(pts, hi)) {
      PetscHashIIterGetKey(pts, hi, p);
      PetscHashIIterNext(pts, hi);
      ierr = PetscSectionGetDof(dofSection, p, &ldof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(dofSection, p, &loff);CHKERRQ(ierr);
      if (0 == ldof) continue;
      for (j = loff; j < ldof + loff; ++j) {
        for (l = 0; l < bs; ++l) {
          PetscInt dof = bs*j + l + subspaceOffset;
          PetscHashIAdd(dofs, dof, 0);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Given two hash tables A and B, compute the keys in B that are not in A, and put them in C */
static PetscErrorCode PCPatchComputeSetDifference_Private(PetscHashI A, PetscHashI B, PetscHashI C)
{
  PetscHashIIter hi;
  PetscInt       key, val;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscHashIClear(C);
  PetscHashIIterBegin(B, hi);
  while (!PetscHashIIterAtEnd(B, hi)) {
    PetscHashIIterGetKeyVal(B, hi, key, val);
    PetscHashIIterNext(B, hi);
    PetscHashIHasKey(A, key, flg);
    if (!flg) {PetscHashIAdd(C, key, val);}
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
 * + cellCounts - Section with counts of cells around each vertex
 * - cells - IS of the cell point indices of cells in each patch
 */
static PetscErrorCode PCPatchCreateCellPatches(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  DM             dm, plex;
  DMLabel        ghost;
  PetscHashI     ht, cht;
  PetscInt      *cellsArray = NULL;
  PetscInt       numCells;
  PetscSection   cellCounts;
  PetscInt       pStart, pEnd, vStart, vEnd, v, cStart, cEnd;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Used to keep track of the cells in the patch. */
  PetscHashICreate(ht);
  PetscHashICreate(cht);

  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch PC\n");
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexGetChart(plex, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);

  if (patch->user_patches) {
    /* compute patch->nuserIS, patch->userIS here */
    ierr = patch->userpatchconstructionop(pc, &patch->nuserIS, &patch->userIS, &patch->iterationSet, patch->userpatchconstructctx);CHKERRQ(ierr);
    vStart = 0;
    vEnd   = patch->nuserIS;
  } else if (patch->codim < 0) { /* codim unset */
    if (patch->dim < 0) { /* dim unset */
      ierr = DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd);CHKERRQ(ierr);
    } else { /* dim set */
      ierr = DMPlexGetDepthStratum(plex, patch->dim, &vStart, &vEnd);CHKERRQ(ierr);
    }
  } else { /* codim set */
    ierr = DMPlexGetHeightStratum(plex, patch->codim, &vStart, &vEnd);CHKERRQ(ierr);
  }

  /* These labels mark the owned points.  We only create patches around points that this process owns. */
  /* Replace this with a check of the SF */
  ierr = DMGetLabel(dm, "pyop2_ghost", &ghost);CHKERRQ(ierr);
  ierr = DMLabelCreateIndex(ghost, pStart, pEnd);CHKERRQ(ierr);

  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->cellCounts);CHKERRQ(ierr);
  cellCounts = patch->cellCounts;
  ierr = PetscSectionSetChart(cellCounts, vStart, vEnd);CHKERRQ(ierr);
  /* Count cells in the patch surrounding each entity */
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIIter hi;
    PetscInt       chtSize;

    if (!patch->user_patches) {
      ierr = DMLabelHasPoint(ghost, v, &flg);CHKERRQ(ierr);
      /* Not an owned entity, don't make a cell patch. */
      if (flg) continue;
    }

    ierr = patch->patchconstructop((void *) patch, dm, v, ht);CHKERRQ(ierr);
    ierr = PCPatchCompleteCellPatch(dm, ht, cht);CHKERRQ(ierr);
    PetscHashISize(cht, chtSize);
    /* empty patch, continue */
    if (chtSize == 0) continue;

    /* safe because size(cht) > 0 from above */
    PetscHashIIterBegin(cht, hi);
    while (!PetscHashIIterAtEnd(cht, hi)) {
      PetscInt point;

      PetscHashIIterGetKey(cht, hi, point);
      if (point >= cStart && point < cEnd) {
        ierr = PetscSectionAddDof(cellCounts, v, 1);CHKERRQ(ierr);
      }
      PetscHashIIterNext(cht, hi);
    }
  }
  ierr = DMLabelDestroyIndex(ghost);CHKERRQ(ierr);

  ierr = PetscSectionSetUp(cellCounts);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(cellCounts, &numCells);CHKERRQ(ierr);
  ierr = PetscMalloc1(numCells, &cellsArray);CHKERRQ(ierr);

  /* Now that we know how much space we need, run through again and actually remember the cells. */
  for (v = vStart; v < vEnd; v++ ) {
    PetscHashIIter hi;
    PetscInt       ndof, off;

    ierr = PetscSectionGetDof(cellCounts, v, &ndof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cellCounts, v, &off);CHKERRQ(ierr);
    if (ndof <= 0) continue;
    ierr = patch->patchconstructop((void *) patch, dm, v, ht);CHKERRQ(ierr);
    ierr = PCPatchCompleteCellPatch(dm, ht, cht);CHKERRQ(ierr);
    ndof = 0;
    PetscHashIIterBegin(cht, hi);
    while (!PetscHashIIterAtEnd(cht, hi)) {
      PetscInt point;

      PetscHashIIterGetKey(cht, hi, point);
      if (point >= cStart && point < cEnd) {
        cellsArray[ndof + off] = point;
        ++ndof;
      }
      PetscHashIIterNext(cht, hi);
    }
  }

  ierr = ISCreateGeneral(PETSC_COMM_SELF, numCells, cellsArray, PETSC_OWN_POINTER, &patch->cells);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd);CHKERRQ(ierr);
  patch->npatch = pEnd - pStart;
  PetscHashIDestroy(ht);
  PetscHashIDestroy(cht);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
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
 * + dofs - IS of local dof numbers of each cell in the patch
 * . gtolCounts - Section with counts of dofs per cell patch
 * - gtol - IS mapping from global dofs to local dofs for each patch.
 */
static PetscErrorCode PCPatchCreateCellPatchDiscretisationInfo(PC pc)
{
  PC_PATCH       *patch           = (PC_PATCH *) pc->data;
  PetscSection    cellCounts      = patch->cellCounts;
  PetscSection    gtolCounts;
  IS              cells           = patch->cells;
  PetscSection    cellNumbering   = patch->cellNumbering;
  PetscInt        numCells;
  PetscInt        numDofs;
  PetscInt        numGlobalDofs;
  PetscInt        totalDofsPerCell = patch->totalDofsPerCell;
  PetscInt        vStart, vEnd, v;
  const PetscInt *cellsArray;
  PetscInt       *newCellsArray   = NULL;
  PetscInt       *dofsArray       = NULL;
  PetscInt       *asmArray        = NULL;
  PetscInt       *globalDofsArray = NULL;
  PetscInt        globalIndex     = 0;
  PetscInt        key             = 0;
  PetscInt        asmKey          = 0;
  PetscHashI      ht;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* dofcounts section is cellcounts section * dofPerCell */
  ierr = PetscSectionGetStorageSize(cellCounts, &numCells);CHKERRQ(ierr);
  numDofs = numCells * totalDofsPerCell;
  ierr = PetscMalloc1(numDofs, &dofsArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(numDofs, &asmArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(numCells, &newCellsArray);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(cellCounts, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCounts);CHKERRQ(ierr);
  gtolCounts = patch->gtolCounts;
  ierr = PetscSectionSetChart(gtolCounts, vStart, vEnd);CHKERRQ(ierr);

  ierr = ISGetIndices(cells, &cellsArray);CHKERRQ(ierr);
  PetscHashICreate(ht);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt localIndex = 0;
    PetscInt dof, off, i, j, k, l;

    PetscHashIClear(ht);
    ierr = PetscSectionGetDof(cellCounts, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cellCounts, v, &off);CHKERRQ(ierr);
    if (dof <= 0) continue;

    for (k = 0; k < patch->nsubspaces; ++k) {
      const PetscInt *cellNodeMap    = patch->cellNodeMap[k];
      PetscInt        nodesPerCell   = patch->nodesPerCell[k];
      PetscInt        subspaceOffset = patch->subspaceOffsets[k];
      PetscInt        bs             = patch->bs[k];

      for (i = off; i < off + dof; ++i) {
        /* Walk over the cells in this patch. */
        const PetscInt c = cellsArray[i];
        PetscInt       cell;

        ierr = PetscSectionGetDof(cellNumbering, c, &cell);CHKERRQ(ierr);
        if (cell <= 0) SETERRQ1(PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_OUTOFRANGE, "Cell %D doesn't appear in cell numbering map", c);
        ierr = PetscSectionGetOffset(cellNumbering, c, &cell);CHKERRQ(ierr);
        newCellsArray[i] = cell;
        for (j = 0; j < nodesPerCell; ++j) {
          /* For each global dof, map it into contiguous local storage. */
          const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + subspaceOffset;
          /* finally, loop over block size */
          for (l = 0; l < bs; ++l) {
            PetscInt localDof;

            PetscHashIMap(ht, globalDof + l, localDof);
            if (localDof == -1) {
              localDof = localIndex++;
              PetscHashIAdd(ht, globalDof + l, localDof);
            }
            if (globalIndex >= numDofs) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
            /* And store. */
            dofsArray[globalIndex++] = localDof;
          }
        }
      }
    }
    /* How many local dofs in this patch? */
    PetscHashISize(ht, dof);
    ierr = PetscSectionSetDof(gtolCounts, v, dof);CHKERRQ(ierr);
  }
  if (globalIndex != numDofs) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Expected number of dofs (%d) doesn't match found number (%d)", numDofs, globalIndex);
  ierr = PetscSectionSetUp(gtolCounts);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(gtolCounts, &numGlobalDofs);CHKERRQ(ierr);
  ierr = PetscMalloc1(numGlobalDofs, &globalDofsArray);CHKERRQ(ierr);

  /* Now populate the global to local map.  This could be merged into the above loop if we were willing to deal with reallocs. */
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIIter hi;
    PetscInt       dof, off, i, j, k, l;

    PetscHashIClear(ht);
    ierr = PetscSectionGetDof(cellCounts, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cellCounts, v, &off);CHKERRQ(ierr);
    if (dof <= 0) continue;

    for (k = 0; k < patch->nsubspaces; ++k) {
      const PetscInt *cellNodeMap    = patch->cellNodeMap[k];
      PetscInt        nodesPerCell   = patch->nodesPerCell[k];
      PetscInt        subspaceOffset = patch->subspaceOffsets[k];
      PetscInt        bs             = patch->bs[k];

      for (i = off; i < off + dof; ++i) {
        /* Reconstruct mapping of global-to-local on this patch. */
        const PetscInt c = cellsArray[i];
        PetscInt       cell;

        ierr = PetscSectionGetOffset(cellNumbering, c, &cell);CHKERRQ(ierr);
        for (j = 0; j < nodesPerCell; ++j) {
          for (l = 0; l < bs; ++l) {
            const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + subspaceOffset + l;
            const PetscInt localDof  = dofsArray[key];

            key += 1;
            PetscHashIAdd(ht, globalDof, localDof);
          }
        }
      }
      if (dof > 0) {
        /* Shove it in the output data structure. */
        PetscInt goff;

        ierr = PetscSectionGetOffset(gtolCounts, v, &goff);CHKERRQ(ierr);
        PetscHashIIterBegin(ht, hi);
        while (!PetscHashIIterAtEnd(ht, hi)) {
          PetscInt globalDof, localDof;

          PetscHashIIterGetKeyVal(ht, hi, globalDof, localDof);
          if (globalDof >= 0) globalDofsArray[goff + localDof] = globalDof;
          PetscHashIIterNext(ht, hi);
        }
      }
    }

    /* At this point, we have a hash table ht built that maps globalDof -> localDof.
     We need to create the dof table laid out cellwise first, then by subspace,
     as the assembler assembles cell-wise and we need to stuff the different
     contributions of the different function spaces to the right places. So we loop
     over cells, then over subspaces. */
    if (patch->nsubspaces > 1) { /* for nsubspaces = 1, data we need is already in dofsArray */
      for (i = off; i < off + dof; ++i) {
        const PetscInt c = cellsArray[i];
        PetscInt       cell;

        ierr = PetscSectionGetOffset(cellNumbering, c, &cell);CHKERRQ(ierr);
        for (k = 0; k < patch->nsubspaces; ++k) {
          const PetscInt *cellNodeMap    = patch->cellNodeMap[k];
          PetscInt        nodesPerCell   = patch->nodesPerCell[k];
          PetscInt        subspaceOffset = patch->subspaceOffsets[k];
          PetscInt        bs             = patch->bs[k];

          for (j = 0; j < nodesPerCell; ++j) {
            for (l = 0; l < bs; ++l) {
              const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + subspaceOffset + l;
              PetscInt       localDof;

              PetscHashIMap(ht, globalDof, localDof);
              asmArray[asmKey++] = localDof;
            }
          }
        }
      }
    }
  }
  if (1 == patch->nsubspaces) {ierr = PetscMemcpy(asmArray, dofsArray, numDofs * sizeof(PetscInt));CHKERRQ(ierr);}

  PetscHashIDestroy(ht);
  ierr = ISRestoreIndices(cells, &cellsArray);CHKERRQ(ierr);
  ierr = PetscFree(dofsArray);CHKERRQ(ierr);
  /* Replace cell indices with firedrake-numbered ones. */
  ierr = ISGeneralSetIndices(cells, numCells, (const PetscInt *) newCellsArray, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofs, globalDofsArray, PETSC_OWN_POINTER, &patch->gtol);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArray, PETSC_OWN_POINTER, &patch->dofs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchCreateCellPatchBCs(PC pc)
{
  PC_PATCH       *patch      = (PC_PATCH *) pc->data;
  const PetscInt *bcNodes    = NULL;
  PetscSection    gtolCounts = patch->gtolCounts;
  IS              gtol       = patch->gtol;
  DM              dm;
  PetscInt        numBcs;
  PetscSection    bcCounts;
  PetscHashI      globalBcs, localBcs, patchDofs;
  PetscHashI      ownedpts, seenpts, owneddofs, seendofs, artificialbcs;
  PetscHashIIter  hi;
  PetscInt       *bcsArray     = NULL;
  PetscInt       *multBcsArray = NULL;
  const PetscInt *gtolArray;
  PetscInt        vStart, vEnd, v, i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  PetscHashICreate(globalBcs);
  ierr = ISGetIndices(patch->ghostBcNodes, &bcNodes);CHKERRQ(ierr);
  ierr = ISGetSize(patch->ghostBcNodes, &numBcs);CHKERRQ(ierr);
  for (i = 0; i < numBcs; ++i) {
    /* these are already in concatenated numbering */
    PetscHashIAdd(globalBcs, bcNodes[i], 0);
  }
  ierr = ISRestoreIndices(patch->ghostBcNodes, &bcNodes);CHKERRQ(ierr);
  PetscHashICreate(patchDofs);
  PetscHashICreate(localBcs);
  PetscHashICreate(ownedpts);
  PetscHashICreate(seenpts);
  PetscHashICreate(owneddofs);
  PetscHashICreate(seendofs);
  PetscHashICreate(artificialbcs);

  ierr = PetscSectionGetChart(patch->cellCounts, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->bcCounts);CHKERRQ(ierr);
  bcCounts = patch->bcCounts;
  ierr = PetscSectionSetChart(bcCounts, vStart, vEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(vEnd - vStart, &patch->bcs);CHKERRQ(ierr);
  if (patch->multiplicative) {ierr = PetscMalloc1(vEnd - vStart, &patch->multBcs);CHKERRQ(ierr);}

  ierr = ISGetIndices(gtol, &gtolArray);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt bcIndex     = 0;
    PetscInt multBcIndex = 0;
    PetscInt numBcs, dof, off;

    PetscHashIClear(patchDofs);
    PetscHashIClear(localBcs);
    ierr = PetscSectionGetDof(gtolCounts, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(gtolCounts, v, &off);CHKERRQ(ierr);

    if (dof <= 0) {
      patch->bcs[v-vStart] = NULL;
      if (patch->multiplicative) patch->multBcs[v-vStart] = NULL;
      continue;
    }

    for (i = off; i < off + dof; ++i) {
      const PetscInt globalDof = gtolArray[i];
      const PetscInt localDof  = i-off;
      PetscBool      flg;

      PetscHashIAdd(patchDofs, globalDof, localDof);
      PetscHashIHasKey(globalBcs, globalDof, flg);
      if (flg) {PetscHashIAdd(localBcs, localDof, 0);}
    }

    /* If we're doing multiplicative, make the BC data structures now corresponding solely to actual globally imposed Dirichlet BCs */
    if (patch->multiplicative) {
      PetscHashISize(localBcs, numBcs);
      ierr = PetscMalloc1(numBcs, &multBcsArray);CHKERRQ(ierr);
      PetscHashIGetKeys(localBcs, &multBcIndex, multBcsArray);
      ierr = PetscSortInt(numBcs, multBcsArray);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF, numBcs, multBcsArray, PETSC_OWN_POINTER, &(patch->multBcs[v-vStart]));CHKERRQ(ierr);
    }

    /* Now figure out the artificial BCs: the set difference of {dofs on entities I see on the patch}\{dofs I am responsible for updating} */
    ierr = patch->patchconstructop((void*) patch, dm, v, ownedpts);CHKERRQ(ierr);
    ierr = PCPatchCompleteCellPatch(dm, ownedpts, seenpts);CHKERRQ(ierr);
    ierr = PCPatchGetPointDofs(patch, ownedpts, owneddofs, v, patch->exclude_subspace);CHKERRQ(ierr);
    ierr = PCPatchGetPointDofs(patch, seenpts, seendofs, v, -1);CHKERRQ(ierr);
    ierr = PCPatchComputeSetDifference_Private(owneddofs, seendofs, artificialbcs);CHKERRQ(ierr);

    if (patch->print_patches) {
      PetscHashI globalbcdofs;
      MPI_Comm   comm;

      PetscHashICreate(globalbcdofs);

      ierr = PetscObjectGetComm((PetscObject) pc, &comm);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: owned dofs:\n", v);CHKERRQ(ierr);
      PetscHashIIterBegin(owneddofs, hi);
      while (!PetscHashIIterAtEnd(owneddofs, hi)) {
        PetscInt globalDof;

        PetscHashIIterGetKey(owneddofs, hi, globalDof);
        PetscHashIIterNext(owneddofs, hi);
        ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof);CHKERRQ(ierr);
      }
      ierr = PetscSynchronizedPrintf(comm, "\n");CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: seen dofs:\n", v);CHKERRQ(ierr);
      PetscHashIIterBegin(seendofs, hi);
      while (!PetscHashIIterAtEnd(seendofs, hi)) {
        PetscInt globalDof;
        PetscBool flg;

        PetscHashIIterGetKey(seendofs, hi, globalDof);
        PetscHashIIterNext(seendofs, hi);
        ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof);CHKERRQ(ierr);
        PetscHashIHasKey(globalBcs, globalDof, flg);
        if (flg) {PetscHashIAdd(globalbcdofs, globalDof, 0);}
      }
      ierr = PetscSynchronizedPrintf(comm, "\n");CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: global BCs:\n", v);CHKERRQ(ierr);
      PetscHashISize(globalbcdofs, numBcs);
      if (numBcs > 0) {
        PetscHashIIterBegin(globalbcdofs, hi);
        while (!PetscHashIIterAtEnd(globalbcdofs, hi)) {
          PetscInt globalDof;
          PetscHashIIterGetKey(globalbcdofs, hi, globalDof);
          PetscHashIIterNext(globalbcdofs, hi);
          ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof);CHKERRQ(ierr);
        }
      }
      ierr = PetscSynchronizedPrintf(comm, "\n");CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: artificial BCs:\n", v);CHKERRQ(ierr);
      PetscHashISize(artificialbcs, numBcs);
      if (numBcs > 0) {
        PetscHashIIterBegin(artificialbcs, hi);
        while (!PetscHashIIterAtEnd(artificialbcs, hi)) {
          PetscInt globalDof;
          PetscHashIIterGetKey(artificialbcs, hi, globalDof);
          PetscHashIIterNext(artificialbcs, hi);
          ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof);CHKERRQ(ierr);
        }
      }
      ierr = PetscSynchronizedPrintf(comm, "\n\n");CHKERRQ(ierr);
      ierr = PetscSynchronizedFlush(comm, PETSC_STDOUT);CHKERRQ(ierr);
      PetscHashIDestroy(globalbcdofs);
    }

    PetscHashISize(artificialbcs, numBcs);
    if (numBcs > 0) {
      PetscHashIIterBegin(artificialbcs, hi);
      while (!PetscHashIIterAtEnd(artificialbcs, hi)) {
        PetscInt globalDof, localDof;
        PetscHashIIterGetKey(artificialbcs, hi, globalDof);
        PetscHashIIterNext(artificialbcs, hi);
        PetscHashIMap(patchDofs, globalDof, localDof);
        if (localDof == -1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Patch %d Didn't find dof %d in patch\n", v-vStart, globalDof);
        PetscHashIAdd(localBcs, localDof, 0);
      }
    }

    /* OK, now we have a hash table with all the bcs indicated by the artificial and global bcs */
    PetscHashISize(localBcs, numBcs);
    ierr = PetscSectionSetDof(bcCounts, v, numBcs);CHKERRQ(ierr);
    ierr = PetscMalloc1(numBcs, &bcsArray);CHKERRQ(ierr);
    PetscHashIGetKeys(localBcs, &bcIndex, bcsArray);
    ierr = PetscSortInt(numBcs, bcsArray);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numBcs, bcsArray, PETSC_OWN_POINTER, &(patch->bcs[v - vStart]));CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(gtol, &gtolArray);CHKERRQ(ierr);
  PetscHashIDestroy(artificialbcs);
  PetscHashIDestroy(seendofs);
  PetscHashIDestroy(owneddofs);
  PetscHashIDestroy(seenpts);
  PetscHashIDestroy(ownedpts);
  PetscHashIDestroy(localBcs);
  PetscHashIDestroy(patchDofs);
  PetscHashIDestroy(globalBcs);
  ierr = ISDestroy(&patch->ghostBcNodes);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(bcCounts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchZeroFillMatrix_Private(Mat mat, const PetscInt ncell, const PetscInt ndof, const PetscInt *dof)
{
  const PetscScalar *values = NULL;
  PetscInt           rows, c, i;
  PetscErrorCode     ierr;

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
  ierr = MatAppendOptionsPrefix(*mat, "sub_");CHKERRQ(ierr);
  if (patch->sub_mat_type) {ierr = MatSetType(*mat, patch->sub_mat_type);CHKERRQ(ierr);}
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
        for (j = 0; j < patch->totalDofsPerCell; ++j) {
          const PetscInt col = idx[j];
          const PetscInt key = row*rsize + col;
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

static PetscErrorCode PCPatchComputeOperator_Private(PC pc, Mat mat, Mat multMat, PetscInt point)
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
  ierr = patch->usercomputeop(pc, mat, ncell, cellsArray + offset, ncell*patch->totalDofsPerCell, dofsArray + offset*patch->totalDofsPerCell, patch->usercomputectx);CHKERRQ(ierr);
  PetscStackPop;
  ierr = ISRestoreIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(patch->cells, &cellsArray);CHKERRQ(ierr);
  /* Apply boundary conditions.  Could also do this through the local_to_patch guy. */
  if (patch->multiplicative) {
    ierr = MatCopy(mat, multMat, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatZeroRowsColumnsIS(multMat, patch->multBcs[point-pStart], (PetscScalar) 1.0, NULL, NULL);CHKERRQ(ierr);
  }
  ierr = MatZeroRowsColumnsIS(mat, patch->bcs[point-pStart], (PetscScalar) 1.0, NULL, NULL);CHKERRQ(ierr);
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
  PetscScalar    *patchX  = NULL;
  const PetscInt *bcNodes = NULL;
  PetscInt        numBcs, i, j;
  const char     *prefix;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscInt pStart, pEnd, p;
    PetscInt localSize;

    ierr = PetscLogEventBegin(PC_Patch_CreatePatches, pc, 0, 0, 0);CHKERRQ(ierr);

    localSize = patch->subspaceOffsets[patch->nsubspaces];
    ierr = VecCreateSeq(PETSC_COMM_SELF, localSize, &patch->localX);CHKERRQ(ierr);
    ierr = VecSetUp(patch->localX);CHKERRQ(ierr);
    ierr = VecDuplicate(patch->localX, &patch->localY);CHKERRQ(ierr);
    ierr = PCPatchCreateCellPatches(pc);CHKERRQ(ierr);
    ierr = PCPatchCreateCellPatchDiscretisationInfo(pc);CHKERRQ(ierr);
    ierr = PCPatchCreateCellPatchBCs(pc);CHKERRQ(ierr);

    /* OK, now build the work vectors */
    ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(patch->npatch, &patch->patchX);CHKERRQ(ierr);
    ierr = PetscMalloc1(patch->npatch, &patch->patchY);CHKERRQ(ierr);
    if (patch->partition_of_unity && patch->multiplicative) {ierr = PetscMalloc1(patch->npatch, &patch->patch_dof_weights);CHKERRQ(ierr);}
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof;

      ierr = PetscSectionGetDof(patch->gtolCounts, p, &dof);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF, dof, &patch->patchX[p-pStart]);CHKERRQ(ierr);
      ierr = VecSetUp(patch->patchX[p-pStart]);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF, dof, &patch->patchY[p-pStart]);CHKERRQ(ierr);
      ierr = VecSetUp(patch->patchY[p-pStart]);CHKERRQ(ierr);
      if (patch->partition_of_unity && patch->multiplicative) {
        ierr = VecCreateSeq(PETSC_COMM_SELF, dof, &patch->patch_dof_weights[p-pStart]);CHKERRQ(ierr);
        ierr = VecSetUp(patch->patch_dof_weights[p-pStart]);CHKERRQ(ierr);
      }
    }
    ierr = PetscMalloc1(patch->npatch, &patch->ksp);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc, &prefix);CHKERRQ(ierr);
    for (i = 0; i < patch->npatch; ++i) {
      ierr = KSPCreate(PETSC_COMM_SELF, &patch->ksp[i]);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(patch->ksp[i], prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(patch->ksp[i], "sub_");CHKERRQ(ierr);
    }
    if (patch->save_operators) {
      ierr = PetscMalloc1(patch->npatch, &patch->mat);CHKERRQ(ierr);
      if (patch->multiplicative) {ierr = PetscMalloc1(patch->npatch, &patch->multMat);CHKERRQ(ierr);}
      for (i = 0; i < patch->npatch; ++i) {
        ierr = PCPatchCreateMatrix_Private(pc, i, &patch->mat[i]);CHKERRQ(ierr);
        if (patch->multiplicative) {ierr = MatDuplicate(patch->mat[i], MAT_SHARE_NONZERO_PATTERN, &patch->multMat[i]);CHKERRQ(ierr);}
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
        /* TODO: Do we need different scatters for X and Y? */
        ierr = VecGetArray(patch->patchX[i], &patchX);CHKERRQ(ierr);
        /* Apply bcs to patchX (zero entries) */
        ierr = ISGetLocalSize(patch->bcs[i], &numBcs);CHKERRQ(ierr);
        ierr = ISGetIndices(patch->bcs[i], &bcNodes);CHKERRQ(ierr);
        for (j = 0; j < numBcs; ++j) patchX[bcNodes[j]] = 0;
        ierr = ISRestoreIndices(patch->bcs[i], &bcNodes);CHKERRQ(ierr);
        ierr = VecRestoreArray(patch->patchX[i], &patchX);CHKERRQ(ierr);

        ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchX[i], patch->dof_weights, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
      }
      ierr = VecReciprocal(patch->dof_weights);CHKERRQ(ierr);
      if (patch->partition_of_unity && patch->multiplicative) {
        for (i = 0; i < patch->npatch; ++i) {
          ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->dof_weights, patch->patch_dof_weights[i], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
        }
      }
    }
  }
  if (patch->save_operators) {
    for (i = 0; i < patch->npatch; ++i) {
      ierr = MatZeroEntries(patch->mat[i]);CHKERRQ(ierr);
      if (patch->multiplicative) {ierr = PCPatchComputeOperator_Private(pc, patch->mat[i], patch->multMat[i], i);CHKERRQ(ierr);}
      else                       {ierr = PCPatchComputeOperator_Private(pc, patch->mat[i], NULL, i);CHKERRQ(ierr);}
      ierr = KSPSetOperators(patch->ksp[i], patch->mat[i], patch->mat[i]);CHKERRQ(ierr);
    }
  }
  if (!pc->setupcalled) {
    /* TODO: Only call if SetFromOptions was called on this PC */
    for (i = 0; i < patch->npatch; ++i) {ierr = KSPSetFromOptions(patch->ksp[i]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PATCH(PC pc, Vec x, Vec y)
{
  PC_PATCH          *patch    = (PC_PATCH *) pc->data;
  const PetscScalar *globalX  = NULL;
  PetscScalar       *localX   = NULL;
  PetscScalar       *globalY  = NULL;
  PetscScalar       *patchX   = NULL;
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
      Mat      multMat = NULL;
      PetscInt i       = patch->user_patches ? iterationSet[j] : j;
      PetscInt start, len;

      ierr = PetscSectionGetDof(patch->gtolCounts, i+pStart, &len);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(patch->gtolCounts, i+pStart, &start);CHKERRQ(ierr);
      /* TODO: Squash out these guys in the setup as well. */
      if (len <= 0) continue;
      /* TODO: Do we need different scatters for X and Y? */
      ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localX, patch->patchX[i], INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
      /* Apply bcs to patchX (zero entries) */
      ierr = VecGetArray(patch->patchX[i], &patchX);CHKERRQ(ierr);
      ierr = ISGetLocalSize(patch->bcs[i], &numBcs);CHKERRQ(ierr);
      ierr = ISGetIndices(patch->bcs[i], &bcNodes);CHKERRQ(ierr);
      for (bc = 0; bc < numBcs; ++bc) patchX[bcNodes[bc]] = 0;
      ierr = ISRestoreIndices(patch->bcs[i], &bcNodes);CHKERRQ(ierr);
      ierr = VecRestoreArray(patch->patchX[i], &patchX);CHKERRQ(ierr);
      if (!patch->save_operators) {
        Mat mat;

        ierr = PCPatchCreateMatrix_Private(pc, i, &mat);CHKERRQ(ierr);
        if (patch->multiplicative) {ierr = MatDuplicate(mat, MAT_SHARE_NONZERO_PATTERN, &multMat);CHKERRQ(ierr);}
        /* Populate operator here. */
        ierr = PCPatchComputeOperator_Private(pc, mat, multMat, i);CHKERRQ(ierr);
        ierr = KSPSetOperators(patch->ksp[i], mat, mat);
        /* Drop reference so the KSPSetOperators below will blow it away. */
        ierr = MatDestroy(&mat);CHKERRQ(ierr);
      } else if (patch->multiplicative) {
        multMat = patch->multMat[i];
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

      if (patch->partition_of_unity && patch->multiplicative) {
        ierr = VecPointwiseMult(patch->patchY[i], patch->patchY[i], patch->patch_dof_weights[i]);CHKERRQ(ierr);
      }
      ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchY[i], patch->localY, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);

      /* Get the matrix on the patch but with only global bcs applied.
       * This matrix is then multiplied with the result from the previous solve
       * to obtain by how much the residual changes. */
      if (patch->multiplicative) {
        ierr = MatMult(multMat, patch->patchY[i], patch->patchX[i]);CHKERRQ(ierr);
        ierr = VecScale(patch->patchX[i], -1.0);CHKERRQ(ierr);
        ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchX[i], patch->localX, ADD_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
        if (!patch->save_operators) {ierr = MatDestroy(&multMat);CHKERRQ(ierr);}
      }
    }
  }
  if (patch->user_patches) {ierr = ISRestoreIndices(patch->iterationSet, &iterationSet);CHKERRQ(ierr);}
  /* XXX: should we do this on the global vector? */
  if (patch->partition_of_unity && !patch->multiplicative) {
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
  ierr = PetscSectionDestroy(&patch->cellNumbering);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->gtolCounts);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->bcCounts);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->gtol);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->cells);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->dofs);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->ghostBcNodes);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->globalBcNodes);CHKERRQ(ierr);

  if (patch->dofSection) {
    for (i = 0; i < patch->nsubspaces; i++) {
      ierr = PetscSectionDestroy(&patch->dofSection[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(patch->dofSection);CHKERRQ(ierr);
  ierr = PetscFree(patch->bs);CHKERRQ(ierr);
  ierr = PetscFree(patch->nodesPerCell);CHKERRQ(ierr);
  ierr = PetscFree(patch->cellNodeMap);CHKERRQ(ierr);
  ierr = PetscFree(patch->subspaceOffsets);CHKERRQ(ierr);

  if (patch->bcs) {
    for (i = 0; i < patch->npatch; ++i) {
      ierr = ISDestroy(&patch->bcs[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(patch->bcs);CHKERRQ(ierr);
  }
  if (patch->multBcs) {
    for (i = 0; i < patch->npatch; ++i) {
      ierr = ISDestroy(&patch->multBcs[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(patch->multBcs);CHKERRQ(ierr);
  }
  if (patch->ksp) {
    for (i = 0; i < patch->npatch; ++i) {
      ierr = KSPReset(patch->ksp[i]);CHKERRQ(ierr);
    }
  }

  ierr = VecDestroy(&patch->localX);CHKERRQ(ierr);
  ierr = VecDestroy(&patch->localY);CHKERRQ(ierr);
  if (patch->patchX) {
    for (i = 0; i < patch->npatch; ++i) {
      ierr = VecDestroy(&patch->patchX[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(patch->patchX);CHKERRQ(ierr);
  }
  if (patch->patchY) {
    for (i = 0; i < patch->npatch; ++i) {
      ierr = VecDestroy(&patch->patchY[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(patch->patchY);CHKERRQ(ierr);
  }
  if (patch->partition_of_unity) {
    ierr = VecDestroy(&patch->dof_weights);CHKERRQ(ierr);
  }
  if (patch->patch_dof_weights) {
    for (i = 0; i < patch->npatch; ++i) {
      ierr = VecDestroy(&patch->patch_dof_weights[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(patch->patch_dof_weights);CHKERRQ(ierr);
  }
  if (patch->mat) {
    for (i = 0; i < patch->npatch; ++i) {
      ierr = MatDestroy(&patch->mat[i]);CHKERRQ(ierr);
      if (patch->multiplicative) {
        ierr = MatDestroy(&patch->multMat[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(patch->mat);CHKERRQ(ierr);
    if (patch->multiplicative) {
      ierr = PetscFree(patch->multMat);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(patch->sub_mat_type);CHKERRQ(ierr);

  patch->bs = 0;
  patch->cellNodeMap = NULL;

  if (patch->user_patches) {
    for (i = 0; i < patch->nuserIS; ++i) {
      ierr = ISDestroy(&patch->userIS[i]);CHKERRQ(ierr);
    }
    PetscFree(patch->userIS);
    patch->nuserIS = 0;
  }
  ierr = ISDestroy(&patch->iterationSet);CHKERRQ(ierr);
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
  char                 sub_mat_type[256];
  PetscBool            flg, dimflg, codimflg;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "Vertex-patch Additive Schwarz options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_patch_save_operators",  "Store all patch operators for lifetime of PC?", "PCPatchSetSaveOperators", patch->save_operators, &patch->save_operators, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_patch_partition_of_unity", "Weight contributions by dof multiplicity?", "PCPatchSetPartitionOfUnity", patch->partition_of_unity, &patch->partition_of_unity, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_patch_multiplicative", "Gauss-Seidel instead of Jacobi?", "PCPatchSetMultiplicative", patch->multiplicative, &patch->multiplicative, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_patch_construct_dim", "What dimension of mesh point to construct patches by? (0 = vertices)", "PCSetFromOptions_PATCH", patch->dim, &patch->dim, &dimflg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_patch_construct_codim", "What co-dimension of mesh point to construct patches by? (0 = cells)", "PCSetFromOptions_PATCH", patch->codim, &patch->codim, &codimflg);CHKERRQ(ierr);
  if (dimflg && codimflg) SETERRQ(PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONG, "Can only set one of dimension or co-dimension");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_patch_construct_type", "How should the patches be constructed?", "PCPatchSetConstructType", PCPatchConstructTypes, (PetscEnum) patchConstructionType, (PetscEnum *) &patchConstructionType, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_patch_vanka_dim", "Topological dimension of entities for Vanka to ignore", "PCSetFromOptions_PATCH", patch->vankadim, &patch->vankadim, &flg);
  if (flg) {ierr = PCPatchSetConstructType(pc, patchConstructionType, NULL, NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsFList("-pc_patch_sub_mat_type", "Matrix type for patch solves", "PCPatchSetSubMatType", MatList, NULL, sub_mat_type, 256, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PCPatchSetSubMatType(pc, sub_mat_type);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-pc_patch_print_patches", "Print out information during patch construction?", "PCSetFromOptions_PATCH", patch->print_patches, &patch->print_patches, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_patch_symmetrise_sweep", "Go start->end, end->start?", "PCSetFromOptions_PATCH", patch->symmetrise_sweep, &patch->symmetrise_sweep, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_patch_exclude_subspace", "What subspace (if any) to exclude in construction?", "PCSetFromOptions_PATCH", patch->exclude_subspace, &patch->exclude_subspace, &flg);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  if (patch->multiplicative) {ierr = PetscViewerASCIIPrintf(viewer, "Schwarz type: multiplicative\n");CHKERRQ(ierr);}
  else                       {ierr = PetscViewerASCIIPrintf(viewer, "Schwarz type: additive\n");CHKERRQ(ierr);}
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

PETSC_EXTERN PetscErrorCode PCCreate_PATCH(PC pc)
{
  PC_PATCH      *patch;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc, &patch);CHKERRQ(ierr);

  /* Set some defaults */
  patch->sub_mat_type       = NULL;
  patch->save_operators     = PETSC_TRUE;
  patch->partition_of_unity = PETSC_FALSE;
  patch->multiplicative     = PETSC_FALSE;
  patch->codim              = -1;
  patch->dim                = -1;
  patch->exclude_subspace   = -1;
  patch->vankadim           = -1;
  patch->patchconstructop   = PCPatchConstruct_Star;
  patch->print_patches      = PETSC_FALSE;
  patch->symmetrise_sweep   = PETSC_FALSE;
  patch->nuserIS            = 0;
  patch->userIS             = NULL;
  patch->iterationSet       = NULL;
  patch->user_patches       = PETSC_FALSE;

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
