#include <petsc/private/pcpatchimpl.h>     /*I "petscpc.h" I*/
#include <petsc/private/kspimpl.h>         /* For ksp->setfromoptionscalled */
#include <petsc/private/vecimpl.h>         /* For vec->map */
#include <petsc/private/dmpleximpl.h> /* For DMPlexComputeJacobian_Patch_Internal() */
#include <petscsf.h>
#include <petscbt.h>
#include <petscds.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

PetscLogEvent PC_Patch_CreatePatches, PC_Patch_ComputeOp, PC_Patch_Solve, PC_Patch_Apply, PC_Patch_Prealloc;

static inline PetscErrorCode ObjectView(PetscObject obj, PetscViewer viewer, PetscViewerFormat format)
{

  PetscCall(PetscViewerPushFormat(viewer, format));
  PetscCall(PetscObjectView(obj, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  return(0);
}

static PetscErrorCode PCPatchConstruct_Star(void *vpatch, DM dm, PetscInt point, PetscHSetI ht)
{
  PetscInt       starSize;
  PetscInt      *star = NULL, si;

  PetscFunctionBegin;
  PetscCall(PetscHSetIClear(ht));
  /* To start with, add the point we care about */
  PetscCall(PetscHSetIAdd(ht, point));
  /* Loop over all the points that this point connects to */
  PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
  for (si = 0; si < starSize*2; si += 2) PetscCall(PetscHSetIAdd(ht, star[si]));
  PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchConstruct_Vanka(void *vpatch, DM dm, PetscInt point, PetscHSetI ht)
{
  PC_PATCH      *patch = (PC_PATCH *) vpatch;
  PetscInt       starSize;
  PetscInt      *star = NULL;
  PetscBool      shouldIgnore = PETSC_FALSE;
  PetscInt       cStart, cEnd, iStart, iEnd, si;

  PetscFunctionBegin;
  PetscCall(PetscHSetIClear(ht));
  /* To start with, add the point we care about */
  PetscCall(PetscHSetIAdd(ht, point));
  /* Should we ignore any points of a certain dimension? */
  if (patch->vankadim >= 0) {
    shouldIgnore = PETSC_TRUE;
    PetscCall(DMPlexGetDepthStratum(dm, patch->vankadim, &iStart, &iEnd));
  }
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  /* Loop over all the cells that this point connects to */
  PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
  for (si = 0; si < starSize*2; si += 2) {
    const PetscInt cell = star[si];
    PetscInt       closureSize;
    PetscInt      *closure = NULL, ci;

    if (cell < cStart || cell >= cEnd) continue;
    /* now loop over all entities in the closure of that cell */
    PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    for (ci = 0; ci < closureSize*2; ci += 2) {
      const PetscInt newpoint = closure[ci];

      /* We've been told to ignore entities of this type.*/
      if (shouldIgnore && newpoint >= iStart && newpoint < iEnd) continue;
      PetscCall(PetscHSetIAdd(ht, newpoint));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchConstruct_Pardecomp(void *vpatch, DM dm, PetscInt point, PetscHSetI ht)
{
  PC_PATCH       *patch = (PC_PATCH *) vpatch;
  DMLabel         ghost = NULL;
  const PetscInt *leaves;
  PetscInt        nleaves, pStart, pEnd, loc;
  PetscBool       isFiredrake;
  PetscBool       flg;
  PetscInt        starSize;
  PetscInt       *star = NULL;
  PetscInt        opoint, overlapi;

  PetscFunctionBegin;
  PetscCall(PetscHSetIClear(ht));

  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));

  PetscCall(DMHasLabel(dm, "pyop2_ghost", &isFiredrake));
  if (isFiredrake) {
    PetscCall(DMGetLabel(dm, "pyop2_ghost", &ghost));
    PetscCall(DMLabelCreateIndex(ghost, pStart, pEnd));
  } else {
    PetscSF sf;
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL));
    nleaves = PetscMax(nleaves, 0);
  }

  for (opoint = pStart; opoint < pEnd; ++opoint) {
    if (ghost) PetscCall(DMLabelHasPoint(ghost, opoint, &flg));
    else       {PetscCall(PetscFindInt(opoint, nleaves, leaves, &loc)); flg = loc >=0 ? PETSC_TRUE : PETSC_FALSE;}
    /* Not an owned entity, don't make a cell patch. */
    if (flg) continue;
    PetscCall(PetscHSetIAdd(ht, opoint));
  }

  /* Now build the overlap for the patch */
  for (overlapi = 0; overlapi < patch->pardecomp_overlap; ++overlapi) {
    PetscInt index = 0;
    PetscInt *htpoints = NULL;
    PetscInt htsize;
    PetscInt i;

    PetscCall(PetscHSetIGetSize(ht, &htsize));
    PetscCall(PetscMalloc1(htsize, &htpoints));
    PetscCall(PetscHSetIGetElems(ht, &index, htpoints));

    for (i = 0; i < htsize; ++i) {
      PetscInt hpoint = htpoints[i];
      PetscInt si;

      PetscCall(DMPlexGetTransitiveClosure(dm, hpoint, PETSC_FALSE, &starSize, &star));
      for (si = 0; si < starSize*2; si += 2) {
        const PetscInt starp = star[si];
        PetscInt       closureSize;
        PetscInt      *closure = NULL, ci;

        /* now loop over all entities in the closure of starp */
        PetscCall(DMPlexGetTransitiveClosure(dm, starp, PETSC_TRUE, &closureSize, &closure));
        for (ci = 0; ci < closureSize*2; ci += 2) {
          const PetscInt closstarp = closure[ci];
          PetscCall(PetscHSetIAdd(ht, closstarp));
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm, starp, PETSC_TRUE, &closureSize, &closure));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, hpoint, PETSC_FALSE, &starSize, &star));
    }
    PetscCall(PetscFree(htpoints));
  }

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

  PetscFunctionBegin;
  PetscCall(PetscHSetIClear(ht));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(ISGetLocalSize(patchis, &n));
  PetscCall(ISGetIndices(patchis, &patchdata));
  for (i = 0; i < n; ++i) {
    const PetscInt ownedpoint = patchdata[i];

    if (ownedpoint < pStart || ownedpoint >= pEnd) {
      SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D was not in [%D, %D)", ownedpoint, pStart, pEnd);
    }
    PetscCall(PetscHSetIAdd(ht, ownedpoint));
  }
  PetscCall(ISRestoreIndices(patchis, &patchdata));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchCreateDefaultSF_Private(PC pc, PetscInt n, const PetscSF *sf, const PetscInt *bs)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (n == 1 && bs[0] == 1) {
    patch->sectionSF = sf[0];
    PetscCall(PetscObjectReference((PetscObject) patch->sectionSF));
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

      PetscCall(PetscSFGetGraph(sf[i], &nroots, &nleaves, NULL, NULL));
      allRoots  += nroots * bs[i];
      allLeaves += nleaves * bs[i];
    }
    PetscCall(PetscMalloc1(allLeaves, &ilocal));
    PetscCall(PetscMalloc1(allLeaves, &iremote));
    /* Now build an SF that just contains process connectivity. */
    PetscCall(PetscHSetICreate(&ranksUniq));
    for (i = 0; i < n; ++i) {
      const PetscMPIInt *ranks = NULL;
      PetscInt           nranks, j;

      PetscCall(PetscSFSetUp(sf[i]));
      PetscCall(PetscSFGetRootRanks(sf[i], &nranks, &ranks, NULL, NULL, NULL));
      /* These are all the ranks who communicate with me. */
      for (j = 0; j < nranks; ++j) {
        PetscCall(PetscHSetIAdd(ranksUniq, (PetscInt) ranks[j]));
      }
    }
    PetscCall(PetscHSetIGetSize(ranksUniq, &numRanks));
    PetscCall(PetscMalloc1(numRanks, &remote));
    PetscCall(PetscMalloc1(numRanks, &ranks));
    PetscCall(PetscHSetIGetElems(ranksUniq, &index, ranks));

    PetscCall(PetscHMapICreate(&rankToIndex));
    for (i = 0; i < numRanks; ++i) {
      remote[i].rank  = ranks[i];
      remote[i].index = 0;
      PetscCall(PetscHMapISet(rankToIndex, ranks[i], i));
    }
    PetscCall(PetscFree(ranks));
    PetscCall(PetscHSetIDestroy(&ranksUniq));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject) pc), &rankSF));
    PetscCall(PetscSFSetGraph(rankSF, 1, numRanks, NULL, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER));
    PetscCall(PetscSFSetUp(rankSF));
    /* OK, use it to communicate the root offset on the remote
     * processes for each subspace. */
    PetscCall(PetscMalloc1(n, &offsets));
    PetscCall(PetscMalloc1(n*numRanks, &remoteOffsets));

    offsets[0] = 0;
    for (i = 1; i < n; ++i) {
      PetscInt nroots;

      PetscCall(PetscSFGetGraph(sf[i-1], &nroots, NULL, NULL, NULL));
      offsets[i] = offsets[i-1] + nroots*bs[i-1];
    }
    /* Offsets are the offsets on the current process of the
     * global dof numbering for the subspaces. */
    PetscCallMPI(MPI_Type_contiguous(n, MPIU_INT, &contig));
    PetscCallMPI(MPI_Type_commit(&contig));

    PetscCall(PetscSFBcastBegin(rankSF, contig, offsets, remoteOffsets,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(rankSF, contig, offsets, remoteOffsets,MPI_REPLACE));
    PetscCallMPI(MPI_Type_free(&contig));
    PetscCall(PetscFree(offsets));
    PetscCall(PetscSFDestroy(&rankSF));
    /* Now remoteOffsets contains the offsets on the remote
     * processes who communicate with me.  So now we can
     * concatenate the list of SFs into a single one. */
    index = 0;
    for (i = 0; i < n; ++i) {
      const PetscSFNode *remote = NULL;
      const PetscInt    *local  = NULL;
      PetscInt           nroots, nleaves, j;

      PetscCall(PetscSFGetGraph(sf[i], &nroots, &nleaves, &local, &remote));
      for (j = 0; j < nleaves; ++j) {
        PetscInt rank = remote[j].rank;
        PetscInt idx, rootOffset, k;

        PetscCall(PetscHMapIGet(rankToIndex, rank, &idx));
        PetscCheckFalse(idx == -1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Didn't find rank, huh?");
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
    PetscCall(PetscHMapIDestroy(&rankToIndex));
    PetscCall(PetscFree(remoteOffsets));
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)pc), &patch->sectionSF));
    PetscCall(PetscSFSetGraph(patch->sectionSF, allRoots, allLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCPatchSetDenseInverse(PC pc, PetscBool flg)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  patch->denseinverse = flg;
  PetscFunctionReturn(0);
}

PetscErrorCode PCPatchGetDenseInverse(PC pc, PetscBool *flg)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  *flg = patch->denseinverse;
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
PetscErrorCode PCPatchSetPrecomputeElementTensors(PC pc, PetscBool flg)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  patch->precomputeElementTensors = flg;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchGetPrecomputeElementTensors(PC pc, PetscBool *flg)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  *flg = patch->precomputeElementTensors;
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
PetscErrorCode PCPatchSetLocalComposition(PC pc, PCCompositeType type)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  PetscCheckFalse(type != PC_COMPOSITE_ADDITIVE && type != PC_COMPOSITE_MULTIPLICATIVE,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Only supports additive or multiplicative as the local type");
  patch->local_composition_type = type;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchGetLocalComposition(PC pc, PCCompositeType *type)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;
  PetscFunctionBegin;
  *type = patch->local_composition_type;
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetSubMatType(PC pc, MatType sub_mat_type)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  if (patch->sub_mat_type) PetscCall(PetscFree(patch->sub_mat_type));
  PetscCall(PetscStrallocpy(sub_mat_type, (char **) &patch->sub_mat_type));
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

  PetscFunctionBegin;
  patch->cellNumbering = cellNumbering;
  PetscCall(PetscObjectReference((PetscObject) cellNumbering));
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
  case PC_PATCH_PARDECOMP:
    patch->user_patches     = PETSC_FALSE;
    patch->patchconstructop = PCPatchConstruct_Pardecomp;
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
    SETERRQ(PetscObjectComm((PetscObject) pc), PETSC_ERR_USER, "Unknown patch construction type %D", (PetscInt) patch->ctype);
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
  case PC_PATCH_PARDECOMP:
    break;
  case PC_PATCH_USER:
  case PC_PATCH_PYTHON:
    *func = patch->userpatchconstructionop;
    *ctx  = patch->userpatchconstructctx;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject) pc), PETSC_ERR_USER, "Unknown patch construction type %D", (PetscInt) patch->ctype);
  }
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetDiscretisationInfo(PC pc, PetscInt nsubspaces, DM *dms, PetscInt *bs, PetscInt *nodesPerCell, const PetscInt **cellNodeMap,
                                            const PetscInt *subspaceOffsets, PetscInt numGhostBcs, const PetscInt *ghostBcNodes, PetscInt numGlobalBcs, const PetscInt *globalBcNodes)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  DM             dm, plex;
  PetscSF       *sfs;
  PetscInt       cStart, cEnd, i, j;

  PetscFunctionBegin;
  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscMalloc1(nsubspaces, &sfs));
  PetscCall(PetscMalloc1(nsubspaces, &patch->dofSection));
  PetscCall(PetscMalloc1(nsubspaces, &patch->bs));
  PetscCall(PetscMalloc1(nsubspaces, &patch->nodesPerCell));
  PetscCall(PetscMalloc1(nsubspaces, &patch->cellNodeMap));
  PetscCall(PetscMalloc1(nsubspaces+1, &patch->subspaceOffsets));

  patch->nsubspaces       = nsubspaces;
  patch->totalDofsPerCell = 0;
  for (i = 0; i < nsubspaces; ++i) {
    PetscCall(DMGetLocalSection(dms[i], &patch->dofSection[i]));
    PetscCall(PetscObjectReference((PetscObject) patch->dofSection[i]));
    PetscCall(DMGetSectionSF(dms[i], &sfs[i]));
    patch->bs[i]              = bs[i];
    patch->nodesPerCell[i]    = nodesPerCell[i];
    patch->totalDofsPerCell  += nodesPerCell[i]*bs[i];
    PetscCall(PetscMalloc1((cEnd-cStart)*nodesPerCell[i], &patch->cellNodeMap[i]));
    for (j = 0; j < (cEnd-cStart)*nodesPerCell[i]; ++j) patch->cellNodeMap[i][j] = cellNodeMap[i][j];
    patch->subspaceOffsets[i] = subspaceOffsets[i];
  }
  PetscCall(PCPatchCreateDefaultSF_Private(pc, nsubspaces, sfs, patch->bs));
  PetscCall(PetscFree(sfs));

  patch->subspaceOffsets[nsubspaces] = subspaceOffsets[nsubspaces];
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numGhostBcs, ghostBcNodes, PETSC_COPY_VALUES, &patch->ghostBcNodes));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numGlobalBcs, globalBcNodes, PETSC_COPY_VALUES, &patch->globalBcNodes));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetDiscretisationInfoCombined(PC pc, DM dm, PetscInt *nodesPerCell, const PetscInt **cellNodeMap, PetscInt numGhostBcs, const PetscInt *ghostBcNodes, PetscInt numGlobalBcs, const PetscInt *globalBcNodes)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       cStart, cEnd, i, j;

  PetscFunctionBegin;
  patch->combined = PETSC_TRUE;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetNumFields(dm, &patch->nsubspaces));
  PetscCall(PetscCalloc1(patch->nsubspaces, &patch->dofSection));
  PetscCall(PetscMalloc1(patch->nsubspaces, &patch->bs));
  PetscCall(PetscMalloc1(patch->nsubspaces, &patch->nodesPerCell));
  PetscCall(PetscMalloc1(patch->nsubspaces, &patch->cellNodeMap));
  PetscCall(PetscCalloc1(patch->nsubspaces+1, &patch->subspaceOffsets));
  PetscCall(DMGetLocalSection(dm, &patch->dofSection[0]));
  PetscCall(PetscObjectReference((PetscObject) patch->dofSection[0]));
  PetscCall(PetscSectionGetStorageSize(patch->dofSection[0], &patch->subspaceOffsets[patch->nsubspaces]));
  patch->totalDofsPerCell = 0;
  for (i = 0; i < patch->nsubspaces; ++i) {
    patch->bs[i]             = 1;
    patch->nodesPerCell[i]   = nodesPerCell[i];
    patch->totalDofsPerCell += nodesPerCell[i];
    PetscCall(PetscMalloc1((cEnd-cStart)*nodesPerCell[i], &patch->cellNodeMap[i]));
    for (j = 0; j < (cEnd-cStart)*nodesPerCell[i]; ++j) patch->cellNodeMap[i][j] = cellNodeMap[i][j];
  }
  PetscCall(DMGetSectionSF(dm, &patch->sectionSF));
  PetscCall(PetscObjectReference((PetscObject) patch->sectionSF));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numGhostBcs, ghostBcNodes, PETSC_COPY_VALUES, &patch->ghostBcNodes));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numGlobalBcs, globalBcNodes, PETSC_COPY_VALUES, &patch->globalBcNodes));
  PetscFunctionReturn(0);
}

/*@C

  PCPatchSetComputeFunction - Set the callback used to compute patch residuals

  Logically collective on PC

  Input Parameters:
+ pc   - The PC
. func - The callback
- ctx  - The user context

  Calling sequence of func:
$   func (PC pc,PetscInt point,Vec x,Vec f,IS cellIS,PetscInt n,const PetscInt* dofsArray,const PetscInt* dofsArrayWithAll,void* ctx)

+  pc               - The PC
.  point            - The point
.  x                - The input solution (not used in linear problems)
.  f                - The patch residual vector
.  cellIS           - An array of the cell numbers
.  n                - The size of dofsArray
.  dofsArray        - The dofmap for the dofs to be solved for
.  dofsArrayWithAll - The dofmap for all dofs on the patch
-  ctx              - The user context

  Level: advanced

  Notes:
  The entries of F (the output residual vector) have been set to zero before the call.

.seealso: PCPatchSetComputeOperator(), PCPatchGetComputeOperator(), PCPatchSetDiscretisationInfo(), PCPatchSetComputeFunctionInteriorFacets()
@*/
PetscErrorCode PCPatchSetComputeFunction(PC pc, PetscErrorCode (*func)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  patch->usercomputef    = func;
  patch->usercomputefctx = ctx;
  PetscFunctionReturn(0);
}

/*@C

  PCPatchSetComputeFunctionInteriorFacets - Set the callback used to compute facet integrals for patch residuals

  Logically collective on PC

  Input Parameters:
+ pc   - The PC
. func - The callback
- ctx  - The user context

  Calling sequence of func:
$   func (PC pc,PetscInt point,Vec x,Vec f,IS facetIS,PetscInt n,const PetscInt* dofsArray,const PetscInt* dofsArrayWithAll,void* ctx)

+  pc               - The PC
.  point            - The point
.  x                - The input solution (not used in linear problems)
.  f                - The patch residual vector
.  facetIS          - An array of the facet numbers
.  n                - The size of dofsArray
.  dofsArray        - The dofmap for the dofs to be solved for
.  dofsArrayWithAll - The dofmap for all dofs on the patch
-  ctx              - The user context

  Level: advanced

  Notes:
  The entries of F (the output residual vector) have been set to zero before the call.

.seealso: PCPatchSetComputeOperator(), PCPatchGetComputeOperator(), PCPatchSetDiscretisationInfo(), PCPatchSetComputeFunction()
@*/
PetscErrorCode PCPatchSetComputeFunctionInteriorFacets(PC pc, PetscErrorCode (*func)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  patch->usercomputefintfacet    = func;
  patch->usercomputefintfacetctx = ctx;
  PetscFunctionReturn(0);
}

/*@C

  PCPatchSetComputeOperator - Set the callback used to compute patch matrices

  Logically collective on PC

  Input Parameters:
+ pc   - The PC
. func - The callback
- ctx  - The user context

  Calling sequence of func:
$   func (PC pc,PetscInt point,Vec x,Mat mat,IS facetIS,PetscInt n,const PetscInt* dofsArray,const PetscInt* dofsArrayWithAll,void* ctx)

+  pc               - The PC
.  point            - The point
.  x                - The input solution (not used in linear problems)
.  mat              - The patch matrix
.  cellIS           - An array of the cell numbers
.  n                - The size of dofsArray
.  dofsArray        - The dofmap for the dofs to be solved for
.  dofsArrayWithAll - The dofmap for all dofs on the patch
-  ctx              - The user context

  Level: advanced

  Notes:
  The matrix entries have been set to zero before the call.

.seealso: PCPatchGetComputeOperator(), PCPatchSetComputeFunction(), PCPatchSetDiscretisationInfo()
@*/
PetscErrorCode PCPatchSetComputeOperator(PC pc, PetscErrorCode (*func)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  patch->usercomputeop    = func;
  patch->usercomputeopctx = ctx;
  PetscFunctionReturn(0);
}

/*@C

  PCPatchSetComputeOperatorInteriorFacets - Set the callback used to compute facet integrals for patch matrices

  Logically collective on PC

  Input Parameters:
+ pc   - The PC
. func - The callback
- ctx  - The user context

  Calling sequence of func:
$   func (PC pc,PetscInt point,Vec x,Mat mat,IS facetIS,PetscInt n,const PetscInt* dofsArray,const PetscInt* dofsArrayWithAll,void* ctx)

+  pc               - The PC
.  point            - The point
.  x                - The input solution (not used in linear problems)
.  mat              - The patch matrix
.  facetIS          - An array of the facet numbers
.  n                - The size of dofsArray
.  dofsArray        - The dofmap for the dofs to be solved for
.  dofsArrayWithAll - The dofmap for all dofs on the patch
-  ctx              - The user context

  Level: advanced

  Notes:
  The matrix entries have been set to zero before the call.

.seealso: PCPatchGetComputeOperator(), PCPatchSetComputeFunction(), PCPatchSetDiscretisationInfo()
@*/
PetscErrorCode PCPatchSetComputeOperatorInteriorFacets(PC pc, PetscErrorCode (*func)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *ctx)
{
  PC_PATCH *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  patch->usercomputeopintfacet    = func;
  patch->usercomputeopintfacetctx = ctx;
  PetscFunctionReturn(0);
}

/* On entry, ht contains the topological entities whose dofs we are responsible for solving for;
   on exit, cht contains all the topological entities we need to compute their residuals.
   In full generality this should incorporate knowledge of the sparsity pattern of the matrix;
   here we assume a standard FE sparsity pattern.*/
/* TODO: Use DMPlexGetAdjacency() */
static PetscErrorCode PCPatchCompleteCellPatch(PC pc, PetscHSetI ht, PetscHSetI cht)
{
  DM             dm, plex;
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscHashIter  hi;
  PetscInt       point;
  PetscInt      *star = NULL, *closure = NULL;
  PetscInt       ignoredim, iStart = 0, iEnd = -1, starSize, closureSize, si, ci;
  PetscInt      *fStar = NULL, *fClosure = NULL;
  PetscInt       fBegin, fEnd, fsi, fci, fStarSize, fClosureSize;

  PetscFunctionBegin;
  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fBegin, &fEnd));
  PetscCall(PCPatchGetIgnoreDim(pc, &ignoredim));
  if (ignoredim >= 0) PetscCall(DMPlexGetDepthStratum(dm, ignoredim, &iStart, &iEnd));
  PetscCall(PetscHSetIClear(cht));
  PetscHashIterBegin(ht, hi);
  while (!PetscHashIterAtEnd(ht, hi)) {

    PetscHashIterGetKey(ht, hi, point);
    PetscHashIterNext(ht, hi);

    /* Loop over all the cells that this point connects to */
    PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
    for (si = 0; si < starSize*2; si += 2) {
      const PetscInt ownedpoint = star[si];
      /* TODO Check for point in cht before running through closure again */
      /* now loop over all entities in the closure of that cell */
      PetscCall(DMPlexGetTransitiveClosure(dm, ownedpoint, PETSC_TRUE, &closureSize, &closure));
      for (ci = 0; ci < closureSize*2; ci += 2) {
        const PetscInt seenpoint = closure[ci];
        if (ignoredim >= 0 && seenpoint >= iStart && seenpoint < iEnd) continue;
        PetscCall(PetscHSetIAdd(cht, seenpoint));
        /* Facet integrals couple dofs across facets, so in that case for each of
         * the facets we need to add all dofs on the other side of the facet to
         * the seen dofs. */
        if (patch->usercomputeopintfacet) {
          if (fBegin <= seenpoint && seenpoint < fEnd) {
            PetscCall(DMPlexGetTransitiveClosure(dm, seenpoint, PETSC_FALSE, &fStarSize, &fStar));
            for (fsi = 0; fsi < fStarSize*2; fsi += 2) {
              PetscCall(DMPlexGetTransitiveClosure(dm, fStar[fsi], PETSC_TRUE, &fClosureSize, &fClosure));
              for (fci = 0; fci < fClosureSize*2; fci += 2) {
                PetscCall(PetscHSetIAdd(cht, fClosure[fci]));
              }
              PetscCall(DMPlexRestoreTransitiveClosure(dm, fStar[fsi], PETSC_TRUE, NULL, &fClosure));
            }
            PetscCall(DMPlexRestoreTransitiveClosure(dm, seenpoint, PETSC_FALSE, NULL, &fStar));
          }
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, ownedpoint, PETSC_TRUE, NULL, &closure));
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, NULL, &star));
  }
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchGetGlobalDofs(PC pc, PetscSection dofSection[], PetscInt f, PetscBool combined, PetscInt p, PetscInt *dof, PetscInt *off)
{
  PetscFunctionBegin;
  if (combined) {
    if (f < 0) {
      if (dof) PetscCall(PetscSectionGetDof(dofSection[0], p, dof));
      if (off) PetscCall(PetscSectionGetOffset(dofSection[0], p, off));
    } else {
      if (dof) PetscCall(PetscSectionGetFieldDof(dofSection[0], p, f, dof));
      if (off) PetscCall(PetscSectionGetFieldOffset(dofSection[0], p, f, off));
    }
  } else {
    if (f < 0) {
      PC_PATCH *patch = (PC_PATCH *) pc->data;
      PetscInt  fdof, g;

      if (dof) {
        *dof = 0;
        for (g = 0; g < patch->nsubspaces; ++g) {
          PetscCall(PetscSectionGetDof(dofSection[g], p, &fdof));
          *dof += fdof;
        }
      }
      if (off) {
        *off = 0;
        for (g = 0; g < patch->nsubspaces; ++g) {
          PetscCall(PetscSectionGetOffset(dofSection[g], p, &fdof));
          *off += fdof;
        }
      }
    } else {
      if (dof) PetscCall(PetscSectionGetDof(dofSection[f], p, dof));
      if (off) PetscCall(PetscSectionGetOffset(dofSection[f], p, off));
    }
  }
  PetscFunctionReturn(0);
}

/* Given a hash table with a set of topological entities (pts), compute the degrees of
   freedom in global concatenated numbering on those entities.
   For Vanka smoothing, this needs to do something special: ignore dofs of the
   constraint subspace on entities that aren't the base entity we're building the patch
   around. */
static PetscErrorCode PCPatchGetPointDofs(PC pc, PetscHSetI pts, PetscHSetI dofs, PetscInt base, PetscHSetI* subspaces_to_exclude)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscHashIter  hi;
  PetscInt       ldof, loff;
  PetscInt       k, p;

  PetscFunctionBegin;
  PetscCall(PetscHSetIClear(dofs));
  for (k = 0; k < patch->nsubspaces; ++k) {
    PetscInt subspaceOffset = patch->subspaceOffsets[k];
    PetscInt bs             = patch->bs[k];
    PetscInt j, l;

    if (subspaces_to_exclude != NULL) {
      PetscBool should_exclude_k = PETSC_FALSE;
      PetscCall(PetscHSetIHas(*subspaces_to_exclude, k, &should_exclude_k));
      if (should_exclude_k) {
        /* only get this subspace dofs at the base entity, not any others */
        PetscCall(PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, base, &ldof, &loff));
        if (0 == ldof) continue;
        for (j = loff; j < ldof + loff; ++j) {
          for (l = 0; l < bs; ++l) {
            PetscInt dof = bs*j + l + subspaceOffset;
            PetscCall(PetscHSetIAdd(dofs, dof));
          }
        }
        continue; /* skip the other dofs of this subspace */
      }
    }

    PetscHashIterBegin(pts, hi);
    while (!PetscHashIterAtEnd(pts, hi)) {
      PetscHashIterGetKey(pts, hi, p);
      PetscHashIterNext(pts, hi);
      PetscCall(PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, p, &ldof, &loff));
      if (0 == ldof) continue;
      for (j = loff; j < ldof + loff; ++j) {
        for (l = 0; l < bs; ++l) {
          PetscInt dof = bs*j + l + subspaceOffset;
          PetscCall(PetscHSetIAdd(dofs, dof));
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

  PetscFunctionBegin;
  PetscCall(PetscHSetIClear(C));
  PetscHashIterBegin(B, hi);
  while (!PetscHashIterAtEnd(B, hi)) {
    PetscHashIterGetKey(B, hi, key);
    PetscHashIterNext(B, hi);
    PetscCall(PetscHSetIHas(A, key, &flg));
    if (!flg) PetscCall(PetscHSetIAdd(C, key));
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
  PetscSection    cellCounts,  pointCounts, intFacetCounts, extFacetCounts;
  PetscInt       *cellsArray, *pointsArray, *intFacetsArray, *extFacetsArray, *intFacetsToPatchCell;
  PetscInt        numCells, numPoints, numIntFacets, numExtFacets;
  const PetscInt *leaves;
  PetscInt        nleaves, pStart, pEnd, cStart, cEnd, vStart, vEnd, fStart, fEnd, v;
  PetscBool       isFiredrake;

  PetscFunctionBegin;
  /* Used to keep track of the cells in the patch. */
  PetscCall(PetscHSetICreate(&ht));
  PetscCall(PetscHSetICreate(&cht));

  PetscCall(PCGetDM(pc, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch PC");
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  if (patch->user_patches) {
    PetscCall(patch->userpatchconstructionop(pc, &patch->npatch, &patch->userIS, &patch->iterationSet, patch->userpatchconstructctx));
    vStart = 0; vEnd = patch->npatch;
  } else if (patch->ctype == PC_PATCH_PARDECOMP) {
    vStart = 0; vEnd = 1;
  } else if (patch->codim < 0) {
    if (patch->dim < 0) PetscCall(DMPlexGetDepthStratum(dm,  0,            &vStart, &vEnd));
    else                PetscCall(DMPlexGetDepthStratum(dm,  patch->dim,   &vStart, &vEnd));
  } else                PetscCall(DMPlexGetHeightStratum(dm, patch->codim, &vStart, &vEnd));
  patch->npatch = vEnd - vStart;

  /* These labels mark the owned points.  We only create patches around points that this process owns. */
  PetscCall(DMHasLabel(dm, "pyop2_ghost", &isFiredrake));
  if (isFiredrake) {
    PetscCall(DMGetLabel(dm, "pyop2_ghost", &ghost));
    PetscCall(DMLabelCreateIndex(ghost, pStart, pEnd));
  } else {
    PetscSF sf;

    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL));
    nleaves = PetscMax(nleaves, 0);
  }

  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &patch->cellCounts));
  PetscCall(PetscObjectSetName((PetscObject) patch->cellCounts, "Patch Cell Layout"));
  cellCounts = patch->cellCounts;
  PetscCall(PetscSectionSetChart(cellCounts, vStart, vEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &patch->pointCounts));
  PetscCall(PetscObjectSetName((PetscObject) patch->pointCounts, "Patch Point Layout"));
  pointCounts = patch->pointCounts;
  PetscCall(PetscSectionSetChart(pointCounts, vStart, vEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &patch->extFacetCounts));
  PetscCall(PetscObjectSetName((PetscObject) patch->extFacetCounts, "Patch Exterior Facet Layout"));
  extFacetCounts = patch->extFacetCounts;
  PetscCall(PetscSectionSetChart(extFacetCounts, vStart, vEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &patch->intFacetCounts));
  PetscCall(PetscObjectSetName((PetscObject) patch->intFacetCounts, "Patch Interior Facet Layout"));
  intFacetCounts = patch->intFacetCounts;
  PetscCall(PetscSectionSetChart(intFacetCounts, vStart, vEnd));
  /* Count cells and points in the patch surrounding each entity */
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIter hi;
    PetscInt       chtSize, loc = -1;
    PetscBool      flg;

    if (!patch->user_patches && patch->ctype != PC_PATCH_PARDECOMP) {
      if (ghost) PetscCall(DMLabelHasPoint(ghost, v, &flg));
      else       {PetscCall(PetscFindInt(v, nleaves, leaves, &loc)); flg = loc >=0 ? PETSC_TRUE : PETSC_FALSE;}
      /* Not an owned entity, don't make a cell patch. */
      if (flg) continue;
    }

    PetscCall(patch->patchconstructop((void *) patch, dm, v, ht));
    PetscCall(PCPatchCompleteCellPatch(pc, ht, cht));
    PetscCall(PetscHSetIGetSize(cht, &chtSize));
    /* empty patch, continue */
    if (chtSize == 0) continue;

    /* safe because size(cht) > 0 from above */
    PetscHashIterBegin(cht, hi);
    while (!PetscHashIterAtEnd(cht, hi)) {
      PetscInt point, pdof;

      PetscHashIterGetKey(cht, hi, point);
      if (fStart <= point && point < fEnd) {
        const PetscInt *support;
        PetscInt supportSize, p;
        PetscBool interior = PETSC_TRUE;
        PetscCall(DMPlexGetSupport(dm, point, &support));
        PetscCall(DMPlexGetSupportSize(dm, point, &supportSize));
        if (supportSize == 1) {
          interior = PETSC_FALSE;
        } else {
          for (p = 0; p < supportSize; p++) {
            PetscBool found;
            /* FIXME: can I do this while iterating over cht? */
            PetscCall(PetscHSetIHas(cht, support[p], &found));
            if (!found) {
              interior = PETSC_FALSE;
              break;
            }
          }
        }
        if (interior) {
          PetscCall(PetscSectionAddDof(intFacetCounts, v, 1));
        } else {
          PetscCall(PetscSectionAddDof(extFacetCounts, v, 1));
        }
      }
      PetscCall(PCPatchGetGlobalDofs(pc, patch->dofSection, -1, patch->combined, point, &pdof, NULL));
      if (pdof)                            PetscCall(PetscSectionAddDof(pointCounts, v, 1));
      if (point >= cStart && point < cEnd) PetscCall(PetscSectionAddDof(cellCounts, v, 1));
      PetscHashIterNext(cht, hi);
    }
  }
  if (isFiredrake) PetscCall(DMLabelDestroyIndex(ghost));

  PetscCall(PetscSectionSetUp(cellCounts));
  PetscCall(PetscSectionGetStorageSize(cellCounts, &numCells));
  PetscCall(PetscMalloc1(numCells, &cellsArray));
  PetscCall(PetscSectionSetUp(pointCounts));
  PetscCall(PetscSectionGetStorageSize(pointCounts, &numPoints));
  PetscCall(PetscMalloc1(numPoints, &pointsArray));

  PetscCall(PetscSectionSetUp(intFacetCounts));
  PetscCall(PetscSectionSetUp(extFacetCounts));
  PetscCall(PetscSectionGetStorageSize(intFacetCounts, &numIntFacets));
  PetscCall(PetscSectionGetStorageSize(extFacetCounts, &numExtFacets));
  PetscCall(PetscMalloc1(numIntFacets, &intFacetsArray));
  PetscCall(PetscMalloc1(numIntFacets*2, &intFacetsToPatchCell));
  PetscCall(PetscMalloc1(numExtFacets, &extFacetsArray));

  /* Now that we know how much space we need, run through again and actually remember the cells. */
  for (v = vStart; v < vEnd; v++) {
    PetscHashIter hi;
    PetscInt       dof, off, cdof, coff, efdof, efoff, ifdof, ifoff, pdof, n = 0, cn = 0, ifn = 0, efn = 0;

    PetscCall(PetscSectionGetDof(pointCounts, v, &dof));
    PetscCall(PetscSectionGetOffset(pointCounts, v, &off));
    PetscCall(PetscSectionGetDof(cellCounts, v, &cdof));
    PetscCall(PetscSectionGetOffset(cellCounts, v, &coff));
    PetscCall(PetscSectionGetDof(intFacetCounts, v, &ifdof));
    PetscCall(PetscSectionGetOffset(intFacetCounts, v, &ifoff));
    PetscCall(PetscSectionGetDof(extFacetCounts, v, &efdof));
    PetscCall(PetscSectionGetOffset(extFacetCounts, v, &efoff));
    if (dof <= 0) continue;
    PetscCall(patch->patchconstructop((void *) patch, dm, v, ht));
    PetscCall(PCPatchCompleteCellPatch(pc, ht, cht));
    PetscHashIterBegin(cht, hi);
    while (!PetscHashIterAtEnd(cht, hi)) {
      PetscInt point;

      PetscHashIterGetKey(cht, hi, point);
      if (fStart <= point && point < fEnd) {
        const PetscInt *support;
        PetscInt       supportSize, p;
        PetscBool      interior = PETSC_TRUE;
        PetscCall(DMPlexGetSupport(dm, point, &support));
        PetscCall(DMPlexGetSupportSize(dm, point, &supportSize));
        if (supportSize == 1) {
          interior = PETSC_FALSE;
        } else {
          for (p = 0; p < supportSize; p++) {
            PetscBool found;
            /* FIXME: can I do this while iterating over cht? */
            PetscCall(PetscHSetIHas(cht, support[p], &found));
            if (!found) {
              interior = PETSC_FALSE;
              break;
            }
          }
        }
        if (interior) {
          intFacetsToPatchCell[2*(ifoff + ifn)] = support[0];
          intFacetsToPatchCell[2*(ifoff + ifn) + 1] = support[1];
          intFacetsArray[ifoff + ifn++] = point;
        } else {
          extFacetsArray[efoff + efn++] = point;
        }
      }
      PetscCall(PCPatchGetGlobalDofs(pc, patch->dofSection, -1, patch->combined, point, &pdof, NULL));
      if (pdof)                            {pointsArray[off + n++] = point;}
      if (point >= cStart && point < cEnd) {cellsArray[coff + cn++] = point;}
      PetscHashIterNext(cht, hi);
    }
    PetscCheckFalse(ifn != ifdof,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of interior facets in patch %D is %D, but should be %D", v, ifn, ifdof);
    PetscCheckFalse(efn != efdof,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of exterior facets in patch %D is %D, but should be %D", v, efn, efdof);
    PetscCheckFalse(cn != cdof,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of cells in patch %D is %D, but should be %D", v, cn, cdof);
    PetscCheckFalse(n  != dof,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of points in patch %D is %D, but should be %D", v, n, dof);

    for (ifn = 0; ifn < ifdof; ifn++) {
      PetscInt  cell0 = intFacetsToPatchCell[2*(ifoff + ifn)];
      PetscInt  cell1 = intFacetsToPatchCell[2*(ifoff + ifn) + 1];
      PetscBool found0 = PETSC_FALSE, found1 = PETSC_FALSE;
      for (n = 0; n < cdof; n++) {
        if (!found0 && cell0 == cellsArray[coff + n]) {
          intFacetsToPatchCell[2*(ifoff + ifn)] = n;
          found0 = PETSC_TRUE;
        }
        if (!found1 && cell1 == cellsArray[coff + n]) {
          intFacetsToPatchCell[2*(ifoff + ifn) + 1] = n;
          found1 = PETSC_TRUE;
        }
        if (found0 && found1) break;
      }
      PetscCheck(found0 && found1,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Didn't manage to find local point numbers for facet support");
    }
  }
  PetscCall(PetscHSetIDestroy(&ht));
  PetscCall(PetscHSetIDestroy(&cht));

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numCells,  cellsArray,  PETSC_OWN_POINTER, &patch->cells));
  PetscCall(PetscObjectSetName((PetscObject) patch->cells,  "Patch Cells"));
  if (patch->viewCells) {
    PetscCall(ObjectView((PetscObject) patch->cellCounts, patch->viewerCells, patch->formatCells));
    PetscCall(ObjectView((PetscObject) patch->cells,      patch->viewerCells, patch->formatCells));
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numIntFacets,  intFacetsArray,  PETSC_OWN_POINTER, &patch->intFacets));
  PetscCall(PetscObjectSetName((PetscObject) patch->intFacets,  "Patch Interior Facets"));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 2*numIntFacets, intFacetsToPatchCell, PETSC_OWN_POINTER, &patch->intFacetsToPatchCell));
  PetscCall(PetscObjectSetName((PetscObject) patch->intFacetsToPatchCell,  "Patch Interior Facets local support"));
  if (patch->viewIntFacets) {
    PetscCall(ObjectView((PetscObject) patch->intFacetCounts,       patch->viewerIntFacets, patch->formatIntFacets));
    PetscCall(ObjectView((PetscObject) patch->intFacets,            patch->viewerIntFacets, patch->formatIntFacets));
    PetscCall(ObjectView((PetscObject) patch->intFacetsToPatchCell, patch->viewerIntFacets, patch->formatIntFacets));
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numExtFacets,  extFacetsArray,  PETSC_OWN_POINTER, &patch->extFacets));
  PetscCall(PetscObjectSetName((PetscObject) patch->extFacets,  "Patch Exterior Facets"));
  if (patch->viewExtFacets) {
    PetscCall(ObjectView((PetscObject) patch->extFacetCounts, patch->viewerExtFacets, patch->formatExtFacets));
    PetscCall(ObjectView((PetscObject) patch->extFacets,      patch->viewerExtFacets, patch->formatExtFacets));
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numPoints, pointsArray, PETSC_OWN_POINTER, &patch->points));
  PetscCall(PetscObjectSetName((PetscObject) patch->points, "Patch Points"));
  if (patch->viewPoints) {
    PetscCall(ObjectView((PetscObject) patch->pointCounts, patch->viewerPoints, patch->formatPoints));
    PetscCall(ObjectView((PetscObject) patch->points,      patch->viewerPoints, patch->formatPoints));
  }
  PetscCall(DMDestroy(&dm));
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
  PetscSection    gtolCounts, gtolCountsWithArtificial = NULL, gtolCountsWithAll = NULL;
  IS              cells           = patch->cells;
  IS              points          = patch->points;
  PetscSection    cellNumbering   = patch->cellNumbering;
  PetscInt        Nf              = patch->nsubspaces;
  PetscInt        numCells, numPoints;
  PetscInt        numDofs;
  PetscInt        numGlobalDofs, numGlobalDofsWithArtificial, numGlobalDofsWithAll;
  PetscInt        totalDofsPerCell = patch->totalDofsPerCell;
  PetscInt        vStart, vEnd, v;
  const PetscInt *cellsArray, *pointsArray;
  PetscInt       *newCellsArray   = NULL;
  PetscInt       *dofsArray       = NULL;
  PetscInt       *dofsArrayWithArtificial = NULL;
  PetscInt       *dofsArrayWithAll = NULL;
  PetscInt       *offsArray       = NULL;
  PetscInt       *offsArrayWithArtificial = NULL;
  PetscInt       *offsArrayWithAll = NULL;
  PetscInt       *asmArray        = NULL;
  PetscInt       *asmArrayWithArtificial = NULL;
  PetscInt       *asmArrayWithAll = NULL;
  PetscInt       *globalDofsArray = NULL;
  PetscInt       *globalDofsArrayWithArtificial = NULL;
  PetscInt       *globalDofsArrayWithAll = NULL;
  PetscInt        globalIndex     = 0;
  PetscInt        key             = 0;
  PetscInt        asmKey          = 0;
  DM              dm              = NULL, plex;
  const PetscInt *bcNodes         = NULL;
  PetscHMapI      ht;
  PetscHMapI      htWithArtificial;
  PetscHMapI      htWithAll;
  PetscHSetI      globalBcs;
  PetscInt        numBcs;
  PetscHSetI      ownedpts, seenpts, owneddofs, seendofs, artificialbcs;
  PetscInt        pStart, pEnd, p, i;
  char            option[PETSC_MAX_PATH_LEN];
  PetscBool       isNonlinear;

  PetscFunctionBegin;

  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  /* dofcounts section is cellcounts section * dofPerCell */
  PetscCall(PetscSectionGetStorageSize(cellCounts, &numCells));
  PetscCall(PetscSectionGetStorageSize(patch->pointCounts, &numPoints));
  numDofs = numCells * totalDofsPerCell;
  PetscCall(PetscMalloc1(numDofs, &dofsArray));
  PetscCall(PetscMalloc1(numPoints*Nf, &offsArray));
  PetscCall(PetscMalloc1(numDofs, &asmArray));
  PetscCall(PetscMalloc1(numCells, &newCellsArray));
  PetscCall(PetscSectionGetChart(cellCounts, &vStart, &vEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCounts));
  gtolCounts = patch->gtolCounts;
  PetscCall(PetscSectionSetChart(gtolCounts, vStart, vEnd));
  PetscCall(PetscObjectSetName((PetscObject) patch->gtolCounts, "Patch Global Index Section"));

  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    PetscCall(PetscMalloc1(numPoints*Nf, &offsArrayWithArtificial));
    PetscCall(PetscMalloc1(numDofs, &asmArrayWithArtificial));
    PetscCall(PetscMalloc1(numDofs, &dofsArrayWithArtificial));
    PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCountsWithArtificial));
    gtolCountsWithArtificial = patch->gtolCountsWithArtificial;
    PetscCall(PetscSectionSetChart(gtolCountsWithArtificial, vStart, vEnd));
    PetscCall(PetscObjectSetName((PetscObject) patch->gtolCountsWithArtificial, "Patch Global Index Section Including Artificial BCs"));
  }

  isNonlinear = patch->isNonlinear;
  if (isNonlinear) {
    PetscCall(PetscMalloc1(numPoints*Nf, &offsArrayWithAll));
    PetscCall(PetscMalloc1(numDofs, &asmArrayWithAll));
    PetscCall(PetscMalloc1(numDofs, &dofsArrayWithAll));
    PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCountsWithAll));
    gtolCountsWithAll = patch->gtolCountsWithAll;
    PetscCall(PetscSectionSetChart(gtolCountsWithAll, vStart, vEnd));
    PetscCall(PetscObjectSetName((PetscObject) patch->gtolCountsWithAll, "Patch Global Index Section Including All BCs"));
  }

  /* Outside the patch loop, get the dofs that are globally-enforced Dirichlet
   conditions */
  PetscCall(PetscHSetICreate(&globalBcs));
  PetscCall(ISGetIndices(patch->ghostBcNodes, &bcNodes));
  PetscCall(ISGetSize(patch->ghostBcNodes, &numBcs));
  for (i = 0; i < numBcs; ++i) {
    PetscCall(PetscHSetIAdd(globalBcs, bcNodes[i])); /* these are already in concatenated numbering */
  }
  PetscCall(ISRestoreIndices(patch->ghostBcNodes, &bcNodes));
  PetscCall(ISDestroy(&patch->ghostBcNodes)); /* memory optimisation */

  /* Hash tables for artificial BC construction */
  PetscCall(PetscHSetICreate(&ownedpts));
  PetscCall(PetscHSetICreate(&seenpts));
  PetscCall(PetscHSetICreate(&owneddofs));
  PetscCall(PetscHSetICreate(&seendofs));
  PetscCall(PetscHSetICreate(&artificialbcs));

  PetscCall(ISGetIndices(cells, &cellsArray));
  PetscCall(ISGetIndices(points, &pointsArray));
  PetscCall(PetscHMapICreate(&ht));
  PetscCall(PetscHMapICreate(&htWithArtificial));
  PetscCall(PetscHMapICreate(&htWithAll));
  for (v = vStart; v < vEnd; ++v) {
    PetscInt localIndex = 0;
    PetscInt localIndexWithArtificial = 0;
    PetscInt localIndexWithAll = 0;
    PetscInt dof, off, i, j, k, l;

    PetscCall(PetscHMapIClear(ht));
    PetscCall(PetscHMapIClear(htWithArtificial));
    PetscCall(PetscHMapIClear(htWithAll));
    PetscCall(PetscSectionGetDof(cellCounts, v, &dof));
    PetscCall(PetscSectionGetOffset(cellCounts, v, &off));
    if (dof <= 0) continue;

    /* Calculate the global numbers of the artificial BC dofs here first */
    PetscCall(patch->patchconstructop((void*)patch, dm, v, ownedpts));
    PetscCall(PCPatchCompleteCellPatch(pc, ownedpts, seenpts));
    PetscCall(PCPatchGetPointDofs(pc, ownedpts, owneddofs, v, &patch->subspaces_to_exclude));
    PetscCall(PCPatchGetPointDofs(pc, seenpts, seendofs, v, NULL));
    PetscCall(PCPatchComputeSetDifference_Private(owneddofs, seendofs, artificialbcs));
    if (patch->viewPatches) {
      PetscHSetI    globalbcdofs;
      PetscHashIter hi;
      MPI_Comm      comm = PetscObjectComm((PetscObject)pc);

      PetscCall(PetscHSetICreate(&globalbcdofs));
      PetscCall(PetscSynchronizedPrintf(comm, "Patch %d: owned dofs:\n", v));
      PetscHashIterBegin(owneddofs, hi);
      while (!PetscHashIterAtEnd(owneddofs, hi)) {
        PetscInt globalDof;

        PetscHashIterGetKey(owneddofs, hi, globalDof);
        PetscHashIterNext(owneddofs, hi);
        PetscCall(PetscSynchronizedPrintf(comm, "%d ", globalDof));
      }
      PetscCall(PetscSynchronizedPrintf(comm, "\n"));
      PetscCall(PetscSynchronizedPrintf(comm, "Patch %d: seen dofs:\n", v));
      PetscHashIterBegin(seendofs, hi);
      while (!PetscHashIterAtEnd(seendofs, hi)) {
        PetscInt globalDof;
        PetscBool flg;

        PetscHashIterGetKey(seendofs, hi, globalDof);
        PetscHashIterNext(seendofs, hi);
        PetscCall(PetscSynchronizedPrintf(comm, "%d ", globalDof));

        PetscCall(PetscHSetIHas(globalBcs, globalDof, &flg));
        if (flg) PetscCall(PetscHSetIAdd(globalbcdofs, globalDof));
      }
      PetscCall(PetscSynchronizedPrintf(comm, "\n"));
      PetscCall(PetscSynchronizedPrintf(comm, "Patch %d: global BCs:\n", v));
      PetscCall(PetscHSetIGetSize(globalbcdofs, &numBcs));
      if (numBcs > 0) {
        PetscHashIterBegin(globalbcdofs, hi);
        while (!PetscHashIterAtEnd(globalbcdofs, hi)) {
          PetscInt globalDof;
          PetscHashIterGetKey(globalbcdofs, hi, globalDof);
          PetscHashIterNext(globalbcdofs, hi);
          PetscCall(PetscSynchronizedPrintf(comm, "%d ", globalDof));
        }
      }
      PetscCall(PetscSynchronizedPrintf(comm, "\n"));
      PetscCall(PetscSynchronizedPrintf(comm, "Patch %d: artificial BCs:\n", v));
      PetscCall(PetscHSetIGetSize(artificialbcs, &numBcs));
      if (numBcs > 0) {
        PetscHashIterBegin(artificialbcs, hi);
        while (!PetscHashIterAtEnd(artificialbcs, hi)) {
          PetscInt globalDof;
          PetscHashIterGetKey(artificialbcs, hi, globalDof);
          PetscHashIterNext(artificialbcs, hi);
          PetscCall(PetscSynchronizedPrintf(comm, "%d ", globalDof));
        }
      }
      PetscCall(PetscSynchronizedPrintf(comm, "\n\n"));
      PetscCall(PetscHSetIDestroy(&globalbcdofs));
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
          PetscCall(PetscSectionGetDof(cellNumbering, c, &cell));
          PetscCheckFalse(cell <= 0,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_OUTOFRANGE, "Cell %D doesn't appear in cell numbering map", c);
          PetscCall(PetscSectionGetOffset(cellNumbering, c, &cell));
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
            PetscCall(PetscHSetIHas(globalBcs, globalDof + l, &isGlobalBcDof));
            PetscCall(PetscHSetIHas(artificialbcs, globalDof + l, &isArtificialBcDof));

            /* if it's either, don't ever give it a local dof number */
            if (isGlobalBcDof || isArtificialBcDof) {
              dofsArray[globalIndex] = -1; /* don't use this in assembly in this patch */
            } else {
              PetscCall(PetscHMapIGet(ht, globalDof + l, &localDof));
              if (localDof == -1) {
                localDof = localIndex++;
                PetscCall(PetscHMapISet(ht, globalDof + l, localDof));
              }
              PetscCheckFalse(globalIndex >= numDofs,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
              /* And store. */
              dofsArray[globalIndex] = localDof;
            }

            if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
              if (isGlobalBcDof) {
                dofsArrayWithArtificial[globalIndex] = -1; /* don't use this in assembly in this patch */
              } else {
                PetscCall(PetscHMapIGet(htWithArtificial, globalDof + l, &localDof));
                if (localDof == -1) {
                  localDof = localIndexWithArtificial++;
                  PetscCall(PetscHMapISet(htWithArtificial, globalDof + l, localDof));
                }
                PetscCheckFalse(globalIndex >= numDofs,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
                /* And store.*/
                dofsArrayWithArtificial[globalIndex] = localDof;
              }
            }

            if (isNonlinear) {
              /* Build the dofmap for the function space with _all_ dofs,
                 including those in any kind of boundary condition */
              PetscCall(PetscHMapIGet(htWithAll, globalDof + l, &localDof));
              if (localDof == -1) {
                localDof = localIndexWithAll++;
                PetscCall(PetscHMapISet(htWithAll, globalDof + l, localDof));
              }
              PetscCheckFalse(globalIndex >= numDofs,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
              /* And store.*/
              dofsArrayWithAll[globalIndex] = localDof;
            }
            globalIndex++;
          }
        }
      }
    }
     /*How many local dofs in this patch? */
   if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
     PetscCall(PetscHMapIGetSize(htWithArtificial, &dof));
     PetscCall(PetscSectionSetDof(gtolCountsWithArtificial, v, dof));
   }
   if (isNonlinear) {
     PetscCall(PetscHMapIGetSize(htWithAll, &dof));
     PetscCall(PetscSectionSetDof(gtolCountsWithAll, v, dof));
   }
    PetscCall(PetscHMapIGetSize(ht, &dof));
    PetscCall(PetscSectionSetDof(gtolCounts, v, dof));
  }

  PetscCall(DMDestroy(&dm));
  PetscCheckFalse(globalIndex != numDofs,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Expected number of dofs (%d) doesn't match found number (%d)", numDofs, globalIndex);
  PetscCall(PetscSectionSetUp(gtolCounts));
  PetscCall(PetscSectionGetStorageSize(gtolCounts, &numGlobalDofs));
  PetscCall(PetscMalloc1(numGlobalDofs, &globalDofsArray));

  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    PetscCall(PetscSectionSetUp(gtolCountsWithArtificial));
    PetscCall(PetscSectionGetStorageSize(gtolCountsWithArtificial, &numGlobalDofsWithArtificial));
    PetscCall(PetscMalloc1(numGlobalDofsWithArtificial, &globalDofsArrayWithArtificial));
  }
  if (isNonlinear) {
    PetscCall(PetscSectionSetUp(gtolCountsWithAll));
    PetscCall(PetscSectionGetStorageSize(gtolCountsWithAll, &numGlobalDofsWithAll));
    PetscCall(PetscMalloc1(numGlobalDofsWithAll, &globalDofsArrayWithAll));
  }
  /* Now populate the global to local map.  This could be merged into the above loop if we were willing to deal with reallocs. */
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIter hi;
    PetscInt      dof, off, Np, ooff, i, j, k, l;

    PetscCall(PetscHMapIClear(ht));
    PetscCall(PetscHMapIClear(htWithArtificial));
    PetscCall(PetscHMapIClear(htWithAll));
    PetscCall(PetscSectionGetDof(cellCounts, v, &dof));
    PetscCall(PetscSectionGetOffset(cellCounts, v, &off));
    PetscCall(PetscSectionGetDof(pointCounts, v, &Np));
    PetscCall(PetscSectionGetOffset(pointCounts, v, &ooff));
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

        if (cellNumbering) PetscCall(PetscSectionGetOffset(cellNumbering, c, &cell));
        for (j = 0; j < nodesPerCell; ++j) {
          for (l = 0; l < bs; ++l) {
            const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + l + subspaceOffset;
            const PetscInt localDof  = dofsArray[key];
            if (localDof >= 0) PetscCall(PetscHMapISet(ht, globalDof, localDof));
            if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
              const PetscInt localDofWithArtificial = dofsArrayWithArtificial[key];
              if (localDofWithArtificial >= 0) {
                PetscCall(PetscHMapISet(htWithArtificial, globalDof, localDofWithArtificial));
              }
            }
            if (isNonlinear) {
              const PetscInt localDofWithAll = dofsArrayWithAll[key];
              if (localDofWithAll >= 0) {
                PetscCall(PetscHMapISet(htWithAll, globalDof, localDofWithAll));
              }
            }
            key++;
          }
        }
      }

      /* Shove it in the output data structure. */
      PetscCall(PetscSectionGetOffset(gtolCounts, v, &goff));
      PetscHashIterBegin(ht, hi);
      while (!PetscHashIterAtEnd(ht, hi)) {
        PetscInt globalDof, localDof;

        PetscHashIterGetKey(ht, hi, globalDof);
        PetscHashIterGetVal(ht, hi, localDof);
        if (globalDof >= 0) globalDofsArray[goff + localDof] = globalDof;
        PetscHashIterNext(ht, hi);
      }

      if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
        PetscCall(PetscSectionGetOffset(gtolCountsWithArtificial, v, &goff));
        PetscHashIterBegin(htWithArtificial, hi);
        while (!PetscHashIterAtEnd(htWithArtificial, hi)) {
          PetscInt globalDof, localDof;
          PetscHashIterGetKey(htWithArtificial, hi, globalDof);
          PetscHashIterGetVal(htWithArtificial, hi, localDof);
          if (globalDof >= 0) globalDofsArrayWithArtificial[goff + localDof] = globalDof;
          PetscHashIterNext(htWithArtificial, hi);
        }
      }
      if (isNonlinear) {
        PetscCall(PetscSectionGetOffset(gtolCountsWithAll, v, &goff));
        PetscHashIterBegin(htWithAll, hi);
        while (!PetscHashIterAtEnd(htWithAll, hi)) {
          PetscInt globalDof, localDof;
          PetscHashIterGetKey(htWithAll, hi, globalDof);
          PetscHashIterGetVal(htWithAll, hi, localDof);
          if (globalDof >= 0) globalDofsArrayWithAll[goff + localDof] = globalDof;
          PetscHashIterNext(htWithAll, hi);
        }
      }

      for (p = 0; p < Np; ++p) {
        const PetscInt point = pointsArray[ooff + p];
        PetscInt       globalDof, localDof;

        PetscCall(PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, point, NULL, &globalDof));
        PetscCall(PetscHMapIGet(ht, globalDof, &localDof));
        offsArray[(ooff + p)*Nf + k] = localDof;
        if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
          PetscCall(PetscHMapIGet(htWithArtificial, globalDof, &localDof));
          offsArrayWithArtificial[(ooff + p)*Nf + k] = localDof;
        }
        if (isNonlinear) {
          PetscCall(PetscHMapIGet(htWithAll, globalDof, &localDof));
          offsArrayWithAll[(ooff + p)*Nf + k] = localDof;
        }
      }
    }

    PetscCall(PetscHSetIDestroy(&globalBcs));
    PetscCall(PetscHSetIDestroy(&ownedpts));
    PetscCall(PetscHSetIDestroy(&seenpts));
    PetscCall(PetscHSetIDestroy(&owneddofs));
    PetscCall(PetscHSetIDestroy(&seendofs));
    PetscCall(PetscHSetIDestroy(&artificialbcs));

      /* At this point, we have a hash table ht built that maps globalDof -> localDof.
     We need to create the dof table laid out cellwise first, then by subspace,
     as the assembler assembles cell-wise and we need to stuff the different
     contributions of the different function spaces to the right places. So we loop
     over cells, then over subspaces. */
    if (patch->nsubspaces > 1) { /* for nsubspaces = 1, data we need is already in dofsArray */
      for (i = off; i < off + dof; ++i) {
        const PetscInt c    = cellsArray[i];
        PetscInt       cell = c;

        if (cellNumbering) PetscCall(PetscSectionGetOffset(cellNumbering, c, &cell));
        for (k = 0; k < patch->nsubspaces; ++k) {
          const PetscInt *cellNodeMap    = patch->cellNodeMap[k];
          PetscInt        nodesPerCell   = patch->nodesPerCell[k];
          PetscInt        subspaceOffset = patch->subspaceOffsets[k];
          PetscInt        bs             = patch->bs[k];

          for (j = 0; j < nodesPerCell; ++j) {
            for (l = 0; l < bs; ++l) {
              const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + l + subspaceOffset;
              PetscInt       localDof;

              PetscCall(PetscHMapIGet(ht, globalDof, &localDof));
              /* If it's not in the hash table, i.e. is a BC dof,
               then the PetscHSetIMap above gives -1, which matches
               exactly the convention for PETSc's matrix assembly to
               ignore the dof. So we don't need to do anything here */
              asmArray[asmKey] = localDof;
              if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
                PetscCall(PetscHMapIGet(htWithArtificial, globalDof, &localDof));
                asmArrayWithArtificial[asmKey] = localDof;
              }
              if (isNonlinear) {
                PetscCall(PetscHMapIGet(htWithAll, globalDof, &localDof));
                asmArrayWithAll[asmKey] = localDof;
              }
              asmKey++;
            }
          }
        }
      }
    }
  }
  if (1 == patch->nsubspaces) {
    PetscCall(PetscArraycpy(asmArray, dofsArray, numDofs));
    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
      PetscCall(PetscArraycpy(asmArrayWithArtificial, dofsArrayWithArtificial, numDofs));
    }
    if (isNonlinear) {
      PetscCall(PetscArraycpy(asmArrayWithAll, dofsArrayWithAll, numDofs));
    }
  }

  PetscCall(PetscHMapIDestroy(&ht));
  PetscCall(PetscHMapIDestroy(&htWithArtificial));
  PetscCall(PetscHMapIDestroy(&htWithAll));
  PetscCall(ISRestoreIndices(cells, &cellsArray));
  PetscCall(ISRestoreIndices(points, &pointsArray));
  PetscCall(PetscFree(dofsArray));
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    PetscCall(PetscFree(dofsArrayWithArtificial));
  }
  if (isNonlinear) {
    PetscCall(PetscFree(dofsArrayWithAll));
  }
  /* Create placeholder section for map from points to patch dofs */
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &patch->patchSection));
  PetscCall(PetscSectionSetNumFields(patch->patchSection, patch->nsubspaces));
  if (patch->combined) {
    PetscInt numFields;
    PetscCall(PetscSectionGetNumFields(patch->dofSection[0], &numFields));
    PetscCheckFalse(numFields != patch->nsubspaces,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Mismatch between number of section fields %D and number of subspaces %D", numFields, patch->nsubspaces);
    PetscCall(PetscSectionGetChart(patch->dofSection[0], &pStart, &pEnd));
    PetscCall(PetscSectionSetChart(patch->patchSection, pStart, pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, fdof, f;

      PetscCall(PetscSectionGetDof(patch->dofSection[0], p, &dof));
      PetscCall(PetscSectionSetDof(patch->patchSection, p, dof));
      for (f = 0; f < patch->nsubspaces; ++f) {
        PetscCall(PetscSectionGetFieldDof(patch->dofSection[0], p, f, &fdof));
        PetscCall(PetscSectionSetFieldDof(patch->patchSection, p, f, fdof));
      }
    }
  } else {
    PetscInt pStartf, pEndf, f;
    pStart = PETSC_MAX_INT;
    pEnd = PETSC_MIN_INT;
    for (f = 0; f < patch->nsubspaces; ++f) {
      PetscCall(PetscSectionGetChart(patch->dofSection[f], &pStartf, &pEndf));
      pStart = PetscMin(pStart, pStartf);
      pEnd = PetscMax(pEnd, pEndf);
    }
    PetscCall(PetscSectionSetChart(patch->patchSection, pStart, pEnd));
    for (f = 0; f < patch->nsubspaces; ++f) {
      PetscCall(PetscSectionGetChart(patch->dofSection[f], &pStartf, &pEndf));
      for (p = pStartf; p < pEndf; ++p) {
        PetscInt fdof;
        PetscCall(PetscSectionGetDof(patch->dofSection[f], p, &fdof));
        PetscCall(PetscSectionAddDof(patch->patchSection, p, fdof));
        PetscCall(PetscSectionSetFieldDof(patch->patchSection, p, f, fdof));
      }
    }
  }
  PetscCall(PetscSectionSetUp(patch->patchSection));
  PetscCall(PetscSectionSetUseFieldOffsets(patch->patchSection, PETSC_TRUE));
  /* Replace cell indices with firedrake-numbered ones. */
  PetscCall(ISGeneralSetIndices(cells, numCells, (const PetscInt *) newCellsArray, PETSC_OWN_POINTER));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofs, globalDofsArray, PETSC_OWN_POINTER, &patch->gtol));
  PetscCall(PetscObjectSetName((PetscObject) patch->gtol, "Global Indices"));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_g2l_view", patch->classname));
  PetscCall(PetscSectionViewFromOptions(patch->gtolCounts, (PetscObject) pc, option));
  PetscCall(ISViewFromOptions(patch->gtol, (PetscObject) pc, option));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArray, PETSC_OWN_POINTER, &patch->dofs));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArray, PETSC_OWN_POINTER, &patch->offs));
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofsWithArtificial, globalDofsArrayWithArtificial, PETSC_OWN_POINTER, &patch->gtolWithArtificial));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArrayWithArtificial, PETSC_OWN_POINTER, &patch->dofsWithArtificial));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArrayWithArtificial, PETSC_OWN_POINTER, &patch->offsWithArtificial));
  }
  if (isNonlinear) {
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofsWithAll, globalDofsArrayWithAll, PETSC_OWN_POINTER, &patch->gtolWithAll));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArrayWithAll, PETSC_OWN_POINTER, &patch->dofsWithAll));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArrayWithAll, PETSC_OWN_POINTER, &patch->offsWithAll));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchCreateMatrix_Private(PC pc, PetscInt point, Mat *mat, PetscBool withArtificial)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscBool      flg;
  PetscInt       csize, rsize;
  const char    *prefix = NULL;

  PetscFunctionBegin;
  if (withArtificial) {
    /* would be nice if we could create a rectangular matrix of size numDofsWithArtificial x numDofs here */
    PetscInt pStart;
    PetscCall(PetscSectionGetChart(patch->gtolCountsWithArtificial, &pStart, NULL));
    PetscCall(PetscSectionGetDof(patch->gtolCountsWithArtificial, point + pStart, &rsize));
    csize = rsize;
  } else {
    PetscInt pStart;
    PetscCall(PetscSectionGetChart(patch->gtolCounts, &pStart, NULL));
    PetscCall(PetscSectionGetDof(patch->gtolCounts, point + pStart, &rsize));
    csize = rsize;
  }

  PetscCall(MatCreate(PETSC_COMM_SELF, mat));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(MatSetOptionsPrefix(*mat, prefix));
  PetscCall(MatAppendOptionsPrefix(*mat, "pc_patch_sub_"));
  if (patch->sub_mat_type)       PetscCall(MatSetType(*mat, patch->sub_mat_type));
  else if (!patch->sub_mat_type) PetscCall(MatSetType(*mat, MATDENSE));
  PetscCall(MatSetSizes(*mat, rsize, csize, rsize, csize));
  PetscCall(PetscObjectTypeCompare((PetscObject) *mat, MATDENSE, &flg));
  if (!flg) PetscCall(PetscObjectTypeCompare((PetscObject)*mat, MATSEQDENSE, &flg));
  /* Sparse patch matrices */
  if (!flg) {
    PetscBT         bt;
    PetscInt       *dnnz      = NULL;
    const PetscInt *dofsArray = NULL;
    PetscInt        pStart, pEnd, ncell, offset, c, i, j;

    if (withArtificial) {
      PetscCall(ISGetIndices(patch->dofsWithArtificial, &dofsArray));
    } else {
      PetscCall(ISGetIndices(patch->dofs, &dofsArray));
    }
    PetscCall(PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd));
    point += pStart;
    PetscCheckFalse(point >= pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Operator point %D not in [%D, %D)", point, pStart, pEnd);
    PetscCall(PetscSectionGetDof(patch->cellCounts, point, &ncell));
    PetscCall(PetscSectionGetOffset(patch->cellCounts, point, &offset));
    PetscCall(PetscLogEventBegin(PC_Patch_Prealloc, pc, 0, 0, 0));
    /* A PetscBT uses N^2 bits to store the sparsity pattern on a
     * patch. This is probably OK if the patches are not too big,
     * but uses too much memory. We therefore switch based on rsize. */
    if (rsize < 3000) { /* FIXME: I picked this switch value out of my hat */
      PetscScalar *zeroes;
      PetscInt rows;

      PetscCall(PetscCalloc1(rsize, &dnnz));
      PetscCall(PetscBTCreate(rsize*rsize, &bt));
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

      if (patch->usercomputeopintfacet) {
        const PetscInt *intFacetsArray = NULL;
        PetscInt        i, numIntFacets, intFacetOffset;
        const PetscInt *facetCells = NULL;

        PetscCall(PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets));
        PetscCall(PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset));
        PetscCall(ISGetIndices(patch->intFacetsToPatchCell, &facetCells));
        PetscCall(ISGetIndices(patch->intFacets, &intFacetsArray));
        for (i = 0; i < numIntFacets; i++) {
          const PetscInt cell0 = facetCells[2*(intFacetOffset + i) + 0];
          const PetscInt cell1 = facetCells[2*(intFacetOffset + i) + 1];
          PetscInt       celli, cellj;

          for (celli = 0; celli < patch->totalDofsPerCell; celli++) {
            const PetscInt row = dofsArray[(offset + cell0)*patch->totalDofsPerCell + celli];
            if (row < 0) continue;
            for (cellj = 0; cellj < patch->totalDofsPerCell; cellj++) {
              const PetscInt col = dofsArray[(offset + cell1)*patch->totalDofsPerCell + cellj];
              const PetscInt key = row*rsize + col;
              if (col < 0) continue;
              if (!PetscBTLookupSet(bt, key)) ++dnnz[row];
            }
          }

          for (celli = 0; celli < patch->totalDofsPerCell; celli++) {
            const PetscInt row = dofsArray[(offset + cell1)*patch->totalDofsPerCell + celli];
            if (row < 0) continue;
            for (cellj = 0; cellj < patch->totalDofsPerCell; cellj++) {
              const PetscInt col = dofsArray[(offset + cell0)*patch->totalDofsPerCell + cellj];
              const PetscInt key = row*rsize + col;
              if (col < 0) continue;
              if (!PetscBTLookupSet(bt, key)) ++dnnz[row];
            }
          }
        }
      }
      PetscCall(PetscBTDestroy(&bt));
      PetscCall(MatXAIJSetPreallocation(*mat, 1, dnnz, NULL, NULL, NULL));
      PetscCall(PetscFree(dnnz));

      PetscCall(PetscCalloc1(patch->totalDofsPerCell*patch->totalDofsPerCell, &zeroes));
      for (c = 0; c < ncell; ++c) {
        const PetscInt *idx = &dofsArray[(offset + c)*patch->totalDofsPerCell];
        PetscCall(MatSetValues(*mat, patch->totalDofsPerCell, idx, patch->totalDofsPerCell, idx, zeroes, INSERT_VALUES));
      }
      PetscCall(MatGetLocalSize(*mat, &rows, NULL));
      for (i = 0; i < rows; ++i) {
        PetscCall(MatSetValues(*mat, 1, &i, 1, &i, zeroes, INSERT_VALUES));
      }

      if (patch->usercomputeopintfacet) {
        const PetscInt *intFacetsArray = NULL;
        PetscInt i, numIntFacets, intFacetOffset;
        const PetscInt *facetCells = NULL;

        PetscCall(PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets));
        PetscCall(PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset));
        PetscCall(ISGetIndices(patch->intFacetsToPatchCell, &facetCells));
        PetscCall(ISGetIndices(patch->intFacets, &intFacetsArray));
        for (i = 0; i < numIntFacets; i++) {
          const PetscInt cell0 = facetCells[2*(intFacetOffset + i) + 0];
          const PetscInt cell1 = facetCells[2*(intFacetOffset + i) + 1];
          const PetscInt *cell0idx = &dofsArray[(offset + cell0)*patch->totalDofsPerCell];
          const PetscInt *cell1idx = &dofsArray[(offset + cell1)*patch->totalDofsPerCell];
          PetscCall(MatSetValues(*mat, patch->totalDofsPerCell, cell0idx, patch->totalDofsPerCell, cell1idx, zeroes, INSERT_VALUES));
          PetscCall(MatSetValues(*mat, patch->totalDofsPerCell, cell1idx, patch->totalDofsPerCell, cell0idx, zeroes, INSERT_VALUES));
        }
      }

      PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));

      PetscCall(PetscFree(zeroes));

    } else { /* rsize too big, use MATPREALLOCATOR */
      Mat preallocator;
      PetscScalar* vals;

      PetscCall(PetscCalloc1(patch->totalDofsPerCell*patch->totalDofsPerCell, &vals));
      PetscCall(MatCreate(PETSC_COMM_SELF, &preallocator));
      PetscCall(MatSetType(preallocator, MATPREALLOCATOR));
      PetscCall(MatSetSizes(preallocator, rsize, rsize, rsize, rsize));
      PetscCall(MatSetUp(preallocator));

      for (c = 0; c < ncell; ++c) {
        const PetscInt *idx = dofsArray + (offset + c)*patch->totalDofsPerCell;
        PetscCall(MatSetValues(preallocator, patch->totalDofsPerCell, idx, patch->totalDofsPerCell, idx, vals, INSERT_VALUES));
      }

      if (patch->usercomputeopintfacet) {
        const PetscInt *intFacetsArray = NULL;
        PetscInt        i, numIntFacets, intFacetOffset;
        const PetscInt *facetCells = NULL;

        PetscCall(PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets));
        PetscCall(PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset));
        PetscCall(ISGetIndices(patch->intFacetsToPatchCell, &facetCells));
        PetscCall(ISGetIndices(patch->intFacets, &intFacetsArray));
        for (i = 0; i < numIntFacets; i++) {
          const PetscInt cell0 = facetCells[2*(intFacetOffset + i) + 0];
          const PetscInt cell1 = facetCells[2*(intFacetOffset + i) + 1];
          const PetscInt *cell0idx = &dofsArray[(offset + cell0)*patch->totalDofsPerCell];
          const PetscInt *cell1idx = &dofsArray[(offset + cell1)*patch->totalDofsPerCell];
          PetscCall(MatSetValues(preallocator, patch->totalDofsPerCell, cell0idx, patch->totalDofsPerCell, cell1idx, vals, INSERT_VALUES));
          PetscCall(MatSetValues(preallocator, patch->totalDofsPerCell, cell1idx, patch->totalDofsPerCell, cell0idx, vals, INSERT_VALUES));
        }
      }

      PetscCall(PetscFree(vals));
      PetscCall(MatAssemblyBegin(preallocator, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(preallocator, MAT_FINAL_ASSEMBLY));
      PetscCall(MatPreallocatorPreallocate(preallocator, PETSC_TRUE, *mat));
      PetscCall(MatDestroy(&preallocator));
    }
    PetscCall(PetscLogEventEnd(PC_Patch_Prealloc, pc, 0, 0, 0));
    if (withArtificial) {
      PetscCall(ISRestoreIndices(patch->dofsWithArtificial, &dofsArray));
    } else {
      PetscCall(ISRestoreIndices(patch->dofs, &dofsArray));
    }
  }
  PetscCall(MatSetUp(*mat));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchComputeFunction_DMPlex_Private(PC pc, PetscInt patchNum, Vec x, Vec F, IS cellIS, PetscInt n, const PetscInt *l2p, const PetscInt *l2pWithAll, void *ctx)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  DM              dm, plex;
  PetscSection    s;
  const PetscInt *parray, *oarray;
  PetscInt        Nf = patch->nsubspaces, Np, poff, p, f;

  PetscFunctionBegin;
  PetscCheck(!patch->precomputeElementTensors,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Precomputing element tensors not implemented with DMPlex compute operator");
  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  PetscCall(DMGetLocalSection(dm, &s));
  /* Set offset into patch */
  PetscCall(PetscSectionGetDof(patch->pointCounts, patchNum, &Np));
  PetscCall(PetscSectionGetOffset(patch->pointCounts, patchNum, &poff));
  PetscCall(ISGetIndices(patch->points, &parray));
  PetscCall(ISGetIndices(patch->offs,   &oarray));
  for (f = 0; f < Nf; ++f) {
    for (p = 0; p < Np; ++p) {
      const PetscInt point = parray[poff+p];
      PetscInt       dof;

      PetscCall(PetscSectionGetFieldDof(patch->patchSection, point, f, &dof));
      PetscCall(PetscSectionSetFieldOffset(patch->patchSection, point, f, oarray[(poff+p)*Nf+f]));
      if (patch->nsubspaces == 1) PetscCall(PetscSectionSetOffset(patch->patchSection, point, oarray[(poff+p)*Nf+f]));
      else                        PetscCall(PetscSectionSetOffset(patch->patchSection, point, -1));
    }
  }
  PetscCall(ISRestoreIndices(patch->points, &parray));
  PetscCall(ISRestoreIndices(patch->offs,   &oarray));
  if (patch->viewSection) PetscCall(ObjectView((PetscObject) patch->patchSection, patch->viewerSection, patch->formatSection));
  PetscCall(DMPlexComputeResidual_Patch_Internal(dm, patch->patchSection, cellIS, 0.0, x, NULL, F, ctx));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

PetscErrorCode PCPatchComputeFunction_Internal(PC pc, Vec x, Vec F, PetscInt point)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  const PetscInt *dofsArray;
  const PetscInt *dofsArrayWithAll;
  const PetscInt *cellsArray;
  PetscInt        ncell, offset, pStart, pEnd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0));
  PetscCheck(patch->usercomputeop,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PCPatchSetComputeOperator() to set user callback");
  PetscCall(ISGetIndices(patch->dofs, &dofsArray));
  PetscCall(ISGetIndices(patch->dofsWithAll, &dofsArrayWithAll));
  PetscCall(ISGetIndices(patch->cells, &cellsArray));
  PetscCall(PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd));

  point += pStart;
  PetscCheckFalse(point >= pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Operator point %D not in [%D, %D)", point, pStart, pEnd);

  PetscCall(PetscSectionGetDof(patch->cellCounts, point, &ncell));
  PetscCall(PetscSectionGetOffset(patch->cellCounts, point, &offset));
  if (ncell <= 0) {
    PetscCall(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));
    PetscFunctionReturn(0);
  }
  PetscCall(VecSet(F, 0.0));
  PetscStackPush("PCPatch user callback");
  /* Cannot reuse the same IS because the geometry info is being cached in it */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray + offset, PETSC_USE_POINTER, &patch->cellIS));
  ierr = patch->usercomputef(pc, point, x, F, patch->cellIS, ncell*patch->totalDofsPerCell, dofsArray + offset*patch->totalDofsPerCell,
                                                                                            dofsArrayWithAll + offset*patch->totalDofsPerCell,
                                                                                            patch->usercomputefctx);PetscCall(ierr);
  PetscStackPop;
  PetscCall(ISDestroy(&patch->cellIS));
  PetscCall(ISRestoreIndices(patch->dofs, &dofsArray));
  PetscCall(ISRestoreIndices(patch->dofsWithAll, &dofsArrayWithAll));
  PetscCall(ISRestoreIndices(patch->cells, &cellsArray));
  if (patch->viewMatrix) {
    char name[PETSC_MAX_PATH_LEN];

    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "Patch vector for Point %D", point));
    PetscCall(PetscObjectSetName((PetscObject) F, name));
    PetscCall(ObjectView((PetscObject) F, patch->viewerMatrix, patch->formatMatrix));
  }
  PetscCall(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchComputeOperator_DMPlex_Private(PC pc, PetscInt patchNum, Vec x, Mat J, IS cellIS, PetscInt n, const PetscInt *l2p, const PetscInt *l2pWithAll, void *ctx)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  DM              dm, plex;
  PetscSection    s;
  const PetscInt *parray, *oarray;
  PetscInt        Nf = patch->nsubspaces, Np, poff, p, f;

  PetscFunctionBegin;
  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  PetscCall(DMGetLocalSection(dm, &s));
  /* Set offset into patch */
  PetscCall(PetscSectionGetDof(patch->pointCounts, patchNum, &Np));
  PetscCall(PetscSectionGetOffset(patch->pointCounts, patchNum, &poff));
  PetscCall(ISGetIndices(patch->points, &parray));
  PetscCall(ISGetIndices(patch->offs,   &oarray));
  for (f = 0; f < Nf; ++f) {
    for (p = 0; p < Np; ++p) {
      const PetscInt point = parray[poff+p];
      PetscInt       dof;

      PetscCall(PetscSectionGetFieldDof(patch->patchSection, point, f, &dof));
      PetscCall(PetscSectionSetFieldOffset(patch->patchSection, point, f, oarray[(poff+p)*Nf+f]));
      if (patch->nsubspaces == 1) PetscCall(PetscSectionSetOffset(patch->patchSection, point, oarray[(poff+p)*Nf+f]));
      else                        PetscCall(PetscSectionSetOffset(patch->patchSection, point, -1));
    }
  }
  PetscCall(ISRestoreIndices(patch->points, &parray));
  PetscCall(ISRestoreIndices(patch->offs,   &oarray));
  if (patch->viewSection) PetscCall(ObjectView((PetscObject) patch->patchSection, patch->viewerSection, patch->formatSection));
  /* TODO Shut off MatViewFromOptions() in MatAssemblyEnd() here */
  PetscCall(DMPlexComputeJacobian_Patch_Internal(dm, patch->patchSection, patch->patchSection, cellIS, 0.0, 0.0, x, NULL, J, J, ctx));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

/* This function zeros mat on entry */
PetscErrorCode PCPatchComputeOperator_Internal(PC pc, Vec x, Mat mat, PetscInt point, PetscBool withArtificial)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  const PetscInt *dofsArray;
  const PetscInt *dofsArrayWithAll = NULL;
  const PetscInt *cellsArray;
  PetscInt        ncell, offset, pStart, pEnd, numIntFacets, intFacetOffset;
  PetscBool       isNonlinear;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0));
  isNonlinear = patch->isNonlinear;
  PetscCheck(patch->usercomputeop,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PCPatchSetComputeOperator() to set user callback");
  if (withArtificial) {
    PetscCall(ISGetIndices(patch->dofsWithArtificial, &dofsArray));
  } else {
    PetscCall(ISGetIndices(patch->dofs, &dofsArray));
  }
  if (isNonlinear) {
    PetscCall(ISGetIndices(patch->dofsWithAll, &dofsArrayWithAll));
  }
  PetscCall(ISGetIndices(patch->cells, &cellsArray));
  PetscCall(PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd));

  point += pStart;
  PetscCheckFalse(point >= pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Operator point %D not in [%D, %D)", point, pStart, pEnd);

  PetscCall(PetscSectionGetDof(patch->cellCounts, point, &ncell));
  PetscCall(PetscSectionGetOffset(patch->cellCounts, point, &offset));
  if (ncell <= 0) {
    PetscCall(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));
    PetscFunctionReturn(0);
  }
  PetscCall(MatZeroEntries(mat));
  if (patch->precomputeElementTensors) {
    PetscInt           i;
    PetscInt           ndof = patch->totalDofsPerCell;
    const PetscScalar *elementTensors;

    PetscCall(VecGetArrayRead(patch->cellMats, &elementTensors));
    for (i = 0; i < ncell; i++) {
      const PetscInt     cell = cellsArray[i + offset];
      const PetscInt    *idx  = dofsArray + (offset + i)*ndof;
      const PetscScalar *v    = elementTensors + patch->precomputedTensorLocations[cell]*ndof*ndof;
      PetscCall(MatSetValues(mat, ndof, idx, ndof, idx, v, ADD_VALUES));
    }
    PetscCall(VecRestoreArrayRead(patch->cellMats, &elementTensors));
    PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  } else {
    PetscStackPush("PCPatch user callback");
    /* Cannot reuse the same IS because the geometry info is being cached in it */
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray + offset, PETSC_USE_POINTER, &patch->cellIS));
    PetscCall(patch->usercomputeop(pc, point, x, mat, patch->cellIS, ncell*patch->totalDofsPerCell, dofsArray + offset*patch->totalDofsPerCell, dofsArrayWithAll ? dofsArrayWithAll + offset*patch->totalDofsPerCell : NULL, patch->usercomputeopctx));
  }
  if (patch->usercomputeopintfacet) {
    PetscCall(PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets));
    PetscCall(PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset));
    if (numIntFacets > 0) {
      /* For each interior facet, grab the two cells (in local numbering, and concatenate dof numberings for those cells) */
      PetscInt       *facetDofs = NULL, *facetDofsWithAll = NULL;
      const PetscInt *intFacetsArray = NULL;
      PetscInt        idx = 0;
      PetscInt        i, c, d;
      PetscInt        fStart;
      DM              dm, plex;
      IS              facetIS = NULL;
      const PetscInt *facetCells = NULL;

      PetscCall(ISGetIndices(patch->intFacetsToPatchCell, &facetCells));
      PetscCall(ISGetIndices(patch->intFacets, &intFacetsArray));
      PetscCall(PCGetDM(pc, &dm));
      PetscCall(DMConvert(dm, DMPLEX, &plex));
      dm = plex;
      PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, NULL));
      /* FIXME: Pull this malloc out. */
      PetscCall(PetscMalloc1(2 * patch->totalDofsPerCell * numIntFacets, &facetDofs));
      if (dofsArrayWithAll) {
        PetscCall(PetscMalloc1(2 * patch->totalDofsPerCell * numIntFacets, &facetDofsWithAll));
      }
      if (patch->precomputeElementTensors) {
        PetscInt           nFacetDof = 2*patch->totalDofsPerCell;
        const PetscScalar *elementTensors;

        PetscCall(VecGetArrayRead(patch->intFacetMats, &elementTensors));

        for (i = 0; i < numIntFacets; i++) {
          const PetscInt     facet = intFacetsArray[i + intFacetOffset];
          const PetscScalar *v     = elementTensors + patch->precomputedIntFacetTensorLocations[facet - fStart]*nFacetDof*nFacetDof;
          idx = 0;
          /*
           * 0--1
           * |\-|
           * |+\|
           * 2--3
           * [0, 2, 3, 0, 1, 3]
           */
          for (c = 0; c < 2; c++) {
            const PetscInt cell = facetCells[2*(intFacetOffset + i) + c];
            for (d = 0; d < patch->totalDofsPerCell; d++) {
              facetDofs[idx] = dofsArray[(offset + cell)*patch->totalDofsPerCell + d];
              idx++;
            }
          }
          PetscCall(MatSetValues(mat, nFacetDof, facetDofs, nFacetDof, facetDofs, v, ADD_VALUES));
        }
        PetscCall(VecRestoreArrayRead(patch->intFacetMats, &elementTensors));
      } else {
        /*
         * 0--1
         * |\-|
         * |+\|
         * 2--3
         * [0, 2, 3, 0, 1, 3]
         */
        for (i = 0; i < numIntFacets; i++) {
          for (c = 0; c < 2; c++) {
            const PetscInt cell = facetCells[2*(intFacetOffset + i) + c];
            for (d = 0; d < patch->totalDofsPerCell; d++) {
              facetDofs[idx] = dofsArray[(offset + cell)*patch->totalDofsPerCell + d];
              if (dofsArrayWithAll) {
                facetDofsWithAll[idx] = dofsArrayWithAll[(offset + cell)*patch->totalDofsPerCell + d];
              }
              idx++;
            }
          }
        }
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numIntFacets, intFacetsArray + intFacetOffset, PETSC_USE_POINTER, &facetIS));
        PetscCall(patch->usercomputeopintfacet(pc, point, x, mat, facetIS, 2*numIntFacets*patch->totalDofsPerCell, facetDofs, facetDofsWithAll, patch->usercomputeopintfacetctx));
        PetscCall(ISDestroy(&facetIS));
      }
      PetscCall(ISRestoreIndices(patch->intFacetsToPatchCell, &facetCells));
      PetscCall(ISRestoreIndices(patch->intFacets, &intFacetsArray));
      PetscCall(PetscFree(facetDofs));
      PetscCall(PetscFree(facetDofsWithAll));
      PetscCall(DMDestroy(&dm));
    }
  }

  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  if (!(withArtificial || isNonlinear) && patch->denseinverse) {
    MatFactorInfo info;
    PetscBool     flg;
    PetscCall(PetscObjectTypeCompare((PetscObject)mat, MATSEQDENSE, &flg));
    PetscCheck(flg,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Invalid Mat type for dense inverse");
    PetscCall(MatFactorInfoInitialize(&info));
    PetscCall(MatLUFactor(mat, NULL, NULL, &info));
    PetscCall(MatSeqDenseInvertFactors_Private(mat));
  }
  PetscStackPop;
  PetscCall(ISDestroy(&patch->cellIS));
  if (withArtificial) {
    PetscCall(ISRestoreIndices(patch->dofsWithArtificial, &dofsArray));
  } else {
    PetscCall(ISRestoreIndices(patch->dofs, &dofsArray));
  }
  if (isNonlinear) {
    PetscCall(ISRestoreIndices(patch->dofsWithAll, &dofsArrayWithAll));
  }
  PetscCall(ISRestoreIndices(patch->cells, &cellsArray));
  if (patch->viewMatrix) {
    char name[PETSC_MAX_PATH_LEN];

    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "Patch matrix for Point %D", point));
    PetscCall(PetscObjectSetName((PetscObject) mat, name));
    PetscCall(ObjectView((PetscObject) mat, patch->viewerMatrix, patch->formatMatrix));
  }
  PetscCall(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_PCPatch_Private(Mat mat, PetscInt m, const PetscInt idxm[],
                                                   PetscInt n, const PetscInt idxn[], const PetscScalar *v, InsertMode addv)
{
  Vec            data;
  PetscScalar   *array;
  PetscInt       bs, nz, i, j, cell;

  PetscCall(MatShellGetContext(mat, &data));
  PetscCall(VecGetBlockSize(data, &bs));
  PetscCall(VecGetSize(data, &nz));
  PetscCall(VecGetArray(data, &array));
  PetscCheckFalse(m != n,PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONG, "Only for square insertion");
  cell = (PetscInt)(idxm[0]/bs); /* use the fact that this is called once per cell */
  for (i = 0; i < m; i++) {
    PetscCheckFalse(idxm[i] != idxn[i],PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONG, "Row and column indices must match!");
    for (j = 0; j < n; j++) {
      const PetscScalar v_ = v[i*bs + j];
      /* Indexing is special to the data structure we have! */
      if (addv == INSERT_VALUES) {
        array[cell*bs*bs + i*bs + j] = v_;
      } else {
        array[cell*bs*bs + i*bs + j] += v_;
      }
    }
  }
  PetscCall(VecRestoreArray(data, &array));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchPrecomputePatchTensors_Private(PC pc)
{
  PC_PATCH       *patch = (PC_PATCH *)pc->data;
  const PetscInt *cellsArray;
  PetscInt        ncell, offset;
  const PetscInt *dofMapArray;
  PetscInt        i, j;
  IS              dofMap;
  IS              cellIS;
  const PetscInt  ndof  = patch->totalDofsPerCell;
  PetscErrorCode  ierr;
  Mat             vecMat;
  PetscInt        cStart, cEnd;
  DM              dm, plex;

  PetscCall(ISGetSize(patch->cells, &ncell));
  if (!ncell) { /* No cells to assemble over -> skip */
    PetscFunctionReturn(0);
  }

  PetscCall(PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0));

  PetscCall(PCGetDM(pc, &dm));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  if (!patch->allCells) {
    PetscHSetI      cells;
    PetscHashIter   hi;
    PetscInt        pStart, pEnd;
    PetscInt        *allCells = NULL;
    PetscCall(PetscHSetICreate(&cells));
    PetscCall(ISGetIndices(patch->cells, &cellsArray));
    PetscCall(PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd));
    for (i = pStart; i < pEnd; i++) {
      PetscCall(PetscSectionGetDof(patch->cellCounts, i, &ncell));
      PetscCall(PetscSectionGetOffset(patch->cellCounts, i, &offset));
      if (ncell <= 0) continue;
      for (j = 0; j < ncell; j++) {
        PetscCall(PetscHSetIAdd(cells, cellsArray[offset + j]));
      }
    }
    PetscCall(ISRestoreIndices(patch->cells, &cellsArray));
    PetscCall(PetscHSetIGetSize(cells, &ncell));
    PetscCall(PetscMalloc1(ncell, &allCells));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(PetscMalloc1(cEnd-cStart, &patch->precomputedTensorLocations));
    i = 0;
    PetscHashIterBegin(cells, hi);
    while (!PetscHashIterAtEnd(cells, hi)) {
      PetscHashIterGetKey(cells, hi, allCells[i]);
      patch->precomputedTensorLocations[allCells[i]] = i;
      PetscHashIterNext(cells, hi);
      i++;
    }
    PetscCall(PetscHSetIDestroy(&cells));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ncell, allCells, PETSC_OWN_POINTER, &patch->allCells));
  }
  PetscCall(ISGetSize(patch->allCells, &ncell));
  if (!patch->cellMats) {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, ncell*ndof*ndof, &patch->cellMats));
    PetscCall(VecSetBlockSize(patch->cellMats, ndof));
  }
  PetscCall(VecSet(patch->cellMats, 0));

  ierr = MatCreateShell(PETSC_COMM_SELF, ncell*ndof, ncell*ndof, ncell*ndof, ncell*ndof,
                        (void*)patch->cellMats, &vecMat);PetscCall(ierr);
  PetscCall(MatShellSetOperation(vecMat, MATOP_SET_VALUES, (void(*)(void))&MatSetValues_PCPatch_Private));
  PetscCall(ISGetSize(patch->allCells, &ncell));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, ndof*ncell, 0, 1, &dofMap));
  PetscCall(ISGetIndices(dofMap, &dofMapArray));
  PetscCall(ISGetIndices(patch->allCells, &cellsArray));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray, PETSC_USE_POINTER, &cellIS));
  PetscStackPush("PCPatch user callback");
  /* TODO: Fix for DMPlex compute op, this bypasses a lot of the machinery and just assembles every element tensor. */
  PetscCall(patch->usercomputeop(pc, -1, NULL, vecMat, cellIS, ndof*ncell, dofMapArray, NULL, patch->usercomputeopctx));
  PetscStackPop;
  PetscCall(ISDestroy(&cellIS));
  PetscCall(MatDestroy(&vecMat));
  PetscCall(ISRestoreIndices(patch->allCells, &cellsArray));
  PetscCall(ISRestoreIndices(dofMap, &dofMapArray));
  PetscCall(ISDestroy(&dofMap));

  if (patch->usercomputeopintfacet) {
    PetscInt nIntFacets;
    IS       intFacetsIS;
    const PetscInt *intFacetsArray = NULL;
    if (!patch->allIntFacets) {
      PetscHSetI      facets;
      PetscHashIter   hi;
      PetscInt        pStart, pEnd, fStart, fEnd;
      PetscInt        *allIntFacets = NULL;
      PetscCall(PetscHSetICreate(&facets));
      PetscCall(ISGetIndices(patch->intFacets, &intFacetsArray));
      PetscCall(PetscSectionGetChart(patch->intFacetCounts, &pStart, &pEnd));
      PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
      for (i = pStart; i < pEnd; i++) {
        PetscCall(PetscSectionGetDof(patch->intFacetCounts, i, &nIntFacets));
        PetscCall(PetscSectionGetOffset(patch->intFacetCounts, i, &offset));
        if (nIntFacets <= 0) continue;
        for (j = 0; j < nIntFacets; j++) {
          PetscCall(PetscHSetIAdd(facets, intFacetsArray[offset + j]));
        }
      }
      PetscCall(ISRestoreIndices(patch->intFacets, &intFacetsArray));
      PetscCall(PetscHSetIGetSize(facets, &nIntFacets));
      PetscCall(PetscMalloc1(nIntFacets, &allIntFacets));
      PetscCall(PetscMalloc1(fEnd-fStart, &patch->precomputedIntFacetTensorLocations));
      i = 0;
      PetscHashIterBegin(facets, hi);
      while (!PetscHashIterAtEnd(facets, hi)) {
        PetscHashIterGetKey(facets, hi, allIntFacets[i]);
        patch->precomputedIntFacetTensorLocations[allIntFacets[i] - fStart] = i;
        PetscHashIterNext(facets, hi);
        i++;
      }
      PetscCall(PetscHSetIDestroy(&facets));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nIntFacets, allIntFacets, PETSC_OWN_POINTER, &patch->allIntFacets));
    }
    PetscCall(ISGetSize(patch->allIntFacets, &nIntFacets));
    if (!patch->intFacetMats) {
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, nIntFacets*ndof*ndof*4, &patch->intFacetMats));
      PetscCall(VecSetBlockSize(patch->intFacetMats, ndof*2));
    }
    PetscCall(VecSet(patch->intFacetMats, 0));

    ierr = MatCreateShell(PETSC_COMM_SELF, nIntFacets*ndof*2, nIntFacets*ndof*2, nIntFacets*ndof*2, nIntFacets*ndof*2,
                          (void*)patch->intFacetMats, &vecMat);PetscCall(ierr);
    PetscCall(MatShellSetOperation(vecMat, MATOP_SET_VALUES, (void(*)(void))&MatSetValues_PCPatch_Private));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 2*ndof*nIntFacets, 0, 1, &dofMap));
    PetscCall(ISGetIndices(dofMap, &dofMapArray));
    PetscCall(ISGetIndices(patch->allIntFacets, &intFacetsArray));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nIntFacets, intFacetsArray, PETSC_USE_POINTER, &intFacetsIS));
    PetscStackPush("PCPatch user callback (interior facets)");
    /* TODO: Fix for DMPlex compute op, this bypasses a lot of the machinery and just assembles every element tensor. */
    PetscCall(patch->usercomputeopintfacet(pc, -1, NULL, vecMat, intFacetsIS, 2*ndof*nIntFacets, dofMapArray, NULL, patch->usercomputeopintfacetctx));
    PetscStackPop;
    PetscCall(ISDestroy(&intFacetsIS));
    PetscCall(MatDestroy(&vecMat));
    PetscCall(ISRestoreIndices(patch->allIntFacets, &intFacetsArray));
    PetscCall(ISRestoreIndices(dofMap, &dofMapArray));
    PetscCall(ISDestroy(&dofMap));
  }
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));

  PetscFunctionReturn(0);
}

PetscErrorCode PCPatch_ScatterLocal_Private(PC pc, PetscInt p, Vec x, Vec y, InsertMode mode, ScatterMode scat, PatchScatterType scattertype)
{
  PC_PATCH          *patch     = (PC_PATCH *) pc->data;
  const PetscScalar *xArray    = NULL;
  PetscScalar       *yArray    = NULL;
  const PetscInt    *gtolArray = NULL;
  PetscInt           dof, offset, lidx;

  PetscFunctionBeginHot;
  PetscCall(VecGetArrayRead(x, &xArray));
  PetscCall(VecGetArray(y, &yArray));
  if (scattertype == SCATTER_WITHARTIFICIAL) {
    PetscCall(PetscSectionGetDof(patch->gtolCountsWithArtificial, p, &dof));
    PetscCall(PetscSectionGetOffset(patch->gtolCountsWithArtificial, p, &offset));
    PetscCall(ISGetIndices(patch->gtolWithArtificial, &gtolArray));
  } else if (scattertype == SCATTER_WITHALL) {
    PetscCall(PetscSectionGetDof(patch->gtolCountsWithAll, p, &dof));
    PetscCall(PetscSectionGetOffset(patch->gtolCountsWithAll, p, &offset));
    PetscCall(ISGetIndices(patch->gtolWithAll, &gtolArray));
  } else {
    PetscCall(PetscSectionGetDof(patch->gtolCounts, p, &dof));
    PetscCall(PetscSectionGetOffset(patch->gtolCounts, p, &offset));
    PetscCall(ISGetIndices(patch->gtol, &gtolArray));
  }
  PetscCheckFalse(mode == INSERT_VALUES && scat != SCATTER_FORWARD,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't insert if not scattering forward");
  PetscCheckFalse(mode == ADD_VALUES    && scat != SCATTER_REVERSE,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't add if not scattering reverse");
  for (lidx = 0; lidx < dof; ++lidx) {
    const PetscInt gidx = gtolArray[offset+lidx];

    if (mode == INSERT_VALUES) yArray[lidx]  = xArray[gidx]; /* Forward */
    else                       yArray[gidx] += xArray[lidx]; /* Reverse */
  }
  if (scattertype == SCATTER_WITHARTIFICIAL) {
    PetscCall(ISRestoreIndices(patch->gtolWithArtificial, &gtolArray));
  } else if (scattertype == SCATTER_WITHALL) {
    PetscCall(ISRestoreIndices(patch->gtolWithAll, &gtolArray));
  } else {
    PetscCall(ISRestoreIndices(patch->gtol, &gtolArray));
  }
  PetscCall(VecRestoreArrayRead(x, &xArray));
  PetscCall(VecRestoreArray(y, &yArray));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_PATCH_Linear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  const char    *prefix;
  PetscInt       i;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscCheck(patch->save_operators || !patch->denseinverse,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Can't have dense inverse without save operators");
    if (!patch->denseinverse) {
      PetscCall(PetscMalloc1(patch->npatch, &patch->solver));
      PetscCall(PCGetOptionsPrefix(pc, &prefix));
      for (i = 0; i < patch->npatch; ++i) {
        KSP ksp;
        PC  subpc;

        PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));
        PetscCall(KSPSetErrorIfNotConverged(ksp, pc->erroriffailure));
        PetscCall(KSPSetOptionsPrefix(ksp, prefix));
        PetscCall(KSPAppendOptionsPrefix(ksp, "sub_"));
        PetscCall(PetscObjectIncrementTabLevel((PetscObject) ksp, (PetscObject) pc, 1));
        PetscCall(KSPGetPC(ksp, &subpc));
        PetscCall(PetscObjectIncrementTabLevel((PetscObject) subpc, (PetscObject) pc, 1));
        PetscCall(PetscLogObjectParent((PetscObject) pc, (PetscObject) ksp));
        patch->solver[i] = (PetscObject) ksp;
      }
    }
  }
  if (patch->save_operators) {
    if (patch->precomputeElementTensors) {
      PetscCall(PCPatchPrecomputePatchTensors_Private(pc));
    }
    for (i = 0; i < patch->npatch; ++i) {
      PetscCall(PCPatchComputeOperator_Internal(pc, NULL, patch->mat[i], i, PETSC_FALSE));
      if (!patch->denseinverse) {
        PetscCall(KSPSetOperators((KSP) patch->solver[i], patch->mat[i], patch->mat[i]));
      } else if (patch->mat[i] && !patch->densesolve) {
        /* Setup matmult callback */
        PetscCall(MatGetOperation(patch->mat[i], MATOP_MULT, (void (**)(void))&patch->densesolve));
      }
    }
  }
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    for (i = 0; i < patch->npatch; ++i) {
      /* Instead of padding patch->patchUpdate with zeros to get */
      /* patch->patchUpdateWithArtificial and then multiplying with the matrix, */
      /* just get rid of the columns that correspond to the dofs with */
      /* artificial bcs. That's of course fairly inefficient, hopefully we */
      /* can just assemble the rectangular matrix in the first place. */
      Mat matSquare;
      IS rowis;
      PetscInt dof;

      PetscCall(MatGetSize(patch->mat[i], &dof, NULL));
      if (dof == 0) {
        patch->matWithArtificial[i] = NULL;
        continue;
      }

      PetscCall(PCPatchCreateMatrix_Private(pc, i, &matSquare, PETSC_TRUE));
      PetscCall(PCPatchComputeOperator_Internal(pc, NULL, matSquare, i, PETSC_TRUE));

      PetscCall(MatGetSize(matSquare, &dof, NULL));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, dof, 0, 1, &rowis));
      if (pc->setupcalled) {
        PetscCall(MatCreateSubMatrix(matSquare, rowis, patch->dofMappingWithoutToWithArtificial[i], MAT_REUSE_MATRIX, &patch->matWithArtificial[i]));
      } else {
        PetscCall(MatCreateSubMatrix(matSquare, rowis, patch->dofMappingWithoutToWithArtificial[i], MAT_INITIAL_MATRIX, &patch->matWithArtificial[i]));
      }
      PetscCall(ISDestroy(&rowis));
      PetscCall(MatDestroy(&matSquare));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_PATCH(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;
  PetscBool      isNonlinear;
  PetscInt       maxDof = -1, maxDofWithArtificial = -1;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscInt pStart, pEnd, p;
    PetscInt localSize;

    PetscCall(PetscLogEventBegin(PC_Patch_CreatePatches, pc, 0, 0, 0));

    isNonlinear = patch->isNonlinear;
    if (!patch->nsubspaces) {
      DM           dm, plex;
      PetscSection s;
      PetscInt     cStart, cEnd, c, Nf, f, numGlobalBcs = 0, *globalBcs, *Nb, **cellDofs;

      PetscCall(PCGetDM(pc, &dm));
      PetscCheck(dm,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONG, "Must set DM for PCPATCH or call PCPatchSetDiscretisationInfo()");
      PetscCall(DMConvert(dm, DMPLEX, &plex));
      dm = plex;
      PetscCall(DMGetLocalSection(dm, &s));
      PetscCall(PetscSectionGetNumFields(s, &Nf));
      PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
      for (p = pStart; p < pEnd; ++p) {
        PetscInt cdof;
        PetscCall(PetscSectionGetConstraintDof(s, p, &cdof));
        numGlobalBcs += cdof;
      }
      PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
      PetscCall(PetscMalloc3(Nf, &Nb, Nf, &cellDofs, numGlobalBcs, &globalBcs));
      for (f = 0; f < Nf; ++f) {
        PetscFE        fe;
        PetscDualSpace sp;
        PetscInt       cdoff = 0;

        PetscCall(DMGetField(dm, f, NULL, (PetscObject *) &fe));
        /* PetscCall(PetscFEGetNumComponents(fe, &Nc[f])); */
        PetscCall(PetscFEGetDualSpace(fe, &sp));
        PetscCall(PetscDualSpaceGetDimension(sp, &Nb[f]));

        PetscCall(PetscMalloc1((cEnd-cStart)*Nb[f], &cellDofs[f]));
        for (c = cStart; c < cEnd; ++c) {
          PetscInt *closure = NULL;
          PetscInt  clSize  = 0, cl;

          PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
          for (cl = 0; cl < clSize*2; cl += 2) {
            const PetscInt p = closure[cl];
            PetscInt       fdof, d, foff;

            PetscCall(PetscSectionGetFieldDof(s, p, f, &fdof));
            PetscCall(PetscSectionGetFieldOffset(s, p, f, &foff));
            for (d = 0; d < fdof; ++d, ++cdoff) cellDofs[f][cdoff] = foff + d;
          }
          PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
        }
        PetscCheckFalse(cdoff != (cEnd-cStart)*Nb[f],PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_SIZ, "Total number of cellDofs %D for field %D should be Nc (%D) * cellDof (%D)", cdoff, f, cEnd-cStart, Nb[f]);
      }
      numGlobalBcs = 0;
      for (p = pStart; p < pEnd; ++p) {
        const PetscInt *ind;
        PetscInt        off, cdof, d;

        PetscCall(PetscSectionGetOffset(s, p, &off));
        PetscCall(PetscSectionGetConstraintDof(s, p, &cdof));
        PetscCall(PetscSectionGetConstraintIndices(s, p, &ind));
        for (d = 0; d < cdof; ++d) globalBcs[numGlobalBcs++] = off + ind[d];
      }

      PetscCall(PCPatchSetDiscretisationInfoCombined(pc, dm, Nb, (const PetscInt **) cellDofs, numGlobalBcs, globalBcs, numGlobalBcs, globalBcs));
      for (f = 0; f < Nf; ++f) {
        PetscCall(PetscFree(cellDofs[f]));
      }
      PetscCall(PetscFree3(Nb, cellDofs, globalBcs));
      PetscCall(PCPatchSetComputeFunction(pc, PCPatchComputeFunction_DMPlex_Private, NULL));
      PetscCall(PCPatchSetComputeOperator(pc, PCPatchComputeOperator_DMPlex_Private, NULL));
      PetscCall(DMDestroy(&dm));
    }

    localSize = patch->subspaceOffsets[patch->nsubspaces];
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, localSize, &patch->localRHS));
    PetscCall(VecSetUp(patch->localRHS));
    PetscCall(VecDuplicate(patch->localRHS, &patch->localUpdate));
    PetscCall(PCPatchCreateCellPatches(pc));
    PetscCall(PCPatchCreateCellPatchDiscretisationInfo(pc));

    /* OK, now build the work vectors */
    PetscCall(PetscSectionGetChart(patch->gtolCounts, &pStart, &pEnd));

    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
      PetscCall(PetscMalloc1(patch->npatch, &patch->dofMappingWithoutToWithArtificial));
    }
    if (isNonlinear) {
      PetscCall(PetscMalloc1(patch->npatch, &patch->dofMappingWithoutToWithAll));
    }
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof;

      PetscCall(PetscSectionGetDof(patch->gtolCounts, p, &dof));
      maxDof = PetscMax(maxDof, dof);
      if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
        const PetscInt    *gtolArray, *gtolArrayWithArtificial = NULL;
        PetscInt           numPatchDofs, offset;
        PetscInt           numPatchDofsWithArtificial, offsetWithArtificial;
        PetscInt           dofWithoutArtificialCounter = 0;
        PetscInt          *patchWithoutArtificialToWithArtificialArray;

        PetscCall(PetscSectionGetDof(patch->gtolCountsWithArtificial, p, &dof));
        maxDofWithArtificial = PetscMax(maxDofWithArtificial, dof);

        /* Now build the mapping that for a dof in a patch WITHOUT dofs that have artificial bcs gives the */
        /* the index in the patch with all dofs */
        PetscCall(ISGetIndices(patch->gtol, &gtolArray));

        PetscCall(PetscSectionGetDof(patch->gtolCounts, p, &numPatchDofs));
        if (numPatchDofs == 0) {
          patch->dofMappingWithoutToWithArtificial[p-pStart] = NULL;
          continue;
        }

        PetscCall(PetscSectionGetOffset(patch->gtolCounts, p, &offset));
        PetscCall(ISGetIndices(patch->gtolWithArtificial, &gtolArrayWithArtificial));
        PetscCall(PetscSectionGetDof(patch->gtolCountsWithArtificial, p, &numPatchDofsWithArtificial));
        PetscCall(PetscSectionGetOffset(patch->gtolCountsWithArtificial, p, &offsetWithArtificial));

        PetscCall(PetscMalloc1(numPatchDofs, &patchWithoutArtificialToWithArtificialArray));
        for (i=0; i<numPatchDofsWithArtificial; i++) {
          if (gtolArrayWithArtificial[i+offsetWithArtificial] == gtolArray[offset+dofWithoutArtificialCounter]) {
            patchWithoutArtificialToWithArtificialArray[dofWithoutArtificialCounter] = i;
            dofWithoutArtificialCounter++;
            if (dofWithoutArtificialCounter == numPatchDofs)
              break;
          }
        }
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numPatchDofs, patchWithoutArtificialToWithArtificialArray, PETSC_OWN_POINTER, &patch->dofMappingWithoutToWithArtificial[p-pStart]));
        PetscCall(ISRestoreIndices(patch->gtol, &gtolArray));
        PetscCall(ISRestoreIndices(patch->gtolWithArtificial, &gtolArrayWithArtificial));
      }
      if (isNonlinear) {
        const PetscInt    *gtolArray, *gtolArrayWithAll = NULL;
        PetscInt           numPatchDofs, offset;
        PetscInt           numPatchDofsWithAll, offsetWithAll;
        PetscInt           dofWithoutAllCounter = 0;
        PetscInt          *patchWithoutAllToWithAllArray;

        /* Now build the mapping that for a dof in a patch WITHOUT dofs that have artificial bcs gives the */
        /* the index in the patch with all dofs */
        PetscCall(ISGetIndices(patch->gtol, &gtolArray));

        PetscCall(PetscSectionGetDof(patch->gtolCounts, p, &numPatchDofs));
        if (numPatchDofs == 0) {
          patch->dofMappingWithoutToWithAll[p-pStart] = NULL;
          continue;
        }

        PetscCall(PetscSectionGetOffset(patch->gtolCounts, p, &offset));
        PetscCall(ISGetIndices(patch->gtolWithAll, &gtolArrayWithAll));
        PetscCall(PetscSectionGetDof(patch->gtolCountsWithAll, p, &numPatchDofsWithAll));
        PetscCall(PetscSectionGetOffset(patch->gtolCountsWithAll, p, &offsetWithAll));

        PetscCall(PetscMalloc1(numPatchDofs, &patchWithoutAllToWithAllArray));

        for (i=0; i<numPatchDofsWithAll; i++) {
          if (gtolArrayWithAll[i+offsetWithAll] == gtolArray[offset+dofWithoutAllCounter]) {
            patchWithoutAllToWithAllArray[dofWithoutAllCounter] = i;
            dofWithoutAllCounter++;
            if (dofWithoutAllCounter == numPatchDofs)
              break;
          }
        }
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, numPatchDofs, patchWithoutAllToWithAllArray, PETSC_OWN_POINTER, &patch->dofMappingWithoutToWithAll[p-pStart]));
        PetscCall(ISRestoreIndices(patch->gtol, &gtolArray));
        PetscCall(ISRestoreIndices(patch->gtolWithAll, &gtolArrayWithAll));
      }
    }
    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, maxDofWithArtificial, &patch->patchRHSWithArtificial));
      PetscCall(VecSetUp(patch->patchRHSWithArtificial));
    }
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, maxDof, &patch->patchRHS));
    PetscCall(VecSetUp(patch->patchRHS));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, maxDof, &patch->patchUpdate));
    PetscCall(VecSetUp(patch->patchUpdate));
    if (patch->save_operators) {
      PetscCall(PetscMalloc1(patch->npatch, &patch->mat));
      for (i = 0; i < patch->npatch; ++i) {
        PetscCall(PCPatchCreateMatrix_Private(pc, i, &patch->mat[i], PETSC_FALSE));
      }
    }
    PetscCall(PetscLogEventEnd(PC_Patch_CreatePatches, pc, 0, 0, 0));

    /* If desired, calculate weights for dof multiplicity */
    if (patch->partition_of_unity) {
      PetscScalar *input = NULL;
      PetscScalar *output = NULL;
      Vec         global;

      PetscCall(VecDuplicate(patch->localRHS, &patch->dof_weights));
      if (patch->local_composition_type == PC_COMPOSITE_ADDITIVE) {
        for (i = 0; i < patch->npatch; ++i) {
          PetscInt dof;

          PetscCall(PetscSectionGetDof(patch->gtolCounts, i+pStart, &dof));
          if (dof <= 0) continue;
          PetscCall(VecSet(patch->patchRHS, 1.0));
          PetscCall(PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchRHS, patch->dof_weights, ADD_VALUES, SCATTER_REVERSE, SCATTER_INTERIOR));
        }
      } else {
        /* multiplicative is actually only locally multiplicative and globally additive. need the pou where the mesh decomposition overlaps */
        PetscCall(VecSet(patch->dof_weights, 1.0));
      }

      VecDuplicate(patch->dof_weights, &global);
      VecSet(global, 0.);

      PetscCall(VecGetArray(patch->dof_weights, &input));
      PetscCall(VecGetArray(global, &output));
      PetscCall(PetscSFReduceBegin(patch->sectionSF, MPIU_SCALAR, input, output, MPI_SUM));
      PetscCall(PetscSFReduceEnd(patch->sectionSF, MPIU_SCALAR, input, output, MPI_SUM));
      PetscCall(VecRestoreArray(patch->dof_weights, &input));
      PetscCall(VecRestoreArray(global, &output));

      PetscCall(VecReciprocal(global));

      PetscCall(VecGetArray(patch->dof_weights, &output));
      PetscCall(VecGetArray(global, &input));
      PetscCall(PetscSFBcastBegin(patch->sectionSF, MPIU_SCALAR, input, output,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(patch->sectionSF, MPIU_SCALAR, input, output,MPI_REPLACE));
      PetscCall(VecRestoreArray(patch->dof_weights, &output));
      PetscCall(VecRestoreArray(global, &input));
      PetscCall(VecDestroy(&global));
    }
    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE && patch->save_operators) {
      PetscCall(PetscMalloc1(patch->npatch, &patch->matWithArtificial));
    }
  }
  PetscCall((*patch->setupsolver)(pc));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PATCH_Linear(PC pc, PetscInt i, Vec x, Vec y)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  KSP            ksp;
  Mat            op;
  PetscInt       m, n;

  PetscFunctionBegin;
  if (patch->denseinverse) {
    PetscCall((*patch->densesolve)(patch->mat[i], x, y));
    PetscFunctionReturn(0);
  }
  ksp = (KSP) patch->solver[i];
  if (!patch->save_operators) {
    Mat mat;

    PetscCall(PCPatchCreateMatrix_Private(pc, i, &mat, PETSC_FALSE));
    /* Populate operator here. */
    PetscCall(PCPatchComputeOperator_Internal(pc, NULL, mat, i, PETSC_FALSE));
    PetscCall(KSPSetOperators(ksp, mat, mat));
    /* Drop reference so the KSPSetOperators below will blow it away. */
    PetscCall(MatDestroy(&mat));
  }
  PetscCall(PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0));
  if (!ksp->setfromoptionscalled) {
    PetscCall(KSPSetFromOptions(ksp));
  }
  /* Disgusting trick to reuse work vectors */
  PetscCall(KSPGetOperators(ksp, &op, NULL));
  PetscCall(MatGetLocalSize(op, &m, &n));
  x->map->n = m;
  y->map->n = n;
  x->map->N = m;
  y->map->N = n;
  PetscCall(KSPSolve(ksp, x, y));
  PetscCall(KSPCheckSolve(ksp, pc, y));
  PetscCall(PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0));
  if (!patch->save_operators) {
    PC pc;
    PetscCall(KSPSetOperators(ksp, NULL, NULL));
    PetscCall(KSPGetPC(ksp, &pc));
    /* Destroy PC context too, otherwise the factored matrix hangs around. */
    PetscCall(PCReset(pc));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCUpdateMultiplicative_PATCH_Linear(PC pc, PetscInt i, PetscInt pStart)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  Mat            multMat;
  PetscInt       n, m;

  PetscFunctionBegin;

  if (patch->save_operators) {
    multMat = patch->matWithArtificial[i];
  } else {
    /*Very inefficient, hopefully we can just assemble the rectangular matrix in the first place.*/
    Mat      matSquare;
    PetscInt dof;
    IS       rowis;
    PetscCall(PCPatchCreateMatrix_Private(pc, i, &matSquare, PETSC_TRUE));
    PetscCall(PCPatchComputeOperator_Internal(pc, NULL, matSquare, i, PETSC_TRUE));
    PetscCall(MatGetSize(matSquare, &dof, NULL));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, dof, 0, 1, &rowis));
    PetscCall(MatCreateSubMatrix(matSquare, rowis, patch->dofMappingWithoutToWithArtificial[i], MAT_INITIAL_MATRIX, &multMat));
    PetscCall(MatDestroy(&matSquare));
    PetscCall(ISDestroy(&rowis));
  }
  /* Disgusting trick to reuse work vectors */
  PetscCall(MatGetLocalSize(multMat, &m, &n));
  patch->patchUpdate->map->n = n;
  patch->patchRHSWithArtificial->map->n = m;
  patch->patchUpdate->map->N = n;
  patch->patchRHSWithArtificial->map->N = m;
  PetscCall(MatMult(multMat, patch->patchUpdate, patch->patchRHSWithArtificial));
  PetscCall(VecScale(patch->patchRHSWithArtificial, -1.0));
  PetscCall(PCPatch_ScatterLocal_Private(pc, i + pStart, patch->patchRHSWithArtificial, patch->localRHS, ADD_VALUES, SCATTER_REVERSE, SCATTER_WITHARTIFICIAL));
  if (!patch->save_operators) {
    PetscCall(MatDestroy(&multMat));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PATCH(PC pc, Vec x, Vec y)
{
  PC_PATCH          *patch    = (PC_PATCH *) pc->data;
  const PetscScalar *globalRHS  = NULL;
  PetscScalar       *localRHS   = NULL;
  PetscScalar       *globalUpdate  = NULL;
  const PetscInt    *bcNodes  = NULL;
  PetscInt           nsweep   = patch->symmetrise_sweep ? 2 : 1;
  PetscInt           start[2] = {0, 0};
  PetscInt           end[2]   = {-1, -1};
  const PetscInt     inc[2]   = {1, -1};
  const PetscScalar *localUpdate;
  const PetscInt    *iterationSet;
  PetscInt           pStart, numBcs, n, sweep, bc, j;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PC_Patch_Apply, pc, 0, 0, 0));
  PetscCall(PetscOptionsPushGetViewerOff(PETSC_TRUE));
  /* start, end, inc have 2 entries to manage a second backward sweep if we symmetrize */
  end[0]   = patch->npatch;
  start[1] = patch->npatch-1;
  if (patch->user_patches) {
    PetscCall(ISGetLocalSize(patch->iterationSet, &end[0]));
    start[1] = end[0] - 1;
    PetscCall(ISGetIndices(patch->iterationSet, &iterationSet));
  }
  /* Scatter from global space into overlapped local spaces */
  PetscCall(VecGetArrayRead(x, &globalRHS));
  PetscCall(VecGetArray(patch->localRHS, &localRHS));
  PetscCall(PetscSFBcastBegin(patch->sectionSF, MPIU_SCALAR, globalRHS, localRHS,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(patch->sectionSF, MPIU_SCALAR, globalRHS, localRHS,MPI_REPLACE));
  PetscCall(VecRestoreArrayRead(x, &globalRHS));
  PetscCall(VecRestoreArray(patch->localRHS, &localRHS));

  PetscCall(VecSet(patch->localUpdate, 0.0));
  PetscCall(PetscSectionGetChart(patch->gtolCounts, &pStart, NULL));
  PetscCall(PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0));
  for (sweep = 0; sweep < nsweep; sweep++) {
    for (j = start[sweep]; j*inc[sweep] < end[sweep]*inc[sweep]; j += inc[sweep]) {
      PetscInt i = patch->user_patches ? iterationSet[j] : j;
      PetscInt start, len;

      PetscCall(PetscSectionGetDof(patch->gtolCounts, i+pStart, &len));
      PetscCall(PetscSectionGetOffset(patch->gtolCounts, i+pStart, &start));
      /* TODO: Squash out these guys in the setup as well. */
      if (len <= 0) continue;
      /* TODO: Do we need different scatters for X and Y? */
      PetscCall(PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localRHS, patch->patchRHS, INSERT_VALUES, SCATTER_FORWARD, SCATTER_INTERIOR));
      PetscCall((*patch->applysolver)(pc, i, patch->patchRHS, patch->patchUpdate));
      PetscCall(PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchUpdate, patch->localUpdate, ADD_VALUES, SCATTER_REVERSE, SCATTER_INTERIOR));
      if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
        PetscCall((*patch->updatemultiplicative)(pc, i, pStart));
      }
    }
  }
  PetscCall(PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0));
  if (patch->user_patches) PetscCall(ISRestoreIndices(patch->iterationSet, &iterationSet));
  /* XXX: should we do this on the global vector? */
  if (patch->partition_of_unity) {
    PetscCall(VecPointwiseMult(patch->localUpdate, patch->localUpdate, patch->dof_weights));
  }
  /* Now patch->localUpdate contains the solution of the patch solves, so we need to combine them all. */
  PetscCall(VecSet(y, 0.0));
  PetscCall(VecGetArray(y, &globalUpdate));
  PetscCall(VecGetArrayRead(patch->localUpdate, &localUpdate));
  PetscCall(PetscSFReduceBegin(patch->sectionSF, MPIU_SCALAR, localUpdate, globalUpdate, MPI_SUM));
  PetscCall(PetscSFReduceEnd(patch->sectionSF, MPIU_SCALAR, localUpdate, globalUpdate, MPI_SUM));
  PetscCall(VecRestoreArrayRead(patch->localUpdate, &localUpdate));

  /* Now we need to send the global BC values through */
  PetscCall(VecGetArrayRead(x, &globalRHS));
  PetscCall(ISGetSize(patch->globalBcNodes, &numBcs));
  PetscCall(ISGetIndices(patch->globalBcNodes, &bcNodes));
  PetscCall(VecGetLocalSize(x, &n));
  for (bc = 0; bc < numBcs; ++bc) {
    const PetscInt idx = bcNodes[bc];
    if (idx < n) globalUpdate[idx] = globalRHS[idx];
  }

  PetscCall(ISRestoreIndices(patch->globalBcNodes, &bcNodes));
  PetscCall(VecRestoreArrayRead(x, &globalRHS));
  PetscCall(VecRestoreArray(y, &globalUpdate));

  PetscCall(PetscOptionsPopGetViewerOff());
  PetscCall(PetscLogEventEnd(PC_Patch_Apply, pc, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH_Linear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(KSPReset((KSP) patch->solver[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;

  PetscCall(PetscSFDestroy(&patch->sectionSF));
  PetscCall(PetscSectionDestroy(&patch->cellCounts));
  PetscCall(PetscSectionDestroy(&patch->pointCounts));
  PetscCall(PetscSectionDestroy(&patch->cellNumbering));
  PetscCall(PetscSectionDestroy(&patch->gtolCounts));
  PetscCall(ISDestroy(&patch->gtol));
  PetscCall(ISDestroy(&patch->cells));
  PetscCall(ISDestroy(&patch->points));
  PetscCall(ISDestroy(&patch->dofs));
  PetscCall(ISDestroy(&patch->offs));
  PetscCall(PetscSectionDestroy(&patch->patchSection));
  PetscCall(ISDestroy(&patch->ghostBcNodes));
  PetscCall(ISDestroy(&patch->globalBcNodes));
  PetscCall(PetscSectionDestroy(&patch->gtolCountsWithArtificial));
  PetscCall(ISDestroy(&patch->gtolWithArtificial));
  PetscCall(ISDestroy(&patch->dofsWithArtificial));
  PetscCall(ISDestroy(&patch->offsWithArtificial));
  PetscCall(PetscSectionDestroy(&patch->gtolCountsWithAll));
  PetscCall(ISDestroy(&patch->gtolWithAll));
  PetscCall(ISDestroy(&patch->dofsWithAll));
  PetscCall(ISDestroy(&patch->offsWithAll));
  PetscCall(VecDestroy(&patch->cellMats));
  PetscCall(VecDestroy(&patch->intFacetMats));
  PetscCall(ISDestroy(&patch->allCells));
  PetscCall(ISDestroy(&patch->intFacets));
  PetscCall(ISDestroy(&patch->extFacets));
  PetscCall(ISDestroy(&patch->intFacetsToPatchCell));
  PetscCall(PetscSectionDestroy(&patch->intFacetCounts));
  PetscCall(PetscSectionDestroy(&patch->extFacetCounts));

  if (patch->dofSection) for (i = 0; i < patch->nsubspaces; i++) PetscCall(PetscSectionDestroy(&patch->dofSection[i]));
  PetscCall(PetscFree(patch->dofSection));
  PetscCall(PetscFree(patch->bs));
  PetscCall(PetscFree(patch->nodesPerCell));
  if (patch->cellNodeMap) for (i = 0; i < patch->nsubspaces; i++) PetscCall(PetscFree(patch->cellNodeMap[i]));
  PetscCall(PetscFree(patch->cellNodeMap));
  PetscCall(PetscFree(patch->subspaceOffsets));

  PetscCall((*patch->resetsolver)(pc));

  if (patch->subspaces_to_exclude) {
    PetscCall(PetscHSetIDestroy(&patch->subspaces_to_exclude));
  }

  PetscCall(VecDestroy(&patch->localRHS));
  PetscCall(VecDestroy(&patch->localUpdate));
  PetscCall(VecDestroy(&patch->patchRHS));
  PetscCall(VecDestroy(&patch->patchUpdate));
  PetscCall(VecDestroy(&patch->dof_weights));
  if (patch->patch_dof_weights) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(VecDestroy(&patch->patch_dof_weights[i]));
    PetscCall(PetscFree(patch->patch_dof_weights));
  }
  if (patch->mat) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(MatDestroy(&patch->mat[i]));
    PetscCall(PetscFree(patch->mat));
  }
  if (patch->matWithArtificial) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(MatDestroy(&patch->matWithArtificial[i]));
    PetscCall(PetscFree(patch->matWithArtificial));
  }
  PetscCall(VecDestroy(&patch->patchRHSWithArtificial));
  if (patch->dofMappingWithoutToWithArtificial) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(ISDestroy(&patch->dofMappingWithoutToWithArtificial[i]));
    PetscCall(PetscFree(patch->dofMappingWithoutToWithArtificial));

  }
  if (patch->dofMappingWithoutToWithAll) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(ISDestroy(&patch->dofMappingWithoutToWithAll[i]));
    PetscCall(PetscFree(patch->dofMappingWithoutToWithAll));

  }
  PetscCall(PetscFree(patch->sub_mat_type));
  if (patch->userIS) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(ISDestroy(&patch->userIS[i]));
    PetscCall(PetscFree(patch->userIS));
  }
  PetscCall(PetscFree(patch->precomputedTensorLocations));
  PetscCall(PetscFree(patch->precomputedIntFacetTensorLocations));

  patch->bs          = NULL;
  patch->cellNodeMap = NULL;
  patch->nsubspaces  = 0;
  PetscCall(ISDestroy(&patch->iterationSet));

  PetscCall(PetscViewerDestroy(&patch->viewerSection));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH_Linear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) PetscCall(KSPDestroy((KSP *) &patch->solver[i]));
    PetscCall(PetscFree(patch->solver));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_PATCH(pc));
  PetscCall((*patch->destroysolver)(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_PATCH(PetscOptionItems *PetscOptionsObject, PC pc)
{
  PC_PATCH            *patch = (PC_PATCH *) pc->data;
  PCPatchConstructType patchConstructionType = PC_PATCH_STAR;
  char                 sub_mat_type[PETSC_MAX_PATH_LEN];
  char                 option[PETSC_MAX_PATH_LEN];
  const char          *prefix;
  PetscBool            flg, dimflg, codimflg;
  MPI_Comm             comm;
  PetscInt            *ifields, nfields, k;
  PCCompositeType      loctype = PC_COMPOSITE_ADDITIVE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) pc, &comm));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject) pc, &prefix));
  PetscCall(PetscOptionsHead(PetscOptionsObject, "Patch solver options"));

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_save_operators", patch->classname));
  PetscCall(PetscOptionsBool(option,  "Store all patch operators for lifetime of object?", "PCPatchSetSaveOperators", patch->save_operators, &patch->save_operators, &flg));

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_precompute_element_tensors", patch->classname));
  PetscCall(PetscOptionsBool(option,  "Compute each element tensor only once?", "PCPatchSetPrecomputeElementTensors", patch->precomputeElementTensors, &patch->precomputeElementTensors, &flg));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_partition_of_unity", patch->classname));
  PetscCall(PetscOptionsBool(option, "Weight contributions by dof multiplicity?", "PCPatchSetPartitionOfUnity", patch->partition_of_unity, &patch->partition_of_unity, &flg));

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_local_type", patch->classname));
  PetscCall(PetscOptionsEnum(option,"Type of local solver composition (additive or multiplicative)","PCPatchSetLocalComposition",PCCompositeTypes,(PetscEnum)loctype,(PetscEnum*)&loctype,&flg));
  if (flg) PetscCall(PCPatchSetLocalComposition(pc, loctype));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_dense_inverse", patch->classname));
  PetscCall(PetscOptionsBool(option, "Compute inverses of patch matrices and apply directly? Ignores KSP/PC settings on patch.", "PCPatchSetDenseInverse", patch->denseinverse, &patch->denseinverse, &flg));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_construct_dim", patch->classname));
  PetscCall(PetscOptionsInt(option, "What dimension of mesh point to construct patches by? (0 = vertices)", "PCPATCH", patch->dim, &patch->dim, &dimflg));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_construct_codim", patch->classname));
  PetscCall(PetscOptionsInt(option, "What co-dimension of mesh point to construct patches by? (0 = cells)", "PCPATCH", patch->codim, &patch->codim, &codimflg));
  PetscCheckFalse(dimflg && codimflg,comm, PETSC_ERR_ARG_WRONG, "Can only set one of dimension or co-dimension");

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_construct_type", patch->classname));
  PetscCall(PetscOptionsEnum(option, "How should the patches be constructed?", "PCPatchSetConstructType", PCPatchConstructTypes, (PetscEnum) patchConstructionType, (PetscEnum *) &patchConstructionType, &flg));
  if (flg) PetscCall(PCPatchSetConstructType(pc, patchConstructionType, NULL, NULL));

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_vanka_dim", patch->classname));
  PetscCall(PetscOptionsInt(option, "Topological dimension of entities for Vanka to ignore", "PCPATCH", patch->vankadim, &patch->vankadim, &flg));

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_ignore_dim", patch->classname));
  PetscCall(PetscOptionsInt(option, "Topological dimension of entities for completion to ignore", "PCPATCH", patch->ignoredim, &patch->ignoredim, &flg));

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_pardecomp_overlap", patch->classname));
  PetscCall(PetscOptionsInt(option, "What overlap should we use in construct type pardecomp?", "PCPATCH", patch->pardecomp_overlap, &patch->pardecomp_overlap, &flg));

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_sub_mat_type", patch->classname));
  PetscCall(PetscOptionsFList(option, "Matrix type for patch solves", "PCPatchSetSubMatType", MatList, NULL, sub_mat_type, PETSC_MAX_PATH_LEN, &flg));
  if (flg) PetscCall(PCPatchSetSubMatType(pc, sub_mat_type));

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_symmetrise_sweep", patch->classname));
  PetscCall(PetscOptionsBool(option, "Go start->end, end->start?", "PCPATCH", patch->symmetrise_sweep, &patch->symmetrise_sweep, &flg));

  /* If the user has set the number of subspaces, use that for the buffer size,
     otherwise use a large number */
  if (patch->nsubspaces <= 0) {
    nfields = 128;
  } else {
    nfields = patch->nsubspaces;
  }
  PetscCall(PetscMalloc1(nfields, &ifields));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_exclude_subspaces", patch->classname));
  PetscCall(PetscOptionsGetIntArray(((PetscObject)pc)->options,((PetscObject)pc)->prefix,option,ifields,&nfields,&flg));
  PetscCheckFalse(flg && (patchConstructionType == PC_PATCH_USER),comm, PETSC_ERR_ARG_INCOMP, "We cannot support excluding a subspace with user patches because we do not index patches with a mesh point");
  if (flg) {
    PetscCall(PetscHSetIClear(patch->subspaces_to_exclude));
    for (k = 0; k < nfields; k++) {
      PetscCall(PetscHSetIAdd(patch->subspaces_to_exclude, ifields[k]));
    }
  }
  PetscCall(PetscFree(ifields));

  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_patches_view", patch->classname));
  PetscCall(PetscOptionsBool(option, "Print out information during patch construction", "PCPATCH", patch->viewPatches, &patch->viewPatches, &flg));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_cells_view", patch->classname));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerCells, &patch->formatCells, &patch->viewCells));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_interior_facets_view", patch->classname));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerIntFacets, &patch->formatIntFacets, &patch->viewIntFacets));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_exterior_facets_view", patch->classname));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerExtFacets, &patch->formatExtFacets, &patch->viewExtFacets));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_points_view", patch->classname));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerPoints, &patch->formatPoints, &patch->viewPoints));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_section_view", patch->classname));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerSection, &patch->formatSection, &patch->viewSection));
  PetscCall(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_mat_view", patch->classname));
  PetscCall(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerMatrix, &patch->formatMatrix, &patch->viewMatrix));
  PetscCall(PetscOptionsTail());
  patch->optionsSet = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUpOnBlocks_PATCH(PC pc)
{
  PC_PATCH          *patch = (PC_PATCH*) pc->data;
  KSPConvergedReason reason;
  PetscInt           i;

  PetscFunctionBegin;
  if (!patch->save_operators) {
    /* Can't do this here because the sub KSPs don't have an operator attached yet. */
    PetscFunctionReturn(0);
  }
  if (patch->denseinverse) {
    /* No solvers */
    PetscFunctionReturn(0);
  }
  for (i = 0; i < patch->npatch; ++i) {
    if (!((KSP) patch->solver[i])->setfromoptionscalled) {
      PetscCall(KSPSetFromOptions((KSP) patch->solver[i]));
    }
    PetscCall(KSPSetUp((KSP) patch->solver[i]));
    PetscCall(KSPGetConvergedReason((KSP) patch->solver[i], &reason));
    if (reason == KSP_DIVERGED_PC_FAILED) pc->failedreason = PC_SUBPC_ERROR;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_PATCH(PC pc, PetscViewer viewer)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscViewer    sviewer;
  PetscBool      isascii;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  /* TODO Redo tabbing with set tbas in new style */
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  if (!isascii) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) pc), &rank));
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Subspace Correction preconditioner with %d patches\n", patch->npatch));
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Schwarz type: multiplicative\n"));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Schwarz type: additive\n"));
  }
  if (patch->partition_of_unity) PetscCall(PetscViewerASCIIPrintf(viewer, "Weighting by partition of unity\n"));
  else                           PetscCall(PetscViewerASCIIPrintf(viewer, "Not weighting by partition of unity\n"));
  if (patch->symmetrise_sweep) PetscCall(PetscViewerASCIIPrintf(viewer, "Symmetrising sweep (start->end, then end->start)\n"));
  else                         PetscCall(PetscViewerASCIIPrintf(viewer, "Not symmetrising sweep\n"));
  if (!patch->precomputeElementTensors) PetscCall(PetscViewerASCIIPrintf(viewer, "Not precomputing element tensors (overlapping cells rebuilt in every patch assembly)\n"));
  else                            PetscCall(PetscViewerASCIIPrintf(viewer, "Precomputing element tensors (each cell assembled only once)\n"));
  if (!patch->save_operators) PetscCall(PetscViewerASCIIPrintf(viewer, "Not saving patch operators (rebuilt every PCApply)\n"));
  else                        PetscCall(PetscViewerASCIIPrintf(viewer, "Saving patch operators (rebuilt every PCSetUp)\n"));
  if (patch->patchconstructop == PCPatchConstruct_Star)       PetscCall(PetscViewerASCIIPrintf(viewer, "Patch construction operator: star\n"));
  else if (patch->patchconstructop == PCPatchConstruct_Vanka) PetscCall(PetscViewerASCIIPrintf(viewer, "Patch construction operator: Vanka\n"));
  else if (patch->patchconstructop == PCPatchConstruct_User)  PetscCall(PetscViewerASCIIPrintf(viewer, "Patch construction operator: user-specified\n"));
  else                                                        PetscCall(PetscViewerASCIIPrintf(viewer, "Patch construction operator: unknown\n"));

  if (patch->denseinverse) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Explicitly forming dense inverse and applying patch solver via MatMult.\n"));
  } else {
    if (patch->isNonlinear) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "SNES on patches (all same):\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "KSP on patches (all same):\n"));
    }
    if (patch->solver) {
      PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
      if (rank == 0) {
        PetscCall(PetscViewerASCIIPushTab(sviewer));
        PetscCall(PetscObjectView(patch->solver[0], sviewer));
        PetscCall(PetscViewerASCIIPopTab(sviewer));
      }
      PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
    } else {
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Solver not yet set.\n"));
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
  }
  PetscCall(PetscViewerASCIIPopTab(viewer));
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

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc, &patch));

  if (patch->subspaces_to_exclude) {
    PetscCall(PetscHSetIDestroy(&patch->subspaces_to_exclude));
  }
  PetscCall(PetscHSetICreate(&patch->subspaces_to_exclude));

  patch->classname = "pc";
  patch->isNonlinear = PETSC_FALSE;

  /* Set some defaults */
  patch->combined           = PETSC_FALSE;
  patch->save_operators     = PETSC_TRUE;
  patch->local_composition_type = PC_COMPOSITE_ADDITIVE;
  patch->precomputeElementTensors = PETSC_FALSE;
  patch->partition_of_unity = PETSC_FALSE;
  patch->codim              = -1;
  patch->dim                = -1;
  patch->vankadim           = -1;
  patch->ignoredim          = -1;
  patch->pardecomp_overlap  = 0;
  patch->patchconstructop   = PCPatchConstruct_Star;
  patch->symmetrise_sweep   = PETSC_FALSE;
  patch->npatch             = 0;
  patch->userIS             = NULL;
  patch->optionsSet         = PETSC_FALSE;
  patch->iterationSet       = NULL;
  patch->user_patches       = PETSC_FALSE;
  PetscCall(PetscStrallocpy(MATDENSE, (char **) &patch->sub_mat_type));
  patch->viewPatches        = PETSC_FALSE;
  patch->viewCells          = PETSC_FALSE;
  patch->viewPoints         = PETSC_FALSE;
  patch->viewSection        = PETSC_FALSE;
  patch->viewMatrix         = PETSC_FALSE;
  patch->densesolve         = NULL;
  patch->setupsolver        = PCSetUp_PATCH_Linear;
  patch->applysolver        = PCApply_PATCH_Linear;
  patch->resetsolver        = PCReset_PATCH_Linear;
  patch->destroysolver      = PCDestroy_PATCH_Linear;
  patch->updatemultiplicative = PCUpdateMultiplicative_PATCH_Linear;
  patch->dofMappingWithoutToWithArtificial = NULL;
  patch->dofMappingWithoutToWithAll = NULL;

  pc->data                 = (void *) patch;
  pc->ops->apply           = PCApply_PATCH;
  pc->ops->applytranspose  = NULL; /* PCApplyTranspose_PATCH; */
  pc->ops->setup           = PCSetUp_PATCH;
  pc->ops->reset           = PCReset_PATCH;
  pc->ops->destroy         = PCDestroy_PATCH;
  pc->ops->setfromoptions  = PCSetFromOptions_PATCH;
  pc->ops->setuponblocks   = PCSetUpOnBlocks_PATCH;
  pc->ops->view            = PCView_PATCH;
  pc->ops->applyrichardson = NULL;

  PetscFunctionReturn(0);
}
