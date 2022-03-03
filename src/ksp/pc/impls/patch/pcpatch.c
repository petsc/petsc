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

  CHKERRQ(PetscViewerPushFormat(viewer, format));
  CHKERRQ(PetscObjectView(obj, viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));
  return(0);
}

static PetscErrorCode PCPatchConstruct_Star(void *vpatch, DM dm, PetscInt point, PetscHSetI ht)
{
  PetscInt       starSize;
  PetscInt      *star = NULL, si;

  PetscFunctionBegin;
  CHKERRQ(PetscHSetIClear(ht));
  /* To start with, add the point we care about */
  CHKERRQ(PetscHSetIAdd(ht, point));
  /* Loop over all the points that this point connects to */
  CHKERRQ(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
  for (si = 0; si < starSize*2; si += 2) CHKERRQ(PetscHSetIAdd(ht, star[si]));
  CHKERRQ(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
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
  CHKERRQ(PetscHSetIClear(ht));
  /* To start with, add the point we care about */
  CHKERRQ(PetscHSetIAdd(ht, point));
  /* Should we ignore any points of a certain dimension? */
  if (patch->vankadim >= 0) {
    shouldIgnore = PETSC_TRUE;
    CHKERRQ(DMPlexGetDepthStratum(dm, patch->vankadim, &iStart, &iEnd));
  }
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  /* Loop over all the cells that this point connects to */
  CHKERRQ(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
  for (si = 0; si < starSize*2; si += 2) {
    const PetscInt cell = star[si];
    PetscInt       closureSize;
    PetscInt      *closure = NULL, ci;

    if (cell < cStart || cell >= cEnd) continue;
    /* now loop over all entities in the closure of that cell */
    CHKERRQ(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
    for (ci = 0; ci < closureSize*2; ci += 2) {
      const PetscInt newpoint = closure[ci];

      /* We've been told to ignore entities of this type.*/
      if (shouldIgnore && newpoint >= iStart && newpoint < iEnd) continue;
      CHKERRQ(PetscHSetIAdd(ht, newpoint));
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure));
  }
  CHKERRQ(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
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
  CHKERRQ(PetscHSetIClear(ht));

  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));

  CHKERRQ(DMHasLabel(dm, "pyop2_ghost", &isFiredrake));
  if (isFiredrake) {
    CHKERRQ(DMGetLabel(dm, "pyop2_ghost", &ghost));
    CHKERRQ(DMLabelCreateIndex(ghost, pStart, pEnd));
  } else {
    PetscSF sf;
    CHKERRQ(DMGetPointSF(dm, &sf));
    CHKERRQ(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL));
    nleaves = PetscMax(nleaves, 0);
  }

  for (opoint = pStart; opoint < pEnd; ++opoint) {
    if (ghost) CHKERRQ(DMLabelHasPoint(ghost, opoint, &flg));
    else       {CHKERRQ(PetscFindInt(opoint, nleaves, leaves, &loc)); flg = loc >=0 ? PETSC_TRUE : PETSC_FALSE;}
    /* Not an owned entity, don't make a cell patch. */
    if (flg) continue;
    CHKERRQ(PetscHSetIAdd(ht, opoint));
  }

  /* Now build the overlap for the patch */
  for (overlapi = 0; overlapi < patch->pardecomp_overlap; ++overlapi) {
    PetscInt index = 0;
    PetscInt *htpoints = NULL;
    PetscInt htsize;
    PetscInt i;

    CHKERRQ(PetscHSetIGetSize(ht, &htsize));
    CHKERRQ(PetscMalloc1(htsize, &htpoints));
    CHKERRQ(PetscHSetIGetElems(ht, &index, htpoints));

    for (i = 0; i < htsize; ++i) {
      PetscInt hpoint = htpoints[i];
      PetscInt si;

      CHKERRQ(DMPlexGetTransitiveClosure(dm, hpoint, PETSC_FALSE, &starSize, &star));
      for (si = 0; si < starSize*2; si += 2) {
        const PetscInt starp = star[si];
        PetscInt       closureSize;
        PetscInt      *closure = NULL, ci;

        /* now loop over all entities in the closure of starp */
        CHKERRQ(DMPlexGetTransitiveClosure(dm, starp, PETSC_TRUE, &closureSize, &closure));
        for (ci = 0; ci < closureSize*2; ci += 2) {
          const PetscInt closstarp = closure[ci];
          CHKERRQ(PetscHSetIAdd(ht, closstarp));
        }
        CHKERRQ(DMPlexRestoreTransitiveClosure(dm, starp, PETSC_TRUE, &closureSize, &closure));
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, hpoint, PETSC_FALSE, &starSize, &star));
    }
    CHKERRQ(PetscFree(htpoints));
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
  CHKERRQ(PetscHSetIClear(ht));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(ISGetLocalSize(patchis, &n));
  CHKERRQ(ISGetIndices(patchis, &patchdata));
  for (i = 0; i < n; ++i) {
    const PetscInt ownedpoint = patchdata[i];

    if (ownedpoint < pStart || ownedpoint >= pEnd) {
      SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "Mesh point %D was not in [%D, %D)", ownedpoint, pStart, pEnd);
    }
    CHKERRQ(PetscHSetIAdd(ht, ownedpoint));
  }
  CHKERRQ(ISRestoreIndices(patchis, &patchdata));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchCreateDefaultSF_Private(PC pc, PetscInt n, const PetscSF *sf, const PetscInt *bs)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (n == 1 && bs[0] == 1) {
    patch->sectionSF = sf[0];
    CHKERRQ(PetscObjectReference((PetscObject) patch->sectionSF));
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

      CHKERRQ(PetscSFGetGraph(sf[i], &nroots, &nleaves, NULL, NULL));
      allRoots  += nroots * bs[i];
      allLeaves += nleaves * bs[i];
    }
    CHKERRQ(PetscMalloc1(allLeaves, &ilocal));
    CHKERRQ(PetscMalloc1(allLeaves, &iremote));
    /* Now build an SF that just contains process connectivity. */
    CHKERRQ(PetscHSetICreate(&ranksUniq));
    for (i = 0; i < n; ++i) {
      const PetscMPIInt *ranks = NULL;
      PetscInt           nranks, j;

      CHKERRQ(PetscSFSetUp(sf[i]));
      CHKERRQ(PetscSFGetRootRanks(sf[i], &nranks, &ranks, NULL, NULL, NULL));
      /* These are all the ranks who communicate with me. */
      for (j = 0; j < nranks; ++j) {
        CHKERRQ(PetscHSetIAdd(ranksUniq, (PetscInt) ranks[j]));
      }
    }
    CHKERRQ(PetscHSetIGetSize(ranksUniq, &numRanks));
    CHKERRQ(PetscMalloc1(numRanks, &remote));
    CHKERRQ(PetscMalloc1(numRanks, &ranks));
    CHKERRQ(PetscHSetIGetElems(ranksUniq, &index, ranks));

    CHKERRQ(PetscHMapICreate(&rankToIndex));
    for (i = 0; i < numRanks; ++i) {
      remote[i].rank  = ranks[i];
      remote[i].index = 0;
      CHKERRQ(PetscHMapISet(rankToIndex, ranks[i], i));
    }
    CHKERRQ(PetscFree(ranks));
    CHKERRQ(PetscHSetIDestroy(&ranksUniq));
    CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject) pc), &rankSF));
    CHKERRQ(PetscSFSetGraph(rankSF, 1, numRanks, NULL, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER));
    CHKERRQ(PetscSFSetUp(rankSF));
    /* OK, use it to communicate the root offset on the remote
     * processes for each subspace. */
    CHKERRQ(PetscMalloc1(n, &offsets));
    CHKERRQ(PetscMalloc1(n*numRanks, &remoteOffsets));

    offsets[0] = 0;
    for (i = 1; i < n; ++i) {
      PetscInt nroots;

      CHKERRQ(PetscSFGetGraph(sf[i-1], &nroots, NULL, NULL, NULL));
      offsets[i] = offsets[i-1] + nroots*bs[i-1];
    }
    /* Offsets are the offsets on the current process of the
     * global dof numbering for the subspaces. */
    CHKERRMPI(MPI_Type_contiguous(n, MPIU_INT, &contig));
    CHKERRMPI(MPI_Type_commit(&contig));

    CHKERRQ(PetscSFBcastBegin(rankSF, contig, offsets, remoteOffsets,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(rankSF, contig, offsets, remoteOffsets,MPI_REPLACE));
    CHKERRMPI(MPI_Type_free(&contig));
    CHKERRQ(PetscFree(offsets));
    CHKERRQ(PetscSFDestroy(&rankSF));
    /* Now remoteOffsets contains the offsets on the remote
     * processes who communicate with me.  So now we can
     * concatenate the list of SFs into a single one. */
    index = 0;
    for (i = 0; i < n; ++i) {
      const PetscSFNode *remote = NULL;
      const PetscInt    *local  = NULL;
      PetscInt           nroots, nleaves, j;

      CHKERRQ(PetscSFGetGraph(sf[i], &nroots, &nleaves, &local, &remote));
      for (j = 0; j < nleaves; ++j) {
        PetscInt rank = remote[j].rank;
        PetscInt idx, rootOffset, k;

        CHKERRQ(PetscHMapIGet(rankToIndex, rank, &idx));
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
    CHKERRQ(PetscHMapIDestroy(&rankToIndex));
    CHKERRQ(PetscFree(remoteOffsets));
    CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)pc), &patch->sectionSF));
    CHKERRQ(PetscSFSetGraph(patch->sectionSF, allRoots, allLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
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
  if (patch->sub_mat_type) CHKERRQ(PetscFree(patch->sub_mat_type));
  CHKERRQ(PetscStrallocpy(sub_mat_type, (char **) &patch->sub_mat_type));
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
  CHKERRQ(PetscObjectReference((PetscObject) cellNumbering));
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
  CHKERRQ(PCGetDM(pc, &dm));
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(PetscMalloc1(nsubspaces, &sfs));
  CHKERRQ(PetscMalloc1(nsubspaces, &patch->dofSection));
  CHKERRQ(PetscMalloc1(nsubspaces, &patch->bs));
  CHKERRQ(PetscMalloc1(nsubspaces, &patch->nodesPerCell));
  CHKERRQ(PetscMalloc1(nsubspaces, &patch->cellNodeMap));
  CHKERRQ(PetscMalloc1(nsubspaces+1, &patch->subspaceOffsets));

  patch->nsubspaces       = nsubspaces;
  patch->totalDofsPerCell = 0;
  for (i = 0; i < nsubspaces; ++i) {
    CHKERRQ(DMGetLocalSection(dms[i], &patch->dofSection[i]));
    CHKERRQ(PetscObjectReference((PetscObject) patch->dofSection[i]));
    CHKERRQ(DMGetSectionSF(dms[i], &sfs[i]));
    patch->bs[i]              = bs[i];
    patch->nodesPerCell[i]    = nodesPerCell[i];
    patch->totalDofsPerCell  += nodesPerCell[i]*bs[i];
    CHKERRQ(PetscMalloc1((cEnd-cStart)*nodesPerCell[i], &patch->cellNodeMap[i]));
    for (j = 0; j < (cEnd-cStart)*nodesPerCell[i]; ++j) patch->cellNodeMap[i][j] = cellNodeMap[i][j];
    patch->subspaceOffsets[i] = subspaceOffsets[i];
  }
  CHKERRQ(PCPatchCreateDefaultSF_Private(pc, nsubspaces, sfs, patch->bs));
  CHKERRQ(PetscFree(sfs));

  patch->subspaceOffsets[nsubspaces] = subspaceOffsets[nsubspaces];
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numGhostBcs, ghostBcNodes, PETSC_COPY_VALUES, &patch->ghostBcNodes));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numGlobalBcs, globalBcNodes, PETSC_COPY_VALUES, &patch->globalBcNodes));
  CHKERRQ(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

/* TODO: Docs */
PetscErrorCode PCPatchSetDiscretisationInfoCombined(PC pc, DM dm, PetscInt *nodesPerCell, const PetscInt **cellNodeMap, PetscInt numGhostBcs, const PetscInt *ghostBcNodes, PetscInt numGlobalBcs, const PetscInt *globalBcNodes)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       cStart, cEnd, i, j;

  PetscFunctionBegin;
  patch->combined = PETSC_TRUE;
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMGetNumFields(dm, &patch->nsubspaces));
  CHKERRQ(PetscCalloc1(patch->nsubspaces, &patch->dofSection));
  CHKERRQ(PetscMalloc1(patch->nsubspaces, &patch->bs));
  CHKERRQ(PetscMalloc1(patch->nsubspaces, &patch->nodesPerCell));
  CHKERRQ(PetscMalloc1(patch->nsubspaces, &patch->cellNodeMap));
  CHKERRQ(PetscCalloc1(patch->nsubspaces+1, &patch->subspaceOffsets));
  CHKERRQ(DMGetLocalSection(dm, &patch->dofSection[0]));
  CHKERRQ(PetscObjectReference((PetscObject) patch->dofSection[0]));
  CHKERRQ(PetscSectionGetStorageSize(patch->dofSection[0], &patch->subspaceOffsets[patch->nsubspaces]));
  patch->totalDofsPerCell = 0;
  for (i = 0; i < patch->nsubspaces; ++i) {
    patch->bs[i]             = 1;
    patch->nodesPerCell[i]   = nodesPerCell[i];
    patch->totalDofsPerCell += nodesPerCell[i];
    CHKERRQ(PetscMalloc1((cEnd-cStart)*nodesPerCell[i], &patch->cellNodeMap[i]));
    for (j = 0; j < (cEnd-cStart)*nodesPerCell[i]; ++j) patch->cellNodeMap[i][j] = cellNodeMap[i][j];
  }
  CHKERRQ(DMGetSectionSF(dm, &patch->sectionSF));
  CHKERRQ(PetscObjectReference((PetscObject) patch->sectionSF));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numGhostBcs, ghostBcNodes, PETSC_COPY_VALUES, &patch->ghostBcNodes));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numGlobalBcs, globalBcNodes, PETSC_COPY_VALUES, &patch->globalBcNodes));
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
  CHKERRQ(PCGetDM(pc, &dm));
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fBegin, &fEnd));
  CHKERRQ(PCPatchGetIgnoreDim(pc, &ignoredim));
  if (ignoredim >= 0) CHKERRQ(DMPlexGetDepthStratum(dm, ignoredim, &iStart, &iEnd));
  CHKERRQ(PetscHSetIClear(cht));
  PetscHashIterBegin(ht, hi);
  while (!PetscHashIterAtEnd(ht, hi)) {

    PetscHashIterGetKey(ht, hi, point);
    PetscHashIterNext(ht, hi);

    /* Loop over all the cells that this point connects to */
    CHKERRQ(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &starSize, &star));
    for (si = 0; si < starSize*2; si += 2) {
      const PetscInt ownedpoint = star[si];
      /* TODO Check for point in cht before running through closure again */
      /* now loop over all entities in the closure of that cell */
      CHKERRQ(DMPlexGetTransitiveClosure(dm, ownedpoint, PETSC_TRUE, &closureSize, &closure));
      for (ci = 0; ci < closureSize*2; ci += 2) {
        const PetscInt seenpoint = closure[ci];
        if (ignoredim >= 0 && seenpoint >= iStart && seenpoint < iEnd) continue;
        CHKERRQ(PetscHSetIAdd(cht, seenpoint));
        /* Facet integrals couple dofs across facets, so in that case for each of
         * the facets we need to add all dofs on the other side of the facet to
         * the seen dofs. */
        if (patch->usercomputeopintfacet) {
          if (fBegin <= seenpoint && seenpoint < fEnd) {
            CHKERRQ(DMPlexGetTransitiveClosure(dm, seenpoint, PETSC_FALSE, &fStarSize, &fStar));
            for (fsi = 0; fsi < fStarSize*2; fsi += 2) {
              CHKERRQ(DMPlexGetTransitiveClosure(dm, fStar[fsi], PETSC_TRUE, &fClosureSize, &fClosure));
              for (fci = 0; fci < fClosureSize*2; fci += 2) {
                CHKERRQ(PetscHSetIAdd(cht, fClosure[fci]));
              }
              CHKERRQ(DMPlexRestoreTransitiveClosure(dm, fStar[fsi], PETSC_TRUE, NULL, &fClosure));
            }
            CHKERRQ(DMPlexRestoreTransitiveClosure(dm, seenpoint, PETSC_FALSE, NULL, &fStar));
          }
        }
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, ownedpoint, PETSC_TRUE, NULL, &closure));
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, NULL, &star));
  }
  CHKERRQ(DMDestroy(&dm));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchGetGlobalDofs(PC pc, PetscSection dofSection[], PetscInt f, PetscBool combined, PetscInt p, PetscInt *dof, PetscInt *off)
{
  PetscFunctionBegin;
  if (combined) {
    if (f < 0) {
      if (dof) CHKERRQ(PetscSectionGetDof(dofSection[0], p, dof));
      if (off) CHKERRQ(PetscSectionGetOffset(dofSection[0], p, off));
    } else {
      if (dof) CHKERRQ(PetscSectionGetFieldDof(dofSection[0], p, f, dof));
      if (off) CHKERRQ(PetscSectionGetFieldOffset(dofSection[0], p, f, off));
    }
  } else {
    if (f < 0) {
      PC_PATCH *patch = (PC_PATCH *) pc->data;
      PetscInt  fdof, g;

      if (dof) {
        *dof = 0;
        for (g = 0; g < patch->nsubspaces; ++g) {
          CHKERRQ(PetscSectionGetDof(dofSection[g], p, &fdof));
          *dof += fdof;
        }
      }
      if (off) {
        *off = 0;
        for (g = 0; g < patch->nsubspaces; ++g) {
          CHKERRQ(PetscSectionGetOffset(dofSection[g], p, &fdof));
          *off += fdof;
        }
      }
    } else {
      if (dof) CHKERRQ(PetscSectionGetDof(dofSection[f], p, dof));
      if (off) CHKERRQ(PetscSectionGetOffset(dofSection[f], p, off));
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
  CHKERRQ(PetscHSetIClear(dofs));
  for (k = 0; k < patch->nsubspaces; ++k) {
    PetscInt subspaceOffset = patch->subspaceOffsets[k];
    PetscInt bs             = patch->bs[k];
    PetscInt j, l;

    if (subspaces_to_exclude != NULL) {
      PetscBool should_exclude_k = PETSC_FALSE;
      CHKERRQ(PetscHSetIHas(*subspaces_to_exclude, k, &should_exclude_k));
      if (should_exclude_k) {
        /* only get this subspace dofs at the base entity, not any others */
        CHKERRQ(PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, base, &ldof, &loff));
        if (0 == ldof) continue;
        for (j = loff; j < ldof + loff; ++j) {
          for (l = 0; l < bs; ++l) {
            PetscInt dof = bs*j + l + subspaceOffset;
            CHKERRQ(PetscHSetIAdd(dofs, dof));
          }
        }
        continue; /* skip the other dofs of this subspace */
      }
    }

    PetscHashIterBegin(pts, hi);
    while (!PetscHashIterAtEnd(pts, hi)) {
      PetscHashIterGetKey(pts, hi, p);
      PetscHashIterNext(pts, hi);
      CHKERRQ(PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, p, &ldof, &loff));
      if (0 == ldof) continue;
      for (j = loff; j < ldof + loff; ++j) {
        for (l = 0; l < bs; ++l) {
          PetscInt dof = bs*j + l + subspaceOffset;
          CHKERRQ(PetscHSetIAdd(dofs, dof));
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
  CHKERRQ(PetscHSetIClear(C));
  PetscHashIterBegin(B, hi);
  while (!PetscHashIterAtEnd(B, hi)) {
    PetscHashIterGetKey(B, hi, key);
    PetscHashIterNext(B, hi);
    CHKERRQ(PetscHSetIHas(A, key, &flg));
    if (!flg) CHKERRQ(PetscHSetIAdd(C, key));
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
  CHKERRQ(PetscHSetICreate(&ht));
  CHKERRQ(PetscHSetICreate(&cht));

  CHKERRQ(PCGetDM(pc, &dm));
  PetscCheck(dm,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch PC");
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

  if (patch->user_patches) {
    CHKERRQ(patch->userpatchconstructionop(pc, &patch->npatch, &patch->userIS, &patch->iterationSet, patch->userpatchconstructctx));
    vStart = 0; vEnd = patch->npatch;
  } else if (patch->ctype == PC_PATCH_PARDECOMP) {
    vStart = 0; vEnd = 1;
  } else if (patch->codim < 0) {
    if (patch->dim < 0) CHKERRQ(DMPlexGetDepthStratum(dm,  0,            &vStart, &vEnd));
    else                CHKERRQ(DMPlexGetDepthStratum(dm,  patch->dim,   &vStart, &vEnd));
  } else                CHKERRQ(DMPlexGetHeightStratum(dm, patch->codim, &vStart, &vEnd));
  patch->npatch = vEnd - vStart;

  /* These labels mark the owned points.  We only create patches around points that this process owns. */
  CHKERRQ(DMHasLabel(dm, "pyop2_ghost", &isFiredrake));
  if (isFiredrake) {
    CHKERRQ(DMGetLabel(dm, "pyop2_ghost", &ghost));
    CHKERRQ(DMLabelCreateIndex(ghost, pStart, pEnd));
  } else {
    PetscSF sf;

    CHKERRQ(DMGetPointSF(dm, &sf));
    CHKERRQ(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, NULL));
    nleaves = PetscMax(nleaves, 0);
  }

  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &patch->cellCounts));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->cellCounts, "Patch Cell Layout"));
  cellCounts = patch->cellCounts;
  CHKERRQ(PetscSectionSetChart(cellCounts, vStart, vEnd));
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &patch->pointCounts));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->pointCounts, "Patch Point Layout"));
  pointCounts = patch->pointCounts;
  CHKERRQ(PetscSectionSetChart(pointCounts, vStart, vEnd));
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &patch->extFacetCounts));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->extFacetCounts, "Patch Exterior Facet Layout"));
  extFacetCounts = patch->extFacetCounts;
  CHKERRQ(PetscSectionSetChart(extFacetCounts, vStart, vEnd));
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &patch->intFacetCounts));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->intFacetCounts, "Patch Interior Facet Layout"));
  intFacetCounts = patch->intFacetCounts;
  CHKERRQ(PetscSectionSetChart(intFacetCounts, vStart, vEnd));
  /* Count cells and points in the patch surrounding each entity */
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIter hi;
    PetscInt       chtSize, loc = -1;
    PetscBool      flg;

    if (!patch->user_patches && patch->ctype != PC_PATCH_PARDECOMP) {
      if (ghost) CHKERRQ(DMLabelHasPoint(ghost, v, &flg));
      else       {CHKERRQ(PetscFindInt(v, nleaves, leaves, &loc)); flg = loc >=0 ? PETSC_TRUE : PETSC_FALSE;}
      /* Not an owned entity, don't make a cell patch. */
      if (flg) continue;
    }

    CHKERRQ(patch->patchconstructop((void *) patch, dm, v, ht));
    CHKERRQ(PCPatchCompleteCellPatch(pc, ht, cht));
    CHKERRQ(PetscHSetIGetSize(cht, &chtSize));
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
        CHKERRQ(DMPlexGetSupport(dm, point, &support));
        CHKERRQ(DMPlexGetSupportSize(dm, point, &supportSize));
        if (supportSize == 1) {
          interior = PETSC_FALSE;
        } else {
          for (p = 0; p < supportSize; p++) {
            PetscBool found;
            /* FIXME: can I do this while iterating over cht? */
            CHKERRQ(PetscHSetIHas(cht, support[p], &found));
            if (!found) {
              interior = PETSC_FALSE;
              break;
            }
          }
        }
        if (interior) {
          CHKERRQ(PetscSectionAddDof(intFacetCounts, v, 1));
        } else {
          CHKERRQ(PetscSectionAddDof(extFacetCounts, v, 1));
        }
      }
      CHKERRQ(PCPatchGetGlobalDofs(pc, patch->dofSection, -1, patch->combined, point, &pdof, NULL));
      if (pdof)                            CHKERRQ(PetscSectionAddDof(pointCounts, v, 1));
      if (point >= cStart && point < cEnd) CHKERRQ(PetscSectionAddDof(cellCounts, v, 1));
      PetscHashIterNext(cht, hi);
    }
  }
  if (isFiredrake) CHKERRQ(DMLabelDestroyIndex(ghost));

  CHKERRQ(PetscSectionSetUp(cellCounts));
  CHKERRQ(PetscSectionGetStorageSize(cellCounts, &numCells));
  CHKERRQ(PetscMalloc1(numCells, &cellsArray));
  CHKERRQ(PetscSectionSetUp(pointCounts));
  CHKERRQ(PetscSectionGetStorageSize(pointCounts, &numPoints));
  CHKERRQ(PetscMalloc1(numPoints, &pointsArray));

  CHKERRQ(PetscSectionSetUp(intFacetCounts));
  CHKERRQ(PetscSectionSetUp(extFacetCounts));
  CHKERRQ(PetscSectionGetStorageSize(intFacetCounts, &numIntFacets));
  CHKERRQ(PetscSectionGetStorageSize(extFacetCounts, &numExtFacets));
  CHKERRQ(PetscMalloc1(numIntFacets, &intFacetsArray));
  CHKERRQ(PetscMalloc1(numIntFacets*2, &intFacetsToPatchCell));
  CHKERRQ(PetscMalloc1(numExtFacets, &extFacetsArray));

  /* Now that we know how much space we need, run through again and actually remember the cells. */
  for (v = vStart; v < vEnd; v++) {
    PetscHashIter hi;
    PetscInt       dof, off, cdof, coff, efdof, efoff, ifdof, ifoff, pdof, n = 0, cn = 0, ifn = 0, efn = 0;

    CHKERRQ(PetscSectionGetDof(pointCounts, v, &dof));
    CHKERRQ(PetscSectionGetOffset(pointCounts, v, &off));
    CHKERRQ(PetscSectionGetDof(cellCounts, v, &cdof));
    CHKERRQ(PetscSectionGetOffset(cellCounts, v, &coff));
    CHKERRQ(PetscSectionGetDof(intFacetCounts, v, &ifdof));
    CHKERRQ(PetscSectionGetOffset(intFacetCounts, v, &ifoff));
    CHKERRQ(PetscSectionGetDof(extFacetCounts, v, &efdof));
    CHKERRQ(PetscSectionGetOffset(extFacetCounts, v, &efoff));
    if (dof <= 0) continue;
    CHKERRQ(patch->patchconstructop((void *) patch, dm, v, ht));
    CHKERRQ(PCPatchCompleteCellPatch(pc, ht, cht));
    PetscHashIterBegin(cht, hi);
    while (!PetscHashIterAtEnd(cht, hi)) {
      PetscInt point;

      PetscHashIterGetKey(cht, hi, point);
      if (fStart <= point && point < fEnd) {
        const PetscInt *support;
        PetscInt       supportSize, p;
        PetscBool      interior = PETSC_TRUE;
        CHKERRQ(DMPlexGetSupport(dm, point, &support));
        CHKERRQ(DMPlexGetSupportSize(dm, point, &supportSize));
        if (supportSize == 1) {
          interior = PETSC_FALSE;
        } else {
          for (p = 0; p < supportSize; p++) {
            PetscBool found;
            /* FIXME: can I do this while iterating over cht? */
            CHKERRQ(PetscHSetIHas(cht, support[p], &found));
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
      CHKERRQ(PCPatchGetGlobalDofs(pc, patch->dofSection, -1, patch->combined, point, &pdof, NULL));
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
      PetscCheckFalse(!(found0 && found1),PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Didn't manage to find local point numbers for facet support");
    }
  }
  CHKERRQ(PetscHSetIDestroy(&ht));
  CHKERRQ(PetscHSetIDestroy(&cht));

  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numCells,  cellsArray,  PETSC_OWN_POINTER, &patch->cells));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->cells,  "Patch Cells"));
  if (patch->viewCells) {
    CHKERRQ(ObjectView((PetscObject) patch->cellCounts, patch->viewerCells, patch->formatCells));
    CHKERRQ(ObjectView((PetscObject) patch->cells,      patch->viewerCells, patch->formatCells));
  }
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numIntFacets,  intFacetsArray,  PETSC_OWN_POINTER, &patch->intFacets));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->intFacets,  "Patch Interior Facets"));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, 2*numIntFacets, intFacetsToPatchCell, PETSC_OWN_POINTER, &patch->intFacetsToPatchCell));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->intFacetsToPatchCell,  "Patch Interior Facets local support"));
  if (patch->viewIntFacets) {
    CHKERRQ(ObjectView((PetscObject) patch->intFacetCounts,       patch->viewerIntFacets, patch->formatIntFacets));
    CHKERRQ(ObjectView((PetscObject) patch->intFacets,            patch->viewerIntFacets, patch->formatIntFacets));
    CHKERRQ(ObjectView((PetscObject) patch->intFacetsToPatchCell, patch->viewerIntFacets, patch->formatIntFacets));
  }
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numExtFacets,  extFacetsArray,  PETSC_OWN_POINTER, &patch->extFacets));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->extFacets,  "Patch Exterior Facets"));
  if (patch->viewExtFacets) {
    CHKERRQ(ObjectView((PetscObject) patch->extFacetCounts, patch->viewerExtFacets, patch->formatExtFacets));
    CHKERRQ(ObjectView((PetscObject) patch->extFacets,      patch->viewerExtFacets, patch->formatExtFacets));
  }
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numPoints, pointsArray, PETSC_OWN_POINTER, &patch->points));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->points, "Patch Points"));
  if (patch->viewPoints) {
    CHKERRQ(ObjectView((PetscObject) patch->pointCounts, patch->viewerPoints, patch->formatPoints));
    CHKERRQ(ObjectView((PetscObject) patch->points,      patch->viewerPoints, patch->formatPoints));
  }
  CHKERRQ(DMDestroy(&dm));
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

  CHKERRQ(PCGetDM(pc, &dm));
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  /* dofcounts section is cellcounts section * dofPerCell */
  CHKERRQ(PetscSectionGetStorageSize(cellCounts, &numCells));
  CHKERRQ(PetscSectionGetStorageSize(patch->pointCounts, &numPoints));
  numDofs = numCells * totalDofsPerCell;
  CHKERRQ(PetscMalloc1(numDofs, &dofsArray));
  CHKERRQ(PetscMalloc1(numPoints*Nf, &offsArray));
  CHKERRQ(PetscMalloc1(numDofs, &asmArray));
  CHKERRQ(PetscMalloc1(numCells, &newCellsArray));
  CHKERRQ(PetscSectionGetChart(cellCounts, &vStart, &vEnd));
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCounts));
  gtolCounts = patch->gtolCounts;
  CHKERRQ(PetscSectionSetChart(gtolCounts, vStart, vEnd));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->gtolCounts, "Patch Global Index Section"));

  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    CHKERRQ(PetscMalloc1(numPoints*Nf, &offsArrayWithArtificial));
    CHKERRQ(PetscMalloc1(numDofs, &asmArrayWithArtificial));
    CHKERRQ(PetscMalloc1(numDofs, &dofsArrayWithArtificial));
    CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCountsWithArtificial));
    gtolCountsWithArtificial = patch->gtolCountsWithArtificial;
    CHKERRQ(PetscSectionSetChart(gtolCountsWithArtificial, vStart, vEnd));
    CHKERRQ(PetscObjectSetName((PetscObject) patch->gtolCountsWithArtificial, "Patch Global Index Section Including Artificial BCs"));
  }

  isNonlinear = patch->isNonlinear;
  if (isNonlinear) {
    CHKERRQ(PetscMalloc1(numPoints*Nf, &offsArrayWithAll));
    CHKERRQ(PetscMalloc1(numDofs, &asmArrayWithAll));
    CHKERRQ(PetscMalloc1(numDofs, &dofsArrayWithAll));
    CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCountsWithAll));
    gtolCountsWithAll = patch->gtolCountsWithAll;
    CHKERRQ(PetscSectionSetChart(gtolCountsWithAll, vStart, vEnd));
    CHKERRQ(PetscObjectSetName((PetscObject) patch->gtolCountsWithAll, "Patch Global Index Section Including All BCs"));
  }

  /* Outside the patch loop, get the dofs that are globally-enforced Dirichlet
   conditions */
  CHKERRQ(PetscHSetICreate(&globalBcs));
  CHKERRQ(ISGetIndices(patch->ghostBcNodes, &bcNodes));
  CHKERRQ(ISGetSize(patch->ghostBcNodes, &numBcs));
  for (i = 0; i < numBcs; ++i) {
    CHKERRQ(PetscHSetIAdd(globalBcs, bcNodes[i])); /* these are already in concatenated numbering */
  }
  CHKERRQ(ISRestoreIndices(patch->ghostBcNodes, &bcNodes));
  CHKERRQ(ISDestroy(&patch->ghostBcNodes)); /* memory optimisation */

  /* Hash tables for artificial BC construction */
  CHKERRQ(PetscHSetICreate(&ownedpts));
  CHKERRQ(PetscHSetICreate(&seenpts));
  CHKERRQ(PetscHSetICreate(&owneddofs));
  CHKERRQ(PetscHSetICreate(&seendofs));
  CHKERRQ(PetscHSetICreate(&artificialbcs));

  CHKERRQ(ISGetIndices(cells, &cellsArray));
  CHKERRQ(ISGetIndices(points, &pointsArray));
  CHKERRQ(PetscHMapICreate(&ht));
  CHKERRQ(PetscHMapICreate(&htWithArtificial));
  CHKERRQ(PetscHMapICreate(&htWithAll));
  for (v = vStart; v < vEnd; ++v) {
    PetscInt localIndex = 0;
    PetscInt localIndexWithArtificial = 0;
    PetscInt localIndexWithAll = 0;
    PetscInt dof, off, i, j, k, l;

    CHKERRQ(PetscHMapIClear(ht));
    CHKERRQ(PetscHMapIClear(htWithArtificial));
    CHKERRQ(PetscHMapIClear(htWithAll));
    CHKERRQ(PetscSectionGetDof(cellCounts, v, &dof));
    CHKERRQ(PetscSectionGetOffset(cellCounts, v, &off));
    if (dof <= 0) continue;

    /* Calculate the global numbers of the artificial BC dofs here first */
    CHKERRQ(patch->patchconstructop((void*)patch, dm, v, ownedpts));
    CHKERRQ(PCPatchCompleteCellPatch(pc, ownedpts, seenpts));
    CHKERRQ(PCPatchGetPointDofs(pc, ownedpts, owneddofs, v, &patch->subspaces_to_exclude));
    CHKERRQ(PCPatchGetPointDofs(pc, seenpts, seendofs, v, NULL));
    CHKERRQ(PCPatchComputeSetDifference_Private(owneddofs, seendofs, artificialbcs));
    if (patch->viewPatches) {
      PetscHSetI    globalbcdofs;
      PetscHashIter hi;
      MPI_Comm      comm = PetscObjectComm((PetscObject)pc);

      CHKERRQ(PetscHSetICreate(&globalbcdofs));
      CHKERRQ(PetscSynchronizedPrintf(comm, "Patch %d: owned dofs:\n", v));
      PetscHashIterBegin(owneddofs, hi);
      while (!PetscHashIterAtEnd(owneddofs, hi)) {
        PetscInt globalDof;

        PetscHashIterGetKey(owneddofs, hi, globalDof);
        PetscHashIterNext(owneddofs, hi);
        CHKERRQ(PetscSynchronizedPrintf(comm, "%d ", globalDof));
      }
      CHKERRQ(PetscSynchronizedPrintf(comm, "\n"));
      CHKERRQ(PetscSynchronizedPrintf(comm, "Patch %d: seen dofs:\n", v));
      PetscHashIterBegin(seendofs, hi);
      while (!PetscHashIterAtEnd(seendofs, hi)) {
        PetscInt globalDof;
        PetscBool flg;

        PetscHashIterGetKey(seendofs, hi, globalDof);
        PetscHashIterNext(seendofs, hi);
        CHKERRQ(PetscSynchronizedPrintf(comm, "%d ", globalDof));

        CHKERRQ(PetscHSetIHas(globalBcs, globalDof, &flg));
        if (flg) CHKERRQ(PetscHSetIAdd(globalbcdofs, globalDof));
      }
      CHKERRQ(PetscSynchronizedPrintf(comm, "\n"));
      CHKERRQ(PetscSynchronizedPrintf(comm, "Patch %d: global BCs:\n", v));
      CHKERRQ(PetscHSetIGetSize(globalbcdofs, &numBcs));
      if (numBcs > 0) {
        PetscHashIterBegin(globalbcdofs, hi);
        while (!PetscHashIterAtEnd(globalbcdofs, hi)) {
          PetscInt globalDof;
          PetscHashIterGetKey(globalbcdofs, hi, globalDof);
          PetscHashIterNext(globalbcdofs, hi);
          CHKERRQ(PetscSynchronizedPrintf(comm, "%d ", globalDof));
        }
      }
      CHKERRQ(PetscSynchronizedPrintf(comm, "\n"));
      CHKERRQ(PetscSynchronizedPrintf(comm, "Patch %d: artificial BCs:\n", v));
      CHKERRQ(PetscHSetIGetSize(artificialbcs, &numBcs));
      if (numBcs > 0) {
        PetscHashIterBegin(artificialbcs, hi);
        while (!PetscHashIterAtEnd(artificialbcs, hi)) {
          PetscInt globalDof;
          PetscHashIterGetKey(artificialbcs, hi, globalDof);
          PetscHashIterNext(artificialbcs, hi);
          CHKERRQ(PetscSynchronizedPrintf(comm, "%d ", globalDof));
        }
      }
      CHKERRQ(PetscSynchronizedPrintf(comm, "\n\n"));
      CHKERRQ(PetscHSetIDestroy(&globalbcdofs));
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
          CHKERRQ(PetscSectionGetDof(cellNumbering, c, &cell));
          PetscCheckFalse(cell <= 0,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_OUTOFRANGE, "Cell %D doesn't appear in cell numbering map", c);
          CHKERRQ(PetscSectionGetOffset(cellNumbering, c, &cell));
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
            CHKERRQ(PetscHSetIHas(globalBcs, globalDof + l, &isGlobalBcDof));
            CHKERRQ(PetscHSetIHas(artificialbcs, globalDof + l, &isArtificialBcDof));

            /* if it's either, don't ever give it a local dof number */
            if (isGlobalBcDof || isArtificialBcDof) {
              dofsArray[globalIndex] = -1; /* don't use this in assembly in this patch */
            } else {
              CHKERRQ(PetscHMapIGet(ht, globalDof + l, &localDof));
              if (localDof == -1) {
                localDof = localIndex++;
                CHKERRQ(PetscHMapISet(ht, globalDof + l, localDof));
              }
              PetscCheckFalse(globalIndex >= numDofs,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
              /* And store. */
              dofsArray[globalIndex] = localDof;
            }

            if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
              if (isGlobalBcDof) {
                dofsArrayWithArtificial[globalIndex] = -1; /* don't use this in assembly in this patch */
              } else {
                CHKERRQ(PetscHMapIGet(htWithArtificial, globalDof + l, &localDof));
                if (localDof == -1) {
                  localDof = localIndexWithArtificial++;
                  CHKERRQ(PetscHMapISet(htWithArtificial, globalDof + l, localDof));
                }
                PetscCheckFalse(globalIndex >= numDofs,PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
                /* And store.*/
                dofsArrayWithArtificial[globalIndex] = localDof;
              }
            }

            if (isNonlinear) {
              /* Build the dofmap for the function space with _all_ dofs,
                 including those in any kind of boundary condition */
              CHKERRQ(PetscHMapIGet(htWithAll, globalDof + l, &localDof));
              if (localDof == -1) {
                localDof = localIndexWithAll++;
                CHKERRQ(PetscHMapISet(htWithAll, globalDof + l, localDof));
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
     CHKERRQ(PetscHMapIGetSize(htWithArtificial, &dof));
     CHKERRQ(PetscSectionSetDof(gtolCountsWithArtificial, v, dof));
   }
   if (isNonlinear) {
     CHKERRQ(PetscHMapIGetSize(htWithAll, &dof));
     CHKERRQ(PetscSectionSetDof(gtolCountsWithAll, v, dof));
   }
    CHKERRQ(PetscHMapIGetSize(ht, &dof));
    CHKERRQ(PetscSectionSetDof(gtolCounts, v, dof));
  }

  CHKERRQ(DMDestroy(&dm));
  PetscCheckFalse(globalIndex != numDofs,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Expected number of dofs (%d) doesn't match found number (%d)", numDofs, globalIndex);
  CHKERRQ(PetscSectionSetUp(gtolCounts));
  CHKERRQ(PetscSectionGetStorageSize(gtolCounts, &numGlobalDofs));
  CHKERRQ(PetscMalloc1(numGlobalDofs, &globalDofsArray));

  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    CHKERRQ(PetscSectionSetUp(gtolCountsWithArtificial));
    CHKERRQ(PetscSectionGetStorageSize(gtolCountsWithArtificial, &numGlobalDofsWithArtificial));
    CHKERRQ(PetscMalloc1(numGlobalDofsWithArtificial, &globalDofsArrayWithArtificial));
  }
  if (isNonlinear) {
    CHKERRQ(PetscSectionSetUp(gtolCountsWithAll));
    CHKERRQ(PetscSectionGetStorageSize(gtolCountsWithAll, &numGlobalDofsWithAll));
    CHKERRQ(PetscMalloc1(numGlobalDofsWithAll, &globalDofsArrayWithAll));
  }
  /* Now populate the global to local map.  This could be merged into the above loop if we were willing to deal with reallocs. */
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIter hi;
    PetscInt      dof, off, Np, ooff, i, j, k, l;

    CHKERRQ(PetscHMapIClear(ht));
    CHKERRQ(PetscHMapIClear(htWithArtificial));
    CHKERRQ(PetscHMapIClear(htWithAll));
    CHKERRQ(PetscSectionGetDof(cellCounts, v, &dof));
    CHKERRQ(PetscSectionGetOffset(cellCounts, v, &off));
    CHKERRQ(PetscSectionGetDof(pointCounts, v, &Np));
    CHKERRQ(PetscSectionGetOffset(pointCounts, v, &ooff));
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

        if (cellNumbering) CHKERRQ(PetscSectionGetOffset(cellNumbering, c, &cell));
        for (j = 0; j < nodesPerCell; ++j) {
          for (l = 0; l < bs; ++l) {
            const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + l + subspaceOffset;
            const PetscInt localDof  = dofsArray[key];
            if (localDof >= 0) CHKERRQ(PetscHMapISet(ht, globalDof, localDof));
            if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
              const PetscInt localDofWithArtificial = dofsArrayWithArtificial[key];
              if (localDofWithArtificial >= 0) {
                CHKERRQ(PetscHMapISet(htWithArtificial, globalDof, localDofWithArtificial));
              }
            }
            if (isNonlinear) {
              const PetscInt localDofWithAll = dofsArrayWithAll[key];
              if (localDofWithAll >= 0) {
                CHKERRQ(PetscHMapISet(htWithAll, globalDof, localDofWithAll));
              }
            }
            key++;
          }
        }
      }

      /* Shove it in the output data structure. */
      CHKERRQ(PetscSectionGetOffset(gtolCounts, v, &goff));
      PetscHashIterBegin(ht, hi);
      while (!PetscHashIterAtEnd(ht, hi)) {
        PetscInt globalDof, localDof;

        PetscHashIterGetKey(ht, hi, globalDof);
        PetscHashIterGetVal(ht, hi, localDof);
        if (globalDof >= 0) globalDofsArray[goff + localDof] = globalDof;
        PetscHashIterNext(ht, hi);
      }

      if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
        CHKERRQ(PetscSectionGetOffset(gtolCountsWithArtificial, v, &goff));
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
        CHKERRQ(PetscSectionGetOffset(gtolCountsWithAll, v, &goff));
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

        CHKERRQ(PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, point, NULL, &globalDof));
        CHKERRQ(PetscHMapIGet(ht, globalDof, &localDof));
        offsArray[(ooff + p)*Nf + k] = localDof;
        if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
          CHKERRQ(PetscHMapIGet(htWithArtificial, globalDof, &localDof));
          offsArrayWithArtificial[(ooff + p)*Nf + k] = localDof;
        }
        if (isNonlinear) {
          CHKERRQ(PetscHMapIGet(htWithAll, globalDof, &localDof));
          offsArrayWithAll[(ooff + p)*Nf + k] = localDof;
        }
      }
    }

    CHKERRQ(PetscHSetIDestroy(&globalBcs));
    CHKERRQ(PetscHSetIDestroy(&ownedpts));
    CHKERRQ(PetscHSetIDestroy(&seenpts));
    CHKERRQ(PetscHSetIDestroy(&owneddofs));
    CHKERRQ(PetscHSetIDestroy(&seendofs));
    CHKERRQ(PetscHSetIDestroy(&artificialbcs));

      /* At this point, we have a hash table ht built that maps globalDof -> localDof.
     We need to create the dof table laid out cellwise first, then by subspace,
     as the assembler assembles cell-wise and we need to stuff the different
     contributions of the different function spaces to the right places. So we loop
     over cells, then over subspaces. */
    if (patch->nsubspaces > 1) { /* for nsubspaces = 1, data we need is already in dofsArray */
      for (i = off; i < off + dof; ++i) {
        const PetscInt c    = cellsArray[i];
        PetscInt       cell = c;

        if (cellNumbering) CHKERRQ(PetscSectionGetOffset(cellNumbering, c, &cell));
        for (k = 0; k < patch->nsubspaces; ++k) {
          const PetscInt *cellNodeMap    = patch->cellNodeMap[k];
          PetscInt        nodesPerCell   = patch->nodesPerCell[k];
          PetscInt        subspaceOffset = patch->subspaceOffsets[k];
          PetscInt        bs             = patch->bs[k];

          for (j = 0; j < nodesPerCell; ++j) {
            for (l = 0; l < bs; ++l) {
              const PetscInt globalDof = cellNodeMap[cell*nodesPerCell + j]*bs + l + subspaceOffset;
              PetscInt       localDof;

              CHKERRQ(PetscHMapIGet(ht, globalDof, &localDof));
              /* If it's not in the hash table, i.e. is a BC dof,
               then the PetscHSetIMap above gives -1, which matches
               exactly the convention for PETSc's matrix assembly to
               ignore the dof. So we don't need to do anything here */
              asmArray[asmKey] = localDof;
              if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
                CHKERRQ(PetscHMapIGet(htWithArtificial, globalDof, &localDof));
                asmArrayWithArtificial[asmKey] = localDof;
              }
              if (isNonlinear) {
                CHKERRQ(PetscHMapIGet(htWithAll, globalDof, &localDof));
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
    CHKERRQ(PetscArraycpy(asmArray, dofsArray, numDofs));
    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
      CHKERRQ(PetscArraycpy(asmArrayWithArtificial, dofsArrayWithArtificial, numDofs));
    }
    if (isNonlinear) {
      CHKERRQ(PetscArraycpy(asmArrayWithAll, dofsArrayWithAll, numDofs));
    }
  }

  CHKERRQ(PetscHMapIDestroy(&ht));
  CHKERRQ(PetscHMapIDestroy(&htWithArtificial));
  CHKERRQ(PetscHMapIDestroy(&htWithAll));
  CHKERRQ(ISRestoreIndices(cells, &cellsArray));
  CHKERRQ(ISRestoreIndices(points, &pointsArray));
  CHKERRQ(PetscFree(dofsArray));
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    CHKERRQ(PetscFree(dofsArrayWithArtificial));
  }
  if (isNonlinear) {
    CHKERRQ(PetscFree(dofsArrayWithAll));
  }
  /* Create placeholder section for map from points to patch dofs */
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &patch->patchSection));
  CHKERRQ(PetscSectionSetNumFields(patch->patchSection, patch->nsubspaces));
  if (patch->combined) {
    PetscInt numFields;
    CHKERRQ(PetscSectionGetNumFields(patch->dofSection[0], &numFields));
    PetscCheckFalse(numFields != patch->nsubspaces,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Mismatch between number of section fields %D and number of subspaces %D", numFields, patch->nsubspaces);
    CHKERRQ(PetscSectionGetChart(patch->dofSection[0], &pStart, &pEnd));
    CHKERRQ(PetscSectionSetChart(patch->patchSection, pStart, pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, fdof, f;

      CHKERRQ(PetscSectionGetDof(patch->dofSection[0], p, &dof));
      CHKERRQ(PetscSectionSetDof(patch->patchSection, p, dof));
      for (f = 0; f < patch->nsubspaces; ++f) {
        CHKERRQ(PetscSectionGetFieldDof(patch->dofSection[0], p, f, &fdof));
        CHKERRQ(PetscSectionSetFieldDof(patch->patchSection, p, f, fdof));
      }
    }
  } else {
    PetscInt pStartf, pEndf, f;
    pStart = PETSC_MAX_INT;
    pEnd = PETSC_MIN_INT;
    for (f = 0; f < patch->nsubspaces; ++f) {
      CHKERRQ(PetscSectionGetChart(patch->dofSection[f], &pStartf, &pEndf));
      pStart = PetscMin(pStart, pStartf);
      pEnd = PetscMax(pEnd, pEndf);
    }
    CHKERRQ(PetscSectionSetChart(patch->patchSection, pStart, pEnd));
    for (f = 0; f < patch->nsubspaces; ++f) {
      CHKERRQ(PetscSectionGetChart(patch->dofSection[f], &pStartf, &pEndf));
      for (p = pStartf; p < pEndf; ++p) {
        PetscInt fdof;
        CHKERRQ(PetscSectionGetDof(patch->dofSection[f], p, &fdof));
        CHKERRQ(PetscSectionAddDof(patch->patchSection, p, fdof));
        CHKERRQ(PetscSectionSetFieldDof(patch->patchSection, p, f, fdof));
      }
    }
  }
  CHKERRQ(PetscSectionSetUp(patch->patchSection));
  CHKERRQ(PetscSectionSetUseFieldOffsets(patch->patchSection, PETSC_TRUE));
  /* Replace cell indices with firedrake-numbered ones. */
  CHKERRQ(ISGeneralSetIndices(cells, numCells, (const PetscInt *) newCellsArray, PETSC_OWN_POINTER));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofs, globalDofsArray, PETSC_OWN_POINTER, &patch->gtol));
  CHKERRQ(PetscObjectSetName((PetscObject) patch->gtol, "Global Indices"));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_g2l_view", patch->classname));
  CHKERRQ(PetscSectionViewFromOptions(patch->gtolCounts, (PetscObject) pc, option));
  CHKERRQ(ISViewFromOptions(patch->gtol, (PetscObject) pc, option));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArray, PETSC_OWN_POINTER, &patch->dofs));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArray, PETSC_OWN_POINTER, &patch->offs));
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofsWithArtificial, globalDofsArrayWithArtificial, PETSC_OWN_POINTER, &patch->gtolWithArtificial));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArrayWithArtificial, PETSC_OWN_POINTER, &patch->dofsWithArtificial));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArrayWithArtificial, PETSC_OWN_POINTER, &patch->offsWithArtificial));
  }
  if (isNonlinear) {
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofsWithAll, globalDofsArrayWithAll, PETSC_OWN_POINTER, &patch->gtolWithAll));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArrayWithAll, PETSC_OWN_POINTER, &patch->dofsWithAll));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArrayWithAll, PETSC_OWN_POINTER, &patch->offsWithAll));
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
    CHKERRQ(PetscSectionGetChart(patch->gtolCountsWithArtificial, &pStart, NULL));
    CHKERRQ(PetscSectionGetDof(patch->gtolCountsWithArtificial, point + pStart, &rsize));
    csize = rsize;
  } else {
    PetscInt pStart;
    CHKERRQ(PetscSectionGetChart(patch->gtolCounts, &pStart, NULL));
    CHKERRQ(PetscSectionGetDof(patch->gtolCounts, point + pStart, &rsize));
    csize = rsize;
  }

  CHKERRQ(MatCreate(PETSC_COMM_SELF, mat));
  CHKERRQ(PCGetOptionsPrefix(pc, &prefix));
  CHKERRQ(MatSetOptionsPrefix(*mat, prefix));
  CHKERRQ(MatAppendOptionsPrefix(*mat, "pc_patch_sub_"));
  if (patch->sub_mat_type)       CHKERRQ(MatSetType(*mat, patch->sub_mat_type));
  else if (!patch->sub_mat_type) CHKERRQ(MatSetType(*mat, MATDENSE));
  CHKERRQ(MatSetSizes(*mat, rsize, csize, rsize, csize));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) *mat, MATDENSE, &flg));
  if (!flg) CHKERRQ(PetscObjectTypeCompare((PetscObject)*mat, MATSEQDENSE, &flg));
  /* Sparse patch matrices */
  if (!flg) {
    PetscBT         bt;
    PetscInt       *dnnz      = NULL;
    const PetscInt *dofsArray = NULL;
    PetscInt        pStart, pEnd, ncell, offset, c, i, j;

    if (withArtificial) {
      CHKERRQ(ISGetIndices(patch->dofsWithArtificial, &dofsArray));
    } else {
      CHKERRQ(ISGetIndices(patch->dofs, &dofsArray));
    }
    CHKERRQ(PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd));
    point += pStart;
    PetscCheckFalse(point >= pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Operator point %D not in [%D, %D)", point, pStart, pEnd);
    CHKERRQ(PetscSectionGetDof(patch->cellCounts, point, &ncell));
    CHKERRQ(PetscSectionGetOffset(patch->cellCounts, point, &offset));
    CHKERRQ(PetscLogEventBegin(PC_Patch_Prealloc, pc, 0, 0, 0));
    /* A PetscBT uses N^2 bits to store the sparsity pattern on a
     * patch. This is probably OK if the patches are not too big,
     * but uses too much memory. We therefore switch based on rsize. */
    if (rsize < 3000) { /* FIXME: I picked this switch value out of my hat */
      PetscScalar *zeroes;
      PetscInt rows;

      CHKERRQ(PetscCalloc1(rsize, &dnnz));
      CHKERRQ(PetscBTCreate(rsize*rsize, &bt));
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

        CHKERRQ(PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets));
        CHKERRQ(PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset));
        CHKERRQ(ISGetIndices(patch->intFacetsToPatchCell, &facetCells));
        CHKERRQ(ISGetIndices(patch->intFacets, &intFacetsArray));
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
      CHKERRQ(PetscBTDestroy(&bt));
      CHKERRQ(MatXAIJSetPreallocation(*mat, 1, dnnz, NULL, NULL, NULL));
      CHKERRQ(PetscFree(dnnz));

      CHKERRQ(PetscCalloc1(patch->totalDofsPerCell*patch->totalDofsPerCell, &zeroes));
      for (c = 0; c < ncell; ++c) {
        const PetscInt *idx = &dofsArray[(offset + c)*patch->totalDofsPerCell];
        CHKERRQ(MatSetValues(*mat, patch->totalDofsPerCell, idx, patch->totalDofsPerCell, idx, zeroes, INSERT_VALUES));
      }
      CHKERRQ(MatGetLocalSize(*mat, &rows, NULL));
      for (i = 0; i < rows; ++i) {
        CHKERRQ(MatSetValues(*mat, 1, &i, 1, &i, zeroes, INSERT_VALUES));
      }

      if (patch->usercomputeopintfacet) {
        const PetscInt *intFacetsArray = NULL;
        PetscInt i, numIntFacets, intFacetOffset;
        const PetscInt *facetCells = NULL;

        CHKERRQ(PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets));
        CHKERRQ(PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset));
        CHKERRQ(ISGetIndices(patch->intFacetsToPatchCell, &facetCells));
        CHKERRQ(ISGetIndices(patch->intFacets, &intFacetsArray));
        for (i = 0; i < numIntFacets; i++) {
          const PetscInt cell0 = facetCells[2*(intFacetOffset + i) + 0];
          const PetscInt cell1 = facetCells[2*(intFacetOffset + i) + 1];
          const PetscInt *cell0idx = &dofsArray[(offset + cell0)*patch->totalDofsPerCell];
          const PetscInt *cell1idx = &dofsArray[(offset + cell1)*patch->totalDofsPerCell];
          CHKERRQ(MatSetValues(*mat, patch->totalDofsPerCell, cell0idx, patch->totalDofsPerCell, cell1idx, zeroes, INSERT_VALUES));
          CHKERRQ(MatSetValues(*mat, patch->totalDofsPerCell, cell1idx, patch->totalDofsPerCell, cell0idx, zeroes, INSERT_VALUES));
        }
      }

      CHKERRQ(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));

      CHKERRQ(PetscFree(zeroes));

    } else { /* rsize too big, use MATPREALLOCATOR */
      Mat preallocator;
      PetscScalar* vals;

      CHKERRQ(PetscCalloc1(patch->totalDofsPerCell*patch->totalDofsPerCell, &vals));
      CHKERRQ(MatCreate(PETSC_COMM_SELF, &preallocator));
      CHKERRQ(MatSetType(preallocator, MATPREALLOCATOR));
      CHKERRQ(MatSetSizes(preallocator, rsize, rsize, rsize, rsize));
      CHKERRQ(MatSetUp(preallocator));

      for (c = 0; c < ncell; ++c) {
        const PetscInt *idx = dofsArray + (offset + c)*patch->totalDofsPerCell;
        CHKERRQ(MatSetValues(preallocator, patch->totalDofsPerCell, idx, patch->totalDofsPerCell, idx, vals, INSERT_VALUES));
      }

      if (patch->usercomputeopintfacet) {
        const PetscInt *intFacetsArray = NULL;
        PetscInt        i, numIntFacets, intFacetOffset;
        const PetscInt *facetCells = NULL;

        CHKERRQ(PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets));
        CHKERRQ(PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset));
        CHKERRQ(ISGetIndices(patch->intFacetsToPatchCell, &facetCells));
        CHKERRQ(ISGetIndices(patch->intFacets, &intFacetsArray));
        for (i = 0; i < numIntFacets; i++) {
          const PetscInt cell0 = facetCells[2*(intFacetOffset + i) + 0];
          const PetscInt cell1 = facetCells[2*(intFacetOffset + i) + 1];
          const PetscInt *cell0idx = &dofsArray[(offset + cell0)*patch->totalDofsPerCell];
          const PetscInt *cell1idx = &dofsArray[(offset + cell1)*patch->totalDofsPerCell];
          CHKERRQ(MatSetValues(preallocator, patch->totalDofsPerCell, cell0idx, patch->totalDofsPerCell, cell1idx, vals, INSERT_VALUES));
          CHKERRQ(MatSetValues(preallocator, patch->totalDofsPerCell, cell1idx, patch->totalDofsPerCell, cell0idx, vals, INSERT_VALUES));
        }
      }

      CHKERRQ(PetscFree(vals));
      CHKERRQ(MatAssemblyBegin(preallocator, MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(preallocator, MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatPreallocatorPreallocate(preallocator, PETSC_TRUE, *mat));
      CHKERRQ(MatDestroy(&preallocator));
    }
    CHKERRQ(PetscLogEventEnd(PC_Patch_Prealloc, pc, 0, 0, 0));
    if (withArtificial) {
      CHKERRQ(ISRestoreIndices(patch->dofsWithArtificial, &dofsArray));
    } else {
      CHKERRQ(ISRestoreIndices(patch->dofs, &dofsArray));
    }
  }
  CHKERRQ(MatSetUp(*mat));
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
  CHKERRQ(PCGetDM(pc, &dm));
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  CHKERRQ(DMGetLocalSection(dm, &s));
  /* Set offset into patch */
  CHKERRQ(PetscSectionGetDof(patch->pointCounts, patchNum, &Np));
  CHKERRQ(PetscSectionGetOffset(patch->pointCounts, patchNum, &poff));
  CHKERRQ(ISGetIndices(patch->points, &parray));
  CHKERRQ(ISGetIndices(patch->offs,   &oarray));
  for (f = 0; f < Nf; ++f) {
    for (p = 0; p < Np; ++p) {
      const PetscInt point = parray[poff+p];
      PetscInt       dof;

      CHKERRQ(PetscSectionGetFieldDof(patch->patchSection, point, f, &dof));
      CHKERRQ(PetscSectionSetFieldOffset(patch->patchSection, point, f, oarray[(poff+p)*Nf+f]));
      if (patch->nsubspaces == 1) CHKERRQ(PetscSectionSetOffset(patch->patchSection, point, oarray[(poff+p)*Nf+f]));
      else                        CHKERRQ(PetscSectionSetOffset(patch->patchSection, point, -1));
    }
  }
  CHKERRQ(ISRestoreIndices(patch->points, &parray));
  CHKERRQ(ISRestoreIndices(patch->offs,   &oarray));
  if (patch->viewSection) CHKERRQ(ObjectView((PetscObject) patch->patchSection, patch->viewerSection, patch->formatSection));
  CHKERRQ(DMPlexComputeResidual_Patch_Internal(dm, patch->patchSection, cellIS, 0.0, x, NULL, F, ctx));
  CHKERRQ(DMDestroy(&dm));
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
  CHKERRQ(PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0));
  PetscCheck(patch->usercomputeop,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PCPatchSetComputeOperator() to set user callback");
  CHKERRQ(ISGetIndices(patch->dofs, &dofsArray));
  CHKERRQ(ISGetIndices(patch->dofsWithAll, &dofsArrayWithAll));
  CHKERRQ(ISGetIndices(patch->cells, &cellsArray));
  CHKERRQ(PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd));

  point += pStart;
  PetscCheckFalse(point >= pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Operator point %D not in [%D, %D)", point, pStart, pEnd);

  CHKERRQ(PetscSectionGetDof(patch->cellCounts, point, &ncell));
  CHKERRQ(PetscSectionGetOffset(patch->cellCounts, point, &offset));
  if (ncell <= 0) {
    CHKERRQ(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));
    PetscFunctionReturn(0);
  }
  CHKERRQ(VecSet(F, 0.0));
  PetscStackPush("PCPatch user callback");
  /* Cannot reuse the same IS because the geometry info is being cached in it */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray + offset, PETSC_USE_POINTER, &patch->cellIS));
  ierr = patch->usercomputef(pc, point, x, F, patch->cellIS, ncell*patch->totalDofsPerCell, dofsArray + offset*patch->totalDofsPerCell,
                                                                                            dofsArrayWithAll + offset*patch->totalDofsPerCell,
                                                                                            patch->usercomputefctx);CHKERRQ(ierr);
  PetscStackPop;
  CHKERRQ(ISDestroy(&patch->cellIS));
  CHKERRQ(ISRestoreIndices(patch->dofs, &dofsArray));
  CHKERRQ(ISRestoreIndices(patch->dofsWithAll, &dofsArrayWithAll));
  CHKERRQ(ISRestoreIndices(patch->cells, &cellsArray));
  if (patch->viewMatrix) {
    char name[PETSC_MAX_PATH_LEN];

    CHKERRQ(PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "Patch vector for Point %D", point));
    CHKERRQ(PetscObjectSetName((PetscObject) F, name));
    CHKERRQ(ObjectView((PetscObject) F, patch->viewerMatrix, patch->formatMatrix));
  }
  CHKERRQ(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));
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
  CHKERRQ(PCGetDM(pc, &dm));
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  CHKERRQ(DMGetLocalSection(dm, &s));
  /* Set offset into patch */
  CHKERRQ(PetscSectionGetDof(patch->pointCounts, patchNum, &Np));
  CHKERRQ(PetscSectionGetOffset(patch->pointCounts, patchNum, &poff));
  CHKERRQ(ISGetIndices(patch->points, &parray));
  CHKERRQ(ISGetIndices(patch->offs,   &oarray));
  for (f = 0; f < Nf; ++f) {
    for (p = 0; p < Np; ++p) {
      const PetscInt point = parray[poff+p];
      PetscInt       dof;

      CHKERRQ(PetscSectionGetFieldDof(patch->patchSection, point, f, &dof));
      CHKERRQ(PetscSectionSetFieldOffset(patch->patchSection, point, f, oarray[(poff+p)*Nf+f]));
      if (patch->nsubspaces == 1) CHKERRQ(PetscSectionSetOffset(patch->patchSection, point, oarray[(poff+p)*Nf+f]));
      else                        CHKERRQ(PetscSectionSetOffset(patch->patchSection, point, -1));
    }
  }
  CHKERRQ(ISRestoreIndices(patch->points, &parray));
  CHKERRQ(ISRestoreIndices(patch->offs,   &oarray));
  if (patch->viewSection) CHKERRQ(ObjectView((PetscObject) patch->patchSection, patch->viewerSection, patch->formatSection));
  /* TODO Shut off MatViewFromOptions() in MatAssemblyEnd() here */
  CHKERRQ(DMPlexComputeJacobian_Patch_Internal(dm, patch->patchSection, patch->patchSection, cellIS, 0.0, 0.0, x, NULL, J, J, ctx));
  CHKERRQ(DMDestroy(&dm));
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
  CHKERRQ(PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0));
  isNonlinear = patch->isNonlinear;
  PetscCheck(patch->usercomputeop,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PCPatchSetComputeOperator() to set user callback");
  if (withArtificial) {
    CHKERRQ(ISGetIndices(patch->dofsWithArtificial, &dofsArray));
  } else {
    CHKERRQ(ISGetIndices(patch->dofs, &dofsArray));
  }
  if (isNonlinear) {
    CHKERRQ(ISGetIndices(patch->dofsWithAll, &dofsArrayWithAll));
  }
  CHKERRQ(ISGetIndices(patch->cells, &cellsArray));
  CHKERRQ(PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd));

  point += pStart;
  PetscCheckFalse(point >= pEnd,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Operator point %D not in [%D, %D)", point, pStart, pEnd);

  CHKERRQ(PetscSectionGetDof(patch->cellCounts, point, &ncell));
  CHKERRQ(PetscSectionGetOffset(patch->cellCounts, point, &offset));
  if (ncell <= 0) {
    CHKERRQ(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));
    PetscFunctionReturn(0);
  }
  CHKERRQ(MatZeroEntries(mat));
  if (patch->precomputeElementTensors) {
    PetscInt           i;
    PetscInt           ndof = patch->totalDofsPerCell;
    const PetscScalar *elementTensors;

    CHKERRQ(VecGetArrayRead(patch->cellMats, &elementTensors));
    for (i = 0; i < ncell; i++) {
      const PetscInt     cell = cellsArray[i + offset];
      const PetscInt    *idx  = dofsArray + (offset + i)*ndof;
      const PetscScalar *v    = elementTensors + patch->precomputedTensorLocations[cell]*ndof*ndof;
      CHKERRQ(MatSetValues(mat, ndof, idx, ndof, idx, v, ADD_VALUES));
    }
    CHKERRQ(VecRestoreArrayRead(patch->cellMats, &elementTensors));
    CHKERRQ(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  } else {
    PetscStackPush("PCPatch user callback");
    /* Cannot reuse the same IS because the geometry info is being cached in it */
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray + offset, PETSC_USE_POINTER, &patch->cellIS));
    CHKERRQ(patch->usercomputeop(pc, point, x, mat, patch->cellIS, ncell*patch->totalDofsPerCell, dofsArray + offset*patch->totalDofsPerCell, dofsArrayWithAll ? dofsArrayWithAll + offset*patch->totalDofsPerCell : NULL, patch->usercomputeopctx));
  }
  if (patch->usercomputeopintfacet) {
    CHKERRQ(PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets));
    CHKERRQ(PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset));
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

      CHKERRQ(ISGetIndices(patch->intFacetsToPatchCell, &facetCells));
      CHKERRQ(ISGetIndices(patch->intFacets, &intFacetsArray));
      CHKERRQ(PCGetDM(pc, &dm));
      CHKERRQ(DMConvert(dm, DMPLEX, &plex));
      dm = plex;
      CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, NULL));
      /* FIXME: Pull this malloc out. */
      CHKERRQ(PetscMalloc1(2 * patch->totalDofsPerCell * numIntFacets, &facetDofs));
      if (dofsArrayWithAll) {
        CHKERRQ(PetscMalloc1(2 * patch->totalDofsPerCell * numIntFacets, &facetDofsWithAll));
      }
      if (patch->precomputeElementTensors) {
        PetscInt           nFacetDof = 2*patch->totalDofsPerCell;
        const PetscScalar *elementTensors;

        CHKERRQ(VecGetArrayRead(patch->intFacetMats, &elementTensors));

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
          CHKERRQ(MatSetValues(mat, nFacetDof, facetDofs, nFacetDof, facetDofs, v, ADD_VALUES));
        }
        CHKERRQ(VecRestoreArrayRead(patch->intFacetMats, &elementTensors));
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
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numIntFacets, intFacetsArray + intFacetOffset, PETSC_USE_POINTER, &facetIS));
        CHKERRQ(patch->usercomputeopintfacet(pc, point, x, mat, facetIS, 2*numIntFacets*patch->totalDofsPerCell, facetDofs, facetDofsWithAll, patch->usercomputeopintfacetctx));
        CHKERRQ(ISDestroy(&facetIS));
      }
      CHKERRQ(ISRestoreIndices(patch->intFacetsToPatchCell, &facetCells));
      CHKERRQ(ISRestoreIndices(patch->intFacets, &intFacetsArray));
      CHKERRQ(PetscFree(facetDofs));
      CHKERRQ(PetscFree(facetDofsWithAll));
      CHKERRQ(DMDestroy(&dm));
    }
  }

  CHKERRQ(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  if (!(withArtificial || isNonlinear) && patch->denseinverse) {
    MatFactorInfo info;
    PetscBool     flg;
    CHKERRQ(PetscObjectTypeCompare((PetscObject)mat, MATSEQDENSE, &flg));
    PetscCheck(flg,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Invalid Mat type for dense inverse");
    CHKERRQ(MatFactorInfoInitialize(&info));
    CHKERRQ(MatLUFactor(mat, NULL, NULL, &info));
    CHKERRQ(MatSeqDenseInvertFactors_Private(mat));
  }
  PetscStackPop;
  CHKERRQ(ISDestroy(&patch->cellIS));
  if (withArtificial) {
    CHKERRQ(ISRestoreIndices(patch->dofsWithArtificial, &dofsArray));
  } else {
    CHKERRQ(ISRestoreIndices(patch->dofs, &dofsArray));
  }
  if (isNonlinear) {
    CHKERRQ(ISRestoreIndices(patch->dofsWithAll, &dofsArrayWithAll));
  }
  CHKERRQ(ISRestoreIndices(patch->cells, &cellsArray));
  if (patch->viewMatrix) {
    char name[PETSC_MAX_PATH_LEN];

    CHKERRQ(PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "Patch matrix for Point %D", point));
    CHKERRQ(PetscObjectSetName((PetscObject) mat, name));
    CHKERRQ(ObjectView((PetscObject) mat, patch->viewerMatrix, patch->formatMatrix));
  }
  CHKERRQ(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_PCPatch_Private(Mat mat, PetscInt m, const PetscInt idxm[],
                                                   PetscInt n, const PetscInt idxn[], const PetscScalar *v, InsertMode addv)
{
  Vec            data;
  PetscScalar   *array;
  PetscInt       bs, nz, i, j, cell;

  CHKERRQ(MatShellGetContext(mat, &data));
  CHKERRQ(VecGetBlockSize(data, &bs));
  CHKERRQ(VecGetSize(data, &nz));
  CHKERRQ(VecGetArray(data, &array));
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
  CHKERRQ(VecRestoreArray(data, &array));
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

  CHKERRQ(ISGetSize(patch->cells, &ncell));
  if (!ncell) { /* No cells to assemble over -> skip */
    PetscFunctionReturn(0);
  }

  CHKERRQ(PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0));

  CHKERRQ(PCGetDM(pc, &dm));
  CHKERRQ(DMConvert(dm, DMPLEX, &plex));
  dm = plex;
  if (!patch->allCells) {
    PetscHSetI      cells;
    PetscHashIter   hi;
    PetscInt        pStart, pEnd;
    PetscInt        *allCells = NULL;
    CHKERRQ(PetscHSetICreate(&cells));
    CHKERRQ(ISGetIndices(patch->cells, &cellsArray));
    CHKERRQ(PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd));
    for (i = pStart; i < pEnd; i++) {
      CHKERRQ(PetscSectionGetDof(patch->cellCounts, i, &ncell));
      CHKERRQ(PetscSectionGetOffset(patch->cellCounts, i, &offset));
      if (ncell <= 0) continue;
      for (j = 0; j < ncell; j++) {
        CHKERRQ(PetscHSetIAdd(cells, cellsArray[offset + j]));
      }
    }
    CHKERRQ(ISRestoreIndices(patch->cells, &cellsArray));
    CHKERRQ(PetscHSetIGetSize(cells, &ncell));
    CHKERRQ(PetscMalloc1(ncell, &allCells));
    CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    CHKERRQ(PetscMalloc1(cEnd-cStart, &patch->precomputedTensorLocations));
    i = 0;
    PetscHashIterBegin(cells, hi);
    while (!PetscHashIterAtEnd(cells, hi)) {
      PetscHashIterGetKey(cells, hi, allCells[i]);
      patch->precomputedTensorLocations[allCells[i]] = i;
      PetscHashIterNext(cells, hi);
      i++;
    }
    CHKERRQ(PetscHSetIDestroy(&cells));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, ncell, allCells, PETSC_OWN_POINTER, &patch->allCells));
  }
  CHKERRQ(ISGetSize(patch->allCells, &ncell));
  if (!patch->cellMats) {
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, ncell*ndof*ndof, &patch->cellMats));
    CHKERRQ(VecSetBlockSize(patch->cellMats, ndof));
  }
  CHKERRQ(VecSet(patch->cellMats, 0));

  ierr = MatCreateShell(PETSC_COMM_SELF, ncell*ndof, ncell*ndof, ncell*ndof, ncell*ndof,
                        (void*)patch->cellMats, &vecMat);CHKERRQ(ierr);
  CHKERRQ(MatShellSetOperation(vecMat, MATOP_SET_VALUES, (void(*)(void))&MatSetValues_PCPatch_Private));
  CHKERRQ(ISGetSize(patch->allCells, &ncell));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, ndof*ncell, 0, 1, &dofMap));
  CHKERRQ(ISGetIndices(dofMap, &dofMapArray));
  CHKERRQ(ISGetIndices(patch->allCells, &cellsArray));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray, PETSC_USE_POINTER, &cellIS));
  PetscStackPush("PCPatch user callback");
  /* TODO: Fix for DMPlex compute op, this bypasses a lot of the machinery and just assembles every element tensor. */
  CHKERRQ(patch->usercomputeop(pc, -1, NULL, vecMat, cellIS, ndof*ncell, dofMapArray, NULL, patch->usercomputeopctx));
  PetscStackPop;
  CHKERRQ(ISDestroy(&cellIS));
  CHKERRQ(MatDestroy(&vecMat));
  CHKERRQ(ISRestoreIndices(patch->allCells, &cellsArray));
  CHKERRQ(ISRestoreIndices(dofMap, &dofMapArray));
  CHKERRQ(ISDestroy(&dofMap));

  if (patch->usercomputeopintfacet) {
    PetscInt nIntFacets;
    IS       intFacetsIS;
    const PetscInt *intFacetsArray = NULL;
    if (!patch->allIntFacets) {
      PetscHSetI      facets;
      PetscHashIter   hi;
      PetscInt        pStart, pEnd, fStart, fEnd;
      PetscInt        *allIntFacets = NULL;
      CHKERRQ(PetscHSetICreate(&facets));
      CHKERRQ(ISGetIndices(patch->intFacets, &intFacetsArray));
      CHKERRQ(PetscSectionGetChart(patch->intFacetCounts, &pStart, &pEnd));
      CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
      for (i = pStart; i < pEnd; i++) {
        CHKERRQ(PetscSectionGetDof(patch->intFacetCounts, i, &nIntFacets));
        CHKERRQ(PetscSectionGetOffset(patch->intFacetCounts, i, &offset));
        if (nIntFacets <= 0) continue;
        for (j = 0; j < nIntFacets; j++) {
          CHKERRQ(PetscHSetIAdd(facets, intFacetsArray[offset + j]));
        }
      }
      CHKERRQ(ISRestoreIndices(patch->intFacets, &intFacetsArray));
      CHKERRQ(PetscHSetIGetSize(facets, &nIntFacets));
      CHKERRQ(PetscMalloc1(nIntFacets, &allIntFacets));
      CHKERRQ(PetscMalloc1(fEnd-fStart, &patch->precomputedIntFacetTensorLocations));
      i = 0;
      PetscHashIterBegin(facets, hi);
      while (!PetscHashIterAtEnd(facets, hi)) {
        PetscHashIterGetKey(facets, hi, allIntFacets[i]);
        patch->precomputedIntFacetTensorLocations[allIntFacets[i] - fStart] = i;
        PetscHashIterNext(facets, hi);
        i++;
      }
      CHKERRQ(PetscHSetIDestroy(&facets));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, nIntFacets, allIntFacets, PETSC_OWN_POINTER, &patch->allIntFacets));
    }
    CHKERRQ(ISGetSize(patch->allIntFacets, &nIntFacets));
    if (!patch->intFacetMats) {
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, nIntFacets*ndof*ndof*4, &patch->intFacetMats));
      CHKERRQ(VecSetBlockSize(patch->intFacetMats, ndof*2));
    }
    CHKERRQ(VecSet(patch->intFacetMats, 0));

    ierr = MatCreateShell(PETSC_COMM_SELF, nIntFacets*ndof*2, nIntFacets*ndof*2, nIntFacets*ndof*2, nIntFacets*ndof*2,
                          (void*)patch->intFacetMats, &vecMat);CHKERRQ(ierr);
    CHKERRQ(MatShellSetOperation(vecMat, MATOP_SET_VALUES, (void(*)(void))&MatSetValues_PCPatch_Private));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF, 2*ndof*nIntFacets, 0, 1, &dofMap));
    CHKERRQ(ISGetIndices(dofMap, &dofMapArray));
    CHKERRQ(ISGetIndices(patch->allIntFacets, &intFacetsArray));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, nIntFacets, intFacetsArray, PETSC_USE_POINTER, &intFacetsIS));
    PetscStackPush("PCPatch user callback (interior facets)");
    /* TODO: Fix for DMPlex compute op, this bypasses a lot of the machinery and just assembles every element tensor. */
    CHKERRQ(patch->usercomputeopintfacet(pc, -1, NULL, vecMat, intFacetsIS, 2*ndof*nIntFacets, dofMapArray, NULL, patch->usercomputeopintfacetctx));
    PetscStackPop;
    CHKERRQ(ISDestroy(&intFacetsIS));
    CHKERRQ(MatDestroy(&vecMat));
    CHKERRQ(ISRestoreIndices(patch->allIntFacets, &intFacetsArray));
    CHKERRQ(ISRestoreIndices(dofMap, &dofMapArray));
    CHKERRQ(ISDestroy(&dofMap));
  }
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0));

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
  CHKERRQ(VecGetArrayRead(x, &xArray));
  CHKERRQ(VecGetArray(y, &yArray));
  if (scattertype == SCATTER_WITHARTIFICIAL) {
    CHKERRQ(PetscSectionGetDof(patch->gtolCountsWithArtificial, p, &dof));
    CHKERRQ(PetscSectionGetOffset(patch->gtolCountsWithArtificial, p, &offset));
    CHKERRQ(ISGetIndices(patch->gtolWithArtificial, &gtolArray));
  } else if (scattertype == SCATTER_WITHALL) {
    CHKERRQ(PetscSectionGetDof(patch->gtolCountsWithAll, p, &dof));
    CHKERRQ(PetscSectionGetOffset(patch->gtolCountsWithAll, p, &offset));
    CHKERRQ(ISGetIndices(patch->gtolWithAll, &gtolArray));
  } else {
    CHKERRQ(PetscSectionGetDof(patch->gtolCounts, p, &dof));
    CHKERRQ(PetscSectionGetOffset(patch->gtolCounts, p, &offset));
    CHKERRQ(ISGetIndices(patch->gtol, &gtolArray));
  }
  PetscCheckFalse(mode == INSERT_VALUES && scat != SCATTER_FORWARD,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't insert if not scattering forward");
  PetscCheckFalse(mode == ADD_VALUES    && scat != SCATTER_REVERSE,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't add if not scattering reverse");
  for (lidx = 0; lidx < dof; ++lidx) {
    const PetscInt gidx = gtolArray[offset+lidx];

    if (mode == INSERT_VALUES) yArray[lidx]  = xArray[gidx]; /* Forward */
    else                       yArray[gidx] += xArray[lidx]; /* Reverse */
  }
  if (scattertype == SCATTER_WITHARTIFICIAL) {
    CHKERRQ(ISRestoreIndices(patch->gtolWithArtificial, &gtolArray));
  } else if (scattertype == SCATTER_WITHALL) {
    CHKERRQ(ISRestoreIndices(patch->gtolWithAll, &gtolArray));
  } else {
    CHKERRQ(ISRestoreIndices(patch->gtol, &gtolArray));
  }
  CHKERRQ(VecRestoreArrayRead(x, &xArray));
  CHKERRQ(VecRestoreArray(y, &yArray));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_PATCH_Linear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  const char    *prefix;
  PetscInt       i;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscCheckFalse(!patch->save_operators && patch->denseinverse,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Can't have dense inverse without save operators");
    if (!patch->denseinverse) {
      CHKERRQ(PetscMalloc1(patch->npatch, &patch->solver));
      CHKERRQ(PCGetOptionsPrefix(pc, &prefix));
      for (i = 0; i < patch->npatch; ++i) {
        KSP ksp;
        PC  subpc;

        CHKERRQ(KSPCreate(PETSC_COMM_SELF, &ksp));
        CHKERRQ(KSPSetErrorIfNotConverged(ksp, pc->erroriffailure));
        CHKERRQ(KSPSetOptionsPrefix(ksp, prefix));
        CHKERRQ(KSPAppendOptionsPrefix(ksp, "sub_"));
        CHKERRQ(PetscObjectIncrementTabLevel((PetscObject) ksp, (PetscObject) pc, 1));
        CHKERRQ(KSPGetPC(ksp, &subpc));
        CHKERRQ(PetscObjectIncrementTabLevel((PetscObject) subpc, (PetscObject) pc, 1));
        CHKERRQ(PetscLogObjectParent((PetscObject) pc, (PetscObject) ksp));
        patch->solver[i] = (PetscObject) ksp;
      }
    }
  }
  if (patch->save_operators) {
    if (patch->precomputeElementTensors) {
      CHKERRQ(PCPatchPrecomputePatchTensors_Private(pc));
    }
    for (i = 0; i < patch->npatch; ++i) {
      CHKERRQ(PCPatchComputeOperator_Internal(pc, NULL, patch->mat[i], i, PETSC_FALSE));
      if (!patch->denseinverse) {
        CHKERRQ(KSPSetOperators((KSP) patch->solver[i], patch->mat[i], patch->mat[i]));
      } else if (patch->mat[i] && !patch->densesolve) {
        /* Setup matmult callback */
        CHKERRQ(MatGetOperation(patch->mat[i], MATOP_MULT, (void (**)(void))&patch->densesolve));
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

      CHKERRQ(MatGetSize(patch->mat[i], &dof, NULL));
      if (dof == 0) {
        patch->matWithArtificial[i] = NULL;
        continue;
      }

      CHKERRQ(PCPatchCreateMatrix_Private(pc, i, &matSquare, PETSC_TRUE));
      CHKERRQ(PCPatchComputeOperator_Internal(pc, NULL, matSquare, i, PETSC_TRUE));

      CHKERRQ(MatGetSize(matSquare, &dof, NULL));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF, dof, 0, 1, &rowis));
      if (pc->setupcalled) {
        CHKERRQ(MatCreateSubMatrix(matSquare, rowis, patch->dofMappingWithoutToWithArtificial[i], MAT_REUSE_MATRIX, &patch->matWithArtificial[i]));
      } else {
        CHKERRQ(MatCreateSubMatrix(matSquare, rowis, patch->dofMappingWithoutToWithArtificial[i], MAT_INITIAL_MATRIX, &patch->matWithArtificial[i]));
      }
      CHKERRQ(ISDestroy(&rowis));
      CHKERRQ(MatDestroy(&matSquare));
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

    CHKERRQ(PetscLogEventBegin(PC_Patch_CreatePatches, pc, 0, 0, 0));

    isNonlinear = patch->isNonlinear;
    if (!patch->nsubspaces) {
      DM           dm, plex;
      PetscSection s;
      PetscInt     cStart, cEnd, c, Nf, f, numGlobalBcs = 0, *globalBcs, *Nb, **cellDofs;

      CHKERRQ(PCGetDM(pc, &dm));
      PetscCheck(dm,PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONG, "Must set DM for PCPATCH or call PCPatchSetDiscretisationInfo()");
      CHKERRQ(DMConvert(dm, DMPLEX, &plex));
      dm = plex;
      CHKERRQ(DMGetLocalSection(dm, &s));
      CHKERRQ(PetscSectionGetNumFields(s, &Nf));
      CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
      for (p = pStart; p < pEnd; ++p) {
        PetscInt cdof;
        CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
        numGlobalBcs += cdof;
      }
      CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
      CHKERRQ(PetscMalloc3(Nf, &Nb, Nf, &cellDofs, numGlobalBcs, &globalBcs));
      for (f = 0; f < Nf; ++f) {
        PetscFE        fe;
        PetscDualSpace sp;
        PetscInt       cdoff = 0;

        CHKERRQ(DMGetField(dm, f, NULL, (PetscObject *) &fe));
        /* CHKERRQ(PetscFEGetNumComponents(fe, &Nc[f])); */
        CHKERRQ(PetscFEGetDualSpace(fe, &sp));
        CHKERRQ(PetscDualSpaceGetDimension(sp, &Nb[f]));

        CHKERRQ(PetscMalloc1((cEnd-cStart)*Nb[f], &cellDofs[f]));
        for (c = cStart; c < cEnd; ++c) {
          PetscInt *closure = NULL;
          PetscInt  clSize  = 0, cl;

          CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
          for (cl = 0; cl < clSize*2; cl += 2) {
            const PetscInt p = closure[cl];
            PetscInt       fdof, d, foff;

            CHKERRQ(PetscSectionGetFieldDof(s, p, f, &fdof));
            CHKERRQ(PetscSectionGetFieldOffset(s, p, f, &foff));
            for (d = 0; d < fdof; ++d, ++cdoff) cellDofs[f][cdoff] = foff + d;
          }
          CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &clSize, &closure));
        }
        PetscCheckFalse(cdoff != (cEnd-cStart)*Nb[f],PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_SIZ, "Total number of cellDofs %D for field %D should be Nc (%D) * cellDof (%D)", cdoff, f, cEnd-cStart, Nb[f]);
      }
      numGlobalBcs = 0;
      for (p = pStart; p < pEnd; ++p) {
        const PetscInt *ind;
        PetscInt        off, cdof, d;

        CHKERRQ(PetscSectionGetOffset(s, p, &off));
        CHKERRQ(PetscSectionGetConstraintDof(s, p, &cdof));
        CHKERRQ(PetscSectionGetConstraintIndices(s, p, &ind));
        for (d = 0; d < cdof; ++d) globalBcs[numGlobalBcs++] = off + ind[d];
      }

      CHKERRQ(PCPatchSetDiscretisationInfoCombined(pc, dm, Nb, (const PetscInt **) cellDofs, numGlobalBcs, globalBcs, numGlobalBcs, globalBcs));
      for (f = 0; f < Nf; ++f) {
        CHKERRQ(PetscFree(cellDofs[f]));
      }
      CHKERRQ(PetscFree3(Nb, cellDofs, globalBcs));
      CHKERRQ(PCPatchSetComputeFunction(pc, PCPatchComputeFunction_DMPlex_Private, NULL));
      CHKERRQ(PCPatchSetComputeOperator(pc, PCPatchComputeOperator_DMPlex_Private, NULL));
      CHKERRQ(DMDestroy(&dm));
    }

    localSize = patch->subspaceOffsets[patch->nsubspaces];
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, localSize, &patch->localRHS));
    CHKERRQ(VecSetUp(patch->localRHS));
    CHKERRQ(VecDuplicate(patch->localRHS, &patch->localUpdate));
    CHKERRQ(PCPatchCreateCellPatches(pc));
    CHKERRQ(PCPatchCreateCellPatchDiscretisationInfo(pc));

    /* OK, now build the work vectors */
    CHKERRQ(PetscSectionGetChart(patch->gtolCounts, &pStart, &pEnd));

    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
      CHKERRQ(PetscMalloc1(patch->npatch, &patch->dofMappingWithoutToWithArtificial));
    }
    if (isNonlinear) {
      CHKERRQ(PetscMalloc1(patch->npatch, &patch->dofMappingWithoutToWithAll));
    }
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof;

      CHKERRQ(PetscSectionGetDof(patch->gtolCounts, p, &dof));
      maxDof = PetscMax(maxDof, dof);
      if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
        const PetscInt    *gtolArray, *gtolArrayWithArtificial = NULL;
        PetscInt           numPatchDofs, offset;
        PetscInt           numPatchDofsWithArtificial, offsetWithArtificial;
        PetscInt           dofWithoutArtificialCounter = 0;
        PetscInt          *patchWithoutArtificialToWithArtificialArray;

        CHKERRQ(PetscSectionGetDof(patch->gtolCountsWithArtificial, p, &dof));
        maxDofWithArtificial = PetscMax(maxDofWithArtificial, dof);

        /* Now build the mapping that for a dof in a patch WITHOUT dofs that have artificial bcs gives the */
        /* the index in the patch with all dofs */
        CHKERRQ(ISGetIndices(patch->gtol, &gtolArray));

        CHKERRQ(PetscSectionGetDof(patch->gtolCounts, p, &numPatchDofs));
        if (numPatchDofs == 0) {
          patch->dofMappingWithoutToWithArtificial[p-pStart] = NULL;
          continue;
        }

        CHKERRQ(PetscSectionGetOffset(patch->gtolCounts, p, &offset));
        CHKERRQ(ISGetIndices(patch->gtolWithArtificial, &gtolArrayWithArtificial));
        CHKERRQ(PetscSectionGetDof(patch->gtolCountsWithArtificial, p, &numPatchDofsWithArtificial));
        CHKERRQ(PetscSectionGetOffset(patch->gtolCountsWithArtificial, p, &offsetWithArtificial));

        CHKERRQ(PetscMalloc1(numPatchDofs, &patchWithoutArtificialToWithArtificialArray));
        for (i=0; i<numPatchDofsWithArtificial; i++) {
          if (gtolArrayWithArtificial[i+offsetWithArtificial] == gtolArray[offset+dofWithoutArtificialCounter]) {
            patchWithoutArtificialToWithArtificialArray[dofWithoutArtificialCounter] = i;
            dofWithoutArtificialCounter++;
            if (dofWithoutArtificialCounter == numPatchDofs)
              break;
          }
        }
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numPatchDofs, patchWithoutArtificialToWithArtificialArray, PETSC_OWN_POINTER, &patch->dofMappingWithoutToWithArtificial[p-pStart]));
        CHKERRQ(ISRestoreIndices(patch->gtol, &gtolArray));
        CHKERRQ(ISRestoreIndices(patch->gtolWithArtificial, &gtolArrayWithArtificial));
      }
      if (isNonlinear) {
        const PetscInt    *gtolArray, *gtolArrayWithAll = NULL;
        PetscInt           numPatchDofs, offset;
        PetscInt           numPatchDofsWithAll, offsetWithAll;
        PetscInt           dofWithoutAllCounter = 0;
        PetscInt          *patchWithoutAllToWithAllArray;

        /* Now build the mapping that for a dof in a patch WITHOUT dofs that have artificial bcs gives the */
        /* the index in the patch with all dofs */
        CHKERRQ(ISGetIndices(patch->gtol, &gtolArray));

        CHKERRQ(PetscSectionGetDof(patch->gtolCounts, p, &numPatchDofs));
        if (numPatchDofs == 0) {
          patch->dofMappingWithoutToWithAll[p-pStart] = NULL;
          continue;
        }

        CHKERRQ(PetscSectionGetOffset(patch->gtolCounts, p, &offset));
        CHKERRQ(ISGetIndices(patch->gtolWithAll, &gtolArrayWithAll));
        CHKERRQ(PetscSectionGetDof(patch->gtolCountsWithAll, p, &numPatchDofsWithAll));
        CHKERRQ(PetscSectionGetOffset(patch->gtolCountsWithAll, p, &offsetWithAll));

        CHKERRQ(PetscMalloc1(numPatchDofs, &patchWithoutAllToWithAllArray));

        for (i=0; i<numPatchDofsWithAll; i++) {
          if (gtolArrayWithAll[i+offsetWithAll] == gtolArray[offset+dofWithoutAllCounter]) {
            patchWithoutAllToWithAllArray[dofWithoutAllCounter] = i;
            dofWithoutAllCounter++;
            if (dofWithoutAllCounter == numPatchDofs)
              break;
          }
        }
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, numPatchDofs, patchWithoutAllToWithAllArray, PETSC_OWN_POINTER, &patch->dofMappingWithoutToWithAll[p-pStart]));
        CHKERRQ(ISRestoreIndices(patch->gtol, &gtolArray));
        CHKERRQ(ISRestoreIndices(patch->gtolWithAll, &gtolArrayWithAll));
      }
    }
    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, maxDofWithArtificial, &patch->patchRHSWithArtificial));
      CHKERRQ(VecSetUp(patch->patchRHSWithArtificial));
    }
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, maxDof, &patch->patchRHS));
    CHKERRQ(VecSetUp(patch->patchRHS));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF, maxDof, &patch->patchUpdate));
    CHKERRQ(VecSetUp(patch->patchUpdate));
    if (patch->save_operators) {
      CHKERRQ(PetscMalloc1(patch->npatch, &patch->mat));
      for (i = 0; i < patch->npatch; ++i) {
        CHKERRQ(PCPatchCreateMatrix_Private(pc, i, &patch->mat[i], PETSC_FALSE));
      }
    }
    CHKERRQ(PetscLogEventEnd(PC_Patch_CreatePatches, pc, 0, 0, 0));

    /* If desired, calculate weights for dof multiplicity */
    if (patch->partition_of_unity) {
      PetscScalar *input = NULL;
      PetscScalar *output = NULL;
      Vec         global;

      CHKERRQ(VecDuplicate(patch->localRHS, &patch->dof_weights));
      if (patch->local_composition_type == PC_COMPOSITE_ADDITIVE) {
        for (i = 0; i < patch->npatch; ++i) {
          PetscInt dof;

          CHKERRQ(PetscSectionGetDof(patch->gtolCounts, i+pStart, &dof));
          if (dof <= 0) continue;
          CHKERRQ(VecSet(patch->patchRHS, 1.0));
          CHKERRQ(PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchRHS, patch->dof_weights, ADD_VALUES, SCATTER_REVERSE, SCATTER_INTERIOR));
        }
      } else {
        /* multiplicative is actually only locally multiplicative and globally additive. need the pou where the mesh decomposition overlaps */
        CHKERRQ(VecSet(patch->dof_weights, 1.0));
      }

      VecDuplicate(patch->dof_weights, &global);
      VecSet(global, 0.);

      CHKERRQ(VecGetArray(patch->dof_weights, &input));
      CHKERRQ(VecGetArray(global, &output));
      CHKERRQ(PetscSFReduceBegin(patch->sectionSF, MPIU_SCALAR, input, output, MPI_SUM));
      CHKERRQ(PetscSFReduceEnd(patch->sectionSF, MPIU_SCALAR, input, output, MPI_SUM));
      CHKERRQ(VecRestoreArray(patch->dof_weights, &input));
      CHKERRQ(VecRestoreArray(global, &output));

      CHKERRQ(VecReciprocal(global));

      CHKERRQ(VecGetArray(patch->dof_weights, &output));
      CHKERRQ(VecGetArray(global, &input));
      CHKERRQ(PetscSFBcastBegin(patch->sectionSF, MPIU_SCALAR, input, output,MPI_REPLACE));
      CHKERRQ(PetscSFBcastEnd(patch->sectionSF, MPIU_SCALAR, input, output,MPI_REPLACE));
      CHKERRQ(VecRestoreArray(patch->dof_weights, &output));
      CHKERRQ(VecRestoreArray(global, &input));
      CHKERRQ(VecDestroy(&global));
    }
    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE && patch->save_operators) {
      CHKERRQ(PetscMalloc1(patch->npatch, &patch->matWithArtificial));
    }
  }
  CHKERRQ((*patch->setupsolver)(pc));
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
    CHKERRQ((*patch->densesolve)(patch->mat[i], x, y));
    PetscFunctionReturn(0);
  }
  ksp = (KSP) patch->solver[i];
  if (!patch->save_operators) {
    Mat mat;

    CHKERRQ(PCPatchCreateMatrix_Private(pc, i, &mat, PETSC_FALSE));
    /* Populate operator here. */
    CHKERRQ(PCPatchComputeOperator_Internal(pc, NULL, mat, i, PETSC_FALSE));
    CHKERRQ(KSPSetOperators(ksp, mat, mat));
    /* Drop reference so the KSPSetOperators below will blow it away. */
    CHKERRQ(MatDestroy(&mat));
  }
  CHKERRQ(PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0));
  if (!ksp->setfromoptionscalled) {
    CHKERRQ(KSPSetFromOptions(ksp));
  }
  /* Disgusting trick to reuse work vectors */
  CHKERRQ(KSPGetOperators(ksp, &op, NULL));
  CHKERRQ(MatGetLocalSize(op, &m, &n));
  x->map->n = m;
  y->map->n = n;
  x->map->N = m;
  y->map->N = n;
  CHKERRQ(KSPSolve(ksp, x, y));
  CHKERRQ(KSPCheckSolve(ksp, pc, y));
  CHKERRQ(PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0));
  if (!patch->save_operators) {
    PC pc;
    CHKERRQ(KSPSetOperators(ksp, NULL, NULL));
    CHKERRQ(KSPGetPC(ksp, &pc));
    /* Destroy PC context too, otherwise the factored matrix hangs around. */
    CHKERRQ(PCReset(pc));
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
    CHKERRQ(PCPatchCreateMatrix_Private(pc, i, &matSquare, PETSC_TRUE));
    CHKERRQ(PCPatchComputeOperator_Internal(pc, NULL, matSquare, i, PETSC_TRUE));
    CHKERRQ(MatGetSize(matSquare, &dof, NULL));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF, dof, 0, 1, &rowis));
    CHKERRQ(MatCreateSubMatrix(matSquare, rowis, patch->dofMappingWithoutToWithArtificial[i], MAT_INITIAL_MATRIX, &multMat));
    CHKERRQ(MatDestroy(&matSquare));
    CHKERRQ(ISDestroy(&rowis));
  }
  /* Disgusting trick to reuse work vectors */
  CHKERRQ(MatGetLocalSize(multMat, &m, &n));
  patch->patchUpdate->map->n = n;
  patch->patchRHSWithArtificial->map->n = m;
  patch->patchUpdate->map->N = n;
  patch->patchRHSWithArtificial->map->N = m;
  CHKERRQ(MatMult(multMat, patch->patchUpdate, patch->patchRHSWithArtificial));
  CHKERRQ(VecScale(patch->patchRHSWithArtificial, -1.0));
  CHKERRQ(PCPatch_ScatterLocal_Private(pc, i + pStart, patch->patchRHSWithArtificial, patch->localRHS, ADD_VALUES, SCATTER_REVERSE, SCATTER_WITHARTIFICIAL));
  if (!patch->save_operators) {
    CHKERRQ(MatDestroy(&multMat));
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
  CHKERRQ(PetscLogEventBegin(PC_Patch_Apply, pc, 0, 0, 0));
  CHKERRQ(PetscOptionsPushGetViewerOff(PETSC_TRUE));
  /* start, end, inc have 2 entries to manage a second backward sweep if we symmetrize */
  end[0]   = patch->npatch;
  start[1] = patch->npatch-1;
  if (patch->user_patches) {
    CHKERRQ(ISGetLocalSize(patch->iterationSet, &end[0]));
    start[1] = end[0] - 1;
    CHKERRQ(ISGetIndices(patch->iterationSet, &iterationSet));
  }
  /* Scatter from global space into overlapped local spaces */
  CHKERRQ(VecGetArrayRead(x, &globalRHS));
  CHKERRQ(VecGetArray(patch->localRHS, &localRHS));
  CHKERRQ(PetscSFBcastBegin(patch->sectionSF, MPIU_SCALAR, globalRHS, localRHS,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(patch->sectionSF, MPIU_SCALAR, globalRHS, localRHS,MPI_REPLACE));
  CHKERRQ(VecRestoreArrayRead(x, &globalRHS));
  CHKERRQ(VecRestoreArray(patch->localRHS, &localRHS));

  CHKERRQ(VecSet(patch->localUpdate, 0.0));
  CHKERRQ(PetscSectionGetChart(patch->gtolCounts, &pStart, NULL));
  CHKERRQ(PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0));
  for (sweep = 0; sweep < nsweep; sweep++) {
    for (j = start[sweep]; j*inc[sweep] < end[sweep]*inc[sweep]; j += inc[sweep]) {
      PetscInt i = patch->user_patches ? iterationSet[j] : j;
      PetscInt start, len;

      CHKERRQ(PetscSectionGetDof(patch->gtolCounts, i+pStart, &len));
      CHKERRQ(PetscSectionGetOffset(patch->gtolCounts, i+pStart, &start));
      /* TODO: Squash out these guys in the setup as well. */
      if (len <= 0) continue;
      /* TODO: Do we need different scatters for X and Y? */
      CHKERRQ(PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localRHS, patch->patchRHS, INSERT_VALUES, SCATTER_FORWARD, SCATTER_INTERIOR));
      CHKERRQ((*patch->applysolver)(pc, i, patch->patchRHS, patch->patchUpdate));
      CHKERRQ(PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchUpdate, patch->localUpdate, ADD_VALUES, SCATTER_REVERSE, SCATTER_INTERIOR));
      if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
        CHKERRQ((*patch->updatemultiplicative)(pc, i, pStart));
      }
    }
  }
  CHKERRQ(PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0));
  if (patch->user_patches) CHKERRQ(ISRestoreIndices(patch->iterationSet, &iterationSet));
  /* XXX: should we do this on the global vector? */
  if (patch->partition_of_unity) {
    CHKERRQ(VecPointwiseMult(patch->localUpdate, patch->localUpdate, patch->dof_weights));
  }
  /* Now patch->localUpdate contains the solution of the patch solves, so we need to combine them all. */
  CHKERRQ(VecSet(y, 0.0));
  CHKERRQ(VecGetArray(y, &globalUpdate));
  CHKERRQ(VecGetArrayRead(patch->localUpdate, &localUpdate));
  CHKERRQ(PetscSFReduceBegin(patch->sectionSF, MPIU_SCALAR, localUpdate, globalUpdate, MPI_SUM));
  CHKERRQ(PetscSFReduceEnd(patch->sectionSF, MPIU_SCALAR, localUpdate, globalUpdate, MPI_SUM));
  CHKERRQ(VecRestoreArrayRead(patch->localUpdate, &localUpdate));

  /* Now we need to send the global BC values through */
  CHKERRQ(VecGetArrayRead(x, &globalRHS));
  CHKERRQ(ISGetSize(patch->globalBcNodes, &numBcs));
  CHKERRQ(ISGetIndices(patch->globalBcNodes, &bcNodes));
  CHKERRQ(VecGetLocalSize(x, &n));
  for (bc = 0; bc < numBcs; ++bc) {
    const PetscInt idx = bcNodes[bc];
    if (idx < n) globalUpdate[idx] = globalRHS[idx];
  }

  CHKERRQ(ISRestoreIndices(patch->globalBcNodes, &bcNodes));
  CHKERRQ(VecRestoreArrayRead(x, &globalRHS));
  CHKERRQ(VecRestoreArray(y, &globalUpdate));

  CHKERRQ(PetscOptionsPopGetViewerOff());
  CHKERRQ(PetscLogEventEnd(PC_Patch_Apply, pc, 0, 0, 0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH_Linear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(KSPReset((KSP) patch->solver[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;

  CHKERRQ(PetscSFDestroy(&patch->sectionSF));
  CHKERRQ(PetscSectionDestroy(&patch->cellCounts));
  CHKERRQ(PetscSectionDestroy(&patch->pointCounts));
  CHKERRQ(PetscSectionDestroy(&patch->cellNumbering));
  CHKERRQ(PetscSectionDestroy(&patch->gtolCounts));
  CHKERRQ(ISDestroy(&patch->gtol));
  CHKERRQ(ISDestroy(&patch->cells));
  CHKERRQ(ISDestroy(&patch->points));
  CHKERRQ(ISDestroy(&patch->dofs));
  CHKERRQ(ISDestroy(&patch->offs));
  CHKERRQ(PetscSectionDestroy(&patch->patchSection));
  CHKERRQ(ISDestroy(&patch->ghostBcNodes));
  CHKERRQ(ISDestroy(&patch->globalBcNodes));
  CHKERRQ(PetscSectionDestroy(&patch->gtolCountsWithArtificial));
  CHKERRQ(ISDestroy(&patch->gtolWithArtificial));
  CHKERRQ(ISDestroy(&patch->dofsWithArtificial));
  CHKERRQ(ISDestroy(&patch->offsWithArtificial));
  CHKERRQ(PetscSectionDestroy(&patch->gtolCountsWithAll));
  CHKERRQ(ISDestroy(&patch->gtolWithAll));
  CHKERRQ(ISDestroy(&patch->dofsWithAll));
  CHKERRQ(ISDestroy(&patch->offsWithAll));
  CHKERRQ(VecDestroy(&patch->cellMats));
  CHKERRQ(VecDestroy(&patch->intFacetMats));
  CHKERRQ(ISDestroy(&patch->allCells));
  CHKERRQ(ISDestroy(&patch->intFacets));
  CHKERRQ(ISDestroy(&patch->extFacets));
  CHKERRQ(ISDestroy(&patch->intFacetsToPatchCell));
  CHKERRQ(PetscSectionDestroy(&patch->intFacetCounts));
  CHKERRQ(PetscSectionDestroy(&patch->extFacetCounts));

  if (patch->dofSection) for (i = 0; i < patch->nsubspaces; i++) CHKERRQ(PetscSectionDestroy(&patch->dofSection[i]));
  CHKERRQ(PetscFree(patch->dofSection));
  CHKERRQ(PetscFree(patch->bs));
  CHKERRQ(PetscFree(patch->nodesPerCell));
  if (patch->cellNodeMap) for (i = 0; i < patch->nsubspaces; i++) CHKERRQ(PetscFree(patch->cellNodeMap[i]));
  CHKERRQ(PetscFree(patch->cellNodeMap));
  CHKERRQ(PetscFree(patch->subspaceOffsets));

  CHKERRQ((*patch->resetsolver)(pc));

  if (patch->subspaces_to_exclude) {
    CHKERRQ(PetscHSetIDestroy(&patch->subspaces_to_exclude));
  }

  CHKERRQ(VecDestroy(&patch->localRHS));
  CHKERRQ(VecDestroy(&patch->localUpdate));
  CHKERRQ(VecDestroy(&patch->patchRHS));
  CHKERRQ(VecDestroy(&patch->patchUpdate));
  CHKERRQ(VecDestroy(&patch->dof_weights));
  if (patch->patch_dof_weights) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(VecDestroy(&patch->patch_dof_weights[i]));
    CHKERRQ(PetscFree(patch->patch_dof_weights));
  }
  if (patch->mat) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(MatDestroy(&patch->mat[i]));
    CHKERRQ(PetscFree(patch->mat));
  }
  if (patch->matWithArtificial) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(MatDestroy(&patch->matWithArtificial[i]));
    CHKERRQ(PetscFree(patch->matWithArtificial));
  }
  CHKERRQ(VecDestroy(&patch->patchRHSWithArtificial));
  if (patch->dofMappingWithoutToWithArtificial) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(ISDestroy(&patch->dofMappingWithoutToWithArtificial[i]));
    CHKERRQ(PetscFree(patch->dofMappingWithoutToWithArtificial));

  }
  if (patch->dofMappingWithoutToWithAll) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(ISDestroy(&patch->dofMappingWithoutToWithAll[i]));
    CHKERRQ(PetscFree(patch->dofMappingWithoutToWithAll));

  }
  CHKERRQ(PetscFree(patch->sub_mat_type));
  if (patch->userIS) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(ISDestroy(&patch->userIS[i]));
    CHKERRQ(PetscFree(patch->userIS));
  }
  CHKERRQ(PetscFree(patch->precomputedTensorLocations));
  CHKERRQ(PetscFree(patch->precomputedIntFacetTensorLocations));

  patch->bs          = NULL;
  patch->cellNodeMap = NULL;
  patch->nsubspaces  = 0;
  CHKERRQ(ISDestroy(&patch->iterationSet));

  CHKERRQ(PetscViewerDestroy(&patch->viewerSection));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH_Linear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) CHKERRQ(KSPDestroy((KSP *) &patch->solver[i]));
    CHKERRQ(PetscFree(patch->solver));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;

  PetscFunctionBegin;
  CHKERRQ(PCReset_PATCH(pc));
  CHKERRQ((*patch->destroysolver)(pc));
  CHKERRQ(PetscFree(pc->data));
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
  CHKERRQ(PetscObjectGetComm((PetscObject) pc, &comm));
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject) pc, &prefix));
  CHKERRQ(PetscOptionsHead(PetscOptionsObject, "Patch solver options"));

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_save_operators", patch->classname));
  CHKERRQ(PetscOptionsBool(option,  "Store all patch operators for lifetime of object?", "PCPatchSetSaveOperators", patch->save_operators, &patch->save_operators, &flg));

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_precompute_element_tensors", patch->classname));
  CHKERRQ(PetscOptionsBool(option,  "Compute each element tensor only once?", "PCPatchSetPrecomputeElementTensors", patch->precomputeElementTensors, &patch->precomputeElementTensors, &flg));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_partition_of_unity", patch->classname));
  CHKERRQ(PetscOptionsBool(option, "Weight contributions by dof multiplicity?", "PCPatchSetPartitionOfUnity", patch->partition_of_unity, &patch->partition_of_unity, &flg));

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_local_type", patch->classname));
  CHKERRQ(PetscOptionsEnum(option,"Type of local solver composition (additive or multiplicative)","PCPatchSetLocalComposition",PCCompositeTypes,(PetscEnum)loctype,(PetscEnum*)&loctype,&flg));
  if (flg) CHKERRQ(PCPatchSetLocalComposition(pc, loctype));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_dense_inverse", patch->classname));
  CHKERRQ(PetscOptionsBool(option, "Compute inverses of patch matrices and apply directly? Ignores KSP/PC settings on patch.", "PCPatchSetDenseInverse", patch->denseinverse, &patch->denseinverse, &flg));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_construct_dim", patch->classname));
  CHKERRQ(PetscOptionsInt(option, "What dimension of mesh point to construct patches by? (0 = vertices)", "PCPATCH", patch->dim, &patch->dim, &dimflg));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_construct_codim", patch->classname));
  CHKERRQ(PetscOptionsInt(option, "What co-dimension of mesh point to construct patches by? (0 = cells)", "PCPATCH", patch->codim, &patch->codim, &codimflg));
  PetscCheckFalse(dimflg && codimflg,comm, PETSC_ERR_ARG_WRONG, "Can only set one of dimension or co-dimension");

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_construct_type", patch->classname));
  CHKERRQ(PetscOptionsEnum(option, "How should the patches be constructed?", "PCPatchSetConstructType", PCPatchConstructTypes, (PetscEnum) patchConstructionType, (PetscEnum *) &patchConstructionType, &flg));
  if (flg) CHKERRQ(PCPatchSetConstructType(pc, patchConstructionType, NULL, NULL));

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_vanka_dim", patch->classname));
  CHKERRQ(PetscOptionsInt(option, "Topological dimension of entities for Vanka to ignore", "PCPATCH", patch->vankadim, &patch->vankadim, &flg));

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_ignore_dim", patch->classname));
  CHKERRQ(PetscOptionsInt(option, "Topological dimension of entities for completion to ignore", "PCPATCH", patch->ignoredim, &patch->ignoredim, &flg));

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_pardecomp_overlap", patch->classname));
  CHKERRQ(PetscOptionsInt(option, "What overlap should we use in construct type pardecomp?", "PCPATCH", patch->pardecomp_overlap, &patch->pardecomp_overlap, &flg));

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_sub_mat_type", patch->classname));
  CHKERRQ(PetscOptionsFList(option, "Matrix type for patch solves", "PCPatchSetSubMatType", MatList, NULL, sub_mat_type, PETSC_MAX_PATH_LEN, &flg));
  if (flg) CHKERRQ(PCPatchSetSubMatType(pc, sub_mat_type));

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_symmetrise_sweep", patch->classname));
  CHKERRQ(PetscOptionsBool(option, "Go start->end, end->start?", "PCPATCH", patch->symmetrise_sweep, &patch->symmetrise_sweep, &flg));

  /* If the user has set the number of subspaces, use that for the buffer size,
     otherwise use a large number */
  if (patch->nsubspaces <= 0) {
    nfields = 128;
  } else {
    nfields = patch->nsubspaces;
  }
  CHKERRQ(PetscMalloc1(nfields, &ifields));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_exclude_subspaces", patch->classname));
  CHKERRQ(PetscOptionsGetIntArray(((PetscObject)pc)->options,((PetscObject)pc)->prefix,option,ifields,&nfields,&flg));
  PetscCheckFalse(flg && (patchConstructionType == PC_PATCH_USER),comm, PETSC_ERR_ARG_INCOMP, "We cannot support excluding a subspace with user patches because we do not index patches with a mesh point");
  if (flg) {
    CHKERRQ(PetscHSetIClear(patch->subspaces_to_exclude));
    for (k = 0; k < nfields; k++) {
      CHKERRQ(PetscHSetIAdd(patch->subspaces_to_exclude, ifields[k]));
    }
  }
  CHKERRQ(PetscFree(ifields));

  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_patches_view", patch->classname));
  CHKERRQ(PetscOptionsBool(option, "Print out information during patch construction", "PCPATCH", patch->viewPatches, &patch->viewPatches, &flg));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_cells_view", patch->classname));
  CHKERRQ(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerCells, &patch->formatCells, &patch->viewCells));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_interior_facets_view", patch->classname));
  CHKERRQ(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerIntFacets, &patch->formatIntFacets, &patch->viewIntFacets));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_exterior_facets_view", patch->classname));
  CHKERRQ(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerExtFacets, &patch->formatExtFacets, &patch->viewExtFacets));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_points_view", patch->classname));
  CHKERRQ(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerPoints, &patch->formatPoints, &patch->viewPoints));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_section_view", patch->classname));
  CHKERRQ(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerSection, &patch->formatSection, &patch->viewSection));
  CHKERRQ(PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_mat_view", patch->classname));
  CHKERRQ(PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerMatrix, &patch->formatMatrix, &patch->viewMatrix));
  CHKERRQ(PetscOptionsTail());
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
      CHKERRQ(KSPSetFromOptions((KSP) patch->solver[i]));
    }
    CHKERRQ(KSPSetUp((KSP) patch->solver[i]));
    CHKERRQ(KSPGetConvergedReason((KSP) patch->solver[i], &reason));
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  if (!isascii) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) pc), &rank));
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Subspace Correction preconditioner with %d patches\n", patch->npatch));
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Schwarz type: multiplicative\n"));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Schwarz type: additive\n"));
  }
  if (patch->partition_of_unity) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Weighting by partition of unity\n"));
  else                           CHKERRQ(PetscViewerASCIIPrintf(viewer, "Not weighting by partition of unity\n"));
  if (patch->symmetrise_sweep) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Symmetrising sweep (start->end, then end->start)\n"));
  else                         CHKERRQ(PetscViewerASCIIPrintf(viewer, "Not symmetrising sweep\n"));
  if (!patch->precomputeElementTensors) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Not precomputing element tensors (overlapping cells rebuilt in every patch assembly)\n"));
  else                            CHKERRQ(PetscViewerASCIIPrintf(viewer, "Precomputing element tensors (each cell assembled only once)\n"));
  if (!patch->save_operators) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Not saving patch operators (rebuilt every PCApply)\n"));
  else                        CHKERRQ(PetscViewerASCIIPrintf(viewer, "Saving patch operators (rebuilt every PCSetUp)\n"));
  if (patch->patchconstructop == PCPatchConstruct_Star)       CHKERRQ(PetscViewerASCIIPrintf(viewer, "Patch construction operator: star\n"));
  else if (patch->patchconstructop == PCPatchConstruct_Vanka) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Patch construction operator: Vanka\n"));
  else if (patch->patchconstructop == PCPatchConstruct_User)  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Patch construction operator: user-specified\n"));
  else                                                        CHKERRQ(PetscViewerASCIIPrintf(viewer, "Patch construction operator: unknown\n"));

  if (patch->denseinverse) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Explicitly forming dense inverse and applying patch solver via MatMult.\n"));
  } else {
    if (patch->isNonlinear) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "SNES on patches (all same):\n"));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "KSP on patches (all same):\n"));
    }
    if (patch->solver) {
      CHKERRQ(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
      if (rank == 0) {
        CHKERRQ(PetscViewerASCIIPushTab(sviewer));
        CHKERRQ(PetscObjectView(patch->solver[0], sviewer));
        CHKERRQ(PetscViewerASCIIPopTab(sviewer));
      }
      CHKERRQ(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
    } else {
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      CHKERRQ(PetscViewerASCIIPrintf(viewer, "Solver not yet set.\n"));
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
  }
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
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
  CHKERRQ(PetscNewLog(pc, &patch));

  if (patch->subspaces_to_exclude) {
    CHKERRQ(PetscHSetIDestroy(&patch->subspaces_to_exclude));
  }
  CHKERRQ(PetscHSetICreate(&patch->subspaces_to_exclude));

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
  CHKERRQ(PetscStrallocpy(MATDENSE, (char **) &patch->sub_mat_type));
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
