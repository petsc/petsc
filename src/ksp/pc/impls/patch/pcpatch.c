#include <petsc/private/pcpatchimpl.h>     /*I "petscpc.h" I*/
#include <petsc/private/kspimpl.h>         /* For ksp->setfromoptionscalled */
#include <petsc/private/vecimpl.h>         /* For vec->map */
#include <petsc/private/dmpleximpl.h> /* For DMPlexComputeJacobian_Patch_Internal() */
#include <petscsf.h>
#include <petscbt.h>
#include <petscds.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

PetscLogEvent PC_Patch_CreatePatches, PC_Patch_ComputeOp, PC_Patch_Solve, PC_Patch_Apply, PC_Patch_Prealloc;

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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscHSetIClear(ht);

  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);

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

  for (opoint = pStart; opoint < pEnd; ++opoint) {
    if (ghost) {ierr = DMLabelHasPoint(ghost, opoint, &flg);CHKERRQ(ierr);}
    else       {ierr = PetscFindInt(opoint, nleaves, leaves, &loc);CHKERRQ(ierr); flg = loc >=0 ? PETSC_TRUE : PETSC_FALSE;}
    /* Not an owned entity, don't make a cell patch. */
    if (flg) continue;
    ierr = PetscHSetIAdd(ht, opoint);CHKERRQ(ierr);
  }

  /* Now build the overlap for the patch */
  for (overlapi = 0; overlapi < patch->pardecomp_overlap; ++overlapi) {
    PetscInt index = 0;
    PetscInt *htpoints = NULL;
    PetscInt htsize;
    PetscInt i;

    ierr = PetscHSetIGetSize(ht, &htsize);CHKERRQ(ierr);
    ierr = PetscMalloc1(htsize, &htpoints);CHKERRQ(ierr);
    ierr = PetscHSetIGetElems(ht, &index, htpoints);CHKERRQ(ierr);

    for (i = 0; i < htsize; ++i) {
      PetscInt hpoint = htpoints[i];
      PetscInt si;

      ierr = DMPlexGetTransitiveClosure(dm, hpoint, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      for (si = 0; si < starSize*2; si += 2) {
        const PetscInt starp = star[si];
        PetscInt       closureSize;
        PetscInt      *closure = NULL, ci;

        /* now loop over all entities in the closure of starp */
        ierr = DMPlexGetTransitiveClosure(dm, starp, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        for (ci = 0; ci < closureSize*2; ci += 2) {
          const PetscInt closstarp = closure[ci];
          ierr = PetscHSetIAdd(ht, closstarp);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, starp, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, hpoint, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    }
    ierr = PetscFree(htpoints);CHKERRQ(ierr);
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
    patch->sectionSF = sf[0];
    ierr = PetscObjectReference((PetscObject) patch->sectionSF);CHKERRQ(ierr);
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
      ierr = PetscSFGetRootRanks(sf[i], &nranks, &ranks, NULL, NULL, NULL);CHKERRQ(ierr);
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
    ierr = MPI_Type_contiguous(n, MPIU_INT, &contig);CHKERRMPI(ierr);
    ierr = MPI_Type_commit(&contig);CHKERRMPI(ierr);

    ierr = PetscSFBcastBegin(rankSF, contig, offsets, remoteOffsets,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(rankSF, contig, offsets, remoteOffsets,MPI_REPLACE);CHKERRQ(ierr);
    ierr = MPI_Type_free(&contig);CHKERRMPI(ierr);
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
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)pc), &patch->sectionSF);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(patch->sectionSF, allRoots, allLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
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
  if (type != PC_COMPOSITE_ADDITIVE && type != PC_COMPOSITE_MULTIPLICATIVE) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Only supports additive or multiplicative as the local type");
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
  case PC_PATCH_PARDECOMP:
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
  DM             dm, plex;
  PetscSF       *sfs;
  PetscInt       cStart, cEnd, i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  dm = plex;
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
    ierr = DMGetLocalSection(dms[i], &patch->dofSection[i]);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) patch->dofSection[i]);CHKERRQ(ierr);
    ierr = DMGetSectionSF(dms[i], &sfs[i]);CHKERRQ(ierr);
    patch->bs[i]              = bs[i];
    patch->nodesPerCell[i]    = nodesPerCell[i];
    patch->totalDofsPerCell  += nodesPerCell[i]*bs[i];
    ierr = PetscMalloc1((cEnd-cStart)*nodesPerCell[i], &patch->cellNodeMap[i]);CHKERRQ(ierr);
    for (j = 0; j < (cEnd-cStart)*nodesPerCell[i]; ++j) patch->cellNodeMap[i][j] = cellNodeMap[i][j];
    patch->subspaceOffsets[i] = subspaceOffsets[i];
  }
  ierr = PCPatchCreateDefaultSF_Private(pc, nsubspaces, sfs, patch->bs);CHKERRQ(ierr);
  ierr = PetscFree(sfs);CHKERRQ(ierr);

  patch->subspaceOffsets[nsubspaces] = subspaceOffsets[nsubspaces];
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGhostBcs, ghostBcNodes, PETSC_COPY_VALUES, &patch->ghostBcNodes);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalBcs, globalBcNodes, PETSC_COPY_VALUES, &patch->globalBcNodes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
  ierr = DMGetLocalSection(dm, &patch->dofSection[0]);CHKERRQ(ierr);
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
  ierr = DMGetSectionSF(dm, &patch->sectionSF);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) patch->sectionSF);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGhostBcs, ghostBcNodes, PETSC_COPY_VALUES, &patch->ghostBcNodes);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalBcs, globalBcNodes, PETSC_COPY_VALUES, &patch->globalBcNodes);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  dm = plex;
  ierr = DMPlexGetHeightStratum(dm, 1, &fBegin, &fEnd);CHKERRQ(ierr);
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
        /* Facet integrals couple dofs across facets, so in that case for each of
         * the facets we need to add all dofs on the other side of the facet to
         * the seen dofs. */
        if (patch->usercomputeopintfacet) {
          if (fBegin <= seenpoint && seenpoint < fEnd) {
            ierr = DMPlexGetTransitiveClosure(dm, seenpoint, PETSC_FALSE, &fStarSize, &fStar);CHKERRQ(ierr);
            for (fsi = 0; fsi < fStarSize*2; fsi += 2) {
              ierr = DMPlexGetTransitiveClosure(dm, fStar[fsi], PETSC_TRUE, &fClosureSize, &fClosure);CHKERRQ(ierr);
              for (fci = 0; fci < fClosureSize*2; fci += 2) {
                ierr = PetscHSetIAdd(cht, fClosure[fci]);CHKERRQ(ierr);
              }
              ierr = DMPlexRestoreTransitiveClosure(dm, fStar[fsi], PETSC_TRUE, NULL, &fClosure);CHKERRQ(ierr);
            }
            ierr = DMPlexRestoreTransitiveClosure(dm, seenpoint, PETSC_FALSE, NULL, &fStar);CHKERRQ(ierr);
          }
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, ownedpoint, PETSC_TRUE, NULL, &closure);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, NULL, &star);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
      if (off) {
        *off = 0;
        for (g = 0; g < patch->nsubspaces; ++g) {
          ierr = PetscSectionGetOffset(dofSection[g], p, &fdof);CHKERRQ(ierr);
          *off += fdof;
        }
      }
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
static PetscErrorCode PCPatchGetPointDofs(PC pc, PetscHSetI pts, PetscHSetI dofs, PetscInt base, PetscHSetI* subspaces_to_exclude)
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

    if (subspaces_to_exclude != NULL) {
      PetscBool should_exclude_k = PETSC_FALSE;
      PetscHSetIHas(*subspaces_to_exclude, k, &should_exclude_k);
      if (should_exclude_k) {
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
  PetscSection    cellCounts,  pointCounts, intFacetCounts, extFacetCounts;
  PetscInt       *cellsArray, *pointsArray, *intFacetsArray, *extFacetsArray, *intFacetsToPatchCell;
  PetscInt        numCells, numPoints, numIntFacets, numExtFacets;
  const PetscInt *leaves;
  PetscInt        nleaves, pStart, pEnd, cStart, cEnd, vStart, vEnd, fStart, fEnd, v;
  PetscBool       isFiredrake;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Used to keep track of the cells in the patch. */
  ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&cht);CHKERRQ(ierr);

  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  if (!dm) SETERRQ(PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONGSTATE, "DM not yet set on patch PC\n");
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  dm = plex;
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  if (patch->user_patches) {
    ierr = patch->userpatchconstructionop(pc, &patch->npatch, &patch->userIS, &patch->iterationSet, patch->userpatchconstructctx);CHKERRQ(ierr);
    vStart = 0; vEnd = patch->npatch;
  } else if (patch->ctype == PC_PATCH_PARDECOMP) {
    vStart = 0; vEnd = 1;
  } else if (patch->codim < 0) {
    if (patch->dim < 0) {ierr = DMPlexGetDepthStratum(dm,  0,            &vStart, &vEnd);CHKERRQ(ierr);}
    else                {ierr = DMPlexGetDepthStratum(dm,  patch->dim,   &vStart, &vEnd);CHKERRQ(ierr);}
  } else                {ierr = DMPlexGetHeightStratum(dm, patch->codim, &vStart, &vEnd);CHKERRQ(ierr);}
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
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->extFacetCounts);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->extFacetCounts, "Patch Exterior Facet Layout");CHKERRQ(ierr);
  extFacetCounts = patch->extFacetCounts;
  ierr = PetscSectionSetChart(extFacetCounts, vStart, vEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->intFacetCounts);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->intFacetCounts, "Patch Interior Facet Layout");CHKERRQ(ierr);
  intFacetCounts = patch->intFacetCounts;
  ierr = PetscSectionSetChart(intFacetCounts, vStart, vEnd);CHKERRQ(ierr);
  /* Count cells and points in the patch surrounding each entity */
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIter hi;
    PetscInt       chtSize, loc = -1;
    PetscBool      flg;

    if (!patch->user_patches && patch->ctype != PC_PATCH_PARDECOMP) {
      if (ghost) {ierr = DMLabelHasPoint(ghost, v, &flg);CHKERRQ(ierr);}
      else       {ierr = PetscFindInt(v, nleaves, leaves, &loc);CHKERRQ(ierr); flg = loc >=0 ? PETSC_TRUE : PETSC_FALSE;}
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
      if (fStart <= point && point < fEnd) {
        const PetscInt *support;
        PetscInt supportSize, p;
        PetscBool interior = PETSC_TRUE;
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, point, &supportSize);CHKERRQ(ierr);
        if (supportSize == 1) {
          interior = PETSC_FALSE;
        } else {
          for (p = 0; p < supportSize; p++) {
            PetscBool found;
            /* FIXME: can I do this while iterating over cht? */
            PetscHSetIHas(cht, support[p], &found);
            if (!found) {
              interior = PETSC_FALSE;
              break;
            }
          }
        }
        if (interior) {
          ierr = PetscSectionAddDof(intFacetCounts, v, 1);CHKERRQ(ierr);
        } else {
          ierr = PetscSectionAddDof(extFacetCounts, v, 1);CHKERRQ(ierr);
        }
      }
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

  ierr = PetscSectionSetUp(intFacetCounts);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(extFacetCounts);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(intFacetCounts, &numIntFacets);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(extFacetCounts, &numExtFacets);CHKERRQ(ierr);
  ierr = PetscMalloc1(numIntFacets, &intFacetsArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(numIntFacets*2, &intFacetsToPatchCell);CHKERRQ(ierr);
  ierr = PetscMalloc1(numExtFacets, &extFacetsArray);CHKERRQ(ierr);

  /* Now that we know how much space we need, run through again and actually remember the cells. */
  for (v = vStart; v < vEnd; v++) {
    PetscHashIter hi;
    PetscInt       dof, off, cdof, coff, efdof, efoff, ifdof, ifoff, pdof, n = 0, cn = 0, ifn = 0, efn = 0;

    ierr = PetscSectionGetDof(pointCounts, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(pointCounts, v, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(cellCounts, v, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cellCounts, v, &coff);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(intFacetCounts, v, &ifdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(intFacetCounts, v, &ifoff);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(extFacetCounts, v, &efdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(extFacetCounts, v, &efoff);CHKERRQ(ierr);
    if (dof <= 0) continue;
    ierr = patch->patchconstructop((void *) patch, dm, v, ht);CHKERRQ(ierr);
    ierr = PCPatchCompleteCellPatch(pc, ht, cht);CHKERRQ(ierr);
    PetscHashIterBegin(cht, hi);
    while (!PetscHashIterAtEnd(cht, hi)) {
      PetscInt point;

      PetscHashIterGetKey(cht, hi, point);
      if (fStart <= point && point < fEnd) {
        const PetscInt *support;
        PetscInt supportSize, p;
        PetscBool interior = PETSC_TRUE;
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, point, &supportSize);CHKERRQ(ierr);
        if (supportSize == 1) {
          interior = PETSC_FALSE;
        } else {
          for (p = 0; p < supportSize; p++) {
            PetscBool found;
            /* FIXME: can I do this while iterating over cht? */
            PetscHSetIHas(cht, support[p], &found);
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
      ierr = PCPatchGetGlobalDofs(pc, patch->dofSection, -1, patch->combined, point, &pdof, NULL);CHKERRQ(ierr);
      if (pdof)                            {pointsArray[off + n++] = point;}
      if (point >= cStart && point < cEnd) {cellsArray[coff + cn++] = point;}
      PetscHashIterNext(cht, hi);
    }
    if (ifn != ifdof) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of interior facets in patch %D is %D, but should be %D", v, ifn, ifdof);
    if (efn != efdof) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of exterior facets in patch %D is %D, but should be %D", v, efn, efdof);
    if (cn != cdof) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of cells in patch %D is %D, but should be %D", v, cn, cdof);
    if (n  != dof)  SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of points in patch %D is %D, but should be %D", v, n, dof);

    for (ifn = 0; ifn < ifdof; ifn++) {
      PetscInt cell0 = intFacetsToPatchCell[2*(ifoff + ifn)];
      PetscInt cell1 = intFacetsToPatchCell[2*(ifoff + ifn) + 1];
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
      if (!(found0 && found1)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Didn't manage to find local point numbers for facet support");
    }
  }
  ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);
  ierr = PetscHSetIDestroy(&cht);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_SELF, numCells,  cellsArray,  PETSC_OWN_POINTER, &patch->cells);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->cells,  "Patch Cells");CHKERRQ(ierr);
  if (patch->viewCells) {
    ierr = ObjectView((PetscObject) patch->cellCounts, patch->viewerCells, patch->formatCells);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) patch->cells,      patch->viewerCells, patch->formatCells);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numIntFacets,  intFacetsArray,  PETSC_OWN_POINTER, &patch->intFacets);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->intFacets,  "Patch Interior Facets");CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, 2*numIntFacets, intFacetsToPatchCell, PETSC_OWN_POINTER, &patch->intFacetsToPatchCell);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->intFacetsToPatchCell,  "Patch Interior Facets local support");CHKERRQ(ierr);
  if (patch->viewIntFacets) {
    ierr = ObjectView((PetscObject) patch->intFacetCounts,       patch->viewerIntFacets, patch->formatIntFacets);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) patch->intFacets,            patch->viewerIntFacets, patch->formatIntFacets);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) patch->intFacetsToPatchCell, patch->viewerIntFacets, patch->formatIntFacets);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numExtFacets,  extFacetsArray,  PETSC_OWN_POINTER, &patch->extFacets);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->extFacets,  "Patch Exterior Facets");CHKERRQ(ierr);
  if (patch->viewExtFacets) {
    ierr = ObjectView((PetscObject) patch->extFacetCounts, patch->viewerExtFacets, patch->formatExtFacets);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) patch->extFacets,      patch->viewerExtFacets, patch->formatExtFacets);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numPoints, pointsArray, PETSC_OWN_POINTER, &patch->points);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->points, "Patch Points");CHKERRQ(ierr);
  if (patch->viewPoints) {
    ierr = ObjectView((PetscObject) patch->pointCounts, patch->viewerPoints, patch->formatPoints);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) patch->points,      patch->viewerPoints, patch->formatPoints);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  dm = plex;
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

  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    ierr = PetscMalloc1(numPoints*Nf, &offsArrayWithArtificial);CHKERRQ(ierr);
    ierr = PetscMalloc1(numDofs, &asmArrayWithArtificial);CHKERRQ(ierr);
    ierr = PetscMalloc1(numDofs, &dofsArrayWithArtificial);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCountsWithArtificial);CHKERRQ(ierr);
    gtolCountsWithArtificial = patch->gtolCountsWithArtificial;
    ierr = PetscSectionSetChart(gtolCountsWithArtificial, vStart, vEnd);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) patch->gtolCountsWithArtificial, "Patch Global Index Section Including Artificial BCs");CHKERRQ(ierr);
  }

  isNonlinear = patch->isNonlinear;
  if (isNonlinear) {
    ierr = PetscMalloc1(numPoints*Nf, &offsArrayWithAll);CHKERRQ(ierr);
    ierr = PetscMalloc1(numDofs, &asmArrayWithAll);CHKERRQ(ierr);
    ierr = PetscMalloc1(numDofs, &dofsArrayWithAll);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->gtolCountsWithAll);CHKERRQ(ierr);
    gtolCountsWithAll = patch->gtolCountsWithAll;
    ierr = PetscSectionSetChart(gtolCountsWithAll, vStart, vEnd);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) patch->gtolCountsWithAll, "Patch Global Index Section Including All BCs");CHKERRQ(ierr);
  }

  /* Outside the patch loop, get the dofs that are globally-enforced Dirichlet
   conditions */
  ierr = PetscHSetICreate(&globalBcs);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->ghostBcNodes, &bcNodes);CHKERRQ(ierr);
  ierr = ISGetSize(patch->ghostBcNodes, &numBcs);CHKERRQ(ierr);
  for (i = 0; i < numBcs; ++i) {
    ierr = PetscHSetIAdd(globalBcs, bcNodes[i]);CHKERRQ(ierr); /* these are already in concatenated numbering */
  }
  ierr = ISRestoreIndices(patch->ghostBcNodes, &bcNodes);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->ghostBcNodes);CHKERRQ(ierr); /* memory optimisation */

  /* Hash tables for artificial BC construction */
  ierr = PetscHSetICreate(&ownedpts);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&seenpts);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&owneddofs);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&seendofs);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&artificialbcs);CHKERRQ(ierr);

  ierr = ISGetIndices(cells, &cellsArray);CHKERRQ(ierr);
  ierr = ISGetIndices(points, &pointsArray);CHKERRQ(ierr);
  ierr = PetscHMapICreate(&ht);CHKERRQ(ierr);
  ierr = PetscHMapICreate(&htWithArtificial);CHKERRQ(ierr);
  ierr = PetscHMapICreate(&htWithAll);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt localIndex = 0;
    PetscInt localIndexWithArtificial = 0;
    PetscInt localIndexWithAll = 0;
    PetscInt dof, off, i, j, k, l;

    ierr = PetscHMapIClear(ht);CHKERRQ(ierr);
    ierr = PetscHMapIClear(htWithArtificial);CHKERRQ(ierr);
    ierr = PetscHMapIClear(htWithAll);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(cellCounts, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(cellCounts, v, &off);CHKERRQ(ierr);
    if (dof <= 0) continue;

    /* Calculate the global numbers of the artificial BC dofs here first */
    ierr = patch->patchconstructop((void*)patch, dm, v, ownedpts);CHKERRQ(ierr);
    ierr = PCPatchCompleteCellPatch(pc, ownedpts, seenpts);CHKERRQ(ierr);
    ierr = PCPatchGetPointDofs(pc, ownedpts, owneddofs, v, &patch->subspaces_to_exclude);CHKERRQ(ierr);
    ierr = PCPatchGetPointDofs(pc, seenpts, seendofs, v, NULL);CHKERRQ(ierr);
    ierr = PCPatchComputeSetDifference_Private(owneddofs, seendofs, artificialbcs);CHKERRQ(ierr);
    if (patch->viewPatches) {
      PetscHSetI globalbcdofs;
      PetscHashIter hi;
      MPI_Comm comm = PetscObjectComm((PetscObject)pc);

      ierr = PetscHSetICreate(&globalbcdofs);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: owned dofs:\n", v);CHKERRQ(ierr);
      PetscHashIterBegin(owneddofs, hi);
      while (!PetscHashIterAtEnd(owneddofs, hi)) {
        PetscInt globalDof;

        PetscHashIterGetKey(owneddofs, hi, globalDof);
        PetscHashIterNext(owneddofs, hi);
        ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof);CHKERRQ(ierr);
      }
      ierr = PetscSynchronizedPrintf(comm, "\n");CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: seen dofs:\n", v);CHKERRQ(ierr);
      PetscHashIterBegin(seendofs, hi);
      while (!PetscHashIterAtEnd(seendofs, hi)) {
        PetscInt globalDof;
        PetscBool flg;

        PetscHashIterGetKey(seendofs, hi, globalDof);
        PetscHashIterNext(seendofs, hi);
        ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof);CHKERRQ(ierr);

        ierr = PetscHSetIHas(globalBcs, globalDof, &flg);CHKERRQ(ierr);
        if (flg) {ierr = PetscHSetIAdd(globalbcdofs, globalDof);CHKERRQ(ierr);}
      }
      ierr = PetscSynchronizedPrintf(comm, "\n");CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "Patch %d: global BCs:\n", v);CHKERRQ(ierr);
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
          ierr = PetscSynchronizedPrintf(comm, "%d ", globalDof);CHKERRQ(ierr);
        }
      }
      ierr = PetscSynchronizedPrintf(comm, "\n\n");CHKERRQ(ierr);
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
              dofsArray[globalIndex] = -1; /* don't use this in assembly in this patch */
            } else {
              ierr = PetscHMapIGet(ht, globalDof + l, &localDof);CHKERRQ(ierr);
              if (localDof == -1) {
                localDof = localIndex++;
                ierr = PetscHMapISet(ht, globalDof + l, localDof);CHKERRQ(ierr);
              }
              if (globalIndex >= numDofs) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
              /* And store. */
              dofsArray[globalIndex] = localDof;
            }

            if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
              if (isGlobalBcDof) {
                dofsArrayWithArtificial[globalIndex] = -1; /* don't use this in assembly in this patch */
              } else {
                ierr = PetscHMapIGet(htWithArtificial, globalDof + l, &localDof);CHKERRQ(ierr);
                if (localDof == -1) {
                  localDof = localIndexWithArtificial++;
                  ierr = PetscHMapISet(htWithArtificial, globalDof + l, localDof);CHKERRQ(ierr);
                }
                if (globalIndex >= numDofs) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
                /* And store.*/
                dofsArrayWithArtificial[globalIndex] = localDof;
              }
            }

            if (isNonlinear) {
              /* Build the dofmap for the function space with _all_ dofs,
                 including those in any kind of boundary condition */
              ierr = PetscHMapIGet(htWithAll, globalDof + l, &localDof);CHKERRQ(ierr);
              if (localDof == -1) {
                localDof = localIndexWithAll++;
                ierr = PetscHMapISet(htWithAll, globalDof + l, localDof);CHKERRQ(ierr);
              }
              if (globalIndex >= numDofs) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Found more dofs %D than expected %D", globalIndex+1, numDofs);
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
     ierr = PetscHMapIGetSize(htWithArtificial, &dof);CHKERRQ(ierr);
     ierr = PetscSectionSetDof(gtolCountsWithArtificial, v, dof);CHKERRQ(ierr);
   }
   if (isNonlinear) {
     ierr = PetscHMapIGetSize(htWithAll, &dof);CHKERRQ(ierr);
     ierr = PetscSectionSetDof(gtolCountsWithAll, v, dof);CHKERRQ(ierr);
   }
    ierr = PetscHMapIGetSize(ht, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(gtolCounts, v, dof);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  if (globalIndex != numDofs) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Expected number of dofs (%d) doesn't match found number (%d)", numDofs, globalIndex);
  ierr = PetscSectionSetUp(gtolCounts);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(gtolCounts, &numGlobalDofs);CHKERRQ(ierr);
  ierr = PetscMalloc1(numGlobalDofs, &globalDofsArray);CHKERRQ(ierr);

  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    ierr = PetscSectionSetUp(gtolCountsWithArtificial);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(gtolCountsWithArtificial, &numGlobalDofsWithArtificial);CHKERRQ(ierr);
    ierr = PetscMalloc1(numGlobalDofsWithArtificial, &globalDofsArrayWithArtificial);CHKERRQ(ierr);
  }
  if (isNonlinear) {
    ierr = PetscSectionSetUp(gtolCountsWithAll);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(gtolCountsWithAll, &numGlobalDofsWithAll);CHKERRQ(ierr);
    ierr = PetscMalloc1(numGlobalDofsWithAll, &globalDofsArrayWithAll);CHKERRQ(ierr);
  }
  /* Now populate the global to local map.  This could be merged into the above loop if we were willing to deal with reallocs. */
  for (v = vStart; v < vEnd; ++v) {
    PetscHashIter hi;
    PetscInt      dof, off, Np, ooff, i, j, k, l;

    ierr = PetscHMapIClear(ht);CHKERRQ(ierr);
    ierr = PetscHMapIClear(htWithArtificial);CHKERRQ(ierr);
    ierr = PetscHMapIClear(htWithAll);CHKERRQ(ierr);
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
            const PetscInt localDof  = dofsArray[key];
            if (localDof >= 0) {ierr = PetscHMapISet(ht, globalDof, localDof);CHKERRQ(ierr);}
            if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
              const PetscInt localDofWithArtificial = dofsArrayWithArtificial[key];
              if (localDofWithArtificial >= 0) {
                ierr = PetscHMapISet(htWithArtificial, globalDof, localDofWithArtificial);CHKERRQ(ierr);
              }
            }
            if (isNonlinear) {
              const PetscInt localDofWithAll = dofsArrayWithAll[key];
              if (localDofWithAll >= 0) {
                ierr = PetscHMapISet(htWithAll, globalDof, localDofWithAll);CHKERRQ(ierr);
              }
            }
            key++;
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

      if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
        ierr = PetscSectionGetOffset(gtolCountsWithArtificial, v, &goff);CHKERRQ(ierr);
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
        ierr = PetscSectionGetOffset(gtolCountsWithAll, v, &goff);CHKERRQ(ierr);
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

        ierr = PCPatchGetGlobalDofs(pc, patch->dofSection, k, patch->combined, point, NULL, &globalDof);CHKERRQ(ierr);
        ierr = PetscHMapIGet(ht, globalDof, &localDof);CHKERRQ(ierr);
        offsArray[(ooff + p)*Nf + k] = localDof;
        if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
          ierr = PetscHMapIGet(htWithArtificial, globalDof, &localDof);CHKERRQ(ierr);
          offsArrayWithArtificial[(ooff + p)*Nf + k] = localDof;
        }
        if (isNonlinear) {
          ierr = PetscHMapIGet(htWithAll, globalDof, &localDof);CHKERRQ(ierr);
          offsArrayWithAll[(ooff + p)*Nf + k] = localDof;
        }
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
              asmArray[asmKey] = localDof;
              if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
                ierr = PetscHMapIGet(htWithArtificial, globalDof, &localDof);CHKERRQ(ierr);
                asmArrayWithArtificial[asmKey] = localDof;
              }
              if (isNonlinear) {
                ierr = PetscHMapIGet(htWithAll, globalDof, &localDof);CHKERRQ(ierr);
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
    ierr = PetscArraycpy(asmArray, dofsArray, numDofs);CHKERRQ(ierr);
    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
      ierr = PetscArraycpy(asmArrayWithArtificial, dofsArrayWithArtificial, numDofs);CHKERRQ(ierr);
    }
    if (isNonlinear) {
      ierr = PetscArraycpy(asmArrayWithAll, dofsArrayWithAll, numDofs);CHKERRQ(ierr);
    }
  }

  ierr = PetscHMapIDestroy(&ht);CHKERRQ(ierr);
  ierr = PetscHMapIDestroy(&htWithArtificial);CHKERRQ(ierr);
  ierr = PetscHMapIDestroy(&htWithAll);CHKERRQ(ierr);
  ierr = ISRestoreIndices(cells, &cellsArray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(points, &pointsArray);CHKERRQ(ierr);
  ierr = PetscFree(dofsArray);CHKERRQ(ierr);
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    ierr = PetscFree(dofsArrayWithArtificial);CHKERRQ(ierr);
  }
  if (isNonlinear) {
    ierr = PetscFree(dofsArrayWithAll);CHKERRQ(ierr);
  }
  /* Create placeholder section for map from points to patch dofs */
  ierr = PetscSectionCreate(PETSC_COMM_SELF, &patch->patchSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(patch->patchSection, patch->nsubspaces);CHKERRQ(ierr);
  if (patch->combined) {
    PetscInt numFields;
    ierr = PetscSectionGetNumFields(patch->dofSection[0], &numFields);CHKERRQ(ierr);
    if (numFields != patch->nsubspaces) SETERRQ2(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Mismatch between number of section fields %D and number of subspaces %D", numFields, patch->nsubspaces);
    ierr = PetscSectionGetChart(patch->dofSection[0], &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(patch->patchSection, pStart, pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof, fdof, f;

      ierr = PetscSectionGetDof(patch->dofSection[0], p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(patch->patchSection, p, dof);CHKERRQ(ierr);
      for (f = 0; f < patch->nsubspaces; ++f) {
        ierr = PetscSectionGetFieldDof(patch->dofSection[0], p, f, &fdof);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(patch->patchSection, p, f, fdof);CHKERRQ(ierr);
      }
    }
  } else {
    PetscInt pStartf, pEndf, f;
    pStart = PETSC_MAX_INT;
    pEnd = PETSC_MIN_INT;
    for (f = 0; f < patch->nsubspaces; ++f) {
      ierr = PetscSectionGetChart(patch->dofSection[f], &pStartf, &pEndf);CHKERRQ(ierr);
      pStart = PetscMin(pStart, pStartf);
      pEnd = PetscMax(pEnd, pEndf);
    }
    ierr = PetscSectionSetChart(patch->patchSection, pStart, pEnd);CHKERRQ(ierr);
    for (f = 0; f < patch->nsubspaces; ++f) {
      ierr = PetscSectionGetChart(patch->dofSection[f], &pStartf, &pEndf);CHKERRQ(ierr);
      for (p = pStartf; p < pEndf; ++p) {
        PetscInt fdof;
        ierr = PetscSectionGetDof(patch->dofSection[f], p, &fdof);CHKERRQ(ierr);
        ierr = PetscSectionAddDof(patch->patchSection, p, fdof);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(patch->patchSection, p, f, fdof);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(patch->patchSection);CHKERRQ(ierr);
  ierr = PetscSectionSetUseFieldOffsets(patch->patchSection, PETSC_TRUE);CHKERRQ(ierr);
  /* Replace cell indices with firedrake-numbered ones. */
  ierr = ISGeneralSetIndices(cells, numCells, (const PetscInt *) newCellsArray, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofs, globalDofsArray, PETSC_OWN_POINTER, &patch->gtol);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) patch->gtol, "Global Indices");CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_g2l_view", patch->classname);CHKERRQ(ierr);
  ierr = PetscSectionViewFromOptions(patch->gtolCounts, (PetscObject) pc, option);CHKERRQ(ierr);
  ierr = ISViewFromOptions(patch->gtol, (PetscObject) pc, option);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArray, PETSC_OWN_POINTER, &patch->dofs);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArray, PETSC_OWN_POINTER, &patch->offs);CHKERRQ(ierr);
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofsWithArtificial, globalDofsArrayWithArtificial, PETSC_OWN_POINTER, &patch->gtolWithArtificial);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArrayWithArtificial, PETSC_OWN_POINTER, &patch->dofsWithArtificial);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArrayWithArtificial, PETSC_OWN_POINTER, &patch->offsWithArtificial);CHKERRQ(ierr);
  }
  if (isNonlinear) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numGlobalDofsWithAll, globalDofsArrayWithAll, PETSC_OWN_POINTER, &patch->gtolWithAll);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numDofs, asmArrayWithAll, PETSC_OWN_POINTER, &patch->dofsWithAll);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, numPoints*Nf, offsArrayWithAll, PETSC_OWN_POINTER, &patch->offsWithAll);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchCreateMatrix_Private(PC pc, PetscInt point, Mat *mat, PetscBool withArtificial)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscBool      flg;
  PetscInt       csize, rsize;
  const char    *prefix = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (withArtificial) {
    /* would be nice if we could create a rectangular matrix of size numDofsWithArtificial x numDofs here */
    PetscInt pStart;
    ierr = PetscSectionGetChart(patch->gtolCountsWithArtificial, &pStart, NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(patch->gtolCountsWithArtificial, point + pStart, &rsize);CHKERRQ(ierr);
    csize = rsize;
  } else {
    PetscInt pStart;
    ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, NULL);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(patch->gtolCounts, point + pStart, &rsize);CHKERRQ(ierr);
    csize = rsize;
  }

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

    if (withArtificial) {
      ierr = ISGetIndices(patch->dofsWithArtificial, &dofsArray);CHKERRQ(ierr);
    } else {
      ierr = ISGetIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
    }
    ierr = PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd);CHKERRQ(ierr);
    point += pStart;
    if (point >= pEnd) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Operator point %D not in [%D, %D)\n", point, pStart, pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(patch->cellCounts, point, &ncell);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(patch->cellCounts, point, &offset);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(PC_Patch_Prealloc, pc, 0, 0, 0);CHKERRQ(ierr);
    /* A PetscBT uses N^2 bits to store the sparsity pattern on a
     * patch. This is probably OK if the patches are not too big,
     * but uses too much memory. We therefore switch based on rsize. */
    if (rsize < 3000) { /* FIXME: I picked this switch value out of my hat */
      PetscScalar *zeroes;
      PetscInt rows;

      ierr = PetscCalloc1(rsize, &dnnz);CHKERRQ(ierr);
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

      if (patch->usercomputeopintfacet) {
        const PetscInt *intFacetsArray = NULL;
        PetscInt        i, numIntFacets, intFacetOffset;
        const PetscInt *facetCells = NULL;

        ierr = PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset);CHKERRQ(ierr);
        ierr = ISGetIndices(patch->intFacetsToPatchCell, &facetCells);CHKERRQ(ierr);
        ierr = ISGetIndices(patch->intFacets, &intFacetsArray);CHKERRQ(ierr);
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
      ierr = PetscBTDestroy(&bt);CHKERRQ(ierr);
      ierr = MatXAIJSetPreallocation(*mat, 1, dnnz, NULL, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscFree(dnnz);CHKERRQ(ierr);

      ierr = PetscCalloc1(patch->totalDofsPerCell*patch->totalDofsPerCell, &zeroes);CHKERRQ(ierr);
      for (c = 0; c < ncell; ++c) {
        const PetscInt *idx = &dofsArray[(offset + c)*patch->totalDofsPerCell];
        ierr = MatSetValues(*mat, patch->totalDofsPerCell, idx, patch->totalDofsPerCell, idx, zeroes, INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatGetLocalSize(*mat, &rows, NULL);CHKERRQ(ierr);
      for (i = 0; i < rows; ++i) {
        ierr = MatSetValues(*mat, 1, &i, 1, &i, zeroes, INSERT_VALUES);CHKERRQ(ierr);
      }

      if (patch->usercomputeopintfacet) {
        const PetscInt *intFacetsArray = NULL;
        PetscInt i, numIntFacets, intFacetOffset;
        const PetscInt *facetCells = NULL;

        ierr = PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset);CHKERRQ(ierr);
        ierr = ISGetIndices(patch->intFacetsToPatchCell, &facetCells);CHKERRQ(ierr);
        ierr = ISGetIndices(patch->intFacets, &intFacetsArray);CHKERRQ(ierr);
        for (i = 0; i < numIntFacets; i++) {
          const PetscInt cell0 = facetCells[2*(intFacetOffset + i) + 0];
          const PetscInt cell1 = facetCells[2*(intFacetOffset + i) + 1];
          const PetscInt *cell0idx = &dofsArray[(offset + cell0)*patch->totalDofsPerCell];
          const PetscInt *cell1idx = &dofsArray[(offset + cell1)*patch->totalDofsPerCell];
          ierr = MatSetValues(*mat, patch->totalDofsPerCell, cell0idx, patch->totalDofsPerCell, cell1idx, zeroes, INSERT_VALUES);CHKERRQ(ierr);
          ierr = MatSetValues(*mat, patch->totalDofsPerCell, cell1idx, patch->totalDofsPerCell, cell0idx, zeroes, INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      ierr = MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      ierr = PetscFree(zeroes);CHKERRQ(ierr);

    } else { /* rsize too big, use MATPREALLOCATOR */
      Mat preallocator;
      PetscScalar* vals;

      ierr = PetscCalloc1(patch->totalDofsPerCell*patch->totalDofsPerCell, &vals);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_SELF, &preallocator);CHKERRQ(ierr);
      ierr = MatSetType(preallocator, MATPREALLOCATOR);CHKERRQ(ierr);
      ierr = MatSetSizes(preallocator, rsize, rsize, rsize, rsize);CHKERRQ(ierr);
      ierr = MatSetUp(preallocator);CHKERRQ(ierr);

      for (c = 0; c < ncell; ++c) {
        const PetscInt *idx = dofsArray + (offset + c)*patch->totalDofsPerCell;
        ierr = MatSetValues(preallocator, patch->totalDofsPerCell, idx, patch->totalDofsPerCell, idx, vals, INSERT_VALUES);CHKERRQ(ierr);
      }

      if (patch->usercomputeopintfacet) {
        const PetscInt *intFacetsArray = NULL;
        PetscInt        i, numIntFacets, intFacetOffset;
        const PetscInt *facetCells = NULL;

        ierr = PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset);CHKERRQ(ierr);
        ierr = ISGetIndices(patch->intFacetsToPatchCell, &facetCells);CHKERRQ(ierr);
        ierr = ISGetIndices(patch->intFacets, &intFacetsArray);CHKERRQ(ierr);
        for (i = 0; i < numIntFacets; i++) {
          const PetscInt cell0 = facetCells[2*(intFacetOffset + i) + 0];
          const PetscInt cell1 = facetCells[2*(intFacetOffset + i) + 1];
          const PetscInt *cell0idx = &dofsArray[(offset + cell0)*patch->totalDofsPerCell];
          const PetscInt *cell1idx = &dofsArray[(offset + cell1)*patch->totalDofsPerCell];
          ierr = MatSetValues(preallocator, patch->totalDofsPerCell, cell0idx, patch->totalDofsPerCell, cell1idx, vals, INSERT_VALUES);CHKERRQ(ierr);
          ierr = MatSetValues(preallocator, patch->totalDofsPerCell, cell1idx, patch->totalDofsPerCell, cell0idx, vals, INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      ierr = PetscFree(vals);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(preallocator, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(preallocator, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatPreallocatorPreallocate(preallocator, PETSC_TRUE, *mat);CHKERRQ(ierr);
      ierr = MatDestroy(&preallocator);CHKERRQ(ierr);
    }
    ierr = PetscLogEventEnd(PC_Patch_Prealloc, pc, 0, 0, 0);CHKERRQ(ierr);
    if (withArtificial) {
      ierr = ISRestoreIndices(patch->dofsWithArtificial, &dofsArray);CHKERRQ(ierr);
    } else {
      ierr = ISRestoreIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
    }
  }
  ierr = MatSetUp(*mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchComputeFunction_DMPlex_Private(PC pc, PetscInt patchNum, Vec x, Vec F, IS cellIS, PetscInt n, const PetscInt *l2p, const PetscInt *l2pWithAll, void *ctx)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  DM              dm, plex;
  PetscSection    s;
  const PetscInt *parray, *oarray;
  PetscInt        Nf = patch->nsubspaces, Np, poff, p, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (patch->precomputeElementTensors) SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Precomputing element tensors not implemented with DMPlex compute operator");
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  dm = plex;
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
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
  ierr = DMPlexComputeResidual_Patch_Internal(dm, patch->patchSection, cellIS, 0.0, x, NULL, F, ctx);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
  ierr = PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0);CHKERRQ(ierr);
  if (!patch->usercomputeop) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PCPatchSetComputeOperator() to set user callback\n");
  ierr = ISGetIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->dofsWithAll, &dofsArrayWithAll);CHKERRQ(ierr);
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
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  PetscStackPush("PCPatch user callback");
  /* Cannot reuse the same IS because the geometry info is being cached in it */
  ierr = ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray + offset, PETSC_USE_POINTER, &patch->cellIS);CHKERRQ(ierr);
  ierr = patch->usercomputef(pc, point, x, F, patch->cellIS, ncell*patch->totalDofsPerCell, dofsArray + offset*patch->totalDofsPerCell,
                                                                                            dofsArrayWithAll + offset*patch->totalDofsPerCell,
                                                                                            patch->usercomputefctx);CHKERRQ(ierr);
  PetscStackPop;
  ierr = ISDestroy(&patch->cellIS);CHKERRQ(ierr);
  ierr = ISRestoreIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(patch->dofsWithAll, &dofsArrayWithAll);CHKERRQ(ierr);
  ierr = ISRestoreIndices(patch->cells, &cellsArray);CHKERRQ(ierr);
  if (patch->viewMatrix) {
    char name[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(name, PETSC_MAX_PATH_LEN-1, "Patch vector for Point %D", point);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) F, name);CHKERRQ(ierr);
    ierr = ObjectView((PetscObject) F, patch->viewerMatrix, patch->formatMatrix);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPatchComputeOperator_DMPlex_Private(PC pc, PetscInt patchNum, Vec x, Mat J, IS cellIS, PetscInt n, const PetscInt *l2p, const PetscInt *l2pWithAll, void *ctx)
{
  PC_PATCH       *patch = (PC_PATCH *) pc->data;
  DM              dm, plex;
  PetscSection    s;
  const PetscInt *parray, *oarray;
  PetscInt        Nf = patch->nsubspaces, Np, poff, p, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  dm = plex;
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
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
  ierr = DMPlexComputeJacobian_Patch_Internal(dm, patch->patchSection, patch->patchSection, cellIS, 0.0, 0.0, x, NULL, J, J, ctx);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0);CHKERRQ(ierr);
  isNonlinear = patch->isNonlinear;
  if (!patch->usercomputeop) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PCPatchSetComputeOperator() to set user callback\n");
  if (withArtificial) {
    ierr = ISGetIndices(patch->dofsWithArtificial, &dofsArray);CHKERRQ(ierr);
  } else {
    ierr = ISGetIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
  }
  if (isNonlinear) {
    ierr = ISGetIndices(patch->dofsWithAll, &dofsArrayWithAll);CHKERRQ(ierr);
  }
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
  ierr = MatZeroEntries(mat);CHKERRQ(ierr);
  if (patch->precomputeElementTensors) {
    PetscInt           i;
    PetscInt           ndof = patch->totalDofsPerCell;
    const PetscScalar *elementTensors;

    ierr = VecGetArrayRead(patch->cellMats, &elementTensors);CHKERRQ(ierr);
    for (i = 0; i < ncell; i++) {
      const PetscInt     cell = cellsArray[i + offset];
      const PetscInt    *idx  = dofsArray + (offset + i)*ndof;
      const PetscScalar *v    = elementTensors + patch->precomputedTensorLocations[cell]*ndof*ndof;
      ierr = MatSetValues(mat, ndof, idx, ndof, idx, v, ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(patch->cellMats, &elementTensors);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    PetscStackPush("PCPatch user callback");
    /* Cannot reuse the same IS because the geometry info is being cached in it */
    ierr = ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray + offset, PETSC_USE_POINTER, &patch->cellIS);CHKERRQ(ierr);
    ierr = patch->usercomputeop(pc, point, x, mat, patch->cellIS, ncell*patch->totalDofsPerCell, dofsArray + offset*patch->totalDofsPerCell, dofsArrayWithAll ? dofsArrayWithAll + offset*patch->totalDofsPerCell : NULL, patch->usercomputeopctx);CHKERRQ(ierr);
  }
  if (patch->usercomputeopintfacet) {
    ierr = PetscSectionGetDof(patch->intFacetCounts, point, &numIntFacets);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(patch->intFacetCounts, point, &intFacetOffset);CHKERRQ(ierr);
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

      ierr = ISGetIndices(patch->intFacetsToPatchCell, &facetCells);CHKERRQ(ierr);
      ierr = ISGetIndices(patch->intFacets, &intFacetsArray);CHKERRQ(ierr);
      ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
      ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
      dm = plex;
      ierr = DMPlexGetHeightStratum(dm, 1, &fStart, NULL);CHKERRQ(ierr);
      /* FIXME: Pull this malloc out. */
      ierr = PetscMalloc1(2 * patch->totalDofsPerCell * numIntFacets, &facetDofs);CHKERRQ(ierr);
      if (dofsArrayWithAll) {
        ierr = PetscMalloc1(2 * patch->totalDofsPerCell * numIntFacets, &facetDofsWithAll);CHKERRQ(ierr);
      }
      if (patch->precomputeElementTensors) {
        PetscInt           nFacetDof = 2*patch->totalDofsPerCell;
        const PetscScalar *elementTensors;

        ierr = VecGetArrayRead(patch->intFacetMats, &elementTensors);CHKERRQ(ierr);

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
          ierr = MatSetValues(mat, nFacetDof, facetDofs, nFacetDof, facetDofs, v, ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = VecRestoreArrayRead(patch->intFacetMats, &elementTensors);CHKERRQ(ierr);
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
        ierr = ISCreateGeneral(PETSC_COMM_SELF, numIntFacets, intFacetsArray + intFacetOffset, PETSC_USE_POINTER, &facetIS);CHKERRQ(ierr);
        ierr = patch->usercomputeopintfacet(pc, point, x, mat, facetIS, 2*numIntFacets*patch->totalDofsPerCell, facetDofs, facetDofsWithAll, patch->usercomputeopintfacetctx);CHKERRQ(ierr);
        ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(patch->intFacetsToPatchCell, &facetCells);CHKERRQ(ierr);
      ierr = ISRestoreIndices(patch->intFacets, &intFacetsArray);CHKERRQ(ierr);
      ierr = PetscFree(facetDofs);CHKERRQ(ierr);
      ierr = PetscFree(facetDofsWithAll);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (!(withArtificial || isNonlinear) && patch->denseinverse) {
    MatFactorInfo info;
    PetscBool     flg;
    ierr = PetscObjectTypeCompare((PetscObject)mat, MATSEQDENSE, &flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Invalid Mat type for dense inverse");
    ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
    ierr = MatLUFactor(mat, NULL, NULL, &info);CHKERRQ(ierr);
    ierr = MatSeqDenseInvertFactors_Private(mat);CHKERRQ(ierr);
  }
  PetscStackPop;
  ierr = ISDestroy(&patch->cellIS);CHKERRQ(ierr);
  if (withArtificial) {
    ierr = ISRestoreIndices(patch->dofsWithArtificial, &dofsArray);CHKERRQ(ierr);
  } else {
    ierr = ISRestoreIndices(patch->dofs, &dofsArray);CHKERRQ(ierr);
  }
  if (isNonlinear) {
    ierr = ISRestoreIndices(patch->dofsWithAll, &dofsArrayWithAll);CHKERRQ(ierr);
  }
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

static PetscErrorCode MatSetValues_PCPatch_Private(Mat mat, PetscInt m, const PetscInt idxm[],
                                                   PetscInt n, const PetscInt idxn[], const PetscScalar *v, InsertMode addv)
{
  Vec            data;
  PetscScalar   *array;
  PetscInt       bs, nz, i, j, cell;
  PetscErrorCode ierr;

  ierr = MatShellGetContext(mat, &data);CHKERRQ(ierr);
  ierr = VecGetBlockSize(data, &bs);CHKERRQ(ierr);
  ierr = VecGetSize(data, &nz);CHKERRQ(ierr);
  ierr = VecGetArray(data, &array);CHKERRQ(ierr);
  if (m != n) SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONG, "Only for square insertion");
  cell = (PetscInt)(idxm[0]/bs); /* use the fact that this is called once per cell */
  for (i = 0; i < m; i++) {
    if (idxm[i] != idxn[i]) SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_WRONG, "Row and column indices must match!");
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
  ierr = VecRestoreArray(data, &array);CHKERRQ(ierr);
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

  ierr = ISGetSize(patch->cells, &ncell);CHKERRQ(ierr);
  if (!ncell) { /* No cells to assemble over -> skip */
    PetscFunctionReturn(0);
  }

  ierr = PetscLogEventBegin(PC_Patch_ComputeOp, pc, 0, 0, 0);CHKERRQ(ierr);

  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  dm = plex;
  if (!patch->allCells) {
    PetscHSetI      cells;
    PetscHashIter   hi;
    PetscInt pStart, pEnd;
    PetscInt *allCells = NULL;
    ierr = PetscHSetICreate(&cells);CHKERRQ(ierr);
    ierr = ISGetIndices(patch->cells, &cellsArray);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(patch->cellCounts, &pStart, &pEnd);CHKERRQ(ierr);
    for (i = pStart; i < pEnd; i++) {
      ierr = PetscSectionGetDof(patch->cellCounts, i, &ncell);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(patch->cellCounts, i, &offset);CHKERRQ(ierr);
      if (ncell <= 0) continue;
      for (j = 0; j < ncell; j++) {
        PetscHSetIAdd(cells, cellsArray[offset + j]);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(patch->cells, &cellsArray);CHKERRQ(ierr);
    ierr = PetscHSetIGetSize(cells, &ncell);CHKERRQ(ierr);
    ierr = PetscMalloc1(ncell, &allCells);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(cEnd-cStart, &patch->precomputedTensorLocations);CHKERRQ(ierr);
    i = 0;
    PetscHashIterBegin(cells, hi);
    while (!PetscHashIterAtEnd(cells, hi)) {
      PetscHashIterGetKey(cells, hi, allCells[i]);
      patch->precomputedTensorLocations[allCells[i]] = i;
      PetscHashIterNext(cells, hi);
      i++;
    }
    ierr = PetscHSetIDestroy(&cells);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, ncell, allCells, PETSC_OWN_POINTER, &patch->allCells);CHKERRQ(ierr);
  }
  ierr = ISGetSize(patch->allCells, &ncell);CHKERRQ(ierr);
  if (!patch->cellMats) {
    ierr = VecCreateSeq(PETSC_COMM_SELF, ncell*ndof*ndof, &patch->cellMats);CHKERRQ(ierr);
    ierr = VecSetBlockSize(patch->cellMats, ndof);CHKERRQ(ierr);
  }
  ierr = VecSet(patch->cellMats, 0);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_SELF, ncell*ndof, ncell*ndof, ncell*ndof, ncell*ndof,
                        (void*)patch->cellMats, &vecMat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(vecMat, MATOP_SET_VALUES, (void(*)(void))&MatSetValues_PCPatch_Private);CHKERRQ(ierr);
  ierr = ISGetSize(patch->allCells, &ncell);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, ndof*ncell, 0, 1, &dofMap);CHKERRQ(ierr);
  ierr = ISGetIndices(dofMap, &dofMapArray);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->allCells, &cellsArray);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, ncell, cellsArray, PETSC_USE_POINTER, &cellIS);CHKERRQ(ierr);
  PetscStackPush("PCPatch user callback");
  /* TODO: Fix for DMPlex compute op, this bypasses a lot of the machinery and just assembles every element tensor. */
  ierr = patch->usercomputeop(pc, -1, NULL, vecMat, cellIS, ndof*ncell, dofMapArray, NULL, patch->usercomputeopctx);CHKERRQ(ierr);
  PetscStackPop;
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = MatDestroy(&vecMat);CHKERRQ(ierr);
  ierr = ISRestoreIndices(patch->allCells, &cellsArray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(dofMap, &dofMapArray);CHKERRQ(ierr);
  ierr = ISDestroy(&dofMap);CHKERRQ(ierr);

  if (patch->usercomputeopintfacet) {
    PetscInt nIntFacets;
    IS       intFacetsIS;
    const PetscInt *intFacetsArray = NULL;
    if (!patch->allIntFacets) {
      PetscHSetI      facets;
      PetscHashIter   hi;
      PetscInt pStart, pEnd, fStart, fEnd;
      PetscInt *allIntFacets = NULL;
      ierr = PetscHSetICreate(&facets);CHKERRQ(ierr);
      ierr = ISGetIndices(patch->intFacets, &intFacetsArray);CHKERRQ(ierr);
      ierr = PetscSectionGetChart(patch->intFacetCounts, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
      for (i = pStart; i < pEnd; i++) {
        ierr = PetscSectionGetDof(patch->intFacetCounts, i, &nIntFacets);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(patch->intFacetCounts, i, &offset);CHKERRQ(ierr);
        if (nIntFacets <= 0) continue;
        for (j = 0; j < nIntFacets; j++) {
          PetscHSetIAdd(facets, intFacetsArray[offset + j]);CHKERRQ(ierr);
        }
      }
      ierr = ISRestoreIndices(patch->intFacets, &intFacetsArray);CHKERRQ(ierr);
      ierr = PetscHSetIGetSize(facets, &nIntFacets);CHKERRQ(ierr);
      ierr = PetscMalloc1(nIntFacets, &allIntFacets);CHKERRQ(ierr);
      ierr = PetscMalloc1(fEnd-fStart, &patch->precomputedIntFacetTensorLocations);CHKERRQ(ierr);
      i = 0;
      PetscHashIterBegin(facets, hi);
      while (!PetscHashIterAtEnd(facets, hi)) {
        PetscHashIterGetKey(facets, hi, allIntFacets[i]);
        patch->precomputedIntFacetTensorLocations[allIntFacets[i] - fStart] = i;
        PetscHashIterNext(facets, hi);
        i++;
      }
      ierr = PetscHSetIDestroy(&facets);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF, nIntFacets, allIntFacets, PETSC_OWN_POINTER, &patch->allIntFacets);CHKERRQ(ierr);
    }
    ierr = ISGetSize(patch->allIntFacets, &nIntFacets);CHKERRQ(ierr);
    if (!patch->intFacetMats) {
      ierr = VecCreateSeq(PETSC_COMM_SELF, nIntFacets*ndof*ndof*4, &patch->intFacetMats);CHKERRQ(ierr);
      ierr = VecSetBlockSize(patch->intFacetMats, ndof*2);CHKERRQ(ierr);
    }
    ierr = VecSet(patch->intFacetMats, 0);CHKERRQ(ierr);

    ierr = MatCreateShell(PETSC_COMM_SELF, nIntFacets*ndof*2, nIntFacets*ndof*2, nIntFacets*ndof*2, nIntFacets*ndof*2,
                          (void*)patch->intFacetMats, &vecMat);CHKERRQ(ierr);
    ierr = MatShellSetOperation(vecMat, MATOP_SET_VALUES, (void(*)(void))&MatSetValues_PCPatch_Private);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF, 2*ndof*nIntFacets, 0, 1, &dofMap);CHKERRQ(ierr);
    ierr = ISGetIndices(dofMap, &dofMapArray);CHKERRQ(ierr);
    ierr = ISGetIndices(patch->allIntFacets, &intFacetsArray);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF, nIntFacets, intFacetsArray, PETSC_USE_POINTER, &intFacetsIS);CHKERRQ(ierr);
    PetscStackPush("PCPatch user callback (interior facets)");
    /* TODO: Fix for DMPlex compute op, this bypasses a lot of the machinery and just assembles every element tensor. */
    ierr = patch->usercomputeopintfacet(pc, -1, NULL, vecMat, intFacetsIS, 2*ndof*nIntFacets, dofMapArray, NULL, patch->usercomputeopintfacetctx);CHKERRQ(ierr);
    PetscStackPop;
    ierr = ISDestroy(&intFacetsIS);CHKERRQ(ierr);
    ierr = MatDestroy(&vecMat);CHKERRQ(ierr);
    ierr = ISRestoreIndices(patch->allIntFacets, &intFacetsArray);CHKERRQ(ierr);
    ierr = ISRestoreIndices(dofMap, &dofMapArray);CHKERRQ(ierr);
    ierr = ISDestroy(&dofMap);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_Patch_ComputeOp, pc, 0, 0, 0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PCPatch_ScatterLocal_Private(PC pc, PetscInt p, Vec x, Vec y, InsertMode mode, ScatterMode scat, PatchScatterType scattertype)
{
  PC_PATCH          *patch     = (PC_PATCH *) pc->data;
  const PetscScalar *xArray    = NULL;
  PetscScalar       *yArray    = NULL;
  const PetscInt    *gtolArray = NULL;
  PetscInt           dof, offset, lidx;
  PetscErrorCode     ierr;

  PetscFunctionBeginHot;
  ierr = VecGetArrayRead(x, &xArray);CHKERRQ(ierr);
  ierr = VecGetArray(y, &yArray);CHKERRQ(ierr);
  if (scattertype == SCATTER_WITHARTIFICIAL) {
    ierr = PetscSectionGetDof(patch->gtolCountsWithArtificial, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(patch->gtolCountsWithArtificial, p, &offset);CHKERRQ(ierr);
    ierr = ISGetIndices(patch->gtolWithArtificial, &gtolArray);CHKERRQ(ierr);
  } else if (scattertype == SCATTER_WITHALL) {
    ierr = PetscSectionGetDof(patch->gtolCountsWithAll, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(patch->gtolCountsWithAll, p, &offset);CHKERRQ(ierr);
    ierr = ISGetIndices(patch->gtolWithAll, &gtolArray);CHKERRQ(ierr);
  } else {
    ierr = PetscSectionGetDof(patch->gtolCounts, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(patch->gtolCounts, p, &offset);CHKERRQ(ierr);
    ierr = ISGetIndices(patch->gtol, &gtolArray);CHKERRQ(ierr);
  }
  if (mode == INSERT_VALUES && scat != SCATTER_FORWARD) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't insert if not scattering forward\n");
  if (mode == ADD_VALUES    && scat != SCATTER_REVERSE) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Can't add if not scattering reverse\n");
  for (lidx = 0; lidx < dof; ++lidx) {
    const PetscInt gidx = gtolArray[offset+lidx];

    if (mode == INSERT_VALUES) yArray[lidx]  = xArray[gidx]; /* Forward */
    else                       yArray[gidx] += xArray[lidx]; /* Reverse */
  }
  if (scattertype == SCATTER_WITHARTIFICIAL) {
    ierr = ISRestoreIndices(patch->gtolWithArtificial, &gtolArray);CHKERRQ(ierr);
  } else if (scattertype == SCATTER_WITHALL) {
    ierr = ISRestoreIndices(patch->gtolWithAll, &gtolArray);CHKERRQ(ierr);
  } else {
    ierr = ISRestoreIndices(patch->gtol, &gtolArray);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(x, &xArray);CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &yArray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_PATCH_Linear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  const char    *prefix;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    if (!patch->save_operators && patch->denseinverse) SETERRQ(PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "Can't have dense inverse without save operators");
    if (!patch->denseinverse) {
      ierr = PetscMalloc1(patch->npatch, &patch->solver);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc, &prefix);CHKERRQ(ierr);
      for (i = 0; i < patch->npatch; ++i) {
        KSP ksp;
        PC  subpc;

        ierr = KSPCreate(PETSC_COMM_SELF, &ksp);CHKERRQ(ierr);
        ierr = KSPSetErrorIfNotConverged(ksp, pc->erroriffailure);CHKERRQ(ierr);
        ierr = KSPSetOptionsPrefix(ksp, prefix);CHKERRQ(ierr);
        ierr = KSPAppendOptionsPrefix(ksp, "sub_");CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject) ksp, (PetscObject) pc, 1);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp, &subpc);CHKERRQ(ierr);
        ierr = PetscObjectIncrementTabLevel((PetscObject) subpc, (PetscObject) pc, 1);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject) pc, (PetscObject) ksp);CHKERRQ(ierr);
        patch->solver[i] = (PetscObject) ksp;
      }
    }
  }
  if (patch->save_operators) {
    if (patch->precomputeElementTensors) {
      ierr = PCPatchPrecomputePatchTensors_Private(pc);CHKERRQ(ierr);
    }
    for (i = 0; i < patch->npatch; ++i) {
      ierr = PCPatchComputeOperator_Internal(pc, NULL, patch->mat[i], i, PETSC_FALSE);CHKERRQ(ierr);
      if (!patch->denseinverse) {
        ierr = KSPSetOperators((KSP) patch->solver[i], patch->mat[i], patch->mat[i]);CHKERRQ(ierr);
      } else if (patch->mat[i] && !patch->densesolve) {
        /* Setup matmult callback */
        ierr = MatGetOperation(patch->mat[i], MATOP_MULT, (void (**)(void))&patch->densesolve);CHKERRQ(ierr);
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

      ierr = MatGetSize(patch->mat[i], &dof, NULL);CHKERRQ(ierr);
      if (dof == 0) {
        patch->matWithArtificial[i] = NULL;
        continue;
      }

      ierr = PCPatchCreateMatrix_Private(pc, i, &matSquare, PETSC_TRUE);CHKERRQ(ierr);
      ierr = PCPatchComputeOperator_Internal(pc, NULL, matSquare, i, PETSC_TRUE);CHKERRQ(ierr);

      ierr = MatGetSize(matSquare, &dof, NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF, dof, 0, 1, &rowis);CHKERRQ(ierr);
      if (pc->setupcalled) {
        ierr = MatCreateSubMatrix(matSquare, rowis, patch->dofMappingWithoutToWithArtificial[i], MAT_REUSE_MATRIX, &patch->matWithArtificial[i]);CHKERRQ(ierr);
      } else {
        ierr = MatCreateSubMatrix(matSquare, rowis, patch->dofMappingWithoutToWithArtificial[i], MAT_INITIAL_MATRIX, &patch->matWithArtificial[i]);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&rowis);CHKERRQ(ierr);
      ierr = MatDestroy(&matSquare);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    PetscInt pStart, pEnd, p;
    PetscInt localSize;

    ierr = PetscLogEventBegin(PC_Patch_CreatePatches, pc, 0, 0, 0);CHKERRQ(ierr);

    isNonlinear = patch->isNonlinear;
    if (!patch->nsubspaces) {
      DM           dm, plex;
      PetscSection s;
      PetscInt     cStart, cEnd, c, Nf, f, numGlobalBcs = 0, *globalBcs, *Nb, totNb = 0, **cellDofs;

      ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
      if (!dm) SETERRQ(PetscObjectComm((PetscObject) pc), PETSC_ERR_ARG_WRONG, "Must set DM for PCPATCH or call PCPatchSetDiscretisationInfo()");
      ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
      dm = plex;
      ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
      ierr = PetscSectionGetNumFields(s, &Nf);CHKERRQ(ierr);
      ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        PetscInt cdof;
        ierr = PetscSectionGetConstraintDof(s, p, &cdof);CHKERRQ(ierr);
        numGlobalBcs += cdof;
      }
      ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
      ierr = PetscMalloc3(Nf, &Nb, Nf, &cellDofs, numGlobalBcs, &globalBcs);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscFE        fe;
        PetscDualSpace sp;
        PetscInt       cdoff = 0;

        ierr = DMGetField(dm, f, NULL, (PetscObject *) &fe);CHKERRQ(ierr);
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
      ierr = PCPatchSetComputeFunction(pc, PCPatchComputeFunction_DMPlex_Private, NULL);CHKERRQ(ierr);
      ierr = PCPatchSetComputeOperator(pc, PCPatchComputeOperator_DMPlex_Private, NULL);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
    }

    localSize = patch->subspaceOffsets[patch->nsubspaces];
    ierr = VecCreateSeq(PETSC_COMM_SELF, localSize, &patch->localRHS);CHKERRQ(ierr);
    ierr = VecSetUp(patch->localRHS);CHKERRQ(ierr);
    ierr = VecDuplicate(patch->localRHS, &patch->localUpdate);CHKERRQ(ierr);
    ierr = PCPatchCreateCellPatches(pc);CHKERRQ(ierr);
    ierr = PCPatchCreateCellPatchDiscretisationInfo(pc);CHKERRQ(ierr);

    /* OK, now build the work vectors */
    ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, &pEnd);CHKERRQ(ierr);

    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
      ierr = PetscMalloc1(patch->npatch, &patch->dofMappingWithoutToWithArtificial);CHKERRQ(ierr);
    }
    if (isNonlinear) {
      ierr = PetscMalloc1(patch->npatch, &patch->dofMappingWithoutToWithAll);CHKERRQ(ierr);
    }
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dof;

      ierr = PetscSectionGetDof(patch->gtolCounts, p, &dof);CHKERRQ(ierr);
      maxDof = PetscMax(maxDof, dof);CHKERRQ(ierr);
      if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
        const PetscInt    *gtolArray, *gtolArrayWithArtificial = NULL;
        PetscInt           numPatchDofs, offset;
        PetscInt           numPatchDofsWithArtificial, offsetWithArtificial;
        PetscInt           dofWithoutArtificialCounter = 0;
        PetscInt          *patchWithoutArtificialToWithArtificialArray;

        ierr = PetscSectionGetDof(patch->gtolCountsWithArtificial, p, &dof);CHKERRQ(ierr);
        maxDofWithArtificial = PetscMax(maxDofWithArtificial, dof);

        /* Now build the mapping that for a dof in a patch WITHOUT dofs that have artificial bcs gives the */
        /* the index in the patch with all dofs */
        ierr = ISGetIndices(patch->gtol, &gtolArray);CHKERRQ(ierr);

        ierr = PetscSectionGetDof(patch->gtolCounts, p, &numPatchDofs);CHKERRQ(ierr);
        if (numPatchDofs == 0) {
          patch->dofMappingWithoutToWithArtificial[p-pStart] = NULL;
          continue;
        }

        ierr = PetscSectionGetOffset(patch->gtolCounts, p, &offset);CHKERRQ(ierr);
        ierr = ISGetIndices(patch->gtolWithArtificial, &gtolArrayWithArtificial);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(patch->gtolCountsWithArtificial, p, &numPatchDofsWithArtificial);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(patch->gtolCountsWithArtificial, p, &offsetWithArtificial);CHKERRQ(ierr);

        ierr = PetscMalloc1(numPatchDofs, &patchWithoutArtificialToWithArtificialArray);CHKERRQ(ierr);
        for (i=0; i<numPatchDofsWithArtificial; i++) {
          if (gtolArrayWithArtificial[i+offsetWithArtificial] == gtolArray[offset+dofWithoutArtificialCounter]) {
            patchWithoutArtificialToWithArtificialArray[dofWithoutArtificialCounter] = i;
            dofWithoutArtificialCounter++;
            if (dofWithoutArtificialCounter == numPatchDofs)
              break;
          }
        }
        ierr = ISCreateGeneral(PETSC_COMM_SELF, numPatchDofs, patchWithoutArtificialToWithArtificialArray, PETSC_OWN_POINTER, &patch->dofMappingWithoutToWithArtificial[p-pStart]);CHKERRQ(ierr);
        ierr = ISRestoreIndices(patch->gtol, &gtolArray);CHKERRQ(ierr);
        ierr = ISRestoreIndices(patch->gtolWithArtificial, &gtolArrayWithArtificial);CHKERRQ(ierr);
      }
      if (isNonlinear) {
        const PetscInt    *gtolArray, *gtolArrayWithAll = NULL;
        PetscInt           numPatchDofs, offset;
        PetscInt           numPatchDofsWithAll, offsetWithAll;
        PetscInt           dofWithoutAllCounter = 0;
        PetscInt          *patchWithoutAllToWithAllArray;

        /* Now build the mapping that for a dof in a patch WITHOUT dofs that have artificial bcs gives the */
        /* the index in the patch with all dofs */
        ierr = ISGetIndices(patch->gtol, &gtolArray);CHKERRQ(ierr);

        ierr = PetscSectionGetDof(patch->gtolCounts, p, &numPatchDofs);CHKERRQ(ierr);
        if (numPatchDofs == 0) {
          patch->dofMappingWithoutToWithAll[p-pStart] = NULL;
          continue;
        }

        ierr = PetscSectionGetOffset(patch->gtolCounts, p, &offset);CHKERRQ(ierr);
        ierr = ISGetIndices(patch->gtolWithAll, &gtolArrayWithAll);CHKERRQ(ierr);
        ierr = PetscSectionGetDof(patch->gtolCountsWithAll, p, &numPatchDofsWithAll);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(patch->gtolCountsWithAll, p, &offsetWithAll);CHKERRQ(ierr);

        ierr = PetscMalloc1(numPatchDofs, &patchWithoutAllToWithAllArray);CHKERRQ(ierr);

        for (i=0; i<numPatchDofsWithAll; i++) {
          if (gtolArrayWithAll[i+offsetWithAll] == gtolArray[offset+dofWithoutAllCounter]) {
            patchWithoutAllToWithAllArray[dofWithoutAllCounter] = i;
            dofWithoutAllCounter++;
            if (dofWithoutAllCounter == numPatchDofs)
              break;
          }
        }
        ierr = ISCreateGeneral(PETSC_COMM_SELF, numPatchDofs, patchWithoutAllToWithAllArray, PETSC_OWN_POINTER, &patch->dofMappingWithoutToWithAll[p-pStart]);CHKERRQ(ierr);
        ierr = ISRestoreIndices(patch->gtol, &gtolArray);CHKERRQ(ierr);
        ierr = ISRestoreIndices(patch->gtolWithAll, &gtolArrayWithAll);CHKERRQ(ierr);
      }
    }
    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
      ierr = VecCreateSeq(PETSC_COMM_SELF, maxDofWithArtificial, &patch->patchRHSWithArtificial);CHKERRQ(ierr);
      ierr = VecSetUp(patch->patchRHSWithArtificial);CHKERRQ(ierr);
    }
    ierr = VecCreateSeq(PETSC_COMM_SELF, maxDof, &patch->patchRHS);CHKERRQ(ierr);
    ierr = VecSetUp(patch->patchRHS);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF, maxDof, &patch->patchUpdate);CHKERRQ(ierr);
    ierr = VecSetUp(patch->patchUpdate);CHKERRQ(ierr);
    if (patch->save_operators) {
      ierr = PetscMalloc1(patch->npatch, &patch->mat);CHKERRQ(ierr);
      for (i = 0; i < patch->npatch; ++i) {
        ierr = PCPatchCreateMatrix_Private(pc, i, &patch->mat[i], PETSC_FALSE);CHKERRQ(ierr);
      }
    }
    ierr = PetscLogEventEnd(PC_Patch_CreatePatches, pc, 0, 0, 0);CHKERRQ(ierr);

    /* If desired, calculate weights for dof multiplicity */
    if (patch->partition_of_unity) {
      PetscScalar *input = NULL;
      PetscScalar *output = NULL;
      Vec global;

      ierr = VecDuplicate(patch->localRHS, &patch->dof_weights);CHKERRQ(ierr);
      if (patch->local_composition_type == PC_COMPOSITE_ADDITIVE) {
        for (i = 0; i < patch->npatch; ++i) {
          PetscInt dof;

          ierr = PetscSectionGetDof(patch->gtolCounts, i+pStart, &dof);CHKERRQ(ierr);
          if (dof <= 0) continue;
          ierr = VecSet(patch->patchRHS, 1.0);CHKERRQ(ierr);
          ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchRHS, patch->dof_weights, ADD_VALUES, SCATTER_REVERSE, SCATTER_INTERIOR);CHKERRQ(ierr);
        }
      } else {
        /* multiplicative is actually only locally multiplicative and globally additive. need the pou where the mesh decomposition overlaps */
        ierr = VecSet(patch->dof_weights, 1.0);CHKERRQ(ierr);
      }

      VecDuplicate(patch->dof_weights, &global);
      VecSet(global, 0.);

      ierr = VecGetArray(patch->dof_weights, &input);CHKERRQ(ierr);
      ierr = VecGetArray(global, &output);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(patch->sectionSF, MPIU_SCALAR, input, output, MPI_SUM);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(patch->sectionSF, MPIU_SCALAR, input, output, MPI_SUM);CHKERRQ(ierr);
      ierr = VecRestoreArray(patch->dof_weights, &input);CHKERRQ(ierr);
      ierr = VecRestoreArray(global, &output);CHKERRQ(ierr);

      ierr = VecReciprocal(global);CHKERRQ(ierr);

      ierr = VecGetArray(patch->dof_weights, &output);CHKERRQ(ierr);
      ierr = VecGetArray(global, &input);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(patch->sectionSF, MPIU_SCALAR, input, output,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(patch->sectionSF, MPIU_SCALAR, input, output,MPI_REPLACE);CHKERRQ(ierr);
      ierr = VecRestoreArray(patch->dof_weights, &output);CHKERRQ(ierr);
      ierr = VecRestoreArray(global, &input);CHKERRQ(ierr);
      ierr = VecDestroy(&global);CHKERRQ(ierr);
    }
    if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE && patch->save_operators) {
      ierr = PetscMalloc1(patch->npatch, &patch->matWithArtificial);CHKERRQ(ierr);
    }
  }
  ierr = (*patch->setupsolver)(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_PATCH_Linear(PC pc, PetscInt i, Vec x, Vec y)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  KSP            ksp;
  Mat            op;
  PetscInt       m, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (patch->denseinverse) {
    ierr = (*patch->densesolve)(patch->mat[i], x, y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ksp = (KSP) patch->solver[i];
  if (!patch->save_operators) {
    Mat mat;

    ierr = PCPatchCreateMatrix_Private(pc, i, &mat, PETSC_FALSE);CHKERRQ(ierr);
    /* Populate operator here. */
    ierr = PCPatchComputeOperator_Internal(pc, NULL, mat, i, PETSC_FALSE);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, mat, mat);CHKERRQ(ierr);
    /* Drop reference so the KSPSetOperators below will blow it away. */
    ierr = MatDestroy(&mat);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0);CHKERRQ(ierr);
  if (!ksp->setfromoptionscalled) {
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  }
  /* Disgusting trick to reuse work vectors */
  ierr = KSPGetOperators(ksp, &op, NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(op, &m, &n);CHKERRQ(ierr);
  x->map->n = m;
  y->map->n = n;
  x->map->N = m;
  y->map->N = n;
  ierr = KSPSolve(ksp, x, y);CHKERRQ(ierr);
  ierr = KSPCheckSolve(ksp, pc, y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0);CHKERRQ(ierr);
  if (!patch->save_operators) {
    PC pc;
    ierr = KSPSetOperators(ksp, NULL, NULL);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    /* Destroy PC context too, otherwise the factored matrix hangs around. */
    ierr = PCReset(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCUpdateMultiplicative_PATCH_Linear(PC pc, PetscInt i, PetscInt pStart)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  Mat multMat;
  PetscInt n, m;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (patch->save_operators) {
    multMat = patch->matWithArtificial[i];
  } else {
    /*Very inefficient, hopefully we can just assemble the rectangular matrix in the first place.*/
    Mat matSquare;
    PetscInt dof;
    IS rowis;
    ierr = PCPatchCreateMatrix_Private(pc, i, &matSquare, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PCPatchComputeOperator_Internal(pc, NULL, matSquare, i, PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatGetSize(matSquare, &dof, NULL);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF, dof, 0, 1, &rowis);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(matSquare, rowis, patch->dofMappingWithoutToWithArtificial[i], MAT_INITIAL_MATRIX, &multMat);CHKERRQ(ierr);
    ierr = MatDestroy(&matSquare);CHKERRQ(ierr);
    ierr = ISDestroy(&rowis);CHKERRQ(ierr);
  }
  /* Disgusting trick to reuse work vectors */
  ierr = MatGetLocalSize(multMat, &m, &n);CHKERRQ(ierr);
  patch->patchUpdate->map->n = n;
  patch->patchRHSWithArtificial->map->n = m;
  patch->patchUpdate->map->N = n;
  patch->patchRHSWithArtificial->map->N = m;
  ierr = MatMult(multMat, patch->patchUpdate, patch->patchRHSWithArtificial);CHKERRQ(ierr);
  ierr = VecScale(patch->patchRHSWithArtificial, -1.0);CHKERRQ(ierr);
  ierr = PCPatch_ScatterLocal_Private(pc, i + pStart, patch->patchRHSWithArtificial, patch->localRHS, ADD_VALUES, SCATTER_REVERSE, SCATTER_WITHARTIFICIAL);CHKERRQ(ierr);
  if (!patch->save_operators) {
    ierr = MatDestroy(&multMat);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_Patch_Apply, pc, 0, 0, 0);CHKERRQ(ierr);
  ierr = PetscOptionsPushGetViewerOff(PETSC_TRUE);CHKERRQ(ierr);
  /* start, end, inc have 2 entries to manage a second backward sweep if we symmetrize */
  end[0]   = patch->npatch;
  start[1] = patch->npatch-1;
  if (patch->user_patches) {
    ierr = ISGetLocalSize(patch->iterationSet, &end[0]);CHKERRQ(ierr);
    start[1] = end[0] - 1;
    ierr = ISGetIndices(patch->iterationSet, &iterationSet);CHKERRQ(ierr);
  }
  /* Scatter from global space into overlapped local spaces */
  ierr = VecGetArrayRead(x, &globalRHS);CHKERRQ(ierr);
  ierr = VecGetArray(patch->localRHS, &localRHS);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(patch->sectionSF, MPIU_SCALAR, globalRHS, localRHS,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(patch->sectionSF, MPIU_SCALAR, globalRHS, localRHS,MPI_REPLACE);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &globalRHS);CHKERRQ(ierr);
  ierr = VecRestoreArray(patch->localRHS, &localRHS);CHKERRQ(ierr);

  ierr = VecSet(patch->localUpdate, 0.0);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(patch->gtolCounts, &pStart, NULL);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PC_Patch_Solve, pc, 0, 0, 0);CHKERRQ(ierr);
  for (sweep = 0; sweep < nsweep; sweep++) {
    for (j = start[sweep]; j*inc[sweep] < end[sweep]*inc[sweep]; j += inc[sweep]) {
      PetscInt i       = patch->user_patches ? iterationSet[j] : j;
      PetscInt start, len;

      ierr = PetscSectionGetDof(patch->gtolCounts, i+pStart, &len);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(patch->gtolCounts, i+pStart, &start);CHKERRQ(ierr);
      /* TODO: Squash out these guys in the setup as well. */
      if (len <= 0) continue;
      /* TODO: Do we need different scatters for X and Y? */
      ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->localRHS, patch->patchRHS, INSERT_VALUES, SCATTER_FORWARD, SCATTER_INTERIOR);CHKERRQ(ierr);
      ierr = (*patch->applysolver)(pc, i, patch->patchRHS, patch->patchUpdate);CHKERRQ(ierr);
      ierr = PCPatch_ScatterLocal_Private(pc, i+pStart, patch->patchUpdate, patch->localUpdate, ADD_VALUES, SCATTER_REVERSE, SCATTER_INTERIOR);CHKERRQ(ierr);
      if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
        ierr = (*patch->updatemultiplicative)(pc, i, pStart);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscLogEventEnd(PC_Patch_Solve, pc, 0, 0, 0);CHKERRQ(ierr);
  if (patch->user_patches) {ierr = ISRestoreIndices(patch->iterationSet, &iterationSet);CHKERRQ(ierr);}
  /* XXX: should we do this on the global vector? */
  if (patch->partition_of_unity) {
    ierr = VecPointwiseMult(patch->localUpdate, patch->localUpdate, patch->dof_weights);CHKERRQ(ierr);
  }
  /* Now patch->localUpdate contains the solution of the patch solves, so we need to combine them all. */
  ierr = VecSet(y, 0.0);CHKERRQ(ierr);
  ierr = VecGetArray(y, &globalUpdate);CHKERRQ(ierr);
  ierr = VecGetArrayRead(patch->localUpdate, &localUpdate);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(patch->sectionSF, MPIU_SCALAR, localUpdate, globalUpdate, MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(patch->sectionSF, MPIU_SCALAR, localUpdate, globalUpdate, MPI_SUM);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(patch->localUpdate, &localUpdate);CHKERRQ(ierr);

  /* Now we need to send the global BC values through */
  ierr = VecGetArrayRead(x, &globalRHS);CHKERRQ(ierr);
  ierr = ISGetSize(patch->globalBcNodes, &numBcs);CHKERRQ(ierr);
  ierr = ISGetIndices(patch->globalBcNodes, &bcNodes);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x, &n);CHKERRQ(ierr);
  for (bc = 0; bc < numBcs; ++bc) {
    const PetscInt idx = bcNodes[bc];
    if (idx < n) globalUpdate[idx] = globalRHS[idx];
  }

  ierr = ISRestoreIndices(patch->globalBcNodes, &bcNodes);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &globalRHS);CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &globalUpdate);CHKERRQ(ierr);

  ierr = PetscOptionsPopGetViewerOff();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_Patch_Apply, pc, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH_Linear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) {ierr = KSPReset((KSP) patch->solver[i]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_PATCH(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscSFDestroy(&patch->sectionSF);CHKERRQ(ierr);
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
  ierr = PetscSectionDestroy(&patch->gtolCountsWithArtificial);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->gtolWithArtificial);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->dofsWithArtificial);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->offsWithArtificial);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->gtolCountsWithAll);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->gtolWithAll);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->dofsWithAll);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->offsWithAll);CHKERRQ(ierr);
  ierr = VecDestroy(&patch->cellMats);CHKERRQ(ierr);
  ierr = VecDestroy(&patch->intFacetMats);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->allCells);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->intFacets);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->extFacets);CHKERRQ(ierr);
  ierr = ISDestroy(&patch->intFacetsToPatchCell);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->intFacetCounts);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&patch->extFacetCounts);CHKERRQ(ierr);

  if (patch->dofSection) for (i = 0; i < patch->nsubspaces; i++) {ierr = PetscSectionDestroy(&patch->dofSection[i]);CHKERRQ(ierr);}
  ierr = PetscFree(patch->dofSection);CHKERRQ(ierr);
  ierr = PetscFree(patch->bs);CHKERRQ(ierr);
  ierr = PetscFree(patch->nodesPerCell);CHKERRQ(ierr);
  if (patch->cellNodeMap) for (i = 0; i < patch->nsubspaces; i++) {ierr = PetscFree(patch->cellNodeMap[i]);CHKERRQ(ierr);}
  ierr = PetscFree(patch->cellNodeMap);CHKERRQ(ierr);
  ierr = PetscFree(patch->subspaceOffsets);CHKERRQ(ierr);

  ierr = (*patch->resetsolver)(pc);CHKERRQ(ierr);

  if (patch->subspaces_to_exclude) {
    PetscHSetIDestroy(&patch->subspaces_to_exclude);
  }

  ierr = VecDestroy(&patch->localRHS);CHKERRQ(ierr);
  ierr = VecDestroy(&patch->localUpdate);CHKERRQ(ierr);
  ierr = VecDestroy(&patch->patchRHS);CHKERRQ(ierr);
  ierr = VecDestroy(&patch->patchUpdate);CHKERRQ(ierr);
  ierr = VecDestroy(&patch->dof_weights);CHKERRQ(ierr);
  if (patch->patch_dof_weights) {
    for (i = 0; i < patch->npatch; ++i) {ierr = VecDestroy(&patch->patch_dof_weights[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->patch_dof_weights);CHKERRQ(ierr);
  }
  if (patch->mat) {
    for (i = 0; i < patch->npatch; ++i) {ierr = MatDestroy(&patch->mat[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->mat);CHKERRQ(ierr);
  }
  if (patch->matWithArtificial) {
    for (i = 0; i < patch->npatch; ++i) {ierr = MatDestroy(&patch->matWithArtificial[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->matWithArtificial);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&patch->patchRHSWithArtificial);CHKERRQ(ierr);
  if (patch->dofMappingWithoutToWithArtificial) {
    for (i = 0; i < patch->npatch; ++i) {ierr = ISDestroy(&patch->dofMappingWithoutToWithArtificial[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->dofMappingWithoutToWithArtificial);CHKERRQ(ierr);

  }
  if (patch->dofMappingWithoutToWithAll) {
    for (i = 0; i < patch->npatch; ++i) {ierr = ISDestroy(&patch->dofMappingWithoutToWithAll[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->dofMappingWithoutToWithAll);CHKERRQ(ierr);

  }
  ierr = PetscFree(patch->sub_mat_type);CHKERRQ(ierr);
  if (patch->userIS) {
    for (i = 0; i < patch->npatch; ++i) {ierr = ISDestroy(&patch->userIS[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->userIS);CHKERRQ(ierr);
  }
  ierr = PetscFree(patch->precomputedTensorLocations);CHKERRQ(ierr);
  ierr = PetscFree(patch->precomputedIntFacetTensorLocations);CHKERRQ(ierr);

  patch->bs          = NULL;
  patch->cellNodeMap = NULL;
  patch->nsubspaces  = 0;
  ierr = ISDestroy(&patch->iterationSet);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&patch->viewerSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH_Linear(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (patch->solver) {
    for (i = 0; i < patch->npatch; ++i) {ierr = KSPDestroy((KSP *) &patch->solver[i]);CHKERRQ(ierr);}
    ierr = PetscFree(patch->solver);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_PATCH(PC pc)
{
  PC_PATCH      *patch = (PC_PATCH *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_PATCH(pc);CHKERRQ(ierr);
  ierr = (*patch->destroysolver)(pc);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
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
  PetscErrorCode       ierr;
  PCCompositeType loctype = PC_COMPOSITE_ADDITIVE;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) pc, &comm);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject) pc, &prefix);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject, "Patch solver options");CHKERRQ(ierr);

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_save_operators", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsBool(option,  "Store all patch operators for lifetime of object?", "PCPatchSetSaveOperators", patch->save_operators, &patch->save_operators, &flg);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_precompute_element_tensors", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsBool(option,  "Compute each element tensor only once?", "PCPatchSetPrecomputeElementTensors", patch->precomputeElementTensors, &patch->precomputeElementTensors, &flg);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_partition_of_unity", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsBool(option, "Weight contributions by dof multiplicity?", "PCPatchSetPartitionOfUnity", patch->partition_of_unity, &patch->partition_of_unity, &flg);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_local_type", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsEnum(option,"Type of local solver composition (additive or multiplicative)","PCPatchSetLocalComposition",PCCompositeTypes,(PetscEnum)loctype,(PetscEnum*)&loctype,&flg);CHKERRQ(ierr);
  if (flg) { ierr = PCPatchSetLocalComposition(pc, loctype);CHKERRQ(ierr);}
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_dense_inverse", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsBool(option, "Compute inverses of patch matrices and apply directly? Ignores KSP/PC settings on patch.", "PCPatchSetDenseInverse", patch->denseinverse, &patch->denseinverse, &flg);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_construct_dim", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsInt(option, "What dimension of mesh point to construct patches by? (0 = vertices)", "PCPATCH", patch->dim, &patch->dim, &dimflg);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_construct_codim", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsInt(option, "What co-dimension of mesh point to construct patches by? (0 = cells)", "PCPATCH", patch->codim, &patch->codim, &codimflg);CHKERRQ(ierr);
  if (dimflg && codimflg) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Can only set one of dimension or co-dimension");

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_construct_type", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsEnum(option, "How should the patches be constructed?", "PCPatchSetConstructType", PCPatchConstructTypes, (PetscEnum) patchConstructionType, (PetscEnum *) &patchConstructionType, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PCPatchSetConstructType(pc, patchConstructionType, NULL, NULL);CHKERRQ(ierr);}

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_vanka_dim", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsInt(option, "Topological dimension of entities for Vanka to ignore", "PCPATCH", patch->vankadim, &patch->vankadim, &flg);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_ignore_dim", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsInt(option, "Topological dimension of entities for completion to ignore", "PCPATCH", patch->ignoredim, &patch->ignoredim, &flg);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_pardecomp_overlap", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsInt(option, "What overlap should we use in construct type pardecomp?", "PCPATCH", patch->pardecomp_overlap, &patch->pardecomp_overlap, &flg);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_sub_mat_type", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsFList(option, "Matrix type for patch solves", "PCPatchSetSubMatType", MatList, NULL, sub_mat_type, PETSC_MAX_PATH_LEN, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PCPatchSetSubMatType(pc, sub_mat_type);CHKERRQ(ierr);}

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_symmetrise_sweep", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsBool(option, "Go start->end, end->start?", "PCPATCH", patch->symmetrise_sweep, &patch->symmetrise_sweep, &flg);CHKERRQ(ierr);

  /* If the user has set the number of subspaces, use that for the buffer size,
     otherwise use a large number */
  if (patch->nsubspaces <= 0) {
    nfields = 128;
  } else {
    nfields = patch->nsubspaces;
  }
  ierr = PetscMalloc1(nfields, &ifields);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_exclude_subspaces", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray(((PetscObject)pc)->options,((PetscObject)pc)->prefix,option,ifields,&nfields,&flg);CHKERRQ(ierr);
  if (flg && (patchConstructionType == PC_PATCH_USER)) SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "We cannot support excluding a subspace with user patches because we do not index patches with a mesh point");
  if (flg) {
    PetscHSetIClear(patch->subspaces_to_exclude);
    for (k = 0; k < nfields; k++) {
      PetscHSetIAdd(patch->subspaces_to_exclude, ifields[k]);
    }
  }
  ierr = PetscFree(ifields);CHKERRQ(ierr);

  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_patches_view", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsBool(option, "Print out information during patch construction", "PCPATCH", patch->viewPatches, &patch->viewPatches, &flg);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_cells_view", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerCells, &patch->formatCells, &patch->viewCells);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_interior_facets_view", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerIntFacets, &patch->formatIntFacets, &patch->viewIntFacets);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_exterior_facets_view", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerExtFacets, &patch->formatExtFacets, &patch->viewExtFacets);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_points_view", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerPoints, &patch->formatPoints, &patch->viewPoints);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_section_view", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerSection, &patch->formatSection, &patch->viewSection);CHKERRQ(ierr);
  ierr = PetscSNPrintf(option, PETSC_MAX_PATH_LEN, "-%s_patch_mat_view", patch->classname);CHKERRQ(ierr);
  ierr = PetscOptionsGetViewer(comm,((PetscObject) pc)->options, prefix, option, &patch->viewerMatrix, &patch->formatMatrix, &patch->viewMatrix);CHKERRQ(ierr);
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
      ierr = KSPSetFromOptions((KSP) patch->solver[i]);CHKERRQ(ierr);
    }
    ierr = KSPSetUp((KSP) patch->solver[i]);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason((KSP) patch->solver[i], &reason);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* TODO Redo tabbing with set tbas in new style */
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) pc), &rank);CHKERRMPI(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Subspace Correction preconditioner with %d patches\n", patch->npatch);CHKERRQ(ierr);
  if (patch->local_composition_type == PC_COMPOSITE_MULTIPLICATIVE) {
    ierr = PetscViewerASCIIPrintf(viewer, "Schwarz type: multiplicative\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "Schwarz type: additive\n");CHKERRQ(ierr);
  }
  if (patch->partition_of_unity) {ierr = PetscViewerASCIIPrintf(viewer, "Weighting by partition of unity\n");CHKERRQ(ierr);}
  else                           {ierr = PetscViewerASCIIPrintf(viewer, "Not weighting by partition of unity\n");CHKERRQ(ierr);}
  if (patch->symmetrise_sweep) {ierr = PetscViewerASCIIPrintf(viewer, "Symmetrising sweep (start->end, then end->start)\n");CHKERRQ(ierr);}
  else                         {ierr = PetscViewerASCIIPrintf(viewer, "Not symmetrising sweep\n");CHKERRQ(ierr);}
  if (!patch->precomputeElementTensors) {ierr = PetscViewerASCIIPrintf(viewer, "Not precomputing element tensors (overlapping cells rebuilt in every patch assembly)\n");CHKERRQ(ierr);}
  else                            {ierr = PetscViewerASCIIPrintf(viewer, "Precomputing element tensors (each cell assembled only once)\n");CHKERRQ(ierr);}
  if (!patch->save_operators) {ierr = PetscViewerASCIIPrintf(viewer, "Not saving patch operators (rebuilt every PCApply)\n");CHKERRQ(ierr);}
  else                        {ierr = PetscViewerASCIIPrintf(viewer, "Saving patch operators (rebuilt every PCSetUp)\n");CHKERRQ(ierr);}
  if (patch->patchconstructop == PCPatchConstruct_Star)       {ierr = PetscViewerASCIIPrintf(viewer, "Patch construction operator: star\n");CHKERRQ(ierr);}
  else if (patch->patchconstructop == PCPatchConstruct_Vanka) {ierr = PetscViewerASCIIPrintf(viewer, "Patch construction operator: Vanka\n");CHKERRQ(ierr);}
  else if (patch->patchconstructop == PCPatchConstruct_User)  {ierr = PetscViewerASCIIPrintf(viewer, "Patch construction operator: user-specified\n");CHKERRQ(ierr);}
  else                                                        {ierr = PetscViewerASCIIPrintf(viewer, "Patch construction operator: unknown\n");CHKERRQ(ierr);}

  if (patch->denseinverse) {
    ierr = PetscViewerASCIIPrintf(viewer, "Explicitly forming dense inverse and applying patch solver via MatMult.\n");CHKERRQ(ierr);
  } else {
    if (patch->isNonlinear) {
      ierr = PetscViewerASCIIPrintf(viewer, "SNES on patches (all same):\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "KSP on patches (all same):\n");CHKERRQ(ierr);
    }
    if (patch->solver) {
      ierr = PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sviewer);CHKERRQ(ierr);
      if (!rank) {
        ierr = PetscViewerASCIIPushTab(sviewer);CHKERRQ(ierr);
        ierr = PetscObjectView(patch->solver[0], sviewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(sviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sviewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "Solver not yet set.\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
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

  if (patch->subspaces_to_exclude) {
    PetscHSetIDestroy(&patch->subspaces_to_exclude);
  }
  PetscHSetICreate(&patch->subspaces_to_exclude);

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
  ierr = PetscStrallocpy(MATDENSE, (char **) &patch->sub_mat_type);CHKERRQ(ierr);
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
