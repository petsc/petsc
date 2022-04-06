#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/isimpl.h>
#include <petscsf.h>
#include <petscds.h>

/* get adjacencies due to point-to-point constraints that can't be found with DMPlexGetAdjacency() */
static PetscErrorCode DMPlexComputeAnchorAdjacencies(DM dm, PetscBool useCone, PetscBool useClosure, PetscSection *anchorSectionAdj, PetscInt *anchorAdj[])
{
  PetscInt       pStart, pEnd;
  PetscSection   section, sectionGlobal, adjSec, aSec;
  IS             aIS;

  PetscFunctionBegin;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &sectionGlobal));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) section), &adjSec));
  PetscCall(PetscSectionGetChart(section,&pStart,&pEnd));
  PetscCall(PetscSectionSetChart(adjSec,pStart,pEnd));

  PetscCall(DMPlexGetAnchors(dm,&aSec,&aIS));
  if (aSec) {
    const PetscInt *anchors;
    PetscInt       p, q, a, aSize, *offsets, aStart, aEnd, *inverse, iSize, *adj, adjSize;
    PetscInt       *tmpAdjP = NULL, *tmpAdjQ = NULL;
    PetscSection   inverseSec;

    /* invert the constraint-to-anchor map */
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)aSec),&inverseSec));
    PetscCall(PetscSectionSetChart(inverseSec,pStart,pEnd));
    PetscCall(ISGetLocalSize(aIS, &aSize));
    PetscCall(ISGetIndices(aIS, &anchors));

    for (p = 0; p < aSize; p++) {
      PetscInt a = anchors[p];

      PetscCall(PetscSectionAddDof(inverseSec,a,1));
    }
    PetscCall(PetscSectionSetUp(inverseSec));
    PetscCall(PetscSectionGetStorageSize(inverseSec,&iSize));
    PetscCall(PetscMalloc1(iSize,&inverse));
    PetscCall(PetscCalloc1(pEnd-pStart,&offsets));
    PetscCall(PetscSectionGetChart(aSec,&aStart,&aEnd));
    for (p = aStart; p < aEnd; p++) {
      PetscInt dof, off;

      PetscCall(PetscSectionGetDof(aSec, p, &dof));
      PetscCall(PetscSectionGetOffset(aSec, p, &off));

      for (q = 0; q < dof; q++) {
        PetscInt iOff;

        a = anchors[off + q];
        PetscCall(PetscSectionGetOffset(inverseSec, a, &iOff));
        inverse[iOff + offsets[a-pStart]++] = p;
      }
    }
    PetscCall(ISRestoreIndices(aIS, &anchors));
    PetscCall(PetscFree(offsets));

    /* construct anchorAdj and adjSec
     *
     * loop over anchors:
     *   construct anchor adjacency
     *   loop over constrained:
     *     construct constrained adjacency
     *     if not in anchor adjacency, add to dofs
     * setup adjSec, allocate anchorAdj
     * loop over anchors:
     *   construct anchor adjacency
     *   loop over constrained:
     *     construct constrained adjacency
     *     if not in anchor adjacency
     *       if not already in list, put in list
     *   sort, unique, reduce dof count
     * optional: compactify
     */
    for (p = pStart; p < pEnd; p++) {
      PetscInt iDof, iOff, i, r, s, numAdjP = PETSC_DETERMINE;

      PetscCall(PetscSectionGetDof(inverseSec,p,&iDof));
      if (!iDof) continue;
      PetscCall(PetscSectionGetOffset(inverseSec,p,&iOff));
      PetscCall(DMPlexGetAdjacency_Internal(dm,p,useCone,useClosure,PETSC_TRUE,&numAdjP,&tmpAdjP));
      for (i = 0; i < iDof; i++) {
        PetscInt iNew = 0, qAdj, qAdjDof, qAdjCDof, numAdjQ = PETSC_DETERMINE;

        q = inverse[iOff + i];
        PetscCall(DMPlexGetAdjacency_Internal(dm,q,useCone,useClosure,PETSC_TRUE,&numAdjQ,&tmpAdjQ));
        for (r = 0; r < numAdjQ; r++) {
          qAdj = tmpAdjQ[r];
          if ((qAdj < pStart) || (qAdj >= pEnd)) continue;
          for (s = 0; s < numAdjP; s++) {
            if (qAdj == tmpAdjP[s]) break;
          }
          if (s < numAdjP) continue;
          PetscCall(PetscSectionGetDof(section,qAdj,&qAdjDof));
          PetscCall(PetscSectionGetConstraintDof(section,qAdj,&qAdjCDof));
          iNew += qAdjDof - qAdjCDof;
        }
        PetscCall(PetscSectionAddDof(adjSec,p,iNew));
      }
    }

    PetscCall(PetscSectionSetUp(adjSec));
    PetscCall(PetscSectionGetStorageSize(adjSec,&adjSize));
    PetscCall(PetscMalloc1(adjSize,&adj));

    for (p = pStart; p < pEnd; p++) {
      PetscInt iDof, iOff, i, r, s, aOff, aOffOrig, aDof, numAdjP = PETSC_DETERMINE;

      PetscCall(PetscSectionGetDof(inverseSec,p,&iDof));
      if (!iDof) continue;
      PetscCall(PetscSectionGetOffset(inverseSec,p,&iOff));
      PetscCall(DMPlexGetAdjacency_Internal(dm,p,useCone,useClosure,PETSC_TRUE,&numAdjP,&tmpAdjP));
      PetscCall(PetscSectionGetDof(adjSec,p,&aDof));
      PetscCall(PetscSectionGetOffset(adjSec,p,&aOff));
      aOffOrig = aOff;
      for (i = 0; i < iDof; i++) {
        PetscInt qAdj, qAdjDof, qAdjCDof, qAdjOff, nd, numAdjQ = PETSC_DETERMINE;

        q = inverse[iOff + i];
        PetscCall(DMPlexGetAdjacency_Internal(dm,q,useCone,useClosure,PETSC_TRUE,&numAdjQ,&tmpAdjQ));
        for (r = 0; r < numAdjQ; r++) {
          qAdj = tmpAdjQ[r];
          if ((qAdj < pStart) || (qAdj >= pEnd)) continue;
          for (s = 0; s < numAdjP; s++) {
            if (qAdj == tmpAdjP[s]) break;
          }
          if (s < numAdjP) continue;
          PetscCall(PetscSectionGetDof(section,qAdj,&qAdjDof));
          PetscCall(PetscSectionGetConstraintDof(section,qAdj,&qAdjCDof));
          PetscCall(PetscSectionGetOffset(sectionGlobal,qAdj,&qAdjOff));
          for (nd = 0; nd < qAdjDof-qAdjCDof; ++nd) {
            adj[aOff++] = (qAdjOff < 0 ? -(qAdjOff+1) : qAdjOff) + nd;
          }
        }
      }
      PetscCall(PetscSortRemoveDupsInt(&aDof,&adj[aOffOrig]));
      PetscCall(PetscSectionSetDof(adjSec,p,aDof));
    }
    *anchorAdj = adj;

    /* clean up */
    PetscCall(PetscSectionDestroy(&inverseSec));
    PetscCall(PetscFree(inverse));
    PetscCall(PetscFree(tmpAdjP));
    PetscCall(PetscFree(tmpAdjQ));
  }
  else {
    *anchorAdj = NULL;
    PetscCall(PetscSectionSetUp(adjSec));
  }
  *anchorSectionAdj = adjSec;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateAdjacencySection_Static(DM dm, PetscInt bs, PetscSF sfDof, PetscBool useCone, PetscBool useClosure, PetscBool useAnchors, PetscSection *sA, PetscInt **colIdx)
{
  MPI_Comm           comm;
  PetscMPIInt        size;
  PetscBool          doCommLocal, doComm, debug = PETSC_FALSE;
  PetscSF            sf, sfAdj;
  PetscSection       section, sectionGlobal, leafSectionAdj, rootSectionAdj, sectionAdj, anchorSectionAdj;
  PetscInt           nroots, nleaves, l, p, r;
  const PetscInt    *leaves;
  const PetscSFNode *remotes;
  PetscInt           dim, pStart, pEnd, numDof, globalOffStart, globalOffEnd, numCols;
  PetscInt          *tmpAdj = NULL, *adj, *rootAdj, *anchorAdj = NULL, *cols, *remoteOffsets;
  PetscInt           adjSize;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-dm_view_preallocation", &debug, NULL));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &sectionGlobal));
  PetscCall(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL));
  doCommLocal = (size > 1) && (nroots >= 0) ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(MPIU_Allreduce(&doCommLocal, &doComm, 1, MPIU_BOOL, MPI_LAND, comm));
  /* Create section for dof adjacency (dof ==> # adj dof) */
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  PetscCall(PetscSectionGetStorageSize(section, &numDof));
  PetscCall(PetscSectionCreate(comm, &leafSectionAdj));
  PetscCall(PetscSectionSetChart(leafSectionAdj, 0, numDof));
  PetscCall(PetscSectionCreate(comm, &rootSectionAdj));
  PetscCall(PetscSectionSetChart(rootSectionAdj, 0, numDof));
  /*   Fill in the ghost dofs on the interface */
  PetscCall(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, &remotes));
  /*
   section        - maps points to (# dofs, local dofs)
   sectionGlobal  - maps points to (# dofs, global dofs)
   leafSectionAdj - maps unowned local dofs to # adj dofs
   rootSectionAdj - maps   owned local dofs to # adj dofs
   adj            - adj global dofs indexed by leafSectionAdj
   rootAdj        - adj global dofs indexed by rootSectionAdj
   sf    - describes shared points across procs
   sfDof - describes shared dofs across procs
   sfAdj - describes shared adjacent dofs across procs
   ** The bootstrapping process involves six rounds with similar structure of visiting neighbors of each point.
  (0). If there are point-to-point constraints, add the adjacencies of constrained points to anchors in anchorAdj
       (This is done in DMPlexComputeAnchorAdjacencies())
    1. Visit unowned points on interface, count adjacencies placing in leafSectionAdj
       Reduce those counts to rootSectionAdj (now redundantly counting some interface points)
    2. Visit owned points on interface, count adjacencies placing in rootSectionAdj
       Create sfAdj connecting rootSectionAdj and leafSectionAdj
    3. Visit unowned points on interface, write adjacencies to adj
       Gather adj to rootAdj (note that there is redundancy in rootAdj when multiple procs find the same adjacencies)
    4. Visit owned points on interface, write adjacencies to rootAdj
       Remove redundancy in rootAdj
   ** The last two traversals use transitive closure
    5. Visit all owned points in the subdomain, count dofs for each point (sectionAdj)
       Allocate memory addressed by sectionAdj (cols)
    6. Visit all owned points in the subdomain, insert dof adjacencies into cols
   ** Knowing all the column adjacencies, check ownership and sum into dnz and onz
  */
  PetscCall(DMPlexComputeAnchorAdjacencies(dm, useCone, useClosure, &anchorSectionAdj, &anchorAdj));
  for (l = 0; l < nleaves; ++l) {
    PetscInt dof, off, d, q, anDof;
    PetscInt p = leaves[l], numAdj = PETSC_DETERMINE;

    if ((p < pStart) || (p >= pEnd)) continue;
    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetOffset(section, p, &off));
    PetscCall(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    for (q = 0; q < numAdj; ++q) {
      const PetscInt padj = tmpAdj[q];
      PetscInt ndof, ncdof;

      if ((padj < pStart) || (padj >= pEnd)) continue;
      PetscCall(PetscSectionGetDof(section, padj, &ndof));
      PetscCall(PetscSectionGetConstraintDof(section, padj, &ncdof));
      for (d = off; d < off+dof; ++d) {
        PetscCall(PetscSectionAddDof(leafSectionAdj, d, ndof-ncdof));
      }
    }
    PetscCall(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    if (anDof) {
      for (d = off; d < off+dof; ++d) {
        PetscCall(PetscSectionAddDof(leafSectionAdj, d, anDof));
      }
    }
  }
  PetscCall(PetscSectionSetUp(leafSectionAdj));
  if (debug) {
    PetscCall(PetscPrintf(comm, "Adjacency Section for Preallocation on Leaves:\n"));
    PetscCall(PetscSectionView(leafSectionAdj, NULL));
  }
  /* Get maximum remote adjacency sizes for owned dofs on interface (roots) */
  if (doComm) {
    PetscCall(PetscSFReduceBegin(sfDof, MPIU_INT, leafSectionAdj->atlasDof, rootSectionAdj->atlasDof, MPI_SUM));
    PetscCall(PetscSFReduceEnd(sfDof, MPIU_INT, leafSectionAdj->atlasDof, rootSectionAdj->atlasDof, MPI_SUM));
    PetscCall(PetscSectionInvalidateMaxDof_Internal(rootSectionAdj));
  }
  if (debug) {
    PetscCall(PetscPrintf(comm, "Adjancency Section for Preallocation on Roots:\n"));
    PetscCall(PetscSectionView(rootSectionAdj, NULL));
  }
  /* Add in local adjacency sizes for owned dofs on interface (roots) */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt numAdj = PETSC_DETERMINE, adof, dof, off, d, q, anDof;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetOffset(section, p, &off));
    if (!dof) continue;
    PetscCall(PetscSectionGetDof(rootSectionAdj, off, &adof));
    if (adof <= 0) continue;
    PetscCall(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    for (q = 0; q < numAdj; ++q) {
      const PetscInt padj = tmpAdj[q];
      PetscInt ndof, ncdof;

      if ((padj < pStart) || (padj >= pEnd)) continue;
      PetscCall(PetscSectionGetDof(section, padj, &ndof));
      PetscCall(PetscSectionGetConstraintDof(section, padj, &ncdof));
      for (d = off; d < off+dof; ++d) {
        PetscCall(PetscSectionAddDof(rootSectionAdj, d, ndof-ncdof));
      }
    }
    PetscCall(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    if (anDof) {
      for (d = off; d < off+dof; ++d) {
        PetscCall(PetscSectionAddDof(rootSectionAdj, d, anDof));
      }
    }
  }
  PetscCall(PetscSectionSetUp(rootSectionAdj));
  if (debug) {
    PetscCall(PetscPrintf(comm, "Adjancency Section for Preallocation on Roots after local additions:\n"));
    PetscCall(PetscSectionView(rootSectionAdj, NULL));
  }
  /* Create adj SF based on dof SF */
  PetscCall(PetscSFCreateRemoteOffsets(sfDof, rootSectionAdj, leafSectionAdj, &remoteOffsets));
  PetscCall(PetscSFCreateSectionSF(sfDof, rootSectionAdj, remoteOffsets, leafSectionAdj, &sfAdj));
  PetscCall(PetscFree(remoteOffsets));
  if (debug && size > 1) {
    PetscCall(PetscPrintf(comm, "Adjacency SF for Preallocation:\n"));
    PetscCall(PetscSFView(sfAdj, NULL));
  }
  /* Create leaf adjacency */
  PetscCall(PetscSectionSetUp(leafSectionAdj));
  PetscCall(PetscSectionGetStorageSize(leafSectionAdj, &adjSize));
  PetscCall(PetscCalloc1(adjSize, &adj));
  for (l = 0; l < nleaves; ++l) {
    PetscInt dof, off, d, q, anDof, anOff;
    PetscInt p = leaves[l], numAdj = PETSC_DETERMINE;

    if ((p < pStart) || (p >= pEnd)) continue;
    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetOffset(section, p, &off));
    PetscCall(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    PetscCall(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    PetscCall(PetscSectionGetOffset(anchorSectionAdj, p, &anOff));
    for (d = off; d < off+dof; ++d) {
      PetscInt aoff, i = 0;

      PetscCall(PetscSectionGetOffset(leafSectionAdj, d, &aoff));
      for (q = 0; q < numAdj; ++q) {
        const PetscInt padj = tmpAdj[q];
        PetscInt ndof, ncdof, ngoff, nd;

        if ((padj < pStart) || (padj >= pEnd)) continue;
        PetscCall(PetscSectionGetDof(section, padj, &ndof));
        PetscCall(PetscSectionGetConstraintDof(section, padj, &ncdof));
        PetscCall(PetscSectionGetOffset(sectionGlobal, padj, &ngoff));
        for (nd = 0; nd < ndof-ncdof; ++nd) {
          adj[aoff+i] = (ngoff < 0 ? -(ngoff+1) : ngoff) + nd;
          ++i;
        }
      }
      for (q = 0; q < anDof; q++) {
        adj[aoff+i] = anchorAdj[anOff+q];
        ++i;
      }
    }
  }
  /* Debugging */
  if (debug) {
    IS tmp;
    PetscCall(PetscPrintf(comm, "Leaf adjacency indices\n"));
    PetscCall(ISCreateGeneral(comm, adjSize, adj, PETSC_USE_POINTER, &tmp));
    PetscCall(ISView(tmp, NULL));
    PetscCall(ISDestroy(&tmp));
  }
  /* Gather adjacent indices to root */
  PetscCall(PetscSectionGetStorageSize(rootSectionAdj, &adjSize));
  PetscCall(PetscMalloc1(adjSize, &rootAdj));
  for (r = 0; r < adjSize; ++r) rootAdj[r] = -1;
  if (doComm) {
    const PetscInt *indegree;
    PetscInt       *remoteadj, radjsize = 0;

    PetscCall(PetscSFComputeDegreeBegin(sfAdj, &indegree));
    PetscCall(PetscSFComputeDegreeEnd(sfAdj, &indegree));
    for (p = 0; p < adjSize; ++p) radjsize += indegree[p];
    PetscCall(PetscMalloc1(radjsize, &remoteadj));
    PetscCall(PetscSFGatherBegin(sfAdj, MPIU_INT, adj, remoteadj));
    PetscCall(PetscSFGatherEnd(sfAdj, MPIU_INT, adj, remoteadj));
    for (p = 0, l = 0, r = 0; p < adjSize; ++p, l = PetscMax(p, l + indegree[p-1])) {
      PetscInt s;
      for (s = 0; s < indegree[p]; ++s, ++r) rootAdj[l+s] = remoteadj[r];
    }
    PetscCheck(r == radjsize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistency in communication %" PetscInt_FMT " != %" PetscInt_FMT, r, radjsize);
    PetscCheck(l == adjSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistency in communication %" PetscInt_FMT " != %" PetscInt_FMT, l, adjSize);
    PetscCall(PetscFree(remoteadj));
  }
  PetscCall(PetscSFDestroy(&sfAdj));
  PetscCall(PetscFree(adj));
  /* Debugging */
  if (debug) {
    IS tmp;
    PetscCall(PetscPrintf(comm, "Root adjacency indices after gather\n"));
    PetscCall(ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp));
    PetscCall(ISView(tmp, NULL));
    PetscCall(ISDestroy(&tmp));
  }
  /* Add in local adjacency indices for owned dofs on interface (roots) */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt numAdj = PETSC_DETERMINE, adof, dof, off, d, q, anDof, anOff;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetOffset(section, p, &off));
    if (!dof) continue;
    PetscCall(PetscSectionGetDof(rootSectionAdj, off, &adof));
    if (adof <= 0) continue;
    PetscCall(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    PetscCall(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    PetscCall(PetscSectionGetOffset(anchorSectionAdj, p, &anOff));
    for (d = off; d < off+dof; ++d) {
      PetscInt adof, aoff, i;

      PetscCall(PetscSectionGetDof(rootSectionAdj, d, &adof));
      PetscCall(PetscSectionGetOffset(rootSectionAdj, d, &aoff));
      i    = adof-1;
      for (q = 0; q < anDof; q++) {
        rootAdj[aoff+i] = anchorAdj[anOff+q];
        --i;
      }
      for (q = 0; q < numAdj; ++q) {
        const PetscInt padj = tmpAdj[q];
        PetscInt ndof, ncdof, ngoff, nd;

        if ((padj < pStart) || (padj >= pEnd)) continue;
        PetscCall(PetscSectionGetDof(section, padj, &ndof));
        PetscCall(PetscSectionGetConstraintDof(section, padj, &ncdof));
        PetscCall(PetscSectionGetOffset(sectionGlobal, padj, &ngoff));
        for (nd = 0; nd < ndof-ncdof; ++nd) {
          rootAdj[aoff+i] = ngoff < 0 ? -(ngoff+1)+nd : ngoff+nd;
          --i;
        }
      }
    }
  }
  /* Debugging */
  if (debug) {
    IS tmp;
    PetscCall(PetscPrintf(comm, "Root adjacency indices\n"));
    PetscCall(ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp));
    PetscCall(ISView(tmp, NULL));
    PetscCall(ISDestroy(&tmp));
  }
  /* Compress indices */
  PetscCall(PetscSectionSetUp(rootSectionAdj));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, off, d;
    PetscInt adof, aoff;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetConstraintDof(section, p, &cdof));
    PetscCall(PetscSectionGetOffset(section, p, &off));
    if (!dof) continue;
    PetscCall(PetscSectionGetDof(rootSectionAdj, off, &adof));
    if (adof <= 0) continue;
    for (d = off; d < off+dof-cdof; ++d) {
      PetscCall(PetscSectionGetDof(rootSectionAdj, d, &adof));
      PetscCall(PetscSectionGetOffset(rootSectionAdj, d, &aoff));
      PetscCall(PetscSortRemoveDupsInt(&adof, &rootAdj[aoff]));
      PetscCall(PetscSectionSetDof(rootSectionAdj, d, adof));
    }
  }
  /* Debugging */
  if (debug) {
    IS tmp;
    PetscCall(PetscPrintf(comm, "Adjancency Section for Preallocation on Roots after compression:\n"));
    PetscCall(PetscSectionView(rootSectionAdj, NULL));
    PetscCall(PetscPrintf(comm, "Root adjacency indices after compression\n"));
    PetscCall(ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp));
    PetscCall(ISView(tmp, NULL));
    PetscCall(ISDestroy(&tmp));
  }
  /* Build adjacency section: Maps global indices to sets of adjacent global indices */
  PetscCall(PetscSectionGetOffsetRange(sectionGlobal, &globalOffStart, &globalOffEnd));
  PetscCall(PetscSectionCreate(comm, &sectionAdj));
  PetscCall(PetscSectionSetChart(sectionAdj, globalOffStart, globalOffEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  numAdj = PETSC_DETERMINE, dof, cdof, off, goff, d, q, anDof;
    PetscBool found  = PETSC_TRUE;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetConstraintDof(section, p, &cdof));
    PetscCall(PetscSectionGetOffset(section, p, &off));
    PetscCall(PetscSectionGetOffset(sectionGlobal, p, &goff));
    for (d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      PetscCall(PetscSectionGetDof(leafSectionAdj, off+d, &ldof));
      PetscCall(PetscSectionGetDof(rootSectionAdj, off+d, &rdof));
      if (ldof > 0) {
        /* We do not own this point */
      } else if (rdof > 0) {
        PetscCall(PetscSectionSetDof(sectionAdj, goff+d, rdof));
      } else {
        found = PETSC_FALSE;
      }
    }
    if (found) continue;
    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetOffset(sectionGlobal, p, &goff));
    PetscCall(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    for (q = 0; q < numAdj; ++q) {
      const PetscInt padj = tmpAdj[q];
      PetscInt ndof, ncdof, noff;

      if ((padj < pStart) || (padj >= pEnd)) continue;
      PetscCall(PetscSectionGetDof(section, padj, &ndof));
      PetscCall(PetscSectionGetConstraintDof(section, padj, &ncdof));
      PetscCall(PetscSectionGetOffset(section, padj, &noff));
      for (d = goff; d < goff+dof-cdof; ++d) {
        PetscCall(PetscSectionAddDof(sectionAdj, d, ndof-ncdof));
      }
    }
    PetscCall(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    if (anDof) {
      for (d = goff; d < goff+dof-cdof; ++d) {
        PetscCall(PetscSectionAddDof(sectionAdj, d, anDof));
      }
    }
  }
  PetscCall(PetscSectionSetUp(sectionAdj));
  if (debug) {
    PetscCall(PetscPrintf(comm, "Adjacency Section for Preallocation:\n"));
    PetscCall(PetscSectionView(sectionAdj, NULL));
  }
  /* Get adjacent indices */
  PetscCall(PetscSectionGetStorageSize(sectionAdj, &numCols));
  PetscCall(PetscMalloc1(numCols, &cols));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  numAdj = PETSC_DETERMINE, dof, cdof, off, goff, d, q, anDof, anOff;
    PetscBool found  = PETSC_TRUE;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    PetscCall(PetscSectionGetConstraintDof(section, p, &cdof));
    PetscCall(PetscSectionGetOffset(section, p, &off));
    PetscCall(PetscSectionGetOffset(sectionGlobal, p, &goff));
    for (d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      PetscCall(PetscSectionGetDof(leafSectionAdj, off+d, &ldof));
      PetscCall(PetscSectionGetDof(rootSectionAdj, off+d, &rdof));
      if (ldof > 0) {
        /* We do not own this point */
      } else if (rdof > 0) {
        PetscInt aoff, roff;

        PetscCall(PetscSectionGetOffset(sectionAdj, goff+d, &aoff));
        PetscCall(PetscSectionGetOffset(rootSectionAdj, off+d, &roff));
        PetscCall(PetscArraycpy(&cols[aoff], &rootAdj[roff], rdof));
      } else {
        found = PETSC_FALSE;
      }
    }
    if (found) continue;
    PetscCall(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    PetscCall(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    PetscCall(PetscSectionGetOffset(anchorSectionAdj, p, &anOff));
    for (d = goff; d < goff+dof-cdof; ++d) {
      PetscInt adof, aoff, i = 0;

      PetscCall(PetscSectionGetDof(sectionAdj, d, &adof));
      PetscCall(PetscSectionGetOffset(sectionAdj, d, &aoff));
      for (q = 0; q < numAdj; ++q) {
        const PetscInt  padj = tmpAdj[q];
        PetscInt        ndof, ncdof, ngoff, nd;
        const PetscInt *ncind;

        /* Adjacent points may not be in the section chart */
        if ((padj < pStart) || (padj >= pEnd)) continue;
        PetscCall(PetscSectionGetDof(section, padj, &ndof));
        PetscCall(PetscSectionGetConstraintDof(section, padj, &ncdof));
        PetscCall(PetscSectionGetConstraintIndices(section, padj, &ncind));
        PetscCall(PetscSectionGetOffset(sectionGlobal, padj, &ngoff));
        for (nd = 0; nd < ndof-ncdof; ++nd, ++i) {
          cols[aoff+i] = ngoff < 0 ? -(ngoff+1)+nd : ngoff+nd;
        }
      }
      for (q = 0; q < anDof; q++, i++) {
        cols[aoff+i] = anchorAdj[anOff + q];
      }
      PetscCheck(i == adof,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of entries %" PetscInt_FMT " != %" PetscInt_FMT " for dof %" PetscInt_FMT " (point %" PetscInt_FMT ")", i, adof, d, p);
    }
  }
  PetscCall(PetscSectionDestroy(&anchorSectionAdj));
  PetscCall(PetscSectionDestroy(&leafSectionAdj));
  PetscCall(PetscSectionDestroy(&rootSectionAdj));
  PetscCall(PetscFree(anchorAdj));
  PetscCall(PetscFree(rootAdj));
  PetscCall(PetscFree(tmpAdj));
  /* Debugging */
  if (debug) {
    IS tmp;
    PetscCall(PetscPrintf(comm, "Column indices\n"));
    PetscCall(ISCreateGeneral(comm, numCols, cols, PETSC_USE_POINTER, &tmp));
    PetscCall(ISView(tmp, NULL));
    PetscCall(ISDestroy(&tmp));
  }

  *sA     = sectionAdj;
  *colIdx = cols;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexUpdateAllocation_Static(DM dm, PetscLayout rLayout, PetscInt bs, PetscInt f, PetscSection sectionAdj, const PetscInt cols[], PetscInt dnz[], PetscInt onz[], PetscInt dnzu[], PetscInt onzu[])
{
  PetscSection   section;
  PetscInt       rStart, rEnd, r, pStart, pEnd, p;

  PetscFunctionBegin;
  /* This loop needs to change to a loop over points, then field dofs, which means we need to look both sections */
  PetscCall(PetscLayoutGetRange(rLayout, &rStart, &rEnd));
  PetscCheckFalse(rStart%bs || rEnd%bs,PetscObjectComm((PetscObject) rLayout), PETSC_ERR_ARG_WRONG, "Invalid layout [%" PetscInt_FMT ", %" PetscInt_FMT ") for matrix, must be divisible by block size %" PetscInt_FMT, rStart, rEnd, bs);
  if (f >= 0 && bs == 1) {
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt rS, rE;

      PetscCall(DMGetGlobalFieldOffset_Private(dm, p, f, &rS, &rE));
      for (r = rS; r < rE; ++r) {
        PetscInt numCols, cStart, c;

        PetscCall(PetscSectionGetDof(sectionAdj, r, &numCols));
        PetscCall(PetscSectionGetOffset(sectionAdj, r, &cStart));
        for (c = cStart; c < cStart+numCols; ++c) {
          if ((cols[c] >= rStart) && (cols[c] < rEnd)) {
            ++dnz[r-rStart];
            if (cols[c] >= r) ++dnzu[r-rStart];
          } else {
            ++onz[r-rStart];
            if (cols[c] >= r) ++onzu[r-rStart];
          }
        }
      }
    }
  } else {
    /* Only loop over blocks of rows */
    for (r = rStart/bs; r < rEnd/bs; ++r) {
      const PetscInt row = r*bs;
      PetscInt       numCols, cStart, c;

      PetscCall(PetscSectionGetDof(sectionAdj, row, &numCols));
      PetscCall(PetscSectionGetOffset(sectionAdj, row, &cStart));
      for (c = cStart; c < cStart+numCols; ++c) {
        if ((cols[c] >= rStart) && (cols[c] < rEnd)) {
          ++dnz[r-rStart/bs];
          if (cols[c] >= row) ++dnzu[r-rStart/bs];
        } else {
          ++onz[r-rStart/bs];
          if (cols[c] >= row) ++onzu[r-rStart/bs];
        }
      }
    }
    for (r = 0; r < (rEnd - rStart)/bs; ++r) {
      dnz[r]  /= bs;
      onz[r]  /= bs;
      dnzu[r] /= bs;
      onzu[r] /= bs;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexFillMatrix_Static(DM dm, PetscLayout rLayout, PetscInt bs, PetscInt f, PetscSection sectionAdj, const PetscInt cols[], Mat A)
{
  PetscSection   section;
  PetscScalar   *values;
  PetscInt       rStart, rEnd, r, pStart, pEnd, p, len, maxRowLen = 0;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetRange(rLayout, &rStart, &rEnd));
  for (r = rStart; r < rEnd; ++r) {
    PetscCall(PetscSectionGetDof(sectionAdj, r, &len));
    maxRowLen = PetscMax(maxRowLen, len);
  }
  PetscCall(PetscCalloc1(maxRowLen, &values));
  if (f >=0 && bs == 1) {
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt rS, rE;

      PetscCall(DMGetGlobalFieldOffset_Private(dm, p, f, &rS, &rE));
      for (r = rS; r < rE; ++r) {
        PetscInt numCols, cStart;

        PetscCall(PetscSectionGetDof(sectionAdj, r, &numCols));
        PetscCall(PetscSectionGetOffset(sectionAdj, r, &cStart));
        PetscCall(MatSetValues(A, 1, &r, numCols, &cols[cStart], values, INSERT_VALUES));
      }
    }
  } else {
    for (r = rStart; r < rEnd; ++r) {
      PetscInt numCols, cStart;

      PetscCall(PetscSectionGetDof(sectionAdj, r, &numCols));
      PetscCall(PetscSectionGetOffset(sectionAdj, r, &cStart));
      PetscCall(MatSetValues(A, 1, &r, numCols, &cols[cStart], values, INSERT_VALUES));
    }
  }
  PetscCall(PetscFree(values));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexPreallocateOperator - Calculate the matrix nonzero pattern based upon the information in the DM,
  the PetscDS it contains, and the default PetscSection.

  Collective

  Input Parameters:
+ dm   - The DMPlex
. bs   - The matrix blocksize
. dnz  - An array to hold the number of nonzeros in the diagonal block
. onz  - An array to hold the number of nonzeros in the off-diagonal block
. dnzu - An array to hold the number of nonzeros in the upper triangle of the diagonal block
. onzu - An array to hold the number of nonzeros in the upper triangle of the off-diagonal block
- fillMatrix - If PETSC_TRUE, fill the matrix with zeros

  Output Parameter:
. A - The preallocated matrix

  Level: advanced

.seealso: DMCreateMatrix()
@*/
PetscErrorCode DMPlexPreallocateOperator(DM dm, PetscInt bs, PetscInt dnz[], PetscInt onz[], PetscInt dnzu[], PetscInt onzu[], Mat A, PetscBool fillMatrix)
{
  MPI_Comm       comm;
  PetscDS        prob;
  MatType        mtype;
  PetscSF        sf, sfDof;
  PetscSection   section;
  PetscInt      *remoteOffsets;
  PetscSection   sectionAdj[4] = {NULL, NULL, NULL, NULL};
  PetscInt      *cols[4]       = {NULL, NULL, NULL, NULL};
  PetscBool      useCone, useClosure;
  PetscInt       Nf, f, idx, locRows;
  PetscLayout    rLayout;
  PetscBool      isSymBlock, isSymSeqBlock, isSymMPIBlock, debug = PETSC_FALSE;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 7);
  if (dnz) PetscValidIntPointer(dnz,3);
  if (onz) PetscValidIntPointer(onz,4);
  if (dnzu) PetscValidIntPointer(dnzu,5);
  if (onzu) PetscValidIntPointer(onzu,6);
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-dm_view_preallocation", &debug, NULL));
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscLogEventBegin(DMPLEX_Preallocate,dm,0,0,0));
  /* Create dof SF based on point SF */
  if (debug) {
    PetscSection section, sectionGlobal;
    PetscSF      sf;

    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(DMGetLocalSection(dm, &section));
    PetscCall(DMGetGlobalSection(dm, &sectionGlobal));
    PetscCall(PetscPrintf(comm, "Input Section for Preallocation:\n"));
    PetscCall(PetscSectionView(section, NULL));
    PetscCall(PetscPrintf(comm, "Input Global Section for Preallocation:\n"));
    PetscCall(PetscSectionView(sectionGlobal, NULL));
    if (size > 1) {
      PetscCall(PetscPrintf(comm, "Input SF for Preallocation:\n"));
      PetscCall(PetscSFView(sf, NULL));
    }
  }
  PetscCall(PetscSFCreateRemoteOffsets(sf, section, section, &remoteOffsets));
  PetscCall(PetscSFCreateSectionSF(sf, section, remoteOffsets, section, &sfDof));
  PetscCall(PetscFree(remoteOffsets));
  if (debug && size > 1) {
    PetscCall(PetscPrintf(comm, "Dof SF for Preallocation:\n"));
    PetscCall(PetscSFView(sfDof, NULL));
  }
  /* Create allocation vectors from adjacency graph */
  PetscCall(MatGetLocalSize(A, &locRows, NULL));
  PetscCall(PetscLayoutCreate(comm, &rLayout));
  PetscCall(PetscLayoutSetLocalSize(rLayout, locRows));
  PetscCall(PetscLayoutSetBlockSize(rLayout, 1));
  PetscCall(PetscLayoutSetUp(rLayout));
  /* There are 4 types of adjacency */
  PetscCall(PetscSectionGetNumFields(section, &Nf));
  if (Nf < 1 || bs > 1) {
    PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
    idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
    PetscCall(DMPlexCreateAdjacencySection_Static(dm, bs, sfDof, useCone, useClosure, PETSC_TRUE, &sectionAdj[idx], &cols[idx]));
    PetscCall(DMPlexUpdateAllocation_Static(dm, rLayout, bs, -1, sectionAdj[idx], cols[idx], dnz, onz, dnzu, onzu));
  } else {
    for (f = 0; f < Nf; ++f) {
      PetscCall(DMGetAdjacency(dm, f, &useCone, &useClosure));
      idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
      if (!sectionAdj[idx]) PetscCall(DMPlexCreateAdjacencySection_Static(dm, bs, sfDof, useCone, useClosure, PETSC_TRUE, &sectionAdj[idx], &cols[idx]));
      PetscCall(DMPlexUpdateAllocation_Static(dm, rLayout, bs, f, sectionAdj[idx], cols[idx], dnz, onz, dnzu, onzu));
    }
  }
  PetscCall(PetscSFDestroy(&sfDof));
  /* Set matrix pattern */
  PetscCall(MatXAIJSetPreallocation(A, bs, dnz, onz, dnzu, onzu));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  /* Check for symmetric storage */
  PetscCall(MatGetType(A, &mtype));
  PetscCall(PetscStrcmp(mtype, MATSBAIJ, &isSymBlock));
  PetscCall(PetscStrcmp(mtype, MATSEQSBAIJ, &isSymSeqBlock));
  PetscCall(PetscStrcmp(mtype, MATMPISBAIJ, &isSymMPIBlock));
  if (isSymBlock || isSymSeqBlock || isSymMPIBlock) PetscCall(MatSetOption(A, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
  /* Fill matrix with zeros */
  if (fillMatrix) {
    if (Nf < 1 || bs > 1) {
      PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
      idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
      PetscCall(DMPlexFillMatrix_Static(dm, rLayout, bs, -1, sectionAdj[idx], cols[idx], A));
    } else {
      for (f = 0; f < Nf; ++f) {
        PetscCall(DMGetAdjacency(dm, f, &useCone, &useClosure));
        idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
        PetscCall(DMPlexFillMatrix_Static(dm, rLayout, bs, f, sectionAdj[idx], cols[idx], A));
      }
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscCall(PetscLayoutDestroy(&rLayout));
  for (idx = 0; idx < 4; ++idx) {PetscCall(PetscSectionDestroy(&sectionAdj[idx])); PetscCall(PetscFree(cols[idx]));}
  PetscCall(PetscLogEventEnd(DMPLEX_Preallocate,dm,0,0,0));
  PetscFunctionReturn(0);
}

#if 0
PetscErrorCode DMPlexPreallocateOperator_2(DM dm, PetscInt bs, PetscSection section, PetscSection sectionGlobal, PetscInt dnz[], PetscInt onz[], PetscInt dnzu[], PetscInt onzu[], Mat A, PetscBool fillMatrix)
{
  PetscInt       *tmpClosure,*tmpAdj,*visits;
  PetscInt        c,cStart,cEnd,pStart,pEnd;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));

  maxClosureSize = 2*PetscMax(PetscPowInt(mesh->maxConeSize,depth+1),PetscPowInt(mesh->maxSupportSize,depth+1));

  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  npoints = pEnd - pStart;

  PetscCall(PetscMalloc3(maxClosureSize,&tmpClosure,npoints,&lvisits,npoints,&visits));
  PetscCall(PetscArrayzero(lvisits,pEnd-pStart));
  PetscCall(PetscArrayzero(visits,pEnd-pStart));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c=cStart; c<cEnd; c++) {
    PetscInt *support = tmpClosure;
    PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_FALSE, &supportSize, (PetscInt**)&support));
    for (p=0; p<supportSize; p++) lvisits[support[p]]++;
  }
  PetscCall(PetscSFReduceBegin(sf,MPIU_INT,lvisits,visits,MPI_SUM));
  PetscCall(PetscSFReduceEnd  (sf,MPIU_INT,lvisits,visits,MPI_SUM));
  PetscCall(PetscSFBcastBegin(sf,MPIU_INT,visits,lvisits,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd  (sf,MPIU_INT,visits,lvisits));

  PetscCall(PetscSFGetRootRanks());

  PetscCall(PetscMalloc2(maxClosureSize*maxClosureSize,&cellmat,npoints,&owner));
  for (c=cStart; c<cEnd; c++) {
    PetscCall(PetscArrayzero(cellmat,maxClosureSize*maxClosureSize));
    /*
     Depth-first walk of transitive closure.
     At each leaf frame f of transitive closure that we see, add 1/visits[f] to each pair (p,q) not marked as done in cellmat.
     This contribution is added to dnz if owning ranks of p and q match, to onz otherwise.
     */
  }

  PetscCall(PetscSFReduceBegin(sf,MPIU_INT,ldnz,dnz,MPI_SUM));
  PetscCall(PetscSFReduceEnd  (sf,MPIU_INT,lonz,onz,MPI_SUM));
  PetscFunctionReturn(0);
}
#endif
