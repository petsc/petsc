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
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetGlobalSection(dm, &sectionGlobal));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) section), &adjSec));
  CHKERRQ(PetscSectionGetChart(section,&pStart,&pEnd));
  CHKERRQ(PetscSectionSetChart(adjSec,pStart,pEnd));

  CHKERRQ(DMPlexGetAnchors(dm,&aSec,&aIS));
  if (aSec) {
    const PetscInt *anchors;
    PetscInt       p, q, a, aSize, *offsets, aStart, aEnd, *inverse, iSize, *adj, adjSize;
    PetscInt       *tmpAdjP = NULL, *tmpAdjQ = NULL;
    PetscSection   inverseSec;

    /* invert the constraint-to-anchor map */
    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)aSec),&inverseSec));
    CHKERRQ(PetscSectionSetChart(inverseSec,pStart,pEnd));
    CHKERRQ(ISGetLocalSize(aIS, &aSize));
    CHKERRQ(ISGetIndices(aIS, &anchors));

    for (p = 0; p < aSize; p++) {
      PetscInt a = anchors[p];

      CHKERRQ(PetscSectionAddDof(inverseSec,a,1));
    }
    CHKERRQ(PetscSectionSetUp(inverseSec));
    CHKERRQ(PetscSectionGetStorageSize(inverseSec,&iSize));
    CHKERRQ(PetscMalloc1(iSize,&inverse));
    CHKERRQ(PetscCalloc1(pEnd-pStart,&offsets));
    CHKERRQ(PetscSectionGetChart(aSec,&aStart,&aEnd));
    for (p = aStart; p < aEnd; p++) {
      PetscInt dof, off;

      CHKERRQ(PetscSectionGetDof(aSec, p, &dof));
      CHKERRQ(PetscSectionGetOffset(aSec, p, &off));

      for (q = 0; q < dof; q++) {
        PetscInt iOff;

        a = anchors[off + q];
        CHKERRQ(PetscSectionGetOffset(inverseSec, a, &iOff));
        inverse[iOff + offsets[a-pStart]++] = p;
      }
    }
    CHKERRQ(ISRestoreIndices(aIS, &anchors));
    CHKERRQ(PetscFree(offsets));

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

      CHKERRQ(PetscSectionGetDof(inverseSec,p,&iDof));
      if (!iDof) continue;
      CHKERRQ(PetscSectionGetOffset(inverseSec,p,&iOff));
      CHKERRQ(DMPlexGetAdjacency_Internal(dm,p,useCone,useClosure,PETSC_TRUE,&numAdjP,&tmpAdjP));
      for (i = 0; i < iDof; i++) {
        PetscInt iNew = 0, qAdj, qAdjDof, qAdjCDof, numAdjQ = PETSC_DETERMINE;

        q = inverse[iOff + i];
        CHKERRQ(DMPlexGetAdjacency_Internal(dm,q,useCone,useClosure,PETSC_TRUE,&numAdjQ,&tmpAdjQ));
        for (r = 0; r < numAdjQ; r++) {
          qAdj = tmpAdjQ[r];
          if ((qAdj < pStart) || (qAdj >= pEnd)) continue;
          for (s = 0; s < numAdjP; s++) {
            if (qAdj == tmpAdjP[s]) break;
          }
          if (s < numAdjP) continue;
          CHKERRQ(PetscSectionGetDof(section,qAdj,&qAdjDof));
          CHKERRQ(PetscSectionGetConstraintDof(section,qAdj,&qAdjCDof));
          iNew += qAdjDof - qAdjCDof;
        }
        CHKERRQ(PetscSectionAddDof(adjSec,p,iNew));
      }
    }

    CHKERRQ(PetscSectionSetUp(adjSec));
    CHKERRQ(PetscSectionGetStorageSize(adjSec,&adjSize));
    CHKERRQ(PetscMalloc1(adjSize,&adj));

    for (p = pStart; p < pEnd; p++) {
      PetscInt iDof, iOff, i, r, s, aOff, aOffOrig, aDof, numAdjP = PETSC_DETERMINE;

      CHKERRQ(PetscSectionGetDof(inverseSec,p,&iDof));
      if (!iDof) continue;
      CHKERRQ(PetscSectionGetOffset(inverseSec,p,&iOff));
      CHKERRQ(DMPlexGetAdjacency_Internal(dm,p,useCone,useClosure,PETSC_TRUE,&numAdjP,&tmpAdjP));
      CHKERRQ(PetscSectionGetDof(adjSec,p,&aDof));
      CHKERRQ(PetscSectionGetOffset(adjSec,p,&aOff));
      aOffOrig = aOff;
      for (i = 0; i < iDof; i++) {
        PetscInt qAdj, qAdjDof, qAdjCDof, qAdjOff, nd, numAdjQ = PETSC_DETERMINE;

        q = inverse[iOff + i];
        CHKERRQ(DMPlexGetAdjacency_Internal(dm,q,useCone,useClosure,PETSC_TRUE,&numAdjQ,&tmpAdjQ));
        for (r = 0; r < numAdjQ; r++) {
          qAdj = tmpAdjQ[r];
          if ((qAdj < pStart) || (qAdj >= pEnd)) continue;
          for (s = 0; s < numAdjP; s++) {
            if (qAdj == tmpAdjP[s]) break;
          }
          if (s < numAdjP) continue;
          CHKERRQ(PetscSectionGetDof(section,qAdj,&qAdjDof));
          CHKERRQ(PetscSectionGetConstraintDof(section,qAdj,&qAdjCDof));
          CHKERRQ(PetscSectionGetOffset(sectionGlobal,qAdj,&qAdjOff));
          for (nd = 0; nd < qAdjDof-qAdjCDof; ++nd) {
            adj[aOff++] = (qAdjOff < 0 ? -(qAdjOff+1) : qAdjOff) + nd;
          }
        }
      }
      CHKERRQ(PetscSortRemoveDupsInt(&aDof,&adj[aOffOrig]));
      CHKERRQ(PetscSectionSetDof(adjSec,p,aDof));
    }
    *anchorAdj = adj;

    /* clean up */
    CHKERRQ(PetscSectionDestroy(&inverseSec));
    CHKERRQ(PetscFree(inverse));
    CHKERRQ(PetscFree(tmpAdjP));
    CHKERRQ(PetscFree(tmpAdjQ));
  }
  else {
    *anchorAdj = NULL;
    CHKERRQ(PetscSectionSetUp(adjSec));
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
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-dm_view_preallocation", &debug, NULL));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(DMGetGlobalSection(dm, &sectionGlobal));
  CHKERRQ(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL));
  doCommLocal = (size > 1) && (nroots >= 0) ? PETSC_TRUE : PETSC_FALSE;
  CHKERRMPI(MPIU_Allreduce(&doCommLocal, &doComm, 1, MPIU_BOOL, MPI_LAND, comm));
  /* Create section for dof adjacency (dof ==> # adj dof) */
  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  CHKERRQ(PetscSectionGetStorageSize(section, &numDof));
  CHKERRQ(PetscSectionCreate(comm, &leafSectionAdj));
  CHKERRQ(PetscSectionSetChart(leafSectionAdj, 0, numDof));
  CHKERRQ(PetscSectionCreate(comm, &rootSectionAdj));
  CHKERRQ(PetscSectionSetChart(rootSectionAdj, 0, numDof));
  /*   Fill in the ghost dofs on the interface */
  CHKERRQ(PetscSFGetGraph(sf, NULL, &nleaves, &leaves, &remotes));
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
  CHKERRQ(DMPlexComputeAnchorAdjacencies(dm, useCone, useClosure, &anchorSectionAdj, &anchorAdj));
  for (l = 0; l < nleaves; ++l) {
    PetscInt dof, off, d, q, anDof;
    PetscInt p = leaves[l], numAdj = PETSC_DETERMINE;

    if ((p < pStart) || (p >= pEnd)) continue;
    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    CHKERRQ(PetscSectionGetOffset(section, p, &off));
    CHKERRQ(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    for (q = 0; q < numAdj; ++q) {
      const PetscInt padj = tmpAdj[q];
      PetscInt ndof, ncdof;

      if ((padj < pStart) || (padj >= pEnd)) continue;
      CHKERRQ(PetscSectionGetDof(section, padj, &ndof));
      CHKERRQ(PetscSectionGetConstraintDof(section, padj, &ncdof));
      for (d = off; d < off+dof; ++d) {
        CHKERRQ(PetscSectionAddDof(leafSectionAdj, d, ndof-ncdof));
      }
    }
    CHKERRQ(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    if (anDof) {
      for (d = off; d < off+dof; ++d) {
        CHKERRQ(PetscSectionAddDof(leafSectionAdj, d, anDof));
      }
    }
  }
  CHKERRQ(PetscSectionSetUp(leafSectionAdj));
  if (debug) {
    CHKERRQ(PetscPrintf(comm, "Adjacency Section for Preallocation on Leaves:\n"));
    CHKERRQ(PetscSectionView(leafSectionAdj, NULL));
  }
  /* Get maximum remote adjacency sizes for owned dofs on interface (roots) */
  if (doComm) {
    CHKERRQ(PetscSFReduceBegin(sfDof, MPIU_INT, leafSectionAdj->atlasDof, rootSectionAdj->atlasDof, MPI_SUM));
    CHKERRQ(PetscSFReduceEnd(sfDof, MPIU_INT, leafSectionAdj->atlasDof, rootSectionAdj->atlasDof, MPI_SUM));
  }
  if (debug) {
    CHKERRQ(PetscPrintf(comm, "Adjancency Section for Preallocation on Roots:\n"));
    CHKERRQ(PetscSectionView(rootSectionAdj, NULL));
  }
  /* Add in local adjacency sizes for owned dofs on interface (roots) */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt numAdj = PETSC_DETERMINE, adof, dof, off, d, q, anDof;

    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    CHKERRQ(PetscSectionGetOffset(section, p, &off));
    if (!dof) continue;
    CHKERRQ(PetscSectionGetDof(rootSectionAdj, off, &adof));
    if (adof <= 0) continue;
    CHKERRQ(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    for (q = 0; q < numAdj; ++q) {
      const PetscInt padj = tmpAdj[q];
      PetscInt ndof, ncdof;

      if ((padj < pStart) || (padj >= pEnd)) continue;
      CHKERRQ(PetscSectionGetDof(section, padj, &ndof));
      CHKERRQ(PetscSectionGetConstraintDof(section, padj, &ncdof));
      for (d = off; d < off+dof; ++d) {
        CHKERRQ(PetscSectionAddDof(rootSectionAdj, d, ndof-ncdof));
      }
    }
    CHKERRQ(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    if (anDof) {
      for (d = off; d < off+dof; ++d) {
        CHKERRQ(PetscSectionAddDof(rootSectionAdj, d, anDof));
      }
    }
  }
  CHKERRQ(PetscSectionSetUp(rootSectionAdj));
  if (debug) {
    CHKERRQ(PetscPrintf(comm, "Adjancency Section for Preallocation on Roots after local additions:\n"));
    CHKERRQ(PetscSectionView(rootSectionAdj, NULL));
  }
  /* Create adj SF based on dof SF */
  CHKERRQ(PetscSFCreateRemoteOffsets(sfDof, rootSectionAdj, leafSectionAdj, &remoteOffsets));
  CHKERRQ(PetscSFCreateSectionSF(sfDof, rootSectionAdj, remoteOffsets, leafSectionAdj, &sfAdj));
  CHKERRQ(PetscFree(remoteOffsets));
  if (debug && size > 1) {
    CHKERRQ(PetscPrintf(comm, "Adjacency SF for Preallocation:\n"));
    CHKERRQ(PetscSFView(sfAdj, NULL));
  }
  /* Create leaf adjacency */
  CHKERRQ(PetscSectionSetUp(leafSectionAdj));
  CHKERRQ(PetscSectionGetStorageSize(leafSectionAdj, &adjSize));
  CHKERRQ(PetscCalloc1(adjSize, &adj));
  for (l = 0; l < nleaves; ++l) {
    PetscInt dof, off, d, q, anDof, anOff;
    PetscInt p = leaves[l], numAdj = PETSC_DETERMINE;

    if ((p < pStart) || (p >= pEnd)) continue;
    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    CHKERRQ(PetscSectionGetOffset(section, p, &off));
    CHKERRQ(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    CHKERRQ(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    CHKERRQ(PetscSectionGetOffset(anchorSectionAdj, p, &anOff));
    for (d = off; d < off+dof; ++d) {
      PetscInt aoff, i = 0;

      CHKERRQ(PetscSectionGetOffset(leafSectionAdj, d, &aoff));
      for (q = 0; q < numAdj; ++q) {
        const PetscInt padj = tmpAdj[q];
        PetscInt ndof, ncdof, ngoff, nd;

        if ((padj < pStart) || (padj >= pEnd)) continue;
        CHKERRQ(PetscSectionGetDof(section, padj, &ndof));
        CHKERRQ(PetscSectionGetConstraintDof(section, padj, &ncdof));
        CHKERRQ(PetscSectionGetOffset(sectionGlobal, padj, &ngoff));
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
    CHKERRQ(PetscPrintf(comm, "Leaf adjacency indices\n"));
    CHKERRQ(ISCreateGeneral(comm, adjSize, adj, PETSC_USE_POINTER, &tmp));
    CHKERRQ(ISView(tmp, NULL));
    CHKERRQ(ISDestroy(&tmp));
  }
  /* Gather adjacent indices to root */
  CHKERRQ(PetscSectionGetStorageSize(rootSectionAdj, &adjSize));
  CHKERRQ(PetscMalloc1(adjSize, &rootAdj));
  for (r = 0; r < adjSize; ++r) rootAdj[r] = -1;
  if (doComm) {
    const PetscInt *indegree;
    PetscInt       *remoteadj, radjsize = 0;

    CHKERRQ(PetscSFComputeDegreeBegin(sfAdj, &indegree));
    CHKERRQ(PetscSFComputeDegreeEnd(sfAdj, &indegree));
    for (p = 0; p < adjSize; ++p) radjsize += indegree[p];
    CHKERRQ(PetscMalloc1(radjsize, &remoteadj));
    CHKERRQ(PetscSFGatherBegin(sfAdj, MPIU_INT, adj, remoteadj));
    CHKERRQ(PetscSFGatherEnd(sfAdj, MPIU_INT, adj, remoteadj));
    for (p = 0, l = 0, r = 0; p < adjSize; ++p, l = PetscMax(p, l + indegree[p-1])) {
      PetscInt s;
      for (s = 0; s < indegree[p]; ++s, ++r) rootAdj[l+s] = remoteadj[r];
    }
    PetscCheckFalse(r != radjsize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistency in communication %d != %d", r, radjsize);
    PetscCheckFalse(l != adjSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistency in communication %d != %d", l, adjSize);
    CHKERRQ(PetscFree(remoteadj));
  }
  CHKERRQ(PetscSFDestroy(&sfAdj));
  CHKERRQ(PetscFree(adj));
  /* Debugging */
  if (debug) {
    IS tmp;
    CHKERRQ(PetscPrintf(comm, "Root adjacency indices after gather\n"));
    CHKERRQ(ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp));
    CHKERRQ(ISView(tmp, NULL));
    CHKERRQ(ISDestroy(&tmp));
  }
  /* Add in local adjacency indices for owned dofs on interface (roots) */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt numAdj = PETSC_DETERMINE, adof, dof, off, d, q, anDof, anOff;

    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    CHKERRQ(PetscSectionGetOffset(section, p, &off));
    if (!dof) continue;
    CHKERRQ(PetscSectionGetDof(rootSectionAdj, off, &adof));
    if (adof <= 0) continue;
    CHKERRQ(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    CHKERRQ(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    CHKERRQ(PetscSectionGetOffset(anchorSectionAdj, p, &anOff));
    for (d = off; d < off+dof; ++d) {
      PetscInt adof, aoff, i;

      CHKERRQ(PetscSectionGetDof(rootSectionAdj, d, &adof));
      CHKERRQ(PetscSectionGetOffset(rootSectionAdj, d, &aoff));
      i    = adof-1;
      for (q = 0; q < anDof; q++) {
        rootAdj[aoff+i] = anchorAdj[anOff+q];
        --i;
      }
      for (q = 0; q < numAdj; ++q) {
        const PetscInt padj = tmpAdj[q];
        PetscInt ndof, ncdof, ngoff, nd;

        if ((padj < pStart) || (padj >= pEnd)) continue;
        CHKERRQ(PetscSectionGetDof(section, padj, &ndof));
        CHKERRQ(PetscSectionGetConstraintDof(section, padj, &ncdof));
        CHKERRQ(PetscSectionGetOffset(sectionGlobal, padj, &ngoff));
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
    CHKERRQ(PetscPrintf(comm, "Root adjacency indices\n"));
    CHKERRQ(ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp));
    CHKERRQ(ISView(tmp, NULL));
    CHKERRQ(ISDestroy(&tmp));
  }
  /* Compress indices */
  CHKERRQ(PetscSectionSetUp(rootSectionAdj));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, off, d;
    PetscInt adof, aoff;

    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    CHKERRQ(PetscSectionGetConstraintDof(section, p, &cdof));
    CHKERRQ(PetscSectionGetOffset(section, p, &off));
    if (!dof) continue;
    CHKERRQ(PetscSectionGetDof(rootSectionAdj, off, &adof));
    if (adof <= 0) continue;
    for (d = off; d < off+dof-cdof; ++d) {
      CHKERRQ(PetscSectionGetDof(rootSectionAdj, d, &adof));
      CHKERRQ(PetscSectionGetOffset(rootSectionAdj, d, &aoff));
      CHKERRQ(PetscSortRemoveDupsInt(&adof, &rootAdj[aoff]));
      CHKERRQ(PetscSectionSetDof(rootSectionAdj, d, adof));
    }
  }
  /* Debugging */
  if (debug) {
    IS tmp;
    CHKERRQ(PetscPrintf(comm, "Adjancency Section for Preallocation on Roots after compression:\n"));
    CHKERRQ(PetscSectionView(rootSectionAdj, NULL));
    CHKERRQ(PetscPrintf(comm, "Root adjacency indices after compression\n"));
    CHKERRQ(ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp));
    CHKERRQ(ISView(tmp, NULL));
    CHKERRQ(ISDestroy(&tmp));
  }
  /* Build adjacency section: Maps global indices to sets of adjacent global indices */
  CHKERRQ(PetscSectionGetOffsetRange(sectionGlobal, &globalOffStart, &globalOffEnd));
  CHKERRQ(PetscSectionCreate(comm, &sectionAdj));
  CHKERRQ(PetscSectionSetChart(sectionAdj, globalOffStart, globalOffEnd));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  numAdj = PETSC_DETERMINE, dof, cdof, off, goff, d, q, anDof;
    PetscBool found  = PETSC_TRUE;

    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    CHKERRQ(PetscSectionGetConstraintDof(section, p, &cdof));
    CHKERRQ(PetscSectionGetOffset(section, p, &off));
    CHKERRQ(PetscSectionGetOffset(sectionGlobal, p, &goff));
    for (d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      CHKERRQ(PetscSectionGetDof(leafSectionAdj, off+d, &ldof));
      CHKERRQ(PetscSectionGetDof(rootSectionAdj, off+d, &rdof));
      if (ldof > 0) {
        /* We do not own this point */
      } else if (rdof > 0) {
        CHKERRQ(PetscSectionSetDof(sectionAdj, goff+d, rdof));
      } else {
        found = PETSC_FALSE;
      }
    }
    if (found) continue;
    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    CHKERRQ(PetscSectionGetOffset(sectionGlobal, p, &goff));
    CHKERRQ(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    for (q = 0; q < numAdj; ++q) {
      const PetscInt padj = tmpAdj[q];
      PetscInt ndof, ncdof, noff;

      if ((padj < pStart) || (padj >= pEnd)) continue;
      CHKERRQ(PetscSectionGetDof(section, padj, &ndof));
      CHKERRQ(PetscSectionGetConstraintDof(section, padj, &ncdof));
      CHKERRQ(PetscSectionGetOffset(section, padj, &noff));
      for (d = goff; d < goff+dof-cdof; ++d) {
        CHKERRQ(PetscSectionAddDof(sectionAdj, d, ndof-ncdof));
      }
    }
    CHKERRQ(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    if (anDof) {
      for (d = goff; d < goff+dof-cdof; ++d) {
        CHKERRQ(PetscSectionAddDof(sectionAdj, d, anDof));
      }
    }
  }
  CHKERRQ(PetscSectionSetUp(sectionAdj));
  if (debug) {
    CHKERRQ(PetscPrintf(comm, "Adjacency Section for Preallocation:\n"));
    CHKERRQ(PetscSectionView(sectionAdj, NULL));
  }
  /* Get adjacent indices */
  CHKERRQ(PetscSectionGetStorageSize(sectionAdj, &numCols));
  CHKERRQ(PetscMalloc1(numCols, &cols));
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  numAdj = PETSC_DETERMINE, dof, cdof, off, goff, d, q, anDof, anOff;
    PetscBool found  = PETSC_TRUE;

    CHKERRQ(PetscSectionGetDof(section, p, &dof));
    CHKERRQ(PetscSectionGetConstraintDof(section, p, &cdof));
    CHKERRQ(PetscSectionGetOffset(section, p, &off));
    CHKERRQ(PetscSectionGetOffset(sectionGlobal, p, &goff));
    for (d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      CHKERRQ(PetscSectionGetDof(leafSectionAdj, off+d, &ldof));
      CHKERRQ(PetscSectionGetDof(rootSectionAdj, off+d, &rdof));
      if (ldof > 0) {
        /* We do not own this point */
      } else if (rdof > 0) {
        PetscInt aoff, roff;

        CHKERRQ(PetscSectionGetOffset(sectionAdj, goff+d, &aoff));
        CHKERRQ(PetscSectionGetOffset(rootSectionAdj, off+d, &roff));
        CHKERRQ(PetscArraycpy(&cols[aoff], &rootAdj[roff], rdof));
      } else {
        found = PETSC_FALSE;
      }
    }
    if (found) continue;
    CHKERRQ(DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj));
    CHKERRQ(PetscSectionGetDof(anchorSectionAdj, p, &anDof));
    CHKERRQ(PetscSectionGetOffset(anchorSectionAdj, p, &anOff));
    for (d = goff; d < goff+dof-cdof; ++d) {
      PetscInt adof, aoff, i = 0;

      CHKERRQ(PetscSectionGetDof(sectionAdj, d, &adof));
      CHKERRQ(PetscSectionGetOffset(sectionAdj, d, &aoff));
      for (q = 0; q < numAdj; ++q) {
        const PetscInt  padj = tmpAdj[q];
        PetscInt        ndof, ncdof, ngoff, nd;
        const PetscInt *ncind;

        /* Adjacent points may not be in the section chart */
        if ((padj < pStart) || (padj >= pEnd)) continue;
        CHKERRQ(PetscSectionGetDof(section, padj, &ndof));
        CHKERRQ(PetscSectionGetConstraintDof(section, padj, &ncdof));
        CHKERRQ(PetscSectionGetConstraintIndices(section, padj, &ncind));
        CHKERRQ(PetscSectionGetOffset(sectionGlobal, padj, &ngoff));
        for (nd = 0; nd < ndof-ncdof; ++nd, ++i) {
          cols[aoff+i] = ngoff < 0 ? -(ngoff+1)+nd : ngoff+nd;
        }
      }
      for (q = 0; q < anDof; q++, i++) {
        cols[aoff+i] = anchorAdj[anOff + q];
      }
      PetscCheckFalse(i != adof,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of entries %D != %D for dof %D (point %D)", i, adof, d, p);
    }
  }
  CHKERRQ(PetscSectionDestroy(&anchorSectionAdj));
  CHKERRQ(PetscSectionDestroy(&leafSectionAdj));
  CHKERRQ(PetscSectionDestroy(&rootSectionAdj));
  CHKERRQ(PetscFree(anchorAdj));
  CHKERRQ(PetscFree(rootAdj));
  CHKERRQ(PetscFree(tmpAdj));
  /* Debugging */
  if (debug) {
    IS tmp;
    CHKERRQ(PetscPrintf(comm, "Column indices\n"));
    CHKERRQ(ISCreateGeneral(comm, numCols, cols, PETSC_USE_POINTER, &tmp));
    CHKERRQ(ISView(tmp, NULL));
    CHKERRQ(ISDestroy(&tmp));
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
  CHKERRQ(PetscLayoutGetRange(rLayout, &rStart, &rEnd));
  PetscCheckFalse(rStart%bs || rEnd%bs,PetscObjectComm((PetscObject) rLayout), PETSC_ERR_ARG_WRONG, "Invalid layout [%d, %d) for matrix, must be divisible by block size %d", rStart, rEnd, bs);
  if (f >= 0 && bs == 1) {
    CHKERRQ(DMGetLocalSection(dm, &section));
    CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt rS, rE;

      CHKERRQ(DMGetGlobalFieldOffset_Private(dm, p, f, &rS, &rE));
      for (r = rS; r < rE; ++r) {
        PetscInt numCols, cStart, c;

        CHKERRQ(PetscSectionGetDof(sectionAdj, r, &numCols));
        CHKERRQ(PetscSectionGetOffset(sectionAdj, r, &cStart));
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

      CHKERRQ(PetscSectionGetDof(sectionAdj, row, &numCols));
      CHKERRQ(PetscSectionGetOffset(sectionAdj, row, &cStart));
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
  CHKERRQ(PetscLayoutGetRange(rLayout, &rStart, &rEnd));
  for (r = rStart; r < rEnd; ++r) {
    CHKERRQ(PetscSectionGetDof(sectionAdj, r, &len));
    maxRowLen = PetscMax(maxRowLen, len);
  }
  CHKERRQ(PetscCalloc1(maxRowLen, &values));
  if (f >=0 && bs == 1) {
    CHKERRQ(DMGetLocalSection(dm, &section));
    CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt rS, rE;

      CHKERRQ(DMGetGlobalFieldOffset_Private(dm, p, f, &rS, &rE));
      for (r = rS; r < rE; ++r) {
        PetscInt numCols, cStart;

        CHKERRQ(PetscSectionGetDof(sectionAdj, r, &numCols));
        CHKERRQ(PetscSectionGetOffset(sectionAdj, r, &cStart));
        CHKERRQ(MatSetValues(A, 1, &r, numCols, &cols[cStart], values, INSERT_VALUES));
      }
    }
  } else {
    for (r = rStart; r < rEnd; ++r) {
      PetscInt numCols, cStart;

      CHKERRQ(PetscSectionGetDof(sectionAdj, r, &numCols));
      CHKERRQ(PetscSectionGetOffset(sectionAdj, r, &cStart));
      CHKERRQ(MatSetValues(A, 1, &r, numCols, &cols[cStart], values, INSERT_VALUES));
    }
  }
  CHKERRQ(PetscFree(values));
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
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(DMGetLocalSection(dm, &section));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-dm_view_preallocation", &debug, NULL));
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(PetscLogEventBegin(DMPLEX_Preallocate,dm,0,0,0));
  /* Create dof SF based on point SF */
  if (debug) {
    PetscSection section, sectionGlobal;
    PetscSF      sf;

    CHKERRQ(DMGetPointSF(dm, &sf));
    CHKERRQ(DMGetLocalSection(dm, &section));
    CHKERRQ(DMGetGlobalSection(dm, &sectionGlobal));
    CHKERRQ(PetscPrintf(comm, "Input Section for Preallocation:\n"));
    CHKERRQ(PetscSectionView(section, NULL));
    CHKERRQ(PetscPrintf(comm, "Input Global Section for Preallocation:\n"));
    CHKERRQ(PetscSectionView(sectionGlobal, NULL));
    if (size > 1) {
      CHKERRQ(PetscPrintf(comm, "Input SF for Preallocation:\n"));
      CHKERRQ(PetscSFView(sf, NULL));
    }
  }
  CHKERRQ(PetscSFCreateRemoteOffsets(sf, section, section, &remoteOffsets));
  CHKERRQ(PetscSFCreateSectionSF(sf, section, remoteOffsets, section, &sfDof));
  CHKERRQ(PetscFree(remoteOffsets));
  if (debug && size > 1) {
    CHKERRQ(PetscPrintf(comm, "Dof SF for Preallocation:\n"));
    CHKERRQ(PetscSFView(sfDof, NULL));
  }
  /* Create allocation vectors from adjacency graph */
  CHKERRQ(MatGetLocalSize(A, &locRows, NULL));
  CHKERRQ(PetscLayoutCreate(comm, &rLayout));
  CHKERRQ(PetscLayoutSetLocalSize(rLayout, locRows));
  CHKERRQ(PetscLayoutSetBlockSize(rLayout, 1));
  CHKERRQ(PetscLayoutSetUp(rLayout));
  /* There are 4 types of adjacency */
  CHKERRQ(PetscSectionGetNumFields(section, &Nf));
  if (Nf < 1 || bs > 1) {
    CHKERRQ(DMGetBasicAdjacency(dm, &useCone, &useClosure));
    idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
    CHKERRQ(DMPlexCreateAdjacencySection_Static(dm, bs, sfDof, useCone, useClosure, PETSC_TRUE, &sectionAdj[idx], &cols[idx]));
    CHKERRQ(DMPlexUpdateAllocation_Static(dm, rLayout, bs, -1, sectionAdj[idx], cols[idx], dnz, onz, dnzu, onzu));
  } else {
    for (f = 0; f < Nf; ++f) {
      CHKERRQ(DMGetAdjacency(dm, f, &useCone, &useClosure));
      idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
      if (!sectionAdj[idx]) CHKERRQ(DMPlexCreateAdjacencySection_Static(dm, bs, sfDof, useCone, useClosure, PETSC_TRUE, &sectionAdj[idx], &cols[idx]));
      CHKERRQ(DMPlexUpdateAllocation_Static(dm, rLayout, bs, f, sectionAdj[idx], cols[idx], dnz, onz, dnzu, onzu));
    }
  }
  CHKERRQ(PetscSFDestroy(&sfDof));
  /* Set matrix pattern */
  CHKERRQ(MatXAIJSetPreallocation(A, bs, dnz, onz, dnzu, onzu));
  CHKERRQ(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  /* Check for symmetric storage */
  CHKERRQ(MatGetType(A, &mtype));
  CHKERRQ(PetscStrcmp(mtype, MATSBAIJ, &isSymBlock));
  CHKERRQ(PetscStrcmp(mtype, MATSEQSBAIJ, &isSymSeqBlock));
  CHKERRQ(PetscStrcmp(mtype, MATMPISBAIJ, &isSymMPIBlock));
  if (isSymBlock || isSymSeqBlock || isSymMPIBlock) CHKERRQ(MatSetOption(A, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
  /* Fill matrix with zeros */
  if (fillMatrix) {
    if (Nf < 1 || bs > 1) {
      CHKERRQ(DMGetBasicAdjacency(dm, &useCone, &useClosure));
      idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
      CHKERRQ(DMPlexFillMatrix_Static(dm, rLayout, bs, -1, sectionAdj[idx], cols[idx], A));
    } else {
      for (f = 0; f < Nf; ++f) {
        CHKERRQ(DMGetAdjacency(dm, f, &useCone, &useClosure));
        idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
        CHKERRQ(DMPlexFillMatrix_Static(dm, rLayout, bs, f, sectionAdj[idx], cols[idx], A));
      }
    }
    CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  CHKERRQ(PetscLayoutDestroy(&rLayout));
  for (idx = 0; idx < 4; ++idx) {CHKERRQ(PetscSectionDestroy(&sectionAdj[idx])); CHKERRQ(PetscFree(cols[idx]));}
  CHKERRQ(PetscLogEventEnd(DMPLEX_Preallocate,dm,0,0,0));
  PetscFunctionReturn(0);
}

#if 0
PetscErrorCode DMPlexPreallocateOperator_2(DM dm, PetscInt bs, PetscSection section, PetscSection sectionGlobal, PetscInt dnz[], PetscInt onz[], PetscInt dnzu[], PetscInt onzu[], Mat A, PetscBool fillMatrix)
{
  PetscInt       *tmpClosure,*tmpAdj,*visits;
  PetscInt        c,cStart,cEnd,pStart,pEnd;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize));

  maxClosureSize = 2*PetscMax(PetscPowInt(mesh->maxConeSize,depth+1),PetscPowInt(mesh->maxSupportSize,depth+1));

  CHKERRQ(PetscSectionGetChart(section, &pStart, &pEnd));
  npoints = pEnd - pStart;

  CHKERRQ(PetscMalloc3(maxClosureSize,&tmpClosure,npoints,&lvisits,npoints,&visits));
  CHKERRQ(PetscArrayzero(lvisits,pEnd-pStart));
  CHKERRQ(PetscArrayzero(visits,pEnd-pStart));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c=cStart; c<cEnd; c++) {
    PetscInt *support = tmpClosure;
    CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_FALSE, &supportSize, (PetscInt**)&support));
    for (p=0; p<supportSize; p++) lvisits[support[p]]++;
  }
  CHKERRQ(PetscSFReduceBegin(sf,MPIU_INT,lvisits,visits,MPI_SUM));
  CHKERRQ(PetscSFReduceEnd  (sf,MPIU_INT,lvisits,visits,MPI_SUM));
  CHKERRQ(PetscSFBcastBegin(sf,MPIU_INT,visits,lvisits,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd  (sf,MPIU_INT,visits,lvisits));

  CHKERRQ(PetscSFGetRootRanks());

  CHKERRQ(PetscMalloc2(maxClosureSize*maxClosureSize,&cellmat,npoints,&owner));
  for (c=cStart; c<cEnd; c++) {
    CHKERRQ(PetscArrayzero(cellmat,maxClosureSize*maxClosureSize));
    /*
     Depth-first walk of transitive closure.
     At each leaf frame f of transitive closure that we see, add 1/visits[f] to each pair (p,q) not marked as done in cellmat.
     This contribution is added to dnz if owning ranks of p and q match, to onz otherwise.
     */
  }

  CHKERRQ(PetscSFReduceBegin(sf,MPIU_INT,ldnz,dnz,MPI_SUM));
  CHKERRQ(PetscSFReduceEnd  (sf,MPIU_INT,lonz,onz,MPI_SUM));
  PetscFunctionReturn(0);
}
#endif
