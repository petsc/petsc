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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) section), &adjSec);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(adjSec,pStart,pEnd);CHKERRQ(ierr);

  ierr = DMPlexGetAnchors(dm,&aSec,&aIS);CHKERRQ(ierr);
  if (aSec) {
    const PetscInt *anchors;
    PetscInt       p, q, a, aSize, *offsets, aStart, aEnd, *inverse, iSize, *adj, adjSize;
    PetscInt       *tmpAdjP = NULL, *tmpAdjQ = NULL;
    PetscSection   inverseSec;

    /* invert the constraint-to-anchor map */
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)aSec),&inverseSec);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(inverseSec,pStart,pEnd);CHKERRQ(ierr);
    ierr = ISGetLocalSize(aIS, &aSize);CHKERRQ(ierr);
    ierr = ISGetIndices(aIS, &anchors);CHKERRQ(ierr);

    for (p = 0; p < aSize; p++) {
      PetscInt a = anchors[p];

      ierr = PetscSectionAddDof(inverseSec,a,1);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(inverseSec);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(inverseSec,&iSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(iSize,&inverse);CHKERRQ(ierr);
    ierr = PetscCalloc1(pEnd-pStart,&offsets);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(aSec,&aStart,&aEnd);CHKERRQ(ierr);
    for (p = aStart; p < aEnd; p++) {
      PetscInt dof, off;

      ierr = PetscSectionGetDof(aSec, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(aSec, p, &off);CHKERRQ(ierr);

      for (q = 0; q < dof; q++) {
        PetscInt iOff;

        a = anchors[off + q];
        ierr = PetscSectionGetOffset(inverseSec, a, &iOff);CHKERRQ(ierr);
        inverse[iOff + offsets[a-pStart]++] = p;
      }
    }
    ierr = ISRestoreIndices(aIS, &anchors);CHKERRQ(ierr);
    ierr = PetscFree(offsets);CHKERRQ(ierr);

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

      ierr = PetscSectionGetDof(inverseSec,p,&iDof);CHKERRQ(ierr);
      if (!iDof) continue;
      ierr = PetscSectionGetOffset(inverseSec,p,&iOff);CHKERRQ(ierr);
      ierr = DMPlexGetAdjacency_Internal(dm,p,useCone,useClosure,PETSC_TRUE,&numAdjP,&tmpAdjP);CHKERRQ(ierr);
      for (i = 0; i < iDof; i++) {
        PetscInt iNew = 0, qAdj, qAdjDof, qAdjCDof, numAdjQ = PETSC_DETERMINE;

        q = inverse[iOff + i];
        ierr = DMPlexGetAdjacency_Internal(dm,q,useCone,useClosure,PETSC_TRUE,&numAdjQ,&tmpAdjQ);CHKERRQ(ierr);
        for (r = 0; r < numAdjQ; r++) {
          qAdj = tmpAdjQ[r];
          if ((qAdj < pStart) || (qAdj >= pEnd)) continue;
          for (s = 0; s < numAdjP; s++) {
            if (qAdj == tmpAdjP[s]) break;
          }
          if (s < numAdjP) continue;
          ierr  = PetscSectionGetDof(section,qAdj,&qAdjDof);CHKERRQ(ierr);
          ierr  = PetscSectionGetConstraintDof(section,qAdj,&qAdjCDof);CHKERRQ(ierr);
          iNew += qAdjDof - qAdjCDof;
        }
        ierr = PetscSectionAddDof(adjSec,p,iNew);CHKERRQ(ierr);
      }
    }

    ierr = PetscSectionSetUp(adjSec);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(adjSec,&adjSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(adjSize,&adj);CHKERRQ(ierr);

    for (p = pStart; p < pEnd; p++) {
      PetscInt iDof, iOff, i, r, s, aOff, aOffOrig, aDof, numAdjP = PETSC_DETERMINE;

      ierr = PetscSectionGetDof(inverseSec,p,&iDof);CHKERRQ(ierr);
      if (!iDof) continue;
      ierr = PetscSectionGetOffset(inverseSec,p,&iOff);CHKERRQ(ierr);
      ierr = DMPlexGetAdjacency_Internal(dm,p,useCone,useClosure,PETSC_TRUE,&numAdjP,&tmpAdjP);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(adjSec,p,&aDof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(adjSec,p,&aOff);CHKERRQ(ierr);
      aOffOrig = aOff;
      for (i = 0; i < iDof; i++) {
        PetscInt qAdj, qAdjDof, qAdjCDof, qAdjOff, nd, numAdjQ = PETSC_DETERMINE;

        q = inverse[iOff + i];
        ierr = DMPlexGetAdjacency_Internal(dm,q,useCone,useClosure,PETSC_TRUE,&numAdjQ,&tmpAdjQ);CHKERRQ(ierr);
        for (r = 0; r < numAdjQ; r++) {
          qAdj = tmpAdjQ[r];
          if ((qAdj < pStart) || (qAdj >= pEnd)) continue;
          for (s = 0; s < numAdjP; s++) {
            if (qAdj == tmpAdjP[s]) break;
          }
          if (s < numAdjP) continue;
          ierr  = PetscSectionGetDof(section,qAdj,&qAdjDof);CHKERRQ(ierr);
          ierr  = PetscSectionGetConstraintDof(section,qAdj,&qAdjCDof);CHKERRQ(ierr);
          ierr  = PetscSectionGetOffset(sectionGlobal,qAdj,&qAdjOff);CHKERRQ(ierr);
          for (nd = 0; nd < qAdjDof-qAdjCDof; ++nd) {
            adj[aOff++] = (qAdjOff < 0 ? -(qAdjOff+1) : qAdjOff) + nd;
          }
        }
      }
      ierr = PetscSortRemoveDupsInt(&aDof,&adj[aOffOrig]);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(adjSec,p,aDof);CHKERRQ(ierr);
    }
    *anchorAdj = adj;

    /* clean up */
    ierr = PetscSectionDestroy(&inverseSec);CHKERRQ(ierr);
    ierr = PetscFree(inverse);CHKERRQ(ierr);
    ierr = PetscFree(tmpAdjP);CHKERRQ(ierr);
    ierr = PetscFree(tmpAdjQ);CHKERRQ(ierr);
  }
  else {
    *anchorAdj = NULL;
    ierr = PetscSectionSetUp(adjSec);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL, "-dm_view_preallocation", &debug, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  doCommLocal = (size > 1) && (nroots >= 0) ? PETSC_TRUE : PETSC_FALSE;
  ierr = MPIU_Allreduce(&doCommLocal, &doComm, 1, MPIU_BOOL, MPI_LAND, comm);CHKERRQ(ierr);
  /* Create section for dof adjacency (dof ==> # adj dof) */
  ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &numDof);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &leafSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(leafSectionAdj, 0, numDof);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &rootSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(rootSectionAdj, 0, numDof);CHKERRQ(ierr);
  /*   Fill in the ghost dofs on the interface */
  ierr = PetscSFGetGraph(sf, NULL, &nleaves, &leaves, &remotes);CHKERRQ(ierr);
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
  ierr = DMPlexComputeAnchorAdjacencies(dm, useCone, useClosure, &anchorSectionAdj, &anchorAdj);CHKERRQ(ierr);
  for (l = 0; l < nleaves; ++l) {
    PetscInt dof, off, d, q, anDof;
    PetscInt p = leaves[l], numAdj = PETSC_DETERMINE;

    if ((p < pStart) || (p >= pEnd)) continue;
    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj);CHKERRQ(ierr);
    for (q = 0; q < numAdj; ++q) {
      const PetscInt padj = tmpAdj[q];
      PetscInt ndof, ncdof;

      if ((padj < pStart) || (padj >= pEnd)) continue;
      ierr = PetscSectionGetDof(section, padj, &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, padj, &ncdof);CHKERRQ(ierr);
      for (d = off; d < off+dof; ++d) {
        ierr = PetscSectionAddDof(leafSectionAdj, d, ndof-ncdof);CHKERRQ(ierr);
      }
    }
    ierr = PetscSectionGetDof(anchorSectionAdj, p, &anDof);CHKERRQ(ierr);
    if (anDof) {
      for (d = off; d < off+dof; ++d) {
        ierr = PetscSectionAddDof(leafSectionAdj, d, anDof);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(leafSectionAdj);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Adjacency Section for Preallocation on Leaves:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(leafSectionAdj, NULL);CHKERRQ(ierr);
  }
  /* Get maximum remote adjacency sizes for owned dofs on interface (roots) */
  if (doComm) {
    ierr = PetscSFReduceBegin(sfDof, MPIU_INT, leafSectionAdj->atlasDof, rootSectionAdj->atlasDof, MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sfDof, MPIU_INT, leafSectionAdj->atlasDof, rootSectionAdj->atlasDof, MPI_SUM);CHKERRQ(ierr);
  }
  if (debug) {
    ierr = PetscPrintf(comm, "Adjancency Section for Preallocation on Roots:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(rootSectionAdj, NULL);CHKERRQ(ierr);
  }
  /* Add in local adjacency sizes for owned dofs on interface (roots) */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt numAdj = PETSC_DETERMINE, adof, dof, off, d, q, anDof;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    if (!dof) continue;
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    ierr = DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj);CHKERRQ(ierr);
    for (q = 0; q < numAdj; ++q) {
      const PetscInt padj = tmpAdj[q];
      PetscInt ndof, ncdof;

      if ((padj < pStart) || (padj >= pEnd)) continue;
      ierr = PetscSectionGetDof(section, padj, &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, padj, &ncdof);CHKERRQ(ierr);
      for (d = off; d < off+dof; ++d) {
        ierr = PetscSectionAddDof(rootSectionAdj, d, ndof-ncdof);CHKERRQ(ierr);
      }
    }
    ierr = PetscSectionGetDof(anchorSectionAdj, p, &anDof);CHKERRQ(ierr);
    if (anDof) {
      for (d = off; d < off+dof; ++d) {
        ierr = PetscSectionAddDof(rootSectionAdj, d, anDof);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(rootSectionAdj);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Adjancency Section for Preallocation on Roots after local additions:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(rootSectionAdj, NULL);CHKERRQ(ierr);
  }
  /* Create adj SF based on dof SF */
  ierr = PetscSFCreateRemoteOffsets(sfDof, rootSectionAdj, leafSectionAdj, &remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(sfDof, rootSectionAdj, remoteOffsets, leafSectionAdj, &sfAdj);CHKERRQ(ierr);
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  if (debug && size > 1) {
    ierr = PetscPrintf(comm, "Adjacency SF for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSFView(sfAdj, NULL);CHKERRQ(ierr);
  }
  /* Create leaf adjacency */
  ierr = PetscSectionSetUp(leafSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(leafSectionAdj, &adjSize);CHKERRQ(ierr);
  ierr = PetscCalloc1(adjSize, &adj);CHKERRQ(ierr);
  for (l = 0; l < nleaves; ++l) {
    PetscInt dof, off, d, q, anDof, anOff;
    PetscInt p = leaves[l], numAdj = PETSC_DETERMINE;

    if ((p < pStart) || (p >= pEnd)) continue;
    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(anchorSectionAdj, p, &anDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(anchorSectionAdj, p, &anOff);CHKERRQ(ierr);
    for (d = off; d < off+dof; ++d) {
      PetscInt aoff, i = 0;

      ierr = PetscSectionGetOffset(leafSectionAdj, d, &aoff);CHKERRQ(ierr);
      for (q = 0; q < numAdj; ++q) {
        const PetscInt padj = tmpAdj[q];
        PetscInt ndof, ncdof, ngoff, nd;

        if ((padj < pStart) || (padj >= pEnd)) continue;
        ierr = PetscSectionGetDof(section, padj, &ndof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, padj, &ncdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, padj, &ngoff);CHKERRQ(ierr);
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
    ierr = PetscPrintf(comm, "Leaf adjacency indices\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, adjSize, adj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, NULL);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
  }
  /* Gather adjacent indices to root */
  ierr = PetscSectionGetStorageSize(rootSectionAdj, &adjSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(adjSize, &rootAdj);CHKERRQ(ierr);
  for (r = 0; r < adjSize; ++r) rootAdj[r] = -1;
  if (doComm) {
    const PetscInt *indegree;
    PetscInt       *remoteadj, radjsize = 0;

    ierr = PetscSFComputeDegreeBegin(sfAdj, &indegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sfAdj, &indegree);CHKERRQ(ierr);
    for (p = 0; p < adjSize; ++p) radjsize += indegree[p];
    ierr = PetscMalloc1(radjsize, &remoteadj);CHKERRQ(ierr);
    ierr = PetscSFGatherBegin(sfAdj, MPIU_INT, adj, remoteadj);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(sfAdj, MPIU_INT, adj, remoteadj);CHKERRQ(ierr);
    for (p = 0, l = 0, r = 0; p < adjSize; ++p, l = PetscMax(p, l + indegree[p-1])) {
      PetscInt s;
      for (s = 0; s < indegree[p]; ++s, ++r) rootAdj[l+s] = remoteadj[r];
    }
    if (r != radjsize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistency in communication %d != %d", r, radjsize);
    if (l != adjSize)  SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistency in communication %d != %d", l, adjSize);
    ierr = PetscFree(remoteadj);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sfAdj);CHKERRQ(ierr);
  ierr = PetscFree(adj);CHKERRQ(ierr);
  /* Debugging */
  if (debug) {
    IS tmp;
    ierr = PetscPrintf(comm, "Root adjacency indices after gather\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, NULL);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
  }
  /* Add in local adjacency indices for owned dofs on interface (roots) */
  for (p = pStart; p < pEnd; ++p) {
    PetscInt numAdj = PETSC_DETERMINE, adof, dof, off, d, q, anDof, anOff;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    if (!dof) continue;
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    ierr = DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(anchorSectionAdj, p, &anDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(anchorSectionAdj, p, &anOff);CHKERRQ(ierr);
    for (d = off; d < off+dof; ++d) {
      PetscInt adof, aoff, i;

      ierr = PetscSectionGetDof(rootSectionAdj, d, &adof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootSectionAdj, d, &aoff);CHKERRQ(ierr);
      i    = adof-1;
      for (q = 0; q < anDof; q++) {
        rootAdj[aoff+i] = anchorAdj[anOff+q];
        --i;
      }
      for (q = 0; q < numAdj; ++q) {
        const PetscInt padj = tmpAdj[q];
        PetscInt ndof, ncdof, ngoff, nd;

        if ((padj < pStart) || (padj >= pEnd)) continue;
        ierr = PetscSectionGetDof(section, padj, &ndof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, padj, &ncdof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, padj, &ngoff);CHKERRQ(ierr);
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
    ierr = PetscPrintf(comm, "Root adjacency indices\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, NULL);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
  }
  /* Compress indices */
  ierr = PetscSectionSetUp(rootSectionAdj);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt dof, cdof, off, d;
    PetscInt adof, aoff;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    if (!dof) continue;
    ierr = PetscSectionGetDof(rootSectionAdj, off, &adof);CHKERRQ(ierr);
    if (adof <= 0) continue;
    for (d = off; d < off+dof-cdof; ++d) {
      ierr = PetscSectionGetDof(rootSectionAdj, d, &adof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootSectionAdj, d, &aoff);CHKERRQ(ierr);
      ierr = PetscSortRemoveDupsInt(&adof, &rootAdj[aoff]);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(rootSectionAdj, d, adof);CHKERRQ(ierr);
    }
  }
  /* Debugging */
  if (debug) {
    IS tmp;
    ierr = PetscPrintf(comm, "Adjancency Section for Preallocation on Roots after compression:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(rootSectionAdj, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Root adjacency indices after compression\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, adjSize, rootAdj, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, NULL);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
  }
  /* Build adjacency section: Maps global indices to sets of adjacent global indices */
  ierr = PetscSectionGetOffsetRange(sectionGlobal, &globalOffStart, &globalOffEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &sectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionAdj, globalOffStart, globalOffEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  numAdj = PETSC_DETERMINE, dof, cdof, off, goff, d, q, anDof;
    PetscBool found  = PETSC_TRUE;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    for (d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      ierr = PetscSectionGetDof(leafSectionAdj, off+d, &ldof);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(rootSectionAdj, off+d, &rdof);CHKERRQ(ierr);
      if (ldof > 0) {
        /* We do not own this point */
      } else if (rdof > 0) {
        ierr = PetscSectionSetDof(sectionAdj, goff+d, rdof);CHKERRQ(ierr);
      } else {
        found = PETSC_FALSE;
      }
    }
    if (found) continue;
    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    ierr = DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj);CHKERRQ(ierr);
    for (q = 0; q < numAdj; ++q) {
      const PetscInt padj = tmpAdj[q];
      PetscInt ndof, ncdof, noff;

      if ((padj < pStart) || (padj >= pEnd)) continue;
      ierr = PetscSectionGetDof(section, padj, &ndof);CHKERRQ(ierr);
      ierr = PetscSectionGetConstraintDof(section, padj, &ncdof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, padj, &noff);CHKERRQ(ierr);
      for (d = goff; d < goff+dof-cdof; ++d) {
        ierr = PetscSectionAddDof(sectionAdj, d, ndof-ncdof);CHKERRQ(ierr);
      }
    }
    ierr = PetscSectionGetDof(anchorSectionAdj, p, &anDof);CHKERRQ(ierr);
    if (anDof) {
      for (d = goff; d < goff+dof-cdof; ++d) {
        ierr = PetscSectionAddDof(sectionAdj, d, anDof);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(sectionAdj);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(comm, "Adjacency Section for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(sectionAdj, NULL);CHKERRQ(ierr);
  }
  /* Get adjacent indices */
  ierr = PetscSectionGetStorageSize(sectionAdj, &numCols);CHKERRQ(ierr);
  ierr = PetscMalloc1(numCols, &cols);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    PetscInt  numAdj = PETSC_DETERMINE, dof, cdof, off, goff, d, q, anDof, anOff;
    PetscBool found  = PETSC_TRUE;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetConstraintDof(section, p, &cdof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sectionGlobal, p, &goff);CHKERRQ(ierr);
    for (d = 0; d < dof-cdof; ++d) {
      PetscInt ldof, rdof;

      ierr = PetscSectionGetDof(leafSectionAdj, off+d, &ldof);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(rootSectionAdj, off+d, &rdof);CHKERRQ(ierr);
      if (ldof > 0) {
        /* We do not own this point */
      } else if (rdof > 0) {
        PetscInt aoff, roff;

        ierr = PetscSectionGetOffset(sectionAdj, goff+d, &aoff);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(rootSectionAdj, off+d, &roff);CHKERRQ(ierr);
        ierr = PetscArraycpy(&cols[aoff], &rootAdj[roff], rdof);CHKERRQ(ierr);
      } else {
        found = PETSC_FALSE;
      }
    }
    if (found) continue;
    ierr = DMPlexGetAdjacency_Internal(dm, p, useCone, useClosure, useAnchors, &numAdj, &tmpAdj);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(anchorSectionAdj, p, &anDof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(anchorSectionAdj, p, &anOff);CHKERRQ(ierr);
    for (d = goff; d < goff+dof-cdof; ++d) {
      PetscInt adof, aoff, i = 0;

      ierr = PetscSectionGetDof(sectionAdj, d, &adof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sectionAdj, d, &aoff);CHKERRQ(ierr);
      for (q = 0; q < numAdj; ++q) {
        const PetscInt  padj = tmpAdj[q];
        PetscInt        ndof, ncdof, ngoff, nd;
        const PetscInt *ncind;

        /* Adjacent points may not be in the section chart */
        if ((padj < pStart) || (padj >= pEnd)) continue;
        ierr = PetscSectionGetDof(section, padj, &ndof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintDof(section, padj, &ncdof);CHKERRQ(ierr);
        ierr = PetscSectionGetConstraintIndices(section, padj, &ncind);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionGlobal, padj, &ngoff);CHKERRQ(ierr);
        for (nd = 0; nd < ndof-ncdof; ++nd, ++i) {
          cols[aoff+i] = ngoff < 0 ? -(ngoff+1)+nd : ngoff+nd;
        }
      }
      for (q = 0; q < anDof; q++, i++) {
        cols[aoff+i] = anchorAdj[anOff + q];
      }
      if (i != adof) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of entries %D != %D for dof %D (point %D)", i, adof, d, p);
    }
  }
  ierr = PetscSectionDestroy(&anchorSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafSectionAdj);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&rootSectionAdj);CHKERRQ(ierr);
  ierr = PetscFree(anchorAdj);CHKERRQ(ierr);
  ierr = PetscFree(rootAdj);CHKERRQ(ierr);
  ierr = PetscFree(tmpAdj);CHKERRQ(ierr);
  /* Debugging */
  if (debug) {
    IS tmp;
    ierr = PetscPrintf(comm, "Column indices\n");CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, numCols, cols, PETSC_USE_POINTER, &tmp);CHKERRQ(ierr);
    ierr = ISView(tmp, NULL);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
  }

  *sA     = sectionAdj;
  *colIdx = cols;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexUpdateAllocation_Static(DM dm, PetscLayout rLayout, PetscInt bs, PetscInt f, PetscSection sectionAdj, const PetscInt cols[], PetscInt dnz[], PetscInt onz[], PetscInt dnzu[], PetscInt onzu[])
{
  PetscSection   section;
  PetscInt       rStart, rEnd, r, pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* This loop needs to change to a loop over points, then field dofs, which means we need to look both sections */
  ierr = PetscLayoutGetRange(rLayout, &rStart, &rEnd);CHKERRQ(ierr);
  if (rStart%bs || rEnd%bs) SETERRQ3(PetscObjectComm((PetscObject) rLayout), PETSC_ERR_ARG_WRONG, "Invalid layout [%d, %d) for matrix, must be divisible by block size %d", rStart, rEnd, bs);
  if (f >= 0 && bs == 1) {
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt rS, rE;

      ierr = DMGetGlobalFieldOffset_Private(dm, p, f, &rS, &rE);CHKERRQ(ierr);
      for (r = rS; r < rE; ++r) {
        PetscInt numCols, cStart, c;

        ierr = PetscSectionGetDof(sectionAdj, r, &numCols);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionAdj, r, &cStart);CHKERRQ(ierr);
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

      ierr = PetscSectionGetDof(sectionAdj, row, &numCols);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sectionAdj, row, &cStart);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutGetRange(rLayout, &rStart, &rEnd);CHKERRQ(ierr);
  for (r = rStart; r < rEnd; ++r) {
    ierr      = PetscSectionGetDof(sectionAdj, r, &len);CHKERRQ(ierr);
    maxRowLen = PetscMax(maxRowLen, len);
  }
  ierr = PetscCalloc1(maxRowLen, &values);CHKERRQ(ierr);
  if (f >=0 && bs == 1) {
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt rS, rE;

      ierr = DMGetGlobalFieldOffset_Private(dm, p, f, &rS, &rE);CHKERRQ(ierr);
      for (r = rS; r < rE; ++r) {
        PetscInt numCols, cStart;

        ierr = PetscSectionGetDof(sectionAdj, r, &numCols);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(sectionAdj, r, &cStart);CHKERRQ(ierr);
        ierr = MatSetValues(A, 1, &r, numCols, &cols[cStart], values, INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  } else {
    for (r = rStart; r < rEnd; ++r) {
      PetscInt numCols, cStart;

      ierr = PetscSectionGetDof(sectionAdj, r, &numCols);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(sectionAdj, r, &cStart);CHKERRQ(ierr);
      ierr = MatSetValues(A, 1, &r, numCols, &cols[cStart], values, INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexPreallocateOperator - Calculate the matrix nonzero pattern based upon the information in the DM,
  the PetscDS it contains, and the default PetscSection.

  Collective

  Input Arguments:
+ dm   - The DMPlex
. bs   - The matrix blocksize
. dnz  - An array to hold the number of nonzeros in the diagonal block
. onz  - An array to hold the number of nonzeros in the off-diagonal block
. dnzu - An array to hold the number of nonzeros in the upper triangle of the diagonal block
. onzu - An array to hold the number of nonzeros in the upper triangle of the off-diagonal block
- fillMatrix - If PETSC_TRUE, fill the matrix with zeros

  Ouput Argument:
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 9);
  if (dnz)  PetscValidPointer(dnz,5);  if (onz)  PetscValidPointer(onz,6);
  if (dnzu) PetscValidPointer(dnzu,7); if (onzu) PetscValidPointer(onzu,8);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL, "-dm_view_preallocation", &debug, NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_Preallocate,dm,0,0,0);CHKERRQ(ierr);
  /* Create dof SF based on point SF */
  if (debug) {
    PetscSection section, sectionGlobal;
    PetscSF      sf;

    ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
    ierr = DMGetGlobalSection(dm, &sectionGlobal);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Input Section for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(section, NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Input Global Section for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSectionView(sectionGlobal, NULL);CHKERRQ(ierr);
    if (size > 1) {
      ierr = PetscPrintf(comm, "Input SF for Preallocation:\n");CHKERRQ(ierr);
      ierr = PetscSFView(sf, NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscSFCreateRemoteOffsets(sf, section, section, &remoteOffsets);CHKERRQ(ierr);
  ierr = PetscSFCreateSectionSF(sf, section, remoteOffsets, section, &sfDof);CHKERRQ(ierr);
  ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  if (debug && size > 1) {
    ierr = PetscPrintf(comm, "Dof SF for Preallocation:\n");CHKERRQ(ierr);
    ierr = PetscSFView(sfDof, NULL);CHKERRQ(ierr);
  }
  /* Create allocation vectors from adjacency graph */
  ierr = MatGetLocalSize(A, &locRows, NULL);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm, &rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(rLayout, locRows);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(rLayout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rLayout);CHKERRQ(ierr);
  /* There are 4 types of adjacency */
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  if (Nf < 1 || bs > 1) {
    ierr = DMGetBasicAdjacency(dm, &useCone, &useClosure);CHKERRQ(ierr);
    idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
    ierr = DMPlexCreateAdjacencySection_Static(dm, bs, sfDof, useCone, useClosure, PETSC_TRUE, &sectionAdj[idx], &cols[idx]);CHKERRQ(ierr);
    ierr = DMPlexUpdateAllocation_Static(dm, rLayout, bs, -1, sectionAdj[idx], cols[idx], dnz, onz, dnzu, onzu);CHKERRQ(ierr);
  } else {
    for (f = 0; f < Nf; ++f) {
      ierr = DMGetAdjacency(dm, f, &useCone, &useClosure);CHKERRQ(ierr);
      idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
      if (!sectionAdj[idx]) {ierr = DMPlexCreateAdjacencySection_Static(dm, bs, sfDof, useCone, useClosure, PETSC_TRUE, &sectionAdj[idx], &cols[idx]);CHKERRQ(ierr);}
      ierr = DMPlexUpdateAllocation_Static(dm, rLayout, bs, f, sectionAdj[idx], cols[idx], dnz, onz, dnzu, onzu);CHKERRQ(ierr);
    }
  }
  ierr = PetscSFDestroy(&sfDof);CHKERRQ(ierr);
  /* Set matrix pattern */
  ierr = MatXAIJSetPreallocation(A, bs, dnz, onz, dnzu, onzu);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  /* Check for symmetric storage */
  ierr = MatGetType(A, &mtype);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSBAIJ, &isSymBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATSEQSBAIJ, &isSymSeqBlock);CHKERRQ(ierr);
  ierr = PetscStrcmp(mtype, MATMPISBAIJ, &isSymMPIBlock);CHKERRQ(ierr);
  if (isSymBlock || isSymSeqBlock || isSymMPIBlock) {ierr = MatSetOption(A, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE);CHKERRQ(ierr);}
  /* Fill matrix with zeros */
  if (fillMatrix) {
    if (Nf < 1 || bs > 1) {
      ierr = DMGetBasicAdjacency(dm, &useCone, &useClosure);CHKERRQ(ierr);
      idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
      ierr = DMPlexFillMatrix_Static(dm, rLayout, bs, -1, sectionAdj[idx], cols[idx], A);CHKERRQ(ierr);
    } else {
      for (f = 0; f < Nf; ++f) {
        ierr = DMGetAdjacency(dm, f, &useCone, &useClosure);CHKERRQ(ierr);
        idx  = (useCone ? 1 : 0) + (useClosure ? 2 : 0);
        ierr = DMPlexFillMatrix_Static(dm, rLayout, bs, f, sectionAdj[idx], cols[idx], A);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = PetscLayoutDestroy(&rLayout);CHKERRQ(ierr);
  for (idx = 0; idx < 4; ++idx) {ierr = PetscSectionDestroy(&sectionAdj[idx]);CHKERRQ(ierr); ierr = PetscFree(cols[idx]);CHKERRQ(ierr);}
  ierr = PetscLogEventEnd(DMPLEX_Preallocate,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
PetscErrorCode DMPlexPreallocateOperator_2(DM dm, PetscInt bs, PetscSection section, PetscSection sectionGlobal, PetscInt dnz[], PetscInt onz[], PetscInt dnzu[], PetscInt onzu[], Mat A, PetscBool fillMatrix)
{
  PetscInt       *tmpClosure,*tmpAdj,*visits;
  PetscInt        c,cStart,cEnd,pStart,pEnd;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);

  maxClosureSize = 2*PetscMax(PetscPowInt(mesh->maxConeSize,depth+1),PetscPowInt(mesh->maxSupportSize,depth+1));

  ierr    = PetscSectionGetChart(section, &pStart, &pEnd);CHKERRQ(ierr);
  npoints = pEnd - pStart;

  ierr = PetscMalloc3(maxClosureSize,&tmpClosure,npoints,&lvisits,npoints,&visits);CHKERRQ(ierr);
  ierr = PetscArrayzero(lvisits,pEnd-pStart);CHKERRQ(ierr);
  ierr = PetscArrayzero(visits,pEnd-pStart);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    PetscInt *support = tmpClosure;
    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_FALSE, &supportSize, (PetscInt**)&support);CHKERRQ(ierr);
    for (p=0; p<supportSize; p++) lvisits[support[p]]++;
  }
  ierr = PetscSFReduceBegin(sf,MPIU_INT,lvisits,visits,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd  (sf,MPIU_INT,lvisits,visits,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf,MPIU_INT,visits,lvisits);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd  (sf,MPIU_INT,visits,lvisits);CHKERRQ(ierr);

  ierr = PetscSFGetRootRanks();CHKERRQ(ierr);


  ierr = PetscMalloc2(maxClosureSize*maxClosureSize,&cellmat,npoints,&owner);CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    ierr = PetscArrayzero(cellmat,maxClosureSize*maxClosureSize);CHKERRQ(ierr);
    /*
     Depth-first walk of transitive closure.
     At each leaf frame f of transitive closure that we see, add 1/visits[f] to each pair (p,q) not marked as done in cellmat.
     This contribution is added to dnz if owning ranks of p and q match, to onz otherwise.
     */
  }

  ierr = PetscSFReduceBegin(sf,MPIU_INT,ldnz,dnz,MPI_SUM);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd  (sf,MPIU_INT,lonz,onz,MPI_SUM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
