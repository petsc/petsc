#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

/*@
  DMPlexOrientPoint - Act with the given orientation on the cone points of this mesh point, and update its use in the mesh.

  Not Collective

  Input Parameters:
+ dm - The DM
. p  - The mesh point
- o  - The orientation

  Level: intermediate

.seealso: DMPlexOrient(), DMPlexGetCone(), DMPlexGetConeOrientation(), DMPlexInterpolate(), DMPlexGetChart()
@*/
PetscErrorCode DMPlexOrientPoint(DM dm, PetscInt p, PetscInt o)
{
  DMPolytopeType  ct;
  const PetscInt *arr, *cone, *ornt, *support;
  PetscInt       *newcone, *newornt;
  PetscInt        coneSize, c, supportSize, s;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
  arr  = DMPolytopeTypeGetArrangment(ct, o);
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, p, &ornt);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, coneSize, MPIU_INT, &newcone);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, coneSize, MPIU_INT, &newornt);CHKERRQ(ierr);
  for (c = 0; c < coneSize; ++c) {
    DMPolytopeType ft;
    PetscInt       nO;

    ierr = DMPlexGetCellType(dm, cone[c], &ft);CHKERRQ(ierr);
    nO   = DMPolytopeTypeGetNumArrangments(ft)/2;
    newcone[c] = cone[arr[c*2+0]];
    newornt[c] = DMPolytopeTypeComposeOrientation(ft, arr[c*2+1], ornt[arr[c*2+0]]);
    if (newornt[c] && (newornt[c] >= nO || newornt[c] < -nO)) SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid orientation %D not in [%D,%D) for %s %D", newornt[c], -nO, nO, DMPolytopeTypes[ft], cone[c]);
  }
  ierr = DMPlexSetCone(dm, p, newcone);CHKERRQ(ierr);
  ierr = DMPlexSetConeOrientation(dm, p, newornt);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, coneSize, MPIU_INT, &newcone);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, coneSize, MPIU_INT, &newornt);CHKERRQ(ierr);
  /* Update orientation of this point in the support points */
  ierr = DMPlexGetSupportSize(dm, p, &supportSize);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, p, &support);CHKERRQ(ierr);
  for (s = 0; s < supportSize; ++s) {
    ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
    for (c = 0; c < coneSize; ++c) {
      PetscInt po;

      if (cone[c] != p) continue;
      /* ornt[c] * 0 = target = po * o so that po = ornt[c] * o^{-1} */
      po   = DMPolytopeTypeComposeOrientationInv(ct, ornt[c], o);
      ierr = DMPlexInsertConeOrientation(dm, support[s], c, po);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
  - Checks face match
    - Flips non-matching
  - Inserts faces of support cells in FIFO
*/
static PetscErrorCode DMPlexCheckFace_Internal(DM dm, PetscInt *faceFIFO, PetscInt *fTop, PetscInt *fBottom, PetscInt cStart, PetscInt fStart, PetscInt fEnd, PetscBT seenCells, PetscBT flippedCells, PetscBT seenFaces)
{
  const PetscInt *support, *coneA, *coneB, *coneOA, *coneOB;
  PetscInt        supportSize, coneSizeA, coneSizeB, posA = -1, posB = -1;
  PetscInt        face, dim, seenA, flippedA, seenB, flippedB, mismatch, c;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  face = faceFIFO[(*fTop)++];
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, face, &support);CHKERRQ(ierr);
  if (supportSize < 2) PetscFunctionReturn(0);
  if (supportSize != 2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Faces should separate only two cells, not %d", supportSize);
  seenA    = PetscBTLookup(seenCells,    support[0]-cStart);
  flippedA = PetscBTLookup(flippedCells, support[0]-cStart) ? 1 : 0;
  seenB    = PetscBTLookup(seenCells,    support[1]-cStart);
  flippedB = PetscBTLookup(flippedCells, support[1]-cStart) ? 1 : 0;

  ierr = DMPlexGetConeSize(dm, support[0], &coneSizeA);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, support[1], &coneSizeB);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, support[0], &coneA);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, support[1], &coneB);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, support[0], &coneOA);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, support[1], &coneOB);CHKERRQ(ierr);
  for (c = 0; c < coneSizeA; ++c) {
    if (!PetscBTLookup(seenFaces, coneA[c]-fStart)) {
      faceFIFO[(*fBottom)++] = coneA[c];
      ierr = PetscBTSet(seenFaces, coneA[c]-fStart);CHKERRQ(ierr);
    }
    if (coneA[c] == face) posA = c;
    if (*fBottom > fEnd-fStart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %d was pushed exceeding capacity %d > %d", coneA[c], *fBottom, fEnd-fStart);
  }
  if (posA < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d could not be located in cell %d", face, support[0]);
  for (c = 0; c < coneSizeB; ++c) {
    if (!PetscBTLookup(seenFaces, coneB[c]-fStart)) {
      faceFIFO[(*fBottom)++] = coneB[c];
      ierr = PetscBTSet(seenFaces, coneB[c]-fStart);CHKERRQ(ierr);
    }
    if (coneB[c] == face) posB = c;
    if (*fBottom > fEnd-fStart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %d was pushed exceeding capacity %d > %d", coneA[c], *fBottom, fEnd-fStart);
  }
  if (posB < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d could not be located in cell %d", face, support[1]);

  if (dim == 1) {
    mismatch = posA == posB;
  } else {
    mismatch = coneOA[posA] == coneOB[posB];
  }

  if (mismatch ^ (flippedA ^ flippedB)) {
    if (seenA && seenB) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Previously seen cells %d and %d do not match: Fault mesh is non-orientable", support[0], support[1]);
    if (!seenA && !flippedA) {
      ierr = PetscBTSet(flippedCells, support[0]-cStart);CHKERRQ(ierr);
    } else if (!seenB && !flippedB) {
      ierr = PetscBTSet(flippedCells, support[1]-cStart);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent mesh orientation: Fault mesh is non-orientable");
  } else if (mismatch && flippedA && flippedB) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to flip already flipped cell: Fault mesh is non-orientable");
  ierr = PetscBTSet(seenCells, support[0]-cStart);CHKERRQ(ierr);
  ierr = PetscBTSet(seenCells, support[1]-cStart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexOrient - Give a consistent orientation to the input mesh

  Input Parameters:
. dm - The DM

  Note: The orientation data for the DM are change in-place.
$ This routine will fail for non-orientable surfaces, such as the Moebius strip.

  Level: advanced

.seealso: DMCreate(), DMPLEX
@*/
PetscErrorCode DMPlexOrient(DM dm)
{
  MPI_Comm           comm;
  PetscSF            sf;
  const PetscInt    *lpoints;
  const PetscSFNode *rpoints;
  PetscSFNode       *rorntComp = NULL, *lorntComp = NULL;
  PetscInt          *numNeighbors, **neighbors;
  PetscSFNode       *nrankComp;
  PetscBool         *match, *flipped;
  PetscBT            seenCells, flippedCells, seenFaces;
  PetscInt          *faceFIFO, fTop, fBottom, *cellComp, *faceComp;
  PetscInt           numLeaves, numRoots, dim, h, cStart, cEnd, c, cell, fStart, fEnd, face, off, totNeighbors = 0;
  PetscMPIInt        rank, size, numComponents, comp = 0;
  PetscBool          flg, flg2;
  PetscViewer        viewer = NULL, selfviewer = NULL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-orientation_view", &flg);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-orientation_view_synchronized", &flg2);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &numRoots, &numLeaves, &lpoints, &rpoints);CHKERRQ(ierr);
  /* Truth Table
     mismatch    flips   do action   mismatch   flipA ^ flipB   action
         F       0 flips     no         F             F           F
         F       1 flip      yes        F             T           T
         F       2 flips     no         T             F           T
         T       0 flips     yes        T             T           F
         T       1 flip      no
         T       2 flips     yes
  */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetVTKCellHeight(dm, &h);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, h,   &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, h+1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = PetscBTCreate(cEnd - cStart, &seenCells);CHKERRQ(ierr);
  ierr = PetscBTMemzero(cEnd - cStart, seenCells);CHKERRQ(ierr);
  ierr = PetscBTCreate(cEnd - cStart, &flippedCells);CHKERRQ(ierr);
  ierr = PetscBTMemzero(cEnd - cStart, flippedCells);CHKERRQ(ierr);
  ierr = PetscBTCreate(fEnd - fStart, &seenFaces);CHKERRQ(ierr);
  ierr = PetscBTMemzero(fEnd - fStart, seenFaces);CHKERRQ(ierr);
  ierr = PetscCalloc3(fEnd - fStart, &faceFIFO, cEnd-cStart, &cellComp, fEnd-fStart, &faceComp);CHKERRQ(ierr);
  /*
   OLD STYLE
   - Add an integer array over cells and faces (component) for connected component number
   Foreach component
     - Mark the initial cell as seen
     - Process component as usual
     - Set component for all seenCells
     - Wipe seenCells and seenFaces (flippedCells can stay)
   - Generate parallel adjacency for component using SF and seenFaces
   - Collect numComponents adj data from each proc to 0
   - Build same serial graph
   - Use same solver
   - Use Scatterv to to send back flipped flags for each component
   - Negate flippedCells by component

   NEW STYLE
   - Create the adj on each process
   - Bootstrap to complete graph on proc 0
  */
  /* Loop over components */
  for (cell = cStart; cell < cEnd; ++cell) cellComp[cell-cStart] = -1;
  do {
    /* Look for first unmarked cell */
    for (cell = cStart; cell < cEnd; ++cell) if (cellComp[cell-cStart] < 0) break;
    if (cell >= cEnd) break;
    /* Initialize FIFO with first cell in component */
    {
      const PetscInt *cone;
      PetscInt        coneSize;

      fTop = fBottom = 0;
      ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        faceFIFO[fBottom++] = cone[c];
        ierr = PetscBTSet(seenFaces, cone[c]-fStart);CHKERRQ(ierr);
      }
      ierr = PetscBTSet(seenCells, cell-cStart);CHKERRQ(ierr);
    }
    /* Consider each face in FIFO */
    while (fTop < fBottom) {
      ierr = DMPlexCheckFace_Internal(dm, faceFIFO, &fTop, &fBottom, cStart, fStart, fEnd, seenCells, flippedCells, seenFaces);CHKERRQ(ierr);
    }
    /* Set component for cells and faces */
    for (cell = 0; cell < cEnd-cStart; ++cell) {
      if (PetscBTLookup(seenCells, cell)) cellComp[cell] = comp;
    }
    for (face = 0; face < fEnd-fStart; ++face) {
      if (PetscBTLookup(seenFaces, face)) faceComp[face] = comp;
    }
    /* Wipe seenCells and seenFaces for next component */
    ierr = PetscBTMemzero(fEnd - fStart, seenFaces);CHKERRQ(ierr);
    ierr = PetscBTMemzero(cEnd - cStart, seenCells);CHKERRQ(ierr);
    ++comp;
  } while (1);
  numComponents = comp;
  if (flg) {
    PetscViewer v;

    ierr = PetscViewerASCIIGetStdout(comm, &v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for serial flipped cells:\n", rank);CHKERRQ(ierr);
    ierr = PetscBTView(cEnd-cStart, flippedCells, v);CHKERRQ(ierr);
    ierr = PetscViewerFlush(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(v);CHKERRQ(ierr);
  }
  /* Now all subdomains are oriented, but we need a consistent parallel orientation */
  if (numLeaves >= 0) {
    /* Store orientations of boundary faces*/
    ierr = PetscCalloc2(numRoots,&rorntComp,numRoots,&lorntComp);CHKERRQ(ierr);
    for (face = fStart; face < fEnd; ++face) {
      const PetscInt *cone, *support, *ornt;
      PetscInt        coneSize, supportSize;

      ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
      if (supportSize != 1) continue;
      ierr = DMPlexGetSupport(dm, face, &support);CHKERRQ(ierr);

      ierr = DMPlexGetCone(dm, support[0], &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, support[0], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, support[0], &ornt);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) if (cone[c] == face) break;
      if (dim == 1) {
        /* Use cone position instead, shifted to -1 or 1 */
        if (PetscBTLookup(flippedCells, support[0]-cStart)) rorntComp[face].rank = 1-c*2;
        else                                                rorntComp[face].rank = c*2-1;
      } else {
        if (PetscBTLookup(flippedCells, support[0]-cStart)) rorntComp[face].rank = ornt[c] < 0 ? -1 :  1;
        else                                                rorntComp[face].rank = ornt[c] < 0 ?  1 : -1;
      }
      rorntComp[face].index = faceComp[face-fStart];
    }
    /* Communicate boundary edge orientations */
    ierr = PetscSFBcastBegin(sf, MPIU_2INT, rorntComp, lorntComp,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf, MPIU_2INT, rorntComp, lorntComp,MPI_REPLACE);CHKERRQ(ierr);
  }
  /* Get process adjacency */
  ierr = PetscMalloc2(numComponents, &numNeighbors, numComponents, &neighbors);CHKERRQ(ierr);
  viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm));
  if (flg2) {ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);}
  ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&selfviewer);CHKERRQ(ierr);
  for (comp = 0; comp < numComponents; ++comp) {
    PetscInt  l, n;

    numNeighbors[comp] = 0;
    ierr = PetscMalloc1(PetscMax(numLeaves, 0), &neighbors[comp]);CHKERRQ(ierr);
    /* I know this is p^2 time in general, but for bounded degree its alright */
    for (l = 0; l < numLeaves; ++l) {
      const PetscInt face = lpoints[l];

      /* Find a representative face (edge) separating pairs of procs */
      if ((face >= fStart) && (face < fEnd) && (faceComp[face-fStart] == comp)) {
        const PetscInt rrank = rpoints[l].rank;
        const PetscInt rcomp = lorntComp[face].index;

        for (n = 0; n < numNeighbors[comp]; ++n) if ((rrank == rpoints[neighbors[comp][n]].rank) && (rcomp == lorntComp[lpoints[neighbors[comp][n]]].index)) break;
        if (n >= numNeighbors[comp]) {
          PetscInt supportSize;

          ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
          if (supportSize != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Boundary faces should see one cell, not %d", supportSize);
          if (flg) {ierr = PetscViewerASCIIPrintf(selfviewer, "[%d]: component %d, Found representative leaf %d (face %d) connecting to face %d on (%d, %d) with orientation %d\n", rank, comp, l, face, rpoints[l].index, rrank, rcomp, lorntComp[face].rank);CHKERRQ(ierr);}
          neighbors[comp][numNeighbors[comp]++] = l;
        }
      }
    }
    totNeighbors += numNeighbors[comp];
  }
  ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&selfviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  if (flg2) {ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);}
  ierr = PetscMalloc2(totNeighbors, &nrankComp, totNeighbors, &match);CHKERRQ(ierr);
  for (comp = 0, off = 0; comp < numComponents; ++comp) {
    PetscInt n;

    for (n = 0; n < numNeighbors[comp]; ++n, ++off) {
      const PetscInt face = lpoints[neighbors[comp][n]];
      const PetscInt o    = rorntComp[face].rank*lorntComp[face].rank;

      if      (o < 0) match[off] = PETSC_TRUE;
      else if (o > 0) match[off] = PETSC_FALSE;
      else SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid face %d (%d, %d) neighbor: %d comp: %d", face, rorntComp[face], lorntComp[face], neighbors[comp][n], comp);
      nrankComp[off].rank  = rpoints[neighbors[comp][n]].rank;
      nrankComp[off].index = lorntComp[lpoints[neighbors[comp][n]]].index;
    }
    ierr = PetscFree(neighbors[comp]);CHKERRQ(ierr);
  }
  /* Collect the graph on 0 */
  if (numLeaves >= 0) {
    Mat          G;
    PetscBT      seenProcs, flippedProcs;
    PetscInt    *procFIFO, pTop, pBottom;
    PetscInt    *N   = NULL, *Noff;
    PetscSFNode *adj = NULL;
    PetscBool   *val = NULL;
    PetscMPIInt *recvcounts = NULL, *displs = NULL, *Nc, p, o;
    PetscMPIInt  size = 0;

    ierr = PetscCalloc1(numComponents, &flipped);CHKERRQ(ierr);
    if (!rank) {ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);}
    ierr = PetscCalloc4(size, &recvcounts, size+1, &displs, size, &Nc, size+1, &Noff);CHKERRQ(ierr);
    ierr = MPI_Gather(&numComponents, 1, MPI_INT, Nc, 1, MPI_INT, 0, comm);CHKERRMPI(ierr);
    for (p = 0; p < size; ++p) {
      displs[p+1] = displs[p] + Nc[p];
    }
    if (!rank) {ierr = PetscMalloc1(displs[size],&N);CHKERRQ(ierr);}
    ierr = MPI_Gatherv(numNeighbors, numComponents, MPIU_INT, N, Nc, displs, MPIU_INT, 0, comm);CHKERRMPI(ierr);
    for (p = 0, o = 0; p < size; ++p) {
      recvcounts[p] = 0;
      for (c = 0; c < Nc[p]; ++c, ++o) recvcounts[p] += N[o];
      displs[p+1] = displs[p] + recvcounts[p];
    }
    if (!rank) {ierr = PetscMalloc2(displs[size], &adj, displs[size], &val);CHKERRQ(ierr);}
    ierr = MPI_Gatherv(nrankComp, totNeighbors, MPIU_2INT, adj, recvcounts, displs, MPIU_2INT, 0, comm);CHKERRMPI(ierr);
    ierr = MPI_Gatherv(match, totNeighbors, MPIU_BOOL, val, recvcounts, displs, MPIU_BOOL, 0, comm);CHKERRMPI(ierr);
    ierr = PetscFree2(numNeighbors, neighbors);CHKERRQ(ierr);
    if (!rank) {
      for (p = 1; p <= size; ++p) {Noff[p] = Noff[p-1] + Nc[p-1];}
      if (flg) {
        PetscInt n;

        for (p = 0, off = 0; p < size; ++p) {
          for (c = 0; c < Nc[p]; ++c) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "Proc %d Comp %d:\n", p, c);CHKERRQ(ierr);
            for (n = 0; n < N[Noff[p]+c]; ++n, ++off) {
              ierr = PetscPrintf(PETSC_COMM_SELF, "  edge (%d, %d) (%d):\n", adj[off].rank, adj[off].index, val[off]);CHKERRQ(ierr);
            }
          }
        }
      }
      /* Symmetrize the graph */
      ierr = MatCreate(PETSC_COMM_SELF, &G);CHKERRQ(ierr);
      ierr = MatSetSizes(G, Noff[size], Noff[size], Noff[size], Noff[size]);CHKERRQ(ierr);
      ierr = MatSetUp(G);CHKERRQ(ierr);
      for (p = 0, off = 0; p < size; ++p) {
        for (c = 0; c < Nc[p]; ++c) {
          const PetscInt r = Noff[p]+c;
          PetscInt       n;

          for (n = 0; n < N[r]; ++n, ++off) {
            const PetscInt    q = Noff[adj[off].rank] + adj[off].index;
            const PetscScalar o = val[off] ? 1.0 : 0.0;

            ierr = MatSetValues(G, 1, &r, 1, &q, &o, INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(G, 1, &q, 1, &r, &o, INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
      ierr = MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      ierr = PetscBTCreate(Noff[size], &seenProcs);CHKERRQ(ierr);
      ierr = PetscBTMemzero(Noff[size], seenProcs);CHKERRQ(ierr);
      ierr = PetscBTCreate(Noff[size], &flippedProcs);CHKERRQ(ierr);
      ierr = PetscBTMemzero(Noff[size], flippedProcs);CHKERRQ(ierr);
      ierr = PetscMalloc1(Noff[size], &procFIFO);CHKERRQ(ierr);
      pTop = pBottom = 0;
      for (p = 0; p < Noff[size]; ++p) {
        if (PetscBTLookup(seenProcs, p)) continue;
        /* Initialize FIFO with next proc */
        procFIFO[pBottom++] = p;
        ierr = PetscBTSet(seenProcs, p);CHKERRQ(ierr);
        /* Consider each proc in FIFO */
        while (pTop < pBottom) {
          const PetscScalar *ornt;
          const PetscInt    *neighbors;
          PetscInt           proc, nproc, seen, flippedA, flippedB, mismatch, numNeighbors, n;

          proc     = procFIFO[pTop++];
          flippedA = PetscBTLookup(flippedProcs, proc) ? 1 : 0;
          ierr = MatGetRow(G, proc, &numNeighbors, &neighbors, &ornt);CHKERRQ(ierr);
          /* Loop over neighboring procs */
          for (n = 0; n < numNeighbors; ++n) {
            nproc    = neighbors[n];
            mismatch = PetscRealPart(ornt[n]) > 0.5 ? 0 : 1;
            seen     = PetscBTLookup(seenProcs, nproc);
            flippedB = PetscBTLookup(flippedProcs, nproc) ? 1 : 0;

            if (mismatch ^ (flippedA ^ flippedB)) {
              if (seen) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Previously seen procs %d and %d do not match: Fault mesh is non-orientable", proc, nproc);
              if (!flippedB) {
                ierr = PetscBTSet(flippedProcs, nproc);CHKERRQ(ierr);
              } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent mesh orientation: Fault mesh is non-orientable");
            } else if (mismatch && flippedA && flippedB) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to flip already flipped cell: Fault mesh is non-orientable");
            if (!seen) {
              procFIFO[pBottom++] = nproc;
              ierr = PetscBTSet(seenProcs, nproc);CHKERRQ(ierr);
            }
          }
        }
      }
      ierr = PetscFree(procFIFO);CHKERRQ(ierr);
      ierr = MatDestroy(&G);CHKERRQ(ierr);
      ierr = PetscFree2(adj, val);CHKERRQ(ierr);
      ierr = PetscBTDestroy(&seenProcs);CHKERRQ(ierr);
    }
    /* Scatter flip flags */
    {
      PetscBool *flips = NULL;

      if (!rank) {
        ierr = PetscMalloc1(Noff[size], &flips);CHKERRQ(ierr);
        for (p = 0; p < Noff[size]; ++p) {
          flips[p] = PetscBTLookup(flippedProcs, p) ? PETSC_TRUE : PETSC_FALSE;
          if (flg && flips[p]) {ierr = PetscPrintf(comm, "Flipping Proc+Comp %d:\n", p);CHKERRQ(ierr);}
        }
        for (p = 0; p < size; ++p) {
          displs[p+1] = displs[p] + Nc[p];
        }
      }
      ierr = MPI_Scatterv(flips, Nc, displs, MPIU_BOOL, flipped, numComponents, MPIU_BOOL, 0, comm);CHKERRMPI(ierr);
      ierr = PetscFree(flips);CHKERRQ(ierr);
    }
    if (!rank) {ierr = PetscBTDestroy(&flippedProcs);CHKERRQ(ierr);}
    ierr = PetscFree(N);CHKERRQ(ierr);
    ierr = PetscFree4(recvcounts, displs, Nc, Noff);CHKERRQ(ierr);
    ierr = PetscFree2(nrankComp, match);CHKERRQ(ierr);

    /* Decide whether to flip cells in each component */
    for (c = 0; c < cEnd-cStart; ++c) {if (flipped[cellComp[c]]) {ierr = PetscBTNegate(flippedCells, c);CHKERRQ(ierr);}}
    ierr = PetscFree(flipped);CHKERRQ(ierr);
  }
  if (flg) {
    PetscViewer v;

    ierr = PetscViewerASCIIGetStdout(comm, &v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for parallel flipped cells:\n", rank);CHKERRQ(ierr);
    ierr = PetscBTView(cEnd-cStart, flippedCells, v);CHKERRQ(ierr);
    ierr = PetscViewerFlush(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(v);CHKERRQ(ierr);
  }
  /* Reverse flipped cells in the mesh */
  for (c = cStart; c < cEnd; ++c) {
    if (PetscBTLookup(flippedCells, c-cStart)) {
      ierr = DMPlexOrientPoint(dm, c, -1);CHKERRQ(ierr);
    }
  }
  ierr = PetscBTDestroy(&seenCells);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&flippedCells);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&seenFaces);CHKERRQ(ierr);
  ierr = PetscFree2(numNeighbors, neighbors);CHKERRQ(ierr);
  ierr = PetscFree2(rorntComp, lorntComp);CHKERRQ(ierr);
  ierr = PetscFree3(faceFIFO, cellComp, faceComp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
