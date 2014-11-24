#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "DMPlexReverseCell"
/*@
  DMPlexReverseCell - Give a mesh cell the opposite orientation

  Input Parameters:
+ dm   - The DM
- cell - The cell number

  Note: The modification of the DM is done in-place.

  Level: advanced

.seealso: DMPlexOrient(), DMCreate(), DMPLEX
@*/
PetscErrorCode DMPlexReverseCell(DM dm, PetscInt cell)
{
  /* Note that the reverse orientation ro of a face with orientation o is:

       ro = o >= 0 ? -(faceSize - o) : faceSize + o

     where faceSize is the size of the cone for the face.
  */
  const PetscInt *cone,    *coneO, *support;
  PetscInt       *revcone, *revconeO;
  PetscInt        maxConeSize, coneSize, supportSize, faceSize, cp, sp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, PETSC_INT, &revcone);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, maxConeSize, PETSC_INT, &revconeO);CHKERRQ(ierr);
  /* Reverse cone, and reverse orientations of faces */
  ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, cell, &coneO);CHKERRQ(ierr);
  for (cp = 0; cp < coneSize; ++cp) {
    const PetscInt rcp = coneSize-cp-1;

    ierr = DMPlexGetConeSize(dm, cone[rcp], &faceSize);CHKERRQ(ierr);
    revcone[cp]  = cone[rcp];
    revconeO[cp] = coneO[rcp] >= 0 ? -(faceSize-coneO[rcp]) : faceSize+coneO[rcp];
  }
  ierr = DMPlexSetCone(dm, cell, revcone);CHKERRQ(ierr);
  ierr = DMPlexSetConeOrientation(dm, cell, revconeO);CHKERRQ(ierr);
  /* Reverse orientation of this cell in the support hypercells */
  faceSize = coneSize;
  ierr = DMPlexGetSupportSize(dm, cell, &supportSize);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, cell, &support);CHKERRQ(ierr);
  for (sp = 0; sp < supportSize; ++sp) {
    ierr = DMPlexGetConeSize(dm, support[sp], &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, support[sp], &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, support[sp], &coneO);CHKERRQ(ierr);
    for (cp = 0; cp < coneSize; ++cp) {
      if (cone[cp] != cell) continue;
      ierr = DMPlexInsertConeOrientation(dm, support[sp], cp, coneO[cp] >= 0 ? -(faceSize-coneO[cp]) : faceSize+coneO[cp]);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreWorkArray(dm, maxConeSize, PETSC_INT, &revcone);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, maxConeSize, PETSC_INT, &revconeO);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexOrient"
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
  MPI_Comm       comm;
  PetscBT        seenCells, flippedCells, seenFaces;
  PetscInt      *faceFIFO, fTop, fBottom;
  PetscInt       dim, h, cStart, cEnd, c, fStart, fEnd, face;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(((PetscObject) dm)->prefix, "-orientation_view", &flg);CHKERRQ(ierr);
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
  ierr = PetscMalloc1((fEnd - fStart), &faceFIFO);CHKERRQ(ierr);
  fTop = fBottom = 0;
  /* Initialize FIFO with first cell */
  if (cEnd > cStart) {
    const PetscInt *cone;
    PetscInt        coneSize;

    ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, cStart, &cone);CHKERRQ(ierr);
    for (c = 0; c < coneSize; ++c) {
      faceFIFO[fBottom++] = cone[c];
      ierr = PetscBTSet(seenFaces, cone[c]-fStart);CHKERRQ(ierr);
    }
  }
  /* Consider each face in FIFO */
  while (fTop < fBottom) {
    const PetscInt *support, *coneA, *coneB, *coneOA, *coneOB;
    PetscInt        supportSize, coneSizeA, coneSizeB, posA = -1, posB = -1;
    PetscInt        seenA, flippedA, seenB, flippedB, mismatch;

    face = faceFIFO[fTop++];
    ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, face, &support);CHKERRQ(ierr);
    if (supportSize < 2) continue;
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
        faceFIFO[fBottom++] = coneA[c];
        ierr = PetscBTSet(seenFaces, coneA[c]-fStart);CHKERRQ(ierr);
      }
      if (coneA[c] == face) posA = c;
      if (fBottom > fEnd-fStart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %d was pushed exceeding capacity %d > %d", coneA[c], fBottom, fEnd-fStart);
    }
    if (posA < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d could not be located in cell %d", face, support[0]);
    for (c = 0; c < coneSizeB; ++c) {
      if (!PetscBTLookup(seenFaces, coneB[c]-fStart)) {
        faceFIFO[fBottom++] = coneB[c];
        ierr = PetscBTSet(seenFaces, coneB[c]-fStart);CHKERRQ(ierr);
      }
      if (coneB[c] == face) posB = c;
      if (fBottom > fEnd-fStart) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %d was pushed exceeding capacity %d > %d", coneA[c], fBottom, fEnd-fStart);
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
  }
  if (flg) {
    PetscViewer v;
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetStdout(comm, &v);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(v, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for serial seen faces:\n", rank);CHKERRQ(ierr);
    ierr = PetscBTView(fEnd-fStart, seenFaces,    v);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(v, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for serial flipped cells:\n", rank);CHKERRQ(ierr);
    ierr = PetscBTView(cEnd-cStart, flippedCells, v);CHKERRQ(ierr);
  }
  /* Now all subdomains are oriented, but we need a consistent parallel orientation */
  {
    /* Find a representative face (edge) separating pairs of procs */
    PetscSF            sf;
    const PetscInt    *lpoints;
    const PetscSFNode *rpoints;
    PetscInt          *neighbors, *nranks;
    PetscInt           numLeaves, numRoots, numNeighbors = 0, l, n;

    ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sf, &numRoots, &numLeaves, &lpoints, &rpoints);CHKERRQ(ierr);
    if (numLeaves >= 0) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneSize, supportSize;
      int            *rornt, *lornt; /* PetscSF cannot handle smaller than int */
      PetscBool      *match, flipped = PETSC_FALSE;

      ierr = PetscMalloc1(numLeaves,&neighbors);CHKERRQ(ierr);
      /* I know this is p^2 time in general, but for bounded degree its alright */
      for (l = 0; l < numLeaves; ++l) {
        const PetscInt face = lpoints[l];
        if ((face >= fStart) && (face < fEnd)) {
          const PetscInt rank = rpoints[l].rank;
          for (n = 0; n < numNeighbors; ++n) if (rank == rpoints[neighbors[n]].rank) break;
          if (n >= numNeighbors) {
            PetscInt supportSize;
            ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
            if (supportSize != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Boundary faces should see one cell, not %d", supportSize);
            neighbors[numNeighbors++] = l;
          }
        }
      }
      ierr = PetscCalloc4(numNeighbors,&match,numNeighbors,&nranks,numRoots,&rornt,numRoots,&lornt);CHKERRQ(ierr);
      for (face = fStart; face < fEnd; ++face) {
        ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
        if (supportSize != 1) continue;
        ierr = DMPlexGetSupport(dm, face, &support);CHKERRQ(ierr);

        ierr = DMPlexGetCone(dm, support[0], &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeSize(dm, support[0], &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, support[0], &ornt);CHKERRQ(ierr);
        for (c = 0; c < coneSize; ++c) if (cone[c] == face) break;
        if (dim == 1) {
          /* Use cone position instead, shifted to -1 or 1 */
          rornt[face] = c*2-1;
        } else {
          if (PetscBTLookup(flippedCells, support[0]-cStart)) rornt[face] = ornt[c] < 0 ? -1 :  1;
          else                                                rornt[face] = ornt[c] < 0 ?  1 : -1;
        }
      }
      /* Mark each edge with match or nomatch */
      ierr = PetscSFBcastBegin(sf, MPI_INT, rornt, lornt);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf, MPI_INT, rornt, lornt);CHKERRQ(ierr);
      for (n = 0; n < numNeighbors; ++n) {
        const PetscInt face = lpoints[neighbors[n]];

        if (rornt[face]*lornt[face] < 0) match[n] = PETSC_TRUE;
        else                             match[n] = PETSC_FALSE;
        nranks[n] = rpoints[neighbors[n]].rank;
      }
      /* Collect the graph on 0 */
      {
        Mat          G;
        PetscBT      seenProcs, flippedProcs;
        PetscInt    *procFIFO, pTop, pBottom;
        PetscInt    *adj = NULL;
        PetscBool   *val = NULL;
        PetscMPIInt *recvcounts = NULL, *displs = NULL, p;
        PetscMPIInt  N = numNeighbors, numProcs = 0, rank;
        PetscInt     debug = 0;

        ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
        if (!rank) {ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);}
        ierr = PetscCalloc2(numProcs,&recvcounts,numProcs+1,&displs);CHKERRQ(ierr);
        ierr = MPI_Gather(&N, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);CHKERRQ(ierr);
        for (p = 0; p < numProcs; ++p) {
          displs[p+1] = displs[p] + recvcounts[p];
        }
        if (!rank) {ierr = PetscMalloc2(displs[numProcs],&adj,displs[numProcs],&val);CHKERRQ(ierr);}
        ierr = MPI_Gatherv(nranks, numNeighbors, MPIU_INT, adj, recvcounts, displs, MPIU_INT, 0, comm);CHKERRQ(ierr);
        ierr = MPI_Gatherv(match, numNeighbors, MPIU_BOOL, val, recvcounts, displs, MPIU_BOOL, 0, comm);CHKERRQ(ierr);
        if (debug) {
          for (p = 0; p < numProcs; ++p) {
            ierr = PetscPrintf(comm, "Proc %d:\n", p);
            for (n = 0; n < recvcounts[p]; ++n) {
              ierr = PetscPrintf(comm, "  edge %d (%d):\n", adj[displs[p]+n], val[displs[p]+n]);
            }
          }
        }
        /* Symmetrize the graph */
        ierr = MatCreate(PETSC_COMM_SELF, &G);CHKERRQ(ierr);
        ierr = MatSetSizes(G, numProcs, numProcs, numProcs, numProcs);CHKERRQ(ierr);
        ierr = MatSetUp(G);CHKERRQ(ierr);
        for (p = 0; p < numProcs; ++p) {
          for (n = 0; n < recvcounts[p]; ++n) {
            const PetscInt    r = p;
            const PetscInt    q = adj[displs[p]+n];
            const PetscScalar o = val[displs[p]+n] ? 1.0 : 0.0;

            ierr = MatSetValues(G, 1, &r, 1, &q, &o, INSERT_VALUES);CHKERRQ(ierr);
            ierr = MatSetValues(G, 1, &q, 1, &r, &o, INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        ierr = MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

        ierr = PetscBTCreate(numProcs, &seenProcs);CHKERRQ(ierr);
        ierr = PetscBTMemzero(numProcs, seenProcs);CHKERRQ(ierr);
        ierr = PetscBTCreate(numProcs, &flippedProcs);CHKERRQ(ierr);
        ierr = PetscBTMemzero(numProcs, flippedProcs);CHKERRQ(ierr);
        ierr = PetscMalloc1(numProcs,&procFIFO);CHKERRQ(ierr);
        pTop = pBottom = 0;
        for (p = 0; p < numProcs; ++p) {
          if (PetscBTLookup(seenProcs, p)) continue;
          /* Initialize FIFO with next proc */
          procFIFO[pBottom++] = p;
          ierr = PetscBTSet(seenProcs, p);CHKERRQ(ierr);
          /* Consider each proc in FIFO */
          while (pTop < pBottom) {
            const PetscScalar *ornt;
            const PetscInt    *neighbors;
            PetscInt           proc, nproc, seen, flippedA, flippedB, mismatch, numNeighbors;

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

        ierr = PetscFree2(recvcounts,displs);CHKERRQ(ierr);
        ierr = PetscFree2(adj,val);CHKERRQ(ierr);
        {
          PetscBool *flips;

          ierr = PetscMalloc1(numProcs,&flips);CHKERRQ(ierr);
          for (p = 0; p < numProcs; ++p) {
            flips[p] = PetscBTLookup(flippedProcs, p) ? PETSC_TRUE : PETSC_FALSE;
            if (debug && flips[p]) {ierr = PetscPrintf(comm, "Flipping Proc %d:\n", p);}
          }
          ierr = MPI_Scatter(flips, 1, MPIU_BOOL, &flipped, 1, MPIU_BOOL, 0, comm);CHKERRQ(ierr);
          ierr = PetscFree(flips);CHKERRQ(ierr);
        }
        ierr = PetscBTDestroy(&seenProcs);CHKERRQ(ierr);
        ierr = PetscBTDestroy(&flippedProcs);CHKERRQ(ierr);
      }
      ierr = PetscFree4(match,nranks,rornt,lornt);CHKERRQ(ierr);
      ierr = PetscFree(neighbors);CHKERRQ(ierr);
      if (flipped) {for (c = cStart; c < cEnd; ++c) {ierr = PetscBTNegate(flippedCells, c-cStart);CHKERRQ(ierr);}}
    }
  }
  if (flg) {
    PetscViewer v;
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetStdout(comm, &v);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(v, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for parallel flipped cells:\n", rank);CHKERRQ(ierr);
    ierr = PetscBTView(cEnd-cStart, flippedCells, v);CHKERRQ(ierr);
  }
  /* Reverse flipped cells in the mesh */
  for (c = cStart; c < cEnd; ++c) {
    if (PetscBTLookup(flippedCells, c-cStart)) {ierr = DMPlexReverseCell(dm, c);CHKERRQ(ierr);}
  }
  ierr = PetscBTDestroy(&seenCells);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&flippedCells);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&seenFaces);CHKERRQ(ierr);
  ierr = PetscFree(faceFIFO);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
