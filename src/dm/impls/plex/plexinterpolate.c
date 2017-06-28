#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

/*
  DMPlexGetFaces_Internal - Gets groups of vertices that correspond to faces for the given cell
*/
PetscErrorCode DMPlexGetFaces_Internal(DM dm, PetscInt dim, PetscInt p, PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  const PetscInt *cone = NULL;
  PetscInt        maxConeSize, maxSupportSize, coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetRawFaces_Internal(dm, dim, coneSize, cone, numFaces, faceSize, faces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexRestoreFaces_Internal - Restores the array
*/
PetscErrorCode DMPlexRestoreFaces_Internal(DM dm, PetscInt dim, PetscInt p, PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMRestoreWorkArray(dm, 0, PETSC_INT, (void *) faces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexGetRawFaces_Internal - Gets groups of vertices that correspond to faces for the given cone
*/
PetscErrorCode DMPlexGetRawFaces_Internal(DM dm, PetscInt dim, PetscInt coneSize, const PetscInt cone[], PetscInt *numFaces, PetscInt *faceSize, const PetscInt *faces[])
{
  PetscInt       *facesTmp;
  PetscInt        maxConeSize, maxSupportSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  if (faces) {ierr = DMGetWorkArray(dm, PetscSqr(PetscMax(maxConeSize, maxSupportSize)), PETSC_INT, &facesTmp);CHKERRQ(ierr);}
  switch (dim) {
  case 1:
    switch (coneSize) {
    case 2:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 2;
      if (faceSize) *faceSize         = 1;
      break;
    default:
      SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  case 2:
    switch (coneSize) {
    case 3:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[0];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 3;
      if (faceSize) *faceSize         = 2;
      break;
    case 4:
      /* Vertices follow right hand rule */
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[3];
        facesTmp[6] = cone[3]; facesTmp[7] = cone[0];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 4;
      if (faceSize) *faceSize         = 2;
      break;
    default:
      SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  case 3:
    switch (coneSize) {
    case 3:
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1] = cone[1];
        facesTmp[2] = cone[1]; facesTmp[3] = cone[2];
        facesTmp[4] = cone[2]; facesTmp[5] = cone[0];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 3;
      if (faceSize) *faceSize         = 2;
      break;
    case 4:
      /* Vertices of first face follow right hand rule and normal points away from last vertex */
      if (faces) {
        facesTmp[0] = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2];
        facesTmp[3] = cone[0]; facesTmp[4]  = cone[3]; facesTmp[5]  = cone[1];
        facesTmp[6] = cone[0]; facesTmp[7]  = cone[2]; facesTmp[8]  = cone[3];
        facesTmp[9] = cone[2]; facesTmp[10] = cone[1]; facesTmp[11] = cone[3];
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 4;
      if (faceSize) *faceSize         = 3;
      break;
    case 8:
      if (faces) {
        facesTmp[0]  = cone[0]; facesTmp[1]  = cone[1]; facesTmp[2]  = cone[2]; facesTmp[3]  = cone[3]; /* Bottom */
        facesTmp[4]  = cone[4]; facesTmp[5]  = cone[5]; facesTmp[6]  = cone[6]; facesTmp[7]  = cone[7]; /* Top */
        facesTmp[8]  = cone[0]; facesTmp[9]  = cone[3]; facesTmp[10] = cone[5]; facesTmp[11] = cone[4]; /* Front */
        facesTmp[12] = cone[2]; facesTmp[13] = cone[1]; facesTmp[14] = cone[7]; facesTmp[15] = cone[6]; /* Back */
        facesTmp[16] = cone[3]; facesTmp[17] = cone[2]; facesTmp[18] = cone[6]; facesTmp[19] = cone[5]; /* Right */
        facesTmp[20] = cone[0]; facesTmp[21] = cone[4]; facesTmp[22] = cone[7]; facesTmp[23] = cone[1]; /* Left */
        *faces = facesTmp;
      }
      if (numFaces) *numFaces         = 6;
      if (faceSize) *faceSize         = 4;
      break;
    default:
      SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Cone size %D not supported for dimension %D", coneSize, dim);
    }
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not supported", dim);
  }
  PetscFunctionReturn(0);
}

/* This interpolates faces for cells at some stratum */
static PetscErrorCode DMPlexInterpolateFaces_Internal(DM dm, PetscInt cellDepth, DM idm)
{
  DMLabel        subpointMap;
  PetscHashIJKL  faceTable;
  PetscInt      *pStart, *pEnd;
  PetscInt       cellDim, depth, faceDepth = cellDepth, numPoints = 0, faceSizeAll = 0, face, c, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &cellDim);CHKERRQ(ierr);
  /* HACK: I need a better way to determine face dimension, or an alternative to GetFaces() */
  ierr = DMPlexGetSubpointMap(dm, &subpointMap);CHKERRQ(ierr);
  if (subpointMap) ++cellDim;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ++depth;
  ++cellDepth;
  cellDim -= depth - cellDepth;
  ierr = PetscMalloc2(depth+1,&pStart,depth+1,&pEnd);CHKERRQ(ierr);
  for (d = depth-1; d >= faceDepth; --d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d+1], &pEnd[d+1]);CHKERRQ(ierr);
  }
  ierr = DMPlexGetDepthStratum(dm, -1, NULL, &pStart[faceDepth]);CHKERRQ(ierr);
  pEnd[faceDepth] = pStart[faceDepth];
  for (d = faceDepth-1; d >= 0; --d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
  }
  if (pEnd[cellDepth] > pStart[cellDepth]) {ierr = DMPlexGetFaces_Internal(dm, cellDim, pStart[cellDepth], NULL, &faceSizeAll, NULL);CHKERRQ(ierr);}
  if (faceSizeAll > 4) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Do not support interpolation of meshes with faces of %D vertices", faceSizeAll);
  ierr = PetscHashIJKLCreate(&faceTable);CHKERRQ(ierr);
  for (c = pStart[cellDepth], face = pStart[faceDepth]; c < pEnd[cellDepth]; ++c) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf;

    ierr = DMPlexGetFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != faceSizeAll) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent face for cell %D of size %D != %D", c, faceSize, faceSizeAll);
    for (cf = 0; cf < numCellFaces; ++cf) {
      const PetscInt   *cellFace = &cellFaces[cf*faceSize];
      PetscHashIJKLKey  key;
      PetscHashIJKLIter missing, iter;

      if (faceSize == 2) {
        key.i = PetscMin(cellFace[0], cellFace[1]);
        key.j = PetscMax(cellFace[0], cellFace[1]);
        key.k = 0;
        key.l = 0;
      } else {
        key.i = cellFace[0]; key.j = cellFace[1]; key.k = cellFace[2]; key.l = faceSize > 3 ? cellFace[3] : 0;
        ierr = PetscSortInt(faceSize, (PetscInt *) &key);CHKERRQ(ierr);
      }
      ierr = PetscHashIJKLPut(faceTable, key, &missing, &iter);CHKERRQ(ierr);
      if (missing) {ierr = PetscHashIJKLSet(faceTable, iter, face++);CHKERRQ(ierr);}
    }
    ierr = DMPlexRestoreFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
  }
  pEnd[faceDepth] = face;
  ierr = PetscHashIJKLDestroy(&faceTable);CHKERRQ(ierr);
  /* Count new points */
  for (d = 0; d <= depth; ++d) {
    numPoints += pEnd[d]-pStart[d];
  }
  ierr = DMPlexSetChart(idm, 0, numPoints);CHKERRQ(ierr);
  /* Set cone sizes */
  for (d = 0; d <= depth; ++d) {
    PetscInt coneSize, p;

    if (d == faceDepth) {
      for (p = pStart[d]; p < pEnd[d]; ++p) {
        /* I see no way to do this if we admit faces of different shapes */
        ierr = DMPlexSetConeSize(idm, p, faceSizeAll);CHKERRQ(ierr);
      }
    } else if (d == cellDepth) {
      for (p = pStart[d]; p < pEnd[d]; ++p) {
        /* Number of cell faces may be different from number of cell vertices*/
        ierr = DMPlexGetFaces_Internal(dm, cellDim, p, &coneSize, NULL, NULL);CHKERRQ(ierr);
        ierr = DMPlexSetConeSize(idm, p, coneSize);CHKERRQ(ierr);
      }
    } else {
      for (p = pStart[d]; p < pEnd[d]; ++p) {
        ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexSetConeSize(idm, p, coneSize);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMSetUp(idm);CHKERRQ(ierr);
  /* Get face cones from subsets of cell vertices */
  if (faceSizeAll > 4) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Do not support interpolation of meshes with faces of %D vertices", faceSizeAll);
  ierr = PetscHashIJKLCreate(&faceTable);CHKERRQ(ierr);
  for (d = depth; d > cellDepth; --d) {
    const PetscInt *cone;
    PetscInt        p;

    for (p = pStart[d]; p < pEnd[d]; ++p) {
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexSetCone(idm, p, cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(idm, p, cone);CHKERRQ(ierr);
    }
  }
  for (c = pStart[cellDepth], face = pStart[faceDepth]; c < pEnd[cellDepth]; ++c) {
    const PetscInt *cellFaces;
    PetscInt        numCellFaces, faceSize, cf;

    ierr = DMPlexGetFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
    if (faceSize != faceSizeAll) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent face for cell %D of size %D != %D", c, faceSize, faceSizeAll);
    for (cf = 0; cf < numCellFaces; ++cf) {
      const PetscInt  *cellFace = &cellFaces[cf*faceSize];
      PetscHashIJKLKey key;
      PetscHashIJKLIter missing, iter;

      if (faceSize == 2) {
        key.i = PetscMin(cellFace[0], cellFace[1]);
        key.j = PetscMax(cellFace[0], cellFace[1]);
        key.k = 0;
        key.l = 0;
      } else {
        key.i = cellFace[0]; key.j = cellFace[1]; key.k = cellFace[2]; key.l = faceSize > 3 ? cellFace[3] : 0;
        ierr = PetscSortInt(faceSize, (PetscInt *) &key);CHKERRQ(ierr);
      }
      ierr = PetscHashIJKLPut(faceTable, key, &missing, &iter);CHKERRQ(ierr);
      if (missing) {
        ierr = DMPlexSetCone(idm, face, cellFace);CHKERRQ(ierr);
        ierr = PetscHashIJKLSet(faceTable, iter, face);CHKERRQ(ierr);
        ierr = DMPlexInsertCone(idm, c, cf, face++);CHKERRQ(ierr);
      } else {
        const PetscInt *cone;
        PetscInt        coneSize, ornt, i, j, f;

        ierr = PetscHashIJKLGet(faceTable, iter, &f);CHKERRQ(ierr);
        ierr = DMPlexInsertCone(idm, c, cf, f);CHKERRQ(ierr);
        /* Orient face: Do not allow reverse orientation at the first vertex */
        ierr = DMPlexGetConeSize(idm, f, &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(idm, f, &cone);CHKERRQ(ierr);
        if (coneSize != faceSize) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid number of face vertices %D for face %D should be %D", coneSize, f, faceSize);
        /* - First find the initial vertex */
        for (i = 0; i < faceSize; ++i) if (cellFace[0] == cone[i]) break;
        /* - Try forward comparison */
        for (j = 0; j < faceSize; ++j) if (cellFace[j] != cone[(i+j)%faceSize]) break;
        if (j == faceSize) {
          if ((faceSize == 2) && (i == 1)) ornt = -2;
          else                             ornt = i;
        } else {
          /* - Try backward comparison */
          for (j = 0; j < faceSize; ++j) if (cellFace[j] != cone[(i+faceSize-j)%faceSize]) break;
          if (j == faceSize) {
            if (i == 0) ornt = -faceSize;
            else        ornt = -i;
          } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not determine face orientation");
        }
        ierr = DMPlexInsertConeOrientation(idm, c, cf, ornt);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreFaces_Internal(dm, cellDim, c, &numCellFaces, &faceSize, &cellFaces);CHKERRQ(ierr);
  }
  if (face != pEnd[faceDepth]) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Invalid number of faces %D should be %D", face-pStart[faceDepth], pEnd[faceDepth]-pStart[faceDepth]);
  ierr = PetscFree2(pStart,pEnd);CHKERRQ(ierr);
  ierr = PetscHashIJKLDestroy(&faceTable);CHKERRQ(ierr);
  ierr = PetscFree2(pStart,pEnd);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(idm);CHKERRQ(ierr);
  ierr = DMPlexStratify(idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This interpolates the PointSF in parallel following local interpolation */
static PetscErrorCode DMPlexInterpolatePointSF(DM dm, PetscSF pointSF, PetscInt depth)
{
  PetscMPIInt        size, rank;
  PetscInt           p, c, d, dof, offset;
  PetscInt           numLeaves, numRoots, candidatesSize, candidatesRemoteSize;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscSFNode       *candidates, *candidatesRemote, *claims;
  PetscSection       candidateSection, candidateSectionRemote, claimSection;
  PetscHashI         leafhash;
  PetscHashIJ        roothash;
  PetscHashIJKey     key;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(pointSF, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (size < 2 || numRoots < 0) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(DMPLEX_InterpolateSF,dm,0,0,0);CHKERRQ(ierr);
  /* Build hashes of points in the SF for efficient lookup */
  PetscHashICreate(leafhash);
  PetscHashIJCreate(&roothash);
  ierr = PetscHashIJSetMultivalued(roothash, PETSC_FALSE);CHKERRQ(ierr);
  for (p = 0; p < numLeaves; ++p) {
    PetscHashIAdd(leafhash, localPoints[p], p);
    key.i = remotePoints[p].index; key.j = remotePoints[p].rank;
    PetscHashIJAdd(roothash, key, p);
  }
  /* Build a section / SFNode array of candidate points in the single-level adjacency of leaves,
     where each candidate is defined by the root entry for the other vertex that defines the edge. */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &candidateSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(candidateSection, 0, numRoots);CHKERRQ(ierr);
  {
    PetscInt leaf, root, idx, a, *adj = NULL;
    for (p = 0; p < numLeaves; ++p) {
      PetscInt adjSize = PETSC_DETERMINE;
      ierr = DMPlexGetAdjacency_Internal(dm, localPoints[p], PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &adjSize, &adj);CHKERRQ(ierr);
      for (a = 0; a < adjSize; ++a) {
        PetscHashIMap(leafhash, adj[a], leaf);
        if (leaf >= 0) {ierr = PetscSectionAddDof(candidateSection, localPoints[p], 1);CHKERRQ(ierr);}
      }
    }
    ierr = PetscSectionSetUp(candidateSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(candidateSection, &candidatesSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(candidatesSize, &candidates);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      PetscInt adjSize = PETSC_DETERMINE;
      ierr = PetscSectionGetOffset(candidateSection, localPoints[p], &offset);CHKERRQ(ierr);
      ierr = DMPlexGetAdjacency_Internal(dm, localPoints[p], PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &adjSize, &adj);CHKERRQ(ierr);
      for (idx = 0, a = 0; a < adjSize; ++a) {
        PetscHashIMap(leafhash, adj[a], root);
        if (root >= 0) candidates[offset+idx++] = remotePoints[root];
      }
    }
    ierr = PetscFree(adj);CHKERRQ(ierr);
  }
  /* Gather candidate section / array pair into the root partition via inverse(multi(pointSF)). */
  {
    PetscSF   sfMulti, sfInverse, sfCandidates;
    PetscInt *remoteOffsets;
    ierr = PetscSFGetMultiSF(pointSF, &sfMulti);CHKERRQ(ierr);
    ierr = PetscSFCreateInverseSF(sfMulti, &sfInverse);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &candidateSectionRemote);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfInverse, candidateSection, &remoteOffsets, candidateSectionRemote);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(sfInverse, candidateSection, remoteOffsets, candidateSectionRemote, &sfCandidates);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(candidateSectionRemote, &candidatesRemoteSize);CHKERRQ(ierr);
    ierr = PetscMalloc1(candidatesRemoteSize, &candidatesRemote);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sfCandidates, MPIU_2INT, candidates, candidatesRemote);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfCandidates, MPIU_2INT, candidates, candidatesRemote);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfInverse);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfCandidates);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
  }
  /* Walk local roots and check for each remote candidate whether we know all required points,
     either from owning it or having a root entry in the point SF. If we do we place a claim
     by replacing the vertex number with our edge ID. */
  {
    PetscInt        idx, root, joinSize, vertices[2];
    const PetscInt *rootdegree, *join = NULL;
    ierr = PetscSFComputeDegreeBegin(pointSF, &rootdegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(pointSF, &rootdegree);CHKERRQ(ierr);
    /* Loop remote edge connections and put in a claim if both vertices are known */
    for (idx = 0, p = 0; p < numRoots; ++p) {
      for (d = 0; d < rootdegree[p]; ++d) {
        ierr = PetscSectionGetDof(candidateSectionRemote, idx, &dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(candidateSectionRemote, idx, &offset);CHKERRQ(ierr);
        for (c = 0; c < dof; ++c) {
          /* We own both vertices, so we claim the edge by replacing vertex with edge */
          if (candidatesRemote[offset+c].rank == rank) {
            vertices[0] = p; vertices[1] = candidatesRemote[offset+c].index;
            ierr = DMPlexGetJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
            if (joinSize == 1) candidatesRemote[offset+c].index = join[0];
            ierr = DMPlexRestoreJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
            continue;
          }
          /* If we own one vertex and share a root with the other, we claim it */
          key.i = candidatesRemote[offset+c].index; key.j = candidatesRemote[offset+c].rank;
          PetscHashIJGet(roothash, key, &root);
          if (root >= 0) {
            vertices[0] = p; vertices[1] = localPoints[root];
            ierr = DMPlexGetJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
            if (joinSize == 1) {
              candidatesRemote[offset+c].index = join[0];
              candidatesRemote[offset+c].rank = rank;
            }
            ierr = DMPlexRestoreJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
          }
        }
        idx++;
      }
    }
  }
  /* Push claims back to receiver via the MultiSF and derive new pointSF mapping on receiver */
  {
    PetscSF         sfMulti, sfClaims, sfPointNew;
    PetscHashI      claimshash;
    PetscInt        size, pStart, pEnd, root, joinSize, numLocalNew;
    PetscInt       *remoteOffsets, *localPointsNew, vertices[2];
    const PetscInt *join = NULL;
    PetscSFNode    *remotePointsNew;
    ierr = PetscSFGetMultiSF(pointSF, &sfMulti);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &claimSection);CHKERRQ(ierr);
    ierr = PetscSFDistributeSection(sfMulti, candidateSectionRemote, &remoteOffsets, claimSection);CHKERRQ(ierr);
    ierr = PetscSFCreateSectionSF(sfMulti, candidateSectionRemote, remoteOffsets, claimSection, &sfClaims);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(claimSection, &size);CHKERRQ(ierr);
    ierr = PetscMalloc1(size, &claims);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sfClaims, MPIU_2INT, candidatesRemote, claims);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfClaims, MPIU_2INT, candidatesRemote, claims);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfClaims);CHKERRQ(ierr);
    ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
    /* Walk the original section of local supports and add an SF entry for each updated item */
    PetscHashICreate(claimshash);
    for (p = 0; p < numRoots; ++p) {
      ierr = PetscSectionGetDof(candidateSection, p, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(candidateSection, p, &offset);CHKERRQ(ierr);
      for (d = 0; d < dof; ++d) {
        if (candidates[offset+d].index != claims[offset+d].index) {
          key.i = candidates[offset+d].index; key.j = candidates[offset+d].rank;
          PetscHashIJGet(roothash, key, &root);
          if (root >= 0) {
            vertices[0] = p; vertices[1] = localPoints[root];
            ierr = DMPlexGetJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
            if (joinSize == 1) PetscHashIAdd(claimshash, join[0], offset+d);
            ierr = DMPlexRestoreJoin(dm, 2, vertices, &joinSize, &join);CHKERRQ(ierr);
          }
        }
      }
    }
    /* Create new pointSF from hashed claims */
    PetscHashISize(claimshash, numLocalNew);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeaves + numLocalNew, &localPointsNew);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeaves + numLocalNew, &remotePointsNew);CHKERRQ(ierr);
    for (p = 0; p < numLeaves; ++p) {
      localPointsNew[p] = localPoints[p];
      remotePointsNew[p].index = remotePoints[p].index;
      remotePointsNew[p].rank = remotePoints[p].rank;
    }
    p = numLeaves;
    ierr = PetscHashIGetKeys(claimshash, &p, localPointsNew);CHKERRQ(ierr);
    ierr = PetscSortInt(numLocalNew,&localPointsNew[numLeaves]);CHKERRQ(ierr);
    for (p = numLeaves; p < numLeaves + numLocalNew; ++p) {
      PetscHashIMap(claimshash, localPointsNew[p], offset);
      remotePointsNew[p] = claims[offset];
    }
    ierr = PetscSFCreate(PetscObjectComm((PetscObject) dm), &sfPointNew);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sfPointNew, pEnd-pStart, numLeaves+numLocalNew, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = DMSetPointSF(dm, sfPointNew);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sfPointNew);CHKERRQ(ierr);
    PetscHashIDestroy(claimshash);
  }
  PetscHashIDestroy(leafhash);
  ierr = PetscHashIJDestroy(&roothash);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&candidateSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&candidateSectionRemote);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&claimSection);CHKERRQ(ierr);
  ierr = PetscFree(candidates);CHKERRQ(ierr);
  ierr = PetscFree(candidatesRemote);CHKERRQ(ierr);
  ierr = PetscFree(claims);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_InterpolateSF,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexInterpolate - Take in a cell-vertex mesh and return one with all intermediate faces, edges, etc.

  Collective on DM

  Input Parameters:
+ dm - The DMPlex object with only cells and vertices
- dmInt - If NULL a new DM is created, otherwise the interpolated DM is put into the given DM

  Output Parameter:
. dmInt - The complete DMPlex object

  Level: intermediate

.keywords: mesh
.seealso: DMPlexUninterpolate(), DMPlexCreateFromCellList()
@*/
PetscErrorCode DMPlexInterpolate(DM dm, DM *dmInt)
{
  DM             idm, odm = dm;
  PetscSF        sfPoint;
  PetscInt       depth, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_Interpolate,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim <= 1) {
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    idm  = dm;
  }
  for (d = 1; d < dim; ++d) {
    /* Create interpolated mesh */
    if ((d == dim-1) && *dmInt) {idm  = *dmInt;}
    else                        {ierr = DMCreate(PetscObjectComm((PetscObject)dm), &idm);CHKERRQ(ierr);}
    ierr = DMSetType(idm, DMPLEX);CHKERRQ(ierr);
    ierr = DMSetDimension(idm, dim);CHKERRQ(ierr);
    if (depth > 0) {
      ierr = DMPlexInterpolateFaces_Internal(odm, 1, idm);CHKERRQ(ierr);
      ierr = DMGetPointSF(odm, &sfPoint);CHKERRQ(ierr);
      ierr = DMPlexInterpolatePointSF(idm, sfPoint, depth);CHKERRQ(ierr);
    }
    if (odm != dm) {ierr = DMDestroy(&odm);CHKERRQ(ierr);}
    odm  = idm;
  }
  *dmInt = idm;
  ierr = PetscLogEventEnd(DMPLEX_Interpolate,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexCopyCoordinates - Copy coordinates from one mesh to another with the same vertices

  Collective on DM

  Input Parameter:
. dmA - The DMPlex object with initial coordinates

  Output Parameter:
. dmB - The DMPlex object with copied coordinates

  Level: intermediate

  Note: This is typically used when adding pieces other than vertices to a mesh

.keywords: mesh
.seealso: DMCopyLabels(), DMGetCoordinates(), DMGetCoordinatesLocal(), DMGetCoordinateDM(), DMGetCoordinateSection()
@*/
PetscErrorCode DMPlexCopyCoordinates(DM dmA, DM dmB)
{
  Vec            coordinatesA, coordinatesB;
  PetscSection   coordSectionA, coordSectionB;
  PetscScalar   *coordsA, *coordsB;
  PetscInt       spaceDim, vStartA, vStartB, vEndA, vEndB, coordSizeB, v, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dmA == dmB) PetscFunctionReturn(0);
  ierr = DMPlexGetDepthStratum(dmA, 0, &vStartA, &vEndA);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dmB, 0, &vStartB, &vEndB);CHKERRQ(ierr);
  if ((vEndA-vStartA) != (vEndB-vStartB)) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The number of vertices in first DM %d != %d in the second DM", vEndA-vStartA, vEndB-vStartB);
  ierr = DMGetCoordinateSection(dmA, &coordSectionA);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dmB, &coordSectionB);CHKERRQ(ierr);
  if (coordSectionA == coordSectionB) PetscFunctionReturn(0);
  if (!coordSectionB) {
    PetscInt dim;

    ierr = PetscSectionCreate(PetscObjectComm((PetscObject) coordSectionA), &coordSectionB);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dmA, &dim);CHKERRQ(ierr);
    ierr = DMSetCoordinateSection(dmB, dim, coordSectionB);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject) coordSectionB);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetNumFields(coordSectionB, 1);CHKERRQ(ierr);
  ierr = PetscSectionGetFieldComponents(coordSectionA, 0, &spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSectionB, 0, spaceDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSectionB, vStartB, vEndB);CHKERRQ(ierr);
  for (v = vStartB; v < vEndB; ++v) {
    ierr = PetscSectionSetDof(coordSectionB, v, spaceDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSectionB, v, 0, spaceDim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSectionB);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSectionB, &coordSizeB);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dmA, &coordinatesA);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinatesB);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinatesB, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinatesB, coordSizeB, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(coordinatesB,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesA, &coordsA);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesB, &coordsB);CHKERRQ(ierr);
  for (v = 0; v < vEndB-vStartB; ++v) {
    for (d = 0; d < spaceDim; ++d) {
      coordsB[v*spaceDim+d] = coordsA[v*spaceDim+d];
    }
  }
  ierr = VecRestoreArray(coordinatesA, &coordsA);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinatesB, &coordsB);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmB, coordinatesB);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinatesB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexUninterpolate - Take in a mesh with all intermediate faces, edges, etc. and return a cell-vertex mesh

  Collective on DM

  Input Parameter:
. dm - The complete DMPlex object

  Output Parameter:
. dmUnint - The DMPlex object with only cells and vertices

  Level: intermediate

.keywords: mesh
.seealso: DMPlexInterpolate(), DMPlexCreateFromCellList()
@*/
PetscErrorCode DMPlexUninterpolate(DM dm, DM *dmUnint)
{
  DM             udm;
  PetscInt       dim, vStart, vEnd, cStart, cEnd, c, maxConeSize = 0, *cone;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim <= 1) {
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
    *dmUnint = dm;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &udm);CHKERRQ(ierr);
  ierr = DMSetType(udm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetDimension(udm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetChart(udm, cStart, vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) ++coneSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = DMPlexSetConeSize(udm, c, coneSize);CHKERRQ(ierr);
    maxConeSize = PetscMax(maxConeSize, coneSize);
  }
  ierr = DMSetUp(udm);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxConeSize, &cone);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt *closure = NULL, closureSize, cl, coneSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt p = closure[cl];

      if ((p >= vStart) && (p < vEnd)) cone[coneSize++] = p;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = DMPlexSetCone(udm, c, cone);CHKERRQ(ierr);
  }
  ierr = PetscFree(cone);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(udm);CHKERRQ(ierr);
  ierr = DMPlexStratify(udm);CHKERRQ(ierr);
  /* Reduce SF */
  {
    PetscSF            sfPoint, sfPointUn;
    const PetscSFNode *remotePoints;
    const PetscInt    *localPoints;
    PetscSFNode       *remotePointsUn;
    PetscInt          *localPointsUn;
    PetscInt           vEnd, numRoots, numLeaves, l;
    PetscInt           numLeavesUn = 0, n = 0;
    PetscErrorCode     ierr;

    /* Get original SF information */
    ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
    ierr = DMGetPointSF(udm, &sfPointUn);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, NULL, &vEnd);CHKERRQ(ierr);
    ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
    /* Allocate space for cells and vertices */
    for (l = 0; l < numLeaves; ++l) if (localPoints[l] < vEnd) numLeavesUn++;
    /* Fill in leaves */
    if (vEnd >= 0) {
      ierr = PetscMalloc1(numLeavesUn, &remotePointsUn);CHKERRQ(ierr);
      ierr = PetscMalloc1(numLeavesUn, &localPointsUn);CHKERRQ(ierr);
      for (l = 0; l < numLeaves; l++) {
        if (localPoints[l] < vEnd) {
          localPointsUn[n]        = localPoints[l];
          remotePointsUn[n].rank  = remotePoints[l].rank;
          remotePointsUn[n].index = remotePoints[l].index;
          ++n;
        }
      }
      if (n != numLeavesUn) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent number of leaves %d != %d", n, numLeavesUn);
      ierr = PetscSFSetGraph(sfPointUn, vEnd, numLeavesUn, localPointsUn, PETSC_OWN_POINTER, remotePointsUn, PETSC_OWN_POINTER);CHKERRQ(ierr);
    }
  }
  *dmUnint = udm;
  PetscFunctionReturn(0);
}
