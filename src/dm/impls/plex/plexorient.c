#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petscsf.h>
#include <petsc/private/hashsetij.h>

/*@
  DMPlexOrientPoint - Act with the given orientation on the cone points of this mesh point, and update its use in the mesh.

  Not Collective

  Input Parameters:
+ dm - The `DM`
. p  - The mesh point
- o  - The orientation

  Level: intermediate

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMPlexOrient()`, `DMPlexGetCone()`, `DMPlexGetConeOrientation()`, `DMPlexInterpolate()`, `DMPlexGetChart()`
@*/
PetscErrorCode DMPlexOrientPoint(DM dm, PetscInt p, PetscInt o)
{
  DMPolytopeType  ct;
  const PetscInt *arr, *cone, *ornt, *support;
  PetscInt       *newcone, *newornt;
  PetscInt        coneSize, c, supportSize, s;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMPlexGetCellType(dm, p, &ct));
  arr = DMPolytopeTypeGetArrangement(ct, o);
  if (!arr) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
  PetscCall(DMPlexGetCone(dm, p, &cone));
  PetscCall(DMPlexGetConeOrientation(dm, p, &ornt));
  PetscCall(DMGetWorkArray(dm, coneSize, MPIU_INT, &newcone));
  PetscCall(DMGetWorkArray(dm, coneSize, MPIU_INT, &newornt));
  for (c = 0; c < coneSize; ++c) {
    DMPolytopeType ft;
    PetscInt       nO;

    PetscCall(DMPlexGetCellType(dm, cone[c], &ft));
    nO         = DMPolytopeTypeGetNumArrangements(ft) / 2;
    newcone[c] = cone[arr[c * 2 + 0]];
    newornt[c] = DMPolytopeTypeComposeOrientation(ft, arr[c * 2 + 1], ornt[arr[c * 2 + 0]]);
    PetscCheck(!newornt[c] || !(newornt[c] >= nO || newornt[c] < -nO), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid orientation %" PetscInt_FMT " not in [%" PetscInt_FMT ",%" PetscInt_FMT ") for %s %" PetscInt_FMT, newornt[c], -nO, nO, DMPolytopeTypes[ft], cone[c]);
  }
  PetscCall(DMPlexSetCone(dm, p, newcone));
  PetscCall(DMPlexSetConeOrientation(dm, p, newornt));
  PetscCall(DMRestoreWorkArray(dm, coneSize, MPIU_INT, &newcone));
  PetscCall(DMRestoreWorkArray(dm, coneSize, MPIU_INT, &newornt));
  /* Update orientation of this point in the support points */
  PetscCall(DMPlexGetSupportSize(dm, p, &supportSize));
  PetscCall(DMPlexGetSupport(dm, p, &support));
  for (s = 0; s < supportSize; ++s) {
    PetscCall(DMPlexGetConeSize(dm, support[s], &coneSize));
    PetscCall(DMPlexGetCone(dm, support[s], &cone));
    PetscCall(DMPlexGetConeOrientation(dm, support[s], &ornt));
    for (c = 0; c < coneSize; ++c) {
      PetscInt po;

      if (cone[c] != p) continue;
      /* ornt[c] * 0 = target = po * o so that po = ornt[c] * o^{-1} */
      po = DMPolytopeTypeComposeOrientationInv(ct, ornt[c], o);
      PetscCall(DMPlexInsertConeOrientation(dm, support[s], c, po));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscInt GetPointIndex(PetscInt point, PetscInt pStart, PetscInt pEnd, const PetscInt points[])
{
  if (points) {
    PetscInt loc;

    PetscCallAbort(PETSC_COMM_SELF, PetscFindInt(point, pEnd - pStart, points, &loc));
    if (loc >= 0) return loc;
  } else {
    if (point >= pStart && point < pEnd) return point - pStart;
  }
  return -1;
}

/*
  - Checks face match
    - Flips non-matching
  - Inserts faces of support cells in FIFO
*/
static PetscErrorCode DMPlexCheckFace_Internal(DM dm, PetscInt *faceFIFO, PetscInt *fTop, PetscInt *fBottom, IS cellIS, IS faceIS, PetscBT seenCells, PetscBT flippedCells, PetscBT seenFaces)
{
  const PetscInt *supp, *coneA, *coneB, *coneOA, *coneOB;
  PetscInt        suppSize, Ns = 0, coneSizeA, coneSizeB, posA = -1, posB = -1;
  PetscInt        face, dim, indC[3], indS[3], seenA, flippedA, seenB, flippedB, mismatch;
  const PetscInt *cells, *faces;
  PetscInt        cStart, cEnd, fStart, fEnd;

  PetscFunctionBegin;
  face = faceFIFO[(*fTop)++];
  PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(DMPlexGetPointDepth(dm, cells ? cells[cStart] : cStart, &dim));
  PetscCall(DMPlexGetSupportSize(dm, face, &suppSize));
  PetscCall(DMPlexGetSupport(dm, face, &supp));
  // Filter the support
  for (PetscInt s = 0; s < suppSize; ++s) {
    // Filter support
    indC[Ns] = GetPointIndex(supp[s], cStart, cEnd, cells);
    indS[Ns] = s;
    if (indC[Ns] >= 0) ++Ns;
  }
  if (Ns < 2) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(Ns == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Faces should separate only two cells, not %" PetscInt_FMT, Ns);
  PetscCheck(indC[0] >= 0 && indC[1] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Support cells %" PetscInt_FMT " (%" PetscInt_FMT ") and %" PetscInt_FMT " (%" PetscInt_FMT ") are not both valid", supp[0], indC[0], supp[1], indC[1]);
  seenA    = PetscBTLookup(seenCells, indC[0]);
  flippedA = PetscBTLookup(flippedCells, indC[0]) ? 1 : 0;
  seenB    = PetscBTLookup(seenCells, indC[1]);
  flippedB = PetscBTLookup(flippedCells, indC[1]) ? 1 : 0;

  PetscCall(DMPlexGetConeSize(dm, supp[indS[0]], &coneSizeA));
  PetscCall(DMPlexGetConeSize(dm, supp[indS[1]], &coneSizeB));
  PetscCall(DMPlexGetCone(dm, supp[indS[0]], &coneA));
  PetscCall(DMPlexGetCone(dm, supp[indS[1]], &coneB));
  PetscCall(DMPlexGetConeOrientation(dm, supp[indS[0]], &coneOA));
  PetscCall(DMPlexGetConeOrientation(dm, supp[indS[1]], &coneOB));
  for (PetscInt c = 0; c < coneSizeA; ++c) {
    const PetscInt indF = GetPointIndex(coneA[c], fStart, fEnd, faces);

    // Filter cone
    if (indF < 0) continue;
    if (!PetscBTLookup(seenFaces, indF)) {
      faceFIFO[(*fBottom)++] = coneA[c];
      PetscCall(PetscBTSet(seenFaces, indF));
    }
    if (coneA[c] == face) posA = c;
    PetscCheck(*fBottom <= fEnd - fStart, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %" PetscInt_FMT " was pushed exceeding capacity %" PetscInt_FMT " > %" PetscInt_FMT, coneA[c], *fBottom, fEnd - fStart);
  }
  PetscCheck(posA >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " could not be located in cell %" PetscInt_FMT, face, supp[indS[0]]);
  for (PetscInt c = 0; c < coneSizeB; ++c) {
    const PetscInt indF = GetPointIndex(coneB[c], fStart, fEnd, faces);

    // Filter cone
    if (indF < 0) continue;
    if (!PetscBTLookup(seenFaces, indF)) {
      faceFIFO[(*fBottom)++] = coneB[c];
      PetscCall(PetscBTSet(seenFaces, indF));
    }
    if (coneB[c] == face) posB = c;
    PetscCheck(*fBottom <= fEnd - fStart, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %" PetscInt_FMT " was pushed exceeding capacity %" PetscInt_FMT " > %" PetscInt_FMT, coneA[c], *fBottom, fEnd - fStart);
  }
  PetscCheck(posB >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " could not be located in cell %" PetscInt_FMT, face, supp[indS[1]]);

  if (dim == 1) {
    mismatch = posA == posB;
  } else {
    mismatch = coneOA[posA] == coneOB[posB];
  }

  if (mismatch ^ (flippedA ^ flippedB)) {
    PetscCheck(!seenA || !seenB, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Previously seen cells %" PetscInt_FMT " and %" PetscInt_FMT " do not match: Fault mesh is non-orientable", supp[indS[0]], supp[indS[1]]);
    if (!seenA && !flippedA) PetscCall(PetscBTSet(flippedCells, indC[0]));
    else {
      PetscCheck(!seenB && !flippedB, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent mesh orientation: Fault mesh is non-orientable");
      PetscCall(PetscBTSet(flippedCells, indC[1]));
    }
  } else PetscCheck(!mismatch || !flippedA || !flippedB, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to flip already flipped cell: Fault mesh is non-orientable");
  PetscCall(PetscBTSet(seenCells, indC[0]));
  PetscCall(PetscBTSet(seenCells, indC[1]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCheckFace_Old_Internal(DM dm, PetscInt *faceFIFO, PetscInt *fTop, PetscInt *fBottom, PetscInt cStart, PetscInt fStart, PetscInt fEnd, PetscBT seenCells, PetscBT flippedCells, PetscBT seenFaces)
{
  const PetscInt *support, *coneA, *coneB, *coneOA, *coneOB;
  PetscInt        supportSize, coneSizeA, coneSizeB, posA = -1, posB = -1;
  PetscInt        face, dim, seenA, flippedA, seenB, flippedB, mismatch, c;

  PetscFunctionBegin;
  face = faceFIFO[(*fTop)++];
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetSupportSize(dm, face, &supportSize));
  PetscCall(DMPlexGetSupport(dm, face, &support));
  if (supportSize < 2) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(supportSize == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Faces should separate only two cells, not %" PetscInt_FMT, supportSize);
  seenA    = PetscBTLookup(seenCells, support[0] - cStart);
  flippedA = PetscBTLookup(flippedCells, support[0] - cStart) ? 1 : 0;
  seenB    = PetscBTLookup(seenCells, support[1] - cStart);
  flippedB = PetscBTLookup(flippedCells, support[1] - cStart) ? 1 : 0;

  PetscCall(DMPlexGetConeSize(dm, support[0], &coneSizeA));
  PetscCall(DMPlexGetConeSize(dm, support[1], &coneSizeB));
  PetscCall(DMPlexGetCone(dm, support[0], &coneA));
  PetscCall(DMPlexGetCone(dm, support[1], &coneB));
  PetscCall(DMPlexGetConeOrientation(dm, support[0], &coneOA));
  PetscCall(DMPlexGetConeOrientation(dm, support[1], &coneOB));
  for (c = 0; c < coneSizeA; ++c) {
    if (!PetscBTLookup(seenFaces, coneA[c] - fStart)) {
      faceFIFO[(*fBottom)++] = coneA[c];
      PetscCall(PetscBTSet(seenFaces, coneA[c] - fStart));
    }
    if (coneA[c] == face) posA = c;
    PetscCheck(*fBottom <= fEnd - fStart, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %" PetscInt_FMT " was pushed exceeding capacity %" PetscInt_FMT " > %" PetscInt_FMT, coneA[c], *fBottom, fEnd - fStart);
  }
  PetscCheck(posA >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " could not be located in cell %" PetscInt_FMT, face, support[0]);
  for (c = 0; c < coneSizeB; ++c) {
    if (!PetscBTLookup(seenFaces, coneB[c] - fStart)) {
      faceFIFO[(*fBottom)++] = coneB[c];
      PetscCall(PetscBTSet(seenFaces, coneB[c] - fStart));
    }
    if (coneB[c] == face) posB = c;
    PetscCheck(*fBottom <= fEnd - fStart, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Face %" PetscInt_FMT " was pushed exceeding capacity %" PetscInt_FMT " > %" PetscInt_FMT, coneA[c], *fBottom, fEnd - fStart);
  }
  PetscCheck(posB >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " could not be located in cell %" PetscInt_FMT, face, support[1]);

  if (dim == 1) {
    mismatch = posA == posB;
  } else {
    mismatch = coneOA[posA] == coneOB[posB];
  }

  if (mismatch ^ (flippedA ^ flippedB)) {
    PetscCheck(!seenA || !seenB, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Previously seen cells %" PetscInt_FMT " and %" PetscInt_FMT " do not match: Fault mesh is non-orientable", support[0], support[1]);
    if (!seenA && !flippedA) {
      PetscCall(PetscBTSet(flippedCells, support[0] - cStart));
    } else if (!seenB && !flippedB) {
      PetscCall(PetscBTSet(flippedCells, support[1] - cStart));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent mesh orientation: Fault mesh is non-orientable");
  } else PetscCheck(!mismatch || !flippedA || !flippedB, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to flip already flipped cell: Fault mesh is non-orientable");
  PetscCall(PetscBTSet(seenCells, support[0] - cStart));
  PetscCall(PetscBTSet(seenCells, support[1] - cStart));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexOrient_Serial - Compute valid orientation for local connected components

  Not collective

  Input Parameters:
  + dm         - The `DM`
  - cellHeight - The height of k-cells to be oriented

  Output Parameters:
  + Ncomp        - The number of connected component
  . cellComp     - The connected component for each local cell
  . faceComp     - The connected component for each local face
  - flippedCells - Marked cells should be inverted

  Level: developer

.seealso: `DMPlexOrient()`
*/
static PetscErrorCode DMPlexOrient_Serial(DM dm, IS cellIS, IS faceIS, PetscInt *Ncomp, PetscInt cellComp[], PetscInt faceComp[], PetscBT flippedCells)
{
  PetscBT         seenCells, seenFaces;
  PetscInt       *faceFIFO;
  const PetscInt *cells = NULL, *faces = NULL;
  PetscInt        cStart = 0, cEnd = 0, fStart = 0, fEnd = 0;

  PetscFunctionBegin;
  /* Truth Table
     mismatch    flips   do action   mismatch   flipA ^ flipB   action
         F       0 flips     no         F             F           F
         F       1 flip      yes        F             T           T
         F       2 flips     no         T             F           T
         T       0 flips     yes        T             T           F
         T       1 flip      no
         T       2 flips     yes
  */
  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(PetscBTCreate(cEnd - cStart, &seenCells));
  PetscCall(PetscBTMemzero(cEnd - cStart, seenCells));
  PetscCall(PetscBTCreate(fEnd - fStart, &seenFaces));
  PetscCall(PetscBTMemzero(fEnd - fStart, seenFaces));
  PetscCall(PetscMalloc1(fEnd - fStart, &faceFIFO));
  *Ncomp = 0;
  for (PetscInt c = 0; c < cEnd - cStart; ++c) cellComp[c] = -1;
  do {
    PetscInt cc, fTop, fBottom;

    // Look for first unmarked cell
    for (cc = cStart; cc < cEnd; ++cc)
      if (cellComp[cc - cStart] < 0) break;
    if (cc >= cEnd) break;
    // Initialize FIFO with first cell in component
    {
      const PetscInt  cell = cells ? cells[cc] : cc;
      const PetscInt *cone;
      PetscInt        coneSize;

      fTop = fBottom = 0;
      PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
      PetscCall(DMPlexGetCone(dm, cell, &cone));
      for (PetscInt c = 0; c < coneSize; ++c) {
        const PetscInt idx = GetPointIndex(cone[c], fStart, fEnd, faces);

        // Cell faces are guaranteed to be in the face set
        PetscCheck(idx >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " of cell %" PetscInt_FMT " is not present in the label", cone[c], cell);
        faceFIFO[fBottom++] = cone[c];
        PetscCall(PetscBTSet(seenFaces, idx));
      }
      PetscCall(PetscBTSet(seenCells, cc - cStart));
    }
    // Consider each face in FIFO
    while (fTop < fBottom) PetscCall(DMPlexCheckFace_Internal(dm, faceFIFO, &fTop, &fBottom, cellIS, faceIS, seenCells, flippedCells, seenFaces));
    // Set component for cells and faces
    for (PetscInt c = 0; c < cEnd - cStart; ++c) {
      if (PetscBTLookup(seenCells, c)) cellComp[c] = *Ncomp;
    }
    for (PetscInt f = 0; f < fEnd - fStart; ++f) {
      if (PetscBTLookup(seenFaces, f)) faceComp[f] = *Ncomp;
    }
    // Wipe seenCells and seenFaces for next component
    PetscCall(PetscBTMemzero(fEnd - fStart, seenFaces));
    PetscCall(PetscBTMemzero(cEnd - cStart, seenCells));
    ++(*Ncomp);
  } while (1);
  PetscCall(PetscBTDestroy(&seenCells));
  PetscCall(PetscBTDestroy(&seenFaces));
  PetscCall(PetscFree(faceFIFO));
  for (PetscInt c = 0; c < cEnd - cStart; ++c)
    PetscCheck(0 <= cellComp[c] && cellComp[c] < *Ncomp, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid component %" PetscInt_FMT " for cell %" PetscInt_FMT " (%" PetscInt_FMT ")", cellComp[c], cells ? cells[c] : c, c);
  for (PetscInt f = 0; f < fEnd - fStart; ++f)
    PetscCheck(0 <= faceComp[f] && faceComp[f] < *Ncomp, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid component %" PetscInt_FMT " for face %" PetscInt_FMT " (%" PetscInt_FMT ")", faceComp[f], faces ? faces[f] : f, f);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexOrient - Give a consistent orientation to the input mesh

  Input Parameter:
. dm - The `DM`

  Notes:
  The orientation data for the `DM` are changed in-place.

  This routine will fail for non-orientable surfaces, such as the Moebius strip.

  Level: advanced

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMCreate()`, `DMPlexOrientLabel()`
@*/
PetscErrorCode DMPlexOrient(DM dm)
{
#if 0
  IS cellIS, faceIS;

  PetscFunctionBegin;
  PetscCall(DMPlexGetAllCells_Internal(dm, &cellIS));
  PetscCall(DMPlexGetAllFaces_Internal(dm, &faceIS));
  PetscCall(DMPlexOrientCells_Internal(dm, cellIS, faceIS));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(ISDestroy(&faceIS));
  PetscFunctionReturn(PETSC_SUCCESS);
#else
  MPI_Comm           comm;
  PetscSF            sf;
  const PetscInt    *lpoints;
  const PetscSFNode *rpoints;
  PetscSFNode       *rorntComp = NULL, *lorntComp = NULL;
  PetscInt          *numNeighbors, **neighbors, *locSupport = NULL;
  PetscSFNode       *nrankComp;
  PetscBool         *match, *flipped;
  PetscBT            seenCells, flippedCells, seenFaces;
  PetscInt          *faceFIFO, fTop, fBottom, *cellComp, *faceComp;
  PetscInt           numLeaves, numRoots, dim, h, cStart, cEnd, c, cell, fStart, fEnd, face, off, totNeighbors = 0;
  PetscMPIInt        rank, size, numComponents, comp = 0;
  PetscBool          flg, flg2;
  PetscViewer        viewer = NULL, selfviewer = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscOptionsHasName(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-orientation_view", &flg));
  PetscCall(PetscOptionsHasName(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-orientation_view_synchronized", &flg2));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, &numRoots, &numLeaves, &lpoints, &rpoints));
  /* Truth Table
     mismatch    flips   do action   mismatch   flipA ^ flipB   action
         F       0 flips     no         F             F           F
         F       1 flip      yes        F             T           T
         F       2 flips     no         T             F           T
         T       0 flips     yes        T             T           F
         T       1 flip      no
         T       2 flips     yes
  */
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetVTKCellHeight(dm, &h));
  PetscCall(DMPlexGetHeightStratum(dm, h, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(dm, h + 1, &fStart, &fEnd));
  PetscCall(PetscBTCreate(cEnd - cStart, &seenCells));
  PetscCall(PetscBTMemzero(cEnd - cStart, seenCells));
  PetscCall(PetscBTCreate(cEnd - cStart, &flippedCells));
  PetscCall(PetscBTMemzero(cEnd - cStart, flippedCells));
  PetscCall(PetscBTCreate(fEnd - fStart, &seenFaces));
  PetscCall(PetscBTMemzero(fEnd - fStart, seenFaces));
  PetscCall(PetscCalloc3(fEnd - fStart, &faceFIFO, cEnd - cStart, &cellComp, fEnd - fStart, &faceComp));
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
   - Use Scatterv to send back flipped flags for each component
   - Negate flippedCells by component

   NEW STYLE
   - Create the adj on each process
   - Bootstrap to complete graph on proc 0
  */
  /* Loop over components */
  for (cell = cStart; cell < cEnd; ++cell) cellComp[cell - cStart] = -1;
  do {
    /* Look for first unmarked cell */
    for (cell = cStart; cell < cEnd; ++cell)
      if (cellComp[cell - cStart] < 0) break;
    if (cell >= cEnd) break;
    /* Initialize FIFO with first cell in component */
    {
      const PetscInt *cone;
      PetscInt        coneSize;

      fTop = fBottom = 0;
      PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
      PetscCall(DMPlexGetCone(dm, cell, &cone));
      for (c = 0; c < coneSize; ++c) {
        faceFIFO[fBottom++] = cone[c];
        PetscCall(PetscBTSet(seenFaces, cone[c] - fStart));
      }
      PetscCall(PetscBTSet(seenCells, cell - cStart));
    }
    /* Consider each face in FIFO */
    while (fTop < fBottom) PetscCall(DMPlexCheckFace_Old_Internal(dm, faceFIFO, &fTop, &fBottom, cStart, fStart, fEnd, seenCells, flippedCells, seenFaces));
    /* Set component for cells and faces */
    for (cell = 0; cell < cEnd - cStart; ++cell) {
      if (PetscBTLookup(seenCells, cell)) cellComp[cell] = comp;
    }
    for (face = 0; face < fEnd - fStart; ++face) {
      if (PetscBTLookup(seenFaces, face)) faceComp[face] = comp;
    }
    /* Wipe seenCells and seenFaces for next component */
    PetscCall(PetscBTMemzero(fEnd - fStart, seenFaces));
    PetscCall(PetscBTMemzero(cEnd - cStart, seenCells));
    ++comp;
  } while (1);
  numComponents = comp;
  if (flg) {
    PetscViewer v;

    PetscCall(PetscViewerASCIIGetStdout(comm, &v));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
    PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for serial flipped cells:\n", rank));
    PetscCall(PetscBTView(cEnd - cStart, flippedCells, v));
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  }
  /* Now all subdomains are oriented, but we need a consistent parallel orientation */
  if (numLeaves >= 0) {
    PetscInt maxSupportSize, neighbor;

    /* Store orientations of boundary faces*/
    PetscCall(DMPlexGetMaxSizes(dm, NULL, &maxSupportSize));
    PetscCall(PetscCalloc3(numRoots, &rorntComp, numRoots, &lorntComp, maxSupportSize, &locSupport));
    for (face = fStart; face < fEnd; ++face) {
      const PetscInt *cone, *support, *ornt;
      PetscInt        coneSize, supportSize, Ns = 0, s, l;

      PetscCall(DMPlexGetSupportSize(dm, face, &supportSize));
      /* Ignore overlapping cells */
      PetscCall(DMPlexGetSupport(dm, face, &support));
      for (s = 0; s < supportSize; ++s) {
        if (lpoints) PetscCall(PetscFindInt(support[s], numLeaves, lpoints, &l));
        else {
          if (support[s] >= 0 && support[s] < numLeaves) l = support[s];
          else l = -1;
        }
        if (l >= 0) continue;
        locSupport[Ns++] = support[s];
      }
      if (Ns != 1) continue;
      neighbor = locSupport[0];
      PetscCall(DMPlexGetCone(dm, neighbor, &cone));
      PetscCall(DMPlexGetConeSize(dm, neighbor, &coneSize));
      PetscCall(DMPlexGetConeOrientation(dm, neighbor, &ornt));
      for (c = 0; c < coneSize; ++c)
        if (cone[c] == face) break;
      if (dim == 1) {
        /* Use cone position instead, shifted to -1 or 1 */
        if (PetscBTLookup(flippedCells, neighbor - cStart)) rorntComp[face].rank = 1 - c * 2;
        else rorntComp[face].rank = c * 2 - 1;
      } else {
        if (PetscBTLookup(flippedCells, neighbor - cStart)) rorntComp[face].rank = ornt[c] < 0 ? -1 : 1;
        else rorntComp[face].rank = ornt[c] < 0 ? 1 : -1;
      }
      rorntComp[face].index = faceComp[face - fStart];
    }
    /* Communicate boundary edge orientations */
    PetscCall(PetscSFBcastBegin(sf, MPIU_SF_NODE, rorntComp, lorntComp, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_SF_NODE, rorntComp, lorntComp, MPI_REPLACE));
  }
  /* Get process adjacency */
  PetscCall(PetscMalloc2(numComponents, &numNeighbors, numComponents, &neighbors));
  viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm));
  if (flg2) PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
  for (comp = 0; comp < numComponents; ++comp) {
    PetscInt n;

    numNeighbors[comp] = 0;
    PetscCall(PetscMalloc1(PetscMax(numLeaves, 0), &neighbors[comp]));
    /* I know this is p^2 time in general, but for bounded degree its alright */
    for (PetscInt l = 0; l < numLeaves; ++l) {
      const PetscInt face = lpoints ? lpoints[l] : l;

      /* Find a representative face (edge) separating pairs of procs */
      if ((face >= fStart) && (face < fEnd) && (faceComp[face - fStart] == comp) && rorntComp[face].rank) {
        const PetscInt rrank = rpoints[l].rank;
        const PetscInt rcomp = lorntComp[face].index;

        for (n = 0; n < numNeighbors[comp]; ++n)
          if ((rrank == rpoints[neighbors[comp][n]].rank) && (rcomp == lorntComp[lpoints[neighbors[comp][n]]].index)) break;
        if (n >= numNeighbors[comp]) {
          PetscInt supportSize;

          PetscCall(DMPlexGetSupportSize(dm, face, &supportSize));
          // We can have internal faces in the SF if we have cells in the SF
          if (supportSize > 1) continue;
          if (flg)
            PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]: component %d, Found representative leaf %" PetscInt_FMT " (face %" PetscInt_FMT ") connecting to face %" PetscInt_FMT " on (%" PetscInt_FMT ", %" PetscInt_FMT ") with orientation %" PetscInt_FMT "\n", rank, comp, l, face,
                                             rpoints[l].index, rrank, rcomp, lorntComp[face].rank));
          neighbors[comp][numNeighbors[comp]++] = l;
        }
      }
    }
    totNeighbors += numNeighbors[comp];
  }
  PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
  if (flg2) PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscMalloc2(totNeighbors, &nrankComp, totNeighbors, &match));
  for (comp = 0, off = 0; comp < numComponents; ++comp) {
    PetscInt n;

    for (n = 0; n < numNeighbors[comp]; ++n, ++off) {
      const PetscInt face = lpoints ? lpoints[neighbors[comp][n]] : neighbors[comp][n];
      const PetscInt o    = rorntComp[face].rank * lorntComp[face].rank;

      if (o < 0) match[off] = PETSC_TRUE;
      else if (o > 0) match[off] = PETSC_FALSE;
      else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid face %" PetscInt_FMT " (%" PetscInt_FMT ", %" PetscInt_FMT ") neighbor: %" PetscInt_FMT " comp: %d", face, rorntComp[face].rank, lorntComp[face].rank, neighbors[comp][n], comp);
      nrankComp[off].rank  = rpoints[neighbors[comp][n]].rank;
      nrankComp[off].index = lorntComp[lpoints ? lpoints[neighbors[comp][n]] : neighbors[comp][n]].index;
    }
    PetscCall(PetscFree(neighbors[comp]));
  }
  /* Collect the graph on 0 */
  if (numLeaves >= 0) {
    Mat          G;
    PetscBT      seenProcs, flippedProcs;
    PetscInt    *procFIFO, pTop, pBottom;
    PetscInt    *N          = NULL, *Noff;
    PetscSFNode *adj        = NULL;
    PetscBool   *val        = NULL;
    PetscMPIInt *recvcounts = NULL, *displs = NULL, *Nc, p, o, itotNeighbors;
    PetscMPIInt  size = 0;

    PetscCall(PetscCalloc1(numComponents, &flipped));
    if (rank == 0) PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCall(PetscCalloc4(size, &recvcounts, size + 1, &displs, size, &Nc, size + 1, &Noff));
    PetscCallMPI(MPI_Gather(&numComponents, 1, MPI_INT, Nc, 1, MPI_INT, 0, comm));
    for (p = 0; p < size; ++p) displs[p + 1] = displs[p] + Nc[p];
    if (rank == 0) PetscCall(PetscMalloc1(displs[size], &N));
    PetscCallMPI(MPI_Gatherv(numNeighbors, numComponents, MPIU_INT, N, Nc, displs, MPIU_INT, 0, comm));
    for (p = 0, o = 0; p < size; ++p) {
      recvcounts[p] = 0;
      for (c = 0; c < Nc[p]; ++c, ++o) recvcounts[p] += N[o];
      displs[p + 1] = displs[p] + recvcounts[p];
    }
    if (rank == 0) PetscCall(PetscMalloc2(displs[size], &adj, displs[size], &val));
    PetscCall(PetscMPIIntCast(totNeighbors, &itotNeighbors));
    PetscCallMPI(MPI_Gatherv(nrankComp, itotNeighbors, MPIU_SF_NODE, adj, recvcounts, displs, MPIU_SF_NODE, 0, comm));
    PetscCallMPI(MPI_Gatherv(match, itotNeighbors, MPI_C_BOOL, val, recvcounts, displs, MPI_C_BOOL, 0, comm));
    PetscCall(PetscFree2(numNeighbors, neighbors));
    if (rank == 0) {
      for (p = 1; p <= size; ++p) Noff[p] = Noff[p - 1] + Nc[p - 1];
      if (flg) {
        PetscInt n;

        for (p = 0, off = 0; p < size; ++p) {
          for (c = 0; c < Nc[p]; ++c) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "Proc %d Comp %" PetscInt_FMT ":\n", p, c));
            for (n = 0; n < N[Noff[p] + c]; ++n, ++off) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  edge (%" PetscInt_FMT ", %" PetscInt_FMT ") (%s):\n", adj[off].rank, adj[off].index, PetscBools[val[off]]));
          }
        }
      }
      /* Symmetrize the graph */
      PetscCall(MatCreate(PETSC_COMM_SELF, &G));
      PetscCall(MatSetSizes(G, Noff[size], Noff[size], Noff[size], Noff[size]));
      PetscCall(MatSetUp(G));
      for (p = 0, off = 0; p < size; ++p) {
        for (c = 0; c < Nc[p]; ++c) {
          const PetscInt r = Noff[p] + c;

          for (PetscInt n = 0; n < N[r]; ++n, ++off) {
            const PetscInt    q = Noff[adj[off].rank] + adj[off].index;
            const PetscScalar o = val[off] ? 1.0 : 0.0;

            PetscCall(MatSetValues(G, 1, &r, 1, &q, &o, INSERT_VALUES));
            PetscCall(MatSetValues(G, 1, &q, 1, &r, &o, INSERT_VALUES));
          }
        }
      }
      PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));

      PetscCall(PetscBTCreate(Noff[size], &seenProcs));
      PetscCall(PetscBTMemzero(Noff[size], seenProcs));
      PetscCall(PetscBTCreate(Noff[size], &flippedProcs));
      PetscCall(PetscBTMemzero(Noff[size], flippedProcs));
      PetscCall(PetscMalloc1(Noff[size], &procFIFO));
      pTop = pBottom = 0;
      for (p = 0; p < Noff[size]; ++p) {
        if (PetscBTLookup(seenProcs, p)) continue;
        /* Initialize FIFO with next proc */
        procFIFO[pBottom++] = p;
        PetscCall(PetscBTSet(seenProcs, p));
        /* Consider each proc in FIFO */
        while (pTop < pBottom) {
          const PetscScalar *ornt;
          const PetscInt    *neighbors;
          PetscInt           proc, nproc, seen, flippedA, flippedB, mismatch, numNeighbors, n;

          proc     = procFIFO[pTop++];
          flippedA = PetscBTLookup(flippedProcs, proc) ? 1 : 0;
          PetscCall(MatGetRow(G, proc, &numNeighbors, &neighbors, &ornt));
          /* Loop over neighboring procs */
          for (n = 0; n < numNeighbors; ++n) {
            nproc    = neighbors[n];
            mismatch = PetscRealPart(ornt[n]) > 0.5 ? 0 : 1;
            seen     = PetscBTLookup(seenProcs, nproc);
            flippedB = PetscBTLookup(flippedProcs, nproc) ? 1 : 0;

            if (mismatch ^ (flippedA ^ flippedB)) {
              PetscCheck(!seen, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Previously seen procs %" PetscInt_FMT " and %" PetscInt_FMT " do not match: Fault mesh is non-orientable", proc, nproc);
              PetscCheck(!flippedB, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent mesh orientation: Fault mesh is non-orientable");
              PetscCall(PetscBTSet(flippedProcs, nproc));
            } else PetscCheck(!mismatch || !flippedA || !flippedB, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to flip already flipped cell: Fault mesh is non-orientable");
            if (!seen) {
              procFIFO[pBottom++] = nproc;
              PetscCall(PetscBTSet(seenProcs, nproc));
            }
          }
        }
      }
      PetscCall(PetscFree(procFIFO));
      PetscCall(MatDestroy(&G));
      PetscCall(PetscFree2(adj, val));
      PetscCall(PetscBTDestroy(&seenProcs));
    }
    /* Scatter flip flags */
    {
      PetscBool *flips = NULL;

      if (rank == 0) {
        PetscCall(PetscMalloc1(Noff[size], &flips));
        for (p = 0; p < Noff[size]; ++p) {
          flips[p] = PetscBTLookup(flippedProcs, p) ? PETSC_TRUE : PETSC_FALSE;
          if (flg && flips[p]) PetscCall(PetscPrintf(comm, "Flipping Proc+Comp %d:\n", p));
        }
        for (p = 0; p < size; ++p) displs[p + 1] = displs[p] + Nc[p];
      }
      PetscCallMPI(MPI_Scatterv(flips, Nc, displs, MPI_C_BOOL, flipped, numComponents, MPI_C_BOOL, 0, comm));
      PetscCall(PetscFree(flips));
    }
    if (rank == 0) PetscCall(PetscBTDestroy(&flippedProcs));
    PetscCall(PetscFree(N));
    PetscCall(PetscFree4(recvcounts, displs, Nc, Noff));
    PetscCall(PetscFree2(nrankComp, match));

    /* Decide whether to flip cells in each component */
    for (c = 0; c < cEnd - cStart; ++c) {
      if (flipped[cellComp[c]]) PetscCall(PetscBTNegate(flippedCells, c));
    }
    PetscCall(PetscFree(flipped));
  }
  if (flg) {
    PetscViewer v;

    PetscCall(PetscViewerASCIIGetStdout(comm, &v));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
    PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for parallel flipped cells:\n", rank));
    PetscCall(PetscBTView(cEnd - cStart, flippedCells, v));
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  }
  /* Reverse flipped cells in the mesh */
  for (c = cStart; c < cEnd; ++c) {
    if (PetscBTLookup(flippedCells, c - cStart)) PetscCall(DMPlexOrientPoint(dm, c, -1));
  }
  PetscCall(PetscBTDestroy(&seenCells));
  PetscCall(PetscBTDestroy(&flippedCells));
  PetscCall(PetscBTDestroy(&seenFaces));
  PetscCall(PetscFree2(numNeighbors, neighbors));
  PetscCall(PetscFree3(rorntComp, lorntComp, locSupport));
  PetscCall(PetscFree3(faceFIFO, cellComp, faceComp));
  PetscFunctionReturn(PETSC_SUCCESS);
#endif
}

static PetscErrorCode CreateCellAndFaceIS_Private(DM dm, DMLabel label, IS *cellIS, IS *faceIS)
{
  IS              valueIS;
  const PetscInt *values;
  PetscInt        Nv, depth = 0;

  PetscFunctionBegin;
  PetscCall(DMLabelGetValueIS(label, &valueIS));
  PetscCall(ISGetLocalSize(valueIS, &Nv));
  PetscCall(ISGetIndices(valueIS, &values));
  for (PetscInt v = 0; v < Nv; ++v) {
    const PetscInt val = values[v] < 0 || values[v] >= 100 ? 0 : values[v];
    PetscInt       n;

    PetscCall(DMLabelGetStratumSize(label, val, &n));
    if (!n) continue;
    depth = PetscMax(val, depth);
  }
  PetscCall(ISDestroy(&valueIS));
  PetscCheck(depth >= 1 || !Nv, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Depth for interface must be at least 1, not %" PetscInt_FMT, depth);
  PetscCall(DMLabelGetStratumIS(label, depth, cellIS));
  PetscCall(DMLabelGetStratumIS(label, depth - 1, faceIS));
  if (!*cellIS) PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, cellIS));
  if (!*faceIS) PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, faceIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexOrientLabel - Give a consistent orientation to the hypersurface marked by the `DMLabel` in the input mesh

  Collective on dm

  Input Parameters:
+ dm    - The `DM`
- label - The `DMLabel`

  Notes:
  The orientation data for the `DM` are changed in-place.

  This routine will fail for non-orientable surfaces, such as the Moebius strip.

  Level: advanced

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMCreate()`, `DMPlexOrient()`
@*/
PetscErrorCode DMPlexOrientLabel(DM dm, DMLabel label)
{
  IS cellIS, faceIS;

  PetscFunctionBegin;
  PetscCall(CreateCellAndFaceIS_Private(dm, label, &cellIS, &faceIS));
  PetscCall(DMPlexOrientCells_Internal(dm, cellIS, faceIS));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(ISDestroy(&faceIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt comp; // connected component number
  PetscInt cell; // cell determining face orientation
  PetscInt ornt; // face orientation
} SharedFace;

static PetscErrorCode DMPlexOrientCreateSharedFaces_Internal(DM dm, IS cellIS, IS faceIS, PetscInt Ncomp, const PetscInt faceComp[], PetscBT cellFlip, SharedFace **localFace, SharedFace **remoteFace)
{
  const PetscInt     debug = ((DM_Plex *)dm->data)->printOrient;
  const PetscInt    *cells = NULL, *faces = NULL;
  PetscInt           cStart = 0, cEnd = 0, fStart = 0, fEnd = 0;
  PetscSF            sf;
  const PetscInt    *lpoints, *rootdegree;
  const PetscSFNode *rpoints;
  PetscInt           Nr, Nl;
  PetscInt           depth, fdepth;
  PetscBool          faceIsVertex = PETSC_FALSE;
  MPI_Datatype       MPIU_3INT;
  PetscViewer        viewer = NULL, selfviewer = NULL;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, &Nr, &Nl, &lpoints, &rpoints));
  PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegree));
  PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegree));
  PetscCall(PetscMalloc2(Nr, localFace, Nr, remoteFace));
  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(DMPlexGetPointDepth(dm, faces ? faces[fStart] : fStart, &fdepth));
  if (!fdepth) faceIsVertex = PETSC_TRUE;
  for (PetscInt r = 0; r < Nr; ++r) {
    (*localFace)[r].comp  = -1;
    (*remoteFace)[r].comp = -1;
  }
  // Get information for shared faces
  if (debug) {
    viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
  }
  for (PetscInt f = fStart; f < fEnd; ++f) {
    const PetscInt  face = faces ? faces[f] : f;
    const PetscInt *supp, *cone, *ornt;
    PetscInt        lf, sS, cS, cind = -1, c;

    (*localFace)[face].comp = -1;
    (*localFace)[face].cell = -1;
    (*localFace)[face].ornt = 0;
    PetscCall(PetscFindInt(face, Nl, lpoints, &lf));
    if (!rootdegree[face] && lf < 0) continue;
    PetscCall(DMPlexGetSupportSize(dm, face, &sS));
    PetscCall(DMPlexGetSupport(dm, face, &supp));
    for (PetscInt s = 0; s < sS; ++s) {
      PetscInt pdepth, l;

      // Filter support
      cind = GetPointIndex(supp[s], cStart, cEnd, cells);
      if (cind < 0) continue;
      // Ignore overlapping cells, but not for embedded manifolds
      PetscCall(DMPlexGetPointDepth(dm, supp[s], &pdepth));
      PetscCall(PetscFindInt(supp[s], Nl, lpoints, &l));
      if (pdepth == depth && l >= 0) continue;
      (*localFace)[face].cell = supp[s];
      break;
    }
    (*localFace)[face].comp = faceComp[f - fStart];
    // Cannot determine orientation without a cell
    if (cind < 0) continue;
    PetscCheck(0 <= faceComp[f - fStart] && faceComp[f - fStart] < Ncomp, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid component %" PetscInt_FMT " for face %" PetscInt_FMT " (%" PetscInt_FMT ")", faceComp[f - fStart], face, f);
    PetscCall(DMPlexGetOrientedCone(dm, (*localFace)[face].cell, &cone, &ornt));
    PetscCall(DMPlexGetConeSize(dm, (*localFace)[face].cell, &cS));
    for (c = 0; c < cS; ++c)
      if (cone[c] == face) break;
    PetscCheck(c < cS, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " not found in cone of cell %" PetscInt_FMT, face, (*localFace)[face].cell);
    if (faceIsVertex) {
      // Use cone position, shifted to -1 or 1
      if (PetscBTLookup(cellFlip, cind)) (*localFace)[face].ornt = 1 - c * 2;
      else (*localFace)[face].ornt = c * 2 - 1;
    } else {
      // Use orientation sense
      if (PetscBTLookup(cellFlip, cind)) (*localFace)[face].ornt = ornt[c] < 0 ? -1 : 1;
      else (*localFace)[face].ornt = ornt[c] < 0 ? 1 : -1;
    }
    PetscCall(DMPlexRestoreOrientedCone(dm, (*localFace)[face].cell, &cone, &ornt));
    if (debug)
      PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]: Local shared face %" PetscInt_FMT " component %" PetscInt_FMT " cell %" PetscInt_FMT " orientation %" PetscInt_FMT "\n", rank, face, (*localFace)[face].comp, (*localFace)[face].cell,
                                       (*localFace)[face].ornt));
  }
  if (debug) {
    PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  }
  // Get information from owners
  PetscCallMPI(MPI_Type_contiguous(3, MPIU_INT, &MPIU_3INT));
  PetscCallMPI(MPI_Type_commit(&MPIU_3INT));
  PetscCall(PetscSFBcastBegin(sf, MPIU_3INT, *localFace, *remoteFace, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_3INT, *localFace, *remoteFace, MPI_REPLACE));
  PetscCallMPI(MPI_Type_free(&MPIU_3INT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt lface; // local shared face
  PetscInt lcell; // local cell determining face orientation
  PetscInt lornt; // local face orientation
  PetscInt rrank; // remote rank
  PetscInt rcomp; // remote connected component
  PetscInt rface; // remote shared face
  PetscInt rcell; // remote cell determining face orientation
  PetscInt rornt; // remote face orientation
} Neighbor;

static PetscErrorCode DMPlexOrientCreateNeighbors_Internal(DM dm, IS cellIS, IS faceIS, PetscInt Ncomp, const PetscInt faceComp[], const SharedFace localFace[], const SharedFace remoteFace[], PetscInt **Nneigh, Neighbor ***neighbors)
{
  const PetscInt     debug = ((DM_Plex *)dm->data)->printOrient;
  const PetscInt    *cells = NULL, *faces = NULL;
  PetscInt           cStart = 0, cEnd = 0, fStart = 0, fEnd = 0;
  PetscSF            sf;
  const PetscInt    *lpoints;
  const PetscSFNode *rpoints;
  PetscInt           Nl;
  PetscHSetIJ       *ht;
  PetscInt          *counts;
  PetscViewer        viewer = NULL, selfviewer = NULL;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  if (debug) {
    viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
  }
  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &lpoints, &rpoints));
  PetscCall(PetscCalloc2(Ncomp, &ht, Ncomp, &counts));
  for (PetscInt c = 0; c < Ncomp; ++c) PetscCall(PetscHSetIJCreate(&ht[c]));
  PetscCall(PetscCalloc2(Ncomp, Nneigh, Ncomp, neighbors));
  // Count the number of unique connections between process/component
  for (PetscInt f = fStart; f < fEnd; ++f) {
    const PetscInt face = faces ? faces[f] : f;
    PetscHashIJKey key;
    PetscInt       l;

    PetscCall(PetscFindInt(face, Nl, lpoints, &l));
    if (l < 0) continue;
    // Remote face is not part of the label
    if (remoteFace[face].comp < 0) continue;
    // Remote face is isolated from any surface cell, so cannot get orientation
    if (!remoteFace[face].ornt) continue;
    key.i = rpoints[l].rank;
    key.j = remoteFace[face].comp;
    PetscCheck(0 <= remoteFace[face].comp, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid component %" PetscInt_FMT " on remote face %" PetscInt_FMT " from rank %" PetscInt_FMT " for local face %" PetscInt_FMT, remoteFace[face].comp, rpoints[l].index,
               rpoints[l].rank, face);
    PetscCall(PetscHSetIJAdd(ht[faceComp[f - fStart]], key));
  }
  // Get the sizes from each hash
  for (PetscInt c = 0; c < Ncomp; ++c) {
    PetscCall(PetscHSetIJGetSize(ht[c], &(*Nneigh)[c]));
    PetscCall(PetscMalloc1((*Nneigh)[c], &(*neighbors)[c]));
    PetscCall(PetscHSetIJClear(ht[c]));
    if (debug) PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]: component %" PetscInt_FMT ", Found %" PetscInt_FMT " connections\n", rank, c, (*Nneigh)[c]));
  }
  // Gather the neighbor information
  for (PetscInt f = fStart; f < fEnd; ++f) {
    const PetscInt face = faces ? faces[f] : f;
    const PetscInt comp = faceComp[f - fStart];
    const PetscInt ind  = counts[comp];
    PetscHashIJKey key;
    PetscBool      missing;
    PetscInt       l;

    PetscCheck(0 <= comp && comp < Ncomp, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid component %" PetscInt_FMT, comp);
    PetscCall(PetscFindInt(face, Nl, lpoints, &l));
    if (l < 0) continue;
    if (remoteFace[face].comp < 0) continue;
    if (!remoteFace[face].ornt) continue;
    key.i = rpoints[l].rank;
    key.j = remoteFace[face].comp;
    PetscCall(PetscHSetIJQueryAdd(ht[comp], key, &missing));
    if (!missing) continue;
    PetscCheck(localFace[face].ornt != 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid local face %" PetscInt_FMT " orientation: %" PetscInt_FMT, face, localFace[face].ornt);
    (*neighbors)[comp][ind].lface = face;
    (*neighbors)[comp][ind].lcell = localFace[face].cell;
    (*neighbors)[comp][ind].lornt = localFace[face].ornt;
    (*neighbors)[comp][ind].rrank = rpoints[l].rank;
    (*neighbors)[comp][ind].rcomp = remoteFace[face].comp;
    (*neighbors)[comp][ind].rface = rpoints[l].index;
    (*neighbors)[comp][ind].rcell = remoteFace[face].cell;
    (*neighbors)[comp][ind].rornt = remoteFace[face].ornt;
    if (debug)
      PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]: component %" PetscInt_FMT ", Found representative %" PetscInt_FMT " leaf %" PetscInt_FMT " (face %" PetscInt_FMT ") connecting to face %" PetscInt_FMT " on (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") with orientation %" PetscInt_FMT "\n", rank,
                                       remoteFace[face].comp, ind, l, face, (*neighbors)[comp][ind].rface, (*neighbors)[comp][ind].rrank, (*neighbors)[comp][ind].rcomp, (*neighbors)[comp][ind].rcell, (*neighbors)[comp][ind].rornt));
    ++counts[comp];
  }
  // Cleanup
  if (debug) {
    PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  }
  for (PetscInt c = 0; c < Ncomp; ++c) {
    PetscCheck(counts[c] == (*Nneigh)[c], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Neigh count for component %" PetscInt_FMT ": %" PetscInt_FMT " != %" PetscInt_FMT " allocated size", c, counts[c], (*Nneigh)[c]);
    PetscCall(PetscHSetIJDestroy(&ht[c]));
  }
  PetscCall(PetscFree2(ht, counts));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexOrientCreateProcessGraph_Internal(DM dm, IS faceIS, PetscInt Ncomp, const PetscInt Nneigh[], Neighbor *neighbors[], PetscSFNode **neighborAdj, PetscBool **neighborVal)
{
  const PetscInt    *faces  = NULL;
  PetscInt           fStart = 0, fEnd = 0;
  PetscSFNode       *nrankComp; // The (rank, comp) of each neighbor
  PetscBool         *match;     // Whether neighbors currently match
  PetscSF            sf;
  const PetscInt    *lpoints;
  const PetscSFNode *rpoints;
  PetscInt           totNeighbors = 0, Nl, fdepth;
  PetscBool          faceIsVertex = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &lpoints, &rpoints));
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(DMPlexGetPointDepth(dm, faces ? faces[fStart] : fStart, &fdepth));
  if (!fdepth) faceIsVertex = PETSC_TRUE;
  for (PetscInt c = 0; c < Ncomp; ++c) totNeighbors += Nneigh[c];
  PetscCall(PetscMalloc2(totNeighbors, neighborAdj, totNeighbors, neighborVal));
  nrankComp = *neighborAdj;
  match     = *neighborVal;
  for (PetscInt c = 0, off = 0; c < Ncomp; ++c) {
    for (PetscInt n = 0; n < Nneigh[c]; ++n, ++off) {
      PetscInt l;

      if (faceIsVertex) {
        match[off] = neighbors[c][n].lornt != neighbors[c][n].rornt ? PETSC_TRUE : PETSC_FALSE;
      } else {
        const PetscInt o = neighbors[c][n].lornt * neighbors[c][n].rornt;

        if (o < 0) match[off] = PETSC_TRUE;
        else match[off] = PETSC_FALSE;
      }
      // Flip sense if we are matching from an unowned cell
      PetscCall(PetscFindInt(neighbors[c][n].lcell, Nl, lpoints, &l));
      if (l >= 0) {
        if (rpoints[l].rank == neighbors[c][n].rrank && rpoints[l].index == neighbors[c][n].rcell) match[off] = match[off] ? PETSC_FALSE : PETSC_TRUE;
      }
      nrankComp[off].rank  = neighbors[c][n].rrank;
      nrankComp[off].index = neighbors[c][n].rcomp;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexOrientSolveProcessGraph_Internal(DM dm, IS cellIS, PetscInt Ncomp, const PetscInt cellComp[], const PetscInt Nneigh[], const PetscSFNode nrankComp[], const PetscBool match[], PetscBT cellFlip)
{
  const PetscInt     debug  = ((DM_Plex *)dm->data)->printOrient;
  const PetscInt    *cells  = NULL;
  PetscInt           cStart = 0, cEnd = 0;
  PetscSF            sf;
  const PetscInt    *lpoints;
  const PetscSFNode *rpoints;
  PetscInt           Nl, totNeighbors = 0;
  PetscBool         *flipped;
  MPI_Comm           comm;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &lpoints, &rpoints));
  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  /* Collect the graph on 0 */
  for (PetscInt c = 0; c < Ncomp; ++c) totNeighbors += Nneigh[c];
  if (Nl >= 0) {
    Mat          G;
    PetscBT      seenProcs, flippedProcs;
    PetscInt    *procFIFO, pTop, pBottom;
    PetscInt    *N          = NULL, *Noff;
    PetscSFNode *adj        = NULL;
    PetscBool   *val        = NULL;
    PetscMPIInt *recvcounts = NULL, *displs = NULL, *Nc;
    PetscMPIInt  size = 0, iNcomp, itotNeighbors;

    PetscCall(PetscCalloc1(Ncomp, &flipped));
    if (rank == 0) PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCall(PetscCalloc4(size, &recvcounts, size + 1, &displs, size, &Nc, size + 1, &Noff));
    PetscCallMPI(MPI_Gather(&Ncomp, 1, MPI_INT, Nc, 1, MPI_INT, 0, comm));
    for (PetscInt p = 0; p < size; ++p) displs[p + 1] = displs[p] + Nc[p];
    if (rank == 0) PetscCall(PetscMalloc1(displs[size], &N));
    PetscCall(PetscMPIIntCast(Ncomp, &iNcomp));
    PetscCallMPI(MPI_Gatherv(Nneigh, iNcomp, MPIU_INT, N, Nc, displs, MPIU_INT, 0, comm));
    for (PetscInt p = 0, o = 0; p < size; ++p) {
      recvcounts[p] = 0;
      for (PetscInt c = 0; c < Nc[p]; ++c, ++o) recvcounts[p] += N[o];
      displs[p + 1] = displs[p] + recvcounts[p];
    }
    if (rank == 0) PetscCall(PetscMalloc2(displs[size], &adj, displs[size], &val));
    PetscCall(PetscMPIIntCast(totNeighbors, &itotNeighbors));
    PetscCallMPI(MPI_Gatherv(nrankComp, itotNeighbors, MPIU_SF_NODE, adj, recvcounts, displs, MPIU_SF_NODE, 0, comm));
    PetscCallMPI(MPI_Gatherv(match, itotNeighbors, MPI_C_BOOL, val, recvcounts, displs, MPI_C_BOOL, 0, comm));
    if (rank == 0) {
      for (PetscInt p = 1; p <= size; ++p) Noff[p] = Noff[p - 1] + Nc[p - 1];
      if (debug) {
        for (PetscInt p = 0, off = 0; p < size; ++p) {
          for (PetscInt c = 0; c < Nc[p]; ++c) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "Proc %" PetscInt_FMT " Comp %" PetscInt_FMT ":\n", p, c));
            for (PetscInt n = 0; n < N[Noff[p] + c]; ++n, ++off) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  edge (%" PetscInt_FMT ", %" PetscInt_FMT ") (%s):\n", adj[off].rank, adj[off].index, PetscBools[val[off]]));
          }
        }
      }
      /* Symmetrize the graph */
      PetscCall(MatCreate(PETSC_COMM_SELF, &G));
      PetscCall(MatSetSizes(G, Noff[size], Noff[size], Noff[size], Noff[size]));
      PetscCall(MatSetUp(G));
      for (PetscInt p = 0, off = 0; p < size; ++p) {
        for (PetscInt c = 0; c < Nc[p]; ++c) {
          const PetscInt r = Noff[p] + c;

          for (PetscInt n = 0; n < N[r]; ++n, ++off) {
            const PetscInt    q = Noff[adj[off].rank] + adj[off].index;
            const PetscScalar o = val[off] ? 1.0 : 0.0;

            // Do not set values for processes that have no face to orient
            if (!Nc[adj[off].rank]) continue;
            PetscCall(MatSetValues(G, 1, &r, 1, &q, &o, INSERT_VALUES));
            PetscCall(MatSetValues(G, 1, &q, 1, &r, &o, INSERT_VALUES));
          }
        }
      }
      PetscCall(MatAssemblyBegin(G, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(G, MAT_FINAL_ASSEMBLY));

      PetscCall(PetscBTCreate(Noff[size], &seenProcs));
      PetscCall(PetscBTMemzero(Noff[size], seenProcs));
      PetscCall(PetscBTCreate(Noff[size], &flippedProcs));
      PetscCall(PetscBTMemzero(Noff[size], flippedProcs));
      PetscCall(PetscMalloc1(Noff[size], &procFIFO));
      pTop = pBottom = 0;
      for (PetscInt p = 0; p < Noff[size]; ++p) {
        if (PetscBTLookup(seenProcs, p)) continue;
        /* Initialize FIFO with next proc */
        procFIFO[pBottom++] = p;
        PetscCall(PetscBTSet(seenProcs, p));
        /* Consider each proc in FIFO */
        while (pTop < pBottom) {
          const PetscScalar *ornt;
          const PetscInt    *neighbors;
          PetscInt           proc, nproc, seen, flippedA, flippedB, mismatch, numNeighbors;

          proc     = procFIFO[pTop++];
          flippedA = PetscBTLookup(flippedProcs, proc) ? 1 : 0;
          PetscCall(MatGetRow(G, proc, &numNeighbors, &neighbors, &ornt));
          /* Loop over neighboring procs */
          for (PetscInt n = 0; n < numNeighbors; ++n) {
            nproc    = neighbors[n];
            mismatch = PetscRealPart(ornt[n]) > 0.5 ? 0 : 1;
            seen     = PetscBTLookup(seenProcs, nproc);
            flippedB = PetscBTLookup(flippedProcs, nproc) ? 1 : 0;

            if (mismatch ^ (flippedA ^ flippedB)) {
              PetscCheck(!seen, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Previously seen procs %" PetscInt_FMT " and %" PetscInt_FMT " do not match: Fault mesh is non-orientable", proc, nproc);
              PetscCheck(!flippedB, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent mesh orientation: Fault mesh is non-orientable");
              PetscCall(PetscBTSet(flippedProcs, nproc));
            } else PetscCheck(!mismatch || !flippedA || !flippedB, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Attempt to flip already flipped cell: Fault mesh is non-orientable");
            if (!seen) {
              procFIFO[pBottom++] = nproc;
              PetscCall(PetscBTSet(seenProcs, nproc));
            }
          }
        }
      }
      PetscCall(PetscFree(procFIFO));
      PetscCall(MatDestroy(&G));
      PetscCall(PetscFree2(adj, val));
      PetscCall(PetscBTDestroy(&seenProcs));
    }
    /* Scatter flip flags */
    {
      PetscBool *flips = NULL;

      if (rank == 0) {
        PetscCall(PetscMalloc1(Noff[size], &flips));
        for (PetscInt p = 0; p < Noff[size]; ++p) {
          flips[p] = PetscBTLookup(flippedProcs, p) ? PETSC_TRUE : PETSC_FALSE;
          if (debug && flips[p]) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Flipping Proc+Comp %" PetscInt_FMT ":\n", p));
        }
        for (PetscInt p = 0; p < size; ++p) displs[p + 1] = displs[p] + Nc[p];
      }
      PetscCall(PetscMPIIntCast(Ncomp, &iNcomp));
      PetscCallMPI(MPI_Scatterv(flips, Nc, displs, MPI_C_BOOL, flipped, iNcomp, MPI_C_BOOL, 0, comm));
      PetscCall(PetscFree(flips));
    }
    if (rank == 0) PetscCall(PetscBTDestroy(&flippedProcs));
    PetscCall(PetscFree(N));
    PetscCall(PetscFree4(recvcounts, displs, Nc, Noff));

    /* Decide whether to flip cells in each component */
    for (PetscInt c = 0; c < cEnd - cStart; ++c) {
      if (flipped[cellComp[c]]) PetscCall(PetscBTNegate(cellFlip, c));
    }
    PetscCall(PetscFree(flipped));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Reverse flipped cells in the mesh
static PetscErrorCode DMPlexOrientPlex_Internal(DM dm, IS cellIS, PetscBT cellFlip)
{
  const PetscInt  debug  = ((DM_Plex *)dm->data)->printOrient;
  const PetscInt *cells  = NULL;
  PetscInt        cStart = 0, cEnd = 0;
  PetscSF         sf;
  const PetscInt *rootdegree = NULL;
  PetscInt       *points     = NULL;
  PetscInt        pStart, pEnd, Nr, depth, cdepth = -1;
  PetscViewer     v;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  if (debug) {
    PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)dm), &v));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
  }
  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  PetscCall(DMPlexGetDepth(dm, &depth));
  if (cEnd > cStart) PetscCall(DMPlexGetPointDepth(dm, cells ? cells[cStart] : cStart, &cdepth));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &cdepth, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, &Nr, NULL, NULL, NULL));
  if (Nr >= 0) {
    PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegree));
    PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegree));
  }
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  if (cdepth == depth) PetscCall(PetscCalloc1(pEnd - pStart, &points));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    if (PetscBTLookup(cellFlip, c - cStart)) {
      const PetscInt cell = cells ? cells[c] : c;

      PetscCall(DMPlexOrientPoint(dm, cell, -1));
      if (points && rootdegree && rootdegree[cell]) points[cell] = 1;
      if (debug) PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]Flipping cell %" PetscInt_FMT "%s\n", rank, cell, points && rootdegree && rootdegree[cell] ? " and sending to overlap" : ""));
    }
  }
  // Propagate flips for volumetric cells in the overlap
  if (cdepth == depth) {
    if (Nr >= 0) {
      PetscCall(PetscSFBcastBegin(sf, MPIU_INT, points, points, MPI_SUM));
      PetscCall(PetscSFBcastEnd(sf, MPIU_INT, points, points, MPI_SUM));
    }
    for (PetscInt c = cStart; c < cEnd; ++c) {
      const PetscInt cell = cells ? cells[c] : c;

      if (points[cell] && !PetscBTLookup(cellFlip, c - cStart)) {
        PetscCall(DMPlexOrientPoint(dm, cell, -1));
        if (debug) PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]Flipping cell %" PetscInt_FMT " through overlap\n", rank, cell));
      }
    }
    PetscCall(PetscFree(points));
  }
  if (debug) {
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The orientation operation needs to perform two functions:

  1) Orient a bulk mesh. Here we assume that faces are in the correct order, but may
     be reversed.

  2) Orient a manifold embedded in a given mesh. This introduces many complications, since
     lower-dimensional pieces of the manifold might be shared with other processes.

  We will divide the operation into phases

  1) Each process orients the local mesh or manifold, and also counts the connected components.

  2) Send information about all neighboring faces to leaves

  3) Determine which processes are connected to each component, and pick a representative shared face connecting them

     For volumetric meshes (ignoring overlap cells), shared faces connect a single cell on each process.

     For manifolds, interior faces can be shared, so multiple cells can be connected. We must determine whether the cells
     we are using to compute orientation are identified.

  4) Determine whether the orientations match across each process boundary, to make the local piece of the process graph

     For manifolds, if the assoicated cells are identified, we flip the sense of the comparison.

  5) Gather the process graph to process 0, compute a satisfying orientation, communicate the orientation

  6) Flip the relevant cells and for volumetric meshes, propagate reordering to overlap cells
*/
PetscErrorCode DMPlexOrientCells_Internal(DM dm, IS cellIS, IS faceIS)
{
  const PetscInt  debug = ((DM_Plex *)dm->data)->printOrient;
  const PetscInt *cells = NULL, *faces = NULL;
  PetscInt        cStart = 0, cEnd = 0, fStart = 0, fEnd = 0;
  PetscBT         cellFlip;    // The bit is true if a cell should have its orientation reversed
  PetscInt       *cellComp;    // The connected component number of each cell
  PetscInt       *faceComp;    // The connected component number of each face
  PetscInt        Ncomp = 0;   // The number of local connected components
  SharedFace     *localFace;   // Holds local information for owned and ghost shared faces
  SharedFace     *remoteFace;  // Holds remote information from owners of shared faces
  PetscInt       *Nneigh;      // The number of neighboring processes for each component
  Neighbor      **neighbors;   // The neighbor for each component
  PetscSFNode    *neighborAdj; // The (rank, comp) of each neighbor
  PetscBool      *neighborVal; // Whether neighbors currently match
  MPI_Comm        comm;
  PetscMPIInt     rank, size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(PetscBTCreate(cEnd - cStart, &cellFlip));
  PetscCall(PetscBTMemzero(cEnd - cStart, cellFlip));
  PetscCall(PetscCalloc2(cEnd - cStart, &cellComp, fEnd - fStart, &faceComp));
  // Phase 1: Serial Orientation
  PetscCall(DMPlexOrient_Serial(dm, cellIS, faceIS, &Ncomp, cellComp, faceComp, cellFlip));
  if (debug) {
    PetscViewer v;
    PetscInt    cdepth = -1;

    PetscCall(PetscViewerASCIIGetStdout(comm, &v));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
    if (cEnd > cStart) PetscCall(DMPlexGetPointDepth(dm, cells ? cells[cStart] : cStart, &cdepth));
    PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]New Orientation %" PetscInt_FMT " cells (depth %" PetscInt_FMT ") and %" PetscInt_FMT " faces\n", rank, cEnd - cStart, cdepth, fEnd - fStart));
    PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for serial flipped cells:\n", rank));
    PetscCall(PetscBTView(cEnd - cStart, cellFlip, v));
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  }
  if (size == 1) goto end;
  // Phase 2
  PetscCall(DMPlexOrientCreateSharedFaces_Internal(dm, cellIS, faceIS, Ncomp, faceComp, cellFlip, &localFace, &remoteFace));
  // Phase 3
  PetscCall(DMPlexOrientCreateNeighbors_Internal(dm, cellIS, faceIS, Ncomp, faceComp, localFace, remoteFace, &Nneigh, &neighbors));
  PetscCall(PetscFree2(localFace, remoteFace));
  // Phase 4
  PetscCall(DMPlexOrientCreateProcessGraph_Internal(dm, faceIS, Ncomp, Nneigh, neighbors, &neighborAdj, &neighborVal));
  // Phase 5
  PetscCall(DMPlexOrientSolveProcessGraph_Internal(dm, cellIS, Ncomp, cellComp, Nneigh, neighborAdj, neighborVal, cellFlip));
  if (debug) {
    PetscViewer v;

    PetscCall(PetscViewerASCIIGetStdout(comm, &v));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
    PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for parallel flipped cells:\n", rank));
    PetscCall(PetscBTView(cEnd - cStart, cellFlip, v));
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  }
  for (PetscInt c = 0; c < Ncomp; ++c) PetscCall(PetscFree(neighbors[c]));
  PetscCall(PetscFree2(Nneigh, neighbors));
  PetscCall(PetscFree2(neighborAdj, neighborVal));
end:
  // Phase 6
  PetscCall(DMPlexOrientPlex_Internal(dm, cellIS, cellFlip));
  PetscCall(PetscBTDestroy(&cellFlip));
  PetscCall(PetscFree2(cellComp, faceComp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexCheckOrientation_Internal(DM dm, IS cellIS, IS faceIS)
{
  const PetscInt     debug  = ((DM_Plex *)dm->data)->printOrient;
  PetscViewer        viewer = NULL, selfviewer = NULL;
  PetscSF            sf;
  const PetscInt    *lpoints, *rootdegree = NULL;
  const PetscSFNode *rpoints;
  const PetscInt    *cells = NULL, *faces = NULL;
  PetscSFNode       *oornt, *rornt, *lornt;
  PetscInt           cStart = 0, cEnd = 0, fStart = 0, fEnd = 0;
  PetscInt           pdepth, Nr, Nl;
  PetscBool          faceIsVertex = PETSC_FALSE, valid = PETSC_TRUE;
  MPI_Comm           comm;
  PetscMPIInt        size, rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (debug) {
    viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm));
    PetscCall(PetscViewerASCIIPushSynchronized(viewer));
    PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
  }

  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, &Nr, &Nl, &lpoints, &rpoints));
  if (Nr >= 0) {
    PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegree));
    PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegree));
  } else {
    sf = NULL;
    Nr = 0;
    Nl = 0;
  }
  PetscCall(PetscCalloc3(Nr, &oornt, Nr, &rornt, Nl, &lornt));
  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(DMPlexGetPointDepth(dm, faces ? faces[fStart] : fStart, &pdepth));
  if (!pdepth) faceIsVertex = PETSC_TRUE;
  if (debug) {
    PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]Checking orientation of %" PetscInt_FMT " cells and %" PetscInt_FMT " faces\n", rank, cEnd - cStart, fEnd - fStart));
    PetscCall(PetscViewerASCIIPushTab(selfviewer));
  }
  for (PetscInt f = fStart; f < fEnd; ++f) {
    const PetscInt  face  = faces ? faces[f] : f;
    const PetscBool owner = rootdegree && rootdegree[face] ? PETSC_TRUE : PETSC_FALSE;
    const PetscInt *supp;
    PetscInt        neighbors[2], o[2] = {0, 0};
    PetscInt        lsS = 0, sS;

    PetscCall(DMPlexGetSupport(dm, face, &supp));
    PetscCall(DMPlexGetSupportSize(dm, face, &sS));
    // Filter support for local cells
    for (PetscInt s = 0; s < sS; ++s) {
      PetscInt ind;

      ind = GetPointIndex(supp[s], cStart, cEnd, cells);
      if (ind >= 0) {
        neighbors[PetscMin(lsS, 1)] = supp[s];
        ++lsS;
      }
    }
    PetscCheck(lsS < 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " has support size %" PetscInt_FMT " > 2", face, lsS);
    // Extract orientations
    for (PetscInt s = 0; s < lsS; ++s) {
      const PetscInt *cone, *ornt;
      PetscInt        cS;

      PetscCall(DMPlexGetConeSize(dm, neighbors[s], &cS));
      PetscCall(DMPlexGetOrientedCone(dm, neighbors[s], &cone, &ornt));
      for (PetscInt c = 0; c < cS; ++c) {
        if (cone[c] == face) {
          if (faceIsVertex) o[s] = c * 2 - 1;
          else o[s] = ornt[c] < 0 ? -1 : 1;
          break;
        }
      }
      PetscCall(DMPlexRestoreOrientedCone(dm, neighbors[s], &cone, &ornt));
    }
    if (lsS == 2) {
      // Check internal face
      if (o[0] * o[1] >= 0) {
        valid = PETSC_FALSE;
        if (debug)
          PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]Internal Face %" PetscInt_FMT " is mismatched: cell %" PetscInt_FMT " (%" PetscInt_FMT ") ~ cell %" PetscInt_FMT " (%" PetscInt_FMT ")\n", rank, face, neighbors[0], o[0], neighbors[1], o[1]));
      } else if (debug > 1) PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]Internal Face %" PetscInt_FMT " valid\n", rank, face));
    } else {
      // Check shared and boundary faces
      PetscInt l;

      PetscCall(PetscFindInt(face, Nl, lpoints, &l));
      // Boundary face
      if (l < 0 && !owner) {
        if (debug > 1) PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]Boundary Face %" PetscInt_FMT " valid\n", rank, face));
        continue;
      }
      if (l >= 0) {
        lornt[l].index = neighbors[0];
        lornt[l].rank  = o[0];
        if (debug > 1) PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]Ghost Face %" PetscInt_FMT " (%" PetscInt_FMT ") was stored\n", rank, face, o[0]));
      }
    }
    if (owner) {
      // Store shared face orientation and cell from owner
      oornt[face].index = neighbors[0];
      oornt[face].rank  = o[0];
      if (debug > 1) PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]Owned Face %" PetscInt_FMT " (%" PetscInt_FMT ") was stored\n", rank, face, o[0]));
    }
  }
  // Communicate shared face orientations from owner
  if (sf) {
    PetscCall(PetscSFBcastBegin(sf, MPIU_SF_NODE, oornt, rornt, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_SF_NODE, oornt, rornt, MPI_REPLACE));
  }
  // Check unowned shared faces
  for (PetscInt l = 0; l < Nl; ++l) {
    const PetscInt face = lpoints ? lpoints[l] : l;
    PetscBool      flip = PETSC_FALSE;
    PetscInt       ind, cl, o[2] = {0, 0};

    // Filter for local faces
    ind = GetPointIndex(face, fStart, fEnd, faces);
    if (ind < 0) continue;
    // Check for shared cell
    PetscCall(PetscFindInt(lornt[l].index, Nl, lpoints, &cl));
    if (cl >= 0) {
      if (rpoints[cl].index == rornt[face].index) {
        flip = PETSC_TRUE;
        if (debug > 1) PetscCall(PetscViewerASCIIPrintf(selfviewer, "Shared cell %" PetscInt_FMT " maps to %" PetscInt_FMT " (%" PetscInt_FMT ") so we reverse the orientation check\n", lornt[l].index, rpoints[cl].index, rpoints[cl].rank));
      } else {
        if (debug > 1)
          PetscCall(PetscViewerASCIIPrintf(selfviewer, "Shared cell %" PetscInt_FMT " maps to %" PetscInt_FMT " (%" PetscInt_FMT ") instead of %" PetscInt_FMT " so we do not reverse the orientation check\n", lornt[l].index, rpoints[cl].index,
                                           rpoints[cl].rank, rornt[face].index));
      }
    }
    o[0] = lornt[l].rank;
    o[1] = rornt[face].rank;
    // This was not a shared face
    if (!o[0]) continue;
    if ((o[0] * o[1] >= 0 && !flip) || (o[0] * o[1] < 0 && flip)) {
      valid = PETSC_FALSE;
      if (debug)
        PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]Ghost Face %" PetscInt_FMT " (%" PetscInt_FMT ") does not match face %" PetscInt_FMT " rank %" PetscInt_FMT " (%" PetscInt_FMT ")\n", rank, face, o[0], rpoints[l].index, rpoints[l].rank, o[1]));
    } else if (debug > 1) PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]Ghost Face %" PetscInt_FMT " matches %" PetscInt_FMT " (%" PetscInt_FMT ") valid\n", rank, face, rpoints[l].index, rpoints[l].rank));
  }
  // Cleanup
  PetscCall(PetscFree3(oornt, rornt, lornt));
  if (debug) {
    PetscCall(PetscViewerASCIIPopTab(selfviewer));
    PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
    PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_C_BOOL, MPI_LAND, comm));
  PetscCheck(valid, comm, PETSC_ERR_ARG_WRONGSTATE, "Mesh was not properly oriented");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexCheckOrientationLabel - Check that the surface defined by the given `DMLabel` is oriented

  Collective

  Input Parameters:
+ dm    - The `DM`
- label - The `DMLabel` defining the embedded surface

  Level: advanced

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMPlexOrient()`
@*/
PetscErrorCode DMPlexCheckOrientationLabel(DM dm, DMLabel label)
{
  IS cellIS, faceIS;

  PetscFunctionBegin;
  PetscCall(CreateCellAndFaceIS_Private(dm, label, &cellIS, &faceIS));
  PetscCall(DMPlexCheckOrientation_Internal(dm, cellIS, faceIS));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(ISDestroy(&faceIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
