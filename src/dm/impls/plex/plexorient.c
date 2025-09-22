#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

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
    if (!seenA && !flippedA) {
      PetscCall(PetscBTSet(flippedCells, indC[0]));
    } else if (!seenB && !flippedB) {
      PetscCall(PetscBTSet(flippedCells, indC[1]));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Inconsistent mesh orientation: Fault mesh is non-orientable");
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexOrient - Give a consistent orientation to the input mesh

  Input Parameter:
. dm - The `DM`

  Note:
  The orientation data for the `DM` are change in-place.

  This routine will fail for non-orientable surfaces, such as the Moebius strip.

  Level: advanced

.seealso: [](ch_unstructured), `DM`, `DMPLEX`, `DMCreate()`
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
    PetscInt l, n;

    numNeighbors[comp] = 0;
    PetscCall(PetscMalloc1(PetscMax(numLeaves, 0), &neighbors[comp]));
    /* I know this is p^2 time in general, but for bounded degree its alright */
    for (l = 0; l < numLeaves; ++l) {
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
          PetscInt       n;

          for (n = 0; n < N[r]; ++n, ++off) {
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

PetscErrorCode DMPlexOrientCells_Internal(DM dm, IS cellIS, IS faceIS)
{
  MPI_Comm           comm;
  PetscSF            sf;
  const PetscInt    *lpoints;
  const PetscSFNode *rpoints;
  PetscSFNode       *rorntComp = NULL, *lorntComp = NULL;
  PetscInt          *numNeighbors, **neighbors, *locSupp = NULL;
  PetscSFNode       *nrankComp;
  PetscBool         *match, *flipped;
  PetscBT            flippedCells;
  PetscInt          *cellComp, *faceComp;
  const PetscInt    *cells = NULL, *faces = NULL;
  PetscInt           cStart = 0, cEnd = 0, fStart = 0, fEnd = 0;
  PetscInt           numLeaves, numRoots, dim, Ncomp, totNeighbors = 0;
  PetscMPIInt        rank, size;
  PetscBool          view, viewSync;
  PetscViewer        viewer = NULL, selfviewer = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscOptionsHasName(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-orientation_view", &view));
  PetscCall(PetscOptionsHasName(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-orientation_view_synchronized", &viewSync));

  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
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
  PetscCall(PetscBTCreate(cEnd - cStart, &flippedCells));
  PetscCall(PetscBTMemzero(cEnd - cStart, flippedCells));
  PetscCall(PetscCalloc2(cEnd - cStart, &cellComp, fEnd - fStart, &faceComp));
  /*
   OLD STYLE
   - Add an integer array over cells and faces (component) for connected component number
   Foreach component
     - Mark the initial cell as seen
     - Process component as usual
     - Set component for all seenCells
     - Wipe seenCells and seenFaces (flippedCells can stay)
   - Generate parallel adjacency for component using SF and seenFaces
   - Collect Ncomp adj data from each proc to 0
   - Build same serial graph
   - Use same solver
   - Use Scatterv to send back flipped flags for each component
   - Negate flippedCells by component

   NEW STYLE
   - Create the adj on each process
   - Bootstrap to complete graph on proc 0
  */
  PetscCall(DMPlexOrient_Serial(dm, cellIS, faceIS, &Ncomp, cellComp, faceComp, flippedCells));
  if (view) {
    PetscViewer v;
    PetscInt    cdepth = -1;

    PetscCall(PetscViewerASCIIGetStdout(comm, &v));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
    if (cEnd > cStart) PetscCall(DMPlexGetPointDepth(dm, cells ? cells[cStart] : cStart, &cdepth));
    PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]New Orientation %" PetscInt_FMT " cells (depth %" PetscInt_FMT ") and %" PetscInt_FMT " faces\n", rank, cEnd - cStart, cdepth, fEnd - fStart));
    PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for serial flipped cells:\n", rank));
    PetscCall(PetscBTView(cEnd - cStart, flippedCells, v));
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  }
  /* Now all subdomains are oriented, but we need a consistent parallel orientation */
  // TODO: This all has to be rewritten to filter cones/supports to the ISes
  if (numLeaves >= 0) {
    PetscInt maxSuppSize, neighbor;

    // Store orientations of boundary faces
    PetscCall(DMPlexGetMaxSizes(dm, NULL, &maxSuppSize));
    PetscCall(PetscCalloc3(numRoots, &rorntComp, numRoots, &lorntComp, maxSuppSize, &locSupp));
    for (PetscInt f = fStart; f < fEnd; ++f) {
      const PetscInt  face = faces ? faces[f] : f;
      const PetscInt *cone, *supp, *ornt;
      PetscInt        coneSize, suppSize, nind, c, Ns = 0;

      PetscCall(DMPlexGetSupportSize(dm, face, &suppSize));
      PetscCall(DMPlexGetSupport(dm, face, &supp));
      for (PetscInt s = 0; s < suppSize; ++s) {
        PetscInt ind, l;

        // Filter support
        ind = GetPointIndex(supp[s], cStart, cEnd, cells);
        if (ind < 0) continue;
        // Ignore overlapping cells
        PetscCall(PetscFindInt(supp[s], numLeaves, lpoints, &l));
        if (l >= 0) continue;
        locSupp[Ns++] = supp[s];
      }
      PetscCheck(Ns < maxSuppSize, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Index %" PetscInt_FMT " exceeds array size %" PetscInt_FMT, Ns, maxSuppSize);
      if (Ns != 1) continue;
      neighbor = locSupp[0];
      nind     = GetPointIndex(neighbor, cStart, cEnd, cells);
      PetscCall(DMPlexGetCone(dm, neighbor, &cone));
      PetscCall(DMPlexGetConeSize(dm, neighbor, &coneSize));
      PetscCall(DMPlexGetConeOrientation(dm, neighbor, &ornt));
      for (c = 0; c < coneSize; ++c)
        if (cone[c] == face) break;
      if (dim == 1) {
        /* Use cone position instead, shifted to -1 or 1 */
        if (PetscBTLookup(flippedCells, nind)) rorntComp[face].rank = 1 - c * 2;
        else rorntComp[face].rank = c * 2 - 1;
      } else {
        if (PetscBTLookup(flippedCells, nind)) rorntComp[face].rank = ornt[c] < 0 ? -1 : 1;
        else rorntComp[face].rank = ornt[c] < 0 ? 1 : -1;
      }
      rorntComp[face].index = faceComp[GetPointIndex(face, fStart, fEnd, faces)];
    }
    // Communicate boundary edge orientations
    PetscCall(PetscSFBcastBegin(sf, MPIU_SF_NODE, rorntComp, lorntComp, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_SF_NODE, rorntComp, lorntComp, MPI_REPLACE));
  }
  /* Get process adjacency */
  PetscCall(PetscMalloc2(Ncomp, &numNeighbors, Ncomp, &neighbors));
  viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)dm));
  if (viewSync) PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
  for (PetscInt comp = 0; comp < Ncomp; ++comp) {
    PetscInt n;

    numNeighbors[comp] = 0;
    PetscCall(PetscMalloc1(PetscMax(numLeaves, 0), &neighbors[comp]));
    /* I know this is p^2 time in general, but for bounded degree its alright */
    for (PetscInt l = 0; l < numLeaves; ++l) {
      const PetscInt face = lpoints[l];
      PetscInt       find;

      /* Find a representative face (edge) separating pairs of procs */
      find = GetPointIndex(face, fStart, fEnd, faces);
      if ((find >= 0) && (faceComp[find] == comp) && rorntComp[face].rank) {
        const PetscInt rrank = rpoints[l].rank;
        const PetscInt rcomp = lorntComp[face].index;

        for (n = 0; n < numNeighbors[comp]; ++n)
          if ((rrank == rpoints[neighbors[comp][n]].rank) && (rcomp == lorntComp[lpoints[neighbors[comp][n]]].index)) break;
        if (n >= numNeighbors[comp]) {
          const PetscInt *supp;
          PetscInt        suppSize, Ns = 0;

          PetscCall(DMPlexGetSupport(dm, face, &supp));
          PetscCall(DMPlexGetSupportSize(dm, face, &suppSize));
          for (PetscInt s = 0; s < suppSize; ++s) {
            // Filter support
            if (GetPointIndex(supp[s], cStart, cEnd, cells) >= 0) ++Ns;
          }
          PetscCheck(Ns == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Boundary face %" PetscInt_FMT " should see one cell, not %" PetscInt_FMT, face, Ns);
          if (view)
            PetscCall(PetscViewerASCIIPrintf(selfviewer, "[%d]: component %" PetscInt_FMT ", Found representative leaf %" PetscInt_FMT " (face %" PetscInt_FMT ") connecting to face %" PetscInt_FMT " on (%" PetscInt_FMT ", %" PetscInt_FMT ") with orientation %" PetscInt_FMT "\n", rank, comp, l, face,
                                             rpoints[l].index, rrank, rcomp, lorntComp[face].rank));
          neighbors[comp][numNeighbors[comp]++] = l;
        }
      }
    }
    totNeighbors += numNeighbors[comp];
  }
  PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &selfviewer));
  if (viewSync) PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscMalloc2(totNeighbors, &nrankComp, totNeighbors, &match));
  for (PetscInt comp = 0, off = 0; comp < Ncomp; ++comp) {
    for (PetscInt n = 0; n < numNeighbors[comp]; ++n, ++off) {
      const PetscInt face = lpoints[neighbors[comp][n]];
      const PetscInt o    = rorntComp[face].rank * lorntComp[face].rank;

      if (o < 0) match[off] = PETSC_TRUE;
      else if (o > 0) match[off] = PETSC_FALSE;
      else
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid face %" PetscInt_FMT " (%" PetscInt_FMT ", %" PetscInt_FMT ") neighbor: %" PetscInt_FMT " comp: %" PetscInt_FMT, face, rorntComp[face].rank, lorntComp[face].rank, neighbors[comp][n], comp);
      nrankComp[off].rank  = rpoints[neighbors[comp][n]].rank;
      nrankComp[off].index = lorntComp[lpoints[neighbors[comp][n]]].index;
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
    PetscMPIInt *recvcounts = NULL, *displs = NULL, *Nc;
    PetscMPIInt  size = 0, iNcomp, itotNeighbors;

    PetscCall(PetscCalloc1(Ncomp, &flipped));
    if (rank == 0) PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCall(PetscCalloc4(size, &recvcounts, size + 1, &displs, size, &Nc, size + 1, &Noff));
    PetscCallMPI(MPI_Gather(&Ncomp, 1, MPI_INT, Nc, 1, MPI_INT, 0, comm));
    for (PetscInt p = 0; p < size; ++p) displs[p + 1] = displs[p] + Nc[p];
    if (rank == 0) PetscCall(PetscMalloc1(displs[size], &N));
    PetscCall(PetscMPIIntCast(Ncomp, &iNcomp));
    PetscCallMPI(MPI_Gatherv(numNeighbors, iNcomp, MPIU_INT, N, Nc, displs, MPIU_INT, 0, comm));
    for (PetscInt p = 0, o = 0; p < size; ++p) {
      recvcounts[p] = 0;
      for (PetscInt c = 0; c < Nc[p]; ++c, ++o) recvcounts[p] += N[o];
      displs[p + 1] = displs[p] + recvcounts[p];
    }
    if (rank == 0) PetscCall(PetscMalloc2(displs[size], &adj, displs[size], &val));
    PetscCall(PetscMPIIntCast(totNeighbors, &itotNeighbors));
    PetscCallMPI(MPI_Gatherv(nrankComp, itotNeighbors, MPIU_SF_NODE, adj, recvcounts, displs, MPIU_SF_NODE, 0, comm));
    PetscCallMPI(MPI_Gatherv(match, itotNeighbors, MPI_C_BOOL, val, recvcounts, displs, MPI_C_BOOL, 0, comm));
    PetscCall(PetscFree2(numNeighbors, neighbors));
    if (rank == 0) {
      for (PetscInt p = 1; p <= size; ++p) Noff[p] = Noff[p - 1] + Nc[p - 1];
      if (view) {
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
          if (view && flips[p]) PetscCall(PetscPrintf(comm, "Flipping Proc+Comp %" PetscInt_FMT ":\n", p));
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
    PetscCall(PetscFree2(nrankComp, match));

    /* Decide whether to flip cells in each component */
    for (PetscInt c = 0; c < cEnd - cStart; ++c) {
      if (flipped[cellComp[c]]) PetscCall(PetscBTNegate(flippedCells, c));
    }
    PetscCall(PetscFree(flipped));
  }
  if (view) {
    PetscViewer v;

    PetscCall(PetscViewerASCIIGetStdout(comm, &v));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
    PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for parallel flipped cells:\n", rank));
    PetscCall(PetscBTView(cEnd - cStart, flippedCells, v));
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  }
  // Reverse flipped cells in the mesh
  PetscViewer     v;
  const PetscInt *degree = NULL;
  PetscInt       *points;
  PetscInt        pStart, pEnd;

  if (view) {
    PetscCall(PetscViewerASCIIGetStdout(comm, &v));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
  }
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  if (numRoots >= 0) {
    PetscCall(PetscSFComputeDegreeBegin(sf, &degree));
    PetscCall(PetscSFComputeDegreeEnd(sf, &degree));
  }
  PetscCall(PetscCalloc1(pEnd - pStart, &points));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    if (PetscBTLookup(flippedCells, c - cStart)) {
      const PetscInt cell = cells ? cells[c] : c;

      PetscCall(DMPlexOrientPoint(dm, cell, -1));
      if (degree && degree[cell]) points[cell] = 1;
      if (view) PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]Flipping cell %" PetscInt_FMT "%s\n", rank, cell, degree && degree[cell] ? " and sending to overlap" : ""));
    }
  }
  // Must propagate flips for cells in the overlap
  if (numRoots >= 0) {
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, points, points, MPI_SUM));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, points, points, MPI_SUM));
  }
  for (PetscInt c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;

    if (points[cell] && !PetscBTLookup(flippedCells, c - cStart)) {
      PetscCall(DMPlexOrientPoint(dm, cell, -1));
      if (view) PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]Flipping cell %" PetscInt_FMT " through overlap\n", rank, cell));
    }
  }
  if (view) {
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  }
  PetscCall(PetscFree(points));
  PetscCall(PetscBTDestroy(&flippedCells));
  PetscCall(PetscFree2(numNeighbors, neighbors));
  PetscCall(PetscFree3(rorntComp, lorntComp, locSupp));
  PetscCall(PetscFree2(cellComp, faceComp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
