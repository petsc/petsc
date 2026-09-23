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

/*
  DMPlexOrient_Serial - Compute valid orientation for local connected components

  Not collective

  Input Parameters:
  + dm     - The `DM`
  . cellIS - The cells to orient
  - faceIS - The faces between the cells

  Output Parameters:
  + Ncomp        - The number of connected component
  . cellComp     - The connected component for each local cell
  - flippedCells - Marked cells should be inverted

  Level: developer

.seealso: `DMPlexOrient()`
*/
static PetscErrorCode DMPlexOrient_Serial(DM dm, IS cellIS, IS faceIS, PetscInt *Ncomp, PetscInt cellComp[], PetscBT flippedCells)
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
    // Set component for cells
    for (PetscInt c = 0; c < cEnd - cStart; ++c) {
      if (PetscBTLookup(seenCells, c)) cellComp[c] = *Ncomp;
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

// A local cell next to a face, with the orientation that it induces on the face
typedef struct {
  PetscInt    rank;  // process holding the cell
  PetscInt    comp;  // connected component of the cell on that process
  PetscInt    ornt;  // orientation of the face in the cell, 1 or -1, or 0 if there is no cell
  PetscSFNode owner; // owner of the cell, which identifies the copies of the cell on other processes
} FaceSample;

// Copies of the same cell induce the same orientation on a face, and two different cells induce opposite orientations
static PetscBool FaceSamplesMatch(const FaceSample *a, const FaceSample *b)
{
  const PetscBool same = a->owner.rank == b->owner.rank && a->owner.index == b->owner.index ? PETSC_TRUE : PETSC_FALSE;

  return (a->ornt == b->ornt) == same ? PETSC_TRUE : PETSC_FALSE;
}

/*
  DMPlexGetFaceSamples_Private - Collect on the owner of each face the samples from all processes that hold the face

  Collective

  Each process samples the local cells of cellIS next to each face of faceIS, with the orientation of cellFlip. For each
  face faces[f] that this process owns, the samples are samples[off[f]] to samples[off[f + 1]]. With overlap, a cell can
  have copies on several processes, so a face can have more than two samples.
*/
static PetscErrorCode DMPlexGetFaceSamples_Private(DM dm, IS cellIS, IS faceIS, const PetscInt cellComp[], PetscBT cellFlip, PetscInt **off, FaceSample **samples)
{
  PetscSF            sf;
  const PetscInt    *lpoints, *rootdegree = NULL, *cells = NULL, *faces = NULL;
  const PetscSFNode *rpoints;
  FaceSample        *local, *remote = NULL;
  PetscInt          *roff;
  PetscInt           cStart = 0, cEnd = 0, fStart = 0, fEnd = 0, pEnd, Nr, Nl;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(DMPlexGetChart(dm, NULL, &pEnd));
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, &Nr, &Nl, &lpoints, &rpoints));
  if (Nr < 0) {
    sf = NULL;
    Nl = 0;
  }
  PetscCall(PetscCalloc2(2 * pEnd, &local, pEnd + 1, &roff));
  for (PetscInt f = fStart; f < fEnd; ++f) {
    const PetscInt  face = faces ? faces[f] : f;
    const PetscInt *supp;
    PetscInt        sS, depth, n = 0;

    PetscCall(DMPlexGetPointDepth(dm, face, &depth));
    PetscCall(DMPlexGetSupportSize(dm, face, &sS));
    PetscCall(DMPlexGetSupport(dm, face, &supp));
    for (PetscInt s = 0; s < sS; ++s) {
      const PetscInt  cind = GetPointIndex(supp[s], cStart, cEnd, cells);
      const PetscInt  l    = GetPointIndex(supp[s], 0, Nl, lpoints);
      FaceSample     *fs   = &local[2 * face + n];
      const PetscInt *cone, *ornt;
      PetscInt        cS, c;

      if (cind < 0) continue;
      PetscCheck(n < 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %" PetscInt_FMT " separates more than two cells", face);
      PetscCall(DMPlexGetConeSize(dm, supp[s], &cS));
      PetscCall(DMPlexGetOrientedCone(dm, supp[s], &cone, &ornt));
      for (c = 0; c < cS; ++c)
        if (cone[c] == face) break;
      // A vertex is oriented by its position in the cone of an edge
      fs->ornt = depth ? (ornt[c] < 0 ? -1 : 1) : 2 * c - 1;
      PetscCall(DMPlexRestoreOrientedCone(dm, supp[s], &cone, &ornt));
      if (cellFlip && PetscBTLookup(cellFlip, cind)) fs->ornt = -fs->ornt;
      fs->rank = rank;
      fs->comp = cellComp ? cellComp[cind] : 0;
      if (l >= 0) fs->owner = rpoints[l];
      else {
        fs->owner.rank  = rank;
        fs->owner.index = supp[s];
      }
      ++n;
    }
  }
  if (sf) {
    MPI_Datatype MPIU_10INT;

    PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegree));
    PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegree));
    for (PetscInt p = 0; p < pEnd; ++p) roff[p + 1] = roff[p] + (p < Nr ? rootdegree[p] : 0);
    PetscCall(PetscMalloc1(2 * roff[pEnd], &remote));
    PetscCallMPI(MPI_Type_contiguous(10, MPIU_INT, &MPIU_10INT));
    PetscCallMPI(MPI_Type_commit(&MPIU_10INT));
    PetscCall(PetscSFGatherBegin(sf, MPIU_10INT, local, remote));
    PetscCall(PetscSFGatherEnd(sf, MPIU_10INT, local, remote));
    PetscCallMPI(MPI_Type_free(&MPIU_10INT));
  }
  // An owned face has its local samples, and two from each process holding a copy of it
  PetscCall(PetscMalloc1(fEnd - fStart + 1, off));
  (*off)[0] = 0;
  for (PetscInt f = fStart; f < fEnd; ++f) {
    const PetscInt face = faces ? faces[f] : f;

    (*off)[f - fStart + 1] = (*off)[f - fStart] + (GetPointIndex(face, 0, Nl, lpoints) < 0 ? 2 * (1 + roff[face + 1] - roff[face]) : 0);
  }
  PetscCall(PetscMalloc1((*off)[fEnd - fStart], samples));
  for (PetscInt f = fStart; f < fEnd; ++f) {
    const PetscInt face = faces ? faces[f] : f;
    FaceSample    *fs;

    if ((*off)[f - fStart + 1] == (*off)[f - fStart]) continue;
    fs = PetscSafePointerPlusOffset(*samples, (*off)[f - fStart]);
    PetscCall(PetscArraycpy(fs, PetscSafePointerPlusOffset(local, 2 * face), 2));
    if (remote) PetscCall(PetscArraycpy(&fs[2], PetscSafePointerPlusOffset(remote, 2 * roff[face]), 2 * (roff[face + 1] - roff[face])));
  }
  PetscCall(PetscFree(remote));
  PetscCall(PetscFree2(local, roff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMPlexOrientSolve_Private - Choose which components to flip, so that all edges of the process graph match

  Collective

  Each edge links component (rank, comp) with component (rank, comp) and says whether their orientations match now.
  Process 0 gathers the graph and flips the components breadth first, and flipped[c] is true if the local component c
  must be flipped.
*/
static PetscErrorCode DMPlexOrientSolve_Private(DM dm, PetscInt Ncomp, PetscInt Ne, const PetscInt edges[], PetscBool flipped[])
{
  const PetscInt debug = ((DM_Plex *)dm->data)->printOrient;
  MPI_Comm       comm;
  PetscMPIInt    rank, size, iNcomp, iNe;
  PetscMPIInt   *Nc = NULL, *coff = NULL, *ecounts = NULL, *eoff = NULL;
  PetscInt      *allEdges = NULL, *adj = NULL, *aoff = NULL, *queue = NULL;
  PetscBool     *flips = NULL, *seen = NULL, orientable = PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscMPIIntCast(Ncomp, &iNcomp));
  PetscCall(PetscMPIIntCast(5 * Ne, &iNe));
  if (rank == 0) PetscCall(PetscCalloc4(size, &Nc, size + 1, &coff, size, &ecounts, size + 1, &eoff));
  PetscCallMPI(MPI_Gather(&iNcomp, 1, MPI_INT, Nc, 1, MPI_INT, 0, comm));
  PetscCallMPI(MPI_Gather(&iNe, 1, MPI_INT, ecounts, 1, MPI_INT, 0, comm));
  if (rank == 0) {
    for (PetscMPIInt p = 0; p < size; ++p) {
      coff[p + 1] = coff[p] + Nc[p];
      eoff[p + 1] = eoff[p] + ecounts[p];
    }
    PetscCall(PetscMalloc1(eoff[size], &allEdges));
  }
  PetscCallMPI(MPI_Gatherv(edges, iNe, MPIU_INT, allEdges, ecounts, eoff, MPIU_INT, 0, comm));
  if (rank == 0) {
    const PetscInt N = coff[size], E = eoff[size] / 5;
    PetscInt       qTop = 0, qBottom = 0;

    // Store both directions of each edge, as the neighbor and whether it matches
    PetscCall(PetscCalloc5(N + 1, &aoff, 4 * E, &adj, N, &queue, N, &flips, N, &seen));
    for (PetscInt e = 0; e < E; ++e) {
      ++aoff[coff[allEdges[5 * e + 0]] + allEdges[5 * e + 1] + 1];
      ++aoff[coff[allEdges[5 * e + 2]] + allEdges[5 * e + 3] + 1];
    }
    for (PetscInt n = 0; n < N; ++n) aoff[n + 1] += aoff[n];
    for (PetscInt e = 0; e < E; ++e) {
      const PetscInt a = coff[allEdges[5 * e + 0]] + allEdges[5 * e + 1], b = coff[allEdges[5 * e + 2]] + allEdges[5 * e + 3];

      if (debug)
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Edge (%" PetscInt_FMT ", %" PetscInt_FMT ") ~ (%" PetscInt_FMT ", %" PetscInt_FMT ") (%s)\n", allEdges[5 * e + 0], allEdges[5 * e + 1], allEdges[5 * e + 2], allEdges[5 * e + 3], PetscBools[allEdges[5 * e + 4]]));
      adj[2 * aoff[a]]     = b;
      adj[2 * aoff[a] + 1] = allEdges[5 * e + 4];
      ++aoff[a];
      adj[2 * aoff[b]]     = a;
      adj[2 * aoff[b] + 1] = allEdges[5 * e + 4];
      ++aoff[b];
    }
    for (PetscInt n = N; n > 0; --n) aoff[n] = aoff[n - 1];
    aoff[0] = 0;
    // A component is flipped relative to its neighbor if their orientations do not match
    for (PetscInt n = 0; n < N; ++n) {
      if (seen[n]) continue;
      seen[n]          = PETSC_TRUE;
      queue[qBottom++] = n;
      while (qTop < qBottom) {
        const PetscInt a = queue[qTop++];

        for (PetscInt i = aoff[a]; i < aoff[a + 1]; ++i) {
          const PetscInt  b    = adj[2 * i];
          const PetscBool flip = adj[2 * i + 1] ? flips[a] : (PetscBool)!flips[a];

          if (!seen[b]) {
            seen[b]          = PETSC_TRUE;
            flips[b]         = flip;
            queue[qBottom++] = b;
          } else if (flips[b] != flip) orientable = PETSC_FALSE;
        }
      }
    }
  }
  PetscCallMPI(MPI_Bcast(&orientable, 1, MPI_C_BOOL, 0, comm));
  PetscCheck(orientable, comm, PETSC_ERR_ARG_WRONG, "Mesh is non-orientable");
  PetscCallMPI(MPI_Scatterv(flips, Nc, coff, MPI_C_BOOL, flipped, iNcomp, MPI_C_BOOL, 0, comm));
  if (rank == 0) {
    PetscCall(PetscFree5(aoff, adj, queue, flips, seen));
    PetscCall(PetscFree(allEdges));
    PetscCall(PetscFree4(Nc, coff, ecounts, eoff));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The orientation operation needs to orient both a bulk mesh and a manifold embedded in a mesh, whose faces can be
  shared by several processes, and whose cells can have copies on several processes when the mesh has overlap.

  1) Each process orients its local cells, and counts the connected components.

  2) The owner of each face gathers the orientation that every copy of every cell next to the face induces on it.

  3) The owner links the components of these cells into a process graph, whose edges say whether two components match.

  4) Process 0 gathers the graph, finds which components to flip, and returns the result to each process.

  5) Each process flips the cells of the flipped components.
*/
PetscErrorCode DMPlexOrientCells_Internal(DM dm, IS cellIS, IS faceIS)
{
  const PetscInt  debug = ((DM_Plex *)dm->data)->printOrient;
  const PetscInt *cells = NULL, *faces = NULL;
  PetscInt        cStart = 0, cEnd = 0, fStart = 0, fEnd = 0;
  PetscBT         cellFlip;  // The bit is true if a cell should have its orientation reversed
  PetscInt       *cellComp;  // The connected component number of each cell
  PetscInt        Ncomp = 0; // The number of local connected components
  PetscInt       *off, *edges, Ne = 0;
  FaceSample     *samples;
  PetscBool      *flipped;
  MPI_Comm        comm;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (cellIS) PetscCall(ISGetPointRange(cellIS, &cStart, &cEnd, &cells));
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(PetscBTCreate(cEnd - cStart, &cellFlip));
  PetscCall(PetscBTMemzero(cEnd - cStart, cellFlip));
  PetscCall(PetscMalloc1(cEnd - cStart, &cellComp));
  // Phase 1: Serial Orientation
  PetscCall(DMPlexOrient_Serial(dm, cellIS, faceIS, &Ncomp, cellComp, cellFlip));
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
  // Phase 2
  PetscCall(DMPlexGetFaceSamples_Private(dm, cellIS, faceIS, cellComp, cellFlip, &off, &samples));
  // Phase 3: Link the first sample of each face to the samples from other components
  for (PetscInt pass = 0; pass < 2; ++pass) {
    PetscInt e = 0;

    for (PetscInt f = 0; f < fEnd - fStart; ++f) {
      const FaceSample *first = NULL;

      for (PetscInt i = off[f]; i < off[f + 1]; ++i) {
        const FaceSample *s = &samples[i];

        if (!s->ornt) continue;
        if (!first) first = s;
        else if (s->rank != first->rank || s->comp != first->comp) {
          if (pass) {
            edges[5 * e + 0] = first->rank;
            edges[5 * e + 1] = first->comp;
            edges[5 * e + 2] = s->rank;
            edges[5 * e + 3] = s->comp;
            edges[5 * e + 4] = FaceSamplesMatch(first, s);
          }
          ++e;
        }
      }
    }
    if (!pass) {
      Ne = e;
      PetscCall(PetscMalloc1(5 * Ne, &edges));
    }
  }
  // Phase 4
  PetscCall(PetscMalloc1(Ncomp, &flipped));
  PetscCall(DMPlexOrientSolve_Private(dm, Ncomp, Ne, edges, flipped));
  // Phase 5
  for (PetscInt c = 0; c < cEnd - cStart; ++c) {
    if (flipped[cellComp[c]]) PetscCall(PetscBTNegate(cellFlip, c));
    if (PetscBTLookup(cellFlip, c)) PetscCall(DMPlexOrientPoint(dm, cells ? cells[cStart + c] : cStart + c, -1));
  }
  if (debug) {
    PetscViewer v;

    PetscCall(PetscViewerASCIIGetStdout(comm, &v));
    PetscCall(PetscViewerASCIIPushSynchronized(v));
    PetscCall(PetscViewerASCIISynchronizedPrintf(v, "[%d]BT for parallel flipped cells:\n", rank));
    PetscCall(PetscBTView(cEnd - cStart, cellFlip, v));
    PetscCall(PetscViewerFlush(v));
    PetscCall(PetscViewerASCIIPopSynchronized(v));
  }
  PetscCall(PetscFree(flipped));
  PetscCall(PetscFree(edges));
  PetscCall(PetscFree(off));
  PetscCall(PetscFree(samples));
  PetscCall(PetscBTDestroy(&cellFlip));
  PetscCall(PetscFree(cellComp));
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
  IS       cellIS, faceIS;
  PetscInt h, cStart, cEnd, fStart, fEnd;

  PetscFunctionBegin;
  PetscCall(DMPlexGetVTKCellHeight(dm, &h));
  PetscCall(DMPlexGetHeightStratum(dm, h, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(dm, h + 1, &fStart, &fEnd));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, cEnd - cStart, cStart, 1, &cellIS));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, fEnd - fStart, fStart, 1, &faceIS));
  PetscCall(DMPlexOrientCells_Internal(dm, cellIS, faceIS));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(ISDestroy(&faceIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The depth of the surface is the largest depth in the label on any process, since a process can hold only part of its closure
static PetscErrorCode CreateCellAndFaceIS_Private(DM dm, DMLabel label, IS *cellIS, IS *faceIS)
{
  IS              valueIS;
  const PetscInt *values;
  PetscInt        Nv, depth;

  PetscFunctionBegin;
  PetscCall(DMLabelGetValueIS(label, &valueIS));
  PetscCall(ISGetLocalSize(valueIS, &Nv));
  PetscCall(ISGetIndices(valueIS, &values));
  depth = Nv ? 0 : -1;
  for (PetscInt v = 0; v < Nv; ++v) {
    const PetscInt val = values[v] < 0 || values[v] >= 100 ? 0 : values[v];
    PetscInt       n;

    PetscCall(DMLabelGetStratumSize(label, val, &n));
    if (!n) continue;
    depth = PetscMax(val, depth);
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &depth, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)dm)));
  PetscCheck(depth, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Depth for interface must be at least 1, not %" PetscInt_FMT, depth);
  *cellIS = *faceIS = NULL;
  if (depth > 0) {
    PetscCall(DMLabelGetStratumIS(label, depth, cellIS));
    PetscCall(DMLabelGetStratumIS(label, depth - 1, faceIS));
  }
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

// Every pair of samples of a face must match
static PetscErrorCode DMPlexCheckOrientation_Internal(DM dm, IS cellIS, IS faceIS)
{
  const PetscInt  debug  = ((DM_Plex *)dm->data)->printOrient;
  const PetscInt *faces  = NULL;
  PetscInt        fStart = 0, fEnd = 0, *off;
  FaceSample     *samples;
  PetscBool       valid = PETSC_TRUE;

  PetscFunctionBegin;
  if (faceIS) PetscCall(ISGetPointRange(faceIS, &fStart, &fEnd, &faces));
  PetscCall(DMPlexGetFaceSamples_Private(dm, cellIS, faceIS, NULL, NULL, &off, &samples));
  for (PetscInt f = 0; f < fEnd - fStart; ++f) {
    for (PetscInt i = off[f]; i < off[f + 1]; ++i) {
      for (PetscInt j = i + 1; j < off[f + 1]; ++j) {
        if (!samples[i].ornt || !samples[j].ornt || FaceSamplesMatch(&samples[i], &samples[j])) continue;
        valid = PETSC_FALSE;
        if (debug)
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "Face %" PetscInt_FMT " is mismatched: cell (%" PetscInt_FMT ", %" PetscInt_FMT ") (%" PetscInt_FMT ") ~ cell (%" PetscInt_FMT ", %" PetscInt_FMT ") (%" PetscInt_FMT ")\n", faces ? faces[fStart + f] : fStart + f,
                                samples[i].owner.rank, samples[i].owner.index, samples[i].ornt, samples[j].owner.rank, samples[j].owner.index, samples[j].ornt));
      }
    }
  }
  PetscCall(PetscFree(off));
  PetscCall(PetscFree(samples));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_C_BOOL, MPI_LAND, PetscObjectComm((PetscObject)dm)));
  PetscCheck(valid, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Mesh was not properly oriented");
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
