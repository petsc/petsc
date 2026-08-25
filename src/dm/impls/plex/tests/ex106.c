static char help[] = "Test space-filling-curve reorder of a distributed cell list\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

static PetscErrorCode SetGlobalCellTags(MPI_Comm comm, PetscInt numCells, PetscInt tags[])
{
  PetscInt    off = 0;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Exscan(&numCells, &off, 1, MPIU_INT, MPI_SUM, comm));
  if (!rank) off = 0;
  for (PetscInt c = 0; c < numCells; ++c) tags[c] = off + c;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Verify that migrationSF maps the input cells onto the output cells one-to-one. Each input cell
// carries a globally unique tag. After migration every tag must appear exactly once.
static PetscErrorCode CheckMigrationIsPermutation(MPI_Comm comm, PetscSF migrationSF, PetscInt numCells, PetscInt newNumCells, PetscInt NCells)
{
  PetscInt *tags, *newtags, *hist;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc2(numCells, &tags, newNumCells, &newtags));
  PetscCall(PetscCalloc1(NCells, &hist));
  PetscCall(SetGlobalCellTags(comm, numCells, tags));
  PetscCall(PetscSFBcastBegin(migrationSF, MPIU_INT, tags, newtags, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(migrationSF, MPIU_INT, tags, newtags, MPI_REPLACE));
  for (PetscInt c = 0; c < newNumCells; ++c) {
    PetscCheck(newtags[c] >= 0 && newtags[c] < NCells, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Migrated tag %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", newtags[c], NCells);
    ++hist[newtags[c]];
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, hist, NCells, MPIU_INT, MPI_SUM, comm));
  for (PetscInt c = 0; c < NCells; ++c) PetscCheck(hist[c] == 1, comm, PETSC_ERR_PLIB, "Cell tag %" PetscInt_FMT " appears %" PetscInt_FMT " times, not once", c, hist[c]);
  PetscCall(PetscFree(hist));
  PetscCall(PetscFree2(tags, newtags));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Independent Morton encoder, so the test does not reuse the implementation it checks. It must
// mirror the contract of DMPLEXCURVEMORTON: 21 bits per axis over the global bounding box, with
// axis 0 in the highest of each interleaved triple.
static uint64_t TestZEncode1(PetscInt t)
{
  uint64_t z = (uint64_t)t & 0x1fffff;

  z = (z | (z << 32)) & UINT64_C(0x1f00000000ffff);
  z = (z | (z << 16)) & UINT64_C(0x1f0000ff0000ff);
  z = (z | (z << 8)) & UINT64_C(0x100f00f00f00f00f);
  z = (z | (z << 4)) & UINT64_C(0x10c30c30c30c30c3);
  z = (z | (z << 2)) & UINT64_C(0x1249249249249249);
  return z;
}

// Verify that the reordered cells form one ascending run of the curve. If useTags is true on every
// process, use tags as the tie-breaker for equal codes, matching the implementation's globally
// unique cell number. Empty processes may pass NULL for tags.
static PetscErrorCode CheckGloballyCurveSorted(MPI_Comm comm, PetscInt spaceDim, PetscInt n, const PetscReal centroids[], PetscBool useTags, const PetscInt tags[])
{
  PetscReal      lo[3], hi[3], span[3];
  PetscInt64    *codes;
  PetscInt64    *bounds;
  PetscInt64     prevmax = PETSC_INT64_MIN;
  PetscInt      *counts;
  PetscInt      *tagbounds = NULL;
  PetscInt       prevtag   = PETSC_INT_MIN;
  const PetscInt maxidx    = (1 << 21) - 1;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscAssert(!useTags || !n || tags, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "useTags is set but tags is NULL on a process with %" PetscInt_FMT " cells", n);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  for (PetscInt d = 0; d < 3; ++d) {
    lo[d] = PETSC_MAX_REAL;
    hi[d] = PETSC_MIN_REAL;
  }
  for (PetscInt c = 0; c < n; ++c) {
    for (PetscInt d = 0; d < spaceDim; ++d) {
      lo[d] = PetscMin(lo[d], centroids[c * spaceDim + d]);
      hi[d] = PetscMax(hi[d], centroids[c * spaceDim + d]);
    }
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, lo, 3, MPIU_REAL, MPIU_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, hi, 3, MPIU_REAL, MPIU_MAX, comm));
  for (PetscInt d = 0; d < 3; ++d) {
    if (lo[d] > hi[d]) {
      lo[d] = 0.;
      hi[d] = 0.;
    }
  }
  for (PetscInt d = 0; d < 3; ++d) span[d] = hi[d] > lo[d] ? hi[d] - lo[d] : 1.;

  PetscCall(PetscMalloc1(PetscMax(1, n), &codes));
  for (PetscInt c = 0; c < n; ++c) {
    PetscInt q[3] = {0, 0, 0};

    for (PetscInt d = 0; d < spaceDim; ++d) {
      const PetscReal t = (centroids[c * spaceDim + d] - lo[d]) / span[d];

      q[d] = PetscMax(0, PetscMin(maxidx, (PetscInt)(t * (PetscReal)maxidx)));
    }
    codes[c] = (PetscInt64)((TestZEncode1(q[0]) << 2) | (TestZEncode1(q[1]) << 1) | TestZEncode1(q[2]));
  }
  for (PetscInt c = 1; c < n; ++c) PetscCheck(codes[c - 1] < codes[c] || (codes[c - 1] == codes[c] && (!useTags || tags[c - 1] < tags[c])), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local curve keys not ascending at %" PetscInt_FMT, c);

  // Exchange each rank's key range. The counts identify empty ranks without reserving a sentinel
  // value, because PETSC_INT64_MAX is itself a valid Morton code.
  PetscCall(PetscMalloc1(2 * size, &bounds));
  PetscCall(PetscMalloc1(size, &counts));
  PetscCallMPI(MPI_Allgather(&n, 1, MPIU_INT, counts, 1, MPIU_INT, comm));
  {
    PetscInt64 mine[2];

    mine[0] = n ? codes[0] : 0;
    mine[1] = n ? codes[n - 1] : 0;
    PetscCallMPI(MPI_Allgather(mine, 2, MPIU_INT64, bounds, 2, MPIU_INT64, comm));
  }
  if (useTags) {
    PetscInt mine[2];

    PetscCall(PetscMalloc1(2 * size, &tagbounds));
    mine[0] = n ? tags[0] : 0;
    mine[1] = n ? tags[n - 1] : 0;
    PetscCallMPI(MPI_Allgather(mine, 2, MPIU_INT, tagbounds, 2, MPIU_INT, comm));
  }
  for (PetscMPIInt r = 0; r < size; ++r) {
    PetscBool ordered;

    if (!counts[r]) continue;
    ordered = prevmax < bounds[2 * r] || (prevmax == bounds[2 * r] && (!useTags || prevtag < tagbounds[2 * r]));
    PetscCheck(ordered, comm, PETSC_ERR_PLIB, "Curve order broken across ranks before rank %d", r);
    prevmax = bounds[2 * r + 1];
    if (useTags) prevtag = tagbounds[2 * r + 1];
  }
  PetscCall(PetscFree(tagbounds));
  PetscCall(PetscFree(counts));
  PetscCall(PetscFree(bounds));
  PetscCall(PetscFree(codes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part A: reorder a lattice of synthetic 3-D centroids. The initial distribution is strided, so
// every rank starts with cells spread over the whole domain.
static PetscErrorCode TestFromCentroids(MPI_Comm comm, PetscInt N, PetscBool allOnRank0)
{
  PetscSF     sf;
  PetscReal  *centroids, *newcentroids;
  PetscInt   *tags, *newtags;
  PetscInt    numCells = 0, newNumCells, NCells = N * N * N, gnew;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  // Two input distributions. The strided one gives every rank cells spread over the whole domain.
  // The other puts every cell on rank 0, which is what reading a mesh serially and building in
  // parallel produces, and which leaves every other rank with nothing to sample.
  for (PetscInt g = 0; g < NCells; ++g)
    if (allOnRank0 ? rank == 0 : g % size == rank) ++numCells;
  PetscCall(PetscMalloc2(PetscMax(1, numCells) * 3, &centroids, PetscMax(1, numCells), &tags));
  {
    PetscInt c = 0;

    for (PetscInt g = 0; g < NCells; ++g) {
      if (allOnRank0 ? rank != 0 : g % size != rank) continue;
      centroids[c * 3 + 0] = (PetscReal)(g % N) + 0.5;
      centroids[c * 3 + 1] = (PetscReal)((g / N) % N) + 0.5;
      centroids[c * 3 + 2] = (PetscReal)(g / (N * N)) + 0.5;
      ++c;
    }
  }
  PetscCall(SetGlobalCellTags(comm, numCells, tags));
  PetscCall(DMPlexReorderCellListByCurveFromCentroids(comm, DMPLEXCURVEMORTON, 3, numCells, centroids, &sf, &newNumCells));
  gnew = newNumCells;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &gnew, 1, MPIU_INT, MPI_SUM, comm));
  PetscCheck(gnew == NCells, comm, PETSC_ERR_PLIB, "Global cell count changed from %" PetscInt_FMT " to %" PetscInt_FMT, NCells, gnew);
  PetscCall(CheckMigrationIsPermutation(comm, sf, numCells, newNumCells, NCells));

  // Migrate the centroids themselves so the locality of the new distribution can be measured.
  PetscCall(PetscMalloc2(newNumCells * 3, &newcentroids, newNumCells, &newtags));
  {
    MPI_Datatype ctype;

    PetscCallMPI(MPI_Type_contiguous(3, MPIU_REAL, &ctype));
    PetscCallMPI(MPI_Type_commit(&ctype));
    PetscCall(PetscSFBcastBegin(sf, ctype, centroids, newcentroids, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, ctype, centroids, newcentroids, MPI_REPLACE));
    PetscCallMPI(MPI_Type_free(&ctype));
  }
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, tags, newtags, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, tags, newtags, MPI_REPLACE));
  PetscCall(CheckGloballyCurveSorted(comm, 3, newNumCells, newcentroids, PETSC_TRUE, newtags));
  // The reorder must redistribute, not merely permute in place, and it must split the cells evenly.
  // The second pass equidistributes from the global position of each cell, so the counts differ by
  // at most one. The splitters alone only bound the busiest rank at twice the average.
  {
    PetscInt hi = newNumCells, lo = newNumCells;

    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &hi, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &lo, 1, MPIU_INT, MPI_MIN, comm));
    PetscCheck(hi - lo <= 1, comm, PETSC_ERR_PLIB, "Cells per rank run from %" PetscInt_FMT " to %" PetscInt_FMT ", which is not an exact split", lo, hi);
    // Every rank must receive cells once there are enough to go round.
    PetscCheck(NCells < (PetscInt)size || lo > 0, comm, PETSC_ERR_PLIB, "A rank received no cells");
  }
  PetscCall(PetscPrintf(comm, "FromCentroids: N=%" PetscInt_FMT " cells=%" PetscInt_FMT " permutation ok, globally curve sorted, balanced\n", N, NCells));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree2(newcentroids, newtags));
  PetscCall(PetscFree2(centroids, tags));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part B: reorder the connectivity of an N x N quadrilateral grid, then build a DMPlex from the
// reordered list. The vertex distribution stays in natural order, as the reorder requires.
static PetscErrorCode TestCellList(MPI_Comm comm, PetscInt N)
{
  DM          dm;
  PetscSF     sf;
  PetscInt   *cells, *newcells, *cellsSaved = NULL;
  PetscReal  *coords;
  PetscLayout vlayout;
  PetscInt    numCells = 0, newNumCells, NCells = N * N, NVertices = (N + 1) * (N + 1);
  PetscInt    numVertices, vStart, vEnd, cStart, cEnd, gcells;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  for (PetscInt g = 0; g < NCells; ++g)
    if (g % size == rank) ++numCells;
  PetscCall(PetscMalloc1(numCells * 4, &cells));
  {
    PetscInt c = 0;

    for (PetscInt g = 0; g < NCells; ++g) {
      const PetscInt i = g % N, j = g / N;

      if (g % size != rank) continue;
      cells[c * 4 + 0] = j * (N + 1) + i;
      cells[c * 4 + 1] = j * (N + 1) + i + 1;
      cells[c * 4 + 2] = (j + 1) * (N + 1) + i + 1;
      cells[c * 4 + 3] = (j + 1) * (N + 1) + i;
      ++c;
    }
  }
  // Own a contiguous slice of the vertices, matching what DMPlexCreateFromCellListParallelPetsc() expects.
  PetscCall(PetscLayoutCreate(comm, &vlayout));
  PetscCall(PetscLayoutSetSize(vlayout, NVertices));
  PetscCall(PetscLayoutSetBlockSize(vlayout, 1));
  PetscCall(PetscLayoutSetUp(vlayout));
  PetscCall(PetscLayoutGetRange(vlayout, &vStart, &vEnd));
  PetscCall(PetscLayoutDestroy(&vlayout));
  numVertices = vEnd - vStart;
  PetscCall(PetscMalloc1(numVertices * 2, &coords));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    coords[(v - vStart) * 2 + 0] = (PetscReal)(v % (N + 1));
    coords[(v - vStart) * 2 + 1] = (PetscReal)(v / (N + 1));
  }

  PetscCall(DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, numCells, 4, cells, 2, numVertices, NVertices, coords, &sf, &newNumCells, &newcells));
  PetscCall(CheckMigrationIsPermutation(comm, sf, numCells, newNumCells, NCells));
  for (PetscInt i = 0; i < newNumCells * 4; ++i) PetscCheck(newcells[i] >= 0 && newcells[i] < NVertices, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Reordered connectivity entry %" PetscInt_FMT " out of range", newcells[i]);

  // The reordered list must build a valid, interpolated, distributed plex.
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, 2, newNumCells, numVertices, NVertices, 4, PETSC_TRUE, newcells, 2, coords, NULL, &cellsSaved, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  gcells = cEnd - cStart;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &gcells, 1, MPIU_INT, MPI_SUM, comm));
  PetscCheck(gcells == NCells, comm, PETSC_ERR_PLIB, "Built plex has %" PetscInt_FMT " cells, expected %" PetscInt_FMT, gcells, NCells);
  PetscCall(DMPlexCheck(dm));
  PetscCall(PetscPrintf(comm, "CellList: N=%" PetscInt_FMT " cells=%" PetscInt_FMT " plex built and checked\n", N, NCells));
  PetscCall(PetscFree(cellsSaved));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(newcells));
  PetscCall(PetscFree(coords));
  PetscCall(PetscFree(cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part C: the reorder exists to make parallel interpolation cheap. After interpolation the number
// of shared points measures how much of the mesh sits on rank boundaries. Reordering must not
// increase it.
static PetscErrorCode BuildAndCountSharedPoints(MPI_Comm comm, PetscInt numCells, const PetscInt cells[], PetscInt numVertices, PetscInt NVertices, const PetscReal coords[], PetscInt *nshared)
{
  DM        dm;
  PetscSF   pointSF;
  PetscInt *cellsSaved = NULL;
  PetscInt  nleaves;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, 2, numCells, numVertices, NVertices, 4, PETSC_TRUE, cells, 2, coords, NULL, &cellsSaved, &dm));
  PetscCall(DMGetPointSF(dm, &pointSF));
  PetscCall(PetscSFGetGraph(pointSF, NULL, &nleaves, NULL, NULL));
  *nshared = nleaves;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, nshared, 1, MPIU_INT, MPI_SUM, comm));
  PetscCall(PetscFree(cellsSaved));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestLocalityImproves(MPI_Comm comm, PetscInt N)
{
  PetscSF     sf;
  PetscInt   *cells, *newcells;
  PetscReal  *coords;
  PetscLayout vlayout;
  PetscInt    numCells = 0, newNumCells, NCells = N * N, NVertices = (N + 1) * (N + 1);
  PetscInt    numVertices, vStart, vEnd, sharedBefore, sharedAfter;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  for (PetscInt g = 0; g < NCells; ++g)
    if (g % size == rank) ++numCells;
  PetscCall(PetscMalloc1(numCells * 4, &cells));
  {
    PetscInt c = 0;

    for (PetscInt g = 0; g < NCells; ++g) {
      const PetscInt i = g % N, j = g / N;

      if (g % size != rank) continue;
      cells[c * 4 + 0] = j * (N + 1) + i;
      cells[c * 4 + 1] = j * (N + 1) + i + 1;
      cells[c * 4 + 2] = (j + 1) * (N + 1) + i + 1;
      cells[c * 4 + 3] = (j + 1) * (N + 1) + i;
      ++c;
    }
  }
  PetscCall(PetscLayoutCreate(comm, &vlayout));
  PetscCall(PetscLayoutSetSize(vlayout, NVertices));
  PetscCall(PetscLayoutSetBlockSize(vlayout, 1));
  PetscCall(PetscLayoutSetUp(vlayout));
  PetscCall(PetscLayoutGetRange(vlayout, &vStart, &vEnd));
  PetscCall(PetscLayoutDestroy(&vlayout));
  numVertices = vEnd - vStart;
  PetscCall(PetscMalloc1(numVertices * 2, &coords));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    coords[(v - vStart) * 2 + 0] = (PetscReal)(v % (N + 1));
    coords[(v - vStart) * 2 + 1] = (PetscReal)(v / (N + 1));
  }

  PetscCall(BuildAndCountSharedPoints(comm, numCells, cells, numVertices, NVertices, coords, &sharedBefore));
  PetscCall(DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, numCells, 4, cells, 2, numVertices, NVertices, coords, &sf, &newNumCells, &newcells));
  PetscCall(BuildAndCountSharedPoints(comm, newNumCells, newcells, numVertices, NVertices, coords, &sharedAfter));
  PetscCheck(sharedAfter <= sharedBefore, comm, PETSC_ERR_PLIB, "Shared points grew from %" PetscInt_FMT " to %" PetscInt_FMT, sharedBefore, sharedAfter);
  // A strided input distribution shares almost every point. The reorder must cut that sharply, not
  // merely avoid making it worse, so require at least a factor of two on more than one process.
  // How much there is to gain depends on how many cells each process receives. With fewer than a
  // couple of cells per process every cell touches a boundary and nothing can improve, so only
  // require that the reorder does no harm. Above that, require a modest reduction, and once each
  // process holds a real block require a factor of two. Measured values: a 2x2 grid over 8
  // processes gives no change, an 8x8 grid over 8 processes gives 1.68, and a 16x16 grid gives
  // more than 4.
  if (size > 1 && NCells >= 8 * (PetscInt)size) {
    PetscCheck(4 * sharedAfter <= 3 * sharedBefore, comm, PETSC_ERR_PLIB, "Shared points only fell from %" PetscInt_FMT " to %" PetscInt_FMT ", less than a quarter", sharedBefore, sharedAfter);
    PetscCheck(NCells < 32 * (PetscInt)size || 2 * sharedAfter <= sharedBefore, comm, PETSC_ERR_PLIB, "Shared points only fell from %" PetscInt_FMT " to %" PetscInt_FMT ", less than a factor of two", sharedBefore, sharedAfter);
  }
  // The counts depend on the number of processes, so keep them out of the reference output.
  PetscCall(PetscPrintf(comm, "Locality: reorder cuts shared points\n"));
  PetscCall(PetscInfo(NULL, "shared points %" PetscInt_FMT " -> %" PetscInt_FMT "\n", sharedBefore, sharedAfter));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(newcells));
  PetscCall(PetscFree(coords));
  PetscCall(PetscFree(cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part D: degenerate geometry. A curve code alone cannot order cells that quantize to the same grid
// point, and one distant node is enough to make the whole bulk of a mesh do that. The reorder must
// still balance, because a rank that receives every cell is worse than no reorder at all.
static PetscErrorCode TestDegenerateGeometry(MPI_Comm comm, PetscInt N)
{
  const char    *names[] = {"identical centroids", "outlier stretches box", "mostly coincident", "corner on every rank"};
  const PetscInt nkinds  = (PetscInt)(sizeof(names) / sizeof(names[0]));
  PetscMPIInt    size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  for (PetscInt kind = 0; kind < nkinds; ++kind) {
    PetscSF    sf;
    PetscReal *cent, *newcent;
    PetscInt  *tags, *newtags;
    PetscInt   NCells = kind == 3 ? 4095 : N * N * N, numCells = 0, newNumCells, lo, hi, tot;

    for (PetscInt g = 0; g < NCells; ++g)
      if (g % size == rank) ++numCells;
    PetscCall(PetscMalloc2(PetscMax(1, numCells) * 3, &cent, PetscMax(1, numCells), &tags));
    {
      PetscInt c = 0;

      for (PetscInt g = 0; g < NCells; ++g) {
        if (g % size != rank) continue;
        if (kind == 0) { // every centroid the same point
          cent[c * 3 + 0] = 1.5;
          cent[c * 3 + 1] = 2.5;
          cent[c * 3 + 2] = 3.5;
        } else if (kind == 1) { // a tight mesh plus one very distant node
          cent[c * 3 + 0] = (PetscReal)(g % N) * 1e-3;
          cent[c * 3 + 1] = (PetscReal)((g / N) % N) * 1e-3;
          cent[c * 3 + 2] = (PetscReal)(g / (N * N)) * 1e-3;
          if (g == 0) {
            cent[c * 3 + 0] = 1e6;
            cent[c * 3 + 1] = 1e6;
            cent[c * 3 + 2] = 1e6;
          }
        } else if (kind == 3) {
          // Every rank holds cells across the whole box, and its first cell sits on the corner, so
          // every rank's own smallest curve code equals the global smallest. A mesh file in
          // generator order looks like this. Too few samples then place a splitter on that shared
          // smallest value, and nearly every cell lands on one rank. The cell count is deliberately
          // indivisible so the per-rank sample counts differ after rounding.
          if (c == 0) {
            cent[c * 3 + 0] = 0.;
            cent[c * 3 + 1] = 0.;
            cent[c * 3 + 2] = 0.;
          } else {
            cent[c * 3 + 0] = (PetscReal)((g * 7919) % 64);
            cent[c * 3 + 1] = (PetscReal)((g * 6271) % 64);
            cent[c * 3 + 2] = (PetscReal)((g * 4643) % 64);
          }
        } else { // three quarters of the cells on one point
          if (g % 4) {
            cent[c * 3 + 0] = 5.;
            cent[c * 3 + 1] = 5.;
            cent[c * 3 + 2] = 5.;
          } else {
            cent[c * 3 + 0] = (PetscReal)(g % N);
            cent[c * 3 + 1] = (PetscReal)((g / N) % N);
            cent[c * 3 + 2] = (PetscReal)(g / (N * N));
          }
        }
        ++c;
      }
    }
    PetscCall(SetGlobalCellTags(comm, numCells, tags));
    PetscCall(DMPlexReorderCellListByCurveFromCentroids(comm, DMPLEXCURVEMORTON, 3, numCells, cent, &sf, &newNumCells));
    PetscCall(CheckMigrationIsPermutation(comm, sf, numCells, newNumCells, NCells));
    PetscCall(PetscMalloc2(newNumCells * 3, &newcent, newNumCells, &newtags));
    {
      MPI_Datatype ctype;

      PetscCallMPI(MPI_Type_contiguous(3, MPIU_REAL, &ctype));
      PetscCallMPI(MPI_Type_commit(&ctype));
      PetscCall(PetscSFBcastBegin(sf, ctype, cent, newcent, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf, ctype, cent, newcent, MPI_REPLACE));
      PetscCallMPI(MPI_Type_free(&ctype));
    }
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, tags, newtags, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, tags, newtags, MPI_REPLACE));
    PetscCall(CheckGloballyCurveSorted(comm, 3, newNumCells, newcent, PETSC_TRUE, newtags));
    lo  = newNumCells;
    hi  = newNumCells;
    tot = newNumCells;
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &lo, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &hi, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &tot, 1, MPIU_INT, MPI_SUM, comm));
    PetscCheck(tot == NCells, comm, PETSC_ERR_PLIB, "%s: cell count changed from %" PetscInt_FMT " to %" PetscInt_FMT, names[kind], NCells, tot);
    // The split must be exact whatever the geometry does to the curve codes. The permutation and
    // key-order checks above use this same degenerate input, because final balance alone cannot see
    // a first-pass defect after the exact split.
    PetscCheck(hi - lo <= 1, comm, PETSC_ERR_PLIB, "%s: cells per rank run from %" PetscInt_FMT " to %" PetscInt_FMT ", which is not an exact split", names[kind], lo, hi);
    // Every rank must receive cells. The curve codes here are all equal or nearly so, so a split
    // taken from the codes alone would leave one rank with every cell and the rest with none.
    PetscCheck(NCells < (PetscInt)size || lo > 0, comm, PETSC_ERR_PLIB, "%s: a rank received no cells; the curve codes do not separate these centroids, so the split must come from the cell numbering", names[kind]);
    PetscCall(PetscSFDestroy(&sf));
    PetscCall(PetscFree2(newcent, newtags));
    PetscCall(PetscFree2(cent, tags));
  }
  PetscCall(PetscPrintf(comm, "Degenerate geometry: %" PetscInt_FMT " cases permutation ok, globally key sorted, balanced\n", nkinds));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part E: the curve is selected by name. An unregistered name must be rejected, because a silent
// fall back to a default curve would order the cells along a curve the caller did not ask for.
static PetscErrorCode TestUnknownCurveType(MPI_Comm comm)
{
  const char    *bad[]       = {"hilbert", "morton ", "", NULL};
  const PetscInt nbad        = (PetscInt)(sizeof(bad) / sizeof(bad[0]));
  PetscReal      cent[3]     = {0.5, 0.5, 0.5};
  PetscSF        sf          = NULL;
  PetscInt      *newcells    = NULL;
  PetscInt       newNumCells = -1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < nbad; ++i) {
    // Every process passes the same name, so the collective check fails on all of them together.
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = DMPlexReorderCellListByCurveFromCentroids(comm, bad[i], 3, 1, cent, &sf, &newNumCells);
    PetscCall(PetscPopErrorHandler());
    PetscCheck(ierr == PETSC_ERR_ARG_UNKNOWN_TYPE, comm, PETSC_ERR_PLIB, "DMPlexReorderCellListByCurveFromCentroids() returned %d for curve \"%s\", not PETSC_ERR_ARG_UNKNOWN_TYPE", (int)ierr, bad[i] ? bad[i] : "(null)");
    // The connectivity interface checks the name through the centroid routine, so it gathers the
    // corner coordinates first. Pass no cells, which keeps that gather empty.
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = DMPlexReorderCellListByCurve(comm, bad[i], 0, 4, NULL, 3, 0, 0, NULL, &sf, &newNumCells, &newcells);
    PetscCall(PetscPopErrorHandler());
    PetscCheck(ierr == PETSC_ERR_ARG_UNKNOWN_TYPE, comm, PETSC_ERR_PLIB, "DMPlexReorderCellListByCurve() returned %d for curve \"%s\", not PETSC_ERR_ARG_UNKNOWN_TYPE", (int)ierr, bad[i] ? bad[i] : "(null)");
  }
  // A rejected call must leave the output arguments alone, so the caller frees nothing.
  PetscCheck(!sf && !newcells && newNumCells == -1, comm, PETSC_ERR_PLIB, "A rejected call wrote to its output arguments");
  PetscCall(PetscPrintf(comm, "Unknown curve: %" PetscInt_FMT " names rejected by both interfaces\n", nbad));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part F: one-dimensional coordinates. The Morton encoder interleaves three axes, and Parts A to D
// use two or three of them, so a single axis exercises a path of its own. In one dimension the
// curve is the axis itself, so the reordered cells must come out in ascending coordinate order.
static PetscErrorCode TestOneDimensional(MPI_Comm comm, PetscInt N)
{
  DM          dm;
  PetscSF     sf, sfCells;
  PetscReal  *cent, *newcent, *coords;
  PetscInt   *cells, *newcells, *cellsSaved = NULL;
  PetscLayout vlayout;
  PetscInt    numCells = 0, nnewCent, nnewList, NCells = N * N * N, NVertices = N * N * N + 1;
  PetscInt    numVertices, vStart, vEnd, gnew, cStart, cEnd, gcells;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  // A line of segments, distributed with a stride so that every process starts with cells spread
  // over the whole line.
  for (PetscInt g = 0; g < NCells; ++g)
    if (g % size == rank) ++numCells;
  PetscCall(PetscMalloc2(PetscMax(1, numCells), &cent, PetscMax(1, numCells) * 2, &cells));
  {
    PetscInt c = 0;

    for (PetscInt g = 0; g < NCells; ++g) {
      if (g % size != rank) continue;
      cent[c]          = (PetscReal)g + 0.5;
      cells[c * 2 + 0] = g;
      cells[c * 2 + 1] = g + 1;
      ++c;
    }
  }

  // The centroid interface in one dimension.
  PetscCall(DMPlexReorderCellListByCurveFromCentroids(comm, DMPLEXCURVEMORTON, 1, numCells, cent, &sf, &nnewCent));
  gnew = nnewCent;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &gnew, 1, MPIU_INT, MPI_SUM, comm));
  PetscCheck(gnew == NCells, comm, PETSC_ERR_PLIB, "Global cell count changed from %" PetscInt_FMT " to %" PetscInt_FMT, NCells, gnew);
  PetscCall(CheckMigrationIsPermutation(comm, sf, numCells, nnewCent, NCells));
  PetscCall(PetscMalloc1(PetscMax(1, nnewCent), &newcent));
  PetscCall(PetscSFBcastBegin(sf, MPIU_REAL, cent, newcent, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_REAL, cent, newcent, MPI_REPLACE));
  PetscCall(CheckGloballyCurveSorted(comm, 1, nnewCent, newcent, PETSC_FALSE, NULL));
  // On one axis the centroids are distinct and the curve is the axis, so the order is strict.
  for (PetscInt c = 1; c < nnewCent; ++c) PetscCheck(newcent[c - 1] < newcent[c], PETSC_COMM_SELF, PETSC_ERR_PLIB, "One-dimensional centroids not strictly ascending at %" PetscInt_FMT, c);

  // The connectivity interface in one dimension. Own a contiguous slice of the vertices.
  PetscCall(PetscLayoutCreate(comm, &vlayout));
  PetscCall(PetscLayoutSetSize(vlayout, NVertices));
  PetscCall(PetscLayoutSetBlockSize(vlayout, 1));
  PetscCall(PetscLayoutSetUp(vlayout));
  PetscCall(PetscLayoutGetRange(vlayout, &vStart, &vEnd));
  PetscCall(PetscLayoutDestroy(&vlayout));
  numVertices = vEnd - vStart;
  PetscCall(PetscMalloc1(PetscMax(1, numVertices), &coords));
  for (PetscInt v = vStart; v < vEnd; ++v) coords[v - vStart] = (PetscReal)v;
  // Pass PETSC_DECIDE for the global vertex count, which the routine accepts and which Part B does
  // not exercise. The layout then sums the local counts, which gives the same NVertices.
  PetscCall(DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, numCells, 2, cells, 1, numVertices, PETSC_DECIDE, coords, &sfCells, &nnewList, &newcells));
  PetscCheck(nnewList == nnewCent, comm, PETSC_ERR_PLIB, "The two interfaces split the same cells differently, %" PetscInt_FMT " against %" PetscInt_FMT, nnewList, nnewCent);
  PetscCall(CheckMigrationIsPermutation(comm, sfCells, numCells, nnewList, NCells));
  // Each segment must arrive whole, and the segments must arrive in ascending order.
  for (PetscInt c = 0; c < nnewList; ++c) {
    PetscCheck(newcells[c * 2 + 1] == newcells[c * 2] + 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Segment %" PetscInt_FMT " has vertices %" PetscInt_FMT " and %" PetscInt_FMT ", not consecutive", c, newcells[c * 2], newcells[c * 2 + 1]);
    PetscCheck(!c || newcells[(c - 1) * 2] < newcells[c * 2], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Segments not ascending at %" PetscInt_FMT, c);
  }

  // The reordered list must build a valid one-dimensional plex.
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, 1, nnewList, numVertices, NVertices, 2, PETSC_TRUE, newcells, 1, coords, NULL, &cellsSaved, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  gcells = cEnd - cStart;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &gcells, 1, MPIU_INT, MPI_SUM, comm));
  PetscCheck(gcells == NCells, comm, PETSC_ERR_PLIB, "Built plex has %" PetscInt_FMT " cells, expected %" PetscInt_FMT, gcells, NCells);
  PetscCall(DMPlexCheck(dm));
  PetscCall(PetscPrintf(comm, "OneDimensional: cells ascending on one axis, plex built and checked\n"));
  PetscCall(PetscFree(cellsSaved));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscSFDestroy(&sfCells));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(newcells));
  PetscCall(PetscFree(coords));
  PetscCall(PetscFree(newcent));
  PetscCall(PetscFree2(cent, cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part G: argument validation. Every process passes the same invalid value, so each call fails
// collectively and the error handler returns the code instead of aborting.
static PetscErrorCode TestArgumentValidation(MPI_Comm comm)
{
  const PetscInt baddim[]    = {0, 4, -1};
  const PetscInt ncorner[]   = {0, -3};
  const PetscInt ndim        = (PetscInt)(sizeof(baddim) / sizeof(baddim[0]));
  const PetscInt ncorn       = (PetscInt)(sizeof(ncorner) / sizeof(ncorner[0]));
  PetscReal      cent[3]     = {0.5, 0.5, 0.5};
  PetscSF        sf          = NULL;
  PetscInt      *newcells    = NULL;
  PetscInt       newNumCells = -1, nrejected = 0;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  // The centroid interface: spaceDim must be in [1, 3] and numCells must not be negative.
  for (PetscInt i = 0; i < ndim; ++i) {
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = DMPlexReorderCellListByCurveFromCentroids(comm, DMPLEXCURVEMORTON, baddim[i], 1, cent, &sf, &newNumCells);
    PetscCall(PetscPopErrorHandler());
    PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, comm, PETSC_ERR_PLIB, "DMPlexReorderCellListByCurveFromCentroids() returned %d for spaceDim %" PetscInt_FMT ", not PETSC_ERR_ARG_OUTOFRANGE", (int)ierr, baddim[i]);
    ++nrejected;
  }
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = DMPlexReorderCellListByCurveFromCentroids(comm, DMPLEXCURVEMORTON, 3, -1, cent, &sf, &newNumCells);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, comm, PETSC_ERR_PLIB, "DMPlexReorderCellListByCurveFromCentroids() returned %d for a negative numCells, not PETSC_ERR_ARG_OUTOFRANGE", (int)ierr);
  ++nrejected;

  // The connectivity interface: the same two ranges, and numCorners must be positive. Each check
  // runs before the routine allocates, so a rejected call leaks nothing.
  for (PetscInt i = 0; i < ndim; ++i) {
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, 0, 4, NULL, baddim[i], 0, 0, NULL, &sf, &newNumCells, &newcells);
    PetscCall(PetscPopErrorHandler());
    PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, comm, PETSC_ERR_PLIB, "DMPlexReorderCellListByCurve() returned %d for spaceDim %" PetscInt_FMT ", not PETSC_ERR_ARG_OUTOFRANGE", (int)ierr, baddim[i]);
    ++nrejected;
  }
  for (PetscInt i = 0; i < ncorn; ++i) {
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, 0, ncorner[i], NULL, 3, 0, 0, NULL, &sf, &newNumCells, &newcells);
    PetscCall(PetscPopErrorHandler());
    PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, comm, PETSC_ERR_PLIB, "DMPlexReorderCellListByCurve() returned %d for numCorners %" PetscInt_FMT ", not PETSC_ERR_ARG_OUTOFRANGE", (int)ierr, ncorner[i]);
    ++nrejected;
  }
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, -1, 4, NULL, 3, 0, 0, NULL, &sf, &newNumCells, &newcells);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, comm, PETSC_ERR_PLIB, "DMPlexReorderCellListByCurve() returned %d for a negative numCells, not PETSC_ERR_ARG_OUTOFRANGE", (int)ierr);
  ++nrejected;

  // The corner buffer of DMPlexReorderCellListByCurve() must fit in a PetscInt. The check runs
  // before the routine reads the connectivity, so the array stays NULL and nothing is allocated.
  // With 64-bit indices no reachable count passes PETSC_INT_MAX, so this case is 32-bit only.
  if (!PetscDefined(USE_64BIT_INDICES)) {
    PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, 300000000, 4, NULL, 2, 0, 0, NULL, &sf, &newNumCells, &newcells);
    PetscCall(PetscPopErrorHandler());
    PetscCheck(ierr == PETSC_ERR_SUP, comm, PETSC_ERR_PLIB, "DMPlexReorderCellListByCurve() returned %d for a corner buffer of 2.4e9 reals, not PETSC_ERR_SUP", (int)ierr);
    ++nrejected;
  }

  PetscCheck(!sf && !newcells && newNumCells == -1, comm, PETSC_ERR_PLIB, "A rejected call wrote to its output arguments");
  // Keep the count out of the message. It differs between a 32-bit and a 64-bit index build.
  PetscCall(PetscPrintf(comm, "Validation: every invalid argument set rejected\n"));
  PetscCall(PetscInfo(NULL, "invalid argument sets rejected %" PetscInt_FMT "\n", nrejected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part H: no cells anywhere. A reader that finds no cells of the requested type hands the reorder
// an empty list on every process. There is then no sample to take, so no splitter can come from the
// data, and both interfaces must return an empty migration rather than fail.
static PetscErrorCode TestNoCells(MPI_Comm comm)
{
  PetscSF   sf;
  PetscInt *newcells    = NULL;
  PetscInt  newNumCells = -1, nroots, nleaves;

  PetscFunctionBeginUser;
  PetscCall(DMPlexReorderCellListByCurveFromCentroids(comm, DMPLEXCURVEMORTON, 3, 0, NULL, &sf, &newNumCells));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, NULL, NULL));
  PetscCheck(newNumCells == 0 && nroots == 0 && nleaves == 0, comm, PETSC_ERR_PLIB, "Empty input gave %" PetscInt_FMT " cells with %" PetscInt_FMT " roots and %" PetscInt_FMT " leaves", newNumCells, nroots, nleaves);
  PetscCall(PetscSFDestroy(&sf));

  newNumCells = -1;
  PetscCall(DMPlexReorderCellListByCurve(comm, DMPLEXCURVEMORTON, 0, 4, NULL, 2, 0, PETSC_DECIDE, NULL, &sf, &newNumCells, &newcells));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, NULL, NULL));
  PetscCheck(newNumCells == 0 && nroots == 0 && nleaves == 0, comm, PETSC_ERR_PLIB, "Empty connectivity gave %" PetscInt_FMT " cells with %" PetscInt_FMT " roots and %" PetscInt_FMT " leaves", newNumCells, nroots, nleaves);
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(newcells));
  PetscCall(PetscPrintf(comm, "NoCells: empty input gives an empty migration\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part I: a mesh embedded in more than three dimensions. The curve interleaves three axes and the
// bounding box holds three values, so DMPlexGetOrdering() must reject a higher-dimensional mesh
// rather than write past that box. The check runs before the work arrays exist, so the rejected call
// frees everything, which every test proves because the suite runs with -malloc_dump.
static PetscErrorCode TestHighCoordinateDim(MPI_Comm comm)
{
  DM             dm;
  IS             perm = NULL;
  PetscLayout    vlayout;
  PetscReal     *coords;
  PetscInt      *cells, *cellsSaved = NULL;
  PetscInt       numCells = 0, numVertices, vStart, vEnd;
  const PetscInt N = 2, NCells = 4, NVertices = 9;
  PetscErrorCode ierr;
  PetscMPIInt    size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  // A 2 x 2 quadrilateral grid whose vertices carry four coordinates each.
  for (PetscInt g = 0; g < NCells; ++g)
    if (g % size == rank) ++numCells;
  PetscCall(PetscMalloc1(PetscMax(1, numCells) * 4, &cells));
  {
    PetscInt c = 0;

    for (PetscInt g = 0; g < NCells; ++g) {
      const PetscInt i = g % N, j = g / N;

      if (g % size != rank) continue;
      cells[c * 4 + 0] = j * (N + 1) + i;
      cells[c * 4 + 1] = j * (N + 1) + i + 1;
      cells[c * 4 + 2] = (j + 1) * (N + 1) + i + 1;
      cells[c * 4 + 3] = (j + 1) * (N + 1) + i;
      ++c;
    }
  }
  PetscCall(PetscLayoutCreate(comm, &vlayout));
  PetscCall(PetscLayoutSetSize(vlayout, NVertices));
  PetscCall(PetscLayoutSetBlockSize(vlayout, 1));
  PetscCall(PetscLayoutSetUp(vlayout));
  PetscCall(PetscLayoutGetRange(vlayout, &vStart, &vEnd));
  PetscCall(PetscLayoutDestroy(&vlayout));
  numVertices = vEnd - vStart;
  PetscCall(PetscMalloc1(PetscMax(1, numVertices) * 4, &coords));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    coords[(v - vStart) * 4 + 0] = (PetscReal)(v % (N + 1));
    coords[(v - vStart) * 4 + 1] = (PetscReal)(v / (N + 1));
    coords[(v - vStart) * 4 + 2] = 0.;
    coords[(v - vStart) * 4 + 3] = 0.;
  }
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, 2, numCells, numVertices, NVertices, 4, PETSC_TRUE, cells, 4, coords, NULL, &cellsSaved, &dm));

  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = DMPlexGetOrdering(dm, DMPLEXCURVEMORTON, NULL, &perm);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_ARG_OUTOFRANGE, comm, PETSC_ERR_PLIB, "DMPlexGetOrdering() returned %d for a mesh in four dimensions, not PETSC_ERR_ARG_OUTOFRANGE", (int)ierr);
  PetscCheck(!perm, comm, PETSC_ERR_PLIB, "The rejected call returned a permutation");
  PetscCall(PetscPrintf(comm, "HighCoordinateDim: four coordinates per vertex rejected\n"));
  PetscCall(PetscFree(cellsSaved));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFree(coords));
  PetscCall(PetscFree(cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part J: the sample-index numerator passes 2^31. DMPlexZCodeSelectSplitters() forms that product in
// 64 bits; a 32-bit product would wrap to a negative index, which its range check reports. Reaching
// that scale needs 32*size*NCells above 2^31, which is 4.3 million cells on 16 processes and 35
// million on two. That costs either processes or memory, so this part is off by default.
// The sample_overflow suffix turns it on. Run it by hand with, for example:
//
//   mpiexec -n 32 ./ex106 -n 4 -sample_overflow -overflow_cells 2500000
static PetscErrorCode TestSampleOverflow(MPI_Comm comm, PetscInt NCells)
{
  PetscSF     sf;
  PetscReal  *cent, *newcent;
  PetscInt64  maxProduct;
  PetscInt   *tags, *newtags;
  PetscInt    numCells, newNumCells, lo, hi, tot;
  PetscMPIInt size, rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  // Every cell starts on rank 0, the distribution that asks one rank for every sample. That rank
  // then takes 32*size samples, and the largest sample-index numerator is that count minus one
  // times the local cell count.
  numCells   = rank == 0 ? NCells : 0;
  maxProduct = (32 * (PetscInt64)size - 1) * (PetscInt64)NCells;
  PetscCheck(maxProduct > 2147483647, comm, PETSC_ERR_ARG_OUTOFRANGE, "The largest sample-index product here is %" PetscInt64_FMT ", which does not pass 2^31. Use more processes or more cells: 32*size*cells must pass 2^31", maxProduct);
  PetscCall(PetscMalloc2(PetscMax(1, numCells), &cent, PetscMax(1, numCells), &tags));
  // One axis, one cell per unit. The curve quantizes to 21 bits, so cells beyond 2^21 share a code
  // and the global cell number separates them.
  for (PetscInt c = 0; c < numCells; ++c) cent[c] = (PetscReal)c;
  PetscCall(SetGlobalCellTags(comm, numCells, tags));
  PetscCall(DMPlexReorderCellListByCurveFromCentroids(comm, DMPLEXCURVEMORTON, 1, numCells, cent, &sf, &newNumCells));
  PetscCall(PetscMalloc2(newNumCells, &newcent, newNumCells, &newtags));
  PetscCall(PetscSFBcastBegin(sf, MPIU_REAL, cent, newcent, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_REAL, cent, newcent, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, tags, newtags, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, tags, newtags, MPI_REPLACE));
  for (PetscInt c = 0; c < newNumCells; ++c) PetscCheck(newtags[c] >= 0 && newtags[c] < NCells, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Migrated tag %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", newtags[c], NCells);
  PetscCall(CheckGloballyCurveSorted(comm, 1, newNumCells, newcent, PETSC_TRUE, newtags));
  lo  = newNumCells;
  hi  = newNumCells;
  tot = newNumCells;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &lo, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &hi, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &tot, 1, MPIU_INT, MPI_SUM, comm));
  PetscCheck(tot == NCells, comm, PETSC_ERR_PLIB, "Cell count changed from %" PetscInt_FMT " to %" PetscInt_FMT, NCells, tot);
  // The exact pass hides splitter quality from the counts, so also verify the complete
  // lexicographic order above.
  PetscCheck(lo > 0, comm, PETSC_ERR_PLIB, "A rank received no cells");
  PetscCheck(hi - lo <= 1, comm, PETSC_ERR_PLIB, "Cells per rank run from %" PetscInt_FMT " to %" PetscInt_FMT ", which is not an exact split", lo, hi);
  PetscCall(PetscPrintf(comm, "SampleOverflow: largest sample-index product %" PetscInt64_FMT " above 2^31, split balanced from %" PetscInt_FMT " to %" PetscInt_FMT " cells\n", maxProduct, lo, hi));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree2(newcent, newtags));
  PetscCall(PetscFree2(cent, tags));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Part K: Morton ordering needs cell coordinates. A topology-only DMPLEX must be rejected before
// DMPlexGetCellCoordinates() reaches its coordinate-vector closure path.
static PetscErrorCode TestNoCoordinates(MPI_Comm comm)
{
  DM             dm;
  IS             perm    = NULL;
  PetscInt       cone[4] = {1, 2, 3, 4};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreate(comm, &dm));
  PetscCall(DMSetDimension(dm, 2));
  PetscCall(DMPlexSetChart(dm, 0, 5));
  PetscCall(DMPlexSetConeSize(dm, 0, 4));
  PetscCall(DMSetUp(dm));
  PetscCall(DMPlexSetCone(dm, 0, cone));
  PetscCall(DMPlexSymmetrize(dm));
  PetscCall(DMPlexStratify(dm));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = DMPlexGetOrdering(dm, DMPLEXCURVEMORTON, NULL, &perm);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_ARG_WRONGSTATE, comm, PETSC_ERR_PLIB, "DMPlexGetOrdering() returned %d for a mesh without coordinates, not PETSC_ERR_ARG_WRONGSTATE", (int)ierr);
  PetscCheck(!perm, comm, PETSC_ERR_PLIB, "The rejected call returned a permutation");
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscPrintf(comm, "NoCoordinates: Morton ordering rejected a mesh without coordinates\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MPI_Comm  comm;
  PetscBool overflow = PETSC_FALSE;
  PetscInt  N = 4, overflowCells = 2500000;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, "", "SFC cell-list reorder test options", "DMPLEX");
  PetscCall(PetscOptionsInt("-n", "Cells per side", "ex106.c", N, &N, NULL));
  PetscCall(PetscOptionsBool("-sample_overflow", "Run the sample-index case, which needs many processes and millions of cells", "ex106.c", overflow, &overflow, NULL));
  PetscCall(PetscOptionsInt("-overflow_cells", "Cells for -sample_overflow", "ex106.c", overflowCells, &overflowCells, NULL));
  PetscOptionsEnd();
  PetscCheck(N > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-n must be positive");
  PetscCheck(overflowCells > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-overflow_cells must be positive");
  PetscCall(TestFromCentroids(comm, N, PETSC_FALSE));
  PetscCall(TestFromCentroids(comm, N, PETSC_TRUE));
  PetscCall(TestCellList(comm, N));
  PetscCall(TestLocalityImproves(comm, N));
  PetscCall(TestDegenerateGeometry(comm, N));
  PetscCall(TestUnknownCurveType(comm));
  PetscCall(TestOneDimensional(comm, N));
  PetscCall(TestArgumentValidation(comm));
  PetscCall(TestNoCells(comm));
  PetscCall(TestHighCoordinateDim(comm));
  PetscCall(TestNoCoordinates(comm));
  if (overflow) PetscCall(TestSampleOverflow(comm, overflowCells));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    nsize: {{1 2 3 4}}
    args: -n 8

  test:
    suffix: 1
    nsize: {{2 5 7}}
    args: -n 16

  # Small and empty-rank cases. At 2 the locality check has only two cells per process. At 8 the
  # three-dimensional parts give every rank one cell. At 16 those parts leave half the ranks empty,
  # which exercises collective key checks with locally NULL tag arrays.
  test:
    suffix: empty_ranks
    nsize: {{2 8 16}}
    args: -n 2

  # Every cell starts on rank 0, the distribution a serial read produces. Check that the output is
  # still a sorted, balanced permutation when the first exchange starts from one process.
  test:
    suffix: skewed_input
    nsize: {{2 4 8}}
    args: -n 8

  # Degenerate geometry: coincident centroids, one distant node, and a corner cell on every rank.
  # Apply the permutation and lexicographic key-order invariants directly to these inputs.
  test:
    suffix: degenerate
    nsize: {{2 4 8}}
    args: -n 6

  # The sample-index product passes 2^31. This needs 32*size*cells above 2^31, so it costs either
  # processes or memory: 16 processes with 4.3 million cells takes 0.3 s and 200 MB on the loaded
  # process. A 32-bit product would wrap to a negative index at this scale, so this guards the 64-bit
  # cast in DMPlexZCodeSelectSplitters(). A build with 64-bit indices cannot form a 32-bit product,
  # so it skips the test and keeps the memory.
  test:
    suffix: sample_overflow
    nsize: 16
    requires: !defined(PETSC_USE_64BIT_INDICES)
    args: -n 4 -sample_overflow -overflow_cells 4300000

TEST*/
