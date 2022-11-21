#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <petsc/private/partitionerimpl.h>
#include <petsc/private/hashseti.h>

const char *const DMPlexCSRAlgorithms[] = {"mat", "graph", "overlap", "DMPlexCSRAlgorithm", "DM_PLEX_CSR_", NULL};

static inline PetscInt DMPlex_GlobalID(PetscInt point)
{
  return point >= 0 ? point : -(point + 1);
}

static PetscErrorCode DMPlexCreatePartitionerGraph_Overlap(DM dm, PetscInt height, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency, IS *globalNumbering)
{
  DM              ovdm;
  PetscSF         sfPoint;
  IS              cellNumbering;
  const PetscInt *cellNum;
  PetscInt       *adj = NULL, *vOffsets = NULL, *vAdj = NULL;
  PetscBool       useCone, useClosure;
  PetscInt        dim, depth, overlap, cStart, cEnd, c, v;
  PetscMPIInt     rank, size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  if (dim != depth) {
    /* We do not handle the uninterpolated case here */
    PetscCall(DMPlexCreateNeighborCSR(dm, height, numVertices, offsets, adjacency));
    /* DMPlexCreateNeighborCSR does not make a numbering */
    if (globalNumbering) PetscCall(DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, globalNumbering));
    /* Different behavior for empty graphs */
    if (!*numVertices) {
      PetscCall(PetscMalloc1(1, offsets));
      (*offsets)[0] = 0;
    }
    /* Broken in parallel */
    if (rank) PetscCheck(!*numVertices, PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel partitioning of uninterpolated meshes not supported");
    PetscFunctionReturn(0);
  }
  /* Always use FVM adjacency to create partitioner graph */
  PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
  /* Need overlap >= 1 */
  PetscCall(DMPlexGetOverlap(dm, &overlap));
  if (size && overlap < 1) {
    PetscCall(DMPlexDistributeOverlap(dm, 1, NULL, &ovdm));
  } else {
    PetscCall(PetscObjectReference((PetscObject)dm));
    ovdm = dm;
  }
  PetscCall(DMGetPointSF(ovdm, &sfPoint));
  PetscCall(DMPlexGetHeightStratum(ovdm, height, &cStart, &cEnd));
  PetscCall(DMPlexCreateNumbering_Plex(ovdm, cStart, cEnd, 0, NULL, sfPoint, &cellNumbering));
  if (globalNumbering) {
    PetscCall(PetscObjectReference((PetscObject)cellNumbering));
    *globalNumbering = cellNumbering;
  }
  PetscCall(ISGetIndices(cellNumbering, &cellNum));
  /* Determine sizes */
  for (*numVertices = 0, c = cStart; c < cEnd; ++c) {
    /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
    if (cellNum[c - cStart] < 0) continue;
    (*numVertices)++;
  }
  PetscCall(PetscCalloc1(*numVertices + 1, &vOffsets));
  for (c = cStart, v = 0; c < cEnd; ++c) {
    PetscInt adjSize = PETSC_DETERMINE, a, vsize = 0;

    if (cellNum[c - cStart] < 0) continue;
    PetscCall(DMPlexGetAdjacency(ovdm, c, &adjSize, &adj));
    for (a = 0; a < adjSize; ++a) {
      const PetscInt point = adj[a];
      if (point != c && cStart <= point && point < cEnd) ++vsize;
    }
    vOffsets[v + 1] = vOffsets[v] + vsize;
    ++v;
  }
  /* Determine adjacency */
  PetscCall(PetscMalloc1(vOffsets[*numVertices], &vAdj));
  for (c = cStart, v = 0; c < cEnd; ++c) {
    PetscInt adjSize = PETSC_DETERMINE, a, off = vOffsets[v];

    /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
    if (cellNum[c - cStart] < 0) continue;
    PetscCall(DMPlexGetAdjacency(ovdm, c, &adjSize, &adj));
    for (a = 0; a < adjSize; ++a) {
      const PetscInt point = adj[a];
      if (point != c && cStart <= point && point < cEnd) vAdj[off++] = DMPlex_GlobalID(cellNum[point - cStart]);
    }
    PetscCheck(off == vOffsets[v + 1], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Offsets %" PetscInt_FMT " should be %" PetscInt_FMT, off, vOffsets[v + 1]);
    /* Sort adjacencies (not strictly necessary) */
    PetscCall(PetscSortInt(off - vOffsets[v], &vAdj[vOffsets[v]]));
    ++v;
  }
  PetscCall(PetscFree(adj));
  PetscCall(ISRestoreIndices(cellNumbering, &cellNum));
  PetscCall(ISDestroy(&cellNumbering));
  PetscCall(DMSetBasicAdjacency(dm, useCone, useClosure));
  PetscCall(DMDestroy(&ovdm));
  if (offsets) {
    *offsets = vOffsets;
  } else PetscCall(PetscFree(vOffsets));
  if (adjacency) {
    *adjacency = vAdj;
  } else PetscCall(PetscFree(vAdj));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreatePartitionerGraph_Native(DM dm, PetscInt height, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency, IS *globalNumbering)
{
  PetscInt        dim, depth, p, pStart, pEnd, a, adjSize, idx, size;
  PetscInt       *adj = NULL, *vOffsets = NULL, *graph = NULL;
  IS              cellNumbering;
  const PetscInt *cellNum;
  PetscBool       useCone, useClosure;
  PetscSection    section;
  PetscSegBuffer  adjBuffer;
  PetscSF         sfPoint;
  PetscInt       *adjCells = NULL, *remoteCells = NULL;
  const PetscInt *local;
  PetscInt        nroots, nleaves, l;
  PetscMPIInt     rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  if (dim != depth) {
    /* We do not handle the uninterpolated case here */
    PetscCall(DMPlexCreateNeighborCSR(dm, height, numVertices, offsets, adjacency));
    /* DMPlexCreateNeighborCSR does not make a numbering */
    if (globalNumbering) PetscCall(DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, globalNumbering));
    /* Different behavior for empty graphs */
    if (!*numVertices) {
      PetscCall(PetscMalloc1(1, offsets));
      (*offsets)[0] = 0;
    }
    /* Broken in parallel */
    if (rank) PetscCheck(!*numVertices, PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel partitioning of uninterpolated meshes not supported");
    PetscFunctionReturn(0);
  }
  PetscCall(DMGetPointSF(dm, &sfPoint));
  PetscCall(DMPlexGetHeightStratum(dm, height, &pStart, &pEnd));
  /* Build adjacency graph via a section/segbuffer */
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &section));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));
  PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 1000, &adjBuffer));
  /* Always use FVM adjacency to create partitioner graph */
  PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
  PetscCall(DMPlexCreateNumbering_Plex(dm, pStart, pEnd, 0, NULL, sfPoint, &cellNumbering));
  if (globalNumbering) {
    PetscCall(PetscObjectReference((PetscObject)cellNumbering));
    *globalNumbering = cellNumbering;
  }
  PetscCall(ISGetIndices(cellNumbering, &cellNum));
  /* For all boundary faces (including faces adjacent to a ghost cell), record the local cell in adjCells
     Broadcast adjCells to remoteCells (to get cells from roots) and Reduce adjCells to remoteCells (to get cells from leaves)
   */
  PetscCall(PetscSFGetGraph(sfPoint, &nroots, &nleaves, &local, NULL));
  if (nroots >= 0) {
    PetscInt fStart, fEnd, f;

    PetscCall(PetscCalloc2(nroots, &adjCells, nroots, &remoteCells));
    PetscCall(DMPlexGetHeightStratum(dm, height + 1, &fStart, &fEnd));
    for (l = 0; l < nroots; ++l) adjCells[l] = -3;
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *support;
      PetscInt        supportSize;

      PetscCall(DMPlexGetSupport(dm, f, &support));
      PetscCall(DMPlexGetSupportSize(dm, f, &supportSize));
      if (supportSize == 1) adjCells[f] = DMPlex_GlobalID(cellNum[support[0] - pStart]);
      else if (supportSize == 2) {
        PetscCall(PetscFindInt(support[0], nleaves, local, &p));
        if (p >= 0) adjCells[f] = DMPlex_GlobalID(cellNum[support[1] - pStart]);
        PetscCall(PetscFindInt(support[1], nleaves, local, &p));
        if (p >= 0) adjCells[f] = DMPlex_GlobalID(cellNum[support[0] - pStart]);
      }
      /* Handle non-conforming meshes */
      if (supportSize > 2) {
        PetscInt        numChildren, i;
        const PetscInt *children;

        PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, &children));
        for (i = 0; i < numChildren; ++i) {
          const PetscInt child = children[i];
          if (fStart <= child && child < fEnd) {
            PetscCall(DMPlexGetSupport(dm, child, &support));
            PetscCall(DMPlexGetSupportSize(dm, child, &supportSize));
            if (supportSize == 1) adjCells[child] = DMPlex_GlobalID(cellNum[support[0] - pStart]);
            else if (supportSize == 2) {
              PetscCall(PetscFindInt(support[0], nleaves, local, &p));
              if (p >= 0) adjCells[child] = DMPlex_GlobalID(cellNum[support[1] - pStart]);
              PetscCall(PetscFindInt(support[1], nleaves, local, &p));
              if (p >= 0) adjCells[child] = DMPlex_GlobalID(cellNum[support[0] - pStart]);
            }
          }
        }
      }
    }
    for (l = 0; l < nroots; ++l) remoteCells[l] = -1;
    PetscCall(PetscSFBcastBegin(dm->sf, MPIU_INT, adjCells, remoteCells, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(dm->sf, MPIU_INT, adjCells, remoteCells, MPI_REPLACE));
    PetscCall(PetscSFReduceBegin(dm->sf, MPIU_INT, adjCells, remoteCells, MPI_MAX));
    PetscCall(PetscSFReduceEnd(dm->sf, MPIU_INT, adjCells, remoteCells, MPI_MAX));
  }
  /* Combine local and global adjacencies */
  for (*numVertices = 0, p = pStart; p < pEnd; p++) {
    /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
    if (nroots > 0) {
      if (cellNum[p - pStart] < 0) continue;
    }
    /* Add remote cells */
    if (remoteCells) {
      const PetscInt  gp = DMPlex_GlobalID(cellNum[p - pStart]);
      PetscInt        coneSize, numChildren, c, i;
      const PetscInt *cone, *children;

      PetscCall(DMPlexGetCone(dm, p, &cone));
      PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
      for (c = 0; c < coneSize; ++c) {
        const PetscInt point = cone[c];
        if (remoteCells[point] >= 0 && remoteCells[point] != gp) {
          PetscInt *PETSC_RESTRICT pBuf;
          PetscCall(PetscSectionAddDof(section, p, 1));
          PetscCall(PetscSegBufferGetInts(adjBuffer, 1, &pBuf));
          *pBuf = remoteCells[point];
        }
        /* Handle non-conforming meshes */
        PetscCall(DMPlexGetTreeChildren(dm, point, &numChildren, &children));
        for (i = 0; i < numChildren; ++i) {
          const PetscInt child = children[i];
          if (remoteCells[child] >= 0 && remoteCells[child] != gp) {
            PetscInt *PETSC_RESTRICT pBuf;
            PetscCall(PetscSectionAddDof(section, p, 1));
            PetscCall(PetscSegBufferGetInts(adjBuffer, 1, &pBuf));
            *pBuf = remoteCells[child];
          }
        }
      }
    }
    /* Add local cells */
    adjSize = PETSC_DETERMINE;
    PetscCall(DMPlexGetAdjacency(dm, p, &adjSize, &adj));
    for (a = 0; a < adjSize; ++a) {
      const PetscInt point = adj[a];
      if (point != p && pStart <= point && point < pEnd) {
        PetscInt *PETSC_RESTRICT pBuf;
        PetscCall(PetscSectionAddDof(section, p, 1));
        PetscCall(PetscSegBufferGetInts(adjBuffer, 1, &pBuf));
        *pBuf = DMPlex_GlobalID(cellNum[point - pStart]);
      }
    }
    (*numVertices)++;
  }
  PetscCall(PetscFree(adj));
  PetscCall(PetscFree2(adjCells, remoteCells));
  PetscCall(DMSetBasicAdjacency(dm, useCone, useClosure));

  /* Derive CSR graph from section/segbuffer */
  PetscCall(PetscSectionSetUp(section));
  PetscCall(PetscSectionGetStorageSize(section, &size));
  PetscCall(PetscMalloc1(*numVertices + 1, &vOffsets));
  for (idx = 0, p = pStart; p < pEnd; p++) {
    if (nroots > 0) {
      if (cellNum[p - pStart] < 0) continue;
    }
    PetscCall(PetscSectionGetOffset(section, p, &(vOffsets[idx++])));
  }
  vOffsets[*numVertices] = size;
  PetscCall(PetscSegBufferExtractAlloc(adjBuffer, &graph));

  if (nroots >= 0) {
    /* Filter out duplicate edges using section/segbuffer */
    PetscCall(PetscSectionReset(section));
    PetscCall(PetscSectionSetChart(section, 0, *numVertices));
    for (p = 0; p < *numVertices; p++) {
      PetscInt start = vOffsets[p], end = vOffsets[p + 1];
      PetscInt numEdges = end - start, *PETSC_RESTRICT edges;
      PetscCall(PetscSortRemoveDupsInt(&numEdges, &graph[start]));
      PetscCall(PetscSectionSetDof(section, p, numEdges));
      PetscCall(PetscSegBufferGetInts(adjBuffer, numEdges, &edges));
      PetscCall(PetscArraycpy(edges, &graph[start], numEdges));
    }
    PetscCall(PetscFree(vOffsets));
    PetscCall(PetscFree(graph));
    /* Derive CSR graph from section/segbuffer */
    PetscCall(PetscSectionSetUp(section));
    PetscCall(PetscSectionGetStorageSize(section, &size));
    PetscCall(PetscMalloc1(*numVertices + 1, &vOffsets));
    for (idx = 0, p = 0; p < *numVertices; p++) PetscCall(PetscSectionGetOffset(section, p, &(vOffsets[idx++])));
    vOffsets[*numVertices] = size;
    PetscCall(PetscSegBufferExtractAlloc(adjBuffer, &graph));
  } else {
    /* Sort adjacencies (not strictly necessary) */
    for (p = 0; p < *numVertices; p++) {
      PetscInt start = vOffsets[p], end = vOffsets[p + 1];
      PetscCall(PetscSortInt(end - start, &graph[start]));
    }
  }

  if (offsets) *offsets = vOffsets;
  if (adjacency) *adjacency = graph;

  /* Cleanup */
  PetscCall(PetscSegBufferDestroy(&adjBuffer));
  PetscCall(PetscSectionDestroy(&section));
  PetscCall(ISRestoreIndices(cellNumbering, &cellNum));
  PetscCall(ISDestroy(&cellNumbering));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreatePartitionerGraph_ViaMat(DM dm, PetscInt height, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency, IS *globalNumbering)
{
  Mat             conn, CSR;
  IS              fis, cis, cis_own;
  PetscSF         sfPoint;
  const PetscInt *rows, *cols, *ii, *jj;
  PetscInt       *idxs, *idxs2;
  PetscInt        dim, depth, floc, cloc, i, M, N, c, lm, m, cStart, cEnd, fStart, fEnd;
  PetscMPIInt     rank;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  if (dim != depth) {
    /* We do not handle the uninterpolated case here */
    PetscCall(DMPlexCreateNeighborCSR(dm, height, numVertices, offsets, adjacency));
    /* DMPlexCreateNeighborCSR does not make a numbering */
    if (globalNumbering) PetscCall(DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, globalNumbering));
    /* Different behavior for empty graphs */
    if (!*numVertices) {
      PetscCall(PetscMalloc1(1, offsets));
      (*offsets)[0] = 0;
    }
    /* Broken in parallel */
    if (rank) PetscCheck(!*numVertices, PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel partitioning of uninterpolated meshes not supported");
    PetscFunctionReturn(0);
  }
  /* Interpolated and parallel case */

  /* numbering */
  PetscCall(DMGetPointSF(dm, &sfPoint));
  PetscCall(DMPlexGetHeightStratum(dm, height, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(dm, height + 1, &fStart, &fEnd));
  PetscCall(DMPlexCreateNumbering_Plex(dm, cStart, cEnd, 0, &N, sfPoint, &cis));
  PetscCall(DMPlexCreateNumbering_Plex(dm, fStart, fEnd, 0, &M, sfPoint, &fis));
  if (globalNumbering) PetscCall(ISDuplicate(cis, globalNumbering));

  /* get positive global ids and local sizes for facets and cells */
  PetscCall(ISGetLocalSize(fis, &m));
  PetscCall(ISGetIndices(fis, &rows));
  PetscCall(PetscMalloc1(m, &idxs));
  for (i = 0, floc = 0; i < m; i++) {
    const PetscInt p = rows[i];

    if (p < 0) {
      idxs[i] = -(p + 1);
    } else {
      idxs[i] = p;
      floc += 1;
    }
  }
  PetscCall(ISRestoreIndices(fis, &rows));
  PetscCall(ISDestroy(&fis));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, idxs, PETSC_OWN_POINTER, &fis));

  PetscCall(ISGetLocalSize(cis, &m));
  PetscCall(ISGetIndices(cis, &cols));
  PetscCall(PetscMalloc1(m, &idxs));
  PetscCall(PetscMalloc1(m, &idxs2));
  for (i = 0, cloc = 0; i < m; i++) {
    const PetscInt p = cols[i];

    if (p < 0) {
      idxs[i] = -(p + 1);
    } else {
      idxs[i]       = p;
      idxs2[cloc++] = p;
    }
  }
  PetscCall(ISRestoreIndices(cis, &cols));
  PetscCall(ISDestroy(&cis));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), m, idxs, PETSC_OWN_POINTER, &cis));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), cloc, idxs2, PETSC_OWN_POINTER, &cis_own));

  /* Create matrix to hold F-C connectivity (MatMatTranspose Mult not supported for MPIAIJ) */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm), &conn));
  PetscCall(MatSetSizes(conn, floc, cloc, M, N));
  PetscCall(MatSetType(conn, MATMPIAIJ));
  PetscCall(DMPlexGetMaxSizes(dm, NULL, &lm));
  PetscCallMPI(MPI_Allreduce(&lm, &m, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)));
  PetscCall(MatMPIAIJSetPreallocation(conn, m, NULL, m, NULL));

  /* Assemble matrix */
  PetscCall(ISGetIndices(fis, &rows));
  PetscCall(ISGetIndices(cis, &cols));
  for (c = cStart; c < cEnd; c++) {
    const PetscInt *cone;
    PetscInt        coneSize, row, col, f;

    col = cols[c - cStart];
    PetscCall(DMPlexGetCone(dm, c, &cone));
    PetscCall(DMPlexGetConeSize(dm, c, &coneSize));
    for (f = 0; f < coneSize; f++) {
      const PetscScalar v = 1.0;
      const PetscInt   *children;
      PetscInt          numChildren, ch;

      row = rows[cone[f] - fStart];
      PetscCall(MatSetValues(conn, 1, &row, 1, &col, &v, INSERT_VALUES));

      /* non-conforming meshes */
      PetscCall(DMPlexGetTreeChildren(dm, cone[f], &numChildren, &children));
      for (ch = 0; ch < numChildren; ch++) {
        const PetscInt child = children[ch];

        if (child < fStart || child >= fEnd) continue;
        row = rows[child - fStart];
        PetscCall(MatSetValues(conn, 1, &row, 1, &col, &v, INSERT_VALUES));
      }
    }
  }
  PetscCall(ISRestoreIndices(fis, &rows));
  PetscCall(ISRestoreIndices(cis, &cols));
  PetscCall(ISDestroy(&fis));
  PetscCall(ISDestroy(&cis));
  PetscCall(MatAssemblyBegin(conn, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(conn, MAT_FINAL_ASSEMBLY));

  /* Get parallel CSR by doing conn^T * conn */
  PetscCall(MatTransposeMatMult(conn, conn, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CSR));
  PetscCall(MatDestroy(&conn));

  /* extract local part of the CSR */
  PetscCall(MatMPIAIJGetLocalMat(CSR, MAT_INITIAL_MATRIX, &conn));
  PetscCall(MatDestroy(&CSR));
  PetscCall(MatGetRowIJ(conn, 0, PETSC_FALSE, PETSC_FALSE, &m, &ii, &jj, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No IJ format");

  /* get back requested output */
  if (numVertices) *numVertices = m;
  if (offsets) {
    PetscCall(PetscCalloc1(m + 1, &idxs));
    for (i = 1; i < m + 1; i++) idxs[i] = ii[i] - i; /* ParMetis does not like self-connectivity */
    *offsets = idxs;
  }
  if (adjacency) {
    PetscCall(PetscMalloc1(ii[m] - m, &idxs));
    PetscCall(ISGetIndices(cis_own, &rows));
    for (i = 0, c = 0; i < m; i++) {
      PetscInt j, g = rows[i];

      for (j = ii[i]; j < ii[i + 1]; j++) {
        if (jj[j] == g) continue; /* again, self-connectivity */
        idxs[c++] = jj[j];
      }
    }
    PetscCheck(c == ii[m] - m, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected %" PetscInt_FMT " != %" PetscInt_FMT, c, ii[m] - m);
    PetscCall(ISRestoreIndices(cis_own, &rows));
    *adjacency = idxs;
  }

  /* cleanup */
  PetscCall(ISDestroy(&cis_own));
  PetscCall(MatRestoreRowIJ(conn, 0, PETSC_FALSE, PETSC_FALSE, &m, &ii, &jj, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No IJ format");
  PetscCall(MatDestroy(&conn));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreatePartitionerGraph - Create a CSR graph of point connections for the partitioner

  Collective on dm

  Input Parameters:
+ dm      - The mesh `DM`
- height  - Height of the strata from which to construct the graph

  Output Parameters:
+ numVertices     - Number of vertices in the graph
. offsets         - Point offsets in the graph
. adjacency       - Point connectivity in the graph
- globalNumbering - A map from the local cell numbering to the global numbering used in "adjacency".  Negative indicates that the cell is a duplicate from another process.

  Options Database Keys:
. -dm_plex_csr_alg <mat,graph,overlap> - Choose the algorithm for computing the CSR graph

  Level: developer

  Note:
  The user can control the definition of adjacency for the mesh using `DMSetAdjacency()`. They should choose the combination appropriate for the function
  representation on the mesh. If requested, globalNumbering needs to be destroyed by the caller; offsets and adjacency need to be freed with PetscFree().

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `PetscPartitionerGetType()`, `PetscPartitionerCreate()`, `DMSetAdjacency()`
@*/
PetscErrorCode DMPlexCreatePartitionerGraph(DM dm, PetscInt height, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency, IS *globalNumbering)
{
  DMPlexCSRAlgorithm alg = DM_PLEX_CSR_GRAPH;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetEnum(((PetscObject)dm)->options, ((PetscObject)dm)->prefix, "-dm_plex_csr_alg", DMPlexCSRAlgorithms, (PetscEnum *)&alg, NULL));
  switch (alg) {
  case DM_PLEX_CSR_MAT:
    PetscCall(DMPlexCreatePartitionerGraph_ViaMat(dm, height, numVertices, offsets, adjacency, globalNumbering));
    break;
  case DM_PLEX_CSR_GRAPH:
    PetscCall(DMPlexCreatePartitionerGraph_Native(dm, height, numVertices, offsets, adjacency, globalNumbering));
    break;
  case DM_PLEX_CSR_OVERLAP:
    PetscCall(DMPlexCreatePartitionerGraph_Overlap(dm, height, numVertices, offsets, adjacency, globalNumbering));
    break;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateNeighborCSR - Create a mesh graph (cell-cell adjacency) in parallel CSR format.

  Collective on dm

  Input Parameters:
+ dm - The `DMPLEX`
- cellHeight - The height of mesh points to treat as cells (default should be 0)

  Output Parameters:
+ numVertices - The number of local vertices in the graph, or cells in the mesh.
. offsets     - The offset to the adjacency list for each cell
- adjacency   - The adjacency list for all cells

  Level: advanced

  Note:
  This is suitable for input to a mesh partitioner like ParMetis.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexCreate()`
@*/
PetscErrorCode DMPlexCreateNeighborCSR(DM dm, PetscInt cellHeight, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency)
{
  const PetscInt maxFaceCases = 30;
  PetscInt       numFaceCases = 0;
  PetscInt       numFaceVertices[30]; /* maxFaceCases, C89 sucks sucks sucks */
  PetscInt      *off, *adj;
  PetscInt      *neighborCells = NULL;
  PetscInt       dim, cellDim, depth = 0, faceDepth, cStart, cEnd, c, numCells, cell;

  PetscFunctionBegin;
  /* For parallel partitioning, I think you have to communicate supports */
  PetscCall(DMGetDimension(dm, &dim));
  cellDim = dim - cellHeight;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  if (cEnd - cStart == 0) {
    if (numVertices) *numVertices = 0;
    if (offsets) *offsets = NULL;
    if (adjacency) *adjacency = NULL;
    PetscFunctionReturn(0);
  }
  numCells  = cEnd - cStart;
  faceDepth = depth - cellHeight;
  if (dim == depth) {
    PetscInt f, fStart, fEnd;

    PetscCall(PetscCalloc1(numCells + 1, &off));
    /* Count neighboring cells */
    PetscCall(DMPlexGetHeightStratum(dm, cellHeight + 1, &fStart, &fEnd));
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *support;
      PetscInt        supportSize;
      PetscCall(DMPlexGetSupportSize(dm, f, &supportSize));
      PetscCall(DMPlexGetSupport(dm, f, &support));
      if (supportSize == 2) {
        PetscInt numChildren;

        PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, NULL));
        if (!numChildren) {
          ++off[support[0] - cStart + 1];
          ++off[support[1] - cStart + 1];
        }
      }
    }
    /* Prefix sum */
    for (c = 1; c <= numCells; ++c) off[c] += off[c - 1];
    if (adjacency) {
      PetscInt *tmp;

      PetscCall(PetscMalloc1(off[numCells], &adj));
      PetscCall(PetscMalloc1(numCells + 1, &tmp));
      PetscCall(PetscArraycpy(tmp, off, numCells + 1));
      /* Get neighboring cells */
      for (f = fStart; f < fEnd; ++f) {
        const PetscInt *support;
        PetscInt        supportSize;
        PetscCall(DMPlexGetSupportSize(dm, f, &supportSize));
        PetscCall(DMPlexGetSupport(dm, f, &support));
        if (supportSize == 2) {
          PetscInt numChildren;

          PetscCall(DMPlexGetTreeChildren(dm, f, &numChildren, NULL));
          if (!numChildren) {
            adj[tmp[support[0] - cStart]++] = support[1];
            adj[tmp[support[1] - cStart]++] = support[0];
          }
        }
      }
      for (c = 0; c < cEnd - cStart; ++c) PetscAssert(tmp[c] == off[c + 1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offset %" PetscInt_FMT " != %" PetscInt_FMT " for cell %" PetscInt_FMT, tmp[c], off[c], c + cStart);
      PetscCall(PetscFree(tmp));
    }
    if (numVertices) *numVertices = numCells;
    if (offsets) *offsets = off;
    if (adjacency) *adjacency = adj;
    PetscFunctionReturn(0);
  }
  /* Setup face recognition */
  if (faceDepth == 1) {
    PetscInt cornersSeen[30] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; /* Could use PetscBT */

    for (c = cStart; c < cEnd; ++c) {
      PetscInt corners;

      PetscCall(DMPlexGetConeSize(dm, c, &corners));
      if (!cornersSeen[corners]) {
        PetscInt nFV;

        PetscCheck(numFaceCases < maxFaceCases, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Exceeded maximum number of face recognition cases");
        cornersSeen[corners] = 1;

        PetscCall(DMPlexGetNumFaceVertices(dm, cellDim, corners, &nFV));

        numFaceVertices[numFaceCases++] = nFV;
      }
    }
  }
  PetscCall(PetscCalloc1(numCells + 1, &off));
  /* Count neighboring cells */
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt numNeighbors = PETSC_DETERMINE, n;

    PetscCall(DMPlexGetAdjacency_Internal(dm, cell, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &numNeighbors, &neighborCells));
    /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
    for (n = 0; n < numNeighbors; ++n) {
      PetscInt        cellPair[2];
      PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
      PetscInt        meetSize = 0;
      const PetscInt *meet     = NULL;

      cellPair[0] = cell;
      cellPair[1] = neighborCells[n];
      if (cellPair[0] == cellPair[1]) continue;
      if (!found) {
        PetscCall(DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet));
        if (meetSize) {
          PetscInt f;

          for (f = 0; f < numFaceCases; ++f) {
            if (numFaceVertices[f] == meetSize) {
              found = PETSC_TRUE;
              break;
            }
          }
        }
        PetscCall(DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet));
      }
      if (found) ++off[cell - cStart + 1];
    }
  }
  /* Prefix sum */
  for (cell = 1; cell <= numCells; ++cell) off[cell] += off[cell - 1];

  if (adjacency) {
    PetscCall(PetscMalloc1(off[numCells], &adj));
    /* Get neighboring cells */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt numNeighbors = PETSC_DETERMINE, n;
      PetscInt cellOffset   = 0;

      PetscCall(DMPlexGetAdjacency_Internal(dm, cell, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &numNeighbors, &neighborCells));
      /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
      for (n = 0; n < numNeighbors; ++n) {
        PetscInt        cellPair[2];
        PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
        PetscInt        meetSize = 0;
        const PetscInt *meet     = NULL;

        cellPair[0] = cell;
        cellPair[1] = neighborCells[n];
        if (cellPair[0] == cellPair[1]) continue;
        if (!found) {
          PetscCall(DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet));
          if (meetSize) {
            PetscInt f;

            for (f = 0; f < numFaceCases; ++f) {
              if (numFaceVertices[f] == meetSize) {
                found = PETSC_TRUE;
                break;
              }
            }
          }
          PetscCall(DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet));
        }
        if (found) {
          adj[off[cell - cStart] + cellOffset] = neighborCells[n];
          ++cellOffset;
        }
      }
    }
  }
  PetscCall(PetscFree(neighborCells));
  if (numVertices) *numVertices = numCells;
  if (offsets) *offsets = off;
  if (adjacency) *adjacency = adj;
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerDMPlexPartition - Create a non-overlapping partition of the cells in the mesh

  Collective on part

  Input Parameters:
+ part    - The `PetscPartitioner`
. targetSection - The `PetscSection` describing the absolute weight of each partition (can be NULL)
- dm      - The mesh `DM`

  Output Parameters:
+ partSection     - The `PetscSection` giving the division of points by partition
- partition       - The list of points by partition

  Level: developer

  Note:
  If the `DM` has a local section associated, each point to be partitioned will be weighted by the total number of dofs identified
  by the section in the transitive closure of the point.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `PetscPartitioner`, `PetscSection`, `DMPlexDistribute()`, `PetscPartitionerCreate()`, `PetscSectionCreate()`,
         `PetscSectionSetChart()`, `PetscPartitionerPartition()`
@*/
PetscErrorCode PetscPartitionerDMPlexPartition(PetscPartitioner part, DM dm, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscMPIInt  size;
  PetscBool    isplex;
  PetscSection vertSection = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (targetSection) PetscValidHeaderSpecific(targetSection, PETSC_SECTION_CLASSID, 3);
  PetscValidHeaderSpecific(partSection, PETSC_SECTION_CLASSID, 4);
  PetscValidPointer(partition, 5);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isplex));
  PetscCheck(isplex, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not for type %s", ((PetscObject)dm)->type_name);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)part), &size));
  if (size == 1) {
    PetscInt *points;
    PetscInt  cStart, cEnd, c;

    PetscCall(DMPlexGetHeightStratum(dm, part->height, &cStart, &cEnd));
    PetscCall(PetscSectionReset(partSection));
    PetscCall(PetscSectionSetChart(partSection, 0, size));
    PetscCall(PetscSectionSetDof(partSection, 0, cEnd - cStart));
    PetscCall(PetscSectionSetUp(partSection));
    PetscCall(PetscMalloc1(cEnd - cStart, &points));
    for (c = cStart; c < cEnd; ++c) points[c] = c;
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)part), cEnd - cStart, points, PETSC_OWN_POINTER, partition));
    PetscFunctionReturn(0);
  }
  if (part->height == 0) {
    PetscInt  numVertices = 0;
    PetscInt *start       = NULL;
    PetscInt *adjacency   = NULL;
    IS        globalNumbering;

    if (!part->noGraph || part->viewGraph) {
      PetscCall(DMPlexCreatePartitionerGraph(dm, part->height, &numVertices, &start, &adjacency, &globalNumbering));
    } else { /* only compute the number of owned local vertices */
      const PetscInt *idxs;
      PetscInt        p, pStart, pEnd;

      PetscCall(DMPlexGetHeightStratum(dm, part->height, &pStart, &pEnd));
      PetscCall(DMPlexCreateNumbering_Plex(dm, pStart, pEnd, 0, NULL, dm->sf, &globalNumbering));
      PetscCall(ISGetIndices(globalNumbering, &idxs));
      for (p = 0; p < pEnd - pStart; p++) numVertices += idxs[p] < 0 ? 0 : 1;
      PetscCall(ISRestoreIndices(globalNumbering, &idxs));
    }
    if (part->usevwgt) {
      PetscSection    section = dm->localSection, clSection = NULL;
      IS              clPoints = NULL;
      const PetscInt *gid, *clIdx;
      PetscInt        v, p, pStart, pEnd;

      /* dm->localSection encodes degrees of freedom per point, not per cell. We need to get the closure index to properly specify cell weights (aka dofs) */
      /* We do this only if the local section has been set */
      if (section) {
        PetscCall(PetscSectionGetClosureIndex(section, (PetscObject)dm, &clSection, NULL));
        if (!clSection) PetscCall(DMPlexCreateClosureIndex(dm, NULL));
        PetscCall(PetscSectionGetClosureIndex(section, (PetscObject)dm, &clSection, &clPoints));
        PetscCall(ISGetIndices(clPoints, &clIdx));
      }
      PetscCall(DMPlexGetHeightStratum(dm, part->height, &pStart, &pEnd));
      PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &vertSection));
      PetscCall(PetscSectionSetChart(vertSection, 0, numVertices));
      if (globalNumbering) {
        PetscCall(ISGetIndices(globalNumbering, &gid));
      } else gid = NULL;
      for (p = pStart, v = 0; p < pEnd; ++p) {
        PetscInt dof = 1;

        /* skip cells in the overlap */
        if (gid && gid[p - pStart] < 0) continue;

        if (section) {
          PetscInt cl, clSize, clOff;

          dof = 0;
          PetscCall(PetscSectionGetDof(clSection, p, &clSize));
          PetscCall(PetscSectionGetOffset(clSection, p, &clOff));
          for (cl = 0; cl < clSize; cl += 2) {
            PetscInt clDof, clPoint = clIdx[clOff + cl]; /* odd indices are reserved for orientations */

            PetscCall(PetscSectionGetDof(section, clPoint, &clDof));
            dof += clDof;
          }
        }
        PetscCheck(dof, PETSC_COMM_SELF, PETSC_ERR_SUP, "Number of dofs for point %" PetscInt_FMT " in the local section should be positive", p);
        PetscCall(PetscSectionSetDof(vertSection, v, dof));
        v++;
      }
      if (globalNumbering) PetscCall(ISRestoreIndices(globalNumbering, &gid));
      if (clPoints) PetscCall(ISRestoreIndices(clPoints, &clIdx));
      PetscCall(PetscSectionSetUp(vertSection));
    }
    PetscCall(PetscPartitionerPartition(part, size, numVertices, start, adjacency, vertSection, targetSection, partSection, partition));
    PetscCall(PetscFree(start));
    PetscCall(PetscFree(adjacency));
    if (globalNumbering) { /* partition is wrt global unique numbering: change this to be wrt local numbering */
      const PetscInt *globalNum;
      const PetscInt *partIdx;
      PetscInt       *map, cStart, cEnd;
      PetscInt       *adjusted, i, localSize, offset;
      IS              newPartition;

      PetscCall(ISGetLocalSize(*partition, &localSize));
      PetscCall(PetscMalloc1(localSize, &adjusted));
      PetscCall(ISGetIndices(globalNumbering, &globalNum));
      PetscCall(ISGetIndices(*partition, &partIdx));
      PetscCall(PetscMalloc1(localSize, &map));
      PetscCall(DMPlexGetHeightStratum(dm, part->height, &cStart, &cEnd));
      for (i = cStart, offset = 0; i < cEnd; i++) {
        if (globalNum[i - cStart] >= 0) map[offset++] = i;
      }
      for (i = 0; i < localSize; i++) adjusted[i] = map[partIdx[i]];
      PetscCall(PetscFree(map));
      PetscCall(ISRestoreIndices(*partition, &partIdx));
      PetscCall(ISRestoreIndices(globalNumbering, &globalNum));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, localSize, adjusted, PETSC_OWN_POINTER, &newPartition));
      PetscCall(ISDestroy(&globalNumbering));
      PetscCall(ISDestroy(partition));
      *partition = newPartition;
    }
  } else SETERRQ(PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_OUTOFRANGE, "Invalid height %" PetscInt_FMT " for points to partition", part->height);
  PetscCall(PetscSectionDestroy(&vertSection));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetPartitioner - Get the mesh partitioner

  Not collective

  Input Parameter:
. dm - The `DM`

  Output Parameter:
. part - The `PetscPartitioner`

  Level: developer

  Note:
  This gets a borrowed reference, so the user should not destroy this `PetscPartitioner`.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `PetscPartitioner`, `PetscSection`, `DMPlexDistribute()`, `DMPlexSetPartitioner()`, `PetscPartitionerDMPlexPartition()`, `PetscPartitionerCreate()`
@*/
PetscErrorCode DMPlexGetPartitioner(DM dm, PetscPartitioner *part)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(part, 2);
  *part = mesh->partitioner;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetPartitioner - Set the mesh partitioner

  logically collective on dm

  Input Parameters:
+ dm - The `DM`
- part - The partitioner

  Level: developer

  Note:
  Any existing `PetscPartitioner` will be destroyed.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `PetscPartitioner`,`DMPlexDistribute()`, `DMPlexGetPartitioner()`, `PetscPartitionerCreate()`
@*/
PetscErrorCode DMPlexSetPartitioner(DM dm, PetscPartitioner part)
{
  DM_Plex *mesh = (DM_Plex *)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)part));
  PetscCall(PetscPartitionerDestroy(&mesh->partitioner));
  mesh->partitioner = part;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosure_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  const PetscInt *cone;
  PetscInt        coneSize, c;
  PetscBool       missing;

  PetscFunctionBeginHot;
  PetscCall(PetscHSetIQueryAdd(ht, point, &missing));
  if (missing) {
    PetscCall(DMPlexGetCone(dm, point, &cone));
    PetscCall(DMPlexGetConeSize(dm, point, &coneSize));
    for (c = 0; c < coneSize; c++) PetscCall(DMPlexAddClosure_Private(dm, ht, cone[c]));
  }
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode DMPlexAddClosure_Tree(DM dm, PetscHSetI ht, PetscInt point, PetscBool up, PetscBool down)
{
  PetscFunctionBegin;
  if (up) {
    PetscInt parent;

    PetscCall(DMPlexGetTreeParent(dm, point, &parent, NULL));
    if (parent != point) {
      PetscInt closureSize, *closure = NULL, i;

      PetscCall(DMPlexGetTransitiveClosure(dm, parent, PETSC_TRUE, &closureSize, &closure));
      for (i = 0; i < closureSize; i++) {
        PetscInt cpoint = closure[2 * i];

        PetscCall(PetscHSetIAdd(ht, cpoint));
        PetscCall(DMPlexAddClosure_Tree(dm, ht, cpoint, PETSC_TRUE, PETSC_FALSE));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, parent, PETSC_TRUE, &closureSize, &closure));
    }
  }
  if (down) {
    PetscInt        numChildren;
    const PetscInt *children;

    PetscCall(DMPlexGetTreeChildren(dm, point, &numChildren, &children));
    if (numChildren) {
      PetscInt i;

      for (i = 0; i < numChildren; i++) {
        PetscInt cpoint = children[i];

        PetscCall(PetscHSetIAdd(ht, cpoint));
        PetscCall(DMPlexAddClosure_Tree(dm, ht, cpoint, PETSC_FALSE, PETSC_TRUE));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosureTree_Up_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  PetscInt parent;

  PetscFunctionBeginHot;
  PetscCall(DMPlexGetTreeParent(dm, point, &parent, NULL));
  if (point != parent) {
    const PetscInt *cone;
    PetscInt        coneSize, c;

    PetscCall(DMPlexAddClosureTree_Up_Private(dm, ht, parent));
    PetscCall(DMPlexAddClosure_Private(dm, ht, parent));
    PetscCall(DMPlexGetCone(dm, parent, &cone));
    PetscCall(DMPlexGetConeSize(dm, parent, &coneSize));
    for (c = 0; c < coneSize; c++) {
      const PetscInt cp = cone[c];

      PetscCall(DMPlexAddClosureTree_Up_Private(dm, ht, cp));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosureTree_Down_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  PetscInt        i, numChildren;
  const PetscInt *children;

  PetscFunctionBeginHot;
  PetscCall(DMPlexGetTreeChildren(dm, point, &numChildren, &children));
  for (i = 0; i < numChildren; i++) PetscCall(PetscHSetIAdd(ht, children[i]));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosureTree_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  const PetscInt *cone;
  PetscInt        coneSize, c;

  PetscFunctionBeginHot;
  PetscCall(PetscHSetIAdd(ht, point));
  PetscCall(DMPlexAddClosureTree_Up_Private(dm, ht, point));
  PetscCall(DMPlexAddClosureTree_Down_Private(dm, ht, point));
  PetscCall(DMPlexGetCone(dm, point, &cone));
  PetscCall(DMPlexGetConeSize(dm, point, &coneSize));
  for (c = 0; c < coneSize; c++) PetscCall(DMPlexAddClosureTree_Private(dm, ht, cone[c]));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexClosurePoints_Private(DM dm, PetscInt numPoints, const PetscInt points[], IS *closureIS)
{
  DM_Plex        *mesh    = (DM_Plex *)dm->data;
  const PetscBool hasTree = (mesh->parentSection || mesh->childSection) ? PETSC_TRUE : PETSC_FALSE;
  PetscInt        nelems, *elems, off = 0, p;
  PetscHSetI      ht = NULL;

  PetscFunctionBegin;
  PetscCall(PetscHSetICreate(&ht));
  PetscCall(PetscHSetIResize(ht, numPoints * 16));
  if (!hasTree) {
    for (p = 0; p < numPoints; ++p) PetscCall(DMPlexAddClosure_Private(dm, ht, points[p]));
  } else {
#if 1
    for (p = 0; p < numPoints; ++p) PetscCall(DMPlexAddClosureTree_Private(dm, ht, points[p]));
#else
    PetscInt *closure = NULL, closureSize, c;
    for (p = 0; p < numPoints; ++p) {
      PetscCall(DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure));
      for (c = 0; c < closureSize * 2; c += 2) {
        PetscCall(PetscHSetIAdd(ht, closure[c]));
        if (hasTree) PetscCall(DMPlexAddClosure_Tree(dm, ht, closure[c], PETSC_TRUE, PETSC_TRUE));
      }
    }
    if (closure) PetscCall(DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, NULL, &closure));
#endif
  }
  PetscCall(PetscHSetIGetSize(ht, &nelems));
  PetscCall(PetscMalloc1(nelems, &elems));
  PetscCall(PetscHSetIGetElems(ht, &off, elems));
  PetscCall(PetscHSetIDestroy(&ht));
  PetscCall(PetscSortInt(nelems, elems));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nelems, elems, PETSC_OWN_POINTER, closureIS));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelClosure - Add the closure of all points to the partition label

  Input Parameters:
+ dm     - The `DM`
- label  - `DMLabel` assigning ranks to remote roots

  Level: developer

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMLabel`, `DMPlexPartitionLabelCreateSF()`, `DMPlexDistribute()`, `DMPlexCreateOverlap()`
@*/
PetscErrorCode DMPlexPartitionLabelClosure(DM dm, DMLabel label)
{
  IS              rankIS, pointIS, closureIS;
  const PetscInt *ranks, *points;
  PetscInt        numRanks, numPoints, r;

  PetscFunctionBegin;
  PetscCall(DMLabelGetValueIS(label, &rankIS));
  PetscCall(ISGetLocalSize(rankIS, &numRanks));
  PetscCall(ISGetIndices(rankIS, &ranks));
  for (r = 0; r < numRanks; ++r) {
    const PetscInt rank = ranks[r];
    PetscCall(DMLabelGetStratumIS(label, rank, &pointIS));
    PetscCall(ISGetLocalSize(pointIS, &numPoints));
    PetscCall(ISGetIndices(pointIS, &points));
    PetscCall(DMPlexClosurePoints_Private(dm, numPoints, points, &closureIS));
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
    PetscCall(DMLabelSetStratumIS(label, rank, closureIS));
    PetscCall(ISDestroy(&closureIS));
  }
  PetscCall(ISRestoreIndices(rankIS, &ranks));
  PetscCall(ISDestroy(&rankIS));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelAdjacency - Add one level of adjacent points to the partition label

  Input Parameters:
+ dm     - The `DM`
- label  - `DMLabel` assigning ranks to remote roots

  Level: developer

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMLabel`, `DMPlexPartitionLabelCreateSF()`, `DMPlexDistribute()`, `DMPlexCreateOverlap()`
@*/
PetscErrorCode DMPlexPartitionLabelAdjacency(DM dm, DMLabel label)
{
  IS              rankIS, pointIS;
  const PetscInt *ranks, *points;
  PetscInt        numRanks, numPoints, r, p, a, adjSize;
  PetscInt       *adj = NULL;

  PetscFunctionBegin;
  PetscCall(DMLabelGetValueIS(label, &rankIS));
  PetscCall(ISGetLocalSize(rankIS, &numRanks));
  PetscCall(ISGetIndices(rankIS, &ranks));
  for (r = 0; r < numRanks; ++r) {
    const PetscInt rank = ranks[r];

    PetscCall(DMLabelGetStratumIS(label, rank, &pointIS));
    PetscCall(ISGetLocalSize(pointIS, &numPoints));
    PetscCall(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      adjSize = PETSC_DETERMINE;
      PetscCall(DMPlexGetAdjacency(dm, points[p], &adjSize, &adj));
      for (a = 0; a < adjSize; ++a) PetscCall(DMLabelSetValue(label, adj[a], rank));
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(rankIS, &ranks));
  PetscCall(ISDestroy(&rankIS));
  PetscCall(PetscFree(adj));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelPropagate - Propagate points in a partition label over the point `PetscSF`

  Input Parameters:
+ dm     - The `DM`
- label  - `DMLabel` assigning ranks to remote roots

  Level: developer

  Note:
  This is required when generating multi-level overlaps to capture
  overlap points from non-neighbouring partitions.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMLabel`, `DMPlexPartitionLabelCreateSF()`, `DMPlexDistribute()`, `DMPlexCreateOverlap()`
@*/
PetscErrorCode DMPlexPartitionLabelPropagate(DM dm, DMLabel label)
{
  MPI_Comm        comm;
  PetscMPIInt     rank;
  PetscSF         sfPoint;
  DMLabel         lblRoots, lblLeaves;
  IS              rankIS, pointIS;
  const PetscInt *ranks;
  PetscInt        numRanks, r;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetPointSF(dm, &sfPoint));
  /* Pull point contributions from remote leaves into local roots */
  PetscCall(DMLabelGather(label, sfPoint, &lblLeaves));
  PetscCall(DMLabelGetValueIS(lblLeaves, &rankIS));
  PetscCall(ISGetLocalSize(rankIS, &numRanks));
  PetscCall(ISGetIndices(rankIS, &ranks));
  for (r = 0; r < numRanks; ++r) {
    const PetscInt remoteRank = ranks[r];
    if (remoteRank == rank) continue;
    PetscCall(DMLabelGetStratumIS(lblLeaves, remoteRank, &pointIS));
    PetscCall(DMLabelInsertIS(label, pointIS, remoteRank));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(rankIS, &ranks));
  PetscCall(ISDestroy(&rankIS));
  PetscCall(DMLabelDestroy(&lblLeaves));
  /* Push point contributions from roots into remote leaves */
  PetscCall(DMLabelDistribute(label, sfPoint, &lblRoots));
  PetscCall(DMLabelGetValueIS(lblRoots, &rankIS));
  PetscCall(ISGetLocalSize(rankIS, &numRanks));
  PetscCall(ISGetIndices(rankIS, &ranks));
  for (r = 0; r < numRanks; ++r) {
    const PetscInt remoteRank = ranks[r];
    if (remoteRank == rank) continue;
    PetscCall(DMLabelGetStratumIS(lblRoots, remoteRank, &pointIS));
    PetscCall(DMLabelInsertIS(label, pointIS, remoteRank));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(rankIS, &ranks));
  PetscCall(ISDestroy(&rankIS));
  PetscCall(DMLabelDestroy(&lblRoots));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelInvert - Create a partition label of remote roots from a local root label

  Input Parameters:
+ dm        - The `DM`
. rootLabel - `DMLabel` assigning ranks to local roots
- processSF - A star forest mapping into the local index on each remote rank

  Output Parameter:
. leafLabel - `DMLabel `assigning ranks to remote roots

  Level: developer

  Note:
  The rootLabel defines a send pattern by mapping local points to remote target ranks. The
  resulting leafLabel is a receiver mapping of remote roots to their parent rank.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexPartitionLabelCreateSF()`, `DMPlexDistribute()`, `DMPlexCreateOverlap()`
@*/
PetscErrorCode DMPlexPartitionLabelInvert(DM dm, DMLabel rootLabel, PetscSF processSF, DMLabel leafLabel)
{
  MPI_Comm           comm;
  PetscMPIInt        rank, size, r;
  PetscInt           p, n, numNeighbors, numPoints, dof, off, rootSize, l, nleaves, leafSize;
  PetscSF            sfPoint;
  PetscSection       rootSection;
  PetscSFNode       *rootPoints, *leafPoints;
  const PetscSFNode *remote;
  const PetscInt    *local, *neighbors;
  IS                 valueIS;
  PetscBool          mpiOverflow = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_PartLabelInvert, dm, 0, 0, 0));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetPointSF(dm, &sfPoint));

  /* Convert to (point, rank) and use actual owners */
  PetscCall(PetscSectionCreate(comm, &rootSection));
  PetscCall(PetscSectionSetChart(rootSection, 0, size));
  PetscCall(DMLabelGetValueIS(rootLabel, &valueIS));
  PetscCall(ISGetLocalSize(valueIS, &numNeighbors));
  PetscCall(ISGetIndices(valueIS, &neighbors));
  for (n = 0; n < numNeighbors; ++n) {
    PetscCall(DMLabelGetStratumSize(rootLabel, neighbors[n], &numPoints));
    PetscCall(PetscSectionAddDof(rootSection, neighbors[n], numPoints));
  }
  PetscCall(PetscSectionSetUp(rootSection));
  PetscCall(PetscSectionGetStorageSize(rootSection, &rootSize));
  PetscCall(PetscMalloc1(rootSize, &rootPoints));
  PetscCall(PetscSFGetGraph(sfPoint, NULL, &nleaves, &local, &remote));
  for (n = 0; n < numNeighbors; ++n) {
    IS              pointIS;
    const PetscInt *points;

    PetscCall(PetscSectionGetOffset(rootSection, neighbors[n], &off));
    PetscCall(DMLabelGetStratumIS(rootLabel, neighbors[n], &pointIS));
    PetscCall(ISGetLocalSize(pointIS, &numPoints));
    PetscCall(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      if (local) PetscCall(PetscFindInt(points[p], nleaves, local, &l));
      else l = -1;
      if (l >= 0) {
        rootPoints[off + p] = remote[l];
      } else {
        rootPoints[off + p].index = points[p];
        rootPoints[off + p].rank  = rank;
      }
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }

  /* Try to communicate overlap using All-to-All */
  if (!processSF) {
    PetscInt64   counter     = 0;
    PetscBool    locOverflow = PETSC_FALSE;
    PetscMPIInt *scounts, *sdispls, *rcounts, *rdispls;

    PetscCall(PetscCalloc4(size, &scounts, size, &sdispls, size, &rcounts, size, &rdispls));
    for (n = 0; n < numNeighbors; ++n) {
      PetscCall(PetscSectionGetDof(rootSection, neighbors[n], &dof));
      PetscCall(PetscSectionGetOffset(rootSection, neighbors[n], &off));
#if defined(PETSC_USE_64BIT_INDICES)
      if (dof > PETSC_MPI_INT_MAX) {
        locOverflow = PETSC_TRUE;
        break;
      }
      if (off > PETSC_MPI_INT_MAX) {
        locOverflow = PETSC_TRUE;
        break;
      }
#endif
      scounts[neighbors[n]] = (PetscMPIInt)dof;
      sdispls[neighbors[n]] = (PetscMPIInt)off;
    }
    PetscCallMPI(MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, comm));
    for (r = 0; r < size; ++r) {
      rdispls[r] = (int)counter;
      counter += rcounts[r];
    }
    if (counter > PETSC_MPI_INT_MAX) locOverflow = PETSC_TRUE;
    PetscCallMPI(MPI_Allreduce(&locOverflow, &mpiOverflow, 1, MPIU_BOOL, MPI_LOR, comm));
    if (!mpiOverflow) {
      PetscCall(PetscInfo(dm, "Using Alltoallv for mesh distribution\n"));
      leafSize = (PetscInt)counter;
      PetscCall(PetscMalloc1(leafSize, &leafPoints));
      PetscCallMPI(MPI_Alltoallv(rootPoints, scounts, sdispls, MPIU_2INT, leafPoints, rcounts, rdispls, MPIU_2INT, comm));
    }
    PetscCall(PetscFree4(scounts, sdispls, rcounts, rdispls));
  }

  /* Communicate overlap using process star forest */
  if (processSF || mpiOverflow) {
    PetscSF      procSF;
    PetscSection leafSection;

    if (processSF) {
      PetscCall(PetscInfo(dm, "Using processSF for mesh distribution\n"));
      PetscCall(PetscObjectReference((PetscObject)processSF));
      procSF = processSF;
    } else {
      PetscCall(PetscInfo(dm, "Using processSF for mesh distribution (MPI overflow)\n"));
      PetscCall(PetscSFCreate(comm, &procSF));
      PetscCall(PetscSFSetGraphWithPattern(procSF, NULL, PETSCSF_PATTERN_ALLTOALL));
    }

    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &leafSection));
    PetscCall(DMPlexDistributeData(dm, procSF, rootSection, MPIU_2INT, rootPoints, leafSection, (void **)&leafPoints));
    PetscCall(PetscSectionGetStorageSize(leafSection, &leafSize));
    PetscCall(PetscSectionDestroy(&leafSection));
    PetscCall(PetscSFDestroy(&procSF));
  }

  for (p = 0; p < leafSize; p++) PetscCall(DMLabelSetValue(leafLabel, leafPoints[p].index, leafPoints[p].rank));

  PetscCall(ISRestoreIndices(valueIS, &neighbors));
  PetscCall(ISDestroy(&valueIS));
  PetscCall(PetscSectionDestroy(&rootSection));
  PetscCall(PetscFree(rootPoints));
  PetscCall(PetscFree(leafPoints));
  PetscCall(PetscLogEventEnd(DMPLEX_PartLabelInvert, dm, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelCreateSF - Create a star forest from a label that assigns ranks to points

  Input Parameters:
+ dm    - The `DM`
- label - `DMLabel` assigning ranks to remote roots

  Output Parameter:
. sf    - The star forest communication context encapsulating the defined mapping

  Level: developer

  Note:
  The incoming label is a receiver mapping of remote points to their parent rank.

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMLabel`, `PetscSF`, `DMPlexDistribute()`, `DMPlexCreateOverlap()`
@*/
PetscErrorCode DMPlexPartitionLabelCreateSF(DM dm, DMLabel label, PetscSF *sf)
{
  PetscMPIInt     rank;
  PetscInt        n, numRemote, p, numPoints, pStart, pEnd, idx = 0, nNeighbors;
  PetscSFNode    *remotePoints;
  IS              remoteRootIS, neighborsIS;
  const PetscInt *remoteRoots, *neighbors;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DMPLEX_PartLabelCreateSF, dm, 0, 0, 0));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));

  PetscCall(DMLabelGetValueIS(label, &neighborsIS));
#if 0
  {
    IS is;
    PetscCall(ISDuplicate(neighborsIS, &is));
    PetscCall(ISSort(is));
    PetscCall(ISDestroy(&neighborsIS));
    neighborsIS = is;
  }
#endif
  PetscCall(ISGetLocalSize(neighborsIS, &nNeighbors));
  PetscCall(ISGetIndices(neighborsIS, &neighbors));
  for (numRemote = 0, n = 0; n < nNeighbors; ++n) {
    PetscCall(DMLabelGetStratumSize(label, neighbors[n], &numPoints));
    numRemote += numPoints;
  }
  PetscCall(PetscMalloc1(numRemote, &remotePoints));
  /* Put owned points first */
  PetscCall(DMLabelGetStratumSize(label, rank, &numPoints));
  if (numPoints > 0) {
    PetscCall(DMLabelGetStratumIS(label, rank, &remoteRootIS));
    PetscCall(ISGetIndices(remoteRootIS, &remoteRoots));
    for (p = 0; p < numPoints; p++) {
      remotePoints[idx].index = remoteRoots[p];
      remotePoints[idx].rank  = rank;
      idx++;
    }
    PetscCall(ISRestoreIndices(remoteRootIS, &remoteRoots));
    PetscCall(ISDestroy(&remoteRootIS));
  }
  /* Now add remote points */
  for (n = 0; n < nNeighbors; ++n) {
    const PetscInt nn = neighbors[n];

    PetscCall(DMLabelGetStratumSize(label, nn, &numPoints));
    if (nn == rank || numPoints <= 0) continue;
    PetscCall(DMLabelGetStratumIS(label, nn, &remoteRootIS));
    PetscCall(ISGetIndices(remoteRootIS, &remoteRoots));
    for (p = 0; p < numPoints; p++) {
      remotePoints[idx].index = remoteRoots[p];
      remotePoints[idx].rank  = nn;
      idx++;
    }
    PetscCall(ISRestoreIndices(remoteRootIS, &remoteRoots));
    PetscCall(ISDestroy(&remoteRootIS));
  }
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm), sf));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSFSetGraph(*sf, pEnd - pStart, numRemote, NULL, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(*sf));
  PetscCall(ISDestroy(&neighborsIS));
  PetscCall(PetscLogEventEnd(DMPLEX_PartLabelCreateSF, dm, 0, 0, 0));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PARMETIS)
  #include <parmetis.h>
#endif

/* The two functions below are used by DMPlexRebalanceSharedPoints which errors
 * when PETSc is built without ParMETIS. To avoid -Wunused-function, we take
 * them out in that case. */
#if defined(PETSC_HAVE_PARMETIS)
/*@C

  DMPlexRewriteSF - Rewrites the ownership of the `PetsSF` of a `DM` (in place).

  Input parameters:
+ dm                - The `DMPLEX` object.
. n                 - The number of points.
. pointsToRewrite   - The points in the `PetscSF` whose ownership will change.
. targetOwners      - New owner for each element in pointsToRewrite.
- degrees           - Degrees of the points in the `PetscSF` as obtained by `PetscSFComputeDegreeBegin()`/`PetscSFComputeDegreeEnd()`.

  Level: developer

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMLabel`, `PetscSF`, `DMPlexDistribute()`, `DMPlexCreateOverlap()`
@*/
static PetscErrorCode DMPlexRewriteSF(DM dm, PetscInt n, PetscInt *pointsToRewrite, PetscInt *targetOwners, const PetscInt *degrees)
{
  PetscInt           pStart, pEnd, i, j, counter, leafCounter, sumDegrees, nroots, nleafs;
  PetscInt          *cumSumDegrees, *newOwners, *newNumbers, *rankOnLeafs, *locationsOfLeafs, *remoteLocalPointOfLeafs, *points, *leafsNew;
  PetscSFNode       *leafLocationsNew;
  const PetscSFNode *iremote;
  const PetscInt    *ilocal;
  PetscBool         *isLeaf;
  PetscSF            sf;
  MPI_Comm           comm;
  PetscMPIInt        rank, size;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));

  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleafs, &ilocal, &iremote));
  PetscCall(PetscMalloc1(pEnd - pStart, &isLeaf));
  for (i = 0; i < pEnd - pStart; i++) isLeaf[i] = PETSC_FALSE;
  for (i = 0; i < nleafs; i++) isLeaf[ilocal[i] - pStart] = PETSC_TRUE;

  PetscCall(PetscMalloc1(pEnd - pStart + 1, &cumSumDegrees));
  cumSumDegrees[0] = 0;
  for (i = 1; i <= pEnd - pStart; i++) cumSumDegrees[i] = cumSumDegrees[i - 1] + degrees[i - 1];
  sumDegrees = cumSumDegrees[pEnd - pStart];
  /* get the location of my leafs (we have sumDegrees many leafs pointing at our roots) */

  PetscCall(PetscMalloc1(sumDegrees, &locationsOfLeafs));
  PetscCall(PetscMalloc1(pEnd - pStart, &rankOnLeafs));
  for (i = 0; i < pEnd - pStart; i++) rankOnLeafs[i] = rank;
  PetscCall(PetscSFGatherBegin(sf, MPIU_INT, rankOnLeafs, locationsOfLeafs));
  PetscCall(PetscSFGatherEnd(sf, MPIU_INT, rankOnLeafs, locationsOfLeafs));
  PetscCall(PetscFree(rankOnLeafs));

  /* get the remote local points of my leaves */
  PetscCall(PetscMalloc1(sumDegrees, &remoteLocalPointOfLeafs));
  PetscCall(PetscMalloc1(pEnd - pStart, &points));
  for (i = 0; i < pEnd - pStart; i++) points[i] = pStart + i;
  PetscCall(PetscSFGatherBegin(sf, MPIU_INT, points, remoteLocalPointOfLeafs));
  PetscCall(PetscSFGatherEnd(sf, MPIU_INT, points, remoteLocalPointOfLeafs));
  PetscCall(PetscFree(points));
  /* Figure out the new owners of the vertices that are up for grabs and their numbers on the new owners */
  PetscCall(PetscMalloc1(pEnd - pStart, &newOwners));
  PetscCall(PetscMalloc1(pEnd - pStart, &newNumbers));
  for (i = 0; i < pEnd - pStart; i++) {
    newOwners[i]  = -1;
    newNumbers[i] = -1;
  }
  {
    PetscInt oldNumber, newNumber, oldOwner, newOwner;
    for (i = 0; i < n; i++) {
      oldNumber = pointsToRewrite[i];
      newNumber = -1;
      oldOwner  = rank;
      newOwner  = targetOwners[i];
      if (oldOwner == newOwner) {
        newNumber = oldNumber;
      } else {
        for (j = 0; j < degrees[oldNumber]; j++) {
          if (locationsOfLeafs[cumSumDegrees[oldNumber] + j] == newOwner) {
            newNumber = remoteLocalPointOfLeafs[cumSumDegrees[oldNumber] + j];
            break;
          }
        }
      }
      PetscCheck(newNumber != -1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Couldn't find the new owner of vertex.");

      newOwners[oldNumber]  = newOwner;
      newNumbers[oldNumber] = newNumber;
    }
  }
  PetscCall(PetscFree(cumSumDegrees));
  PetscCall(PetscFree(locationsOfLeafs));
  PetscCall(PetscFree(remoteLocalPointOfLeafs));

  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, newOwners, newOwners, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, newOwners, newOwners, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, newNumbers, newNumbers, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, newNumbers, newNumbers, MPI_REPLACE));

  /* Now count how many leafs we have on each processor. */
  leafCounter = 0;
  for (i = 0; i < pEnd - pStart; i++) {
    if (newOwners[i] >= 0) {
      if (newOwners[i] != rank) leafCounter++;
    } else {
      if (isLeaf[i]) leafCounter++;
    }
  }

  /* Now set up the new sf by creating the leaf arrays */
  PetscCall(PetscMalloc1(leafCounter, &leafsNew));
  PetscCall(PetscMalloc1(leafCounter, &leafLocationsNew));

  leafCounter = 0;
  counter     = 0;
  for (i = 0; i < pEnd - pStart; i++) {
    if (newOwners[i] >= 0) {
      if (newOwners[i] != rank) {
        leafsNew[leafCounter]               = i;
        leafLocationsNew[leafCounter].rank  = newOwners[i];
        leafLocationsNew[leafCounter].index = newNumbers[i];
        leafCounter++;
      }
    } else {
      if (isLeaf[i]) {
        leafsNew[leafCounter]               = i;
        leafLocationsNew[leafCounter].rank  = iremote[counter].rank;
        leafLocationsNew[leafCounter].index = iremote[counter].index;
        leafCounter++;
      }
    }
    if (isLeaf[i]) counter++;
  }

  PetscCall(PetscSFSetGraph(sf, nroots, leafCounter, leafsNew, PETSC_OWN_POINTER, leafLocationsNew, PETSC_OWN_POINTER));
  PetscCall(PetscFree(newOwners));
  PetscCall(PetscFree(newNumbers));
  PetscCall(PetscFree(isLeaf));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexViewDistribution(MPI_Comm comm, PetscInt n, PetscInt skip, PetscInt *vtxwgt, PetscInt *part, PetscViewer viewer)
{
  PetscInt   *distribution, min, max, sum;
  PetscMPIInt rank, size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscCalloc1(size, &distribution));
  for (PetscInt i = 0; i < n; i++) {
    if (part) distribution[part[i]] += vtxwgt[skip * i];
    else distribution[rank] += vtxwgt[skip * i];
  }
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, distribution, size, MPIU_INT, MPI_SUM, comm));
  min = distribution[0];
  max = distribution[0];
  sum = distribution[0];
  for (PetscInt i = 1; i < size; i++) {
    if (distribution[i] < min) min = distribution[i];
    if (distribution[i] > max) max = distribution[i];
    sum += distribution[i];
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "Min: %" PetscInt_FMT ", Avg: %" PetscInt_FMT ", Max: %" PetscInt_FMT ", Balance: %f\n", min, sum / size, max, (max * 1. * size) / sum));
  PetscCall(PetscFree(distribution));
  PetscFunctionReturn(0);
}

#endif

/*@
  DMPlexRebalanceSharedPoints - Redistribute points in the plex that are shared in order to achieve better balancing. This routine updates the `PointSF` of the `DM` inplace.

  Input parameters:
+ dm               - The `DMPLEX` object.
. entityDepth      - depth of the entity to balance (0 -> balance vertices).
. useInitialGuess  - whether to use the current distribution as initial guess (only used by ParMETIS).
- parallel         - whether to use ParMETIS and do the partition in parallel or whether to gather the graph onto a single process and use METIS.

  Output parameters:
. success          - whether the graph partitioning was successful or not, optional. Unsuccessful simply means no change to the partitioning

  Options Database Keys:
+  -dm_plex_rebalance_shared_points_parmetis - Use ParMetis instead of Metis for the partitioner
.  -dm_plex_rebalance_shared_points_use_initial_guess - Use current partition to bootstrap ParMetis partition
.  -dm_plex_rebalance_shared_points_use_mat_partitioning - Use the MatPartitioning object to perform the partition, the prefix for those operations is -dm_plex_rebalance_shared_points_
-  -dm_plex_rebalance_shared_points_monitor - Monitor the shared points rebalance process

  Level: intermediate

.seealso: [](chapter_unstructured), `DM`, `DMPLEX`, `DMPlexDistribute()`, `DMPlexCreateOverlap()`
@*/
PetscErrorCode DMPlexRebalanceSharedPoints(DM dm, PetscInt entityDepth, PetscBool useInitialGuess, PetscBool parallel, PetscBool *success)
{
#if defined(PETSC_HAVE_PARMETIS)
  PetscSF            sf;
  PetscInt           ierr, i, j, idx, jdx;
  PetscInt           eBegin, eEnd, nroots, nleafs, pStart, pEnd;
  const PetscInt    *degrees, *ilocal;
  const PetscSFNode *iremote;
  PetscBool         *toBalance, *isLeaf, *isExclusivelyOwned, *isNonExclusivelyOwned;
  PetscInt           numExclusivelyOwned, numNonExclusivelyOwned;
  PetscMPIInt        rank, size;
  PetscInt          *globalNumbersOfLocalOwnedVertices, *leafGlobalNumbers;
  const PetscInt    *cumSumVertices;
  PetscInt           offset, counter;
  PetscInt          *vtxwgt;
  const PetscInt    *xadj, *adjncy;
  PetscInt          *part, *options;
  PetscInt           nparts, wgtflag, numflag, ncon, edgecut;
  real_t            *ubvec;
  PetscInt          *firstVertices, *renumbering;
  PetscInt           failed, failedGlobal;
  MPI_Comm           comm;
  Mat                A;
  PetscViewer        viewer;
  PetscViewerFormat  format;
  PetscLayout        layout;
  real_t            *tpwgts;
  PetscMPIInt       *counts, *mpiCumSumVertices;
  PetscInt          *pointsToRewrite;
  PetscInt           numRows;
  PetscBool          done, usematpartitioning = PETSC_FALSE;
  IS                 ispart = NULL;
  MatPartitioning    mp;
  const char        *prefix;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    if (success) *success = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  if (success) *success = PETSC_FALSE;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  parallel        = PETSC_FALSE;
  useInitialGuess = PETSC_FALSE;
  PetscObjectOptionsBegin((PetscObject)dm);
  PetscCall(PetscOptionsName("-dm_plex_rebalance_shared_points_parmetis", "Use ParMetis instead of Metis for the partitioner", "DMPlexRebalanceSharedPoints", &parallel));
  PetscCall(PetscOptionsBool("-dm_plex_rebalance_shared_points_use_initial_guess", "Use current partition to bootstrap ParMetis partition", "DMPlexRebalanceSharedPoints", useInitialGuess, &useInitialGuess, NULL));
  PetscCall(PetscOptionsBool("-dm_plex_rebalance_shared_points_use_mat_partitioning", "Use the MatPartitioning object to partition", "DMPlexRebalanceSharedPoints", usematpartitioning, &usematpartitioning, NULL));
  PetscCall(PetscOptionsViewer("-dm_plex_rebalance_shared_points_monitor", "Monitor the shared points rebalance process", "DMPlexRebalanceSharedPoints", &viewer, &format, NULL));
  PetscOptionsEnd();
  if (viewer) PetscCall(PetscViewerPushFormat(viewer, format));

  PetscCall(PetscLogEventBegin(DMPLEX_RebalanceSharedPoints, dm, 0, 0, 0));

  PetscCall(DMGetOptionsPrefix(dm, &prefix));
  PetscCall(PetscOptionsGetViewer(comm, ((PetscObject)dm)->options, prefix, "-dm_rebalance_partition_view", &viewer, &format, NULL));
  if (viewer) PetscCall(PetscViewerPushFormat(viewer, format));

  /* Figure out all points in the plex that we are interested in balancing. */
  PetscCall(DMPlexGetDepthStratum(dm, entityDepth, &eBegin, &eEnd));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscMalloc1(pEnd - pStart, &toBalance));
  for (i = 0; i < pEnd - pStart; i++) toBalance[i] = (PetscBool)(i >= eBegin && i < eEnd);

  /* There are three types of points:
   * exclusivelyOwned: points that are owned by this process and only seen by this process
   * nonExclusivelyOwned: points that are owned by this process but seen by at least another process
   * leaf: a point that is seen by this process but owned by a different process
   */
  PetscCall(DMGetPointSF(dm, &sf));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleafs, &ilocal, &iremote));
  PetscCall(PetscMalloc1(pEnd - pStart, &isLeaf));
  PetscCall(PetscMalloc1(pEnd - pStart, &isNonExclusivelyOwned));
  PetscCall(PetscMalloc1(pEnd - pStart, &isExclusivelyOwned));
  for (i = 0; i < pEnd - pStart; i++) {
    isNonExclusivelyOwned[i] = PETSC_FALSE;
    isExclusivelyOwned[i]    = PETSC_FALSE;
    isLeaf[i]                = PETSC_FALSE;
  }

  /* mark all the leafs */
  for (i = 0; i < nleafs; i++) isLeaf[ilocal[i] - pStart] = PETSC_TRUE;

  /* for an owned point, we can figure out whether another processor sees it or
   * not by calculating its degree */
  PetscCall(PetscSFComputeDegreeBegin(sf, &degrees));
  PetscCall(PetscSFComputeDegreeEnd(sf, &degrees));
  numExclusivelyOwned    = 0;
  numNonExclusivelyOwned = 0;
  for (i = 0; i < pEnd - pStart; i++) {
    if (toBalance[i]) {
      if (degrees[i] > 0) {
        isNonExclusivelyOwned[i] = PETSC_TRUE;
        numNonExclusivelyOwned += 1;
      } else {
        if (!isLeaf[i]) {
          isExclusivelyOwned[i] = PETSC_TRUE;
          numExclusivelyOwned += 1;
        }
      }
    }
  }

  /* Build a graph with one vertex per core representing the
   * exclusively owned points and then one vertex per nonExclusively owned
   * point. */
  PetscCall(PetscLayoutCreate(comm, &layout));
  PetscCall(PetscLayoutSetLocalSize(layout, 1 + numNonExclusivelyOwned));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetRanges(layout, &cumSumVertices));
  PetscCall(PetscMalloc1(pEnd - pStart, &globalNumbersOfLocalOwnedVertices));
  for (i = 0; i < pEnd - pStart; i++) globalNumbersOfLocalOwnedVertices[i] = pStart - 1;
  offset  = cumSumVertices[rank];
  counter = 0;
  for (i = 0; i < pEnd - pStart; i++) {
    if (toBalance[i]) {
      if (degrees[i] > 0) {
        globalNumbersOfLocalOwnedVertices[i] = counter + 1 + offset;
        counter++;
      }
    }
  }

  /* send the global numbers of vertices I own to the leafs so that they know to connect to it */
  PetscCall(PetscMalloc1(pEnd - pStart, &leafGlobalNumbers));
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, globalNumbersOfLocalOwnedVertices, leafGlobalNumbers, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, globalNumbersOfLocalOwnedVertices, leafGlobalNumbers, MPI_REPLACE));

  /* Build the graph for partitioning */
  numRows = 1 + numNonExclusivelyOwned;
  PetscCall(PetscLogEventBegin(DMPLEX_RebalBuildGraph, dm, 0, 0, 0));
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetType(A, MATMPIADJ));
  PetscCall(MatSetSizes(A, numRows, numRows, cumSumVertices[size], cumSumVertices[size]));
  idx = cumSumVertices[rank];
  for (i = 0; i < pEnd - pStart; i++) {
    if (toBalance[i]) {
      if (isNonExclusivelyOwned[i]) jdx = globalNumbersOfLocalOwnedVertices[i];
      else if (isLeaf[i]) jdx = leafGlobalNumbers[i];
      else continue;
      PetscCall(MatSetValue(A, idx, jdx, 1, INSERT_VALUES));
      PetscCall(MatSetValue(A, jdx, idx, 1, INSERT_VALUES));
    }
  }
  PetscCall(PetscFree(globalNumbersOfLocalOwnedVertices));
  PetscCall(PetscFree(leafGlobalNumbers));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogEventEnd(DMPLEX_RebalBuildGraph, dm, 0, 0, 0));

  nparts = size;
  ncon   = 1;
  PetscCall(PetscMalloc1(ncon * nparts, &tpwgts));
  for (i = 0; i < ncon * nparts; i++) tpwgts[i] = 1. / (nparts);
  PetscCall(PetscMalloc1(ncon, &ubvec));
  ubvec[0] = 1.05;
  ubvec[1] = 1.05;

  PetscCall(PetscMalloc1(ncon * (1 + numNonExclusivelyOwned), &vtxwgt));
  if (ncon == 2) {
    vtxwgt[0] = numExclusivelyOwned;
    vtxwgt[1] = 1;
    for (i = 0; i < numNonExclusivelyOwned; i++) {
      vtxwgt[ncon * (i + 1)]     = 1;
      vtxwgt[ncon * (i + 1) + 1] = 0;
    }
  } else {
    PetscInt base, ms;
    PetscCallMPI(MPI_Allreduce(&numExclusivelyOwned, &base, 1, MPIU_INT, MPIU_MAX, PetscObjectComm((PetscObject)dm)));
    PetscCall(MatGetSize(A, &ms, NULL));
    ms -= size;
    base      = PetscMax(base, ms);
    vtxwgt[0] = base + numExclusivelyOwned;
    for (i = 0; i < numNonExclusivelyOwned; i++) vtxwgt[i + 1] = 1;
  }

  if (viewer) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Attempt rebalancing of shared points of depth %" PetscInt_FMT " on interface of mesh distribution.\n", entityDepth));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Size of generated auxiliary graph: %" PetscInt_FMT "\n", cumSumVertices[size]));
  }
  /* TODO: Drop the parallel/sequential choice here and just use MatPartioner for much more flexibility */
  if (usematpartitioning) {
    const char *prefix;

    PetscCall(MatPartitioningCreate(PetscObjectComm((PetscObject)dm), &mp));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)mp, "dm_plex_rebalance_shared_points_"));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix));
    PetscCall(PetscObjectPrependOptionsPrefix((PetscObject)mp, prefix));
    PetscCall(MatPartitioningSetAdjacency(mp, A));
    PetscCall(MatPartitioningSetNumberVertexWeights(mp, ncon));
    PetscCall(MatPartitioningSetVertexWeights(mp, vtxwgt));
    PetscCall(MatPartitioningSetFromOptions(mp));
    PetscCall(MatPartitioningApply(mp, &ispart));
    PetscCall(ISGetIndices(ispart, (const PetscInt **)&part));
  } else if (parallel) {
    if (viewer) PetscCall(PetscViewerASCIIPrintf(viewer, "Using ParMETIS to partition graph.\n"));
    PetscCall(PetscMalloc1(cumSumVertices[rank + 1] - cumSumVertices[rank], &part));
    PetscCall(MatGetRowIJ(A, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &numRows, &xadj, &adjncy, &done));
    PetscCheck(done, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Could not get adjacency information");
    PetscCall(PetscMalloc1(4, &options));
    options[0] = 1;
    options[1] = 0; /* Verbosity */
    if (viewer) options[1] = 1;
    options[2] = 0;                    /* Seed */
    options[3] = PARMETIS_PSR_COUPLED; /* Seed */
    wgtflag    = 2;
    numflag    = 0;
    if (useInitialGuess) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "THIS DOES NOT WORK! I don't know why. Using current distribution of points as initial guess.\n"));
      for (i = 0; i < numRows; i++) part[i] = rank;
      if (viewer) PetscCall(PetscViewerASCIIPrintf(viewer, "Using current distribution of points as initial guess.\n"));
      PetscStackPushExternal("ParMETIS_V3_RefineKway");
      PetscCall(PetscLogEventBegin(DMPLEX_RebalPartition, 0, 0, 0, 0));
      ierr = ParMETIS_V3_RefineKway((PetscInt *)cumSumVertices, (idx_t *)xadj, (idx_t *)adjncy, vtxwgt, NULL, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);
      PetscCall(PetscLogEventEnd(DMPLEX_RebalPartition, 0, 0, 0, 0));
      PetscStackPop;
      PetscCheck(ierr == METIS_OK, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in ParMETIS_V3_RefineKway()");
    } else {
      PetscStackPushExternal("ParMETIS_V3_PartKway");
      PetscCall(PetscLogEventBegin(DMPLEX_RebalPartition, 0, 0, 0, 0));
      ierr = ParMETIS_V3_PartKway((PetscInt *)cumSumVertices, (idx_t *)xadj, (idx_t *)adjncy, vtxwgt, NULL, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);
      PetscCall(PetscLogEventEnd(DMPLEX_RebalPartition, 0, 0, 0, 0));
      PetscStackPop;
      PetscCheck(ierr == METIS_OK, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in ParMETIS_V3_PartKway()");
    }
    PetscCall(MatRestoreRowIJ(A, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &numRows, &xadj, &adjncy, &done));
    PetscCall(PetscFree(options));
  } else {
    if (viewer) PetscCall(PetscViewerASCIIPrintf(viewer, "Using METIS to partition graph.\n"));
    Mat       As;
    PetscInt *partGlobal;
    PetscInt *numExclusivelyOwnedAll;

    PetscCall(PetscMalloc1(cumSumVertices[rank + 1] - cumSumVertices[rank], &part));
    PetscCall(MatGetSize(A, &numRows, NULL));
    PetscCall(PetscLogEventBegin(DMPLEX_RebalGatherGraph, dm, 0, 0, 0));
    PetscCall(MatMPIAdjToSeqRankZero(A, &As));
    PetscCall(PetscLogEventEnd(DMPLEX_RebalGatherGraph, dm, 0, 0, 0));

    PetscCall(PetscMalloc1(size, &numExclusivelyOwnedAll));
    numExclusivelyOwnedAll[rank] = numExclusivelyOwned;
    PetscCallMPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, numExclusivelyOwnedAll, 1, MPIU_INT, comm));

    PetscCall(PetscMalloc1(numRows, &partGlobal));
    PetscCall(PetscLogEventBegin(DMPLEX_RebalPartition, 0, 0, 0, 0));
    if (rank == 0) {
      PetscInt *vtxwgt_g, numRows_g;

      PetscCall(MatGetRowIJ(As, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &numRows_g, &xadj, &adjncy, &done));
      PetscCall(PetscMalloc1(2 * numRows_g, &vtxwgt_g));
      for (i = 0; i < size; i++) {
        vtxwgt_g[ncon * cumSumVertices[i]] = numExclusivelyOwnedAll[i];
        if (ncon > 1) vtxwgt_g[ncon * cumSumVertices[i] + 1] = 1;
        for (j = cumSumVertices[i] + 1; j < cumSumVertices[i + 1]; j++) {
          vtxwgt_g[ncon * j] = 1;
          if (ncon > 1) vtxwgt_g[2 * j + 1] = 0;
        }
      }

      PetscCall(PetscMalloc1(64, &options));
      ierr = METIS_SetDefaultOptions(options); /* initialize all defaults */
      PetscCheck(ierr == METIS_OK, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_SetDefaultOptions()");
      options[METIS_OPTION_CONTIG] = 1;
      PetscStackPushExternal("METIS_PartGraphKway");
      ierr = METIS_PartGraphKway(&numRows_g, &ncon, (idx_t *)xadj, (idx_t *)adjncy, vtxwgt_g, NULL, NULL, &nparts, tpwgts, ubvec, options, &edgecut, partGlobal);
      PetscStackPop;
      PetscCheck(ierr == METIS_OK, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_PartGraphKway()");
      PetscCall(PetscFree(options));
      PetscCall(PetscFree(vtxwgt_g));
      PetscCall(MatRestoreRowIJ(As, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &numRows_g, &xadj, &adjncy, &done));
      PetscCall(MatDestroy(&As));
    }
    PetscCall(PetscBarrier((PetscObject)dm));
    PetscCall(PetscLogEventEnd(DMPLEX_RebalPartition, 0, 0, 0, 0));
    PetscCall(PetscFree(numExclusivelyOwnedAll));

    /* scatter the partitioning information to ranks */
    PetscCall(PetscLogEventBegin(DMPLEX_RebalScatterPart, 0, 0, 0, 0));
    PetscCall(PetscMalloc1(size, &counts));
    PetscCall(PetscMalloc1(size + 1, &mpiCumSumVertices));
    for (i = 0; i < size; i++) PetscCall(PetscMPIIntCast(cumSumVertices[i + 1] - cumSumVertices[i], &(counts[i])));
    for (i = 0; i <= size; i++) PetscCall(PetscMPIIntCast(cumSumVertices[i], &(mpiCumSumVertices[i])));
    PetscCallMPI(MPI_Scatterv(partGlobal, counts, mpiCumSumVertices, MPIU_INT, part, counts[rank], MPIU_INT, 0, comm));
    PetscCall(PetscFree(counts));
    PetscCall(PetscFree(mpiCumSumVertices));
    PetscCall(PetscFree(partGlobal));
    PetscCall(PetscLogEventEnd(DMPLEX_RebalScatterPart, 0, 0, 0, 0));
  }
  PetscCall(PetscFree(ubvec));
  PetscCall(PetscFree(tpwgts));

  /* Rename the result so that the vertex resembling the exclusively owned points stays on the same rank */
  PetscCall(PetscMalloc2(size, &firstVertices, size, &renumbering));
  firstVertices[rank] = part[0];
  PetscCallMPI(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, firstVertices, 1, MPIU_INT, comm));
  for (i = 0; i < size; i++) renumbering[firstVertices[i]] = i;
  for (i = 0; i < cumSumVertices[rank + 1] - cumSumVertices[rank]; i++) part[i] = renumbering[part[i]];
  PetscCall(PetscFree2(firstVertices, renumbering));

  /* Check if the renumbering worked (this can fail when ParMETIS gives fewer partitions than there are processes) */
  failed = (PetscInt)(part[0] != rank);
  PetscCallMPI(MPI_Allreduce(&failed, &failedGlobal, 1, MPIU_INT, MPI_SUM, comm));
  if (failedGlobal > 0) {
    PetscCheck(failedGlobal <= 0, comm, PETSC_ERR_LIB, "Metis/Parmetis returned a bad partion");
    PetscCall(PetscFree(vtxwgt));
    PetscCall(PetscFree(toBalance));
    PetscCall(PetscFree(isLeaf));
    PetscCall(PetscFree(isNonExclusivelyOwned));
    PetscCall(PetscFree(isExclusivelyOwned));
    if (usematpartitioning) {
      PetscCall(ISRestoreIndices(ispart, (const PetscInt **)&part));
      PetscCall(ISDestroy(&ispart));
    } else PetscCall(PetscFree(part));
    if (viewer) {
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    PetscCall(PetscLogEventEnd(DMPLEX_RebalanceSharedPoints, dm, 0, 0, 0));
    PetscFunctionReturn(0);
  }

  /* Check how well we did distributing points*/
  if (viewer) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Number of owned entities of depth %" PetscInt_FMT ".\n", entityDepth));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Initial      "));
    PetscCall(DMPlexViewDistribution(comm, cumSumVertices[rank + 1] - cumSumVertices[rank], ncon, vtxwgt, NULL, viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Rebalanced   "));
    PetscCall(DMPlexViewDistribution(comm, cumSumVertices[rank + 1] - cumSumVertices[rank], ncon, vtxwgt, part, viewer));
  }

  /* Check that every vertex is owned by a process that it is actually connected to. */
  PetscCall(MatGetRowIJ(A, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &numRows, (const PetscInt **)&xadj, (const PetscInt **)&adjncy, &done));
  for (i = 1; i <= numNonExclusivelyOwned; i++) {
    PetscInt loc = 0;
    PetscCall(PetscFindInt(cumSumVertices[part[i]], xadj[i + 1] - xadj[i], &adjncy[xadj[i]], &loc));
    /* If not, then just set the owner to the original owner (hopefully a rare event, it means that a vertex has been isolated) */
    if (loc < 0) part[i] = rank;
  }
  PetscCall(MatRestoreRowIJ(A, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, &numRows, (const PetscInt **)&xadj, (const PetscInt **)&adjncy, &done));
  PetscCall(MatDestroy(&A));

  /* See how significant the influences of the previous fixing up step was.*/
  if (viewer) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "After fix    "));
    PetscCall(DMPlexViewDistribution(comm, cumSumVertices[rank + 1] - cumSumVertices[rank], ncon, vtxwgt, part, viewer));
  }
  if (!usematpartitioning) PetscCall(PetscFree(vtxwgt));
  else PetscCall(MatPartitioningDestroy(&mp));

  PetscCall(PetscLayoutDestroy(&layout));

  PetscCall(PetscLogEventBegin(DMPLEX_RebalRewriteSF, dm, 0, 0, 0));
  /* Rewrite the SF to reflect the new ownership. */
  PetscCall(PetscMalloc1(numNonExclusivelyOwned, &pointsToRewrite));
  counter = 0;
  for (i = 0; i < pEnd - pStart; i++) {
    if (toBalance[i]) {
      if (isNonExclusivelyOwned[i]) {
        pointsToRewrite[counter] = i + pStart;
        counter++;
      }
    }
  }
  PetscCall(DMPlexRewriteSF(dm, numNonExclusivelyOwned, pointsToRewrite, part + 1, degrees));
  PetscCall(PetscFree(pointsToRewrite));
  PetscCall(PetscLogEventEnd(DMPLEX_RebalRewriteSF, dm, 0, 0, 0));

  PetscCall(PetscFree(toBalance));
  PetscCall(PetscFree(isLeaf));
  PetscCall(PetscFree(isNonExclusivelyOwned));
  PetscCall(PetscFree(isExclusivelyOwned));
  if (usematpartitioning) {
    PetscCall(ISRestoreIndices(ispart, (const PetscInt **)&part));
    PetscCall(ISDestroy(&ispart));
  } else PetscCall(PetscFree(part));
  if (viewer) {
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  if (success) *success = PETSC_TRUE;
  PetscCall(PetscLogEventEnd(DMPLEX_RebalanceSharedPoints, dm, 0, 0, 0));
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-parmetis.");
#endif
}
