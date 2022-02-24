#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/partitionerimpl.h>
#include <petsc/private/hashseti.h>

const char * const DMPlexCSRAlgorithms[] = {"mat", "graph", "overlap", "DMPlexCSRAlgorithm", "DM_PLEX_CSR_",NULL};

static inline PetscInt DMPlex_GlobalID(PetscInt point) { return point >= 0 ? point : -(point+1); }

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
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  if (dim != depth) {
    /* We do not handle the uninterpolated case here */
    CHKERRQ(DMPlexCreateNeighborCSR(dm, height, numVertices, offsets, adjacency));
    /* DMPlexCreateNeighborCSR does not make a numbering */
    if (globalNumbering) CHKERRQ(DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, globalNumbering));
    /* Different behavior for empty graphs */
    if (!*numVertices) {
      CHKERRQ(PetscMalloc1(1, offsets));
      (*offsets)[0] = 0;
    }
    /* Broken in parallel */
    if (rank) PetscCheck(!*numVertices,PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel partitioning of uninterpolated meshes not supported");
    PetscFunctionReturn(0);
  }
  /* Always use FVM adjacency to create partitioner graph */
  CHKERRQ(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  CHKERRQ(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
  /* Need overlap >= 1 */
  CHKERRQ(DMPlexGetOverlap(dm, &overlap));
  if (size && overlap < 1) {
    CHKERRQ(DMPlexDistributeOverlap(dm, 1, NULL, &ovdm));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject) dm));
    ovdm = dm;
  }
  CHKERRQ(DMGetPointSF(ovdm, &sfPoint));
  CHKERRQ(DMPlexGetHeightStratum(ovdm, height, &cStart, &cEnd));
  CHKERRQ(DMPlexCreateNumbering_Plex(ovdm, cStart, cEnd, 0, NULL, sfPoint, &cellNumbering));
  if (globalNumbering) {
    CHKERRQ(PetscObjectReference((PetscObject) cellNumbering));
    *globalNumbering = cellNumbering;
  }
  CHKERRQ(ISGetIndices(cellNumbering, &cellNum));
  /* Determine sizes */
  for (*numVertices = 0, c = cStart; c < cEnd; ++c) {
    /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
    if (cellNum[c] < 0) continue;
    (*numVertices)++;
  }
  CHKERRQ(PetscCalloc1(*numVertices+1, &vOffsets));
  for (c = cStart, v = 0; c < cEnd; ++c) {
    PetscInt adjSize = PETSC_DETERMINE, a, vsize = 0;

    if (cellNum[c] < 0) continue;
    CHKERRQ(DMPlexGetAdjacency(ovdm, c, &adjSize, &adj));
    for (a = 0; a < adjSize; ++a) {
      const PetscInt point = adj[a];
      if (point != c && cStart <= point && point < cEnd) ++vsize;
    }
    vOffsets[v+1] = vOffsets[v] + vsize;
    ++v;
  }
  /* Determine adjacency */
  CHKERRQ(PetscMalloc1(vOffsets[*numVertices], &vAdj));
  for (c = cStart, v = 0; c < cEnd; ++c) {
    PetscInt adjSize = PETSC_DETERMINE, a, off = vOffsets[v];

    /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
    if (cellNum[c] < 0) continue;
    CHKERRQ(DMPlexGetAdjacency(ovdm, c, &adjSize, &adj));
    for (a = 0; a < adjSize; ++a) {
      const PetscInt point = adj[a];
      if (point != c && cStart <= point && point < cEnd) {
        vAdj[off++] = DMPlex_GlobalID(cellNum[point]);
      }
    }
    PetscCheck(off == vOffsets[v+1],PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Offsets %D should be %D", off, vOffsets[v+1]);
    /* Sort adjacencies (not strictly necessary) */
    CHKERRQ(PetscSortInt(off-vOffsets[v], &vAdj[vOffsets[v]]));
    ++v;
  }
  CHKERRQ(PetscFree(adj));
  CHKERRQ(ISRestoreIndices(cellNumbering, &cellNum));
  CHKERRQ(ISDestroy(&cellNumbering));
  CHKERRQ(DMSetBasicAdjacency(dm, useCone, useClosure));
  CHKERRQ(DMDestroy(&ovdm));
  if (offsets)   {*offsets = vOffsets;}
  else           CHKERRQ(PetscFree(vOffsets));
  if (adjacency) {*adjacency = vAdj;}
  else           CHKERRQ(PetscFree(vAdj));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreatePartitionerGraph_Native(DM dm, PetscInt height, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency, IS *globalNumbering)
{
  PetscInt       dim, depth, p, pStart, pEnd, a, adjSize, idx, size;
  PetscInt      *adj = NULL, *vOffsets = NULL, *graph = NULL;
  IS             cellNumbering;
  const PetscInt *cellNum;
  PetscBool      useCone, useClosure;
  PetscSection   section;
  PetscSegBuffer adjBuffer;
  PetscSF        sfPoint;
  PetscInt       *adjCells = NULL, *remoteCells = NULL;
  const PetscInt *local;
  PetscInt       nroots, nleaves, l;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  if (dim != depth) {
    /* We do not handle the uninterpolated case here */
    CHKERRQ(DMPlexCreateNeighborCSR(dm, height, numVertices, offsets, adjacency));
    /* DMPlexCreateNeighborCSR does not make a numbering */
    if (globalNumbering) CHKERRQ(DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, globalNumbering));
    /* Different behavior for empty graphs */
    if (!*numVertices) {
      CHKERRQ(PetscMalloc1(1, offsets));
      (*offsets)[0] = 0;
    }
    /* Broken in parallel */
    if (rank) PetscCheck(!*numVertices,PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel partitioning of uninterpolated meshes not supported");
    PetscFunctionReturn(0);
  }
  CHKERRQ(DMGetPointSF(dm, &sfPoint));
  CHKERRQ(DMPlexGetHeightStratum(dm, height, &pStart, &pEnd));
  /* Build adjacency graph via a section/segbuffer */
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section));
  CHKERRQ(PetscSectionSetChart(section, pStart, pEnd));
  CHKERRQ(PetscSegBufferCreate(sizeof(PetscInt),1000,&adjBuffer));
  /* Always use FVM adjacency to create partitioner graph */
  CHKERRQ(DMGetBasicAdjacency(dm, &useCone, &useClosure));
  CHKERRQ(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
  CHKERRQ(DMPlexCreateNumbering_Plex(dm, pStart, pEnd, 0, NULL, sfPoint, &cellNumbering));
  if (globalNumbering) {
    CHKERRQ(PetscObjectReference((PetscObject)cellNumbering));
    *globalNumbering = cellNumbering;
  }
  CHKERRQ(ISGetIndices(cellNumbering, &cellNum));
  /* For all boundary faces (including faces adjacent to a ghost cell), record the local cell in adjCells
     Broadcast adjCells to remoteCells (to get cells from roots) and Reduce adjCells to remoteCells (to get cells from leaves)
   */
  CHKERRQ(PetscSFGetGraph(sfPoint, &nroots, &nleaves, &local, NULL));
  if (nroots >= 0) {
    PetscInt fStart, fEnd, f;

    CHKERRQ(PetscCalloc2(nroots, &adjCells, nroots, &remoteCells));
    CHKERRQ(DMPlexGetHeightStratum(dm, height+1, &fStart, &fEnd));
    for (l = 0; l < nroots; ++l) adjCells[l] = -3;
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *support;
      PetscInt        supportSize;

      CHKERRQ(DMPlexGetSupport(dm, f, &support));
      CHKERRQ(DMPlexGetSupportSize(dm, f, &supportSize));
      if (supportSize == 1) adjCells[f] = DMPlex_GlobalID(cellNum[support[0]]);
      else if (supportSize == 2) {
        CHKERRQ(PetscFindInt(support[0], nleaves, local, &p));
        if (p >= 0) adjCells[f] = DMPlex_GlobalID(cellNum[support[1]]);
        CHKERRQ(PetscFindInt(support[1], nleaves, local, &p));
        if (p >= 0) adjCells[f] = DMPlex_GlobalID(cellNum[support[0]]);
      }
      /* Handle non-conforming meshes */
      if (supportSize > 2) {
        PetscInt        numChildren, i;
        const PetscInt *children;

        CHKERRQ(DMPlexGetTreeChildren(dm, f, &numChildren, &children));
        for (i = 0; i < numChildren; ++i) {
          const PetscInt child = children[i];
          if (fStart <= child && child < fEnd) {
            CHKERRQ(DMPlexGetSupport(dm, child, &support));
            CHKERRQ(DMPlexGetSupportSize(dm, child, &supportSize));
            if (supportSize == 1) adjCells[child] = DMPlex_GlobalID(cellNum[support[0]]);
            else if (supportSize == 2) {
              CHKERRQ(PetscFindInt(support[0], nleaves, local, &p));
              if (p >= 0) adjCells[child] = DMPlex_GlobalID(cellNum[support[1]]);
              CHKERRQ(PetscFindInt(support[1], nleaves, local, &p));
              if (p >= 0) adjCells[child] = DMPlex_GlobalID(cellNum[support[0]]);
            }
          }
        }
      }
    }
    for (l = 0; l < nroots; ++l) remoteCells[l] = -1;
    CHKERRQ(PetscSFBcastBegin(dm->sf, MPIU_INT, adjCells, remoteCells,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(dm->sf, MPIU_INT, adjCells, remoteCells,MPI_REPLACE));
    CHKERRQ(PetscSFReduceBegin(dm->sf, MPIU_INT, adjCells, remoteCells, MPI_MAX));
    CHKERRQ(PetscSFReduceEnd(dm->sf, MPIU_INT, adjCells, remoteCells, MPI_MAX));
  }
  /* Combine local and global adjacencies */
  for (*numVertices = 0, p = pStart; p < pEnd; p++) {
    /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
    if (nroots > 0) {if (cellNum[p] < 0) continue;}
    /* Add remote cells */
    if (remoteCells) {
      const PetscInt gp = DMPlex_GlobalID(cellNum[p]);
      PetscInt       coneSize, numChildren, c, i;
      const PetscInt *cone, *children;

      CHKERRQ(DMPlexGetCone(dm, p, &cone));
      CHKERRQ(DMPlexGetConeSize(dm, p, &coneSize));
      for (c = 0; c < coneSize; ++c) {
        const PetscInt point = cone[c];
        if (remoteCells[point] >= 0 && remoteCells[point] != gp) {
          PetscInt *PETSC_RESTRICT pBuf;
          CHKERRQ(PetscSectionAddDof(section, p, 1));
          CHKERRQ(PetscSegBufferGetInts(adjBuffer, 1, &pBuf));
          *pBuf = remoteCells[point];
        }
        /* Handle non-conforming meshes */
        CHKERRQ(DMPlexGetTreeChildren(dm, point, &numChildren, &children));
        for (i = 0; i < numChildren; ++i) {
          const PetscInt child = children[i];
          if (remoteCells[child] >= 0 && remoteCells[child] != gp) {
            PetscInt *PETSC_RESTRICT pBuf;
            CHKERRQ(PetscSectionAddDof(section, p, 1));
            CHKERRQ(PetscSegBufferGetInts(adjBuffer, 1, &pBuf));
            *pBuf = remoteCells[child];
          }
        }
      }
    }
    /* Add local cells */
    adjSize = PETSC_DETERMINE;
    CHKERRQ(DMPlexGetAdjacency(dm, p, &adjSize, &adj));
    for (a = 0; a < adjSize; ++a) {
      const PetscInt point = adj[a];
      if (point != p && pStart <= point && point < pEnd) {
        PetscInt *PETSC_RESTRICT pBuf;
        CHKERRQ(PetscSectionAddDof(section, p, 1));
        CHKERRQ(PetscSegBufferGetInts(adjBuffer, 1, &pBuf));
        *pBuf = DMPlex_GlobalID(cellNum[point]);
      }
    }
    (*numVertices)++;
  }
  CHKERRQ(PetscFree(adj));
  CHKERRQ(PetscFree2(adjCells, remoteCells));
  CHKERRQ(DMSetBasicAdjacency(dm, useCone, useClosure));

  /* Derive CSR graph from section/segbuffer */
  CHKERRQ(PetscSectionSetUp(section));
  CHKERRQ(PetscSectionGetStorageSize(section, &size));
  CHKERRQ(PetscMalloc1(*numVertices+1, &vOffsets));
  for (idx = 0, p = pStart; p < pEnd; p++) {
    if (nroots > 0) {if (cellNum[p] < 0) continue;}
    CHKERRQ(PetscSectionGetOffset(section, p, &(vOffsets[idx++])));
  }
  vOffsets[*numVertices] = size;
  CHKERRQ(PetscSegBufferExtractAlloc(adjBuffer, &graph));

  if (nroots >= 0) {
    /* Filter out duplicate edges using section/segbuffer */
    CHKERRQ(PetscSectionReset(section));
    CHKERRQ(PetscSectionSetChart(section, 0, *numVertices));
    for (p = 0; p < *numVertices; p++) {
      PetscInt start = vOffsets[p], end = vOffsets[p+1];
      PetscInt numEdges = end-start, *PETSC_RESTRICT edges;
      CHKERRQ(PetscSortRemoveDupsInt(&numEdges, &graph[start]));
      CHKERRQ(PetscSectionSetDof(section, p, numEdges));
      CHKERRQ(PetscSegBufferGetInts(adjBuffer, numEdges, &edges));
      CHKERRQ(PetscArraycpy(edges, &graph[start], numEdges));
    }
    CHKERRQ(PetscFree(vOffsets));
    CHKERRQ(PetscFree(graph));
    /* Derive CSR graph from section/segbuffer */
    CHKERRQ(PetscSectionSetUp(section));
    CHKERRQ(PetscSectionGetStorageSize(section, &size));
    CHKERRQ(PetscMalloc1(*numVertices+1, &vOffsets));
    for (idx = 0, p = 0; p < *numVertices; p++) {
      CHKERRQ(PetscSectionGetOffset(section, p, &(vOffsets[idx++])));
    }
    vOffsets[*numVertices] = size;
    CHKERRQ(PetscSegBufferExtractAlloc(adjBuffer, &graph));
  } else {
    /* Sort adjacencies (not strictly necessary) */
    for (p = 0; p < *numVertices; p++) {
      PetscInt start = vOffsets[p], end = vOffsets[p+1];
      CHKERRQ(PetscSortInt(end-start, &graph[start]));
    }
  }

  if (offsets) *offsets = vOffsets;
  if (adjacency) *adjacency = graph;

  /* Cleanup */
  CHKERRQ(PetscSegBufferDestroy(&adjBuffer));
  CHKERRQ(PetscSectionDestroy(&section));
  CHKERRQ(ISRestoreIndices(cellNumbering, &cellNum));
  CHKERRQ(ISDestroy(&cellNumbering));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreatePartitionerGraph_ViaMat(DM dm, PetscInt height, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency, IS *globalNumbering)
{
  Mat            conn, CSR;
  IS             fis, cis, cis_own;
  PetscSF        sfPoint;
  const PetscInt *rows, *cols, *ii, *jj;
  PetscInt       *idxs,*idxs2;
  PetscInt       dim, depth, floc, cloc, i, M, N, c, lm, m, cStart, cEnd, fStart, fEnd;
  PetscMPIInt    rank;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  if (dim != depth) {
    /* We do not handle the uninterpolated case here */
    CHKERRQ(DMPlexCreateNeighborCSR(dm, height, numVertices, offsets, adjacency));
    /* DMPlexCreateNeighborCSR does not make a numbering */
    if (globalNumbering) CHKERRQ(DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, globalNumbering));
    /* Different behavior for empty graphs */
    if (!*numVertices) {
      CHKERRQ(PetscMalloc1(1, offsets));
      (*offsets)[0] = 0;
    }
    /* Broken in parallel */
    if (rank) PetscCheck(!*numVertices,PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel partitioning of uninterpolated meshes not supported");
    PetscFunctionReturn(0);
  }
  /* Interpolated and parallel case */

  /* numbering */
  CHKERRQ(DMGetPointSF(dm, &sfPoint));
  CHKERRQ(DMPlexGetHeightStratum(dm, height, &cStart, &cEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, height+1, &fStart, &fEnd));
  CHKERRQ(DMPlexCreateNumbering_Plex(dm, cStart, cEnd, 0, &N, sfPoint, &cis));
  CHKERRQ(DMPlexCreateNumbering_Plex(dm, fStart, fEnd, 0, &M, sfPoint, &fis));
  if (globalNumbering) {
    CHKERRQ(ISDuplicate(cis, globalNumbering));
  }

  /* get positive global ids and local sizes for facets and cells */
  CHKERRQ(ISGetLocalSize(fis, &m));
  CHKERRQ(ISGetIndices(fis, &rows));
  CHKERRQ(PetscMalloc1(m, &idxs));
  for (i = 0, floc = 0; i < m; i++) {
    const PetscInt p = rows[i];

    if (p < 0) {
      idxs[i] = -(p+1);
    } else {
      idxs[i] = p;
      floc   += 1;
    }
  }
  CHKERRQ(ISRestoreIndices(fis, &rows));
  CHKERRQ(ISDestroy(&fis));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, m, idxs, PETSC_OWN_POINTER, &fis));

  CHKERRQ(ISGetLocalSize(cis, &m));
  CHKERRQ(ISGetIndices(cis, &cols));
  CHKERRQ(PetscMalloc1(m, &idxs));
  CHKERRQ(PetscMalloc1(m, &idxs2));
  for (i = 0, cloc = 0; i < m; i++) {
    const PetscInt p = cols[i];

    if (p < 0) {
      idxs[i] = -(p+1);
    } else {
      idxs[i]       = p;
      idxs2[cloc++] = p;
    }
  }
  CHKERRQ(ISRestoreIndices(cis, &cols));
  CHKERRQ(ISDestroy(&cis));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)dm), m, idxs, PETSC_OWN_POINTER, &cis));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)dm), cloc, idxs2, PETSC_OWN_POINTER, &cis_own));

  /* Create matrix to hold F-C connectivity (MatMatTranspose Mult not supported for MPIAIJ) */
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)dm), &conn));
  CHKERRQ(MatSetSizes(conn, floc, cloc, M, N));
  CHKERRQ(MatSetType(conn, MATMPIAIJ));
  CHKERRQ(DMPlexGetMaxSizes(dm, NULL, &lm));
  CHKERRMPI(MPI_Allreduce(&lm, &m, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) dm)));
  CHKERRQ(MatMPIAIJSetPreallocation(conn, m, NULL, m, NULL));

  /* Assemble matrix */
  CHKERRQ(ISGetIndices(fis, &rows));
  CHKERRQ(ISGetIndices(cis, &cols));
  for (c = cStart; c < cEnd; c++) {
    const PetscInt *cone;
    PetscInt        coneSize, row, col, f;

    col  = cols[c-cStart];
    CHKERRQ(DMPlexGetCone(dm, c, &cone));
    CHKERRQ(DMPlexGetConeSize(dm, c, &coneSize));
    for (f = 0; f < coneSize; f++) {
      const PetscScalar v = 1.0;
      const PetscInt *children;
      PetscInt        numChildren, ch;

      row  = rows[cone[f]-fStart];
      CHKERRQ(MatSetValues(conn, 1, &row, 1, &col, &v, INSERT_VALUES));

      /* non-conforming meshes */
      CHKERRQ(DMPlexGetTreeChildren(dm, cone[f], &numChildren, &children));
      for (ch = 0; ch < numChildren; ch++) {
        const PetscInt child = children[ch];

        if (child < fStart || child >= fEnd) continue;
        row  = rows[child-fStart];
        CHKERRQ(MatSetValues(conn, 1, &row, 1, &col, &v, INSERT_VALUES));
      }
    }
  }
  CHKERRQ(ISRestoreIndices(fis, &rows));
  CHKERRQ(ISRestoreIndices(cis, &cols));
  CHKERRQ(ISDestroy(&fis));
  CHKERRQ(ISDestroy(&cis));
  CHKERRQ(MatAssemblyBegin(conn, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(conn, MAT_FINAL_ASSEMBLY));

  /* Get parallel CSR by doing conn^T * conn */
  CHKERRQ(MatTransposeMatMult(conn, conn, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CSR));
  CHKERRQ(MatDestroy(&conn));

  /* extract local part of the CSR */
  CHKERRQ(MatMPIAIJGetLocalMat(CSR, MAT_INITIAL_MATRIX, &conn));
  CHKERRQ(MatDestroy(&CSR));
  CHKERRQ(MatGetRowIJ(conn, 0, PETSC_FALSE, PETSC_FALSE, &m, &ii, &jj, &flg));
  PetscCheck(flg,PETSC_COMM_SELF, PETSC_ERR_PLIB, "No IJ format");

  /* get back requested output */
  if (numVertices) *numVertices = m;
  if (offsets) {
    CHKERRQ(PetscCalloc1(m+1, &idxs));
    for (i = 1; i < m+1; i++) idxs[i] = ii[i] - i; /* ParMetis does not like self-connectivity */
    *offsets = idxs;
  }
  if (adjacency) {
    CHKERRQ(PetscMalloc1(ii[m] - m, &idxs));
    CHKERRQ(ISGetIndices(cis_own, &rows));
    for (i = 0, c = 0; i < m; i++) {
      PetscInt j, g = rows[i];

      for (j = ii[i]; j < ii[i+1]; j++) {
        if (jj[j] == g) continue; /* again, self-connectivity */
        idxs[c++] = jj[j];
      }
    }
    PetscCheck(c == ii[m] - m,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected %D != %D",c,ii[m]-m);
    CHKERRQ(ISRestoreIndices(cis_own, &rows));
    *adjacency = idxs;
  }

  /* cleanup */
  CHKERRQ(ISDestroy(&cis_own));
  CHKERRQ(MatRestoreRowIJ(conn, 0, PETSC_FALSE, PETSC_FALSE, &m, &ii, &jj, &flg));
  PetscCheck(flg,PETSC_COMM_SELF, PETSC_ERR_PLIB, "No IJ format");
  CHKERRQ(MatDestroy(&conn));
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreatePartitionerGraph - Create a CSR graph of point connections for the partitioner

  Input Parameters:
+ dm      - The mesh DM dm
- height  - Height of the strata from which to construct the graph

  Output Parameters:
+ numVertices     - Number of vertices in the graph
. offsets         - Point offsets in the graph
. adjacency       - Point connectivity in the graph
- globalNumbering - A map from the local cell numbering to the global numbering used in "adjacency".  Negative indicates that the cell is a duplicate from another process.

  The user can control the definition of adjacency for the mesh using DMSetAdjacency(). They should choose the combination appropriate for the function
  representation on the mesh. If requested, globalNumbering needs to be destroyed by the caller; offsets and adjacency need to be freed with PetscFree().

  Options Database Keys:
. -dm_plex_csr_alg <mat,graph,overlap> - Choose the algorithm for computing the CSR graph

  Level: developer

.seealso: PetscPartitionerGetType(), PetscPartitionerCreate(), DMSetAdjacency()
@*/
PetscErrorCode DMPlexCreatePartitionerGraph(DM dm, PetscInt height, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency, IS *globalNumbering)
{
  DMPlexCSRAlgorithm alg = DM_PLEX_CSR_GRAPH;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetEnum(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_csr_alg", DMPlexCSRAlgorithms, (PetscEnum *) &alg, NULL));
  switch (alg) {
    case DM_PLEX_CSR_MAT:
      CHKERRQ(DMPlexCreatePartitionerGraph_ViaMat(dm, height, numVertices, offsets, adjacency, globalNumbering));break;
    case DM_PLEX_CSR_GRAPH:
      CHKERRQ(DMPlexCreatePartitionerGraph_Native(dm, height, numVertices, offsets, adjacency, globalNumbering));break;
    case DM_PLEX_CSR_OVERLAP:
      CHKERRQ(DMPlexCreatePartitionerGraph_Overlap(dm, height, numVertices, offsets, adjacency, globalNumbering));break;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateNeighborCSR - Create a mesh graph (cell-cell adjacency) in parallel CSR format.

  Collective on DM

  Input Parameters:
+ dm - The DMPlex
- cellHeight - The height of mesh points to treat as cells (default should be 0)

  Output Parameters:
+ numVertices - The number of local vertices in the graph, or cells in the mesh.
. offsets     - The offset to the adjacency list for each cell
- adjacency   - The adjacency list for all cells

  Note: This is suitable for input to a mesh partitioner like ParMetis.

  Level: advanced

.seealso: DMPlexCreate()
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
  CHKERRQ(DMGetDimension(dm, &dim));
  cellDim = dim - cellHeight;
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd));
  if (cEnd - cStart == 0) {
    if (numVertices) *numVertices = 0;
    if (offsets)   *offsets   = NULL;
    if (adjacency) *adjacency = NULL;
    PetscFunctionReturn(0);
  }
  numCells  = cEnd - cStart;
  faceDepth = depth - cellHeight;
  if (dim == depth) {
    PetscInt f, fStart, fEnd;

    CHKERRQ(PetscCalloc1(numCells+1, &off));
    /* Count neighboring cells */
    CHKERRQ(DMPlexGetHeightStratum(dm, cellHeight+1, &fStart, &fEnd));
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *support;
      PetscInt        supportSize;
      CHKERRQ(DMPlexGetSupportSize(dm, f, &supportSize));
      CHKERRQ(DMPlexGetSupport(dm, f, &support));
      if (supportSize == 2) {
        PetscInt numChildren;

        CHKERRQ(DMPlexGetTreeChildren(dm,f,&numChildren,NULL));
        if (!numChildren) {
          ++off[support[0]-cStart+1];
          ++off[support[1]-cStart+1];
        }
      }
    }
    /* Prefix sum */
    for (c = 1; c <= numCells; ++c) off[c] += off[c-1];
    if (adjacency) {
      PetscInt *tmp;

      CHKERRQ(PetscMalloc1(off[numCells], &adj));
      CHKERRQ(PetscMalloc1(numCells+1, &tmp));
      CHKERRQ(PetscArraycpy(tmp, off, numCells+1));
      /* Get neighboring cells */
      for (f = fStart; f < fEnd; ++f) {
        const PetscInt *support;
        PetscInt        supportSize;
        CHKERRQ(DMPlexGetSupportSize(dm, f, &supportSize));
        CHKERRQ(DMPlexGetSupport(dm, f, &support));
        if (supportSize == 2) {
          PetscInt numChildren;

          CHKERRQ(DMPlexGetTreeChildren(dm,f,&numChildren,NULL));
          if (!numChildren) {
            adj[tmp[support[0]-cStart]++] = support[1];
            adj[tmp[support[1]-cStart]++] = support[0];
          }
        }
      }
      for (c = 0; c < cEnd-cStart; ++c) PetscAssert(tmp[c] == off[c+1],PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offset %d != %d for cell %d", tmp[c], off[c], c+cStart);
      CHKERRQ(PetscFree(tmp));
    }
    if (numVertices) *numVertices = numCells;
    if (offsets)   *offsets   = off;
    if (adjacency) *adjacency = adj;
    PetscFunctionReturn(0);
  }
  /* Setup face recognition */
  if (faceDepth == 1) {
    PetscInt cornersSeen[30] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; /* Could use PetscBT */

    for (c = cStart; c < cEnd; ++c) {
      PetscInt corners;

      CHKERRQ(DMPlexGetConeSize(dm, c, &corners));
      if (!cornersSeen[corners]) {
        PetscInt nFV;

        PetscCheck(numFaceCases < maxFaceCases,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Exceeded maximum number of face recognition cases");
        cornersSeen[corners] = 1;

        CHKERRQ(DMPlexGetNumFaceVertices(dm, cellDim, corners, &nFV));

        numFaceVertices[numFaceCases++] = nFV;
      }
    }
  }
  CHKERRQ(PetscCalloc1(numCells+1, &off));
  /* Count neighboring cells */
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt numNeighbors = PETSC_DETERMINE, n;

    CHKERRQ(DMPlexGetAdjacency_Internal(dm, cell, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &numNeighbors, &neighborCells));
    /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
    for (n = 0; n < numNeighbors; ++n) {
      PetscInt        cellPair[2];
      PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
      PetscInt        meetSize = 0;
      const PetscInt *meet    = NULL;

      cellPair[0] = cell; cellPair[1] = neighborCells[n];
      if (cellPair[0] == cellPair[1]) continue;
      if (!found) {
        CHKERRQ(DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet));
        if (meetSize) {
          PetscInt f;

          for (f = 0; f < numFaceCases; ++f) {
            if (numFaceVertices[f] == meetSize) {
              found = PETSC_TRUE;
              break;
            }
          }
        }
        CHKERRQ(DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet));
      }
      if (found) ++off[cell-cStart+1];
    }
  }
  /* Prefix sum */
  for (cell = 1; cell <= numCells; ++cell) off[cell] += off[cell-1];

  if (adjacency) {
    CHKERRQ(PetscMalloc1(off[numCells], &adj));
    /* Get neighboring cells */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt numNeighbors = PETSC_DETERMINE, n;
      PetscInt cellOffset   = 0;

      CHKERRQ(DMPlexGetAdjacency_Internal(dm, cell, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &numNeighbors, &neighborCells));
      /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
      for (n = 0; n < numNeighbors; ++n) {
        PetscInt        cellPair[2];
        PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
        PetscInt        meetSize = 0;
        const PetscInt *meet    = NULL;

        cellPair[0] = cell; cellPair[1] = neighborCells[n];
        if (cellPair[0] == cellPair[1]) continue;
        if (!found) {
          CHKERRQ(DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet));
          if (meetSize) {
            PetscInt f;

            for (f = 0; f < numFaceCases; ++f) {
              if (numFaceVertices[f] == meetSize) {
                found = PETSC_TRUE;
                break;
              }
            }
          }
          CHKERRQ(DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet));
        }
        if (found) {
          adj[off[cell-cStart]+cellOffset] = neighborCells[n];
          ++cellOffset;
        }
      }
    }
  }
  CHKERRQ(PetscFree(neighborCells));
  if (numVertices) *numVertices = numCells;
  if (offsets)   *offsets   = off;
  if (adjacency) *adjacency = adj;
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerDMPlexPartition - Create a non-overlapping partition of the cells in the mesh

  Collective on PetscPartitioner

  Input Parameters:
+ part    - The PetscPartitioner
. targetSection - The PetscSection describing the absolute weight of each partition (can be NULL)
- dm      - The mesh DM

  Output Parameters:
+ partSection     - The PetscSection giving the division of points by partition
- partition       - The list of points by partition

  Notes:
    If the DM has a local section associated, each point to be partitioned will be weighted by the total number of dofs identified
    by the section in the transitive closure of the point.

  Level: developer

.seealso DMPlexDistribute(), PetscPartitionerCreate(), PetscSectionCreate(), PetscSectionSetChart(), PetscPartitionerPartition()
@*/
PetscErrorCode PetscPartitionerDMPlexPartition(PetscPartitioner part, DM dm, PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscMPIInt    size;
  PetscBool      isplex;
  PetscSection   vertSection = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (targetSection) PetscValidHeaderSpecific(targetSection, PETSC_SECTION_CLASSID, 3);
  PetscValidHeaderSpecific(partSection, PETSC_SECTION_CLASSID, 4);
  PetscValidPointer(partition, 5);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isplex));
  PetscCheck(isplex,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not for type %s",((PetscObject)dm)->type_name);
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject) part), &size));
  if (size == 1) {
    PetscInt *points;
    PetscInt  cStart, cEnd, c;

    CHKERRQ(DMPlexGetHeightStratum(dm, part->height, &cStart, &cEnd));
    CHKERRQ(PetscSectionReset(partSection));
    CHKERRQ(PetscSectionSetChart(partSection, 0, size));
    CHKERRQ(PetscSectionSetDof(partSection, 0, cEnd-cStart));
    CHKERRQ(PetscSectionSetUp(partSection));
    CHKERRQ(PetscMalloc1(cEnd-cStart, &points));
    for (c = cStart; c < cEnd; ++c) points[c] = c;
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject) part), cEnd-cStart, points, PETSC_OWN_POINTER, partition));
    PetscFunctionReturn(0);
  }
  if (part->height == 0) {
    PetscInt numVertices = 0;
    PetscInt *start     = NULL;
    PetscInt *adjacency = NULL;
    IS       globalNumbering;

    if (!part->noGraph || part->viewGraph) {
      CHKERRQ(DMPlexCreatePartitionerGraph(dm, part->height, &numVertices, &start, &adjacency, &globalNumbering));
    } else { /* only compute the number of owned local vertices */
      const PetscInt *idxs;
      PetscInt       p, pStart, pEnd;

      CHKERRQ(DMPlexGetHeightStratum(dm, part->height, &pStart, &pEnd));
      CHKERRQ(DMPlexCreateNumbering_Plex(dm, pStart, pEnd, 0, NULL, dm->sf, &globalNumbering));
      CHKERRQ(ISGetIndices(globalNumbering, &idxs));
      for (p = 0; p < pEnd - pStart; p++) numVertices += idxs[p] < 0 ? 0 : 1;
      CHKERRQ(ISRestoreIndices(globalNumbering, &idxs));
    }
    if (part->usevwgt) {
      PetscSection   section = dm->localSection, clSection = NULL;
      IS             clPoints = NULL;
      const PetscInt *gid,*clIdx;
      PetscInt       v, p, pStart, pEnd;

      /* dm->localSection encodes degrees of freedom per point, not per cell. We need to get the closure index to properly specify cell weights (aka dofs) */
      /* We do this only if the local section has been set */
      if (section) {
        CHKERRQ(PetscSectionGetClosureIndex(section, (PetscObject)dm, &clSection, NULL));
        if (!clSection) {
          CHKERRQ(DMPlexCreateClosureIndex(dm,NULL));
        }
        CHKERRQ(PetscSectionGetClosureIndex(section, (PetscObject)dm, &clSection, &clPoints));
        CHKERRQ(ISGetIndices(clPoints,&clIdx));
      }
      CHKERRQ(DMPlexGetHeightStratum(dm, part->height, &pStart, &pEnd));
      CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &vertSection));
      CHKERRQ(PetscSectionSetChart(vertSection, 0, numVertices));
      if (globalNumbering) {
        CHKERRQ(ISGetIndices(globalNumbering,&gid));
      } else gid = NULL;
      for (p = pStart, v = 0; p < pEnd; ++p) {
        PetscInt dof = 1;

        /* skip cells in the overlap */
        if (gid && gid[p-pStart] < 0) continue;

        if (section) {
          PetscInt cl, clSize, clOff;

          dof  = 0;
          CHKERRQ(PetscSectionGetDof(clSection, p, &clSize));
          CHKERRQ(PetscSectionGetOffset(clSection, p, &clOff));
          for (cl = 0; cl < clSize; cl+=2) {
            PetscInt clDof, clPoint = clIdx[clOff + cl]; /* odd indices are reserved for orientations */

            CHKERRQ(PetscSectionGetDof(section, clPoint, &clDof));
            dof += clDof;
          }
        }
        PetscCheck(dof,PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of dofs for point %D in the local section should be positive",p);
        CHKERRQ(PetscSectionSetDof(vertSection, v, dof));
        v++;
      }
      if (globalNumbering) {
        CHKERRQ(ISRestoreIndices(globalNumbering,&gid));
      }
      if (clPoints) {
        CHKERRQ(ISRestoreIndices(clPoints,&clIdx));
      }
      CHKERRQ(PetscSectionSetUp(vertSection));
    }
    CHKERRQ(PetscPartitionerPartition(part, size, numVertices, start, adjacency, vertSection, targetSection, partSection, partition));
    CHKERRQ(PetscFree(start));
    CHKERRQ(PetscFree(adjacency));
    if (globalNumbering) { /* partition is wrt global unique numbering: change this to be wrt local numbering */
      const PetscInt *globalNum;
      const PetscInt *partIdx;
      PetscInt       *map, cStart, cEnd;
      PetscInt       *adjusted, i, localSize, offset;
      IS             newPartition;

      CHKERRQ(ISGetLocalSize(*partition,&localSize));
      CHKERRQ(PetscMalloc1(localSize,&adjusted));
      CHKERRQ(ISGetIndices(globalNumbering,&globalNum));
      CHKERRQ(ISGetIndices(*partition,&partIdx));
      CHKERRQ(PetscMalloc1(localSize,&map));
      CHKERRQ(DMPlexGetHeightStratum(dm, part->height, &cStart, &cEnd));
      for (i = cStart, offset = 0; i < cEnd; i++) {
        if (globalNum[i - cStart] >= 0) map[offset++] = i;
      }
      for (i = 0; i < localSize; i++) {
        adjusted[i] = map[partIdx[i]];
      }
      CHKERRQ(PetscFree(map));
      CHKERRQ(ISRestoreIndices(*partition,&partIdx));
      CHKERRQ(ISRestoreIndices(globalNumbering,&globalNum));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,localSize,adjusted,PETSC_OWN_POINTER,&newPartition));
      CHKERRQ(ISDestroy(&globalNumbering));
      CHKERRQ(ISDestroy(partition));
      *partition = newPartition;
    }
  } else SETERRQ(PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_OUTOFRANGE, "Invalid height %D for points to partition", part->height);
  CHKERRQ(PetscSectionDestroy(&vertSection));
  PetscFunctionReturn(0);
}

/*@
  DMPlexGetPartitioner - Get the mesh partitioner

  Not collective

  Input Parameter:
. dm - The DM

  Output Parameter:
. part - The PetscPartitioner

  Level: developer

  Note: This gets a borrowed reference, so the user should not destroy this PetscPartitioner.

.seealso DMPlexDistribute(), DMPlexSetPartitioner(), PetscPartitionerDMPlexPartition(), PetscPartitionerCreate()
@*/
PetscErrorCode DMPlexGetPartitioner(DM dm, PetscPartitioner *part)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(part, 2);
  *part = mesh->partitioner;
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetPartitioner - Set the mesh partitioner

  logically collective on DM

  Input Parameters:
+ dm - The DM
- part - The partitioner

  Level: developer

  Note: Any existing PetscPartitioner will be destroyed.

.seealso DMPlexDistribute(), DMPlexGetPartitioner(), PetscPartitionerCreate()
@*/
PetscErrorCode DMPlexSetPartitioner(DM dm, PetscPartitioner part)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 2);
  CHKERRQ(PetscObjectReference((PetscObject)part));
  CHKERRQ(PetscPartitionerDestroy(&mesh->partitioner));
  mesh->partitioner = part;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosure_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  const PetscInt *cone;
  PetscInt       coneSize, c;
  PetscBool      missing;

  PetscFunctionBeginHot;
  CHKERRQ(PetscHSetIQueryAdd(ht, point, &missing));
  if (missing) {
    CHKERRQ(DMPlexGetCone(dm, point, &cone));
    CHKERRQ(DMPlexGetConeSize(dm, point, &coneSize));
    for (c = 0; c < coneSize; c++) {
      CHKERRQ(DMPlexAddClosure_Private(dm, ht, cone[c]));
    }
  }
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode DMPlexAddClosure_Tree(DM dm, PetscHSetI ht, PetscInt point, PetscBool up, PetscBool down)
{
  PetscFunctionBegin;
  if (up) {
    PetscInt parent;

    CHKERRQ(DMPlexGetTreeParent(dm,point,&parent,NULL));
    if (parent != point) {
      PetscInt closureSize, *closure = NULL, i;

      CHKERRQ(DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
      for (i = 0; i < closureSize; i++) {
        PetscInt cpoint = closure[2*i];

        CHKERRQ(PetscHSetIAdd(ht, cpoint));
        CHKERRQ(DMPlexAddClosure_Tree(dm,ht,cpoint,PETSC_TRUE,PETSC_FALSE));
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure));
    }
  }
  if (down) {
    PetscInt numChildren;
    const PetscInt *children;

    CHKERRQ(DMPlexGetTreeChildren(dm,point,&numChildren,&children));
    if (numChildren) {
      PetscInt i;

      for (i = 0; i < numChildren; i++) {
        PetscInt cpoint = children[i];

        CHKERRQ(PetscHSetIAdd(ht, cpoint));
        CHKERRQ(DMPlexAddClosure_Tree(dm,ht,cpoint,PETSC_FALSE,PETSC_TRUE));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosureTree_Up_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  PetscInt       parent;

  PetscFunctionBeginHot;
  CHKERRQ(DMPlexGetTreeParent(dm, point, &parent,NULL));
  if (point != parent) {
    const PetscInt *cone;
    PetscInt       coneSize, c;

    CHKERRQ(DMPlexAddClosureTree_Up_Private(dm, ht, parent));
    CHKERRQ(DMPlexAddClosure_Private(dm, ht, parent));
    CHKERRQ(DMPlexGetCone(dm, parent, &cone));
    CHKERRQ(DMPlexGetConeSize(dm, parent, &coneSize));
    for (c = 0; c < coneSize; c++) {
      const PetscInt cp = cone[c];

      CHKERRQ(DMPlexAddClosureTree_Up_Private(dm, ht, cp));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosureTree_Down_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  PetscInt       i, numChildren;
  const PetscInt *children;

  PetscFunctionBeginHot;
  CHKERRQ(DMPlexGetTreeChildren(dm, point, &numChildren, &children));
  for (i = 0; i < numChildren; i++) {
    CHKERRQ(PetscHSetIAdd(ht, children[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosureTree_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  const PetscInt *cone;
  PetscInt       coneSize, c;

  PetscFunctionBeginHot;
  CHKERRQ(PetscHSetIAdd(ht, point));
  CHKERRQ(DMPlexAddClosureTree_Up_Private(dm, ht, point));
  CHKERRQ(DMPlexAddClosureTree_Down_Private(dm, ht, point));
  CHKERRQ(DMPlexGetCone(dm, point, &cone));
  CHKERRQ(DMPlexGetConeSize(dm, point, &coneSize));
  for (c = 0; c < coneSize; c++) {
    CHKERRQ(DMPlexAddClosureTree_Private(dm, ht, cone[c]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexClosurePoints_Private(DM dm, PetscInt numPoints, const PetscInt points[], IS *closureIS)
{
  DM_Plex         *mesh = (DM_Plex *)dm->data;
  const PetscBool hasTree = (mesh->parentSection || mesh->childSection) ? PETSC_TRUE : PETSC_FALSE;
  PetscInt        nelems, *elems, off = 0, p;
  PetscHSetI      ht = NULL;

  PetscFunctionBegin;
  CHKERRQ(PetscHSetICreate(&ht));
  CHKERRQ(PetscHSetIResize(ht, numPoints*16));
  if (!hasTree) {
    for (p = 0; p < numPoints; ++p) {
      CHKERRQ(DMPlexAddClosure_Private(dm, ht, points[p]));
    }
  } else {
#if 1
    for (p = 0; p < numPoints; ++p) {
      CHKERRQ(DMPlexAddClosureTree_Private(dm, ht, points[p]));
    }
#else
    PetscInt  *closure = NULL, closureSize, c;
    for (p = 0; p < numPoints; ++p) {
      CHKERRQ(DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure));
      for (c = 0; c < closureSize*2; c += 2) {
        CHKERRQ(PetscHSetIAdd(ht, closure[c]));
        if (hasTree) CHKERRQ(DMPlexAddClosure_Tree(dm, ht, closure[c], PETSC_TRUE, PETSC_TRUE));
      }
    }
    if (closure) CHKERRQ(DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, NULL, &closure));
#endif
  }
  CHKERRQ(PetscHSetIGetSize(ht, &nelems));
  CHKERRQ(PetscMalloc1(nelems, &elems));
  CHKERRQ(PetscHSetIGetElems(ht, &off, elems));
  CHKERRQ(PetscHSetIDestroy(&ht));
  CHKERRQ(PetscSortInt(nelems, elems));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, nelems, elems, PETSC_OWN_POINTER, closureIS));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelClosure - Add the closure of all points to the partition label

  Input Parameters:
+ dm     - The DM
- label  - DMLabel assigning ranks to remote roots

  Level: developer

.seealso: DMPlexPartitionLabelCreateSF(), DMPlexDistribute(), DMPlexCreateOverlap()
@*/
PetscErrorCode DMPlexPartitionLabelClosure(DM dm, DMLabel label)
{
  IS              rankIS,   pointIS, closureIS;
  const PetscInt *ranks,   *points;
  PetscInt        numRanks, numPoints, r;

  PetscFunctionBegin;
  CHKERRQ(DMLabelGetValueIS(label, &rankIS));
  CHKERRQ(ISGetLocalSize(rankIS, &numRanks));
  CHKERRQ(ISGetIndices(rankIS, &ranks));
  for (r = 0; r < numRanks; ++r) {
    const PetscInt rank = ranks[r];
    CHKERRQ(DMLabelGetStratumIS(label, rank, &pointIS));
    CHKERRQ(ISGetLocalSize(pointIS, &numPoints));
    CHKERRQ(ISGetIndices(pointIS, &points));
    CHKERRQ(DMPlexClosurePoints_Private(dm, numPoints, points, &closureIS));
    CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
    CHKERRQ(DMLabelSetStratumIS(label, rank, closureIS));
    CHKERRQ(ISDestroy(&closureIS));
  }
  CHKERRQ(ISRestoreIndices(rankIS, &ranks));
  CHKERRQ(ISDestroy(&rankIS));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelAdjacency - Add one level of adjacent points to the partition label

  Input Parameters:
+ dm     - The DM
- label  - DMLabel assigning ranks to remote roots

  Level: developer

.seealso: DMPlexPartitionLabelCreateSF(), DMPlexDistribute(), DMPlexCreateOverlap()
@*/
PetscErrorCode DMPlexPartitionLabelAdjacency(DM dm, DMLabel label)
{
  IS              rankIS,   pointIS;
  const PetscInt *ranks,   *points;
  PetscInt        numRanks, numPoints, r, p, a, adjSize;
  PetscInt       *adj = NULL;

  PetscFunctionBegin;
  CHKERRQ(DMLabelGetValueIS(label, &rankIS));
  CHKERRQ(ISGetLocalSize(rankIS, &numRanks));
  CHKERRQ(ISGetIndices(rankIS, &ranks));
  for (r = 0; r < numRanks; ++r) {
    const PetscInt rank = ranks[r];

    CHKERRQ(DMLabelGetStratumIS(label, rank, &pointIS));
    CHKERRQ(ISGetLocalSize(pointIS, &numPoints));
    CHKERRQ(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      adjSize = PETSC_DETERMINE;
      CHKERRQ(DMPlexGetAdjacency(dm, points[p], &adjSize, &adj));
      for (a = 0; a < adjSize; ++a) CHKERRQ(DMLabelSetValue(label, adj[a], rank));
    }
    CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
  }
  CHKERRQ(ISRestoreIndices(rankIS, &ranks));
  CHKERRQ(ISDestroy(&rankIS));
  CHKERRQ(PetscFree(adj));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelPropagate - Propagate points in a partition label over the point SF

  Input Parameters:
+ dm     - The DM
- label  - DMLabel assigning ranks to remote roots

  Level: developer

  Note: This is required when generating multi-level overlaps to capture
  overlap points from non-neighbouring partitions.

.seealso: DMPlexPartitionLabelCreateSF(), DMPlexDistribute(), DMPlexCreateOverlap()
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
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMGetPointSF(dm, &sfPoint));
  /* Pull point contributions from remote leaves into local roots */
  CHKERRQ(DMLabelGather(label, sfPoint, &lblLeaves));
  CHKERRQ(DMLabelGetValueIS(lblLeaves, &rankIS));
  CHKERRQ(ISGetLocalSize(rankIS, &numRanks));
  CHKERRQ(ISGetIndices(rankIS, &ranks));
  for (r = 0; r < numRanks; ++r) {
    const PetscInt remoteRank = ranks[r];
    if (remoteRank == rank) continue;
    CHKERRQ(DMLabelGetStratumIS(lblLeaves, remoteRank, &pointIS));
    CHKERRQ(DMLabelInsertIS(label, pointIS, remoteRank));
    CHKERRQ(ISDestroy(&pointIS));
  }
  CHKERRQ(ISRestoreIndices(rankIS, &ranks));
  CHKERRQ(ISDestroy(&rankIS));
  CHKERRQ(DMLabelDestroy(&lblLeaves));
  /* Push point contributions from roots into remote leaves */
  CHKERRQ(DMLabelDistribute(label, sfPoint, &lblRoots));
  CHKERRQ(DMLabelGetValueIS(lblRoots, &rankIS));
  CHKERRQ(ISGetLocalSize(rankIS, &numRanks));
  CHKERRQ(ISGetIndices(rankIS, &ranks));
  for (r = 0; r < numRanks; ++r) {
    const PetscInt remoteRank = ranks[r];
    if (remoteRank == rank) continue;
    CHKERRQ(DMLabelGetStratumIS(lblRoots, remoteRank, &pointIS));
    CHKERRQ(DMLabelInsertIS(label, pointIS, remoteRank));
    CHKERRQ(ISDestroy(&pointIS));
  }
  CHKERRQ(ISRestoreIndices(rankIS, &ranks));
  CHKERRQ(ISDestroy(&rankIS));
  CHKERRQ(DMLabelDestroy(&lblRoots));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelInvert - Create a partition label of remote roots from a local root label

  Input Parameters:
+ dm        - The DM
. rootLabel - DMLabel assigning ranks to local roots
- processSF - A star forest mapping into the local index on each remote rank

  Output Parameter:
. leafLabel - DMLabel assigning ranks to remote roots

  Note: The rootLabel defines a send pattern by mapping local points to remote target ranks. The
  resulting leafLabel is a receiver mapping of remote roots to their parent rank.

  Level: developer

.seealso: DMPlexPartitionLabelCreateSF(), DMPlexDistribute(), DMPlexCreateOverlap()
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
  CHKERRQ(PetscLogEventBegin(DMPLEX_PartLabelInvert,dm,0,0,0));
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(DMGetPointSF(dm, &sfPoint));

  /* Convert to (point, rank) and use actual owners */
  CHKERRQ(PetscSectionCreate(comm, &rootSection));
  CHKERRQ(PetscSectionSetChart(rootSection, 0, size));
  CHKERRQ(DMLabelGetValueIS(rootLabel, &valueIS));
  CHKERRQ(ISGetLocalSize(valueIS, &numNeighbors));
  CHKERRQ(ISGetIndices(valueIS, &neighbors));
  for (n = 0; n < numNeighbors; ++n) {
    CHKERRQ(DMLabelGetStratumSize(rootLabel, neighbors[n], &numPoints));
    CHKERRQ(PetscSectionAddDof(rootSection, neighbors[n], numPoints));
  }
  CHKERRQ(PetscSectionSetUp(rootSection));
  CHKERRQ(PetscSectionGetStorageSize(rootSection, &rootSize));
  CHKERRQ(PetscMalloc1(rootSize, &rootPoints));
  CHKERRQ(PetscSFGetGraph(sfPoint, NULL, &nleaves, &local, &remote));
  for (n = 0; n < numNeighbors; ++n) {
    IS              pointIS;
    const PetscInt *points;

    CHKERRQ(PetscSectionGetOffset(rootSection, neighbors[n], &off));
    CHKERRQ(DMLabelGetStratumIS(rootLabel, neighbors[n], &pointIS));
    CHKERRQ(ISGetLocalSize(pointIS, &numPoints));
    CHKERRQ(ISGetIndices(pointIS, &points));
    for (p = 0; p < numPoints; ++p) {
      if (local) CHKERRQ(PetscFindInt(points[p], nleaves, local, &l));
      else       {l = -1;}
      if (l >= 0) {rootPoints[off+p] = remote[l];}
      else        {rootPoints[off+p].index = points[p]; rootPoints[off+p].rank = rank;}
    }
    CHKERRQ(ISRestoreIndices(pointIS, &points));
    CHKERRQ(ISDestroy(&pointIS));
  }

  /* Try to communicate overlap using All-to-All */
  if (!processSF) {
    PetscInt64  counter = 0;
    PetscBool   locOverflow = PETSC_FALSE;
    PetscMPIInt *scounts, *sdispls, *rcounts, *rdispls;

    CHKERRQ(PetscCalloc4(size, &scounts, size, &sdispls, size, &rcounts, size, &rdispls));
    for (n = 0; n < numNeighbors; ++n) {
      CHKERRQ(PetscSectionGetDof(rootSection, neighbors[n], &dof));
      CHKERRQ(PetscSectionGetOffset(rootSection, neighbors[n], &off));
#if defined(PETSC_USE_64BIT_INDICES)
      if (dof > PETSC_MPI_INT_MAX) {locOverflow = PETSC_TRUE; break;}
      if (off > PETSC_MPI_INT_MAX) {locOverflow = PETSC_TRUE; break;}
#endif
      scounts[neighbors[n]] = (PetscMPIInt) dof;
      sdispls[neighbors[n]] = (PetscMPIInt) off;
    }
    CHKERRMPI(MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, comm));
    for (r = 0; r < size; ++r) { rdispls[r] = (int)counter; counter += rcounts[r]; }
    if (counter > PETSC_MPI_INT_MAX) locOverflow = PETSC_TRUE;
    CHKERRMPI(MPI_Allreduce(&locOverflow, &mpiOverflow, 1, MPIU_BOOL, MPI_LOR, comm));
    if (!mpiOverflow) {
      CHKERRQ(PetscInfo(dm,"Using Alltoallv for mesh distribution\n"));
      leafSize = (PetscInt) counter;
      CHKERRQ(PetscMalloc1(leafSize, &leafPoints));
      CHKERRMPI(MPI_Alltoallv(rootPoints, scounts, sdispls, MPIU_2INT, leafPoints, rcounts, rdispls, MPIU_2INT, comm));
    }
    CHKERRQ(PetscFree4(scounts, sdispls, rcounts, rdispls));
  }

  /* Communicate overlap using process star forest */
  if (processSF || mpiOverflow) {
    PetscSF      procSF;
    PetscSection leafSection;

    if (processSF) {
      CHKERRQ(PetscInfo(dm,"Using processSF for mesh distribution\n"));
      CHKERRQ(PetscObjectReference((PetscObject)processSF));
      procSF = processSF;
    } else {
      CHKERRQ(PetscInfo(dm,"Using processSF for mesh distribution (MPI overflow)\n"));
      CHKERRQ(PetscSFCreate(comm,&procSF));
      CHKERRQ(PetscSFSetGraphWithPattern(procSF,NULL,PETSCSF_PATTERN_ALLTOALL));
    }

    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)dm), &leafSection));
    CHKERRQ(DMPlexDistributeData(dm, procSF, rootSection, MPIU_2INT, rootPoints, leafSection, (void**) &leafPoints));
    CHKERRQ(PetscSectionGetStorageSize(leafSection, &leafSize));
    CHKERRQ(PetscSectionDestroy(&leafSection));
    CHKERRQ(PetscSFDestroy(&procSF));
  }

  for (p = 0; p < leafSize; p++) {
    CHKERRQ(DMLabelSetValue(leafLabel, leafPoints[p].index, leafPoints[p].rank));
  }

  CHKERRQ(ISRestoreIndices(valueIS, &neighbors));
  CHKERRQ(ISDestroy(&valueIS));
  CHKERRQ(PetscSectionDestroy(&rootSection));
  CHKERRQ(PetscFree(rootPoints));
  CHKERRQ(PetscFree(leafPoints));
  CHKERRQ(PetscLogEventEnd(DMPLEX_PartLabelInvert,dm,0,0,0));
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelCreateSF - Create a star forest from a label that assigns ranks to points

  Input Parameters:
+ dm    - The DM
- label - DMLabel assigning ranks to remote roots

  Output Parameter:
. sf    - The star forest communication context encapsulating the defined mapping

  Note: The incoming label is a receiver mapping of remote points to their parent rank.

  Level: developer

.seealso: DMPlexDistribute(), DMPlexCreateOverlap()
@*/
PetscErrorCode DMPlexPartitionLabelCreateSF(DM dm, DMLabel label, PetscSF *sf)
{
  PetscMPIInt     rank;
  PetscInt        n, numRemote, p, numPoints, pStart, pEnd, idx = 0, nNeighbors;
  PetscSFNode    *remotePoints;
  IS              remoteRootIS, neighborsIS;
  const PetscInt *remoteRoots, *neighbors;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(DMPLEX_PartLabelCreateSF,dm,0,0,0));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank));

  CHKERRQ(DMLabelGetValueIS(label, &neighborsIS));
#if 0
  {
    IS is;
    CHKERRQ(ISDuplicate(neighborsIS, &is));
    CHKERRQ(ISSort(is));
    CHKERRQ(ISDestroy(&neighborsIS));
    neighborsIS = is;
  }
#endif
  CHKERRQ(ISGetLocalSize(neighborsIS, &nNeighbors));
  CHKERRQ(ISGetIndices(neighborsIS, &neighbors));
  for (numRemote = 0, n = 0; n < nNeighbors; ++n) {
    CHKERRQ(DMLabelGetStratumSize(label, neighbors[n], &numPoints));
    numRemote += numPoints;
  }
  CHKERRQ(PetscMalloc1(numRemote, &remotePoints));
  /* Put owned points first */
  CHKERRQ(DMLabelGetStratumSize(label, rank, &numPoints));
  if (numPoints > 0) {
    CHKERRQ(DMLabelGetStratumIS(label, rank, &remoteRootIS));
    CHKERRQ(ISGetIndices(remoteRootIS, &remoteRoots));
    for (p = 0; p < numPoints; p++) {
      remotePoints[idx].index = remoteRoots[p];
      remotePoints[idx].rank = rank;
      idx++;
    }
    CHKERRQ(ISRestoreIndices(remoteRootIS, &remoteRoots));
    CHKERRQ(ISDestroy(&remoteRootIS));
  }
  /* Now add remote points */
  for (n = 0; n < nNeighbors; ++n) {
    const PetscInt nn = neighbors[n];

    CHKERRQ(DMLabelGetStratumSize(label, nn, &numPoints));
    if (nn == rank || numPoints <= 0) continue;
    CHKERRQ(DMLabelGetStratumIS(label, nn, &remoteRootIS));
    CHKERRQ(ISGetIndices(remoteRootIS, &remoteRoots));
    for (p = 0; p < numPoints; p++) {
      remotePoints[idx].index = remoteRoots[p];
      remotePoints[idx].rank = nn;
      idx++;
    }
    CHKERRQ(ISRestoreIndices(remoteRootIS, &remoteRoots));
    CHKERRQ(ISDestroy(&remoteRootIS));
  }
  CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject) dm), sf));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(PetscSFSetGraph(*sf, pEnd-pStart, numRemote, NULL, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetUp(*sf));
  CHKERRQ(ISDestroy(&neighborsIS));
  CHKERRQ(PetscLogEventEnd(DMPLEX_PartLabelCreateSF,dm,0,0,0));
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

  DMPlexRewriteSF - Rewrites the ownership of the SF of a DM (in place).

  Input parameters:
+ dm                - The DMPlex object.
. n                 - The number of points.
. pointsToRewrite   - The points in the SF whose ownership will change.
. targetOwners      - New owner for each element in pointsToRewrite.
- degrees           - Degrees of the points in the SF as obtained by PetscSFComputeDegreeBegin/PetscSFComputeDegreeEnd.

  Level: developer

@*/
static PetscErrorCode DMPlexRewriteSF(DM dm, PetscInt n, PetscInt *pointsToRewrite, PetscInt *targetOwners, const PetscInt *degrees)
{
  PetscInt      pStart, pEnd, i, j, counter, leafCounter, sumDegrees, nroots, nleafs;
  PetscInt     *cumSumDegrees, *newOwners, *newNumbers, *rankOnLeafs, *locationsOfLeafs, *remoteLocalPointOfLeafs, *points, *leafsNew;
  PetscSFNode  *leafLocationsNew;
  const         PetscSFNode *iremote;
  const         PetscInt *ilocal;
  PetscBool    *isLeaf;
  PetscSF       sf;
  MPI_Comm      comm;
  PetscMPIInt   rank, size;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));

  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(PetscSFGetGraph(sf, &nroots, &nleafs, &ilocal, &iremote));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &isLeaf));
  for (i=0; i<pEnd-pStart; i++) {
    isLeaf[i] = PETSC_FALSE;
  }
  for (i=0; i<nleafs; i++) {
    isLeaf[ilocal[i]-pStart] = PETSC_TRUE;
  }

  CHKERRQ(PetscMalloc1(pEnd-pStart+1, &cumSumDegrees));
  cumSumDegrees[0] = 0;
  for (i=1; i<=pEnd-pStart; i++) {
    cumSumDegrees[i] = cumSumDegrees[i-1] + degrees[i-1];
  }
  sumDegrees = cumSumDegrees[pEnd-pStart];
  /* get the location of my leafs (we have sumDegrees many leafs pointing at our roots) */

  CHKERRQ(PetscMalloc1(sumDegrees, &locationsOfLeafs));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &rankOnLeafs));
  for (i=0; i<pEnd-pStart; i++) {
    rankOnLeafs[i] = rank;
  }
  CHKERRQ(PetscSFGatherBegin(sf, MPIU_INT, rankOnLeafs, locationsOfLeafs));
  CHKERRQ(PetscSFGatherEnd(sf, MPIU_INT, rankOnLeafs, locationsOfLeafs));
  CHKERRQ(PetscFree(rankOnLeafs));

  /* get the remote local points of my leaves */
  CHKERRQ(PetscMalloc1(sumDegrees, &remoteLocalPointOfLeafs));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &points));
  for (i=0; i<pEnd-pStart; i++) {
    points[i] = pStart+i;
  }
  CHKERRQ(PetscSFGatherBegin(sf, MPIU_INT, points, remoteLocalPointOfLeafs));
  CHKERRQ(PetscSFGatherEnd(sf, MPIU_INT, points, remoteLocalPointOfLeafs));
  CHKERRQ(PetscFree(points));
  /* Figure out the new owners of the vertices that are up for grabs and their numbers on the new owners */
  CHKERRQ(PetscMalloc1(pEnd-pStart, &newOwners));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &newNumbers));
  for (i=0; i<pEnd-pStart; i++) {
    newOwners[i] = -1;
    newNumbers[i] = -1;
  }
  {
    PetscInt oldNumber, newNumber, oldOwner, newOwner;
    for (i=0; i<n; i++) {
      oldNumber = pointsToRewrite[i];
      newNumber = -1;
      oldOwner = rank;
      newOwner = targetOwners[i];
      if (oldOwner == newOwner) {
        newNumber = oldNumber;
      } else {
        for (j=0; j<degrees[oldNumber]; j++) {
          if (locationsOfLeafs[cumSumDegrees[oldNumber]+j] == newOwner) {
            newNumber = remoteLocalPointOfLeafs[cumSumDegrees[oldNumber]+j];
            break;
          }
        }
      }
      PetscCheck(newNumber != -1,PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Couldn't find the new owner of vertex.");

      newOwners[oldNumber] = newOwner;
      newNumbers[oldNumber] = newNumber;
    }
  }
  CHKERRQ(PetscFree(cumSumDegrees));
  CHKERRQ(PetscFree(locationsOfLeafs));
  CHKERRQ(PetscFree(remoteLocalPointOfLeafs));

  CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, newOwners, newOwners,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, newOwners, newOwners,MPI_REPLACE));
  CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, newNumbers, newNumbers,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, newNumbers, newNumbers,MPI_REPLACE));

  /* Now count how many leafs we have on each processor. */
  leafCounter=0;
  for (i=0; i<pEnd-pStart; i++) {
    if (newOwners[i] >= 0) {
      if (newOwners[i] != rank) {
        leafCounter++;
      }
    } else {
      if (isLeaf[i]) {
        leafCounter++;
      }
    }
  }

  /* Now set up the new sf by creating the leaf arrays */
  CHKERRQ(PetscMalloc1(leafCounter, &leafsNew));
  CHKERRQ(PetscMalloc1(leafCounter, &leafLocationsNew));

  leafCounter = 0;
  counter = 0;
  for (i=0; i<pEnd-pStart; i++) {
    if (newOwners[i] >= 0) {
      if (newOwners[i] != rank) {
        leafsNew[leafCounter] = i;
        leafLocationsNew[leafCounter].rank = newOwners[i];
        leafLocationsNew[leafCounter].index = newNumbers[i];
        leafCounter++;
      }
    } else {
      if (isLeaf[i]) {
        leafsNew[leafCounter] = i;
        leafLocationsNew[leafCounter].rank = iremote[counter].rank;
        leafLocationsNew[leafCounter].index = iremote[counter].index;
        leafCounter++;
      }
    }
    if (isLeaf[i]) {
      counter++;
    }
  }

  CHKERRQ(PetscSFSetGraph(sf, nroots, leafCounter, leafsNew, PETSC_OWN_POINTER, leafLocationsNew, PETSC_OWN_POINTER));
  CHKERRQ(PetscFree(newOwners));
  CHKERRQ(PetscFree(newNumbers));
  CHKERRQ(PetscFree(isLeaf));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexViewDistribution(MPI_Comm comm, PetscInt n, PetscInt skip, PetscInt *vtxwgt, PetscInt *part, PetscViewer viewer)
{
  PetscInt    *distribution, min, max, sum;
  PetscMPIInt rank, size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscCalloc1(size, &distribution));
  for (PetscInt i=0; i<n; i++) {
    if (part) distribution[part[i]] += vtxwgt[skip*i];
    else distribution[rank] += vtxwgt[skip*i];
  }
  CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, distribution, size, MPIU_INT, MPI_SUM, comm));
  min = distribution[0];
  max = distribution[0];
  sum = distribution[0];
  for (PetscInt i=1; i<size; i++) {
    if (distribution[i]<min) min=distribution[i];
    if (distribution[i]>max) max=distribution[i];
    sum += distribution[i];
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Min: %D, Avg: %D, Max: %D, Balance: %f\n", min, sum/size, max, (max*1.*size)/sum));
  CHKERRQ(PetscFree(distribution));
  PetscFunctionReturn(0);
}

#endif

/*@
  DMPlexRebalanceSharedPoints - Redistribute points in the plex that are shared in order to achieve better balancing. This routine updates the PointSF of the DM inplace.

  Input parameters:
+ dm               - The DMPlex object.
. entityDepth      - depth of the entity to balance (0 -> balance vertices).
. useInitialGuess  - whether to use the current distribution as initial guess (only used by ParMETIS).
- parallel         - whether to use ParMETIS and do the partition in parallel or whether to gather the graph onto a single process and use METIS.

  Output parameters:
. success          - whether the graph partitioning was successful or not. If not, try useInitialGuess=True and parallel=True.

  Level: intermediate

@*/

PetscErrorCode DMPlexRebalanceSharedPoints(DM dm, PetscInt entityDepth, PetscBool useInitialGuess, PetscBool parallel, PetscBool *success)
{
#if defined(PETSC_HAVE_PARMETIS)
  PetscSF     sf;
  PetscInt    ierr, i, j, idx, jdx;
  PetscInt    eBegin, eEnd, nroots, nleafs, pStart, pEnd;
  const       PetscInt *degrees, *ilocal;
  const       PetscSFNode *iremote;
  PetscBool   *toBalance, *isLeaf, *isExclusivelyOwned, *isNonExclusivelyOwned;
  PetscInt    numExclusivelyOwned, numNonExclusivelyOwned;
  PetscMPIInt rank, size;
  PetscInt    *globalNumbersOfLocalOwnedVertices, *leafGlobalNumbers;
  const       PetscInt *cumSumVertices;
  PetscInt    offset, counter;
  PetscInt    lenadjncy;
  PetscInt    *xadj, *adjncy, *vtxwgt;
  PetscInt    lenxadj;
  PetscInt    *adjwgt = NULL;
  PetscInt    *part, *options;
  PetscInt    nparts, wgtflag, numflag, ncon, edgecut;
  real_t      *ubvec;
  PetscInt    *firstVertices, *renumbering;
  PetscInt    failed, failedGlobal;
  MPI_Comm    comm;
  Mat         A, Apre;
  const char *prefix = NULL;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscLayout layout;

  PetscFunctionBegin;
  if (success) *success = PETSC_FALSE;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  if (size==1) PetscFunctionReturn(0);

  CHKERRQ(PetscLogEventBegin(DMPLEX_RebalanceSharedPoints, dm, 0, 0, 0));

  CHKERRQ(PetscOptionsGetViewer(comm,((PetscObject)dm)->options, prefix,"-dm_rebalance_partition_view",&viewer,&format,NULL));
  if (viewer) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
  }

  /* Figure out all points in the plex that we are interested in balancing. */
  CHKERRQ(DMPlexGetDepthStratum(dm, entityDepth, &eBegin, &eEnd));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &toBalance));

  for (i=0; i<pEnd-pStart; i++) {
    toBalance[i] = (PetscBool)(i-pStart>=eBegin && i-pStart<eEnd);
  }

  /* There are three types of points:
   * exclusivelyOwned: points that are owned by this process and only seen by this process
   * nonExclusivelyOwned: points that are owned by this process but seen by at least another process
   * leaf: a point that is seen by this process but owned by a different process
   */
  CHKERRQ(DMGetPointSF(dm, &sf));
  CHKERRQ(PetscSFGetGraph(sf, &nroots, &nleafs, &ilocal, &iremote));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &isLeaf));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &isNonExclusivelyOwned));
  CHKERRQ(PetscMalloc1(pEnd-pStart, &isExclusivelyOwned));
  for (i=0; i<pEnd-pStart; i++) {
    isNonExclusivelyOwned[i] = PETSC_FALSE;
    isExclusivelyOwned[i] = PETSC_FALSE;
    isLeaf[i] = PETSC_FALSE;
  }

  /* start by marking all the leafs */
  for (i=0; i<nleafs; i++) {
    isLeaf[ilocal[i]-pStart] = PETSC_TRUE;
  }

  /* for an owned point, we can figure out whether another processor sees it or
   * not by calculating its degree */
  CHKERRQ(PetscSFComputeDegreeBegin(sf, &degrees));
  CHKERRQ(PetscSFComputeDegreeEnd(sf, &degrees));

  numExclusivelyOwned = 0;
  numNonExclusivelyOwned = 0;
  for (i=0; i<pEnd-pStart; i++) {
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

  /* We are going to build a graph with one vertex per core representing the
   * exclusively owned points and then one vertex per nonExclusively owned
   * point. */

  CHKERRQ(PetscLayoutCreate(comm, &layout));
  CHKERRQ(PetscLayoutSetLocalSize(layout, 1 + numNonExclusivelyOwned));
  CHKERRQ(PetscLayoutSetUp(layout));
  CHKERRQ(PetscLayoutGetRanges(layout, &cumSumVertices));

  CHKERRQ(PetscMalloc1(pEnd-pStart, &globalNumbersOfLocalOwnedVertices));
  for (i=0; i<pEnd-pStart; i++) {globalNumbersOfLocalOwnedVertices[i] = pStart - 1;}
  offset = cumSumVertices[rank];
  counter = 0;
  for (i=0; i<pEnd-pStart; i++) {
    if (toBalance[i]) {
      if (degrees[i] > 0) {
        globalNumbersOfLocalOwnedVertices[i] = counter + 1 + offset;
        counter++;
      }
    }
  }

  /* send the global numbers of vertices I own to the leafs so that they know to connect to it */
  CHKERRQ(PetscMalloc1(pEnd-pStart, &leafGlobalNumbers));
  CHKERRQ(PetscSFBcastBegin(sf, MPIU_INT, globalNumbersOfLocalOwnedVertices, leafGlobalNumbers,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sf, MPIU_INT, globalNumbersOfLocalOwnedVertices, leafGlobalNumbers,MPI_REPLACE));

  /* Now start building the data structure for ParMETIS */

  CHKERRQ(MatCreate(comm, &Apre));
  CHKERRQ(MatSetType(Apre, MATPREALLOCATOR));
  CHKERRQ(MatSetSizes(Apre, 1+numNonExclusivelyOwned, 1+numNonExclusivelyOwned, cumSumVertices[size], cumSumVertices[size]));
  CHKERRQ(MatSetUp(Apre));

  for (i=0; i<pEnd-pStart; i++) {
    if (toBalance[i]) {
      idx = cumSumVertices[rank];
      if (isNonExclusivelyOwned[i]) jdx = globalNumbersOfLocalOwnedVertices[i];
      else if (isLeaf[i]) jdx = leafGlobalNumbers[i];
      else continue;
      CHKERRQ(MatSetValue(Apre, idx, jdx, 1, INSERT_VALUES));
      CHKERRQ(MatSetValue(Apre, jdx, idx, 1, INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(Apre, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Apre, MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(comm, &A));
  CHKERRQ(MatSetType(A, MATMPIAIJ));
  CHKERRQ(MatSetSizes(A, 1+numNonExclusivelyOwned, 1+numNonExclusivelyOwned, cumSumVertices[size], cumSumVertices[size]));
  CHKERRQ(MatPreallocatorPreallocate(Apre, PETSC_FALSE, A));
  CHKERRQ(MatDestroy(&Apre));

  for (i=0; i<pEnd-pStart; i++) {
    if (toBalance[i]) {
      idx = cumSumVertices[rank];
      if (isNonExclusivelyOwned[i]) jdx = globalNumbersOfLocalOwnedVertices[i];
      else if (isLeaf[i]) jdx = leafGlobalNumbers[i];
      else continue;
      CHKERRQ(MatSetValue(A, idx, jdx, 1, INSERT_VALUES));
      CHKERRQ(MatSetValue(A, jdx, idx, 1, INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscFree(leafGlobalNumbers));
  CHKERRQ(PetscFree(globalNumbersOfLocalOwnedVertices));

  nparts = size;
  wgtflag = 2;
  numflag = 0;
  ncon = 2;
  real_t *tpwgts;
  CHKERRQ(PetscMalloc1(ncon * nparts, &tpwgts));
  for (i=0; i<ncon*nparts; i++) {
    tpwgts[i] = 1./(nparts);
  }

  CHKERRQ(PetscMalloc1(ncon, &ubvec));
  ubvec[0] = 1.01;
  ubvec[1] = 1.01;
  lenadjncy = 0;
  for (i=0; i<1+numNonExclusivelyOwned; i++) {
    PetscInt temp=0;
    CHKERRQ(MatGetRow(A, cumSumVertices[rank] + i, &temp, NULL, NULL));
    lenadjncy += temp;
    CHKERRQ(MatRestoreRow(A, cumSumVertices[rank] + i, &temp, NULL, NULL));
  }
  CHKERRQ(PetscMalloc1(lenadjncy, &adjncy));
  lenxadj = 2 + numNonExclusivelyOwned;
  CHKERRQ(PetscMalloc1(lenxadj, &xadj));
  xadj[0] = 0;
  counter = 0;
  for (i=0; i<1+numNonExclusivelyOwned; i++) {
    PetscInt        temp=0;
    const PetscInt *cols;
    CHKERRQ(MatGetRow(A, cumSumVertices[rank] + i, &temp, &cols, NULL));
    CHKERRQ(PetscArraycpy(&adjncy[counter], cols, temp));
    counter += temp;
    xadj[i+1] = counter;
    CHKERRQ(MatRestoreRow(A, cumSumVertices[rank] + i, &temp, &cols, NULL));
  }

  CHKERRQ(PetscMalloc1(cumSumVertices[rank+1]-cumSumVertices[rank], &part));
  CHKERRQ(PetscMalloc1(ncon*(1 + numNonExclusivelyOwned), &vtxwgt));
  vtxwgt[0] = numExclusivelyOwned;
  if (ncon>1) vtxwgt[1] = 1;
  for (i=0; i<numNonExclusivelyOwned; i++) {
    vtxwgt[ncon*(i+1)] = 1;
    if (ncon>1) vtxwgt[ncon*(i+1)+1] = 0;
  }

  if (viewer) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Attempt rebalancing of shared points of depth %D on interface of mesh distribution.\n", entityDepth));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Size of generated auxiliary graph: %D\n", cumSumVertices[size]));
  }
  if (parallel) {
    CHKERRQ(PetscMalloc1(4, &options));
    options[0] = 1;
    options[1] = 0; /* Verbosity */
    options[2] = 0; /* Seed */
    options[3] = PARMETIS_PSR_COUPLED; /* Seed */
    if (viewer) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Using ParMETIS to partition graph.\n"));
    if (useInitialGuess) {
      if (viewer) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Using current distribution of points as initial guess.\n"));
      PetscStackPush("ParMETIS_V3_RefineKway");
      ierr = ParMETIS_V3_RefineKway((PetscInt*)cumSumVertices, xadj, adjncy, vtxwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);
      PetscCheck(ierr == METIS_OK,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in ParMETIS_V3_RefineKway()");
      PetscStackPop;
    } else {
      PetscStackPush("ParMETIS_V3_PartKway");
      ierr = ParMETIS_V3_PartKway((PetscInt*)cumSumVertices, xadj, adjncy, vtxwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);
      PetscStackPop;
      PetscCheck(ierr == METIS_OK,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in ParMETIS_V3_PartKway()");
    }
    CHKERRQ(PetscFree(options));
  } else {
    if (viewer) CHKERRQ(PetscViewerASCIIPrintf(viewer, "Using METIS to partition graph.\n"));
    Mat As;
    PetscInt numRows;
    PetscInt *partGlobal;
    CHKERRQ(MatCreateRedundantMatrix(A, size, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &As));

    PetscInt *numExclusivelyOwnedAll;
    CHKERRQ(PetscMalloc1(size, &numExclusivelyOwnedAll));
    numExclusivelyOwnedAll[rank] = numExclusivelyOwned;
    CHKERRMPI(MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,numExclusivelyOwnedAll,1,MPIU_INT,comm));

    CHKERRQ(MatGetSize(As, &numRows, NULL));
    CHKERRQ(PetscMalloc1(numRows, &partGlobal));
    if (rank == 0) {
      PetscInt *adjncy_g, *xadj_g, *vtxwgt_g;
      lenadjncy = 0;

      for (i=0; i<numRows; i++) {
        PetscInt temp=0;
        CHKERRQ(MatGetRow(As, i, &temp, NULL, NULL));
        lenadjncy += temp;
        CHKERRQ(MatRestoreRow(As, i, &temp, NULL, NULL));
      }
      CHKERRQ(PetscMalloc1(lenadjncy, &adjncy_g));
      lenxadj = 1 + numRows;
      CHKERRQ(PetscMalloc1(lenxadj, &xadj_g));
      xadj_g[0] = 0;
      counter = 0;
      for (i=0; i<numRows; i++) {
        PetscInt        temp=0;
        const PetscInt *cols;
        CHKERRQ(MatGetRow(As, i, &temp, &cols, NULL));
        CHKERRQ(PetscArraycpy(&adjncy_g[counter], cols, temp));
        counter += temp;
        xadj_g[i+1] = counter;
        CHKERRQ(MatRestoreRow(As, i, &temp, &cols, NULL));
      }
      CHKERRQ(PetscMalloc1(2*numRows, &vtxwgt_g));
      for (i=0; i<size; i++) {
        vtxwgt_g[ncon*cumSumVertices[i]] = numExclusivelyOwnedAll[i];
        if (ncon>1) vtxwgt_g[ncon*cumSumVertices[i]+1] = 1;
        for (j=cumSumVertices[i]+1; j<cumSumVertices[i+1]; j++) {
          vtxwgt_g[ncon*j] = 1;
          if (ncon>1) vtxwgt_g[2*j+1] = 0;
        }
      }
      CHKERRQ(PetscMalloc1(64, &options));
      ierr = METIS_SetDefaultOptions(options); /* initialize all defaults */
      PetscCheck(ierr == METIS_OK,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_SetDefaultOptions()");
      options[METIS_OPTION_CONTIG] = 1;
      PetscStackPush("METIS_PartGraphKway");
      ierr = METIS_PartGraphKway(&numRows, &ncon, xadj_g, adjncy_g, vtxwgt_g, NULL, NULL, &nparts, tpwgts, ubvec, options, &edgecut, partGlobal);
      PetscStackPop;
      PetscCheck(ierr == METIS_OK,PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_PartGraphKway()");
      CHKERRQ(PetscFree(options));
      CHKERRQ(PetscFree(xadj_g));
      CHKERRQ(PetscFree(adjncy_g));
      CHKERRQ(PetscFree(vtxwgt_g));
    }
    CHKERRQ(PetscFree(numExclusivelyOwnedAll));

    /* Now scatter the parts array. */
    {
      PetscMPIInt *counts, *mpiCumSumVertices;
      CHKERRQ(PetscMalloc1(size, &counts));
      CHKERRQ(PetscMalloc1(size+1, &mpiCumSumVertices));
      for (i=0; i<size; i++) {
        CHKERRQ(PetscMPIIntCast(cumSumVertices[i+1] - cumSumVertices[i], &(counts[i])));
      }
      for (i=0; i<=size; i++) {
        CHKERRQ(PetscMPIIntCast(cumSumVertices[i], &(mpiCumSumVertices[i])));
      }
      CHKERRMPI(MPI_Scatterv(partGlobal, counts, mpiCumSumVertices, MPIU_INT, part, counts[rank], MPIU_INT, 0, comm));
      CHKERRQ(PetscFree(counts));
      CHKERRQ(PetscFree(mpiCumSumVertices));
    }

    CHKERRQ(PetscFree(partGlobal));
    CHKERRQ(MatDestroy(&As));
  }

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFree(ubvec));
  CHKERRQ(PetscFree(tpwgts));

  /* Now rename the result so that the vertex resembling the exclusively owned points stays on the same rank */

  CHKERRQ(PetscMalloc1(size, &firstVertices));
  CHKERRQ(PetscMalloc1(size, &renumbering));
  firstVertices[rank] = part[0];
  CHKERRMPI(MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,firstVertices,1,MPIU_INT,comm));
  for (i=0; i<size; i++) {
    renumbering[firstVertices[i]] = i;
  }
  for (i=0; i<cumSumVertices[rank+1]-cumSumVertices[rank]; i++) {
    part[i] = renumbering[part[i]];
  }
  /* Check if the renumbering worked (this can fail when ParMETIS gives fewer partitions than there are processes) */
  failed = (PetscInt)(part[0] != rank);
  CHKERRMPI(MPI_Allreduce(&failed, &failedGlobal, 1, MPIU_INT, MPI_SUM, comm));

  CHKERRQ(PetscFree(firstVertices));
  CHKERRQ(PetscFree(renumbering));

  if (failedGlobal > 0) {
    CHKERRQ(PetscLayoutDestroy(&layout));
    CHKERRQ(PetscFree(xadj));
    CHKERRQ(PetscFree(adjncy));
    CHKERRQ(PetscFree(vtxwgt));
    CHKERRQ(PetscFree(toBalance));
    CHKERRQ(PetscFree(isLeaf));
    CHKERRQ(PetscFree(isNonExclusivelyOwned));
    CHKERRQ(PetscFree(isExclusivelyOwned));
    CHKERRQ(PetscFree(part));
    if (viewer) {
      CHKERRQ(PetscViewerPopFormat(viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
    }
    CHKERRQ(PetscLogEventEnd(DMPLEX_RebalanceSharedPoints, dm, 0, 0, 0));
    PetscFunctionReturn(0);
  }

  /*Let's check how well we did distributing points*/
  if (viewer) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Comparing number of owned entities of depth %D on each process before rebalancing, after rebalancing, and after consistency checks.\n", entityDepth));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Initial.     "));
    CHKERRQ(DMPlexViewDistribution(comm, cumSumVertices[rank+1]-cumSumVertices[rank], ncon, vtxwgt, NULL, viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Rebalanced.  "));
    CHKERRQ(DMPlexViewDistribution(comm, cumSumVertices[rank+1]-cumSumVertices[rank], ncon, vtxwgt, part, viewer));
  }

  /* Now check that every vertex is owned by a process that it is actually connected to. */
  for (i=1; i<=numNonExclusivelyOwned; i++) {
    PetscInt loc = 0;
    CHKERRQ(PetscFindInt(cumSumVertices[part[i]], xadj[i+1]-xadj[i], &adjncy[xadj[i]], &loc));
    /* If not, then just set the owner to the original owner (hopefully a rare event, it means that a vertex has been isolated) */
    if (loc<0) {
      part[i] = rank;
    }
  }

  /* Let's see how significant the influences of the previous fixing up step was.*/
  if (viewer) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "After.       "));
    CHKERRQ(DMPlexViewDistribution(comm, cumSumVertices[rank+1]-cumSumVertices[rank], ncon, vtxwgt, part, viewer));
  }

  CHKERRQ(PetscLayoutDestroy(&layout));
  CHKERRQ(PetscFree(xadj));
  CHKERRQ(PetscFree(adjncy));
  CHKERRQ(PetscFree(vtxwgt));

  /* Almost done, now rewrite the SF to reflect the new ownership. */
  {
    PetscInt *pointsToRewrite;
    CHKERRQ(PetscMalloc1(numNonExclusivelyOwned, &pointsToRewrite));
    counter = 0;
    for (i=0; i<pEnd-pStart; i++) {
      if (toBalance[i]) {
        if (isNonExclusivelyOwned[i]) {
          pointsToRewrite[counter] = i + pStart;
          counter++;
        }
      }
    }
    CHKERRQ(DMPlexRewriteSF(dm, numNonExclusivelyOwned, pointsToRewrite, part+1, degrees));
    CHKERRQ(PetscFree(pointsToRewrite));
  }

  CHKERRQ(PetscFree(toBalance));
  CHKERRQ(PetscFree(isLeaf));
  CHKERRQ(PetscFree(isNonExclusivelyOwned));
  CHKERRQ(PetscFree(isExclusivelyOwned));
  CHKERRQ(PetscFree(part));
  if (viewer) {
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
  if (success) *success = PETSC_TRUE;
  CHKERRQ(PetscLogEventEnd(DMPLEX_RebalanceSharedPoints, dm, 0, 0, 0));
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-parmetis.");
#endif
}
