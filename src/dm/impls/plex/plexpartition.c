#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/partitionerimpl.h>
#include <petsc/private/hashseti.h>

PETSC_STATIC_INLINE PetscInt DMPlex_GlobalID(PetscInt point) { return point >= 0 ? point : -(point+1); }

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (dim != depth) {
    /* We do not handle the uninterpolated case here */
    ierr = DMPlexCreateNeighborCSR(dm, height, numVertices, offsets, adjacency);CHKERRQ(ierr);
    /* DMPlexCreateNeighborCSR does not make a numbering */
    if (globalNumbering) {ierr = DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, globalNumbering);CHKERRQ(ierr);}
    /* Different behavior for empty graphs */
    if (!*numVertices) {
      ierr = PetscMalloc1(1, offsets);CHKERRQ(ierr);
      (*offsets)[0] = 0;
    }
    /* Broken in parallel */
    if (rank && *numVertices) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel partitioning of uninterpolated meshes not supported");
    PetscFunctionReturn(0);
  }
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, height, &pStart, &pEnd);CHKERRQ(ierr);
  /* Build adjacency graph via a section/segbuffer */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscInt),1000,&adjBuffer);CHKERRQ(ierr);
  /* Always use FVM adjacency to create partitioner graph */
  ierr = DMGetBasicAdjacency(dm, &useCone, &useClosure);CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMPlexCreateNumbering_Plex(dm, pStart, pEnd, 0, NULL, sfPoint, &cellNumbering);CHKERRQ(ierr);
  if (globalNumbering) {
    ierr = PetscObjectReference((PetscObject)cellNumbering);CHKERRQ(ierr);
    *globalNumbering = cellNumbering;
  }
  ierr = ISGetIndices(cellNumbering, &cellNum);CHKERRQ(ierr);
  /* For all boundary faces (including faces adjacent to a ghost cell), record the local cell in adjCells
     Broadcast adjCells to remoteCells (to get cells from roots) and Reduce adjCells to remoteCells (to get cells from leaves)
   */
  ierr = PetscSFGetGraph(sfPoint, &nroots, &nleaves, &local, NULL);CHKERRQ(ierr);
  if (nroots >= 0) {
    PetscInt fStart, fEnd, f;

    ierr = PetscCalloc2(nroots, &adjCells, nroots, &remoteCells);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, height+1, &fStart, &fEnd);CHKERRQ(ierr);
    for (l = 0; l < nroots; ++l) adjCells[l] = -3;
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *support;
      PetscInt        supportSize;

      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      if (supportSize == 1) adjCells[f] = DMPlex_GlobalID(cellNum[support[0]]);
      else if (supportSize == 2) {
        ierr = PetscFindInt(support[0], nleaves, local, &p);CHKERRQ(ierr);
        if (p >= 0) adjCells[f] = DMPlex_GlobalID(cellNum[support[1]]);
        ierr = PetscFindInt(support[1], nleaves, local, &p);CHKERRQ(ierr);
        if (p >= 0) adjCells[f] = DMPlex_GlobalID(cellNum[support[0]]);
      }
      /* Handle non-conforming meshes */
      if (supportSize > 2) {
        PetscInt        numChildren, i;
        const PetscInt *children;

        ierr = DMPlexGetTreeChildren(dm, f, &numChildren, &children);CHKERRQ(ierr);
        for (i = 0; i < numChildren; ++i) {
          const PetscInt child = children[i];
          if (fStart <= child && child < fEnd) {
            ierr = DMPlexGetSupport(dm, child, &support);CHKERRQ(ierr);
            ierr = DMPlexGetSupportSize(dm, child, &supportSize);CHKERRQ(ierr);
            if (supportSize == 1) adjCells[child] = DMPlex_GlobalID(cellNum[support[0]]);
            else if (supportSize == 2) {
              ierr = PetscFindInt(support[0], nleaves, local, &p);CHKERRQ(ierr);
              if (p >= 0) adjCells[child] = DMPlex_GlobalID(cellNum[support[1]]);
              ierr = PetscFindInt(support[1], nleaves, local, &p);CHKERRQ(ierr);
              if (p >= 0) adjCells[child] = DMPlex_GlobalID(cellNum[support[0]]);
            }
          }
        }
      }
    }
    for (l = 0; l < nroots; ++l) remoteCells[l] = -1;
    ierr = PetscSFBcastBegin(dm->sf, MPIU_INT, adjCells, remoteCells);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(dm->sf, MPIU_INT, adjCells, remoteCells);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(dm->sf, MPIU_INT, adjCells, remoteCells, MPI_MAX);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(dm->sf, MPIU_INT, adjCells, remoteCells, MPI_MAX);CHKERRQ(ierr);
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

      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        const PetscInt point = cone[c];
        if (remoteCells[point] >= 0 && remoteCells[point] != gp) {
          PetscInt *PETSC_RESTRICT pBuf;
          ierr = PetscSectionAddDof(section, p, 1);CHKERRQ(ierr);
          ierr = PetscSegBufferGetInts(adjBuffer, 1, &pBuf);CHKERRQ(ierr);
          *pBuf = remoteCells[point];
        }
        /* Handle non-conforming meshes */
        ierr = DMPlexGetTreeChildren(dm, point, &numChildren, &children);CHKERRQ(ierr);
        for (i = 0; i < numChildren; ++i) {
          const PetscInt child = children[i];
          if (remoteCells[child] >= 0 && remoteCells[child] != gp) {
            PetscInt *PETSC_RESTRICT pBuf;
            ierr = PetscSectionAddDof(section, p, 1);CHKERRQ(ierr);
            ierr = PetscSegBufferGetInts(adjBuffer, 1, &pBuf);CHKERRQ(ierr);
            *pBuf = remoteCells[child];
          }
        }
      }
    }
    /* Add local cells */
    adjSize = PETSC_DETERMINE;
    ierr = DMPlexGetAdjacency(dm, p, &adjSize, &adj);CHKERRQ(ierr);
    for (a = 0; a < adjSize; ++a) {
      const PetscInt point = adj[a];
      if (point != p && pStart <= point && point < pEnd) {
        PetscInt *PETSC_RESTRICT pBuf;
        ierr = PetscSectionAddDof(section, p, 1);CHKERRQ(ierr);
        ierr = PetscSegBufferGetInts(adjBuffer, 1, &pBuf);CHKERRQ(ierr);
        *pBuf = DMPlex_GlobalID(cellNum[point]);
      }
    }
    (*numVertices)++;
  }
  ierr = PetscFree(adj);CHKERRQ(ierr);
  ierr = PetscFree2(adjCells, remoteCells);CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm, useCone, useClosure);CHKERRQ(ierr);

  /* Derive CSR graph from section/segbuffer */
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &size);CHKERRQ(ierr);
  ierr = PetscMalloc1(*numVertices+1, &vOffsets);CHKERRQ(ierr);
  for (idx = 0, p = pStart; p < pEnd; p++) {
    if (nroots > 0) {if (cellNum[p] < 0) continue;}
    ierr = PetscSectionGetOffset(section, p, &(vOffsets[idx++]));CHKERRQ(ierr);
  }
  vOffsets[*numVertices] = size;
  ierr = PetscSegBufferExtractAlloc(adjBuffer, &graph);CHKERRQ(ierr);

  if (nroots >= 0) {
    /* Filter out duplicate edges using section/segbuffer */
    ierr = PetscSectionReset(section);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(section, 0, *numVertices);CHKERRQ(ierr);
    for (p = 0; p < *numVertices; p++) {
      PetscInt start = vOffsets[p], end = vOffsets[p+1];
      PetscInt numEdges = end-start, *PETSC_RESTRICT edges;
      ierr = PetscSortRemoveDupsInt(&numEdges, &graph[start]);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(section, p, numEdges);CHKERRQ(ierr);
      ierr = PetscSegBufferGetInts(adjBuffer, numEdges, &edges);CHKERRQ(ierr);
      ierr = PetscArraycpy(edges, &graph[start], numEdges);CHKERRQ(ierr);
    }
    ierr = PetscFree(vOffsets);CHKERRQ(ierr);
    ierr = PetscFree(graph);CHKERRQ(ierr);
    /* Derive CSR graph from section/segbuffer */
    ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(section, &size);CHKERRQ(ierr);
    ierr = PetscMalloc1(*numVertices+1, &vOffsets);CHKERRQ(ierr);
    for (idx = 0, p = 0; p < *numVertices; p++) {
      ierr = PetscSectionGetOffset(section, p, &(vOffsets[idx++]));CHKERRQ(ierr);
    }
    vOffsets[*numVertices] = size;
    ierr = PetscSegBufferExtractAlloc(adjBuffer, &graph);CHKERRQ(ierr);
  } else {
    /* Sort adjacencies (not strictly necessary) */
    for (p = 0; p < *numVertices; p++) {
      PetscInt start = vOffsets[p], end = vOffsets[p+1];
      ierr = PetscSortInt(end-start, &graph[start]);CHKERRQ(ierr);
    }
  }

  if (offsets) *offsets = vOffsets;
  if (adjacency) *adjacency = graph;

  /* Cleanup */
  ierr = PetscSegBufferDestroy(&adjBuffer);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = ISRestoreIndices(cellNumbering, &cellNum);CHKERRQ(ierr);
  ierr = ISDestroy(&cellNumbering);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (dim != depth) {
    /* We do not handle the uninterpolated case here */
    ierr = DMPlexCreateNeighborCSR(dm, height, numVertices, offsets, adjacency);CHKERRQ(ierr);
    /* DMPlexCreateNeighborCSR does not make a numbering */
    if (globalNumbering) {ierr = DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, globalNumbering);CHKERRQ(ierr);}
    /* Different behavior for empty graphs */
    if (!*numVertices) {
      ierr = PetscMalloc1(1, offsets);CHKERRQ(ierr);
      (*offsets)[0] = 0;
    }
    /* Broken in parallel */
    if (rank && *numVertices) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Parallel partitioning of uninterpolated meshes not supported");
    PetscFunctionReturn(0);
  }
  /* Interpolated and parallel case */

  /* numbering */
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, height, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, height+1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexCreateNumbering_Plex(dm, cStart, cEnd, 0, &N, sfPoint, &cis);CHKERRQ(ierr);
  ierr = DMPlexCreateNumbering_Plex(dm, fStart, fEnd, 0, &M, sfPoint, &fis);CHKERRQ(ierr);
  if (globalNumbering) {
    ierr = ISDuplicate(cis, globalNumbering);CHKERRQ(ierr);
  }

  /* get positive global ids and local sizes for facets and cells */
  ierr = ISGetLocalSize(fis, &m);CHKERRQ(ierr);
  ierr = ISGetIndices(fis, &rows);CHKERRQ(ierr);
  ierr = PetscMalloc1(m, &idxs);CHKERRQ(ierr);
  for (i = 0, floc = 0; i < m; i++) {
    const PetscInt p = rows[i];

    if (p < 0) {
      idxs[i] = -(p+1);
    } else {
      idxs[i] = p;
      floc   += 1;
    }
  }
  ierr = ISRestoreIndices(fis, &rows);CHKERRQ(ierr);
  ierr = ISDestroy(&fis);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, m, idxs, PETSC_OWN_POINTER, &fis);CHKERRQ(ierr);

  ierr = ISGetLocalSize(cis, &m);CHKERRQ(ierr);
  ierr = ISGetIndices(cis, &cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(m, &idxs);CHKERRQ(ierr);
  ierr = PetscMalloc1(m, &idxs2);CHKERRQ(ierr);
  for (i = 0, cloc = 0; i < m; i++) {
    const PetscInt p = cols[i];

    if (p < 0) {
      idxs[i] = -(p+1);
    } else {
      idxs[i]       = p;
      idxs2[cloc++] = p;
    }
  }
  ierr = ISRestoreIndices(cis, &cols);CHKERRQ(ierr);
  ierr = ISDestroy(&cis);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), m, idxs, PETSC_OWN_POINTER, &cis);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), cloc, idxs2, PETSC_OWN_POINTER, &cis_own);CHKERRQ(ierr);

  /* Create matrix to hold F-C connectivity (MatMatTranspose Mult not supported for MPIAIJ) */
  ierr = MatCreate(PetscObjectComm((PetscObject)dm), &conn);CHKERRQ(ierr);
  ierr = MatSetSizes(conn, floc, cloc, M, N);CHKERRQ(ierr);
  ierr = MatSetType(conn, MATMPIAIJ);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, NULL, &lm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lm, &m, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(conn, m, NULL, m, NULL);CHKERRQ(ierr);

  /* Assemble matrix */
  ierr = ISGetIndices(fis, &rows);CHKERRQ(ierr);
  ierr = ISGetIndices(cis, &cols);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; c++) {
    const PetscInt *cone;
    PetscInt        coneSize, row, col, f;

    col  = cols[c-cStart];
    ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, c, &coneSize);CHKERRQ(ierr);
    for (f = 0; f < coneSize; f++) {
      const PetscScalar v = 1.0;
      const PetscInt *children;
      PetscInt        numChildren, ch;

      row  = rows[cone[f]-fStart];
      ierr = MatSetValues(conn, 1, &row, 1, &col, &v, INSERT_VALUES);CHKERRQ(ierr);

      /* non-conforming meshes */
      ierr = DMPlexGetTreeChildren(dm, cone[f], &numChildren, &children);CHKERRQ(ierr);
      for (ch = 0; ch < numChildren; ch++) {
        const PetscInt child = children[ch];

        if (child < fStart || child >= fEnd) continue;
        row  = rows[child-fStart];
        ierr = MatSetValues(conn, 1, &row, 1, &col, &v, INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISRestoreIndices(fis, &rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(cis, &cols);CHKERRQ(ierr);
  ierr = ISDestroy(&fis);CHKERRQ(ierr);
  ierr = ISDestroy(&cis);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(conn, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(conn, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Get parallel CSR by doing conn^T * conn */
  ierr = MatTransposeMatMult(conn, conn, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &CSR);CHKERRQ(ierr);
  ierr = MatDestroy(&conn);CHKERRQ(ierr);

  /* extract local part of the CSR */
  ierr = MatMPIAIJGetLocalMat(CSR, MAT_INITIAL_MATRIX, &conn);CHKERRQ(ierr);
  ierr = MatDestroy(&CSR);CHKERRQ(ierr);
  ierr = MatGetRowIJ(conn, 0, PETSC_FALSE, PETSC_FALSE, &m, &ii, &jj, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "No IJ format");

  /* get back requested output */
  if (numVertices) *numVertices = m;
  if (offsets) {
    ierr = PetscCalloc1(m+1, &idxs);CHKERRQ(ierr);
    for (i = 1; i < m+1; i++) idxs[i] = ii[i] - i; /* ParMetis does not like self-connectivity */
    *offsets = idxs;
  }
  if (adjacency) {
    ierr = PetscMalloc1(ii[m] - m, &idxs);CHKERRQ(ierr);
    ierr = ISGetIndices(cis_own, &rows);CHKERRQ(ierr);
    for (i = 0, c = 0; i < m; i++) {
      PetscInt j, g = rows[i];

      for (j = ii[i]; j < ii[i+1]; j++) {
        if (jj[j] == g) continue; /* again, self-connectivity */
        idxs[c++] = jj[j];
      }
    }
    if (c != ii[m] - m) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected %D != %D",c,ii[m]-m);
    ierr = ISRestoreIndices(cis_own, &rows);CHKERRQ(ierr);
    *adjacency = idxs;
  }

  /* cleanup */
  ierr = ISDestroy(&cis_own);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(conn, 0, PETSC_FALSE, PETSC_FALSE, &m, &ii, &jj, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "No IJ format");
  ierr = MatDestroy(&conn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreatePartitionerGraph - Create a CSR graph of point connections for the partitioner

  Input Parameters:
+ dm      - The mesh DM dm
- height  - Height of the strata from which to construct the graph

  Output Parameter:
+ numVertices     - Number of vertices in the graph
. offsets         - Point offsets in the graph
. adjacency       - Point connectivity in the graph
- globalNumbering - A map from the local cell numbering to the global numbering used in "adjacency".  Negative indicates that the cell is a duplicate from another process.

  The user can control the definition of adjacency for the mesh using DMSetAdjacency(). They should choose the combination appropriate for the function
  representation on the mesh. If requested, globalNumbering needs to be destroyed by the caller; offsets and adjacency need to be freed with PetscFree().

  Level: developer

.seealso: PetscPartitionerGetType(), PetscPartitionerCreate(), DMSetAdjacency()
@*/
PetscErrorCode DMPlexCreatePartitionerGraph(DM dm, PetscInt height, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency, IS *globalNumbering)
{
  PetscErrorCode ierr;
  PetscBool      usemat = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(((PetscObject) dm)->options,((PetscObject) dm)->prefix, "-dm_plex_csr_via_mat", &usemat, NULL);CHKERRQ(ierr);
  if (usemat) {
    ierr = DMPlexCreatePartitionerGraph_ViaMat(dm, height, numVertices, offsets, adjacency, globalNumbering);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreatePartitionerGraph_Native(dm, height, numVertices, offsets, adjacency, globalNumbering);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateNeighborCSR - Create a mesh graph (cell-cell adjacency) in parallel CSR format.

  Collective on DM

  Input Arguments:
+ dm - The DMPlex
- cellHeight - The height of mesh points to treat as cells (default should be 0)

  Output Arguments:
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* For parallel partitioning, I think you have to communicate supports */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  cellDim = dim - cellHeight;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
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

    ierr = PetscCalloc1(numCells+1, &off);CHKERRQ(ierr);
    /* Count neighboring cells */
    ierr = DMPlexGetHeightStratum(dm, cellHeight+1, &fStart, &fEnd);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *support;
      PetscInt        supportSize;
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      if (supportSize == 2) {
        PetscInt numChildren;

        ierr = DMPlexGetTreeChildren(dm,f,&numChildren,NULL);CHKERRQ(ierr);
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

      ierr = PetscMalloc1(off[numCells], &adj);CHKERRQ(ierr);
      ierr = PetscMalloc1(numCells+1, &tmp);CHKERRQ(ierr);
      ierr = PetscArraycpy(tmp, off, numCells+1);CHKERRQ(ierr);
      /* Get neighboring cells */
      for (f = fStart; f < fEnd; ++f) {
        const PetscInt *support;
        PetscInt        supportSize;
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        if (supportSize == 2) {
          PetscInt numChildren;

          ierr = DMPlexGetTreeChildren(dm,f,&numChildren,NULL);CHKERRQ(ierr);
          if (!numChildren) {
            adj[tmp[support[0]-cStart]++] = support[1];
            adj[tmp[support[1]-cStart]++] = support[0];
          }
        }
      }
      if (PetscDefined(USE_DEBUG)) {
        for (c = 0; c < cEnd-cStart; ++c) if (tmp[c] != off[c+1]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offset %d != %d for cell %d", tmp[c], off[c], c+cStart);
      }
      ierr = PetscFree(tmp);CHKERRQ(ierr);
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

      ierr = DMPlexGetConeSize(dm, c, &corners);CHKERRQ(ierr);
      if (!cornersSeen[corners]) {
        PetscInt nFV;

        if (numFaceCases >= maxFaceCases) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Exceeded maximum number of face recognition cases");
        cornersSeen[corners] = 1;

        ierr = DMPlexGetNumFaceVertices(dm, cellDim, corners, &nFV);CHKERRQ(ierr);

        numFaceVertices[numFaceCases++] = nFV;
      }
    }
  }
  ierr = PetscCalloc1(numCells+1, &off);CHKERRQ(ierr);
  /* Count neighboring cells */
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt numNeighbors = PETSC_DETERMINE, n;

    ierr = DMPlexGetAdjacency_Internal(dm, cell, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &numNeighbors, &neighborCells);CHKERRQ(ierr);
    /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
    for (n = 0; n < numNeighbors; ++n) {
      PetscInt        cellPair[2];
      PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
      PetscInt        meetSize = 0;
      const PetscInt *meet    = NULL;

      cellPair[0] = cell; cellPair[1] = neighborCells[n];
      if (cellPair[0] == cellPair[1]) continue;
      if (!found) {
        ierr = DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
        if (meetSize) {
          PetscInt f;

          for (f = 0; f < numFaceCases; ++f) {
            if (numFaceVertices[f] == meetSize) {
              found = PETSC_TRUE;
              break;
            }
          }
        }
        ierr = DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
      }
      if (found) ++off[cell-cStart+1];
    }
  }
  /* Prefix sum */
  for (cell = 1; cell <= numCells; ++cell) off[cell] += off[cell-1];

  if (adjacency) {
    ierr = PetscMalloc1(off[numCells], &adj);CHKERRQ(ierr);
    /* Get neighboring cells */
    for (cell = cStart; cell < cEnd; ++cell) {
      PetscInt numNeighbors = PETSC_DETERMINE, n;
      PetscInt cellOffset   = 0;

      ierr = DMPlexGetAdjacency_Internal(dm, cell, PETSC_TRUE, PETSC_FALSE, PETSC_FALSE, &numNeighbors, &neighborCells);CHKERRQ(ierr);
      /* Get meet with each cell, and check with recognizer (could optimize to check each pair only once) */
      for (n = 0; n < numNeighbors; ++n) {
        PetscInt        cellPair[2];
        PetscBool       found    = faceDepth > 1 ? PETSC_TRUE : PETSC_FALSE;
        PetscInt        meetSize = 0;
        const PetscInt *meet    = NULL;

        cellPair[0] = cell; cellPair[1] = neighborCells[n];
        if (cellPair[0] == cellPair[1]) continue;
        if (!found) {
          ierr = DMPlexGetMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
          if (meetSize) {
            PetscInt f;

            for (f = 0; f < numFaceCases; ++f) {
              if (numFaceVertices[f] == meetSize) {
                found = PETSC_TRUE;
                break;
              }
            }
          }
          ierr = DMPlexRestoreMeet(dm, 2, cellPair, &meetSize, &meet);CHKERRQ(ierr);
        }
        if (found) {
          adj[off[cell-cStart]+cellOffset] = neighborCells[n];
          ++cellOffset;
        }
      }
    }
  }
  ierr = PetscFree(neighborCells);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscSection   vertSection = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (targetSection) PetscValidHeaderSpecific(targetSection, PETSC_SECTION_CLASSID, 3);
  PetscValidHeaderSpecific(partSection, PETSC_SECTION_CLASSID, 4);
  PetscValidPointer(partition, 5);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isplex);CHKERRQ(ierr);
  if (!isplex) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not for type %s",((PetscObject)dm)->type_name);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) part), &size);CHKERRQ(ierr);
  if (size == 1) {
    PetscInt *points;
    PetscInt  cStart, cEnd, c;

    ierr = DMPlexGetHeightStratum(dm, part->height, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = PetscSectionReset(partSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(partSection, 0, size);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(partSection, 0, cEnd-cStart);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
    ierr = PetscMalloc1(cEnd-cStart, &points);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) points[c] = c;
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject) part), cEnd-cStart, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (part->height == 0) {
    PetscInt numVertices = 0;
    PetscInt *start     = NULL;
    PetscInt *adjacency = NULL;
    IS       globalNumbering;

    if (!part->noGraph || part->viewGraph) {
      ierr = DMPlexCreatePartitionerGraph(dm, part->height, &numVertices, &start, &adjacency, &globalNumbering);CHKERRQ(ierr);
    } else { /* only compute the number of owned local vertices */
      const PetscInt *idxs;
      PetscInt       p, pStart, pEnd;

      ierr = DMPlexGetHeightStratum(dm, part->height, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = DMPlexCreateNumbering_Plex(dm, pStart, pEnd, 0, NULL, dm->sf, &globalNumbering);CHKERRQ(ierr);
      ierr = ISGetIndices(globalNumbering, &idxs);CHKERRQ(ierr);
      for (p = 0; p < pEnd - pStart; p++) numVertices += idxs[p] < 0 ? 0 : 1;
      ierr = ISRestoreIndices(globalNumbering, &idxs);CHKERRQ(ierr);
    }
    if (part->usevwgt) {
      PetscSection   section = dm->localSection, clSection = NULL;
      IS             clPoints = NULL;
      const PetscInt *gid,*clIdx;
      PetscInt       v, p, pStart, pEnd;

      /* dm->localSection encodes degrees of freedom per point, not per cell. We need to get the closure index to properly specify cell weights (aka dofs) */
      /* We do this only if the local section has been set */
      if (section) {
        ierr = PetscSectionGetClosureIndex(section, (PetscObject)dm, &clSection, NULL);CHKERRQ(ierr);
        if (!clSection) {
          ierr = DMPlexCreateClosureIndex(dm,NULL);CHKERRQ(ierr);
        }
        ierr = PetscSectionGetClosureIndex(section, (PetscObject)dm, &clSection, &clPoints);CHKERRQ(ierr);
        ierr = ISGetIndices(clPoints,&clIdx);CHKERRQ(ierr);
      }
      ierr = DMPlexGetHeightStratum(dm, part->height, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = PetscSectionCreate(PETSC_COMM_SELF, &vertSection);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(vertSection, 0, numVertices);CHKERRQ(ierr);
      if (globalNumbering) {
        ierr = ISGetIndices(globalNumbering,&gid);CHKERRQ(ierr);
      } else gid = NULL;
      for (p = pStart, v = 0; p < pEnd; ++p) {
        PetscInt dof = 1;

        /* skip cells in the overlap */
        if (gid && gid[p-pStart] < 0) continue;

        if (section) {
          PetscInt cl, clSize, clOff;

          dof  = 0;
          ierr = PetscSectionGetDof(clSection, p, &clSize);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(clSection, p, &clOff);CHKERRQ(ierr);
          for (cl = 0; cl < clSize; cl+=2) {
            PetscInt clDof, clPoint = clIdx[clOff + cl]; /* odd indices are reserved for orientations */

            ierr = PetscSectionGetDof(section, clPoint, &clDof);CHKERRQ(ierr);
            dof += clDof;
          }
        }
        if (!dof) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Number of dofs for point %D in the local section should be positive",p);
        ierr = PetscSectionSetDof(vertSection, v, dof);CHKERRQ(ierr);
        v++;
      }
      if (globalNumbering) {
        ierr = ISRestoreIndices(globalNumbering,&gid);CHKERRQ(ierr);
      }
      if (clPoints) {
        ierr = ISRestoreIndices(clPoints,&clIdx);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetUp(vertSection);CHKERRQ(ierr);
    }
    ierr = PetscPartitionerPartition(part, size, numVertices, start, adjacency, vertSection, targetSection, partSection, partition);CHKERRQ(ierr);
    ierr = PetscFree(start);CHKERRQ(ierr);
    ierr = PetscFree(adjacency);CHKERRQ(ierr);
    if (globalNumbering) { /* partition is wrt global unique numbering: change this to be wrt local numbering */
      const PetscInt *globalNum;
      const PetscInt *partIdx;
      PetscInt       *map, cStart, cEnd;
      PetscInt       *adjusted, i, localSize, offset;
      IS             newPartition;

      ierr = ISGetLocalSize(*partition,&localSize);CHKERRQ(ierr);
      ierr = PetscMalloc1(localSize,&adjusted);CHKERRQ(ierr);
      ierr = ISGetIndices(globalNumbering,&globalNum);CHKERRQ(ierr);
      ierr = ISGetIndices(*partition,&partIdx);CHKERRQ(ierr);
      ierr = PetscMalloc1(localSize,&map);CHKERRQ(ierr);
      ierr = DMPlexGetHeightStratum(dm, part->height, &cStart, &cEnd);CHKERRQ(ierr);
      for (i = cStart, offset = 0; i < cEnd; i++) {
        if (globalNum[i - cStart] >= 0) map[offset++] = i;
      }
      for (i = 0; i < localSize; i++) {
        adjusted[i] = map[partIdx[i]];
      }
      ierr = PetscFree(map);CHKERRQ(ierr);
      ierr = ISRestoreIndices(*partition,&partIdx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(globalNumbering,&globalNum);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,localSize,adjusted,PETSC_OWN_POINTER,&newPartition);CHKERRQ(ierr);
      ierr = ISDestroy(&globalNumbering);CHKERRQ(ierr);
      ierr = ISDestroy(partition);CHKERRQ(ierr);
      *partition = newPartition;
    }
  } else SETERRQ1(PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_OUTOFRANGE, "Invalid height %D for points to partition", part->height);
  ierr = PetscSectionDestroy(&vertSection);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 2);
  ierr = PetscObjectReference((PetscObject)part);CHKERRQ(ierr);
  ierr = PetscPartitionerDestroy(&mesh->partitioner);CHKERRQ(ierr);
  mesh->partitioner = part;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosure_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  const PetscInt *cone;
  PetscInt       coneSize, c;
  PetscBool      missing;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscHSetIQueryAdd(ht, point, &missing);CHKERRQ(ierr);
  if (missing) {
    ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
    for (c = 0; c < coneSize; c++) {
      ierr = DMPlexAddClosure_Private(dm, ht, cone[c]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode DMPlexAddClosure_Tree(DM dm, PetscHSetI ht, PetscInt point, PetscBool up, PetscBool down)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (up) {
    PetscInt parent;

    ierr = DMPlexGetTreeParent(dm,point,&parent,NULL);CHKERRQ(ierr);
    if (parent != point) {
      PetscInt closureSize, *closure = NULL, i;

      ierr = DMPlexGetTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (i = 0; i < closureSize; i++) {
        PetscInt cpoint = closure[2*i];

        ierr = PetscHSetIAdd(ht, cpoint);CHKERRQ(ierr);
        ierr = DMPlexAddClosure_Tree(dm,ht,cpoint,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm,parent,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    }
  }
  if (down) {
    PetscInt numChildren;
    const PetscInt *children;

    ierr = DMPlexGetTreeChildren(dm,point,&numChildren,&children);CHKERRQ(ierr);
    if (numChildren) {
      PetscInt i;

      for (i = 0; i < numChildren; i++) {
        PetscInt cpoint = children[i];

        ierr = PetscHSetIAdd(ht, cpoint);CHKERRQ(ierr);
        ierr = DMPlexAddClosure_Tree(dm,ht,cpoint,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosureTree_Up_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  PetscInt       parent;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetTreeParent(dm, point, &parent,NULL);CHKERRQ(ierr);
  if (point != parent) {
    const PetscInt *cone;
    PetscInt       coneSize, c;

    ierr = DMPlexAddClosureTree_Up_Private(dm, ht, parent);CHKERRQ(ierr);
    ierr = DMPlexAddClosure_Private(dm, ht, parent);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, parent, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, parent, &coneSize);CHKERRQ(ierr);
    for (c = 0; c < coneSize; c++) {
      const PetscInt cp = cone[c];

      ierr = DMPlexAddClosureTree_Up_Private(dm, ht, cp);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosureTree_Down_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  PetscInt       i, numChildren;
  const PetscInt *children;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = DMPlexGetTreeChildren(dm, point, &numChildren, &children);CHKERRQ(ierr);
  for (i = 0; i < numChildren; i++) {
    ierr = PetscHSetIAdd(ht, children[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexAddClosureTree_Private(DM dm, PetscHSetI ht, PetscInt point)
{
  const PetscInt *cone;
  PetscInt       coneSize, c;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscHSetIAdd(ht, point);CHKERRQ(ierr);
  ierr = DMPlexAddClosureTree_Up_Private(dm, ht, point);CHKERRQ(ierr);
  ierr = DMPlexAddClosureTree_Down_Private(dm, ht, point);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
  for (c = 0; c < coneSize; c++) {
    ierr = DMPlexAddClosureTree_Private(dm, ht, cone[c]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexClosurePoints_Private(DM dm, PetscInt numPoints, const PetscInt points[], IS *closureIS)
{
  DM_Plex         *mesh = (DM_Plex *)dm->data;
  const PetscBool hasTree = (mesh->parentSection || mesh->childSection) ? PETSC_TRUE : PETSC_FALSE;
  PetscInt        nelems, *elems, off = 0, p;
  PetscHSetI      ht;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
  ierr = PetscHSetIResize(ht, numPoints*16);CHKERRQ(ierr);
  if (!hasTree) {
    for (p = 0; p < numPoints; ++p) {
      ierr = DMPlexAddClosure_Private(dm, ht, points[p]);CHKERRQ(ierr);
    }
  } else {
#if 1
    for (p = 0; p < numPoints; ++p) {
      ierr = DMPlexAddClosureTree_Private(dm, ht, points[p]);CHKERRQ(ierr);
    }
#else
    PetscInt  *closure = NULL, closureSize, c;
    for (p = 0; p < numPoints; ++p) {
      ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (c = 0; c < closureSize*2; c += 2) {
        ierr = PetscHSetIAdd(ht, closure[c]);CHKERRQ(ierr);
        if (hasTree) {ierr = DMPlexAddClosure_Tree(dm, ht, closure[c], PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);}
      }
    }
    if (closure) {ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, NULL, &closure);CHKERRQ(ierr);}
#endif
  }
  ierr = PetscHSetIGetSize(ht, &nelems);CHKERRQ(ierr);
  ierr = PetscMalloc1(nelems, &elems);CHKERRQ(ierr);
  ierr = PetscHSetIGetElems(ht, &off, elems);CHKERRQ(ierr);
  ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);
  ierr = PetscSortInt(nelems, elems);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, nelems, elems, PETSC_OWN_POINTER, closureIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelClosure - Add the closure of all points to the partition label

  Input Parameters:
+ dm     - The DM
- label  - DMLabel assinging ranks to remote roots

  Level: developer

.seealso: DMPlexPartitionLabelCreateSF(), DMPlexDistribute(), DMPlexCreateOverlap()
@*/
PetscErrorCode DMPlexPartitionLabelClosure(DM dm, DMLabel label)
{
  IS              rankIS,   pointIS, closureIS;
  const PetscInt *ranks,   *points;
  PetscInt        numRanks, numPoints, r;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMLabelGetValueIS(label, &rankIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(rankIS, &numRanks);CHKERRQ(ierr);
  ierr = ISGetIndices(rankIS, &ranks);CHKERRQ(ierr);
  for (r = 0; r < numRanks; ++r) {
    const PetscInt rank = ranks[r];
    ierr = DMLabelGetStratumIS(label, rank, &pointIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = DMPlexClosurePoints_Private(dm, numPoints, points, &closureIS);CHKERRQ(ierr);
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    ierr = DMLabelSetStratumIS(label, rank, closureIS);CHKERRQ(ierr);
    ierr = ISDestroy(&closureIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(rankIS, &ranks);CHKERRQ(ierr);
  ierr = ISDestroy(&rankIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelAdjacency - Add one level of adjacent points to the partition label

  Input Parameters:
+ dm     - The DM
- label  - DMLabel assinging ranks to remote roots

  Level: developer

.seealso: DMPlexPartitionLabelCreateSF(), DMPlexDistribute(), DMPlexCreateOverlap()
@*/
PetscErrorCode DMPlexPartitionLabelAdjacency(DM dm, DMLabel label)
{
  IS              rankIS,   pointIS;
  const PetscInt *ranks,   *points;
  PetscInt        numRanks, numPoints, r, p, a, adjSize;
  PetscInt       *adj = NULL;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMLabelGetValueIS(label, &rankIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(rankIS, &numRanks);CHKERRQ(ierr);
  ierr = ISGetIndices(rankIS, &ranks);CHKERRQ(ierr);
  for (r = 0; r < numRanks; ++r) {
    const PetscInt rank = ranks[r];

    ierr = DMLabelGetStratumIS(label, rank, &pointIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      adjSize = PETSC_DETERMINE;
      ierr = DMPlexGetAdjacency(dm, points[p], &adjSize, &adj);CHKERRQ(ierr);
      for (a = 0; a < adjSize; ++a) {ierr = DMLabelSetValue(label, adj[a], rank);CHKERRQ(ierr);}
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(rankIS, &ranks);CHKERRQ(ierr);
  ierr = ISDestroy(&rankIS);CHKERRQ(ierr);
  ierr = PetscFree(adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelPropagate - Propagate points in a partition label over the point SF

  Input Parameters:
+ dm     - The DM
- label  - DMLabel assinging ranks to remote roots

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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  /* Pull point contributions from remote leaves into local roots */
  ierr = DMLabelGather(label, sfPoint, &lblLeaves);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(lblLeaves, &rankIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(rankIS, &numRanks);CHKERRQ(ierr);
  ierr = ISGetIndices(rankIS, &ranks);CHKERRQ(ierr);
  for (r = 0; r < numRanks; ++r) {
    const PetscInt remoteRank = ranks[r];
    if (remoteRank == rank) continue;
    ierr = DMLabelGetStratumIS(lblLeaves, remoteRank, &pointIS);CHKERRQ(ierr);
    ierr = DMLabelInsertIS(label, pointIS, remoteRank);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(rankIS, &ranks);CHKERRQ(ierr);
  ierr = ISDestroy(&rankIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&lblLeaves);CHKERRQ(ierr);
  /* Push point contributions from roots into remote leaves */
  ierr = DMLabelDistribute(label, sfPoint, &lblRoots);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(lblRoots, &rankIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(rankIS, &numRanks);CHKERRQ(ierr);
  ierr = ISGetIndices(rankIS, &ranks);CHKERRQ(ierr);
  for (r = 0; r < numRanks; ++r) {
    const PetscInt remoteRank = ranks[r];
    if (remoteRank == rank) continue;
    ierr = DMLabelGetStratumIS(lblRoots, remoteRank, &pointIS);CHKERRQ(ierr);
    ierr = DMLabelInsertIS(label, pointIS, remoteRank);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(rankIS, &ranks);CHKERRQ(ierr);
  ierr = ISDestroy(&rankIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&lblRoots);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelInvert - Create a partition label of remote roots from a local root label

  Input Parameters:
+ dm        - The DM
. rootLabel - DMLabel assinging ranks to local roots
- processSF - A star forest mapping into the local index on each remote rank

  Output Parameter:
. leafLabel - DMLabel assinging ranks to remote roots

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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_PartLabelInvert,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);

  /* Convert to (point, rank) and use actual owners */
  ierr = PetscSectionCreate(comm, &rootSection);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(rootSection, 0, size);CHKERRQ(ierr);
  ierr = DMLabelGetValueIS(rootLabel, &valueIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(valueIS, &numNeighbors);CHKERRQ(ierr);
  ierr = ISGetIndices(valueIS, &neighbors);CHKERRQ(ierr);
  for (n = 0; n < numNeighbors; ++n) {
    ierr = DMLabelGetStratumSize(rootLabel, neighbors[n], &numPoints);CHKERRQ(ierr);
    ierr = PetscSectionAddDof(rootSection, neighbors[n], numPoints);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(rootSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(rootSection, &rootSize);CHKERRQ(ierr);
  ierr = PetscMalloc1(rootSize, &rootPoints);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, NULL, &nleaves, &local, &remote);CHKERRQ(ierr);
  for (n = 0; n < numNeighbors; ++n) {
    IS              pointIS;
    const PetscInt *points;

    ierr = PetscSectionGetOffset(rootSection, neighbors[n], &off);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(rootLabel, neighbors[n], &pointIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    for (p = 0; p < numPoints; ++p) {
      if (local) {ierr = PetscFindInt(points[p], nleaves, local, &l);CHKERRQ(ierr);}
      else       {l = -1;}
      if (l >= 0) {rootPoints[off+p] = remote[l];}
      else        {rootPoints[off+p].index = points[p]; rootPoints[off+p].rank = rank;}
    }
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
  }

  /* Try to communicate overlap using All-to-All */
  if (!processSF) {
    PetscInt64  counter = 0;
    PetscBool   locOverflow = PETSC_FALSE;
    PetscMPIInt *scounts, *sdispls, *rcounts, *rdispls;

    ierr = PetscCalloc4(size, &scounts, size, &sdispls, size, &rcounts, size, &rdispls);CHKERRQ(ierr);
    for (n = 0; n < numNeighbors; ++n) {
      ierr = PetscSectionGetDof(rootSection, neighbors[n], &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(rootSection, neighbors[n], &off);CHKERRQ(ierr);
#if defined(PETSC_USE_64BIT_INDICES)
      if (dof > PETSC_MPI_INT_MAX) {locOverflow = PETSC_TRUE; break;}
      if (off > PETSC_MPI_INT_MAX) {locOverflow = PETSC_TRUE; break;}
#endif
      scounts[neighbors[n]] = (PetscMPIInt) dof;
      sdispls[neighbors[n]] = (PetscMPIInt) off;
    }
    ierr = MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, comm);CHKERRQ(ierr);
    for (r = 0; r < size; ++r) { rdispls[r] = (int)counter; counter += rcounts[r]; }
    if (counter > PETSC_MPI_INT_MAX) locOverflow = PETSC_TRUE;
    ierr = MPI_Allreduce(&locOverflow, &mpiOverflow, 1, MPIU_BOOL, MPI_LOR, comm);CHKERRQ(ierr);
    if (!mpiOverflow) {
      ierr = PetscInfo(dm,"Using Alltoallv for mesh distribution\n");CHKERRQ(ierr);
      leafSize = (PetscInt) counter;
      ierr = PetscMalloc1(leafSize, &leafPoints);CHKERRQ(ierr);
      ierr = MPI_Alltoallv(rootPoints, scounts, sdispls, MPIU_2INT, leafPoints, rcounts, rdispls, MPIU_2INT, comm);CHKERRQ(ierr);
    }
    ierr = PetscFree4(scounts, sdispls, rcounts, rdispls);CHKERRQ(ierr);
  }

  /* Communicate overlap using process star forest */
  if (processSF || mpiOverflow) {
    PetscSF      procSF;
    PetscSection leafSection;

    if (processSF) {
      ierr = PetscInfo(dm,"Using processSF for mesh distribution\n");CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)processSF);CHKERRQ(ierr);
      procSF = processSF;
    } else {
      ierr = PetscInfo(dm,"Using processSF for mesh distribution (MPI overflow)\n");CHKERRQ(ierr);
      ierr = PetscSFCreate(comm,&procSF);CHKERRQ(ierr);
      ierr = PetscSFSetGraphWithPattern(procSF,NULL,PETSCSF_PATTERN_ALLTOALL);CHKERRQ(ierr);
    }

    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &leafSection);CHKERRQ(ierr);
    ierr = DMPlexDistributeData(dm, procSF, rootSection, MPIU_2INT, rootPoints, leafSection, (void**) &leafPoints);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(leafSection, &leafSize);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&leafSection);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&procSF);CHKERRQ(ierr);
  }

  for (p = 0; p < leafSize; p++) {
    ierr = DMLabelSetValue(leafLabel, leafPoints[p].index, leafPoints[p].rank);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(valueIS, &neighbors);CHKERRQ(ierr);
  ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&rootSection);CHKERRQ(ierr);
  ierr = PetscFree(rootPoints);CHKERRQ(ierr);
  ierr = PetscFree(leafPoints);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_PartLabelInvert,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelCreateSF - Create a star forest from a label that assigns ranks to points

  Input Parameters:
+ dm    - The DM
- label - DMLabel assinging ranks to remote roots

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_PartLabelCreateSF,dm,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);

  ierr = DMLabelGetValueIS(label, &neighborsIS);CHKERRQ(ierr);
#if 0
  {
    IS is;
    ierr = ISDuplicate(neighborsIS, &is);CHKERRQ(ierr);
    ierr = ISSort(is);CHKERRQ(ierr);
    ierr = ISDestroy(&neighborsIS);CHKERRQ(ierr);
    neighborsIS = is;
  }
#endif
  ierr = ISGetLocalSize(neighborsIS, &nNeighbors);CHKERRQ(ierr);
  ierr = ISGetIndices(neighborsIS, &neighbors);CHKERRQ(ierr);
  for (numRemote = 0, n = 0; n < nNeighbors; ++n) {
    ierr = DMLabelGetStratumSize(label, neighbors[n], &numPoints);CHKERRQ(ierr);
    numRemote += numPoints;
  }
  ierr = PetscMalloc1(numRemote, &remotePoints);CHKERRQ(ierr);
  /* Put owned points first */
  ierr = DMLabelGetStratumSize(label, rank, &numPoints);CHKERRQ(ierr);
  if (numPoints > 0) {
    ierr = DMLabelGetStratumIS(label, rank, &remoteRootIS);CHKERRQ(ierr);
    ierr = ISGetIndices(remoteRootIS, &remoteRoots);CHKERRQ(ierr);
    for (p = 0; p < numPoints; p++) {
      remotePoints[idx].index = remoteRoots[p];
      remotePoints[idx].rank = rank;
      idx++;
    }
    ierr = ISRestoreIndices(remoteRootIS, &remoteRoots);CHKERRQ(ierr);
    ierr = ISDestroy(&remoteRootIS);CHKERRQ(ierr);
  }
  /* Now add remote points */
  for (n = 0; n < nNeighbors; ++n) {
    const PetscInt nn = neighbors[n];

    ierr = DMLabelGetStratumSize(label, nn, &numPoints);CHKERRQ(ierr);
    if (nn == rank || numPoints <= 0) continue;
    ierr = DMLabelGetStratumIS(label, nn, &remoteRootIS);CHKERRQ(ierr);
    ierr = ISGetIndices(remoteRootIS, &remoteRoots);CHKERRQ(ierr);
    for (p = 0; p < numPoints; p++) {
      remotePoints[idx].index = remoteRoots[p];
      remotePoints[idx].rank = nn;
      idx++;
    }
    ierr = ISRestoreIndices(remoteRootIS, &remoteRoots);CHKERRQ(ierr);
    ierr = ISDestroy(&remoteRootIS);CHKERRQ(ierr);
  }
  ierr = PetscSFCreate(PetscObjectComm((PetscObject) dm), sf);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sf, pEnd-pStart, numRemote, NULL, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(*sf);CHKERRQ(ierr);
  ierr = ISDestroy(&neighborsIS);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_PartLabelCreateSF,dm,0,0,0);CHKERRQ(ierr);
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
  PetscInt      ierr, pStart, pEnd, i, j, counter, leafCounter, sumDegrees, nroots, nleafs;
  PetscInt     *cumSumDegrees, *newOwners, *newNumbers, *rankOnLeafs, *locationsOfLeafs, *remoteLocalPointOfLeafs, *points, *leafsNew;
  PetscSFNode  *leafLocationsNew;
  const         PetscSFNode *iremote;
  const         PetscInt *ilocal;
  PetscBool    *isLeaf;
  PetscSF       sf;
  MPI_Comm      comm;
  PetscMPIInt   rank, size;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);

  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, &nleafs, &ilocal, &iremote);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &isLeaf);CHKERRQ(ierr);
  for (i=0; i<pEnd-pStart; i++) {
    isLeaf[i] = PETSC_FALSE;
  }
  for (i=0; i<nleafs; i++) {
    isLeaf[ilocal[i]-pStart] = PETSC_TRUE;
  }

  ierr = PetscMalloc1(pEnd-pStart+1, &cumSumDegrees);CHKERRQ(ierr);
  cumSumDegrees[0] = 0;
  for (i=1; i<=pEnd-pStart; i++) {
    cumSumDegrees[i] = cumSumDegrees[i-1] + degrees[i-1];
  }
  sumDegrees = cumSumDegrees[pEnd-pStart];
  /* get the location of my leafs (we have sumDegrees many leafs pointing at our roots) */

  ierr = PetscMalloc1(sumDegrees, &locationsOfLeafs);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &rankOnLeafs);CHKERRQ(ierr);
  for (i=0; i<pEnd-pStart; i++) {
    rankOnLeafs[i] = rank;
  }
  ierr = PetscSFGatherBegin(sf, MPIU_INT, rankOnLeafs, locationsOfLeafs);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(sf, MPIU_INT, rankOnLeafs, locationsOfLeafs);CHKERRQ(ierr);
  ierr = PetscFree(rankOnLeafs);CHKERRQ(ierr);

  /* get the remote local points of my leaves */
  ierr = PetscMalloc1(sumDegrees, &remoteLocalPointOfLeafs);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &points);CHKERRQ(ierr);
  for (i=0; i<pEnd-pStart; i++) {
    points[i] = pStart+i;
  }
  ierr = PetscSFGatherBegin(sf, MPIU_INT, points, remoteLocalPointOfLeafs);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(sf, MPIU_INT, points, remoteLocalPointOfLeafs);CHKERRQ(ierr);
  ierr = PetscFree(points);CHKERRQ(ierr);
  /* Figure out the new owners of the vertices that are up for grabs and their numbers on the new owners */
  ierr = PetscMalloc1(pEnd-pStart, &newOwners);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &newNumbers);CHKERRQ(ierr);
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
      if (newNumber == -1) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Couldn't find the new owner of vertex.");

      newOwners[oldNumber] = newOwner;
      newNumbers[oldNumber] = newNumber;
    }
  }
  ierr = PetscFree(cumSumDegrees);CHKERRQ(ierr);
  ierr = PetscFree(locationsOfLeafs);CHKERRQ(ierr);
  ierr = PetscFree(remoteLocalPointOfLeafs);CHKERRQ(ierr);

  ierr = PetscSFBcastBegin(sf, MPIU_INT, newOwners, newOwners);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_INT, newOwners, newOwners);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPIU_INT, newNumbers, newNumbers);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_INT, newNumbers, newNumbers);CHKERRQ(ierr);

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
  ierr = PetscMalloc1(leafCounter, &leafsNew);CHKERRQ(ierr);
  ierr = PetscMalloc1(leafCounter, &leafLocationsNew);CHKERRQ(ierr);

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

  ierr = PetscSFSetGraph(sf, nroots, leafCounter, leafsNew, PETSC_OWN_POINTER, leafLocationsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscFree(newOwners);CHKERRQ(ierr);
  ierr = PetscFree(newNumbers);CHKERRQ(ierr);
  ierr = PetscFree(isLeaf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexViewDistribution(MPI_Comm comm, PetscInt n, PetscInt skip, PetscInt *vtxwgt, PetscInt *part, PetscViewer viewer)
{
  PetscInt *distribution, min, max, sum, i, ierr;
  PetscMPIInt rank, size;
  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscCalloc1(size, &distribution);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (part) distribution[part[i]] += vtxwgt[skip*i];
    else distribution[rank] += vtxwgt[skip*i];
  }
  ierr = MPI_Allreduce(MPI_IN_PLACE, distribution, size, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
  min = distribution[0];
  max = distribution[0];
  sum = distribution[0];
  for (i=1; i<size; i++) {
    if (distribution[i]<min) min=distribution[i];
    if (distribution[i]>max) max=distribution[i];
    sum += distribution[i];
  }
  ierr = PetscViewerASCIIPrintf(viewer, "Min: %D, Avg: %D, Max: %D, Balance: %f\n", min, sum/size, max, (max*1.*size)/sum);CHKERRQ(ierr);
  ierr = PetscFree(distribution);CHKERRQ(ierr);
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
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size==1) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(DMPLEX_RebalanceSharedPoints, dm, 0, 0, 0);CHKERRQ(ierr);

  ierr = PetscOptionsGetViewer(comm,((PetscObject)dm)->options, prefix,"-dm_rebalance_partition_view",&viewer,&format,NULL);CHKERRQ(ierr);
  if (viewer) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
  }

  /* Figure out all points in the plex that we are interested in balancing. */
  ierr = DMPlexGetDepthStratum(dm, entityDepth, &eBegin, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &toBalance);CHKERRQ(ierr);

  for (i=0; i<pEnd-pStart; i++) {
    toBalance[i] = (PetscBool)(i-pStart>=eBegin && i-pStart<eEnd);
  }

  /* There are three types of points:
   * exclusivelyOwned: points that are owned by this process and only seen by this process
   * nonExclusivelyOwned: points that are owned by this process but seen by at least another process
   * leaf: a point that is seen by this process but owned by a different process
   */
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, &nleafs, &ilocal, &iremote);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &isLeaf);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &isNonExclusivelyOwned);CHKERRQ(ierr);
  ierr = PetscMalloc1(pEnd-pStart, &isExclusivelyOwned);CHKERRQ(ierr);
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
  ierr = PetscSFComputeDegreeBegin(sf, &degrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf, &degrees);CHKERRQ(ierr);

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

  ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(layout, 1 + numNonExclusivelyOwned);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(layout, &cumSumVertices);CHKERRQ(ierr);

  ierr = PetscMalloc1(pEnd-pStart, &globalNumbersOfLocalOwnedVertices);CHKERRQ(ierr);
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
  ierr = PetscMalloc1(pEnd-pStart, &leafGlobalNumbers);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sf, MPIU_INT, globalNumbersOfLocalOwnedVertices, leafGlobalNumbers);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf, MPIU_INT, globalNumbersOfLocalOwnedVertices, leafGlobalNumbers);CHKERRQ(ierr);

  /* Now start building the data structure for ParMETIS */

  ierr = MatCreate(comm, &Apre);CHKERRQ(ierr);
  ierr = MatSetType(Apre, MATPREALLOCATOR);CHKERRQ(ierr);
  ierr = MatSetSizes(Apre, 1+numNonExclusivelyOwned, 1+numNonExclusivelyOwned, cumSumVertices[size], cumSumVertices[size]);CHKERRQ(ierr);
  ierr = MatSetUp(Apre);CHKERRQ(ierr);

  for (i=0; i<pEnd-pStart; i++) {
    if (toBalance[i]) {
      idx = cumSumVertices[rank];
      if (isNonExclusivelyOwned[i]) jdx = globalNumbersOfLocalOwnedVertices[i];
      else if (isLeaf[i]) jdx = leafGlobalNumbers[i];
      else continue;
      ierr = MatSetValue(Apre, idx, jdx, 1, INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(Apre, jdx, idx, 1, INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(Apre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Apre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(comm, &A);CHKERRQ(ierr);
  ierr = MatSetType(A, MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A, 1+numNonExclusivelyOwned, 1+numNonExclusivelyOwned, cumSumVertices[size], cumSumVertices[size]);CHKERRQ(ierr);
  ierr = MatPreallocatorPreallocate(Apre, PETSC_FALSE, A);CHKERRQ(ierr);
  ierr = MatDestroy(&Apre);CHKERRQ(ierr);

  for (i=0; i<pEnd-pStart; i++) {
    if (toBalance[i]) {
      idx = cumSumVertices[rank];
      if (isNonExclusivelyOwned[i]) jdx = globalNumbersOfLocalOwnedVertices[i];
      else if (isLeaf[i]) jdx = leafGlobalNumbers[i];
      else continue;
      ierr = MatSetValue(A, idx, jdx, 1, INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValue(A, jdx, idx, 1, INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(leafGlobalNumbers);CHKERRQ(ierr);
  ierr = PetscFree(globalNumbersOfLocalOwnedVertices);CHKERRQ(ierr);

  nparts = size;
  wgtflag = 2;
  numflag = 0;
  ncon = 2;
  real_t *tpwgts;
  ierr = PetscMalloc1(ncon * nparts, &tpwgts);CHKERRQ(ierr);
  for (i=0; i<ncon*nparts; i++) {
    tpwgts[i] = 1./(nparts);
  }

  ierr = PetscMalloc1(ncon, &ubvec);CHKERRQ(ierr);
  ubvec[0] = 1.01;
  ubvec[1] = 1.01;
  lenadjncy = 0;
  for (i=0; i<1+numNonExclusivelyOwned; i++) {
    PetscInt temp=0;
    ierr = MatGetRow(A, cumSumVertices[rank] + i, &temp, NULL, NULL);CHKERRQ(ierr);
    lenadjncy += temp;
    ierr = MatRestoreRow(A, cumSumVertices[rank] + i, &temp, NULL, NULL);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(lenadjncy, &adjncy);CHKERRQ(ierr);
  lenxadj = 2 + numNonExclusivelyOwned;
  ierr = PetscMalloc1(lenxadj, &xadj);CHKERRQ(ierr);
  xadj[0] = 0;
  counter = 0;
  for (i=0; i<1+numNonExclusivelyOwned; i++) {
    PetscInt        temp=0;
    const PetscInt *cols;
    ierr = MatGetRow(A, cumSumVertices[rank] + i, &temp, &cols, NULL);CHKERRQ(ierr);
    ierr = PetscArraycpy(&adjncy[counter], cols, temp);CHKERRQ(ierr);
    counter += temp;
    xadj[i+1] = counter;
    ierr = MatRestoreRow(A, cumSumVertices[rank] + i, &temp, &cols, NULL);CHKERRQ(ierr);
  }

  ierr = PetscMalloc1(cumSumVertices[rank+1]-cumSumVertices[rank], &part);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncon*(1 + numNonExclusivelyOwned), &vtxwgt);CHKERRQ(ierr);
  vtxwgt[0] = numExclusivelyOwned;
  if (ncon>1) vtxwgt[1] = 1;
  for (i=0; i<numNonExclusivelyOwned; i++) {
    vtxwgt[ncon*(i+1)] = 1;
    if (ncon>1) vtxwgt[ncon*(i+1)+1] = 0;
  }

  if (viewer) {
    ierr = PetscViewerASCIIPrintf(viewer, "Attempt rebalancing of shared points of depth %D on interface of mesh distribution.\n", entityDepth);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Size of generated auxiliary graph: %D\n", cumSumVertices[size]);CHKERRQ(ierr);
  }
  if (parallel) {
    ierr = PetscMalloc1(4, &options);CHKERRQ(ierr);
    options[0] = 1;
    options[1] = 0; /* Verbosity */
    options[2] = 0; /* Seed */
    options[3] = PARMETIS_PSR_COUPLED; /* Seed */
    if (viewer) { ierr = PetscViewerASCIIPrintf(viewer, "Using ParMETIS to partition graph.\n");CHKERRQ(ierr); }
    if (useInitialGuess) {
      if (viewer) { ierr = PetscViewerASCIIPrintf(viewer, "Using current distribution of points as initial guess.\n");CHKERRQ(ierr); }
      PetscStackPush("ParMETIS_V3_RefineKway");
      ierr = ParMETIS_V3_RefineKway((PetscInt*)cumSumVertices, xadj, adjncy, vtxwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);
      if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in ParMETIS_V3_RefineKway()");
      PetscStackPop;
    } else {
      PetscStackPush("ParMETIS_V3_PartKway");
      ierr = ParMETIS_V3_PartKway((PetscInt*)cumSumVertices, xadj, adjncy, vtxwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, part, &comm);
      PetscStackPop;
      if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in ParMETIS_V3_PartKway()");
    }
    ierr = PetscFree(options);CHKERRQ(ierr);
  } else {
    if (viewer) { ierr = PetscViewerASCIIPrintf(viewer, "Using METIS to partition graph.\n");CHKERRQ(ierr); }
    Mat As;
    PetscInt numRows;
    PetscInt *partGlobal;
    ierr = MatCreateRedundantMatrix(A, size, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &As);CHKERRQ(ierr);

    PetscInt *numExclusivelyOwnedAll;
    ierr = PetscMalloc1(size, &numExclusivelyOwnedAll);CHKERRQ(ierr);
    numExclusivelyOwnedAll[rank] = numExclusivelyOwned;
    ierr = MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,numExclusivelyOwnedAll,1,MPIU_INT,comm);CHKERRQ(ierr);

    ierr = MatGetSize(As, &numRows, NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(numRows, &partGlobal);CHKERRQ(ierr);
    if (!rank) {
      PetscInt *adjncy_g, *xadj_g, *vtxwgt_g;
      lenadjncy = 0;

      for (i=0; i<numRows; i++) {
        PetscInt temp=0;
        ierr = MatGetRow(As, i, &temp, NULL, NULL);CHKERRQ(ierr);
        lenadjncy += temp;
        ierr = MatRestoreRow(As, i, &temp, NULL, NULL);CHKERRQ(ierr);
      }
      ierr = PetscMalloc1(lenadjncy, &adjncy_g);CHKERRQ(ierr);
      lenxadj = 1 + numRows;
      ierr = PetscMalloc1(lenxadj, &xadj_g);CHKERRQ(ierr);
      xadj_g[0] = 0;
      counter = 0;
      for (i=0; i<numRows; i++) {
        PetscInt        temp=0;
        const PetscInt *cols;
        ierr = MatGetRow(As, i, &temp, &cols, NULL);CHKERRQ(ierr);
        ierr = PetscArraycpy(&adjncy_g[counter], cols, temp);CHKERRQ(ierr);
        counter += temp;
        xadj_g[i+1] = counter;
        ierr = MatRestoreRow(As, i, &temp, &cols, NULL);CHKERRQ(ierr);
      }
      ierr = PetscMalloc1(2*numRows, &vtxwgt_g);CHKERRQ(ierr);
      for (i=0; i<size; i++){
        vtxwgt_g[ncon*cumSumVertices[i]] = numExclusivelyOwnedAll[i];
        if (ncon>1) vtxwgt_g[ncon*cumSumVertices[i]+1] = 1;
        for (j=cumSumVertices[i]+1; j<cumSumVertices[i+1]; j++) {
          vtxwgt_g[ncon*j] = 1;
          if (ncon>1) vtxwgt_g[2*j+1] = 0;
        }
      }
      ierr = PetscMalloc1(64, &options);CHKERRQ(ierr);
      ierr = METIS_SetDefaultOptions(options); /* initialize all defaults */
      if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_SetDefaultOptions()");
      options[METIS_OPTION_CONTIG] = 1;
      PetscStackPush("METIS_PartGraphKway");
      ierr = METIS_PartGraphKway(&numRows, &ncon, xadj_g, adjncy_g, vtxwgt_g, NULL, NULL, &nparts, tpwgts, ubvec, options, &edgecut, partGlobal);
      PetscStackPop;
      if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_PartGraphKway()");
      ierr = PetscFree(options);CHKERRQ(ierr);
      ierr = PetscFree(xadj_g);CHKERRQ(ierr);
      ierr = PetscFree(adjncy_g);CHKERRQ(ierr);
      ierr = PetscFree(vtxwgt_g);CHKERRQ(ierr);
    }
    ierr = PetscFree(numExclusivelyOwnedAll);CHKERRQ(ierr);

    /* Now scatter the parts array. */
    {
      PetscMPIInt *counts, *mpiCumSumVertices;
      ierr = PetscMalloc1(size, &counts);CHKERRQ(ierr);
      ierr = PetscMalloc1(size+1, &mpiCumSumVertices);CHKERRQ(ierr);
      for (i=0; i<size; i++) {
        ierr = PetscMPIIntCast(cumSumVertices[i+1] - cumSumVertices[i], &(counts[i]));CHKERRQ(ierr);
      }
      for (i=0; i<=size; i++) {
        ierr = PetscMPIIntCast(cumSumVertices[i], &(mpiCumSumVertices[i]));CHKERRQ(ierr);
      }
      ierr = MPI_Scatterv(partGlobal, counts, mpiCumSumVertices, MPIU_INT, part, counts[rank], MPIU_INT, 0, comm);CHKERRQ(ierr);
      ierr = PetscFree(counts);CHKERRQ(ierr);
      ierr = PetscFree(mpiCumSumVertices);CHKERRQ(ierr);
    }

    ierr = PetscFree(partGlobal);CHKERRQ(ierr);
    ierr = MatDestroy(&As);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree(ubvec);CHKERRQ(ierr);
  ierr = PetscFree(tpwgts);CHKERRQ(ierr);

  /* Now rename the result so that the vertex resembling the exclusively owned points stays on the same rank */

  ierr = PetscMalloc1(size, &firstVertices);CHKERRQ(ierr);
  ierr = PetscMalloc1(size, &renumbering);CHKERRQ(ierr);
  firstVertices[rank] = part[0];
  ierr = MPI_Allgather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,firstVertices,1,MPIU_INT,comm);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    renumbering[firstVertices[i]] = i;
  }
  for (i=0; i<cumSumVertices[rank+1]-cumSumVertices[rank]; i++) {
    part[i] = renumbering[part[i]];
  }
  /* Check if the renumbering worked (this can fail when ParMETIS gives fewer partitions than there are processes) */
  failed = (PetscInt)(part[0] != rank);
  ierr = MPI_Allreduce(&failed, &failedGlobal, 1, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);

  ierr = PetscFree(firstVertices);CHKERRQ(ierr);
  ierr = PetscFree(renumbering);CHKERRQ(ierr);

  if (failedGlobal > 0) {
    ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
    ierr = PetscFree(xadj);CHKERRQ(ierr);
    ierr = PetscFree(adjncy);CHKERRQ(ierr);
    ierr = PetscFree(vtxwgt);CHKERRQ(ierr);
    ierr = PetscFree(toBalance);CHKERRQ(ierr);
    ierr = PetscFree(isLeaf);CHKERRQ(ierr);
    ierr = PetscFree(isNonExclusivelyOwned);CHKERRQ(ierr);
    ierr = PetscFree(isExclusivelyOwned);CHKERRQ(ierr);
    ierr = PetscFree(part);CHKERRQ(ierr);
    if (viewer) {
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    ierr = PetscLogEventEnd(DMPLEX_RebalanceSharedPoints, dm, 0, 0, 0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /*Let's check how well we did distributing points*/
  if (viewer) {
    ierr = PetscViewerASCIIPrintf(viewer, "Comparing number of owned entities of depth %D on each process before rebalancing, after rebalancing, and after consistency checks.\n", entityDepth);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Initial.     ");CHKERRQ(ierr);
    ierr = DMPlexViewDistribution(comm, cumSumVertices[rank+1]-cumSumVertices[rank], ncon, vtxwgt, NULL, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Rebalanced.  ");CHKERRQ(ierr);
    ierr = DMPlexViewDistribution(comm, cumSumVertices[rank+1]-cumSumVertices[rank], ncon, vtxwgt, part, viewer);CHKERRQ(ierr);
  }

  /* Now check that every vertex is owned by a process that it is actually connected to. */
  for (i=1; i<=numNonExclusivelyOwned; i++) {
    PetscInt loc = 0;
    ierr = PetscFindInt(cumSumVertices[part[i]], xadj[i+1]-xadj[i], &adjncy[xadj[i]], &loc);CHKERRQ(ierr);
    /* If not, then just set the owner to the original owner (hopefully a rare event, it means that a vertex has been isolated) */
    if (loc<0) {
      part[i] = rank;
    }
  }

  /* Let's see how significant the influences of the previous fixing up step was.*/
  if (viewer) {
    ierr = PetscViewerASCIIPrintf(viewer, "After.       ");CHKERRQ(ierr);
    ierr = DMPlexViewDistribution(comm, cumSumVertices[rank+1]-cumSumVertices[rank], ncon, vtxwgt, part, viewer);CHKERRQ(ierr);
  }

  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  ierr = PetscFree(xadj);CHKERRQ(ierr);
  ierr = PetscFree(adjncy);CHKERRQ(ierr);
  ierr = PetscFree(vtxwgt);CHKERRQ(ierr);

  /* Almost done, now rewrite the SF to reflect the new ownership. */
  {
    PetscInt *pointsToRewrite;
    ierr = PetscMalloc1(numNonExclusivelyOwned, &pointsToRewrite);CHKERRQ(ierr);
    counter = 0;
    for (i=0; i<pEnd-pStart; i++) {
      if (toBalance[i]) {
        if (isNonExclusivelyOwned[i]) {
          pointsToRewrite[counter] = i + pStart;
          counter++;
        }
      }
    }
    ierr = DMPlexRewriteSF(dm, numNonExclusivelyOwned, pointsToRewrite, part+1, degrees);CHKERRQ(ierr);
    ierr = PetscFree(pointsToRewrite);CHKERRQ(ierr);
  }

  ierr = PetscFree(toBalance);CHKERRQ(ierr);
  ierr = PetscFree(isLeaf);CHKERRQ(ierr);
  ierr = PetscFree(isNonExclusivelyOwned);CHKERRQ(ierr);
  ierr = PetscFree(isExclusivelyOwned);CHKERRQ(ierr);
  ierr = PetscFree(part);CHKERRQ(ierr);
  if (viewer) {
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  if (success) *success = PETSC_TRUE;
  ierr = PetscLogEventEnd(DMPLEX_RebalanceSharedPoints, dm, 0, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-parmetis.");
#endif
}
