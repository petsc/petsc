#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petsc/private/hashseti.h>

PetscClassId PETSCPARTITIONER_CLASSID = 0;

PetscFunctionList PetscPartitionerList              = NULL;
PetscBool         PetscPartitionerRegisterAllCalled = PETSC_FALSE;

PetscBool ChacoPartitionercite = PETSC_FALSE;
const char ChacoPartitionerCitation[] = "@inproceedings{Chaco95,\n"
                               "  author    = {Bruce Hendrickson and Robert Leland},\n"
                               "  title     = {A multilevel algorithm for partitioning graphs},\n"
                               "  booktitle = {Supercomputing '95: Proceedings of the 1995 ACM/IEEE Conference on Supercomputing (CDROM)},"
                               "  isbn      = {0-89791-816-9},\n"
                               "  pages     = {28},\n"
                               "  doi       = {http://doi.acm.org/10.1145/224170.224228},\n"
                               "  publisher = {ACM Press},\n"
                               "  address   = {New York},\n"
                               "  year      = {1995}\n}\n";

PetscBool ParMetisPartitionercite = PETSC_FALSE;
const char ParMetisPartitionerCitation[] = "@article{KarypisKumar98,\n"
                               "  author  = {George Karypis and Vipin Kumar},\n"
                               "  title   = {A Parallel Algorithm for Multilevel Graph Partitioning and Sparse Matrix Ordering},\n"
                               "  journal = {Journal of Parallel and Distributed Computing},\n"
                               "  volume  = {48},\n"
                               "  pages   = {71--85},\n"
                               "  year    = {1998}\n}\n";

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
  representation on the mesh.

  Level: developer

.seealso: PetscPartitionerGetType(), PetscPartitionerCreate(), DMSetAdjacency()
@*/
PetscErrorCode DMPlexCreatePartitionerGraph(DM dm, PetscInt height, PetscInt *numVertices, PetscInt **offsets, PetscInt **adjacency, IS *globalNumbering)
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
  ierr = DMPlexGetHeightStratum(dm, height, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  /* Build adjacency graph via a section/segbuffer */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscInt),1000,&adjBuffer);CHKERRQ(ierr);
  /* Always use FVM adjacency to create partitioner graph */
  ierr = DMGetBasicAdjacency(dm, &useCone, &useClosure);CHKERRQ(ierr);
  ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, &cellNumbering);CHKERRQ(ierr);
  if (globalNumbering) {
    ierr = PetscObjectReference((PetscObject)cellNumbering);CHKERRQ(ierr);
    *globalNumbering = cellNumbering;
  }
  ierr = ISGetIndices(cellNumbering, &cellNum);CHKERRQ(ierr);
  /* For all boundary faces (including faces adjacent to a ghost cell), record the local cell in adjCells
     Broadcast adjCells to remoteCells (to get cells from roots) and Reduce adjCells to remoteCells (to get cells from leaves)
   */
  ierr = PetscSFGetGraph(dm->sf, &nroots, &nleaves, &local, NULL);CHKERRQ(ierr);
  if (nroots >= 0) {
    PetscInt fStart, fEnd, f;

    ierr = PetscCalloc2(nroots, &adjCells, nroots, &remoteCells);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    for (l = 0; l < nroots; ++l) adjCells[l] = -3;
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *support;
      PetscInt        supportSize;

      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      if (supportSize == 1) adjCells[f] = cellNum[support[0]];
      else if (supportSize == 2) {
        ierr = PetscFindInt(support[0], nleaves, local, &p);CHKERRQ(ierr);
        if (p >= 0) adjCells[f] = cellNum[support[1]];
        ierr = PetscFindInt(support[1], nleaves, local, &p);CHKERRQ(ierr);
        if (p >= 0) adjCells[f] = cellNum[support[0]];
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
    const PetscInt *cone;
    PetscInt        coneSize, c;

    /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
    if (nroots > 0) {if (cellNum[p] < 0) continue;}
    /* Add remote cells */
    if (remoteCells) {
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
      for (c = 0; c < coneSize; ++c) {
        if (remoteCells[cone[c]] != -1) {
          PetscInt *PETSC_RESTRICT pBuf;

          ierr = PetscSectionAddDof(section, p, 1);CHKERRQ(ierr);
          ierr = PetscSegBufferGetInts(adjBuffer, 1, &pBuf);CHKERRQ(ierr);
          *pBuf = remoteCells[cone[c]];
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
        *pBuf = cellNum[point];
      }
    }
    (*numVertices)++;
  }
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
  if (offsets) *offsets = vOffsets;
  ierr = PetscSegBufferExtractAlloc(adjBuffer, &graph);CHKERRQ(ierr);
  ierr = ISRestoreIndices(cellNumbering, &cellNum);CHKERRQ(ierr);
  ierr = ISDestroy(&cellNumbering);CHKERRQ(ierr);
  if (adjacency) *adjacency = graph;
  /* Clean up */
  ierr = PetscSegBufferDestroy(&adjBuffer);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = PetscFree(adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateNeighborCSR - Create a mesh graph (cell-cell adjacency) in parallel CSR format.

  Collective

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
      ierr = PetscMemcpy(tmp, off, (numCells+1) * sizeof(PetscInt));CHKERRQ(ierr);
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
#if defined(PETSC_USE_DEBUG)
      for (c = 0; c < cEnd-cStart; ++c) if (tmp[c] != off[c+1]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Offset %d != %d for cell %d", tmp[c], off[c], c+cStart);
#endif
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

/*@C
  PetscPartitionerRegister - Adds a new PetscPartitioner implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Notes:
  PetscPartitionerRegister() may be called multiple times to add several user-defined PetscPartitioners

  Sample usage:
.vb
    PetscPartitionerRegister("my_part", MyPetscPartitionerCreate);
.ve

  Then, your PetscPartitioner type can be chosen with the procedural interface via
.vb
    PetscPartitionerCreate(MPI_Comm, PetscPartitioner *);
    PetscPartitionerSetType(PetscPartitioner, "my_part");
.ve
   or at runtime via the option
.vb
    -petscpartitioner_type my_part
.ve

  Level: advanced

.keywords: PetscPartitioner, register
.seealso: PetscPartitionerRegisterAll(), PetscPartitionerRegisterDestroy()

@*/
PetscErrorCode PetscPartitionerRegister(const char sname[], PetscErrorCode (*function)(PetscPartitioner))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PetscPartitionerList, sname, function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscPartitionerSetType - Builds a particular PetscPartitioner

  Collective on PetscPartitioner

  Input Parameters:
+ part - The PetscPartitioner object
- name - The kind of partitioner

  Options Database Key:
. -petscpartitioner_type <type> - Sets the PetscPartitioner type; use -help for a list of available types

  Level: intermediate

.keywords: PetscPartitioner, set, type
.seealso: PetscPartitionerGetType(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerSetType(PetscPartitioner part, PetscPartitionerType name)
{
  PetscErrorCode (*r)(PetscPartitioner);
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr = PetscObjectTypeCompare((PetscObject) part, name, &match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscPartitionerRegisterAll();CHKERRQ(ierr);
  ierr = PetscFunctionListFind(PetscPartitionerList, name, &r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscPartitioner type: %s", name);

  if (part->ops->destroy) {
    ierr              = (*part->ops->destroy)(part);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(part->ops, sizeof(struct _PetscPartitionerOps));CHKERRQ(ierr);
  ierr = (*r)(part);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) part, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscPartitionerGetType - Gets the PetscPartitioner type name (as a string) from the object.

  Not Collective

  Input Parameter:
. part - The PetscPartitioner

  Output Parameter:
. name - The PetscPartitioner type name

  Level: intermediate

.keywords: PetscPartitioner, get, type, name
.seealso: PetscPartitionerSetType(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerGetType(PetscPartitioner part, PetscPartitionerType *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidPointer(name, 2);
  ierr = PetscPartitionerRegisterAll();CHKERRQ(ierr);
  *name = ((PetscObject) part)->type_name;
  PetscFunctionReturn(0);
}

/*@C
  PetscPartitionerView - Views a PetscPartitioner

  Collective on PetscPartitioner

  Input Parameter:
+ part - the PetscPartitioner object to view
- v    - the viewer

  Level: developer

.seealso: PetscPartitionerDestroy()
@*/
PetscErrorCode PetscPartitionerView(PetscPartitioner part, PetscViewer v)
{
  PetscMPIInt    size;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (!v) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) part), &v);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject) part), &size);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "Graph Partitioner: %d MPI Process%s\n", size, size > 1 ? "es" : "");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "  type: %s\n", part->hdr.type_name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "edge cut: %D\n", part->edgeCut);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(v, "balance:  %.2g\n", part->balance);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }
  if (part->ops->view) {ierr = (*part->ops->view)(part, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerGetDefaultType(const char *currentType, const char **defaultType)
{
  PetscFunctionBegin;
  if (!currentType) {
#if defined(PETSC_HAVE_CHACO)
    *defaultType = PETSCPARTITIONERCHACO;
#elif defined(PETSC_HAVE_PARMETIS)
    *defaultType = PETSCPARTITIONERPARMETIS;
#elif defined(PETSC_HAVE_PTSCOTCH)
    *defaultType = PETSCPARTITIONERPTSCOTCH;
#else
    *defaultType = PETSCPARTITIONERSIMPLE;
#endif
  } else {
    *defaultType = currentType;
  }
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerSetFromOptions - sets parameters in a PetscPartitioner from the options database

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to set options for

  Level: developer

.seealso: PetscPartitionerView()
@*/
PetscErrorCode PetscPartitionerSetFromOptions(PetscPartitioner part)
{
  const char    *defaultType;
  char           name[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr = PetscPartitionerRegisterAll();CHKERRQ(ierr);
  ierr = PetscPartitionerGetDefaultType(((PetscObject) part)->type_name,&defaultType);CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject) part);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-petscpartitioner_type", "Graph partitioner", "PetscPartitionerSetType", PetscPartitionerList, defaultType, name, sizeof(name), &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPartitionerSetType(part, name);CHKERRQ(ierr);
  } else if (!((PetscObject) part)->type_name) {
    ierr = PetscPartitionerSetType(part, defaultType);CHKERRQ(ierr);
  }
  if (part->ops->setfromoptions) {
    ierr = (*part->ops->setfromoptions)(PetscOptionsObject,part);CHKERRQ(ierr);
  }
  ierr = PetscOptionsGetViewer(((PetscObject) part)->comm, ((PetscObject) part)->options, ((PetscObject) part)->prefix, "-petscpartitioner_view_graph", &part->viewerGraph, &part->formatGraph, &part->viewGraph);CHKERRQ(ierr);
  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) part);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscPartitionerSetUp - Construct data structures for the PetscPartitioner

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to setup

  Level: developer

.seealso: PetscPartitionerView(), PetscPartitionerDestroy()
@*/
PetscErrorCode PetscPartitionerSetUp(PetscPartitioner part)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (part->ops->setup) {ierr = (*part->ops->setup)(part);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerDestroy - Destroys a PetscPartitioner object

  Collective on PetscPartitioner

  Input Parameter:
. part - the PetscPartitioner object to destroy

  Level: developer

.seealso: PetscPartitionerView()
@*/
PetscErrorCode PetscPartitionerDestroy(PetscPartitioner *part)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*part) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*part), PETSCPARTITIONER_CLASSID, 1);

  if (--((PetscObject)(*part))->refct > 0) {*part = 0; PetscFunctionReturn(0);}
  ((PetscObject) (*part))->refct = 0;

  ierr = PetscViewerDestroy(&(*part)->viewerGraph);CHKERRQ(ierr);
  if ((*part)->ops->destroy) {ierr = (*(*part)->ops->destroy)(*part);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerCreate - Creates an empty PetscPartitioner object. The type can then be set with PetscPartitionerSetType().

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the PetscPartitioner object

  Output Parameter:
. part - The PetscPartitioner object

  Level: beginner

.seealso: PetscPartitionerSetType(), PETSCPARTITIONERCHACO, PETSCPARTITIONERPARMETIS, PETSCPARTITIONERSHELL, PETSCPARTITIONERSIMPLE, PETSCPARTITIONERGATHER
@*/
PetscErrorCode PetscPartitionerCreate(MPI_Comm comm, PetscPartitioner *part)
{
  PetscPartitioner p;
  const char       *partitionerType = NULL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidPointer(part, 2);
  *part = NULL;
  ierr = DMInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(p, PETSCPARTITIONER_CLASSID, "PetscPartitioner", "Graph Partitioner", "PetscPartitioner", comm, PetscPartitionerDestroy, PetscPartitionerView);CHKERRQ(ierr);
  ierr = PetscPartitionerGetDefaultType(NULL,&partitionerType);CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(p,partitionerType);CHKERRQ(ierr);

  p->edgeCut = 0;
  p->balance = 0.0;

  *part = p;
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerPartition - Create a non-overlapping partition of the cells in the mesh

  Collective on DM

  Input Parameters:
+ part    - The PetscPartitioner
- dm      - The mesh DM

  Output Parameters:
+ partSection     - The PetscSection giving the division of points by partition
- partition       - The list of points by partition

  Options Database:
. -petscpartitioner_view_graph - View the graph we are partitioning

  Note: Instead of cells, points at a given height can be partitioned by calling PetscPartitionerSetPointHeight()

  Level: developer

.seealso DMPlexDistribute(), PetscPartitionerSetPointHeight(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerPartition(PetscPartitioner part, DM dm, PetscSection partSection, IS *partition)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(partSection, PETSC_SECTION_CLASSID, 4);
  PetscValidPointer(partition, 5);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) part), &size);CHKERRQ(ierr);
  if (size == 1) {
    PetscInt *points;
    PetscInt  cStart, cEnd, c;

    ierr = DMPlexGetHeightStratum(dm, part->height, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(partSection, 0, size);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(partSection, 0, cEnd-cStart);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
    ierr = PetscMalloc1(cEnd-cStart, &points);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) points[c] = c;
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject) part), cEnd-cStart, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (part->height == 0) {
    PetscInt numVertices;
    PetscInt *start     = NULL;
    PetscInt *adjacency = NULL;
    IS       globalNumbering;

    ierr = DMPlexCreatePartitionerGraph(dm, 0, &numVertices, &start, &adjacency, &globalNumbering);CHKERRQ(ierr);
    if (part->viewGraph) {
      PetscViewer viewer = part->viewerGraph;
      PetscBool   isascii;
      PetscInt    v, i;
      PetscMPIInt rank;

      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
      if (isascii) {
        ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]Nv: %D\n", rank, numVertices);CHKERRQ(ierr);
        for (v = 0; v < numVertices; ++v) {
          const PetscInt s = start[v];
          const PetscInt e = start[v+1];

          ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d]  ", rank);CHKERRQ(ierr);
          for (i = s; i < e; ++i) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%D ", adjacency[i]);CHKERRQ(ierr);}
          ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%D-%D)\n", s, e);CHKERRQ(ierr);
        }
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
      }
    }
    if (!part->ops->partition) SETERRQ(PetscObjectComm((PetscObject) part), PETSC_ERR_ARG_WRONGSTATE, "PetscPartitioner has no type");
    ierr = (*part->ops->partition)(part, dm, size, numVertices, start, adjacency, partSection, partition);CHKERRQ(ierr);
    ierr = PetscFree(start);CHKERRQ(ierr);
    ierr = PetscFree(adjacency);CHKERRQ(ierr);
    if (globalNumbering) { /* partition is wrt global unique numbering: change this to be wrt local numbering */
      const PetscInt *globalNum;
      const PetscInt *partIdx;
      PetscInt *map, cStart, cEnd;
      PetscInt *adjusted, i, localSize, offset;
      IS    newPartition;

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
  ierr = PetscPartitionerViewFromOptions(part, NULL, "-petscpartitioner_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerDestroy_Shell(PetscPartitioner part)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscSectionDestroy(&p->section);CHKERRQ(ierr);
  ierr = ISDestroy(&p->partition);CHKERRQ(ierr);
  ierr = PetscFree(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Shell_Ascii(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (p->random) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "using random partition\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Shell(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_Shell_Ascii(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_Shell(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "PetscPartitioner Shell Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscpartitioner_shell_random", "Use a random partition", "PetscPartitionerView", PETSC_FALSE, &p->random, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Shell(PetscPartitioner part, DM dm, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection partSection, IS *partition)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscInt                np;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (p->random) {
    PetscRandom r;
    PetscInt   *sizes, *points, v, p;
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(r, 0.0, (PetscScalar) nparts);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);
    ierr = PetscCalloc2(nparts, &sizes, numVertices, &points);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {points[v] = v;}
    for (p = 0; p < nparts; ++p) {sizes[p] = numVertices/nparts + (PetscInt) (p < numVertices % nparts);}
    for (v = numVertices-1; v > 0; --v) {
      PetscReal val;
      PetscInt  w, tmp;

      ierr = PetscRandomSetInterval(r, 0.0, (PetscScalar) (v+1));CHKERRQ(ierr);
      ierr = PetscRandomGetValueReal(r, &val);CHKERRQ(ierr);
      w    = PetscFloorReal(val);
      tmp       = points[v];
      points[v] = points[w];
      points[w] = tmp;
    }
    ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);
    ierr = PetscPartitionerShellSetPartition(part, nparts, sizes, points);CHKERRQ(ierr);
    ierr = PetscFree2(sizes, points);CHKERRQ(ierr);
  }
  if (!p->section) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Shell partitioner information not provided. Please call PetscPartitionerShellSetPartition()");
  ierr = PetscSectionGetChart(p->section, NULL, &np);CHKERRQ(ierr);
  if (nparts != np) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of requested partitions %d != configured partitions %d", nparts, np);
  ierr = ISGetLocalSize(p->partition, &np);CHKERRQ(ierr);
  if (numVertices != np) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of input vertices %d != configured vertices %d", numVertices, np);
  ierr = PetscSectionCopy(p->section, partSection);CHKERRQ(ierr);
  *partition = p->partition;
  ierr = PetscObjectReference((PetscObject) p->partition);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerInitialize_Shell(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->ops->view           = PetscPartitionerView_Shell;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_Shell;
  part->ops->destroy        = PetscPartitionerDestroy_Shell;
  part->ops->partition      = PetscPartitionerPartition_Shell;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERSHELL = "shell" - A PetscPartitioner object

  Level: intermediate

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Shell(PetscPartitioner part)
{
  PetscPartitioner_Shell *p;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr       = PetscNewLog(part, &p);CHKERRQ(ierr);
  part->data = p;

  ierr = PetscPartitionerInitialize_Shell(part);CHKERRQ(ierr);
  p->random = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscPartitionerShellSetPartition - Set an artifical partition for a mesh

  Collective on Part

  Input Parameters:
+ part     - The PetscPartitioner
. size - The number of partitions
. sizes    - array of size size (or NULL) providing the number of points in each partition
- points   - array of size sum(sizes) (may be NULL iff sizes is NULL), a permutation of the points that groups those assigned to each partition in order (i.e., partition 0 first, partition 1 next, etc.)

  Level: developer

  Notes:

    It is safe to free the sizes and points arrays after use in this routine.

.seealso DMPlexDistribute(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerShellSetPartition(PetscPartitioner part, PetscInt size, const PetscInt sizes[], const PetscInt points[])
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;
  PetscInt                proc, numPoints;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (sizes)  {PetscValidPointer(sizes, 3);}
  if (points) {PetscValidPointer(points, 4);}
  ierr = PetscSectionDestroy(&p->section);CHKERRQ(ierr);
  ierr = ISDestroy(&p->partition);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) part), &p->section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(p->section, 0, size);CHKERRQ(ierr);
  if (sizes) {
    for (proc = 0; proc < size; ++proc) {
      ierr = PetscSectionSetDof(p->section, proc, sizes[proc]);CHKERRQ(ierr);
    }
  }
  ierr = PetscSectionSetUp(p->section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(p->section, &numPoints);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject) part), numPoints, points, PETSC_COPY_VALUES, &p->partition);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerShellSetRandom - Set the flag to use a random partition

  Collective on Part

  Input Parameters:
+ part   - The PetscPartitioner
- random - The flag to use a random partition

  Level: intermediate

.seealso PetscPartitionerShellGetRandom(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerShellSetRandom(PetscPartitioner part, PetscBool random)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  p->random = random;
  PetscFunctionReturn(0);
}

/*@
  PetscPartitionerShellGetRandom - get the flag to use a random partition

  Collective on Part

  Input Parameter:
. part   - The PetscPartitioner

  Output Parameter
. random - The flag to use a random partition

  Level: intermediate

.seealso PetscPartitionerShellSetRandom(), PetscPartitionerCreate()
@*/
PetscErrorCode PetscPartitionerShellGetRandom(PetscPartitioner part, PetscBool *random)
{
  PetscPartitioner_Shell *p = (PetscPartitioner_Shell *) part->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidPointer(random, 2);
  *random = p->random;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerDestroy_Simple(PetscPartitioner part)
{
  PetscPartitioner_Simple *p = (PetscPartitioner_Simple *) part->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscFree(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Simple_Ascii(PetscPartitioner part, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Simple(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_Simple_Ascii(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Simple(PetscPartitioner part, DM dm, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection partSection, IS *partition)
{
  MPI_Comm       comm;
  PetscInt       np;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  comm = PetscObjectComm((PetscObject)dm);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(partSection, 0, nparts);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, numVertices, 0, 1, partition);CHKERRQ(ierr);
  if (size == 1) {
    for (np = 0; np < nparts; ++np) {ierr = PetscSectionSetDof(partSection, np, numVertices/nparts + ((numVertices % nparts) > np));CHKERRQ(ierr);}
  }
  else {
    PetscMPIInt rank;
    PetscInt nvGlobal, *offsets, myFirst, myLast;

    ierr = PetscMalloc1(size+1,&offsets);CHKERRQ(ierr);
    offsets[0] = 0;
    ierr = MPI_Allgather(&numVertices,1,MPIU_INT,&offsets[1],1,MPIU_INT,comm);CHKERRQ(ierr);
    for (np = 2; np <= size; np++) {
      offsets[np] += offsets[np-1];
    }
    nvGlobal = offsets[size];
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    myFirst = offsets[rank];
    myLast  = offsets[rank + 1] - 1;
    ierr = PetscFree(offsets);CHKERRQ(ierr);
    if (numVertices) {
      PetscInt firstPart = 0, firstLargePart = 0;
      PetscInt lastPart = 0, lastLargePart = 0;
      PetscInt rem = nvGlobal % nparts;
      PetscInt pSmall = nvGlobal/nparts;
      PetscInt pBig = nvGlobal/nparts + 1;


      if (rem) {
        firstLargePart = myFirst / pBig;
        lastLargePart  = myLast  / pBig;

        if (firstLargePart < rem) {
          firstPart = firstLargePart;
        }
        else {
          firstPart = rem + (myFirst - (rem * pBig)) / pSmall;
        }
        if (lastLargePart < rem) {
          lastPart = lastLargePart;
        }
        else {
          lastPart = rem + (myLast - (rem * pBig)) / pSmall;
        }
      }
      else {
        firstPart = myFirst / (nvGlobal/nparts);
        lastPart  = myLast  / (nvGlobal/nparts);
      }

      for (np = firstPart; np <= lastPart; np++) {
        PetscInt PartStart =  np    * (nvGlobal/nparts) + PetscMin(nvGlobal % nparts,np);
        PetscInt PartEnd   = (np+1) * (nvGlobal/nparts) + PetscMin(nvGlobal % nparts,np+1);

        PartStart = PetscMax(PartStart,myFirst);
        PartEnd   = PetscMin(PartEnd,myLast+1);
        ierr = PetscSectionSetDof(partSection,np,PartEnd-PartStart);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerInitialize_Simple(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->ops->view      = PetscPartitionerView_Simple;
  part->ops->destroy   = PetscPartitionerDestroy_Simple;
  part->ops->partition = PetscPartitionerPartition_Simple;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERSIMPLE = "simple" - A PetscPartitioner object

  Level: intermediate

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Simple(PetscPartitioner part)
{
  PetscPartitioner_Simple *p;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr       = PetscNewLog(part, &p);CHKERRQ(ierr);
  part->data = p;

  ierr = PetscPartitionerInitialize_Simple(part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerDestroy_Gather(PetscPartitioner part)
{
  PetscPartitioner_Gather *p = (PetscPartitioner_Gather *) part->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscFree(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Gather_Ascii(PetscPartitioner part, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Gather(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_Gather_Ascii(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_Gather(PetscPartitioner part, DM dm, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection partSection, IS *partition)
{
  PetscInt       np;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSectionSetChart(partSection, 0, nparts);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, numVertices, 0, 1, partition);CHKERRQ(ierr);
  ierr = PetscSectionSetDof(partSection,0,numVertices);CHKERRQ(ierr);
  for (np = 1; np < nparts; ++np) {ierr = PetscSectionSetDof(partSection, np, 0);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerInitialize_Gather(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->ops->view      = PetscPartitionerView_Gather;
  part->ops->destroy   = PetscPartitionerDestroy_Gather;
  part->ops->partition = PetscPartitionerPartition_Gather;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERGATHER = "gather" - A PetscPartitioner object

  Level: intermediate

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Gather(PetscPartitioner part)
{
  PetscPartitioner_Gather *p;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr       = PetscNewLog(part, &p);CHKERRQ(ierr);
  part->data = p;

  ierr = PetscPartitionerInitialize_Gather(part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode PetscPartitionerDestroy_Chaco(PetscPartitioner part)
{
  PetscPartitioner_Chaco *p = (PetscPartitioner_Chaco *) part->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscFree(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Chaco_Ascii(PetscPartitioner part, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_Chaco(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_Chaco_Ascii(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CHACO)
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_CHACO_INT_ASSIGNMENT)
#include <chaco.h>
#else
/* Older versions of Chaco do not have an include file */
PETSC_EXTERN int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);
#endif
extern int FREE_GRAPH;
#endif

static PetscErrorCode PetscPartitionerPartition_Chaco(PetscPartitioner part, DM dm, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection partSection, IS *partition)
{
#if defined(PETSC_HAVE_CHACO)
  enum {DEFAULT_METHOD = 1, INERTIAL_METHOD = 3};
  MPI_Comm       comm;
  int            nvtxs          = numVertices; /* number of vertices in full graph */
  int           *vwgts          = NULL;   /* weights for all vertices */
  float         *ewgts          = NULL;   /* weights for all edges */
  float         *x              = NULL, *y = NULL, *z = NULL; /* coordinates for inertial method */
  char          *outassignname  = NULL;   /*  name of assignment output file */
  char          *outfilename    = NULL;   /* output file name */
  int            architecture   = 1;      /* 0 => hypercube, d => d-dimensional mesh */
  int            ndims_tot      = 0;      /* total number of cube dimensions to divide */
  int            mesh_dims[3];            /* dimensions of mesh of processors */
  double        *goal          = NULL;    /* desired set sizes for each set */
  int            global_method = 1;       /* global partitioning algorithm */
  int            local_method  = 1;       /* local partitioning algorithm */
  int            rqi_flag      = 0;       /* should I use RQI/Symmlq eigensolver? */
  int            vmax          = 200;     /* how many vertices to coarsen down to? */
  int            ndims         = 1;       /* number of eigenvectors (2^d sets) */
  double         eigtol        = 0.001;   /* tolerance on eigenvectors */
  long           seed          = 123636512; /* for random graph mutations */
#if defined(PETSC_HAVE_CHACO_INT_ASSIGNMENT)
  int           *assignment;              /* Output partition */
#else
  short int     *assignment;              /* Output partition */
#endif
  int            fd_stdout, fd_pipe[2];
  PetscInt      *points;
  int            i, v, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
#if defined (PETSC_USE_DEBUG)
  {
    int ival,isum;
    PetscBool distributed;

    ival = (numVertices > 0);
    ierr = MPI_Allreduce(&ival, &isum, 1, MPI_INT, MPI_SUM, comm);CHKERRQ(ierr);
    distributed = (isum > 1) ? PETSC_TRUE : PETSC_FALSE;
    if (distributed) SETERRQ(comm, PETSC_ERR_SUP, "Chaco cannot partition a distributed graph");
  }
#endif
  if (!numVertices) {
    ierr = PetscSectionSetChart(partSection, 0, nparts);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
    ierr = ISCreateGeneral(comm, 0, NULL, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
  for (i = 0; i < start[numVertices]; ++i) ++adjacency[i];

  if (global_method == INERTIAL_METHOD) {
    /* manager.createCellCoordinates(nvtxs, &x, &y, &z); */
    SETERRQ(comm, PETSC_ERR_SUP, "Inertial partitioning not yet supported");
  }
  mesh_dims[0] = nparts;
  mesh_dims[1] = 1;
  mesh_dims[2] = 1;
  ierr = PetscMalloc1(nvtxs, &assignment);CHKERRQ(ierr);
  /* Chaco outputs to stdout. We redirect this to a buffer. */
  /* TODO: check error codes for UNIX calls */
#if defined(PETSC_HAVE_UNISTD_H)
  {
    int piperet;
    piperet = pipe(fd_pipe);
    if (piperet) SETERRQ(comm,PETSC_ERR_SYS,"Could not create pipe");
    fd_stdout = dup(1);
    close(1);
    dup2(fd_pipe[1], 1);
  }
#endif
  ierr = interface(nvtxs, (int*) start, (int*) adjacency, vwgts, ewgts, x, y, z, outassignname, outfilename,
                   assignment, architecture, ndims_tot, mesh_dims, goal, global_method, local_method, rqi_flag,
                   vmax, ndims, eigtol, seed);
#if defined(PETSC_HAVE_UNISTD_H)
  {
    char msgLog[10000];
    int  count;

    fflush(stdout);
    count = read(fd_pipe[0], msgLog, (10000-1)*sizeof(char));
    if (count < 0) count = 0;
    msgLog[count] = 0;
    close(1);
    dup2(fd_stdout, 1);
    close(fd_stdout);
    close(fd_pipe[0]);
    close(fd_pipe[1]);
    if (ierr) SETERRQ1(comm, PETSC_ERR_LIB, "Error in Chaco library: %s", msgLog);
  }
#else
  if (ierr) SETERRQ1(comm, PETSC_ERR_LIB, "Error in Chaco library: %s", "error in stdout");
#endif
  /* Convert to PetscSection+IS */
  ierr = PetscSectionSetChart(partSection, 0, nparts);CHKERRQ(ierr);
  for (v = 0; v < nvtxs; ++v) {
    ierr = PetscSectionAddDof(partSection, assignment[v], 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
  ierr = PetscMalloc1(nvtxs, &points);CHKERRQ(ierr);
  for (p = 0, i = 0; p < nparts; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  if (i != nvtxs) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
  ierr = ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  if (global_method == INERTIAL_METHOD) {
    /* manager.destroyCellCoordinates(nvtxs, &x, &y, &z); */
  }
  ierr = PetscFree(assignment);CHKERRQ(ierr);
  for (i = 0; i < start[numVertices]; ++i) --adjacency[i];
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject) part), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-chaco.");
#endif
}

static PetscErrorCode PetscPartitionerInitialize_Chaco(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->ops->view      = PetscPartitionerView_Chaco;
  part->ops->destroy   = PetscPartitionerDestroy_Chaco;
  part->ops->partition = PetscPartitionerPartition_Chaco;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERCHACO = "chaco" - A PetscPartitioner object using the Chaco library

  Level: intermediate

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Chaco(PetscPartitioner part)
{
  PetscPartitioner_Chaco *p;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr       = PetscNewLog(part, &p);CHKERRQ(ierr);
  part->data = p;

  ierr = PetscPartitionerInitialize_Chaco(part);CHKERRQ(ierr);
  ierr = PetscCitationsRegister(ChacoPartitionerCitation, &ChacoPartitionercite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const char *ptypes[] = {"kway", "rb"};

static PetscErrorCode PetscPartitionerDestroy_ParMetis(PetscPartitioner part)
{
  PetscPartitioner_ParMetis *p = (PetscPartitioner_ParMetis *) part->data;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = PetscFree(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_ParMetis_Ascii(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_ParMetis *p = (PetscPartitioner_ParMetis *) part->data;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "ParMetis type: %s\n", ptypes[p->ptype]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "load imbalance ratio %g\n", (double) p->imbalanceRatio);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "debug flag %D\n", p->debugFlag);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_ParMetis(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_ParMetis_Ascii(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_ParMetis(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_ParMetis *p = (PetscPartitioner_ParMetis *) part->data;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "PetscPartitioner ParMetis Options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-petscpartitioner_parmetis_type", "Partitioning method", "", ptypes, 2, ptypes[p->ptype], &p->ptype, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-petscpartitioner_parmetis_imbalance_ratio", "Load imbalance ratio limit", "", p->imbalanceRatio, &p->imbalanceRatio, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-petscpartitioner_parmetis_debug", "Debugging flag", "", p->debugFlag, &p->debugFlag, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_PARMETIS)
#include <parmetis.h>
#endif

static PetscErrorCode PetscPartitionerPartition_ParMetis(PetscPartitioner part, DM dm, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection partSection, IS *partition)
{
#if defined(PETSC_HAVE_PARMETIS)
  PetscPartitioner_ParMetis *pm = (PetscPartitioner_ParMetis *) part->data;
  MPI_Comm       comm;
  PetscSection   section;
  PetscInt       nvtxs       = numVertices; /* The number of vertices in full graph */
  PetscInt      *vtxdist;                   /* Distribution of vertices across processes */
  PetscInt      *xadj        = start;       /* Start of edge list for each vertex */
  PetscInt      *adjncy      = adjacency;   /* Edge lists for all vertices */
  PetscInt      *vwgt        = NULL;        /* Vertex weights */
  PetscInt      *adjwgt      = NULL;        /* Edge weights */
  PetscInt       wgtflag     = 0;           /* Indicates which weights are present */
  PetscInt       numflag     = 0;           /* Indicates initial offset (0 or 1) */
  PetscInt       ncon        = 1;           /* The number of weights per vertex */
  PetscInt       metis_ptype = pm->ptype;   /* kway or recursive bisection */
  real_t        *tpwgts;                    /* The fraction of vertex weights assigned to each partition */
  real_t        *ubvec;                     /* The balance intolerance for vertex weights */
  PetscInt       options[64];               /* Options */
  /* Outputs */
  PetscInt       v, i, *assignment, *points;
  PetscMPIInt    size, rank, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) part, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Calculate vertex distribution */
  ierr = PetscMalloc5(size+1,&vtxdist,nparts*ncon,&tpwgts,ncon,&ubvec,nvtxs,&assignment,nvtxs,&vwgt);CHKERRQ(ierr);
  vtxdist[0] = 0;
  ierr = MPI_Allgather(&nvtxs, 1, MPIU_INT, &vtxdist[1], 1, MPIU_INT, comm);CHKERRQ(ierr);
  for (p = 2; p <= size; ++p) {
    vtxdist[p] += vtxdist[p-1];
  }
  /* Calculate partition weights */
  for (p = 0; p < nparts; ++p) {
    tpwgts[p] = 1.0/nparts;
  }
  ubvec[0] = pm->imbalanceRatio;
  /* Weight cells by dofs on cell by default */
  ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
  if (section) {
    PetscInt cStart, cEnd, dof;

    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    for (v = cStart; v < cEnd; ++v) {
      ierr = PetscSectionGetDof(section, v, &dof);CHKERRQ(ierr);
      /* WARNING: Assumes that meshes with overlap have the overlapped cells at the end of the stratum. */
      /* To do this properly, we should use the cell numbering created in DMPlexCreatePartitionerGraph. */
      if (v-cStart < numVertices) vwgt[v-cStart] = PetscMax(dof, 1);
    }
  } else {
    for (v = 0; v < nvtxs; ++v) vwgt[v] = 1;
  }
  wgtflag |= 2; /* have weights on graph vertices */

  if (nparts == 1) {
    ierr = PetscMemzero(assignment, nvtxs * sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    for (p = 0; !vtxdist[p+1] && p < size; ++p);
    if (vtxdist[p+1] == vtxdist[size]) {
      if (rank == p) {
        ierr = METIS_SetDefaultOptions(options); /* initialize all defaults */
        if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_SetDefaultOptions()");
        if (metis_ptype == 1) {
          PetscStackPush("METIS_PartGraphRecursive");
          ierr = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt, &nparts, tpwgts, ubvec, options, &part->edgeCut, assignment);
          PetscStackPop;
          if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_PartGraphRecursive()");
        } else {
          /*
           It would be nice to activate the two options below, but they would need some actual testing.
           - Turning on these options may exercise path of the METIS code that have bugs and may break production runs.
           - If CONTIG is set to 1, METIS will exit with error if the graph is disconnected, despite the manual saying the option is ignored in such case.
          */
          /* options[METIS_OPTION_CONTIG]  = 1; */ /* try to produce partitions that are contiguous */
          /* options[METIS_OPTION_MINCONN] = 1; */ /* minimize the maximum degree of the subdomain graph */
          PetscStackPush("METIS_PartGraphKway");
          ierr = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt, &nparts, tpwgts, ubvec, options, &part->edgeCut, assignment);
          PetscStackPop;
          if (ierr != METIS_OK) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in METIS_PartGraphKway()");
        }
      }
    } else {
      options[0] = 1;
      options[1] = pm->debugFlag;
      PetscStackPush("ParMETIS_V3_PartKway");
      ierr = ParMETIS_V3_PartKway(vtxdist, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &part->edgeCut, assignment, &comm);
      PetscStackPop;
      if (ierr != METIS_OK) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Error %d in ParMETIS_V3_PartKway()", ierr);
    }
  }
  /* Convert to PetscSection+IS */
  ierr = PetscSectionSetChart(partSection, 0, nparts);CHKERRQ(ierr);
  for (v = 0; v < nvtxs; ++v) {ierr = PetscSectionAddDof(partSection, assignment[v], 1);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
  ierr = PetscMalloc1(nvtxs, &points);CHKERRQ(ierr);
  for (p = 0, i = 0; p < nparts; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  if (i != nvtxs) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
  ierr = ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);
  ierr = PetscFree5(vtxdist,tpwgts,ubvec,assignment,vwgt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject) part), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-parmetis.");
#endif
}

static PetscErrorCode PetscPartitionerInitialize_ParMetis(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->ops->view           = PetscPartitionerView_ParMetis;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_ParMetis;
  part->ops->destroy        = PetscPartitionerDestroy_ParMetis;
  part->ops->partition      = PetscPartitionerPartition_ParMetis;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERPARMETIS = "parmetis" - A PetscPartitioner object using the ParMetis library

  Level: intermediate

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_ParMetis(PetscPartitioner part)
{
  PetscPartitioner_ParMetis *p;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr       = PetscNewLog(part, &p);CHKERRQ(ierr);
  part->data = p;

  p->ptype          = 0;
  p->imbalanceRatio = 1.05;
  p->debugFlag      = 0;

  ierr = PetscPartitionerInitialize_ParMetis(part);CHKERRQ(ierr);
  ierr = PetscCitationsRegister(ParMetisPartitionerCitation, &ParMetisPartitionercite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscBool PTScotchPartitionercite = PETSC_FALSE;
const char PTScotchPartitionerCitation[] =
  "@article{PTSCOTCH,\n"
  "  author  = {C. Chevalier and F. Pellegrini},\n"
  "  title   = {{PT-SCOTCH}: a tool for efficient parallel graph ordering},\n"
  "  journal = {Parallel Computing},\n"
  "  volume  = {34},\n"
  "  number  = {6},\n"
  "  pages   = {318--331},\n"
  "  year    = {2008},\n"
  "  doi     = {https://doi.org/10.1016/j.parco.2007.12.001}\n"
  "}\n";

typedef struct {
  PetscInt  strategy;
  PetscReal imbalance;
} PetscPartitioner_PTScotch;

static const char *const
PTScotchStrategyList[] = {
  "DEFAULT",
  "QUALITY",
  "SPEED",
  "BALANCE",
  "SAFETY",
  "SCALABILITY",
  "RECURSIVE",
  "REMAP"
};

#if defined(PETSC_HAVE_PTSCOTCH)

EXTERN_C_BEGIN
#include <ptscotch.h>
EXTERN_C_END

#define CHKERRPTSCOTCH(ierr) do { if (ierr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error calling PT-Scotch library"); } while(0)

static int PTScotch_Strategy(PetscInt strategy)
{
  switch (strategy) {
  case  0: return SCOTCH_STRATDEFAULT;
  case  1: return SCOTCH_STRATQUALITY;
  case  2: return SCOTCH_STRATSPEED;
  case  3: return SCOTCH_STRATBALANCE;
  case  4: return SCOTCH_STRATSAFETY;
  case  5: return SCOTCH_STRATSCALABILITY;
  case  6: return SCOTCH_STRATRECURSIVE;
  case  7: return SCOTCH_STRATREMAP;
  default: return SCOTCH_STRATDEFAULT;
  }
}


static PetscErrorCode PTScotch_PartGraph_Seq(SCOTCH_Num strategy, double imbalance, SCOTCH_Num n, SCOTCH_Num xadj[], SCOTCH_Num adjncy[],
                                             SCOTCH_Num vtxwgt[], SCOTCH_Num adjwgt[], SCOTCH_Num nparts, SCOTCH_Num part[])
{
  SCOTCH_Graph   grafdat;
  SCOTCH_Strat   stradat;
  SCOTCH_Num     vertnbr = n;
  SCOTCH_Num     edgenbr = xadj[n];
  SCOTCH_Num*    velotab = vtxwgt;
  SCOTCH_Num*    edlotab = adjwgt;
  SCOTCH_Num     flagval = strategy;
  double         kbalval = imbalance;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  {
    PetscBool flg = PETSC_TRUE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-petscpartititoner_ptscotch_vertex_weight", &flg, NULL);CHKERRQ(ierr);
    if (!flg) velotab = NULL;
  }
  ierr = SCOTCH_graphInit(&grafdat);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_graphBuild(&grafdat, 0, vertnbr, xadj, xadj + 1, velotab, NULL, edgenbr, adjncy, edlotab);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_stratInit(&stradat);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_stratGraphMapBuild(&stradat, flagval, nparts, kbalval);CHKERRPTSCOTCH(ierr);
#if defined(PETSC_USE_DEBUG)
  ierr = SCOTCH_graphCheck(&grafdat);CHKERRPTSCOTCH(ierr);
#endif
  ierr = SCOTCH_graphPart(&grafdat, nparts, &stradat, part);CHKERRPTSCOTCH(ierr);
  SCOTCH_stratExit(&stradat);
  SCOTCH_graphExit(&grafdat);
  PetscFunctionReturn(0);
}

static PetscErrorCode PTScotch_PartGraph_MPI(SCOTCH_Num strategy, double imbalance, SCOTCH_Num vtxdist[], SCOTCH_Num xadj[], SCOTCH_Num adjncy[],
                                             SCOTCH_Num vtxwgt[], SCOTCH_Num adjwgt[], SCOTCH_Num nparts, SCOTCH_Num part[], MPI_Comm comm)
{
  PetscMPIInt     procglbnbr;
  PetscMPIInt     proclocnum;
  SCOTCH_Arch     archdat;
  SCOTCH_Dgraph   grafdat;
  SCOTCH_Dmapping mappdat;
  SCOTCH_Strat    stradat;
  SCOTCH_Num      vertlocnbr;
  SCOTCH_Num      edgelocnbr;
  SCOTCH_Num*     veloloctab = vtxwgt;
  SCOTCH_Num*     edloloctab = adjwgt;
  SCOTCH_Num      flagval = strategy;
  double          kbalval = imbalance;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  {
    PetscBool flg = PETSC_TRUE;
    ierr = PetscOptionsGetBool(NULL, NULL, "-petscpartititoner_ptscotch_vertex_weight", &flg, NULL);CHKERRQ(ierr);
    if (!flg) veloloctab = NULL;
  }
  ierr = MPI_Comm_size(comm, &procglbnbr);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &proclocnum);CHKERRQ(ierr);
  vertlocnbr = vtxdist[proclocnum + 1] - vtxdist[proclocnum];
  edgelocnbr = xadj[vertlocnbr];

  ierr = SCOTCH_dgraphInit(&grafdat, comm);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_dgraphBuild(&grafdat, 0, vertlocnbr, vertlocnbr, xadj, xadj + 1, veloloctab, NULL, edgelocnbr, edgelocnbr, adjncy, NULL, edloloctab);CHKERRPTSCOTCH(ierr);
#if defined(PETSC_USE_DEBUG)
  ierr = SCOTCH_dgraphCheck(&grafdat);CHKERRPTSCOTCH(ierr);
#endif
  ierr = SCOTCH_stratInit(&stradat);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_stratDgraphMapBuild(&stradat, flagval, procglbnbr, nparts, kbalval);CHKERRQ(ierr);
  ierr = SCOTCH_archInit(&archdat);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_archCmplt(&archdat, nparts);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_dgraphMapInit(&grafdat, &mappdat, &archdat, part);CHKERRPTSCOTCH(ierr);
  ierr = SCOTCH_dgraphMapCompute(&grafdat, &mappdat, &stradat);CHKERRPTSCOTCH(ierr);
  SCOTCH_dgraphMapExit(&grafdat, &mappdat);
  SCOTCH_archExit(&archdat);
  SCOTCH_stratExit(&stradat);
  SCOTCH_dgraphExit(&grafdat);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_PTSCOTCH */

static PetscErrorCode PetscPartitionerDestroy_PTScotch(PetscPartitioner part)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *) part->data;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = PetscFree(p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_PTScotch_Ascii(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *) part->data;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "using partitioning strategy %s\n",PTScotchStrategyList[p->strategy]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "using load imbalance ratio %g\n",(double)p->imbalance);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerView_PTScotch(PetscPartitioner part, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscPartitionerView_PTScotch_Ascii(part, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerSetFromOptions_PTScotch(PetscOptionItems *PetscOptionsObject, PetscPartitioner part)
{
  PetscPartitioner_PTScotch *p = (PetscPartitioner_PTScotch *) part->data;
  const char *const         *slist = PTScotchStrategyList;
  PetscInt                  nlist = (PetscInt)(sizeof(PTScotchStrategyList)/sizeof(PTScotchStrategyList[0]));
  PetscBool                 flag;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "PetscPartitioner PTScotch Options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-petscpartitioner_ptscotch_strategy","Partitioning strategy","",slist,nlist,slist[p->strategy],&p->strategy,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-petscpartitioner_ptscotch_imbalance","Load imbalance ratio","",p->imbalance,&p->imbalance,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscPartitionerPartition_PTScotch(PetscPartitioner part, DM dm, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection partSection, IS *partition)
{
#if defined(PETSC_HAVE_PTSCOTCH)
  MPI_Comm       comm       = PetscObjectComm((PetscObject)part);
  PetscInt       nvtxs      = numVertices; /* The number of vertices in full graph */
  PetscInt      *vtxdist;                  /* Distribution of vertices across processes */
  PetscInt      *xadj       = start;       /* Start of edge list for each vertex */
  PetscInt      *adjncy     = adjacency;   /* Edge lists for all vertices */
  PetscInt      *vwgt       = NULL;        /* Vertex weights */
  PetscInt      *adjwgt     = NULL;        /* Edge weights */
  PetscInt       v, i, *assignment, *points;
  PetscMPIInt    size, rank, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscMalloc2(nparts+1,&vtxdist,PetscMax(nvtxs,1),&assignment);CHKERRQ(ierr);

  /* Calculate vertex distribution */
  vtxdist[0] = 0;
  ierr = MPI_Allgather(&nvtxs, 1, MPIU_INT, &vtxdist[1], 1, MPIU_INT, comm);CHKERRQ(ierr);
  for (p = 2; p <= size; ++p) {
    vtxdist[p] += vtxdist[p-1];
  }

  if (nparts == 1) {
    ierr = PetscMemzero(assignment, nvtxs * sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    PetscSection section;
    /* Weight cells by dofs on cell by default */
    ierr = PetscMalloc1(PetscMax(nvtxs,1),&vwgt);CHKERRQ(ierr);
    ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
    if (section) {
      PetscInt vStart, vEnd, dof;
      ierr = DMPlexGetHeightStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
      for (v = vStart; v < vEnd; ++v) {
        ierr = PetscSectionGetDof(section, v, &dof);CHKERRQ(ierr);
        /* WARNING: Assumes that meshes with overlap have the overlapped cells at the end of the stratum. */
        /* To do this properly, we should use the cell numbering created in DMPlexCreatePartitionerGraph. */
        if (v-vStart < numVertices) vwgt[v-vStart] = PetscMax(dof, 1);
      }
    } else {
      for (v = 0; v < nvtxs; ++v) vwgt[v] = 1;
    }
    {
      PetscPartitioner_PTScotch *pts = (PetscPartitioner_PTScotch *) part->data;
      int                       strat = PTScotch_Strategy(pts->strategy);
      double                    imbal = (double)pts->imbalance;

      for (p = 0; !vtxdist[p+1] && p < size; ++p);
      if (vtxdist[p+1] == vtxdist[size]) {
        if (rank == p) {
          ierr = PTScotch_PartGraph_Seq(strat, imbal, nvtxs, xadj, adjncy, vwgt, adjwgt, nparts, assignment);CHKERRQ(ierr);
        }
      } else {
        ierr = PTScotch_PartGraph_MPI(strat, imbal, vtxdist, xadj, adjncy, vwgt, adjwgt, nparts, assignment, comm);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(vwgt);CHKERRQ(ierr);
  }

  /* Convert to PetscSection+IS */
  ierr = PetscSectionSetChart(partSection, 0, nparts);CHKERRQ(ierr);
  for (v = 0; v < nvtxs; ++v) {ierr = PetscSectionAddDof(partSection, assignment[v], 1);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(partSection);CHKERRQ(ierr);
  ierr = PetscMalloc1(nvtxs, &points);CHKERRQ(ierr);
  for (p = 0, i = 0; p < nparts; ++p) {
    for (v = 0; v < nvtxs; ++v) {
      if (assignment[v] == p) points[i++] = v;
    }
  }
  if (i != nvtxs) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of points %D should be %D", i, nvtxs);
  ierr = ISCreateGeneral(comm, nvtxs, points, PETSC_OWN_POINTER, partition);CHKERRQ(ierr);

  ierr = PetscFree2(vtxdist,assignment);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject) part), PETSC_ERR_SUP, "Mesh partitioning needs external package support.\nPlease reconfigure with --download-ptscotch.");
#endif
}

static PetscErrorCode PetscPartitionerInitialize_PTScotch(PetscPartitioner part)
{
  PetscFunctionBegin;
  part->ops->view           = PetscPartitionerView_PTScotch;
  part->ops->destroy        = PetscPartitionerDestroy_PTScotch;
  part->ops->partition      = PetscPartitionerPartition_PTScotch;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_PTScotch;
  PetscFunctionReturn(0);
}

/*MC
  PETSCPARTITIONERPTSCOTCH = "ptscotch" - A PetscPartitioner object using the PT-Scotch library

  Level: intermediate

.seealso: PetscPartitionerType, PetscPartitionerCreate(), PetscPartitionerSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_PTScotch(PetscPartitioner part)
{
  PetscPartitioner_PTScotch *p;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  ierr = PetscNewLog(part, &p);CHKERRQ(ierr);
  part->data = p;

  p->strategy  = 0;
  p->imbalance = 0.01;

  ierr = PetscPartitionerInitialize_PTScotch(part);CHKERRQ(ierr);
  ierr = PetscCitationsRegister(PTScotchPartitionerCitation, &PTScotchPartitionercite);CHKERRQ(ierr);
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

.seealso DMPlexDistribute(), DMPlexSetPartitioner(), PetscPartitionerCreate()
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

  logically collective on dm and part

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

static PetscErrorCode DMPlexAddClosure_Tree(DM dm, PetscHSetI ht, PetscInt point, PetscBool up, PetscBool down)
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

PETSC_INTERN PetscErrorCode DMPlexPartitionLabelClosure_Private(DM,DMLabel,PetscInt,PetscInt,const PetscInt[],IS*);

PetscErrorCode DMPlexPartitionLabelClosure_Private(DM dm, DMLabel label, PetscInt rank, PetscInt numPoints, const PetscInt points[], IS *closureIS)
{
  DM_Plex        *mesh = (DM_Plex *)dm->data;
  PetscBool      hasTree = (mesh->parentSection || mesh->childSection) ? PETSC_TRUE : PETSC_FALSE;
  PetscInt       *closure = NULL, closureSize, nelems, *elems, off = 0, p, c;
  PetscHSetI     ht;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
  ierr = PetscHSetIResize(ht, numPoints*16);CHKERRQ(ierr);
  for (p = 0; p < numPoints; ++p) {
    ierr = DMPlexGetTransitiveClosure(dm, points[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (c = 0; c < closureSize*2; c += 2) {
      ierr = PetscHSetIAdd(ht, closure[c]);CHKERRQ(ierr);
      if (hasTree) {ierr = DMPlexAddClosure_Tree(dm, ht, closure[c], PETSC_TRUE, PETSC_TRUE);CHKERRQ(ierr);}
    }
  }
  if (closure) {ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, NULL, &closure);CHKERRQ(ierr);}
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

.seealso: DMPlexPartitionLabelCreateSF, DMPlexDistribute(), DMPlexCreateOverlap
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
    ierr = DMPlexPartitionLabelClosure_Private(dm, label, rank, numPoints, points, &closureIS);CHKERRQ(ierr);
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

.seealso: DMPlexPartitionLabelCreateSF, DMPlexDistribute(), DMPlexCreateOverlap
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

.seealso: DMPlexPartitionLabelCreateSF, DMPlexDistribute(), DMPlexCreateOverlap
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
. processSF - A star forest mapping into the local index on each remote rank

  Output Parameter:
- leafLabel - DMLabel assinging ranks to remote roots

  Note: The rootLabel defines a send pattern by mapping local points to remote target ranks. The
  resulting leafLabel is a receiver mapping of remote roots to their parent rank.

  Level: developer

.seealso: DMPlexPartitionLabelCreateSF, DMPlexDistribute(), DMPlexCreateOverlap
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
      leafSize = (PetscInt) counter;
      ierr = PetscMalloc1(leafSize, &leafPoints);CHKERRQ(ierr);
      ierr = MPI_Alltoallv(rootPoints, scounts, sdispls, MPIU_2INT, leafPoints, rcounts, rdispls, MPIU_2INT, comm);CHKERRQ(ierr);
    }
    ierr = PetscFree4(scounts, sdispls, rcounts, rdispls);CHKERRQ(ierr);
  }

  /* Communicate overlap using process star forest */
  if (processSF || mpiOverflow) {
    PetscSF      procSF;
    PetscSFNode  *remote;
    PetscSection leafSection;

    if (processSF) {
      ierr = PetscObjectReference((PetscObject)processSF);CHKERRQ(ierr);
      procSF = processSF;
    } else {
      ierr = PetscMalloc1(size, &remote);CHKERRQ(ierr);
      for (r = 0; r < size; ++r) { remote[r].rank  = r; remote[r].index = rank; }
      ierr = PetscSFCreate(comm, &procSF);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(procSF, size, size, NULL, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    }

    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &leafSection);CHKERRQ(ierr);
    ierr = DMPlexDistributeData(dm, processSF, rootSection, MPIU_2INT, rootPoints, leafSection, (void**) &leafPoints);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

/*@
  DMPlexPartitionLabelCreateSF - Create a star forest from a label that assigns ranks to points

  Input Parameters:
+ dm    - The DM
. label - DMLabel assinging ranks to remote roots

  Output Parameter:
- sf    - The star forest communication context encapsulating the defined mapping

  Note: The incoming label is a receiver mapping of remote points to their parent rank.

  Level: developer

.seealso: DMPlexDistribute(), DMPlexCreateOverlap
@*/
PetscErrorCode DMPlexPartitionLabelCreateSF(DM dm, DMLabel label, PetscSF *sf)
{
  PetscMPIInt     rank, size;
  PetscInt        n, numRemote, p, numPoints, pStart, pEnd, idx = 0;
  PetscSFNode    *remotePoints;
  IS              remoteRootIS;
  const PetscInt *remoteRoots;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);

  for (numRemote = 0, n = 0; n < size; ++n) {
    ierr = DMLabelGetStratumSize(label, n, &numPoints);CHKERRQ(ierr);
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
  for (n = 0; n < size; ++n) {
    ierr = DMLabelGetStratumSize(label, n, &numPoints);CHKERRQ(ierr);
    if (numPoints <= 0 || n == rank) continue;
    ierr = DMLabelGetStratumIS(label, n, &remoteRootIS);CHKERRQ(ierr);
    ierr = ISGetIndices(remoteRootIS, &remoteRoots);CHKERRQ(ierr);
    for (p = 0; p < numPoints; p++) {
      remotePoints[idx].index = remoteRoots[p];
      remotePoints[idx].rank = n;
      idx++;
    }
    ierr = ISRestoreIndices(remoteRootIS, &remoteRoots);CHKERRQ(ierr);
    ierr = ISDestroy(&remoteRootIS);CHKERRQ(ierr);
  }
  ierr = PetscSFCreate(PetscObjectComm((PetscObject) dm), sf);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sf, pEnd-pStart, numRemote, NULL, PETSC_OWN_POINTER, remotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
