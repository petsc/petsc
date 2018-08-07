#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

/*@C
  DMPlexCreateGmshFromFile - Create a DMPlex mesh from a Gmsh file

+ comm        - The MPI communicator
. filename    - Name of the Gmsh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Level: beginner

.seealso: DMPlexCreateFromFile(), DMPlexCreateGmsh(), DMPlexCreate()
@*/
PetscErrorCode DMPlexCreateGmshFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscViewer     viewer;
  PetscMPIInt     rank;
  int             fileType;
  PetscViewerType vtype;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  /* Determine Gmsh file type (ASCII or binary) from file header */
  if (!rank) {
    PetscViewer vheader;
    char        line[PETSC_MAX_PATH_LEN];
    PetscBool   match;
    int         snum;
    float       version;

    ierr = PetscViewerCreate(PETSC_COMM_SELF, &vheader);CHKERRQ(ierr);
    ierr = PetscViewerSetType(vheader, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(vheader, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(vheader, filename);CHKERRQ(ierr);
    /* Read only the first two lines of the Gmsh file */
    ierr = PetscViewerRead(vheader, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$MeshFormat", sizeof(line), &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = PetscViewerRead(vheader, line, 2, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%f %d", &version, &fileType);
    if (snum != 2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse Gmsh file header: %s", line);
    if (version < 2.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Gmsh file must be at least version 2.0");
    ierr = PetscViewerDestroy(&vheader);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(&fileType, 1, MPI_INT, 0, comm);CHKERRQ(ierr);
  vtype = (fileType == 0) ? PETSCVIEWERASCII : PETSCVIEWERBINARY;

  /* Create appropriate viewer and build plex */
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, vtype);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = DMPlexCreateGmsh(comm, viewer, interpolate, dm);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateGmsh_ReadNodes(PetscViewer viewer, PetscBool binary, PetscBool byteSwap, int shift, int numVertices, double **coordinates)
{
  int            v,nid;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(numVertices*3, coordinates);CHKERRQ(ierr);
  for (v = 0; v < numVertices; ++v) {
    double *xyz = *coordinates + v*3;
    ierr = PetscViewerRead(viewer, &nid, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(&nid, PETSC_ENUM, 1);CHKERRQ(ierr);}
    if (nid != v+shift) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unexpected node number %d should be %d", nid, v+shift);
    ierr = PetscViewerRead(viewer, xyz, 3, NULL, PETSC_DOUBLE);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(xyz, PETSC_DOUBLE, 3);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateGmsh_ReadElements(PetscViewer viewer, PetscBool binary, PetscBool byteSwap, PETSC_UNUSED int shift, int numCells, GmshElement **gmsh_elems)
{
  GmshElement   *elements;
  int            i, c, p, ibuf[1+4+512];
  int            cellType, dim, numNodes, numNodesIgnore, numElem, numTags;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(numCells, &elements);CHKERRQ(ierr);
  for (c = 0; c < numCells;) {
    ierr = PetscViewerRead(viewer, ibuf, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, 3);CHKERRQ(ierr);}
    if (binary) {
      cellType = ibuf[0];
      numElem = ibuf[1];
      numTags = ibuf[2];
    } else {
      elements[c].id = ibuf[0];
      cellType = ibuf[1];
      numTags = ibuf[2];
      numElem = 1;
    }
    /* http://gmsh.info/doc/texinfo/gmsh.html#MSH-ASCII-file-format */
    numNodesIgnore = 0;
    switch (cellType) {
    case 1: /* 2-node line */
      dim = 1;
      numNodes = 2;
      break;
    case 2: /* 3-node triangle */
      dim = 2;
      numNodes = 3;
      break;
    case 3: /* 4-node quadrangle */
      dim = 2;
      numNodes = 4;
      break;
    case 4: /* 4-node tetrahedron */
      dim  = 3;
      numNodes = 4;
      break;
    case 5: /* 8-node hexahedron */
      dim = 3;
      numNodes = 8;
      break;
    case 6: /* 6-node wedge */
      dim = 3;
      numNodes = 6;
      break;
    case 8: /* 3-node 2nd order line */
      dim = 1;
      numNodes = 2;
      numNodesIgnore = 1;
      break;
    case 9: /* 6-node 2nd order triangle */
      dim = 2;
      numNodes = 3;
      numNodesIgnore = 3;
      break;
    case 13: /* 18-node 2nd wedge */
      dim = 3;
      numNodes = 6;
      numNodesIgnore = 12;
      break;
    case 15: /* 1-node vertex */
      dim = 0;
      numNodes = 1;
      break;
    case 7: /* 5-node pyramid */
    case 10: /* 9-node 2nd order quadrangle */
    case 11: /* 10-node 2nd order tetrahedron */
    case 12: /* 27-node 2nd order hexhedron */
    case 14: /* 14-node 2nd order pyramid */
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported Gmsh element type %d", cellType);
    }
    if (binary) {
      const int nint = 1 + numTags + numNodes + numNodesIgnore;
      /* Loop over element blocks */
      for (i = 0; i < numElem; ++i, ++c) {
        ierr = PetscViewerRead(viewer, ibuf, nint, NULL, PETSC_ENUM);CHKERRQ(ierr);
        if (byteSwap) {ierr = PetscByteSwap(ibuf, PETSC_ENUM, nint);CHKERRQ(ierr);}
        elements[c].dim = dim;
        elements[c].numNodes = numNodes;
        elements[c].numTags = numTags;
        elements[c].id = ibuf[0];
        elements[c].cellType = cellType;
        for (p = 0; p < numTags; p++) elements[c].tags[p] = ibuf[1 + p];
        for (p = 0; p < numNodes; p++) elements[c].nodes[p] = ibuf[1 + numTags + p];
      }
    } else {
      elements[c].dim = dim;
      elements[c].numNodes = numNodes;
      elements[c].numTags = numTags;
      elements[c].cellType = cellType;
      ierr = PetscViewerRead(viewer, elements[c].tags, elements[c].numTags, NULL, PETSC_ENUM);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, elements[c].nodes, elements[c].numNodes, NULL, PETSC_ENUM);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, ibuf, numNodesIgnore, NULL, PETSC_ENUM);CHKERRQ(ierr);
      c++;
    }
  }
  *gmsh_elems = elements;
  PetscFunctionReturn(0);
}

/*@
  DMPlexCreateGmsh - Create a DMPlex mesh from a Gmsh file viewer

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
. viewer - The Viewer associated with a Gmsh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.geuz.org/gmsh/doc/texinfo/#MSH-ASCII-file-format
  and http://www.geuz.org/gmsh/doc/texinfo/#MSH-binary-file-format

  Level: beginner

.keywords: mesh,Gmsh
.seealso: DMPLEX, DMCreate()
@*/
PetscErrorCode DMPlexCreateGmsh(MPI_Comm comm, PetscViewer viewer, PetscBool interpolate, DM *dm)
{
  PetscViewer    parentviewer = NULL;
  double        *coordsIn = NULL;
  GmshElement   *gmsh_elem = NULL;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscBT        periodicV = NULL, periodicC = NULL;
  PetscScalar   *coords;
  PetscInt       dim = 0, embedDim = -1, coordSize, c, v, d, r, cell, *periodicMap = NULL, *periodicMapI = NULL, *hybridMap = NULL, cMax = PETSC_DETERMINE;
  int            i, numVertices = 0, numCells = 0, trueNumCells = 0, numRegions = 0, snum, shift = 1;
  PetscMPIInt    rank;
  char           line[PETSC_MAX_PATH_LEN];
  PetscBool      binary, byteSwap = PETSC_FALSE, zerobase = PETSC_FALSE, periodic = PETSC_FALSE, usemarker = PETSC_FALSE;
  PetscBool      enable_hybrid = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_gmsh_hybrid", &enable_hybrid, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_gmsh_periodic", &periodic, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_gmsh_use_marker", &usemarker, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_gmsh_zero_base", &zerobase, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(((PetscObject) viewer)->options,((PetscObject) viewer)->prefix, "-dm_plex_gmsh_spacedim", &embedDim, NULL);CHKERRQ(ierr);
  if (zerobase) shift = 0;

  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_CreateGmsh,*dm,0,0,0);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &binary);CHKERRQ(ierr);

  /* Binary viewers read on all ranks, get subviewer to read only in rank 0 */
  if (binary) {
    parentviewer = viewer;
    ierr = PetscViewerGetSubViewer(parentviewer, PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
  }

  if (!rank) {
    PetscBool match, hybrid;
    int       fileType, dataSize;
    float     version;

    /* Read header */
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$MeshFormat", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = PetscViewerRead(viewer, line, 3, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%f %d %d", &version, &fileType, &dataSize);
    if (snum != 3) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse Gmsh file header: %s", line);
    if (version < 2.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Gmsh file must be at least version 2.0");
    if (dataSize != sizeof(double)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data size %d is not valid for a Gmsh file", dataSize);
    if (binary) {
      int checkInt;
      ierr = PetscViewerRead(viewer, &checkInt, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
      if (checkInt != 1) {
        ierr = PetscByteSwap(&checkInt, PETSC_ENUM, 1);CHKERRQ(ierr);
        if (checkInt == 1) byteSwap = PETSC_TRUE;
        else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File type %d is not a valid Gmsh binary file", fileType);
      }
    } else if (fileType) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File type %d is not a valid Gmsh ASCII file", fileType);
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$EndMeshFormat", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");

    /* OPTIONAL Read physical names */
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$PhysicalNames", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (match) {
      ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
      snum = sscanf(line, "%d", &numRegions);
      if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
      for (r = 0; r < numRegions; ++r) {
        ierr = PetscViewerRead(viewer, line, 3, NULL, PETSC_STRING);CHKERRQ(ierr);
      }
      ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
      ierr = PetscStrncmp(line, "$EndPhysicalNames", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
      if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
      /* Initial read for vertex section */
      ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    }

    /* Read vertices */
    ierr = PetscStrncmp(line, "$Nodes", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d", &numVertices);
    if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = DMPlexCreateGmsh_ReadNodes(viewer, binary, byteSwap, shift, numVertices, &coordsIn);CHKERRQ(ierr);
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);;
    ierr = PetscStrncmp(line, "$EndNodes", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");

    /* Read cells */
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);;
    ierr = PetscStrncmp(line, "$Elements", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);;
    snum = sscanf(line, "%d", &numCells);
    if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    /* Gmsh elements can be of any dimension/co-dimension, so we need to traverse the
       file contents multiple times to figure out the true number of cells and facets
       in the given mesh. To make this more efficient we read the file contents only
       once and store them in memory, while determining the true number of cells. */
    ierr = DMPlexCreateGmsh_ReadElements(viewer, binary, byteSwap, shift, numCells, &gmsh_elem);CHKERRQ(ierr);
    hybrid = PETSC_FALSE;
    for (trueNumCells = 0, c = 0; c < numCells; ++c) {
      int on = -1;
      if (gmsh_elem[c].dim > dim) {dim = gmsh_elem[c].dim; trueNumCells = 0;}
      if (gmsh_elem[c].dim == dim) {hybrid = (trueNumCells ? (on != gmsh_elem[c].cellType ? on = gmsh_elem[c].cellType,PETSC_TRUE : hybrid) : (on = gmsh_elem[c].cellType, PETSC_FALSE) ); trueNumCells++;}
      /* wedges always indicate an hybrid mesh in PLEX */
      if (on == 6 || on == 13) hybrid = PETSC_TRUE;
    }
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$EndElements", 12, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");

    /* Renumber cells for hybrid grids */
    if (hybrid && enable_hybrid) {
      PetscInt hc1 = 0, hc2 = 0, *hybridCells1 = NULL, *hybridCells2 = NULL;
      PetscInt cell, tn, *tp;
      int      n1 = 0,n2 = 0;

      ierr = PetscMalloc1(trueNumCells, &hybridCells1);CHKERRQ(ierr);
      ierr = PetscMalloc1(trueNumCells, &hybridCells2);CHKERRQ(ierr);
      for (cell = 0, c = 0; c < numCells; ++c) {
        if (gmsh_elem[c].dim == dim) {
          if (!n1) n1 = gmsh_elem[c].cellType;
          else if (!n2 && n1 != gmsh_elem[c].cellType) n2 = gmsh_elem[c].cellType;

          if      (gmsh_elem[c].cellType == n1) hybridCells1[hc1++] = cell;
          else if (gmsh_elem[c].cellType == n2) hybridCells2[hc2++] = cell;
          else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle more than 2 cell types");
          cell++;
        }
      }

      switch (n1) {
      case 2: /* triangles */
      case 9:
        switch (n2) {
        case 0: /* single type mesh */
        case 3: /* quads */
        case 10:
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
        }
        break;
      case 3:
      case 10:
        switch (n2) {
        case 0: /* single type mesh */
        case 2: /* swap since we list simplices first */
        case 9:
          tn  = hc1;
          hc1 = hc2;
          hc2 = tn;

          tp           = hybridCells1;
          hybridCells1 = hybridCells2;
          hybridCells2 = tp;
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
        }
        break;
      case 4: /* tetrahedra */
      case 11:
        switch (n2) {
        case 0: /* single type mesh */
        case 6: /* wedges */
        case 13:
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
        }
        break;
      case 6:
      case 13:
        switch (n2) {
        case 0: /* single type mesh */
        case 4: /* swap since we list simplices first */
        case 11:
          tn  = hc1;
          hc1 = hc2;
          hc2 = tn;

          tp           = hybridCells1;
          hybridCells1 = hybridCells2;
          hybridCells2 = tp;
          break;
        default:
          SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
        }
        break;
      default:
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell types %d and %d",n1, n2);
      }
      cMax = hc1;
      ierr = PetscMalloc1(trueNumCells, &hybridMap);CHKERRQ(ierr);
      for (cell = 0; cell < hc1; cell++) hybridMap[hybridCells1[cell]] = cell;
      for (cell = 0; cell < hc2; cell++) hybridMap[hybridCells2[cell]] = cell + hc1;
      ierr = PetscFree(hybridCells1);CHKERRQ(ierr);
      ierr = PetscFree(hybridCells2);CHKERRQ(ierr);
    }

    /* OPTIONAL Read periodic section */
    if (periodic) {
      PetscInt pVert, *periodicMapT, *aux;
      int      numPeriodic;

      ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
      ierr = PetscStrncmp(line, "$Periodic", 9, &match);CHKERRQ(ierr);
      if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
      ierr = PetscMalloc1(numVertices, &periodicMapT);CHKERRQ(ierr);
      ierr = PetscBTCreate(numVertices, &periodicV);CHKERRQ(ierr);
      for (i = 0; i < numVertices; i++) periodicMapT[i] = i;
      ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
      snum = sscanf(line, "%d", &numPeriodic);
      if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
      for (i = 0; i < numPeriodic; i++) {
        int j, edim = -1, slaveTag = -1, masterTag = -1, nNodes, slaveNode, masterNode;

        ierr = PetscViewerRead(viewer, line, 3, NULL, PETSC_STRING);CHKERRQ(ierr);
        snum = sscanf(line, "%d %d %d", &edim, &slaveTag, &masterTag); /* slaveTag and masterTag are unused */
        if (snum != 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
        ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
        snum = sscanf(line, "%d", &nNodes);
        if (snum != 1) { /* discard tranformation and try again */
          ierr = PetscViewerRead(viewer, line, -PETSC_MAX_PATH_LEN, NULL, PETSC_STRING);CHKERRQ(ierr);
          ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
          snum = sscanf(line, "%d", &nNodes);
          if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
        }
        for (j = 0; j < nNodes; j++) {
          ierr = PetscViewerRead(viewer, line, 2, NULL, PETSC_STRING);CHKERRQ(ierr);
          snum = sscanf(line, "%d %d", &slaveNode, &masterNode);
          if (snum != 2) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
          periodicMapT[slaveNode - shift] = masterNode - shift;
          ierr = PetscBTSet(periodicV, slaveNode - shift);CHKERRQ(ierr);
          ierr = PetscBTSet(periodicV, masterNode - shift);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
      ierr = PetscStrncmp(line, "$EndPeriodic", 12, &match);CHKERRQ(ierr);
      if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
      /* we may have slaves of slaves */
      for (i = 0; i < numVertices; i++) {
        while (periodicMapT[periodicMapT[i]] != periodicMapT[i]) {
          periodicMapT[i] = periodicMapT[periodicMapT[i]];
        }
      }
      /* periodicMap : from old to new numbering (periodic vertices excluded)
         periodicMapI: from new to old numbering */
      ierr = PetscMalloc1(numVertices, &periodicMap);CHKERRQ(ierr);
      ierr = PetscMalloc1(numVertices, &periodicMapI);CHKERRQ(ierr);
      ierr = PetscMalloc1(numVertices, &aux);CHKERRQ(ierr);
      for (i = 0, pVert = 0; i < numVertices; i++) {
        if (periodicMapT[i] != i) {
          pVert++;
        } else {
          aux[i] = i - pVert;
          periodicMapI[i - pVert] = i;
        }
      }
      for (i = 0 ; i < numVertices; i++) {
        periodicMap[i] = aux[periodicMapT[i]];
      }
      ierr = PetscFree(periodicMapT);CHKERRQ(ierr);
      ierr = PetscFree(aux);CHKERRQ(ierr);
      /* remove periodic vertices */
      numVertices = numVertices - pVert;
    }
  }

  if (parentviewer) {
    ierr = PetscViewerRestoreSubViewer(parentviewer, PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
  }

  /* Allocate the cell-vertex mesh */
  ierr = DMPlexSetChart(*dm, 0, trueNumCells+numVertices);CHKERRQ(ierr);
  for (cell = 0, c = 0; c < numCells; ++c) {
    if (gmsh_elem[c].dim == dim) {
      ierr = DMPlexSetConeSize(*dm, hybridMap ? hybridMap[cell] : cell, gmsh_elem[c].numNodes);CHKERRQ(ierr);
      cell++;
    }
  }
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  /* Add cell-vertex connections */
  for (cell = 0, c = 0; c < numCells; ++c) {
    if (gmsh_elem[c].dim == dim) {
      PetscInt pcone[8], corner;
      for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
        const PetscInt cc = gmsh_elem[c].nodes[corner] - shift;
        pcone[corner] = (periodicMap ? periodicMap[cc] : cc) + trueNumCells;
      }
      if (dim == 3) {
        /* Tetrahedra are inverted */
        if (gmsh_elem[c].cellType == 4) {
          PetscInt tmp = pcone[0];
          pcone[0] = pcone[1];
          pcone[1] = tmp;
        }
        /* Hexahedra are inverted */
        if (gmsh_elem[c].cellType == 5) {
          PetscInt tmp = pcone[1];
          pcone[1] = pcone[3];
          pcone[3] = tmp;
        }
        /* Prisms are inverted */
        if (gmsh_elem[c].cellType == 6) {
          PetscInt tmp;

          tmp      = pcone[1];
          pcone[1] = pcone[2];
          pcone[2] = tmp;
          tmp      = pcone[4];
          pcone[4] = pcone[5];
          pcone[5] = tmp;
        }
      } else if (dim == 2 && hybridMap && hybridMap[cell] >= cMax) { /* hybrid cells */
        /* quads are input to PLEX as prisms */
        if (gmsh_elem[c].cellType == 3) {
          PetscInt tmp = pcone[2];
          pcone[2] = pcone[3];
          pcone[3] = tmp;
        }
      }
      ierr = DMPlexSetCone(*dm, hybridMap ? hybridMap[cell] : cell, pcone);CHKERRQ(ierr);
      cell++;
    }
  }
  ierr = MPI_Bcast(&dim, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexSetHybridBounds(*dm, cMax, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  if (interpolate) {
    DM idm;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }

  if (usemarker && !interpolate && dim > 1) SETERRQ(comm,PETSC_ERR_SUP,"Cannot create marker label without interpolation");
  if (!rank && usemarker) {
    PetscInt f, fStart, fEnd;

    ierr = DMCreateLabel(*dm, "marker");CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(*dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      PetscInt suppSize;

      ierr = DMPlexGetSupportSize(*dm, f, &suppSize);CHKERRQ(ierr);
      if (suppSize == 1) {
        PetscInt *cone = NULL, coneSize, p;

        ierr = DMPlexGetTransitiveClosure(*dm, f, PETSC_TRUE, &coneSize, &cone);CHKERRQ(ierr);
        for (p = 0; p < coneSize; p += 2) {
          ierr = DMSetLabelValue(*dm, "marker", cone[p], 1);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreTransitiveClosure(*dm, f, PETSC_TRUE, &coneSize, &cone);CHKERRQ(ierr);
      }
    }
  }

  if (!rank) {
    PetscInt vStart, vEnd;

    ierr = DMPlexGetDepthStratum(*dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    for (cell = 0, c = 0; c < numCells; ++c) {

      /* Create face sets */
      if (interpolate && gmsh_elem[c].dim == dim-1) {
        const PetscInt *join;
        PetscInt        joinSize, pcone[8], corner;
        /* Find the relevant facet with vertex joins */
        for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
          const PetscInt cc = gmsh_elem[c].nodes[corner] - shift;
          pcone[corner] = (periodicMap ? periodicMap[cc] : cc) + vStart;
        }
        ierr = DMPlexGetFullJoin(*dm, gmsh_elem[c].numNodes, pcone, &joinSize, &join);CHKERRQ(ierr);
        if (joinSize != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for element %d", gmsh_elem[c].id);
        ierr = DMSetLabelValue(*dm, "Face Sets", join[0], gmsh_elem[c].tags[0]);CHKERRQ(ierr);
        ierr = DMPlexRestoreJoin(*dm, gmsh_elem[c].numNodes, (const PetscInt *) pcone, &joinSize, &join);CHKERRQ(ierr);
      }

      /* Create cell sets */
      if (gmsh_elem[c].dim == dim) {
        if (gmsh_elem[c].numTags > 0) {
          ierr = DMSetLabelValue(*dm, "Cell Sets", hybridMap ? hybridMap[cell] : cell, gmsh_elem[c].tags[0]);CHKERRQ(ierr);
        }
        cell++;
      }

      /* Create vertex sets */
      if (gmsh_elem[c].dim == 0) {
        if (gmsh_elem[c].numTags > 0) {
          const PetscInt cc = gmsh_elem[c].nodes[0] - shift;
          const PetscInt vid = (periodicMap ? periodicMap[cc] : cc) + vStart;
          ierr = DMSetLabelValue(*dm, "Vertex Sets", vid, gmsh_elem[c].tags[0]);CHKERRQ(ierr);
        }
      }
    }
  }

  /* Create coordinates */
  if (embedDim < 0) embedDim = dim;
  ierr = DMSetCoordinateDim(*dm, embedDim);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, embedDim);CHKERRQ(ierr);
  if (periodicMap) { /* we need to localize coordinates on cells */
    ierr = PetscSectionSetChart(coordSection, 0, trueNumCells + numVertices);CHKERRQ(ierr);
  } else {
    ierr = PetscSectionSetChart(coordSection, trueNumCells, trueNumCells + numVertices);CHKERRQ(ierr);
  }
  for (v = trueNumCells; v < trueNumCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, embedDim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, embedDim);CHKERRQ(ierr);
  }
  if (periodicMap) {
    ierr = PetscBTCreate(trueNumCells, &periodicC);CHKERRQ(ierr);
    for (cell = 0, c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim == dim) {
        PetscInt  corner;
        PetscBool pc = PETSC_FALSE;
        for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
          pc = (PetscBool)(pc || PetscBTLookup(periodicV, gmsh_elem[c].nodes[corner] - shift));
        }
        if (pc) {
          PetscInt dof = gmsh_elem[c].numNodes*embedDim;
          PetscInt ucell = hybridMap ? hybridMap[cell] : cell;
          ierr = PetscBTSet(periodicC, ucell);CHKERRQ(ierr);
          ierr = PetscSectionSetDof(coordSection, ucell, dof);CHKERRQ(ierr);
          ierr = PetscSectionSetFieldDof(coordSection, ucell, 0, dof);CHKERRQ(ierr);
        }
        cell++;
      }
    }
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, embedDim);CHKERRQ(ierr);
  ierr = VecSetType(coordinates, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (periodicMap) {
    PetscInt off;

    for (cell = 0, c = 0; c < numCells; ++c) {
      PetscInt pcone[8], corner;
      if (gmsh_elem[c].dim == dim) {
        PetscInt ucell = hybridMap ? hybridMap[cell] : cell;
        if (PetscUnlikely(PetscBTLookup(periodicC, ucell))) {
          for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
            pcone[corner] = gmsh_elem[c].nodes[corner] - shift;
          }
          if (dim == 3) {
            /* Tetrahedra are inverted */
            if (gmsh_elem[c].cellType == 4) {
              PetscInt tmp = pcone[0];
              pcone[0] = pcone[1];
              pcone[1] = tmp;
            }
            /* Hexahedra are inverted */
            if (gmsh_elem[c].cellType == 5) {
              PetscInt tmp = pcone[1];
              pcone[1] = pcone[3];
              pcone[3] = tmp;
            }
            /* Prisms are inverted */
            if (gmsh_elem[c].cellType == 6) {
              PetscInt tmp;

              tmp      = pcone[1];
              pcone[1] = pcone[2];
              pcone[2] = tmp;
              tmp      = pcone[4];
              pcone[4] = pcone[5];
              pcone[5] = tmp;
            }
          } else if (dim == 2 && hybridMap && hybridMap[cell] >= cMax) { /* hybrid cells */
            /* quads are input to PLEX as prisms */
            if (gmsh_elem[c].cellType == 3) {
              PetscInt tmp = pcone[2];
              pcone[2] = pcone[3];
              pcone[3] = tmp;
            }
          }
          ierr = PetscSectionGetOffset(coordSection, ucell, &off);CHKERRQ(ierr);
          for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
            v = pcone[corner];
            for (d = 0; d < embedDim; ++d) {
              coords[off++] = (PetscReal) coordsIn[v*3+d];
            }
          }
        }
        cell++;
      }
    }
    for (v = 0; v < numVertices; ++v) {
      ierr = PetscSectionGetOffset(coordSection, v + trueNumCells, &off);CHKERRQ(ierr);
      for (d = 0; d < embedDim; ++d) {
        coords[off+d] = (PetscReal) coordsIn[periodicMapI[v]*3+d];
      }
    }
  } else {
    for (v = 0; v < numVertices; ++v) {
      for (d = 0; d < embedDim; ++d) {
        coords[v*embedDim+d] = (PetscReal) coordsIn[v*3+d];
      }
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = DMSetPeriodicity(*dm, periodic, NULL, NULL, NULL);CHKERRQ(ierr);

  ierr = PetscFree(coordsIn);CHKERRQ(ierr);
  ierr = PetscFree(gmsh_elem);CHKERRQ(ierr);
  ierr = PetscFree(hybridMap);CHKERRQ(ierr);
  ierr = PetscFree(periodicMap);CHKERRQ(ierr);
  ierr = PetscFree(periodicMapI);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&periodicV);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&periodicC);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(DMPLEX_CreateGmsh,*dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
