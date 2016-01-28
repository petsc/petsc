#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateGmshFromFile"
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
  PetscViewer     viewer, vheader;
  PetscMPIInt     rank;
  PetscViewerType vtype;
  char            line[PETSC_MAX_PATH_LEN];
  int             snum;
  PetscBool       match;
  int             fT;
  PetscInt        fileType;
  float           version;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* Determine Gmsh file type (ASCII or binary) from file header */
  ierr = PetscViewerCreate(comm, &vheader);CHKERRQ(ierr);
  ierr = PetscViewerSetType(vheader, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(vheader, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(vheader, filename);CHKERRQ(ierr);
  if (!rank) {
    /* Read only the first two lines of the Gmsh file */
    ierr = PetscViewerRead(vheader, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$MeshFormat", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = PetscViewerRead(vheader, line, 2, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%f %d", &version, &fT);
    fileType = (PetscInt) fT;
    if (snum != 2) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unable to parse Gmsh file header: %s", line);
    if (version < 2.0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Gmsh file must be at least version 2.0");
  }
  ierr = MPI_Bcast(&fileType, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  /* Create appropriate viewer and build plex */
  if (fileType == 0) vtype = PETSCVIEWERASCII;
  else vtype = PETSCVIEWERBINARY;
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, vtype);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = DMPlexCreateGmsh(comm, viewer, interpolate, dm);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vheader);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateGmsh"
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
  PetscViewerType vtype;
  GmshElement   *gmsh_elem;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords, *coordsIn = NULL;
  PetscInt       dim = 0, coordSize, c, v, d, cell;
  int            i, numVertices = 0, numCells = 0, trueNumCells = 0, snum;
  PetscMPIInt    num_proc, rank;
  char           line[PETSC_MAX_PATH_LEN];
  PetscBool      match, binary, bswap = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &num_proc);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(DMPLEX_CreateGmsh,*dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscViewerGetType(viewer, &vtype);CHKERRQ(ierr);
  ierr = PetscStrcmp(vtype, PETSCVIEWERBINARY, &binary);CHKERRQ(ierr);
  if (!rank || binary) {
    PetscBool match;
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
        if (checkInt == 1) bswap = PETSC_TRUE;
        else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File type %d is not a valid Gmsh binary file", fileType);
      }
    } else if (fileType) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File type %d is not a valid Gmsh ASCII file", fileType);
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$EndMeshFormat", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    /* Read vertices */
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    ierr = PetscStrncmp(line, "$Nodes", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);
    snum = sscanf(line, "%d", &numVertices);
    if (snum != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    ierr = PetscMalloc1(numVertices*3, &coordsIn);CHKERRQ(ierr);
    if (binary) {
      size_t doubleSize, intSize;
      PetscInt elementSize;
      char *buffer;
      PetscScalar *baseptr;
      ierr = PetscDataTypeGetSize(PETSC_ENUM, &intSize);CHKERRQ(ierr);
      ierr = PetscDataTypeGetSize(PETSC_DOUBLE, &doubleSize);CHKERRQ(ierr);
      elementSize = (intSize + 3*doubleSize);
      ierr = PetscMalloc1(elementSize*numVertices, &buffer);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, buffer, elementSize*numVertices, NULL, PETSC_CHAR);CHKERRQ(ierr);
      if (bswap) ierr = PetscByteSwap(buffer, PETSC_CHAR, elementSize*numVertices);CHKERRQ(ierr);
      for (v = 0; v < numVertices; ++v) {
        baseptr = ((PetscScalar*)(buffer+v*elementSize+intSize));
        coordsIn[v*3+0] = baseptr[0];
        coordsIn[v*3+1] = baseptr[1];
        coordsIn[v*3+2] = baseptr[2];
      }
      ierr = PetscFree(buffer);CHKERRQ(ierr);
    } else {
      for (v = 0; v < numVertices; ++v) {
        ierr = PetscViewerRead(viewer, &i, 1, NULL, PETSC_ENUM);CHKERRQ(ierr);
        ierr = PetscViewerRead(viewer, &(coordsIn[v*3]), 3, NULL, PETSC_DOUBLE);CHKERRQ(ierr);
        if (i != (int)v+1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid node number %d should be %d", i, v+1);
      }
    }
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
  }

  if (!rank || binary) {
    /* Gmsh elements can be of any dimension/co-dimension, so we need to traverse the
       file contents multiple times to figure out the true number of cells and facets
       in the given mesh. To make this more efficient we read the file contents only
       once and store them in memory, while determining the true number of cells. */
    ierr = DMPlexCreateGmsh_ReadElement(viewer, numCells, binary, bswap, &gmsh_elem);CHKERRQ(ierr);
    for (trueNumCells=0, c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim > dim) {dim = gmsh_elem[c].dim; trueNumCells = 0;}
      if (gmsh_elem[c].dim == dim) trueNumCells++;
    }
    ierr = PetscViewerRead(viewer, line, 1, NULL, PETSC_STRING);CHKERRQ(ierr);;
    ierr = PetscStrncmp(line, "$EndElements", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
  }
  /* For binary we read on all ranks, but only build the plex on rank 0 */
  if (binary && rank) {trueNumCells = 0; numVertices = 0;};
  /* Allocate the cell-vertex mesh */
  ierr = DMPlexSetChart(*dm, 0, trueNumCells+numVertices);CHKERRQ(ierr);
  if (!rank) {
    for (cell = 0, c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim == dim) {
        ierr = DMPlexSetConeSize(*dm, cell, gmsh_elem[c].numNodes);CHKERRQ(ierr);
        cell++;
      }
    }
  }
  ierr = DMSetUp(*dm);CHKERRQ(ierr);
  /* Add cell-vertex connections */
  if (!rank) {
    PetscInt pcone[8], corner;
    for (cell = 0, c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim == dim) {
        for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
          pcone[corner] = gmsh_elem[c].nodes[corner] + trueNumCells-1;
        }
        ierr = DMPlexSetCone(*dm, cell, pcone);CHKERRQ(ierr);
        cell++;
      }
    }
  }
  ierr = MPI_Bcast(&dim, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm, dim);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  if (interpolate) {
    DM idm = NULL;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }

  if (!rank) {
    /* Apply boundary IDs by finding the relevant facets with vertex joins */
    PetscInt pcone[8], corner, vStart, vEnd;

    ierr = DMPlexGetDepthStratum(*dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    for (c = 0; c < numCells; ++c) {
      if (gmsh_elem[c].dim == dim-1) {
        PetscInt joinSize;
        const PetscInt *join;
        for (corner = 0; corner < gmsh_elem[c].numNodes; ++corner) {
          pcone[corner] = gmsh_elem[c].nodes[corner] + vStart - 1;
        }
        ierr = DMPlexGetFullJoin(*dm, gmsh_elem[c].numNodes, (const PetscInt *) pcone, &joinSize, &join);CHKERRQ(ierr);
        if (joinSize != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Could not determine Plex facet for element %d", gmsh_elem[c].id);
        ierr = DMSetLabelValue(*dm, "Face Sets", join[0], gmsh_elem[c].tags[0]);CHKERRQ(ierr);
        ierr = DMPlexRestoreJoin(*dm, gmsh_elem[c].numNodes, (const PetscInt *) pcone, &joinSize, &join);CHKERRQ(ierr);
      }
    }
  }

  /* Read coordinates */
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, trueNumCells, trueNumCells + numVertices);CHKERRQ(ierr);
  for (v = trueNumCells; v < trueNumCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(comm, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(coordinates, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (!rank) {
    for (v = 0; v < numVertices; ++v) {
      for (d = 0; d < dim; ++d) {
        coords[v*dim+d] = coordsIn[v*3+d];
      }
    }
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscFree(coordsIn);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
  /* Clean up intermediate storage */
  if (!rank || binary) ierr = PetscFree(gmsh_elem);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_CreateGmsh,*dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateGmsh_ReadElement"
PetscErrorCode DMPlexCreateGmsh_ReadElement(PetscViewer viewer, PetscInt numCells, PetscBool binary, PetscBool byteSwap, GmshElement **gmsh_elems)
{
  PetscInt       c, p;
  GmshElement   *elements;
  int            i, cellType, dim, numNodes, numElem, numTags;
  int            ibuf[16];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(numCells, &elements);CHKERRQ(ierr);
  for (c = 0; c < numCells;) {
    ierr = PetscViewerRead(viewer, &ibuf, 3, NULL, PETSC_ENUM);CHKERRQ(ierr);
    if (byteSwap) ierr = PetscByteSwap(&ibuf, PETSC_ENUM, 3);CHKERRQ(ierr);
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
    case 15: /* 1-node vertex */
      dim = 0;
      numNodes = 1;
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported Gmsh element type %d", cellType);
    }
    if (binary) {
      const PetscInt nint = numNodes + numTags + 1;
      for (i = 0; i < numElem; ++i, ++c) {
        /* Loop over inner binary element block */
        elements[c].dim = dim;
        elements[c].numNodes = numNodes;
        elements[c].numTags = numTags;

        ierr = PetscViewerRead(viewer, &ibuf, nint, NULL, PETSC_ENUM);CHKERRQ(ierr);
        if (byteSwap) ierr = PetscByteSwap( &ibuf, PETSC_ENUM, nint);CHKERRQ(ierr);
        elements[c].id = ibuf[0];
        for (p = 0; p < numTags; p++) elements[c].tags[p] = ibuf[1 + p];
        for (p = 0; p < numNodes; p++) elements[c].nodes[p] = ibuf[1 + numTags + p];
      }
    } else {
      elements[c].dim = dim;
      elements[c].numNodes = numNodes;
      elements[c].numTags = numTags;
      ierr = PetscViewerRead(viewer, elements[c].tags, elements[c].numTags, NULL, PETSC_ENUM);CHKERRQ(ierr);
      ierr = PetscViewerRead(viewer, elements[c].nodes, elements[c].numNodes, NULL, PETSC_ENUM);CHKERRQ(ierr);
      c++;
    }
  }
  *gmsh_elems = elements;
  PetscFunctionReturn(0);
}
