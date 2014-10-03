#define PETSCDM_DLL
#include <petsc-private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateGmsh"
/*@
  DMPlexCreateGmsh - Create a DMPlex mesh from a Gmsh file.

  Collective on comm

  Input Parameters:
+ comm  - The MPI communicator
. viewer - The Viewer associated with a Gmsh file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.geuz.org/gmsh/doc/texinfo/#MSH-ASCII-file-format

  Level: beginner

.keywords: mesh,Gmsh
.seealso: DMPLEX, DMCreate()
@*/
PetscErrorCode DMPlexCreateGmsh(MPI_Comm comm, PetscViewer viewer, PetscBool interpolate, DM *dm)
{
  FILE          *fd;
  GmshElement   *gmsh_elem;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords, *coordsIn = NULL;
  PetscInt       dim = 0, coordSize, c, v, d, cell;
  int            numVertices = 0, numCells = 0, trueNumCells = 0, snum;
  PetscMPIInt    num_proc, rank;
  char           line[PETSC_MAX_PATH_LEN];
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &num_proc);CHKERRQ(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  if (!rank) {
    PetscBool match;
    int       fileType, dataSize;

    ierr = PetscViewerASCIIGetPointer(viewer, &fd);CHKERRQ(ierr);
    /* Read header */
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$MeshFormat\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    snum = fscanf(fd, "2.2 %d %d\n", &fileType, &dataSize);CHKERRQ(snum != 2);
    if (fileType) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File type %d is not a valid Gmsh ASCII file", fileType);
    if (dataSize != sizeof(double)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Data size %d is not valid for a Gmsh file", dataSize);
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$EndMeshFormat\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    /* Read vertices */
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$Nodes\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    snum = fscanf(fd, "%d\n", &numVertices);CHKERRQ(snum != 1);
    ierr = PetscMalloc(numVertices*3 * sizeof(PetscScalar), &coordsIn);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {
      double x, y, z;
      int    i;

      snum = fscanf(fd, "%d %lg %lg %lg\n", &i, &x, &y, &z);CHKERRQ(snum != 4);
      coordsIn[v*3+0] = x; coordsIn[v*3+1] = y; coordsIn[v*3+2] = z;
      if (i != v+1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid node number %d should be %d", i, v+1);
    }
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$EndNodes\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    /* Read cells */
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$Elements\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
    snum = fscanf(fd, "%d\n", &numCells);CHKERRQ(snum != 1);
  }

  if (!rank) {
    /* Gmsh elements can be of any dimension/co-dimension, so we need to traverse the
       file contents multiple times to figure out the true number of cells and facets
       in the given mesh. To make this more efficient we read the file contents only
       once and store them in memory, while determining the true number of cells. */
    ierr = PetscMalloc1(numCells, &gmsh_elem);CHKERRQ(ierr);
    for (trueNumCells=0, c = 0; c < numCells; ++c) {
      ierr = DMPlexCreateGmsh_ReadElement(fd, &gmsh_elem[c]);CHKERRQ(ierr);
      if (gmsh_elem[c].dim > dim) {dim = gmsh_elem[c].dim; trueNumCells = 0;}
      if (gmsh_elem[c].dim == dim) trueNumCells++;
    }
    fgets(line, PETSC_MAX_PATH_LEN, fd);
    ierr = PetscStrncmp(line, "$EndElements\n", PETSC_MAX_PATH_LEN, &match);CHKERRQ(ierr);
    if (!match) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "File is not a valid Gmsh file");
  }
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
        ierr = DMPlexSetLabelValue(*dm, "Face Sets", join[0], gmsh_elem[c].tags[0]);CHKERRQ(ierr);
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
  if (!rank) {
    for (c = 0; c < numCells; ++c) {
      ierr = PetscFree(gmsh_elem[c].nodes);
      ierr = PetscFree(gmsh_elem[c].tags);
    }
    ierr = PetscFree(gmsh_elem);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateGmsh_ReadElement"
PetscErrorCode DMPlexCreateGmsh_ReadElement(FILE *fd, GmshElement *ele)
{
  int            snum, cellType;
  PetscInt       t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snum = fscanf(fd, "%d %d %d", &(ele->id), &cellType, &(ele->numTags));CHKERRQ(snum != 3);
  ierr = PetscMalloc1(ele->numTags, &(ele->tags));CHKERRQ(ierr);
  for (t=0; t<ele->numTags; t++) {snum = fscanf(fd, "%d", &(ele->tags[t]));CHKERRQ(snum != 1);}
  switch (cellType) {
  case 1: /* 2-node line */
    ele->dim = 1;
    ele->numNodes = 2;
    ierr = PetscMalloc1(ele->numNodes, &(ele->nodes));CHKERRQ(ierr);
    snum = fscanf(fd, "%d %d\n", &(ele->nodes[0]), &(ele->nodes[1]));CHKERRQ(snum != ele->numNodes);
    break;
  case 2: /* 3-node triangle */
    ele->dim = 2;
    ele->numNodes = 3;
    ierr = PetscMalloc1(ele->numNodes, &(ele->nodes));CHKERRQ(ierr);
    snum = fscanf(fd, "%d %d %d\n", &(ele->nodes[0]), &(ele->nodes[1]), &(ele->nodes[2]));CHKERRQ(snum != ele->numNodes);
    break;
  case 3: /* 4-node quadrangle */
    ele->dim = 2;
    ele->numNodes = 4;
    ierr = PetscMalloc1(ele->numNodes, &(ele->nodes));CHKERRQ(ierr);
    snum = fscanf(fd, "%d %d %d %d\n", &(ele->nodes[0]), &(ele->nodes[1]), &(ele->nodes[2]), &(ele->nodes[3]));CHKERRQ(snum != ele->numNodes);
    break;
  case 4: /* 4-node tetrahedron */
    ele->dim  = 3;
    ele->numNodes = 4;
    ierr = PetscMalloc1(ele->numNodes, &(ele->nodes));CHKERRQ(ierr);
    snum = fscanf(fd, "%d %d %d %d\n", &(ele->nodes[0]), &(ele->nodes[1]), &(ele->nodes[2]), &(ele->nodes[3]));CHKERRQ(snum != ele->numNodes);
    break;
  case 5: /* 8-node hexahedron */
    ele->dim = 3;
    ele->numNodes = 8;
    ierr = PetscMalloc1(ele->numNodes, &(ele->nodes));CHKERRQ(ierr);
    snum = fscanf(fd, "%d %d %d %d %d %d %d %d\n", &(ele->nodes[0]), &(ele->nodes[1]), &(ele->nodes[2]), &(ele->nodes[3]), &(ele->nodes[4]), &(ele->nodes[5]), &(ele->nodes[6]), &(ele->nodes[7]));CHKERRQ(snum != ele->numNodes);
    break;
  case 15: /* 1-node vertex */
    ele->dim = 0;
    ele->numNodes = 1;
    ierr = PetscMalloc1(ele->numNodes, &(ele->nodes));CHKERRQ(ierr);
    snum = fscanf(fd, "%d\n", &(ele->nodes[0]));CHKERRQ(snum != ele->numNodes);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported Gmsh element type %d", cellType);
  }
  PetscFunctionReturn(0);
}
