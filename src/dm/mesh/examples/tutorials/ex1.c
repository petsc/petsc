/*T
   Concepts: Mesh^loading a mesh
   Concepts: Mesh^partitioning a mesh
   Concepts: Mesh^viewing a mesh
   Processors: n
T*/

/*
  Read in a mesh using the PCICE format:

  connectivity file:
  ------------------
  NumCells
  Cell #   v_0 v_1 ... v_d
  .
  .
  .

  coordinate file:
  ----------------
  NumVertices
  Vertex #  x_0 x_1 ... x_{d-1}
  .
  .
  .

Partition the mesh and distribute it to each process.

Output the mesh in VTK format with a scalar field indicating
the rank of the process owning each cell.
*/

static char help[] = "Reads, partitions, and outputs an unstructured mesh.\n\n";

#include "petscda.h"
#include "petscviewer.h"
#include <stdlib.h>
#include <string.h>

PetscErrorCode ReadConnectivity(const char *, PetscInt, PetscInt **);
PetscErrorCode ReadCoordinates(const char *, PetscInt, PetscScalar **);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  char           vertexFilename[2048];
  char           coordFilename[2048];
  PetscInt      *vertices;
  PetscScalar   *coordinates;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMMG");
    dim = 2;
    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex1.c", 2, &dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(vertexFilename, "lcon.dat");CHKERRQ(ierr);
    ierr = PetscOptionsString("-vertex_file", "The file listing the vertices of each cell", "ex1.c", "lcon.dat", vertexFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(coordFilename, "nodes.dat");CHKERRQ(ierr);
    ierr= PetscOptionsString("-coord_file", "The file listing the coordinates of each vertex", "ex1.c", "nodes.dat", coordFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = ReadConnectivity(vertexFilename, dim, &vertices);CHKERRQ(ierr);
  ierr = ReadCoordinates(coordFilename, dim, &coordinates);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadConnectivity"
PetscErrorCode ReadConnectivity(const char *filename, PetscInt dim, PetscInt **vertices)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       numCells, cellCount = 0;
  PetscInt      *verts;
  char           buf[2048];
  PetscInt       c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Reading connectivity information from %s...\n", filename);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerASCIISetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerSetFilename(viewer, filename);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
  numCells = atoi(fgets(buf, 2048, f));
  ierr = PetscMalloc(numCells*(dim+1) * sizeof(PetscInt), &verts);CHKERRQ(ierr);
  while(fgets(buf, 2048, f) != NULL) {
    const char *v = strtok(buf, " ");

    /* Ignore cell number */
    v = strtok(NULL, " ");
    for(c = 0; c <= dim; c++) {
      verts[cellCount*(dim+1)+c] = atoi(v)-1;
      v = strtok(NULL, " ");
    }
    cellCount++;
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "  Read %d elements\n", numCells);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  *vertices = verts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadCoordinates"
PetscErrorCode ReadCoordinates(const char *filename, PetscInt dim, PetscScalar **coordinates)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       numVertices, vertexCount = 0;
  PetscScalar   *coords;
  char           buf[2048];
  PetscInt       c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Reading coordinate information from %s...\n", filename);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerASCIISetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerSetFilename(viewer, filename);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
  numVertices = atoi(fgets(buf, 2048, f));
  ierr = PetscMalloc(numVertices*dim * sizeof(PetscScalar), &coords);CHKERRQ(ierr);
  while(fgets(buf, 2048, f) != NULL) {
    const char *x = strtok(buf, " ");

    /* Ignore vertex number */
    x = strtok(NULL, " ");
    for(c = 0; c < dim; c++) {
      coords[vertexCount*dim+c] = atof(x);
      x = strtok(NULL, " ");
    }
    vertexCount++;
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD, "  Read %d vertices\n", numVertices);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
