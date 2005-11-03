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

PetscErrorCode ReadConnectivity(MPI_Comm, const char *, PetscInt, PetscTruth, PetscInt *, PetscInt **);
PetscErrorCode ReadCoordinates(MPI_Comm, const char *, PetscInt, PetscInt *, PetscScalar **);
extern int debug;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Mesh           mesh;
  PetscViewer    viewer;
  char           vertexFilename[2048];
  char           coordFilename[2048];
  PetscTruth     useZeroBase;
  PetscInt      *vertices;
  PetscScalar   *coordinates;
  PetscInt       dim, numVertices, numElements;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Options for the inhomogeneous Poisson equation", "DMMG");
    ierr = PetscOptionsInt("-debug", "The debugging flag", "ex1.c", 0, &debug, PETSC_NULL);CHKERRQ(ierr);
    dim = 2;
    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex1.c", 2, &dim, PETSC_NULL);CHKERRQ(ierr);
    useZeroBase = PETSC_FALSE;
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "ex1.c", PETSC_FALSE, &useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(vertexFilename, "lcon.dat");CHKERRQ(ierr);
    ierr = PetscOptionsString("-vertex_file", "The file listing the vertices of each cell", "ex1.c", "lcon.dat", vertexFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(coordFilename, "nodes.dat");CHKERRQ(ierr);
    ierr= PetscOptionsString("-coord_file", "The file listing the coordinates of each vertex", "ex1.c", "nodes.dat", coordFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  comm = PETSC_COMM_WORLD;

  ierr = ReadConnectivity(comm, vertexFilename, dim, useZeroBase, &numElements, &vertices);CHKERRQ(ierr);
  ierr = ReadCoordinates(comm, coordFilename, dim, &numVertices, &coordinates);CHKERRQ(ierr);

  ierr = MeshCreate(comm, &mesh);CHKERRQ(ierr);
  ierr = MeshCreateSeq(mesh, dim, numVertices, numElements, vertices, coordinates);CHKERRQ(ierr);
  //ierr = MeshCreateBoundary(mesh, 8, boundaryVertices); CHKERRQ(ierr);
  ierr = MeshDistribute(mesh);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadConnectivity"
PetscErrorCode ReadConnectivity(MPI_Comm comm, const char *filename, PetscInt dim, PetscTruth useZeroBase, PetscInt *numElements, PetscInt **vertices)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       numCells, cellCount = 0;
  PetscInt      *verts;
  char           buf[2048];
  PetscInt       c;
  PetscInt       commSize, commRank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &commSize); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &commRank); CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "Reading connectivity information on proc 0 of %d procs from file %s...\n", commSize, filename);
  CHKERRQ(ierr);
  if(commRank == 0) {
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
    numCells = atoi(fgets(buf, 2048, f));
    ierr = PetscMalloc(numCells*(dim+1) * sizeof(PetscInt), &verts);CHKERRQ(ierr);
    while(fgets(buf, 2048, f) != NULL) {
      const char *v = strtok(buf, " ");
      
      /* Ignore cell number */
      v = strtok(NULL, " ");
      for(c = 0; c <= dim; c++) {
        int vertex = atoi(v);
        
        if (!useZeroBase) vertex -= 1;
        verts[cellCount*(dim+1)+c] = vertex;
        v = strtok(NULL, " ");
      }
      cellCount++;
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    *numElements = numCells;
    *vertices = verts;
  }
  ierr = PetscPrintf(comm, "  Read %d elements\n", numCells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadCoordinates"
PetscErrorCode ReadCoordinates(MPI_Comm comm, const char *filename, PetscInt dim, PetscInt *numVertices, PetscScalar **coordinates)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       numVerts, vertexCount = 0;
  PetscScalar   *coords;
  char           buf[2048];
  PetscInt       c;
  PetscInt       commSize, commRank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &commSize); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &commRank); CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "Reading coordinate information on proc 0 of %d procs from file %s...\n", commSize, filename); CHKERRQ(ierr);
  if (commRank == 0) {
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
    numVerts = atoi(fgets(buf, 2048, f));
    ierr = PetscMalloc(numVerts*dim * sizeof(PetscScalar), &coords);CHKERRQ(ierr);
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
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    *numVertices = numVerts;
    *coordinates = coords;
  }
  ierr = PetscPrintf(comm, "  Read %d vertices\n", numVerts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
