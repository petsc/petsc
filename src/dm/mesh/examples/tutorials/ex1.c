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

typedef enum {PCICE, PYLITH} FileType;

PetscErrorCode ReadConnectivity(MPI_Comm, FileType, const char *, PetscInt, PetscTruth, PetscInt *, PetscInt **);
PetscErrorCode ReadCoordinates(MPI_Comm, FileType, const char *, PetscInt, PetscInt *, PetscScalar **);
PetscErrorCode ReadConnectivity_PCICE(MPI_Comm, const char *, PetscInt, PetscTruth, PetscInt *, PetscInt **);
PetscErrorCode ReadCoordinates_PCICE(MPI_Comm, const char *, PetscInt, PetscInt *, PetscScalar **);
PetscErrorCode ReadConnectivity_PyLith(MPI_Comm, const char *, PetscInt, PetscTruth, PetscInt *, PetscInt **);
PetscErrorCode ReadCoordinates_PyLith(MPI_Comm, const char *, PetscInt, PetscInt *, PetscScalar **);
PetscErrorCode CreatePartitionVector(Mesh, Vec *);
extern int debug;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Mesh           mesh;
  PetscViewer    viewer;
  Vec            partition;
  char           vertexFilename[2048];
  char           coordFilename[2048];
  PetscTruth     useZeroBase;
  const char    *fileTypes[2] = {"pcice", "pylith"};
  FileType       fileType;
  PetscInt      *vertices;
  PetscScalar   *coordinates;
  PetscInt       dim, numVertices, numElements, ft;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Options for the inhomogeneous Poisson equation", "DMMG");
    ierr = PetscOptionsInt("-debug", "The debugging flag", "ex1.c", 0, &debug, PETSC_NULL);CHKERRQ(ierr);
    dim  = 2;
    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex1.c", 2, &dim, PETSC_NULL);CHKERRQ(ierr);
    useZeroBase = PETSC_FALSE;
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "ex1.c", PETSC_FALSE, &useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ft   = (PetscInt) PCICE;
    ierr = PetscOptionsEList("-file_type", "Type of input files", "ex1.c", fileTypes, 2, fileTypes[0], &ft, PETSC_NULL);CHKERRQ(ierr);
    fileType = (FileType) ft;
    ierr = PetscStrcpy(vertexFilename, "lcon.dat");CHKERRQ(ierr);
    ierr = PetscOptionsString("-vertex_file", "The file listing the vertices of each cell", "ex1.c", "lcon.dat", vertexFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(coordFilename, "nodes.dat");CHKERRQ(ierr);
    ierr = PetscOptionsString("-coord_file", "The file listing the coordinates of each vertex", "ex1.c", "nodes.dat", coordFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  comm = PETSC_COMM_WORLD;

  ierr = ReadConnectivity(comm, fileType, vertexFilename, dim, useZeroBase, &numElements, &vertices);CHKERRQ(ierr);
  ierr = ReadCoordinates(comm, fileType, coordFilename, dim, &numVertices, &coordinates);CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
  ierr = MeshCreate(comm, &mesh);CHKERRQ(ierr);
  ierr = MeshCreateSeq(mesh, dim, numVertices, numElements, vertices, coordinates);CHKERRQ(ierr);
  //ierr = MeshCreateBoundary(mesh, 8, boundaryVertices); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Distributing mesh\n");CHKERRQ(ierr);
  ierr = MeshDistribute(mesh);CHKERRQ(ierr);

  ierr = CreatePartitionVector(mesh, &partition);CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "Creating VTK mesh file\n");CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = VecView(partition, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  if (fileType == PCICE) {
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PCICE);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.lcon");CHKERRQ(ierr);
  } else if (fileType == PYLITH) {
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.connect");CHKERRQ(ierr);
  }
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  ierr = MeshDestroy(mesh);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadConnectivity"
PetscErrorCode ReadConnectivity(MPI_Comm comm, FileType fileType, const char *filename, PetscInt dim, PetscTruth useZeroBase, PetscInt *numElements, PetscInt **vertices)
{
  if (fileType == PCICE) {
    return ReadConnectivity_PCICE(comm, filename, dim, useZeroBase, numElements, vertices);
  } else if (fileType == PYLITH) {
    return ReadConnectivity_PyLith(comm, filename, dim, useZeroBase, numElements, vertices);
  }
  SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "Unknown input file type: %d", fileType);
}

#undef __FUNCT__
#define __FUNCT__ "ReadCoordinates"
PetscErrorCode ReadCoordinates(MPI_Comm comm, FileType fileType, const char *filename, PetscInt dim, PetscInt *numVertices, PetscScalar **coordinates)
{
  if (fileType == PCICE) {
    return ReadCoordinates_PCICE(comm, filename, dim, numVertices, coordinates);
  } else if (fileType == PYLITH) {
    return ReadCoordinates_PyLith(comm, filename, dim, numVertices, coordinates);
  }
  SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "Unknown input file type: %d", fileType);
}

#undef __FUNCT__
#define __FUNCT__ "ReadConnectivity_PCICE"
PetscErrorCode ReadConnectivity_PCICE(MPI_Comm comm, const char *filename, PetscInt dim, PetscTruth useZeroBase, PetscInt *numElements, PetscInt **vertices)
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
  ierr = PetscPrintf(comm, "  Read %d elements\n", *numElements);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadCoordinates_PCICE"
PetscErrorCode ReadCoordinates_PCICE(MPI_Comm comm, const char *filename, PetscInt dim, PetscInt *numVertices, PetscScalar **coordinates)
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
  ierr = PetscPrintf(comm, "  Read %d vertices\n", *numVertices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IgnoreComments_PyLith"
PetscErrorCode IgnoreComments_PyLith(char *buf, PetscInt bufSize, FILE *f)
{
  PetscFunctionBegin;
  while((fgets(buf, bufSize, f) != NULL) && (buf[0] == '#')) {}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadConnectivity_PyLith"
PetscErrorCode ReadConnectivity_PyLith(MPI_Comm comm, const char *filename, PetscInt dim, PetscTruth useZeroBase, PetscInt *numElements, PetscInt **vertices)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       maxCells = 1024, cellCount = 0;
  PetscInt      *verts;
  char           buf[2048];
  PetscInt       c;
  PetscInt       commSize, commRank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &commSize); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &commRank); CHKERRQ(ierr);

  if (dim != 3) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "PyLith only works in 3D");
  }
  ierr = PetscPrintf(comm, "Reading connectivity information on proc 0 of %d procs from file %s...\n", commSize, filename);
  CHKERRQ(ierr);
  if(commRank == 0) {
    ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
    ierr = PetscViewerASCIIGetPointer(viewer, &f);CHKERRQ(ierr);
    /* Ignore comments */
    IgnoreComments_PyLith(buf, 2048, f);
    ierr = PetscMalloc(maxCells*(dim+1) * sizeof(PetscInt), &verts);CHKERRQ(ierr);
    do {
      const char *v = strtok(buf, " ");
      int         elementType;

      if (cellCount == maxCells) {
        PetscInt *vtmp;

        vtmp = verts;
        ierr = PetscMalloc(maxCells*2*(dim+1) * sizeof(PetscInt), &verts);CHKERRQ(ierr);
        ierr = PetscMemcpy(verts, vtmp, maxCells*(dim+1) * sizeof(PetscInt));CHKERRQ(ierr);
        ierr = PetscFree(vtmp);CHKERRQ(ierr);
        maxCells *= 2;
      }
      /* Ignore cell number */
      v = strtok(NULL, " ");
      /* Verify element type is linear tetrahedron */
      elementType = atoi(v);
      if (elementType != 5) {
        SETERRQ(PETSC_ERR_ARG_WRONG, "We only accept linear tetrahedra right now");
      }
      v = strtok(NULL, " ");
      /* Ignore material type */
      v = strtok(NULL, " ");
      /* Ignore infinite domain element code */
      v = strtok(NULL, " ");
      for(c = 0; c <= dim; c++) {
        int vertex = atoi(v);
        
        if (!useZeroBase) vertex -= 1;
        verts[cellCount*(dim+1)+c] = vertex;
        v = strtok(NULL, " ");
      }
      cellCount++;
    } while(fgets(buf, 2048, f) != NULL);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    *numElements = cellCount;
    *vertices = verts;
  }
  ierr = PetscPrintf(comm, "  Read %d elements\n", *numElements);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadCoordinates_PyLith"
PetscErrorCode ReadCoordinates_PyLith(MPI_Comm comm, const char *filename, PetscInt dim, PetscInt *numVertices, PetscScalar **coordinates)
{
  PetscViewer    viewer;
  FILE          *f;
  PetscInt       maxVerts = 1024, vertexCount = 0;
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
    /* Ignore comments and units line */
    IgnoreComments_PyLith(buf, 2048, f);
    ierr = PetscMalloc(maxVerts*dim * sizeof(PetscScalar), &coords);CHKERRQ(ierr);
    while(fgets(buf, 2048, f) != NULL) {
      const char *x = strtok(buf, " ");

      if (vertexCount == maxVerts) {
        PetscScalar *ctmp;

        ctmp = coords;
        ierr = PetscMalloc(maxVerts*2*dim * sizeof(PetscScalar), &coords);CHKERRQ(ierr);
        ierr = PetscMemcpy(coords, ctmp, maxVerts*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = PetscFree(ctmp);CHKERRQ(ierr);
        maxVerts *= 2;
      }
      /* Ignore vertex number */
      x = strtok(NULL, " ");
      for(c = 0; c < dim; c++) {
        coords[vertexCount*dim+c] = atof(x);
        x = strtok(NULL, " ");
      }
      vertexCount++;
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    *numVertices = vertexCount;
    *coordinates = coords;
  }
  ierr = PetscPrintf(comm, "  Read %d vertices\n", *numVertices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <IndexBundle.hh>
extern PetscErrorCode MeshCreateVector(Mesh, ALE::IndexBundle *, int, Vec *);

#undef __FUNCT__
#define __FUNCT__ "CreatePartitionVector"
/*
  Creates a vector whose value is the processor rank on each element
*/
PetscErrorCode CreatePartitionVector(Mesh mesh, Vec *partition)
{
  ALE::Sieve    *topology;
  PetscScalar   *array;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscInt       n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, (void **) &topology);CHKERRQ(ierr);
  ALE::IndexBundle elementBundle(topology);
  elementBundle.setFiberDimensionByHeight(0, 1);
  elementBundle.computeOverlapIndices();
  elementBundle.computeGlobalIndices();
  ierr = MeshCreateVector(mesh, &elementBundle, debug, partition);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*partition, 1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(*partition, &n);CHKERRQ(ierr);
  ierr = VecGetArray(*partition, &array);CHKERRQ(ierr);
  for(i = 0; i < n; i++) {
    array[i] = rank;
  }
  ierr = VecRestoreArray(*partition, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
