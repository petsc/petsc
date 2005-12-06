/*T
   Concepts: Mesh^generating a mesh
   Concepts: Mesh^partitioning a mesh
   Concepts: Mesh^viewing a mesh
   Processors: n
T*/

/*
  Generate a simple square mesh using the builtin mesh generator.

  Partition the mesh and distribute it to each process.

  Output the mesh in VTK format with a scalar field indicating
  the rank of the process owning each cell.
*/

static char help[] = "Generates, partitions, and outputs an unstructured mesh.\n\n";

#include "petscda.h"
#include "petscviewer.h"

#include <IndexBundle.hh>

PetscErrorCode assembleField(ALE::Obj<ALE::IndexBundle>, ALE::Obj<ALE::PreSieve>, Vec, ALE::Point, PetscScalar[], InsertMode);

extern int debug;
PetscErrorCode CreateMeshBoundary(MPI_Comm, Mesh *);
PetscErrorCode CreatePartitionVector(Mesh, Vec *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Mesh           mesh, meshBoundary;
  Vec            partition;
  PetscViewer    viewer;
  PetscInt       dim;
  PetscReal      refinementLimit;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Options for mesh generation", "DMMG");
    ierr = PetscOptionsInt("-debug", "The debugging flag", "ex2.c", 0, &debug, PETSC_NULL);CHKERRQ(ierr);
    dim  = 2;
    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex2.c", 2, &dim, PETSC_NULL);CHKERRQ(ierr);
    refinementLimit = 0.0;
    ierr = PetscOptionsReal("-refinement_limit", "The area of the largest trianglei the mesh", "ex2.c", 1.0, &refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  comm = PETSC_COMM_WORLD;

  ierr = PetscPrintf(comm, "Generating mesh\n");CHKERRQ(ierr);
  ierr = CreateMeshBoundary(comm, &meshBoundary);CHKERRQ(ierr);
  ierr = MeshGenerate(meshBoundary, &mesh);CHKERRQ(ierr);
  ALE::Obj<ALE::Sieve> topology;
  ierr = MeshGetTopology(mesh, &topology);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0).size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0).size());CHKERRQ(ierr);

  ierr = PetscPrintf(comm, "Distributing mesh\n");CHKERRQ(ierr);
  ierr = MeshDistribute(mesh);CHKERRQ(ierr);
  ierr = CreatePartitionVector(mesh, &partition);CHKERRQ(ierr);

  if (refinementLimit > 0.0) {
    Mesh refinedMesh;

    ierr = PetscPrintf(comm, "Refining mesh\n");CHKERRQ(ierr);
    ierr = MeshRefine(mesh, refinementLimit, PETSC_NULL, &refinedMesh);CHKERRQ(ierr);
    mesh = refinedMesh;
  }

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

  ierr = MeshDestroy(mesh);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMeshBoundary"
/*
  Simple square boundary:

  6--14-5--13-4
  |     |     |
  15   19    12
  |     |     |
  7--20-8--18-3
  |     |     |
  16   17    11
  |     |     |
  0--9--1--10-2
*/
PetscErrorCode CreateMeshBoundary(MPI_Comm comm, Mesh *mesh)
{
  ALE::Obj<ALE::Sieve>       topology = ALE::Sieve(comm);
  ALE::Obj<ALE::PreSieve>    orientation = ALE::PreSieve(comm);
  ALE::Obj<ALE::IndexBundle> elementBundle = ALE::IndexBundle(topology);
  ALE::Obj<ALE::IndexBundle> coordBundle = ALE::IndexBundle(topology);
  ALE::Obj<ALE::Sieve>       boundary = ALE::Sieve(comm);
  Mesh              m;
  Vec               coordinates;
  PetscScalar       coords[18] = {0.0, 0.0,
                                  1.0, 0.0,
                                  2.0, 0.0,
                                  2.0, 1.0,
                                  2.0, 2.0,
                                  1.0, 2.0,
                                  0.0, 2.0,
                                  0.0, 1.0,
                                  1.0, 1.0};
  ALE::Point_set    cone;
  ALE::Point        vertices[9];
  ALE::Point        edge;
  PetscInt          embedDim = 2;
  PetscMPIInt       rank;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);
  ierr = MeshCreate(comm, &m);CHKERRQ(ierr);
  /* Create topology and ordering */
  for(int v = 0; v < 9; v++) {
    ALE::Point vertex(0, v);

    vertices[v] = vertex;
    cone.insert(vertex);
    orientation->addCone(cone, vertex);
    cone.clear();
  }
  for(int e = 9; e < 17; e++) {
    edge = ALE::Point(0, e);
    cone.insert(vertices[e-9]);
    cone.insert(vertices[(e-8)%8]);
    topology->addCone(cone, edge);
    cone.clear();
    cone.insert(vertices[e-9]);
    cone.insert(edge);
    orientation->addCone(cone, edge);
    cone.clear();
  }
  edge = ALE::Point(0, 17);
  cone.insert(vertices[1]);
  cone.insert(vertices[8]);
  topology->addCone(cone, edge);
  cone.clear();
  cone.insert(vertices[1]);
  cone.insert(edge);
  orientation->addCone(cone, edge);
  cone.clear();
  edge = ALE::Point(0, 18);
  cone.insert(vertices[3]);
  cone.insert(vertices[8]);
  topology->addCone(cone, edge);
  cone.clear();
  cone.insert(vertices[3]);
  cone.insert(edge);
  orientation->addCone(cone, edge);
  cone.clear();
  edge = ALE::Point(0, 19);
  cone.insert(vertices[5]);
  cone.insert(vertices[8]);
  topology->addCone(cone, edge);
  cone.clear();
  cone.insert(vertices[5]);
  cone.insert(edge);
  orientation->addCone(cone, edge);
  cone.clear();
  edge = ALE::Point(0, 20);
  cone.insert(vertices[7]);
  cone.insert(vertices[8]);
  topology->addCone(cone, edge);
  cone.clear();
  cone.insert(vertices[7]);
  cone.insert(edge);
  orientation->addCone(cone, edge);
  cone.clear();
  ierr = MeshSetTopology(m, topology);CHKERRQ(ierr);
  ierr = MeshSetOrientation(m, orientation);CHKERRQ(ierr);
  /* Create element numbering */
  elementBundle->setFiberDimensionByHeight(0, 1);
  elementBundle->computeOverlapIndices();
  elementBundle->computeGlobalIndices();
  ierr = MeshSetElementBundle(m, elementBundle);CHKERRQ(ierr);
  /* Create vertex coordinates */
  coordBundle->setFiberDimensionByDepth(0, embedDim);
  coordBundle->computeOverlapIndices();
  coordBundle->computeGlobalIndices();
  ierr = MeshSetCoordinateBundle(m, coordBundle);CHKERRQ(ierr);
  int localSize = coordBundle->getLocalSize();
  int globalSize = coordBundle->getGlobalSize();
  ierr = VecCreate(comm, &coordinates);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, embedDim);CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, localSize, globalSize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
  ALE::Obj<ALE::Point_set> vertexPoints = topology->depthStratum(0);
  for(ALE::Point_set::iterator vertex_itor = vertexPoints->begin(); vertex_itor != vertexPoints->end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;
    ierr = assembleField(coordBundle, orientation, coordinates, v, &coords[v.index*embedDim], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(coordinates);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(coordinates);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(m, coordinates);CHKERRQ(ierr);
  /* Create boundary conditions */
  for(int v = 0; v < 8; v++) {
    cone.insert(ALE::Point(0, v));
  }
  for(int e = 9; e < 17; e++) {
    cone.insert(ALE::Point(0, e));
  }
  ALE::Point boundaryPoint(-1, 1);
  boundary->addCone(cone, boundaryPoint);
  ierr = MeshSetBoundary(m, boundary);CHKERRQ(ierr);
  *mesh = m;
  PetscFunctionReturn(0);
}

extern PetscErrorCode MeshCreateVector(Mesh, ALE::IndexBundle *, int, Vec *);

#undef __FUNCT__
#define __FUNCT__ "CreatePartitionVector"
/*
  Creates a vector whose value is the processor rank on each element
*/
PetscErrorCode CreatePartitionVector(Mesh mesh, Vec *partition)
{
  ALE::Obj<ALE::Sieve> topology;
  PetscScalar   *array;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscInt       n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MeshGetTopology(mesh, &topology);CHKERRQ(ierr);
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
