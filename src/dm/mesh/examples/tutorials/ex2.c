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

#include <Mesh.hh>
#include "petscmesh.h"
#include "petscviewer.h"

PetscErrorCode MeshView_Sieve_New(ALE::Obj<ALE::def::Mesh>, PetscViewer);

PetscErrorCode CreateMeshBoundary(ALE::Obj<ALE::def::Mesh>);
PetscErrorCode CreatePartitionVector(Mesh, Vec *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  //Vec            partition;
  PetscViewer    viewer;
  PetscInt       dim, debug;
  PetscReal      refinementLimit;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Options for mesh generation", "DMMG");
    debug = 0;
    ierr = PetscOptionsInt("-debug", "The debugging flag", "ex2.c", 0, &debug, PETSC_NULL);CHKERRQ(ierr);
    dim  = 2;
    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex2.c", 2, &dim, PETSC_NULL);CHKERRQ(ierr);
    refinementLimit = 0.0;
    ierr = PetscOptionsReal("-refinement_limit", "The area of the largest triangle in the mesh", "ex2.c", 1.0, &refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  comm = PETSC_COMM_WORLD;

  ALE::Obj<ALE::def::Mesh> meshBoundary = ALE::def::Mesh(comm, dim-1);
  ALE::Obj<ALE::def::Mesh> mesh;

  try {
    ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Generating mesh\n");CHKERRQ(ierr);
    ierr = CreateMeshBoundary(meshBoundary);CHKERRQ(ierr);
    mesh = ALE::def::Generator::generate(meshBoundary);
    ALE::Obj<ALE::def::Mesh::sieve_type> topology = mesh->getTopology();
    ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0)->size());CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0)->size());CHKERRQ(ierr);
#if 0
    ierr = PetscPrintf(comm, "Distributing mesh\n");CHKERRQ(ierr);
    mesh->distribute();
#endif

    if (refinementLimit > 0.0) {
      stage = ALE::LogStageRegister("MeshRefine");
      ALE::LogStagePush(stage);
      ierr = PetscPrintf(comm, "Refining mesh\n");CHKERRQ(ierr);
      mesh = ALE::def::Generator::refine(mesh, refinementLimit);
      ALE::LogStagePop(stage);
    }
    //ierr = CreatePartitionVector(mesh, &partition);CHKERRQ(ierr);

    stage = ALE::LogStageRegister("MeshOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Creating VTK mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
    ierr = MeshView_Sieve_New(mesh, viewer);CHKERRQ(ierr);
    //ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
    //ierr = VecView(partition, viewer);CHKERRQ(ierr);
    //ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSquareBoundary"
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
PetscErrorCode CreateSquareBoundary(ALE::Obj<ALE::def::Mesh> mesh)
{
  MPI_Comm          comm = mesh->getComm();
  ALE::Obj<ALE::def::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::def::Mesh::sieve_type> orientation = mesh->getOrientation();
  PetscScalar       coords[18] = {0.0, 0.0,
                                  1.0, 0.0,
                                  2.0, 0.0,
                                  2.0, 1.0,
                                  2.0, 2.0,
                                  1.0, 2.0,
                                  0.0, 2.0,
                                  0.0, 1.0,
                                  1.0, 1.0};
  ALE::Obj<ALE::def::PointSet> cone = ALE::def::PointSet();
  ALE::def::Mesh::point_type vertices[9];
  PetscMPIInt       rank;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (rank == 0) {
    ALE::def::Mesh::point_type edge;

    /* Create topology and ordering */
    for(int v = 0; v < 9; v++) {
      vertices[v] = ALE::def::Mesh::point_type(0, v);
    }
    for(int e = 9; e < 17; e++) {
      edge = ALE::def::Mesh::point_type(0, e);
      cone->insert(vertices[e-9]);
      cone->insert(vertices[(e-8)%8]);
      topology->addCone(cone, edge);
      cone->clear();
      cone->insert(vertices[e-9]);
      cone->insert(edge);
      orientation->addCone(cone, edge);
      cone->clear();
    }
    edge = ALE::def::Mesh::point_type(0, 17);
    cone->insert(vertices[1]);
    cone->insert(vertices[8]);
    topology->addCone(cone, edge);
    cone->clear();
    cone->insert(vertices[1]);
    cone->insert(edge);
    orientation->addCone(cone, edge);
    cone->clear();
    edge = ALE::def::Mesh::point_type(0, 18);
    cone->insert(vertices[3]);
    cone->insert(vertices[8]);
    topology->addCone(cone, edge);
    cone->clear();
    cone->insert(vertices[3]);
    cone->insert(edge);
    orientation->addCone(cone, edge);
    cone->clear();
    edge = ALE::def::Mesh::point_type(0, 19);
    cone->insert(vertices[5]);
    cone->insert(vertices[8]);
    topology->addCone(cone, edge);
    cone->clear();
    cone->insert(vertices[5]);
    cone->insert(edge);
    orientation->addCone(cone, edge);
    cone->clear();
    edge = ALE::def::Mesh::point_type(0, 20);
    cone->insert(vertices[7]);
    cone->insert(vertices[8]);
    topology->addCone(cone, edge);
    cone->clear();
    cone->insert(vertices[7]);
    cone->insert(edge);
    orientation->addCone(cone, edge);
    cone->clear();
  }
  topology->stratify();
  mesh->createSerialCoordinates(2, 0, coords);
  /* Create boundary conditions */
  if (rank == 0) {
    for(int v = 0; v < 8; v++) {
      cone->insert(ALE::def::Mesh::point_type(0, v));
    }
    for(int e = 9; e < 17; e++) {
      cone->insert(ALE::def::Mesh::point_type(0, e));
    }
    topology->setMarker(cone, 1);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateCubeBoundary"
/*
  Simple cube boundary:

      7-----6
     /|    /|
    3-----2 |
    | |   | |
    | 4---|-5
    |/    |/
    0-----1
*/
PetscErrorCode CreateCubeBoundary(ALE::Obj<ALE::def::Mesh> mesh)
{
  MPI_Comm          comm = mesh->getComm();
  ALE::Obj<ALE::Sieve>    topology = ALE::Sieve(comm);
  ALE::Obj<ALE::PreSieve> orientation = ALE::PreSieve(comm);
  ALE::Obj<ALE::Sieve>    boundary = ALE::Sieve(comm);
  Mesh              m;
  Vec               coordinates;
  PetscScalar       coords[24] = {0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  1.0, 1.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0,
                                  1.0, 0.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  0.0, 1.0, 1.0};
  ALE::Point_set    cone;
  ALE::Point        vertices[8];
  ALE::Point        edges[18];
  ALE::Point        edge;
  PetscInt          embedDim = 3;
  PetscMPIInt       rank, size;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MeshCreate(comm, &m);CHKERRQ(ierr);
  if (rank == 0) {
    ALE::Point face;

    /* Create topology and ordering */
    for(int v = 0; v < 8; v++) {
      ALE::Point vertex(0, v);

      vertices[v] = vertex;
      cone.insert(vertex);
      orientation->addCone(cone, vertex);
      cone.clear();
    }
    for(int e = 8; e < 12; e++) {
      edge = ALE::Point(0, e);
      edges[e-8] = edge;
      cone.insert(vertices[e-8]);
      cone.insert(vertices[(e-8)%4]);
      topology->addCone(cone, edge);
      cone.clear();
/*       cone.insert(vertices[e-8]); */
/*       cone.insert(edge); */
/*       orientation->addCone(cone, edge); */
/*       cone.clear(); */
    }
    for(int e = 12; e < 16; e++) {
      edge = ALE::Point(0, e);
      edges[e-8] = edge;
      cone.insert(vertices[e-12+4]);
      cone.insert(vertices[(e-12)%4+4]);
      topology->addCone(cone, edge);
      cone.clear();
    }
    for(int e = 16; e < 20; e++) {
      edge = ALE::Point(0, e);
      edges[e-8] = edge;
      cone.insert(vertices[e-16]);
      cone.insert(vertices[e-16+4]);
      topology->addCone(cone, edge);
      cone.clear();
    }
    edge = ALE::Point(0, 20);
    edges[12] = edge;
    cone.insert(vertices[1]);
    cone.insert(vertices[3]);
    topology->addCone(cone, edge);
    cone.clear();
    edge = ALE::Point(0, 21);
    edges[13] = edge;
    cone.insert(vertices[5]);
    cone.insert(vertices[2]);
    topology->addCone(cone, edge);
    cone.clear();
    edge = ALE::Point(0, 22);
    edges[14] = edge;
    cone.insert(vertices[4]);
    cone.insert(vertices[6]);
    topology->addCone(cone, edge);
    cone.clear();
    edge = ALE::Point(0, 23);
    edges[15] = edge;
    cone.insert(vertices[0]);
    cone.insert(vertices[7]);
    topology->addCone(cone, edge);
    cone.clear();
    edge = ALE::Point(0, 24);
    edges[16] = edge;
    cone.insert(vertices[0]);
    cone.insert(vertices[5]);
    topology->addCone(cone, edge);
    cone.clear();
    edge = ALE::Point(0, 25);
    edges[17] = edge;
    cone.insert(vertices[2]);
    cone.insert(vertices[7]);
    topology->addCone(cone, edge);
    cone.clear();

    face = ALE::Point(0, 26);
    cone.insert(vertices[2]);
    cone.insert(vertices[7]);
    cone.insert(vertices[7]);
    topology->addCone(cone, edge);
    cone.clear();
  }
  ierr = MeshSetTopology(m, topology);CHKERRQ(ierr);
  ierr = MeshSetOrientation(m, orientation);CHKERRQ(ierr);
  /* Create element numbering */
  ALE::Obj<ALE::IndexBundle> elementBundle = ALE::IndexBundle(topology);
  if (rank == 0) {
    elementBundle->setFiberDimensionByHeight(0, 1);
  }
  elementBundle->computeOverlapIndices();
  elementBundle->computeGlobalIndices();
  ierr = MeshSetElementBundle(m, elementBundle);CHKERRQ(ierr);
  /* Create vertex coordinates */
  ALE::Obj<ALE::IndexBundle> coordBundle = ALE::IndexBundle(topology);
  if (rank == 0) {
    coordBundle->setFiberDimensionByDepth(0, embedDim);
  }
  coordBundle->computeOverlapIndices();
  coordBundle->computeGlobalIndices();
  ierr = MeshSetCoordinateBundle(m, coordBundle);CHKERRQ(ierr);
  /* Store coordinates */
  int localSize = 0, globalSize = 0;

  ierr = VecCreate(comm, &coordinates);CHKERRQ(ierr);
  if (rank == 0) {
    localSize = coordBundle->getLocalSize();
  }
  globalSize = coordBundle->getGlobalSize();
  ierr = VecSetBlockSize(coordinates, embedDim);CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, localSize, globalSize);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinates);CHKERRQ(ierr);
  ALE::Obj<ALE::Point_set> vertexPoints = topology->depthStratum(0);
  for(ALE::Point_set::iterator vertex_itor = vertexPoints->begin(); vertex_itor != vertexPoints->end(); vertex_itor++) {
    ALE::Point v = *vertex_itor;
    //ierr = assembleField(coordBundle, orientation, coordinates, v, &coords[v.index*embedDim], INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(coordinates);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(coordinates);CHKERRQ(ierr);
  ierr = MeshSetCoordinates(m, coordinates);CHKERRQ(ierr);
  /* Create boundary conditions */
  if (rank == 0) {
    for(int v = 0; v < 8; v++) {
      cone.insert(ALE::Point(0, v));
    }
    for(int e = 9; e < 17; e++) {
      cone.insert(ALE::Point(0, e));
    }
    ALE::Point boundaryPoint(-1, 1);
    boundary->addCone(cone, boundaryPoint);
  }
  ierr = MeshSetBoundary(m, boundary);CHKERRQ(ierr);
  //*mesh = m;
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
PetscErrorCode CreateMeshBoundary(ALE::Obj<ALE::def::Mesh> mesh)
{
  int            dim = mesh->getDimension();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dim == 1) {
    ierr = CreateSquareBoundary(mesh);
  } else if (dim == 2) {
    ierr = CreateCubeBoundary(mesh);
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Cannot construct a boundary of dimension %d", dim);
  }
  PetscFunctionReturn(0);
}

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
  //ierr = MeshCreateVector(mesh, &elementBundle, debug, partition);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*partition, 1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(*partition, &n);CHKERRQ(ierr);
  ierr = VecGetArray(*partition, &array);CHKERRQ(ierr);
  for(i = 0; i < n; i++) {
    array[i] = rank;
  }
  ierr = VecRestoreArray(*partition, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
