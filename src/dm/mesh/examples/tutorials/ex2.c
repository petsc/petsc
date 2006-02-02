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

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve_Newer(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer);
PetscErrorCode CreateMeshBoundary(ALE::Obj<ALE::Two::Mesh>);
PetscErrorCode CreatePartitionVector(ALE::Obj<ALE::def::Mesh>, Vec *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  //Vec            partition;
  PetscViewer    viewer;
  PetscInt       dim, debug;
  PetscReal      refinementLimit;
  PetscTruth     interpolate;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Options for mesh generation", "DMMG");CHKERRQ(ierr);
    debug = 0;
    ierr = PetscOptionsInt("-debug", "The debugging flag", "ex2.c", 0, &debug, PETSC_NULL);CHKERRQ(ierr);
    dim  = 2;
    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex2.c", 2, &dim, PETSC_NULL);CHKERRQ(ierr);
    interpolate = PETSC_TRUE;
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ex2.c", PETSC_TRUE, &interpolate, PETSC_NULL);CHKERRQ(ierr);
    refinementLimit = 0.0;
    ierr = PetscOptionsReal("-refinement_limit", "The area of the largest triangle in the mesh", "ex2.c", 1.0, &refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  comm = PETSC_COMM_WORLD;

  ALE::Obj<ALE::Two::Mesh> meshBoundary = ALE::Two::Mesh(comm, dim-1, debug);
  ALE::Obj<ALE::Two::Mesh> mesh;

  try {
    ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Generating mesh\n");CHKERRQ(ierr);
    ierr = CreateMeshBoundary(meshBoundary);CHKERRQ(ierr);
    mesh = ALE::Two::Generator::generate(meshBoundary);
    ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
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
      mesh = ALE::Two::Generator::refine(mesh, refinementLimit);
      ALE::LogStagePop(stage);
      ierr = PetscPrintf(comm, "  Read %d elements\n", mesh->getTopology()->heightStratum(0)->size());CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "  Read %d vertices\n", mesh->getTopology()->depthStratum(0)->size());CHKERRQ(ierr);
    }
    //ierr = CreatePartitionVector(mesh, &partition);CHKERRQ(ierr);

    stage = ALE::LogStageRegister("MeshOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Creating VTK mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
    ierr = MeshView_Sieve_Newer(mesh, viewer);CHKERRQ(ierr);
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
PetscErrorCode CreateSquareBoundary(ALE::Obj<ALE::Two::Mesh> mesh)
{
  MPI_Comm          comm = mesh->getComm();
  ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
  PetscScalar       coords[18] = {0.0, 0.0,
                                  1.0, 0.0,
                                  2.0, 0.0,
                                  2.0, 1.0,
                                  2.0, 2.0,
                                  1.0, 2.0,
                                  0.0, 2.0,
                                  0.0, 1.0,
                                  1.0, 1.0};
  ALE::Two::Mesh::point_type vertices[9];
  PetscInt          order = 0;
  PetscMPIInt       rank;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (rank == 0) {
    ALE::Two::Mesh::point_type edge;

    /* Create topology and ordering */
    for(int v = 0; v < 9; v++) {
      vertices[v] = ALE::Two::Mesh::point_type(0, v);
    }
    for(int e = 9; e < 17; e++) {
      edge = ALE::Two::Mesh::point_type(0, e);
      topology->addArrow(vertices[e-9],     edge, order++);
      topology->addArrow(vertices[(e-8)%8], edge, order++);
    }
    edge = ALE::Two::Mesh::point_type(0, 17);
    topology->addArrow(vertices[1], edge, order++);
    topology->addArrow(vertices[8], edge, order++);
    edge = ALE::Two::Mesh::point_type(0, 18);
    topology->addArrow(vertices[3], edge, order++);
    topology->addArrow(vertices[8], edge, order++);
    edge = ALE::Two::Mesh::point_type(0, 19);
    topology->addArrow(vertices[5], edge, order++);
    topology->addArrow(vertices[8], edge, order++);
    edge = ALE::Two::Mesh::point_type(0, 20);
    topology->addArrow(vertices[7], edge, order++);
    topology->addArrow(vertices[8], edge, order++);
  }
  topology->stratify();
  mesh->createSerialCoordinates(2, 0, coords);
  /* Create boundary conditions */
  if (rank == 0) {
    ALE::Obj<ALE::def::PointSet> cone = ALE::def::PointSet();

    for(int v = 0; v < 8; v++) {
      cone->insert(ALE::Two::Mesh::point_type(0, v));
    }
    for(int e = 9; e < 17; e++) {
      cone->insert(ALE::Two::Mesh::point_type(0, e));
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
PetscErrorCode CreateCubeBoundary(ALE::Obj<ALE::Two::Mesh> mesh)
{
  MPI_Comm          comm = mesh->getComm();
  ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
  PetscScalar       coords[24] = {0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  1.0, 1.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0,
                                  1.0, 0.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  0.0, 1.0, 1.0};

  ALE::Obj<std::set<ALE::Two::Mesh::point_type> > cone = std::set<ALE::Two::Mesh::point_type>();
  ALE::Two::Mesh::point_type            vertices[8];
  ALE::Two::Mesh::point_type            edges[12];
  ALE::Two::Mesh::point_type            edge;
  PetscInt                              embedDim = 3;
  PetscInt                              order = 0;
  PetscMPIInt                           rank;
  PetscErrorCode                        ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (rank == 0) {
    ALE::Two::Mesh::point_type face;

    /* Create topology and ordering */
    /* Vertices: 0 .. 3 on the bottom of the cube, 4 .. 7 on the top */
    for(int v = 0; v < 8; v++) {
      vertices[v] = ALE::Two::Mesh::point_type(0, v);
    }

    /* Edges on the bottom: Sieve element numbers e = 8 .. 11, edge numbers e - 8 = 0 .. 3 */
    for(int e = 8; e < 12; e++) {
      edge = ALE::Two::Mesh::point_type(0, e);
      edges[e-8] = edge;
      topology->addArrow(vertices[e-8],     edge, order++);
      topology->addArrow(vertices[(e-7)%4], edge, order++);
    }
    /* Edges on the top: Sieve element numbers e = 12 .. 15, edge numbers e - 8 = 4 .. 7 */
    for(int e = 12; e < 16; e++) {
      edge = ALE::Two::Mesh::point_type(0, e); 
      edges[e-8] = edge;
      topology->addArrow(vertices[e-8],        edge, order++);
      topology->addArrow(vertices[(e-11)%4+4], edge, order++);
    }
    /* Edges from bottom to top: Sieve element numbers e = 16 .. 19, edge numbers e - 8 = 8 .. 11 */
    for(int e = 16; e < 20; e++) {
      edge = ALE::Two::Mesh::point_type(0, e); 
      edges[e-8] = edge;
      topology->addArrow(vertices[e-16],   edge, order++);
      topology->addArrow(vertices[e-16+4], edge, order++);
    }

    /* Bottom face */
    face = ALE::Two::Mesh::point_type(0, 20); 
    topology->addArrow(edges[0], face, order++);
    topology->addArrow(edges[1], face, order++);
    topology->addArrow(edges[2], face, order++);
    topology->addArrow(edges[3], face, order++);
    /* Top face */
    face = ALE::Two::Mesh::point_type(0, 21); 
    topology->addArrow(edges[4], face, order++);
    topology->addArrow(edges[5], face, order++);
    topology->addArrow(edges[6], face, order++);
    topology->addArrow(edges[7], face, order++);
    /* Side faces: f = 22 .. 25 */
    for(int f = 22; f < 26; f++) {
      face = ALE::Two::Mesh::point_type(0, f);
      int v = f - 22;
      /* Covered by edges f - 22, f - 22 + 4, f - 22 + 8, (f - 21)%4 + 8 */
      topology->addArrow(edges[v],         face, order++);
      topology->addArrow(edges[(v+1)%4+8], face, order++);
      topology->addArrow(edges[v+4],       face, order++);
      topology->addArrow(edges[v+8],       face, order++);
    }
  }/* if(rank == 0) */
  topology->stratify();
  if (rank == 0) {
    ALE::Obj<ALE::Two::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
    ALE::Obj<ALE::Two::Mesh::bundle_type::PointArray> points = ALE::Two::Mesh::bundle_type::PointArray();
    const std::string orderName("element");
    /* Bottom face */
    ALE::Two::Mesh::point_type face = ALE::Two::Mesh::point_type(0, 20); 
    points->clear();
    points->push_back(vertices[0]);
    points->push_back(vertices[1]);
    points->push_back(vertices[2]);
    points->push_back(vertices[3]);
    vertexBundle->setPatch(orderName, points, face);
    /* Top face */
    face = ALE::Two::Mesh::point_type(0, 21); 
    points->clear();
    points->push_back(vertices[4]);
    points->push_back(vertices[5]);
    points->push_back(vertices[6]);
    points->push_back(vertices[7]);
    vertexBundle->setPatch(orderName, points, face);
    /* Side faces: f = 22 .. 25 */
    for(int f = 22; f < 26; f++) {
      face = ALE::Two::Mesh::point_type(0, f);
      int v = f - 22;
      /* Covered by edges f - 22, f - 22 + 4, f - 22 + 8, (f - 21)%4 + 8 */
      points->clear();
      points->push_back(vertices[v]);
      points->push_back(vertices[(v+1)%4]);
      points->push_back(vertices[(v+1)%4+4]);
      points->push_back(vertices[v+4]);
      vertexBundle->setPatch(orderName, points, face);
    }
  }
  mesh->createSerialCoordinates(embedDim, 0, coords);

  /* Create boundary conditions: set marker 1 to all of the sieve elements, 
     since everything is on the boundary (no internal faces, edges or vertices)  */
  if (rank == 0) {
    /* set marker to the base of the topology sieve -- the faces and the edges */
    topology->setMarker(topology->base(), 1);
    /* set marker to the vertices -- the 0-depth stratum */
    topology->setMarker(topology->depthStratum(0), 1);
  }

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
PetscErrorCode CreateMeshBoundary(ALE::Obj<ALE::Two::Mesh> mesh)
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
PetscErrorCode CreatePartitionVector(ALE::Obj<ALE::def::Mesh> mesh, Vec *partition)
{
  ALE::Obj<ALE::def::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::def::Mesh::bundle_type> elementBundle = mesh->getBundle(mesh->getDimension());
  PetscScalar   *array;
  PetscMPIInt    rank;
  PetscInt       n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(mesh->getComm(), &rank);CHKERRQ(ierr);
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
