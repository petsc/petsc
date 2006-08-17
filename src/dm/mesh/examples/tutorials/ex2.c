/*T
   Concepts: Mesh^generating a mesh
   Concepts: Mesh^partitioning a mesh
   Concepts: Mesh^viewing a mesh
   Processors: n
T*/

/*
  Generate a simple square or cube mesh using the builtin mesh generator.

  Partition the mesh and distribute it to each process.

  Output the mesh in VTK format with a scalar field indicating
  the rank of the process owning each cell.
*/

static char help[] = "Generates, partitions, and outputs an unstructured mesh.\n\n";

#include <Generator.hh>
#include "petscmesh.h"
#include "petscviewer.h"

using ALE::Obj;

typedef struct {
  int        debug;              // The debugging level
  PetscInt   dim;                // The topological mesh dimension
  PetscTruth outputVTK;          // Output the mesh in VTK
  PetscTruth distribute;         // Distribute the mesh among processes
  PetscTruth interpolate;        // Construct missing elements of the mesh
  PetscTruth partition;          // Construct field over cells indicating process number
  PetscReal  refinementLimit;    // The maximum volume of a cell after refinement
} Options;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve(const Obj<ALE::Mesh>&, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT FieldView_Sieve(const Obj<ALE::Mesh>&, const std::string&, PetscViewer);
PetscErrorCode ProcessOptions(MPI_Comm, Options *);
PetscErrorCode CreateMeshBoundary(MPI_Comm, Obj<ALE::Mesh>&, Options *);
PetscErrorCode CreateMesh(const Obj<ALE::Mesh>&, Obj<ALE::Mesh>&, Options *);
PetscErrorCode CreatePartition(const Obj<ALE::Mesh>&);
PetscErrorCode DistributeMesh(Obj<ALE::Mesh>&, Options *);
PetscErrorCode RefineMesh(Obj<ALE::Mesh>&, Options *);
PetscErrorCode OutputVTK(const Obj<ALE::Mesh>&, Options *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  try {
    Obj<ALE::Mesh> meshBoundary;
    Obj<ALE::Mesh> mesh;

    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMeshBoundary(comm, meshBoundary, &options);CHKERRQ(ierr);
    meshBoundary->view("Boundary");
    ierr = CreateMesh(meshBoundary, mesh, &options);CHKERRQ(ierr);
    ierr = DistributeMesh(mesh, &options);CHKERRQ(ierr);
    ierr = RefineMesh(mesh, &options);CHKERRQ(ierr);
    ierr = OutputVTK(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->outputVTK       = PETSC_TRUE;
  options->distribute      = PETSC_TRUE;
  options->interpolate     = PETSC_TRUE;
  options->partition       = PETSC_TRUE;
  options->refinementLimit = -1.0;

  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex2.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-output_vtk", "Output the mesh in VTK", "ex2.c", options->outputVTK, &options->outputVTK, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-distribute", "Distribute the mesh among processes", "ex2.c", options->distribute, &options->distribute, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ex2.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-partition", "Create the partition field", "ex2.c", options->partition, &options->partition, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The maximum cell volume", "ex2.c", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
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

 18--5-17--4--16
  |     |     |
  6    10     3
  |     |     |
 19-11-20--9--15
  |     |     |
  7     8     2
  |     |     |
 12--0-13--1--14
*/
PetscErrorCode CreateSquareBoundary(const ALE::Obj<ALE::Mesh>& mesh)
{
  PetscScalar coords[18] = {0.0, 0.0,
                            1.0, 0.0,
                            2.0, 0.0,
                            2.0, 1.0,
                            2.0, 2.0,
                            1.0, 2.0,
                            0.0, 2.0,
                            0.0, 1.0,
                            1.0, 1.0};
  const ALE::Mesh::topology_type::patch_type patch = 0;
  int                   dim   = 1;
  int                   order = 0;
  ALE::Mesh::point_type vertices[9];
  const Obj<ALE::Mesh::sieve_type>    sieve    = new ALE::Mesh::sieve_type(mesh->comm(), mesh->debug);
  const Obj<ALE::Mesh::topology_type> topology = new ALE::Mesh::topology_type(mesh->comm(), mesh->debug);

  PetscFunctionBegin;
  if (mesh->commRank() == 0) {
    ALE::Mesh::point_type edge;

    /* Create topology and ordering */
    for(int v = 12; v < 21; v++) {
      vertices[v-12] = ALE::Mesh::point_type(v);
    }
    for(int e = 0; e < 8; e++) {
      edge = ALE::Mesh::point_type(e);
      sieve->addArrow(vertices[e],       edge, order++);
      sieve->addArrow(vertices[(e+1)%8], edge, order++);
    }
    edge = ALE::Mesh::point_type(8);
    sieve->addArrow(vertices[1], edge, order++);
    sieve->addArrow(vertices[8], edge, order++);
    edge = ALE::Mesh::point_type(9);
    sieve->addArrow(vertices[3], edge, order++);
    sieve->addArrow(vertices[8], edge, order++);
    edge = ALE::Mesh::point_type(10);
    sieve->addArrow(vertices[5], edge, order++);
    sieve->addArrow(vertices[8], edge, order++);
    edge = ALE::Mesh::point_type(11);
    sieve->addArrow(vertices[7], edge, order++);
    sieve->addArrow(vertices[8], edge, order++);
  }
  sieve->view("Boundary sieve");
  sieve->stratify();
  topology->setPatch(patch, sieve);
  topology->stratify();
  mesh->setTopologyNew(topology);
  ALE::PyLith::Builder::buildCoordinates(mesh->getSection("coordinates"), dim+1, coords);
  /* Create boundary conditions */
  if (mesh->commRank() == 0) {
    const Obj<ALE::Mesh::topology_type::patch_label_type>& markers = topology->createLabel(patch, "marker");

    for(int v = 12; v < 20; v++) {
      topology->setValue(markers, v, 1);
    }
    for(int e = 0; e < 8; e++) {
      topology->setValue(markers, e, 1);
    }
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
PetscErrorCode CreateCubeBoundary(const ALE::Obj<ALE::Mesh>& mesh)
{
#if 0
  ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
  PetscScalar       coords[24] = {0.0, 0.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  1.0, 1.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0,
                                  1.0, 0.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  0.0, 1.0, 1.0};
  PetscInt    connectivity[24] = {0, 1, 2, 3,
                                  7, 6, 5, 4,
                                  0, 4, 5, 1,
                                  1, 5, 6, 2,
                                  2, 6, 7, 3,
                                  3, 7, 4, 0};
  ALE::Obj<std::set<ALE::Mesh::point_type> > cone = std::set<ALE::Mesh::point_type>();
  ALE::Mesh::point_type            vertices[8];
  ALE::Mesh::point_type            edges[12];
  ALE::Mesh::point_type            edge;
  PetscInt                              embedDim = 3;
  PetscInt                              order = 0;
  int                                   rank = mesh->commRank();

  PetscFunctionBegin;
  if (rank == 0) {
    ALE::Mesh::point_type face;

    /* Create topology and ordering */
    /* Vertices: 0 .. 3 on the bottom of the cube, 4 .. 7 on the top */
    for(int v = 0; v < 8; v++) {
      vertices[v] = ALE::Mesh::point_type(0, v);
    }

    /* Edges on the bottom: Sieve element numbers e = 8 .. 11, edge numbers e - 8 = 0 .. 3 */
    for(int e = 8; e < 12; e++) {
      edge = ALE::Mesh::point_type(0, e);
      edges[e-8] = edge;
      topology->addArrow(vertices[e-8],     edge, order++);
      topology->addArrow(vertices[(e-7)%4], edge, order++);
    }
    /* Edges on the top: Sieve element numbers e = 12 .. 15, edge numbers e - 8 = 4 .. 7 */
    for(int e = 12; e < 16; e++) {
      edge = ALE::Mesh::point_type(0, e); 
      edges[e-8] = edge;
      topology->addArrow(vertices[e-8],        edge, order++);
      topology->addArrow(vertices[(e-11)%4+4], edge, order++);
    }
    /* Edges from bottom to top: Sieve element numbers e = 16 .. 19, edge numbers e - 8 = 8 .. 11 */
    for(int e = 16; e < 20; e++) {
      edge = ALE::Mesh::point_type(0, e); 
      edges[e-8] = edge;
      topology->addArrow(vertices[e-16],   edge, order++);
      topology->addArrow(vertices[e-16+4], edge, order++);
    }

    /* Bottom face */
    face = ALE::Mesh::point_type(0, 20); 
    topology->addArrow(edges[0], face, order++);
    topology->addArrow(edges[1], face, order++);
    topology->addArrow(edges[2], face, order++);
    topology->addArrow(edges[3], face, order++);
    /* Top face */
    face = ALE::Mesh::point_type(0, 21); 
    topology->addArrow(edges[4], face, order++);
    topology->addArrow(edges[5], face, order++);
    topology->addArrow(edges[6], face, order++);
    topology->addArrow(edges[7], face, order++);
    /* Side faces: f = 22 .. 25 */
    for(int f = 22; f < 26; f++) {
      face = ALE::Mesh::point_type(0, f);
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
    ALE::Obj<ALE::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
    ALE::Obj<ALE::Mesh::bundle_type::PointArray> points = ALE::Mesh::bundle_type::PointArray();
    const std::string orderName("element");
    /* Bottom face */
    ALE::Mesh::point_type face = ALE::Mesh::point_type(0, 20); 
    points->clear();
    points->push_back(vertices[0]);
    points->push_back(vertices[1]);
    points->push_back(vertices[2]);
    points->push_back(vertices[3]);
    vertexBundle->setPatch(orderName, points, face);
    /* Top face */
    face = ALE::Mesh::point_type(0, 21); 
    points->clear();
    points->push_back(vertices[4]);
    points->push_back(vertices[5]);
    points->push_back(vertices[6]);
    points->push_back(vertices[7]);
    vertexBundle->setPatch(orderName, points, face);
    /* Side faces: f = 22 .. 25 */
    for(int f = 22; f < 26; f++) {
      face = ALE::Mesh::point_type(0, f);
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
  mesh->createVertexBundle(12, connectivity, 8);
  mesh->createSerialCoordinates(embedDim, 0, coords);

  /* Create boundary conditions: set marker 1 to all of the sieve elements, 
     since everything is on the boundary (no internal faces, edges or vertices)  */
  if (rank == 0) {
    /* set marker to the base of the topology sieve -- the faces and the edges */
    topology->setMarker(topology->base(), 1);
    /* set marker to the vertices -- the 0-depth stratum */
    topology->setMarker(topology->depthStratum(0), 1);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMeshBoundary"
PetscErrorCode CreateMeshBoundary(MPI_Comm comm, ALE::Obj<ALE::Mesh>& meshBoundary, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  meshBoundary = new ALE::Mesh(comm, options->dim-1, options->debug);
  if (options->dim == 2) {
    ierr = CreateSquareBoundary(meshBoundary);
  } else if (options->dim == 3) {
    ierr = CreateCubeBoundary(meshBoundary);
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Cannot construct a boundary of dimension %d", options->dim);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(const Obj<ALE::Mesh>& meshBoundary, Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshGeneration");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(meshBoundary->comm(), "Generating mesh\n");CHKERRQ(ierr);
  mesh = ALE::Generator::generateMesh(meshBoundary, options->interpolate);
  ALE::LogStagePop(stage);
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopologyNew();
  ierr = PetscPrintf(mesh->comm(), "  Made %d elements\n", topology->heightStratum(0, 0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(mesh->comm(), "  Made %d vertices\n", topology->depthStratum(0, 0)->size());CHKERRQ(ierr);
  if (options->debug) {
    topology->view("Serial topology");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeMesh"
PetscErrorCode DistributeMesh(Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (options->distribute) {
    ALE::LogStage stage = ALE::LogStageRegister("MeshDistribution");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Distributing mesh\n");CHKERRQ(ierr);
    mesh = ALE::New::Distribution<ALE::Mesh::topology_type>::distributeMesh(mesh);
    if (options->partition) {
      ierr = CreatePartition(mesh);CHKERRQ(ierr);
    }
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RefineMesh"
PetscErrorCode RefineMesh(Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (options->refinementLimit > 0.0) {
    ALE::LogStage stage = ALE::LogStageRegister("MeshRefinement");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Refining mesh\n");CHKERRQ(ierr);
    mesh = ALE::Generator::refineMesh(mesh, options->refinementLimit, options->interpolate);
    if (options->partition) {
      ierr = CreatePartition(mesh);CHKERRQ(ierr);
    }
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(const Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->outputVTK) {
    ALE::LogStage stage = ALE::LogStageRegister("VTKOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Creating VTK mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
    ierr = MeshView_Sieve(mesh, viewer);CHKERRQ(ierr);
    if (options->partition) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = FieldView_Sieve(mesh, "partition", viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
/*
  Creates a field whose value is the processor rank on each element
*/
PetscErrorCode CreatePartition(const Obj<ALE::Mesh>& mesh)
{
  Obj<ALE::Mesh::section_type>        partition = mesh->getSection("partition");
  ALE::Mesh::section_type::patch_type patch     = 0;
  ALE::Mesh::section_type::value_type rank      = mesh->commRank();

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  partition->getAtlas()->setFiberDimensionByHeight(patch, 0, 1);
  partition->getAtlas()->orderPatches();
  partition->allocate();
  const Obj<ALE::Mesh::topology_type::label_sequence>& cells = partition->getAtlas()->getTopology()->heightStratum(patch, 0);

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    partition->update(patch, *c_iter, &rank);
  }
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}
