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
#include "../src/dm/mesh/meshpcice.h"

using ALE::Obj;

typedef enum {PCICE, PYLITH} FileType;

typedef struct {
  int        debug;              // The debugging level
  PetscInt   dim;                // The topological mesh dimension
  PetscTruth inputBd;            // Read mesh boundary from a file
  PetscTruth useZeroBase;        // Use zero-based indexing
  FileType   inputFileType;      // The input file type, e.g. PCICE
  char       baseFilename[2048]; // The base filename for mesh files
  PetscTruth outputVTK;          // Output the mesh in VTK
  PetscTruth distribute;         // Distribute the mesh among processes
  PetscTruth interpolate;        // Construct missing elements of the mesh
  PetscTruth partition;          // Construct field over cells indicating process number
  PetscReal  refinementLimit;    // The maximum volume of a cell after refinement
} Options;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve(const Obj<ALE::Mesh>&, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT FieldView_Sieve(const Obj<ALE::Mesh>&, const std::string&, PetscViewer);
PetscErrorCode ProcessOptions(MPI_Comm, Options *);
PetscErrorCode CreateMeshBoundary(MPI_Comm, Mesh*, Options *);
PetscErrorCode CreateMesh(Mesh, Mesh*, Options *);
PetscErrorCode CreatePartition(Mesh);
PetscErrorCode DistributeMesh(Mesh*, Options *);
PetscErrorCode RefineMesh(Mesh*, Options *);
PetscErrorCode OutputVTK(Mesh, Options *);

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
    Mesh meshBoundary;
    Mesh mesh;

    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMeshBoundary(comm, &meshBoundary, &options);CHKERRQ(ierr);
    ierr = CreateMesh(meshBoundary, &mesh, &options);CHKERRQ(ierr);
    ierr = DistributeMesh(&mesh, &options);CHKERRQ(ierr);
    ierr = RefineMesh(&mesh, &options);CHKERRQ(ierr);
    ierr = OutputVTK(mesh, &options);CHKERRQ(ierr);
    ierr = MeshDestroy(meshBoundary);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
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
  const char    *fileTypes[2] = {"pcice", "pylith"};
  PetscInt       inputFt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->inputBd         = PETSC_FALSE;
  options->useZeroBase     = PETSC_TRUE;
  options->inputFileType   = PCICE;
  ierr = PetscStrcpy(options->baseFilename, "illinois/stbnd_w_lake");CHKERRQ(ierr);
  options->outputVTK       = PETSC_TRUE;
  options->distribute      = PETSC_TRUE;
  options->interpolate     = PETSC_TRUE;
  options->partition       = PETSC_TRUE;
  options->refinementLimit = -1.0;

  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex2.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-input_bd", "Read mesh boundary from a file", "ex2.c", options->inputBd, &options->inputBd, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "ex2.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-file_type", "Type of input files", "ex2.c", fileTypes, 2, fileTypes[0], &inputFt, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex2.c", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-output_vtk", "Output the mesh in VTK", "ex2.c", options->outputVTK, &options->outputVTK, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-distribute", "Distribute the mesh among processes", "ex2.c", options->distribute, &options->distribute, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ex2.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-partition", "Create the partition field", "ex2.c", options->partition, &options->partition, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The maximum cell volume", "ex2.c", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  options->inputFileType = (FileType) inputFt;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSquareBoundary"
/*
  Simple square boundary:

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
  const Obj<ALE::Mesh::sieve_type>    sieve    = new ALE::Mesh::sieve_type(mesh->comm(), mesh->debug());
  const Obj<ALE::Mesh::topology_type> topology = new ALE::Mesh::topology_type(mesh->comm(), mesh->debug());

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
  sieve->stratify();
  topology->setPatch(patch, sieve);
  topology->stratify();
  mesh->setTopology(topology);
  ALE::New::SieveBuilder<ALE::Mesh::sieve_type>::buildCoordinates(mesh->getRealSection("coordinates"), dim+1, coords);
  /* Create boundary conditions */
  const Obj<ALE::Mesh::topology_type::patch_label_type>& markers = topology->createLabel(patch, "marker");

  if (mesh->commRank() == 0) {
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
PetscErrorCode CreateMeshBoundary(MPI_Comm comm, Mesh *meshBoundary, Options *options)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, meshBoundary);CHKERRQ(ierr);
  if (options->inputBd) {
    //m = ALE::PCICE::Builder::readMesh(comm, options->dim, options->baseFilename, options->useZeroBase, options->interpolate, options->debug);
    if (m->commRank() == 0) {
      const Obj<ALE::Mesh::topology_type>&                   topology = m->getTopology();
      const Obj<ALE::Mesh::topology_type::label_sequence>&   vertices = topology->depthStratum(0, 0);
      const Obj<ALE::Mesh::topology_type::label_sequence>&   edges    = topology->depthStratum(0, 1);
      const Obj<ALE::Mesh::topology_type::patch_label_type>& markers  = topology->createLabel(0, "marker");

      for(ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        topology->setValue(markers, *v_iter, 1);
      }
      for(ALE::Mesh::topology_type::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
        topology->setValue(markers, *e_iter, 1);
      }
    }
  } else {
    if (options->dim == 2) {
      double lower[2] = {0.0, 0.0};
      double upper[2] = {2.0, 2.0};
      int    edges[2] = {2, 2};

      m = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options->debug);
    } else if (options->dim == 3) {
      m = new ALE::Mesh(comm, options->dim-1, options->debug);
      ierr = CreateCubeBoundary(m);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP, "Cannot construct a boundary of dimension %d", options->dim);
    }
  }
  if (options->debug) {
    m->view("Boundary");
  }
  ierr = MeshSetMesh(*meshBoundary, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(Mesh meshBoundary, Mesh *mesh, Options *options)
{
  MPI_Comm       comm;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshGeneration");
  ALE::LogStagePush(stage);
  ierr = PetscObjectGetComm((PetscObject) meshBoundary, &comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Generating mesh\n");CHKERRQ(ierr);
  ierr = MeshGenerate(meshBoundary, options->interpolate, mesh);CHKERRQ(ierr);
  ALE::LogStagePop(stage);
  ierr = MeshGetMesh(*mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::topology_type>& topology = m->getTopology();
  ierr = PetscPrintf(m->comm(), "  Made %d elements\n", topology->heightStratum(0, 0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(m->comm(), "  Made %d vertices\n", topology->depthStratum(0, 0)->size());CHKERRQ(ierr);
  if (options->debug) {
    topology->view("Serial topology");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeMesh"
PetscErrorCode DistributeMesh(Mesh *mesh, Options *options)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->distribute) {
    Mesh parallelMesh;

    ALE::LogStage stage = ALE::LogStageRegister("MeshDistribution");
    ALE::LogStagePush(stage);
    ierr = PetscObjectGetComm((PetscObject) *mesh, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Distributing mesh\n");CHKERRQ(ierr);
    ierr = MeshDistribute(*mesh, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(*mesh);CHKERRQ(ierr);
    *mesh = parallelMesh;
    if (options->partition) {
      ierr = CreatePartition(*mesh);CHKERRQ(ierr);
    }
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RefineMesh"
PetscErrorCode RefineMesh(Mesh *mesh, Options *options)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->refinementLimit > 0.0) {
    Mesh refinedMesh;

    ALE::LogStage stage = ALE::LogStageRegister("MeshRefinement");
    ALE::LogStagePush(stage);
    ierr = PetscObjectGetComm((PetscObject) *mesh, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Refining mesh\n");CHKERRQ(ierr);
    ierr = MeshRefine(*mesh, options->refinementLimit, options->interpolate, &refinedMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(*mesh);CHKERRQ(ierr);
    *mesh = refinedMesh;
    if (options->partition) {
      ierr = CreatePartition(*mesh);CHKERRQ(ierr);
    }
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(Mesh mesh, Options *options)
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->outputVTK) {
    ALE::LogStage stage = ALE::LogStageRegister("VTKOutput");
    ALE::LogStagePush(stage);
    ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Creating VTK mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
    ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
    if (options->partition) {
      SectionInt partition;

      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = MeshGetSectionInt(mesh, "partition", &partition);CHKERRQ(ierr);
      ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
      ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(Mesh mesh)
{
  Obj<ALE::Mesh> m;
  SectionInt     partition;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  ierr = MeshGetCellSectionInt(mesh, 1, &partition);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) partition, "partition");CHKERRQ(ierr);
  ierr = MeshSetSectionInt(mesh, partition);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const ALE::Mesh::int_section_type::patch_type            patch    = 0;
  const Obj<ALE::Mesh::topology_type>&                     topology = m->getTopology();
  const Obj<ALE::Mesh::topology_type::label_sequence>&     cells    = topology->heightStratum(patch, 0);
  const ALE::Mesh::topology_type::label_sequence::iterator end      = cells->end();
  const ALE::Mesh::int_section_type::value_type            rank     = m->commRank();

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    ierr = SectionIntUpdate(partition, *c_iter, &rank);CHKERRQ(ierr);
  }
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}
