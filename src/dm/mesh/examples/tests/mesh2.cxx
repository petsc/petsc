static char help[] = "Mesh Tests.\n\n";

#include <petscmesh.h>

#include <Generator.hh>
#include <src/dm/mesh/meshvtk.h>

using ALE::Obj;

typedef struct {
  int        debug;           // The debugging level
  int        dim;             // The topological mesh dimension
  PetscTruth interpolate;     // Construct missing elements of the mesh
  PetscReal  refinementLimit; // The largest allowable cell volume
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;

  ierr = PetscOptionsBegin(comm, "", "Options for mesh test", "Mesh");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level",            "mesh2.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim",   "The topological mesh dimension", "mesh2.cxx", options->dim, &options->dim,   PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "mesh2.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "mesh2.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(const Obj<ALE::Field::Mesh>& m/*Mesh mesh*/, const Obj<ALE::Field::Mesh::int_section_type>& s/*SectionInt *partition*/)
{
  //Obj<ALE::Field::Mesh::int_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //ierr = MeshGetCellSectionInt(mesh, 1, partition);CHKERRQ(ierr);
  //ierr = SectionIntGetSection(*partition, s);CHKERRQ(ierr);
  const Obj<ALE::Field::Mesh::label_sequence>&         cells = m->heightStratum(0);
  const ALE::Field::Mesh::int_section_type::value_type rank  = s->commRank();

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    s->updatePoint(*c_iter, &rank);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewMesh"
PetscErrorCode ViewMesh(const Obj<ALE::Field::Mesh>& m/*Mesh mesh*/, const char filename[])
{
  MPI_Comm       comm = m->comm();
  SectionInt     partition;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
#if 0
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(mesh, &partition);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
#else
  ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
  ierr = VTKViewer::writeVertices(m, viewer);CHKERRQ(ierr);
  ierr = VTKViewer::writeElements(m, viewer);CHKERRQ(ierr);
#endif
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, Options *options)
{
  Mesh           boundary, mesh;
  PetscTruth     view;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = MeshCreate(comm, &boundary);CHKERRQ(ierr);
  if (options->dim == 2) {
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};

    Obj<ALE::Field::Mesh> mB = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options->debug);
    mB->view("Boundary");
    Obj<ALE::Field::Mesh> m = ALE::Generator::generateMesh(mB, options->interpolate);
    m->view("Mesh");
    if (options->refinementLimit > 0.0) {
      Obj<ALE::Field::Mesh> refMesh = ALE::Generator::refineMesh(m, options->refinementLimit, options->interpolate);
      refMesh->view("Refined Mesh");
      m = refMesh;
    }
    if (size > 1) {
      Obj<ALE::Field::Mesh> newMesh = ALE::Distribution<ALE::Field::Mesh>::distributeMesh(m);
      newMesh->view("Parallel Mesh");
      m = newMesh;
    }
    ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
    if (view) {ierr = ViewMesh(m, "mesh.vtk");CHKERRQ(ierr);}
#if 0
    ierr = MeshSetMesh(boundary, mB);CHKERRQ(ierr);
  } else if (options->dim == 3) {
    double lower[3] = {0.0, 0.0, 0.0};
    double upper[3] = {1.0, 1.0, 1.0};
    int    faces[3] = {1, 1, 1};

    Obj<ALE::Mesh> mB = ALE::MeshBuilder::createCubeBoundary(comm, lower, upper, faces, options->debug);
    ierr = MeshSetMesh(boundary, mB);CHKERRQ(ierr);
#endif
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
  }
#if 0
  ierr = MeshGenerate(boundary, options->interpolate, &mesh);CHKERRQ(ierr);
  if (options->refinementLimit > 0.0) {
    Mesh refinedMesh;

    ierr = MeshRefine(mesh, options->refinementLimit, options->interpolate, &refinedMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = refinedMesh;
  }
  ierr = MeshDestroy(boundary);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size > 1) {
    Mesh parallelMesh;

    ierr = MeshDistribute(mesh, PETSC_NULL, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = parallelMesh;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh(mesh, "mesh.vtk");CHKERRQ(ierr);}
#endif
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
#if 0
    Obj<ALE::Field::Mesh> m;
    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    m->view("Mesh");
#endif
  }
  *dm = (DM) mesh;
  PetscFunctionReturn(0);
}

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
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    DM dm;
    ierr = CreateMesh(comm, &dm, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
