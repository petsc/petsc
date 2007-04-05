static char help[] = "Mesh Tests.\n\n";

#include <Mesh.hh>
#include <Generator.hh>
#include <src/dm/mesh/meshvtk.h>

using ALE::Obj;
using ALE::Mesh;

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
PetscErrorCode CreatePartition(const Obj<Mesh>& m, const Obj<Mesh::int_section_type>& s)
{
  PetscFunctionBegin;
  const Obj<Mesh::label_sequence>&         cells = m->heightStratum(0);
  const Mesh::int_section_type::value_type rank  = s->commRank();

  s->setFiberDimension(m->heightStratum(0), 1);
  m->allocate(s);
  for(Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    s->updatePoint(*c_iter, &rank);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewMesh"
PetscErrorCode ViewMesh(const Obj<Mesh>& m, const char filename[])
{
  const Obj<Mesh::int_section_type>& partition = m->getIntSection("partition");
  PetscViewer                        viewer;
  PetscErrorCode                     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(m->comm(), &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
  ierr = VTKViewer::writeVertices(m, viewer);CHKERRQ(ierr);
  ierr = VTKViewer::writeElements(m, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(m, partition);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = VTKViewer::writeField(partition, partition->getName(), 1, m->getFactory()->getNumbering(m, m->depth()), viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Obj<Mesh>& m, Options *options)
{
  Obj<Mesh>      mB;
  PetscTruth     view;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (options->dim == 2) {
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};

    mB = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options->debug);
  } else if (options->dim == 3) {
    double lower[3] = {0.0, 0.0, 0.0};
    double upper[3] = {1.0, 1.0, 1.0};
    int    faces[3] = {1, 1, 1};

    mB = ALE::MeshBuilder::createCubeBoundary(comm, lower, upper, faces, options->debug);
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
  }
  if (view) {mB->view("Boundary");}
  m = ALE::Generator::generateMesh(mB, options->interpolate);
  if (view) {m->view("Mesh");}
  if (options->refinementLimit > 0.0) {
    Obj<Mesh> refMesh = ALE::Generator::refineMesh(m, options->refinementLimit, options->interpolate);
    if (view) {refMesh->view("Refined Mesh");}
    m = refMesh;
  } else if (m->commSize() > 1) {
    Obj<Mesh> newMesh = ALE::Distribution<Mesh>::distributeMesh(m);
    if (view) {newMesh->view("Parallel Mesh");}
    m = newMesh;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh(m, "mesh.vtk");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ClosureTest"
PetscErrorCode ClosureTest(const Obj<Mesh>& m, Options *options)
{
  const Obj<Mesh::label_sequence>& cells = m->heightStratum(0);

  PetscFunctionBegin;
  for(Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    const Obj<Mesh::coneArray> closure = ALE::Closure::closure(m, *c_iter);
    Mesh::coneArray::iterator  end     = closure->end();

    std::cout << "Closure of " << *c_iter << std::endl;
    for(Mesh::coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
      std::cout << "  " << *p_iter << std::endl;
    }
  }
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
    Obj<Mesh> m;

    ierr = CreateMesh(comm, m, &options);CHKERRQ(ierr);
    ierr = ClosureTest(m, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
