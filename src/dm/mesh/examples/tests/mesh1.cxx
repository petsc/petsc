static char help[] = "Mesh Tests.\n\n";

#include "petscmesh_formats.hh"
#include "meshTest.hh"

#include <LabelSifter.hh>

using ALE::Obj;
typedef ALE::Mesh::sieve_type        sieve_type;
typedef ALE::Mesh::real_section_type section_type;
typedef ALE::Mesh::label_type        label_type;
typedef ALE::LabelSifter<int,sieve_type::point_type> new_label_type;

typedef struct {
  int        debug;           // The debugging level
  int        test;            // The testing level
  int        dim;             // The topological mesh dimension
  PetscBool  interpolate;     // Construct missing elements of the mesh
  PetscReal  refinementLimit; // The largest allowable cell volume
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->test            = 0;
  options->dim             = 2;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;

  ierr = PetscOptionsBegin(comm, "", "Options for Mesh test", "Mesh");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "mesh1.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-test", "The testing level", "mesh1.cxx", options->test, &options->test, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim",   "The topological mesh dimension", "mesh1.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Construct missing elements of the mesh", "mesh1.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "mesh1.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode PrintMatrix(MPI_Comm comm, int rank, const std::string& name, const int rows, const int cols, const section_type::value_type matrix[])
{
  PetscFunctionBegin;
  PetscSynchronizedPrintf(comm, "%s", ALE::Mesh::printMatrix(name, rows, cols, matrix, rank).c_str());
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GeometryTest"
PetscErrorCode GeometryTest(const Obj<ALE::Mesh>& mesh, const Obj<section_type>& coordinates, Options *options)
{
  const Obj<ALE::Mesh::label_sequence>&     cells  = mesh->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();
  const int                                 dim    = mesh->getDimension();
  const MPI_Comm                            comm   = mesh->comm();
  const int                                 rank   = mesh->commRank();
  const int                                 debug  = mesh->debug();
  section_type::value_type *v0   = new section_type::value_type[dim];
  section_type::value_type *J    = new section_type::value_type[dim*dim];
  section_type::value_type *invJ = new section_type::value_type[dim*dim];
  section_type::value_type  detJ;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
    const sieve_type::point_type& e = *c_iter;

    if (debug) {
      const std::string elem = ALE::Test::MeshProcessor::printElement(e, dim, mesh->restrictClosure(coordinates, e), rank);
      ierr = PetscSynchronizedPrintf(comm, "%s", elem.c_str());CHKERRQ(ierr);
    }
    mesh->computeElementGeometry(coordinates, e, v0, J, invJ, detJ);
    if (debug) {
      ierr = PrintMatrix(comm, rank, "J",    dim, dim, J);CHKERRQ(ierr);
      ierr = PrintMatrix(comm, rank, "invJ", dim, dim, invJ);CHKERRQ(ierr);
    }
    if (detJ < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Negative Jacobian determinant");
  }
  delete [] v0;
  delete [] J;
  delete [] invJ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LabelTest"
PetscErrorCode LabelTest(const Obj<ALE::Mesh>& mesh, const Obj<label_type>& label, Options *options)
{
  const Obj<ALE::Mesh::label_sequence>&     cells  = mesh->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();

  PetscFunctionBegin;
  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
    const sieve_type::point_type& e = *c_iter;

    if (options->test > 2) {
      mesh->setValue(label, e, 1);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NewLabelTest"
PetscErrorCode NewLabelTest(const Obj<ALE::Mesh>& mesh, const Obj<new_label_type>& label, Options *options)
{
  const Obj<ALE::Mesh::label_sequence>&     cells  = mesh->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();

  PetscFunctionBegin;
  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
    const sieve_type::point_type& e = *c_iter;

    if (options->test > 4) {
      label->setCone(1, e);
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
    Obj<ALE::Mesh> boundary;

    if (options.dim == 2) {
      double lower[2] = {0.0, 0.0};
      double upper[2] = {1.0, 1.0};
      int    edges[2] = {2, 2};

      boundary = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options.debug);
    } else if (options.dim == 3) {
      double lower[3] = {0.0, 0.0, 0.0};
      double upper[3] = {1.0, 1.0, 1.0};
      int    faces[3] = {1, 1, 1};

      boundary = ALE::MeshBuilder::createCubeBoundary(comm, lower, upper, faces, options.debug);
    } else {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Dimension not supported: %d", options.dim);
    }
    Obj<ALE::Mesh> mesh = ALE::Generator::generateMesh(boundary, options.interpolate);

    if (mesh->commSize() > 1) {
      mesh = ALE::Distribution<ALE::Mesh>::distributeMesh(mesh);
    }
    if (options.refinementLimit > 0.0) {
      mesh = ALE::Generator::refineMesh(mesh, options.refinementLimit, options.interpolate);
    }
    if (options.debug) {
      mesh->view("Mesh");
    }
    if (options.test > 0) {
      ierr = GeometryTest(mesh, mesh->getRealSection("coordinates"), &options);CHKERRQ(ierr);
    }
    if (options.test > 1) {
      ierr = LabelTest(mesh, mesh->getLabel("marker"), &options);CHKERRQ(ierr);
    }
    if (options.test > 3) {
      Obj<new_label_type> label = new new_label_type(mesh->comm(), mesh->debug());
      ierr = NewLabelTest(mesh, label, &options);CHKERRQ(ierr);
    }
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
