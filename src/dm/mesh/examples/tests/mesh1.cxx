static char help[] = "Mesh Tests.\n\n";

#include <petsc.h>
#include <Distribution.hh>
#include "meshTest.hh"
#include "../../meshpcice.h"

using ALE::Obj;
typedef ALE::Test::section_type   section_type;
typedef section_type::atlas_type  atlas_type;
typedef atlas_type::topology_type topology_type;
typedef topology_type::sieve_type sieve_type;

typedef struct {
  int        debug;              // The debugging level
  int        dim;                // The topological mesh dimension
  char       baseFilename[2048]; // The base filename for mesh files
  PetscTruth useZeroBase;        // Use zero-based indexing
  PetscTruth interpolate;        // Construct missing elements of the mesh
} Options;

extern PetscErrorCode updateOperator(Mat A, const ALE::Obj<ALE::Mesh::atlas_type>& atlas, const ALE::Obj<ALE::Mesh::numbering_type>& globalOrder, const ALE::Mesh::point_type& e, PetscScalar array[], InsertMode mode);


PetscErrorCode PrintMatrix(MPI_Comm comm, int rank, const std::string& name, const int rows, const int cols, const section_type::value_type matrix[])
{
  PetscFunctionBegin;
  PetscSynchronizedPrintf(comm, "%s", ALE::Test::MeshProcessor::printMatrix(name, rows, cols, matrix, rank).c_str());
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ElementGeometry"
PetscErrorCode ElementGeometry(const Obj<section_type>& coordinates, int dim, const sieve_type::point_type& e, section_type::value_type v0[], section_type::value_type J[], section_type::value_type invJ[], section_type::value_type& detJ)
{
  const int debug = coordinates->debug();

  PetscFunctionBegin;
  if (debug) {PetscSynchronizedPrintf(coordinates->comm(), "%s", ALE::Test::MeshProcessor::printElement(e, dim, coordinates->restrict(0, e), coordinates->commRank()).c_str());}
  ALE::Test::MeshProcessor::computeElementGeometry(coordinates, dim, e, v0, J, invJ, detJ);
  if (debug) {PrintMatrix(coordinates->comm(), coordinates->commRank(), "J", dim, dim, J);}
  if (detJ < 0) {SETERRQ(PETSC_ERR_ARG_WRONG, "Negative Jacobian determinant");}
  if (debug) {PrintMatrix(coordinates->comm(), coordinates->commRank(), "invJ", dim, dim, invJ);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GeometryTest"
PetscErrorCode GeometryTest(const Obj<section_type>& coordinates, Options *options)
{
  const Obj<topology_type::label_sequence>& elements = coordinates->getAtlas()->getTopology()->heightStratum(0, 0);
  section_type::value_type *v0   = new section_type::value_type[options->dim];
  section_type::value_type *J    = new section_type::value_type[options->dim*options->dim];
  section_type::value_type *invJ = new section_type::value_type[options->dim*options->dim];
  section_type::value_type  detJ;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  coordinates->view("Coordinates");
  for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    ierr = ElementGeometry(coordinates, options->dim, *e_iter, v0, J, invJ, detJ); CHKERRQ(ierr);
  }
  delete [] v0;
  delete [] J;
  delete [] invJ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AllocationTest"
PetscErrorCode AllocationTest(const Obj<ALE::Mesh>& mesh, Options *options)
{
  std::string                           name("coordinates");
  const Obj<ALE::Mesh::numbering_type>& order   = mesh->getGlobalOrder(name);
  const Obj<ALE::Mesh::section_type>&   section = mesh->getSection(name);
  Mat                                   A;
  PetscErrorCode                        ierr;

  PetscFunctionBegin;
  ierr = MatCreate(mesh->comm(), &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, order->getLocalSize(), order->getLocalSize(), order->getGlobalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = preallocateMatrix(mesh.objPtr, section, order, A);CHKERRQ(ierr);
  const Obj<ALE::Mesh::topology_type::label_sequence>& elements = mesh->getTopologyNew()->heightStratum(0, 0);
  int          size       = options->dim*options->dim*9;
  PetscScalar *elementMat = new PetscScalar[size];

  for(int i = 0; i < size; ++i) elementMat[i] = 1.0;
  for(ALE::Mesh::topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    ierr = updateOperator(A, section->getAtlas(), order, *e_iter, elementMat, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug       = 0;
  options->dim         = 2;
  ierr = PetscStrcpy(options->baseFilename, "../tutorials/data/ex1_2d");CHKERRQ(ierr);
  options->useZeroBase = PETSC_TRUE;
  options->interpolate = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Options for sifter stress test", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level",            "section1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim",   "The topological mesh dimension", "section1.c", options->dim, &options->dim,   PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "section1.c", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "section1.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ex1.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
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
    Obj<ALE::Mesh> mesh = ALE::PCICE::Builder::readMesh(comm, options.dim, options.baseFilename, options.useZeroBase, options.interpolate, options.debug);

    if (options.debug) {
      mesh->getTopologyNew()->getPatch(0)->view("Mesh");
    }
    mesh = ALE::New::Distribution<ALE::Mesh::topology_type>::distributeMesh(mesh);
    ierr = GeometryTest(mesh->getSection("coordinates"), &options);CHKERRQ(ierr);
    ierr = AllocationTest(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
