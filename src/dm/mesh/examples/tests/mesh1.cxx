static char help[] = "Mesh Tests.\n\n";

#include <petsc.h>
#include "meshTest.hh"

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

PetscErrorCode PrintMatrix(MPI_Comm comm, int rank, const int rows, const int cols, const section_type::value_type matrix[])
{
  PetscFunctionBegin;
  PetscSynchronizedPrintf(comm, "[%d]J = \n", rank);
  for(int r = 0; r < rows; r++) {
    if (r == 0) {
      PetscSynchronizedPrintf(comm, "[%d] /", rank);
    } else if (r == rows-1) {
      PetscSynchronizedPrintf(comm, "[%d] \\", rank);
    } else {
      PetscSynchronizedPrintf(comm, "[%d] |", rank);
    }
    for(int c = 0; c < cols; c++) {
      PetscSynchronizedPrintf(comm, " %g", matrix[r*cols+c]);
    }
    if (r == 0) {
      PetscSynchronizedPrintf(comm, " \\\n");
    } else if (r == rows-1) {
      PetscSynchronizedPrintf(comm, " /\n");
    } else {
      PetscSynchronizedPrintf(comm, " |\n");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ElementGeometry"
PetscErrorCode ElementGeometry(const Obj<section_type>& coordinates, int dim, const sieve_type::point_type& e, section_type::value_type v0[], section_type::value_type J[], section_type::value_type invJ[], section_type::value_type& detJ)
{
  const int                       debug  = coordinates->debug();
  const section_type::patch_type  patch  = 0;
  const section_type::value_type *coords = coordinates->restrict(patch, e);
  PetscReal                       invDet;

  PetscFunctionBegin;
  if (debug) {
    MPI_Comm comm = coordinates->comm();
    int      rank = coordinates->commRank();

    PetscSynchronizedPrintf(comm, "[%d]Element (%d, %d)\n", rank, e.prefix, e.index);
    PetscSynchronizedPrintf(comm, "[%d]Coordinates:\n[%d]  ", rank, rank);
    for(int f = 0; f <= dim; f++) {
      PetscSynchronizedPrintf(comm, " (");
      for(int d = 0; d < dim; d++) {
        if (d > 0) PetscSynchronizedPrintf(comm, ", ");
        PetscSynchronizedPrintf(comm, "%g", coords[f*dim+d]);
      }
      PetscSynchronizedPrintf(comm, ")");
    }
    PetscSynchronizedPrintf(comm, "\n");
  }
  for(int d = 0; d < dim; d++) {
    v0[d] = coords[d];
  }
  for(int d = 0; d < dim; d++) {
    for(int f = 0; f < dim; f++) {
      J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
    }
  }
  if (debug) {PrintMatrix(coordinates->comm(), coordinates->commRank(), dim, dim, J);}
  if (dim == 1) {
    detJ = J[0];
  } else if (dim == 2) {
    detJ = J[0]*J[3] - J[1]*J[2];
  } else if (dim == 3) {
    detJ = J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
           J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
           J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
  }
  invDet = 1.0/detJ;
  if (detJ < 0) {SETERRQ(PETSC_ERR_ARG_WRONG, "Negative Jacobian determinant");}
  if (dim == 2) {
    invJ[0] =  invDet*J[3];
    invJ[1] = -invDet*J[1];
    invJ[2] = -invDet*J[2];
    invJ[3] =  invDet*J[0];
  } else if (dim == 3) {
    // FIX: This may be wrong
    invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
    invJ[0*3+1] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
    invJ[0*3+2] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
    invJ[1*3+0] = invDet*(J[0*3+1]*J[2*3+2] - J[0*3+2]*J[2*3+1]);
    invJ[1*3+1] = invDet*(J[0*3+2]*J[2*3+0] - J[0*3+0]*J[2*3+2]);
    invJ[1*3+2] = invDet*(J[0*3+0]*J[2*3+1] - J[0*3+1]*J[2*3+0]);
    invJ[2*3+0] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
    invJ[2*3+1] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
    invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
  }
  if (debug) {PrintMatrix(coordinates->comm(), coordinates->commRank(), dim, dim, invJ);}
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
    Obj<section_type> coordinates = ALE::Test::MeshBuilder::readMesh(comm, options.dim, options.baseFilename, options.useZeroBase, options.interpolate, options.debug);

    ierr = GeometryTest(coordinates, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
