static char help[] = "Topology Tests.\n\n";

#include <petscsys.h>
#include "topologyTest.hh"

using ALE::Obj;
typedef ALE::Test::topology_type  topology_type;
typedef topology_type::sieve_type sieve_type;

typedef struct {
  int        debug;              // The debugging level
  int        dim;                // The topological mesh dimension
  char       baseFilename[2048]; // The base filename for mesh files
  PetscBool  useZeroBase;        // Use zero-based indexing
  PetscBool  interpolate;        // Construct missing elements of the mesh
} Options;

#undef __FUNCT__
#define __FUNCT__ "StratificationTest"
PetscErrorCode StratificationTest(const Obj<topology_type>& topology, Options *options)
{
  PetscFunctionBegin;
  topology->stratify();
  const topology_type::sheaf_type& patches = topology->getPatches();

  for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
    const topology_type::patch_type& patch = p_iter->first;
    int eulerChi, d, h;

    if (options->dim != topology->depth()) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid topology depth %d, should be %d", topology->depth(), options->dim);
    }
    if (options->dim != topology->height()) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid topology height %d, should be %d", topology->height(), options->dim);
    }
    if (options->debug) {topology->getLabel(patch, "height")->view("height");}
    // Calculate the Euler characteristic
    for(h = 0, eulerChi = 0; h <= topology->height(); h++) {
      if (h%2) {
        eulerChi -= topology->heightStratum(patch, h)->size();
      } else {
        eulerChi += topology->heightStratum(patch, h)->size();
      }
    }
    if (eulerChi != 1) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid Euler characteristic %d, should be %d", eulerChi, 0);
    }
    if (options->debug) {topology->getLabel(patch, "depth")->view("depth");}
    for(d = 0, eulerChi = 0; d <= topology->depth(); d++) {
      if (d%2) {
        eulerChi -= topology->depthStratum(patch, d)->size();
      } else {
        eulerChi += topology->depthStratum(patch, d)->size();
      }
    }
    if (eulerChi != 1) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid Euler characteristic %d, should be %d", eulerChi, 0);
    }
  }
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
    ierr = PetscOptionsBool("-use_zero_base", "Use zero-based indexing", "section1.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Construct missing elements of the mesh", "ex1.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
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
    Obj<topology_type> topology = ALE::Test::TopologyBuilder<topology_type>::readTopology(comm, options.dim, options.baseFilename, options.useZeroBase, options.interpolate, options.debug, false);

    ierr = StratificationTest(topology, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
