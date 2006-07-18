static char help[] = "Sifter Performance Stress Tests.\n\n";

#include <petsc.h>
#include "sectionTest.hh"

typedef ALE::Test::atlas_type   atlas_type;
typedef ALE::Test::section_type section_type;

typedef struct {
  int debug; // The debugging level
} Options;

#undef __FUNCT__
#define __FUNCT__ "LinearTest"
PetscErrorCode LinearTest(const ALE::Obj<section_type>& section, Options *options)
{
  typedef section_type::atlas_type::topology_type topology_type;
  const ALE::Obj<atlas_type>&    atlas    = section->getAtlas();
  const ALE::Obj<topology_type>& topology = atlas->getTopology();
  topology_type::patch_type patch;

  PetscFunctionBegin;
  atlas->setFiberDimensionByDepth(patch, 1, 0);

  int numVertices = topology->getPatch(patch)->depthStratum(0)->size();

  if (atlas->size(patch) != numVertices) {
    SETERRQ2(PETSC_ERR_ARG_SIZ, "Invalid patch size %d should be %d", atlas->size(patch), numVertices);
  }
  ALE::Obj<topology_type::sieve_type::traits::heightSequence> elements = topology->getPatch(patch)->heightStratum(0);

  for(topology_type::sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    int numCorners = topology->getPatch(patch)->nCone(*elements->begin(), topology->getPatch(patch)->depth())->size();

    if (atlas->size(patch, *e_iter) != numCorners) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "Invalid element size %d should be %d", atlas->size(patch, *e_iter), numCorners);
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
  options->debug = 0;

  ierr = PetscOptionsBegin(comm, "", "Options for sifter stress test", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "sifter1.c", 0, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &options);CHKERRQ(ierr);
  {
    ALE::Obj<section_type> section = new section_type(PETSC_COMM_WORLD, options.debug);
    topology_type::patch_type patch;

    section->getAtlas()->getTopology()->setPatch(patch) = ;
    ierr = LinearTest(section, &options);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
