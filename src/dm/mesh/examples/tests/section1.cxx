static char help[] = "Section Tests.\n\n";

#include <petsc.h>
#include "sectionTest.hh"

using ALE::Obj;
typedef ALE::Test::atlas_type     atlas_type;
typedef ALE::Test::section_type   section_type;
typedef atlas_type::topology_type topology_type;
typedef topology_type::sieve_type sieve_type;

typedef struct {
  int        debug;              // The debugging level
  int        dim;                // The topological mesh dimension
  char       baseFilename[2048]; // The base filename for mesh files
  PetscTruth useZeroBase;        // Use zero-based indexing
  PetscTruth interpolate;        // Construct missing elements of the mesh
} Options;

#undef __FUNCT__
#define __FUNCT__ "LinearTest"
PetscErrorCode LinearTest(const Obj<section_type>& section, Options *options)
{
  const Obj<atlas_type>&    atlas    = section->getAtlas();
  const Obj<topology_type>& topology = atlas->getTopology();
  topology_type::patch_type patch    = 0;

  PetscFunctionBegin;
  // Creation
  const Obj<topology_type::label_sequence>& elements = topology->heightStratum(patch, 0);
  const Obj<topology_type::label_sequence>& vertices = topology->depthStratum(patch, 0);
  int depth       = topology->depth();
  int numVertices = vertices->size();
  int numCorners  = topology->getPatch(patch)->nCone(*elements->begin(), depth)->size();
  section_type::value_type *values = new section_type::value_type[numCorners];

  atlas->setFiberDimensionByDepth(patch, 0, 1);
  section->allocate();
  for(int c = 0; c < numCorners; c++) {values[c] = 3.0;}
  for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    section->updateAdd(patch, *e_iter, values);
  }
  // Verification
  if (atlas->size(patch) != numVertices) {
    SETERRQ2(PETSC_ERR_ARG_SIZ, "Linear Test: Invalid patch size %d should be %d", atlas->size(patch), numVertices);
  }
  for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    if (atlas->size(patch, *e_iter) != numCorners) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "Linear Test: Invalid element size %d should be %d", atlas->size(patch, *e_iter), numCorners);
    }
  }
  for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    const section_type::value_type *values = section->restrict(patch, *v_iter);
    int neighbors = topology->getPatch(patch)->nStar(*v_iter, depth)->size();

    if (values[0] == neighbors*3.0) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "Linear Test: Invalid vertex value %g should be %g", values[0], neighbors*3.0);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CubicTest"
/* This only works for triangles right now */
PetscErrorCode CubicTest(const Obj<section_type>& section, Options *options)
{
  const Obj<atlas_type>&    atlas    = section->getAtlas();
  const Obj<topology_type>& topology = atlas->getTopology();
  topology_type::patch_type patch    = 0;

  PetscFunctionBegin;
  // Creation
  const Obj<topology_type::label_sequence>& vertices = topology->depthStratum(patch, 0);
  int depth       = topology->getPatch(patch)->depth();
  int numVertices = vertices->size();
  int numEdges    = topology->heightStratum(patch, 1)->size();
  int numFaces    = topology->heightStratum(patch, 0)->size();
  const section_type::value_type values[10] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0};

  atlas->setFiberDimensionByDepth(patch, 0, 1);
  atlas->setFiberDimensionByDepth(patch, 1, 2);
  atlas->setFiberDimensionByDepth(patch, 2, 1);
  section->allocate();
  const Obj<topology_type::label_sequence>& elements = topology->heightStratum(patch, 0);

  for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    section->updateAdd(patch, *e_iter, values);
  }
  // Verification
  if (atlas->size(patch) != numVertices + numEdges*2 + numFaces) {
    SETERRQ2(PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid patch size %d should be %d", atlas->size(patch), numVertices + numEdges*2 + numFaces);
  }
  for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    const section_type::value_type *values = section->restrictPoint(patch, *e_iter);

    if (atlas->size(patch, *e_iter) != 3 + 3*2 + 1) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid element size %d should be %d", atlas->size(patch, *e_iter), 3 + 3*2 + 1);
    }
    if (values[0] == 3.0) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid cell value %g should be %g", values[0], 3.0);
    }
  }
  for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    const section_type::value_type *values = section->restrict(patch, *v_iter);
    int neighbors = topology->getPatch(patch)->nStar(*v_iter, depth)->size();

    if (values[0] == neighbors*1.0) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid vertex value %g should be %g", values[0], neighbors*1.0);
    }
  }
  const Obj<topology_type::label_sequence>& edges = topology->heightStratum(patch, 1);

  for(topology_type::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
    const section_type::value_type *values = section->restrict(patch, *e_iter);
    int neighbors = topology->getPatch(patch)->nStar(*e_iter, depth-1)->size();

    if (values[0] == neighbors*2.0) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "Linear Test: Invalid first edge value %g should be %g", values[0], neighbors*2.0);
    }
    if (values[1] == neighbors*2.0) {
      SETERRQ2(PETSC_ERR_ARG_SIZ, "Linear Test: Invalid second edge value %g should be %g", values[1], neighbors*2.0);
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
    Obj<section_type> section = new section_type(comm, options.debug);
    Obj<sieve_type>   sieve   = ALE::Test::SieveBuilder<sieve_type>::readSieve(comm, options.dim, options.baseFilename, options.useZeroBase, options.debug);

    section->getAtlas()->getTopology()->setPatch(0, sieve);
    section->getAtlas()->getTopology()->stratify();
    ierr = LinearTest(section, &options);CHKERRQ(ierr);
    ierr = CubicTest(section, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
