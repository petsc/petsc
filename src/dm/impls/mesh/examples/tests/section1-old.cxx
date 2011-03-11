static char help[] = "Section Tests.\n\n";

#include <petscsys.h>
#include "sectionTest.hh"

using ALE::Obj;
typedef ALE::Test::section_type   section_type;
typedef section_type::atlas_type  atlas_type;

typedef ALE::Test::topology_type         topology_type;
typedef topology_type::sieve_type        sieve_type;
typedef ALE::Test::constant_section_type constant_section_type;
typedef ALE::Test::uniform_section_type  uniform_section_type;

typedef struct {
  int        debug;              // The debugging level
  int        dim;                // The topological mesh dimension
  char       baseFilename[2048]; // The base filename for mesh files
  PetscBool  useZeroBase;        // Use zero-based indexing
  PetscBool  interpolate;        // Construct missing elements of the mesh
} Options;

#undef __FUNCT__
#define __FUNCT__ "ConstantSectionTest"
PetscErrorCode ConstantSectionTest(const Obj<topology_type>& topology, Options *options)
{
  const constant_section_type::value_type value   = 5.0;
  const Obj<constant_section_type>        section = new constant_section_type(topology, value);
  const topology_type::sheaf_type&        patches = topology->getPatches();

  PetscFunctionBegin;
  if (options->debug) {PetscPrintf(section->comm(), "Running %s\n", __FUNCT__);}
  for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter)
  {
    const topology_type::patch_type&          patch   = p_iter->first;
    const Obj<topology_type::label_sequence>& stratum = topology->depthStratum(patch, 0);

    section->setFiberDimensionByDepth(patch, 0, 1);
    if (section->size(patch) != (int) stratum->size()) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid section patch size %d should be %d", section->size(patch), stratum->size());
    }
    for(topology_type::label_sequence::iterator d_iter = stratum->begin(); d_iter != stratum->end(); ++d_iter) {
      if (section->size(patch, *d_iter) != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid section point size");
      if (section->restrict(patch, *d_iter)[0] != value) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid section point value");
    }
  }
  if (options->debug) {section->view("Constant Section");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "UniformSectionTest"
PetscErrorCode UniformSectionTest(const Obj<topology_type>& topology, Options *options)
{
  const Obj<uniform_section_type>  section  = new uniform_section_type(topology);
  uniform_section_type::value_type value[2] = {0, 1};
  const topology_type::sheaf_type& patches  = topology->getPatches();

  PetscFunctionBegin;
  if (options->debug) {PetscPrintf(section->comm(), "Running %s\n", __FUNCT__);}
  for(topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter)
  {
    const topology_type::patch_type&          patch   = p_iter->first;
    const Obj<topology_type::label_sequence>& stratum = topology->depthStratum(patch, 0);

    section->setFiberDimensionByDepth(patch, 0, 2);
    if (section->size(patch) != (int) stratum->size()*2) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid section patch size %d should be %d", section->size(patch), stratum->size());
    }
    for(topology_type::label_sequence::iterator d_iter = stratum->begin(); d_iter != stratum->end(); ++d_iter) {
      if (section->size(patch, *d_iter) != 2) {
        SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Invalid section point size %d should be %d", section->size(patch, *d_iter), 2);
      }
      section->update(patch, *d_iter, value);
      value[0]++;
    }
    value[0] = 0;
    for(topology_type::label_sequence::iterator d_iter = stratum->begin(); d_iter != stratum->end(); ++d_iter) {
      if (section->restrict(patch, *d_iter)[0] != value[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid uniform section point value");
      if (section->restrict(patch, *d_iter)[1] != value[1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid uniform section point value");
      value[0]++;
    }
  }
  if (options->debug) {section->view("Uniform Section");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LinearTest"
PetscErrorCode LinearTest(const Obj<section_type>& section, Options *options)
{
  const Obj<topology_type>& topology = section->getTopology();
  topology_type::patch_type patch    = 0;

  PetscFunctionBegin;
  // Creation
  const Obj<topology_type::label_sequence>& elements = topology->heightStratum(patch, 0);
  const Obj<topology_type::label_sequence>& vertices = topology->depthStratum(patch, 0);
  int depth       = topology->depth();
  int numVertices = vertices->size();
  int numCorners  = topology->getPatch(patch)->nCone(*elements->begin(), depth)->size();
  section_type::value_type *values = new section_type::value_type[numCorners];

  section->clear();
  section->setFiberDimensionByDepth(patch, 0, 1);
  //section->orderPatches();
  section->allocate();
  for(int c = 0; c < numCorners; c++) {values[c] = 3.0;}
  for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    section->updateAdd(patch, *e_iter, values);
  }
  // Verification
  if (section->size(patch) != numVertices) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Linear Test: Invalid patch size %d should be %d", section->size(patch), numVertices);
  }
  for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    if (section->size(patch, *e_iter) != numCorners) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Linear Test: Invalid element size %d should be %d", section->size(patch, *e_iter), numCorners);
    }
  }
  for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    const section_type::value_type *values = section->restrict(patch, *v_iter);
    int neighbors = topology->getPatch(patch)->nSupport(*v_iter, depth)->size();

    if (values[0] != neighbors*3.0) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Linear Test: Invalid vertex value %g should be %g", values[0], neighbors*3.0);
    }
  }
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "CubicTest"
/* This only works for triangles right now
   FAILURE: We need an ordered closure to get the indices to match up with values
*/
PetscErrorCode CubicTest(const Obj<section_type>& section, Options *options)
{
  const Obj<atlas_type>&    atlas    = section->getAtlas();
  const Obj<topology_type>& topology = atlas->getTopology();
  topology_type::patch_type patch    = 0;

  PetscFunctionBegin;
  atlas->clearIndices();
  // Creation
  const Obj<topology_type::label_sequence>& vertices = topology->depthStratum(patch, 0);
  int depth       = topology->depth();
  int numVertices = vertices->size();
  int numEdges    = topology->heightStratum(patch, 1)->size();
  int numFaces    = topology->heightStratum(patch, 0)->size();
  const section_type::value_type values[10] = {3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0};

  atlas->setFiberDimensionByDepth(patch, 0, 1);
  atlas->setFiberDimensionByDepth(patch, 1, 2);
  atlas->setFiberDimensionByDepth(patch, 2, 1);
  atlas->orderPatches();
  section->allocate();
  const Obj<topology_type::label_sequence>& elements = topology->heightStratum(patch, 0);

  for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    section->updateAdd(patch, *e_iter, values);
  }
  section->view("Cubic");
  // Verification
  if (atlas->size(patch) != numVertices + numEdges*2 + numFaces) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid patch size %d should be %d", atlas->size(patch), numVertices + numEdges*2 + numFaces);
  }
  for(topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    const section_type::value_type *values = section->restrictPoint(patch, *e_iter);

    if (atlas->size(patch, *e_iter) != 3 + 3*2 + 1) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid element size %d should be %d", atlas->size(patch, *e_iter), 3 + 3*2 + 1);
    }
    if (values[0] != 3.0) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid cell value %g should be %g", values[0], 3.0);
    }
  }
  for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    const section_type::value_type *values = section->restrict(patch, *v_iter);
    int neighbors = topology->getPatch(patch)->nSupport(*v_iter, depth)->size();

    if (values[0] != neighbors*1.0) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid vertex value %g should be %g", values[0], neighbors*1.0);
    }
  }
  const Obj<topology_type::label_sequence>& edges = topology->heightStratum(patch, 1);

  for(topology_type::label_sequence::iterator e_iter = edges->begin(); e_iter != edges->end(); ++e_iter) {
    const section_type::value_type *values = section->restrict(patch, *e_iter);
    int neighbors = topology->getPatch(patch)->nSupport(*e_iter, depth-1)->size();

    if (values[0] != neighbors*2.0) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid first edge value %g should be %g", values[0], neighbors*2.0);
    }
    if (values[1] != neighbors*2.0) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Cubic Test: Invalid second edge value %g should be %g", values[1], neighbors*2.0);
    }
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "GeneralSectionTest"
PetscErrorCode GeneralSectionTest(const Obj<topology_type>& topology, Options *options)
{
  const Obj<section_type> section = new section_type(topology);
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  if (options->debug) {PetscPrintf(section->comm(), "Running %s\n", __FUNCT__);}

  ierr = LinearTest(section, options);CHKERRQ(ierr);
  if (options->debug) {section->view("General Section");}
  //ierr = CubicTest(section, options);CHKERRQ(ierr);
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
    Obj<topology_type> topology = ALE::Test::TopologyBuilder<topology_type>::readTopology(comm, options.dim, options.baseFilename, options.useZeroBase, options.interpolate, options.debug);

    ierr = ConstantSectionTest(topology, &options);CHKERRQ(ierr);
    ierr = UniformSectionTest(topology, &options);CHKERRQ(ierr);
    ierr = GeneralSectionTest(topology, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
