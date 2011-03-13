static char help[] = "Sieve Distribution Tests.\n\n";

#include <petscsys.h>
#include "overlapTest.hh"
#include "meshTest.hh"
#include <Completion.hh>


using ALE::Obj;
typedef ALE::Test::OverlapTest::topology_type                     topology_type;
typedef topology_type::sieve_type                                 sieve_type;
typedef ALE::New::Completion<topology_type, sieve_type::point_type>              sieveCompletion;
typedef ALE::New::Completion<topology_type, ALE::Mesh::section_type::value_type> sectionCompletion;
typedef sectionCompletion::send_overlap_type                      send_overlap_type;
typedef sectionCompletion::recv_overlap_type                      recv_overlap_type;
typedef sectionCompletion::send_section_type                      send_section_type;
typedef sectionCompletion::recv_section_type                      recv_section_type;

typedef struct {
  int        debug;              // The debugging level
  int        dim;                // The topological mesh dimension
  char       baseFilename[2048]; // The base filename for mesh files
  PetscBool  useZeroBase;        // Use zero-based indexing
  PetscBool  interpolate;        // Construct missing elements of the mesh
} Options;

PetscErrorCode SendDistribution(const Obj<ALE::Mesh>& mesh, const Obj<ALE::Mesh>& meshNew)
{
  typedef ALE::New::PatchlessSection<ALE::Mesh::section_type> CoordFiller;
  const int dim   = mesh->getDimension();
  const int debug = mesh->debug;

  PetscFunctionBegin;
  Obj<send_overlap_type> cellOverlap   = sieveCompletion::sendDistribution(mesh->getTopology(), dim, meshNew->getTopology());
  Obj<send_overlap_type> vertexOverlap = new send_overlap_type(mesh->comm(), debug);
  Obj<ALE::Mesh::sieve_type> sieve = mesh->getTopology()->getPatch(0);
  const Obj<send_overlap_type::traits::capSequence> cap = cellOverlap->cap();

  for(send_overlap_type::traits::baseSequence::iterator p_iter = cap->begin(); p_iter != cap->end(); ++p_iter) {
    const Obj<send_overlap_type::traits::supportSequence>& ranks = cellOverlap->support(*p_iter);

    for(send_overlap_type::traits::supportSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
      const Obj<ALE::Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*p_iter);

      for(ALE::Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
        vertexOverlap->addArrow(*c_iter, *r_iter, *c_iter);
      }
    }
  }
  vertexOverlap->view(std::cout, "Send vertex overlap");
  const ALE::Mesh::section_type::patch_type patch = 0;
  const Obj<ALE::Mesh::section_type> coordinates    = mesh->getSection("coordinates");
  const Obj<ALE::Mesh::section_type> coordinatesNew = meshNew->getSection("coordinates");
  const Obj<send_section_type>       sendCoords     = new send_section_type(mesh->comm(), debug);
  const Obj<CoordFiller>             coordFiller    = new CoordFiller(coordinates, patch);
  const int embedDim = coordinates->getAtlas()->getFiberDimension(patch, *mesh->getTopology()->depthStratum(patch, 0)->begin());
  const Obj<sectionCompletion::constant_sizer> constantSizer = new sectionCompletion::constant_sizer(MPI_COMM_SELF, embedDim, debug);

  sectionCompletion::sendSection(vertexOverlap, constantSizer, coordFiller, sendCoords);
  coordinatesNew->getAtlas()->setFiberDimensionByDepth(patch, 0, embedDim);
  coordinatesNew->getAtlas()->orderPatches();
  coordinatesNew->allocate();
  const Obj<ALE::Mesh::topology_type::label_sequence>& vertices = mesh->getTopology()->depthStratum(patch, 0);

  for(ALE::Mesh::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    coordinatesNew->update(patch, *v_iter, coordinates->restrict(patch, *v_iter));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ReceiveDistribution(const Obj<ALE::Mesh>& mesh, const Obj<ALE::Mesh>& meshNew)
{
  PetscFunctionBegin;
  Obj<recv_overlap_type> cellOverlap   = sieveCompletion::receiveDistribution(mesh->getTopology(), meshNew->getTopology());
  Obj<recv_overlap_type> vertexOverlap = new recv_overlap_type(mesh->comm(), mesh->debug);
  Obj<ALE::Mesh::sieve_type> sieveNew = meshNew->getTopology()->getPatch(0);
  const Obj<send_overlap_type::traits::baseSequence> base = cellOverlap->base();

  for(send_overlap_type::traits::baseSequence::iterator p_iter = base->begin(); p_iter != base->end(); ++p_iter) {
    const Obj<send_overlap_type::traits::coneSequence>& ranks = cellOverlap->cone(*p_iter);

    for(send_overlap_type::traits::coneSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
      const Obj<ALE::Mesh::sieve_type::traits::coneSequence>& cone = sieveNew->cone(*p_iter);

      for(ALE::Mesh::sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
        vertexOverlap->addArrow(*r_iter, *c_iter, *c_iter);
      }
    }
  }
  vertexOverlap->view(std::cout, "Receive vertex overlap");
  const Obj<ALE::Mesh::section_type> coordinates    = mesh->getSection("coordinates");
  const Obj<ALE::Mesh::section_type> coordinatesNew = meshNew->getSection("coordinates");
  const Obj<recv_section_type>       recvCoords     = new recv_section_type(mesh->comm(), mesh->debug);
  const ALE::Mesh::section_type::patch_type patch   = 0;

  sectionCompletion::recvSection(vertexOverlap, recvCoords);
  const sectionCompletion::topology_type::sheaf_type& patches = recvCoords->getAtlas()->getTopology()->getPatches();
  const int embedDim = recvCoords->getAtlas()->getFiberDimension(patch, *recvCoords->getAtlas()->getTopology()->depthStratum(patches.begin()->first, 0)->begin());
  coordinatesNew->getAtlas()->setFiberDimensionByDepth(patch, 0, embedDim);
  coordinatesNew->getAtlas()->orderPatches();
  coordinatesNew->allocate();

  for(sectionCompletion::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
    const Obj<sectionCompletion::topology_type::sieve_type::baseSequence>& base = p_iter->second->base();

    for(sectionCompletion::topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      coordinatesNew->update(patch, *b_iter, recvCoords->restrict(p_iter->first, *b_iter));
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributionTest"
// This example does distribution from a central source
PetscErrorCode DistributionTest(const Obj<ALE::Mesh>& mesh, const Obj<ALE::Mesh>& meshNew, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  const Obj<ALE::Mesh::topology_type>& topology = new ALE::Mesh::topology_type(mesh->comm(), mesh->debug);
  const Obj<ALE::Mesh::sieve_type>&    sieve    = new ALE::Mesh::sieve_type(mesh->comm(), mesh->debug);

  topology->setPatch(0, sieve);
  meshNew->setTopology(topology);
  if (mesh->commRank() == 0) {
    ierr = SendDistribution(mesh, meshNew);CHKERRQ(ierr);
  } else {
    ierr = ReceiveDistribution(mesh, meshNew);CHKERRQ(ierr);
  }
  sieve->view("Distributed sieve");
  meshNew->getSection("coordinates")->view("Coordinates");
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
    Obj<ALE::Mesh> mesh = ALE::PCICE::Builder::readMesh(comm, options.dim, options.baseFilename, options.useZeroBase, options.interpolate, options.debug);
    Obj<ALE::Mesh> meshNew = new ALE::Mesh(comm, options.dim, options.debug);

    if (options.debug) {
      mesh->getTopology()->view("Mesh");
    }
    ierr = DistributionTest(mesh, meshNew, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
