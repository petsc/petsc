static char help[] = "Sieve Distribution Tests.\n\n";

#include <petsc.h>
#include "meshTest.hh"
#include "overlapTest.hh"

using ALE::Obj;
typedef ALE::Test::sieve_type sieve_type;
typedef ALE::Test::OverlapTest::dsieve_type       dsieve_type;
typedef ALE::Test::OverlapTest::send_overlap_type send_overlap_type;
typedef ALE::Test::OverlapTest::send_section_type send_section_type;
typedef ALE::Test::OverlapTest::recv_overlap_type recv_overlap_type;
typedef ALE::Test::OverlapTest::recv_section_type recv_section_type;

typedef struct {
  int        debug;              // The debugging level
  int        dim;                // The topological mesh dimension
  char       baseFilename[2048]; // The base filename for mesh files
  PetscTruth useZeroBase;        // Use zero-based indexing
  PetscTruth interpolate;        // Construct missing elements of the mesh
} Options;

PetscErrorCode SendDistribution(const Obj<ALE::Mesh>& mesh, Options *options)
{
  Obj<send_overlap_type> sendOverlap = new send_overlap_type(mesh->comm(), options->debug);
  Obj<send_section_type> sendSizer   = new send_section_type(mesh->comm(), send_section_type::SEND, options->debug);
  Obj<send_section_type> sendSection = new send_section_type(mesh->comm(), send_section_type::SEND, options->debug);

  PetscFunctionBegin;
  // 1) Partition the mesh
  short *assignment = ALE::Test::MeshProcessor::partitionMesh_Chaco(mesh);
  // 2) Form partition point overlap a priori
  //      There are arrows to each rank whose color is the partition point (also the rank)
  for(int p = 1; p < mesh->commSize(); p++) {
    sendOverlap->addCone(mesh->commRank(), p, p);
  }
  sendOverlap->view("Send overlap");
  // 3) Form send section sizer
  for(int p = 1; p < mesh->commSize(); p++) {
    Obj<dsieve_type> sendSieve = new dsieve_type(sendOverlap->cone(p));
    sendSizer->getAtlas()->getTopology()->setPatch(p, sendSieve);
  }
  sendSizer->getAtlas()->getTopology()->stratify();
  sendSizer->construct(1);
  sendSizer->getAtlas()->orderPatches();
  sendSizer->allocate();
  sendSizer->constructCommunication();
  // 4) Form send section
  sendSection->getAtlas()->getTopology()->stratify();
  sendSection->construct(1);
  sendSection->getAtlas()->orderPatches();
  sendSection->allocate();
  sendSection->constructCommunication();
  // 5) Restrict from the send section
  const send_section_type::topology_type::sheaf_type& patches = sendSizer->getAtlas()->getTopology()->getPatches();

  for(send_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
    const Obj<send_section_type::sieve_type::baseSequence>& base = p_iter->second->base();
    int                                                     rank = p_iter->first;

    for(send_section_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      int size = sendSection->getAtlas()->size(rank);
      sendSizer->update(rank, *b_iter, &size);
    }
  }
  sendSizer->view("Send Sizer");
  // 5) Complete the sizer
  sendSizer->startCommunication();
  sendSizer->endCommunication();
  PetscFunctionReturn(0);
}

PetscErrorCode ReceiveDistribution(const Obj<ALE::Mesh>& mesh, Options *options)
{
  Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(mesh->comm(), options->debug);
  Obj<recv_section_type> recvSizer   = new recv_section_type(mesh->comm(), recv_section_type::RECEIVE, options->debug);

  PetscFunctionBegin;
  // 1) Form partition point overlap a priori
  //      The arrow is from rank 0 with partition point 0
  recvOverlap->addCone(0, mesh->commRank(), 0);
  recvOverlap->view("Receive overlap");
  // 3) Form receive overlap section sizer
  //      Want to replace this loop with a slice through color
  Obj<dsieve_type> recvSieve = new dsieve_type();
  const Obj<recv_overlap_type::supportSequence>& points = recvOverlap->support(0);

  for(recv_overlap_type::supportSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
    recvSieve->addPoint(p_iter.color());
  }
  recvSizer->getAtlas()->getTopology()->setPatch(0, recvSieve);
  recvSizer->getAtlas()->getTopology()->stratify();
  recvSizer->construct(1);
  recvSizer->getAtlas()->orderPatches();
  recvSizer->allocate();
  recvSizer->constructCommunication();
  // 4) Complete the sizer
  recvSizer->startCommunication();
  recvSizer->endCommunication();
  recvSizer->view("Receive Sizer");
  // 5) Update to the receive section
  const send_section_type::topology_type::sheaf_type& patches = recvSizer->getAtlas()->getTopology()->getPatches();

  for(send_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
    const Obj<send_section_type::sieve_type::baseSequence>& base = p_iter->second->base();
    int                                                     rank = p_iter->first;

    for(send_section_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributionTest"
// This example does distribution from a central source
PetscErrorCode DistributionTest(const Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mesh->commRank() == 0) {
    ierr = SendDistribution(mesh, options);CHKERRQ(ierr);
  } else {
    ierr = ReceiveDistribution(mesh, options);CHKERRQ(ierr);
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
    Obj<ALE::Mesh> mesh = ALE::PCICE::Builder::readMesh(comm, options.dim, options.baseFilename, options.useZeroBase, options.interpolate, options.debug);

    if (options.debug) {
      mesh->getTopologyNew()->getPatch(0)->view("Mesh");
    }
    ierr = DistributionTest(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
