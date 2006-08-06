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
  typedef ALE::New::ConstantSection<send_section_type::topology_type, send_section_type::value_type> constant_section;
  typedef ALE::Test::PartitionSizeSection<send_section_type::topology_type, short int> partition_size_section;
  typedef ALE::Test::PartitionSection<send_section_type::topology_type, short int>     partition_section;
  typedef ALE::Test::ConeSizeSection<send_section_type::topology_type, ALE::Mesh::sieve_type> cone_size_section;
  typedef ALE::Test::ConeSection<send_section_type::topology_type, ALE::Mesh::sieve_type>     cone_section;
  Obj<send_overlap_type> sendOverlap   = new send_overlap_type(mesh->comm(), options->debug);
  Obj<send_section_type> sendSizer     = new send_section_type(mesh->comm(), send_section_type::SEND, options->debug);
  Obj<send_section_type> sendSection   = new send_section_type(mesh->comm(), send_section_type::SEND, options->debug);
  Obj<constant_section>  constantSizer = new constant_section(mesh->comm(), 1, options->debug);
  int numElements = mesh->getTopologyNew()->heightStratum(0, 0)->size();

  PetscFunctionBegin;
  // 1) Form partition point overlap a priori
  //      There are arrows to each rank whose color is the partition point (also the rank)
  for(int p = 1; p < mesh->commSize(); p++) {
    sendOverlap->addCone(p, p, p);
  }
  sendOverlap->view(std::cout, "Send overlap");
  // 2) Create the sizer section
  ALE::Test::Completion::setupSend(sendOverlap, constantSizer, sendSizer);
  // 3) Partition the mesh
  short *assignment = ALE::Test::MeshProcessor::partitionMesh_Chaco(mesh);
  Obj<partition_size_section> partitionSizeSection = new partition_size_section(sendSizer->getAtlas()->getTopology(), numElements, assignment);
  Obj<partition_section>      partitionSection     = new partition_section(sendSizer->getAtlas()->getTopology(), numElements, assignment);
  // 4) Fill the sizer section and communicate
  ALE::Test::Completion::completeSend(partitionSizeSection, sendSizer);
  // 5) Create the send section
  ALE::Test::Completion::setupSend(sendOverlap, sendSizer, sendSection);
  // 6) Fill up send section and communicate
  ALE::Test::Completion::completeSend(partitionSection, sendSection);
  // 7) Create point overlap
  // Could this potentially be the sendSection itself?
  sendOverlap->clear();
  const send_section_type::topology_type::sheaf_type& patches = sendSection->getAtlas()->getTopology()->getPatches();

  for(send_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
    const Obj<send_section_type::sieve_type::baseSequence>& base = p_iter->second->base();

    for(send_section_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      const send_section_type::value_type *points = sendSection->restrict(p_iter->first, *b_iter);
      int size = sendSection->getAtlas()->size(p_iter->first, *b_iter);

      for(int p = 0; p < size; p++) {
        sendOverlap->addArrow(points[p], p_iter->first, points[p]);
      }
    }
  }
  sendOverlap->view(std::cout, "Send overlap");
  // 8) Create the sizer section
  ALE::Test::Completion::setupSend(sendOverlap, constantSizer, sendSizer);
  // 4) Fill the sizer section and communicate
  Obj<cone_size_section> coneSizeSection = new cone_size_section(sendSizer->getAtlas()->getTopology(), mesh->getTopologyNew()->getPatch(0));
  ALE::Test::Completion::completeSend(coneSizeSection, sendSizer);
  // 5) Create the send section
  ALE::Test::Completion::setupSend(sendOverlap, sendSizer, sendSection);
  // 6) Fill up send section and communicate
  Obj<cone_section>      coneSection     = new cone_section(sendSizer->getAtlas()->getTopology(), mesh->getTopologyNew()->getPatch(0));
  ALE::Test::Completion::completeSend(coneSection, sendSection);
  PetscFunctionReturn(0);
}

PetscErrorCode ReceiveDistribution(const Obj<ALE::Mesh>& mesh, Options *options)
{
  typedef ALE::New::ConstantSection<recv_section_type::topology_type, recv_section_type::value_type> constant_section;
  Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(mesh->comm(), options->debug);
  Obj<recv_section_type> recvSizer   = new recv_section_type(mesh->comm(), recv_section_type::RECEIVE, options->debug);
  Obj<recv_section_type> recvSection = new recv_section_type(mesh->comm(), recv_section_type::RECEIVE, options->debug);
  Obj<constant_section>  constantSizer = new constant_section(mesh->comm(), 1, options->debug);

  PetscFunctionBegin;
  // 1) Form partition point overlap a priori
  //      The arrow is from rank 0 with partition point 0
  recvOverlap->addCone(0, mesh->commRank(), mesh->commRank());
  recvOverlap->view(std::cout, "Receive overlap");
  // 2) Create the sizer section
  ALE::Test::Completion::setupReceive(recvOverlap, constantSizer, recvSizer);
  // 3) Communicate
  ALE::Test::Completion::completeReceive(recvSizer);
  // 4) Update to the receive section
  ALE::Test::Completion::setupReceive(recvOverlap, recvSizer, recvSection);
  // 5) Complete the section
  ALE::Test::Completion::completeReceive(recvSection);
  // 6) Unpack the section into the overlap
  recvOverlap->clear();
  const recv_section_type::topology_type::sheaf_type& patches = recvSizer->getAtlas()->getTopology()->getPatches();

  for(recv_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
    const Obj<recv_section_type::sieve_type::baseSequence>& base = p_iter->second->base();
    int                                                     rank = p_iter->first;

    for(recv_section_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      const recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
      int size = recvSection->getAtlas()->getFiberDimension(rank, *b_iter);

      for(int p = 0; p < size; p++) {
        recvOverlap->addArrow(rank, points[p], points[p]);
      }
    }
  }
  recvOverlap->view(std::cout, "Receive overlap");
  // 7) Create the sizer section
  ALE::Test::Completion::setupReceive(recvOverlap, constantSizer, recvSizer);
  // 8) Communicate
  ALE::Test::Completion::completeReceive(recvSizer);
  // 9) Update to the receive section
  ALE::Test::Completion::setupReceive(recvOverlap, recvSizer, recvSection);
  // 10) Complete the section
  ALE::Test::Completion::completeReceive(recvSection);
  // 11) Unpack the section into the sieve
  const Obj<ALE::Mesh::sieve_type>& sieve = mesh->getTopologyNew()->getPatch(0);

  for(recv_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
    const Obj<recv_section_type::sieve_type::baseSequence>& base = p_iter->second->base();
    int                                                     rank = p_iter->first;

    for(recv_section_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      const recv_section_type::value_type *points = recvSection->restrict(rank, *b_iter);
      int size = recvSection->getAtlas()->getFiberDimension(rank, *b_iter);
      int c = 0;

      for(int p = 0; p < size; p++) {
        sieve->addArrow(*b_iter, points[p], c++);
      }
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
      mesh->getTopologyNew()->view("Mesh");
    }
    ierr = DistributionTest(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
