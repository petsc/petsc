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
  typedef ALE::New::ConstantSection<send_section_type::topology_type, send_section_type::index_type, send_section_type::value_type> constant_section;
  typedef ALE::Test::PartitionSection<send_section_type::topology_type, send_section_type::index_type, short int> partition_section;
  Obj<send_overlap_type> sendOverlap   = new send_overlap_type(mesh->comm(), options->debug);
  Obj<send_section_type> sendSizer     = new send_section_type(mesh->comm(), send_section_type::SEND, options->debug);
  Obj<send_section_type> sendSection   = new send_section_type(mesh->comm(), send_section_type::SEND, options->debug);
  Obj<constant_section>  constantSizer = new constant_section(mesh->comm(), 1, options->debug);
  int numElements = mesh->getTopologyNew()->heightStratum(0, 0)->size();

  PetscFunctionBegin;
  // 1) Form partition point overlap a priori
  //      There are arrows to each rank whose color is the partition point (also the rank)
  for(int p = 1; p < mesh->commSize(); p++) {
    sendOverlap->addCone(mesh->commRank(), p, p);
  }
  sendOverlap->view(std::cout, "Send overlap");
  ALE::Test::Completion::setupSend(sendOverlap, constantSizer, sendSizer);
  // 2) Partition the mesh
  short *assignment = ALE::Test::MeshProcessor::partitionMesh_Chaco(mesh);
  Obj<partition_section> partitionSection = new partition_section(sendSizer->getAtlas()->getTopology(), numElements, assignment);
  ALE::Test::Completion::completeSend(sendOverlap, partitionSection, sendSizer);
//   // 3) Form send section sizer
//   for(int p = 1; p < mesh->commSize(); p++) {
//     Obj<dsieve_type> sendSieve = new dsieve_type(sendOverlap->cone(p));
//     sendSizer->getAtlas()->getTopology()->setPatch(p, sendSieve);
//   }
//   sendSizer->getAtlas()->getTopology()->stratify();
//   sendSizer->construct(1);
//   sendSizer->getAtlas()->orderPatches();
//   sendSizer->allocate();
//   sendSizer->constructCommunication();
  // 4) Form send section
  sendSection->getAtlas()->setTopology(sendSizer->getAtlas()->getTopology());
  for(int p = 1; p < mesh->commSize(); p++) {
    int size = 0;

    for(int e = 0; e < numElements; e++) {
      if (assignment[e] == p) size++;
    }
    sendSection->getAtlas()->setFiberDimension(p, p, size);
  }
  sendSection->getAtlas()->orderPatches();
  sendSection->allocate();
  sendSection->constructCommunication();
//   // 5) Restrict from the send section
//   const send_section_type::topology_type::sheaf_type& patches = sendSizer->getAtlas()->getTopology()->getPatches();

//   for(send_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
//     const Obj<send_section_type::sieve_type::baseSequence>& base = p_iter->second->base();
//     int                                                     rank = p_iter->first;

//     for(send_section_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
//       int size = sendSection->getAtlas()->size(rank);
//       sendSizer->update(rank, *b_iter, &size);
//     }
//   }
//   sendSizer->view("Send Sizer", MPI_COMM_SELF);
//   // 5) Complete the sizer
//   sendSizer->startCommunication();
//   sendSizer->endCommunication();
  // 6) Fill up send section
  for(int p = 1; p < mesh->commSize(); p++) {
    int size = sendSection->getAtlas()->getFiberDimension(p, p);
    send_section_type::value_type *points = new send_section_type::value_type[size];
    int offset = 0;

    for(int e = 0; e < numElements; e++) {
      if (assignment[e] == p) {
        points[offset++] = e;
      }
    }
    sendSection->update(p, p, points);
  }
  sendSection->view("Send Section", MPI_COMM_SELF);
  // 7) Complete the section
  sendSection->startCommunication();
  sendSection->endCommunication();
  // 8) Create point overlap
  sendOverlap->clear();
  for(int e = 0; e < numElements; e++) {
    if (assignment[e] != mesh->commRank()) {
      sendOverlap->addCone(e, assignment[e], e);
    }
  }
  sendOverlap->view(std::cout, "Send overlap");
  PetscFunctionReturn(0);
}

PetscErrorCode ReceiveDistribution(const Obj<ALE::Mesh>& mesh, Options *options)
{
  typedef ALE::New::ConstantSection<recv_section_type::topology_type, recv_section_type::index_type, recv_section_type::value_type> constant_section;
  Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(mesh->comm(), options->debug);
  Obj<recv_section_type> recvSizer   = new recv_section_type(mesh->comm(), recv_section_type::RECEIVE, options->debug);
  Obj<recv_section_type> recvSection = new recv_section_type(mesh->comm(), recv_section_type::RECEIVE, options->debug);
  Obj<constant_section>  constantSizer = new constant_section(mesh->comm(), 1, options->debug);

  PetscFunctionBegin;
  // 1) Form partition point overlap a priori
  //      The arrow is from rank 0 with partition point 0
  recvOverlap->addCone(0, mesh->commRank(), 0);
  recvOverlap->view(std::cout, "Receive overlap");
  ALE::Test::Completion::setupReceive(recvOverlap, constantSizer, recvSizer);
  ALE::Test::Completion::completeReceive(recvOverlap, recvSizer);
//   // 3) Form receive overlap section sizer
//   Obj<dsieve_type> recvSieve = new dsieve_type();
//   const Obj<recv_overlap_type::supportSequence>& points = recvOverlap->support(0);

//   for(recv_overlap_type::supportSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
//     recvSieve->addPoint(p_iter.color());
//   }
//   recvSizer->getAtlas()->getTopology()->setPatch(0, recvSieve);
//   recvSizer->getAtlas()->getTopology()->stratify();
//   recvSizer->construct(1);
//   recvSizer->getAtlas()->orderPatches();
//   recvSizer->allocate();
//   recvSizer->constructCommunication();
//   // 4) Complete the sizer
//   recvSizer->startCommunication();
//   recvSizer->endCommunication();
//   recvSizer->view("Receive Sizer", MPI_COMM_SELF);
  // 5) Update to the receive section
  recvSection->getAtlas()->setTopology(recvSizer->getAtlas()->getTopology());
  const recv_section_type::topology_type::sheaf_type& patches = recvSizer->getAtlas()->getTopology()->getPatches();

  for(recv_section_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
    const Obj<recv_section_type::sieve_type::baseSequence>& base = p_iter->second->base();
    int                                                     rank = p_iter->first;

    for(recv_section_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
      recvSection->getAtlas()->setFiberDimension(rank, *b_iter, *(recvSizer->restrict(rank, *b_iter)));
    }
  }
  recvSection->getAtlas()->orderPatches();
  recvSection->allocate();
  recvSection->constructCommunication();
  // 6) Complete the section
  recvSection->startCommunication();
  recvSection->endCommunication();
  recvSection->view("Receive Section", MPI_COMM_SELF);
  // 7) Unpack the section into the overlap
  recvOverlap->clear();
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
