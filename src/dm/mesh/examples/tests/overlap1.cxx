static char help[] = "Overlap Tests.\n\n";

#include <petscsys.h>
#include "overlapTest.hh"

using ALE::Obj;
typedef ALE::Test::OverlapTest::sieve_type        sieve_type;
typedef ALE::Test::OverlapTest::topology_type     topology_type;
typedef ALE::Test::OverlapTest::dsieve_type       dsieve_type;
typedef ALE::Test::OverlapTest::send_overlap_type send_overlap_type;
typedef ALE::Test::OverlapTest::send_section_type send_section_type;
typedef ALE::Test::OverlapTest::recv_overlap_type recv_overlap_type;
typedef ALE::Test::OverlapTest::recv_section_type recv_section_type;
typedef ALE::Test::SupportSizer                   SupportSizer;

typedef struct {
  int debug; // The debugging level
} Options;

#undef __FUNCT__
#define __FUNCT__ "DoubletTest"
// This test has
//  - A mesh overlapping itself
//  - Single points overlapping single points
PetscErrorCode DoubletTest(const Obj<send_section_type>& sendSection, const Obj<recv_section_type>& recvSection, Options *options)
{
  Obj<topology_type>     topology    = new topology_type(sendSection->comm(), options->debug);
  Obj<send_overlap_type> sendOverlap = new send_overlap_type(sendSection->comm(), options->debug);
  Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(recvSection->comm(), options->debug);
  Obj<dsieve_type>       dSieve      = new dsieve_type();
  Obj<std::set<int> >    cone        = new std::set<int>();
  int                    debug       = options->debug;

  PetscFunctionBegin;
  if (sendSection->commSize() != 2) SETERRQ(PETSC_ERR_SUP, "DoubletTest can only be run with 2 processes");
  // Make the sieve
  ALE::Test::OverlapTest::constructDoublet(topology);
  Obj<sieve_type>   sieve = topology->getPatch(0);
  Obj<SupportSizer> sizer = new ALE::Test::SupportSizer(sieve);
  // Make overlap
  ALE::Test::OverlapTest::constructDoubletOverlap(sendOverlap, recvOverlap);
  // Make discrete sieve
  if (sieve->commRank() == 0) {
    cone->insert(0);cone->insert(2);
    dSieve->addPoints(cone);
    sendSection->getTopology()->setPatch(1, dSieve);
    recvSection->getTopology()->setPatch(1, dSieve);
  } else {
    cone->insert(1);cone->insert(2);
    dSieve->addPoints(cone);
    sendSection->getTopology()->setPatch(0, dSieve);
    recvSection->getTopology()->setPatch(0, dSieve);
  }
  sendSection->getTopology()->stratify();
  recvSection->getTopology()->stratify();
  // Setup sections
  sendSection->construct(sizer);
  recvSection->construct(sizer);
  sendSection->allocate();
  recvSection->allocate();
  sendSection->constructCommunication(send_section_type::SEND);
  recvSection->constructCommunication(recv_section_type::RECEIVE);
  // Fill up sections
  Obj<send_overlap_type::traits::capSequence> sendPoints = sendOverlap->cap();

  if (debug) {std::cout << "Send information" << std::endl;}
  for(send_overlap_type::traits::capSequence::iterator s_iter = sendPoints->begin(); s_iter != sendPoints->end(); ++s_iter) {
    const Obj<send_overlap_type::traits::supportSequence>& sendPatches = sendOverlap->support(*s_iter);
    
    if (debug) {std::cout << "[" << sendOverlap->commRank() << "]Point " << *s_iter << std::endl;}
    for(send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
      sendSection->update(*p_iter, *s_iter, sieve->support(*s_iter));

      if (debug) {
        std::cout << "[" << sendOverlap->commRank() << "]  Receiver " << *p_iter << " Matching Point " << p_iter.color() << std::endl;
        const send_section_type::value_type *values = sendSection->restrictPoint(*p_iter, *s_iter);
        for(int i = 0; i < sendSection->getAtlas()->size(*p_iter, *s_iter); i++) {
          std::cout << "[" << recvOverlap->commRank() << "]    " << values[i] << std::endl;
        }
      }
    }
  }
  if (debug) {sendSection->view("Send section");}
  // Communicate
  sendSection->startCommunication();
  recvSection->startCommunication();
  sendSection->endCommunication();
  recvSection->endCommunication();
  // Read out sections and check answer
  if (debug) {recvSection->view("Receive section");}
  if (debug) {std::cout << "Receive information" << std::endl;}
  Obj<recv_overlap_type::traits::baseSequence> recvPoints = recvOverlap->base();

  for(recv_overlap_type::traits::baseSequence::iterator r_iter = recvPoints->begin(); r_iter != recvPoints->end(); ++r_iter) {
    const Obj<recv_overlap_type::traits::coneSequence>& recvPatches = recvOverlap->cone(*r_iter);
    
    if (debug) {std::cout << "[" << recvOverlap->commRank() << "]Point " << *r_iter << std::endl;}
    for(recv_overlap_type::traits::coneSequence::iterator p_iter = recvPatches->begin(); p_iter != recvPatches->end(); ++p_iter) {
      const recv_section_type::value_type *values = recvSection->restrictPoint(*p_iter, *r_iter);

      if (debug) {
        std::cout << "[" << recvOverlap->commRank() << "]  Sender " << *p_iter << " Matching Point " << p_iter.color() << std::endl;
        for(int i = 0; i < recvSection->getAtlas()->size(*p_iter, *r_iter); i++) {
          std::cout << "[" << recvOverlap->commRank() << "]    " << values[i] << std::endl;
        }
      }
      if (recvOverlap->commRank() == 0) {
        if (*r_iter == 1) {
          if ((values[0] != 3) || (values[1] != 5)) SETERRQ1(PETSC_ERR_PLIB, "Invalid received value for point %d", *r_iter);
        } else {
          if ((values[0] != 4) || (values[1] != 5)) SETERRQ1(PETSC_ERR_PLIB, "Invalid received value for point %d", *r_iter);
        }
      } else {
        if (*r_iter == 0) {
          if ((values[0] != 3) || (values[1] != 4)) SETERRQ1(PETSC_ERR_PLIB, "Invalid received value for point %d", *r_iter);
        } else {
          if ((values[0] != 4) || (values[1] != 5)) SETERRQ1(PETSC_ERR_PLIB, "Invalid received value for point %d", *r_iter);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PartitionTest"
PetscErrorCode PartitionTest(const Obj<send_section_type>& sendSection, const Obj<recv_section_type>& recvSection, Options *options)
{
  // Construct the proto-overlap
  //   We say that partition points overlap those on proc 0 (never appear in Sieve)
  // Construct the send and receive sections
  //   The send size is the number of points to send from proc 0
  //   The recv size is the number coming into proc >0
  // Fill up with points
  // Communicate
  // Insert points into overlap
  // Construct the send and receive sections
  //   The send size is the cone sizes to send from proc 0
  //   The recv size is the incoming cone sizes into proc >0
  // Fill up with cones
  // Communicate
  // Insert points into sieve

  // Operator allocation or application:
  // -----------------------------------
  // Construct the proto-overlap
  //   This is just the traditional overlap of boundary points
  // Construct the send and receive sections
  //   The send size and recv size are the sizes of the remote cone of influence
  // Fill up with points in the cone of influence
  // Communicate
  // Insert influence points into overlap
  // Construct the send and receive sections
  //   The send size and recv size are the value sizes over each point
  // Fill up with values
  // Communicate
  // Combine values into application
  PetscFunctionBegin;
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
    ierr = PetscOptionsInt("-debug", "The debugging level", "overlap1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
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
    Obj<send_section_type> sendSection = new send_section_type(comm, options.debug);
    Obj<recv_section_type> recvSection = new recv_section_type(comm, sendSection->getTag(), options.debug);

    ierr = DoubletTest(sendSection, recvSection, &options);CHKERRQ(ierr);
    ierr = PartitionTest(sendSection, recvSection, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
