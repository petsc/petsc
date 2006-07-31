static char help[] = "Global Ordering Tests.\n\n";

#include <petsc.h>
#include "sectionTest.hh"

using ALE::Obj;
typedef int                                                                        point_type;
typedef ALE::Sieve<point_type, int, int>                                           sieve_type;
typedef ALE::New::Topology<int, sieve_type>                                        topology_type;
typedef ALE::New::DiscreteSieve<point_type>                                        dsieve_type;
typedef ALE::New::Topology<int, dsieve_type>                                       overlap_topology_type;
typedef ALE::New::Atlas<overlap_topology_type, ALE::Point>                         overlap_atlas_type;
typedef ALE::Sifter<int,point_type,point_type>                                     send_overlap_type;
typedef ALE::New::OverlapValues<send_overlap_type, overlap_atlas_type, point_type> send_section_type;
typedef ALE::Sifter<point_type,int,point_type>                                     recv_overlap_type;
typedef ALE::New::OverlapValues<recv_overlap_type, overlap_atlas_type, point_type> recv_section_type;

typedef struct {
  int debug; // The debugging level
} Options;

#undef __FUNCT__
#define __FUNCT__ "DoubletTest"
// The doublet is
//
//       2 | 2
// p0     /|\      p1
//     5 / | \ 4
//      /  |  \     _
//     /   |   \    _
//  0 /   4|5   \ 1
//    \    |    /
//     \ 6 | 6 /
//    3 \  |  / 3
//       \ | /
//        \|/
//       1 | 0
//
// This test has
//  - A mesh overlapping itself
//  - Single points overlapping single points
PetscErrorCode DoubletTest(const Obj<send_section_type>& sendSection, const Obj<recv_section_type>& recvSection, Options *options)
{
  Obj<sieve_type>        sieve       = new sieve_type(sendSection->comm(), options->debug);
  Obj<topology_type>     topology    = new topology_type(sendSection->comm(), options->debug);
  Obj<send_overlap_type> sendOverlap = new send_overlap_type(sendSection->comm(), options->debug);
  Obj<recv_overlap_type> recvOverlap = new recv_overlap_type(recvSection->comm(), options->debug);
  Obj<dsieve_type>       dSieve      = new dsieve_type();
  Obj<std::set<int> >    cone        = new std::set<int>();
  int                    debug       = options->debug;
  std::map<sieve_type::point_type, int> globalOrder;

  PetscFunctionBegin;
  if (sieve->commSize() != 2) SETERRQ(PETSC_ERR_SUP, "DoubletTest can only be run with 2 processes");
  // Make the sieve and topology
  cone->insert(3);cone->insert(4);cone->insert(5);
  sieve->addCone(cone, 6);cone->clear();
  cone->insert(0);cone->insert(1);
  sieve->addCone(cone, 3);cone->clear();
  cone->insert(1);cone->insert(2);
  sieve->addCone(cone, 4);cone->clear();
  cone->insert(2);cone->insert(0);
  sieve->addCone(cone, 5);cone->clear();
  topology->setPatch(0, sieve);
  topology->stratify();
  // Make overlap
  if (sieve->commRank() == 0) {
    // Local point 1 is overlapped by remote point 0 from proc 1
    sendOverlap->addArrow(1, 1, 0);
    recvOverlap->addArrow(1, 1, 0);
    // Local point 2 is overlapped by remote point 2 from proc 1
    sendOverlap->addArrow(2, 1, 2);
    recvOverlap->addArrow(1, 2, 2);
  } else {
    // Local point 0 is overlapped by remote point 1 from proc 0
    sendOverlap->addArrow(0, 0, 1);
    recvOverlap->addArrow(0, 0, 1);
    // Local point 2 is overlapped by remote point 2 from proc 0
    sendOverlap->addArrow(2, 0, 2);
    recvOverlap->addArrow(0, 2, 2);
  }
  // Make discrete sieve
  if (sieve->commRank() == 0) {
    cone->insert(0);cone->insert(2);
    dSieve->addPoints(cone);
    sendSection->getAtlas()->getTopology()->setPatch(1, dSieve);
    recvSection->getAtlas()->getTopology()->setPatch(1, dSieve);
  } else {
    cone->insert(1);cone->insert(2);
    dSieve->addPoints(cone);
    sendSection->getAtlas()->getTopology()->setPatch(0, dSieve);
    recvSection->getAtlas()->getTopology()->setPatch(0, dSieve);
  }
  sendSection->getAtlas()->getTopology()->stratify();
  recvSection->getAtlas()->getTopology()->stratify();
  // Setup sections
  sendSection->construct(sendOverlap, 1);
  recvSection->construct(recvOverlap, 1);
  sendSection->getAtlas()->orderPatches();
  recvSection->getAtlas()->orderPatches();
  sendSection->allocate();
  recvSection->allocate();
  sendSection->constructCommunication();
  recvSection->constructCommunication();
  // Create local piece of the global order
  const Obj<topology_type::label_sequence>& vertices = topology->depthStratum(0, 0);
  int localNumber = 0;
  int offset;

  for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    if (sendOverlap->capContains(*v_iter)) {
      const Obj<send_overlap_type::traits::supportSequence>& sendPatches = sendOverlap->support(*v_iter);
      int minRank = sendOverlap->commSize();

      for(send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
        if (*p_iter < minRank) minRank = *p_iter;
      }
      if (minRank < sendOverlap->commRank()) {
        globalOrder[*v_iter] = -1;
      } else {
        globalOrder[*v_iter] = localNumber++;
      }
    } else {
      globalOrder[*v_iter] = localNumber++;
    }
  }
  MPI_Scan(&localNumber, &offset, 1, MPI_INT, MPI_SUM, sendOverlap->comm());
  offset -= localNumber;
  for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    if (globalOrder[*v_iter] >= 0) {
      globalOrder[*v_iter] += offset;
    }
  }
  // Fill up sections
  Obj<send_overlap_type::traits::capSequence> sendPoints = sendOverlap->cap();

  if (debug) {std::cout << "Send information" << std::endl;}
  for(send_overlap_type::traits::capSequence::iterator s_iter = sendPoints->begin(); s_iter != sendPoints->end(); ++s_iter) {
    const Obj<send_overlap_type::traits::supportSequence>& sendPatches = sendOverlap->support(*s_iter);
    
    if (debug) {std::cout << "[" << sendOverlap->commRank() << "]Point " << *s_iter << std::endl;}
    for(send_overlap_type::traits::supportSequence::iterator p_iter = sendPatches->begin(); p_iter != sendPatches->end(); ++p_iter) {
      sendSection->update(*p_iter, *s_iter, &(globalOrder[*s_iter]));

      if (debug) {
        std::cout << "[" << sendOverlap->commRank() << "]  Receiver " << *p_iter << " Matching Point " << p_iter.color() << std::endl;
        const send_section_type::value_type *values = sendSection->restrict(*p_iter, *s_iter);
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
      const recv_section_type::value_type *values = recvSection->restrict(*p_iter, *r_iter);

      if (debug) {
        std::cout << "[" << recvOverlap->commRank() << "]  Sender " << *p_iter << " Matching Point " << p_iter.color() << std::endl;
        for(int i = 0; i < recvSection->getAtlas()->size(*p_iter, *r_iter); i++) {
          std::cout << "[" << recvOverlap->commRank() << "]    " << values[i] << std::endl;
        }
      }
      if (values[0] >= 0) {
        if (globalOrder[*r_iter] >= 0) SETERRQ1(PETSC_ERR_PLIB, "Multiple indices for point %d", *r_iter);
        globalOrder[*r_iter] = values[0];
      }
    }
  }
  if (debug) {
    const Obj<topology_type::label_sequence>& vertices = topology->depthStratum(0, 0);

    std::cout << "[" << recvOverlap->commRank() << "]  Global Order" << std::endl;
    for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
      std::cout << "[" << recvOverlap->commRank() << "] " << *v_iter << " --> " << globalOrder[*v_iter] << std::endl;
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
    Obj<send_section_type> sendSection = new send_section_type(comm, send_section_type::SEND, options.debug);
    Obj<recv_section_type> recvSection = new recv_section_type(comm, recv_section_type::RECEIVE, sendSection->getTag(), options.debug);

    ierr = DoubletTest(sendSection, recvSection, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
