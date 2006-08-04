#include <Sieve.hh>
#include <src/dm/mesh/meshpcice.h>
#include "overlapTest.hh"

namespace ALE {
  namespace Test {
    class Completion {
    public:
      typedef ALE::Test::OverlapTest::dsieve_type       dsieve_type;
      typedef ALE::Test::OverlapTest::send_overlap_type send_overlap_type;
      typedef ALE::Test::OverlapTest::send_section_type send_section_type;
      typedef ALE::Test::OverlapTest::recv_overlap_type recv_overlap_type;
      typedef ALE::Test::OverlapTest::recv_section_type recv_section_type;
    public:
      template<typename Sizer>
      static void completeSend(const Obj<send_overlap_type>& sendOverlap, const Obj<Sizer>& sendSizer, const Obj<send_section_type>& sendSection) {
        // Create section
        const Obj<send_overlap_type::traits::baseSequence> ranks = sendOverlap->base();

        for(send_overlap_type::traits::baseSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          Obj<dsieve_type> sendSieve = new dsieve_type(sendOverlap->cone(*r_iter));
          sendSection->getAtlas()->getTopology()->setPatch(*r_iter, sendSieve);
        }
        sendSection->getAtlas()->getTopology()->stratify();
        sendSection->construct(sendSizer);
        sendSection->getAtlas()->orderPatches();
        sendSection->allocate();
        sendSection->constructCommunication();
      };
      template<typename Sizer>
      static void completeReceive(const Obj<recv_overlap_type>& recvOverlap, const Obj<Sizer>& recvSizer, const Obj<recv_section_type>& recvSection) {
        // Create section
        const Obj<recv_overlap_type::traits::capSequence> ranks = recvOverlap->cap();

        for(recv_overlap_type::traits::capSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
          Obj<dsieve_type> recvSieve = new dsieve_type();
          const Obj<recv_overlap_type::supportSequence>& points = recvOverlap->support(0);

          // Want to replace this loop with a slice through color
          for(recv_overlap_type::supportSequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
            recvSieve->addPoint(p_iter.color());
          }
          recvSection->getAtlas()->getTopology()->setPatch(0, recvSieve);
        }
        recvSection->getAtlas()->getTopology()->stratify();
        recvSection->construct(recvSizer);
        recvSection->getAtlas()->orderPatches();
        recvSection->allocate();
        recvSection->constructCommunication();
      };
    };
  };
};
