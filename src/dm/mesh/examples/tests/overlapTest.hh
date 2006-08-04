#ifndef included_ALE_overlapTest_hh
#define included_ALE_overlapTest_hh

#include <CoSieve.hh>
#include "sectionTest.hh"

namespace ALE {
  namespace Test {
    class OverlapTest {
    public:
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
    public:
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
      static void constructDoublet(const Obj<topology_type>& topology) {
        Obj<sieve_type>     sieve = new sieve_type(topology->comm(), topology->debug());
        Obj<std::set<int> > cone  = new std::set<int>();

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
      };
      // Send Overlap:
      //  - Cap contains local points
      //  - Base contains ranks
      //  - Arrows decorated with remote points
      // Receive Overlap:
      //  - Base contains local points
      //  - Cap contains ranks
      //  - Arrows decorated with remote points
      static void constructDoubletOverlap(const Obj<send_overlap_type>& sendOverlap, const Obj<recv_overlap_type>& recvOverlap) {
        if (sendOverlap->commRank() == 0) {
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
      };
    };

    class SupportSizer {
    protected:
      Obj<OverlapTest::sieve_type> _sieve;
    public:
      SupportSizer(const Obj<OverlapTest::sieve_type>& sieve) {this->_sieve = sieve;};
    public:
      int size(const OverlapTest::sieve_type::point_type& point) const {
        return this->_sieve->support(point)->size();
      };
    };
  }
}

#endif
