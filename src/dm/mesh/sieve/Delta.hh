#ifndef included_ALE_Delta_hh
#define included_ALE_Delta_hh

#ifndef  included_ALE_CoSieve_hh
#include <CoSieve.hh>
#endif

//
// This file contains classes and methods implementing  the Overlap and Fusion algorithms for Sections.
//
namespace ALE {
  // Overlap operates on sections; if Sifters and Sieves are presented as Sections, will operate on those too;
  // for that to work Sections must support an iterator-based access to values.
  // The idea is to look at the points overlapping in the corresponding Atlases.
  template <typename SectionA_, typename SectionB_, typename Pullback_>
  class Overlap {
  public:
    typedef SectionA_ section_a_type;
    typedef SectionB_ section_b_type;
    typedef Pullback_ pullback_type;
  protected:
    //
    static void computeOverlap(const section_a_type& secA, const section_b_type& secB, const pullback_type& pullback,
                               send_section_type& sendSec, recv_section_type& recvSec) {
    }
  }; // class Overlap

    
} // namespace ALE

#endif
