#include <CoSieve.hh>
#include "sieveTest.hh"

namespace ALE {
  namespace Test {
    typedef ALE::Sieve<Point, int, int>           sieve_type;
    typedef ALE::New::Topology<int, sieve_type>   topology_type;
    typedef ALE::New::Atlas<topology_type, Point> atlas_type;
    typedef ALE::New::Section<atlas_type, double> section_type;

    class SectionTest {
    public:
    };
  };
};
