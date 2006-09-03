#ifndef included_ALE_sectionTest_hh
#define included_ALE_sectionTest_hh

#include <CoSieve.hh>
#include "sieveTest.hh"

namespace ALE {
  namespace Test {
    typedef ALE::Sieve<int, int, int>                           sieve_type;
    typedef ALE::New::Topology<int, sieve_type>                 topology_type;
    typedef ALE::New::NewConstantSection<topology_type, double> constant_section_type;
    typedef ALE::New::UniformSection<topology_type, int, 2>     uniform_section_type;
    typedef ALE::New::Section<topology_type, double>            section_type;

    class SectionTest {
    public:
    };
  };
};

#endif
