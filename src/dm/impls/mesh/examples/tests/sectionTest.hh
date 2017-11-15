#ifndef included_ALE_sectionTest_hh
#define included_ALE_sectionTest_hh

#include <Field.hh>
#include "sieveTest.hh"

namespace ALE {
  namespace Test {
    typedef ALE::Sieve<int, int, int>                sieve_type;
    typedef ALE::ConstantSection<sieve_type, double> constant_section_type;
    typedef ALE::UniformSection<sieve_type, int, 2>  uniform_section_type;
    typedef ALE::Section<sieve_type, double>         section_type;

    class SectionTest {
    public:
    };
  };
};

#endif
