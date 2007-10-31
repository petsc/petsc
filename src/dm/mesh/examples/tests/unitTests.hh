#ifndef included_ALE_unitTests_hh
#define included_ALE_unitTests_hh

#include <Sifter.hh>

namespace ALE {
  namespace Test {
    class SifterBuilder {
    public:
      template<typename SifterType>
      static Obj<SifterType> createHatSifter(MPI_Comm comm, const int baseSize = 10, const int debug = 0) {
        Obj<SifterType> sifter = new SifterType(comm, debug);
        Obj<std::vector<typename SifterType::source_type> > cone = new std::vector<typename SifterType::source_type>();

        for(int b = 0; b < baseSize; b++) {
          cone->clear();
          cone->push_back(b*2-1);
          cone->push_back(b*2+0);
          cone->push_back(b*2+1);
          sifter->addCone(cone, b);
        }
        return sifter;
      };
    };
  };
};

#endif
