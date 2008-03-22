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
      // This create a hat of three points above each base point, adjacent points are shared
      //   There are baseSize base points
      //   There are baseSize+2 unshared cap points
      //   There are baseSize-1 shared cap points
      template<typename ISifter>
      static void createHatISifter(MPI_Comm comm, ISifter& sifter, const int baseSize = 10, const int debug = 0) {
        for(int b = 0; b < baseSize; b++) {
          sifter.setConeSize(b, 3);
          if (b == 0) {
            sifter.setSupportSize(baseSize+1+b*2-1, 1);
          } else {
            sifter.setSupportSize(baseSize+1+b*2-1, 2);
          }
          sifter.setSupportSize(baseSize+1+b*2+0, 1);
          if (b == baseSize-1) {
            sifter.setSupportSize(baseSize+1+b*2+1, 1);
          }
        }
        sifter.allocate();
        for(int b = 0; b < baseSize; b++) {
          typename ISifter::source_type cone[3];
          typename ISifter::source_type support[2];

          cone[0] = baseSize+1+b*2-1;
          cone[1] = baseSize+1+b*2+0;
          cone[2] = baseSize+1+b*2+1;
          sifter.setCone(cone, b);
          support[0] = b;
          support[1] = b-1;
          sifter.setSupport(cone[0], support);
          sifter.setSupport(cone[1], support);
          if (b == baseSize-1) {
            sifter.setSupport(cone[2], support);
          }
        }
      };
    };
  };
};

#endif
