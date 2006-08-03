#ifndef included_ALE_sifterTest_hh
#define included_ALE_sifterTest_hh

#include <Sifter.hh>

namespace ALE {
  namespace Test {
    typedef ALE::Point                     Point;
    typedef ALE::Sifter<Point, Point, int> sifter_type;

    class SifterTest {
    public:
      static Obj<sifter_type> createHatSifter(MPI_Comm comm, const int baseSize = 10, const int debug = 0) {
        Obj<sifter_type>   sifter = new sifter_type(comm, debug);
        Obj<std::vector<Point> > cone = new std::vector<Point>();

        for(int b = 0; b < baseSize; b++) {
          cone->clear();
          cone->push_back(Point(1, b*2-1));
          cone->push_back(Point(1, b*2+0));
          cone->push_back(Point(1, b*2+1));
          sifter->addCone(cone, Point(0, b));
        }
        return sifter;
      };
    };
  };
};

#endif
