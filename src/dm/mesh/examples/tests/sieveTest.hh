#include <Sieve.hh>
#include <src/dm/mesh/meshpcice.h>
#include "sifterTest.hh"

namespace ALE {
  namespace Test {
    template <typename Sieve_>
    class SieveBuilder {
    public:
      typedef Sieve_ sieve_type;
    public:
      static Obj<sieve_type> readSieve(MPI_Comm comm, const int dim, std::string basename, bool useZeroBase, const int debug = 0) {
        Obj<sieve_type> sieve = new sieve_type(comm, debug);
        int    *cells;
        double *coordinates;
        int     numCells = 0, numVertices = 0, numCorners = dim+1;

        ALE::PCICE::Builder::readConnectivity(comm, basename+".lcon", numCorners, useZeroBase, numCells, &cells);
        ALE::PCICE::Builder::readCoordinates(comm, basename+".nodes", dim, numVertices, &coordinates);
        ALE::New::SieveBuilder<sieve_type>::buildTopology(sieve, dim, numCells, cells, numVertices);
        sieve->stratify();
        return sieve;
      };
    };
  };
};
