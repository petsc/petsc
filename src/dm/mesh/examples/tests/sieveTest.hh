#include <Sieve.hh>

namespace ALE {
  namespace Test {
    template<typename Topology_>
    class TopologyBuilder {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::sieve_type sieve_type;
    public:
      static Obj<topology_type> readTopology(MPI_Comm comm, const int dim, const std::string& basename, const bool useZeroBase = true, const bool interpolate = true, const int debug = 0, const bool stratify = true) {
        Obj<topology_type> topology = new topology_type(comm, debug);
        Obj<sieve_type>    sieve    = new sieve_type(comm, debug);
        int    *cells;
        double *coordinates;
        int     numCells = 0, numVertices = 0, numCorners = dim+1;

        ALE::PCICE::Builder::readConnectivity(comm, basename+".lcon", numCorners, useZeroBase, numCells, &cells);
        ALE::PCICE::Builder::readCoordinates(comm, basename+".nodes", dim, numVertices, &coordinates);
        ALE::New::SieveBuilder<sieve_type>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate, numCorners);
        sieve->stratify();
        topology->setPatch(0, sieve);
        if (stratify) {
          topology->stratify();
        }
        return topology;
      };
    };
  };
};
