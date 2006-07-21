#include <Sieve.hh>
#include <src/dm/mesh/meshpcice.h>
#include "sectionTest.hh"

namespace ALE {
  namespace Test {
    class MeshBuilder {
      typedef section_type::atlas_type  atlas_type;
      typedef atlas_type::topology_type topology_type;
      typedef section_type::sieve_type  sieve_type;
    public:
      static void buildCoordinates(Obj<section_type> coords, const int embedDim, const double coordinates[]) {
        const section_type::patch_type patch = 0;
        const Obj<topology_type::label_sequence>& vertices = coords->getAtlas()->getTopology()->depthStratum(patch, 0);
        const int numCells = coords->getAtlas()->getTopology()->heightStratum(patch, 0)->size();

        coords->getAtlas()->setFiberDimensionByDepth(patch, 0, embedDim);
        coords->getAtlas()->orderPatches();
        coords->allocate();
        for(topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          coords->update(patch, *v_iter, &(coordinates[((*v_iter).index - numCells)*embedDim]));
        }
      };
      static Obj<section_type> readMesh(MPI_Comm comm, const int dim, std::string basename, bool useZeroBase = true, bool interpolate = true, const int debug = 0) {
        Obj<sieve_type>   sieve  = new sieve_type(comm, debug);
        Obj<section_type> coords = new section_type(comm, debug);
        int    *cells;
        double *coordinates;
        int     numCells = 0, numVertices = 0, numCorners = dim+1;

        ALE::PCICE::Builder::readConnectivity(comm, basename+".lcon", numCorners, useZeroBase, numCells, &cells);
        ALE::PCICE::Builder::readCoordinates(comm, basename+".nodes", dim, numVertices, &coordinates);
        ALE::Test::SieveBuilder<sieve_type>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate);
        sieve->stratify();
        sieve->view("Mesh");
        coords->getAtlas()->getTopology()->setPatch(0, sieve);
        coords->getAtlas()->getTopology()->stratify();
        ALE::Test::MeshBuilder::buildCoordinates(coords, dim, coordinates);
        return coords;
      };
    };
  };
};
