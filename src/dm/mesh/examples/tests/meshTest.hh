#include <Sieve.hh>
#include <src/dm/mesh/meshpcice.h>
#include "sectionTest.hh"

namespace ALE {
  namespace Test {
    class MeshProcessor {
    public:
      static std::string printMatrix(const int rows, const int cols, const section_type::value_type matrix[], const int rank = -1)
      {
        ostringstream output;
        ostringstream rankStr;

        if (rank >= 0) {
          rankStr << "[" << rank << "]";
        }
        output << rankStr.str() << "J = " << std::endl;
        for(int r = 0; r < rows; r++) {
          if (r == 0) {
            output << rankStr.str() << " /";
          } else if (r == rows-1) {
            output << rankStr.str() << " \\";
          } else {
            output << rankStr.str() << " |";
          }
          for(int c = 0; c < cols; c++) {
            output << " " << matrix[r*cols+c];
          }
          if (r == 0) {
            output << " \\" << std::endl;
          } else if (r == rows-1) {
            output << " /" << std::endl;
          } else {
            output << " |" << std::endl;
          }
        }
        return output.str();
      }
      static std::string printElement(const section_type::point_type& e, const int dim, const section_type::value_type coords[], const int rank = -1) {
        ostringstream output;
        ostringstream r;

        if (rank >= 0) {
          r << "[" << rank << "]";
        }
        output << r.str() << "Element " << e << std::endl;
        output << r.str() << "Coordinates: " << e << std::endl << r.str() << "  ";
        for(int f = 0; f <= dim; f++) {
          output << " (";
          for(int d = 0; d < dim; d++) {
            if (d > 0) output << ", ";
            output << coords[f*dim+d];
          }
          output << ")";
        }
        output << std::endl;
        return output.str();
      };
      static void computeElementGeometry(const Obj<section_type>& coordinates, int dim, const sieve_type::point_type& e, section_type::value_type v0[], section_type::value_type J[], section_type::value_type invJ[], section_type::value_type& detJ)
      {
        const section_type::patch_type  patch  = 0;
        const section_type::value_type *coords = coordinates->restrict(patch, e);
        section_type::value_type        invDet;

        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        if (dim == 1) {
          detJ = J[0];
        } else if (dim == 2) {
          detJ = J[0]*J[3] - J[1]*J[2];
        } else if (dim == 3) {
          detJ = J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
            J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
            J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        }
        invDet = 1.0/detJ;
        if (dim == 2) {
          invJ[0] =  invDet*J[3];
          invJ[1] = -invDet*J[1];
          invJ[2] = -invDet*J[2];
          invJ[3] =  invDet*J[0];
        } else if (dim == 3) {
          // FIX: This may be wrong
          invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
          invJ[0*3+1] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
          invJ[0*3+2] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
          invJ[1*3+0] = invDet*(J[0*3+1]*J[2*3+2] - J[0*3+2]*J[2*3+1]);
          invJ[1*3+1] = invDet*(J[0*3+2]*J[2*3+0] - J[0*3+0]*J[2*3+2]);
          invJ[1*3+2] = invDet*(J[0*3+0]*J[2*3+1] - J[0*3+1]*J[2*3+0]);
          invJ[2*3+0] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
          invJ[2*3+1] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
          invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
        }
      };
    };
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
        ALE::New::SieveBuilder<sieve_type>::buildTopology(sieve, dim, numCells, cells, numVertices, interpolate);
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
