#include <Sieve.hh>
#include <src/dm/mesh/meshpcice.h>
#include "sifterTest.hh"

namespace ALE {
  namespace Test {
    template <typename Sieve_>
    class SieveBuilder {
    public:
      typedef Sieve_                                       sieve_type;
      typedef std::vector<typename sieve_type::point_type> PointArray;
    public:
#if 0
      // For a hex, there are 2d faces
      static void buildHexFaces(int dim, std::map<int, int*> *curSimplex, PointArray *boundary, point_type& simplex) {
        PointArray *faces = NULL;

        if (debug > 1) {std::cout << "  Building hex faces for boundary of " << simplex << " (size " << boundary->size() << "), dim " << dim << std::endl;}
        if (dim > 3) {
          throw ALE::Exception("Cannot do hexes of dimension greater than three");
        } else if (dim > 2) {
          int nodes[24] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                           1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7};
          faces = new PointArray();

          for(int b = 0; b < 6; b++) {
            PointArray faceBoundary = PointArray();
            point_type face;

            for(int c = 0; c < 4; c++) {
              faceBoundary.push_back((*boundary)[nodes[b*4+c]]);
            }
            if (debug > 1) {std::cout << "    boundary point " << (*boundary)[b] << std::endl;}
            this->buildHexFaces(dim-1, curSimplex, &faceBoundary, face);
            faces->push_back(face);
          }
        } else if (dim > 1) {
          int boundarySize = (int) boundary->size();
          faces = new PointArray();

          for(int b = 0; b < boundarySize; b++) {
            PointArray faceBoundary = PointArray();
            point_type face;

            for(int c = 0; c < 2; c++) {
              faceBoundary.push_back((*boundary)[(b+c)%boundarySize]);
            }
            if (debug > 1) {std::cout << "    boundary point " << (*boundary)[b] << std::endl;}
            this->buildHexFaces(dim-1, curSimplex, &faceBoundary, face);
            faces->push_back(face);
          }
        } else {
          if (debug > 1) {std::cout << "  Just set faces to boundary in 1d" << std::endl;}
          faces = boundary;
        }
        if (debug > 1) {
          for(PointArray::iterator f_itor = faces->begin(); f_itor != faces->end(); ++f_itor) {
            std::cout << "  face point " << *f_itor << std::endl;
          }
        }
        // We always create the toplevel, so we could short circuit somehow
        // Should not have to loop here since the meet of just 2 boundary elements is an element
        PointArray::iterator f_itor = faces->begin();
        point_type           start = *f_itor;
        f_itor++;
        point_type           next = *f_itor;
        Obj<sieve_type::supportSet> preElement = this->topology->nJoin(start, next, 1);

        if (preElement->size() > 0) {
          simplex = *preElement->begin();
          if (debug > 1) {std::cout << "  Found old simplex " << simplex << std::endl;}
        } else {
          int color = 0;

          simplex = point_type(0, (*(*curSimplex)[dim])++);
          for(PointArray::iterator f_itor = faces->begin(); f_itor != faces->end(); ++f_itor) {
            this->topology->addArrow(*f_itor, simplex, color++);
          }
          if (debug > 1) {std::cout << "  Added simplex " << simplex << " dim " << dim << std::endl;}
        }
        if (dim > 1) {
          delete faces;
        }
      };
#endif
      static void buildFaces(Obj<sieve_type> sieve, int dim, std::map<int, int*>& curElement, std::map<int,PointArray>& bdVertices, std::map<int,PointArray>& faces, typename sieve_type::point_type& cell) {
        int debug = sieve->debug;

        if (debug > 1) {std::cout << "  Building faces for boundary of " << cell << " (size " << bdVertices[dim].size() << "), dim " << dim << std::endl;}
        faces[dim].clear();
        if (dim > 1) {
          // Use the cone construction
          for(typename PointArray::iterator b_itor = bdVertices[dim].begin(); b_itor != bdVertices[dim].end(); ++b_itor) {
            typename sieve_type::point_type face;

            bdVertices[dim-1].clear();
            for(typename PointArray::iterator i_itor = bdVertices[dim].begin(); i_itor != bdVertices[dim].end(); ++i_itor) {
              if (i_itor != b_itor) {
                bdVertices[dim-1].push_back(*i_itor);
              }
            }
            if (debug > 1) {std::cout << "    boundary point " << *b_itor << std::endl;}
            buildFaces(sieve, dim-1, curElement, bdVertices, faces, face);
            if (debug > 1) {std::cout << "    added face " << face << std::endl;}
            faces[dim].push_back(face);
          }
        } else {
          if (debug > 1) {std::cout << "  Just set faces to boundary in 1d" << std::endl;}
          faces[dim].insert(faces[dim].end(), bdVertices[dim].begin(), bdVertices[dim].end());
        }
        if (debug > 1) {
          for(typename PointArray::iterator f_itor = faces[dim].begin(); f_itor != faces[dim].end(); ++f_itor) {
            std::cout << "  face point " << *f_itor << std::endl;
          }
        }
        // We always create the toplevel, so we could short circuit somehow
        // Should not have to loop here since the meet of just 2 boundary elements is an element
        typename PointArray::iterator   f_itor = faces[dim].begin();
        typename sieve_type::point_type start  = *f_itor;
        typename sieve_type::point_type next   = *(++f_itor);
        Obj<typename sieve_type::supportSet> preElement = sieve->nJoin(start, next, 1);

        if (preElement->size() > 0) {
          cell = *preElement->begin();
          if (debug > 1) {std::cout << "  Found old cell " << cell << std::endl;}
        } else {
          int color = 0;

          cell = typename sieve_type::point_type(0, (*curElement[dim])++);
          for(typename PointArray::iterator f_itor = faces[dim].begin(); f_itor != faces[dim].end(); ++f_itor) {
            sieve->addArrow(*f_itor, cell, color++);
          }
          if (debug > 1) {std::cout << "  Added cell " << cell << " dim " << dim << std::endl;}
        }
      };

      #undef __FUNCT__
      #define __FUNCT__ "buildTopology"
      // Build a topology from a connectivity description
      //   (0, 0)        ... (0, numCells-1):  dim-dimensional cells
      //   (0, numCells) ... (0, numVertices): vertices
      // The other cells are numbered as they are requested
      static void buildTopology(Obj<sieve_type> sieve, int dim, int numCells, int cells[], int numVertices, bool interpolate = true, int corners = -1) {
        int debug = sieve->debug;

        ALE_LOG_EVENT_BEGIN;
        if (sieve->commRank() != 0) {
          ALE_LOG_EVENT_END;
          return;
        }
        // Create a map from dimension to the current element number for that dimension
        std::map<int,int*>       curElement;
        std::map<int,PointArray> bdVertices;
        std::map<int,PointArray> faces;
        int                      curCell    = 0;
        int                      curVertex  = numCells;
        int                      newElement = numCells+numVertices;

        if (corners < 0) corners = dim+1;
        curElement[0]   = &curVertex;
        curElement[dim] = &curCell;
        for(int d = 1; d < dim; d++) {
          curElement[d] = &newElement;
        }
        for(int c = 0; c < numCells; c++) {
          typename sieve_type::point_type cell(0, c);

          // Build the cell
          if (interpolate) {
            bdVertices[dim].clear();
            for(int b = 0; b < corners; b++) {
              typename sieve_type::point_type vertex(0, cells[c*corners+b]+numCells);

              if (debug > 1) {std::cout << "Adding boundary vertex " << vertex << std::endl;}
              bdVertices[dim].push_back(vertex);
            }
            if (debug) {std::cout << "cell " << cell << " num boundary vertices " << bdVertices[dim].size() << std::endl;}

            if (corners != dim+1) {
              //buildHexFaces(sieve, dim, curElement, bdVertices, faces, cell);
            } else {
              buildFaces(sieve, dim, curElement, bdVertices, faces, cell);
            }
          } else {
            for(int b = 0; b < corners; b++) {
              sieve->addArrow(typename sieve_type::point_type(0, cells[c*corners+b]+numCells), cell, b);
            }
          }
        }
        ALE_LOG_EVENT_END;
      };
      static Obj<sieve_type> readSieve(MPI_Comm comm, const int dim, std::string basename, bool useZeroBase, const int debug = 0) {
        Obj<sieve_type> sieve = new sieve_type(comm, debug);
        int    *cells;
        double *coordinates;
        int     numCells = 0, numVertices = 0, numCorners = dim+1;

        ALE::PCICE::Builder::readConnectivity(comm, basename+".lcon", numCorners, useZeroBase, numCells, &cells);
        ALE::PCICE::Builder::readCoordinates(comm, basename+".nodes", dim, numVertices, &coordinates);
        ALE::Test::SieveBuilder<sieve_type>::buildTopology(sieve, dim, numCells, cells, numVertices);
        sieve->stratify();
        return sieve;
      };
    };
  };
};
