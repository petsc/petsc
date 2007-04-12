#ifndef included_ALE_SieveBuilder_hh
#define included_ALE_SieveBuilder_hh

#ifndef  included_ALE_Field_hh
#include <Field.hh>
#endif

namespace ALE {
  template<typename Bundle_>
  class SieveBuilder {
  public:
    typedef Bundle_                                      bundle_type;
    typedef typename bundle_type::sieve_type             sieve_type;
    typedef typename bundle_type::arrow_section_type     arrow_section_type;
    typedef std::vector<typename sieve_type::point_type> PointArray;
  public:
    static void buildHexFaces(Obj<sieve_type> sieve, int dim, std::map<int, int*>& curElement, std::map<int,PointArray>& bdVertices, std::map<int,PointArray>& faces, typename sieve_type::point_type& cell) {
      int debug = sieve->debug();

      if (debug > 1) {std::cout << "  Building hex faces for boundary of " << cell << " (size " << bdVertices[dim].size() << "), dim " << dim << std::endl;}
      faces[dim].clear();
      if (dim > 3) {
        throw ALE::Exception("Cannot do hexes of dimension greater than three");
      } else if (dim > 2) {
        int nodes[24] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                         1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7};

        for(int b = 0; b < 6; b++) {
          typename sieve_type::point_type face;

          bdVertices[dim-1].clear();
          for(int c = 0; c < 4; c++) {
            bdVertices[dim-1].push_back(bdVertices[dim][nodes[b*4+c]]);
          }
          if (debug > 1) {std::cout << "    boundary hex face " << b << std::endl;}
          buildHexFaces(sieve, dim-1, curElement, bdVertices, faces, face);
          if (debug > 1) {std::cout << "    added face " << face << std::endl;}
          faces[dim].push_back(face);
        }
      } else if (dim > 1) {
        int boundarySize = bdVertices[dim].size();

        for(int b = 0; b < boundarySize; b++) {
          typename sieve_type::point_type face;

          bdVertices[dim-1].clear();
          for(int c = 0; c < 2; c++) {
            bdVertices[dim-1].push_back(bdVertices[dim][(b+c)%boundarySize]);
          }
          if (debug > 1) {
            std::cout << "    boundary point " << bdVertices[dim][b] << std::endl;
            std::cout << "      boundary vertices";
            for(int c = 0; c < (int) bdVertices[dim-1].size(); c++) {
              std::cout << " " << bdVertices[dim-1][c];
            }
            std::cout << std::endl;
          }
          buildHexFaces(sieve, dim-1, curElement, bdVertices, faces, face);
          if (debug > 1) {std::cout << "    added face " << face << std::endl;}
          faces[dim].push_back(face);
        }
      } else {
        if (debug > 1) {std::cout << "  Just set faces to boundary in 1d" << std::endl;}
        faces[dim].insert(faces[dim].end(), bdVertices[dim].begin(), bdVertices[dim].end());
      }
      if (debug > 1) {
        for(typename PointArray::iterator f_iter = faces[dim].begin(); f_iter != faces[dim].end(); ++f_iter) {
          std::cout << "  face point " << *f_iter << std::endl;
        }
      }
      // We always create the toplevel, so we could short circuit somehow
      // Should not have to loop here since the meet of just 2 boundary elements is an element
      typename PointArray::iterator          f_itor = faces[dim].begin();
      const typename sieve_type::point_type& start  = *f_itor;
      const typename sieve_type::point_type& next   = *(++f_itor);
      Obj<typename sieve_type::supportSet> preElement = sieve->nJoin(start, next, 1);

      if (preElement->size() > 0) {
        cell = *preElement->begin();
        if (debug > 1) {std::cout << "  Found old cell " << cell << std::endl;}
      } else {
        int color = 0;

        cell = typename sieve_type::point_type((*curElement[dim])++);
        for(typename PointArray::iterator f_itor = faces[dim].begin(); f_itor != faces[dim].end(); ++f_itor) {
          sieve->addArrow(*f_itor, cell, color++);
        }
        if (debug > 1) {std::cout << "  Added cell " << cell << " dim " << dim << std::endl;}
      }
    };
    static void buildFaces(Obj<sieve_type> sieve, int dim, std::map<int, int*>& curElement, std::map<int,PointArray>& bdVertices, std::map<int,PointArray>& faces, typename sieve_type::point_type& cell) {
      int debug = sieve->debug();

      if (debug > 1) {
        if (cell >= 0) {
          std::cout << "  Building faces for boundary of " << cell << " (size " << bdVertices[dim].size() << "), dim " << dim << std::endl;
        } else {
          std::cout << "  Building faces for boundary of undetermined cell (size " << bdVertices[dim].size() << "), dim " << dim << std::endl;
        }
      }
      faces[dim].clear();
      if (dim > 1) {
        // Use the cone construction
        for(typename PointArray::iterator b_itor = bdVertices[dim].begin(); b_itor != bdVertices[dim].end(); ++b_itor) {
          typename sieve_type::point_type face   = -1;

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
        for(typename PointArray::iterator f_iter = faces[dim].begin(); f_iter != faces[dim].end(); ++f_iter) {
          std::cout << "  face point " << *f_iter << std::endl;
        }
      }
      // We always create the toplevel, so we could short circuit somehow
      // Should not have to loop here since the meet of just 2 boundary elements is an element
      typename PointArray::iterator          f_itor = faces[dim].begin();
      const typename sieve_type::point_type& start  = *f_itor;
      const typename sieve_type::point_type& next   = *(++f_itor);
      Obj<typename sieve_type::supportSet> preElement = sieve->nJoin(start, next, 1);

      if (preElement->size() > 0) {
        cell = *preElement->begin();
        if (debug > 1) {std::cout << "  Found old cell " << cell << std::endl;}
      } else {
        int color = 0;

        cell = typename sieve_type::point_type((*curElement[dim])++);
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
    static void buildTopology(Obj<sieve_type> sieve, int dim, int numCells, int cells[], int numVertices, bool interpolate = true, int corners = -1, int firstVertex = -1, Obj<arrow_section_type> orientation = NULL) {
      int debug = sieve->debug();

      ALE_LOG_EVENT_BEGIN;
      if (sieve->commRank() != 0) {
        ALE_LOG_EVENT_END;
        return;
      }
      if (firstVertex < 0) firstVertex = numCells;
      // Create a map from dimension to the current element number for that dimension
      std::map<int,int*>       curElement;
      std::map<int,PointArray> bdVertices;
      std::map<int,PointArray> faces;
      int                      curCell    = 0;
      int                      curVertex  = firstVertex;
      int                      newElement = firstVertex+numVertices;

      if (corners < 0) corners = dim+1;
      curElement[0]   = &curVertex;
      curElement[dim] = &curCell;
      for(int d = 1; d < dim; d++) {
        curElement[d] = &newElement;
      }
      for(int c = 0; c < numCells; c++) {
        typename sieve_type::point_type cell(c);

        // Build the cell
        if (interpolate) {
          bdVertices[dim].clear();
          for(int b = 0; b < corners; b++) {
            // This ordering produces the same vertex order as the uninterpolated mesh
            //typename sieve_type::point_type vertex(cells[c*corners+(b+corners-1)%corners]+firstVertex);
            typename sieve_type::point_type vertex(cells[c*corners+b]+firstVertex);

            if (debug > 1) {std::cout << "Adding boundary vertex " << vertex << std::endl;}
            bdVertices[dim].push_back(vertex);
          }
          if (debug) {std::cout << "cell " << cell << " num boundary vertices " << bdVertices[dim].size() << std::endl;}

          if (corners != dim+1) {
            buildHexFaces(sieve, dim, curElement, bdVertices, faces, cell);
          } else {
            buildFaces(sieve, dim, curElement, bdVertices, faces, cell);
          }
          if ((dim == 2) && (!orientation.isNull())) {
            if (debug > 1) {std::cout << "Orienting cell " << cell << std::endl;}
            const Obj<typename sieve_type::traits::coneSequence>&     cone = sieve->cone(cell);
            const typename sieve_type::traits::coneSequence::iterator end  = cone->end();

            for(typename sieve_type::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
              if (debug > 1) {std::cout << "  edge " << *p_iter << std::endl;}
              const Obj<typename sieve_type::traits::coneSequence>& vertices = sieve->cone(*p_iter);
              typename sieve_type::traits::coneSequence::iterator   vertex   = vertices->begin();
              MinimalArrow<typename sieve_type::point_type,typename sieve_type::point_type> arrow(*p_iter, cell);
              int                                                                           indA, indB, value;

              orientation->addPoint(arrow);
              for(indA = 0; indA < corners; indA++) {if (*vertex == cells[c*corners+indA] + numCells) break;}
              if (debug > 1) {std::cout << "    vertexA " << *vertex << " indA " << indA <<std::endl;}
              ++vertex;
              for(indB = 0; indB < corners; indB++) {if (*vertex == cells[c*corners+indB] + numCells) break;}
              if (debug > 1) {std::cout << "    vertexB " << *vertex << " indB " << indB <<std::endl;}
              if ((indA == corners) || (indB == corners) || (indA == indB)) {throw ALE::Exception("Invalid edge endpoints");}
              if ((indB - indA == 1) || (indA - indB == 2)) {
                value =  1;
              } else {
                value = -1;
              }
              if (debug > 1) {std::cout << "  value " << value <<std::endl;}
              orientation->updatePoint(arrow, &value);
            }
          }
        } else {
          for(int b = 0; b < corners; b++) {
            sieve->addArrow(typename sieve_type::point_type(cells[c*corners+b]+firstVertex), cell, b);
          }
          if (debug) {
            if (debug > 1) {
              for(int b = 0; b < corners; b++) {
                std::cout << "  Adding vertex " << typename sieve_type::point_type(cells[c*corners+b]+firstVertex) << std::endl;
              }
            }
            std::cout << "Adding cell " << cell << " dim " << dim << std::endl;
          }
        }
      }
      ALE_LOG_EVENT_END;
    };
    static void buildCoordinates(const Obj<Bundle_>& bundle, const int embedDim, const double coords[]) {
      const Obj<typename Bundle_::real_section_type>& coordinates = bundle->getRealSection("coordinates");
      const Obj<typename Bundle_::label_sequence>&    vertices    = bundle->depthStratum(0);
      const int numCells = bundle->heightStratum(0)->size();

      coordinates->setFiberDimension(vertices, embedDim);
      bundle->allocate(coordinates);
      for(typename Bundle_::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        coordinates->updatePoint(*v_iter, &(coords[(*v_iter - numCells)*embedDim]));
      }
    };
  };
}

#endif
