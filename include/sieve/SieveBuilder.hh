#ifndef included_ALE_SieveBuilder_hh
#define included_ALE_SieveBuilder_hh

#ifndef  included_ALE_ALE_hh
#include <ALE.hh>
#endif

namespace ALE {
  template<typename Bundle_>
  class SieveBuilder {
  public:
    typedef Bundle_                                      bundle_type;
    typedef typename bundle_type::sieve_type             sieve_type;
    typedef typename bundle_type::arrow_section_type     arrow_section_type;
    typedef std::vector<typename sieve_type::point_type> PointArray;
    typedef std::pair<typename sieve_type::point_type, int> oPoint_type;
    typedef std::vector<oPoint_type>                        oPointArray;
  public:
    static void buildHexFaces(Obj<sieve_type> sieve, Obj<arrow_section_type> orientation, int dim, std::map<int, int*>& curElement, std::map<int,PointArray>& bdVertices, std::map<int,oPointArray>& faces, typename sieve_type::point_type& cell, int& cellOrientation) {
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
          int o = 1;

          bdVertices[dim-1].clear();
          for(int c = 0; c < 4; c++) {
            bdVertices[dim-1].push_back(bdVertices[dim][nodes[b*4+c]]);
          }
          if (debug > 1) {std::cout << "    boundary hex face " << b << std::endl;}
          buildHexFaces(sieve, orientation, dim-1, curElement, bdVertices, faces, face, o);
          if (debug > 1) {std::cout << "    added face " << face << std::endl;}
          faces[dim].push_back(oPoint_type(face, o));
        }
      } else if (dim > 1) {
        int boundarySize = bdVertices[dim].size();

        for(int b = 0; b < boundarySize; b++) {
          typename sieve_type::point_type face;
          int o = 1;

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
          buildHexFaces(sieve, orientation, dim-1, curElement, bdVertices, faces, face, o);
          if (debug > 1) {std::cout << "    added face " << face << std::endl;}
          faces[dim].push_back(oPoint_type(face, o));
        }
      } else {
        if (debug > 1) {std::cout << "  Just set faces to boundary in 1d" << std::endl;}
        typename PointArray::iterator bd_iter = bdVertices[dim].begin();
        faces[dim].push_back(oPoint_type(*bd_iter, 0));++bd_iter;
        faces[dim].push_back(oPoint_type(*bd_iter, 0));
        //faces[dim].insert(faces[dim].end(), bdVertices[dim].begin(), bdVertices[dim].end());
      }
      if (debug > 1) {
        for(typename oPointArray::iterator f_iter = faces[dim].begin(); f_iter != faces[dim].end(); ++f_iter) {
          std::cout << "  face point " << f_iter->first << " orientation " << f_iter->second << std::endl;
        }
      }
      // We always create the toplevel, so we could short circuit somehow
      // Should not have to loop here since the meet of just 2 boundary elements is an element
      typename oPointArray::iterator         f_itor = faces[dim].begin();
      const typename sieve_type::point_type& start  = f_itor->first;
      const typename sieve_type::point_type& next   = (++f_itor)->first;
      Obj<typename sieve_type::supportSet> preElement = sieve->nJoin(start, next, 1);

      if (preElement->size() > 0) {
        cell = *preElement->begin();

        const int size = faces[dim].size();
        const Obj<typename sieve_type::traits::coneSequence>& cone = sieve->cone(cell);
        int       wrap = size > 2 ? size-1 : 0;
        int       indA = 0, indB = 0;

        for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++indA) {
          if (start == *c_iter) break;
        }
        if (debug > 1) {std::cout << "    pointA " << start << " indA " << indA << std::endl;}
        for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++indB) {
          if (next  == *c_iter) break;
        }
        if (debug > 1) {std::cout << "    pointB " << next  << " indB " << indB << std::endl;}
        if ((indB - indA == 1) || (indA - indB == wrap)) {
          if (cellOrientation > 0) {
            cellOrientation = indA+1;
          } else {
            if (dim == 1) {
              cellOrientation = -2;
            } else {
              cellOrientation = -(indA+1);
            }
          }
        } else if ((indA - indB == 1) || (indB - indA == wrap)) {
          if (debug > 1) {std::cout << "      reversing cell orientation" << std::endl;}
          if (cellOrientation > 0) {
            cellOrientation = -(indA+1);
          } else {
            if (dim == 1) {
              cellOrientation = 1;
            } else {
              cellOrientation = indA+1;
            }
          }
        } else {
          throw ALE::Exception("Inconsistent orientation");
        }
        if (debug > 1) {std::cout << "  Found old cell " << cell << " orientation " << cellOrientation << std::endl;}
      } else {
        int color = 0;

        cell = typename sieve_type::point_type((*curElement[dim])++);
        for(typename oPointArray::iterator f_iter = faces[dim].begin(); f_iter != faces[dim].end(); ++f_iter) {
          MinimalArrow<typename sieve_type::point_type,typename sieve_type::point_type> arrow(f_iter->first, cell);

          sieve->addArrow(f_iter->first, cell, color++);
          if (f_iter->second) {
            orientation->addPoint(arrow);
            orientation->updatePoint(arrow, &(f_iter->second));
            if (debug > 1) {std::cout << "    Orienting arrow (" << f_iter->first << ", " << cell << ") to " << f_iter->second << std::endl;}
          }
        }
        if (cellOrientation > 0) {
          cellOrientation = 1;
        } else {
          cellOrientation = -(dim+1);
        }
        if (debug > 1) {std::cout << "  Added cell " << cell << " dim " << dim << std::endl;}
      }
    };
    static void buildFaces(Obj<sieve_type> sieve, Obj<arrow_section_type> orientation, int dim, std::map<int, int*>& curElement, std::map<int,PointArray>& bdVertices, std::map<int,oPointArray>& faces, typename sieve_type::point_type& cell, int& cellOrientation) {
      int debug = sieve->debug();

      if (debug > 1) {
        if (cell >= 0) {
          std::cout << "  Building faces for boundary of " << cell << " (size " << bdVertices[dim].size() << "), dim " << dim << std::endl;
        } else {
          std::cout << "  Building faces for boundary of undetermined cell (size " << bdVertices[dim].size() << "), dim " << dim << std::endl;
        }
      }
      if (dim == 0) return;
      faces[dim].clear();
      if (dim > 1) {
        int b = 0;
        // Use the cone construction
        for(typename PointArray::iterator b_itor = bdVertices[dim].begin(); b_itor != bdVertices[dim].end(); ++b_itor, ++b) {
          typename sieve_type::point_type face = -1;
          int                             o    = b%2 ? -bdVertices[dim].size() : 1;

          bdVertices[dim-1].clear();
          for(typename PointArray::iterator i_itor = bdVertices[dim].begin(); i_itor != bdVertices[dim].end(); ++i_itor) {
            if (i_itor != b_itor) {
              bdVertices[dim-1].push_back(*i_itor);
            }
          }
          if (debug > 1) {std::cout << "    boundary point " << *b_itor << std::endl;}
          buildFaces(sieve, orientation, dim-1, curElement, bdVertices, faces, face, o);
          if (debug > 1) {std::cout << "    added face " << face << std::endl;}
          faces[dim].push_back(oPoint_type(face, o));
        }
      } else {
        if (debug > 1) {std::cout << "  Just set faces to boundary in 1d" << std::endl;}
        typename PointArray::iterator bd_iter = bdVertices[dim].begin();
        faces[dim].push_back(oPoint_type(*bd_iter, 0));++bd_iter;
        faces[dim].push_back(oPoint_type(*bd_iter, 0));
        //faces[dim].insert(faces[dim].end(), bdVertices[dim].begin(), bdVertices[dim].end());
      }
      if (debug > 1) {
        for(typename oPointArray::iterator f_iter = faces[dim].begin(); f_iter != faces[dim].end(); ++f_iter) {
          std::cout << "  face point " << f_iter->first << " orientation " << f_iter->second << std::endl;
        }
      }
      // We always create the toplevel, so we could short circuit somehow
      // Should not have to loop here since the meet of just 2 boundary elements is an element
      typename oPointArray::iterator         f_itor = faces[dim].begin();
      const typename sieve_type::point_type& start  = f_itor->first;
      const typename sieve_type::point_type& next   = (++f_itor)->first;
      Obj<typename sieve_type::supportSet> preElement = sieve->nJoin(start, next, 1);

      if (preElement->size() > 0) {
        cell = *preElement->begin();

        const int size = faces[dim].size();
        const Obj<typename sieve_type::traits::coneSequence>& cone = sieve->cone(cell);
        int       wrap = size > 2 ? size-1 : 0;
        int       indA = 0, indB = 0;

        for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++indA) {
          if (start == *c_iter) break;
        }
        if (debug > 1) {std::cout << "    pointA " << start << " indA " << indA << std::endl;}
        for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++indB) {
          if (next  == *c_iter) break;
        }
        if (debug > 1) {std::cout << "    pointB " << next  << " indB " << indB << std::endl;}
        if ((indB - indA == 1) || (indA - indB == wrap)) {
          if (cellOrientation > 0) {
            cellOrientation = indA+1;
          } else {
            if (dim == 1) {
              cellOrientation = -2;
            } else {
              cellOrientation = -(indA+1);
            }
          }
        } else if ((indA - indB == 1) || (indB - indA == wrap)) {
          if (debug > 1) {std::cout << "      reversing cell orientation" << std::endl;}
          if (cellOrientation > 0) {
            cellOrientation = -(indA+1);
          } else {
            if (dim == 1) {
              cellOrientation = 1;
            } else {
              cellOrientation = indA+1;
            }
          }
        } else {
          throw ALE::Exception("Inconsistent orientation");
        }
        if (debug > 1) {std::cout << "  Found old cell " << cell << " orientation " << cellOrientation << std::endl;}
      } else {
        int color = 0;

        cell = typename sieve_type::point_type((*curElement[dim])++);
        for(typename oPointArray::iterator f_iter = faces[dim].begin(); f_iter != faces[dim].end(); ++f_iter) {
          MinimalArrow<typename sieve_type::point_type,typename sieve_type::point_type> arrow(f_iter->first, cell);

          sieve->addArrow(f_iter->first, cell, color++);
          if (f_iter->second) {
            orientation->addPoint(arrow);
            orientation->updatePoint(arrow, &(f_iter->second));
            if (debug > 1) {std::cout << "    Orienting arrow (" << f_iter->first << ", " << cell << ") to " << f_iter->second << std::endl;}
          }
        }
        if (cellOrientation > 0) {
          cellOrientation = 1;
        } else {
          cellOrientation = -(dim+1);
        }
        if (debug > 1) {std::cout << "  Added cell " << cell << " dim " << dim << " orientation " << cellOrientation << std::endl;}
      }
    };
#if 0
    static void orientTriangle(const typename sieve_type::point_type cell, const int vertices[], const Obj<sieve_type>& sieve, const Obj<arrow_section_type>& orientation, int firstVertex[]) {
      const Obj<typename sieve_type::traits::coneSequence>&     cone = sieve->cone(cell);
      const typename sieve_type::traits::coneSequence::iterator end  = cone->end();
      int debug   = sieve->debug();
      int corners = 3;
      int e       = 0;

      if (debug > 1) {std::cout << "Orienting triangle " << cell << std::endl;}
      for(typename sieve_type::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter, ++e) {
        if (debug > 1) {std::cout << "  edge " << *p_iter << std::endl;}
        const Obj<typename sieve_type::traits::coneSequence>& endpoints = sieve->cone(*p_iter);
        typename sieve_type::traits::coneSequence::iterator   vertex    = endpoints->begin();
        MinimalArrow<typename sieve_type::point_type,typename sieve_type::point_type> arrow(*p_iter, cell);
        int                                                                           indA, indB, value;

        orientation->addPoint(arrow);
        for(indA = 0; indA < corners; indA++) {if (*vertex == vertices[indA]) break;}
        if (debug > 1) {std::cout << "    vertexA " << *vertex << " indA " << indA <<std::endl;}
        firstVertex[e] = *vertex;
        ++vertex;
        for(indB = 0; indB < corners; indB++) {if (*vertex == vertices[indB]) break;}
        if (debug > 1) {std::cout << "    vertexB " << *vertex << " indB " << indB <<std::endl;}
        if ((indA == corners) || (indB == corners) || (indA == indB)) {throw ALE::Exception("Invalid edge endpoints");}
        if ((indB - indA == 1) || (indA - indB == 2)) {
          value =  1;
        } else {
          value = -1;
          firstVertex[e] = *vertex;
        }
        if (debug > 1) {std::cout << "  value " << value <<std::endl;}
        orientation->updatePoint(arrow, &value);
      }
    };
#endif
#undef __FUNCT__
#define __FUNCT__ "buildTopology"
    // Build a topology from a connectivity description
    //   (0, 0)        ... (0, numCells-1):  dim-dimensional cells
    //   (0, numCells) ... (0, numVertices): vertices
    // The other cells are numbered as they are requested
    static void buildTopology(Obj<sieve_type> sieve, int dim, int numCells, int cells[], int numVertices, bool interpolate = true, int corners = -1, int firstVertex = -1, Obj<arrow_section_type> orientation = NULL, int firstCell = 0) {
      ALE_LOG_EVENT_BEGIN;
      if (sieve->commRank() != 0) {
        ALE_LOG_EVENT_END;
        return;
      }
      buildTopology_private(sieve, dim, numCells, cells, numVertices, interpolate, corners, firstVertex, orientation, firstCell);
      ALE_LOG_EVENT_END;
    };
    static void buildTopology_private(Obj<sieve_type> sieve, int dim, int numCells, int cells[], int numVertices, bool interpolate = true, int corners = -1, int firstVertex = -1, Obj<arrow_section_type> orientation = NULL, int firstCell = 0) {
      int debug = sieve->debug();

      if (firstVertex < 0) firstVertex = numCells;
      // Create a map from dimension to the current element number for that dimension
      std::map<int,int*>       curElement;
      std::map<int,PointArray> bdVertices;
      std::map<int,PointArray> faces;
      std::map<int,oPointArray> oFaces;
      int                      curCell    = firstCell;
      int                      curVertex  = firstVertex;
      int                      newElement = firstVertex > firstCell ? firstVertex + numVertices : firstCell + numCells;
      int                      o          = 1;

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
            buildHexFaces(sieve, orientation, dim, curElement, bdVertices, oFaces, cell, o);
          } else {
            buildFaces(sieve, orientation, dim, curElement, bdVertices, oFaces, cell, o);
          }
#if 0
          if ((dim == 2) && (!orientation.isNull())) {
            typename sieve_type::point_type eVertices[3];
            typename sieve_type::point_type fVertices[3];

            for(int v = 0; v < 3; ++v) {
              fVertices[v] = cells[c*corners+v]+numCells;
            }
            orientTriangle(cell, fVertices, sieve, orientation, eVertices);
          } else if ((dim == 3) && (!orientation.isNull())) {
            // The order of vertices in cells[] orients the cell itself
            if (debug > 1) {std::cout << "Orienting tetrahedron " << cell << std::endl;}
            const Obj<typename sieve_type::traits::coneSequence>&     cone = sieve->cone(cell);
            const typename sieve_type::traits::coneSequence::iterator end  = cone->end();
            int f = 0;

            for(typename sieve_type::traits::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter, ++f) {
              if (debug > 1) {std::cout << "  face " << *p_iter << std::endl;}
              const Obj<typename sieve_type::traits::coneSequence>&     fCone = sieve->cone(*p_iter);
              const typename sieve_type::traits::coneSequence::iterator fEnd  = fCone->end();
              typename sieve_type::point_type fVertices[3];
              typename sieve_type::point_type eVertices[3];

              // We will choose the orientation such that the normals are outward
              for(int v = 0, i = 0; v < corners; ++v) {
                if (v == f) continue;
                fVertices[i++] = cells[c*corners+v]+numCells;
              }
              if (f%2) {
                int tmp      = fVertices[0];
                fVertices[0] = fVertices[1];
                fVertices[1] = tmp;
              }
              orientTriangle(*p_iter, fVertices, sieve, orientation, eVertices);
              MinimalArrow<typename sieve_type::point_type,typename sieve_type::point_type> fArrow(*p_iter, cell);
              int                                                                           indC, indD, indE, value;

              orientation->addPoint(fArrow);
              for(indC = 0; indC < corners; indC++) {if (eVertices[0] == fVertices[indC]) break;}
              if (debug > 1) {std::cout << "    vertexC " << eVertices[0] << " indC " << indC <<std::endl;}
              for(indD = 0; indD < corners; indD++) {if (eVertices[1] == fVertices[indD]) break;}
              if (debug > 1) {std::cout << "    vertexD " << eVertices[1] << " indD " << indD <<std::endl;}
              for(indE = 0; indE < corners; indE++) {if (eVertices[2] == fVertices[indE]) break;}
              if (debug > 1) {std::cout << "    vertexE " << eVertices[2] << " indE " << indE <<std::endl;}
              if ((indC == corners) || (indD == corners) || (indE == corners) ||
                  (indC == indD) || (indD == indE) || (indE == indC)) {throw ALE::Exception("Invalid face corners");}
              if ((indD - indC == 1) || (indC - indD == 2)) {
                if (!((indE - indD == 1) || (indD - indE == 2)) ||
                    !((indC - indE == 1) || (indE - indC == 2))) {throw ALE::Exception("Invalid order");}
                value =  1;
              } else {
                value = -1;
              }
              if (debug > 1) {std::cout << "  value " << value <<std::endl;}
              orientation->updatePoint(fArrow, &value);
              orientation->view("Intermediate orientation");
            }
          }
#endif
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
            if ((numCells < 10000) || (c%1000 == 0)) {
              std::cout << "Adding cell " << cell << " dim " << dim << std::endl;
            }
          }
        }
      }
    };
    static void buildCoordinates(const Obj<Bundle_>& bundle, const int embedDim, const double coords[]) {
      const Obj<typename Bundle_::real_section_type>& coordinates = bundle->getRealSection("coordinates");
      const Obj<typename Bundle_::label_sequence>&    vertices    = bundle->depthStratum(0);
      const int numCells = bundle->heightStratum(0)->size();
      const int debug    = bundle->debug();

      bundle->setupCoordinates(coordinates);
      coordinates->setFiberDimension(vertices, embedDim);
      bundle->allocate(coordinates);
      for(typename Bundle_::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        coordinates->updatePoint(*v_iter, &(coords[(*v_iter - numCells)*embedDim]));
        if (debug) {
          if ((numCells < 10000) || ((*v_iter)%1000 == 0)) {
            std::cout << "Adding coordinates for vertex " << *v_iter << std::endl;
          }
        }
      }
    };
    #undef __FUNCT__
    #define __FUNCT__ "buildTopologyMultiple"
    // Build a topology from a connectivity description
    //   (0, 0)        ... (0, numCells-1):  dim-dimensional cells
    //   (0, numCells) ... (0, numVertices): vertices
    // The other cells are numbered as they are requested
    static void buildTopologyMultiple(Obj<sieve_type> sieve, int dim, int numCells, int cells[], int numVertices, bool interpolate = true, int corners = -1, int firstVertex = -1, Obj<arrow_section_type> orientation = NULL) {
      int debug = sieve->debug();

      ALE_LOG_EVENT_BEGIN;
      int *cellOffset = new int[sieve->commSize()+1];
      cellOffset[0] = 0;
      MPI_Allgather(&numCells, 1, MPI_INT, &cellOffset[1], 1, MPI_INT, sieve->comm());
      for(int p = 1; p <= sieve->commSize(); ++p) cellOffset[p] += cellOffset[p-1];
      int *vertexOffset = new int[sieve->commSize()+1];
      vertexOffset[0] = 0;
      MPI_Allgather(&numVertices, 1, MPI_INT, &vertexOffset[1], 1, MPI_INT, sieve->comm());
      for(int p = 1; p <= sieve->commSize(); ++p) vertexOffset[p] += vertexOffset[p-1];
      if (firstVertex < 0) firstVertex = cellOffset[sieve->commSize()] + vertexOffset[sieve->commRank()];
      // Estimate the number of intermediates as (V+C)*(dim-1)
      //   Should include a check for running over the namespace
      // Create a map from dimension to the current element number for that dimension
      std::map<int,int*>       curElement;
      std::map<int,PointArray> bdVertices;
      std::map<int,PointArray> faces;
      std::map<int,oPointArray> oFaces;
      int                      curCell    = cellOffset[sieve->commRank()];
      int                      curVertex  = firstVertex;
      int                      newElement = firstVertex+vertexOffset[sieve->commSize()] + (cellOffset[sieve->commRank()] + vertexOffset[sieve->commRank()])*(dim-1);
      int                      o          = 1;

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
            buildHexFaces(sieve, orientation, dim, curElement, bdVertices, oFaces, cell, o);
          } else {
            buildFaces(sieve, orientation, dim, curElement, bdVertices, oFaces, cell, o);
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
            if ((numCells < 10000) || (c%1000 == 0)) {
              std::cout << "Adding cell " << cell << " dim " << dim << std::endl;
            }
          }
        }
      }

      if (newElement >= firstVertex+vertexOffset[sieve->commSize()] + (cellOffset[sieve->commRank()+1] + vertexOffset[sieve->commRank()+1])*(dim-1)) {
	throw ALE::Exception("Namespace violation during intermediate element construction");
      }
      delete [] cellOffset;
      delete [] vertexOffset;
      ALE_LOG_EVENT_END;
    };
    static void buildCoordinatesMultiple(const Obj<Bundle_>& bundle, const int embedDim, const double coords[]) {
      const Obj<typename Bundle_::real_section_type>& coordinates = bundle->getRealSection("coordinates");
      const Obj<typename Bundle_::label_sequence>&    vertices    = bundle->depthStratum(0);
      const int numCells = bundle->heightStratum(0)->size(), numVertices = vertices->size();
      const int debug    = bundle->debug();
      int       numGlobalCells, offset;

      MPI_Allreduce((void *) &numCells, &numGlobalCells, 1, MPI_INT, MPI_SUM, bundle->comm());
      MPI_Scan((void *) &numVertices, &offset, 1, MPI_INT, MPI_SUM, bundle->comm());
      offset += numGlobalCells - numVertices;
      coordinates->setFiberDimension(vertices, embedDim);
      bundle->allocate(coordinates);
      for(typename Bundle_::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        coordinates->updatePoint(*v_iter, &(coords[(*v_iter - offset)*embedDim]));
        if (debug) {
          if ((numCells < 10000) || ((*v_iter)%1000 == 0)) {
            std::cout << "Adding coordinates for vertex " << *v_iter << std::endl;
          }
        }
      }
    };
  };
}
#endif
