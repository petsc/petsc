#ifndef included_ALE_Selection_hh
#define included_ALE_Selection_hh

#ifndef  included_ALE_SieveAlgorithms_hh
#include <SieveAlgorithms.hh>
#endif

#ifndef  included_ALE_SieveBuilder_hh
#include <SieveBuilder.hh>
#endif

namespace ALE {
  template<typename Mesh_>
  class Selection {
  public:
    typedef Mesh_                                mesh_type;
    typedef typename mesh_type::sieve_type       sieve_type;
    typedef typename mesh_type::point_type       point_type;
    typedef typename mesh_type::int_section_type int_section_type;
    typedef std::set<point_type>                 PointSet;
    typedef std::vector<point_type>              PointArray;
    typedef std::pair<typename sieve_type::point_type, int> oPoint_type;
    typedef std::vector<oPoint_type>                        oPointArray;
    typedef typename ALE::SieveAlg<mesh_type>    sieveAlg;
  public:
    template<typename Processor>
    static void subsets(const Obj<PointArray>& v, const int size, Processor& processor, Obj<PointArray> *out = NULL, const int min = 0) {
      if (size == 0) {
        processor(*out);
        return;
      }
      if (min == 0) {
        out  = new Obj<PointArray>();
        *out = new PointArray();
      }
      for(int i = min; i < (int) v->size(); ++i) {
        (*out)->push_back((*v.ptr())[i]);
        subsets(v, size-1, processor, out, i+1);
        (*out)->pop_back();
      }
      if (min == 0) {delete out;}
    };
    static int numFaceVertices(const point_type& cell, const Obj<mesh_type>& mesh, const int depth = -1) {
      const int    cellDim    = mesh->getDimension();
      const int    meshDepth  = (depth < 0) ? mesh->depth() : depth;
      unsigned int numCorners = mesh->getSieve()->nCone(cell, meshDepth)->size();

      unsigned int _numFaceVertices = 0;
      switch (cellDim) {
      case 0 :
        _numFaceVertices = 0;
        break;
      case 1 :
        _numFaceVertices = 1;
        break;
      case 2:
        switch (numCorners) {
        case 3 : // triangle
          _numFaceVertices = 2; // Edge has 2 vertices
          break;
        case 4 : // quadrilateral
          _numFaceVertices = 2; // Edge has 2 vertices
          break;
        default :
          throw ALE::Exception("Invalid number of face corners");
        }
        break;
      case 3:
        switch (numCorners)	{
        case 4 : // tetradehdron
          _numFaceVertices = 3; // Face has 3 vertices
          break;
        case 8 : // hexahedron
          _numFaceVertices = 4; // Face has 4 vertices
          break;
        default :
          throw ALE::Exception("Invalid number of face corners");
        }
        break;
      default:
        throw ALE::Exception("Invalid cell dimension");
      }
      return _numFaceVertices;
    };
    // We need this method because we do not use interpolates sieves
    //   - Without interpolation, we cannot say what vertex collections are
    //     faces, and how they are oriented
    //   - Now we read off the list of face vertices IN THE ORDER IN WHICH
    //     THEY APPEAR IN THE CELL
    //   - This leads to simple algorithms for simplices and quads to check
    //     orientation since these sets are always valid faces
    //   - This is not true with hexes, so we just sort and check explicit cases
    //   - This means for hexes that we have to alter the vertex container as well
    static bool faceOrientation(const point_type& cell, const Obj<mesh_type>& mesh, const int numCorners,
                                const int indices[], const int oppositeVertex, PointArray *origVertices, PointArray *faceVertices) {
      const int cellDim   = mesh->getDimension();
      const int debug     = mesh->debug();
      bool      posOrient = false;

      // Simplices
      if (cellDim == numCorners-1) {
        posOrient = !(oppositeVertex%2);
      } else if (cellDim == 2) {
        // Quads
        if ((indices[1] > indices[0]) && (indices[1] - indices[0] == 1)) {
          posOrient = true;
        } else if ((indices[0] == 3) && (indices[1] == 0)) {
          posOrient = true;
        } else {
          if (((indices[0] > indices[1]) && (indices[0] - indices[1] == 1)) || ((indices[0] == 0) && (indices[1] == 3))) {
            posOrient = false;
          } else {
            throw ALE::Exception("Invalid quad crossedge");
          }
        }
      } else if (cellDim == 3) {
        // Hexes
        //   A hex is two oriented quads with the normal of the first
        //   pointing up at the second.
        //
        //     7---6
        //    /|  /|
        //   4---5 |
        //   | 3-|-2
        //   |/  |/
        //   0---1
        int  sortedIndices[4];
        bool found = false;

        for(int i = 0; i < 4; ++i) sortedIndices[i] = indices[i];
        std::sort(sortedIndices, sortedIndices+4);
        // Case 1: Bottom quad
        if ((sortedIndices[0] == 0) && (sortedIndices[1] == 1) && (sortedIndices[2] == 2) && (sortedIndices[3] == 3)) {
          if (debug) std::cout << "Bottom quad" << std::endl;
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 3) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 2) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 1) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 0) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          found = true;
        }
        // Case 2: Top quad
        if ((sortedIndices[0] == 4) && (sortedIndices[1] == 5) && (sortedIndices[2] == 6) && (sortedIndices[3] == 7)) {
          if (debug) std::cout << "Top quad" << std::endl;
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 5) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 6) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 7) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 4) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          found = true;
        }
        // Case 3: Front quad
        if ((sortedIndices[0] == 0) && (sortedIndices[1] == 1) && (sortedIndices[2] == 4) && (sortedIndices[3] == 5)) {
          if (debug) std::cout << "Front quad" << std::endl;
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 1) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 5) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 4) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 0) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          found = true;
        }
        // Case 4: Back quad
        if ((sortedIndices[0] == 2) && (sortedIndices[1] == 3) && (sortedIndices[2] == 6) && (sortedIndices[3] == 7)) {
          if (debug) std::cout << "Back quad" << std::endl;
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 7) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 6) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 2) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 3) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          found = true;
        }
        // Case 5: Right quad
        if ((sortedIndices[0] == 1) && (sortedIndices[1] == 2) && (sortedIndices[2] == 5) && (sortedIndices[3] == 6)) {
          if (debug) std::cout << "Right quad" << std::endl;
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 2) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 6) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 5) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 1) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          found = true;
        }
        // Case 6: Left quad
        if ((sortedIndices[0] == 0) && (sortedIndices[1] == 3) && (sortedIndices[2] == 4) && (sortedIndices[3] == 7)) {
          if (debug) std::cout << "Left quad" << std::endl;
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 4) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 7) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 3) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          for(int i = 0; i < 4; ++i) {
            if (indices[i] == 0) {
              faceVertices->push_back((*origVertices)[i]); break;
            }
          }
          found = true;
        }
        if (!found) {throw ALE::Exception("Invalid hex crossface");}
        return true;
      }
      if (!posOrient) {
        if (debug) std::cout << "  Reversing initial face orientation" << std::endl;
        faceVertices->insert(faceVertices->end(), (*origVertices).rbegin(), (*origVertices).rend());
      } else {
        if (debug) std::cout << "  Keeping initial face orientation" << std::endl;
        faceVertices->insert(faceVertices->end(), (*origVertices).begin(), (*origVertices).end());
      }
      return posOrient;
    };
    // Given a cell and a face, as a set of vertices,
    //   return the oriented face, as a set of vertices, in faceVertices
    // The orientation is such that the face normal points out of the cell
    template<typename FaceType>
    static bool getOrientedFace(const Obj<mesh_type>& mesh, const point_type& cell, FaceType face,
                                const int numCorners, int indices[], PointArray *origVertices, PointArray *faceVertices)
    {
      const Obj<typename sieve_type::traits::coneSequence>& cone = mesh->getSieve()->cone(cell);
      const int debug = mesh->debug();
      int       v     = 0;
      int       oppositeVertex;

      origVertices->clear();
      faceVertices->clear();
      for(typename sieve_type::traits::coneSequence::iterator v_iter = cone->begin(); v_iter != cone->end(); ++v_iter, ++v) {
        if (face->find(*v_iter) != face->end()) {
          if (debug) std::cout << "    vertex " << *v_iter << std::endl;
          indices[origVertices->size()] = v;
          origVertices->insert(origVertices->end(), *v_iter);
        } else {
          if (debug) std::cout << "    vertex " << *v_iter << std::endl;
          oppositeVertex = v;
        }
      }
      return faceOrientation(cell, mesh, numCorners, indices, oppositeVertex, origVertices, faceVertices);
    };
    static void insertFace(const Obj<mesh_type>& mesh, const Obj<sieve_type>& subSieve, const Obj<PointSet>& face, point_type& f,
                           const point_type& cell, const int numCorners, int indices[], PointArray *origVertices, PointArray *faceVertices)
    {
      const Obj<typename sieve_type::supportSet> preFace = subSieve->nJoin1(face);
      const int                                  debug   = subSieve->debug();

      if (preFace->size() > 1) {
        throw ALE::Exception("Invalid fault sieve: Multiple faces from vertex set");
      } else if (preFace->size() == 1) {
        // Add the other cell neighbor for this face
        subSieve->addArrow(*preFace->begin(), cell);
      } else if (preFace->size() == 0) {
        if (debug) std::cout << "  Orienting face " << f << std::endl;
        try {
          getOrientedFace(mesh, cell, face, numCorners, indices, origVertices, faceVertices);
          if (debug) std::cout << "  Adding face " << f << std::endl;
          int color = 0;
          for(typename PointArray::const_iterator f_iter = faceVertices->begin(); f_iter != faceVertices->end(); ++f_iter) {
            if (debug) std::cout << "    vertex " << *f_iter << std::endl;
            subSieve->addArrow(*f_iter, f, color++);
          }
          subSieve->addArrow(f, cell);
          f++;
        } catch (ALE::Exception e) {
          if (debug) std::cout << "  Did not add invalid face " << f << std::endl;
        }
      }
    };
  public:
    class FaceInserter {
#if 0
    public:
      typedef Mesh_                                mesh_type;
      typedef typename mesh_type::sieve_type       sieve_type;
      typedef typename mesh_type::point_type       point_type;
      typedef std::set<point_type>                 PointSet;
      typedef std::vector<point_type>              PointArray;
#endif
    protected:
      const Obj<mesh_type>  mesh;
      const Obj<sieve_type> sieve;
      const Obj<sieve_type> subSieve;
      point_type&           f;
      const point_type      cell;
      const int             numCorners;
      int                  *indices;
      PointArray           *origVertices;
      PointArray           *faceVertices;
      PointSet             *subCells;
      const int             debug;
    public:
      FaceInserter(const Obj<mesh_type>& mesh, const Obj<sieve_type>& sieve, const Obj<sieve_type>& subSieve, point_type& f, const point_type& cell, const int numCorners, int indices[], PointArray *origVertices, PointArray *faceVertices, PointSet *subCells) : mesh(mesh), sieve(sieve), subSieve(subSieve), f(f), cell(cell), numCorners(numCorners), indices(indices), origVertices(origVertices), faceVertices(faceVertices), subCells(subCells), debug(mesh->debug()) {};
      virtual ~FaceInserter() {};
    public:
      void operator()(const Obj<PointArray>& face) {
        const Obj<typename sieve_type::supportSet> sievePreFace = sieve->nJoin1(face);

        if (sievePreFace->size() == 1) {
          if (debug) std::cout << "  Contains a boundary face on the submesh" << std::endl;
          PointSet faceSet(face->begin(), face->end());
          ALE::Selection<mesh_type>::insertFace(mesh, subSieve, faceSet, f, cell, numCorners, indices, origVertices, faceVertices);
          subCells->insert(cell);
        }
      };
    };
  public:
    // This takes in a section and creates a submesh from the vertices in the section chart
    //   This is a hyperplane of one dimension lower than the mesh
    static Obj<mesh_type> submesh_uninterpolated(const Obj<mesh_type>& mesh, const Obj<int_section_type>& label, const int dimension = -1, const bool boundaryFaces = true) {
      const int              dim        = (dimension > 0) ? dimension : mesh->getDimension()-1;
      Obj<mesh_type>         submesh    = new mesh_type(mesh->comm(), dim, mesh->debug());
      Obj<sieve_type>        subSieve   = new sieve_type(mesh->comm(), mesh->debug());
      Obj<PointSet>          face       = new PointSet();
      const Obj<sieve_type>& sieve      = mesh->getSieve();
      const int              numCorners = sieve->nCone(*mesh->heightStratum(0)->begin(), mesh->depth())->size();
      const int              faceSize   = numFaceVertices(*mesh->heightStratum(0)->begin(), mesh);
      int                   *indices    = new int[faceSize];
      int                    f          = sieve->base()->size() + sieve->cap()->size();
      const int              debug      = mesh->debug();
      const int              depth      = mesh->depth();
      const int              height     = mesh->height();
      const typename int_section_type::chart_type&          chart    = label->getChart();
      const typename int_section_type::chart_type::iterator chartEnd = chart.end();
      PointSet               submeshVertices, submeshCells;
      PointArray             origVertices, faceVertices;

      for(typename int_section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chartEnd; ++c_iter) {
        //assert(!mesh->depth(*c_iter));
        submeshVertices.insert(*c_iter);
      }
      const typename PointSet::const_iterator svBegin = submeshVertices.begin();
      const typename PointSet::const_iterator svEnd   = submeshVertices.end();

      for(typename PointSet::const_iterator sv_iter = svBegin; sv_iter != svEnd; ++sv_iter) {
        const Obj<typename sieveAlg::supportArray>& cells = sieveAlg::nSupport(mesh, *sv_iter, depth);
        const typename sieveAlg::supportArray::iterator cBegin = cells->begin();
        const typename sieveAlg::supportArray::iterator cEnd   = cells->end();
    
        if (debug) std::cout << "Checking submesh vertex " << *sv_iter << std::endl;
        for(typename sieveAlg::supportArray::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
          if (debug) std::cout << "  Checking cell " << *c_iter << std::endl;
          if (submeshCells.find(*c_iter) != submeshCells.end())	continue;
          const Obj<typename sieveAlg::coneArray>& cone = sieveAlg::nCone(mesh, *c_iter, height);
          const typename sieveAlg::coneArray::iterator vBegin = cone->begin();
          const typename sieveAlg::coneArray::iterator vEnd   = cone->end();

          face->clear();
          for(typename sieveAlg::coneArray::iterator v_iter = vBegin; v_iter != vEnd; ++v_iter) {
            if (submeshVertices.find(*v_iter) != svEnd) {
              if (debug) std::cout << "    contains submesh vertex " << *v_iter << std::endl;
              face->insert(face->end(), *v_iter);
            }
          }
          if ((int) face->size() > faceSize) {
            if (!boundaryFaces) throw ALE::Exception("Invalid fault mesh: Too many vertices of an element on the fault");
            // Here we allow a set of vertices to lie completely on a boundary cell (like a corner tetrahedron)
            //   We have to take all the faces, and discard those in the interior
            FaceInserter inserter(mesh, sieve, subSieve, f, *c_iter,
                                  numCorners, indices, &origVertices, &faceVertices, &submeshCells);
            PointArray faceVec(face->begin(), face->end());

            subsets(faceVec, faceSize, inserter);
          }
          if ((int) face->size() == faceSize) {
            if (debug) std::cout << "  Contains a face on the submesh" << std::endl;
            insertFace(mesh, subSieve, face, f, *c_iter, numCorners, indices, &origVertices, &faceVertices);
            submeshCells.insert(*c_iter);
          }
        }
      }
      submesh->setSieve(subSieve);
      submesh->stratify();
      submeshCells.clear();
      if (debug) submesh->view("Submesh");
      return submesh;
    };
    // This takes in a section and creates a submesh from the vertices in the section chart
    //   This is a hyperplane of one dimension lower than the mesh
    static Obj<mesh_type> submesh_interpolated(const Obj<mesh_type>& mesh, const Obj<int_section_type>& label, const int dimension = -1, const bool boundaryFaces = true) {
      const int debug  = mesh->debug();
      const int depth  = mesh->depth();
      const int height = mesh->height();
      const typename int_section_type::chart_type&          chart        = label->getChart();
      const typename int_section_type::chart_type::iterator chartEnd     = chart.end();
      const Obj<PointSet>                                   submeshFaces = new PointSet();
      PointSet submeshVertices;

      for(typename int_section_type::chart_type::iterator c_iter = chart.begin(); c_iter != chartEnd; ++c_iter) {
        //assert(!mesh->depth(*c_iter));
        submeshVertices.insert(*c_iter);
      }
      const typename PointSet::const_iterator svBegin = submeshVertices.begin();
      const typename PointSet::const_iterator svEnd   = submeshVertices.end();

      for(typename PointSet::const_iterator sv_iter = svBegin; sv_iter != svEnd; ++sv_iter) {
        const Obj<typename sieveAlg::supportArray>& faces = sieveAlg::nSupport(mesh, *sv_iter, depth-1);
        const typename sieveAlg::supportArray::iterator fBegin = faces->begin();
        const typename sieveAlg::supportArray::iterator fEnd   = faces->end();
    
        if (debug) std::cout << "Checking submesh vertex " << *sv_iter << std::endl;
        for(typename sieveAlg::supportArray::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
          if (debug) std::cout << "  Checking face " << *f_iter << std::endl;
          if (submeshFaces->find(*f_iter) != submeshFaces->end())	continue;
          const Obj<typename sieveAlg::coneArray>& cone = sieveAlg::nCone(mesh, *f_iter, height-1);
          const typename sieveAlg::coneArray::iterator vBegin = cone->begin();
          const typename sieveAlg::coneArray::iterator vEnd   = cone->end();
          bool                                         found  = true;

          for(typename sieveAlg::coneArray::iterator v_iter = vBegin; v_iter != vEnd; ++v_iter) {
            if (submeshVertices.find(*v_iter) != svEnd) {
              if (debug) std::cout << "    contains submesh vertex " << *v_iter << std::endl;
            } else {
              found = false;
            }
          }
          if (found) {
            if (boundaryFaces) {throw ALE::Exception("Not finished: should check that it is a boundary face");}
            if (debug) std::cout << "  Is a face on the submesh" << std::endl;
            submeshFaces->insert(*f_iter);
          }
        }
      }
      return submesh(mesh, submeshFaces, mesh->getDimension()-1);
    };
  public:
    // This takes in a section and creates a submesh from the vertices in the section chart
    //   This is a hyperplane of one dimension lower than the mesh
    static Obj<mesh_type> submesh(const Obj<mesh_type>& mesh, const Obj<int_section_type>& label, const int dimension = -1) {
      const int dim   = mesh->getDimension();
      const int depth = mesh->depth();

      if (dim == depth) {
        return submesh_interpolated(mesh, label, dimension, false);
      } else if (depth == 1) {
        return submesh_uninterpolated(mesh, label, dimension);
      }
      throw ALE::Exception("Cannot handle partially interpolated meshes");
    };
    // This creates a submesh consisting of the union of the closures of the given points
    //   This mesh is the same dimension as in the input mesh
    template<typename Points>
    static Obj<mesh_type> submesh(const Obj<mesh_type>& mesh, const Obj<Points>& points, const int dim = -1) {
      const Obj<sieve_type>& sieve     = mesh->getSieve();
      Obj<mesh_type>         newMesh   = new mesh_type(mesh->comm(), dim >= 0 ? dim : mesh->getDimension(), mesh->debug());
      Obj<sieve_type>        newSieve  = new sieve_type(mesh->comm(), mesh->debug());
      Obj<PointSet>          newPoints = new PointSet();
      Obj<PointSet>          modPoints = new PointSet();
      Obj<PointSet>          tmpPoints;

      newMesh->setSieve(newSieve);
      for(typename Points::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        newPoints->insert(*p_iter);
        do {
          modPoints->clear();
          for(typename PointSet::iterator np_iter = newPoints->begin(); np_iter != newPoints->end(); ++np_iter) {
            const Obj<typename sieve_type::traits::coneSequence>&     cone = sieve->cone(*np_iter);
            const typename sieve_type::traits::coneSequence::iterator end  = cone->end();
            int c = 0;

            for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter, ++c) {
              newSieve->addArrow(*c_iter, *np_iter, c);
            }
            modPoints->insert(cone->begin(), cone->end());
          }
          tmpPoints = newPoints;
          newPoints = modPoints;
          modPoints = tmpPoints;
        } while(newPoints->size());
        newPoints->insert(*p_iter);
        do {
          modPoints->clear();
          for(typename PointSet::iterator np_iter = newPoints->begin(); np_iter != newPoints->end(); ++np_iter) {
            const Obj<typename sieve_type::traits::supportSequence>&     support = sieve->support(*np_iter);
            const typename sieve_type::traits::supportSequence::iterator end     = support->end();
            int s = 0;

            for(typename sieve_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != end; ++s_iter, ++s) {
              newSieve->addArrow(*np_iter, *s_iter, s);
            }
            modPoints->insert(support->begin(), support->end());
          }
          tmpPoints = newPoints;
          newPoints = modPoints;
          modPoints = tmpPoints;
        } while(newPoints->size());
      }
      newMesh->stratify();
      newMesh->setRealSection("coordinates", mesh->getRealSection("coordinates"));
      newMesh->setArrowSection("orientation", mesh->getArrowSection("orientation"));
      return newMesh;
    };
  protected:
    static Obj<mesh_type> boundary_uninterpolated(const Obj<mesh_type>& mesh) {
      throw ALE::Exception("Not yet implemented");
      const Obj<typename mesh_type::label_sequence>&     cells    = mesh->heightStratum(0);
      const Obj<sieve_type>&                             sieve    = mesh->getSieve();
      const typename mesh_type::label_sequence::iterator cBegin   = cells->begin();
      const typename mesh_type::label_sequence::iterator cEnd     = cells->end();
      const int                                          faceSize = numFaceVertices(*cBegin, mesh);

      for(typename mesh_type::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
        const Obj<typename sieve_type::traits::coneSequence>&     vertices = sieve->cone(*c_iter);
        const typename sieve_type::traits::coneSequence::iterator vBegin   = vertices->begin();
        const typename sieve_type::traits::coneSequence::iterator vEnd     = vertices->end();
        //PointArray cell(vertices->begin(), vertices->end());

        // For each vertex, gather 
        for(typename sieve_type::traits::coneSequence::iterator v_iter = vBegin; v_iter != vEnd; ++v_iter) {
          const Obj<typename sieve_type::traits::supportSequence>&     neighbors = sieve->support(*v_iter);
          const typename sieve_type::traits::supportSequence::iterator nBegin    = neighbors->begin();
          const typename sieve_type::traits::supportSequence::iterator nEnd      = neighbors->end();

          for(typename sieve_type::traits::supportSequence::iterator n_iter = nBegin; n_iter != nEnd; ++n_iter) {
            const Obj<typename sieve_type::coneSet>& preFace = sieve->nMeet(*c_iter, *n_iter, 1);

            if (preFace->size() == faceSize) {
            }
          }
        }
        // For each face
        // - determine if its legal
        
        // - determine if its part of a neighboring cell
        // - if not, its a boundary face
        //subsets(cell, faceSize, inserter);
      }
    };
    static void addClosure(const Obj<sieve_type>& sieveA, const Obj<sieve_type>& sieveB, const point_type& p, const int depth = 1) {
      Obj<typename sieve_type::coneSet> current = new typename sieve_type::coneSet();
      Obj<typename sieve_type::coneSet> next    = new typename sieve_type::coneSet();
      Obj<typename sieve_type::coneSet> tmp;

      current->insert(p);
      while(current->size()) {
        for(typename sieve_type::coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
          const Obj<typename sieve_type::traits::coneSequence>&     cone  = sieveA->cone(*p_iter);
          const typename sieve_type::traits::coneSequence::iterator begin = cone->begin();
          const typename sieve_type::traits::coneSequence::iterator end   = cone->end();

          for(typename sieve_type::traits::coneSequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
            sieveB->addArrow(*c_iter, *p_iter, c_iter.color());
            next->insert(*c_iter);
          }
        }
        tmp = current; current = next; next = tmp;
        next->clear();
      }
      if (!depth) {
        const Obj<typename sieve_type::traits::supportSequence>&     support = sieveA->support(p);
        const typename sieve_type::traits::supportSequence::iterator begin   = support->begin();
        const typename sieve_type::traits::supportSequence::iterator end     = support->end();
            
        for(typename sieve_type::traits::supportSequence::iterator s_iter = begin; s_iter != end; ++s_iter) {
          sieveB->addArrow(p, *s_iter, s_iter.color());
          next->insert(*s_iter);
        }
      }
    };
    static Obj<mesh_type> boundary_interpolated(const Obj<mesh_type>& mesh, const int faceHeight = 1) {
      Obj<mesh_type>                                     newMesh  = new mesh_type(mesh->comm(), mesh->getDimension(), mesh->debug());
      Obj<sieve_type>                                    newSieve = new sieve_type(mesh->comm(), mesh->debug());
      const Obj<sieve_type>&                             sieve    = mesh->getSieve();
      const Obj<typename mesh_type::label_sequence>&     faces    = mesh->heightStratum(faceHeight);
      const typename mesh_type::label_sequence::iterator fBegin   = faces->begin();
      const typename mesh_type::label_sequence::iterator fEnd     = faces->end();
      const int                                          depth    = faceHeight - mesh->depth();

      for(typename mesh_type::label_sequence::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
        const Obj<typename sieve_type::traits::supportSequence>& support = sieve->support(*f_iter);

        if (support->size() == 1) {
          addClosure(sieve, newSieve, *f_iter, depth);
        }
      }
      newMesh->setSieve(newSieve);
      newMesh->stratify();
      return newMesh;
    };
  public:
    static Obj<mesh_type> boundary(const Obj<mesh_type>& mesh) {
      const int dim   = mesh->getDimension();
      const int depth = mesh->depth();

      if (dim == depth) {
        return boundary_interpolated(mesh);
      } else if (depth == dim+1) {
        return boundary_interpolated(mesh, 2);
      } else if (depth == 1) {
        return boundary_uninterpolated(mesh);
      }
      throw ALE::Exception("Cannot handle partially interpolated meshes");
    };
  public:
    static Obj<mesh_type> interpolateMesh(const Obj<mesh_type>& mesh) {
      const Obj<sieve_type>                              sieve       = mesh->getSieve();
      const int  dim         = mesh->getDimension();
      const int  numVertices = mesh->depthStratum(0)->size();
      const Obj<typename mesh_type::label_sequence>&     cells       = mesh->heightStratum(0);
      const int  numCells    = cells->size();
      const int  corners     = sieve->cone(*cells->begin())->size();
      const int  firstVertex = numCells;
      const int  debug       = sieve->debug();
      Obj<mesh_type>                                     newMesh     = new mesh_type(mesh->comm(), dim, mesh->debug());
      Obj<sieve_type>                                    newSieve    = new sieve_type(mesh->comm(), mesh->debug());
      const Obj<typename mesh_type::arrow_section_type>& orientation = newMesh->getArrowSection("orientation");

      newMesh->setSieve(newSieve);
      // Create a map from dimension to the current element number for that dimension
      std::map<int,point_type*> curElement;
      std::map<int,PointArray>  bdVertices;
      std::map<int,PointArray>  faces;
      std::map<int,oPointArray> oFaces;
      int                       curCell    = 0;
      int                       curVertex  = firstVertex;
      int                       newElement = firstVertex+numVertices;
      int                       o;

      curElement[0]   = &curVertex;
      curElement[dim] = &curCell;
      for(int d = 1; d < dim; d++) {
        curElement[d] = &newElement;
      }
      typename mesh_type::label_sequence::iterator cBegin = cells->begin();
      typename mesh_type::label_sequence::iterator cEnd   = cells->end();

      for(typename mesh_type::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
        typename sieve_type::point_type                           cell   = *c_iter;
        const Obj<typename sieve_type::traits::coneSequence>&     cone   = sieve->cone(cell);
        const typename sieve_type::traits::coneSequence::iterator vBegin = cone->begin();
        const typename sieve_type::traits::coneSequence::iterator vEnd   = cone->end();

        // Build the cell
        bdVertices[dim].clear();
        for(typename sieve_type::traits::coneSequence::iterator v_iter = vBegin; v_iter != vEnd; ++v_iter) {
          typename sieve_type::point_type vertex(*v_iter);

          if (debug > 1) {std::cout << "Adding boundary vertex " << vertex << std::endl;}
          bdVertices[dim].push_back(vertex);
        }
        if (debug) {std::cout << "cell " << cell << " num boundary vertices " << bdVertices[dim].size() << std::endl;}

        if (corners != dim+1) {
          ALE::SieveBuilder<mesh_type>::buildHexFaces(newSieve, dim, curElement, bdVertices, faces, cell);
        } else {
          ALE::SieveBuilder<mesh_type>::buildFaces(newSieve, orientation, dim, curElement, bdVertices, oFaces, cell, o);
        }
      }
      newMesh->stratify();
      newMesh->setRealSection("coordinates", mesh->getRealSection("coordinates"));
      return newMesh;
    };
  };


}

#if 0
namespace ALE {
  class MySelection {
  public:
    typedef ALE::SieveAlg<ALE::Mesh> sieveAlg;
    typedef ALE::Selection<ALE::Mesh> selection;
    typedef ALE::Mesh::sieve_type sieve_type;
    typedef ALE::Mesh::int_section_type int_section_type;
    typedef ALE::Mesh::real_section_type real_section_type;
    typedef std::set<ALE::Mesh::point_type> PointSet;
    typedef std::vector<ALE::Mesh::point_type> PointArray;
  public:
    template<class InputPoints>
    static bool _compatibleOrientation(const Obj<Mesh>& mesh,
                                       const ALE::Mesh::point_type& p,
                                       const ALE::Mesh::point_type& q,
                                       const int numFaultCorners,
                                       const int faultFaceSize,
                                       const int faultDepth,
                                       const Obj<InputPoints>& points,
                                       int indices[],
                                       PointArray *origVertices,
                                       PointArray *faceVertices,
                                       PointArray *neighborVertices)
    {
      typedef ALE::Selection<ALE::Mesh> selection;
      const int debug = mesh->debug();
      bool compatible;

      bool eOrient = selection::getOrientedFace(mesh, p, points, numFaultCorners, indices, origVertices, faceVertices);
      bool nOrient = selection::getOrientedFace(mesh, q, points, numFaultCorners, indices, origVertices, neighborVertices);

      if (faultFaceSize > 1) {
        if (debug) {
          for(PointArray::iterator v_iter = faceVertices->begin(); v_iter != faceVertices->end(); ++v_iter) {
            std::cout << "  face vertex " << *v_iter << std::endl;
          }
          for(PointArray::iterator v_iter = neighborVertices->begin(); v_iter != neighborVertices->end(); ++v_iter) {
            std::cout << "  neighbor vertex " << *v_iter << std::endl;
          }
        }
        compatible = !(*faceVertices->begin() == *neighborVertices->begin());
      } else {
        compatible = !(nOrient == eOrient);
      }
      return compatible;
    };
    static void _replaceCell(const Obj<sieve_type>& sieve,
                             const ALE::Mesh::point_type cell,
                             std::map<int,int> *vertexRenumber,
                             const int debug)
    {
      bool       replace = false;
      PointArray newVertices;

      const Obj<sieve_type::traits::coneSequence>& cCone = sieve->cone(cell);

      for(sieve_type::traits::coneSequence::iterator v_iter = cCone->begin(); v_iter != cCone->end(); ++v_iter) {
        if (vertexRenumber->find(*v_iter) != vertexRenumber->end()) {
          if (debug) std::cout << "    vertex " << (*vertexRenumber)[*v_iter] << std::endl;
          newVertices.insert(newVertices.end(), (*vertexRenumber)[*v_iter]);
          replace = true;
        } else {
          if (debug) std::cout << "    vertex " << *v_iter << std::endl;
          newVertices.insert(newVertices.end(), *v_iter);
        } // if/else
      } // for
      if (replace) {
        if (debug) std::cout << "  Replacing cell " << cell << std::endl;
        sieve->clearCone(cell);
        int color = 0;
        for(PointArray::const_iterator v_iter = newVertices.begin(); v_iter != newVertices.end(); ++v_iter) {
          sieve->addArrow(*v_iter, cell, color++);
        } // for
      }
    };
    template<class InputPoints>
    static void _computeCensoredDepth(const Obj<ALE::Mesh>& mesh,
                                      const Obj<ALE::Mesh::label_type>& depth,
                                      const Obj<ALE::Mesh::sieve_type>& sieve,
                                      const Obj<InputPoints>& points,
                                      const ALE::Mesh::point_type& firstCohesiveCell,
                                      const Obj<std::set<ALE::Mesh::point_type> >& modifiedPoints)
    {
      modifiedPoints->clear();

      for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        if (*p_iter >= firstCohesiveCell) continue;
        // Compute the max depth of the points in the cone of p, and add 1
        int d0 = mesh->getValue(depth, *p_iter, -1);
        int d1 = mesh->getMaxValue(depth, sieve->cone(*p_iter), -1) + 1;

        if(d1 != d0) {
          mesh->setValue(depth, *p_iter, d1);
          modifiedPoints->insert(*p_iter);
        }
      }
      // FIX: We would like to avoid the copy here with support()
      if(modifiedPoints->size() > 0) {
        _computeCensoredDepth(mesh, depth, sieve, sieve->support(modifiedPoints), firstCohesiveCell, modifiedPoints);
      }
    };
    static void create(const Obj<Mesh>& mesh, Obj<Mesh> fault, const Obj<Mesh::int_section_type>& groupField) {
      static PetscEvent CreateFaultMesh_Event = 0, OrientFaultMesh_Event = 0, AddCohesivePoints_Event = 0, SplitMesh_Event = 0;

      if (!CreateFaultMesh_Event) {
        PetscLogEventRegister(&CreateFaultMesh_Event, "CreateFaultMesh", 0);
      }
      if (!OrientFaultMesh_Event) {
        PetscLogEventRegister(&OrientFaultMesh_Event, "OrientFaultMesh", 0);
      }
      if (!AddCohesivePoints_Event) {
        PetscLogEventRegister(&AddCohesivePoints_Event, "AddCohesivePoints", 0);
      }
      if (!SplitMesh_Event) {
        PetscLogEventRegister(&SplitMesh_Event, "SplitMesh", 0);
      }

      const Obj<sieve_type>& sieve = mesh->getSieve();
      const int  debug      = mesh->debug();
      int        numCorners = 0;    // The number of vertices in a mesh cell
      int        faceSize   = 0;    // The number of vertices in a mesh face
      int       *indices    = NULL; // The indices of a face vertex set in a cell
      PointArray origVertices;
      PointArray faceVertices;
      PointArray neighborVertices;
      const bool constraintCell = false;

      if (!mesh->commRank()) {
        numCorners = sieve->nCone(*mesh->heightStratum(0)->begin(), mesh->depth())->size();
        faceSize   = selection::numFaceVertices(*mesh->heightStratum(0)->begin(), mesh);
        indices    = new int[faceSize];
      }

      //int f = sieve->base()->size() + sieve->cap()->size();
      //ALE::Obj<PointSet> face = new PointSet();
  
      // Create a sieve which captures the fault
      PetscLogEventBegin(CreateFaultMesh_Event,0,0,0,0);
      fault = ALE::Selection<ALE::Mesh>::submesh(mesh, groupField);
      if (debug) {fault->view("Fault mesh");}
      PetscLogEventEnd(CreateFaultMesh_Event,0,0,0,0);
      // Orient the fault sieve
      PetscLogEventBegin(OrientFaultMesh_Event,0,0,0,0);
      const Obj<sieve_type>&                faultSieve = fault->getSieve();
      const ALE::Obj<Mesh::label_sequence>& fFaces     = fault->heightStratum(1);
      int faultDepth      = fault->depth()-1; // Depth of fault cells
      int numFaultCorners = 0; // The number of vertices in a fault cell

      if (!fault->commRank()) {
        numFaultCorners = faultSieve->nCone(*fFaces->begin(), faultDepth)->size();
        if (debug) std::cout << "  Fault corners " << numFaultCorners << std::endl;
        assert(numFaultCorners == faceSize);
      }
      PetscLogEventEnd(OrientFaultMesh_Event,0,0,0,0);

      // Add new shadow vertices and possibly Lagrange multipler vertices
      PetscLogEventBegin(AddCohesivePoints_Event,0,0,0,0);
      const ALE::Obj<Mesh::label_sequence>&   fVertices  = fault->depthStratum(0);
      const ALE::Obj<std::set<std::string> >& groupNames = mesh->getIntSections();
      Mesh::point_type newPoint = sieve->base()->size() + sieve->cap()->size();
      std::map<int,int> vertexRenumber;
  
      for(Mesh::label_sequence::iterator v_iter = fVertices->begin(); v_iter != fVertices->end(); ++v_iter, ++newPoint) {
        vertexRenumber[*v_iter] = newPoint;
        if (debug) {std::cout << "Duplicating " << *v_iter << " to " << vertexRenumber[*v_iter] << std::endl;}

        // Add shadow and constraint vertices (if they exist) to group
        // associated with fault
        groupField->addPoint(newPoint, 1);
        if (constraintCell) groupField->addPoint(newPoint+1, 1);

        // Add shadow vertices to other groups, don't add constraint
        // vertices (if they exist) because we don't want BC, etc to act
        // on constraint vertices
        for(std::set<std::string>::const_iterator name = groupNames->begin(); name != groupNames->end(); ++name) {
          const ALE::Obj<int_section_type>& group = mesh->getIntSection(*name);
          if (group->hasPoint(*v_iter)) group->addPoint(newPoint, 1);
        }
        if (constraintCell) newPoint++;
      }
      for(std::set<std::string>::const_iterator name = groupNames->begin(); name != groupNames->end(); ++name) {
        mesh->reallocate(mesh->getIntSection(*name));
      }

      // Split the mesh along the fault sieve and create cohesive elements
      const Obj<ALE::Mesh::label_sequence>&     faces       = fault->depthStratum(1);
      const Obj<ALE::Mesh::arrow_section_type>& orientation = mesh->getArrowSection("orientation");
      int firstCohesiveCell = newPoint;
      PointSet replaceCells;
      PointSet noReplaceCells;

      for(ALE::Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter, ++newPoint) {
        if (debug) std::cout << "Considering fault face " << *f_iter << std::endl;
        const ALE::Obj<sieve_type::traits::supportSequence>& cells = faultSieve->support(*f_iter);
        const ALE::Mesh::arrow_section_type::point_type arrow(*cells->begin(), *f_iter);
        bool reversed = orientation->restrictPoint(arrow)[0] < 0;
        const ALE::Mesh::point_type cell = *cells->begin();

        if (debug) std::cout << "  Checking orientation against cell " << cell << std::endl;
        if (numFaultCorners == 2) reversed = orientation->restrictPoint(arrow)[0] == -2;
        if (reversed) {
          replaceCells.insert(cell);
          noReplaceCells.insert(*(++cells->begin()));
        } else {
          replaceCells.insert(*(++cells->begin()));
          noReplaceCells.insert(cell);
        }
        //selection::getOrientedFace(mesh, cell, &vertexRenumber, numCorners, indices, &origVertices, &faceVertices);
        //const Obj<sieve_type::coneArray> faceCone = faultSieve->nCone(*f_iter, faultDepth);

        // Adding cohesive cell (not interpolated)
        const Obj<sieve_type::coneArray>&     fCone  = faultSieve->nCone(*f_iter, faultDepth);
        const sieve_type::coneArray::iterator fBegin = fCone->begin();
        const sieve_type::coneArray::iterator fEnd   = fCone->end();
        int color = 0;

        if (debug) {std::cout << "  Creating cohesive cell " << newPoint << std::endl;}
        for(sieve_type::coneArray::iterator v_iter = fBegin; v_iter != fEnd; ++v_iter) {
          if (debug) {std::cout << "    vertex " << *v_iter << std::endl;}
          sieve->addArrow(*v_iter, newPoint, color++);
        }
        for(sieve_type::coneArray::iterator v_iter = fBegin; v_iter != fEnd; ++v_iter) {
          if (debug) {std::cout << "    shadow vertex " << vertexRenumber[*v_iter] << std::endl;}
          sieve->addArrow(vertexRenumber[*v_iter], newPoint, color++);
        }
      }
      PetscLogEventEnd(AddCohesivePoints_Event,0,0,0,0);
      // Replace all cells with a vertex on the fault that share a face with this one, or with one that does
      PetscLogEventBegin(SplitMesh_Event,0,0,0,0);
      const int_section_type::chart_type&          chart    = groupField->getChart();
      const int_section_type::chart_type::iterator chartEnd = chart.end();

      for(PointSet::const_iterator v_iter = chart.begin(); v_iter != chartEnd; ++v_iter) {
        bool modified = true;

        while(modified) {
          modified = false;
          const Obj<sieve_type::traits::supportSequence>&     neighbors = sieve->support(*v_iter);
          const sieve_type::traits::supportSequence::iterator end       = neighbors->end();

          for(sieve_type::traits::supportSequence::iterator n_iter = neighbors->begin(); n_iter != end; ++n_iter) {
            if (replaceCells.find(*n_iter)   != replaceCells.end())   continue;
            if (noReplaceCells.find(*n_iter) != noReplaceCells.end()) continue;
            if (*n_iter >= firstCohesiveCell) continue;
            if (debug) std::cout << "  Checking fault neighbor " << *n_iter << std::endl;
            // If neighbors shares a faces with anyone in replaceCells, then add
            for(PointSet::const_iterator c_iter = replaceCells.begin(); c_iter != replaceCells.end(); ++c_iter) {
              const ALE::Obj<sieve_type::coneSet>& preFace = sieve->nMeet(*c_iter, *n_iter, mesh->depth());

              if ((int) preFace->size() == faceSize) {
                if (debug) std::cout << "    Scheduling " << *n_iter << " for replacement" << std::endl;
                replaceCells.insert(*n_iter);
                modified = true;
                break;
              }
            }
          }
        }
      }
      for(PointSet::const_iterator c_iter = replaceCells.begin(); c_iter != replaceCells.end(); ++c_iter) {
        _replaceCell(sieve, *c_iter, &vertexRenumber, debug);
      }
      if (!fault->commRank()) delete [] indices;
      mesh->stratify();
      const ALE::Obj<Mesh::label_type>& label          = mesh->createLabel(std::string("censored depth"));
      const ALE::Obj<PointSet>          modifiedPoints = new PointSet();
      _computeCensoredDepth(mesh, label, mesh->getSieve(), mesh->getSieve()->roots(), firstCohesiveCell, modifiedPoints);
      if (debug) mesh->view("Mesh with Cohesive Elements");

      // Fix coordinates
      const Obj<real_section_type>&         coordinates = mesh->getRealSection("coordinates");
      const Obj<ALE::Mesh::label_sequence>& fVertices2  = fault->depthStratum(0);

      for(ALE::Mesh::label_sequence::iterator v_iter = fVertices2->begin(); v_iter != fVertices2->end(); ++v_iter) {
        coordinates->addPoint(vertexRenumber[*v_iter], coordinates->getFiberDimension(*v_iter));
        if (constraintCell) {
          coordinates->addPoint(vertexRenumber[*v_iter]+1, coordinates->getFiberDimension(*v_iter));
        }
      }
      mesh->reallocate(coordinates);
      for(ALE::Mesh::label_sequence::iterator v_iter = fVertices2->begin(); v_iter != fVertices2->end(); ++v_iter) {
        coordinates->updatePoint(vertexRenumber[*v_iter], coordinates->restrictPoint(*v_iter));
        if (constraintCell) {
        coordinates->updatePoint(vertexRenumber[*v_iter]+1, coordinates->restrictPoint(*v_iter));
        }
      }
      if (debug) coordinates->view("Coordinates with shadow vertices");
      PetscLogEventEnd(SplitMesh_Event,0,0,0,0);
    };
  };
};
#endif

#endif
