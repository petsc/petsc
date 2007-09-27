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
    static Obj<mesh_type> submesh_interpolated(const Obj<mesh_type>& mesh, const Obj<int_section_type>& label, const int dimension = -1) {
      throw ALE::Exception("Not implemented");
    };
  public:
    // This takes in a section and creates a submesh from the vertices in the section chart
    //   This is a hyperplane of one dimension lower than the mesh
    static Obj<mesh_type> submesh(const Obj<mesh_type>& mesh, const Obj<int_section_type>& label, const int dimension = -1) {
      const int dim   = mesh->getDimension();
      const int depth = mesh->depth();

      if (dim == depth) {
        return submesh_interpolated(mesh, label, dimension);
      } else if (depth == 1) {
        return submesh_uninterpolated(mesh, label, dimension);
      }
      throw ALE::Exception("Cannot handle partially interpolated meshes");
    };
    // This creates a submesh consisting of the union of the closures of the given points
    //   This mesh is the same dimension as in the input mesh
    template<typename Points>
    static Obj<mesh_type> submesh(const Obj<mesh_type>& mesh, const Obj<Points>& points) {
      const Obj<sieve_type>& sieve     = mesh->getSieve();
      Obj<mesh_type>         newMesh   = new mesh_type(mesh->comm(), mesh->getDimension(), mesh->debug());
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
            const Obj<typename sieve_type::traits::coneSequence>& cone = sieve->cone(*np_iter);
            int c = 0;

            for(typename sieve_type::traits::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter, ++c) {
              newSieve->addArrow(*c_iter, *np_iter, c);
            }
            modPoints->insert(cone->begin(), cone->end());
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
      const Obj<typename mesh_type::label_sequence>&     cells    = mesh->heightStratum(0);
      const typename mesh_type::label_sequence::iterator cBegin   = cells->begin();
      const typename mesh_type::label_sequence::iterator cEnd     = cells->end();
      const int                                          faceSize = numFaceVertices(*cBegin, mesh);
      const int                                          depth    = mesh->depth();

      for(typename mesh_type::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
        const Obj<typename sieve_type::coneSet>& cone = mesh->getSieve()->nCone(*c_iter, depth);
        PointArray cell(cone->begin(), cone->end());

        // For each face
        // - determine if its legal
        
        // - determine if its part of a neighboring cell
        // - if not, its a boundary face
        //subsets(cell, faceSize, inserter);
      }
    };
    static Obj<mesh_type> boundary_interpolated(const Obj<mesh_type>& mesh) {
      Obj<mesh_type>                                     newMesh  = new mesh_type(mesh->comm(), mesh->getDimension(), mesh->debug());
      Obj<sieve_type>                                    newSieve = new sieve_type(mesh->comm(), mesh->debug());
      const Obj<sieve_type>&                             sieve    = mesh->getSieve();
      const Obj<typename mesh_type::label_sequence>&     faces    = mesh->heightStratum(1);
      const typename mesh_type::label_sequence::iterator fBegin   = faces->begin();
      const typename mesh_type::label_sequence::iterator fEnd     = faces->end();

      for(typename mesh_type::label_sequence::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
        const Obj<typename sieve_type::traits::supportSequence>& support = sieve->support(*f_iter);

        if (support->size() == 1) {
          addClosure(sieve, newSieve, *f_iter);
        }
      }
      newMesh->setSieve(newSieve);
      newMesh->stratify();
    };
  public:
    static Obj<mesh_type> boundary(const Obj<mesh_type>& mesh) {
      const int dim   = mesh->getDimension();
      const int depth = mesh->depth();

      if (dim == depth) {
        return boundary_interpolated(mesh);
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

#endif
