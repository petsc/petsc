#ifndef included_ALE_Selection_hh
#define included_ALE_Selection_hh

#ifndef  included_ALE_SieveAlgorithms_hh
#include <sieve/SieveAlgorithms.hh>
#endif

#ifndef  included_ALE_SieveBuilder_hh
#include <sieve/SieveBuilder.hh>
#endif

#ifndef  included_ALE_Mesh_hh
#include <sieve/Mesh.hh>
#endif

namespace ALE {
  template<typename Mesh_>
  class Selection {
  public:
    typedef ALE::Mesh<PetscInt,PetscScalar>      FlexMesh;
    typedef Mesh_                                mesh_type;
    typedef typename mesh_type::sieve_type       sieve_type;
    typedef typename mesh_type::point_type       point_type;
    typedef typename mesh_type::int_section_type int_section_type;
    typedef std::set<point_type>                 PointSet;
    typedef std::vector<point_type>              PointArray;
    typedef std::pair<typename sieve_type::point_type, int> oPoint_type;
    typedef std::vector<oPoint_type>                        oPointArray;
    typedef typename ALE::SieveAlg<mesh_type>    sieveAlg;
  protected:
    template<typename Sieve, typename FaceType>
    class FaceVisitor {
    public:
      typedef typename Sieve::point_type point_type;
      typedef typename Sieve::arrow_type arrow_type;
    protected:
      const FaceType& face;
      PointArray&     origVertices;
      PointArray&     faceVertices;
      int            *indices;
      const int       debug;
      int             oppositeVertex;
      int             v;
    public:
      FaceVisitor(const FaceType& f, PointArray& oV, PointArray& fV, int *i, const int debug) : face(f), origVertices(oV), faceVertices(fV), indices(i), debug(debug), oppositeVertex(-1), v(0) {};
      void visitPoint(const point_type& point) {
        if (face->find(point) != face->end()) {
          if (debug) std::cout << "    vertex " << point << std::endl;
          indices[origVertices.size()] = v;
          origVertices.insert(origVertices.end(), point);
        } else {
          if (debug) std::cout << "    vertex " << point << std::endl;
          oppositeVertex = v;
        }
        ++v;
      };
      void visitArrow(const arrow_type&) {};
    public:
      int getOppositeVertex() {return this->oppositeVertex;};
    };
  public:
    template<typename Processor>
    static void subsets(const PointArray& v, const int size, Processor& processor, Obj<PointArray> *out = NULL, const int min = 0) {
      if (size == 0) {
        processor(*out);
        return;
      }
      if (min == 0) {
        out  = new Obj<PointArray>();
        *out = new PointArray();
      }
      for(int i = min; i < (int) v.size(); ++i) {
        (*out)->push_back(v[i]);
        subsets(v, size-1, processor, out, i+1);
        (*out)->pop_back();
      }
      if (min == 0) {delete out;}
    };
    template<typename Mesh>
    static int numFaceVertices(const Obj<Mesh>& mesh) {
      return numFaceVertices(mesh, mesh->getNumCellCorners());
    };
    template<typename Mesh>
    static int numFaceVertices(const Obj<Mesh>& mesh, const unsigned int numCorners) {
      //unsigned int numCorners = mesh->getNumCellCorners(cell, depth);
      const    int cellDim          = mesh->getDimension();
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
        case 6 : // quadratic triangle
          _numFaceVertices = 3; // Edge has 3 vertices
          break;
        case 9 : // quadratic quadrilateral
          _numFaceVertices = 3; // Edge has 3 vertices
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
        case 10 : // quadratic tetrahedron
          _numFaceVertices = 6; // Face has 6 vertices
          break;
        case 27 : // quadratic hexahedron
          _numFaceVertices = 9; // Face has 9 vertices
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
    // We need this method because we do not use interpolated sieves
    //   - Without interpolation, we cannot say what vertex collections are
    //     faces, and how they are oriented
    //   - Now we read off the list of face vertices IN THE ORDER IN WHICH
    //     THEY APPEAR IN THE CELL
    //   - This leads to simple algorithms for simplices and quads to check
    //     orientation since these sets are always valid faces
    //   - This is not true with hexes, so we just sort and check explicit cases
    //   - This means for hexes that we have to alter the vertex container as well
    template<typename Mesh>
    static bool faceOrientation(const point_type& cell, const Obj<Mesh>& mesh, const int numCorners,
                                const int indices[], const int oppositeVertex, PointArray *origVertices, PointArray *faceVertices) {
      const int cellDim   = mesh->getDimension();
      const int debug     = mesh->debug();
      bool      posOrient = false;

      if (debug) std::cout << "cellDim: " << cellDim << ", numCorners: " << numCorners << std::endl;

      if (cellDim == numCorners-1) {
        // Simplices
        posOrient = !(oppositeVertex%2);
      } else if (cellDim == 1 && numCorners == 3) {
	posOrient = true;
      } else if (cellDim == 2 && numCorners == 4) {
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
      } else if (cellDim == 2 && numCorners == 6) {
        // Quadratic triangle (I hate this)
        // Edges are determined by the first 2 vertices (corners of edges)
        const int faceSizeTri = 3;
        int  sortedIndices[3];
        bool found = false;
        int faceVerticesTriSorted[9] = {
          0, 3,  4, // bottom
          1, 4,  5, // right
          2, 3,  5, // left
        };
        int faceVerticesTri[9] = {
          0, 3,  4, // bottom
          1, 4,  5, // right
          2, 5,  3, // left
        };

        for(int i = 0; i < faceSizeTri; ++i) sortedIndices[i] = indices[i];
        std::sort(sortedIndices, sortedIndices+faceSizeTri);
        for (int iFace=0; iFace < 4; ++iFace) {
          const int ii = iFace*faceSizeTri;
          if ((sortedIndices[0] == faceVerticesTriSorted[ii+0]) &&
              (sortedIndices[1] == faceVerticesTriSorted[ii+1])) {
            if (debug) {
              if (iFace == 0) std::cout << "Bottom edge" << std::endl;
              else if (iFace == 1) std::cout << "Right edge" << std::endl;
              else if (iFace == 2) std::cout << "Left edge" << std::endl;
            }  // if
            for (int fVertex=0; fVertex < faceSizeTri; ++fVertex)
              for (int cVertex=0; cVertex < faceSizeTri; ++cVertex)
                if (indices[cVertex] == faceVerticesTri[ii+fVertex]) {
                  faceVertices->push_back((*origVertices)[cVertex]); 
                  break;
                } // if
            found = true;
            break;
          } // if
        } // for
        if (!found) {throw ALE::Exception("Invalid tri crossface");}
        return true;
      } else if (cellDim == 2 && numCorners == 9) {
        // Quadratic quad (I hate this)
        // Edges are determined by the first 2 vertices (corners of edges)
        const int faceSizeQuad = 3;
        int  sortedIndices[3];
        bool found = false;
        int faceVerticesQuadSorted[12] = {
          0, 1,  4, // bottom
          1, 2,  5, // right
          2, 3,  6, // top
          0, 3,  7, // left
        };
        int faceVerticesQuad[12] = {
          0, 1,  4, // bottom
          1, 2,  5, // right
          2, 3,  6, // top
          3, 0,  7, // left
        };

        for(int i = 0; i < faceSizeQuad; ++i) sortedIndices[i] = indices[i];
        std::sort(sortedIndices, sortedIndices+faceSizeQuad);
        for (int iFace=0; iFace < 4; ++iFace) {
          const int ii = iFace*faceSizeQuad;
          if ((sortedIndices[0] == faceVerticesQuadSorted[ii+0]) &&
              (sortedIndices[1] == faceVerticesQuadSorted[ii+1])) {
            if (debug) {
              if (iFace == 0) std::cout << "Bottom edge" << std::endl;
              else if (iFace == 1) std::cout << "Right edge" << std::endl;
              else if (iFace == 2) std::cout << "Top edge" << std::endl;
              else if (iFace == 3) std::cout << "Left edge" << std::endl;
            }  // if
            for (int fVertex=0; fVertex < faceSizeQuad; ++fVertex)
              for (int cVertex=0; cVertex < faceSizeQuad; ++cVertex)
                if (indices[cVertex] == faceVerticesQuad[ii+fVertex]) {
                  faceVertices->push_back((*origVertices)[cVertex]); 
                  break;
                } // if
            found = true;
            break;
          } // if
        } // for
        if (!found) {throw ALE::Exception("Invalid quad crossface");}
        return true;
      } else if (cellDim == 3 && numCorners == 8) {
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
        //
        // Faces are determined by the first 4 vertices (corners of faces)
        const int faceSizeHex = 4;
        int  sortedIndices[4];
        bool found = false;
        int faceVerticesHexSorted[24] = {
          0, 1, 2, 3,  // bottom
          4, 5, 6, 7,  // top
          0, 1, 4, 5,  // front
          1, 2, 5, 6,  // right
          2, 3, 6, 7,  // back
          0, 3, 4, 7,  // left
        };
        int faceVerticesHex[24] = {
          3, 2, 1, 0,  // bottom
          4, 5, 6, 7,  // top
          0, 1, 5, 4,  // front
          1, 2, 6, 5,  // right
          2, 3, 7, 6,  // back
          3, 0, 4, 7,  // left
        };

        for(int i = 0; i < faceSizeHex; ++i) sortedIndices[i] = indices[i];
        std::sort(sortedIndices, sortedIndices+faceSizeHex);
        for (int iFace=0; iFace < 6; ++iFace) {
          const int ii = iFace*faceSizeHex;
          if ((sortedIndices[0] == faceVerticesHexSorted[ii+0]) &&
              (sortedIndices[1] == faceVerticesHexSorted[ii+1]) &&
              (sortedIndices[2] == faceVerticesHexSorted[ii+2]) &&
              (sortedIndices[3] == faceVerticesHexSorted[ii+3])) {
            if (debug) {
              if (iFace == 0) std::cout << "Bottom quad" << std::endl;
              else if (iFace == 1) std::cout << "Top quad" << std::endl;
              else if (iFace == 2) std::cout << "Front quad" << std::endl;
              else if (iFace == 3) std::cout << "Right quad" << std::endl;
              else if (iFace == 4) std::cout << "Back quad" << std::endl;
              else if (iFace == 5) std::cout << "Left quad" << std::endl;
            }  // if
            for (int fVertex=0; fVertex < faceSizeHex; ++fVertex)
              for (int cVertex=0; cVertex < faceSizeHex; ++cVertex)
                if (indices[cVertex] == faceVerticesHex[ii+fVertex]) {
                  faceVertices->push_back((*origVertices)[cVertex]); 
                  break;
                } // if
            found = true;
            break;
          } // if
        } // for
        if (!found) {throw ALE::Exception("Invalid hex crossface");}
        return true;
      } else if (cellDim == 3 && numCorners == 10) {
        // Quadratic tet
        // Faces are determined by the first 3 vertices (corners of faces)
        const int faceSizeTet = 6;
        int  sortedIndices[6];
        bool found = false;
        int faceVerticesTetSorted[24] = {
          0, 1, 2,  6, 7, 8, // bottom
          0, 3, 4,  6, 7, 9,  // front
          1, 4, 5,  7, 8, 9,  // right
          2, 3, 5,  6, 8, 9,  // left
        };
        int faceVerticesTet[24] = {
          0, 1, 2,  6, 7, 8, // bottom
          0, 4, 3,  6, 7, 9,  // front
          1, 5, 4,  7, 8, 9,  // right
          2, 3, 5,  8, 6, 9,  // left
        };

        for(int i = 0; i < faceSizeTet; ++i) sortedIndices[i] = indices[i];
        std::sort(sortedIndices, sortedIndices+faceSizeTet);
        for (int iFace=0; iFace < 6; ++iFace) {
          const int ii = iFace*faceSizeTet;
          if ((sortedIndices[0] == faceVerticesTetSorted[ii+0]) &&
              (sortedIndices[1] == faceVerticesTetSorted[ii+1]) &&
              (sortedIndices[2] == faceVerticesTetSorted[ii+2]) &&
              (sortedIndices[3] == faceVerticesTetSorted[ii+3])) {
            if (debug) {
              if (iFace == 0) std::cout << "Bottom tri" << std::endl;
              else if (iFace == 1) std::cout << "Front tri" << std::endl;
              else if (iFace == 2) std::cout << "Right tri" << std::endl;
              else if (iFace == 3) std::cout << "Left tri" << std::endl;
            }  // if
            for (int fVertex=0; fVertex < faceSizeTet; ++fVertex)
              for (int cVertex=0; cVertex < faceSizeTet; ++cVertex)
                if (indices[cVertex] == faceVerticesTet[ii+fVertex]) {
                  faceVertices->push_back((*origVertices)[cVertex]); 
                  break;
                } // if
            found = true;
            break;
          } // if
        } // for
        if (!found) {throw ALE::Exception("Invalid tet crossface");}
        return true;
      } else if (cellDim == 3 && numCorners == 27) {
        // Quadratic hexes (I hate this)
        //   A hex is two oriented quads with the normal of the first
        //   pointing up at the second.
        //
        //     7---6
        //    /|  /|
        //   4---5 |
        //   | 3-|-2
        //   |/  |/
        //   0---1
        //
        // Faces are determined by the first 4 vertices (corners of faces)
        const int faceSizeQuadHex = 9;
        int  sortedIndices[9];
        bool found = false;
        int faceVerticesQuadHexSorted[54] = {
          0, 1, 2, 3,  8, 9, 10, 11,  24, // bottom
          4, 5, 6, 7,  12, 13, 14, 15,  25, // top
          0, 1, 4, 5,  8, 12, 16, 17,  22, // front
          1, 2, 5, 6,  9, 13, 17, 18,  21, // right
          2, 3, 6, 7,  10, 14, 18, 19,  23, // back
          0, 3, 4, 7,  11, 15, 16, 19,  20, // left
        };
        int faceVerticesQuadHex[54] = {
          3, 2, 1, 0,  10, 9, 8, 11,  24, // bottom
          4, 5, 6, 7,  12, 13, 14, 15,  25, // top
          0, 1, 5, 4,  8, 17, 12, 16,  22, // front
          1, 2, 6, 5,  9, 18, 13, 17,  21, // right
          2, 3, 7, 6,  10, 19, 14, 18,  23, // back
          3, 0, 4, 7,  11, 16, 15, 19,  20 // left
        };

        for (int i=0; i < faceSizeQuadHex; ++i) sortedIndices[i] = indices[i];
        std::sort(sortedIndices, sortedIndices+faceSizeQuadHex);
        for (int iFace=0; iFace < 6; ++iFace) {
          const int ii = iFace*faceSizeQuadHex;
          if ((sortedIndices[0] == faceVerticesQuadHexSorted[ii+0]) &&
              (sortedIndices[1] == faceVerticesQuadHexSorted[ii+1]) &&
              (sortedIndices[2] == faceVerticesQuadHexSorted[ii+2]) &&
              (sortedIndices[3] == faceVerticesQuadHexSorted[ii+3])) {
            if (debug) {
              if (iFace == 0) std::cout << "Bottom quad" << std::endl;
              else if (iFace == 1) std::cout << "Top quad" << std::endl;
              else if (iFace == 2) std::cout << "Front quad" << std::endl;
              else if (iFace == 3) std::cout << "Right quad" << std::endl;
              else if (iFace == 4) std::cout << "Back quad" << std::endl;
              else if (iFace == 5) std::cout << "Left quad" << std::endl;
            }  // if
            for (int fVertex=0; fVertex < faceSizeQuadHex; ++fVertex)
              for (int cVertex=0; cVertex < faceSizeQuadHex; ++cVertex)
                if (indices[cVertex] == faceVerticesQuadHex[ii+fVertex]) {
                  faceVertices->push_back((*origVertices)[cVertex]); 
                  break;
                } // if
            found = true;
            break;
          } // if
        } // for
        if (!found) {throw ALE::Exception("Invalid hex crossface");}
        return true;
      } else {
        throw ALE::Exception("Unknown cell type for faceOrientation().");
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
    template<typename Mesh, typename FaceType>
    static bool getOrientedFace(const Obj<Mesh>& mesh, const point_type& cell, FaceType face,
                                const int numCorners, int indices[], PointArray *origVertices, PointArray *faceVertices)
    {
      FaceVisitor<typename Mesh::sieve_type,FaceType> v(face, *origVertices, *faceVertices, indices, mesh->debug());

      origVertices->clear();
      faceVertices->clear();
      mesh->getSieve()->cone(cell, v);
      return faceOrientation(cell, mesh, numCorners, indices, v.getOppositeVertex(), origVertices, faceVertices);
    };
    template<typename FaceType>
    static bool getOrientedFace(const Obj<FlexMesh>& mesh, const point_type& cell, FaceType face,
                                const int numCorners, int indices[], PointArray *origVertices, PointArray *faceVertices)
    {
      const Obj<typename FlexMesh::sieve_type::traits::coneSequence>& cone = mesh->getSieve()->cone(cell);
      const int debug = mesh->debug();
      int       v     = 0;
      int       oppositeVertex;

      origVertices->clear();
      faceVertices->clear();
      for(typename FlexMesh::sieve_type::traits::coneSequence::iterator v_iter = cone->begin(); v_iter != cone->end(); ++v_iter, ++v) {
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
    template<typename Sieve>
    static void insertFace(const Obj<mesh_type>& mesh, const Obj<Sieve>& subSieve, const Obj<PointSet>& face, point_type& f,
                           const point_type& cell, const int numCorners, int indices[], PointArray *origVertices, PointArray *faceVertices)
    {
      const Obj<typename Sieve::supportSet> preFace = subSieve->nJoin1(face);
      const int                             debug   = subSieve->debug();

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
    template<typename Sieve>
    class FaceInserterV {
    protected:
      const Obj<mesh_type>&  mesh;
      const Obj<sieve_type>& sieve;
      const Obj<Sieve>&      subSieve;
      point_type&            f;
      const point_type       cell;
      const int              numCorners;
      int                   *indices;
      PointArray            *origVertices;
      PointArray            *faceVertices;
      PointSet              *subCells;
      const int              debug;
    public:
      FaceInserterV(const Obj<mesh_type>& mesh, const Obj<sieve_type>& sieve, const Obj<Sieve>& subSieve, point_type& f, const point_type& cell, const int numCorners, int indices[], PointArray *origVertices, PointArray *faceVertices, PointSet *subCells) : mesh(mesh), sieve(sieve), subSieve(subSieve), f(f), cell(cell), numCorners(numCorners), indices(indices), origVertices(origVertices), faceVertices(faceVertices), subCells(subCells), debug(mesh->debug()) {};
      virtual ~FaceInserterV() {};
    public:
      void operator()(const Obj<PointArray>& face) {
        ISieveVisitor::PointRetriever<sieve_type> jV(sieve->getMaxSupportSize());

        sieve->join(*face, jV);
        if (jV.getSize() == 1) {
          if (debug) std::cout << "  Contains a boundary face on the submesh" << std::endl;
          PointSet faceSet(face->begin(), face->end());
          ALE::Selection<mesh_type>::insertFace(mesh, subSieve, faceSet, f, cell, numCorners, indices, origVertices, faceVertices);
          subCells->insert(cell);
        }
      };
    };
  protected:
    static int binomial(const int i, const int j) {
      assert(j <= i);
      assert(i < 34);
      if (j == 0) {
        return 1;
      } else if (j == i) {
        return 1;
      } else {
        return binomial(i-1, j) + binomial(i-1, j-1);
      }
    };
  public:
    // This takes in a section and creates a submesh from the vertices in the section chart
    //   This is a hyperplane of one dimension lower than the mesh
    static Obj<mesh_type> submesh_uninterpolated(const Obj<mesh_type>& mesh, const Obj<int_section_type>& label, const int dimension = -1, const bool boundaryFaces = true) {
      // A challenge here is to coordinate the extra numbering of new faces
      //   In serial, it is enough to number after the last point:
      //     Use sieve->base()->size() + sieve->cap()->size(), or determine the greatest point
      //   In parallel, there are two steps:
      //     1) Use the serial result, and reduce either with add (for size) or max (for greatest point)
      //     2) Determine how many faces will be created on each process
      //        This will be bounded by C(numCorners, faceSize)*submeshCells
      //        Thus it looks like we should first accumulate submeshCells, and then create faces
      typedef typename mesh_type::label_type        label_type;
      typedef typename int_section_type::chart_type chart_type;
      const int                  dim        = (dimension > 0) ? dimension : mesh->getDimension()-1;
      const Obj<sieve_type>&     sieve      = mesh->getSieve();
      Obj<mesh_type>             submesh    = new mesh_type(mesh->comm(), dim, mesh->debug());
      Obj<sieve_type>            subSieve   = new sieve_type(mesh->comm(), mesh->debug());
      const bool                 censor     = mesh->hasLabel("censored depth");
      const Obj<label_type>&     depthLabel = censor ? mesh->getLabel("censored depth") : mesh->getLabel("depth");
      const int                  depth      = mesh->depth();
      const int                  height     = mesh->height();
      const chart_type&          chart      = label->getChart();
      const int                  numCorners = sieve->nCone(*mesh->heightStratum(0)->begin(), depth)->size();
      const int                  faceSize   = numFaceVertices(mesh);
      Obj<PointSet>              face       = new PointSet();
      int                        f          = sieve->base()->size() + sieve->cap()->size();
      const int                  debug      = mesh->debug();
      int                       *indices    = new int[faceSize];
      PointArray                 origVertices, faceVertices;
      PointSet                   submeshVertices, submeshCells;


      const typename chart_type::iterator chartEnd = chart.end();
      for(typename chart_type::iterator c_iter = chart.begin(); c_iter != chartEnd; ++c_iter) {
        if (label->getFiberDimension(*c_iter)) submeshVertices.insert(*c_iter);
      }
      const typename PointSet::const_iterator svBegin = submeshVertices.begin();
      const typename PointSet::const_iterator svEnd   = submeshVertices.end();

      for(typename PointSet::const_iterator sv_iter = svBegin; sv_iter != svEnd; ++sv_iter) {
        const Obj<typename sieveAlg::supportArray>&     cells  = sieveAlg::nSupport(mesh, *sv_iter, depth);
        const typename sieveAlg::supportArray::iterator cBegin = cells->begin();
        const typename sieveAlg::supportArray::iterator cEnd   = cells->end();

        if (debug) std::cout << "Checking submesh vertex " << *sv_iter << std::endl;
        for(typename sieveAlg::supportArray::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
          if (debug) std::cout << "  Checking cell " << *c_iter << std::endl;
          if (submeshCells.find(*c_iter) != submeshCells.end())	continue;
          if (censor && (!mesh->getValue(depthLabel, *c_iter))) continue;
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
            if (debug) std::cout << "  Has all boundary faces on the submesh" << std::endl;
            submeshCells.insert(*c_iter);
          }
          if ((int) face->size() == faceSize) {
            if (debug) std::cout << "  Contains a face on the submesh" << std::endl;
            submeshCells.insert(*c_iter);
          }
        }
      }
      if (mesh->commSize() > 1) {
        int localF     = f;
        int localFaces = binomial(numCorners, faceSize)*submeshCells.size();
        int maxFaces;

        MPI_Allreduce(&localF, &f, 1, MPI_INT, MPI_SUM, mesh->comm());
        //     2) Determine how many faces will be created on each process
        //        This will be bounded by faceSize*submeshCells
        //        Thus it looks like we should first accumulate submeshCells, and then create faces
        MPI_Scan(&localFaces, &maxFaces, 1, MPI_INT, MPI_SUM, mesh->comm());
        f += maxFaces - localFaces;
      }
      for(typename PointSet::const_iterator c_iter = submeshCells.begin(); c_iter != submeshCells.end(); ++c_iter) {
        if (debug) std::cout << "  Processing submesh cell " << *c_iter << std::endl;
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
          // Here we allow a set of vertices to lie completely on a boundary cell (like a corner tetrahedron)
          //   We have to take all the faces, and discard those in the interior
          FaceInserter inserter(mesh, sieve, subSieve, f, *c_iter, numCorners, indices, &origVertices, &faceVertices, &submeshCells);
          PointArray   faceVec(face->begin(), face->end());

          subsets(faceVec, faceSize, inserter);
        }
        if ((int) face->size() == faceSize) {
          insertFace(mesh, subSieve, face, f, *c_iter, numCorners, indices, &origVertices, &faceVertices);
        }
      }
      delete [] indices;
      submesh->setSieve(subSieve);
      submesh->stratify();
      if (debug) submesh->view("Submesh");
      return submesh;
    };
    // This takes in a section and creates a submesh from the vertices in the section chart
    //   This is a hyperplane of one dimension lower than the mesh
    static Obj<mesh_type> submesh_interpolated(const Obj<mesh_type>& mesh, const Obj<int_section_type>& label, const int dimension = -1, const bool boundaryFaces = true) {
      const int debug  = mesh->debug();
      const int depth  = mesh->depth();
      const int height = mesh->height();
      const typename int_section_type::chart_type&                chart        = label->getChart();
      const typename int_section_type::chart_type::const_iterator chartEnd     = chart.end();
      const Obj<PointSet>                                         submeshFaces = new PointSet();
      PointSet submeshVertices;

      for(typename int_section_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chartEnd; ++c_iter) {
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
    template<typename output_mesh_type>
    static Obj<output_mesh_type> submeshV_uninterpolated(const Obj<mesh_type>& mesh, const Obj<int_section_type>& label, const int dimension = -1, const bool boundaryFaces = true) {
      typedef typename mesh_type::label_type        label_type;
      typedef typename int_section_type::chart_type chart_type;
      const int                           dim        = (dimension > 0) ? dimension : mesh->getDimension()-1;
      const Obj<sieve_type>&              sieve      = mesh->getSieve();
      Obj<FlexMesh>                       submesh    = new FlexMesh(mesh->comm(), dim, mesh->debug());
      Obj<typename FlexMesh::sieve_type>  subSieve   = new typename FlexMesh::sieve_type(mesh->comm(), mesh->debug());
      const bool                          censor     = mesh->hasLabel("censored depth");
      const Obj<label_type>&              depthLabel = censor ? mesh->getLabel("censored depth") : mesh->getLabel("depth");
      const chart_type&                   chart      = label->getChart();
      const int                           numCorners = mesh->getNumCellCorners();
      const int                           faceSize   = numFaceVertices(mesh);
      Obj<PointSet>                       face       = new PointSet();
      int                                 f          = sieve->getBaseSize() + sieve->getCapSize();
      const int                           debug      = mesh->debug();
      int                                *indices    = new int[faceSize];
      PointArray                          origVertices, faceVertices;
      PointSet                            submeshVertices, submeshCells;

      const typename chart_type::const_iterator chartEnd = chart.end();
      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chartEnd; ++c_iter) {
        if (label->getFiberDimension(*c_iter)) submeshVertices.insert(*c_iter);
      }
      const typename PointSet::const_iterator svBegin = submeshVertices.begin();
      const typename PointSet::const_iterator svEnd   = submeshVertices.end();
      typename ISieveVisitor::PointRetriever<sieve_type> sV(sieve->getMaxSupportSize());
      typename ISieveVisitor::PointRetriever<sieve_type> cV(sieve->getMaxConeSize());

      for(typename PointSet::const_iterator sv_iter = svBegin; sv_iter != svEnd; ++sv_iter) {
        sieve->support(*sv_iter, sV);
        const int         numCells = sV.getSize();
        const point_type *cells    = sV.getPoints();
    
        if (debug) std::cout << "Checking submesh vertex " << *sv_iter << std::endl;
        for(int c = 0; c < numCells; ++c) {
          if (debug) std::cout << "  Checking cell " << cells[c] << std::endl;
          if (submeshCells.find(cells[c]) != submeshCells.end()) continue;
          if (censor && (!mesh->getValue(depthLabel, cells[c]))) continue;
          sieve->cone(cells[c], cV);
          const int         numVertices = cV.getSize();
          const point_type *vertices    = cV.getPoints();

          face->clear();
          for(int v = 0; v < numVertices; ++v) {
            if (submeshVertices.find(vertices[v]) != svEnd) {
              if (debug) std::cout << "    contains submesh vertex " << vertices[v] << std::endl;
              face->insert(face->end(), vertices[v]);
            }
          }
          if ((int) face->size() > faceSize) {
            if (!boundaryFaces) throw ALE::Exception("Invalid fault mesh: Too many vertices of an element on the fault");
            if (debug) std::cout << "  Has all boundary faces on the submesh" << std::endl;
            submeshCells.insert(cells[c]);
          }
          if ((int) face->size() == faceSize) {
            if (debug) std::cout << "  Contains a face on the submesh" << std::endl;
            submeshCells.insert(cells[c]);
          }
          cV.clear();
        }
        sV.clear();
      }
      if (mesh->commSize() > 1) {
        int localF     = f;
        int localFaces = binomial(numCorners, faceSize)*submeshCells.size();
        int maxFaces;

        MPI_Allreduce(&localF, &f, 1, MPI_INT, MPI_SUM, mesh->comm());
        //     2) Determine how many faces will be created on each process
        //        This will be bounded by faceSize*submeshCells
        //        Thus it looks like we should first accumulate submeshCells, and then create faces
        MPI_Scan(&localFaces, &maxFaces, 1, MPI_INT, MPI_SUM, mesh->comm());
        f += maxFaces - localFaces;
      }
      for(typename PointSet::const_iterator c_iter = submeshCells.begin(); c_iter != submeshCells.end(); ++c_iter) {
        if (debug) std::cout << "  Processing submesh cell " << *c_iter << std::endl;
        sieve->cone(*c_iter, cV);
        const int         numVertices = cV.getSize();
        const point_type *vertices    = cV.getPoints();

        face->clear();
        for(int v = 0; v < numVertices; ++v) {
          if (submeshVertices.find(vertices[v]) != svEnd) {
            if (debug) std::cout << "    contains submesh vertex " << vertices[v] << std::endl;
            face->insert(face->end(), vertices[v]);
          }
        }
        if ((int) face->size() > faceSize) {
          if (!boundaryFaces) throw ALE::Exception("Invalid fault mesh: Too many vertices of an element on the fault");
          // Here we allow a set of vertices to lie completely on a boundary cell (like a corner tetrahedron)
          //   We have to take all the faces, and discard those in the interior
          FaceInserterV<FlexMesh::sieve_type> inserter(mesh, sieve, subSieve, f, *c_iter, numCorners, indices, &origVertices, &faceVertices, &submeshCells);
          PointArray                          faceVec(face->begin(), face->end());

          subsets(faceVec, faceSize, inserter);
        }
        if ((int) face->size() == faceSize) {
          insertFace(mesh, subSieve, face, f, *c_iter, numCorners, indices, &origVertices, &faceVertices);
        }
        cV.clear();
      }
      delete [] indices;
      submesh->setSieve(subSieve);
      submesh->stratify();
      if (debug) submesh->view("Submesh");

      Obj<output_mesh_type> isubmesh = new output_mesh_type(submesh->comm(), submesh->getDimension(), submesh->debug());
      Obj<typename output_mesh_type::sieve_type> isieve = new typename output_mesh_type::sieve_type(submesh->comm(), submesh->debug());
      std::map<typename output_mesh_type::point_type,typename output_mesh_type::point_type> renumbering;
      isubmesh->setSieve(isieve);
      ALE::ISieveConverter::convertMesh(*submesh, *isubmesh, renumbering, false);
      return isubmesh;
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
    template<typename output_mesh_type>
    static Obj<output_mesh_type> submeshV(const Obj<mesh_type>& mesh, const Obj<int_section_type>& label, const int dimension = -1) {
      const int dim   = mesh->getDimension();
      const int depth = mesh->depth();

#if 0
      if (dim == depth) {
        //return submesh_interpolated(mesh, label, dimension, false);
        throw ALE::Exception("Cannot handle interpolated meshes");
      } else if (depth == 1) {
        return submeshV_uninterpolated<output_mesh_type>(mesh, label, dimension);
      }
#else
      if (depth == 1) {
        return submeshV_uninterpolated<output_mesh_type>(mesh, label, dimension);
      } else if (dim == depth) {
        //return submesh_interpolated(mesh, label, dimension, false);
        throw ALE::Exception("Cannot handle interpolated meshes");
      }
#endif
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
      const int                                          faceSize = numFaceVertices(mesh);

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
      Obj<mesh_type>                                     newMesh  = new mesh_type(mesh->comm(), mesh->getDimension()-1, mesh->debug());
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
    template<typename SieveTypeA, typename SieveTypeB>
    static void addClosureV(const Obj<SieveTypeA>& sieveA, const Obj<SieveTypeB>& sieveB, const point_type& p, const int depth = 1) {
      typedef std::set<typename SieveTypeA::point_type> coneSet;
      ALE::ISieveVisitor::PointRetriever<SieveTypeA> cV(std::max(1, sieveA->getMaxConeSize()));
      Obj<coneSet> current = new coneSet();
      Obj<coneSet> next    = new coneSet();
      Obj<coneSet> tmp;

      current->insert(p);
      while(current->size()) {
        for(typename coneSet::const_iterator p_iter = current->begin(); p_iter != current->end(); ++p_iter) {
          sieveA->cone(*p_iter, cV);
          const typename SieveTypeA::point_type *cone = cV.getPoints();

          for(int c = 0; c < (int) cV.getSize(); ++c) {
            sieveB->addArrow(cone[c], *p_iter);
            next->insert(cone[c]);
          }
          cV.clear();
        }
        tmp = current; current = next; next = tmp;
        next->clear();
      }
      if (!depth) {
        ALE::ISieveVisitor::PointRetriever<SieveTypeA> sV(std::max(1, sieveA->getMaxSupportSize()));

        sieveA->support(p, sV);
        const typename SieveTypeA::point_type *support = sV.getPoints();
            
        for(int s = 0; s < (int) sV.getSize(); ++s) {
          sieveB->addArrow(p, support[s]);
        }
        sV.clear();
      }
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
      } else if (depth == -1) {
        Obj<mesh_type>  newMesh  = new mesh_type(mesh->comm(), mesh->getDimension()-1, mesh->debug());
        Obj<sieve_type> newSieve = new sieve_type(mesh->comm(), mesh->debug());

        newMesh->setSieve(newSieve);
        newMesh->stratify();
        return newMesh;
      }
      throw ALE::Exception("Cannot handle partially interpolated meshes");
    };
    template<typename MeshTypeQ>
    static Obj<FlexMesh> boundaryV_uninterpolated(const Obj<MeshTypeQ>& mesh, const int faceHeight = 1) {
        throw ALE::Exception("Cannot handle uninterpolated meshes");
    };
    // This method takes in an interpolated mesh, and returns the boundary
    template<typename MeshTypeQ>
    static Obj<FlexMesh> boundaryV_interpolated(const Obj<MeshTypeQ>& mesh, const int faceHeight = 1) {
      Obj<FlexMesh>                                       newMesh  = new FlexMesh(mesh->comm(), mesh->getDimension()-1, mesh->debug());
      Obj<typename FlexMesh::sieve_type>                  newSieve = new typename FlexMesh::sieve_type(mesh->comm(), mesh->debug());
      const Obj<typename MeshTypeQ::sieve_type>&          sieve    = mesh->getSieve();
      const Obj<typename MeshTypeQ::label_sequence>&      faces    = mesh->heightStratum(faceHeight);
      const typename MeshTypeQ::label_sequence::iterator  fBegin   = faces->begin();
      const typename MeshTypeQ::label_sequence::iterator  fEnd     = faces->end();
      const int                                           depth    = faceHeight - mesh->depth();
      ALE::ISieveVisitor::PointRetriever<sieve_type>      sV(std::max(1, sieve->getMaxSupportSize()));

      for(typename MeshTypeQ::label_sequence::iterator f_iter = fBegin; f_iter != fEnd; ++f_iter) {
        sieve->support(*f_iter, sV);

        if (sV.getSize() == 1) {
          addClosureV(sieve, newSieve, *f_iter, depth);
        }
        sV.clear();
      }
      newMesh->setSieve(newSieve);
      newMesh->stratify();
      return newMesh;
    };
    template<typename MeshTypeQ>
    static Obj<FlexMesh> boundaryV(const Obj<MeshTypeQ>& mesh, const int faceHeight = 1) {
      const int dim   = mesh->getDimension();
      const int depth = mesh->depth();

      if (dim == depth) {
        return boundaryV_interpolated(mesh);
      } else if (depth == dim+1) {
        return boundaryV_interpolated(mesh, 2);
      } else if (depth == 1) {
        throw ALE::Exception("Cannot handle uninterpolated meshes");
      } else if (depth == -1) {
        Obj<mesh_type>  newMesh  = new mesh_type(mesh->comm(), mesh->getDimension()-1, mesh->debug());
        Obj<sieve_type> newSieve = new sieve_type(mesh->comm(), mesh->debug());

        newMesh->setSieve(newSieve);
        newMesh->stratify();
        return newMesh;
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
        typename sieve_type::point_type                           cell    = *c_iter;
        const Obj<typename sieve_type::traits::coneSequence>&     cone    = sieve->cone(cell);
        const int                                                 corners = cone->size();
        const typename sieve_type::traits::coneSequence::iterator vBegin  = cone->begin();
        const typename sieve_type::traits::coneSequence::iterator vEnd    = cone->end();

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
          // Will need to handle cohesive cells here
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
      static PetscLogEvent CreateFaultMesh_Event = 0, OrientFaultMesh_Event = 0, AddCohesivePoints_Event = 0, SplitMesh_Event = 0;

      if (!CreateFaultMesh_Event) {
        PetscLogEventRegister("CreateFaultMesh", 0,&CreateFaultMesh_Event);
      }
      if (!OrientFaultMesh_Event) {
        PetscLogEventRegister("OrientFaultMesh", 0,&OrientFaultMesh_Event);
      }
      if (!AddCohesivePoints_Event) {
        PetscLogEventRegister("AddCohesivePoints", 0,&AddCohesivePoints_Event);
      }
      if (!SplitMesh_Event) {
        PetscLogEventRegister("SplitMesh", 0,&SplitMesh_Event);
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
        faceSize   = selection::numFaceVertices(mesh);
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
