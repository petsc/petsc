#ifndef included_ALE_CoSieve_hh
#define included_ALE_CoSieve_hh

#ifndef  included_ALE_Sieve_hh
#include <Sieve.hh>
#endif

namespace ALE {
  class ParallelObject {
  protected:
    int         _debug;
    MPI_Comm    _comm;
    int         _commRank;
    int         _commSize;
    PetscObject _petscObj;
    void __init(MPI_Comm comm) {
      static PetscCookie objType = -1;
      //const char        *id_name = ALE::getClassName<T>();
      const char        *id_name = "ParallelObject";
      PetscErrorCode     ierr;

      if (objType < 0) {
        ierr = PetscLogClassRegister(&objType, id_name);CHKERROR(ierr, "Error in PetscLogClassRegister");
      }
      this->_comm = comm;
      ierr = MPI_Comm_rank(this->_comm, &this->_commRank); CHKERROR(ierr, "Error in MPI_Comm_rank");
      ierr = MPI_Comm_size(this->_comm, &this->_commSize); CHKERROR(ierr, "Error in MPI_Comm_size");
      ierr = PetscObjectCreateGeneric(this->_comm, objType, id_name, &this->_petscObj);CHKERROR(ierr, "Error in PetscObjectCreate");
      //ALE::restoreClassName<T>(id_name);
    };
  public:
    ParallelObject(MPI_Comm comm = PETSC_COMM_SELF, const int debug = 0) : _debug(debug), _petscObj(NULL) {__init(comm);}
    virtual ~ParallelObject() {
      if (this->_petscObj) {
        PetscErrorCode ierr = PetscObjectDestroy(this->_petscObj); CHKERROR(ierr, "Failed in PetscObjectDestroy");
        this->_petscObj = NULL;
      }
    };
  public:
    int         debug()    const {return this->_debug;};
    void        setDebug(const int debug) {this->_debug = debug;};
    MPI_Comm    comm()     const {return this->_comm;};
    int         commSize() const {return this->_commSize;};
    int         commRank() const {return this->_commRank;}
    PetscObject petscObj() const {return this->_petscObj;};
  };

  namespace New {
    template<typename Sieve_>
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
    };

    template<typename Patch_, typename Sieve_>
    class Topology : public ALE::ParallelObject {
    public:
      typedef Patch_                                                    patch_type;
      typedef Sieve_                                                    sieve_type;
      typedef typename sieve_type::point_type                           point_type;
      typedef typename ALE::set<point_type>                             PointSet;
      typedef typename std::map<patch_type, Obj<sieve_type> >           sheaf_type;
      typedef typename ALE::Sifter<int, point_type, int>                patch_label_type;
      typedef typename std::map<patch_type, Obj<patch_label_type> >     label_type;
      typedef typename std::map<const typename std::string, label_type> labels_type;
      typedef typename patch_label_type::supportSequence                label_sequence;
    protected:
      sheaf_type  _sheaf;
      labels_type _labels;
      int         _maxHeight;
      int         _maxDepth;
    public:
      Topology(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _maxHeight(-1) {};
    public: // Verifiers
      void checkPatch(const patch_type& patch) {
        if (this->_sheaf.find(patch) == this->_sheaf.end()) {
          ostringstream msg;
          msg << "Invalid topology patch: " << patch << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
    public: // Accessors
      const Obj<sieve_type>& getPatch(const patch_type& patch) {
        this->checkPatch(patch);
        return this->_sheaf[patch];
      };
      void setPatch(const patch_type& patch, const Obj<sieve_type>& sieve) {
        this->_sheaf[patch] = sieve;
      };
      int getValue (const Obj<patch_label_type>& label, const point_type& point, const int defValue = 0) {
        const Obj<typename patch_label_type::coneSequence>& cone = label->cone(point);

        if (cone->size() == 0) return defValue;
        return *cone->begin();
      };
      template<typename InputPoints>
      int getMaxValue (const Obj<patch_label_type>& label, const Obj<InputPoints>& points, const int defValue = 0) {
        int maxValue = defValue;

        for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          maxValue = std::max(maxValue, this->getValue(label, *p_iter, defValue));
        }
        return maxValue;
      };
      void setValue(const Obj<patch_label_type>& label, const point_type& point, const int value) {
        label->setCone(value, point);
      };
      const Obj<patch_label_type>& getLabel(const patch_type& patch, const std::string& name) {
        return this->_labels[name][patch];
      };
      const Obj<label_sequence>& getLabelStratum(const patch_type& patch, const std::string& name, int label) {
        return this->_labels[name][patch]->support(label);
      };
      const sheaf_type& getPatches() {
        return this->_sheaf;
      };
    public:
      template<class InputPoints>
      void computeHeight(const Obj<patch_label_type>& height, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points) {
        Obj<typename std::set<point_type> > modifiedPoints = new typename std::set<point_type>();

        for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          // Compute the max height of the points in the support of p, and add 1
          int h0 = this->getValue(height, *p_iter, -1);
          int h1 = this->getMaxValue(height, sieve->support(*p_iter), -1) + 1;

          if(h1 != h0) {
            this->setValue(height, *p_iter, h1);
            if (h1 > this->_maxHeight) this->_maxHeight = h1;
            modifiedPoints->insert(*p_iter);
          }
        }
        // FIX: We would like to avoid the copy here with cone()
        if(modifiedPoints->size() > 0) {
          this->computeHeight(height, sieve, sieve->cone(modifiedPoints));
        }
      };
      void computeHeights() {
        const std::string name("height");

        this->_maxHeight = -1;
        for(typename sheaf_type::iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
          Obj<patch_label_type> label = new patch_label_type(this->comm(), this->debug());

          this->computeHeight(label, s_iter->second, s_iter->second->leaves());
          this->_labels[name][s_iter->first] = label;
        }
      };
      int height() {return this->_maxHeight;};
      const Obj<label_sequence>& heightStratum(const patch_type& patch, int height) {
        return this->getLabelStratum(patch, "height", height);
      };
      template<class InputPoints>
      void computeDepth(const Obj<patch_label_type>& depth, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points) {
        Obj<typename std::set<point_type> > modifiedPoints = new typename std::set<point_type>();

        for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          // Compute the max depth of the points in the cone of p, and add 1
          int d0 = this->getValue(depth, *p_iter, -1);
          int d1 = this->getMaxValue(depth, sieve->cone(*p_iter), -1) + 1;

          if(d1 != d0) {
            this->setValue(depth, *p_iter, d1);
            if (d1 > this->_maxDepth) this->_maxDepth = d1;
            modifiedPoints->insert(*p_iter);
          }
        }
        // FIX: We would like to avoid the copy here with support()
        if(modifiedPoints->size() > 0) {
          this->computeDepth(depth, sieve, sieve->support(modifiedPoints));
        }
      };
      void computeDepths() {
        const std::string name("depth");

        this->_maxDepth = -1;
        for(typename sheaf_type::iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
          Obj<patch_label_type> label = new patch_label_type(this->comm(), this->debug());

          this->computeDepth(label, s_iter->second, s_iter->second->roots());
          this->_labels[name][s_iter->first] = label;
        }
      };
      int depth() {return this->_maxDepth;};
      const Obj<label_sequence>& depthStratum(const patch_type& patch, int depth) {
        return this->getLabelStratum(patch, "depth", depth);
      };
      void stratify() {
        this->computeHeights();
        this->computeDepths();
      };
    };

    template<typename Topology_, typename Index_>
    class Atlas : public ALE::ParallelObject {
    public:
      typedef Topology_                                 topology_type;
      typedef typename topology_type::patch_type        patch_type;
      typedef typename topology_type::sieve_type        sieve_type;
      typedef typename topology_type::point_type        point_type;
      typedef Index_                                    index_type;
      typedef std::vector<index_type>                   IndexArray;
      typedef typename std::map<point_type, index_type> chart_type;
      typedef typename std::map<patch_type, chart_type> indices_type;
    protected:
      Obj<topology_type> _topology;
      indices_type       _indices;
      Obj<IndexArray>    _array;
    public:
      Atlas(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
        this->_topology = new topology_type(comm, debug);
        this->_array    = new IndexArray();
      };
      Atlas(const Obj<topology_type>& topology) : ParallelObject(topology->comm(), topology->debug()), _topology(topology) {
        this->_array = new IndexArray();
      };
    public: // Accessors
      const Obj<topology_type>& getTopology() {return this->_topology;};
    public: // Verifiers
      void checkPatch(const patch_type& patch) {
        this->_topology->checkPatch(patch);
        if (this->_indices.find(patch) == this->_indices.end()) {
          ostringstream msg;
          msg << "Invalid atlas patch: " << patch << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
    public: // Sizes
      void setFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        this->_indices[patch][p].prefix = -1;
        this->_indices[patch][p].index  = dim;
      };
      void setFiberDimensionByDepth(const patch_type& patch, int depth, int dim) {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->depthStratum(patch, depth);

        for(typename topology_type::label_sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(patch, *p_iter, dim);
        }
      };
      void setFiberDimensionByHeight(const patch_type& patch, int height, int dim) {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->heightStratum(patch, height);

        for(typename topology_type::label_sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(patch, *p_iter, dim);
        }
      };
      int size(const patch_type& patch) {
        typename chart_type::iterator end = this->_indices[patch].end();
        int size = 0;

        for(typename chart_type::iterator c_iter = this->_indices[patch].begin(); c_iter != end; ++c_iter) {
          size += c_iter->second.index;
        }
        return size;
      };
      int size(const patch_type& patch, const point_type& p) {
        this->checkPatch(patch);
        Obj<typename sieve_type::coneSet>  closure = this->_topology->getPatch(patch)->closure(p);
        typename sieve_type::coneSet::iterator end = closure->end();
        int size = 0;

        for(typename sieve_type::coneSet::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
          size += this->_indices[patch][*c_iter].index;
        }
        return size;
      };
      void orderPoint(chart_type& chart, const Obj<sieve_type>& sieve, const point_type& point, int& offset) {
        const Obj<typename sieve_type::coneSequence>& cone = sieve->cone(point);
        typename sieve_type::coneSequence::iterator end = cone->end();

        if (chart[point].prefix < 0) {
          for(typename sieve_type::coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
            std::cout << "    Recursing to " << *c_iter << std::endl;
            this->orderPoint(chart, sieve, *c_iter, offset);
          }
          std::cout << "  Ordering point " << point << " at " << offset << std::endl;
          chart[point].prefix = offset;
          offset += chart[point].index;
        }
      }
      void orderPatch(const patch_type& patch, int& offset) {
        chart_type& chart = this->_indices[patch];

        for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
          std::cout << "Ordering closure of point " << p_iter->first << std::endl;
          this->orderPoint(chart, this->_topology->getPatch(patch), p_iter->first, offset);
        }
      };
      void orderPatches() {
        int offset = 0;

        for(typename indices_type::iterator i_iter = this->_indices.begin(); i_iter != this->_indices.end(); ++i_iter) {
          std::cout << "Ordering patch " << i_iter->first << std::endl;
          this->orderPatch(i_iter->first, offset);
        }
      };
      void clearIndices() {
        for(typename indices_type::iterator i_iter = this->_indices.begin(); i_iter != this->_indices.end(); ++i_iter) {
          this->_indices[i_iter->first].clear();
        }
      };
    public: // Index retrieval
      const index_type& getIndex(const patch_type& patch, const point_type& p) {
        this->checkPatch(patch);
        return this->_indices[patch][p];
      };
      // Want to return a sequence
      const Obj<IndexArray>& getIndices(const patch_type& patch, const point_type& p, const int level = -1) {
        this->_array->clear();

        if (level == 0) {
          this->_array->push_back(this->getIndex(patch, p));
        } else if ((level == 1) || (this->_topology->getPatch(patch)->height() == 1)) {
          const Obj<typename sieve_type::coneSequence>& cone = this->_topology->getPatch(patch)->cone(p);

          this->_array->push_back(this->getIndex(patch, p));
          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            this->_array->push_back(this->getIndex(patch, *p_iter));
          }
        } else if (level == -1) {
          Obj<typename sieve_type::coneSet> closure = this->_topology->getPatch(patch)->closure(p);

          for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != closure->end(); ++p_iter) {
            this->_array->push_back(this->getIndex(patch, *p_iter));
          }
        } else {
          Obj<typename sieve_type::coneArray> cone = this->_topology->getPatch(patch)->nCone(p, level);

          for(typename sieve_type::coneArray::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            this->_array->push_back(this->getIndex(patch, *p_iter));
          }
        }
        return this->_array;
      };
      const chart_type& getChart(const patch_type& patch) {
        return this->_indices[patch];
      }
    };

    template<typename Atlas_, typename Value_>
    class Section : public ALE::ParallelObject {
    public:
      typedef Atlas_                             atlas_type;
      typedef typename atlas_type::patch_type    patch_type;
      typedef typename atlas_type::sieve_type    sieve_type;
      typedef typename atlas_type::point_type    point_type;
      typedef typename atlas_type::index_type    index_type;
      typedef Value_                             value_type;
      typedef std::map<patch_type, value_type *> values_type;
    protected:
      Obj<atlas_type> _atlas;
      values_type     _arrays;
    public:
      Section(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
        this->_atlas = new atlas_type(comm, debug);
      };
      Section(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas) {};
      virtual ~Section() {
        for(typename values_type::iterator a_iter = this->_arrays.begin(); a_iter != this->_arrays.end(); ++a_iter) {
          delete [] a_iter->second;
          a_iter->second = NULL;
        }
      };
    public: // Verifiers
      void checkPatch(const patch_type& patch) {
        this->_atlas->checkPatch(patch);
        if (this->_arrays.find(patch) == this->_arrays.end()) {
          ostringstream msg;
          msg << "Invalid section patch: " << patch << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
    public: // Accessors
      const Obj<atlas_type>& getAtlas() {return this->_atlas;};
    public:
      void allocate() {
        const typename atlas_type::topology_type::sheaf_type& patches = this->_atlas->getTopology()->getPatches();

        for(typename atlas_type::topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          this->_arrays[p_iter->first] = new value_type[this->_atlas->size(p_iter->first)];
          PetscMemzero(this->_arrays[p_iter->first], this->_atlas->size(p_iter->first) * sizeof(value_type));
        }
      };
      // Return a smart pointer?
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        this->checkPatch(patch);
        const value_type  *a      = this->_arrays[patch];
        static value_type *values = NULL;
        static int         vSize  = 0;
        int                size   = this->_atlas->size(patch, p);

        if (size != vSize) {
          vSize = size;
          if (values) delete [] values;
          values = new value_type[vSize];
        };
        if (this->_atlas->getTopology()->getPatch(patch)->height() == 1) {
          // Only avoids the copy
          const Obj<typename sieve_type::coneSequence>& cone = this->_atlas->getTopology()->getPatch(patch)->cone(p);
          const index_type&                             pInd = this->_atlas->getIndex(patch, p);
          int                                           j    = -1;

          for(int i = pInd.prefix; i < pInd.prefix + pInd.index; ++i) {
            values[++j] = a[i];
          }
          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            const index_type& ind    = this->_atlas->getIndex(patch, *p_iter);
            const int         start  = ind.prefix;
            const int         length = ind.index;

            for(int i = start; i < start + length; ++i) {
              values[++j] = a[i];
            }
          }
        } else {
          const Obj<typename atlas_type::IndexArray>& ind = this->_atlas->getIndices(patch, p);
          int                                         j   = -1;

          for(typename atlas_type::IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int start  = i_iter->prefix;
            const int length = i_iter->index;

            for(int i = start; i < start + length; ++i) {
              values[++j] = a[i];
            }
          }
        }
        return values;
      };
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        this->checkPatch(patch);
        // Using the index structure explicitly
        return &(this->_arrays[patch][this->_atlas->getIndex(patch, p).prefix]);
      };
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        value_type *a = this->_arrays[patch];

        if (this->_atlas->getTopology()->getPatch(patch)->height() == 1) {
          // Only avoids the copy
          const Obj<typename sieve_type::coneSequence>& cone = this->_atlas->getTopology()->getPatch(patch)->cone(p);
          const index_type&                             pInd = this->_atlas->getIndex(patch, p);
          int                                           j    = -1;

          for(int i = pInd.prefix; i < pInd.prefix + pInd.index; ++i) {
            a[i] = v[++j];
          }
          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            const index_type& ind    = this->_atlas->getIndex(patch, *p_iter);
            const int         start  = ind.prefix;
            const int         length = ind.index;

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
            }
          }
        } else {
          const Obj<typename atlas_type::IndexArray>& ind = this->_atlas->getIndices(patch, p);
          int                                         j   = -1;

          for(typename atlas_type::IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int start  = i_iter->prefix;
            const int length = i_iter->index;

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
            }
          }
        }
      };
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        value_type *a = this->_arrays[patch];

        if (this->_atlas->getTopology()->getPatch(patch)->height() == 1) {
          // Only avoids the copy
          const Obj<typename sieve_type::coneSequence>& cone = this->_atlas->getTopology()->getPatch(patch)->cone(p);
          const index_type&                             pInd = this->_atlas->getIndex(patch, p);
          int                                           j    = -1;

          for(int i = pInd.prefix; i < pInd.prefix + pInd.index; ++i) {
            a[i] += v[++j];
          }
          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            const index_type& ind    = this->_atlas->getIndex(patch, *p_iter);
            const int         start  = ind.prefix;
            const int         length = ind.index;

            for(int i = start; i < start + length; ++i) {
              a[i] += v[++j];
            }
          }
        } else {
          const Obj<typename atlas_type::IndexArray>& ind = this->_atlas->getIndices(patch, p);
          int                                         j   = -1;

          for(typename atlas_type::IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int start  = i_iter->prefix;
            const int length = i_iter->index;

            for(int i = start; i < start + length; ++i) {
              a[i] += v[++j];
            }
          }
        }
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        const index_type& ind = this->_atlas->getIndex(patch, p);
        value_type       *a   = &(this->_arrays[patch][ind.first]);

        // Using the index structure explicitly
        for(int i = 0; i < ind.second; ++i) {
          a[i] = v[i];
        }
      };
    public:
      void view(const std::string& name) const {
        ostringstream txt;

        if (name == "") {
          if(this->commRank() == 0) {
            txt << "viewing a Section" << std::endl;
          }
        } else {
          if(this->commRank() == 0) {
            txt << "viewing Section '" << name << "'" << std::endl;
          }
        }
        for(typename values_type::const_iterator a_iter = this->_arrays.begin(); a_iter != this->_arrays.end(); ++a_iter) {
          const patch_type  patch = a_iter->first;
          const value_type *array = a_iter->second;

          txt << "[" << this->commRank() << "]: Patch " << patch << std::endl;
          const typename atlas_type::chart_type& chart = this->_atlas->getChart(patch);

          for(typename atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const typename atlas_type::index_type& idx = c_iter->second;

            if (idx.index != 0) {
              txt << "[" << this->commRank() << "]:   " << c_iter->first << " dim " << idx.index << " offset " << idx.prefix << "  ";
              for(int i = 0; i < idx.index; i++) {
                txt << " " << array[idx.prefix+i];
              }
              txt << std::endl;
            }
          }
        }
        PetscSynchronizedPrintf(this->comm(), txt.str().c_str());
        PetscSynchronizedFlush(this->comm());
      };
    };
  }
}

#endif
