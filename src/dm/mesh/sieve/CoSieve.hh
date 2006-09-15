#ifndef included_ALE_CoSieve_hh
#define included_ALE_CoSieve_hh

#ifndef  included_ALE_Sieve_hh
#include <Sieve.hh>
#endif

extern "C" PetscMPIInt Petsc_DelTag(MPI_Comm comm,PetscMPIInt keyval,void* attr_val,void* extra_state);

#ifdef PETSC_HAVE_CHACO
/* Chaco does not have an include file */
extern "C" {
  extern int interface(int nvtxs, int *start, int *adjacency, int *vwgts,
                       float *ewgts, float *x, float *y, float *z, char *outassignname,
                       char *outfilename, short *assignment, int architecture, int ndims_tot,
                       int mesh_dims[3], double *goal, int global_method, int local_method,
                       int rqi_flag, int vmax, int ndims, double eigtol, long seed);

  extern int FREE_GRAPH;
}
#endif
#ifdef PETSC_HAVE_CHACO
extern "C" {
  extern void METIS_PartGraphKway(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *);
}
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
      static void buildHexFaces(Obj<sieve_type> sieve, int dim, std::map<int, int*>& curElement, std::map<int,PointArray>& bdVertices, std::map<int,PointArray>& faces, typename sieve_type::point_type& cell) {
        int debug = sieve->debug;

        if (debug > 1) {std::cout << "  Building hex faces for boundary of " << cell << " (size " << bdVertices[dim].size() << "), dim " << dim << std::endl;}
        if (dim > 3) {
          throw ALE::Exception("Cannot do hexes of dimension greater than three");
        } else if (dim > 2) {
          int nodes[24] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
                           1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7};

          for(int b = 0; b < 6; b++) {
            typename sieve_type::point_type face;

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

            for(int c = 0; c < 2; c++) {
              bdVertices[dim-1].push_back(bdVertices[dim][(b+c)%boundarySize]);
            }
            if (debug > 1) {std::cout << "    boundary point " << bdVertices[dim][b] << std::endl;}
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
        int debug = sieve->debug;

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
          typename sieve_type::point_type cell(c);

          // Build the cell
          if (interpolate) {
            bdVertices[dim].clear();
            for(int b = 0; b < corners; b++) {
              typename sieve_type::point_type vertex(cells[c*corners+b]+numCells);

              if (debug > 1) {std::cout << "Adding boundary vertex " << vertex << std::endl;}
              bdVertices[dim].push_back(vertex);
            }
            if (debug) {std::cout << "cell " << cell << " num boundary vertices " << bdVertices[dim].size() << std::endl;}

            if (corners != dim+1) {
              buildHexFaces(sieve, dim, curElement, bdVertices, faces, cell);
            } else {
              buildFaces(sieve, dim, curElement, bdVertices, faces, cell);
            }
          } else {
            for(int b = 0; b < corners; b++) {
              sieve->addArrow(typename sieve_type::point_type(cells[c*corners+b]+numCells), cell, b);
            }
            if (debug) {
              if (debug > 1) {
                for(int b = 0; b < corners; b++) {
                  std::cout << "  Adding vertex " << typename sieve_type::point_type(cells[c*corners+b]+numCells) << std::endl;
                }
              }
              std::cout << "Adding cell " << cell << " dim " << dim << std::endl;
            }
          }
        }
        ALE_LOG_EVENT_END;
      };
      template<typename Section>
      static void buildCoordinates(const Obj<Section>& coords, const int embedDim, const double coordinates[]) {
        const typename Section::patch_type                          patch    = 0;
        const Obj<typename Section::topology_type::label_sequence>& vertices = coords->getTopology()->depthStratum(patch, 0);
        const int numCells = coords->getTopology()->heightStratum(patch, 0)->size();

        coords->setFiberDimensionByDepth(patch, 0, embedDim);
        coords->allocate();
        for(typename Section::topology_type::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
          coords->update(patch, *v_iter, &(coordinates[(*v_iter - numCells)*embedDim]));
        }
      };
    };

    // A Topology is a collection of Sieves
    //   Each Sieve has a label, which we call a \emph{patch}
    //   The collection itself we call a \emph{sheaf}
    //   The main operation we provide in Topology is the creation of a \emph{label}
    //     A label is a bidirectional mapping of Sieve points to integers, implemented with a Sifter
    template<typename Patch_, typename Sieve_>
    class Topology : public ALE::ParallelObject {
    public:
      typedef Patch_                                                patch_type;
      typedef Sieve_                                                sieve_type;
      typedef typename sieve_type::point_type                       point_type;
      typedef typename std::map<patch_type, Obj<sieve_type> >       sheaf_type;
      typedef typename ALE::Sifter<int, point_type, int>            patch_label_type;
      typedef typename std::map<patch_type, Obj<patch_label_type> > label_type;
      typedef typename std::map<patch_type, int>                    max_label_type;
      typedef typename std::map<const std::string, label_type>      labels_type;
      typedef typename patch_label_type::supportSequence            label_sequence;
      typedef typename std::set<point_type>                         point_set_type;
    protected:
      sheaf_type     _sheaf;
      labels_type    _labels;
      int            _maxHeight;
      max_label_type _maxHeights;
      int            _maxDepth;
      max_label_type _maxDepths;
      // Work space
      Obj<point_set_type> _modifiedPoints;
    public:
      Topology(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _maxHeight(-1), _maxDepth(-1) {
        this->_modifiedPoints = new point_set_type();
      };
    public: // Verifiers
      void checkPatch(const patch_type& patch) {
        if (this->_sheaf.find(patch) == this->_sheaf.end()) {
          ostringstream msg;
          msg << "Invalid topology patch: " << patch << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
      void checkLabel(const std::string& name, const patch_type& patch) {
        this->checkPatch(patch);
        if ((this->_labels.find(name) == this->_labels.end()) || (this->_labels[name].find(patch) == this->_labels[name].end())) {
          ostringstream msg;
          msg << "Invalid label name: " << name << " for patch " << patch << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
      bool hasLabel(const std::string& name, const patch_type& patch) {
        if ((this->_labels.find(name) != this->_labels.end()) && (this->_labels[name].find(patch) != this->_labels[name].end())) {
          return true;
        }
        return false;
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
      const Obj<patch_label_type>& createLabel(const patch_type& patch, const std::string& name) {
        this->checkPatch(patch);
        if ((this->_labels.find(name) == this->_labels.end()) || (this->_labels[name].find(patch) == this->_labels[name].end())) {
          this->_labels[name][patch] = new patch_label_type(this->comm(), this->debug());
        }
        return this->_labels[name][patch];
      };
      const Obj<patch_label_type>& getLabel(const patch_type& patch, const std::string& name) {
        this->checkLabel(name, patch);
        return this->_labels[name][patch];
      };
      const Obj<label_sequence>& getLabelStratum(const patch_type& patch, const std::string& name, int label) {
        this->checkLabel(name, patch);
        return this->_labels[name][patch]->support(label);
      };
      const sheaf_type& getPatches() {
        return this->_sheaf;
      };
      const labels_type& getLabels() {
        return this->_sheaf;
      };
      void clear() {
        this->_sheaf.clear();
        this->_labels.clear();
        this->_maxHeight = -1;
        this->_maxHeights.clear();
        this->_maxDepth = -1;
        this->_maxDepths.clear();
      };
    public:
      template<class InputPoints>
      void computeHeight(const Obj<patch_label_type>& height, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points, int& maxHeight) {
        this->_modifiedPoints->clear();

        for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          // Compute the max height of the points in the support of p, and add 1
          int h0 = this->getValue(height, *p_iter, -1);
          int h1 = this->getMaxValue(height, sieve->support(*p_iter), -1) + 1;

          if(h1 != h0) {
            this->setValue(height, *p_iter, h1);
            if (h1 > maxHeight) maxHeight = h1;
            this->_modifiedPoints->insert(*p_iter);
          }
        }
        // FIX: We would like to avoid the copy here with cone()
        if(this->_modifiedPoints->size() > 0) {
          this->computeHeight(height, sieve, sieve->cone(this->_modifiedPoints), maxHeight);
        }
      };
      void computeHeights() {
        const std::string name("height");

        this->_maxHeight = -1;
        for(typename sheaf_type::iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
          const Obj<patch_label_type>& label = this->createLabel(s_iter->first, name);

          this->_maxHeights[s_iter->first] = -1;
          this->computeHeight(label, s_iter->second, s_iter->second->leaves(), this->_maxHeights[s_iter->first]);
          if (this->_maxHeights[s_iter->first] > this->_maxHeight) this->_maxHeight = this->_maxHeights[s_iter->first];
        }
      };
      int height() {return this->_maxHeight;};
      int height(const patch_type& patch) {
        this->checkPatch(patch);
        return this->_maxHeights[patch];
      };
      int height(const patch_type& patch, const point_type& point) {
        return this->getValue(this->_labels["height"][patch], point, -1);
      };
      const Obj<label_sequence>& heightStratum(const patch_type& patch, int height) {
        return this->getLabelStratum(patch, "height", height);
      };
      template<class InputPoints>
      void computeDepth(const Obj<patch_label_type>& depth, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points, int& maxDepth) {
        this->_modifiedPoints->clear();

        for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          // Compute the max depth of the points in the cone of p, and add 1
          int d0 = this->getValue(depth, *p_iter, -1);
          int d1 = this->getMaxValue(depth, sieve->cone(*p_iter), -1) + 1;

          if(d1 != d0) {
            this->setValue(depth, *p_iter, d1);
            if (d1 > maxDepth) maxDepth = d1;
            this->_modifiedPoints->insert(*p_iter);
          }
        }
        // FIX: We would like to avoid the copy here with support()
        if(this->_modifiedPoints->size() > 0) {
          this->computeDepth(depth, sieve, sieve->support(this->_modifiedPoints), maxDepth);
        }
      };
      void computeDepths() {
        const std::string name("depth");

        this->_maxDepth = -1;
        for(typename sheaf_type::iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
          const Obj<patch_label_type>& label = this->createLabel(s_iter->first, name);

          this->_maxDepths[s_iter->first] = -1;
          this->computeDepth(label, s_iter->second, s_iter->second->roots(), this->_maxDepths[s_iter->first]);
          if (this->_maxDepths[s_iter->first] > this->_maxDepth) this->_maxDepth = this->_maxDepths[s_iter->first];
        }
      };
      int depth() {return this->_maxDepth;};
      int depth(const patch_type& patch) {
        this->checkPatch(patch);
        return this->_maxDepths[patch];
      };
      int depth(const patch_type& patch, const point_type& point) {
        return this->getValue(this->_labels["depth"][patch], point, -1);
      };
      const Obj<label_sequence>& depthStratum(const patch_type& patch, int depth) {
        return this->getLabelStratum(patch, "depth", depth);
      };
      #undef __FUNCT__
      #define __FUNCT__ "Topology::stratify"
      void stratify() {
        ALE_LOG_EVENT_BEGIN;
        this->computeHeights();
        this->computeDepths();
        ALE_LOG_EVENT_END;
      };
    public: // Viewers
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        if (comm == MPI_COMM_NULL) {
          comm = this->comm();
        }
        if (name == "") {
          PetscPrintf(comm, "viewing a Topology\n");
        } else {
          PetscPrintf(comm, "viewing Topology '%s'\n", name.c_str());
        }
        for(typename sheaf_type::const_iterator s_iter = this->_sheaf.begin(); s_iter != this->_sheaf.end(); ++s_iter) {
          ostringstream txt;

          txt << "Patch " << s_iter->first;
          s_iter->second->view(txt.str().c_str(), comm);
        }
        for(typename labels_type::const_iterator l_iter = this->_labels.begin(); l_iter != this->_labels.end(); ++l_iter) {
          PetscPrintf(comm, "  label %s constructed\n", l_iter->first.c_str());
        }
      };
    };

    template<typename Topology_>
    class Partitioner {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::point_type point_type;
    public:
      #undef __FUNCT__
      #define __FUNCT__ "buildDualCSR"
      // This creates a CSR representation of the adjacency matrix for cells
      static void buildDualCSR(const Obj<topology_type>& topology, const int dim, const patch_type& patch, int **offsets, int **adjacency) {
        ALE_LOG_EVENT_BEGIN;
        const Obj<sieve_type>&                             sieve    = topology->getPatch(patch);
        const Obj<typename topology_type::label_sequence>& elements = topology->heightStratum(patch, 0);
        int numElements = elements->size();
        int corners     = sieve->cone(*elements->begin())->size();
        int *off        = new int[numElements+1];

        std::set<point_type> *neighborCells = new std::set<point_type>[numElements];
        int faceVertices = -1;

        if (topology->depth(patch) != 1) {
          throw ALE::Exception("Not yet implemented for interpolated meshes");
        }
        if (corners == dim+1) {
          faceVertices = dim;
        } else if ((dim == 2) && (corners == 4)) {
          faceVertices = 2;
        } else if ((dim == 3) && (corners == 8)) {
          faceVertices = 4;
        } else {
          throw ALE::Exception("Could not determine number of face vertices");
        }
        for(typename topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
          const Obj<typename sieve_type::traits::coneSequence>& vertices  = sieve->cone(*e_iter);
          typename sieve_type::traits::coneSequence::iterator vEnd = vertices->end();

          for(typename sieve_type::traits::coneSequence::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
            const Obj<typename sieve_type::traits::supportSequence>& neighbors = sieve->support(*v_iter);
            typename sieve_type::traits::supportSequence::iterator nEnd = neighbors->end();

            for(typename sieve_type::traits::supportSequence::iterator n_iter = neighbors->begin(); n_iter != nEnd; ++n_iter) {
              if (*e_iter == *n_iter) continue;
              if ((int) sieve->meet(*e_iter, *n_iter)->size() == faceVertices) {
                neighborCells[*e_iter].insert(*n_iter);
              }
            }
          }
        }
        off[0] = 0;
        for(int e = 1; e <= numElements; e++) {
          off[e] = neighborCells[e-1].size() + off[e-1];
        }
        int *adj    = new int[off[numElements]];
        int  offset = 0;
        for(int e = 0; e < numElements; e++) {
          for(typename std::set<point_type>::iterator n_iter = neighborCells[e].begin(); n_iter != neighborCells[e].end(); ++n_iter) {
            adj[offset++] = *n_iter;
          }
        }
        delete [] neighborCells;
        if (offset != off[numElements]) {
          ostringstream msg;
          msg << "ERROR: Total number of neighbors " << offset << " does not match the offset array " << off[numElements];
          throw ALE::Exception(msg.str().c_str());
        }
        *offsets   = off;
        *adjacency = adj;
        ALE_LOG_EVENT_END;
      };
    };
#ifdef PETSC_HAVE_CHACO
    namespace Chaco {
      template<typename Topology_>
      class Partitioner {
      public:
        typedef Topology_                          topology_type;
        typedef typename topology_type::sieve_type sieve_type;
        typedef typename topology_type::patch_type patch_type;
        typedef typename topology_type::point_type point_type;
        typedef short int                          part_type;
      public:
        #undef __FUNCT__
        #define __FUNCT__ "ChacoPartitionSieve"
        static part_type *partitionSieve(const Obj<topology_type>& topology, const int dim) {
          part_type *assignment = NULL; /* set number of each vtx (length n) */

          ALE_LOG_EVENT_BEGIN;
          if (topology->commRank() == 0) {
            /* arguments for Chaco library */
            FREE_GRAPH = 0;                         /* Do not let Chaco free my memory */
            int nvtxs;                              /* number of vertices in full graph */
            int *start;                             /* start of edge list for each vertex */
            int *adjacency;                         /* = adj -> j; edge list data  */
            int *vwgts = NULL;                      /* weights for all vertices */
            float *ewgts = NULL;                    /* weights for all edges */
            float *x = NULL, *y = NULL, *z = NULL;  /* coordinates for inertial method */
            char *outassignname = NULL;             /*  name of assignment output file */
            char *outfilename = NULL;               /* output file name */
            int architecture = 1;                   /* 0 => hypercube, d => d-dimensional mesh */
            int ndims_tot = 0;                      /* total number of cube dimensions to divide */
            int mesh_dims[3];                       /* dimensions of mesh of processors */
            double *goal = NULL;                    /* desired set sizes for each set */
            int global_method = 1;                  /* global partitioning algorithm */
            int local_method = 1;                   /* local partitioning algorithm */
            int rqi_flag = 0;                       /* should I use RQI/Symmlq eigensolver? */
            int vmax = 200;                         /* how many vertices to coarsen down to? */
            int ndims = 1;                          /* number of eigenvectors (2^d sets) */
            double eigtol = 0.001;                  /* tolerance on eigenvectors */
            long seed = 123636512;                  /* for random graph mutations */
            int patch = 0;
            PetscErrorCode ierr;

            nvtxs = topology->heightStratum(patch, 0)->size();
            mesh_dims[0] = topology->commSize(); mesh_dims[1] = 1; mesh_dims[2] = 1;
            ALE::New::Partitioner<topology_type>::buildDualCSR(topology, dim, patch, &start, &adjacency);
            for(int e = 0; e < start[nvtxs]; e++) {
              adjacency[e]++;
            }
            assignment = new part_type[nvtxs];
            ierr = PetscMemzero(assignment, nvtxs * sizeof(part_type));CHKERROR(ierr, "Error in PetscMemzero");

            /* redirect output to buffer: chaco -> msgLog */
#ifdef PETSC_HAVE_UNISTD_H
            char *msgLog;
            int fd_stdout, fd_pipe[2], count;

            fd_stdout = dup(1);
            pipe(fd_pipe);
            close(1);
            dup2(fd_pipe[1], 1);
            msgLog = new char[16284];
#endif

            ierr = interface(nvtxs, start, adjacency, vwgts, ewgts, x, y, z,
                             outassignname, outfilename, assignment, architecture, ndims_tot,
                             mesh_dims, goal, global_method, local_method, rqi_flag, vmax, ndims,
                             eigtol, seed);

#ifdef PETSC_HAVE_UNISTD_H
            int SIZE_LOG  = 10000;

            fflush(stdout);
            count = read(fd_pipe[0], msgLog, (SIZE_LOG - 1) * sizeof(char));
            if (count < 0) count = 0;
            msgLog[count] = 0;
            close(1);
            dup2(fd_stdout, 1);
            close(fd_stdout);
            close(fd_pipe[0]);
            close(fd_pipe[1]);
            if (topology->debug()) {
              std::cout << msgLog << std::endl;
            }
            delete [] msgLog;
#endif
            delete [] adjacency;
            delete [] start;
          }
          ALE_LOG_EVENT_END;
          return assignment;
        };
      };
    };
#endif
#ifdef PETSC_HAVE_PARMETIS
    namespace ParMetis {
      template<typename Topology_>
      class Partitioner {
      public:
        typedef Topology_                          topology_type;
        typedef typename topology_type::sieve_type sieve_type;
        typedef typename topology_type::patch_type patch_type;
        typedef typename topology_type::point_type point_type;
        typedef int                                part_type;
      public:
        #undef __FUNCT__
        #define __FUNCT__ "ParMetisPartitionSieve"
        static part_type *partitionSieve(const Obj<topology_type>& topology, const int dim) {
          int    nvtxs;      // The number of vertices in full graph
          int   *xadj;       // Start of edge list for each vertex
          int   *adjncy;     // Edge lists for all vertices
          int   *vwgt;       // Vertex weights
          int   *adjwgt;     // Edge weights
          int    wgtflag;    // Indicates which weights are present
          int    numflag;    // Indicates initial offset (0 or 1)
          int    nparts;     // The number of partitions
          int    options[5]; // Options
          // Outputs
          int    edgeCut;    // The number of edges cut by the partition
          int   *assignment; // The vertex partition
          const typename topology_type::patch_type patch = 0;

          if (topology->commRank() == 0) {
            nvtxs = topology->heightStratum(patch, 0)->size();
            vwgt       = NULL;
            adjwgt     = NULL;
            wgtflag    = 0;
            numflag    = 0;
            nparts     = topology->commSize();
            options[0] = 0; // Use all defaults
            assignment = new part_type[nvtxs];
            if (topology->commSize() == 1) {
              PetscMemzero(assignment, nvtxs * sizeof(part_type));
            } else {
              ALE::New::Partitioner<topology_type>::buildDualCSR(topology, dim, patch, &xadj, &adjncy);
              METIS_PartGraphKway(&nvtxs, xadj, adjncy, vwgt, adjwgt, &wgtflag, &numflag, &nparts, options, &edgeCut, assignment);
              delete [] xadj;
              delete [] adjncy;
            }
          }
          return assignment;
        };
      };
    };
#endif

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
      const Obj<topology_type>& getTopology() const {return this->_topology;};
      void setTopology(const Obj<topology_type>& topology) {this->_topology = topology;};
      void copy(const Obj<Atlas>& atlas) {
        const typename topology_type::sheaf_type& sheaf = atlas->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator s_iter = sheaf.begin(); s_iter != sheaf.end(); ++s_iter) {
          const chart_type& chart = atlas->getChart(s_iter->first);

          for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            this->setFiberDimension(s_iter->first, c_iter->first, c_iter->second.index);
          }
        }
      };
      void copyByDepth(const Obj<Atlas>& atlas) {
        this->copyByDepth(atlas, atlas->getTopology());
      };
      template<typename AtlasType, typename TopologyType>
      void copyByDepth(const Obj<AtlasType>& atlas, const Obj<TopologyType>& topology) {
        const typename topology_type::sheaf_type& patches  = topology->getPatches();
        bool *depths = new bool[topology->depth()+1];

        for(int d = 0; d <= topology->depth(); d++) depths[d] = false;
        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const patch_type& patch = p_iter->first;
          const chart_type& chart = atlas->getChart(p_iter->first);

          for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const point_type& point = c_iter->first;
            const int         depth = topology->depth(patch, point);

            if (!depths[depth]) {
              this->setFiberDimensionByDepth(patch, depth, atlas->getFiberDimension(patch, point));
            }
          }
        }
        this->orderPatches();
      }
    public: // Verifiers
      void checkPatch(const patch_type& patch) {
        this->_topology->checkPatch(patch);
        if (this->_indices.find(patch) == this->_indices.end()) {
          ostringstream msg;
          msg << "Invalid atlas patch: " << patch << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
      bool hasPatch(const patch_type& patch) {
        if (this->_indices.find(patch) == this->_indices.end()) return false;
        return true;
      }
      void clear() {
        this->_indices.clear();
      };
    public: // Sizes
      int const getFiberDimension(const patch_type& patch, const point_type& p) {
        return this->_indices[patch][p].index;
      };
      void setFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        this->_indices[patch][p].prefix = -1;
        this->_indices[patch][p].index  = dim;
      };
      void addFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        if (this->hasPatch(patch) && (this->_indices[patch].find(p) != this->_indices[patch].end())) {
          this->_indices[patch][p].index += dim;
        } else {
          this->setFiberDimension(patch, p, dim);
        }
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
      void setFiberDimensionByLabel(const patch_type& patch, const std::string& label, int value, int dim) {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->getLabelStratum(patch, label, value);

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
            if (this->_debug > 1) {std::cout << "    Recursing to " << *c_iter << std::endl;}
            this->orderPoint(chart, sieve, *c_iter, offset);
          }
          if (this->_debug > 1) {std::cout << "  Ordering point " << point << " at " << offset << std::endl;}
          chart[point].prefix = offset;
          offset += chart[point].index;
        }
      }
      void orderPatch(const patch_type& patch, int& offset) {
        chart_type& chart = this->_indices[patch];

        for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
          if (this->_debug > 1) {std::cout << "Ordering closure of point " << p_iter->first << std::endl;}
          this->orderPoint(chart, this->_topology->getPatch(patch), p_iter->first, offset);
        }
      };
      void orderPatches() {
        for(typename indices_type::iterator i_iter = this->_indices.begin(); i_iter != this->_indices.end(); ++i_iter) {
          if (this->_debug > 1) {std::cout << "Ordering patch " << i_iter->first << std::endl;}
          int offset = 0;

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
      template<typename Numbering>
      const index_type getIndex(const patch_type& patch, const point_type& p, const Obj<Numbering>& numbering) {
        this->checkPatch(patch);
        return index_type(numbering->getIndex(p), this->_indices[patch][p].index);
      };
      // Want to return a sequence
      const Obj<IndexArray>& getIndices(const patch_type& patch, const point_type& p, const int level = -1) {
        this->_array->clear();

        if (level == 0) {
          this->_array->push_back(this->getIndex(patch, p));
        } else if ((level == 1) || (this->_topology->height(patch) == 1)) {
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
      template<typename Numbering>
      const Obj<IndexArray>& getIndices(const patch_type& patch, const point_type& p, const Obj<Numbering>& numbering, const int level = -1) {
        this->_array->clear();

        if (level == 0) {
          this->_array->push_back(this->getIndex(patch, p, numbering));
        } else if ((level == 1) || (this->_topology->height(patch) == 1)) {
          const Obj<typename sieve_type::coneSequence>& cone = this->_topology->getPatch(patch)->cone(p);

          this->_array->push_back(this->getIndex(patch, p, numbering));
          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            this->_array->push_back(this->getIndex(patch, *p_iter, numbering));
          }
        } else if (level == -1) {
          Obj<typename sieve_type::coneSet> closure = this->_topology->getPatch(patch)->closure(p);

          for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != closure->end(); ++p_iter) {
            this->_array->push_back(this->getIndex(patch, *p_iter, numbering));
          }
        } else {
          Obj<typename sieve_type::coneArray> cone = this->_topology->getPatch(patch)->nCone(p, level);

          for(typename sieve_type::coneArray::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            this->_array->push_back(this->getIndex(patch, *p_iter, numbering));
          }
        }
        return this->_array;
      };
      const chart_type& getChart(const patch_type& patch) {
        return this->_indices[patch];
      }
    };

    // An AbstractSection is a mapping from Sieve points to sets of values
    //   This is our presentation of a section of a fibre bundle,
    //     in which the Topology is the base space, and
    //     the value sets are vectors in the fiber spaces
    //   The main interface to values is through restrict() and update()
    //     This retrieval uses Sieve closure()
    //     We should call these rawRestrict/rawUpdate
    //   The Section must also be able to set/report the dimension of each fiber
    //     for which we use another section we call an \emph{Atlas}
    //     For some storage schemes, we also want offsets to go with these dimensions
    //   We must have some way of limiting the points associated with values
    //     so each section must support a getPatch() call returning the points with associated fibers
    //     I was using the Chart for this
    //   The Section must be able to participate in \emph{completion}
    //     This means restrict to a provided overlap, and exchange in the restricted sections
    //     Completion does not use hierarchy, so we see the Topology as a DiscreteTopology

    // A ConstantSection is the simplest Section
    //   All fibers are dimension 1
    //   All values are equal to a constant
    //     We need no value storage and no communication for completion
    template<typename Topology_, typename Value_>
    class NewConstantSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef std::set<point_type>               chart_type;
      typedef std::map<patch_type, chart_type>   atlas_type;
      typedef Value_                             value_type;
    protected:
      Obj<topology_type> _topology;
      atlas_type         _atlas;
      value_type         _value;
    public:
      NewConstantSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
        this->_topology = new topology_type(comm, debug);
      };
      NewConstantSection(const Obj<topology_type>& topology) : ParallelObject(topology->comm(), topology->debug()), _topology(topology) {};
      NewConstantSection(const Obj<topology_type>& topology, const value_type& value) : ParallelObject(topology->comm(), topology->debug()), _topology(topology), _value(value) {};
    public: // Verifiers
      void checkPatch(const patch_type& patch) const {
        this->_topology->checkPatch(patch);
        if (this->_atlas.find(patch) == this->_atlas.end()) {
          ostringstream msg;
          msg << "Invalid atlas patch " << patch << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
      void checkDimension(const int& dim) {
        if (dim != 1) {
          ostringstream msg;
          msg << "Invalid fiber dimension " << dim << " must be 1" << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
      bool hasPatch(const patch_type& patch) {
        if (this->_atlas.find(patch) == this->_atlas.end()) {
          return false;
        }
        return true;
      };
      bool hasPoint(const patch_type& patch, const point_type& point) {
        this->checkPatch(patch);
        return this->_atlas[patch].count(point) > 0;
      };
    public: // Accessors
      const Obj<topology_type>& getTopology() const {return this->_topology;};
      void setTopology(const Obj<topology_type>& topology) {this->_topology = topology;};
      const chart_type& getPatch(const patch_type& patch) {
        this->checkPatch(patch);
        return this->_atlas[patch];
      };
      void updatePatch(const patch_type& patch, const point_type& point) {
        this->_atlas[patch].insert(point);
      };
      template<typename Points>
      void updatePatch(const patch_type& patch, const Obj<Points>& points) {
        this->_atlas[patch].insert(points->begin(), points->end());
      };
    public: // Sizes
      void clear() {
        this->_atlas.clear(); 
      };
      int getFiberDimension(const patch_type& patch, const point_type& p) const {return 1;};
      void setFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        this->checkDimension(dim);
        this->updatePatch(patch, p);
      };
      void addFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        if (this->hasPatch(patch) && (this->_atlas[patch].find(p) != this->_atlas[patch].end())) {
          ostringstream msg;
          msg << "Invalid addition to fiber dimension " << dim << " cannot exceed 1" << std::endl;
          throw ALE::Exception(msg.str().c_str());
        } else {
          this->setFiberDimension(patch, p, dim);
        }
      };
      void setFiberDimensionByDepth(const patch_type& patch, int depth, int dim) {
        this->setFiberDimensionByLabel(patch, "depth", depth, dim);
      };
      void setFiberDimensionByHeight(const patch_type& patch, int height, int dim) {
        this->setFiberDimensionByLabel(patch, "height", height, dim);
      };
      void setFiberDimensionByLabel(const patch_type& patch, const std::string& label, int value, int dim) {
        const Obj<typename topology_type::label_sequence>& points = this->_topology->getLabelStratum(patch, label, value);

        for(typename topology_type::label_sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(patch, *p_iter, dim);
        }
      };
      int size(const patch_type& patch) {return this->_atlas[patch].size();};
      int size(const patch_type& patch, const point_type& p) {return 1;};
      void orderPatches() {};
    public: // Restriction
      const value_type *restrict(const patch_type& patch, const point_type& p) const {
        this->checkPatch(patch);
        return &this->_value;
      };
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) const {return this->restrict(patch, p);};
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        this->_value = v[0];
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {return this->update(patch, p, v);};
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        this->_value += v[0];
      };
      void updateAddPoint(const patch_type& patch, const point_type& p, const value_type v[]) {return this->updateAdd(patch, p, v);};
    public:
      void copy(const Obj<NewConstantSection>& section) {
        const typename topology_type::sheaf_type& patches = this->_topology->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const patch_type& patch = p_iter->first;
          if (!section->hasPatch(patch)) continue;
          const chart_type& chart = section->getPatch(patch);

          for(typename chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            this->updatePatch(patch, *c_iter);
          }
          this->_value = section->restrict(patch, *chart.begin())[0];
        }
      };
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        ostringstream txt;
        int rank;

        if (comm == MPI_COMM_NULL) {
          comm = this->comm();
          rank = this->commRank();
        } else {
          MPI_Comm_rank(comm, &rank);
        }
        if (name == "") {
          if(rank == 0) {
            txt << "viewing a ConstantSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing ConstantSection '" << name << "'" << std::endl;
          }
        }
        const typename topology_type::sheaf_type& sheaf = this->_topology->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = sheaf.begin(); p_iter != sheaf.end(); ++p_iter) {
          txt <<"["<<this->commRank()<<"]: Patch " << p_iter->first << std::endl;
          txt <<"["<<this->commRank()<<"]:   Value " << this->_value << std::endl;
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    // A UniformSection often acts as an Atlas
    //   All fibers are the same dimension
    //     Note we can use a ConstantSection for this Atlas
    //   Each point may have a different vector
    //     Thus we need storage for values, and hence must implement completion
    template<typename Topology_, typename Value_, int fiberDim = 1>
    class UniformSection : public ALE::ParallelObject {
    public:
      typedef Topology_                              topology_type;
      typedef typename topology_type::patch_type     patch_type;
      typedef typename topology_type::sieve_type     sieve_type;
      typedef typename topology_type::point_type     point_type;
      typedef NewConstantSection<topology_type, int> atlas_type;
      typedef typename atlas_type::chart_type        chart_type;
      typedef Value_                                 value_type;
      typedef struct {value_type v[fiberDim];}       fiber_type;
      typedef std::map<point_type, fiber_type>       array_type;
      typedef std::map<patch_type, array_type>       values_type;
    protected:
      Obj<atlas_type> _atlas;
      values_type     _arrays;
    public:
      UniformSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
        this->_atlas = new atlas_type(comm, debug);
      };
      UniformSection(const Obj<topology_type>& topology) : ParallelObject(topology->comm(), topology->debug()) {
        this->_atlas = new atlas_type(topology);
      };
      UniformSection(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas) {};
    protected:
      value_type *getRawArray(const int size) {
        static value_type *array   = NULL;
        static int         maxSize = 0;

        if (size > maxSize) {
          maxSize = size;
          if (array) delete [] array;
          array = new value_type[maxSize];
        };
        return array;
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
      bool hasPatch(const patch_type& patch) {
        return this->_atlas->hasPatch(patch);
      };
      bool hasPoint(const patch_type& patch, const point_type& point) {
        return this->_atlas->hasPoint(patch, point);
      };
      void checkDimension(const int& dim) {
        if (dim != fiberDim) {
          ostringstream msg;
          msg << "Invalid fiber dimension " << dim << " must be " << fiberDim << std::endl;
          throw ALE::Exception(msg.str().c_str());
        }
      };
    public: // Accessors
      const Obj<atlas_type>& getAtlas() {return this->_atlas;};
      void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
      const Obj<topology_type>& getTopology() {return this->_atlas->getTopology();};
      void setTopology(const Obj<topology_type>& topology) {this->_atlas->setTopology(topology);};
      const chart_type& getPatch(const patch_type& patch) {
        return this->_atlas->getPatch(patch);
      };
      void updatePatch(const patch_type& patch, const point_type& point) {
        this->setFiberDimension(patch, point, 1);
      };
      template<typename Points>
      void updatePatch(const patch_type& patch, const Obj<Points>& points) {
        for(typename Points::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(patch, *p_iter, 1);
        }
      };
      void copy(const Obj<UniformSection<Topology_, Value_, fiberDim> >& section) {
        this->getAtlas()->copy(section->getAtlas());
        const typename topology_type::sheaf_type& sheaf = section->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator s_iter = sheaf.begin(); s_iter != sheaf.end(); ++s_iter) {
          const patch_type& patch = s_iter->first;
          if (!section->hasPatch(patch)) continue;
          const chart_type& chart = section->getPatch(patch);

          for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            this->update(s_iter->first, *c_iter, section->restrict(s_iter->first, *c_iter));
          }
        }
      };
    public: // Sizes
      void clear() {
        this->_atlas->clear(); 
        this->_arrays.clear();
      };
      int getFiberDimension(const patch_type& patch, const point_type& p) const {
        // Could check for non-existence here
        return this->_atlas->restrictPoint(patch, p)[0];
      };
      void setFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        this->checkDimension(dim);
        this->_atlas->updatePatch(patch, p);
        this->_atlas->updatePoint(patch, p, &dim);
      };
      void addFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        if (this->hasPatch(patch) && (this->_atlas[patch].find(p) != this->_atlas[patch].end())) {
          ostringstream msg;
          msg << "Invalid addition to fiber dimension " << dim << " cannot exceed " << fiberDim << std::endl;
          throw ALE::Exception(msg.str().c_str());
        } else {
          this->setFiberDimension(patch, p, dim);
        }
      };
      void setFiberDimensionByDepth(const patch_type& patch, int depth, int dim) {
        this->setFiberDimensionByLabel(patch, "depth", depth, dim);
      };
      void setFiberDimensionByHeight(const patch_type& patch, int height, int dim) {
        this->setFiberDimensionByLabel(patch, "height", height, dim);
      };
      void setFiberDimensionByLabel(const patch_type& patch, const std::string& label, int value, int dim) {
        const Obj<typename topology_type::label_sequence>& points = this->getTopology()->getLabelStratum(patch, label, value);

        for(typename topology_type::label_sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(patch, *p_iter, dim);
        }
      };
      int size(const patch_type& patch) {
        const typename atlas_type::chart_type& points = this->_atlas->getPatch(patch);
        int size = 0;

        for(typename atlas_type::chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          size += this->getFiberDimension(patch, *p_iter);
        }
        return size;
      };
      int size(const patch_type& patch, const point_type& p) {
        const typename atlas_type::chart_type&  points  = this->_atlas->getPatch(patch);
        const Obj<typename sieve_type::coneSet> closure = this->getTopology()->getPatch(patch)->closure(p);
        typename sieve_type::coneSet::iterator  end     = closure->end();
        int size = 0;

        for(typename sieve_type::coneSet::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
          if (points.count(*c_iter)) {
            size += this->getFiberDimension(patch, *c_iter);
          }
        }
        return size;
      };
      void orderPatches() {};
    public: // Restriction
      const array_type& restrict(const patch_type& patch) {
        this->checkPatch(patch);
        return this->_arrays[patch];
      };
      // Return the values for the closure of this point
      //   use a smart pointer?
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        this->checkPatch(patch);
        const chart_type& chart = this->getPatch(patch);
        array_type& array  = this->_arrays[patch];
        const int   size   = this->size(patch, p);
        value_type *values = this->getRawArray(size);
        int         j      = -1;

        // We could actually ask for the height of the individual point
        if (this->getTopology()->height(patch) < 2) {
          // Only avoids the copy of closure()
          const int& dim = this->_atlas->restrict(patch, p)[0];

          if (chart.count(p)) {
            for(int i = 0; i < dim; ++i) {
              values[++j] = array[p].v[i];
            }
          }
          // Should be closure()
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            if (chart.count(*p_iter)) {
              const int& dim = this->_atlas->restrict(patch, *p_iter)[0];

              for(int i = 0; i < dim; ++i) {
                values[++j] = array[*p_iter].v[i];
              }
            }
          }
        } else {
          throw ALE::Exception("Not yet implemented for interpolated sieves");
        }
        if (j != size-1) {
          ostringstream txt;

          txt << "Invalid restrict to point " << p << std::endl;
          txt << "  j " << j << " should be " << (size-1) << std::endl;
          std::cout << txt.str();
          throw ALE::Exception(txt.str().c_str());
        }
        return values;
      };
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_atlas->checkPatch(patch);
        const chart_type& chart = this->getPatch(patch);
        array_type& array = this->_arrays[patch];
        int         j     = -1;

        if (this->getTopology()->height(patch) < 2) {
          // Only avoids the copy of closure()
          const int& dim = this->_atlas->restrict(patch, p)[0];

          if (chart.count(p)) {
            for(int i = 0; i < dim; ++i) {
              array[p].v[i] = v[++j];
            }
          }
          // Should be closure()
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            if (chart.count(*p_iter)) {
              const int& dim = this->_atlas->restrict(patch, *p_iter)[0];

              for(int i = 0; i < dim; ++i) {
                array[*p_iter].v[i] = v[++j];
              }
            }
          }
        } else {
          throw ALE::Exception("Not yet implemented for interpolated sieves");
        }
      };
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_atlas->checkPatch(patch);
        const chart_type& chart = this->getPatch(patch);
        array_type& array = this->_arrays[patch];
        int         j     = -1;

        if (this->getTopology()->height(patch) < 2) {
          // Only avoids the copy of closure()
          const int& dim = this->_atlas->restrict(patch, p)[0];

          if (chart.count(p)) {
            for(int i = 0; i < dim; ++i) {
              array[p].v[i] += v[++j];
            }
          }
          // Should be closure()
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            if (chart.count(*p_iter)) {
              const int& dim = this->_atlas->restrict(patch, *p_iter)[0];

              for(int i = 0; i < dim; ++i) {
                array[*p_iter].v[i] += v[++j];
              }
            }
          }
        } else {
          throw ALE::Exception("Not yet implemented for interpolated sieves");
        }
      };
      // Return only the values associated to this point, not its closure
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        this->checkPatch(patch);
        return this->_arrays[patch][p].v;
      };
      // Update only the values associated to this point, not its closure
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_atlas->checkPatch(patch);
        for(int i = 0; i < fiberDim; ++i) {
          this->_arrays[patch][p].v[i] = v[i];
        }
      };
      // Update only the values associated to this point, not its closure
      void updateAddPoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->_atlas->checkPatch(patch);
        for(int i = 0; i < fiberDim; ++i) {
          this->_arrays[patch][p].v[i] += v[i];
        }
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
        ostringstream txt;
        int rank;

        if (comm == MPI_COMM_NULL) {
          comm = this->comm();
          rank = this->commRank();
        } else {
          MPI_Comm_rank(comm, &rank);
        }
        if (name == "") {
          if(rank == 0) {
            txt << "viewing a UniformSection" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing UniformSection '" << name << "'" << std::endl;
          }
        }
        for(typename values_type::const_iterator a_iter = this->_arrays.begin(); a_iter != this->_arrays.end(); ++a_iter) {
          const patch_type& patch = a_iter->first;
          array_type&       array = this->_arrays[patch];

          txt << "[" << this->commRank() << "]: Patch " << patch << std::endl;
          const typename atlas_type::chart_type& chart = this->_atlas->getPatch(patch);

          for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
            const point_type&                     point = *p_iter;
            const typename atlas_type::value_type dim   = this->_atlas->restrict(patch, point)[0];

            if (dim != 0) {
              txt << "[" << this->commRank() << "]:   " << point << " dim " << dim << "  ";
              for(int i = 0; i < dim; i++) {
                txt << " " << array[point].v[i];
              }
              txt << std::endl;
            }
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    // A Section is our most general construct (more general ones could be envisioned)
    //   The Atlas is a UniformSection of dimension 1 and value type Point
    //     to hold each fiber dimension and offsets into a contiguous patch array
    template<typename Topology_, typename Value_>
    class Section : public ALE::ParallelObject {
    public:
      typedef Topology_                                 topology_type;
      typedef typename topology_type::patch_type        patch_type;
      typedef typename topology_type::sieve_type        sieve_type;
      typedef typename topology_type::point_type        point_type;
      typedef ALE::Point                                index_type;
      typedef UniformSection<topology_type, index_type> atlas_type;
      typedef typename atlas_type::chart_type           chart_type;
      typedef Value_                                    value_type;
      typedef value_type *                              array_type;
      typedef std::map<patch_type, array_type>          values_type;
      typedef std::vector<index_type>                   IndexArray;
    protected:
      Obj<atlas_type> _atlas;
      Obj<atlas_type> _atlasNew;
      values_type     _arrays;
      Obj<IndexArray> _indexArray;
    public:
      Section(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
        this->_atlas      = new atlas_type(comm, debug);
        this->_atlasNew   = NULL;
        this->_indexArray = new IndexArray();
      };
      Section(const Obj<topology_type>& topology) : ParallelObject(topology->comm(), topology->debug()), _atlasNew(NULL) {
        this->_atlas      = new atlas_type(topology);
        this->_indexArray = new IndexArray();
      };
      Section(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas), _atlasNew(NULL) {
        this->_indexArray = new IndexArray();
      };
      virtual ~Section() {
        for(typename values_type::iterator a_iter = this->_arrays.begin(); a_iter != this->_arrays.end(); ++a_iter) {
          delete [] a_iter->second;
          a_iter->second = NULL;
        }
      };
    protected:
      value_type *getRawArray(const int size) {
        static value_type *array   = NULL;
        static int         maxSize = 0;

        if (size > maxSize) {
          maxSize = size;
          if (array) delete [] array;
          array = new value_type[maxSize];
        };
        return array;
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
      bool hasPatch(const patch_type& patch) {
        return this->_atlas->hasPatch(patch);
      };
    public: // Accessors
      const Obj<atlas_type>& getAtlas() {return this->_atlas;};
      void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
      const Obj<topology_type>& getTopology() {return this->_atlas->getTopology();};
      void setTopology(const Obj<topology_type>& topology) {this->_atlas->setTopology(topology);};
      const chart_type& getPatch(const patch_type& patch) {
        return this->_atlas->getPatch(patch);
      };
    public: // Sizes
      void clear() {
        this->_atlas->clear(); 
        this->_arrays.clear();
      };
      int getFiberDimension(const patch_type& patch, const point_type& p) const {
        // Could check for non-existence here
        return this->_atlas->restrictPoint(patch, p)->prefix;
      };
      int getFiberDimension(const Obj<atlas_type>& atlas, const patch_type& patch, const point_type& p) const {
        // Could check for non-existence here
        return atlas->restrictPoint(patch, p)->prefix;
      };
      void setFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        const index_type idx(dim, -1);
        this->_atlas->updatePatch(patch, p);
        this->_atlas->updatePoint(patch, p, &idx);
      };
      void addFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        const index_type values(dim, -1);
        this->_atlas->updatePatch(patch, p);
        this->_atlas->updateAddPoint(patch, p, &values);
      };
      void setFiberDimensionByDepth(const patch_type& patch, int depth, int dim) {
        this->setFiberDimensionByLabel(patch, "depth", depth, dim);
      };
      void setFiberDimensionByHeight(const patch_type& patch, int height, int dim) {
        this->setFiberDimensionByLabel(patch, "height", height, dim);
      };
      void setFiberDimensionByLabel(const patch_type& patch, const std::string& label, int value, int dim) {
        const Obj<typename topology_type::label_sequence>& points = this->getTopology()->getLabelStratum(patch, label, value);

        for(typename topology_type::label_sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(patch, *p_iter, dim);
        }
      };
      int size(const patch_type& patch) {
        const typename atlas_type::chart_type& points = this->_atlas->getPatch(patch);
        int size = 0;

        for(typename atlas_type::chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          size += this->getFiberDimension(patch, *p_iter);
        }
        return size;
      };
      int size(const patch_type& patch, const point_type& p) {
        const typename atlas_type::chart_type&  points  = this->_atlas->getPatch(patch);
        const Obj<typename sieve_type::coneSet> closure = this->getTopology()->getPatch(patch)->closure(p);
        typename sieve_type::coneSet::iterator  end     = closure->end();
        int size = 0;

        for(typename sieve_type::coneSet::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
          if (points.count(*c_iter)) {
            size += this->getFiberDimension(patch, *c_iter);
          }
        }
        return size;
      };
      int size(const Obj<atlas_type>& atlas, const patch_type& patch) {
        const typename atlas_type::chart_type& points = atlas->getPatch(patch);
        int size = 0;

        for(typename atlas_type::chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          size += this->getFiberDimension(atlas, patch, *p_iter);
        }
        return size;
      };
    public: // Index retrieval
      const index_type& getIndex(const patch_type& patch, const point_type& p) {
        this->checkPatch(patch);
        return this->_atlas->restrictPoint(patch, p)[0];
      };
      template<typename Numbering>
      const index_type getIndex(const patch_type& patch, const point_type& p, const Obj<Numbering>& numbering) {
        this->checkPatch(patch);
        return index_type(this->getFiberDimension(patch, p), numbering->getIndex(p));
      };
      const Obj<IndexArray>& getIndices(const patch_type& patch, const point_type& p, const int level = -1) {
        this->_indexArray->clear();

        if (level == 0) {
          this->_indexArray->push_back(this->getIndex(patch, p));
        } else if ((level == 1) || (this->getTopology()->height(patch) == 1)) {
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          this->_indexArray->push_back(this->getIndex(patch, p));
          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            this->_indexArray->push_back(this->getIndex(patch, *p_iter));
          }
        } else if (level == -1) {
          const Obj<typename sieve_type::coneSet> closure = this->getTopology()->getPatch(patch)->closure(p);
          typename sieve_type::coneSet::iterator  end     = closure->end();

          for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
            this->_indexArray->push_back(this->getIndex(patch, *p_iter));
          }
        } else {
          const Obj<typename sieve_type::coneArray> cone = this->getTopology()->getPatch(patch)->nCone(p, level);
          typename sieve_type::coneArray::iterator  end  = cone->end();

          for(typename sieve_type::coneArray::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            this->_indexArray->push_back(this->getIndex(patch, *p_iter));
          }
        }
        return this->_indexArray;
      };
      template<typename Numbering>
      const Obj<IndexArray>& getIndices(const patch_type& patch, const point_type& p, const Obj<Numbering>& numbering, const int level = -1) {
        this->_indexArray->clear();

        if (level == 0) {
          this->_indexArray->push_back(this->getIndex(patch, p, numbering));
        } else if ((level == 1) || (this->getTopology()->height(patch) == 1)) {
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          this->_indexArray->push_back(this->getIndex(patch, p, numbering));
          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            this->_indexArray->push_back(this->getIndex(patch, *p_iter, numbering));
          }
        } else if (level == -1) {
          const Obj<typename sieve_type::coneSet> closure = this->getTopology()->getPatch(patch)->closure(p);
          typename sieve_type::coneSet::iterator  end     = closure->end();

          for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
            this->_indexArray->push_back(this->getIndex(patch, *p_iter, numbering));
          }
        } else {
          const Obj<typename sieve_type::coneArray> cone = this->getTopology()->getPatch(patch)->nCone(p, level);
          typename sieve_type::coneArray::iterator  end  = cone->end();

          for(typename sieve_type::coneArray::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            this->_indexArray->push_back(this->getIndex(patch, *p_iter, numbering));
          }
        }
        return this->_indexArray;
      };
    public: // Allocation
      void orderPoint(const Obj<atlas_type>& atlas, const Obj<sieve_type>& sieve, const patch_type& patch, const point_type& point, int& offset) {
        const Obj<typename sieve_type::coneSequence>& cone = sieve->cone(point);
        typename sieve_type::coneSequence::iterator   end  = cone->end();
        index_type                                    idx  = atlas->restrictPoint(patch, point)[0];
        const int&                                    dim  = idx.prefix;
        const index_type                              defaultIdx(0, -1);

        if (atlas->getPatch(patch).count(point) == 0) {
          idx = defaultIdx;
        }
        if (idx.index < 0) {
          for(typename sieve_type::coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
            if (this->_debug > 1) {std::cout << "    Recursing to " << *c_iter << std::endl;}
            this->orderPoint(atlas, sieve, patch, *c_iter, offset);
          }
          if (dim > 0) {
            if (this->_debug > 1) {std::cout << "  Ordering point " << point << " at " << offset << std::endl;}
            idx.index = offset;
            atlas->updatePoint(patch, point, &idx);
            offset += dim;
          }
        }
      }
      void orderPatch(const Obj<atlas_type>& atlas, const patch_type& patch, int& offset) {
        const typename atlas_type::chart_type& chart = atlas->getPatch(patch);

        for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
          if (this->_debug > 1) {std::cout << "Ordering closure of point " << *p_iter << std::endl;}
          this->orderPoint(atlas, this->getTopology()->getPatch(patch), patch, *p_iter, offset);
        }
      };
      void orderPatches(const Obj<atlas_type>& atlas) {
        const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          if (this->_debug > 1) {std::cout << "Ordering patch " << p_iter->first << std::endl;}
          int offset = 0;

          if (!atlas->hasPatch(p_iter->first)) continue;
          this->orderPatch(atlas, p_iter->first, offset);
        }
      };
      void orderPatches() {
        this->orderPatches(this->_atlas);
      };
      void allocate() {
        this->orderPatches();
        const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          if (!this->_atlas->hasPatch(p_iter->first)) continue;
          this->_arrays[p_iter->first] = new value_type[this->size(p_iter->first)];
          PetscMemzero(this->_arrays[p_iter->first], this->size(p_iter->first) * sizeof(value_type));
        }
      };
      void addPoint(const patch_type& patch, const point_type& point, const int dim) {
        //const typename atlas_type::chart_type& chart = this->_atlas->getPatch(patch);

        //if (chart.find(point) == chart.end()) {
        if (this->_atlasNew.isNull()) {
          this->_atlasNew = new atlas_type(this->getTopology());
          this->_atlasNew->copy(this->_atlas);
        }
        const index_type idx(dim, -1);
        this->_atlasNew->updatePatch(patch, point);
        this->_atlasNew->update(patch, point, &idx);
      };
      void reallocate() {
        if (this->_atlasNew.isNull()) return;
        this->orderPatches(_atlasNew);
        const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const patch_type&                      patch    = p_iter->first;
          value_type                            *newArray = new value_type[this->size(this->_atlasNew, patch)];

          this->_arrays[patch] = newArray;
          if (!this->_atlas->hasPatch(patch)) continue;
          const typename atlas_type::chart_type& chart    = this->_atlas->getPatch(patch);
          const value_type                      *array    = this->_arrays[patch];

          for(typename atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const index_type& idx       = this->_atlas->restrict(patch, *c_iter)[0];
            const int&        size      = idx.prefix;
            const int&        offset    = idx.index;
            const int&        newOffset = this->_atlasNew->restrict(patch, *c_iter)[0].index;

            for(int i = 0; i < size; ++i) {
              newArray[newOffset+i] = array[offset+i];
            }
          }
          delete [] this->_arrays[patch];
        }
        this->_atlas    = this->_atlasNew;
        this->_atlasNew = NULL;
      };
    public: // Restriction
      // Return a pointer to the entire contiguous storage array
      const value_type *restrict(const patch_type& patch) {
        this->checkPatch(patch);
        return this->_arrays[patch];
      };
      // Update the entire contiguous storage array
      void update(const patch_type& patch, const value_type v[]) {
        const value_type *array = this->_arrays[patch];
        const int         size  = this->size(patch);

        for(int i = 0; i < size; i++) {
          array[i] = v[i];
        }
      };
      // Return the values for the closure of this point
      //   use a smart pointer?
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        this->checkPatch(patch);
        const value_type *a      = this->_arrays[patch];
        const int         size   = this->size(patch, p);
        value_type       *values = this->getRawArray(size);
        int               j      = -1;

        if (this->getTopology()->height(patch) < 2) {
          // Avoids the copy of both
          //   points  in topology->closure()
          //   indices in _atlas->restrict()
          const index_type& pInd = this->_atlas->restrictPoint(patch, p)[0];

          for(int i = pInd.index; i < pInd.prefix + pInd.index; ++i) {
            values[++j] = a[i];
          }
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            const index_type& ind    = this->_atlas->restrictPoint(patch, *p_iter)[0];
            const int&        start  = ind.index;
            const int&        length = ind.prefix;

            for(int i = start; i < start + length; ++i) {
              values[++j] = a[i];
            }
          }
        } else {
          const Obj<IndexArray>& ind = this->getIndices(patch, p);

          for(typename IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int& start  = i_iter->index;
            const int& length = i_iter->prefix;

            for(int i = start; i < start + length; ++i) {
              values[++j] = a[i];
            }
          }
        }
        if (j != size-1) {
          ostringstream txt;

          txt << "Invalid restrict to point " << p << std::endl;
          txt << "  j " << j << " should be " << (size-1) << std::endl;
          std::cout << txt.str();
          throw ALE::Exception(txt.str().c_str());
        }
        return values;
      };
      // Update the values for the closure of this point
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        value_type *a = this->_arrays[patch];
        int         j = -1;

        if (this->getTopology()->height(patch) < 2) {
          // Avoids the copy of both
          //   points  in topology->closure()
          //   indices in _atlas->restrict()
          const index_type& pInd = this->_atlas->restrictPoint(patch, p)[0];

          for(int i = pInd.index; i < pInd.prefix + pInd.index; ++i) {
            a[i] = v[++j];
          }
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            const index_type& ind    = this->_atlas->restrictPoint(patch, *p_iter)[0];
            const int&        start  = ind.index;
            const int&        length = ind.prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
            }
          }
        } else {
          const Obj<IndexArray>& ind = this->getIndices(patch, p);

          for(typename IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int& start  = i_iter->index;
            const int& length = i_iter->prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
            }
          }
        }
      };
      // Update the values for the closure of this point
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        value_type *a = this->_arrays[patch];
        int         j = -1;

        if (this->getTopology()->height(patch) < 2) {
          // Avoids the copy of both
          //   points  in topology->closure()
          //   indices in _atlas->restrict()
          const index_type& pInd = this->_atlas->restrictPoint(patch, p)[0];

          for(int i = pInd.index; i < pInd.prefix + pInd.index; ++i) {
            a[i] += v[++j];
          }
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            const index_type& ind    = this->_atlas->restrictPoint(patch, *p_iter)[0];
            const int&        start  = ind.index;
            const int&        length = ind.prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] += v[++j];
            }
          }
        } else {
          const Obj<IndexArray>& ind = this->getIndices(patch, p);

          for(typename IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int& start  = i_iter->index;
            const int& length = i_iter->prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] += v[++j];
            }
          }
        }
      };
      // Update the values for the closure of this point
      template<typename Input>
      void update(const patch_type& patch, const point_type& p, const Obj<Input>& v) {
        this->checkPatch(patch);
        value_type *a = this->_arrays[patch];

        if (this->getTopology()->height(patch) == 1) {
          // Only avoids the copy
          const index_type& pInd = this->_atlas->restrictPoint(patch, p)[0];
          typename Input::iterator v_iter = v->begin();
          typename Input::iterator v_end  = v->end();

          for(int i = pInd.index; i < pInd.prefix + pInd.index; ++i) {
            a[i] = *v_iter;
            ++v_iter;
          }
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            const index_type& ind    = this->_atlas->restrictPoint(patch, *p_iter)[0];
            const int&        start  = ind.index;
            const int&        length = ind.prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] = *v_iter;
              ++v_iter;
            }
          }
        } else {
          const Obj<typename atlas_type::IndexArray>& ind = this->getIndices(patch, p);
          typename Input::iterator v_iter = v->begin();
          typename Input::iterator v_end  = v->end();

          for(typename atlas_type::IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int& start  = i_iter->index;
            const int& length = i_iter->prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] = *v_iter;
              ++v_iter;
            }
          }
        }
      };
      // Return only the values associated to this point, not its closure
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        this->checkPatch(patch);
        return &(this->_arrays[patch][this->_atlas->restrictPoint(patch, p)[0].index]);
      };
      // Update only the values associated to this point, not its closure
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        const index_type& idx = this->_atlas->restrictPoint(patch, p)[0];
        value_type       *a   = &(this->_arrays[patch][idx.index]);

        for(int i = 0; i < idx.prefix; ++i) {
          a[i] = v[i];
        }
      };
      // Update only the values associated to this point, not its closure
      void updateAddPoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        const index_type& idx = this->_atlas->restrictPoint(patch, p)[0];
        value_type       *a   = &(this->_arrays[patch][idx.index]);

        for(int i = 0; i < idx.prefix; ++i) {
          a[i] += v[i];
        }
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        ostringstream txt;
        int rank;

        if (comm == MPI_COMM_NULL) {
          comm = this->comm();
          rank = this->commRank();
        } else {
          MPI_Comm_rank(comm, &rank);
        }
        if (name == "") {
          if(rank == 0) {
            txt << "viewing a Section" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing Section '" << name << "'" << std::endl;
          }
        }
        for(typename values_type::const_iterator a_iter = this->_arrays.begin(); a_iter != this->_arrays.end(); ++a_iter) {
          const patch_type&  patch = a_iter->first;
          const value_type  *array = a_iter->second;

          txt << "[" << this->commRank() << "]: Patch " << patch << std::endl;
          const typename atlas_type::chart_type& chart = this->_atlas->getPatch(patch);

          for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
            const point_type& point = *p_iter;
            const index_type& idx   = this->_atlas->restrict(patch, point)[0];

            if (idx.prefix != 0) {
              txt << "[" << this->commRank() << "]:   " << point << " dim " << idx.prefix << " offset " << idx.index << "  ";
              for(int i = 0; i < idx.prefix; i++) {
                txt << " " << array[idx.index+i];
              }
              txt << std::endl;
            }
          }
        }
        if (this->_arrays.empty()) {
          txt << "[" << this->commRank() << "]: empty" << std::endl;
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    // An Overlap is a Sifter describing the overlap of two Sieves
    //   Each arrow is local point ---(remote point)---> remote rank right now
    //     For XSifter, this should change to (local patch, local point) ---> (remote rank, remote patch, remote point)

    template<typename Topology_, typename Value_>
    class ConstantSection : public ALE::ParallelObject {
    public:
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Value_                             value_type;
    protected:
      Obj<topology_type> _topology;
      const value_type   _value;
    public:
      ConstantSection(MPI_Comm comm, const value_type value, const int debug = 0) : ParallelObject(comm, debug), _value(value) {
        this->_topology = new topology_type(comm, debug);
      };
      ConstantSection(const Obj<topology_type>& topology, const value_type value) : ParallelObject(topology->comm(), topology->debug()), _topology(topology), _value(value) {};
      virtual ~ConstantSection() {};
    public:
      void allocate() {};
      const value_type *restrict(const patch_type& patch) {return &this->_value;};
      // This should return something the size of the closure
      const value_type *restrict(const patch_type& patch, const point_type& p) {return &this->_value;};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {return &this->_value;};
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a ConstantSection");
      };
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a ConstantSection");
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
        throw ALE::Exception("Cannot update a ConstantSection");
      };
      template<typename Input>
      void update(const patch_type& patch, const point_type& p, const Obj<Input>& v) {
        throw ALE::Exception("Cannot update a ConstantSection");
      };
    public:
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        ostringstream txt;
        int rank;

        if (comm == MPI_COMM_NULL) {
          comm = this->comm();
          rank = this->commRank();
        } else {
          MPI_Comm_rank(comm, &rank);
        }
        if (name == "") {
          if(rank == 0) {
            txt << "viewing a ConstantSection with value " << this->_value << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing ConstantSection '" << name << "' with value " << this->_value << std::endl;
          }
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };

    template<typename Point_>
    class DiscreteSieve {
    public:
      typedef Point_                  point_type;
      typedef std::vector<point_type> coneSequence;
      typedef std::vector<point_type> coneSet;
      typedef std::vector<point_type> coneArray;
      typedef std::vector<point_type> supportSequence;
      typedef std::vector<point_type> supportSet;
      typedef std::vector<point_type> supportArray;
      typedef std::set<point_type>    points_type;
      typedef points_type             baseSequence;
      typedef points_type             capSequence;
    protected:
      Obj<points_type>  _points;
      Obj<coneSequence> _empty;
      Obj<coneSequence> _return;
      void _init() {
        this->_points = new points_type();
        this->_empty  = new coneSequence();
        this->_return = new coneSequence();
      };
    public:
      DiscreteSieve() {
        this->_init();
      };
      template<typename Input>
      DiscreteSieve(const Obj<Input>& points) {
        this->_init();
        this->_points->insert(points->begin(), points->end());
      };
      virtual ~DiscreteSieve() {};
    public:
      void addPoint(const point_type& point) {
        this->_points->insert(point);
      };
      template<typename Input>
      void addPoints(const Obj<Input>& points) {
        this->_points->insert(points->begin(), points->end());
      };
      const Obj<coneSequence>& cone(const point_type& p) {return this->_empty;};
      template<typename Input>
      const Obj<coneSequence>& cone(const Input& p) {return this->_empty;};
      const Obj<coneSet>& nCone(const point_type& p, const int level) {
        if (level == 0) {
          return this->closure(p);
        } else {
          return this->_empty;
        }
      };
      const Obj<coneArray>& closure(const point_type& p) {
        this->_return->clear();
        this->_return->push_back(p);
        return this->_return;
      };
      const Obj<supportSequence>& support(const point_type& p) {return this->_empty;};
      template<typename Input>
      const Obj<supportSequence>& support(const Input& p) {return this->_empty;};
      const Obj<supportSet>& nSupport(const point_type& p, const int level) {
        if (level == 0) {
          return this->star(p);
        } else {
          return this->_empty;
        }
      };
      const Obj<supportArray>& star(const point_type& p) {
        this->_return->clear();
        this->_return->push_back(p);
        return this->_return;
      };
      const Obj<capSequence>& roots() {return this->_points;};
      const Obj<capSequence>& cap() {return this->_points;};
      const Obj<baseSequence>& leaves() {return this->_points;};
      const Obj<baseSequence>& base() {return this->_points;};
      template<typename Color>
      void addArrow(const point_type& p, const point_type& q, const Color& color) {
        throw ALE::Exception("Cannot add an arrow to a DiscreteSieve");
      };
      void stratify() {};
      void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) const {
        ostringstream txt;
        int rank;

        if (comm == MPI_COMM_NULL) {
          comm = MPI_COMM_SELF;
          rank = 0;
        } else {
          MPI_Comm_rank(comm, &rank);
        }
        if (name == "") {
          if(rank == 0) {
            txt << "viewing a DiscreteSieve" << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing DiscreteSieve '" << name << "'" << std::endl;
          }
        }
        for(typename points_type::const_iterator p_iter = this->_points->begin(); p_iter != this->_points->end(); ++p_iter) {
          txt << "  Point " << *p_iter << std::endl;
        }
        PetscSynchronizedPrintf(comm, txt.str().c_str());
        PetscSynchronizedFlush(comm);
      };
    };


    template<typename Overlap_, typename Topology_, typename Value_>
    class OverlapValues : public Section<Topology_, Value_> {
    public:
      typedef Overlap_                          overlap_type;
      typedef Section<Topology_, Value_>        base_type;
      typedef typename base_type::patch_type    patch_type;
      typedef typename base_type::topology_type topology_type;
      typedef typename base_type::atlas_type    atlas_type;
      typedef typename base_type::value_type    value_type;
      typedef enum {SEND, RECEIVE}              request_type;
      typedef std::map<patch_type, MPI_Request> requests_type;
    protected:
      int           _tag;
      MPI_Datatype  _datatype;
      requests_type _requests;
    public:
      OverlapValues(MPI_Comm comm, const int debug = 0) : Section<Topology_, Value_>(comm, debug) {
        this->_tag      = this->getNewTag();
        this->_datatype = this->getMPIDatatype();
      };
      OverlapValues(MPI_Comm comm, const int tag, const int debug) : Section<Topology_, Value_>(comm, debug), _tag(tag) {
        this->_datatype = this->getMPIDatatype();
      };
      virtual ~OverlapValues() {};
    protected:
      MPI_Datatype getMPIDatatype() {
        if (sizeof(value_type) == 4) {
          return MPI_INT;
        } else if (sizeof(value_type) == 8) {
          return MPI_DOUBLE;
        } else if (sizeof(value_type) == 28) {
          int          blen[2];
          MPI_Aint     indices[2];
          MPI_Datatype oldtypes[2], newtype;
          blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_INT;
          blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
          MPI_Type_struct(2, blen, indices, oldtypes, &newtype);
          MPI_Type_commit(&newtype);
          return newtype;
        } else if (sizeof(value_type) == 32) {
          int          blen[2];
          MPI_Aint     indices[2];
          MPI_Datatype oldtypes[2], newtype;
          blen[0] = 1; indices[0] = 0;           oldtypes[0] = MPI_DOUBLE;
          blen[1] = 3; indices[1] = sizeof(int); oldtypes[1] = MPI_DOUBLE;
          MPI_Type_struct(2, blen, indices, oldtypes, &newtype);
          MPI_Type_commit(&newtype);
          return newtype;
        }
        throw ALE::Exception("Cannot determine MPI type for value type");
      };
      int getNewTag() {
        static int tagKeyval = MPI_KEYVAL_INVALID;
        int *tagvalp = NULL, *maxval, flg;

        if (tagKeyval == MPI_KEYVAL_INVALID) {
          PetscMalloc(sizeof(int), &tagvalp);
          MPI_Keyval_create(MPI_NULL_COPY_FN, Petsc_DelTag, &tagKeyval, (void *) NULL);
          MPI_Attr_put(this->_comm, tagKeyval, tagvalp);
          tagvalp[0] = 0;
        }
        MPI_Attr_get(this->_comm, tagKeyval, (void **) &tagvalp, &flg);
        if (tagvalp[0] < 1) {
          MPI_Attr_get(MPI_COMM_WORLD, MPI_TAG_UB, (void **) &maxval, &flg);
          tagvalp[0] = *maxval - 128; // hope that any still active tags were issued right at the beginning of the run
        }
        //std::cout << "[" << this->commRank() << "]Got new tag " << tagvalp[0] << std::endl;
        return tagvalp[0]--;
      };
    public: // Accessors
      int getTag() const {return this->_tag;};
      void setTag(const int tag) {this->_tag = tag;};
    public:
      void construct(const int size) {
        const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                  rank = p_iter->first;

          for(typename topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            this->setFiberDimension(rank, *b_iter, size);
          }
        }
      };
      template<typename Sizer>
      void construct(const Obj<Sizer>& sizer) {
        const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const Obj<typename topology_type::sieve_type::baseSequence>& base = p_iter->second->base();
          int                                  rank = p_iter->first;

          for(typename topology_type::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
            this->setFiberDimension(rank, *b_iter, *(sizer->restrict(rank, *b_iter)));
          }
        }
      };
      void constructCommunication(const request_type& requestType) {
        const typename topology_type::sheaf_type& patches = this->getAtlas()->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const patch_type patch = p_iter->first;
          MPI_Request request;

          if (requestType == RECEIVE) {
            if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Receiving data(" << this->size(patch) << ") from " << patch << " tag " << this->_tag << std::endl;}
            MPI_Recv_init(this->_arrays[patch], this->size(patch), this->_datatype, patch, this->_tag, this->_comm, &request);
          } else {
            if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Sending data (" << this->size(patch) << ") to " << patch << " tag " << this->_tag << std::endl;}
            MPI_Send_init(this->_arrays[patch], this->size(patch), this->_datatype, patch, this->_tag, this->_comm, &request);
          }
          this->_requests[patch] = request;
        }
      };
      void startCommunication() {
        const typename topology_type::sheaf_type& patches = this->getAtlas()->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          MPI_Request request = this->_requests[p_iter->first];

          MPI_Start(&request);
        }
      };
      void endCommunication() {
        const typename topology_type::sheaf_type& patches = this->getAtlas()->getTopology()->getPatches();
        MPI_Status status;

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          MPI_Request request = this->_requests[p_iter->first];

          MPI_Wait(&request, &status);
        }
      };
    };
  }
}

#endif
