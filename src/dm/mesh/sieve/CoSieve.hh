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
        this->_indices[patch][p].index = dim;
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
          int                                           j    = -1;

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            const index_type& ind    = this->_atlas->getIndex(patch, p);
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
      void updateAdd(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        value_type *a = this->_arrays[patch];

        if (this->_atlas->getTopology()->getPatch(patch)->height() == 1) {
          // Only avoids the copy
          const Obj<typename sieve_type::coneSequence>& cone = this->_atlas->getTopology()->getPatch(patch)->cone(p);
          int                                           j    = -1;

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            const index_type& ind    = this->_atlas->getIndex(patch, p);
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
    };
  }
}

#endif
