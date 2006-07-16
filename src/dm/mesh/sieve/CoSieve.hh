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

      if (sifterType < 0) {
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
    int         debug()    const {return this->_debug;};
    void        setDebug(const int debug) {this->_debug = debug;};
    MPI_Comm    comm()     const {return this->_comm;};
    int         commSize() const {return this->_commSize;};
    int         commRank() const {return this->_commRank;}
    PetscObject petscObj() const {return this->_petscObj;};
  };

  namespace New {
    template<typename Patch_, typename Sieve_>
    class Topology {
    public:
      typedef Patch_                                                    patch_type;
      typedef Sieve_                                                    sieve_type;
      typedef typename sieve_type::point_type                           point_type;
      typedef typename ALE::set<point_type>                             PointSet;
      typedef typename std::map<patch_type, Obj<sieve_type> >           sheaf_type;
      typedef typename ALE::Sifter<point_type, int, int>                patch_label_type;
      typedef typename std::map<patch_type, Obj<patch_label_type> >     label_type;
      typedef typename std::map<const typename std::string, label_type> labels_type;
    protected:
      sheaf_type  _sheaf;
      labels_type _labels;
      int         _maxHeight;
    public:
      const Obj<sieve_type>& const getPatch(const patch_type& patch) {return this->_sheaf[patch]};

      int getValue const (const Obj<patch_label_type>& label, const point_type& point, const int defValue = 0) {
        Obj<typename patch_label_type::coneSequence> cone = label->cone(point);

        if (cone->size() == 0) return defValue;
        return *cone->begin();
      };
      template<typename InputPoints>
      int getMaxValue const (const Obj<patch_label_type>& label, const Obj<InputPoints>& points, const int defValue = 0) {
        Obj<typename patch_label_type::coneSequence> cone = label->cone(point);
        int maxValue = defValue;

        for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          maxValue = std::max(maxValue, this->getValue(label, *p_iter, defValue));
        }
        return maxValue;
      };
      void setValue(const Obj<patch_label_type>& label, const point_type& point, const int value) {
        label->setCone(value, point);
      };
      template<class InputPoints>
      void computeHeight(const Obj<patch_label_type>& height, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points) {
        for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          // Compute the max height of the points in the support of p, and add 1
          int h0 = this->getValue(height, *p_iter, -1);
          int h1 = this->getMaxValue(height, this->support(*p_iter), -1) + 1;

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
    };

    template<typename Topology_, typename Index_>
    class Atlas : public ALE::ParallelObject {
    public:
      typedef Topology_                                       topology_type;
      typedef typename topology_type::patch_type              patch_type;
      typedef typename topology_type::point_type              point_type;
      typedef Index_                                          index_type;
      typedef std::vector<index_type>                         IndexArray;
      typedef typename std::map<point_type, index_type>       patch_index_type;
      typedef typename std::map<patch_type, patch_index_type> indices_type;
    protected:
      Obj<topology_type> _topology;
      indices_type       _indices;
      Obj<IndexArray>    _array;
    public:
      Atlas(const Obj<topology_type>& topology) : ParallelObject(topology->comm(), topology->debug()), _topology(topology) {
        _array = new IndexArray();
      };
    public:
      const Obj<topology_type>& const getTopology() {return this->_topology};
      const index_type& const getIndex(const patch_type& patch, const point_type& p) {
        return this->_index[patch][p];
      };
      // Want to return a sequence
      const Obj<IndexArray>& const getIndices(const patch_type& patch, const point_type& p, const int level = -1) {
        array->clear();

        if (level == 0) {
          array->push_back(this->getIndex(patch, p));
        } else if ((level == 1) || (this->_topology->height() == 1)) {
          Obj<typename sieve_type::coneSequence> cone = this->_topology->getPatch(patch)->cone(p);

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            array->push_back(this->getIndex(patch, *p_iter));
          }
        } else if (level == -1) {
          Obj<typename sieve_type::coneSet> closure = this->_topology->getPatch(patch)->closure(p);

          for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != closure->end(); ++p_iter) {
            array->push_back(this->getIndex(patch, *p_iter));
          }
        } else {
          Obj<typename sieve_type::coneSet> cone = this->_topology->getPatch(patch)->nCone(p, level);

          for(typename sieve_type::coneSet::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            array->push_back(this->getIndex(patch, *p_iter));
          }
        }
        return array;
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
      typedef std::map<patch_type, size_t>       sizes_type;
    protected:
      Obj<atlas_type> _atlas;
      values_type     _arrays;
      sizes_type      _sizes;
    public:
      Section(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas) {};
      virtual ~Section() {
        for(typename values_type::iterator a_iter = this->_arrays.begin(); a_iter != this->_arrays.end(); ++a_iter) {
          delete [] a_iter->second;
          a_iter->second              = NULL;
          this->_sizes[a_iter->first] = 0;
        }
      };
    public:
      // Return a smart pointer?
      const value_type *restrict(const patch_type& patch, const point_type& p) {
        static value_type                     *values = NULL;
        static int                             size = 0;
      };
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {
        // Using the index structure explicitly
        return this->_arrays[patch][this->_atlas->getIndex(patch, p).first]
      };
      void update(const patch_type& patch, const point_type& p, const value_type v[]) {
        value_type *a = this->_arrays[patch];

        if (this->_topology->height() == 1) {
          // Only avoids the copy
          Obj<typename sieve_type::coneSequence> cone = this->_atlas->getTopology()->getPatch(patch)->cone(p);
          int                                    j    = -1;

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
            const index_type& ind    = this->_atlas->getIndex(patch, p);
            const int         start  = ind.first;
            const int         length = ind.second;

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
            }
          }
        } else {
          const Obj<typename atlas_type::IndexArray>& ind = this->_atlas->getIndices(patch, p);
          int                                         j   = -1;

          for(typename atlas_type::IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int start  = i_iter->first;
            const int length = i_iter->second;

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
            }
          }
        }
      };
      void updatePoint(const patch_type& patch, const point_type& p, const value_type v[]) {
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
