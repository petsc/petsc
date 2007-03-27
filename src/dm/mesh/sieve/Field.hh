#ifndef included_ALE_Field_hh
#define included_ALE_Field_hh

#ifndef  included_ALE_Sieve_hh
#include <Sieve.hh>
#endif

#ifndef  included_ALE_SieveAlgorithms_hh
#include <SieveAlgorithms.hh>
#endif

// Sieve need point_type
// Section need point_type and value_type
//   size(), restrict(), update() need orientation which is a Bundle (circular)
// Bundle is Sieve+Section

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
namespace ALE {
namespace Field {
  // A ConstantSection is the simplest Section
  //   All fibers are dimension 1
  //   All values are equal to a constant
  //     We need no value storage and no communication for completion
  template<typename Point_, typename Value_>
  class ConstantSection : public ALE::ParallelObject {
  public:
    typedef Point_               point_type;
    typedef std::set<point_type> chart_type;
    typedef Value_               value_type;
  protected:
    chart_type _chart;
    value_type _value;
    value_type _defaultValue;
  public:
    ConstantSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _defaultValue(0) {};
    ConstantSection(MPI_Comm comm, const value_type& value, const int debug = 0) : ParallelObject(comm, debug), _value(value), _defaultValue(value) {};
    ConstantSection(MPI_Comm comm, const value_type& value, const value_type& defaultValue, const int debug = 0) : ParallelObject(comm, debug), _value(value), _defaultValue(defaultValue) {};
  public: // Verifiers
    void checkPoint(const point_type& point) const {
      if (this->_chart.find(point) == this->_chart.end()) {
        ostringstream msg;
        msg << "Invalid section point " << point << std::endl;
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
    bool hasPoint(const point_type& point) const {
      return this->_chart.count(point) > 0;
    };
  public: // Accessors
    const chart_type& getChart() {return this->_chart;};
    void addPoint(const point_type& point) {
      this->_chart.insert(point);
    };
    template<typename Points>
    void addPoint(const Obj<Points>& points) {
      this->_chart.insert(points->begin(), points->end());
    };
    value_type getDefaultValue() {return this->_defaultValue;};
    void setDefaultValue(const value_type value) {this->_defaultValue = value;};
    void copy(const Obj<ConstantSection>& section) {
      const chart_type& chart = section->getChart();

      this->addPoint(chart);
      this->_value = section->restrict(*chart.begin())[0];
    };
  public: // Sizes
    void clear() {
      this->_chart.clear(); 
    };
    int getFiberDimension(const point_type& p) const {
      if (this->hasPoint(p)) return 1;
      return 0;
    };
    void setFiberDimension(const point_type& p, int dim) {
      this->checkDimension(dim);
      this->addPoint(p);
    };
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    };
    void addFiberDimension(const point_type& p, int dim) {
      if (this->_chart.find(p) != this->_chart.end()) {
        ostringstream msg;
        msg << "Invalid addition to fiber dimension " << dim << " cannot exceed 1" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      } else {
        this->setFiberDimension(p, dim);
      }
    };
    int size() {return this->_sheaf.size();};
    int size(const point_type& p) {return this->getFiberDimension(p);};
  public: // Restriction
    const value_type *restrict(const point_type& p) const {
      if (this->hasPoint(p)) {
        return &this->_value;
      }
      return &this->_defaultValue;
    };
    const value_type *restrictPoint(const point_type& p) const {return this->restrict(p);};
    void update(const point_type& p, const value_type v[]) {
      this->_value = v[0];
    };
    void updatePoint(const point_type& p, const value_type v[]) {return this->update(p, v);};
    void updateAdd(const point_type& p, const value_type v[]) {
      this->_value += v[0];
    };
    void updateAddPoint(const point_type& p, const value_type v[]) {return this->updateAdd(p, v);};
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
          txt << "viewing a ConstantSection" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing ConstantSection '" << name << "'" << std::endl;
        }
      }
      txt <<"["<<this->commRank()<<"]: Value " << this->_value << std::endl;
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };

  // A UniformSection often acts as an Atlas
  //   All fibers are the same dimension
  //     Note we can use a ConstantSection for this Atlas
  //   Each point may have a different vector
  //     Thus we need storage for values, and hence must implement completion
  template<typename Point_, typename Value_, int fiberDim = 1>
  class UniformSection : public ALE::ParallelObject {
  public:
    typedef Point_                           point_type;
    typedef ConstantSection<point_type, int> atlas_type;
    typedef typename atlas_type::chart_type  chart_type;
    typedef Value_                           value_type;
    typedef struct {value_type v[fiberDim];} fiber_type;
    typedef std::map<point_type, fiber_type> values_type;
  protected:
    Obj<atlas_type> _atlas;
    values_type     _array;
  public:
    UniformSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      this->_atlas = new atlas_type(comm, fiberDim, 0, debug);
    };
    UniformSection(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas) {
      int dim = fiberDim;
      this->_atlas->update(*this->_atlas->getChart().begin(), &dim);
    };
  public:
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
    bool hasPoint(const point_type& point) {
      return this->_atlas->hasPoint(point);
    };
    void checkDimension(const int& dim) {
      if (dim != fiberDim) {
        ostringstream msg;
        msg << "Invalid fiber dimension " << dim << " must be " << fiberDim << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
  public: // Accessors
    const chart_type& getChart() {return this->_atlas->getChart();};
    const Obj<atlas_type>& getAtlas() {return this->_atlas;};
    void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
    void addPoint(const point_type& point) {
      this->setFiberDimension(point, fiberDim);
    };
    template<typename Points>
    void addPoint(const Obj<Points>& points) {
      for(typename Points::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, fiberDim);
      }
    };
    void copy(const Obj<UniformSection>& section) {
      this->getAtlas()->copy(section->getAtlas());
      const chart_type& chart = section->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        this->updatePoint(*c_iter, section->restrictPoint(*c_iter));
      }
    };
  public: // Sizes
    void clear() {
      this->_atlas->clear(); 
      this->_array.clear();
    };
    int getFiberDimension(const point_type& p) const {
      return this->_atlas->restrictPoint(p)[0];
    };
    void setFiberDimension(const point_type& p, int dim) {
      this->checkDimension(dim);
      this->_atlas->addPoint(p);
      this->_atlas->updatePoint(p, &dim);
    };
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    };
    void setFiberDimension(const std::set<point_type>& points, int dim) {
      for(typename std::set<point_type>::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    };
    void addFiberDimension(const point_type& p, int dim) {
      if (this->hasPoint(p)) {
        ostringstream msg;
        msg << "Invalid addition to fiber dimension " << dim << " cannot exceed " << fiberDim << std::endl;
        throw ALE::Exception(msg.str().c_str());
      } else {
        this->setFiberDimension(p, dim);
      }
    };
    int size() const {
      const chart_type& points = this->_atlas->getChart();
      int               size   = 0;

      for(typename chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getFiberDimension(*p_iter);
      }
      return size;
    };
    int sizeWithBC() const {
      return this->size();
    };
  public: // Restriction
    // Return a pointer to the entire contiguous storage array
    const values_type& restrict() {
      return this->_array;
    };
    // Return only the values associated to this point, not its closure
    const value_type *restrictPoint(const point_type& p) {
      return this->_array[p].v;
    };
    // Update only the values associated to this point, not its closure
    void updatePoint(const point_type& p, const value_type v[]) {
      for(int i = 0; i < fiberDim; ++i) {
        this->_array[p].v[i] = v[i];
      }
    };
    // Update only the values associated to this point, not its closure
    void updateAddPoint(const point_type& p, const value_type v[]) {
      for(int i = 0; i < fiberDim; ++i) {
        this->_array[p].v[i] += v[i];
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
      const typename atlas_type::chart_type& chart = this->_atlas->getChart();
      const values_type&                     array = this->_array;

      for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const point_type&                     point = *p_iter;
        const typename atlas_type::value_type dim   = this->_atlas->restrict(point)[0];

        if (dim != 0) {
          txt << "[" << this->commRank() << "]:   " << point << " dim " << dim << "  ";
          for(int i = 0; i < dim; i++) {
            txt << " " << array[point].v[i];
          }
          txt << std::endl;
        }
      }
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };
  // A Section is our most general construct (more general ones could be envisioned)
  //   The Atlas is a UniformSection of dimension 1 and value type Point
  //     to hold each fiber dimension and offsets into a contiguous patch array
  template<typename Point_, typename Value_>
  class Section : public ALE::ParallelObject {
  public:
    typedef Point_                                 point_type;
    typedef ALE::Point                             index_type;
    typedef UniformSection<point_type, index_type> atlas_type;
    typedef typename atlas_type::chart_type        chart_type;
    typedef Value_                                 value_type;
    typedef value_type *                           values_type;
  protected:
    Obj<atlas_type> _atlas;
    Obj<atlas_type> _atlasNew;
    values_type     _array;
  public:
    Section(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      this->_atlas    = new atlas_type(comm, debug);
      this->_atlasNew = NULL;
      this->_array    = NULL;
    };
    Section(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas), _atlasNew(NULL), _array(NULL) {};
    virtual ~Section() {
      if (!this->_array) {
        delete [] this->_array;
        this->_array = NULL;
      }
    };
  public:
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
    bool hasPoint(const point_type& point) {
      return this->_atlas->hasPoint(point);
    };
  public: // Accessors
    const Obj<atlas_type>& getAtlas() {return this->_atlas;};
    void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
    const Obj<atlas_type>& getNewAtlas() {return this->_atlasNew;};
    void setNewAtlas(const Obj<atlas_type>& atlas) {this->_atlasNew = atlas;};
  public: // Sizes
    void clear() {
      this->_atlas->clear(); 
      delete [] this->_array;
      this->_array = NULL;
    };
    int getFiberDimension(const point_type& p) const {
      return this->_atlas->restrictPoint(p)->prefix;
    };
    int getFiberDimension(const Obj<atlas_type>& atlas, const point_type& p) const {
      return atlas->restrictPoint(p)->prefix;
    };
    void setFiberDimension(const point_type& p, int dim) {
      const index_type idx(dim, -1);
      this->_atlas->addPoint(p);
      this->_atlas->updatePoint(p, &idx);
    };
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    };
    void addFiberDimension(const point_type& p, int dim) {
      if (this->_atlas->hasPoint(p)) {
        const index_type values(dim, 0);
        this->_atlas->updateAddPoint(p, &values);
      } else {
        this->setFiberDimension(p, dim);
      }
    };
    int size() const {
      const typename atlas_type::chart_type& points = this->_atlas->getChart();
      int size = 0;

      for(typename atlas_type::chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += std::max(0, this->getFiberDimension(*p_iter));
      }
      return size;
    };
    int sizeWithBC() const {
      const typename atlas_type::chart_type& points = this->_atlas->getChart();
      int size = 0;

      for(typename atlas_type::chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += std::abs(this->getFiberDimension(*p_iter));
      }
      return size;
    };
#if 0
    int size(const Obj<atlas_type>& atlas) {
      const typename atlas_type::chart_type& points = atlas->getChart();
      int size = 0;

      for(typename atlas_type::chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += std::max(0, this->getFiberDimension(atlas, *p_iter));
      }
      return size;
    };
#endif
  public: // Index retrieval
    const index_type& getIndex(const point_type& p) {
      return this->_atlas->restrictPoint(p)[0];
    };
    template<typename Numbering>
    const index_type getIndex(const point_type& p, const Obj<Numbering>& numbering) {
      return index_type(this->getFiberDimension(p), numbering->getIndex(p));
    };
  public: // Allocation
    void allocateStorage() {
      this->_array = new value_type[this->sizeWithBC()];
      PetscMemzero(this->_array, this->sizeWithBC() * sizeof(value_type));
    };
    void replaceStorage(const value_type *newArray) {
      delete [] this->_array;
      this->_array = newArray;
      this->_atlas    = this->_atlasNew;
      this->_atlasNew = NULL;
    };
    void addPoint(const point_type& point, const int dim) {
      if (dim == 0) return;
      if (this->_atlasNew.isNull()) {
        this->_atlasNew = new atlas_type(this->comm(), this->debug());
        this->_atlasNew->copy(this->_atlas);
      }
      const index_type idx(dim, -1);
      this->_atlasNew->addPoint(point);
      this->_atlasNew->updatePoint(point, &idx);
    };
  public: // Restriction and Update
    // Zero entries
    void zero() {
      memset(this->_array, 0, this->size()* sizeof(value_type));
    };
    // Return a pointer to the entire contiguous storage array
    const value_type *restrict() {
      return this->_array;
    };
    // Update the entire contiguous storage array
    void update(const value_type v[]) {
      const value_type *array = this->_array;
      const int         size  = this->size();

      for(int i = 0; i < size; i++) {
        array[i] = v[i];
      }
    };
    // Return only the values associated to this point, not its closure
    const value_type *restrictPoint(const point_type& p) {
      return &(this->_array[this->_atlas->restrictPoint(p)[0].index]);
    };
    // Update only the values associated to this point, not its closure
    void updatePoint(const point_type& p, const value_type v[]) {
      const index_type& idx = this->_atlas->restrictPoint(p)[0];
      value_type       *a   = &(this->_array[idx.index]);

      for(int i = 0; i < idx.prefix; ++i) {
        a[i] = v[i];
      }
    };
    // Update only the values associated to this point, not its closure
    void updateAddPoint(const point_type& p, const value_type v[]) {
      const index_type& idx = this->_atlas->restrictPoint(p)[0];
      value_type       *a   = &(this->_array[idx.index]);

      for(int i = 0; i < idx.prefix; ++i) {
        a[i] += v[i];
      }
    };
    void updatePointBC(const point_type& p, const value_type v[]) {
      const index_type& idx = this->_atlas->restrictPoint(p)[0];
      value_type       *a   = &(this->_array[idx.index]);

      for(int i = 0; i < std::abs(idx.prefix); ++i) {
        a[i] = v[i];
      }
    };
  public: // BC
    void copyBC(const Obj<Section>& section) {
      const typename atlas_type::chart_type& chart = this->_atlas->getChart();

      for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const index_type& idx = this->_atlas->restrictPoint(*p_iter)[0];

        if (idx.prefix < 0) {
          this->updatePointBC(*p_iter, section->restrictPoint(*p_iter));
        }
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
      const typename atlas_type::chart_type& chart = this->_atlas->getChart();
      const value_type                      *array = this->_array;

      for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const point_type& point = *p_iter;
        const index_type& idx   = this->_atlas->restrictPoint(point)[0];

        if (idx.prefix != 0) {
          txt << "[" << this->commRank() << "]:   " << point << " dim " << idx.prefix << " offset " << idx.index << "  ";
          for(int i = 0; i < std::abs(idx.prefix); i++) {
            txt << " " << array[idx.index+i];
          }
          txt << std::endl;
        }
      }
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };
  template<typename Sieve_, typename Value_ = int>
  class NumberingFactory : ALE::ParallelObject {
  public:
    typedef Sieve_                                          sieve_type;
    typedef typename sieve_type::point_type                 point_type;
    typedef Value_                                          value_type;
    typedef typename ALE::Sifter<int,point_type,point_type> send_overlap_type;
    typedef typename ALE::Sifter<point_type,int,point_type> recv_overlap_type;
  protected:
  protected:
    NumberingFactory(MPI_Comm comm, const int debug = 0) : ALE::ParallelObject(comm, debug) {};
  public:
    ~NumberingFactory() {};
  public:
    static const Obj<NumberingFactory>& singleton(MPI_Comm comm, const int debug, bool cleanup = false) {
      static Obj<NumberingFactory> *_singleton = NULL;

      if (cleanup) {
        if (debug) {std::cout << "Destroying NumberingFactory" << std::endl;}
        if (_singleton) {delete _singleton;}
        _singleton = NULL;
      } else if (_singleton == NULL) {
        if (debug) {std::cout << "Creating new NumberingFactory" << std::endl;}
        _singleton  = new Obj<NumberingFactory>();
        *_singleton = new NumberingFactory(comm, debug);
      }
      return *_singleton;
    };
  public:
    template<typename Atlas_>
    void orderPoint(const Obj<Atlas_>& atlas, const Obj<sieve_type>& sieve, const typename Atlas_::point_type& point, value_type& offset, value_type& bcOffset, const Obj<send_overlap_type>& sendOverlap = NULL) {
      const Obj<typename sieve_type::coneSequence>& cone = sieve->cone(point);
      typename sieve_type::coneSequence::iterator   end  = cone->end();
      typename Atlas_::value_type                   idx  = atlas->restrictPoint(point)[0];
      const value_type&                             dim  = idx.prefix;
      const typename Atlas_::value_type             defaultIdx(0, -1);

      if (atlas->getChart().count(point) == 0) {
        idx = defaultIdx;
      }
      if (idx.index == -1) {
        for(typename sieve_type::coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
          if (this->_debug > 1) {std::cout << "    Recursing to " << *c_iter << std::endl;}
          this->orderPoint(atlas, sieve, *c_iter, offset, bcOffset, sendOverlap);
        }
        if (dim > 0) {
          bool number = true;

          // Maybe use template specialization here
          if (!sendOverlap.isNull() && sendOverlap->capContains(point)) {
            const Obj<typename send_overlap_type::supportSequence>& ranks = sendOverlap->support(point);

            for(typename send_overlap_type::supportSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
              if (this->commRank() > *r_iter) {
                number = false;
                break;
              }
            }
          }
          if (number) {
            if (this->_debug > 1) {std::cout << "  Ordering point " << point << " at " << offset << std::endl;}
            idx.index = offset;
            atlas->updatePoint(point, &idx);
            offset += dim;
          } else {
            if (this->_debug > 1) {std::cout << "  Ignoring ghost point " << point << std::endl;}
          }
        } else if (dim < 0) {
          if (this->_debug > 1) {std::cout << "  Ordering boundary point " << point << " at " << bcOffset << std::endl;}
          idx.index = bcOffset;
          atlas->updatePoint(point, &idx);
          bcOffset += dim;
        }
      }
    };
    template<typename Atlas_>
    void orderPatch(const Obj<Atlas_>& atlas, const Obj<sieve_type>& sieve, const Obj<send_overlap_type>& sendOverlap = NULL, const value_type offset = 0, const value_type bcOffset = -2) {
      const typename Atlas_::chart_type& chart = atlas->getChart();
      int off   = offset;
      int bcOff = bcOffset;

      if (this->_debug > 1) {std::cout << "Ordering patch" << std::endl;}
      for(typename Atlas_::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        if (this->_debug > 1) {std::cout << "Ordering closure of point " << *p_iter << std::endl;}
        this->orderPoint(atlas, sieve, *p_iter, off, bcOff, sendOverlap);
      }
      for(typename Atlas_::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        typename Atlas_::value_type idx = atlas->restrictPoint(*p_iter)[0];
        const int&                  dim = idx.prefix;

        if (dim < 0) {
          if (this->_debug > 1) {std::cout << "Correcting boundary offset of point " << *p_iter << std::endl;}
          idx.index = offset - (idx.index+2);
          atlas->updatePoint(*p_iter, &idx);
        }
      }
    };
  };
  template<typename Sieve_>
  class Bundle : public ALE::ParallelObject {
  public:
    typedef Sieve_                                                    sieve_type;
    typedef typename sieve_type::point_type                           point_type;
    typedef typename ALE::Sifter<int, point_type, int>                label_type;
    typedef typename std::map<const std::string, Obj<label_type> >    labels_type;
    typedef typename label_type::supportSequence                      label_sequence;
    typedef UniformSection<MinimalArrow<point_type, point_type>, int> arrow_section_type;
    typedef std::map<std::string, Obj<arrow_section_type> >           arrow_sections_type;
    typedef Section<point_type, double>                               real_section_type;
    typedef std::map<std::string, Obj<real_section_type> >            real_sections_type;
    typedef Section<point_type, int>                                  int_section_type;
    typedef std::map<std::string, Obj<int_section_type> >             int_sections_type;
    typedef ALE::Point                                                index_type;
    typedef typename sieve_type::coneArray                            coneArray;
    typedef std::vector<index_type>                                   indexArray;
    typedef NumberingFactory<sieve_type>                              NumberingFactory;
    ///typedef NumberingFactory::numbering_type          numbering_type;
    ///typedef NumberingFactory::order_type              order_type;
    typedef typename ALE::Sifter<int,point_type,point_type>           send_overlap_type;
    typedef typename ALE::Sifter<point_type,int,point_type>           recv_overlap_type;
  protected:
    Obj<sieve_type>       _sieve;
    labels_type           _labels;
    int                   _maxHeight;
    int                   _maxDepth;
    arrow_sections_type   _arrowSections;
    real_sections_type    _realSections;
    int_sections_type     _intSections;
    Obj<indexArray>       _indexArray;
    Obj<NumberingFactory> _factory;
    // Work space
    Obj<std::set<point_type> > _modifiedPoints;
  public:
    Bundle(MPI_Comm comm, int debug = 0) : ALE::ParallelObject(comm, debug), _maxHeight(-1), _maxDepth(-1) {
      this->_indexArray     = new indexArray();
      this->_modifiedPoints = new std::set<point_type>();
      this->_factory        = NumberingFactory::singleton(this->comm(), this->debug());
    };
    Bundle(const Obj<sieve_type>& sieve) : ALE::ParallelObject(sieve->comm(), sieve->debug()), _sieve(sieve), _maxHeight(-1), _maxDepth(-1) {
      this->_indexArray     = new indexArray();
      this->_modifiedPoints = new std::set<point_type>();
      this->_factory        = NumberingFactory::singleton(this->comm(), this->debug());
    };
    virtual ~Bundle() {};
  public: // Verifiers
    bool hasLabel(const std::string& name) {
      if (this->_labels.find(name) != this->_labels.end()) {
        return true;
      }
      return false;
    };
    void checkLabel(const std::string& name) {
      if (!this->hasLabel(name)) {
        ostringstream msg;
        msg << "Invalid label name: " << name << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
  public: // Accessors
    const Obj<sieve_type>& getSieve() const {return this->_sieve;};
    void setSieve(const Obj<sieve_type>& sieve) {this->_sieve = sieve;};
    bool hasArrowSection(const std::string& name) const {
      return this->_arrowSections.find(name) != this->_arrowSections.end();
    };
    const Obj<arrow_section_type>& getArrowSection(const std::string& name) {
      if (!this->hasArrowSection(name)) {
        Obj<arrow_section_type> section = new arrow_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new arrow section: " << name << std::endl;}
        this->_arrowSections[name] = section;
      }
      return this->_arrowSections[name];
    };
    void setArrowSection(const std::string& name, const Obj<arrow_section_type>& section) {
      this->_arrowSections[name] = section;
    };
    Obj<std::set<std::string> > getArrowSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename arrow_sections_type::const_iterator s_iter = this->_arrowSections.begin(); s_iter != this->_arrowSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasRealSection(const std::string& name) const {
      return this->_realSections.find(name) != this->_realSections.end();
    };
    const Obj<real_section_type>& getRealSection(const std::string& name) {
      if (!this->hasRealSection(name)) {
        Obj<real_section_type> section = new real_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new real section: " << name << std::endl;}
        this->_realSections[name] = section;
      }
      return this->_realSections[name];
    };
    void setRealSection(const std::string& name, const Obj<real_section_type>& section) {
      this->_realSections[name] = section;
    };
    Obj<std::set<std::string> > getRealSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename real_sections_type::const_iterator s_iter = this->_realSections.begin(); s_iter != this->_realSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
    bool hasIntSection(const std::string& name) const {
      return this->_intSections.find(name) != this->_intSections.end();
    };
    const Obj<int_section_type>& getIntSection(const std::string& name) {
      if (!this->hasIntSection(name)) {
        Obj<int_section_type> section = new int_section_type(this->comm(), this->debug());

        section->setName(name);
        if (this->_debug) {std::cout << "Creating new int section: " << name << std::endl;}
        this->_intSections[name] = section;
      }
      return this->_intSections[name];
    };
    void setIntSection(const std::string& name, const Obj<int_section_type>& section) {
      this->_intSections[name] = section;
    };
    Obj<std::set<std::string> > getIntSections() const {
      Obj<std::set<std::string> > names = std::set<std::string>();

      for(typename int_sections_type::const_iterator s_iter = this->_intSections.begin(); s_iter != this->_intSections.end(); ++s_iter) {
        names->insert(s_iter->first);
      }
      return names;
    };
  public: // Labels
    int getValue (const Obj<label_type>& label, const point_type& point, const int defValue = 0) {
      const Obj<typename label_type::coneSequence>& cone = label->cone(point);

      if (cone->size() == 0) return defValue;
      return *cone->begin();
    };
    void setValue(const Obj<label_type>& label, const point_type& point, const int value) {
      label->setCone(value, point);
    };
    template<typename InputPoints>
    int getMaxValue (const Obj<label_type>& label, const Obj<InputPoints>& points, const int defValue = 0) {
      int maxValue = defValue;

      for(typename InputPoints::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        maxValue = std::max(maxValue, this->getValue(label, *p_iter, defValue));
      }
      return maxValue;
    };
    const Obj<label_type>& createLabel(const std::string& name) {
      this->_labels[name] = new label_type(this->comm(), this->debug());
      return this->_labels[name];
    };
    const Obj<label_type>& getLabel(const std::string& name) {
      this->checkLabel(name);
      return this->_labels[name];
    };
    const Obj<label_sequence>& getLabelStratum(const std::string& name, int value) {
      this->checkLabel(name);
      return this->_labels[name]->support(value);
    };
  public: // Stratification
    template<class InputPoints>
    void computeHeight(const Obj<label_type>& height, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points, int& maxHeight) {
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
      const Obj<label_type>& label = this->createLabel(std::string("height"));

      this->_maxHeight = -1;
      this->computeHeight(label, this->_sieve, this->_sieve->leaves(), this->_maxHeight);
    };
    int height() const {return this->_maxHeight;};
    int height(const point_type& point) {
      return this->getValue(this->_labels["height"], point, -1);
    };
    const Obj<label_sequence>& heightStratum(int height) {
      return this->getLabelStratum("height", height);
    };
    template<class InputPoints>
    void computeDepth(const Obj<label_type>& depth, const Obj<sieve_type>& sieve, const Obj<InputPoints>& points, int& maxDepth) {
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
      const Obj<label_type>& label = this->createLabel(std::string("depth"));

      this->_maxDepth = -1;
      this->computeDepth(label, this->_sieve, this->_sieve->roots(), this->_maxDepth);
    };
    int depth() const {return this->_maxDepth;};
    int depth(const point_type& point) {
      return this->getValue(this->_labels["depth"], point, -1);
    };
    const Obj<label_sequence>& depthStratum(int depth) {
      return this->getLabelStratum("depth", depth);
    };
#undef __FUNCT__
#define __FUNCT__ "Bundle::stratify"
    void stratify() {
      ALE_LOG_EVENT_BEGIN;
      this->computeHeights();
      this->computeDepths();
      ALE_LOG_EVENT_END;
    };
  public: // Size traversal
    template<typename Section_>
    int size(const Obj<Section_>& section, const point_type& p) {
      const typename Section_::chart_type&  chart   = section->getAtlas()->getChart();
      const Obj<coneArray>                  closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
      typename coneArray::iterator          end     = closure->end();
      int                                   size    = 0;

      for(typename coneArray::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
        if (chart.count(*c_iter)) {
          size += std::max(0, section->getFiberDimension(*c_iter));
        }
      }
      return size;
    };
    template<typename Section_>
    int sizeWithBC(const Obj<Section_>& section, const point_type& p) {
      const typename Section_::chart_type&  chart   = section->getAtlas()->getChart();
      const Obj<coneArray>                  closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
      typename coneArray::iterator          end     = closure->end();
      int                                   size    = 0;

      for(typename coneArray::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
        if (chart.count(*c_iter)) {
          size += std::abs(section->getFiberDimension(*c_iter));
        }
      }
      return size;
    };
  public: // Index traversal
    template<typename Section_>
    const Obj<indexArray>& getIndices(const Obj<Section_>& section, const point_type& p, const int level = -1) {
      this->_indexArray->clear();

      if (level == 0) {
        this->_indexArray->push_back(section->getIndex(p));
      } else if ((level == 1) || (this->height() == 1)) {
        const Obj<typename sieve_type::coneSequence>& cone = this->_sieve->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        this->_indexArray->push_back(section->getIndex(p));
        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          this->_indexArray->push_back(section->getIndex(*p_iter));
        }
      } else if (level == -1) {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          this->_indexArray->push_back(section->getIndex(*p_iter));
        }
      } else {
        throw ALE::Exception("Bundle has not yet implemented nCone");
      }
      return this->_indexArray;
    };
    template<typename Section_, typename Numbering>
    const Obj<indexArray>& getIndices(const Obj<Section_>& section, const point_type& p, const Obj<Numbering>& numbering, const int level = -1) {
      this->_indexArray->clear();

      if (level == 0) {
        this->_indexArray->push_back(section->getIndex(p, numbering));
      } else if ((level == 1) || (this->height() == 1)) {
        const Obj<typename sieve_type::coneSequence>& cone = this->_sieve->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        this->_indexArray->push_back(section->getIndex(p, numbering));
        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          this->_indexArray->push_back(section->getIndex(*p_iter, numbering));
        }
      } else if (level == -1) {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          this->_indexArray->push_back(this->getIndex(*p_iter, numbering));
        }
      } else {
        throw ALE::Exception("Bundle has not yet implemented nCone");
      }
      return this->_indexArray;
    };
  public: // Retrieval traversal
    // Return the values for the closure of this point
    //   use a smart pointer?
    template<typename Section_>
    const typename Section_::value_type *restrict(const Obj<Section_>& section, const point_type& p) {
      const int                             size   = this->sizeWithBC(section, p);
      typename Section_::value_type        *values = section->getRawArray(size);
      int                                   j      = -1;

      // We could actually ask for the height of the individual point
      if (this->height() < 2) {
        const int& dim = std::abs(section->getFiberDimension(p));
        const typename Section_::value_type *array = section->restrictPoint(p);

        for(int i = 0; i < dim; ++i) {
          values[++j] = array[i];
        }
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          const int& dim = std::abs(section->getFiberDimension(*p_iter));

          array = section->restrictPoint(*p_iter);
          for(int i = 0; i < dim; ++i) {
            values[++j] = array[i];
          }
        }
      } else {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          const int& dim = std::abs(section->getFiberDimension(*p_iter));
          const typename Section_::value_type *array = section->restrictPoint(*p_iter);

          for(int i = 0; i < dim; ++i) {
            values[++j] = array[i];
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
    template<typename Section_>
    void update(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updatePoint(p, &v[j]);
        j += std::abs(section->getFiberDimension(p));
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updatePoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      } else {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updatePoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      }
    };
    template<typename Section_>
    void updateAdd(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updateAddPoint(p, &v[j]);
        j += std::abs(section->getFiberDimension(p));
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updateAddPoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      } else {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updateAddPoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      }
    };
    template<typename Section_>
    void updateBC(const Obj<Section_>& section, const point_type& p, const typename Section_::value_type v[]) {
      int j = 0;

      if (this->height() < 2) {
        section->updateBCPoint(p, &v[j]);
        j += std::abs(section->getFiberDimension(p));
        const Obj<typename sieve_type::coneSequence>& cone = this->getSieve()->cone(p);

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != cone->end(); ++p_iter) {
          section->updateBCPoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      } else {
        const Obj<coneArray>         closure = ALE::Closure::closure(this, this->getArrowSection("orientation"), p);
        typename coneArray::iterator end     = closure->end();

        for(typename coneArray::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          section->updateBCPoint(*p_iter, &v[j]);
          j += std::abs(section->getFiberDimension(*p_iter));
        }
      }
    };
  public: // Allocation
    template<typename Section_>
    void allocate(const Obj<Section_>& section, const Obj<send_overlap_type>& sendOverlap = NULL) {
      bool doGhosts = !sendOverlap.isNull();

      this->_factory->orderPatch(section->getAtlas(), this->getSieve(), sendOverlap);
      if (doGhosts) {
        if (this->_debug > 1) {std::cout << "Ordering patch for ghosts" << std::endl;}
        const typename Section_::atlas_type::chart_type& points = section->getAtlas()->getChart();
        int offset = 0;

        for(typename Section_::atlas_type::chart_type::iterator point = points.begin(); point != points.end(); ++point) {
          const typename Section_::index_type& idx = section->getIndex(*point);

          offset = std::max(offset, idx.index + std::abs(idx.prefix));
        }
        this->_factory->orderPatch(section->getAtlas(), this->getSieve(), NULL, offset);
        if (offset != section->sizeWithBC()) throw ALE::Exception("Inconsistent array sizes in section");
      }
      section->allocateStorage();
    };
    template<typename Section_>
    void reallocate(const Obj<Section_>& section) {
      if (section->getNewAtlas().isNull()) return;
      // Since copy() preserves offsets, we must reinitialize them before ordering
      const Obj<typename Section_::atlas_type>&        newAtlas = section->getNewAtlas();
      const typename Section_::atlas_type::chart_type& chart    = newAtlas->getChart();
      int                                              newSize  = 0;
      typename Section_::index_type                    defaultIdx(0, -1);

      for(typename Section_::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        defaultIdx.prefix = newAtlas->restrictPoint(*c_iter)[0].prefix;
        newAtlas->updatePoint(*c_iter, &defaultIdx);
        newSize += defaultIdx.prefix;
      }
      this->_factory->orderPatch(newAtlas, this->getSieve());
      // Copy over existing values
      typename Section_::value_type *newArray = new typename Section_::value_type[newSize];

      chart = section->getAtlas()->getChart();
      for(typename Section_::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        const typename Section_::value_type *array  = section->restrictPoint(*c_iter);
        const int&                           dim    = std::abs(section->getFiberDimension(*c_iter));
        const int&                           offset = newAtlas->restrictPoint(*c_iter)[0].index;

        for(int i = 0; i < dim; ++i) {
          newArray[offset+i] = array[i];
        }
      }
      section->replaceStorage(newArray);
    };
  };
  // A Field combines several sections
  template<typename Patch_, typename Section_>
  class Field : public ALE::ParallelObject {
  public:
    typedef Patch_                                          patch_type;
    typedef Section_                                        section_type;
    typedef typename section_type::point_type               point_type;
    typedef std::map<patch_type, Obj<section_type> >        sheaf_type;
    typedef typename ALE::Sifter<int,point_type,point_type> send_overlap_type;
    typedef typename ALE::Sifter<point_type,int,point_type> recv_overlap_type;
  protected:
    sheaf_type _sheaf;
  public:
    Field(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {};
    virtual ~Field() {};
  public: // Verifiers
    void checkPatch(const patch_type& patch) const {
      if (this->_sheaf.find(patch) == this->_sheaf.end()) {
        ostringstream msg;
        msg << "Invalid field patch " << patch << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
    bool hasPatch(const patch_type& patch) {
      if (this->_sheaf.find(patch) == this->_sheaf.end()) {
        return false;
      }
      return true;
    };
  public: // Accessors
    Obj<section_type>& getSection(const patch_type& patch) {return this->_sheaf[patch];};
    void setSection(const patch_type& patch, const Obj<section_type>& section) {this->_sheaf[patch] = section;};
  };
  class Discretization : public ALE::ParallelObject {
  protected:
    std::map<int,int> _dim2dof;
    std::map<int,int> _dim2class;
  public:
    Discretization(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {};
    ~Discretization() {};
  public:
    const double *getQuadraturePoints() {return NULL;};
    const double *getQuadratureWeights() {return NULL;};
    const double *getBasis() {return NULL;};
    const double *getBasisDerivatives() {return NULL;};
    int  getNumDof(const int dim) {return this->_dim2dof[dim];};
    void setNumDof(const int dim, const int numDof) {this->_dim2dof[dim] = numDof;};
    int  getDofClass(const int dim) {return this->_dim2class[dim];};
    void setDofClass(const int dim, const int dofClass) {this->_dim2class[dim] = dofClass;};
  };
  class BoundaryCondition : public ALE::ParallelObject {
  protected:
    std::string _labelName;
    double    (*_func)(const double []);
  public:
    BoundaryCondition(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {};
    ~BoundaryCondition() {};
  public:
    const std::string& getLabelName() {return this->_labelName;};
    void setLabelName(const std::string& name) {this->_labelName = name;};
    void setFunction(double (*func)(const double [])) {this->_func = func;};
  public:
    double evaluate(const double coords[]) {return this->_func(coords);};
  };
  class Mesh : public Bundle<ALE::Sieve<int,int,int> > {
  public:
    typedef Bundle<ALE::Sieve<int,int,int> > base_type;
    typedef base_type::sieve_type            sieve_type;
    typedef sieve_type::point_type           point_type;
    typedef base_type::label_sequence        label_sequence;
    typedef base_type::real_section_type     real_section_type;
    ///typedef base_type::send_overlap_type              send_overlap_type;
    ///typedef base_type::recv_overlap_type              recv_overlap_type;
    ///typedef base_type::send_section_type              send_section_type;
    ///typedef base_type::recv_section_type              recv_section_type;
  protected:
    int                   _dim;
    ///Obj<NumberingFactory> _factory;
    // Discretization
    Obj<Discretization>    _discretization;
    Obj<BoundaryCondition> _boundaryCondition;
  public:
    Mesh(MPI_Comm comm, int dim, int debug = 0) : Bundle<ALE::Sieve<int,int,int> >(comm, debug), _dim(dim) {
      ///this->_factory = NumberingFactory::singleton(debug);
      this->_discretization = new Discretization(comm, debug);
      this->_boundaryCondition = new BoundaryCondition(comm, debug);
    };
  public: // Accessors
    int getDimension() const {return this->_dim;};
    void setDimension(const int dim) {this->_dim = dim;};
    ///const Obj<NumberingFactory>& getFactory() {return this->_factory;};
    const Obj<Discretization>&    getDiscretization() {return this->_discretization;};
    void setDiscretization(const Obj<Discretization>& discretization) {this->_discretization = discretization;};
    const Obj<BoundaryCondition>& getBoundaryCondition() {return this->_boundaryCondition;};
    void setBoundaryCondition(const Obj<BoundaryCondition>& boundaryCondition) {this->_boundaryCondition = boundaryCondition;};
  public: // Mesh geometry
    void computeTriangleGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const double    *coords = this->restrict(coordinates, e);
      const int        dim    = 2;
      double           invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        detJ = J[0]*J[3] - J[1]*J[2];
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
      }
    };
    void computeTetrahedronGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      const double *coords = this->restrict(coordinates, e);
      const int     dim    = 3;
      double        invDet;

      if (v0) {
        for(int d = 0; d < dim; d++) {
          v0[d] = coords[d];
        }
      }
      if (J) {
        for(int d = 0; d < dim; d++) {
          for(int f = 0; f < dim; f++) {
            J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
          }
        }
        detJ = J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
          J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
          J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
      }
      if (invJ) {
        invDet  = 1.0/detJ;
        invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
        invJ[0*3+1] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
        invJ[0*3+2] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        invJ[1*3+0] = invDet*(J[0*3+1]*J[2*3+2] - J[0*3+2]*J[2*3+1]);
        invJ[1*3+1] = invDet*(J[0*3+2]*J[2*3+0] - J[0*3+0]*J[2*3+2]);
        invJ[1*3+2] = invDet*(J[0*3+0]*J[2*3+1] - J[0*3+1]*J[2*3+0]);
        invJ[2*3+0] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
        invJ[2*3+1] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
        invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
      }
    };
    void computeElementGeometry(const Obj<real_section_type>& coordinates, const point_type& e, double v0[], double J[], double invJ[], double& detJ) {
      if (this->_dim == 2) {
        computeTriangleGeometry(coordinates, e, v0, J, invJ, detJ);
      } else if (this->_dim == 3) {
        computeTetrahedronGeometry(coordinates, e, v0, J, invJ, detJ);
      } else {
        throw ALE::Exception("Unsupport dimension for element geometry computation");
      }
    };
    double getMaxVolume() {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      double v0[3], J[9], invJ[9], detJ, maxVolume = 0.0;

      for(label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        maxVolume = std::max(maxVolume, detJ);
      }
      return maxVolume;
    };
    // Find the cell in which this point lies (stupid algorithm)
    point_type locatePoint_2D(const real_section_type::value_type point[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     embedDim    = 2;
      double v0[2], J[4], invJ[4], detJ;

      for(label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        double xi   = invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]);
        double eta  = invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]);

        if ((xi >= 0.0) && (eta >= 0.0) && (xi + eta <= 1.0)) {
          return *c_iter;
        }
      }
      throw ALE::Exception("Could not locate point");
    };
    //   Assume a simplex and 3D
    point_type locatePoint_3D(const real_section_type::value_type point[]) {
      const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
      const Obj<label_sequence>&    cells       = this->heightStratum(0);
      const int                     embedDim    = 3;
      double v0[3], J[9], invJ[9], detJ;

      for(label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
        this->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
        double xi   = invJ[0*embedDim+0]*(point[0] - v0[0]) + invJ[0*embedDim+1]*(point[1] - v0[1]) + invJ[0*embedDim+2]*(point[2] - v0[2]);
        double eta  = invJ[1*embedDim+0]*(point[0] - v0[0]) + invJ[1*embedDim+1]*(point[1] - v0[1]) + invJ[1*embedDim+2]*(point[2] - v0[2]);
        double zeta = invJ[2*embedDim+0]*(point[0] - v0[0]) + invJ[2*embedDim+1]*(point[1] - v0[1]) + invJ[2*embedDim+2]*(point[2] - v0[2]);

        if ((xi >= 0.0) && (eta >= 0.0) && (zeta >= 0.0) && (xi + eta + zeta <= 1.0)) {
          return *c_iter;
        }
      }
      throw ALE::Exception("Could not locate point");
    };
    point_type locatePoint(const real_section_type::value_type point[]) {
      if (this->_dim == 2) {
        return locatePoint_2D(point);
      } else if (this->_dim == 3) {
        return locatePoint_3D(point);
      } else {
        throw ALE::Exception("No point location for mesh dimension");
      }
    };
  public: // Discretization
    void setupField(const Obj<real_section_type>& s, const bool postponeGhosts = false) {
      const std::string& name = this->_boundaryCondition->getLabelName();

      for(int d = 0; d <= this->_dim; ++d) {
        s->setFiberDimension(this->depthStratum(d), this->_discretization->getNumDof(d));
      }
      if (!name.empty()) {
        const Obj<label_sequence>& boundary = this->getLabelStratum(name, 1);

        for(label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
          s->setFiberDimension(*e_iter, -this->_discretization->getNumDof(this->depth(*e_iter)));
        }
      }
      if (postponeGhosts) throw ALE::Exception("Not implemented yet");
      this->allocate(s);
      if (!name.empty()) {
        const Obj<real_section_type>& coordinates = this->getRealSection("coordinates");
        const Obj<label_sequence>&    boundary    = this->getLabelStratum(name, 1);

        for(label_sequence::iterator e_iter = boundary->begin(); e_iter != boundary->end(); ++e_iter) {
          const real_section_type::value_type *coords = coordinates->restrictPoint(*e_iter);
          const PetscScalar                    value  = this->_boundaryCondition->evaluate(coords);

          s->updatePointBC(*e_iter, &value);
        }
      }
    };
  public:
    void view(const std::string& name, MPI_Comm comm = MPI_COMM_NULL) {
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
      }
      if (name == "") {
        PetscPrintf(comm, "viewing a Mesh\n");
      } else {
        PetscPrintf(comm, "viewing Mesh '%s'\n", name.c_str());
      }
      this->getSieve()->view("mesh sieve", comm);
      Obj<std::set<std::string> > sections = this->getRealSections();

      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getRealSection(*name)->view(*name);
      }
      sections = this->getIntSections();
      for(std::set<std::string>::iterator name = sections->begin(); name != sections->end(); ++name) {
        this->getIntSection(*name)->view(*name);
      }
    };
    template<typename value_type>
    static std::string printMatrix(const std::string& name, const int rows, const int cols, const value_type matrix[], const int rank = -1)
    {
      ostringstream output;
      ostringstream rankStr;

      if (rank >= 0) {
        rankStr << "[" << rank << "]";
      }
      output << rankStr.str() << name << " = " << std::endl;
      for(int r = 0; r < rows; r++) {
        if (r == 0) {
          output << rankStr.str() << " /";
        } else if (r == rows-1) {
          output << rankStr.str() << " \\";
        } else {
          output << rankStr.str() << " |";
        }
        for(int c = 0; c < cols; c++) {
          output << " " << matrix[r*cols+c];
        }
        if (r == 0) {
          output << " \\" << std::endl;
        } else if (r == rows-1) {
          output << " /" << std::endl;
        } else {
          output << " |" << std::endl;
        }
      }
      return output.str();
    };
  };
}
}

#endif
