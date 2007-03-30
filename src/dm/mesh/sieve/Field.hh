#ifndef included_ALE_Field_hh
#define included_ALE_Field_hh

#ifndef  included_ALE_Sieve_hh
#include <Sieve.hh>
#endif

#ifndef  included_ALE_SieveAlgorithms_hh
#include <SieveAlgorithms.hh>
#endif

extern "C" PetscMPIInt Petsc_DelTag(MPI_Comm comm,PetscMPIInt keyval,void* attr_val,void* extra_state);

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
    void addPoint(const std::set<point_type>& points) {
      this->_chart.insert(points.begin(), points.end());
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
    const chart_type& getChart() {return this->_atlas->getChart();};
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
    void replaceStorage(value_type *newArray) {
      delete [] this->_array;
      this->_array    = newArray;
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
    void orderPoints(const Obj<atlas_type>& atlas){
      const typename atlas_type::chart_type& chart = atlas->getChart();
      int offset = 0;

      for(typename atlas_type::chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        typename atlas_type::value_type idx = atlas->restrictPoint(*c_iter)[0];
        const int&                      dim = idx.prefix;

        idx.index = offset;
        atlas->updatePoint(*c_iter, &idx);
        offset += dim;
      }
    };
    void allocatePoint() {
      this->orderPoints(this->_atlas);
      this->allocateStorage();
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
  // A Field combines several sections
  template<typename Overlap_, typename Patch_, typename Section_>
  class Field : public ALE::ParallelObject {
  public:
    typedef Overlap_                                 overlap_type;
    typedef Patch_                                   patch_type;
    typedef Section_                                 section_type;
    typedef typename section_type::point_type        point_type;
    typedef typename section_type::chart_type        chart_type;
    typedef typename section_type::value_type        value_type;
    typedef std::map<patch_type, Obj<section_type> > sheaf_type;
    typedef enum {SEND, RECEIVE}                     request_type;
    typedef std::map<patch_type, MPI_Request>        requests_type;
  protected:
    sheaf_type    _sheaf;
    int           _tag;
    MPI_Datatype  _datatype;
    requests_type _requests;
  public:
    Field(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      this->_tag      = this->getNewTag();
      this->_datatype = this->getMPIDatatype();
    };
    Field(MPI_Comm comm, const int tag, const int debug) : ParallelObject(comm, debug), _tag(tag) {
      this->_datatype = this->getMPIDatatype();
    };
    virtual ~Field() {};
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
      if (this->debug()) {
        std::cout << "[" << this->commRank() << "]Got new tag " << tagvalp[0] << std::endl;
      }
      return tagvalp[0]--;
    };
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
    int getTag() const {return this->_tag;};
    void setTag(const int tag) {this->_tag = tag;};
    Obj<section_type>& getSection(const patch_type& patch) {
      if (this->_sheaf.find(patch) == this->_sheaf.end()) {
        this->_sheaf[patch] = new section_type(this->comm(), this->debug());
      }
      return this->_sheaf[patch];
    };
    void setSection(const patch_type& patch, const Obj<section_type>& section) {this->_sheaf[patch] = section;};
    const sheaf_type& getPatches() {
      return this->_sheaf;
    };
    void clear() {
      for(typename sheaf_type::const_iterator p_iter = this->_sheaf.begin(); p_iter != this->_sheaf.end(); ++p_iter) {
        p_iter->second->clear();
      }
    };
  public: //  Adapter
    template<typename Topology_>
    void setTopology(const Obj<Topology_>& topology) {
      const typename Topology_::sheaf_type& patches = topology->getPatches();

      for(typename Topology_::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        int                      rank    = p_iter->first;
        const Obj<section_type>& section = this->getSection(rank);
        const Obj<typename Topology_::sieve_type::baseSequence>& base = p_iter->second->base();

        for(typename Topology_::sieve_type::baseSequence::iterator b_iter = base->begin(); b_iter != base->end(); ++b_iter) {
          section->setFiberDimension(*b_iter, 1);
        }
      }
    };
    void allocate() {
      for(typename sheaf_type::const_iterator p_iter = this->_sheaf.begin(); p_iter != this->_sheaf.end(); ++p_iter) {
        p_iter->second->allocatePoint();
      }
    };
  public: // Communication
    void construct(const int size) {
      const sheaf_type& patches = this->getPatches();

      for(typename sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        const patch_type         rank    = p_iter->first;
        const Obj<section_type>& section = this->getSection(rank);
        const chart_type&        chart   = section->getChart();

        for(typename chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
          section->setFiberDimension(*c_iter, size);
        }
      }
    };
    template<typename Sizer>
    void construct(const Obj<Sizer>& sizer) {
      const sheaf_type& patches = this->getPatches();

      for(typename sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        const patch_type         rank    = p_iter->first;
        const Obj<section_type>& section = this->getSection(rank);
        const chart_type&        chart   = section->getChart();
    
        for(typename chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
          section->setFiberDimension(*c_iter, *(sizer->getSection(rank)->restrictPoint(*c_iter)));
        }
      }
    };
    void constructCommunication(const request_type& requestType) {
      const sheaf_type& patches = this->getPatches();

      for(typename sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        const patch_type         patch   = p_iter->first;
        const Obj<section_type>& section = this->getSection(patch);
        MPI_Request              request;

        if (requestType == RECEIVE) {
          if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Receiving data(" << section->size() << ") from " << patch << " tag " << this->_tag << std::endl;}
          MPI_Recv_init((void *) section->restrict(), section->size(), this->_datatype, patch, this->_tag, this->comm(), &request);
        } else {
          if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Sending data (" << section->size() << ") to " << patch << " tag " << this->_tag << std::endl;}
          MPI_Send_init((void *) section->restrict(), section->size(), this->_datatype, patch, this->_tag, this->comm(), &request);
        }
        this->_requests[patch] = request;
      }
    };
    void startCommunication() {
      const sheaf_type& patches = this->getPatches();

      for(typename sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        MPI_Request request = this->_requests[p_iter->first];

        MPI_Start(&request);
      }
    };
    void endCommunication() {
      const sheaf_type& patches = this->getPatches();
      MPI_Status status;

      for(typename sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        MPI_Request request = this->_requests[p_iter->first];

        MPI_Wait(&request, &status);
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
          txt << "viewing a Field" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing Field '" << name << "'" << std::endl;
        }
      }
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
      for(typename sheaf_type::const_iterator p_iter = this->_sheaf.begin(); p_iter != this->_sheaf.end(); ++p_iter) {
        ostringstream txt1;

        txt1 << "[" << this->commRank() << "]: Patch " << p_iter->first << std::endl;
        PetscSynchronizedPrintf(comm, txt1.str().c_str());
        PetscSynchronizedFlush(comm);
        p_iter->second->view("field section", comm);
      }
    };
  };
}
}

#endif
