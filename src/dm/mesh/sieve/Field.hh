#ifndef included_ALE_Field_hh
#define included_ALE_Field_hh

#ifndef  included_ALE_SieveAlgorithms_hh
#include <SieveAlgorithms.hh>
#endif

extern "C" PetscMPIInt Mesh_DelTag(MPI_Comm comm,PetscMPIInt keyval,void* attr_val,void* extra_state);

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
    ConstantSection(MPI_Comm comm, const value_type& value, const int debug) : ParallelObject(comm, debug), _value(value), _defaultValue(value) {};
    ConstantSection(MPI_Comm comm, const value_type& value, const value_type& defaultValue, const int debug) : ParallelObject(comm, debug), _value(value), _defaultValue(defaultValue) {};
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
    void updatePointAll(const point_type& p, const value_type v[]) {
      this->updatePoint(p, v);
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
      values_type&                           array = this->_array;

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
      if (chart.size() == 0) {
        txt << "[" << this->commRank() << "]: empty" << std::endl;
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
      if (this->_array) {
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
    const chart_type& getChart() const {return this->_atlas->getChart();};
  public: // BC
    // Returns the number of constraints on a point
    const int getConstraintDimension(const point_type& p) const {
      return std::max(0, -this->_atlas->restrictPoint(p)->prefix);
    };
    // Set the number of constraints on a point
    //   We only allow the entire point to be constrained, so these will be the
    //   only dofs on the point
    void setConstraintDimension(const point_type& p, const int numConstraints) {
      this->setFiberDimension(p, -numConstraints);
    };
    void copyBC(const Obj<Section>& section) {
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        if (this->getConstraintDimension(*p_iter)) {
          this->updatePointBC(*p_iter, section->restrictPoint(*p_iter));
        }
      }
    };
    void defaultConstraintDof() {};
  public: // Sizes
    void clear() {
      this->_atlas->clear(); 
      delete [] this->_array;
      this->_array = NULL;
    };
    // Return the total number of dofs on the point (free and constrained)
    int getFiberDimension(const point_type& p) const {
      return std::abs(this->_atlas->restrictPoint(p)->prefix);
    };
    int getFiberDimension(const Obj<atlas_type>& atlas, const point_type& p) const {
      return std::abs(atlas->restrictPoint(p)->prefix);
    };
    // Set the total number of dofs on the point (free and constrained)
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
    // Return the number of constrained dofs on this point
    //   If constrained, this is equal to the fiber dimension
    //   Otherwise, 0
    int getConstrainedFiberDimension(const point_type& p) const {
      return std::max(0, this->_atlas->restrictPoint(p)->prefix);
    };
    // Return the total number of free dofs
    int size() const {
      const chart_type& points = this->getChart();
      int size = 0;

      for(typename chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getConstrainedFiberDimension(*p_iter);
      }
      return size;
    };
    // Return the total number of dofs (free and constrained)
    int sizeWithBC() const {
      const chart_type& points = this->getChart();
      int size = 0;

      for(typename chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getFiberDimension(*p_iter);
      }
      return size;
    };
  public: // Index retrieval
    const int& getIndex(const point_type& p) {
      return this->_atlas->restrictPoint(p)->index;
    };
    void setIndex(const point_type& p, const int& index) {
      ((typename atlas_type::value_type *) this->_atlas->restrictPoint(p))->index = index;
    };
    void setIndexBC(const point_type& p, const int& index) {
      this->setIndex(p, index);
    };
    void getIndices(const point_type& p, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false) {
      this->getIndices(p, this->getIndex(p), indices, indx, orientation, freeOnly);
    };
    template<typename Order_>
    void getIndices(const point_type& p, const Obj<Order_>& order, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false) {
      this->getIndices(p, order->getIndex(p), indices, indx, orientation, freeOnly);
    };
    void getIndices(const point_type& p, const int start, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false) {
      const int& dim   = this->getFiberDimension(p);
      const int& cDim  = this->getConstraintDimension(p);
      const int  end   = start + dim;

      if (!cDim) {
        if (orientation >= 0) {
          for(int i = start; i < end; ++i) {
            indices[(*indx)++] = i;
          }
        } else {
          for(int i = end-1; i >= start; --i) {
            indices[(*indx)++] = i;
          }
        }
      } else {
        if (!freeOnly) {
          for(int i = start; i < end; ++i) {
            indices[(*indx)++] = -1;
          }
        }
      }
    };
  public: // Allocation
    void allocateStorage() {
      const int totalSize = this->sizeWithBC();

      this->_array = new value_type[totalSize];
      PetscMemzero(this->_array, totalSize * sizeof(value_type));
    };
    void replaceStorage(value_type *newArray) {
      delete [] this->_array;
      this->_array    = newArray;
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
    template<typename Sequence>
    void addPoints(const Obj<Sequence>& points, const int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->addPoint(*p_iter, dim);
      }
    };
    void orderPoints(const Obj<atlas_type>& atlas){
      const chart_type& chart    = this->getChart();
      int               offset   = 0;
      int               bcOffset = this->size();

      for(typename chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        typename atlas_type::value_type idx  = atlas->restrictPoint(*c_iter)[0];
        const int&                      dim  = idx.prefix;

        if (dim >= 0) {
          idx.index = offset;
          atlas->updatePoint(*c_iter, &idx);
          offset += dim;
        } else {
          idx.index = bcOffset;
          atlas->updatePoint(*c_iter, &idx);
          bcOffset -= dim;
        }
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
    // Return the free values on a point
    const value_type *restrictPoint(const point_type& p) {
      return &(this->_array[this->_atlas->restrictPoint(p)[0].index]);
    };
    // Update the free values on a point
    void updatePoint(const point_type& p, const value_type v[], const int orientation = 1) {
      const index_type& idx = this->_atlas->restrictPoint(p)[0];
      value_type       *a   = &(this->_array[idx.index]);

      if (orientation >= 0) {
        for(int i = 0; i < idx.prefix; ++i) {
          a[i] = v[i];
        }
      } else {
        const int last = idx.prefix-1;

        for(int i = 0; i < idx.prefix; ++i) {
          a[i] = v[last-i];
        }
      }
    };
    // Update the free values on a point
    void updateAddPoint(const point_type& p, const value_type v[], const int orientation = 1) {
      const index_type& idx = this->_atlas->restrictPoint(p)[0];
      value_type       *a   = &(this->_array[idx.index]);

      if (orientation >= 0) {
        for(int i = 0; i < idx.prefix; ++i) {
          a[i] += v[i];
        }
      } else {
        const int last = idx.prefix-1;

        for(int i = 0; i < idx.prefix; ++i) {
          a[i] += v[last-i];
        }
      }
    };
    // Update only the constrained dofs on a point
    void updatePointBC(const point_type& p, const value_type v[], const int orientation = 1) {
      const index_type& idx = this->_atlas->restrictPoint(p)[0];
      const int         dim = -idx.prefix;
      value_type       *a   = &(this->_array[idx.index]);

      if (orientation >= 0) {
        for(int i = 0; i < dim; ++i) {
          a[i] = v[i];
        }
      } else {
        const int last = dim-1;

        for(int i = 0; i < dim; ++i) {
          a[i] = v[last-i];
        }
      }
    };
    // Update all dofs on a point (free and constrained)
    void updatePointAll(const point_type& p, const value_type v[], const int orientation = 1) {
      const index_type& idx = this->_atlas->restrictPoint(p)[0];
      const int         dim = std::abs(idx.prefix);
      value_type       *a   = &(this->_array[idx.index]);

      if (orientation >= 0) {
        for(int i = 0; i < dim; ++i) {
          a[i] = v[i];
        }
      } else {
        const int last = dim-1;

        for(int i = 0; i < dim; ++i) {
          a[i] = v[last-i];
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
      if(rank == 0) {
        txt << "viewing Section " << this->getName() << std::endl;
        if (name != "") {
          txt << ": " << name << "'";
        }
        txt << std::endl;
      }
      const typename atlas_type::chart_type& chart = this->_atlas->getChart();
      const value_type                      *array = this->_array;

      for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const point_type& point = *p_iter;
        const index_type& idx   = this->_atlas->restrictPoint(point)[0];
        const int         dim   = this->getFiberDimension(point);

        if (idx.prefix != 0) {
          txt << "[" << this->commRank() << "]:   " << point << " dim " << idx.prefix << " offset " << idx.index << "  ";
          for(int i = 0; i < dim; i++) {
            txt << " " << array[idx.index+i];
          }
          txt << std::endl;
        }
      }
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };
  // GeneralSection will support BC on a subset of unknowns on a point
  //   We make a separate constraint Atlas to mark constrained dofs on a point
  //   Storage will be contiguous by node, just as in Section
  //     This allows fast restrict(p)
  //     Then update() is accomplished by skipping constrained unknowns
  //     We must eliminate restrict() since it does not correspond to the constrained system
  //   Numbering will have to be rewritten to calculate correct mappings
  //     I think we can just generate multiple tuples per point
  template<typename Point_, typename Value_>
  class GeneralSection : public ALE::ParallelObject {
  public:
    typedef Point_                                 point_type;
    typedef ALE::Point                             index_type;
    typedef UniformSection<point_type, index_type> atlas_type;
    typedef Section<point_type, int>               bc_type;
    typedef typename atlas_type::chart_type        chart_type;
    typedef Value_                                 value_type;
    typedef value_type *                           values_type;
  protected:
    Obj<atlas_type> _atlas;
    Obj<atlas_type> _atlasNew;
    values_type     _array;
    Obj<bc_type>    _bc;
  public:
    GeneralSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      this->_atlas    = new atlas_type(comm, debug);
      this->_atlasNew = NULL;
      this->_array    = NULL;
      this->_bc       = new bc_type(comm, debug);
    };
    GeneralSection(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas), _atlasNew(NULL), _array(NULL) {
      this->_bc       = new bc_type(comm, debug);
    };
    virtual ~GeneralSection() {
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
    const Obj<atlas_type>& getAtlas() const {return this->_atlas;};
    void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
    const Obj<atlas_type>& getNewAtlas() const {return this->_atlasNew;};
    void setNewAtlas(const Obj<atlas_type>& atlas) {this->_atlasNew = atlas;};
    const Obj<bc_type>& getBC() const {return this->_bc;};
    void setBC(const Obj<bc_type>& bc) {this->_bc = bc;};
    const chart_type& getChart() const {return this->_atlas->getChart();};
  public: // BC
    // Returns the number of constraints on a point
    const int getConstraintDimension(const point_type& p) const {
      if (!this->_bc->hasPoint(p)) return 0;
      return this->_bc->getFiberDimension(p);
    };
    // Set the number of constraints on a point
    void setConstraintDimension(const point_type& p, const int numConstraints) {
      this->_bc->setFiberDimension(p, numConstraints);
    };
    // Return the local dofs which are constrained on a point
    const int *getConstraintDof(const point_type& p) {
      return this->_bc->restrictPoint(p);
    };
    // Set the local dofs which are constrained on a point
    void setConstraintDof(const point_type& p, const int dofs[]) {
      this->_bc->updatePoint(p, dofs);
    };
    void copyBC(const Obj<GeneralSection>& section) {
      this->setBC(section->getBC());
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        if (this->getConstraintDimension(*p_iter)) {
          this->updatePointBC(*p_iter, section->restrictPoint(*p_iter));
        }
      }
    };
    void defaultConstraintDof() {
      const chart_type& chart = this->getChart();
      int size = 0;

      for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        size = std::max(size, this->getConstraintDimension(*p_iter));
      }
      int *dofs = new int[size];
      for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const int cDim = this->getConstraintDimension(*p_iter);

        if (cDim) {
          for(int d = 0; d < cDim; ++d) {
            dofs[d] = d;
          }
          this->_bc->updatePoint(*p_iter, dofs);
        }
      }
    };
  public: // Sizes
    void clear() {
      this->_atlas->clear(); 
      delete [] this->_array;
      this->_array = NULL;
      this->_bc->clear(); 
    };
    // Return the total number of dofs on the point (free and constrained)
    int getFiberDimension(const point_type& p) const {
      return this->_atlas->restrictPoint(p)->prefix;
    };
    int getFiberDimension(const Obj<atlas_type>& atlas, const point_type& p) const {
      return atlas->restrictPoint(p)->prefix;
    };
    // Set the total number of dofs on the point (free and constrained)
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
    // Return the number of constrained dofs on this point
    int getConstrainedFiberDimension(const point_type& p) const {
      return this->getFiberDimension(p) - this->getConstraintDimension(p);
    };
    // Return the total number of free dofs
    int size() const {
      const chart_type& points = this->getChart();
      int               size   = 0;

      for(typename chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getConstrainedFiberDimension(*p_iter);
      }
      return size;
    };
    // Return the total number of dofs (free and constrained)
    int sizeWithBC() const {
      const chart_type& points = this->getChart();
      int               size   = 0;

      for(typename chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getFiberDimension(*p_iter);
      }
      return size;
    };
  public: // Index retrieval
    const int& getIndex(const point_type& p) {
      return this->_atlas->restrictPoint(p)->index;
    };
    void setIndex(const point_type& p, const int& index) {
      ((typename atlas_type::value_type *) this->_atlas->restrictPoint(p))->index = index;
    };
    void setIndexBC(const point_type& p, const int& index) {};
    void getIndices(const point_type& p, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false) {
      this->getIndices(p, this->getIndex(p), indices, indx, orientation, freeOnly);
    };
    template<typename Order_>
    void getIndices(const point_type& p, const Obj<Order_>& order, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false) {
      this->getIndices(p, order->getIndex(p), indices, indx, orientation, freeOnly);
    };
    void getIndices(const point_type& p, const int start, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false) {
      const int& dim   = this->getFiberDimension(p);
      const int& cDim  = this->getConstraintDimension(p);
      const int  end   = start + dim;

      if (!cDim) {
        if (orientation >= 0) {
          for(int i = start; i < end; ++i) {
            indices[(*indx)++] = i;
          }
        } else {
          for(int i = end-1; i >= start; --i) {
            indices[(*indx)++] = i;
          }
        }
      } else {
        const typename bc_type::value_type *cDof = this->getConstraintDof(p);
        int                                 cInd = 0;

        if (orientation >= 0) {
          for(int i = start, k = 0; k < dim; ++k) {
            if ((cInd < cDim) && (k == cDof[cInd])) {
              if (!freeOnly) indices[(*indx)++] = -(k+1);
              ++cInd;
            } else {
              indices[(*indx)++] = i++;
            }
          }
        } else {
          const int tEnd = start + this->getConstrainedFiberDimension(p);

          for(int i = tEnd-1, k = 0; k < dim; ++k) {
            if ((cInd < cDim) && (k == cDof[cInd])) {
              if (!freeOnly) indices[(*indx)++] = -(dim-k+1);
              ++cInd;
            } else {
              indices[(*indx)++] = i--;
            }
          }
        }
      }
    };
  public: // Allocation
    void allocateStorage() {
      const int totalSize = this->sizeWithBC();

      this->_array = new value_type[totalSize];
      PetscMemzero(this->_array, totalSize * sizeof(value_type));
      this->_bc->allocatePoint();
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
      const chart_type& chart  = this->getChart();
      int               offset = 0;

      for(typename chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
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
      //memset(this->_array, 0, this->sizeWithBC()* sizeof(value_type));
      const chart_type& chart = this->getChart();

      for(typename chart_type::iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        value_type *array = (value_type *) this->restrictPoint(*c_iter);
        const int&  dim   = this->getFiberDimension(*c_iter);
        const int&  cDim  = this->getConstraintDimension(*c_iter);

        if (!cDim) {
          memset(array, 0, dim * sizeof(value_type));
        } else {
          const typename bc_type::value_type *cDof = this->getConstraintDof(*c_iter);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = 0.0;
          }
        }
      }
    };
    // Return the free values on a point
    const value_type *restrict() const {
      return this->_array;
    };
    // Return the free values on a point
    const value_type *restrictPoint(const point_type& p) const {
      return &(this->_array[this->_atlas->restrictPoint(p)[0].index]);
    };
    // Update the free values on a point
    void updatePoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  dim   = this->getFiberDimension(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          for(int i = 0; i < dim; ++i) {
            array[i] = v[i];
          }
        } else {
          const int last = dim-1;

          for(int i = 0; i < dim; ++i) {
            array[i] = v[last-i];
          }
        }
      } else {
        const typename bc_type::value_type *cDof = this->getConstraintDof(p);
        int                                 cInd = 0;

        if (orientation >= 0) {
          for(int i = 0, k = -1; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = v[++k];
          }
        } else {
          const int tDim = this->getConstrainedFiberDimension(p);

          for(int i = 0, k = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = v[tDim-k];
            ++k;
          }
        }
      }
    };
    // Update the free values on a point
    void updateAddPoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  dim   = this->getFiberDimension(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          for(int i = 0; i < dim; ++i) {
            array[i] += v[i];
          }
        } else {
          const int last = dim-1;

          for(int i = 0; i < dim; ++i) {
            array[i] += v[last-i];
          }
        }
      } else {
        const typename bc_type::value_type *cDof = this->getConstraintDof(p);
        int                                 cInd = 0;

        if (orientation >= 0) {
          for(int i = 0, k = -1; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] += v[++k];
          }
        } else {
          const int tDim = this->getConstrainedFiberDimension(p);

          for(int i = 0, k = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] += v[tDim-k];
            ++k;
          }
        }
      }
    };
    // Update only the constrained dofs on a point
    void updatePointBC(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  dim   = this->getFiberDimension(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (cDim) {
        const typename bc_type::value_type *cDof = this->getConstraintDof(p);
        int                                 cInd = 0;

        for(int i = 0, k = 0; i < dim; ++i) {
          if (cInd == cDim) break;
          if (i == cDof[cInd]) {
            array[i] = v[k];
            ++cInd;
            ++k;
          }
        }
      }
    };
    // Update all dofs on a point (free and constrained)
    void updatePointAll(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  dim   = this->getFiberDimension(p);

      if (orientation >= 0) {
        for(int i = 0; i < dim; ++i) {
          array[i] = v[i];
        }
      } else {
        const int last = dim-1;

        for(int i = 0; i < dim; ++i) {
          array[i] = v[last-i];
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
          txt << "viewing a GeneralSection" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing GeneralSection '" << name << "'" << std::endl;
        }
      }
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const value_type *array = this->restrictPoint(*p_iter);
        const int&        dim   = this->getFiberDimension(*p_iter);

        if (dim != 0) {
          txt << "[" << this->commRank() << "]:   " << *p_iter << " dim " << dim << " offset " << this->_atlas->restrictPoint(*p_iter)->index << "  ";
          for(int i = 0; i < dim; i++) {
            txt << " " << array[i];
          }
          const int& dim = this->getConstraintDimension(*p_iter);

          if (dim) {
            const typename bc_type::value_type *bcArray = this->_bc->restrictPoint(*p_iter);

            txt << " constrained";
            for(int i = 0; i < dim; ++i) {
              txt << " " << bcArray[i];
            }
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
        tagvalp = (int *) malloc(sizeof(int));
        MPI_Keyval_create(MPI_NULL_COPY_FN, Mesh_DelTag, &tagKeyval, (void *) NULL);
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

namespace ALECompat {
  namespace New {
    using ALE::Obj;
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
    chart_type         _emptyChart;
    value_type         _value;
    value_type         _defaultValue;
  public:
    NewConstantSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _defaultValue(0) {
      this->_topology = new topology_type(comm, debug);
    };
    NewConstantSection(const Obj<topology_type>& topology) : ParallelObject(topology->comm(), topology->debug()), _topology(topology) {};
    NewConstantSection(const Obj<topology_type>& topology, const value_type& value) : ParallelObject(topology->comm(), topology->debug()), _topology(topology), _value(value), _defaultValue(value) {};
    NewConstantSection(const Obj<topology_type>& topology, const value_type& value, const value_type& defaultValue) : ParallelObject(topology->comm(), topology->debug()), _topology(topology), _value(value), _defaultValue(defaultValue) {};
  public: // Verifiers
    void checkPatch(const patch_type& patch) const {
      this->_topology->checkPatch(patch);
      if (this->_atlas.find(patch) == this->_atlas.end()) {
        ostringstream msg;
        msg << "Invalid atlas patch " << patch << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
    void checkPoint(const patch_type& patch, const point_type& point) const {
      this->checkPatch(patch);
      if (this->_atlas.find(patch)->second.find(point) == this->_atlas.find(patch)->second.end()) {
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
    bool hasPatch(const patch_type& patch) {
      if (this->_atlas.find(patch) == this->_atlas.end()) {
        return false;
      }
      return true;
    };
    bool hasPoint(const patch_type& patch, const point_type& point) const {
      this->checkPatch(patch);
      return this->_atlas.find(patch)->second.count(point) > 0;
    };
    bool hasPoint(const point_type& point) const {
      this->checkPatch(0);
      return this->_atlas.find(0)->second.count(point) > 0;
    };
  public: // Accessors
    const Obj<topology_type>& getTopology() const {return this->_topology;};
    void setTopology(const Obj<topology_type>& topology) {this->_topology = topology;};
    const chart_type& getPatch(const patch_type& patch) {
      if (this->hasPatch(patch)) {
        return this->_atlas[patch];
      }
      return this->_emptyChart;
    };
    void updatePatch(const patch_type& patch, const point_type& point) {
      this->_atlas[patch].insert(point);
    };
    template<typename Points>
    void updatePatch(const patch_type& patch, const Obj<Points>& points) {
      this->_atlas[patch].insert(points->begin(), points->end());
    };
    value_type getDefaultValue() {return this->_defaultValue;};
    void setDefaultValue(const value_type value) {this->_defaultValue = value;};
  public: // Sizes
    void clear() {
      this->_atlas.clear(); 
    };
    int getFiberDimension(const patch_type& patch, const point_type& p) const {
      if (this->hasPoint(patch, p)) return 1;
      return 0;
    };
    void setFiberDimension(const patch_type& patch, const point_type& p, int dim) {
      this->checkDimension(dim);
      this->updatePatch(patch, p);
    };
    template<typename Sequence>
    void setFiberDimension(const patch_type& patch, const Obj<Sequence>& points, int dim) {
      for(typename topology_type::label_sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(patch, *p_iter, dim);
      }
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
      this->setFiberDimension(patch, this->_topology->getLabelStratum(patch, "depth", depth), dim);
    };
    void setFiberDimensionByHeight(const patch_type& patch, int height, int dim) {
      this->setFiberDimension(patch, this->_topology->getLabelStratum(patch, "height", height), dim);
    };
    int size(const patch_type& patch) {return this->_atlas[patch].size();};
    int size(const patch_type& patch, const point_type& p) {return this->getFiberDimension(patch, p);};
  public: // Restriction
    const value_type *restrict(const patch_type& patch, const point_type& p) const {
      //std::cout <<"["<<this->commRank()<<"]: Constant restrict ("<<patch<<","<<p<<") from " << std::endl;
      //for(typename chart_type::iterator c_iter = this->_atlas.find(patch)->second.begin(); c_iter != this->_atlas.find(patch)->second.end(); ++c_iter) {
      //  std::cout <<"["<<this->commRank()<<"]:   point " << *c_iter << std::endl;
      //}
      if (this->hasPoint(patch, p)) {
        return &this->_value;
      }
      return &this->_defaultValue;
    };
    const value_type *restrictPoint(const patch_type& patch, const point_type& p) const {return this->restrict(patch, p);};
    const value_type *restrictPoint(const point_type& p) const {return this->restrict(0, p);};
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
          txt << "viewing a NewConstantSection" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing NewConstantSection '" << name << "'" << std::endl;
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
      this->_atlas = new atlas_type(topology, fiberDim, 0);
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
    bool hasPoint(const point_type& point) {
      return this->_atlas->hasPoint(0, point);
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
          this->updatePoint(s_iter->first, *c_iter, section->restrictPoint(s_iter->first, *c_iter));
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
    template<typename Sequence>
    void setFiberDimension(const patch_type& patch, const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(patch, *p_iter, dim);
      }
    };
    void setFiberDimension(const patch_type& patch, const std::set<point_type>& points, int dim) {
      for(typename std::set<point_type>::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        this->setFiberDimension(patch, *p_iter, dim);
      }
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
      this->setFiberDimension(patch, this->getTopology()->getLabelStratum(patch, "depth", depth), dim);
    };
    void setFiberDimensionByHeight(const patch_type& patch, int height, int dim) {
      this->setFiberDimension(patch, this->getTopology()->getLabelStratum(patch, "height", height), dim);
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
    // Return a pointer to the entire contiguous storage array
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
        // Avoids only the copy of closure()
        const int& dim = this->_atlas->restrictPoint(patch, p)[0];

        if (chart.count(p)) {
          for(int i = 0; i < dim; ++i) {
            values[++j] = array[p].v[i];
          }
        }
        // Need only the cone
        const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
        typename sieve_type::coneSequence::iterator   end  = cone->end();

        for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
          if (chart.count(*p_iter)) {
            const int& dim = this->_atlas->restrictPoint(patch, *p_iter)[0];

            for(int i = 0; i < dim; ++i) {
              values[++j] = array[*p_iter].v[i];
            }
          }
        }
      } else {
        // Right now, we have no way of consistently ordering the closure()
        const Obj<typename sieve_type::coneSet>& closure = this->getTopology()->getPatch(patch)->closure(p);
        typename sieve_type::coneSet::iterator   end     = closure->end();

        for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
          if (chart.count(*p_iter)) {
            const int& dim = this->_atlas->restrictPoint(patch, *p_iter)[0];

            for(int i = 0; i < dim; ++i) {
              values[++j] = array[*p_iter].v[i];
            }
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
        throw ALE::Exception("Update is not yet implemented for interpolated sieves");
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
    const value_type *restrictPoint(const point_type& p) {
      this->checkPatch(0);
      return this->_arrays[0][p].v;
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
      bool hasPoint(const patch_type& patch, const point_type& point) {
        return this->_atlas->hasPoint(patch, point);
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
      template<typename Sequence>
      void setFiberDimension(const patch_type& patch, const Obj<Sequence>& points, int dim) {
        for(typename topology_type::label_sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
          this->setFiberDimension(patch, *p_iter, dim);
        }
      };
      void addFiberDimension(const patch_type& patch, const point_type& p, int dim) {
        if (this->_atlas->hasPatch(patch) && this->_atlas->hasPoint(patch, p)) {
          const index_type values(dim, 0);
          this->_atlas->updateAddPoint(patch, p, &values);
        } else {
          this->setFiberDimension(patch, p, dim);
        }
      };
      void setFiberDimensionByDepth(const patch_type& patch, int depth, int dim) {
        this->setFiberDimension(patch, this->getTopology()->getLabelStratum(patch, "depth", depth), dim);
      };
      void setFiberDimensionByHeight(const patch_type& patch, int height, int dim) {
        this->setFiberDimension(patch, this->getTopology()->getLabelStratum(patch, "height", height), dim);
      };
      int size(const patch_type& patch) {
        const typename atlas_type::chart_type& points = this->_atlas->getPatch(patch);
        int size = 0;

        for(typename atlas_type::chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          size += std::max(0, this->getFiberDimension(patch, *p_iter));
        }
        return size;
      };
      int sizeWithBC(const patch_type& patch) {
        const typename atlas_type::chart_type& points = this->_atlas->getPatch(patch);
        int size = 0;

        for(typename atlas_type::chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          size += std::abs(this->getFiberDimension(patch, *p_iter));
        }
        return size;
      };
      int size(const patch_type& patch, const point_type& p) {
        if (this->getTopology()->depth() > 1) throw ALE::Exception("Compatibility layer is not for interpolated meshes");
        const typename atlas_type::chart_type&  points  = this->_atlas->getPatch(patch);
        const Obj<typename sieve_type::coneSequence> closure = this->getTopology()->getPatch(patch)->cone(p);
        typename sieve_type::coneSequence::iterator  end     = closure->end();
        int size = 0;

        size += std::max(0, this->getFiberDimension(patch, p));
        for(typename sieve_type::coneSequence::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
          if (points.count(*c_iter)) {
            size += std::max(0, this->getFiberDimension(patch, *c_iter));
          }
        }
        return size;
      };
      int sizeWithBC(const patch_type& patch, const point_type& p) {
        if (this->getTopology()->depth() > 1) throw ALE::Exception("Compatibility layer is not for interpolated meshes");
        const typename atlas_type::chart_type&  points  = this->_atlas->getPatch(patch);
        const Obj<typename sieve_type::coneSequence> closure = this->getTopology()->getPatch(patch)->cone(p);
        typename sieve_type::coneSequence::iterator  end     = closure->end();
        int size = 0;

        size += std::abs(this->getFiberDimension(patch, p));
        for(typename sieve_type::coneSequence::iterator c_iter = closure->begin(); c_iter != end; ++c_iter) {
          if (points.count(*c_iter)) {
            size += std::abs(this->getFiberDimension(patch, *c_iter));
          }
        }
        return size;
      };
      int size(const Obj<atlas_type>& atlas, const patch_type& patch) {
        const typename atlas_type::chart_type& points = atlas->getPatch(patch);
        int size = 0;

        for(typename atlas_type::chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
          size += std::max(0, this->getFiberDimension(atlas, patch, *p_iter));
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
#if 1
          throw ALE::Exception("Call should be moved to Bundle");
#else
          const Obj<typename sieve_type::coneSet> closure = this->getTopology()->getPatch(patch)->closure(p);
          typename sieve_type::coneSet::iterator  end     = closure->end();

          for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
            this->_indexArray->push_back(this->getIndex(patch, *p_iter));
          }
#endif
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
#if 1
          throw ALE::Exception("Call should be moved to Bundle");
#else
          const Obj<typename sieve_type::coneSet> closure = this->getTopology()->getPatch(patch)->closure(p);
          typename sieve_type::coneSet::iterator  end     = closure->end();

          for(typename sieve_type::coneSet::iterator p_iter = closure->begin(); p_iter != end; ++p_iter) {
            this->_indexArray->push_back(this->getIndex(patch, *p_iter, numbering));
          }
#endif
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
      void orderPoint(const Obj<atlas_type>& atlas, const Obj<sieve_type>& sieve, const patch_type& patch, const point_type& point, int& offset, int& bcOffset, const bool postponeGhosts = false) {
        const Obj<typename sieve_type::coneSequence>& cone = sieve->cone(point);
        typename sieve_type::coneSequence::iterator   end  = cone->end();
        index_type                                    idx  = atlas->restrictPoint(patch, point)[0];
        const int&                                    dim  = idx.prefix;
        const index_type                              defaultIdx(0, -1);

        if (atlas->getPatch(patch).count(point) == 0) {
          idx = defaultIdx;
        }
        if (idx.index == -1) {
          for(typename sieve_type::coneSequence::iterator c_iter = cone->begin(); c_iter != end; ++c_iter) {
            if (this->_debug > 1) {std::cout << "    Recursing to " << *c_iter << std::endl;}
            this->orderPoint(atlas, sieve, patch, *c_iter, offset, bcOffset);
          }
          if (dim > 0) {
            bool number = true;

            // Maybe use template specialization here
            if (postponeGhosts && this->getTopology()->getSendOverlap()->capContains(point)) {
              const Obj<typename topology_type::send_overlap_type::supportSequence>& ranks = this->getTopology()->getSendOverlap()->support(point);

              for(typename topology_type::send_overlap_type::supportSequence::iterator r_iter = ranks->begin(); r_iter != ranks->end(); ++r_iter) {
                if (this->commRank() > *r_iter) {
                  number = false;
                  break;
                }
              }
            }
            if (number) {
              if (this->_debug > 1) {std::cout << "  Ordering point " << point << " at " << offset << std::endl;}
              idx.index = offset;
              atlas->updatePoint(patch, point, &idx);
              offset += dim;
            } else {
              if (this->_debug > 1) {std::cout << "  Ignoring ghost point " << point << std::endl;}
            }
          } else if (dim < 0) {
            if (this->_debug > 1) {std::cout << "  Ordering boundary point " << point << " at " << bcOffset << std::endl;}
            idx.index = bcOffset;
            atlas->updatePoint(patch, point, &idx);
            bcOffset += dim;
          }
        }
      }
      void orderPatch(const Obj<atlas_type>& atlas, const patch_type& patch, int& offset, int& bcOffset, const bool postponeGhosts = false) {
        const typename atlas_type::chart_type& chart = atlas->getPatch(patch);

        for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
          if (this->_debug > 1) {std::cout << "Ordering closure of point " << *p_iter << std::endl;}
          this->orderPoint(atlas, this->getTopology()->getPatch(patch), patch, *p_iter, offset, bcOffset, postponeGhosts);
        }
        for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
          index_type idx = atlas->restrictPoint(patch, *p_iter)[0];
          const int& dim = idx.prefix;

          if (dim < 0) {
            if (this->_debug > 1) {std::cout << "Correcting boundary offset of point " << *p_iter << std::endl;}
            idx.index = offset - (idx.index+2);
            atlas->updatePoint(patch, *p_iter, &idx);
          }
        }
      };
      void orderPatches(const Obj<atlas_type>& atlas, const bool postponeGhosts = false) {
        const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          if (this->_debug > 1) {std::cout << "Ordering patch " << p_iter->first << std::endl;}
          int offset = 0, bcOffset = -2;

          if (!atlas->hasPatch(p_iter->first)) continue;
          this->orderPatch(atlas, p_iter->first, offset, bcOffset, postponeGhosts);
        }
      };
      void orderPatches(const bool postponeGhosts = false) {
        this->orderPatches(this->_atlas, postponeGhosts);
      };
      void allocateStorage() {
        const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          if (!this->_atlas->hasPatch(p_iter->first)) continue;
          this->_arrays[p_iter->first] = new value_type[this->sizeWithBC(p_iter->first)];
          PetscMemzero(this->_arrays[p_iter->first], this->sizeWithBC(p_iter->first) * sizeof(value_type));
        }
      };
      void allocate(const bool postponeGhosts = false) {
        bool doGhosts = false;

        if (postponeGhosts && !this->getTopology()->getSendOverlap().isNull()) {
          doGhosts = true;
        }
        this->orderPatches(doGhosts);
        if (doGhosts) {
          const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

          for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
            if (this->_debug > 1) {std::cout << "Ordering patch " << p_iter->first << " for ghosts" << std::endl;}
            const typename atlas_type::chart_type& points = this->_atlas->getPatch(p_iter->first);
            int offset = 0, bcOffset = -2;

            for(typename atlas_type::chart_type::iterator point = points.begin(); point != points.end(); ++point) {
              const index_type& idx = this->_atlas->restrictPoint(p_iter->first, *point)[0];

              offset = std::max(offset, idx.index + std::abs(idx.prefix));
            }
            if (!this->_atlas->hasPatch(p_iter->first)) continue;
            this->orderPatch(this->_atlas, p_iter->first, offset, bcOffset);
            if (offset != this->sizeWithBC(p_iter->first)) throw ALE::Exception("Inconsistent array sizes in section");
          }
        }
        this->allocateStorage();
      };
      void addPoint(const patch_type& patch, const point_type& point, const int dim) {
        if (dim == 0) return;
        //const typename atlas_type::chart_type& chart = this->_atlas->getPatch(patch);

        //if (chart.find(point) == chart.end()) {
        if (this->_atlasNew.isNull()) {
          this->_atlasNew = new atlas_type(this->getTopology());
          this->_atlasNew->copy(this->_atlas);
        }
        const index_type idx(dim, -1);
        this->_atlasNew->updatePatch(patch, point);
        this->_atlasNew->updatePoint(patch, point, &idx);
      };
      void reallocate() {
        if (this->_atlasNew.isNull()) return;
        const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

        // Since copy() preserves offsets, we must reinitialize them before ordering
        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const patch_type&                      patch = p_iter->first;
          const typename atlas_type::chart_type& chart = this->_atlasNew->getPatch(patch);
          index_type                             defaultIdx(0, -1);

          for(typename atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            defaultIdx.prefix = this->_atlasNew->restrictPoint(patch, *c_iter)[0].prefix;
            this->_atlasNew->updatePoint(patch, *c_iter, &defaultIdx);
          }
        }
        this->orderPatches(this->_atlasNew);
        // Copy over existing values
        for(typename topology_type::sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
          const patch_type&                      patch    = p_iter->first;
          value_type                            *newArray = new value_type[this->size(this->_atlasNew, patch)];

          if (!this->_atlas->hasPatch(patch)) {
            this->_arrays[patch] = newArray;
            continue;
          }
          const typename atlas_type::chart_type& chart    = this->_atlas->getPatch(patch);
          const value_type                      *array    = this->_arrays[patch];

          for(typename atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
            const index_type& idx       = this->_atlas->restrictPoint(patch, *c_iter)[0];
            const int         size      = idx.prefix;
            const int         offset    = idx.index;
            const int&        newOffset = this->_atlasNew->restrictPoint(patch, *c_iter)[0].index;

            for(int i = 0; i < size; ++i) {
              newArray[newOffset+i] = array[offset+i];
            }
          }
          delete [] this->_arrays[patch];
          this->_arrays[patch] = newArray;
        }
        this->_atlas    = this->_atlasNew;
        this->_atlasNew = NULL;
      };
    public: // Restriction and Update
      // Zero entries
      void zero(const patch_type& patch) {
        this->checkPatch(patch);
        memset(this->_arrays[patch], 0, this->size(patch)* sizeof(value_type));
      };
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
        const int         size   = this->sizeWithBC(patch, p);
        value_type       *values = this->getRawArray(size);
        int               j      = -1;

        if (this->getTopology()->height(patch) < 2) {
          // Avoids the copy of both
          //   points  in topology->closure()
          //   indices in _atlas->restrict()
          const index_type& pInd = this->_atlas->restrictPoint(patch, p)[0];

          for(int i = pInd.index; i < std::abs(pInd.prefix) + pInd.index; ++i) {
            values[++j] = a[i];
          }
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            const index_type& ind    = this->_atlas->restrictPoint(patch, *p_iter)[0];
            const int&        start  = ind.index;
            const int&        length = std::abs(ind.prefix);

            for(int i = start; i < start + length; ++i) {
              values[++j] = a[i];
            }
          }
        } else {
          const Obj<IndexArray>& ind = this->getIndices(patch, p);

          for(typename IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int& start  = i_iter->index;
            const int& length = std::abs(i_iter->prefix);

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
          j += std::max(0, -pInd.prefix);
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            const index_type& ind    = this->_atlas->restrictPoint(patch, *p_iter)[0];
            const int&        start  = ind.index;
            const int&        length = ind.prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
            }
            j += std::max(0, -length);
          }
        } else {
          const Obj<IndexArray>& ind = this->getIndices(patch, p);

          for(typename IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int& start  = i_iter->index;
            const int& length = i_iter->prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
            }
            j += std::max(0, -length);
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
          j += std::max(0, -pInd.prefix);
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            const index_type& ind    = this->_atlas->restrictPoint(patch, *p_iter)[0];
            const int&        start  = ind.index;
            const int&        length = ind.prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] += v[++j];
            }
            j += std::max(0, -length);
          }
        } else {
          const Obj<IndexArray>& ind = this->getIndices(patch, p);

          for(typename IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int& start  = i_iter->index;
            const int& length = i_iter->prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] += v[++j];
            }
            j += std::max(0, -length);
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
          const Obj<IndexArray>& ind = this->getIndices(patch, p);
          typename Input::iterator v_iter = v->begin();
          typename Input::iterator v_end  = v->end();

          for(typename IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int& start  = i_iter->index;
            const int& length = i_iter->prefix;

            for(int i = start; i < start + length; ++i) {
              a[i] = *v_iter;
              ++v_iter;
            }
          }
        }
      };
      void updateBC(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        value_type *a = this->_arrays[patch];
        int         j = -1;

        if (this->getTopology()->height(patch) < 2) {
          // Avoids the copy of both
          //   points  in topology->closure()
          //   indices in _atlas->restrict()
          const index_type& pInd = this->_atlas->restrictPoint(patch, p)[0];

          for(int i = pInd.index; i < std::abs(pInd.prefix) + pInd.index; ++i) {
            a[i] = v[++j];
          }
          const Obj<typename sieve_type::coneSequence>& cone = this->getTopology()->getPatch(patch)->cone(p);
          typename sieve_type::coneSequence::iterator   end  = cone->end();

          for(typename sieve_type::coneSequence::iterator p_iter = cone->begin(); p_iter != end; ++p_iter) {
            const index_type& ind    = this->_atlas->restrictPoint(patch, *p_iter)[0];
            const int&        start  = ind.index;
            const int&        length = std::abs(ind.prefix);

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
            }
          }
        } else {
          const Obj<IndexArray>& ind = this->getIndices(patch, p);

          for(typename IndexArray::iterator i_iter = ind->begin(); i_iter != ind->end(); ++i_iter) {
            const int& start  = i_iter->index;
            const int& length = std::abs(i_iter->prefix);

            for(int i = start; i < start + length; ++i) {
              a[i] = v[++j];
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
      void updatePointBC(const patch_type& patch, const point_type& p, const value_type v[]) {
        this->checkPatch(patch);
        const index_type& idx = this->_atlas->restrictPoint(patch, p)[0];
        value_type       *a   = &(this->_arrays[patch][idx.index]);

        for(int i = 0; i < std::abs(idx.prefix); ++i) {
          a[i] = v[i];
        }
      };
    public: // BC
      void copyBC(const Obj<Section>& section) {
        const typename topology_type::sheaf_type& patches = this->getTopology()->getPatches();

        for(typename topology_type::sheaf_type::const_iterator patch_iter = patches.begin(); patch_iter != patches.end(); ++patch_iter) {
          const typename atlas_type::chart_type& chart = this->_atlas->getPatch(patch_iter->first);

          for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
            const index_type& idx = this->_atlas->restrictPoint(patch_iter->first, *p_iter)[0];

            if (idx.prefix < 0) {
              this->updatePointBC(patch_iter->first, *p_iter, section->restrictPoint(patch_iter->first, *p_iter));
            }
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
        for(typename values_type::const_iterator a_iter = this->_arrays.begin(); a_iter != this->_arrays.end(); ++a_iter) {
          const patch_type&  patch = a_iter->first;
          const value_type  *array = a_iter->second;

          txt << "[" << this->commRank() << "]: Patch " << patch << std::endl;
          const typename atlas_type::chart_type& chart = this->_atlas->getPatch(patch);

          for(typename atlas_type::chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
            const point_type& point = *p_iter;
            const index_type& idx   = this->_atlas->restrictPoint(patch, point)[0];

            if (idx.prefix != 0) {
              txt << "[" << this->commRank() << "]:   " << point << " dim " << idx.prefix << " offset " << idx.index << "  ";
              for(int i = 0; i < std::abs(idx.prefix); i++) {
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
    class OldConstantSection : public ALE::ParallelObject {
    public:
      typedef OldConstantSection<Topology_, Value_> section_type;
      typedef Topology_                          topology_type;
      typedef typename topology_type::patch_type patch_type;
      typedef typename topology_type::sieve_type sieve_type;
      typedef typename topology_type::point_type point_type;
      typedef Value_                             value_type;
    protected:
      Obj<topology_type> _topology;
      const value_type   _value;
      Obj<section_type>  _section;
    public:
      OldConstantSection(MPI_Comm comm, const value_type value, const int debug = 0) : ParallelObject(comm, debug), _value(value) {
        this->_topology = new topology_type(comm, debug);
        this->_section  = this;
        this->_section.addRef();
      };
      OldConstantSection(const Obj<topology_type>& topology, const value_type value) : ParallelObject(topology->comm(), topology->debug()), _topology(topology), _value(value) {
        this->_section  = this;
        this->_section.addRef();
      };
      virtual ~OldConstantSection() {};
    public: // Verifiers
      bool hasPoint(const patch_type& patch, const point_type& point) const {return true;};
    public: // Restriction
      const value_type *restrict(const patch_type& patch) {return &this->_value;};
      const value_type *restrictPoint(const patch_type& patch, const point_type& p) {return &this->_value;};
    public: // Adapter
      const Obj<section_type>& getSection(const patch_type& patch) {return this->_section;};
      const value_type *restrictPoint(const point_type& p) {return &this->_value;};
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
            txt << "viewing a OldConstantSection with value " << this->_value << std::endl;
          }
        } else {
          if(rank == 0) {
            txt << "viewing OldConstantSection '" << name << "' with value " << this->_value << std::endl;
          }
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
      typedef typename base_type::topology_type topology_type;
      typedef typename base_type::patch_type    patch_type;
      typedef typename base_type::chart_type    chart_type;
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
          tagvalp = (int *) malloc(sizeof(int));
          MPI_Keyval_create(MPI_NULL_COPY_FN, Mesh_DelTag, &tagKeyval, (void *) NULL);
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
            this->setFiberDimension(rank, *b_iter, *(sizer->restrictPoint(rank, *b_iter)));
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
