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
  template<typename Point_, typename Alloc_ = malloc_allocator<Point_> >
  class DiscreteSieve {
  public:
    typedef Point_                              point_type;
    typedef Alloc_                              alloc_type;
    typedef std::vector<point_type, alloc_type> coneSequence;
    typedef std::vector<point_type, alloc_type> coneSet;
    typedef std::vector<point_type, alloc_type> coneArray;
    typedef std::vector<point_type, alloc_type> supportSequence;
    typedef std::vector<point_type, alloc_type> supportSet;
    typedef std::vector<point_type, alloc_type> supportArray;
    typedef std::set<point_type, std::less<point_type>, alloc_type>   points_type;
    typedef points_type                                               baseSequence;
    typedef points_type                                               capSequence;
    typedef typename alloc_type::template rebind<points_type>::other  points_alloc_type;
    typedef typename points_alloc_type::pointer                       points_ptr;
    typedef typename alloc_type::template rebind<coneSequence>::other coneSequence_alloc_type;
    typedef typename coneSequence_alloc_type::pointer                 coneSequence_ptr;
  protected:
    Obj<points_type>  _points;
    Obj<coneSequence> _empty;
    Obj<coneSequence> _return;
    alloc_type        _allocator;
    void _init() {
      points_ptr pPoints = points_alloc_type(this->_allocator).allocate(1);
      points_alloc_type(this->_allocator).construct(pPoints, points_type());
      this->_points = Obj<points_type>(pPoints, sizeof(points_type));
      ///this->_points = new points_type();
      coneSequence_ptr pEmpty = coneSequence_alloc_type(this->_allocator).allocate(1);
      coneSequence_alloc_type(this->_allocator).construct(pEmpty, coneSequence());
      this->_empty = Obj<coneSequence>(pEmpty, sizeof(coneSequence));
      ///this->_empty  = new coneSequence();
      coneSequence_ptr pReturn = coneSequence_alloc_type(this->_allocator).allocate(1);
      coneSequence_alloc_type(this->_allocator).construct(pReturn, coneSequence());
      this->_return = Obj<coneSequence>(pReturn, sizeof(coneSequence));
      ///this->_return = new coneSequence();
    };
  public:
    DiscreteSieve() {
      this->_init();
    };
    template<typename Input>
    DiscreteSieve(const Obj<Input>& points) {
      this->_init();
      this->_points->insert(points->begin(), points->end());
    }
    virtual ~DiscreteSieve() {};
  public:
    void addPoint(const point_type& point) {
      this->_points->insert(point);
    }
    template<typename Input>
    void addPoints(const Obj<Input>& points) {
      this->_points->insert(points->begin(), points->end());
    }
    const Obj<coneSequence>& cone(const point_type& p) {return this->_empty;}
    template<typename Input>
    const Obj<coneSequence>& cone(const Input& p) {return this->_empty;}
    const Obj<coneSet>& nCone(const point_type& p, const int level) {
      if (level == 0) {
        return this->closure(p);
      } else {
        return this->_empty;
      }
    }
    const Obj<coneArray>& closure(const point_type& p) {
      this->_return->clear();
      this->_return->push_back(p);
      return this->_return;
    }
    const Obj<supportSequence>& support(const point_type& p) {return this->_empty;}
    template<typename Input>
    const Obj<supportSequence>& support(const Input& p) {return this->_empty;}
    const Obj<supportSet>& nSupport(const point_type& p, const int level) {
      if (level == 0) {
        return this->star(p);
      } else {
        return this->_empty;
      }
    }
    const Obj<supportArray>& star(const point_type& p) {
      this->_return->clear();
      this->_return->push_back(p);
      return this->_return;
    }
    const Obj<capSequence>& roots() {return this->_points;}
    const Obj<capSequence>& cap() {return this->_points;}
    const Obj<baseSequence>& leaves() {return this->_points;}
    const Obj<baseSequence>& base() {return this->_points;}
    template<typename Color>
    void addArrow(const point_type& p, const point_type& q, const Color& color) {
      throw ALE::Exception("Cannot add an arrow to a DiscreteSieve");
    }
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
  template<typename Point_, typename Value_, typename Alloc_ = malloc_allocator<Point_> >
  class ConstantSection : public ALE::ParallelObject {
  public:
    typedef Point_                                                  point_type;
    typedef Value_                                                  value_type;
    typedef Alloc_                                                  alloc_type;
    typedef std::set<point_type, std::less<point_type>, alloc_type> chart_type;
  protected:
    chart_type _chart;
    value_type _value[2];
  public:
    ConstantSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      _value[1] = 0;
    };
    ConstantSection(MPI_Comm comm, const value_type& value, const int debug) : ParallelObject(comm, debug) {
      _value[0] = value;
      _value[1] = value;
    };
    ConstantSection(MPI_Comm comm, const value_type& value, const value_type& defaultValue, const int debug) : ParallelObject(comm, debug) {
      _value[0] = value;
      _value[1] = defaultValue;
    };
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
    }
    template<typename Points>
    void addPoint(const Points& points) {
      this->_chart.insert(points.begin(), points.end());
    }
//     void addPoint(const std::set<point_type>& points) {
//       this->_chart.insert(points.begin(), points.end());
//     };
    value_type getDefaultValue() {return this->_value[1];};
    void setDefaultValue(const value_type value) {this->_value[1] = value;};
    void copy(const Obj<ConstantSection>& section) {
      const chart_type& chart = section->getChart();

      this->addPoint(chart);
      this->_value[0] = section->restrictSpace()[0];
      this->setDefaultValue(section->getDefaultValue());
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
    }
    void addFiberDimension(const point_type& p, int dim) {
      if (this->_chart.find(p) != this->_chart.end()) {
        ostringstream msg;
        msg << "Invalid addition to fiber dimension " << dim << " cannot exceed 1" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      } else {
        this->setFiberDimension(p, dim);
      }
    }
    int size(const point_type& p) {return this->getFiberDimension(p);};
    void allocatePoint() {};
  public: // Restriction
    const value_type *restrictSpace() const {
      return this->_value;
    };
    const value_type *restrictPoint(const point_type& p) const {
      if (this->hasPoint(p)) {
        return this->_value;
      }
      return &this->_value[1];
    };
    void updatePoint(const point_type& p, const value_type v[]) {
      this->_value[0] = v[0];
    };
    void updateAddPoint(const point_type& p, const value_type v[]) {
      this->_value[0] += v[0];
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
          txt << "viewing a ConstantSection" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing ConstantSection '" << name << "'" << std::endl;
        }
      }
      txt <<"["<<this->commRank()<<"]: Value " << this->_value[0] << " Default Value " << this->_value[1] << std::endl;
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };
  // A UniformSection often acts as an Atlas
  //   All fibers are the same dimension
  //     Note we can use a ConstantSection for this Atlas
  //   Each point may have a different vector
  //     Thus we need storage for values, and hence must implement completion
  template<typename Point_, typename Value_, int fiberDim = 1, typename Alloc_ = malloc_allocator<Value_> >
  class UniformSection : public ALE::ParallelObject {
  public:
    typedef Point_                                           point_type;
    typedef Value_                                           value_type;
    typedef Alloc_                                           alloc_type;
    typedef typename alloc_type::template rebind<point_type>::other point_alloc_type;
    typedef ConstantSection<point_type, int, point_alloc_type> atlas_type;
    typedef typename atlas_type::chart_type                  chart_type;
    typedef struct {value_type v[fiberDim];}                 fiber_type;
    typedef typename alloc_type::template rebind<std::pair<const point_type, fiber_type> >::other pair_alloc_type;
    typedef std::map<point_type, fiber_type, std::less<point_type>, pair_alloc_type>              values_type;
    typedef typename alloc_type::template rebind<atlas_type>::other                               atlas_alloc_type;
    typedef typename atlas_alloc_type::pointer                                                    atlas_ptr;
  protected:
    size_t          _contiguous_array_size;
    value_type     *_contiguous_array;
    Obj<atlas_type> _atlas;
    values_type     _array;
    fiber_type      _emptyValue;
    alloc_type      _allocator;
  public:
    UniformSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _contiguous_array_size(0), _contiguous_array(NULL) {
      atlas_ptr pAtlas = atlas_alloc_type(this->_allocator).allocate(1);
      atlas_alloc_type(this->_allocator).construct(pAtlas, atlas_type(comm, debug));
      this->_atlas = Obj<atlas_type>(pAtlas, sizeof(atlas_type));
      int dim = fiberDim;
      this->_atlas->updatePoint(*this->_atlas->getChart().begin(), &dim);
      for(int i = 0; i < fiberDim; ++i) this->_emptyValue.v[i] = value_type();
    };
    UniformSection(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _contiguous_array_size(0), _contiguous_array(NULL), _atlas(atlas) {
      int dim = fiberDim;
      this->_atlas->update(*this->_atlas->getChart().begin(), &dim);
      for(int i = 0; i < fiberDim; ++i) this->_emptyValue.v[i] = value_type();
    };
    virtual ~UniformSection() {
      this->_atlas = NULL;
      if (this->_contiguous_array) {
        for(size_t i = 0; i < this->_contiguous_array_size; ++i) {this->_allocator.destroy(this->_contiguous_array+i);}
        this->_allocator.deallocate(this->_contiguous_array, this->_contiguous_array_size);
      }
    };
  public:
    value_type *getRawArray(const int size) {
      static value_type *array   = NULL;
      static int         maxSize = 0;

      if (size > maxSize) {
        const value_type dummy(0);

        if (array) {
          for(int i = 0; i < maxSize; ++i) {this->_allocator.destroy(array+i);}
          this->_allocator.deallocate(array, maxSize);
          ///delete [] array;
        }
        maxSize = size;
        array   = this->_allocator.allocate(maxSize);
        for(int i = 0; i < maxSize; ++i) {this->_allocator.construct(array+i, dummy);}
        ///array = new value_type[maxSize];
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
    }
    void copy(const Obj<UniformSection>& section) {
      this->getAtlas()->copy(section->getAtlas());
      const chart_type& chart = section->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        this->updatePoint(*c_iter, section->restrictPoint(*c_iter));
      }
    }
  public: // Sizes
    void clear() {
      this->_array.clear();
      this->_atlas->clear(); 
    }
    int getFiberDimension(const point_type& p) const {
      return this->_atlas->restrictPoint(p)[0];
    }
    void setFiberDimension(const point_type& p, int dim) {
      this->update();
      this->checkDimension(dim);
      this->_atlas->addPoint(p);
      this->_atlas->updatePoint(p, &dim);
    }
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    }
    void setFiberDimension(const std::set<point_type>& points, int dim) {
      for(typename std::set<point_type>::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
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
    void allocatePoint() {};
  public: // Restriction
    const value_type *restrictSpace() {
      const chart_type& chart = this->getChart();
      const value_type  dummy = 0;
      int               k     = 0;

      if (chart.size() > this->_contiguous_array_size*fiberDim) {
        if (this->_contiguous_array) {
          for(size_t i = 0; i < this->_contiguous_array_size; ++i) {this->_allocator.destroy(this->_contiguous_array+i);}
          this->_allocator.deallocate(this->_contiguous_array, this->_contiguous_array_size);
        }
        this->_contiguous_array_size = chart.size()*fiberDim;
        this->_contiguous_array = this->_allocator.allocate(this->_contiguous_array_size*fiberDim);
        for(size_t i = 0; i < this->_contiguous_array_size; ++i) {this->_allocator.construct(this->_contiguous_array+i, dummy);}
      }
      for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const value_type *a = this->_array[*p_iter].v;

        for(int i = 0; i < fiberDim; ++i, ++k) {
          this->_contiguous_array[k] = a[i];
        }
      }
      return this->_contiguous_array;
    };
    void update() {
      if (this->_contiguous_array) {
        const chart_type& chart = this->getChart();
        int               k     = 0;

        for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
          value_type *a = this->_array[*p_iter].v;

          for(int i = 0; i < fiberDim; ++i, ++k) {
            a[i] = this->_contiguous_array[k];
          }
        }
        for(size_t i = 0; i < this->_contiguous_array_size; ++i) {this->_allocator.destroy(this->_contiguous_array+i);}
        this->_allocator.deallocate(this->_contiguous_array, this->_contiguous_array_size);
        this->_contiguous_array_size = 0;
        this->_contiguous_array      = NULL;
      }
    };
    // Return only the values associated to this point, not its closure
    const value_type *restrictPoint(const point_type& p) {
      if (this->_array.find(p) == this->_array.end()) return this->_emptyValue.v;
      this->update();
      return this->_array[p].v;
    };
    // Update only the values associated to this point, not its closure
    void updatePoint(const point_type& p, const value_type v[]) {
      this->update();
      for(int i = 0; i < fiberDim; ++i) {
        this->_array[p].v[i] = v[i];
      }
    };
    // Update only the values associated to this point, not its closure
    void updateAddPoint(const point_type& p, const value_type v[]) {
      this->update();
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

      this->update();
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
        const typename atlas_type::value_type dim   = this->_atlas->restrictPoint(point)[0];

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

  // A FauxConstantSection is the simplest Section
  //   All fibers are dimension 1
  //   All values are equal to a constant
  //     We need no value storage and no communication for completion
  template<typename Point_, typename Value_, typename Alloc_ = malloc_allocator<Point_> >
  class FauxConstantSection : public ALE::ParallelObject {
  public:
    typedef Point_ point_type;
    typedef Value_ value_type;
    typedef Alloc_ alloc_type;
  protected:
    value_type _value;
  public:
    FauxConstantSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {};
    FauxConstantSection(MPI_Comm comm, const value_type& value, const int debug) : ParallelObject(comm, debug), _value(value) {};
  public: // Verifiers
    void checkDimension(const int& dim) {
      if (dim != 1) {
        ostringstream msg;
        msg << "Invalid fiber dimension " << dim << " must be 1" << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
  public: // Accessors
    void addPoint(const point_type& point) {
    }
    template<typename Points>
    void addPoint(const Obj<Points>& points) {
    }
    template<typename Points>
    void addPoint(const Points& points) {
    }
    void copy(const Obj<FauxConstantSection>& section) {
      this->_value = section->restrictPoint(point_type())[0];
    }
  public: // Sizes
    void clear() {
    };
    int getFiberDimension(const point_type& p) const {
      return 1;
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
    }
    void addFiberDimension(const point_type& p, int dim) {
      this->setFiberDimension(p, dim);
    }
    int size(const point_type& p) {return this->getFiberDimension(p);}
  public: // Restriction
    const value_type *restrictPoint(const point_type& p) const {
      return &this->_value;
    };
    void updatePoint(const point_type& p, const value_type v[]) {
      this->_value = v[0];
    };
    void updateAddPoint(const point_type& p, const value_type v[]) {
      this->_value += v[0];
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
          txt << "viewing a FauxConstantSection" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing FauxConstantSection '" << name << "'" << std::endl;
        }
      }
      txt <<"["<<this->commRank()<<"]: Value " << this->_value << std::endl;
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };
  // Make a simple set from the keys of a map
  template<typename Map>
  class SetFromMap {
  public:
    typedef typename Map::size_type size_type;
    class const_iterator {
    public:
      typedef typename Map::key_type  value_type;
      typedef typename Map::size_type size_type;
    protected:
      typename Map::const_iterator _iter;
    public:
      const_iterator(const typename Map::const_iterator& iter): _iter(iter) {};
      ~const_iterator() {};
    public:
      const_iterator& operator=(const const_iterator& iter) {this->_iter = iter._iter; return this->_iter;};
      bool operator==(const const_iterator& iter) const {return this->_iter == iter._iter;};
      bool operator!=(const const_iterator& iter) const {return this->_iter != iter._iter;};
      const_iterator& operator++() {++this->_iter; return *this;}
      const_iterator operator++(int) {
        const_iterator tmp(*this);
        ++(*this);
        return tmp;
      };
      const_iterator& operator--() {--this->_iter; return *this;}
      const_iterator operator--(int) {
        const_iterator tmp(*this);
        --(*this);
        return tmp;
      };
      value_type operator*() const {return this->_iter->first;};
    };
  protected:
    const Map& _map;
  public:
    SetFromMap(const Map& map): _map(map) {};
  public:
    const_iterator begin() const {return const_iterator(this->_map.begin());};
    const_iterator end()   const {return const_iterator(this->_map.end());};
    size_type      size()  const {return this->_map.size();};
  };
  // A NewUniformSection often acts as an Atlas
  //   All fibers are the same dimension
  //     Note we can use a ConstantSection for this Atlas
  //   Each point may have a different vector
  //     Thus we need storage for values, and hence must implement completion
  template<typename Point_, typename Value_, int fiberDim = 1, typename Alloc_ = malloc_allocator<Value_> >
  class NewUniformSection : public ALE::ParallelObject {
  public:
    typedef Point_                                           point_type;
    typedef Value_                                           value_type;
    typedef Alloc_                                           alloc_type;
    typedef typename alloc_type::template rebind<point_type>::other                               point_alloc_type;
    typedef FauxConstantSection<point_type, int, point_alloc_type>                                atlas_type;
    typedef struct {value_type v[fiberDim];}                                                      fiber_type;
    typedef typename alloc_type::template rebind<std::pair<const point_type, fiber_type> >::other pair_alloc_type;
    typedef std::map<point_type, fiber_type, std::less<point_type>, pair_alloc_type>              values_type;
    typedef SetFromMap<values_type>                                                               chart_type;
    typedef typename alloc_type::template rebind<atlas_type>::other                               atlas_alloc_type;
    typedef typename atlas_alloc_type::pointer                                                    atlas_ptr;
  protected:
    values_type     _array;
    chart_type      _chart;
    size_t          _contiguous_array_size;
    value_type     *_contiguous_array;
    Obj<atlas_type> _atlas;
    fiber_type      _emptyValue;
    alloc_type      _allocator;
  public:
    NewUniformSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug), _chart(_array), _contiguous_array_size(0), _contiguous_array(NULL) {
      atlas_ptr pAtlas = atlas_alloc_type(this->_allocator).allocate(1);
      atlas_alloc_type(this->_allocator).construct(pAtlas, atlas_type(comm, debug));
      this->_atlas = Obj<atlas_type>(pAtlas, sizeof(atlas_type));
      int dim = fiberDim;
      this->_atlas->update(point_type(), &dim);
      for(int i = 0; i < fiberDim; ++i) this->_emptyValue.v[i] = value_type();
    };
    NewUniformSection(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _chart(_array), _contiguous_array_size(0), _contiguous_array(NULL), _atlas(atlas) {
      int dim = fiberDim;
      this->_atlas->update(point_type(), &dim);
      for(int i = 0; i < fiberDim; ++i) this->_emptyValue.v[i] = value_type();
    };
    ~NewUniformSection() {
      this->_atlas = NULL;
      if (this->_contiguous_array) {
        for(size_t i = 0; i < this->_contiguous_array_size; ++i) {this->_allocator.destroy(this->_contiguous_array+i);}
        this->_allocator.deallocate(this->_contiguous_array, this->_contiguous_array_size);
      }
    };
  public:
    value_type *getRawArray(const int size) {
      static value_type *array   = NULL;
      static int         maxSize = 0;

      if (size > maxSize) {
        const value_type dummy(0);

        if (array) {
          for(int i = 0; i < maxSize; ++i) {this->_allocator.destroy(array+i);}
          this->_allocator.deallocate(array, maxSize);
          ///delete [] array;
        }
        maxSize = size;
        array   = this->_allocator.allocate(maxSize);
        for(int i = 0; i < maxSize; ++i) {this->_allocator.construct(array+i, dummy);}
        ///array = new value_type[maxSize];
      };
      return array;
    };
  public: // Verifiers
    bool hasPoint(const point_type& point) {
      return (this->_array.find(point) != this->_array.end());
      ///return this->_atlas->hasPoint(point);
    };
    void checkDimension(const int& dim) {
      if (dim != fiberDim) {
        ostringstream msg;
        msg << "Invalid fiber dimension " << dim << " must be " << fiberDim << std::endl;
        throw ALE::Exception(msg.str().c_str());
      }
    };
  public: // Accessors
    const chart_type& getChart() {return this->_chart;}
    const Obj<atlas_type>& getAtlas() {return this->_atlas;}
    void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;}
    void addPoint(const point_type& point) {
      this->setFiberDimension(point, fiberDim);
    }
    template<typename Points>
    void addPoint(const Obj<Points>& points) {
      for(typename Points::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, fiberDim);
      }
    }
    void copy(const Obj<NewUniformSection>& section) {
      this->getAtlas()->copy(section->getAtlas());
      const chart_type& chart = section->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        this->updatePoint(*c_iter, section->restrictPoint(*c_iter));
      }
    }
  public: // Sizes
    void clear() {
      this->_array.clear();
      this->_atlas->clear(); 
    };
    int getFiberDimension(const point_type& p) const {
      return fiberDim;
    };
    void setFiberDimension(const point_type& p, int dim) {
      this->update();
      this->checkDimension(dim);
      this->_atlas->addPoint(p);
      this->_atlas->updatePoint(p, &dim);
      this->_array[p] = fiber_type();
    };
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim);
      }
    }
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
    int size() {
      const chart_type& points = this->getChart();
      int               size   = 0;

      for(typename chart_type::iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getFiberDimension(*p_iter);
      }
      return size;
    };
    int sizeWithBC() {
      return this->size();
    };
    void allocatePoint() {};
  public: // Restriction
    // Return a pointer to the entire contiguous storage array
    const value_type *restrictSpace() {
      const chart_type& chart = this->getChart();
      const value_type  dummy = 0;
      int               k     = 0;

      if (chart.size() > this->_contiguous_array_size*fiberDim) {
        if (this->_contiguous_array) {
          for(size_t i = 0; i < this->_contiguous_array_size; ++i) {this->_allocator.destroy(this->_contiguous_array+i);}
          this->_allocator.deallocate(this->_contiguous_array, this->_contiguous_array_size);
        }
        this->_contiguous_array_size = chart.size()*fiberDim;
        this->_contiguous_array = this->_allocator.allocate(this->_contiguous_array_size*fiberDim);
        for(size_t i = 0; i < this->_contiguous_array_size; ++i) {this->_allocator.construct(this->_contiguous_array+i, dummy);}
      }
      for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const value_type *a = this->_array[*p_iter].v;

        for(int i = 0; i < fiberDim; ++i, ++k) {
          this->_contiguous_array[k] = a[i];
        }
      }
      return this->_contiguous_array;
    };
    void update() {
      if (this->_contiguous_array) {
        const chart_type& chart = this->getChart();
        int               k     = 0;

        for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
          value_type *a = this->_array[*p_iter].v;

          for(int i = 0; i < fiberDim; ++i, ++k) {
            a[i] = this->_contiguous_array[k];
          }
        }
        for(size_t i = 0; i < this->_contiguous_array_size; ++i) {this->_allocator.destroy(this->_contiguous_array+i);}
        this->_allocator.deallocate(this->_contiguous_array, this->_contiguous_array_size);
        this->_contiguous_array_size = 0;
        this->_contiguous_array      = NULL;
      }
    };
    // Return only the values associated to this point, not its closure
    const value_type *restrictPoint(const point_type& p) {
      if (this->_array.find(p) == this->_array.end()) return this->_emptyValue.v;
      this->update();
      return this->_array[p].v;
    };
    // Update only the values associated to this point, not its closure
    void updatePoint(const point_type& p, const value_type v[]) {
      this->update();
      for(int i = 0; i < fiberDim; ++i) {
        this->_array[p].v[i] = v[i];
      }
    };
    // Update only the values associated to this point, not its closure
    void updateAddPoint(const point_type& p, const value_type v[]) {
      this->update();
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

      this->update();
      if (comm == MPI_COMM_NULL) {
        comm = this->comm();
        rank = this->commRank();
      } else {
        MPI_Comm_rank(comm, &rank);
      }
      if (name == "") {
        if(rank == 0) {
          txt << "viewing a NewUniformSection" << std::endl;
        }
      } else {
        if(rank == 0) {
          txt << "viewing NewUniformSection '" << name << "'" << std::endl;
        }
      }
      const chart_type& chart = this->getChart();
      values_type&      array = this->_array;

      for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        const point_type& point = *p_iter;

        if (fiberDim != 0) {
          txt << "[" << this->commRank() << "]:   " << point << " dim " << fiberDim << "  ";
          for(int i = 0; i < fiberDim; i++) {
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
  template<typename Point_, typename Value_, typename Alloc_ = malloc_allocator<Value_>,
           typename Atlas_ = UniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other> >
  class Section : public ALE::ParallelObject {
  public:
    typedef Point_                                                  point_type;
    typedef Value_                                                  value_type;
    typedef Alloc_                                                  alloc_type;
    typedef Atlas_                                                  atlas_type;
    typedef Point                                                   index_type;
    typedef typename atlas_type::chart_type                         chart_type;
    typedef value_type *                                            values_type;
    typedef typename alloc_type::template rebind<atlas_type>::other atlas_alloc_type;
    typedef typename atlas_alloc_type::pointer                      atlas_ptr;
  protected:
    Obj<atlas_type> _atlas;
    Obj<atlas_type> _atlasNew;
    values_type     _array;
    alloc_type      _allocator;
  public:
    Section(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      atlas_ptr pAtlas = atlas_alloc_type(this->_allocator).allocate(1);
      atlas_alloc_type(this->_allocator).construct(pAtlas, atlas_type(comm, debug));
      this->_atlas    = Obj<atlas_type>(pAtlas, sizeof(atlas_type));
      this->_atlasNew = NULL;
      this->_array    = NULL;
    };
    Section(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas), _atlasNew(NULL), _array(NULL) {};
    virtual ~Section() {
      if (this->_array) {
        const int totalSize = this->sizeWithBC();

        for(int i = 0; i < totalSize; ++i) {this->_allocator.destroy(this->_array+i);}
        this->_allocator.deallocate(this->_array, totalSize);
        ///delete [] this->_array;
        this->_array = NULL;
      }
    };
  public:
    value_type *getRawArray(const int size) {
      static value_type *array   = NULL;
      static int         maxSize = 0;

      if (size > maxSize) {
        const value_type dummy(0);

        if (array) {
          for(int i = 0; i < maxSize; ++i) {this->_allocator.destroy(array+i);}
          this->_allocator.deallocate(array, maxSize);
          ///delete [] array;
        }
        maxSize = size;
        array   = this->_allocator.allocate(maxSize);
        for(int i = 0; i < maxSize; ++i) {this->_allocator.construct(array+i, dummy);}
        ///array = new value_type[maxSize];
      };
      return array;
    };
    int getStorageSize() const {
      return this->sizeWithBC();
    };
  public: // Verifiers
    bool hasPoint(const point_type& point) {
      return this->_atlas->hasPoint(point);
    };
  public: // Accessors
    const chart_type& getChart() {return this->_atlas->getChart();};
    void setChart(chart_type& chart) {};
    const Obj<atlas_type>& getAtlas() {return this->_atlas;};
    void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
    const Obj<atlas_type>& getNewAtlas() {return this->_atlasNew;};
    void setNewAtlas(const Obj<atlas_type>& atlas) {this->_atlasNew = atlas;};
    const chart_type& getChart() const {return this->_atlas->getChart();};
  public: // BC
    // Returns the number of constraints on a point
    int getConstraintDimension(const point_type& p) const {
      return std::max(0, -this->_atlas->restrictPoint(p)->prefix);
    }
    // Set the number of constraints on a point
    //   We only allow the entire point to be constrained, so these will be the
    //   only dofs on the point
    void setConstraintDimension(const point_type& p, const int numConstraints) {
      this->setFiberDimension(p, -numConstraints);
    };
    void addConstraintDimension(const point_type& p, const int numConstraints) {
      throw ALE::Exception("Variable constraint dimensions not available for this Section type");
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
      const int totalSize = this->sizeWithBC();

      for(int i = 0; i < totalSize; ++i) {this->_allocator.destroy(this->_array+i);}
      this->_allocator.deallocate(this->_array, totalSize);
      ///delete [] this->_array;
      this->_array = NULL;
      this->_atlas->clear(); 
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
    }
    void addFiberDimension(const point_type& p, int dim) {
      if (this->_atlas->hasPoint(p)) {
        const index_type values(dim, 0);
        this->_atlas->updateAddPoint(p, &values);
      } else {
        this->setFiberDimension(p, dim);
      }
    }
    // Return the number of constrained dofs on this point
    //   If constrained, this is equal to the fiber dimension
    //   Otherwise, 0
    int getConstrainedFiberDimension(const point_type& p) const {
      return std::max((index_type::prefix_type) 0, this->_atlas->restrictPoint(p)->prefix);
    };
    // Return the total number of free dofs
    int size() const {
      const chart_type& points = this->getChart();
      int size = 0;

      for(typename chart_type::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getConstrainedFiberDimension(*p_iter);
      }
      return size;
    };
    // Return the total number of dofs (free and constrained)
    int sizeWithBC() const {
      const chart_type& points = this->getChart();
      int size = 0;

      for(typename chart_type::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getFiberDimension(*p_iter);
      }
      return size;
    };
  public: // Index retrieval
    const typename index_type::index_type& getIndex(const point_type& p) {
      return this->_atlas->restrictPoint(p)->index;
    };
    void setIndex(const point_type& p, const typename index_type::index_type& index) {
      ((typename atlas_type::value_type *) this->_atlas->restrictPoint(p))->index = index;
    };
    void setIndexBC(const point_type& p, const typename index_type::index_type& index) {
      this->setIndex(p, index);
    };
    void getIndices(const point_type& p, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false, const bool skipConstraints = false) {
      this->getIndices(p, this->getIndex(p), indices, indx, orientation, freeOnly, skipConstraints);
    };
    template<typename Order_>
    void getIndices(const point_type& p, const Obj<Order_>& order, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false, const bool skipConstraints = false) {
      this->getIndices(p, order->getIndex(p), indices, indx, orientation, freeOnly, skipConstraints);
    }
    void getIndices(const point_type& p, const int start, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false, const bool skipConstraints = false) {
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
      const value_type dummy(0);

      this->_array = this->_allocator.allocate(totalSize);
      ///this->_array = new value_type[totalSize];
      for(int i = 0; i < totalSize; ++i) {this->_allocator.construct(this->_array+i, dummy);}
      ///PetscMemzero(this->_array, totalSize * sizeof(value_type));
    };
    void replaceStorage(value_type *newArray) {
      const int totalSize = this->sizeWithBC();

      for(int i = 0; i < totalSize; ++i) {this->_allocator.destroy(this->_array+i);}
      this->_allocator.deallocate(this->_array, totalSize);
      ///delete [] this->_array;
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
    }
    void orderPoints(const Obj<atlas_type>& atlas){
      const chart_type& chart    = this->getChart();
      int               offset   = 0;
      int               bcOffset = this->size();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
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
    const value_type *restrictSpace() {
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
      if (chart.size() == 0) {
        txt << "[" << this->commRank() << "]: empty" << std::endl;
      }
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  public: // Fibrations
    void setConstraintDof(const point_type& p, const int dofs[]) {};
    int getNumSpaces() const {return 1;};
    void addSpace() {};
    int getFiberDimension(const point_type& p, const int space) {return this->getFiberDimension(p);};
    void setFiberDimension(const point_type& p, int dim, const int space) {this->setFiberDimension(p, dim);};
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim, const int space) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim, space);
      }
    }
    void setConstraintDimension(const point_type& p, const int numConstraints, const int space) {
      this->setConstraintDimension(p, numConstraints);
    }
  };
  // GeneralSection will support BC on a subset of unknowns on a point
  //   We make a separate constraint Atlas to mark constrained dofs on a point
  //   Storage will be contiguous by node, just as in Section
  //     This allows fast restrict(p)
  //     Then update() is accomplished by skipping constrained unknowns
  //     We must eliminate restrictSpace() since it does not correspond to the constrained system
  //   Numbering will have to be rewritten to calculate correct mappings
  //     I think we can just generate multiple tuples per point
  template<typename Point_, typename Value_, typename Alloc_ = malloc_allocator<Value_>,
           typename Atlas_ = UniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other>,
           typename BCAtlas_ = Section<Point_, int, typename Alloc_::template rebind<int>::other> >
  class GeneralSection : public ALE::ParallelObject {
  public:
    typedef Point_                                                  point_type;
    typedef Value_                                                  value_type;
    typedef Alloc_                                                  alloc_type;
    typedef Atlas_                                                  atlas_type;
    typedef BCAtlas_                                                bc_type;
    typedef Point                                                   index_type;
    typedef typename atlas_type::chart_type                         chart_type;
    typedef value_type *                                            values_type;
    typedef std::pair<const int *, const int *>                     customAtlasInd_type;
    typedef std::pair<customAtlasInd_type, bool>                    customAtlas_type;
    typedef typename alloc_type::template rebind<atlas_type>::other atlas_alloc_type;
    typedef typename atlas_alloc_type::pointer                      atlas_ptr;
    typedef typename alloc_type::template rebind<bc_type>::other    bc_alloc_type;
    typedef typename bc_alloc_type::pointer                         bc_ptr;
  protected:
    // Describes layout of storage, point --> (# of values, offset into array)
    Obj<atlas_type> _atlas;
    // Spare atlas to allow dynamic updates
    Obj<atlas_type> _atlasNew;
    // Storage
    values_type     _array;
    bool            _sharedStorage;
    int             _sharedStorageSize;
    // A section that maps points to sets of constrained local dofs
    Obj<bc_type>    _bc;
    alloc_type      _allocator;
    // Fibration structures
    //   _spaces is a set of atlases which describe the layout of each in the storage of this section
    //   _bcs is the same as _bc, but for each field
    std::vector<Obj<atlas_type> > _spaces;
    std::vector<Obj<bc_type> >    _bcs;
    // Optimization
    std::vector<customAtlas_type> _customRestrictAtlas;
    std::vector<customAtlas_type> _customUpdateAtlas;
  public:
    GeneralSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      atlas_ptr pAtlas = atlas_alloc_type(this->_allocator).allocate(1);
      atlas_alloc_type(this->_allocator).construct(pAtlas, atlas_type(comm, debug));
      this->_atlas         = Obj<atlas_type>(pAtlas, sizeof(atlas_type));
      bc_ptr pBC           = bc_alloc_type(this->_allocator).allocate(1);
      bc_alloc_type(this->_allocator).construct(pBC, bc_type(comm, debug));
      this->_bc            = Obj<bc_type>(pBC, sizeof(bc_type));
      this->_atlasNew      = NULL;
      this->_array         = NULL;
      this->_sharedStorage = false;
    };
    GeneralSection(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas), _atlasNew(NULL), _array(NULL), _sharedStorage(false), _sharedStorageSize(0) {
      bc_ptr pBC = bc_alloc_type(this->_allocator).allocate(1);
      bc_alloc_type(this->_allocator).construct(pBC, bc_type(this->comm(), this->debug()));
      this->_bc  = Obj<bc_type>(pBC, sizeof(bc_type));
    };
    ~GeneralSection() {
      if (this->_array && !this->_sharedStorage) {
        const int totalSize = this->sizeWithBC();

        for(int i = 0; i < totalSize; ++i) {this->_allocator.destroy(this->_array+i);}
        this->_allocator.deallocate(this->_array, totalSize);
        ///delete [] this->_array;
        this->_array = NULL;
      }
      for(std::vector<customAtlas_type>::iterator a_iter = this->_customRestrictAtlas.begin(); a_iter != this->_customRestrictAtlas.end(); ++a_iter) {
        if (a_iter->second) {
          delete [] a_iter->first.first;
          delete [] a_iter->first.second;
        }
      }
      for(std::vector<customAtlas_type>::iterator a_iter = this->_customUpdateAtlas.begin(); a_iter != this->_customUpdateAtlas.end(); ++a_iter) {
        if (a_iter->second) {
          delete [] a_iter->first.first;
          delete [] a_iter->first.second;
        }
      }
    };
  public:
    value_type *getRawArray(const int size) {
      // Put in a sentinel value that deallocates the array
      static value_type *array   = NULL;
      static int         maxSize = 0;

      if (size > maxSize) {
        const value_type dummy(0);

        if (array) {
          for(int i = 0; i < maxSize; ++i) {this->_allocator.destroy(array+i);}
          this->_allocator.deallocate(array, maxSize);
          ///delete [] array;
        }
        maxSize = size;
        array   = this->_allocator.allocate(maxSize);
        for(int i = 0; i < maxSize; ++i) {this->_allocator.construct(array+i, dummy);}
        ///array = new value_type[maxSize];
      };
      return array;
    };
    int getStorageSize() const {
      if (this->_sharedStorage) {
        return this->_sharedStorageSize;
      }
      return this->sizeWithBC();
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
    void setChart(const chart_type& chart) {throw ALE::Exception("setChart() for GeneralSection is invalid");};
  public: // BC
    // Returns the number of constraints on a point
    int getConstraintDimension(const point_type& p) const {
      if (!this->_bc->hasPoint(p)) return 0;
      return this->_bc->getFiberDimension(p);
    }
    // Set the number of constraints on a point
    void setConstraintDimension(const point_type& p, const int numConstraints) {
      this->_bc->setFiberDimension(p, numConstraints);
    }
    // Increment the number of constraints on a point
    void addConstraintDimension(const point_type& p, const int numConstraints) {
      this->_bc->addFiberDimension(p, numConstraints);
    }
    // Return the local dofs which are constrained on a point
    const int *getConstraintDof(const point_type& p) const {
      return this->_bc->restrictPoint(p);
    }
    // Set the local dofs which are constrained on a point
    void setConstraintDof(const point_type& p, const int dofs[]) {
      this->_bc->updatePoint(p, dofs);
    }
    template<typename OtherSection>
    void copyBC(const Obj<OtherSection>& section) {
      this->setBC(section->getBC());
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator p_iter = chart.begin(); p_iter != chart.end(); ++p_iter) {
        if (this->getConstraintDimension(*p_iter)) {
          this->updatePointBCFull(*p_iter, section->restrictPoint(*p_iter));
        }
      }
      this->copyFibration(section);
    }
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
      delete [] dofs;
    };
  public: // Sizes
    void clear() {
      if (!this->_sharedStorage) {
        const int totalSize = this->sizeWithBC();

        for(int i = 0; i < totalSize; ++i) {this->_allocator.destroy(this->_array+i);}
        this->_allocator.deallocate(this->_array, totalSize);
        ///delete [] this->_array;
      }
      this->_array = NULL;
      this->_atlas->clear(); 
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
    }
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

      for(typename chart_type::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getConstrainedFiberDimension(*p_iter);
      }
      return size;
    };
    // Return the total number of dofs (free and constrained)
    int sizeWithBC() const {
      const chart_type& points = this->getChart();
      int               size   = 0;

      for(typename chart_type::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getFiberDimension(*p_iter);
      }
      return size;
    };
  public: // Index retrieval
    const typename index_type::index_type& getIndex(const point_type& p) {
      return this->_atlas->restrictPoint(p)->index;
    };
    void setIndex(const point_type& p, const typename index_type::index_type& index) {
      ((typename atlas_type::value_type *) this->_atlas->restrictPoint(p))->index = index;
    };
    void setIndexBC(const point_type& p, const typename index_type::index_type& index) {};
    void getIndices(const point_type& p, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false, const bool skipConstraints = true) {
      this->getIndices(p, this->getIndex(p), indices, indx, orientation, freeOnly, skipConstraints);
    };
    template<typename Order_>
    void getIndices(const point_type& p, const Obj<Order_>& order, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false, const bool skipConstraints = false) {
      this->getIndices(p, order->getIndex(p), indices, indx, orientation, freeOnly, skipConstraints);
    }
    void getIndicesRaw(const point_type& p, const int start, PetscInt indices[], PetscInt *indx, const int orientation) {
      if (orientation >= 0) {
        const int& dim = this->getFiberDimension(p);
        const int  end = start + dim;

        for(int i = start; i < end; ++i) {
          indices[(*indx)++] = i;
        }
      } else {
        const int numSpaces = this->getNumSpaces();
        int offset = start;

        for(int space = 0; space < numSpaces; ++space) {
          const int& dim = this->getFiberDimension(p, space);

          for(int i = offset+dim-1; i >= offset; --i) {
            indices[(*indx)++] = i;
          }
          offset += dim;
        }
        if (!numSpaces) {
          const int& dim = this->getFiberDimension(p);

          for(int i = offset+dim-1; i >= offset; --i) {
            indices[(*indx)++] = i;
          }
          offset += dim;
        }
      }
    }
    void getIndices(const point_type& p, const int start, PetscInt indices[], PetscInt *indx, const int orientation = 1, const bool freeOnly = false, const bool skipConstraints = false) {
      const int& cDim = this->getConstraintDimension(p);

      if (!cDim) {
        this->getIndicesRaw(p, start, indices, indx, orientation);
      } else {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = start, k = 0; k < dim; ++k) {
            if ((cInd < cDim) && (k == cDof[cInd])) {
              if (!freeOnly) indices[(*indx)++] = -(k+1);
              if (skipConstraints) ++i;
              ++cInd;
            } else {
              indices[(*indx)++] = i++;
            }
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = -1;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            int       cInd = (dim - tDim)-1;

            j += dim;
            for(int i = 0, k = start+tDim+offset; i < dim; ++i, --j) {
              if ((cInd >= 0) && (j == cDof[cInd+cOffset])) {
                if (!freeOnly) indices[(*indx)++] = -(offset+i+1);
                if (skipConstraints) --k;
                --cInd;
              } else {
                indices[(*indx)++] = --k;
              }
            }
            j       += dim;
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
  public: // Allocation
    void allocateStorage() {
      const int totalSize = this->sizeWithBC();
      const value_type dummy(0) ;

      this->_array             = this->_allocator.allocate(totalSize);
      ///this->_array             = new value_type[totalSize];
      this->_sharedStorage     = false;
      this->_sharedStorageSize = 0;
      for(int i = 0; i < totalSize; ++i) {this->_allocator.construct(this->_array+i, dummy);}
      ///PetscMemzero(this->_array, totalSize * sizeof(value_type));
      this->_bc->allocatePoint();
      for(typename std::vector<Obj<bc_type> >::const_iterator b_iter = this->_bcs.begin(); b_iter != this->_bcs.end(); ++b_iter) {
        (*b_iter)->allocatePoint();;
      }
    };
    void replaceStorage(value_type *newArray, const bool sharedStorage = false, const int sharedStorageSize = 0) {
      if (this->_array && !this->_sharedStorage) {
        const int totalSize = this->sizeWithBC();

        for(int i = 0; i < totalSize; ++i) {this->_allocator.destroy(this->_array+i);}
        this->_allocator.deallocate(this->_array, totalSize);
        ///delete [] this->_array;
      }
      this->_array             = newArray;
      this->_sharedStorage     = sharedStorage;
      this->_sharedStorageSize = sharedStorageSize;
      this->_atlas             = this->_atlasNew;
      this->_atlasNew          = NULL;
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

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
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
      this->set(0.0);
    };
    void zeroWithBC() {
      memset(this->_array, 0, this->sizeWithBC()* sizeof(value_type));
    };
    void set(const value_type value) {
      //memset(this->_array, 0, this->sizeWithBC()* sizeof(value_type));
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        value_type *array = (value_type *) this->restrictPoint(*c_iter);
        const int&  dim   = this->getFiberDimension(*c_iter);
        const int&  cDim  = this->getConstraintDimension(*c_iter);

        if (!cDim) {
          for(int i = 0; i < dim; ++i) {
            array[i] = value;
          }
        } else {
          const typename bc_type::value_type *cDof = this->getConstraintDof(*c_iter);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = value;
          }
        }
      }
    };
    // Add two sections and put the result in a third
    void add(const Obj<GeneralSection>& x, const Obj<GeneralSection>& y) {
      // Check atlases
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        value_type       *array  = (value_type *) this->restrictPoint(*c_iter);
        const value_type *xArray = x->restrictPoint(*c_iter);
        const value_type *yArray = y->restrictPoint(*c_iter);
        const int&        dim    = this->getFiberDimension(*c_iter);
        const int&        cDim   = this->getConstraintDimension(*c_iter);

        if (!cDim) {
          for(int i = 0; i < dim; ++i) {
            array[i] = xArray[i] + yArray[i];
          }
        } else {
          const typename bc_type::value_type *cDof = this->getConstraintDof(*c_iter);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = xArray[i] + yArray[i];
          }
        }
      }
    };
    // this = this + alpha * x
    template<typename OtherSection>
    void axpy(const value_type alpha, const Obj<OtherSection>& x) {
      // Check atlases
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        value_type       *array  = (value_type *) this->restrictPoint(*c_iter);
        const value_type *xArray = x->restrictPoint(*c_iter);
        const int&        dim    = this->getFiberDimension(*c_iter);
        const int&        cDim   = this->getConstraintDimension(*c_iter);

        if (!cDim) {
          for(int i = 0; i < dim; ++i) {
            array[i] += alpha*xArray[i];
          }
        } else {
          const typename bc_type::value_type *cDof = this->getConstraintDof(*c_iter);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] += alpha*xArray[i];
          }
        }
      }
    }
    // Return the free values on a point
    const value_type *restrictSpace() const {
      return this->_array;
    }
    // Return the free values on a point
    const value_type *restrictPoint(const point_type& p) const {
      return &(this->_array[this->_atlas->restrictPoint(p)[0].index]);
    }
    void restrictPoint(const point_type& p, value_type values[], const int size) const {
      assert(this->_atlas->restrictPoint(p)[0].prefix == size);
      const value_type *v = &(this->_array[this->_atlas->restrictPoint(p)[0].index]);

      for(int i = 0; i < size; ++i) {
        values[i] = v[i];
      }
    };
    // Update the free values on a point
    //   Takes a full array and ignores constrained values
    void updatePoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          const int& dim = this->getFiberDimension(p);

          for(int i = 0; i < dim; ++i) {
            array[i] = v[i];
          }
        } else {
          int offset = 0;
          int j      = -1;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int& dim = this->getFiberDimension(p, space);

            for(int i = dim-1; i >= 0; --i) {
              array[++j] = v[i+offset];
            }
            offset += dim;
          }
        }
      } else {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = v[i];
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            const int sDim = dim - tDim;
            int       cInd = 0;

            for(int i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
              if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
              array[j] = v[k];
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update the free values on a point
    //   Takes a full array and ignores constrained values
    void updateAddPoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          const int& dim = this->getFiberDimension(p);

          for(int i = 0; i < dim; ++i) {
            array[i] += v[i];
          }
        } else {
          int offset = 0;
          int j      = -1;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int& dim = this->getFiberDimension(p, space);

            for(int i = dim-1; i >= 0; --i) {
              array[++j] += v[i+offset];
            }
            offset += dim;
          }
        }
      } else {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] += v[i];
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            const int sDim = dim - tDim;
            int       cInd = 0;

            for(int i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
              if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
              array[j] += v[k];
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update the free values on a point
    //   Takes ONLY unconstrained values
    void updateFreePoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          const int& dim = this->getFiberDimension(p);

          for(int i = 0; i < dim; ++i) {
            array[i] = v[i];
          }
        } else {
          int offset = 0;
          int j      = -1;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int& dim = this->getFiberDimension(p, space);

            for(int i = dim-1; i >= 0; --i) {
              array[++j] = v[i+offset];
            }
            offset += dim;
          }
        }
      } else {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0, k = -1; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = v[++k];
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            const int sDim = dim - tDim;
            int       cInd = 0;

            for(int i = 0, k = tDim+offset-1; i < dim; ++i, ++j) {
              if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
              array[j] = v[--k];
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update the free values on a point
    //   Takes ONLY unconstrained values
    void updateFreeAddPoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          const int& dim = this->getFiberDimension(p);

          for(int i = 0; i < dim; ++i) {
            array[i] += v[i];
          }
        } else {
          int offset = 0;
          int j      = -1;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int& dim = this->getFiberDimension(p, space);

            for(int i = dim-1; i >= 0; --i) {
              array[++j] += v[i+offset];
            }
            offset += dim;
          }
        }
      } else {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0, k = -1; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] += v[++k];
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            const int sDim = dim - tDim;
            int       cInd = 0;

            for(int i = 0, k = tDim+offset-1; i < dim; ++i, ++j) {
              if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
              array[j] += v[--k];
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update only the constrained dofs on a point
    //   This takes an array with ONLY bc values
    void updatePointBC(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (cDim) {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if (cInd == cDim) break;
            if (i == cDof[cInd]) {
              array[i] = v[cInd];
              ++cInd;
            }
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            int       cInd = 0;

            for(int i = 0; i < dim; ++i, ++j) {
              if (cInd < 0) break;
              if (j == cDof[cInd+cOffset]) {
                array[j] = v[cInd+cOffset];
                ++cInd;
              }
            }
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update only the constrained dofs on a point
    //   This takes an array with ALL values, not just BC
    void updatePointBCFull(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (cDim) {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if (cInd == cDim) break;
            if (i == cDof[cInd]) {
              array[i] = v[i];
              ++cInd;
            }
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            int       cInd = 0;

            for(int i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
              if (cInd < 0) break;
              if (j == cDof[cInd+cOffset]) {
                array[j] = v[k];
                ++cInd;
              }
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update all dofs on a point (free and constrained)
    void updatePointAll(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);

      if (orientation >= 0) {
        const int& dim = this->getFiberDimension(p);

        for(int i = 0; i < dim; ++i) {
          array[i] = v[i];
        }
      } else {
        int offset = 0;
        int j      = -1;

        for(int space = 0; space < this->getNumSpaces(); ++space) {
          const int& dim = this->getFiberDimension(p, space);

          for(int i = dim-1; i >= 0; --i) {
            array[++j] = v[i+offset];
          }
          offset += dim;
        }
      }
    };
    // Update all dofs on a point (free and constrained)
    void updatePointAllAdd(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);

      if (orientation >= 0) {
        const int& dim = this->getFiberDimension(p);

        for(int i = 0; i < dim; ++i) {
          array[i] += v[i];
        }
      } else {
        int offset = 0;
        int j      = -1;

        for(int space = 0; space < this->getNumSpaces(); ++space) {
          const int& dim = this->getFiberDimension(p, space);

          for(int i = dim-1; i >= 0; --i) {
            array[++j] += v[i+offset];
          }
          offset += dim;
        }
      }
    };
  public: // Fibrations
    int getNumSpaces() const {return this->_spaces.size();};
    const std::vector<Obj<atlas_type> >& getSpaces() {return this->_spaces;};
    const std::vector<Obj<bc_type> >& getBCs() {return this->_bcs;};
    void addSpace() {
      Obj<atlas_type> space = new atlas_type(this->comm(), this->debug());
      Obj<bc_type>    bc    = new bc_type(this->comm(), this->debug());
      this->_spaces.push_back(space);
      this->_bcs.push_back(bc);
    };
    int getFiberDimension(const point_type& p, const int space) const {
      return this->_spaces[space]->restrictPoint(p)->prefix;
    };
    void setFiberDimension(const point_type& p, int dim, const int space) {
      const index_type idx(dim, -1);
      this->_spaces[space]->addPoint(p);
      this->_spaces[space]->updatePoint(p, &idx);
    };
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim, const int space) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim, space);
      }
    }
    int getConstraintDimension(const point_type& p, const int space) const {
      if (!this->_bcs[space]->hasPoint(p)) return 0;
      return this->_bcs[space]->getFiberDimension(p);
    }
    void setConstraintDimension(const point_type& p, const int numConstraints, const int space) {
      this->_bcs[space]->setFiberDimension(p, numConstraints);
    }
    void addConstraintDimension(const point_type& p, const int numConstraints, const int space) {
      this->_bcs[space]->addFiberDimension(p, numConstraints);
    }
    int getConstrainedFiberDimension(const point_type& p, const int space) const {
      return this->getFiberDimension(p, space) - this->getConstraintDimension(p, space);
    }
    // Return the local dofs which are constrained on a point
    const int *getConstraintDof(const point_type& p, const int space) const {
      return this->_bcs[space]->restrictPoint(p);
    }
    // Set the local dofs which are constrained on a point
    void setConstraintDof(const point_type& p, const int dofs[], const int space) {
      this->_bcs[space]->updatePoint(p, dofs);
    }
    // Return the total number of free dofs
    int size(const int space) const {
      const chart_type& points = this->getChart();
      int               size   = 0;

      for(typename chart_type::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getConstrainedFiberDimension(*p_iter, space);
      }
      return size;
    };
    template<typename OtherSection>
    void copyFibration(const Obj<OtherSection>& section) {
      const std::vector<Obj<atlas_type> >& spaces = section->getSpaces();
      const std::vector<Obj<bc_type> >&    bcs    = section->getBCs();

      this->_spaces.clear();
      for(typename std::vector<Obj<atlas_type> >::const_iterator s_iter = spaces.begin(); s_iter != spaces.end(); ++s_iter) {
        this->_spaces.push_back(*s_iter);
      }
      this->_bcs.clear();
      for(typename std::vector<Obj<bc_type> >::const_iterator b_iter = bcs.begin(); b_iter != bcs.end(); ++b_iter) {
        this->_bcs.push_back(*b_iter);
      }
    }
    Obj<GeneralSection> getFibration(const int space) const {
      Obj<GeneralSection> field = new GeneralSection(this->comm(), this->debug());
//     Obj<atlas_type> _atlas;
//     std::vector<Obj<atlas_type> > _spaces;
//     Obj<bc_type>    _bc;
//     std::vector<Obj<bc_type> >    _bcs;
      field->addSpace();
      const chart_type& chart = this->getChart();

      // Copy sizes
      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        const int fDim = this->getFiberDimension(*c_iter, space);
        const int cDim = this->getConstraintDimension(*c_iter, space);

        if (fDim) {
          field->setFiberDimension(*c_iter, fDim);
          field->setFiberDimension(*c_iter, fDim, 0);
        }
        if (cDim) {
          field->setConstraintDimension(*c_iter, cDim);
          field->setConstraintDimension(*c_iter, cDim, 0);
        }
      }
      field->allocateStorage();
      Obj<atlas_type>   newAtlas = new atlas_type(this->comm(), this->debug());
      const chart_type& newChart = field->getChart();

      for(typename chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
        const int cDim   = field->getConstraintDimension(*c_iter);
        const int dof[1] = {0};

        if (cDim) {
          field->setConstraintDof(*c_iter, this->getConstraintDof(*c_iter, space));
        }
      }
      // Copy offsets
      for(typename chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
        index_type idx;

        idx.prefix = field->getFiberDimension(*c_iter);
        idx.index  = this->_atlas->restrictPoint(*c_iter)[0].index;
        for(int s = 0; s < space; ++s) {
          idx.index += this->getFiberDimension(*c_iter, s);
        }
        newAtlas->addPoint(*c_iter);
        newAtlas->updatePoint(*c_iter, &idx);
      }
      field->replaceStorage(this->_array, true, this->getStorageSize());
      field->setAtlas(newAtlas);
      return field;
    };
  public: // Optimization
    void getCustomRestrictAtlas(const int tag, const int *offsets[], const int *indices[]) {
      *offsets = this->_customRestrictAtlas[tag].first.first;
      *indices = this->_customRestrictAtlas[tag].first.second;
    };
    void getCustomUpdateAtlas(const int tag, const int *offsets[], const int *indices[]) {
      *offsets = this->_customUpdateAtlas[tag].first.first;
      *indices = this->_customUpdateAtlas[tag].first.second;
    };
    // This returns the tag assigned to the atlas
    int setCustomAtlas(const int restrictOffsets[], const int restrictIndices[], const int updateOffsets[], const int updateIndices[], bool autoFree = true) {
      this->_customRestrictAtlas.push_back(customAtlas_type(customAtlasInd_type(restrictOffsets, restrictIndices), autoFree));
      this->_customUpdateAtlas.push_back(customAtlas_type(customAtlasInd_type(updateOffsets, updateIndices), autoFree));
      return this->_customUpdateAtlas.size()-1;
    };
    int copyCustomAtlas(const Obj<GeneralSection>& section, const int tag) {
      const int *rOffsets, *rIndices, *uOffsets, *uIndices;

      section->getCustomRestrictAtlas(tag, &rOffsets, &rIndices);
      section->getCustomUpdateAtlas(tag, &uOffsets, &uIndices);
      return this->setCustomAtlas(rOffsets, rIndices, uOffsets, uIndices, false);
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
        if (rank == 0) {
          txt << "viewing GeneralSection '" << name << "'" << std::endl;
        }
      }
      if (rank == 0) {
        txt << "  Fields: " << this->getNumSpaces() << std::endl;
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
      if (chart.size() == 0) {
        txt << "[" << this->commRank() << "]: empty" << std::endl;
      }
      PetscSynchronizedPrintf(comm, txt.str().c_str());
      PetscSynchronizedFlush(comm);
    };
  };
  // FEMSection will support vector BC on a subset of unknowns on a point
  //   We make a separate constraint Section to hold the transformation and projection operators
  //   Storage will be contiguous by node, just as in Section
  //     This allows fast restrict(p)
  //     Then update() is accomplished by projecting constrained unknowns
  template<typename Point_, typename Value_, typename Alloc_ = malloc_allocator<Value_>,
           typename Atlas_ = UniformSection<Point_, Point, 1, typename Alloc_::template rebind<Point>::other>,
           typename BCAtlas_ = UniformSection<Point_, int, 1, typename Alloc_::template rebind<int>::other>,
           typename BC_ = Section<Point_, double, typename Alloc_::template rebind<double>::other> >
  class FEMSection : public ALE::ParallelObject {
  public:
    typedef Point_                                                  point_type;
    typedef Value_                                                  value_type;
    typedef Alloc_                                                  alloc_type;
    typedef Atlas_                                                  atlas_type;
    typedef BCAtlas_                                                bc_atlas_type;
    typedef BC_                                                     bc_type;
    typedef Point                                                   index_type;
    typedef typename atlas_type::chart_type                         chart_type;
    typedef value_type *                                            values_type;
    typedef typename alloc_type::template rebind<atlas_type>::other atlas_alloc_type;
    typedef typename atlas_alloc_type::pointer                      atlas_ptr;
    typedef typename alloc_type::template rebind<bc_type>::other    bc_atlas_alloc_type;
    typedef typename bc_atlas_alloc_type::pointer                   bc_atlas_ptr;
    typedef typename alloc_type::template rebind<bc_type>::other    bc_alloc_type;
    typedef typename bc_alloc_type::pointer                         bc_ptr;
  protected:
    Obj<atlas_type>    _atlas;
    Obj<bc_atlas_type> _bc_atlas;
    Obj<bc_type>       _bc;
    values_type        _array;
    bool               _sharedStorage;
    int                _sharedStorageSize;
    alloc_type         _allocator;
    std::vector<Obj<atlas_type> > _spaces;
    std::vector<Obj<bc_type> >    _bcs;
  public:
    FEMSection(MPI_Comm comm, const int debug = 0) : ParallelObject(comm, debug) {
      atlas_ptr pAtlas      = atlas_alloc_type(this->_allocator).allocate(1);
      atlas_alloc_type(this->_allocator).construct(pAtlas, atlas_type(comm, debug));
      this->_atlas          = Obj<atlas_type>(pAtlas, sizeof(atlas_type));
      bc_atlas_ptr pBCAtlas = bc_atlas_alloc_type(this->_allocator).allocate(1);
      bc_atlas_alloc_type(this->_allocator).construct(pBCAtlas, bc_atlas_type(comm, debug));
      this->_bc_atlas       = Obj<bc_atlas_type>(pBCAtlas, sizeof(bc_atlas_type));
      bc_ptr pBC            = bc_alloc_type(this->_allocator).allocate(1);
      bc_alloc_type(this->_allocator).construct(pBC, bc_type(comm, debug));
      this->_bc             = Obj<bc_type>(pBC, sizeof(bc_type));
      this->_array          = NULL;
      this->_sharedStorage  = false;
    };
    FEMSection(const Obj<atlas_type>& atlas) : ParallelObject(atlas->comm(), atlas->debug()), _atlas(atlas), _array(NULL), _sharedStorage(false), _sharedStorageSize(0) {
      bc_atlas_ptr pBCAtlas = bc_atlas_alloc_type(this->_allocator).allocate(1);
      bc_atlas_alloc_type(this->_allocator).construct(pBCAtlas, bc_atlas_type(this->comm(), this->debug()));
      this->_bc_atlas       = Obj<bc_atlas_type>(pBCAtlas, sizeof(bc_atlas_type));
      bc_ptr pBC            = bc_alloc_type(this->_allocator).allocate(1);
      bc_alloc_type(this->_allocator).construct(pBC, bc_type(this->comm(), this->debug()));
      this->_bc             = Obj<bc_type>(pBC, sizeof(bc_type));
    };
    ~FEMSection() {
      if (this->_array && !this->_sharedStorage) {
        const int totalSize = this->sizeWithBC();

        for(int i = 0; i < totalSize; ++i) {this->_allocator.destroy(this->_array+i);}
        this->_allocator.deallocate(this->_array, totalSize);
        this->_array = NULL;
      }
    };
  public:
    value_type *getRawArray(const int size) {
      // Put in a sentinel value that deallocates the array
      static value_type *array   = NULL;
      static int         maxSize = 0;

      if (size > maxSize) {
        const value_type dummy(0);

        if (array) {
          for(int i = 0; i < maxSize; ++i) {this->_allocator.destroy(array+i);}
          this->_allocator.deallocate(array, maxSize);
        }
        maxSize = size;
        array   = this->_allocator.allocate(maxSize);
        for(int i = 0; i < maxSize; ++i) {this->_allocator.construct(array+i, dummy);}
      };
      return array;
    };
    int getStorageSize() const {
      if (this->_sharedStorage) {
        return this->_sharedStorageSize;
      }
      return this->sizeWithBC();
    };
  public: // Verifiers
    bool hasPoint(const point_type& point) {
      return this->_atlas->hasPoint(point);
    };
  public: // Accessors
    const chart_type& getChart() const {return this->_atlas->getChart();};
    const Obj<atlas_type>& getAtlas() const {return this->_atlas;};
    void setAtlas(const Obj<atlas_type>& atlas) {this->_atlas = atlas;};
    const Obj<bc_type>& getBC() const {return this->_bc;};
    void setBC(const Obj<bc_type>& bc) {this->_bc = bc;};
  public: // BC
    // Returns the number of constraints on a point
    int getConstraintDimension(const point_type& p) const {
      if (!this->_bc_atlas->hasPoint(p)) return 0;
      return this->_bc_atlas->restrictPoint(p)[0];
    };
    // Set the number of constraints on a point
    void setConstraintDimension(const point_type& p, const int numConstraints) {
      this->_bc_atlas->updatePoint(p, &numConstraints);
    };
    // Increment the number of constraints on a point
    void addConstraintDimension(const point_type& p, const int numConstraints) {
      this->_bc_atlas->updatePointAdd(p, &numConstraints);
    };
    const int *getConstraintDof(const point_type& p) const {
      return this->_bc->restrictPoint(p);
    }
  public: // Sizes
    void clear() {
      if (!this->_sharedStorage) {
        const int totalSize = this->sizeWithBC();

        for(int i = 0; i < totalSize; ++i) {this->_allocator.destroy(this->_array+i);}
        this->_allocator.deallocate(this->_array, totalSize);
      }
      this->_array = NULL;
      this->_atlas->clear(); 
      this->_bc_atlas->clear(); 
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
    }
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

      for(typename chart_type::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getConstrainedFiberDimension(*p_iter);
      }
      return size;
    };
    // Return the total number of dofs (free and constrained)
    int sizeWithBC() const {
      const chart_type& points = this->getChart();
      int               size   = 0;

      for(typename chart_type::const_iterator p_iter = points.begin(); p_iter != points.end(); ++p_iter) {
        size += this->getFiberDimension(*p_iter);
      }
      return size;
    };
  public: // Allocation
    void allocateStorage() {
      const int totalSize = this->sizeWithBC();
      const value_type dummy(0) ;

      this->_array             = this->_allocator.allocate(totalSize);
      this->_sharedStorage     = false;
      this->_sharedStorageSize = 0;
      for(int i = 0; i < totalSize; ++i) {this->_allocator.construct(this->_array+i, dummy);}
      this->_bc_atlas->allocatePoint();
      this->_bc->allocatePoint();
    };
    void orderPoints(const Obj<atlas_type>& atlas){
      const chart_type& chart  = this->getChart();
      int               offset = 0;

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
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
      this->set(0.0);
    };
    void set(const value_type value) {
      //memset(this->_array, 0, this->sizeWithBC()* sizeof(value_type));
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        value_type *array = (value_type *) this->restrictPoint(*c_iter);
        const int&  dim   = this->getFiberDimension(*c_iter);
        const int&  cDim  = this->getConstraintDimension(*c_iter);

        if (!cDim) {
          for(int i = 0; i < dim; ++i) {
            array[i] = value;
          }
        } else {
          const typename bc_type::value_type *cDof = this->getConstraintDof(*c_iter);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = value;
          }
        }
      }
    };
    // Add two sections and put the result in a third
    void add(const Obj<FEMSection>& x, const Obj<FEMSection>& y) {
      // Check atlases
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        value_type       *array  = (value_type *) this->restrictPoint(*c_iter);
        const value_type *xArray = x->restrictPoint(*c_iter);
        const value_type *yArray = y->restrictPoint(*c_iter);
        const int&        dim    = this->getFiberDimension(*c_iter);
        const int&        cDim   = this->getConstraintDimension(*c_iter);

        if (!cDim) {
          for(int i = 0; i < dim; ++i) {
            array[i] = xArray[i] + yArray[i];
          }
        } else {
          const typename bc_type::value_type *cDof = this->getConstraintDof(*c_iter);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = xArray[i] + yArray[i];
          }
        }
      }
    };
    // this = this + alpha * x
    void axpy(const value_type alpha, const Obj<FEMSection>& x) {
      // Check atlases
      const chart_type& chart = this->getChart();

      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        value_type       *array  = (value_type *) this->restrictPoint(*c_iter);
        const value_type *xArray = x->restrictPoint(*c_iter);
        const int&        dim    = this->getFiberDimension(*c_iter);
        const int&        cDim   = this->getConstraintDimension(*c_iter);

        if (!cDim) {
          for(int i = 0; i < dim; ++i) {
            array[i] += alpha*xArray[i];
          }
        } else {
          const typename bc_type::value_type *cDof = this->getConstraintDof(*c_iter);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] += alpha*xArray[i];
          }
        }
      }
    };
    // Return the free values on a point
    const value_type *restrictSpace() const {
      return this->_array;
    };
    // Return the free values on a point
    const value_type *restrictPoint(const point_type& p) const {
      return &(this->_array[this->_atlas->restrictPoint(p)[0].index]);
    };
    // Update the free values on a point
    //   Takes a full array and ignores constrained values
    void updatePoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          const int& dim = this->getFiberDimension(p);

          for(int i = 0; i < dim; ++i) {
            array[i] = v[i];
          }
        } else {
          int offset = 0;
          int j      = -1;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int& dim = this->getFiberDimension(p, space);

            for(int i = dim-1; i >= 0; --i) {
              array[++j] = v[i+offset];
            }
            offset += dim;
          }
        }
      } else {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = v[i];
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            const int sDim = dim - tDim;
            int       cInd = 0;

            for(int i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
              if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
              array[j] = v[k];
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update the free values on a point
    //   Takes a full array and ignores constrained values
    void updateAddPoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          const int& dim = this->getFiberDimension(p);

          for(int i = 0; i < dim; ++i) {
            array[i] += v[i];
          }
        } else {
          int offset = 0;
          int j      = -1;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int& dim = this->getFiberDimension(p, space);

            for(int i = dim-1; i >= 0; --i) {
              array[++j] += v[i+offset];
            }
            offset += dim;
          }
        }
      } else {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] += v[i];
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            const int sDim = dim - tDim;
            int       cInd = 0;

            for(int i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
              if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
              array[j] += v[k];
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update the free values on a point
    //   Takes ONLY unconstrained values
    void updateFreePoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          const int& dim = this->getFiberDimension(p);

          for(int i = 0; i < dim; ++i) {
            array[i] = v[i];
          }
        } else {
          int offset = 0;
          int j      = -1;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int& dim = this->getFiberDimension(p, space);

            for(int i = dim-1; i >= 0; --i) {
              array[++j] = v[i+offset];
            }
            offset += dim;
          }
        }
      } else {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0, k = -1; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] = v[++k];
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            const int sDim = dim - tDim;
            int       cInd = 0;

            for(int i = 0, k = tDim+offset-1; i < dim; ++i, ++j) {
              if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
              array[j] = v[--k];
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update the free values on a point
    //   Takes ONLY unconstrained values
    void updateFreeAddPoint(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (!cDim) {
        if (orientation >= 0) {
          const int& dim = this->getFiberDimension(p);

          for(int i = 0; i < dim; ++i) {
            array[i] += v[i];
          }
        } else {
          int offset = 0;
          int j      = -1;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int& dim = this->getFiberDimension(p, space);

            for(int i = dim-1; i >= 0; --i) {
              array[++j] += v[i+offset];
            }
            offset += dim;
          }
        }
      } else {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0, k = -1; i < dim; ++i) {
            if ((cInd < cDim) && (i == cDof[cInd])) {++cInd; continue;}
            array[i] += v[++k];
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            const int sDim = dim - tDim;
            int       cInd = 0;

            for(int i = 0, k = tDim+offset-1; i < dim; ++i, ++j) {
              if ((cInd < sDim) && (j == cDof[cInd+cOffset])) {++cInd; continue;}
              array[j] += v[--k];
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update only the constrained dofs on a point
    //   This takes an array with ONLY bc values
    void updatePointBC(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (cDim) {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if (cInd == cDim) break;
            if (i == cDof[cInd]) {
              array[i] = v[cInd];
              ++cInd;
            }
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            int       cInd = 0;

            for(int i = 0; i < dim; ++i, ++j) {
              if (cInd < 0) break;
              if (j == cDof[cInd+cOffset]) {
                array[j] = v[cInd+cOffset];
                ++cInd;
              }
            }
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update only the constrained dofs on a point
    //   This takes an array with ALL values, not just BC
    void updatePointBCFull(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);
      const int&  cDim  = this->getConstraintDimension(p);

      if (cDim) {
        if (orientation >= 0) {
          const int&                          dim  = this->getFiberDimension(p);
          const typename bc_type::value_type *cDof = this->getConstraintDof(p);
          int                                 cInd = 0;

          for(int i = 0; i < dim; ++i) {
            if (cInd == cDim) break;
            if (i == cDof[cInd]) {
              array[i] = v[i];
              ++cInd;
            }
          }
        } else {
          const typename bc_type::value_type *cDof    = this->getConstraintDof(p);
          int                                 offset  = 0;
          int                                 cOffset = 0;
          int                                 j       = 0;

          for(int space = 0; space < this->getNumSpaces(); ++space) {
            const int  dim = this->getFiberDimension(p, space);
            const int tDim = this->getConstrainedFiberDimension(p, space);
            int       cInd = 0;

            for(int i = 0, k = dim+offset-1; i < dim; ++i, ++j, --k) {
              if (cInd < 0) break;
              if (j == cDof[cInd+cOffset]) {
                array[j] = v[k];
                ++cInd;
              }
            }
            offset  += dim;
            cOffset += dim - tDim;
          }
        }
      }
    };
    // Update all dofs on a point (free and constrained)
    void updatePointAll(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);

      if (orientation >= 0) {
        const int& dim = this->getFiberDimension(p);

        for(int i = 0; i < dim; ++i) {
          array[i] = v[i];
        }
      } else {
        int offset = 0;
        int j      = -1;

        for(int space = 0; space < this->getNumSpaces(); ++space) {
          const int& dim = this->getFiberDimension(p, space);

          for(int i = dim-1; i >= 0; --i) {
            array[++j] = v[i+offset];
          }
          offset += dim;
        }
      }
    };
    // Update all dofs on a point (free and constrained)
    void updatePointAllAdd(const point_type& p, const value_type v[], const int orientation = 1) {
      value_type *array = (value_type *) this->restrictPoint(p);

      if (orientation >= 0) {
        const int& dim = this->getFiberDimension(p);

        for(int i = 0; i < dim; ++i) {
          array[i] += v[i];
        }
      } else {
        int offset = 0;
        int j      = -1;

        for(int space = 0; space < this->getNumSpaces(); ++space) {
          const int& dim = this->getFiberDimension(p, space);

          for(int i = dim-1; i >= 0; --i) {
            array[++j] += v[i+offset];
          }
          offset += dim;
        }
      }
    };
  public: // Fibrations
    int getNumSpaces() const {return this->_spaces.size();};
    const std::vector<Obj<atlas_type> >& getSpaces() {return this->_spaces;};
    const std::vector<Obj<bc_type> >& getBCs() {return this->_bcs;};
    void addSpace() {
      Obj<atlas_type> space = new atlas_type(this->comm(), this->debug());
      Obj<bc_type>    bc    = new bc_type(this->comm(), this->debug());
      this->_spaces.push_back(space);
      this->_bcs.push_back(bc);
    };
    int getFiberDimension(const point_type& p, const int space) const {
      return this->_spaces[space]->restrictPoint(p)->prefix;
    };
    void setFiberDimension(const point_type& p, int dim, const int space) {
      const index_type idx(dim, -1);
      this->_spaces[space]->addPoint(p);
      this->_spaces[space]->updatePoint(p, &idx);
    };
    template<typename Sequence>
    void setFiberDimension(const Obj<Sequence>& points, int dim, const int space) {
      for(typename Sequence::iterator p_iter = points->begin(); p_iter != points->end(); ++p_iter) {
        this->setFiberDimension(*p_iter, dim, space);
      }
    }
    int getConstraintDimension(const point_type& p, const int space) const {
      if (!this->_bcs[space]->hasPoint(p)) return 0;
      return this->_bcs[space]->getFiberDimension(p);
    };
    void setConstraintDimension(const point_type& p, const int numConstraints, const int space) {
      this->_bcs[space]->setFiberDimension(p, numConstraints);
    };
    int getConstrainedFiberDimension(const point_type& p, const int space) const {
      return this->getFiberDimension(p, space) - this->getConstraintDimension(p, space);
    };
    void copyFibration(const Obj<FEMSection>& section) {
      const std::vector<Obj<atlas_type> >& spaces = section->getSpaces();
      const std::vector<Obj<bc_type> >&    bcs    = section->getBCs();

      this->_spaces.clear();
      for(typename std::vector<Obj<atlas_type> >::const_iterator s_iter = spaces.begin(); s_iter != spaces.end(); ++s_iter) {
        this->_spaces.push_back(*s_iter);
      }
      this->_bcs.clear();
      for(typename std::vector<Obj<bc_type> >::const_iterator b_iter = bcs.begin(); b_iter != bcs.end(); ++b_iter) {
        this->_bcs.push_back(*b_iter);
      }
    };
    Obj<FEMSection> getFibration(const int space) const {
      Obj<FEMSection> field = new FEMSection(this->comm(), this->debug());
//     Obj<atlas_type> _atlas;
//     std::vector<Obj<atlas_type> > _spaces;
//     Obj<bc_type>    _bc;
//     std::vector<Obj<bc_type> >    _bcs;
      field->addSpace();
      const chart_type& chart = this->getChart();

      // Copy sizes
      for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
        const int fDim = this->getFiberDimension(*c_iter, space);
        const int cDim = this->getConstraintDimension(*c_iter, space);

        if (fDim) {
          field->setFiberDimension(*c_iter, fDim);
          field->setFiberDimension(*c_iter, fDim, 0);
        }
        if (cDim) {
          field->setConstraintDimension(*c_iter, cDim);
          field->setConstraintDimension(*c_iter, cDim, 0);
        }
      }
      field->allocateStorage();
      Obj<atlas_type>   newAtlas = new atlas_type(this->comm(), this->debug());
      const chart_type& newChart = field->getChart();

      for(typename chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
        const int cDim   = field->getConstraintDimension(*c_iter);
        const int dof[1] = {0};

        if (cDim) {
          field->setConstraintDof(*c_iter, dof);
        }
      }
      // Copy offsets
      for(typename chart_type::const_iterator c_iter = newChart.begin(); c_iter != newChart.end(); ++c_iter) {
        index_type idx;

        idx.prefix = field->getFiberDimension(*c_iter);
        idx.index  = this->_atlas->restrictPoint(*c_iter)[0].index;
        for(int s = 0; s < space; ++s) {
          idx.index += this->getFiberDimension(*c_iter, s);
        }
        newAtlas->addPoint(*c_iter);
        newAtlas->updatePoint(*c_iter, &idx);
      }
      field->replaceStorage(this->_array, true, this->getStorageSize());
      field->setAtlas(newAtlas);
      return field;
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
          txt << "viewing a FEMSection" << std::endl;
        }
      } else {
        if (rank == 0) {
          txt << "viewing FEMSection '" << name << "'" << std::endl;
        }
      }
      if (rank == 0) {
        txt << "  Fields: " << this->getNumSpaces() << std::endl;
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
      if (chart.size() == 0) {
        txt << "[" << this->commRank() << "]: empty" << std::endl;
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
    }
    void allocate() {
      for(typename sheaf_type::const_iterator p_iter = this->_sheaf.begin(); p_iter != this->_sheaf.end(); ++p_iter) {
        p_iter->second->allocatePoint();
      }
    }
  public: // Communication
    void construct(const int size) {
      const sheaf_type& patches = this->getPatches();

      for(typename sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        const patch_type         rank    = p_iter->first;
        const Obj<section_type>& section = this->getSection(rank);
        const chart_type&        chart   = section->getChart();

        for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
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
    
        for(typename chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
          section->setFiberDimension(*c_iter, *(sizer->getSection(rank)->restrictPoint(*c_iter)));
        }
      }
    }
    void constructCommunication(const request_type& requestType) {
      const sheaf_type& patches = this->getPatches();

      for(typename sheaf_type::const_iterator p_iter = patches.begin(); p_iter != patches.end(); ++p_iter) {
        const patch_type         patch   = p_iter->first;
        const Obj<section_type>& section = this->getSection(patch);
        MPI_Request              request;

        if (requestType == RECEIVE) {
          if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Receiving data(" << section->size() << ") from " << patch << " tag " << this->_tag << std::endl;}
          MPI_Recv_init((void *) section->restrictSpace(), section->size(), this->_datatype, patch, this->_tag, this->comm(), &request);
        } else {
          if (this->_debug) {std::cout <<"["<<this->commRank()<<"] Sending data (" << section->size() << ") to " << patch << " tag " << this->_tag << std::endl;}
          MPI_Send_init((void *) section->restrictSpace(), section->size(), this->_datatype, patch, this->_tag, this->comm(), &request);
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
#endif
